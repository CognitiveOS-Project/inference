//go:build cgo

package llm

/*
#cgo LDFLAGS: -L${SRCDIR}/../../vendor/llama.cpp/build -llama -lm -lstdc++
#cgo CFLAGS: -I${SRCDIR}/../../vendor/llama.cpp/include

#include <stdlib.h>
#include "llama.h"
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"
)

type cgoModel struct {
	model *C.struct_llama_model
	ctx   *C.struct_llama_context
}

func initBridge() {
	C.llama_backend_init()
}

func bridgeLoadModel(modelPath string, nCtx, nGPULayers, nThreads int) (*cgoModel, error) {
	cpath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cpath))

	mparams := C.llama_model_default_params()
	mparams.n_gpu_layers = C.int32_t(nGPULayers)
	mparams.check_tensors = true

	model := C.llama_model_load_from_file(cpath, mparams)
	if model == nil {
		return nil, fmt.Errorf("E_MODEL_LOAD_FAILED: llama_model_load_from_file returned null for %q", modelPath)
	}

	cparams := C.llama_context_default_params()
	cparams.n_ctx = C.uint32_t(nCtx)
	if nThreads > 0 {
		cparams.n_threads = C.int32_t(nThreads)
		cparams.n_threads_batch = C.int32_t(nThreads)
	}

	ctx := C.llama_init_from_model(model, cparams)
	if ctx == nil {
		C.llama_model_free(model)
		return nil, fmt.Errorf("E_MODEL_LOAD_FAILED: llama_init_from_model returned null for %q", modelPath)
	}

	cm := &cgoModel{model: model, ctx: ctx}
	runtime.SetFinalizer(cm, func(m *cgoModel) {
		m.close()
	})

	return cm, nil
}

func (m *cgoModel) close() {
	if m.ctx != nil {
		C.llama_free(m.ctx)
		m.ctx = nil
	}
	if m.model != nil {
		C.llama_model_free(m.model)
		m.model = nil
	}
}

func (m *cgoModel) vocab() *C.struct_llama_vocab {
	return C.llama_model_get_vocab(m.model)
}

func (m *cgoModel) nCtx() int {
	return int(C.llama_n_ctx(m.ctx))
}

func (m *cgoModel) modelSize() uint64 {
	return uint64(C.llama_model_size(m.model))
}

func (m *cgoModel) modelDesc() string {
	buf := make([]byte, 256)
	n := C.llama_model_desc(m.model, (*C.char)(unsafe.Pointer(&buf[0])), C.size_t(len(buf)))
	if n < 0 {
		return "unknown"
	}
	return string(buf[:n])
}

func (m *cgoModel) isEOG(token int32) bool {
	return bool(C.llama_vocab_is_eog(m.vocab(), C.llama_token(token)))
}

func bridgeTokenize(vocab *C.struct_llama_vocab, text string, addSpecial bool) ([]int32, error) {
	ctext := C.CString(text)
	defer C.free(unsafe.Pointer(ctext))

	n := C.llama_tokenize(vocab, ctext, C.int32_t(len(text)), nil, 0, C.bool(addSpecial), false)
	if n <= 0 {
		return nil, fmt.Errorf("tokenize failed: text too long or empty")
	}

	ctokens := make([]C.llama_token, n)
	n = C.llama_tokenize(vocab, ctext, C.int32_t(len(text)), &ctokens[0], C.int32_t(len(ctokens)), C.bool(addSpecial), false)
	if n < 0 {
		return nil, fmt.Errorf("tokenize failed on second pass")
	}

	tokens := make([]int32, n)
	for i := 0; i < int(n); i++ {
		tokens[i] = int32(ctokens[i])
	}
	return tokens, nil
}

func bridgeTokenToPiece(vocab *C.struct_llama_vocab, token int32) string {
	buf := make([]byte, 64)
	n := C.llama_token_to_piece(vocab, C.llama_token(token), (*C.char)(unsafe.Pointer(&buf[0])), C.int32_t(len(buf)), 0, false)
	if n < 0 {
		buf = make([]byte, -int(n)+1)
		n = C.llama_token_to_piece(vocab, C.llama_token(token), (*C.char)(unsafe.Pointer(&buf[0])), C.int32_t(len(buf)), 0, false)
	}
	if n < 0 {
		return fmt.Sprintf("<token:%d>", int(token))
	}
	return string(buf[:n])
}

func bridgeDecode(ctx *C.struct_llama_context, tokens []int32) error {
	if len(tokens) == 0 {
		return nil
	}

	ctokens := make([]C.llama_token, len(tokens))
	for i, t := range tokens {
		ctokens[i] = C.llama_token(t)
	}

	batch := C.llama_batch_get_one(&ctokens[0], C.int32_t(len(ctokens)))
	code := C.llama_decode(ctx, batch)
	if code < 0 {
		return fmt.Errorf("E_INTERNAL: llama_decode failed with code %d", int(code))
	}
	return nil
}

func bridgeSample(ctx *C.struct_llama_context, vocab *C.struct_llama_vocab, temp float64, topK int32, topP float64, seed uint32) (int32, error) {
	sparams := C.llama_sampler_chain_default_params()
	chain := C.llama_sampler_chain_init(sparams)
	if chain == nil {
		return -1, fmt.Errorf("E_INTERNAL: failed to create sampler chain")
	}
	defer C.llama_sampler_free(chain)

	if topK > 0 {
		C.llama_sampler_chain_add(chain, C.llama_sampler_init_top_k(C.int32_t(topK)))
	}
	if topP > 0.0 && topP < 1.0 {
		C.llama_sampler_chain_add(chain, C.llama_sampler_init_top_p(C.float(topP), 1))
	}
	C.llama_sampler_chain_add(chain, C.llama_sampler_init_temp(C.float(temp)))

	C.llama_sampler_chain_add(chain, C.llama_sampler_init_dist(C.uint32_t(seed)))

	token := C.llama_sampler_sample(chain, ctx, C.int32_t(-1))
	return int32(token), nil
}
