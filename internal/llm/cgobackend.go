//go:build cgo

package llm

import (
	"fmt"
	"math/rand"
	"os"
	"strings"
	"sync"
	"time"
)

var _ Backend = (*CgoBackend)(nil)

type CgoBackend struct {
	mu        sync.Mutex
	loaded    bool
	modelPath string
	modelInfo *ModelInfo
	cg        *cgoModel
	lastUse   time.Time
}

func NewCgoBackend() *CgoBackend {
	initBridge()
	return &CgoBackend{}
}

func (c *CgoBackend) Name() string { return "cgo" }

func (c *CgoBackend) Load(modelPath string, opts *LoadOptions) (*ModelInfo, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.loaded {
		if err := c.unloadLocked(); err != nil {
			return nil, err
		}
	}

	nCtx := 2048
	nGPULayers := 0
	nThreads := 0
	if opts != nil {
		if opts.NumCtx > 0 {
			nCtx = opts.NumCtx
		}
		if opts.GPULayers > 0 {
			nGPULayers = opts.GPULayers
		}
		if opts.Threads > 0 {
			nThreads = opts.Threads
		}
	}

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("E_MODEL_NOT_FOUND: %s", modelPath)
	}

	f, err := os.Open(modelPath)
	if err != nil {
		return nil, fmt.Errorf("E_MODEL_NOT_FOUND: %s: %w", modelPath, err)
	}
	magic := make([]byte, 4)
	f.Read(magic)
	f.Close()
	if string(magic) != "GGUF" {
		return nil, fmt.Errorf("E_MODEL_LOAD_FAILED: invalid GGUF magic bytes in %s", modelPath)
	}

	cg, err := bridgeLoadModel(modelPath, nCtx, nGPULayers, nThreads)
	if err != nil {
		return nil, err
	}

	c.cg = cg
	c.loaded = true
	c.modelPath = modelPath
	c.lastUse = time.Now()

	c.modelInfo = &ModelInfo{
		Name:          modelPath,
		Path:          modelPath,
		RAMUsageMB:    int64(cg.modelSize() / (1024 * 1024)),
		ContextWindow: nCtx,
		State:         StateReady,
		LoadedAt:      time.Now(),
	}

	return c.modelInfo, nil
}

func (c *CgoBackend) Unload() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.unloadLocked()
}

func (c *CgoBackend) unloadLocked() error {
	if c.cg != nil {
		c.cg.close()
		c.cg = nil
	}
	c.loaded = false
	c.modelPath = ""
	c.modelInfo = nil
	return nil
}

func (c *CgoBackend) IsLoaded() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.loaded
}

func (c *CgoBackend) LoadedModel() *ModelInfo {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.modelInfo
}

func (c *CgoBackend) Generate(req GenerateReq, onToken func(string)) (*GenerateResp, error) {
	c.mu.Lock()
	if !c.loaded || c.cg == nil {
		c.mu.Unlock()
		return nil, fmt.Errorf("E_MODEL_NOT_FOUND: no model loaded")
	}
	cg := c.cg
	c.mu.Unlock()

	vocab := cg.vocab()
	if vocab == nil {
		return nil, fmt.Errorf("E_INTERNAL: model has no vocabulary")
	}

	temp := 0.7
	numPredict := 512
	topK := int32(40)
	topP := 0.9
	if v, ok := req.Options["temperature"].(float64); ok {
		temp = v
	}
	if v, ok := req.Options["num_predict"].(float64); ok {
		numPredict = int(v)
	}
	if v, ok := req.Options["top_k"].(float64); ok {
		topK = int32(v)
	}
	if v, ok := req.Options["top_p"].(float64); ok {
		topP = v
	}

	prompt := req.Prompt
	if req.System != "" {
		prompt = fmt.Sprintf("system: %s\n\nuser: %s\nassistant:", req.System, req.Prompt)
	}

	start := time.Now()

	tokens, err := bridgeTokenize(vocab, prompt, true)
	if err != nil {
		return nil, fmt.Errorf("E_INTERNAL: %w", err)
	}

	if err := bridgeDecode(cg.ctx, tokens); err != nil {
		return nil, err
	}
	nPromptTokens := len(tokens)

	var output strings.Builder
	var generated int
	seed := uint32(rand.Int63())

	for generated < numPredict {
		token, err := bridgeSample(cg.ctx, vocab, temp, topK, topP, seed)
		if err != nil {
			return nil, err
		}

		if cg.isEOG(token) || token < 0 {
			break
		}

		piece := bridgeTokenToPiece(vocab, token)
		output.WriteString(piece)

		if req.Stream && onToken != nil {
			onToken(piece)
		}

		genTokens := []int32{token}
		if err := bridgeDecode(cg.ctx, genTokens); err != nil {
			return nil, err
		}

		generated++
	}

	elapsed := time.Since(start)
	response := output.String()

	return &GenerateResp{
		Response:        response,
		Done:            true,
		TotalDuration:   elapsed.Microseconds(),
		LoadDuration:    0,
		PromptEvalCount: nPromptTokens,
		EvalCount:       generated,
		EvalDuration:    elapsed.Microseconds(),
	}, nil
}

func (c *CgoBackend) Close() error {
	return c.Unload()
}
