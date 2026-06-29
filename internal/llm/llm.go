package llm

import (
	"fmt"
	"math/rand"
	"strings"
	"sync"
	"time"
)

type State string

const (
	StateUnloaded  State = "unloaded"
	StateLoading   State = "loading"
	StateReady     State = "ready"
	StateUnloading State = "unloading"
	StateError     State = "error"
)

type ModelInfo struct {
	Name            string    `json:"name"`
	Path            string    `json:"path"`
	Quantization    string    `json:"quantization,omitempty"`
	RAMUsageMB      int64     `json:"ram_usage_mb"`
	VRAMUsageMB     int64     `json:"vram_usage_mb,omitempty"`
	ContextWindow   int       `json:"context_window"`
	ContextUsed     int       `json:"context_used"`
	TokensPerSecond float64   `json:"tokens_per_second"`
	State           State     `json:"-"`
	LoadedAt        time.Time `json:"-"`
}

type GenerateReq struct {
	Model   string                 `json:"model"`
	Prompt  string                 `json:"prompt"`
	System  string                 `json:"system,omitempty"`
	Options map[string]interface{} `json:"options,omitempty"`
	Stream  bool                   `json:"stream"`
}

type GenerateResp struct {
	Response        string `json:"response"`
	Done            bool   `json:"done"`
	Context         []int  `json:"context,omitempty"`
	TotalDuration   int64  `json:"total_duration,omitempty"`
	LoadDuration    int64  `json:"load_duration,omitempty"`
	PromptEvalCount int    `json:"prompt_eval_count,omitempty"`
	EvalCount       int    `json:"eval_count,omitempty"`
	EvalDuration    int64  `json:"eval_duration,omitempty"`
}

type ChatMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Backend interface {
	Name() string
	Generate(req GenerateReq, onToken func(string)) (*GenerateResp, error)
	Load(modelPath string, opts *LoadOptions) (*ModelInfo, error)
	Unload() error
	IsLoaded() bool
	LoadedModel() *ModelInfo
	Close() error
}

type MockBackend struct {
	mu        sync.Mutex
	loaded    bool
	modelPath string
	modelInfo *ModelInfo
}

func NewMockBackend() *MockBackend {
	return &MockBackend{}
}

func (m *MockBackend) Name() string { return "mock" }

func (m *MockBackend) Load(modelPath string, opts *LoadOptions) (*ModelInfo, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.loaded = true
	m.modelPath = modelPath
	nCtx := 8192
	if opts != nil && opts.NumCtx > 0 {
		nCtx = opts.NumCtx
	}
	m.modelInfo = &ModelInfo{
		Name:          modelPath,
		Path:          modelPath,
		RAMUsageMB:    128,
		ContextWindow: nCtx,
		State:         StateReady,
		LoadedAt:      time.Now(),
	}
	return m.modelInfo, nil
}

func (m *MockBackend) Unload() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.loaded = false
	m.modelPath = ""
	m.modelInfo = nil
	return nil
}

func (m *MockBackend) IsLoaded() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.loaded
}

func (m *MockBackend) LoadedModel() *ModelInfo {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.modelInfo
}

func (m *MockBackend) Generate(req GenerateReq, onToken func(string)) (*GenerateResp, error) {
	m.mu.Lock()
	loaded := m.loaded
	m.mu.Unlock()

	if !loaded {
		return nil, fmt.Errorf("E_MODEL_NOT_FOUND: no model loaded")
	}

	respText := fmt.Sprintf(
		"I understand your request: %s\n\nAs CognitiveOS AI, I can process this. (Mock mode -- real inference would use llama.cpp with an actual GGUF model.)",
		req.Prompt,
	)

	if req.Stream && onToken != nil {
		for _, ch := range respText {
			onToken(string(ch))
			time.Sleep(time.Duration(5+rand.Intn(15)) * time.Millisecond)
		}
	}

	evalCount := len(strings.Fields(respText))
	evalDur := int64(evalCount) * 50000000

	return &GenerateResp{
		Response:        respText,
		Done:            true,
		Context:         []int{1, 2, 3},
		TotalDuration:   evalDur + 100000000,
		LoadDuration:    50000000,
		PromptEvalCount: len(strings.Fields(req.Prompt)),
		EvalCount:       evalCount,
		EvalDuration:    evalDur,
	}, nil
}

func (m *MockBackend) Close() error { return m.Unload() }
