package cloud

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/CognitiveOS-Project/inference/internal/llm"
)

func newTestConfig() *Config {
	return &Config{
		APIBase: "https://api.openai.com",
		ModelID: "gpt-4o-mini",
		APIKey:  "sk-test-key",
	}
}

func TestNew(t *testing.T) {
	b := New(newTestConfig())
	if b.Name() != "cloud" {
		t.Fatalf("expected Name() == 'cloud', got %q", b.Name())
	}
	if b.client == nil {
		t.Fatal("expected non-nil HTTP client")
	}
}

func TestLoadNilConfig(t *testing.T) {
	b := New(nil)
	_, err := b.Load("ignored", llm.DefaultLoadOptions())
	if err == nil {
		t.Fatal("expected error for nil config")
	}
	if err.Error() != "E_CLOUD_NOT_CONFIGURED: cloud backend requires --api-base, --cloud-model, and --api-key" {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestLoadEmptyAPIBase(t *testing.T) {
	b := New(&Config{APIBase: "", ModelID: "gpt-4", APIKey: "key"})
	_, err := b.Load("ignored", llm.DefaultLoadOptions())
	if err == nil {
		t.Fatal("expected error for empty APIBase")
	}
}

func TestLoadSuccess(t *testing.T) {
	b := New(newTestConfig())
	mi, err := b.Load("ignored", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mi.Name != "gpt-4o-mini" {
		t.Fatalf("expected Name 'gpt-4o-mini', got %q", mi.Name)
	}
	if mi.Path != "https://api.openai.com/v1/chat/completions" {
		t.Fatalf("unexpected path: %s", mi.Path)
	}
	if mi.ContextWindow != 4096 {
		t.Fatalf("expected ContextWindow 4096, got %d", mi.ContextWindow)
	}
	if mi.State != llm.StateReady {
		t.Fatalf("expected StateReady, got %v", mi.State)
	}
}

func TestLoadCustomContextWindow(t *testing.T) {
	b := New(newTestConfig())
	opts := &llm.LoadOptions{NumCtx: 8192}
	mi, err := b.Load("ignored", opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mi.ContextWindow != 8192 {
		t.Fatalf("expected ContextWindow 8192, got %d", mi.ContextWindow)
	}
}

func TestLifecycle(t *testing.T) {
	b := New(newTestConfig())
	if b.IsLoaded() {
		t.Fatal("should not be loaded before Load")
	}
	if b.LoadedModel() != nil {
		t.Fatal("LoadedModel should be nil before Load")
	}
	mi, err := b.Load("ignored", llm.DefaultLoadOptions())
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	if !b.IsLoaded() {
		t.Fatal("should be loaded after Load")
	}
	if b.LoadedModel() != mi {
		t.Fatal("LoadedModel mismatch")
	}
	if err := b.Unload(); err != nil {
		t.Fatalf("Unload failed: %v", err)
	}
	if b.IsLoaded() {
		t.Fatal("should not be loaded after Unload")
	}
	if b.LoadedModel() != nil {
		t.Fatal("LoadedModel should be nil after Unload")
	}
}

func TestDoubleUnload(t *testing.T) {
	b := New(newTestConfig())
	b.Load("ignored", llm.DefaultLoadOptions())
	if err := b.Unload(); err != nil {
		t.Fatalf("first Unload failed: %v", err)
	}
	if err := b.Unload(); err != nil {
		t.Fatalf("second Unload should be safe: %v", err)
	}
}

func TestClose(t *testing.T) {
	b := New(newTestConfig())
	b.Load("ignored", llm.DefaultLoadOptions())
	if err := b.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}
	if b.IsLoaded() {
		t.Fatal("should not be loaded after Close")
	}
}

func TestLoadAdapterUnsupported(t *testing.T) {
	b := New(newTestConfig())
	err := b.LoadAdapter("/some/adapter")
	if err == nil {
		t.Fatal("expected error for LoadAdapter")
	}
	if err.Error() != "E_UNSUPPORTED: adapters not supported for cloud backend" {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestGenerateNotLoaded(t *testing.T) {
	b := New(newTestConfig())
	_, err := b.Generate(llm.GenerateReq{Prompt: "hi", Stream: false}, nil)
	if err == nil {
		t.Fatal("expected error when not loaded")
	}
}

func TestGenerateSyncSuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Fatalf("expected POST, got %s", r.Method)
		}
		if r.Header.Get("Authorization") != "Bearer sk-test-key" {
			t.Fatalf("bad auth header: %s", r.Header.Get("Authorization"))
		}
		var req chatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if req.Model != "gpt-4o-mini" {
			t.Fatalf("expected model 'gpt-4o-mini', got %q", req.Model)
		}
		if !req.Stream {
			resp := chatResponse{}
			json.Unmarshal([]byte(`[{"message":{"content":"Hello, world!"},"finish_reason":"stop"}]`), &resp.Choices)
			resp.Usage.PromptTokens = 10
			resp.Usage.CompletionTokens = 3
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(resp)
		}
	}))
	defer srv.Close()

	cfg := &Config{APIBase: srv.URL, ModelID: "gpt-4o-mini", APIKey: "sk-test-key"}
	b := New(cfg)
	b.Load("ignored", llm.DefaultLoadOptions())

	result, err := b.Generate(llm.GenerateReq{Prompt: "test", Stream: false}, nil)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if result.Response != "Hello, world!" {
		t.Fatalf("expected 'Hello, world!', got %q", result.Response)
	}
	if !result.Done {
		t.Fatal("expected Done true")
	}
	if result.PromptEvalCount != 10 {
		t.Fatalf("expected PromptEvalCount 10, got %d", result.PromptEvalCount)
	}
	if result.EvalCount != 3 {
		t.Fatalf("expected EvalCount 3, got %d", result.EvalCount)
	}
}

func TestGenerateSyncWithSystemPrompt(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req chatRequest
		json.NewDecoder(r.Body).Decode(&req)
		if len(req.Messages) != 2 {
			t.Fatalf("expected 2 messages (system+user), got %d", len(req.Messages))
		}
		if req.Messages[0].Role != "system" {
			t.Fatalf("expected system role, got %q", req.Messages[0].Role)
		}
		resp := chatResponse{}
		json.Unmarshal([]byte(`[{"message":{"content":"ok"},"finish_reason":"stop"}]`), &resp.Choices)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	cfg := &Config{APIBase: srv.URL, ModelID: "gpt-4", APIKey: "key"}
	b := New(cfg)
	b.Load("ignored", llm.DefaultLoadOptions())
	_, err := b.Generate(llm.GenerateReq{Prompt: "hello", System: "be concise", Stream: false}, nil)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
}

func TestGenerateSyncHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		w.Write([]byte("rate limited"))
	}))
	defer srv.Close()

	cfg := &Config{APIBase: srv.URL, ModelID: "gpt-4", APIKey: "key"}
	b := New(cfg)
	b.Load("ignored", llm.DefaultLoadOptions())
	_, err := b.Generate(llm.GenerateReq{Prompt: "hi", Stream: false}, nil)
	if err == nil {
		t.Fatal("expected error for HTTP 429")
	}
}

func TestGenerateSyncEmptyChoices(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := chatResponse{}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	cfg := &Config{APIBase: srv.URL, ModelID: "gpt-4", APIKey: "key"}
	b := New(cfg)
	b.Load("ignored", llm.DefaultLoadOptions())
	_, err := b.Generate(llm.GenerateReq{Prompt: "hi", Stream: false}, nil)
	if err == nil || err.Error() != "cloud API returned no choices" {
		t.Fatalf("expected 'no choices' error, got %v", err)
	}
}

func TestGenerateSyncMalformedJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{malformed`))
	}))
	defer srv.Close()

	cfg := &Config{APIBase: srv.URL, ModelID: "gpt-4", APIKey: "key"}
	b := New(cfg)
	b.Load("ignored", llm.DefaultLoadOptions())
	_, err := b.Generate(llm.GenerateReq{Prompt: "hi", Stream: false}, nil)
	if err == nil {
		t.Fatal("expected decode error")
	}
}

func TestGenerateStreamSuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req chatRequest
		json.NewDecoder(r.Body).Decode(&req)
		if req.Stream != true {
			t.Fatal("expected stream=true in request")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("response writer does not support flush")
		}
		for _, token := range []string{"Hello", ", ", "world", "!"} {
			chunk := fmt.Sprintf(`data: {"choices":[{"delta":{"content":"%s"},"finish_reason":null}]}`+"\n\n", token)
			fmt.Fprint(w, chunk)
			flusher.Flush()
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	cfg := &Config{APIBase: srv.URL, ModelID: "gpt-4", APIKey: "key"}
	b := New(cfg)
	b.Load("ignored", llm.DefaultLoadOptions())

	var tokens []string
	result, err := b.Generate(llm.GenerateReq{Prompt: "test", Stream: true}, func(tok string) {
		tokens = append(tokens, tok)
	})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if result.Response != "Hello, world!" {
		t.Fatalf("expected 'Hello, world!', got %q", result.Response)
	}
	if !result.Done {
		t.Fatal("expected Done true")
	}
	if len(tokens) != 4 {
		t.Fatalf("expected 4 tokens, got %d: %v", len(tokens), tokens)
	}
}

func TestGenerateStreamHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte("bad key"))
	}))
	defer srv.Close()

	cfg := &Config{APIBase: srv.URL, ModelID: "gpt-4", APIKey: "bad"}
	b := New(cfg)
	b.Load("ignored", llm.DefaultLoadOptions())
	_, err := b.Generate(llm.GenerateReq{Prompt: "hi", Stream: true}, nil)
	if err == nil {
		t.Fatal("expected error for HTTP 401")
	}
}

func TestGenerateStreamImmediateDone(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("response writer does not support flush")
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	cfg := &Config{APIBase: srv.URL, ModelID: "gpt-4", APIKey: "key"}
	b := New(cfg)
	b.Load("ignored", llm.DefaultLoadOptions())

	result, err := b.Generate(llm.GenerateReq{Prompt: "test", Stream: true}, nil)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if result.Response != "" {
		t.Fatalf("expected empty response, got %q", result.Response)
	}
}

func TestCloudEndpointURLTrimSlash(t *testing.T) {
	cfg := &Config{APIBase: "https://api.openai.com/", ModelID: "gpt-4", APIKey: "key"}
	b := New(cfg)
	b.Load("ignored", llm.DefaultLoadOptions())
	if b.modelInfo.Path != "https://api.openai.com//v1/chat/completions" {
		t.Fatalf("unexpected path: %s", b.modelInfo.Path)
	}
}
