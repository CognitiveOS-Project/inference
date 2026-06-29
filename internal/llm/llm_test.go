package llm

import (
	"testing"
)

func TestMockBackendLoad(t *testing.T) {
	b := NewMockBackend()
	if b.IsLoaded() {
		t.Fatal("expected not loaded initially")
	}

	info, err := b.Load("/cognitiveos/models/test.gguf", nil)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if info == nil {
		t.Fatal("expected model info")
	}
	if !b.IsLoaded() {
		t.Fatal("expected loaded after load")
	}
	if b.LoadedModel() == nil {
		t.Fatal("expected LoadedModel to return info")
	}
}

func TestMockBackendUnload(t *testing.T) {
	b := NewMockBackend()
	b.Load("/cognitiveos/models/test.gguf", nil)
	if err := b.Unload(); err != nil {
		t.Fatalf("unload: %v", err)
	}
	if b.IsLoaded() {
		t.Fatal("expected not loaded after unload")
	}
}

func TestMockBackendClose(t *testing.T) {
	b := NewMockBackend()
	b.Load("/cognitiveos/models/test.gguf", nil)
	if err := b.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if b.IsLoaded() {
		t.Fatal("expected not loaded after close")
	}
}

func TestMockBackendGenerateWithoutLoad(t *testing.T) {
	b := NewMockBackend()
	_, err := b.Generate(GenerateReq{Prompt: "hello"}, nil)
	if err == nil {
		t.Fatal("expected error when generating without loading a model")
	}
}

func TestMockBackendGenerate(t *testing.T) {
	b := NewMockBackend()
	b.Load("/cognitiveos/models/test.gguf", nil)

	resp, err := b.Generate(GenerateReq{Prompt: "What is 2+2?"}, nil)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if !resp.Done {
		t.Fatal("expected done")
	}
	if resp.Response == "" {
		t.Fatal("expected non-empty response")
	}
	if resp.EvalCount == 0 {
		t.Fatal("expected non-zero eval count")
	}
	if resp.TotalDuration == 0 {
		t.Fatal("expected non-zero duration")
	}
}

func TestMockBackendGenerateStreaming(t *testing.T) {
	b := NewMockBackend()
	b.Load("/cognitiveos/models/test.gguf", nil)

	var tokens []string
	onToken := func(tok string) {
		tokens = append(tokens, tok)
	}

	resp, err := b.Generate(GenerateReq{Prompt: "Hello", Stream: true}, onToken)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if !resp.Done {
		t.Fatal("expected done")
	}
	if len(tokens) == 0 {
		t.Fatal("expected token callbacks")
	}
}

func TestMockBackendName(t *testing.T) {
	b := NewMockBackend()
	if b.Name() != "mock" {
		t.Fatalf("expected 'mock', got %q", b.Name())
	}
}
