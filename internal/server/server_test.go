package server

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type testResp map[string]interface{}

func startTestServer(t *testing.T) (*httptest.Server, string) {
	t.Helper()
	dir := t.TempDir()
	modelDir := filepath.Join(dir, "models")
	_ = os.MkdirAll(filepath.Join(modelDir, "wide", "active"), 0755)
	_ = os.WriteFile(filepath.Join(modelDir, "test-model.gguf"), []byte("dummy-model-data"), 0644)

	s := New(modelDir, "mock")
	mux := http.NewServeMux()
	mux.HandleFunc("/api/generate", s.handleGenerate)
	mux.HandleFunc("/api/chat", s.handleChat)
	mux.HandleFunc("/api/tags", s.handleTags)
	mux.HandleFunc("/api/pull", s.handlePull)
	mux.HandleFunc("/api/ps", s.handlePs)
	mux.HandleFunc("/api/delete", s.handleDelete)
	mux.HandleFunc("/api/negotiate", s.handleNegotiate)
	mux.HandleFunc("/cognitiveos/status", s.handleStatus)
	mux.HandleFunc("/cognitiveos/capabilities", s.handleCapabilities)
	mux.HandleFunc("/health", s.handleHealth)
	return httptest.NewServer(mux), modelDir
}

func request(t *testing.T, url, method string, body interface{}) *http.Response {
	t.Helper()
	var buf bytes.Buffer
	if body != nil {
		_ = json.NewEncoder(&buf).Encode(body)
	}
	req, err := http.NewRequest(method, url, &buf)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("do request: %v", err)
	}
	return resp
}

func bodyString(t *testing.T, resp *http.Response) string {
	t.Helper()
	defer func() { _ = resp.Body.Close() }()
	b, _ := io.ReadAll(resp.Body)
	return strings.TrimSpace(string(b))
}

func decodeMap(t *testing.T, resp *http.Response) testResp {
	t.Helper()
	var r testResp
	if err := json.NewDecoder(resp.Body).Decode(&r); err != nil {
		t.Fatalf("decode: %v (body: %s)", err, bodyString(t, resp))
	}
	_ = resp.Body.Close()
	return r
}

func TestHealth(t *testing.T) {
	ts, _ := startTestServer(t)
	defer ts.Close()

	resp := request(t, ts.URL+"/health", "GET", nil)
	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d: %s", resp.StatusCode, bodyString(t, resp))
	}
	_ = resp.Body.Close()
}

func TestStatus(t *testing.T) {
	ts, _ := startTestServer(t)
	defer ts.Close()

	resp := request(t, ts.URL+"/cognitiveos/status", "GET", nil)
	r := decodeMap(t, resp)
	if r["status"] == nil {
		t.Fatal("expected status field")
	}
}

func TestCapabilities(t *testing.T) {
	ts, _ := startTestServer(t)
	defer ts.Close()

	resp := request(t, ts.URL+"/cognitiveos/capabilities", "GET", nil)
	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d: %s", resp.StatusCode, bodyString(t, resp))
	}
	_ = resp.Body.Close()
}

func TestTags(t *testing.T) {
	ts, dir := startTestServer(t)
	defer ts.Close()

	_ = os.WriteFile(filepath.Join(dir, "raw-model.gguf"), []byte("data"), 0644)

	resp := request(t, ts.URL+"/api/tags", "GET", nil)
	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d: %s", resp.StatusCode, bodyString(t, resp))
	}
	r := decodeMap(t, resp)
	models, ok := r["models"].([]interface{})
	if !ok {
		t.Fatalf("expected models array, got %T: %v", r["models"], r)
	}
	if len(models) == 0 {
		t.Fatal("expected at least 1 model")
	}
}

func TestGenerate(t *testing.T) {
	ts, _ := startTestServer(t)
	defer ts.Close()

	resp := request(t, ts.URL+"/api/generate", "POST", map[string]interface{}{
		"model":  "test-model",
		"prompt": "Hello",
	})
	r := decodeMap(t, resp)
	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d: %v", resp.StatusCode, r)
	}
	if r["response"] == nil {
		t.Fatal("expected response field")
	}
}

func TestGenerateWithoutPrompt(t *testing.T) {
	ts, _ := startTestServer(t)
	defer ts.Close()

	resp := request(t, ts.URL+"/api/generate", "POST", map[string]interface{}{
		"model": "test-model",
	})
	if resp.StatusCode == 200 {
		t.Fatal("expected error for missing prompt")
	}
	_ = resp.Body.Close()
}

func TestChat(t *testing.T) {
	ts, _ := startTestServer(t)
	defer ts.Close()

	resp := request(t, ts.URL+"/api/chat", "POST", map[string]interface{}{
		"model": "test-model",
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "Hello"},
		},
	})
	r := decodeMap(t, resp)
	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d: %v", resp.StatusCode, r)
	}
	if r["response"] == nil {
		t.Fatal("expected response field")
	}
}

func TestChatWithoutMessages(t *testing.T) {
	ts, _ := startTestServer(t)
	defer ts.Close()

	resp := request(t, ts.URL+"/api/chat", "POST", map[string]interface{}{
		"model": "test-model",
	})
	if resp.StatusCode == 200 {
		t.Fatal("expected error for no messages")
	}
	_ = resp.Body.Close()
}

func TestPs(t *testing.T) {
	ts, _ := startTestServer(t)
	defer ts.Close()

	resp := request(t, ts.URL+"/api/ps", "GET", nil)
	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d: %s", resp.StatusCode, bodyString(t, resp))
	}
	_ = resp.Body.Close()
}

func TestDelete(t *testing.T) {
	ts, _ := startTestServer(t)
	defer ts.Close()

	resp := request(t, ts.URL+"/api/delete", "DELETE", nil)
	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d: %s", resp.StatusCode, bodyString(t, resp))
	}
	_ = resp.Body.Close()
}

func TestMethodNotAllowed(t *testing.T) {
	ts, _ := startTestServer(t)
	defer ts.Close()

	resp := request(t, ts.URL+"/api/generate", "GET", nil)
	if resp.StatusCode != 405 {
		t.Fatalf("expected 405, got %d: %s", resp.StatusCode, bodyString(t, resp))
	}
	_ = resp.Body.Close()
}

func TestGenerateStream(t *testing.T) {
	ts, _ := startTestServer(t)
	defer ts.Close()

	body := map[string]interface{}{
		"model":  "test-model",
		"prompt": "Hello",
		"stream": true,
	}
	var buf bytes.Buffer
	_ = json.NewEncoder(&buf).Encode(body)

	req, err := http.NewRequest("POST", ts.URL+"/api/generate", &buf)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("do request: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d: %s", resp.StatusCode, bodyString(t, resp))
	}
	ct := resp.Header.Get("Content-Type")
	if ct != "application/x-ndjson" {
		t.Fatalf("expected ndjson, got %q", ct)
	}
}

func TestNegotiate(t *testing.T) {
	ts, dir := startTestServer(t)
	defer ts.Close()

	modelPath := filepath.Join(dir, "test-model.gguf")
	resp := request(t, ts.URL+"/api/negotiate", "POST", map[string]interface{}{
		"model_path": modelPath,
	})
	r := decodeMap(t, resp)
	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d: %v", resp.StatusCode, r)
	}
	if r["status"] != "ok" {
		t.Fatalf("expected status ok, got %v", r)
	}
}

func TestPullLocalFile(t *testing.T) {
	ts, dir := startTestServer(t)
	defer ts.Close()

	modelPath := filepath.Join(dir, "existing-model.gguf")
	_ = os.WriteFile(modelPath, []byte("data"), 0644)

	resp := request(t, ts.URL+"/api/pull", "POST", map[string]interface{}{
		"name": "existing-model",
		"path": modelPath,
	})
	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d: %s", resp.StatusCode, bodyString(t, resp))
	}
	_ = resp.Body.Close()
}
