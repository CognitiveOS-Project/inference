package cloud

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/CognitiveOS-Project/inference/internal/llm"
)

type Config struct {
	APIBase string // e.g. "https://api.openai.com"
	ModelID string // e.g. "gpt-4o-mini"
	APIKey  string // Bearer token
}

type CloudBackend struct {
	mu       sync.Mutex
	config   *Config
	loaded   bool
	modelInfo *llm.ModelInfo
	client   *http.Client
}

func New(cfg *Config) *CloudBackend {
	return &CloudBackend{
		config: cfg,
		client: &http.Client{Timeout: 120 * time.Second},
	}
}

func (c *CloudBackend) Name() string { return "cloud" }

func (c *CloudBackend) Load(modelPath string, opts *llm.LoadOptions) (*llm.ModelInfo, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.config == nil || c.config.APIBase == "" {
		return nil, fmt.Errorf("E_CLOUD_NOT_CONFIGURED: cloud backend requires --api-base, --cloud-model, and --api-key")
	}

	nCtx := 4096
	if opts != nil && opts.NumCtx > 0 {
		nCtx = opts.NumCtx
	}

	c.loaded = true
	c.modelInfo = &llm.ModelInfo{
		Name:          c.config.ModelID,
		Path:          c.config.APIBase + "/v1/chat/completions",
		ContextWindow: nCtx,
		State:         llm.StateReady,
		LoadedAt:      time.Now(),
	}

	return c.modelInfo, nil
}

func (c *CloudBackend) Unload() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.loaded = false
	c.modelInfo = nil
	return nil
}

func (c *CloudBackend) IsLoaded() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.loaded
}

func (c *CloudBackend) LoadedModel() *llm.ModelInfo {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.modelInfo
}

func (c *CloudBackend) LoadAdapter(adapterPath string) error {
	return fmt.Errorf("E_UNSUPPORTED: adapters not supported for cloud backend")
}

func (c *CloudBackend) Close() error {
	return c.Unload()
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
	Stream   bool          `json:"stream"`
}

type chatResponse struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		TotalTokens int `json:"total_tokens"`
		PromptTokens int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

func (c *CloudBackend) Generate(req llm.GenerateReq, onToken func(string)) (*llm.GenerateResp, error) {
	c.mu.Lock()
	if !c.loaded {
		c.mu.Unlock()
		return nil, fmt.Errorf("E_MODEL_NOT_FOUND: no cloud model configured")
	}
	cfg := c.config
	c.mu.Unlock()

	var messages []chatMessage
	if req.System != "" {
		messages = append(messages, chatMessage{Role: "system", Content: req.System})
	}
	messages = append(messages, chatMessage{Role: "user", Content: req.Prompt})

	apiURL := strings.TrimRight(cfg.APIBase, "/") + "/v1/chat/completions"

	body, err := json.Marshal(chatRequest{
		Model:    cfg.ModelID,
		Messages: messages,
		Stream:   req.Stream,
	})
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequest("POST", apiURL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+cfg.APIKey)

	if req.Stream {
		return c.generateStream(httpReq, onToken)
	}
	return c.generateSync(httpReq)
}

func (c *CloudBackend) generateSync(httpReq *http.Request) (*llm.GenerateResp, error) {
	resp, err := c.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("cloud API error: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("cloud API returned %d: %s", resp.StatusCode, string(body))
	}

	var chatResp chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	if len(chatResp.Choices) == 0 {
		return nil, fmt.Errorf("cloud API returned no choices")
	}

	text := chatResp.Choices[0].Message.Content
	tokens := len(strings.Fields(text))

	return &llm.GenerateResp{
		Response:        text,
		Done:            true,
		PromptEvalCount: chatResp.Usage.PromptTokens,
		EvalCount:       chatResp.Usage.CompletionTokens,
		EvalDuration:    int64(tokens) * 50000000,
	}, nil
}

func (c *CloudBackend) generateStream(httpReq *http.Request, onToken func(string)) (*llm.GenerateResp, error) {
	resp, err := c.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("cloud API error: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("cloud API returned %d: %s", resp.StatusCode, string(body))
	}

	var fullResponse strings.Builder
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var chunk chatResponse
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		if len(chunk.Choices) == 0 {
			continue
		}

		token := chunk.Choices[0].Delta.Content
		if token != "" {
			fullResponse.WriteString(token)
			if onToken != nil {
				onToken(token)
			}
		}
	}

	text := fullResponse.String()
	tokens := len(strings.Fields(text))

	return &llm.GenerateResp{
		Response:  text,
		Done:      true,
		EvalCount: tokens,
	}, nil
}
