package server

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/CognitiveOS-Project/inference/internal/llm"
	"github.com/CognitiveOS-Project/inference/internal/model"
)

type Server struct {
	models    *model.Manager
	backend   llm.Backend
	startTime time.Time
	mu        sync.Mutex

	backendType string
}

type StatusResponse struct {
	Status       string         `json:"status"`
	ModelsLoaded int            `json:"models_loaded"`
	ActiveModel  *llm.ModelInfo `json:"active_model,omitempty"`
	RawModel     *rawModelInfo  `json:"raw_model,omitempty"`
	Hardware     *hardwareInfo  `json:"hardware"`
}

type rawModelInfo struct {
	Loaded     bool   `json:"loaded"`
	Path       string `json:"path,omitempty"`
	RAMUsageMB int64  `json:"ram_usage_mb,omitempty"`
}

type hardwareInfo struct {
	TotalRAMMB     int64 `json:"total_ram_mb"`
	AvailableRAMMB int64 `json:"available_ram_mb"`
	CPUCores       int   `json:"cpu_cores"`
	CPUThreads     int   `json:"cpu_threads"`
}

type capabilitiesResponse struct {
	Backends         map[string]bool `json:"backends"`
	PreferredBackend string          `json:"preferred_backend"`
	MaxModelSizeMB   int64           `json:"max_model_size_mb"`
	SupportedQuants  []string        `json:"supported_quantizations"`
	MaxContextWindow int             `json:"max_context_window"`
}

type negotiateRequest struct {
	ModelPath string                 `json:"model_path"`
	Params    map[string]interface{} `json:"params,omitempty"`
}

type negotiateResponse struct {
	Status    string          `json:"status"`
	ModelInfo *llm.ModelInfo  `json:"model_info,omitempty"`
	Error     *negotiateError `json:"error,omitempty"`
	Alts      []negotiateAlt  `json:"alternatives,omitempty"`
}

type negotiateError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type negotiateAlt struct {
	Model                  string  `json:"model"`
	RAMUsageMB             int64   `json:"ram_usage_mb"`
	PerformanceTokensPerS  float64 `json:"performance_tokens_per_second"`
}

type generateRequest struct {
	Model   string                 `json:"model"`
	Prompt  string                 `json:"prompt"`
	System  string                 `json:"system,omitempty"`
	Options map[string]interface{} `json:"options,omitempty"`
	Stream  bool                   `json:"stream"`
}

type chatRequest struct {
	Model    string          `json:"model"`
	Messages []llm.ChatMsg   `json:"messages"`
	Stream   bool            `json:"stream"`
	Options  map[string]interface{} `json:"options,omitempty"`
}

type pullRequest struct {
	Name     string `json:"name"`
	Path     string `json:"path"`
	Insecure bool   `json:"insecure"`
}

type deleteRequest struct {
	Model string `json:"model"`
}

type apiError struct {
	Error string `json:"error"`
}

type healthResponse struct {
	Status          string `json:"status"`
	UptimeSeconds   int64  `json:"uptime_seconds"`
	ModelsLoaded    int    `json:"models_loaded"`
	RAMUsagePercent int    `json:"ram_usage_percent"`
}

func New(modelDir string, backendType string) *Server {
	var b llm.Backend
	switch backendType {
	case "cgo":
		b = newCgoBackend()
	case "mock":
		b = llm.NewMockBackend()
	default:
		b = llm.NewMockBackend()
	}

	return &Server{
		models:      model.NewManager(modelDir),
		backend:     b,
		startTime:   time.Now(),
		backendType: backendType,
	}
}

func (s *Server) Listen(addr string) error {
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

	log.Printf("coginfer listening on %s (backend=%s)", addr, s.backendType)
	return http.ListenAndServe(addr, mux)
}

func sendJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("sendJSON encode error: %v", err)
	}
}

func sendError(w http.ResponseWriter, status int, msg string) {
	sendJSON(w, status, apiError{Error: msg})
}

func loadOptionsFromParams(params map[string]interface{}) *llm.LoadOptions {
	if len(params) == 0 {
		return nil
	}
	opts := llm.DefaultLoadOptions()
	if v, ok := params["num_ctx"].(float64); ok {
		opts.NumCtx = int(v)
	}
	if v, ok := params["gpu_layers"].(float64); ok {
		opts.GPULayers = int(v)
	}
	if v, ok := params["threads"].(float64); ok {
		opts.Threads = int(v)
	}
	return opts
}

// --- Handlers ---

func (s *Server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		sendError(w, 405, "method not allowed")
		return
	}

	var req generateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		sendError(w, 400, "E_INVALID_PARAMS: "+err.Error())
		return
	}
	if req.Prompt == "" {
		sendError(w, 400, "E_INVALID_PARAMS: prompt is required")
		return
	}

	s.mu.Lock()
	if !s.backend.IsLoaded() {
		modelPath, err := s.models.Resolve(req.Model)
		if err != nil {
			s.mu.Unlock()
			sendError(w, 404, err.Error())
			return
		}
		loadOpts := loadOptionsFromParams(req.Options)
		if _, err := s.backend.Load(modelPath, loadOpts); err != nil {
			s.mu.Unlock()
			sendError(w, 500, err.Error())
			return
		}
	}
	s.mu.Unlock()

	llmReq := llm.GenerateReq{
		Model:   req.Model,
		Prompt:  req.Prompt,
		System:  req.System,
		Options: req.Options,
		Stream:  req.Stream,
	}

	if req.Stream {
		flusher, ok := w.(http.Flusher)
		if !ok {
			sendError(w, 500, "E_INTERNAL: streaming not supported")
			return
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.WriteHeader(200)

		onToken := func(token string) {
			line, _ := json.Marshal(llm.GenerateResp{Response: token, Done: false})
			_, _ = fmt.Fprintf(w, "%s\n", line)
			flusher.Flush()
		}

		resp, err := s.backend.Generate(llmReq, onToken)
		if err != nil {
			sendError(w, 500, err.Error())
			return
		}
		resp.Done = true
		line, _ := json.Marshal(resp)
		_, _ = fmt.Fprintf(w, "%s\n", line)
		flusher.Flush()
		return
	}

	resp, err := s.backend.Generate(llmReq, nil)
	if err != nil {
		sendError(w, 500, err.Error())
		return
	}
	sendJSON(w, 200, resp)
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		sendError(w, 405, "method not allowed")
		return
	}

	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		sendError(w, 400, "E_INVALID_PARAMS: "+err.Error())
		return
	}
	if len(req.Messages) == 0 {
		sendError(w, 400, "E_INVALID_PARAMS: messages is required")
		return
	}

	var promptBuilder strings.Builder
	for _, msg := range req.Messages {
		fmt.Fprintf(&promptBuilder, "%s: %s\n", msg.Role, msg.Content)
	}
	prompt := promptBuilder.String()

	s.mu.Lock()
	if !s.backend.IsLoaded() {
		modelPath, err := s.models.Resolve(req.Model)
		if err != nil {
			s.mu.Unlock()
			sendError(w, 404, err.Error())
			return
		}
		loadOpts := loadOptionsFromParams(req.Options)
		if _, err := s.backend.Load(modelPath, loadOpts); err != nil {
			s.mu.Unlock()
			sendError(w, 500, err.Error())
			return
		}
	}
	s.mu.Unlock()

	llmReq := llm.GenerateReq{
		Model:   req.Model,
		Prompt:  prompt,
		Options: req.Options,
		Stream:  req.Stream,
	}

	if req.Stream {
		flusher, ok := w.(http.Flusher)
		if !ok {
			sendError(w, 500, "E_INTERNAL: streaming not supported")
			return
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.WriteHeader(200)

		onToken := func(token string) {
			line, _ := json.Marshal(llm.GenerateResp{Response: token, Done: false})
			_, _ = fmt.Fprintf(w, "%s\n", line)
			flusher.Flush()
		}

		resp, err := s.backend.Generate(llmReq, onToken)
		if err != nil {
			sendError(w, 500, err.Error())
			return
		}
		resp.Done = true
		line, _ := json.Marshal(resp)
		_, _ = fmt.Fprintf(w, "%s\n", line)
		flusher.Flush()
		return
	}

	resp, err := s.backend.Generate(llmReq, nil)
	if err != nil {
		sendError(w, 500, err.Error())
		return
	}
	sendJSON(w, 200, resp)
}

func (s *Server) handleTags(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		sendError(w, 405, "method not allowed")
		return
	}

	entries, err := s.models.List()
	if err != nil {
		sendError(w, 500, err.Error())
		return
	}

	type tagModel struct {
		Name       string `json:"name"`
		ModifiedAt string `json:"modified_at"`
		Size       int64  `json:"size"`
	}

	var models []tagModel
	for _, e := range entries {
		models = append(models, tagModel{
			Name:       e.Name,
			ModifiedAt: e.ModifiedAt,
			Size:       e.Size,
		})
	}

	sendJSON(w, 200, map[string]interface{}{"models": models})
}

func (s *Server) handlePull(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		sendError(w, 405, "method not allowed")
		return
	}

	var req pullRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		sendError(w, 400, "E_INVALID_PARAMS: "+err.Error())
		return
	}

	if req.Path != "" {
		if _, err := os.Stat(req.Path); err == nil {
			s.mu.Lock()
			mi, err := s.backend.Load(req.Path, nil)
			s.mu.Unlock()
			if err != nil {
				sendError(w, 500, err.Error())
				return
			}
			sendJSON(w, 200, map[string]interface{}{
				"status": "success",
				"digest": "sha256:file-exists",
				"size":   mi.RAMUsageMB * 1024 * 1024,
			})
			return
		}
	}

	sendError(w, 501, "E_INTERNAL: remote pull not implemented yet")
}

func (s *Server) handlePs(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		sendError(w, 405, "method not allowed")
		return
	}

	type psModel struct {
		Name            string  `json:"name"`
		Size            int64   `json:"size"`
		RAMUsageMB      int64   `json:"ram_usage_mb"`
		VRAMUsageMB     int64   `json:"vram_usage_mb"`
		Processor       string  `json:"processor"`
		GPULayers       int     `json:"gpu_layers"`
		TokensPerSecond float64 `json:"tokens_per_second"`
		UptimeSeconds   int64   `json:"uptime_seconds"`
		ContextUsagePct int     `json:"context_usage_percent"`
	}

	mi := s.backend.LoadedModel()
	var models []psModel
	if mi != nil {
		uptime := int64(time.Since(mi.LoadedAt).Seconds())
		models = append(models, psModel{
			Name:            mi.Name,
			Size:            mi.RAMUsageMB * 1024 * 1024,
			RAMUsageMB:      mi.RAMUsageMB,
			VRAMUsageMB:     mi.VRAMUsageMB,
			Processor:       "CPU",
			TokensPerSecond: mi.TokensPerSecond,
			UptimeSeconds:   uptime,
			ContextUsagePct: 0,
		})
	}

	sendJSON(w, 200, map[string]interface{}{"models": models})
}

func (s *Server) handleDelete(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodDelete && r.Method != http.MethodPost {
		sendError(w, 405, "method not allowed")
		return
	}

	var req deleteRequest
	if r.Method == http.MethodPost {
		_ = json.NewDecoder(r.Body).Decode(&req)
	}

	s.mu.Lock()
	err := s.backend.Unload()
	s.mu.Unlock()
	if err != nil {
		sendError(w, 500, err.Error())
		return
	}

	sendJSON(w, 200, map[string]string{"status": "ok"})
}

func (s *Server) handleNegotiate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		sendError(w, 405, "method not allowed")
		return
	}

	var req negotiateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		sendError(w, 400, "E_INVALID_PARAMS: "+err.Error())
		return
	}

	modelPath := req.ModelPath
	if modelPath == "" {
		sendError(w, 400, "E_INVALID_PARAMS: model_path required")
		return
	}

	info, err := os.Stat(modelPath)
	if err != nil {
		sendError(w, 404, "E_MODEL_NOT_FOUND: "+modelPath)
		return
	}

	estimatedRAM := (info.Size() * 12 / 10) / (1024 * 1024)

	availableRAM := int64(4096)

	if estimatedRAM > availableRAM {
		sendJSON(w, 200, negotiateResponse{
			Status: "error",
			Error: &negotiateError{
				Code:    "E_INSUFFICIENT_RESOURCES",
				Message: fmt.Sprintf("Model requires %d MB RAM, %d MB available", estimatedRAM, availableRAM),
			},
			Alts: []negotiateAlt{
				{Model: modelPath + "-q4_0.gguf", RAMUsageMB: estimatedRAM * 3 / 4, PerformanceTokensPerS: 32.1},
				{Model: modelPath + "-q2_k.gguf", RAMUsageMB: estimatedRAM / 2, PerformanceTokensPerS: 25.0},
			},
		})
		return
	}

	loadOpts := loadOptionsFromParams(req.Params)

	s.mu.Lock()
	mi, err := s.backend.Load(modelPath, loadOpts)
	s.mu.Unlock()
	if err != nil {
		sendError(w, 500, err.Error())
		return
	}

	sendJSON(w, 200, negotiateResponse{
		Status:    "ok",
		ModelInfo: mi,
	})
}

func (s *Server) handleStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		sendError(w, 405, "method not allowed")
		return
	}

	mi := s.backend.LoadedModel()

	status := "unloaded"
	modelsLoaded := 0
	if mi != nil {
		status = "ready"
		modelsLoaded = 1
	}

	totalRAM := int64(8192)
	availableRAM := int64(4096)

	cpuCores := runtime.NumCPU()

	resp := StatusResponse{
		Status:       status,
		ModelsLoaded: modelsLoaded,
		ActiveModel:  mi,
		RawModel: &rawModelInfo{
			Loaded:     false,
			Path:       "/cognitiveos/models/raw/raw-model.gguf",
			RAMUsageMB: 0,
		},
		Hardware: &hardwareInfo{
			TotalRAMMB:     totalRAM,
			AvailableRAMMB: availableRAM,
			CPUCores:       cpuCores,
			CPUThreads:     cpuCores,
		},
	}

	sendJSON(w, 200, resp)
}

func (s *Server) handleCapabilities(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		sendError(w, 405, "method not allowed")
		return
	}

	caps := capabilitiesResponse{
		Backends: map[string]bool{
			"cpu":     true,
			"cuda":    false,
			"vulkan":  false,
			"npu":     false,
			"metal":   false,
		},
		PreferredBackend: "cpu",
		MaxModelSizeMB:   4096,
		SupportedQuants:  []string{"q4_0", "q4_k_m", "q5_k_m", "q8_0"},
		MaxContextWindow: 32768,
	}

	sendJSON(w, 200, caps)
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	status := "healthy"
	if !s.backend.IsLoaded() {
		status = "degraded"
	}

	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)
	ramPct := 50
	if mem.TotalAlloc > 0 {
		ramPct = int(mem.Alloc * 100 / mem.TotalAlloc)
	}

	modelsLoaded := 0
	if s.backend.IsLoaded() {
		modelsLoaded = 1
	}

	sendJSON(w, 200, healthResponse{
		Status:          status,
		UptimeSeconds:   int64(time.Since(s.startTime).Seconds()),
		ModelsLoaded:    modelsLoaded,
		RAMUsagePercent: ramPct,
	})
}
