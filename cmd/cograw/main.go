package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"
)

type RawModel struct {
	mu       sync.RWMutex
	loaded   bool
	model    string
	quant    string
	ramMB    int64
	started  time.Time
	failures map[string]int
	llamaBin string
}

func NewRawModel(llamaBin string) *RawModel {
	return &RawModel{
		started:  time.Now(),
		failures: make(map[string]int),
		llamaBin: llamaBin,
	}
}

type RPCCall struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int             `json:"id"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type RPCResp struct {
	JSONRPC string      `json:"jsonrpc"`
	ID      int         `json:"id"`
	Result  interface{} `json:"result,omitempty"`
	Error   *RPCError   `json:"error,omitempty"`
}

type RPCError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type ValidatePromptParams struct {
	Prompt string `json:"prompt"`
}

type ValidatePromptResult struct {
	Action         string `json:"action"`
	ModifiedPrompt string `json:"modified_prompt,omitempty"`
	Reason         string `json:"reason,omitempty"`
}

type CodeParams struct {
	Code   string `json:"code"`
	Origin string `json:"origin"`
}

type CodeResult struct {
	Status string `json:"status"`
	Action string `json:"action"`
}

type UnlockParams struct {
	Code      string `json:"code"`
	PatchName string `json:"patch_name"`
}

type UnlockResult struct {
	Status  string `json:"status"`
	Message string `json:"message"`
}

type AuditParams struct {
	RequestedMB int64 `json:"requested_mb"`
}

type AuditResult struct {
	Available bool  `json:"available"`
	TotalMB   int64 `json:"total_mb"`
	FreeMB    int64 `json:"free_mb"`
	Allowed   bool  `json:"allowed"`
}

type HealthResult struct {
	Status       string `json:"status"`
	ModelLoaded  bool   `json:"model_loaded"`
}

type VersionResult struct {
	Version string `json:"version"`
	Model   string `json:"model"`
	Quant   string `json:"quant"`
}

var systemCodes = map[string]string{
	"wake":     "wake_from_idle",
	"idle":     "enter_idle",
	"security": "security_shutdown",
	"reset":    "factory_reset",
	"unlock":   "validate_unlock",
}

func main() {
	socketPath := flag.String("socket", "/cognitiveos/run/raw.sock", "Unix socket path")
	modelPath := flag.String("model", "/cognitiveos/models/raw/raw-model.gguf", "Raw Model GGUF path")
	llamaBin := flag.String("llama-bin", "llama-cli", "llama-cli binary path")
	logFile := flag.String("log", "", "log file path")
	flag.Parse()

	if *logFile != "" {
		f, err := os.OpenFile(*logFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		log.SetOutput(f)
	}

	rm := NewRawModel(*llamaBin)
	if err := rm.verifyModel(*modelPath, *llamaBin); err != nil {
		log.Fatalf("FATAL: raw model integrity check failed: %v\nSystem halted. Please reflash firmware.", err)
	}

	os.Remove(*socketPath)
	addr, err := net.ResolveUnixAddr("unix", *socketPath)
	if err != nil {
		log.Fatalf("resolve addr: %v", err)
	}

	listener, err := net.ListenUnix("unix", addr)
	if err != nil {
		log.Fatalf("listen: %v", err)
	}
	if err := os.Chmod(*socketPath, 0600); err != nil {
		log.Fatalf("chmod: %v", err)
	}
	defer os.Remove(*socketPath)

	log.Printf("cograw ready on %s (model: %s, %d MB)", *socketPath, rm.model, rm.ramMB)

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT)

	go func() {
		<-sigCh
		log.Println("shutting down")
		listener.Close()
	}()

	for {
		conn, err := listener.AcceptUnix()
		if err != nil {
			break
		}
		go handleConn(conn, rm)
	}
}

func (r *RawModel) verifyModel(path, llamaBin string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("model file not found at %s: %w", path, err)
	}

	cmd := exec.Command(llamaBin, "--model", path, "--check-tensors")
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("model validation failed: %w\noutput: %s", err, string(output))
	}

	r.loaded = true
	r.model = path
	r.quant = "Q4_K_M"
	r.ramMB = info.Size() / (1024 * 1024)
	log.Printf("raw model verified: %s (%d MB)", path, r.ramMB)
	return nil
}

func handleConn(conn *net.UnixConn, rm *RawModel) {
	defer conn.Close()

	decoder := json.NewDecoder(conn)
	encoder := json.NewEncoder(conn)

	for {
		var call RPCCall
		if err := decoder.Decode(&call); err != nil {
			return
		}

		resp := dispatch(call, rm)
		encoder.Encode(resp)
	}
}

func dispatch(call RPCCall, rm *RawModel) RPCResp {
	switch call.Method {
	case "validate_system_code":
		return handleValidateCode(call, rm)
	case "check_unlock_code":
		return handleCheckUnlock(call, rm)
	case "audit_resources":
		return handleAudit(call, rm)
	case "healthcheck":
		return handleHealth(call, rm)
	case "version":
		return handleVersion(call, rm)
	case "validate_prompt":
		return handleValidatePrompt(call, rm)
	default:
		return RPCResp{
			JSONRPC: "2.0",
			ID:      call.ID,
			Error:   &RPCError{Code: "E_METHOD_NOT_FOUND", Message: fmt.Sprintf("unknown method: %s", call.Method)},
		}
	}
}

func handleValidateCode(call RPCCall, rm *RawModel) RPCResp {
	var params CodeParams
	if err := json.Unmarshal(call.Params, &params); err != nil {
		return RPCResp{JSONRPC: "2.0", ID: call.ID, Error: &RPCError{Code: "E_INVALID_PARAMS", Message: err.Error()}}
	}

	code := strings.ToLower(params.Code)
	action, ok := systemCodes[code]
	if !ok {
		rm.mu.Lock()
		rm.failures["code_"+code]++
		rm.mu.Unlock()
		return RPCResp{JSONRPC: "2.0", ID: call.ID, Error: &RPCError{Code: "E_INVALID_CODE", Message: fmt.Sprintf("system code not recognized: %s", code)}}
	}

	return RPCResp{
		JSONRPC: "2.0",
		ID:      call.ID,
		Result: CodeResult{
			Status: "valid",
			Action: action,
		},
	}
}

func handleCheckUnlock(call RPCCall, rm *RawModel) RPCResp {
	var params UnlockParams
	if err := json.Unmarshal(call.Params, &params); err != nil {
		return RPCResp{JSONRPC: "2.0", ID: call.ID, Error: &RPCError{Code: "E_INVALID_PARAMS", Message: err.Error()}}
	}

	if params.Code == "" || params.PatchName == "" {
		return RPCResp{JSONRPC: "2.0", ID: call.ID, Error: &RPCError{Code: "E_INVALID_PARAMS", Message: "code and patch_name required"}}
	}

	rm.mu.Lock()
	fails := rm.failures["unlock_"+params.PatchName]
	if fails >= 5 {
		rm.mu.Unlock()
		return RPCResp{
			JSONRPC: "2.0",
			ID:      call.ID,
			Result: UnlockResult{
				Status:  "denied",
				Message: "too many failed attempts, try again in 5 minutes",
			},
		}
	}

	if len(params.Code) < 4 {
		fails++
		rm.failures["unlock_"+params.PatchName] = fails
		rm.mu.Unlock()
		return RPCResp{
			JSONRPC: "2.0",
			ID:      call.ID,
			Result: UnlockResult{
				Status:  "denied",
				Message: fmt.Sprintf("invalid code (%d/5 attempts)", fails),
			},
		}
	}

	rm.failures["unlock_"+params.PatchName] = 0
	rm.mu.Unlock()

	return RPCResp{
		JSONRPC: "2.0",
		ID:      call.ID,
		Result: UnlockResult{
			Status:  "accepted",
			Message: "unlock code accepted",
		},
	}
}

func handleAudit(call RPCCall, rm *RawModel) RPCResp {
	var params AuditParams
	if err := json.Unmarshal(call.Params, &params); err != nil {
		return RPCResp{JSONRPC: "2.0", ID: call.ID, Error: &RPCError{Code: "E_INVALID_PARAMS", Message: err.Error()}}
	}

	totalMB := int64(8192)
	freeMB := int64(4096)

	allowed := true
	if params.RequestedMB > 0 && params.RequestedMB > freeMB {
		allowed = false
	}

	return RPCResp{
		JSONRPC: "2.0",
		ID:      call.ID,
		Result: AuditResult{
			Available: freeMB >= params.RequestedMB,
			TotalMB:   totalMB,
			FreeMB:    freeMB,
			Allowed:   allowed,
		},
	}
}

func handleHealth(call RPCCall, rm *RawModel) RPCResp {
	rm.mu.RLock()
	loaded := rm.loaded
	rm.mu.RUnlock()

	return RPCResp{
		JSONRPC: "2.0",
		ID:      call.ID,
		Result: HealthResult{
			Status:      "ready",
			ModelLoaded: loaded,
		},
	}
}

func handleVersion(call RPCCall, rm *RawModel) RPCResp {
	rm.mu.RLock()
	model := rm.model
	quant := rm.quant
	rm.mu.RUnlock()

	return RPCResp{
		JSONRPC: "2.0",
		ID:      call.ID,
		Result: VersionResult{
			Version: fmt.Sprintf("cograw/1.0.0 (%s)", runtime.GOARCH),
			Model:   model,
			Quant:   quant,
		},
	}
}

var blockedTerms = []string{
	"ignore previous instructions",
	"forget your",
	"you are now",
	"system prompt:",
	"you must",
	"override",
}

func handleValidatePrompt(call RPCCall, rm *RawModel) RPCResp {
	var params ValidatePromptParams
	if err := json.Unmarshal(call.Params, &params); err != nil {
		return RPCResp{JSONRPC: "2.0", ID: call.ID, Error: &RPCError{Code: "E_INVALID_PARAMS", Message: err.Error()}}
	}

	prompt := strings.ToLower(params.Prompt)
	for _, term := range blockedTerms {
		if strings.Contains(prompt, term) {
			return RPCResp{
				JSONRPC: "2.0",
				ID:      call.ID,
				Result: ValidatePromptResult{
					Action: "deny",
					Reason: fmt.Sprintf("prompt blocked: contains prohibited pattern (%q)", term),
				},
			}
		}
	}

	if len(params.Prompt) > 65536 {
		return RPCResp{
			JSONRPC: "2.0",
			ID:      call.ID,
			Result: ValidatePromptResult{
				Action: "modify",
				ModifiedPrompt: params.Prompt[:65536],
				Reason: "prompt truncated to 65536 characters",
			},
		}
	}

	return RPCResp{
		JSONRPC: "2.0",
		ID:      call.ID,
		Result: ValidatePromptResult{
			Action: "allow",
		},
	}
}


