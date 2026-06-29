# CogInfer — CognitiveOS Inference Engine

LLM inference server wrapping llama.cpp as a child process, exposing an Ollama-compatible HTTP API with CognitiveOS extensions.

## Architecture

```
cmd/coginfer             — Wide Model HTTP inference server (Ollama-compatible)
cmd/cograw               — Raw Model RPC server (firmware guardrail)
internal/server/         — HTTP server with all API handlers
  ├── /api/*             — Ollama-compatible: generate, chat, tags, pull, ps, delete
  ├── /cognitiveos/*     — Extensions: status, capabilities
  ├── /api/negotiate     — Resource negotiation
  ├── /health            — Healthcheck for cognitiveosd
  ├── backend_cgo.go     — CgoBackend constructor (build tag: cgo)
  └── backend_stub.go    — Stub fallback (build tag: !cgo)
internal/llm/            — Backend interface + implementations
  ├── llm.go             — Backend interface, MockBackend
  ├── bridge.go          — Single import "C" file wrapping llama.h API (build tag: cgo)
  ├── cgobackend.go      — CgoBackend: Backend impl calling bridge functions (build tag: cgo)
  └── loadopts.go        — LoadOptions{NumCtx, GPULayers, Threads}
internal/model/          — Model scanning and metadata (.gguf file discovery)
vendor/llama.cpp/        — Git submodule, pinned to b9842
```

### coginfer (Wide Model)

LLM inference server linking llama.cpp via a vendored CGo bridge, exposing an Ollama-compatible HTTP API with CognitiveOS extensions.

#### Build

```bash
# With CGo (production — requires cmake + gcc + submodule)
cd vendor/llama.cpp && cmake -B build -DLLAMA_NO_ACCELERATE=1 \
  -DLLAMA_STATIC=1 -DLLAMA_NATIVE=0 \
  -DBUILD_SHARED_LIBS=0 -DLLAMA_BUILD_TESTS=0 \
  -DLLAMA_BUILD_EXAMPLES=0 -DLLAMA_BUILD_SERVER=0
cmake --build build --config Release -j$(nproc)
cd ../.. && CGO_ENABLED=1 go build -tags=cgo -o bin/coginfer ./cmd/coginfer

# Without CGo (CI, development — mock backend only)
CGO_ENABLED=0 go build -o bin/coginfer ./cmd/coginfer
```

## Usage

```bash
# Start with mock backend (default, no llama.cpp needed)
./bin/coginfer --backend mock --models /cognitiveos/models

# Start with CGo llama.cpp bridge (production)
./bin/coginfer --backend cgo --models /cognitiveos/models

# Start with CGo llama.cpp bridge (production)
./bin/coginfer --backend cgo --models /cognitiveos/models

# Custom port and log file
./bin/coginfer --addr 127.0.0.1:11434 --log /cognitiveos/logs/inference.log
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/generate` | Generate completion (stream + non-stream) |
| POST | `/api/chat` | Chat completion |
| GET  | `/api/tags` | List available models |
| POST | `/api/pull` | Load model from path |
| GET  | `/api/ps` | Show loaded model resource usage |
| DELETE | `/api/delete` | Unload model |
| POST | `/api/negotiate` | Resource negotiation before load |
| GET  | `/cognitiveos/status` | Full engine status |
| GET  | `/cognitiveos/capabilities` | Hardware capabilities |
| GET  | `/health` | Healthcheck |

## Backends

- **mock**: Simulated token generation with delays, no external dependencies. Default for development. Always available.
- **cgo**: In-process llama.cpp via CGo bridge. Requires `CGO_ENABLED=1` and `vendor/llama.cpp/build/libllama.a`. Production default.
- **cgo**: In-process llama.cpp via CGo bridge. Requires `CGO_ENABLED=1` and `vendor/llama.cpp/build/libllama.a`. Production default.

#### Build Tags

- `bridge.go`, `cgobackend.go`, `backend_cgo.go` — `//go:build cgo`
- `backend_stub.go` — `//go:build !cgo`
- CI runs `CGO_ENABLED=0`, excludes all CGo files automatically
- `MockBackend` has no build tag — always available

### cograw (Raw Model — Firmware Guardrail)

Always-on, root-level RPC server that provides the firmware guardrail between the human and the Wide Model. Communicates over a Unix socket using JSON-RPC 2.0.

**Key constraint:** cograw has **no knowledge of MCP tools or registries** — it is a pure guardrail GGUF. Tool routing is the daemon's responsibility.

#### Location

`cmd/cograw/main.go`

#### Transport

JSON-RPC 2.0 over Unix socket at `/cognitiveos/run/raw.sock` (mode 0600, root-owned).

#### RPC Methods

| Method | Purpose |
|--------|---------|
| `validate_system_code` | Validate system codes (wake/idle/security/reset/unlock) |
| `check_unlock_code` | Validate unlock codes for paid patches |
| `audit_resources` | Check hardware resources before Wide Model load |
| `healthcheck` | Return running status |
| `version` | Return version, model path, quantization |
| `validate_prompt` | Guardrail check — allow/deny/modify prompts before they reach the Wide Model |

**Full parameter/response schemas**: [product-specs/specs/raw-model.md](https://github.com/CognitiveOS-Project/product-specs/blob/development/specs/raw-model.md#rpc-methods)

#### Build

```bash
# With CGo (production)
cd vendor/llama.cpp && cmake -B build ... && cmake --build build --config Release -j$(nproc)
CGO_ENABLED=1 go build -tags=cgo -o bin/cograw ./cmd/cograw

# Without CGo (mock mode, no real guardrail)
CGO_ENABLED=0 go build -o bin/cograw ./cmd/cograw
```

#### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--socket` | `/cognitiveos/run/raw.sock` | Unix socket path |
| `--model` | `/cognitiveos/models/raw/raw-model.gguf` | Raw model GGUF path |
| `--log` | (stderr) | Log file path |
| `--audit-log` | `/cognitiveos/logs/raw/audit.log` | Audit log file path |

#### Startup Order

cograw starts **before** cognitiveosd. The daemon hard-fails if the raw socket is unavailable at boot.

#### Model File

- Path: `/cognitiveos/models/raw/raw-model.gguf`
- Read-only squashfs partition
- `root:root 0400`
- Wide Model has **no read access**
- Updated only via firmware flash (not `.cgp` packages)

#### Spec Reference

- [raw-model.md](https://github.com/CognitiveOS-Project/product-specs/blob/development/specs/raw-model.md) — authoritative Raw Model specification
- [architecture.md](https://github.com/CognitiveOS-Project/product-specs/blob/development/specs/architecture.md) — system architecture and layer diagram
- [cognitiveosd-api.md](https://github.com/CognitiveOS-Project/product-specs/blob/development/specs/cognitiveosd-api.md) — daemon protocol and message types
