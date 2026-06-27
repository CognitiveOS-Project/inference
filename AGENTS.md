# CognitiveOS Inference Engine

This repo contains two binaries: **coginfer** (Wide Model) and **cograw** (Raw Model firmware guardrail).

## Architecture

```
cmd/coginfer             — Wide Model HTTP inference server (Ollama-compatible)
cmd/cograw               — Raw Model RPC server (firmware guardrail)
internal/server/         — HTTP server with all API handlers
  ├── /api/*             — Ollama-compatible: generate, chat, tags, pull, ps, delete
  ├── /cognitiveos/*     — Extensions: status, capabilities
  ├── /api/negotiate     — Resource negotiation
  └── /health            — Healthcheck for cognitiveosd
internal/llm/            — Backend interface + implementations
  ├── mock               — Simulated generation (for testing/dev)
  └── cli                — llama-cli subprocess (production)
internal/model/          — Model scanning and metadata (.gguf file discovery)
```

### coginfer (Wide Model)

LLM inference server wrapping llama.cpp as a child process, exposing an Ollama-compatible HTTP API with CognitiveOS extensions.

#### Build

```go
go build -o bin/coginfer ./cmd/coginfer
```

#### Usage

```bash
# Start with mock backend (default, no llama.cpp needed)
./bin/coginfer --backend mock --models /cognitiveos/models

# Start with real llama-cli backend
./bin/coginfer --backend cli --models /cognitiveos/models

# Custom port and log file
./bin/coginfer --addr 127.0.0.1:11434 --log /cognitiveos/logs/inference.log
```

#### API

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

#### Backends

- **mock**: Simulated token generation with delays, no external dependencies. Default for development.
- **cli**: Shells out to `llama-cli` for real inference. Pass `--backend cli` in production.

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

**Full parameter/response schemas**: [product-specs/specs/raw-model.md#rpc-methods](https://github.com/CognitiveOS-Project/product-specs/blob/development/specs/raw-model.md#rpc-methods)

#### Build

```go
go build -o bin/cograw ./cmd/cograw
```

#### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--socket` | `/cognitiveos/run/raw.sock` | Unix socket path |
| `--model` | `/cognitiveos/models/raw/raw-model.gguf` | Raw model GGUF path |
| `--llama-bin` | `llama-cli` | llama-cli binary for tensor integrity check |
| `--log` | (stderr) | Log file path |

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
