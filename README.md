# inference — CogInfer + CogRaw

CognitiveOS inference engine — local LLM runtime for both **Raw Model** (firmware guardrail) and **Wide Model** (operational inference). Contains two binaries:

- `coginfer` — Ollama-compatible HTTP inference server for the Wide Model
- `cograw` — Root-level RPC server for the Raw Model (firmware GGUF guardrail)

## Architecture

```
cmd/coginfer             — Wide Model inference server (HTTP, Ollama-compatible)
cmd/cograw               — Raw Model RPC server (Unix socket, JSON-RPC 2.0)
internal/server/         — HTTP server with all API handlers
internal/llm/            — Backend interface + implementations (mock, cgo)
internal/model/          — Model scanning and .gguf metadata discovery
```

## Binaries

### coginfer (Wide Model)

Exposes an Ollama-compatible HTTP API for the general-purpose Wide Model.

#### Build

```bash
# With CGo (production — requires vendored llama.cpp build)
CGO_ENABLED=1 go build -tags=cgo -o bin/coginfer ./cmd/coginfer

# Without CGo (mock backend only, no llama.cpp needed)
CGO_ENABLED=0 go build -o bin/coginfer ./cmd/coginfer
```

#### Usage

```bash
# Start with mock backend (no llama.cpp needed)
./bin/coginfer --backend mock --models /cognitiveos/models

# Start with CGo llama.cpp backend (production)
./bin/coginfer --backend cgo --models /cognitiveos/models

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

- **mock** — Simulated token generation with delays. Default for development.
- **cgo** — In-process llama.cpp via CGo bridge. Requires `CGO_ENABLED=1` and vendored llama.cpp build.

### cograw (Raw Model — Firmware Guardrail)

A small, always-on RPC server that acts as the firmware-level guardrail between the human and the Wide Model. Runs as root, communicates over a Unix socket via JSON-RPC 2.0.

**Key constraint:** cograw has **no knowledge of MCP tools or registries** — it is a pure guardrail GGUF. Tool routing is the daemon's responsibility.

For the authoritative specification, see [raw-model.md](https://github.com/CognitiveOS-Project/product-specs/blob/development/specs/raw-model.md) in the product-specs repo.

#### Build

Requires CGo and vendored llama.cpp. See [cograw build docs](cmd/cograw/) for details.

#### Usage

```bash
./bin/cograw \
  --socket /cognitiveos/run/raw.sock \
  --model /cognitiveos/models/raw/raw-model.gguf
```

#### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--socket` | `/cognitiveos/run/raw.sock` | Unix socket path for JSON-RPC |
| `--model` | `/cognitiveos/models/raw/raw-model.gguf` | Path to raw model GGUF file |
| `--log` | (stderr) | Log file path |

#### RPC Methods

| Method | Description |
|--------|-------------|
| `validate_system_code` | Validate a system code (wake/idle/security/reset/unlock) |
| `check_unlock_code` | Validate an unlock code for a paid patch |
| `audit_resources` | Check hardware resources before Wide Model load |
| `healthcheck` | Return whether the Raw Model is running and ready |
| `version` | Return Raw Model version and model info |
| `validate_prompt` | Guardrail check: allow/deny/modify a prompt before it reaches the Wide Model |

Detailed parameter and response schemas are in the [product-specs RPC methods table](https://github.com/CognitiveOS-Project/product-specs/blob/development/specs/raw-model.md#rpc-methods).

#### Role in System

```
Human → cli → cognitiveosd → cograw (validate) → Wide Model (generate) → cognitiveosd (parse, route MCP) → cli
```

1. Human input arrives at the daemon via CLI
2. Daemon sends the prompt to cograw for guardrail validation
3. If allowed, daemon forwards to Wide Model for inference
4. Daemon parses the Wide Model response and routes tool calls to MCP servers
5. Final output returned to CLI

cograw **must start before** cognitiveosd — the daemon hard-fails if the raw socket is unavailable at boot.

#### Model Protection

The raw model file is stored at `/cognitiveos/models/raw/raw-model.gguf` on a read-only squashfs partition:
- Owner: `root:root`
- Permissions: `0400`
- The Wide Model has **no read access**
- Only a full firmware update (physical reflash or signed update) can change it

## Related

- [Product Specs](https://github.com/CognitiveOS-Project/product-specs) — authoritative specs for raw-model, architecture, API, and security model
- [cognitiveosd](https://github.com/CognitiveOS-Project/cognitiveosd) — daemon that manages the Wide Model and routes tool calls
- [CognitiveOS Project](https://github.com/CognitiveOS-Project) — GitHub organization

## Contributing

1. Branch from `development`, not `main`
2. Use topic branches: `feature/<name>`, `fix/<name>`, `bugfix/<name>`
3. Open a PR to `development` with a clear title and description
4. Merge via squash after review
5. Changes flow to `main` via a release PR

See the [SDLC repo](https://github.com/CognitiveOS-Project/sdlc) for the full contribution guide, code review standards, and testing strategy.

## Author

**Jean Machuca** — [GitHub](https://github.com/jeanmachuca) · [Sponsor](https://github.com/sponsors/jeanmachuca)

## License

MIT
