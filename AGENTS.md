# CogInfer — CognitiveOS Inference Engine

LLM inference server wrapping llama.cpp as a child process, exposing an Ollama-compatible HTTP API with CognitiveOS extensions.

## Architecture

```
cmd/coginfer         — Entry point, flag parsing
internal/server/     — HTTP server with all API handlers
  ├── /api/*         — Ollama-compatible: generate, chat, tags, pull, ps, delete
  ├── /cognitiveos/* — Extensions: status, capabilities
  ├── /api/negotiate — Resource negotiation
  └── /health        — Healthcheck for cognitiveosd
internal/llm/        — Backend interface + implementations
  ├── mock           — Simulated generation (for testing/dev)
  └── cli            — llama-cli subprocess (production)
internal/model/      — Model scanning and metadata (.gguf file discovery)
```

## Build

```go
go build -o bin/coginfer ./cmd/coginfer
```

## Usage

```bash
# Start with mock backend (default, no llama.cpp needed)
./bin/coginfer --backend mock --models /cognitiveos/models

# Start with real llama-cli backend
./bin/coginfer --backend cli --models /cognitiveos/models

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

- **mock**: Simulated token generation with delays, no external dependencies. Default for development.
- **cli**: Shells out to `llama-cli` for real inference. Pass `--backend cli` in production.
