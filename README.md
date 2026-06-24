# inference — CogInfer

CognitiveOS inference engine — local LLM runtime for Raw Model and Wide Model execution. Exposes an Ollama-compatible HTTP API with CognitiveOS extensions.

## Architecture

```
cmd/coginfer             — Entry point, flag parsing
internal/server/         — HTTP server with all API handlers
internal/llm/            — Backend interface + implementations (mock, cli)
internal/model/          — Model scanning and .gguf metadata discovery
```

## Build

```bash
go build -o bin/coginfer ./cmd/coginfer
```

## Usage

```bash
# Start with mock backend (no llama.cpp needed)
./bin/coginfer --backend mock --models /cognitiveos/models

# Start with llama-cli backend (production)
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

- **mock** — Simulated token generation with delays. Default for development.
- **cli** — Shells out to `llama-cli` for real inference on device.

## Related

- [CognitiveOS](https://github.com/CognitiveOS-Project/cognitiveos) — main project repository
- [cognitive-os.org](https://cognitive-os.org) — project website
- [cognitiveosd](https://github.com/CognitiveOS-Project/cognitiveosd) — daemon that manages model lifecycle
- [Product Specs](https://github.com/CognitiveOS-Project/product-specs) — inference API specification
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
