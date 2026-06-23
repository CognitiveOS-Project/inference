# CognitiveOS Inference Engine

Local LLM runtime for Raw Model and Wide Model execution. Wraps llama.cpp with a CognitiveOS-native interface.

## Components

- `coginfer` — Go binary that manages llama.cpp as a child process
- Exposes an Ollama-compatible HTTP API
- Handles model downloads, quantization, and memory management

## Build

```bash
go build -o bin/coginfer ./cmd/coginfer
```

## Model Paths

- Raw Model: `/cognitiveos/models/raw/` (bundled in distro image)
- Wide Models: `/cognitiveos/models/wide/` (installed via cpm)

## Resource Awareness

Before loading a model, `coginfer` checks `/proc/meminfo` and available CPUs. Falls back to smaller quantization if resources are insufficient — no crash, just graceful degradation.
