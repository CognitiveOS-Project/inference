//go:build cgo

package main

import "github.com/CognitiveOS-Project/inference/internal/llm"

func newBackend(backend string) llm.Backend {
	if backend == "mock" {
		return llm.NewMockBackend()
	}
	return llm.NewCgoBackend()
}
