//go:build !cgo

package main

import (
	"log"

	"github.com/CognitiveOS-Project/inference/internal/llm"
)

func newBackend(backend string) llm.Backend {
	if backend != "mock" {
		log.Println("WARN: cgo backend not available (CGO_ENABLED=0); falling back to mock backend. Raw model guardrail is disabled.")
	}
	return llm.NewMockBackend()
}
