//go:build !cgo

package main

import (
	"log"

	"github.com/CognitiveOS-Project/inference/internal/llm"
)

func newBackend() llm.Backend {
	log.Println("WARN: cgo backend not available (CGO_ENABLED=0); using mock backend. Raw model guardrail is disabled.")
	return llm.NewMockBackend()
}
