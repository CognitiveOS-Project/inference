//go:build !cgo

package server

import (
	"log"

	"github.com/CognitiveOS-Project/inference/internal/llm"
)

func newCgoBackend() llm.Backend {
	log.Println("WARN: cgo backend requested but build without CGO_ENABLED=1; falling back to mock")
	return llm.NewMockBackend()
}
