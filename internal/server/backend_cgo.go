//go:build cgo

package server

import "github.com/CognitiveOS-Project/inference/internal/llm"

func newCgoBackend() llm.Backend {
	return llm.NewCgoBackend()
}
