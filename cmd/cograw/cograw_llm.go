//go:build cgo

package main

import "github.com/CognitiveOS-Project/inference/internal/llm"

func newBackend() llm.Backend {
	return llm.NewCgoBackend()
}
