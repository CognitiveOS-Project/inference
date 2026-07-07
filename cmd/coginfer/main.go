package main

import (
	"flag"
	"log"
	"os"

	"github.com/CognitiveOS-Project/inference/internal/server"
)

func main() {
	addr := flag.String("addr", "127.0.0.1:11434", "HTTP listen address")
	modelDir := flag.String("models", "/cognitiveos/models", "model directory")
	backend := flag.String("backend", "mock", "inference backend (mock, cgo)")
	logFile := flag.String("log", "", "log file path")
	flag.Parse()

	if *logFile != "" {
		f, err := os.OpenFile(*logFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			log.Fatal(err)
		}
		defer func() { _ = f.Close() }()
		log.SetOutput(f)
	}

	srv := server.New(*modelDir, *backend)
	log.Printf("coginfer starting on %s (backend=%s, models=%s)", *addr, *backend, *modelDir)
	if err := srv.Listen(*addr); err != nil {
		log.Fatal(err)
	}
}
