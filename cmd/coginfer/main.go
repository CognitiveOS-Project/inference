package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

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
	httpSrv, err := srv.Listen(*addr)
	if err != nil {
		log.Fatal(err)
	}

	go func() {
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("listen error: %v", err)
		}
	}()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT)

	<-sigCh
	log.Println("shutting down coginfer...")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := httpSrv.Shutdown(ctx); err != nil {
		log.Printf("shutdown error: %v", err)
	}
	if err := srv.Shutdown(); err != nil {
		log.Printf("backend unload error: %v", err)
	}
	log.Println("coginfer stopped")
}
