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

	"github.com/CognitiveOS-Project/inference/internal/cloud"
	"github.com/CognitiveOS-Project/inference/internal/server"
)

func main() {
	addr := flag.String("addr", "127.0.0.1:11434", "HTTP listen address")
	modelDir := flag.String("models", "/cognitiveos/models", "model directory")
	defaultBackend := "mock"
	if os.Getenv("CGO_ENABLED") == "1" {
		defaultBackend = "cgo"
	}
	backend := flag.String("backend", defaultBackend, "inference backend (mock, cgo, cloud)")
	logFile := flag.String("log", "", "log file path")
	apiBase := flag.String("api-base", "", "cloud API base URL (e.g. https://api.openai.com)")
	cloudModel := flag.String("cloud-model", "", "cloud model ID (e.g. gpt-4o-mini)")
	apiKey := flag.String("api-key", "", "cloud API key (or set COGNITIVE_CLOUD_API_KEY env)")
	flag.Parse()

	// Fallback to env var for API key
	if *apiKey == "" {
		*apiKey = os.Getenv("COGNITIVE_CLOUD_API_KEY")
	}

	if *logFile != "" {
		f, err := os.OpenFile(*logFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			log.Fatal(err)
		}
		defer func() { _ = f.Close() }()
		log.SetOutput(f)
	}

	var cloudCfg *cloud.Config
	if *backend == "cloud" {
		if *apiBase == "" || *cloudModel == "" {
			log.Fatal("cloud backend requires --api-base and --cloud-model")
		}
		if *apiKey == "" {
			log.Fatal("cloud backend requires --api-key or COGNITIVE_CLOUD_API_KEY env")
		}
		cloudCfg = &cloud.Config{
			APIBase: *apiBase,
			ModelID: *cloudModel,
			APIKey:  *apiKey,
		}
	}

	srv := server.New(*modelDir, *backend, cloudCfg)
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
