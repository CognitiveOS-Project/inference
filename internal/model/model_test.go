package model

import (
	"os"
	"path/filepath"
	"testing"
)

func TestManagerResolveAbsPath(t *testing.T) {
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "test.gguf")
	_ = os.WriteFile(modelPath, []byte("dummy"), 0644)

	m := NewManager(dir)
	got, err := m.Resolve(modelPath)
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if got != modelPath {
		t.Fatalf("expected %q, got %q", modelPath, got)
	}
}

func TestManagerResolveNotFound(t *testing.T) {
	m := NewManager("/nonexistent")
	_, err := m.Resolve("nonexistent-model")
	if err == nil {
		t.Fatal("expected error for nonexistent model")
	}
}

func TestManagerList(t *testing.T) {
	dir := t.TempDir()
	_ = os.MkdirAll(filepath.Join(dir, "wide", "active"), 0755)
	_ = os.WriteFile(filepath.Join(dir, "raw-model.gguf"), []byte("raw"), 0644)
	_ = os.WriteFile(filepath.Join(dir, "wide", "active", "gemma.gguf"), []byte("gemma"), 0644)

	m := NewManager(dir)
	models, err := m.List()
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(models) != 2 {
		t.Fatalf("expected 2 models, got %d", len(models))
	}
}

func TestManagerListEmptyDir(t *testing.T) {
	dir := t.TempDir()
	m := NewManager(dir)
	models, err := m.List()
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(models) != 0 {
		t.Fatalf("expected 0 models, got %d", len(models))
	}
}
