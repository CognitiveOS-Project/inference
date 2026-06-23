package model

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type Entry struct {
	Name       string `json:"name"`
	Path       string `json:"path"`
	Size       int64  `json:"size"`
	ModifiedAt string `json:"modified_at"`
}

type Manager struct {
	ModelDir string
}

func NewManager(modelDir string) *Manager {
	if modelDir == "" {
		modelDir = "/cognitiveos/models"
	}
	return &Manager{ModelDir: modelDir}
}

func (m *Manager) List() ([]Entry, error) {
	var models []Entry

	dirs := []string{
		filepath.Join(m.ModelDir, "raw"),
		filepath.Join(m.ModelDir, "wide", "active"),
		m.ModelDir,
	}

	seen := map[string]bool{}
	for _, dir := range dirs {
		entries, err := os.ReadDir(dir)
		if err != nil {
			continue
		}
		for _, e := range entries {
			if e.IsDir() || !strings.HasSuffix(e.Name(), ".gguf") {
				continue
			}
			if seen[e.Name()] {
				continue
			}
			seen[e.Name()] = true
			info, err := e.Info()
			if err != nil {
				continue
			}
			models = append(models, Entry{
				Name:       strings.TrimSuffix(e.Name(), ".gguf"),
				Path:       filepath.Join(dir, e.Name()),
				Size:       info.Size(),
				ModifiedAt: info.ModTime().UTC().Format("2006-01-02T15:04:05Z"),
			})
		}
	}

	sort.Slice(models, func(i, j int) bool {
		return models[i].Name < models[j].Name
	})

	return models, nil
}

func (m *Manager) Resolve(name string) (string, error) {
	if filepath.IsAbs(name) {
		if _, err := os.Stat(name); err == nil {
			return name, nil
		}
		return "", fmt.Errorf("E_MODEL_NOT_FOUND: %s", name)
	}

	baseName := name
	if !strings.HasSuffix(baseName, ".gguf") {
		baseName += ".gguf"
	}

	candidates := []string{
		filepath.Join(m.ModelDir, "raw", baseName),
		filepath.Join(m.ModelDir, "wide", "active", baseName),
		filepath.Join(m.ModelDir, baseName),
		filepath.Join(m.ModelDir, "wide", "active", name+".gguf"),
	}

	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}

	// Search recursively
	var found string
	filepath.Walk(m.ModelDir, func(path string, info os.FileInfo, err error) error {
		if err != nil || found != "" {
			return nil
		}
		if !info.IsDir() && strings.HasSuffix(info.Name(), ".gguf") {
			stem := strings.TrimSuffix(info.Name(), ".gguf")
			if stem == name {
				found = path
			}
		}
		return nil
	})

	if found != "" {
		return found, nil
	}

	return "", fmt.Errorf("E_MODEL_NOT_FOUND: model %q not found in %s", name, m.ModelDir)
}
