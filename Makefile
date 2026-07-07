SHELL := /bin/sh
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.DELETE_ON_ERROR:

BUILD_DIR := build
BIN_DIR := $(BUILD_DIR)/bin

.PHONY: build test lint clean build-llama

build: $(BIN_DIR)/cognitiveos-inference $(BIN_DIR)/cograw

build-llama:
	@if [ -f vendor/llama.cpp/CMakeLists.txt ] && [ ! -f vendor/llama.cpp/build/libllama.a ]; then
		cd vendor/llama.cpp
		cmake -B build -DLLAMA_NATIVE=0 -DBUILD_SHARED_LIBS=0 \
			-DLLAMA_BUILD_TESTS=0 -DLLAMA_BUILD_EXAMPLES=0 -DLLAMA_BUILD_SERVER=0 \
			-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY="$$(pwd)/build"
		cmake --build build --config Release --target llama -j"$$(nproc)"
	fi

$(BIN_DIR)/cognitiveos-inference $(BIN_DIR)/cograw: build-llama
	@mkdir -p $(BIN_DIR)
	@echo "  building with CGo (llama.cpp)"
	cd "$(CURDIR)"
	CGO_ENABLED=1 CGO_CFLAGS="-Ivendor/llama.cpp/ggml/include" \
		CGO_LDFLAGS="-Lvendor/llama.cpp/build/src -lllama $$(for lib in vendor/llama.cpp/build/libggml*.a; do libname=$$(basename "$$lib" .a | sed 's/^lib//'); echo -n " -l$$libname"; done) -lgomp" \
		go build -tags=cgo -ldflags="-s -w" -o $(BIN_DIR)/cognitiveos-inference ./cmd/coginfer
	CGO_ENABLED=1 CGO_CFLAGS="-Ivendor/llama.cpp/ggml/include" \
		CGO_LDFLAGS="-Lvendor/llama.cpp/build/src -lllama $$(for lib in vendor/llama.cpp/build/libggml*.a; do libname=$$(basename "$$lib" .a | sed 's/^lib//'); echo -n " -l$$libname"; done) -lgomp" \
		go build -tags=cgo -ldflags="-s -w" -o $(BIN_DIR)/cograw ./cmd/cograw

test:
	CGO_ENABLED=0 go test ./... -v -count=1

lint:
	CGO_ENABLED=0 go vet ./... 

clean:
	rm -rf $(BUILD_DIR)
