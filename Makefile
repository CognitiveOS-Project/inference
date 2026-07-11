SHELL := /bin/sh
.SHELLFLAGS := -eu -c
.ONESHELL:
.DELETE_ON_ERROR:

BUILD_DIR := build
BIN_DIR := $(BUILD_DIR)/bin

.PHONY: build build-mock test lint clean build-llama pack publish

build:
	@if [ "$$CGO_ENABLED" = "0" ]; then \
		echo "  CGO_ENABLED=0: building mock backend"; \
		$(MAKE) build-mock; \
	else \
		echo "  CGO_ENABLED=1: building production backend"; \
		$(MAKE) $(BIN_DIR)/coginfer $(BIN_DIR)/cograw; \
	fi

build-mock:
	@mkdir -p $(BIN_DIR)
	CGO_ENABLED=0 go build -ldflags="-s -w" -o $(BIN_DIR)/coginfer ./cmd/coginfer
	CGO_ENABLED=0 go build -ldflags="-s -w" -o $(BIN_DIR)/cograw ./cmd/cograw

build-llama:
	@if [ ! -d vendor/llama.cpp ]; then \
		echo "  Cloning llama.cpp into vendor/..."; \
		mkdir -p vendor; \
		git clone --depth=1 https://github.com/ggerganov/llama.cpp.git vendor/llama.cpp; \
	fi
	@if [ ! -f vendor/llama.cpp/build/libllama.a ]; then
		cd vendor/llama.cpp
		cmake -B build -DLLAMA_NATIVE=0 -DBUILD_SHARED_LIBS=0 \
			-DLLAMA_BUILD_TESTS=0 -DLLAMA_BUILD_EXAMPLES=0 -DLLAMA_BUILD_SERVER=0 \
			-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY="$$(pwd)/build"
		cmake --build build --config Release --target llama -j"$$(nproc)"
	fi

$(BIN_DIR)/coginfer $(BIN_DIR)/cograw: build-llama
	@mkdir -p $(BIN_DIR)
	@echo "  building with CGo (llama.cpp)"
	cd "$(CURDIR)"
	CGO_ENABLED=1 CGO_CFLAGS="-Ivendor/llama.cpp/ggml/include" \
		CGO_LDFLAGS="-Lvendor/llama.cpp/build/src -lllama $$(for lib in vendor/llama.cpp/build/libggml*.a; do libname=$$(basename "$$lib" .a | sed 's/^lib//'); echo -n " -l$$libname"; done) -lgomp" \
		go build -tags=cgo -ldflags="-s -w" -o $(BIN_DIR)/coginfer ./cmd/coginfer
	CGO_ENABLED=1 CGO_CFLAGS="-Ivendor/llama.cpp/ggml/include" \
		CGO_LDFLAGS="-Lvendor/llama.cpp/build/src -lllama $$(for lib in vendor/llama.cpp/build/libggml*.a; do libname=$$(basename "$$lib" .a | sed 's/^lib//'); echo -n " -l$$libname"; done) -lgomp" \
		go build -tags=cgo -ldflags="-s -w" -o $(BIN_DIR)/cograw ./cmd/cograw

pack: build
	@VERSION=$$(git describe --tags --abbrev=0 2>/dev/null || echo "dev")
	@CPM=/workspace/cpm/build/bin/cpm
	@mkdir -p $(BIN_DIR)
	@$${CPM} pack --bin $(BIN_DIR)/coginfer --name coginfer --version $$VERSION --os linux --arch amd64 --description "CognitiveOS Wide Model inference engine"
	@$${CPM} pack --bin $(BIN_DIR)/cograw --name cograw --version $$VERSION --os linux --arch amd64 --description "CognitiveOS Raw Model firmware guardrail"

publish: pack
	@if [ -z "$${REGISTRY_TOKEN}" ]; then \
		echo "  ERROR: REGISTRY_TOKEN not set"; exit 1; \
	fi
	@VERSION=$$(git describe --tags --abbrev=0 2>/dev/null || echo "dev")
	@for cgp in *.cgp; do \
		[ -f "$$cgp" ] || continue; \
		URL="https://github.com/CognitiveOS-Project/inference/releases/download/$$VERSION/$$(basename $$cgp)"; \
		/workspace/cpm/build/bin/cpm publish "$$cgp" --download-url "$$URL"; \
		rm "$$cgp"; \
	done

test:
	CGO_ENABLED=0 go test ./... -v -count=1

lint:
	shellcheck scripts/build.sh
	CGO_ENABLED=0 go vet ./... 

clean:
	rm -rf $(BUILD_DIR)
