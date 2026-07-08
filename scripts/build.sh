#!/bin/sh
set -euo pipefail

if ! command -v go &>/dev/null; then
    if [ ! -f /tmp/go/bin/go ]; then
        echo "Installing Go..."
        curl -sL https://go.dev/dl/go1.24.linux-amd64.tar.gz | tar -C /tmp -xz
    fi
    export PATH="/tmp/go/bin:$PATH"
fi

BUILD_DIR="$(cd "$(dirname "$0")/.." && pwd)/build"
BIN_DIR="${BUILD_DIR}/bin"
mkdir -p "${BIN_DIR}"

# Build llama.cpp if vendored and not already built
LLAMA_CPP_DIR="$(cd "$(dirname "$0")/.." && pwd)/vendor/llama.cpp"
CGO_FLAGS=""
if [ -f "${LLAMA_CPP_DIR}/CMakeLists.txt" ]; then
    if [ ! -f "${LLAMA_CPP_DIR}/build/libllama.a" ]; then
        echo "Building llama.cpp..."
        cd "${LLAMA_CPP_DIR}"
        cmake -B build -DLLAMA_NATIVE=0 \
            -DBUILD_SHARED_LIBS=0 -DLLAMA_BUILD_TESTS=0 \
            -DLLAMA_BUILD_EXAMPLES=0 -DLLAMA_BUILD_SERVER=0 \
            -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY="${LLAMA_CPP_DIR}/build"
        cmake --build build --config Release --target llama -j"$(nproc)"
        echo "  -> llama.cpp built"
    fi

    LLAMA_INC="${LLAMA_CPP_DIR}/ggml/include"
    CGO_LDFLAGS="-L${LLAMA_CPP_DIR}/build/src -lllama"
    while IFS= read -r lib; do
        libname=$(basename "${lib}" .a | sed 's/^lib//')
        CGO_LDFLAGS="${CGO_LDFLAGS} -l${libname}"
    done < <(find "${LLAMA_CPP_DIR}/build" -name "libggml*.a" -type f)
    CGO_FLAGS="CGO_ENABLED=1 CGO_CFLAGS=-I${LLAMA_INC} CGO_LDFLAGS=${CGO_LDFLAGS} -lgomp"
fi

echo "Building coginfer..."
eval "${CGO_FLAGS}" go build -tags=cgo -ldflags="-s -w" -o "${BIN_DIR}/cognitiveos-inference" ./cmd/coginfer
echo "  -> ${BIN_DIR}/cognitiveos-inference"

echo "Building cograw..."
eval "${CGO_FLAGS}" go build -tags=cgo -ldflags="-s -w" -o "${BIN_DIR}/cograw" ./cmd/cograw
echo "  -> ${BIN_DIR}/cograw"
