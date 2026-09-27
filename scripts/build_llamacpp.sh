#!/bin/bash
# Build llama.cpp (CUDA, Orin sm_87) at a pinned commit. Record the hash in models/MODELS.md.
set -euo pipefail

COMMIT=d2e54583c7452353eb35d40431281f6ee984332f  # ggml-org/llama.cpp v0.5.0
DEST="$HOME/llama.cpp"

if [ ! -d "$DEST/.git" ]; then
    git clone https://github.com/ggml-org/llama.cpp "$DEST"
fi
cd "$DEST"
git fetch origin "$COMMIT" --quiet
git checkout --quiet "$COMMIT"
echo "Building llama.cpp at $(git rev-parse HEAD)"

cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DGGML_NATIVE=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4 --target llama-server llama-cli llama-mtmd-cli

echo "Done: $DEST/build/bin/{llama-server,llama-cli,llama-mtmd-cli} at $(git rev-parse --short HEAD)"
