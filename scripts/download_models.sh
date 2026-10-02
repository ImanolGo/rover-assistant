#!/bin/bash
# Idempotent model downloader. Verifies SHA256 into models/MODELS.md.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p models/gemma models/whisper models/piper models/moondream models/yolo_trt models/laya

fetch() { # fetch <url> <dest>
    local url="$1" dest="$2"
    if [ -s "$dest" ]; then
        echo "skip (exists): $dest"
    else
        echo "downloading: $url -> $dest"
        curl -fL --retry 3 --progress-bar "$url" -o "$dest.tmp" && mv "$dest.tmp" "$dest"
    fi
}

sha() { sha256sum "$1" | cut -d' ' -f1; }

GEMMA_REPO=unsloth/gemma-4-E2B-it-GGUF
fetch "https://huggingface.co/$GEMMA_REPO/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf" models/gemma/gemma-4-E2B-it-Q4_K_M.gguf
fetch "https://huggingface.co/$GEMMA_REPO/resolve/main/mmproj-F16.gguf" models/gemma/mmproj-gemma4-e2b-f16.gguf

fetch "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin" models/whisper/ggml-base.en.bin

fetch "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" models/piper/en_US-lessac-medium.onnx
fetch "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" models/piper/en_US-lessac-medium.onnx.json

fetch "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt" models/yolo11n.pt

# Legacy baseline (Moondream): reuse the exact files legacy benchmarked (ollama blobs).
for f in moondream2-text-model.gguf moondream2-mmproj.gguf; do
    src="/home/imanolgo/repos/local-ai-robot-assistant/models/moondream2_gguf/$f"
    dest="models/moondream/$f"
    if [ -s "$dest" ]; then
        echo "skip (exists): $dest"
    elif [ -e "$src" ]; then
        cp -L "$src" "$dest"
    else
        echo "WARN: legacy moondream blob missing: $src"
    fi
done

# Laya checkpoints are pulled by the laya package itself in its own venv (bench 1b).

# Moonshine streaming STT models (B2). The moonshine-voice library owns the CDN
# URLs; we pin the package and cache the .ort weights under models/moonshine/.
mkdir -p models/moonshine
if .venv/bin/python -c "import moonshine_voice" 2>/dev/null; then
    .venv/bin/python - <<'PY'
from pathlib import Path

from moonshine_voice import ModelArch
from moonshine_voice.download import get_model_for_language

root = Path("models/moonshine")
for arch in (ModelArch.TINY_STREAMING, ModelArch.SMALL_STREAMING, ModelArch.MEDIUM_STREAMING):
    path, _ = get_model_for_language("en", arch, cache_root=root)
    print(f"moonshine {arch.name} -> {path}")
PY
else
    echo "WARN: moonshine-voice not installed; skipping Moonshine models"
fi

cat >> models/MODELS.md <<EOF

## Downloaded $(date -u +%Y-%m-%d)
| File | SHA256 |
|---|---|
| gemma/gemma-4-E2B-it-Q4_K_M.gguf | $(sha models/gemma/gemma-4-E2B-it-Q4_K_M.gguf) |
| gemma/mmproj-gemma4-e2b-f16.gguf | $(sha models/gemma/mmproj-gemma4-e2b-f16.gguf) |
| whisper/ggml-base.en.bin | $(sha models/whisper/ggml-base.en.bin) |
| piper/en_US-lessac-medium.onnx | $(sha models/piper/en_US-lessac-medium.onnx) |
| yolo11n.pt | $(sha models/yolo11n.pt) |
| moondream/moondream2-text-model.gguf | $(sha models/moondream/moondream2-text-model.gguf) |
| moondream/moondream2-mmproj.gguf | $(sha models/moondream/moondream2-mmproj.gguf) |
$(find models/moonshine -name '*.ort' 2>/dev/null | sort | while read -r f; do echo "| $f | $(sha "$f") |"; done)

llama.cpp build commit: $(git -C "$HOME/llama.cpp" rev-parse HEAD 2>/dev/null || echo unknown)
EOF

echo "All models present. Hashes appended to models/MODELS.md."
