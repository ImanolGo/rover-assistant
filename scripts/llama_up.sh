#!/bin/bash
# Start llama-server with drop_caches and a retry loop.
# Pulled forward from Phase 7 (doctor.sh): on JetPack the mmproj load fails with
# NvMapMemAllocInternalTagged error 12 on many boots and needs drop_caches + a
# retry (STATUS known issue 1). Usage: scripts/llama_up.sh [tmux-session-name]
set -euo pipefail
cd "$(dirname "$0")/.."

BIN="${LLAMA_BIN:-$HOME/llama.cpp/build/bin/llama-server}"
MODEL="models/gemma/gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ="models/gemma/mmproj-gemma4-e2b-q8_0.gguf"
ARGS="-c 2048 -ngl 99 --flash-attn on --jinja -np 1 --cache-ram 0 --image-min-tokens 70 --image-max-tokens 70 --host 127.0.0.1 --port 8080"
LOG="bench/results/raw/logs/llama.log"
SESSION="${1:-llama}"
HEALTH_URL="http://127.0.0.1:8080/health"

mkdir -p "$(dirname "$LOG")"

drop_caches() {
    sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory' \
        2>/dev/null || true
}

health() { curl -sf --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; }

start_once() {
    if command -v tmux >/dev/null; then
        tmux kill-session -t "$SESSION" 2>/dev/null || true
        tmux new-session -d -s "$SESSION" \
            "cd $(pwd) && '$BIN' -m $MODEL --mmproj $MMPROJ $ARGS > '$LOG' 2>&1"
    else
        "$BIN" -m "$MODEL" --mmproj "$MMPROJ" $ARGS >"$LOG" 2>&1 &
    fi
}

for attempt in 1 2 3 4; do
    drop_caches
    start_once
    for _ in $(seq 1 60); do
        if health; then
            sleep 5
            if health; then
                echo "llama-server healthy (attempt $attempt)"
                exit 0
            fi
            break
        fi
        sleep 1
    done
    echo "llama-server attempt $attempt failed; retrying" >&2
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    sleep 2
done

echo "llama-server failed to start after 4 attempts; see $LOG" >&2
exit 1
