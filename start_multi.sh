#!/usr/bin/env bash
set -euo pipefail

# Multi-instance serving: one serve.py per model on separate ports.
#
# When to use multi-process vs the default single-process dynamic server:
#
#   Workload type                         → Recommendation
#   ──────────────────────────────────────────────────────
#   Single model at a time                → single-process (default)
#   Mixed mini+small, high concurrency    → multi-process (this script)
#   Mixed medium+large, high concurrency  → single-process with locked-eval
#
# Why: Apple Silicon has one GPU, and MLX serializes individual `mx.eval()`
# calls. In single-process mode a threading lock protects eval(), which lets
# CPU graph building overlap across threads but still queues GPU submissions
# one at a time. Small Metal commands from `mini` end up stuck behind large
# ones from `small`, so mini latency inflates to match small's.
#
# Running the two models in separate processes lets Metal's own scheduler
# interleave at finer granularity — effectively removing head-of-line blocking.
#
# Empirical results (M1 Max, 4-way concurrent, 10 mini + 10 small, 64 tokens):
#
#   Mode            Wall    Throughput  Mini p50   Small p50
#   single-process  9.78s    73.7 t/s    1.88s      1.89s
#   multi-process   7.38s    97.7 t/s    0.53s      2.14s   ← this script
#
# Multi-process wins ~25% aggregate throughput on mixed mini+small workloads.
# Mini requests are 3.5× faster per-request because they no longer wait on
# small's eval serialization. Memory overhead is ~350 MB (a second Python
# interpreter + MLX JIT cache).
#
# For medium/large models where generation dominates and a single request
# already saturates the GPU, the multi-process speedup disappears and the
# single-process dynamic server (./start.sh) is the simpler choice.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Run setup.sh first."
    exit 1
fi

source "$VENV_DIR/bin/activate"

# Default: fast-tier pair (mini + small) — the combination that benefits most
# from multi-process. Override by setting MODELS="tier:port,tier:port,..."
# A routing proxy on PROXY_PORT (default 8800) gives clients a unified endpoint
# while each model runs as a separate process for true GPU parallelism.
MODELS="${MODELS:-mini:8810,small:8811}"
PROXY_PORT="${PROXY_PORT:-8800}"

declare -a PIDS=()

cleanup() {
    echo ""
    echo "Shutting down all instances..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo "All instances stopped."
}
trap cleanup EXIT INT TERM

echo "============================================"
echo "  MLX Multi-Instance Serving"
echo "============================================"
echo ""

LOG_DIR="${LOG_DIR:-/tmp/mlx_multi}"
mkdir -p "$LOG_DIR"
echo "  Logs: $LOG_DIR/"

IFS=',' read -ra PAIRS <<< "$MODELS"
for pair in "${PAIRS[@]}"; do
    tier="${pair%%:*}"
    port="${pair##*:}"
    LOG_FILE="$LOG_DIR/${tier}_${port}.log"
    echo "  $tier → http://localhost:$port  (log: $LOG_FILE)"
    DEBUG_REQUESTS="${DEBUG_REQUESTS:-1}" STRIP_THINK=1 DFLASH=0 DDTREE=0 \
        MLX_PORT="$port" python3 "$SCRIPT_DIR/serve.py" --preload "$tier" \
        > "$LOG_FILE" 2>&1 &
    PIDS+=($!)
done

# Wait briefly for backends to start binding their ports
sleep 2

echo ""
PROXY_LOG="$LOG_DIR/proxy_${PROXY_PORT}.log"
echo "  proxy → http://localhost:$PROXY_PORT (unified endpoint, log: $PROXY_LOG)"
ROUTES="$MODELS" PROXY_PORT="$PROXY_PORT" \
    python3 "$SCRIPT_DIR/route_proxy.py" > "$PROXY_LOG" 2>&1 &
PIDS+=($!)

echo ""
echo "${#PIDS[@]} processes started (${#PAIRS[@]} model workers + 1 proxy)"
echo "Point clients (Hermes etc.) at http://localhost:$PROXY_PORT"
echo "(Ctrl+C to stop all)"
wait
