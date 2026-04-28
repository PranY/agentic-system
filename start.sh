#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Run setup.sh first."
    exit 1
fi

source "$VENV_DIR/bin/activate"

PORT="${MLX_PORT:-8800}"

# Default: pre-load `huge` (Qwen3.6-35B-A3B MoE) — top quality + fast generation
# (only 3B active params per token, so generation is much faster than dense 27B).
# Override with --preload to pick something else, or use --preload "" to disable.
#
# Examples:
#   ./start.sh                              # pre-loads huge by default
#   ./start.sh --preload large              # use 27B dense instead
#   ./start.sh --preload mini,small,huge    # multiple
#   ./start.sh --preload ""                 # no preload, all on-demand

# If user passes --preload anywhere in args, respect it; otherwise default to huge.
HAS_PRELOAD=0
for arg in "$@"; do
    if [ "$arg" = "--preload" ]; then
        HAS_PRELOAD=1
        break
    fi
done

if [ "$HAS_PRELOAD" -eq 0 ]; then
    set -- --preload huge "$@"
fi

exec python3 "$SCRIPT_DIR/serve.py" --port "$PORT" "$@"
