#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Run setup.sh first."
    exit 1
fi

source "$VENV_DIR/bin/activate"

# All target models + DFlash draft models
MODELS=(
    # Target models (5 tiers)
    "mlx-community/Qwen3.5-0.8B-MLX-4bit"
    "mlx-community/Qwen3.5-4B-MLX-4bit"
    "mlx-community/Qwen3.5-9B-MLX-4bit"
    "mlx-community/Qwen3.5-27B-4bit"
    "mlx-community/Qwen3.6-35B-A3B-4bit"
    # DFlash draft models
    "z-lab/Qwen3.5-4B-DFlash"
    "z-lab/Qwen3.5-9B-DFlash"
    "z-lab/Qwen3.5-27B-DFlash"
    "z-lab/Qwen3.6-35B-A3B-DFlash"
)

echo "============================================"
echo "  Download All Models"
echo "============================================"
echo ""
echo "  ${#MODELS[@]} models to check/download."
echo ""

for model in "${MODELS[@]}"; do
    echo "--- $model ---"
    python3 -c "
import socket
# Force IPv4 — HF CloudFront IPv6 hangs on SYN_SENT
_orig = socket.getaddrinfo
def ipv4_only(*args, **kwargs):
    return [r for r in _orig(*args, **kwargs) if r[0] == socket.AF_INET]
socket.getaddrinfo = ipv4_only

from huggingface_hub import snapshot_download, try_to_load_from_cache
import os, sys

model_id = '$model'

# Check if already cached by looking for config.json
cached = try_to_load_from_cache(model_id, 'config.json')
if cached is not None:
    cache_dir = os.path.dirname(cached)
    print(f'  Already downloaded: {cache_dir}')
    sys.exit(0)

print(f'  Downloading...')
path = snapshot_download(model_id, max_workers=4)
print(f'  Done: {path}')
"
    echo ""
done

echo "============================================"
echo "  All models downloaded."
echo "============================================"
