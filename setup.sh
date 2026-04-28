#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "============================================"
echo "  MLX Local Inference Setup for Apple Silicon"
echo "============================================"
echo ""

# Require uv for fast dependency management
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    brew install uv || curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Step 1: Create venv with Python 3.12+
if [ -d "$VENV_DIR" ]; then
    EXISTING_VER=$("$VENV_DIR/bin/python3" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
    EXISTING_MINOR=$(echo "$EXISTING_VER" | cut -d. -f2)
    if [ "$EXISTING_MINOR" -lt 11 ]; then
        echo "  Existing venv has Python $EXISTING_VER (need 3.11+), recreating..."
        rm -rf "$VENV_DIR"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "[1/6] Creating virtual environment (Python 3.12)..."
    uv venv --python 3.12 "$VENV_DIR"
else
    echo "[1/6] Virtual environment already exists."
fi

source "$VENV_DIR/bin/activate"
echo "  Python: $(python3 --version)"

# Step 2: Install MLX stack + DFlash + DDTree
echo ""
echo "[2/6] Installing MLX inference stack + DFlash speculative decoding..."
uv pip install \
    "mlx>=0.25.0" \
    "mlx-lm>=0.31.2" \
    "dflash-mlx" \
    huggingface_hub \
    transformers \
    fastapi \
    "uvicorn[standard]" \
    httpx \
    pyobjc-framework-Cocoa

# ddtree-mlx is not on PyPI — install from GitHub
echo "  Installing ddtree-mlx from GitHub..."
uv pip install "ddtree-mlx @ git+https://github.com/humanrouter/ddtree-mlx.git" 2>/dev/null \
    || echo "  (ddtree-mlx install failed — optional, DFlash still works without it)"

MLX_VER=$(python3 -c 'import mlx.core; print(mlx.core.__version__)' 2>/dev/null || echo 'unknown')
MLX_LM_VER=$(python3 -c 'import mlx_lm; print(mlx_lm.__version__)' 2>/dev/null || echo 'unknown')
DFLASH_VER=$(python3 -c 'import dflash_mlx; print(dflash_mlx.__version__)' 2>/dev/null || echo 'NOT INSTALLED')
DDTREE_OK=$(python3 -c 'import ddtree_mlx; print("installed")' 2>/dev/null || echo 'NOT INSTALLED')
echo "  ✓ mlx $MLX_VER"
echo "  ✓ mlx-lm $MLX_LM_VER"
if [ "$DFLASH_VER" = "NOT INSTALLED" ]; then
    echo "  ✗ dflash-mlx NOT INSTALLED"
else
    echo "  ✓ dflash-mlx $DFLASH_VER"
fi
if [ "$DDTREE_OK" = "NOT INSTALLED" ]; then
    echo "  ⚠ ddtree-mlx not available (optional — DFlash works without it)"
else
    echo "  ✓ ddtree-mlx $DDTREE_OK"
fi
echo "  ✓ Native 4-bit KV cache quantization (built into mlx-lm)"

# Step 3: Dump system config
echo ""
echo "[3/6] Detecting system configuration..."
python3 "$SCRIPT_DIR/dump_system_config.py"

# Step 4: Run model selection optimizer (dense vs MoE aware)
echo ""
echo "[4/6] Running model selection optimizer..."
python3 "$SCRIPT_DIR/select_models.py"

# Step 5: Download core models (small + medium)
# Other tiers (mini, large, huge) download on-demand when first requested.
echo ""
echo "[5/6] Downloading core models from Hugging Face..."

python3 -c "
import json, sys, time, gc

with open('$SCRIPT_DIR/model_selection.json') as f:
    sel = json.load(f)

core_tiers = ['small', 'medium']
for cat in core_tiers:
    info = sel['selected_models'].get(cat)
    if not info:
        continue
    name = info['name']
    print(f'\n  [{cat}] Downloading: {name} (~{info[\"mem_gb\"]}GB)')
    sys.stdout.flush()

    from mlx_lm import load
    start = time.time()
    model, tokenizer = load(name)
    elapsed = time.time() - start
    print(f'  ✓ {name} ready ({elapsed:.1f}s)')

    del model, tokenizer
    gc.collect()

print('\n  Other tiers (mini, large, huge) will download on first use.')
"

# Step 6: Verify inference
echo ""
echo "[6/6] Running verification tests..."
python3 -c "
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import json

with open('$SCRIPT_DIR/model_selection.json') as f:
    sel = json.load(f)

small = sel['selected_models']['small']['name']
model, tokenizer = load(small)
prompt = tokenizer.apply_chat_template(
    [{'role': 'user', 'content': 'Say hello in one word.'}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
result = generate(model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)
print(f'  ✓ Inference: {result.strip()[:60]}')

result2 = generate(model, tokenizer, prompt=prompt, max_tokens=10,
                   sampler=make_sampler(temp=0.1), kv_bits=4, verbose=False)
print(f'  ✓ KV cache quantization (4-bit) verified')
"

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "  Start the server:"
echo "    ./start.sh                         # models load on-demand"
echo "    ./start.sh --preload small,medium  # pre-load at startup"
echo ""
echo "  Models: mini(0.5G) small(2.5G) medium(5.5G) large(14G) huge(20G)"
echo "  DFlash: 2-4x lossless speedup via block diffusion speculative decoding"
echo "  DDTree: +10-15% on top of DFlash for code/structured content"
echo "  Models load automatically on first request and evict after 15min idle."
echo ""
echo "  Dashboard: http://localhost:8800/stats/live"
echo "  API docs:  http://localhost:8800/docs"
echo ""
