#!/usr/bin/env python3
"""
Model catalog and selection for MLX local inference.

Five tiers: mini, small, medium, large, huge — each with ranked candidates.
The top candidate per tier is the default; the server can load any model
from any tier on demand.

Dense vs MoE on Apple Silicon MLX:
  Dense wins for throughput per GB (3-10x better than MoE on Metal).
  MoE only justified when you need quality requiring >27B active params.
"""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "system_config.json")
SELECTION_PATH = os.path.join(SCRIPT_DIR, "model_selection.json")

# ═══════════════════════════════════════════════════════════════════
# Model Catalog — MLX models (updated Apr 2026)
# mini/small/medium: Qwen3.5 (Gated DeltaNet, 75% linear attention)
# large: Qwen3.5-27B (dense, dflash spec decode)
# huge: Qwen3.6-35B-A3B (MoE, latest open-weight Qwen)
#
# DFlash speculative decoding: block diffusion draft models from z-lab
# give 1.3-4.4x lossless speedup. Draft models are ~1B params each.
# ═══════════════════════════════════════════════════════════════════

# DFlash draft model registry — maps target HF IDs to their block diffusion
# draft models. Draft models are ~1B params trained specifically for each target.
# Set to None for models without a dflash draft (e.g. mini is already fast enough).
DFLASH_DRAFT_REGISTRY = {
    "mlx-community/Qwen3.5-4B-MLX-4bit": "z-lab/Qwen3.5-4B-DFlash",
    "mlx-community/Qwen3.5-9B-MLX-4bit": "z-lab/Qwen3.5-9B-DFlash",
    "mlx-community/Qwen3.5-27B-4bit": "z-lab/Qwen3.5-27B-DFlash",
    "mlx-community/Qwen3.5-35B-A3B-4bit": "z-lab/Qwen3.5-35B-A3B-DFlash",
    "mlx-community/Qwen3.6-35B-A3B-4bit": "z-lab/Qwen3.6-35B-A3B-DFlash",
}

MODEL_CATALOG = {
    # ── MINI: ultra-fast, embeddings, simple classification ──
    "mini": [
        {
            "name": "mlx-community/Qwen3.5-0.8B-MLX-4bit",
            "params_b": 0.8,
            "active_params_b": 0.8,
            "architecture": "dense",
            "quant": "4bit",
            "mem_gb": 0.5,
            "context_length": 262144,
            "tool_calling": True,
            "thinking_mode": True,
            "mmlu_approx": 58,
            "tool_calling_accuracy": 80.0,
            "strengths": [
                "ultra-fast inference (~160 tok/s on M1 Max)",
                "262K context",
                "0.5 GB — fits alongside any combination",
                "thinking/non-thinking dual mode",
            ],
            "capabilities": {
                "classification": "strong",
                "routing": "strong",
                "summarization": "good",
                "extraction": "strong",
                "code": "basic",
                "reasoning": "basic",
                "tool_calling": "good",
                "structured_output": "good",
                "math": "basic",
            },
        },
    ],

    # ── SMALL: fast routing, classification, tool calling ──
    "small": [
        {
            "name": "mlx-community/Qwen3.5-4B-MLX-4bit",
            "params_b": 4.0,
            "active_params_b": 4.0,
            "architecture": "dense",
            "quant": "4bit",
            "mem_gb": 2.5,
            "context_length": 262144,
            "tool_calling": True,
            "thinking_mode": True,
            "mmlu_approx": 76,
            "tool_calling_accuracy": 97.5,
            "strengths": [
                "97.5% tool calling accuracy (best in class)",
                "Gated DeltaNet: 75% linear attention, low KV cache",
                "262K native context",
                "thinking/non-thinking dual mode",
                "multilingual",
            ],
            "capabilities": {
                "classification": "excellent",
                "routing": "excellent",
                "summarization": "strong",
                "extraction": "excellent",
                "code": "strong",
                "reasoning": "strong",
                "tool_calling": "excellent",
                "structured_output": "excellent",
                "math": "strong",
            },
        },
        {
            "name": "mlx-community/Qwen3.5-2B-MLX-4bit",
            "params_b": 2.0,
            "active_params_b": 2.0,
            "architecture": "dense",
            "quant": "4bit",
            "mem_gb": 1.2,
            "context_length": 262144,
            "tool_calling": True,
            "thinking_mode": True,
            "mmlu_approx": 68,
            "tool_calling_accuracy": 90.0,
            "strengths": [
                "very fast (~100 tok/s on M1 Max)",
                "262K context",
                "1.2 GB footprint",
            ],
            "capabilities": {
                "classification": "strong",
                "routing": "strong",
                "summarization": "strong",
                "extraction": "strong",
                "code": "good",
                "reasoning": "good",
                "tool_calling": "strong",
                "structured_output": "strong",
                "math": "good",
            },
        },
    ],

    # ── MEDIUM: agentic workhorse, tool calling, planning ──
    "medium": [
        {
            "name": "mlx-community/Qwen3.5-9B-MLX-4bit",
            "params_b": 9.0,
            "active_params_b": 9.0,
            "architecture": "dense",
            "quant": "4bit",
            "mem_gb": 5.5,
            "context_length": 262144,
            "tool_calling": True,
            "thinking_mode": True,
            "mmlu_approx": 82,
            "tool_calling_accuracy": 97.5,
            "strengths": [
                "97.5% tool calling accuracy (best in class)",
                "Gated DeltaNet: 75% linear attention, low KV cache",
                "262K native context",
                "MMLU-Pro 82.5, GPQA Diamond 81.7",
                "excellent agentic and multi-step reasoning",
                "multilingual",
            ],
            "capabilities": {
                "classification": "excellent",
                "routing": "excellent",
                "summarization": "excellent",
                "extraction": "excellent",
                "code": "excellent",
                "reasoning": "excellent",
                "tool_calling": "exceptional",
                "structured_output": "excellent",
                "math": "excellent",
                "planning": "excellent",
                "agentic": "excellent",
            },
        },
    ],

    # ── LARGE: deep reasoning + agentic, with dflash spec decode ──
    # Qwen3.5-27B (base, not Opus-distilled). The Opus-distilled version had
    # 100% JSON parse failure; the base model handles structured output fine.
    # With dflash speculative decoding: 1.3-2.4x speedup.
    "large": [
        {
            "name": "mlx-community/Qwen3.5-27B-4bit",
            "params_b": 27.0,
            "active_params_b": 27.0,
            "architecture": "dense",
            "quant": "4bit",
            "mem_gb": 14.0,
            "context_length": 262144,
            "tool_calling": True,
            "thinking_mode": True,
            "mmlu_approx": 85,
            "tool_calling_accuracy": 97.5,
            "strengths": [
                "97.5% tool calling accuracy (best in class at this scale)",
                "GPQA Diamond 85.5, AIME 91.3%",
                "Gated DeltaNet: 75% linear attention, low KV cache",
                "dflash spec decode: 1.3-2.4x speedup (z-lab/Qwen3.5-27B-DFlash)",
                "ddtree tree-based decode: +10-15% on top of dflash for code",
                "262K native context",
            ],
            "capabilities": {
                "classification": "excellent",
                "routing": "excellent",
                "summarization": "excellent",
                "extraction": "excellent",
                "code": "exceptional",
                "reasoning": "exceptional",
                "tool_calling": "exceptional",
                "structured_output": "excellent",
                "math": "exceptional",
                "planning": "exceptional",
                "agentic": "exceptional",
                "analysis": "exceptional",
            },
        },
    ],

    # ── HUGE: maximum quality, Qwen 3.6 (latest open-weight) ──
    # Qwen3.6-35B-A3B: vision+text MoE, 256 experts, 8 routed + 1 shared.
    # SWE-bench 73.4%, AIME 92.7%, GPQA 86.0%, MCPMark 37%.
    # Uses qwen3_coder tool parser. dflash draft: 1.3-2.2x speedup.
    "huge": [
        {
            "name": "mlx-community/Qwen3.6-35B-A3B-4bit",
            "params_b": 35.0,
            "active_params_b": 3.0,
            "architecture": "moe",
            "quant": "4bit",
            "mem_gb": 20.0,
            "context_length": 262144,
            "tool_calling": True,
            "thinking_mode": True,
            "mmlu_approx": 93,
            "tool_calling_accuracy": 97.0,
            "strengths": [
                "Qwen 3.6 — latest open-weight, released Apr 2026",
                "SWE-bench Verified 73.4%, AIME 2026 92.7%, GPQA Diamond 86.0%",
                "MCPMark tool use 37% (2x Gemma 4-31B's 18.1%)",
                "MoE: 256 experts, 3B active — quality of 35B, compute of 3B",
                "dflash spec decode: 1.3-2.2x speedup (z-lab/Qwen3.6-35B-A3B-DFlash)",
                "262K native, extensible to 1M tokens via YaRN",
                "agentic coding: repository-level reasoning, frontend workflows",
            ],
            "capabilities": {
                "classification": "excellent",
                "routing": "excellent",
                "summarization": "excellent",
                "extraction": "excellent",
                "code": "exceptional",
                "reasoning": "exceptional",
                "tool_calling": "exceptional",
                "structured_output": "excellent",
                "math": "exceptional",
                "planning": "exceptional",
                "agentic": "exceptional",
                "analysis": "exceptional",
            },
        },
    ],
}

# Ordered from smallest to largest for memory management
TIER_ORDER = ["mini", "small", "medium", "large", "huge"]


def get_default_model(tier: str) -> dict | None:
    """Return the top (first) candidate for a tier, or None if tier unknown."""
    candidates = MODEL_CATALOG.get(tier)
    if not candidates:
        return None
    return candidates[0]


def get_all_defaults() -> dict[str, dict]:
    """Return {tier: top_candidate} for all tiers."""
    return {tier: candidates[0] for tier, candidates in MODEL_CATALOG.items()}


def load_system_config():
    if not os.path.exists(CONFIG_PATH):
        print(f"ERROR: {CONFIG_PATH} not found. Run dump_system_config.py first.")
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        return json.load(f)


def select_models(config):
    """Run the model selection optimizer and write model_selection.json."""
    available_mem = config["memory"]["available_for_models_gb"]
    bandwidth = config["bandwidth_gbps"]

    print(f"\n{'='*65}")
    print(f"  Model Catalog — 5 Tiers")
    print(f"{'='*65}")
    print(f"  Chip:              {config['chip']['brand']}")
    print(f"  Available memory:  {available_mem} GB")
    print(f"  Memory bandwidth:  {bandwidth} GB/s")
    print(f"{'='*65}\n")

    selections = {}
    for tier in TIER_ORDER:
        default = get_default_model(tier)
        if default:
            selections[tier] = default
            tps_est = bandwidth / default["mem_gb"]
            print(f"  {tier:8s} → {default['name']:60s} {default['mem_gb']:5.1f} GB  ~{tps_est:.0f} tok/s")

    total_mem = sum(m["mem_gb"] for m in selections.values())

    result = {
        "system_summary": {
            "chip": config["chip"]["brand"],
            "memory_gb": config["memory"]["total_gb"],
            "bandwidth_gbps": bandwidth,
            "available_for_models_gb": available_mem,
        },
        "catalog": {
            tier: {
                "default": candidates[0]["name"],
                "candidates": [c["name"] for c in candidates],
                "default_mem_gb": candidates[0]["mem_gb"],
            }
            for tier, candidates in MODEL_CATALOG.items()
        },
        "selected_models": {
            tier: {
                "name": m["name"],
                "params_b": m["params_b"],
                "active_params_b": m["active_params_b"],
                "architecture": m["architecture"],
                "quant": m["quant"],
                "mem_gb": m["mem_gb"],
                "context_length": m["context_length"],
                "tool_calling": m.get("tool_calling", False),
                "thinking_mode": m.get("thinking_mode", False),
                "capabilities": m["capabilities"],
                "strengths": m["strengths"],
                "estimated_tok_per_sec": round(bandwidth / m["mem_gb"], 1),
            }
            for tier, m in selections.items()
        },
        "deployment": {
            "all_models_total_gb": round(total_mem, 1),
            "memory_available_gb": available_mem,
            "note": "Models are loaded on-demand, not all at once. "
                    "The server manages memory dynamically with LRU eviction.",
        },
    }

    with open(SELECTION_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Total if all loaded: {total_mem:.1f} GB / {available_mem} GB")
    print(f"  (Models load on-demand — not all at once)")
    print(f"  Written to: {SELECTION_PATH}")
    print(f"{'='*65}")

    return result


def main():
    config = load_system_config()
    select_models(config)


if __name__ == "__main__":
    main()
