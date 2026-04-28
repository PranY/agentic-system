#!/usr/bin/env python3
"""
Dumps Apple Silicon system configuration to JSON for model selection optimization.
Detects chip type, memory, GPU cores, and estimates unified memory bandwidth.
"""

import json
import subprocess
import re
import os
import sys


def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
    except subprocess.CalledProcessError:
        return ""


def get_chip_info():
    brand = run_cmd("sysctl -n machdep.cpu.brand_string")
    # Extract chip family (M1, M2, M3, M4) and variant (Pro, Max, Ultra)
    chip_match = re.search(r"Apple (M\d+)(\s+(Pro|Max|Ultra))?", brand)
    if chip_match:
        chip_gen = chip_match.group(1)
        chip_variant = chip_match.group(3) or "Base"
    else:
        chip_gen = "Unknown"
        chip_variant = "Unknown"
    return brand, chip_gen, chip_variant


def get_memory_gb():
    mem_bytes = int(run_cmd("sysctl -n hw.memsize") or 0)
    return mem_bytes / (1024 ** 3)


def get_gpu_cores():
    gpu_info = run_cmd("system_profiler SPDisplaysDataType 2>/dev/null")
    cores_match = re.search(r"Total Number of Cores:\s*(\d+)", gpu_info)
    return int(cores_match.group(1)) if cores_match else 0


def get_cpu_cores():
    perf = int(run_cmd("sysctl -n hw.perflevel0.logicalcpu") or run_cmd("sysctl -n hw.ncpu") or "0")
    eff = int(run_cmd("sysctl -n hw.perflevel1.logicalcpu") or "0")
    total = int(run_cmd("sysctl -n hw.ncpu") or "0")
    return {"performance": perf, "efficiency": eff, "total": total}


def estimate_memory_bandwidth_gbps(chip_gen, chip_variant):
    """Estimate unified memory bandwidth in GB/s based on chip."""
    bandwidth_map = {
        ("M1", "Base"): 68.25,
        ("M1", "Pro"): 200,
        ("M1", "Max"): 400,
        ("M1", "Ultra"): 800,
        ("M2", "Base"): 100,
        ("M2", "Pro"): 200,
        ("M2", "Max"): 400,
        ("M2", "Ultra"): 800,
        ("M3", "Base"): 100,
        ("M3", "Pro"): 150,
        ("M3", "Max"): 400,
        ("M3", "Ultra"): 800,
        ("M4", "Base"): 120,
        ("M4", "Pro"): 273,
        ("M4", "Max"): 546,
        ("M4", "Ultra"): 819,
    }
    return bandwidth_map.get((chip_gen, chip_variant), 100)


def estimate_max_concurrent_models(memory_gb, bandwidth_gbps):
    """
    Estimate how many models can run concurrently with reasonable throughput.
    MLX loads models into unified memory. We need to reserve ~4GB for OS + overhead.
    For decent token/s, each model needs ~50 GB/s bandwidth minimum.
    """
    available_memory_gb = memory_gb - 4
    # Bandwidth-limited concurrency (each model needs ~50 GB/s for decent speed)
    bandwidth_concurrency = max(1, int(bandwidth_gbps / 50))
    # Memory-limited concurrency depends on model sizes (estimated later)
    return {
        "available_memory_gb": round(available_memory_gb, 1),
        "bandwidth_gbps": bandwidth_gbps,
        "max_bandwidth_limited_slots": bandwidth_concurrency,
    }


def main():
    brand, chip_gen, chip_variant = get_chip_info()
    memory_gb = get_memory_gb()
    gpu_cores = get_gpu_cores()
    cpu_cores = get_cpu_cores()
    bandwidth = estimate_memory_bandwidth_gbps(chip_gen, chip_variant)
    concurrency = estimate_max_concurrent_models(memory_gb, bandwidth)

    config = {
        "chip": {
            "brand": brand,
            "generation": chip_gen,
            "variant": chip_variant,
        },
        "memory": {
            "total_gb": round(memory_gb, 1),
            "available_for_models_gb": concurrency["available_memory_gb"],
        },
        "gpu": {
            "cores": gpu_cores,
        },
        "cpu": cpu_cores,
        "bandwidth_gbps": bandwidth,
        "concurrency": {
            "max_bandwidth_limited_slots": concurrency["max_bandwidth_limited_slots"],
            "note": "Actual concurrency depends on selected model sizes fitting in available memory",
        },
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "system_config.json")
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"System config written to {out_path}")
    print(json.dumps(config, indent=2))
    return config


if __name__ == "__main__":
    main()
