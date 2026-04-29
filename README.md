# Local Agentic Inference on Apple Silicon

A local OpenAI-compatible inference server for Apple Silicon (M1/M2/M3/M4) that powers AI agent harnesses like [Hermes Agent](https://hermes-agent.nousresearch.com/). Five Qwen tiers from 0.8B to 35B, dynamic on-demand loading, OpenAI-compatible streaming + tool calling, multi-process mode for true GPU parallelism.

Built on [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm). No cloud, no API keys, your hardware only.

---

## Why this exists

OpenAI-compatible servers for Apple Silicon (Ollama, LM Studio, plain `mlx_lm.server`) work, but most fall short for agent harnesses that:
- Send `stream: true` for the chat loop (most expect streaming)
- Send `tools: [...]` and expect OpenAI-format `tool_calls` deltas back
- Probe `GET /v1/models/{id}` and other auxiliary endpoints
- Hit several different model tiers concurrently in one session

This server is a thin focused layer that handles all of that, plus dynamic loading/unloading so you can keep five model tiers available without holding all of them in memory at once.

---

## Hardware requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- 32 GB unified memory minimum (64 GB recommended for `large` and `huge`)
- ~50 GB free disk for all five tiers cached

---

## Quick start

```bash
# 1. Clone and enter
git clone <this-repo> agentic-system && cd agentic-system

# 2. One-time setup: install mlx-lm + dependencies, detect hardware, pick models
./setup.sh

# 3. (Optional) Pre-download all models — recommended on first install (~50 GB total)
./download_models.sh

# 4. Start the server (auto-preloads `large` for fast first response)
./start.sh

# Server is now at http://localhost:8800/v1 — OpenAI-compatible
# Live dashboard: http://localhost:8800/stats/live
```

Test it:

```bash
curl http://localhost:8800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "small",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

---

## Model tiers

| Tier | Model | Memory | When to use |
|---|---|---|---|
| `mini` | Qwen3.5-0.8B (4-bit) | 0.5 GB | Routing, classification, ultra-fast subagents, conversation titles |
| `small` | Qwen3.5-4B (4-bit) | 2.5 GB | Fast tool calling (97.5% accuracy), web extract, vision |
| `medium` | Qwen3.5-9B (4-bit) | 5.5 GB | Agentic workhorse, MMLU-Pro 82.5 |
| `large` | Qwen3.5-27B (4-bit) | 14 GB | **Default for synthesis-heavy work.** GPQA 85.5, AIME 91.3 |
| `huge` | Qwen3.6-35B-A3B (4-bit) | 20 GB | Latest Qwen (Apr 2026). SWE-bench 73.4. MoE — 3B active |

Models load on-demand and unload after 15 minutes idle. The catalog is in [`select_models.py`](select_models.py).

> **Note on model size vs quality.** Smaller local models (≤9B) hallucinate placeholders (`[Title]`, `[Authors]`) when tool outputs are incomplete. They can call tools correctly but can't reason about gaps. For agent loops doing real synthesis, default to `large` or `huge`. Tool-calling correctness ≠ output usefulness.

---

## Setting up Hermes Agent (or any OpenAI-compatible client)

[Hermes Agent](https://github.com/NousResearch/hermes-agent) is a self-improving AI agent framework from Nous Research. It speaks the OpenAI API and consumes our local server natively. Setup walkthrough:

### 1. Install Hermes

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

### 2. Copy our pre-tuned config + agent persona

The repo includes [`hermes-config.yaml.example`](hermes-config.yaml.example) (multi-tier model routing) and [`hermes-SOUL.md.example`](hermes-SOUL.md.example) (an agent persona that prevents tool-selection death loops on local models):

```bash
mkdir -p ~/.hermes
cp hermes-config.yaml.example ~/.hermes/config.yaml
cp hermes-SOUL.md.example ~/.hermes/SOUL.md
```

The config maps Hermes's roles to local tiers:
- **Main agent** → `huge` (Qwen3.6-35B-A3B MoE) — fast generation despite size, since only 3B params are active per token
- **Fallback** → `large` (Qwen3.5-27B dense) — for cases where huge gets stuck
- **Subagent delegation** → `small` (4B) — fast for narrow scoped subtasks
- **Auxiliary** (titles, compression, vision, web extract, session search) → `mini` or `small`

The SOUL.md adds tool-routing guardrails: prefer structured skills (`arxiv`, `web_search`, `web_extract`) over raw `curl` chains, fail-fast after 2 retries, no fabricated table entries. Without this, local agents tend to spin in `curl` loops trying to scrape data when a proper skill exists.

### 3. Tell Hermes the endpoint

```bash
cat > ~/.hermes/.env <<EOF
OPENAI_API_KEY=local
OPENAI_BASE_URL=http://localhost:8800/v1
EOF
```

### 4. Run

```bash
# Terminal 1: start the inference server (preloads `large`)
./start.sh

# Terminal 2: start Hermes
hermes
```

That's it. You now have a fully local AI agent with self-improving skills, three-layer memory, MCP support, terminal access, and 100+ bundled tools — all running on your Mac, no cloud.

### Multi-process mode (if you want true parallel models)

If you want `mini`, `small`, and `medium` to run concurrently (e.g. medium handling tool calls while mini extracts content), use `start_multi.sh` instead. It launches one process per tier plus a routing proxy:

```bash
MODELS="mini:8810,small:8811,medium:8812" ./start_multi.sh
```

Hermes still connects to `http://localhost:8800/v1` — the proxy routes by model name. ~25% aggregate throughput win on mixed mini+small workloads. See `USAGE.md` for details.

---

## Documentation

- **[USAGE.md](USAGE.md)** — full API reference, env vars, streaming, tool calling, prefix cache, DFlash speculative decoding
- **[hermes-config.yaml.example](hermes-config.yaml.example)** — Hermes config template

---

## Other clients

Anything OpenAI-compatible works. Set `base_url` to `http://localhost:8800/v1` and pick a tier name as the model:

- **Continue** (VS Code extension): set provider to OpenAI, base URL `http://localhost:8800/v1`, model `medium`
- **Aider**: `aider --model openai/medium --openai-api-base http://localhost:8800/v1`
- **Open WebUI**: add a custom OpenAI provider
- **Python OpenAI SDK**: `OpenAI(base_url="http://localhost:8800/v1", api_key="local")`

---

## How this compares to other Apple Silicon inference servers

Several open-source projects already do local MLX inference. Quick honest comparison so you can pick the right one:

### [oMLX](https://github.com/jundot/omlx) — feature-rich, paged SSD KV cache

oMLX is a more sophisticated server: paged SSD KV caching (KV blocks persist to disk and survive restarts), continuous batching, native macOS menu-bar app, admin web UI, support for VLMs / OCR / embeddings / rerankers. Apache 2.0, actively maintained.

We measured both head-to-head (M1 Max, Qwen3.5-9B, fresh state). The two cache strategies optimize for different workloads:

| Workload | this server | oMLX |
|---|---|---|
| Multi-turn agent loop, growing context (Hermes-shape) | **5× speedup** on cached turns | no per-turn improvement |
| Identical long prompt repeated (best-of-N shape) | no cache | **3.6× speedup** on repeats |
| Tool calling on Qwen3.5 | works, lean SSE output | works, more verbose SSE output |
| Tool calling on Qwen3.6 | **works** | **broken** — emits 0 `tool_calls` chunks (parser doesn't handle `qwen3_coder` format yet) |
| VLM / OCR / embeddings | not supported | supported |
| Persistent cache across restarts | no | yes (SSD-paged) |

**Use this server if** you're running Hermes-style agent loops with stable system prompts, want multi-tier semantic routing (`mini`/`small`/`medium`/`large`/`huge`) baked in, and need Qwen3.6 tool calling to work reliably.

**Use oMLX if** you do a lot of best-of-N evaluation, batch RAG with identical retrieved-chunk prefixes, need VLM/OCR/embedding support, want persistent cache across restarts, or prefer a more polished GUI.

**Use both** if you have a workload that mixes these patterns — e.g. an outer GEPA optimization loop (best-of-N shape) wrapping inner agent runs (Hermes-shape). A small routing proxy in front of both can pick per-request which backend's cache will land. We use that pattern for an upcoming simulator project; the recipe is in the sandbox repo.

### [vllm-metal](https://github.com/vllm-project/vllm-metal) and [vllm-mlx](https://github.com/waybarrios/vllm-mlx)

Two paths to vLLM on Apple Silicon. Strong on continuous batching at high concurrency. As of April 2026, vllm-metal disables automatic prefix caching for hybrid Mamba/GatedDeltaNet models (Qwen3.5/3.6 are hybrids), which is the single biggest server-side optimization for agent loops. Worth tracking — when that lands, vllm-metal becomes a serious alternative for production.

### [Ollama](https://ollama.com), [llama.cpp](https://github.com/ggml-org/llama.cpp), [mlx_lm.server](https://github.com/ml-explore/mlx-lm)

Single-model servers. Mature, fast, well-tested. Right answer if you only need one model behind one endpoint. Don't support multi-tier dynamic loading — to serve five tiers you'd run five processes plus your own router.

---

## License & contributions

[See LICENSE]. PRs welcome — particularly for support of newer Qwen releases, additional MLX tool parsers, and benchmarks on different Apple Silicon variants.

## Acknowledgements

- [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) by Apple
- [Qwen](https://huggingface.co/Qwen) team for the model family
- [mlx-community](https://huggingface.co/mlx-community) for the 4-bit MLX conversions
- [Nous Research](https://hermes-agent.nousresearch.com/) for Hermes Agent
- [DFlash](https://github.com/bstnxbt/dflash-mlx) and [DDTree](https://github.com/humanrouter/ddtree-mlx) for speculative decoding
