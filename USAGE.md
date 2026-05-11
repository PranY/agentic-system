# Local MLX Inference — Dynamic Model Loading + DFlash Speculative Decoding

## Quick Start

```bash
# One-time setup (installs mlx-lm + dflash-mlx + ddtree-mlx)
./setup.sh

# Start server (models load on-demand with DFlash drafts)
./start.sh

# Or pre-load specific models at startup
./start.sh --preload small,medium
```

The server starts immediately with no models loaded. Models load automatically on first request (with their DFlash draft model) and unload after 15 minutes of idle time.

## Models

Five Qwen tiers spanning 0.8B to 35B parameters. All emit OpenAI-compatible tool calls via the `<tool_call>...</tool_call>` markers in their chat templates.

| Tier | Model | Memory | Use case |
|---|---|---|---|
| `mini` | Qwen3.5-0.8B-MLX-4bit | 0.5 GB | Routing, titles, classification, ultra-fast subagents |
| `small` | Qwen3.5-4B-MLX-4bit | 2.5 GB | Fast tool calling (97.5% accuracy), web extract, vision |
| `medium` | Qwen3.5-9B-MLX-4bit | 5.5 GB | Agentic workhorse, MMLU-Pro 82.5 |
| `large` | Qwen3.6-27B-4bit | 16 GB | Deep reasoning. Terminal-Bench 2.0 59.3, SWE-bench Verified 77.2, AIME 2026 94.1, GPQA 87.8. **Best output quality without placeholder hallucination** — use as default for synthesis-heavy work |
| `huge` | Qwen3.6-35B-A3B-4bit-DWQ | 21 GB | Latest Qwen (Apr 2026), DWQ-calibrated to avoid multi-turn tool-call drift. SWE-bench 73.4, AIME 92.7. MoE — 3B active despite 35B total |

All models: 262K native context, thinking/non-thinking dual mode, Gated DeltaNet + attention hybrid architecture, `qwen3_5` / `qwen3_coder` tool parser.

**Choosing a tier**: smaller models hallucinate placeholders (`[Title]`, `[Authors]`) in long structured outputs because they can't reason about gaps in tool results. For agent loops doing synthesis (research, summaries, comparisons), default to `huge` (preferred — MoE generation is fast despite the 35B total) or `large`. For narrow well-scoped subtasks (routing, extraction, simple tool calls), `mini` and `small` are fine.

**Streaming + system-prompt prefix cache**: the `/v1/chat/completions` streaming path (used by Hermes and most agent harnesses) automatically detects the stable system-prompt portion of a conversation via three-way LCP across two dummy-marker chat-template renders, prefills that portion once, and reuses the KV state across all subsequent turns. Measured: ~92% cache hit rate on long agent loops, dropping per-turn latency by 4-5×.

### Optional: DFlash speculative decoding

DFlash gives 2-4x lossless speedup but is **disabled by default** (`DFLASH=0`) until end-to-end verification with current model lineup. Enable with `DFLASH=1` once you've confirmed it works for your workload. See the DFlash section below.

## API Reference

### Chat Completions

```
POST /v1/chat/completions
```

```json
{
  "model": "small",
  "messages": [{"role": "user", "content": "..."}],
  "temperature": 0.7,
  "max_tokens": 2048,
  "stream": false,
  "enable_thinking": false,
  "strip_thinking": false,
  "response_format": {"type": "json_object"},
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
      }
    }
  ]
}
```

The model auto-loads if not already in memory. First request to an unloaded model takes 5-30 seconds (download + load + Metal shader compilation).

**Streaming**: `stream: true` returns OpenAI-compatible SSE chunks (`data: {...}\n\n`) ending with `data: [DONE]`. The streaming path also reuses a persistent rolling KV cache across calls — successive turns of the same conversation skip prefill of the shared prefix.

**Tool calling**: Pass OpenAI-format `tools` and the model emits `<tool_call>...</tool_call>` blocks that the server parses back into `tool_calls` deltas. `function.arguments` is normalized between OpenAI's JSON-string format and the dict format Qwen's chat template expects.

**Model aliases**: Common external model names map to local tiers automatically:
- `gpt-4o-mini`, `gpt-3.5-turbo`, `claude-3-haiku` → `mini`
- `gpt-4o` → `small`
- `gpt-4`, `gpt-4-turbo`, `claude-3-sonnet` → `medium`
- `claude-3-opus` → `large`

### Batch Completions

```
POST /v1/chat/completions/batch
```

```json
{
  "model": "mini",
  "requests": [
    {"messages": [{"role": "user", "content": "classify: printer jam"}], "max_tokens": 256},
    {"messages": [{"role": "user", "content": "classify: vpn issues"}], "max_tokens": 256},
    {"messages": [{"role": "user", "content": "classify: pay stub error"}], "max_tokens": 256}
  ],
  "temperature": 0.7,
  "enable_thinking": false
}
```

Processes multiple prompts in a single batched GPU pass using MLX `batch_generate`. More efficient than sending individual requests — eliminates per-request overhead (context switching, KV cache setup). Timeout scales with batch size.

### Load Model (pre-warm)

```
POST /v1/models/load
```

```json
{"model": "large"}
```

Returns immediately with status `"loading"` or `"ready"`. Use this to pre-warm models before a batch of requests.

Response:
```json
{
  "status": "loading",
  "model": "large",
  "load_id": "load-a1b2c3d4",
  "estimated_seconds": 28
}
```

### Poll Load Status

```
GET /v1/models/load/{load_id}
```

### Unload Model

```
POST /v1/models/unload
```

```json
{"model": "large"}
```

Frees memory immediately. Returns 409 if model has in-flight requests.

### List Loaded Models

```
GET /v1/models
```

Shows loaded models with capabilities, idle time, ref count.

### Model Catalog

```
GET /v1/models/catalog
```

Full catalog of all 5 tiers with load status (`loaded`, `loading`, `available`).

### Memory

```
GET /v1/memory
```

Detailed memory breakdown: total budget, KV cache reserve, per-model usage, idle times.

### Health

```
GET /health
```

### Stats

```
GET /stats       # JSON
GET /stats/live  # Auto-refreshing HTML dashboard
```

The dashboard shows memory usage, model catalog with load/unload buttons, per-model metrics, and thermal state.

## Dynamic Model Management

### How It Works

1. **On-demand loading**: First request to `{"model": "medium"}` triggers download (if needed) + load + Metal warm-up
2. **Memory management**: Server tracks total memory usage. If a new model won't fit, it evicts the least-recently-used idle model first
3. **Idle eviction**: Models not used for 15 minutes are automatically unloaded (configurable via `IDLE_TIMEOUT_SEC`)
4. **Ref counting**: Models with in-flight requests cannot be evicted
5. **Pre-warming**: Use `POST /v1/models/load` to load models before they're needed
6. **DFlash speculative decoding**: Each model loads its paired `z-lab/*-DFlash` draft model for 2-4x lossless speedup via block diffusion. DDTree adds 10-15% on top for code tasks

### Memory Budget

| Component | GB |
|---|---|
| System total | 64 |
| Available for models | 60 |
| KV cache reserve | 8 |
| Usable for weights | 52 |

You can load any combination that fits in 52 GB. Examples:

- mini + small + medium + large = 22.5 GB (all 4 dense models)
- small + medium = 8 GB (lightweight setup)
- large alone = 14 GB (maximum quality per request)
- huge alone = 20 GB (maximum quality, MoE)

## Using from Python

```python
import httpx

BASE = "http://localhost:8800"

# Model loads on first request — no setup needed
def ask(model: str, prompt: str, **kwargs) -> str:
    resp = httpx.post(f"{BASE}/v1/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": kwargs.get("max_tokens", 1024),
        "temperature": kwargs.get("temperature", 0.7),
    }, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# Pre-load for faster first response
httpx.post(f"{BASE}/v1/models/load", json={"model": "medium"})

# Use any tier
route = ask("mini", "Route to 'code' or 'docs': fix the login bug")
code = ask("medium", "Write a Python ISO 8601 date parser")
analysis = ask("large", "Analyze: microservices vs monolith for 5-person team")
```

## Configuration

| File | Purpose |
|---|---|
| `serve.py` | FastAPI server: dynamic loading, streaming SSE, tool calling, IPv4 fix |
| `model_manager.py` | Model lifecycle: load, unload, evict, ref counting, per-load HF offline toggle |
| `select_models.py` | Model catalog (5 tiers), selection, DFlash draft registry |
| `route_proxy.py` | Routing proxy for multi-instance mode — unifies several backends behind one port |
| `setup.sh` | One-time install |
| `start.sh` | Start single-process server (auto-preloads `large` by default) |
| `start_multi.sh` | Multi-process mode in foreground: one server per tier + routing proxy |
| `start_tmux.sh` | Same as `start_multi.sh` but inside a tmux session, one pane per backend |
| `download_models.sh` | Pre-download all models (idempotent, IPv4-forced) |
| `hermes-config.yaml.example` | Drop-in template for Hermes Agent (copy to `~/.hermes/config.yaml`) |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MLX_PORT` | `8800` | Server port |
| `KV_BITS` | `4` | KV cache quantization (0/4/8) |
| `FORCE_IPV4` | `1` | Force IPv4 for HF Hub (avoids macOS IPv6 SYN_SENT hangs) |
| `DFLASH` | `0` | DFlash block diffusion spec decode (0/1) — disabled by default |
| `DDTREE` | `0` | DDTree tree-based spec decode (0/1) — disabled by default |
| `SPEC_DECODE` | `1` | Legacy mlx-lm spec decode, used when DFlash unavailable (0/1) |
| `SPEC_DECODE_DRAFT_TOKENS` | `4` | Draft tokens for legacy spec decode |
| `MAX_GENERATION_TIME` | `120` | Per-request timeout in seconds (0=disabled) |
| `MAX_CONCURRENT_GENERATIONS` | `4` | Max parallel generations across models (locked eval) |
| `IDLE_TIMEOUT_SEC` | `900` | Idle model eviction timeout (seconds) |
| `KV_CACHE_RESERVE_GB` | `8` | GB reserved for KV cache |
| `THERMAL_GOVERNOR` | `1` | Thermal throttling (0/1) |
| `STRIP_THINK` | `1` | Strip `<think>` blocks from responses (0/1). On by default — clients like Hermes treat raw `<think>` as malformed |
| `DEBUG_REQUESTS` | `0` | Log incoming chat-completion requests to stderr (0/1) |
| `PROMPT_CACHE_SIZE` | `4` | Prefix-cache entries per model (0=disable) |
| `PROMPT_CACHE_MAX_MB` | `2048` | Prefix-cache byte cap per model |
| `PROMPT_CACHE_MIN_TOKENS` | `64` | Minimum shared-prefix length to cache |

### DFlash speculative decoding (2-4x lossless speedup)

[DFlash](https://github.com/bstnxbt/dflash-mlx) uses block diffusion draft models to generate 16 tokens in parallel, then verifies them in a single target forward pass. Output is lossless — bit-for-bit identical to standard autoregressive decoding.

Each model tier (except mini) has a paired `z-lab/*-DFlash` draft model (~1B params) that loads automatically alongside the target. Draft acceptance rates are typically 85-91%.

Benchmarks (M5 Max, 64GB):

| Model | Baseline | DFlash | Speedup |
|-------|----------|--------|---------|
| Qwen3.5-4B | 53 tok/s | 153-196 tok/s | 3.0-3.7x |
| Qwen3.5-9B | 30 tok/s | 67-135 tok/s | 2.2-4.4x |
| Qwen3.5-27B | 33 tok/s | 45-79 tok/s | 1.3-2.4x |
| Qwen3.6-35B-A3B | 134 tok/s | 177-300 tok/s | 1.3-2.2x |

DFlash activates automatically when: a draft model is loaded, `temperature <= 0.01`, and logprobs are not requested. Responses include `_decode_method: "dflash"` or `"ddtree"` in usage.

[DDTree](https://github.com/humanrouter/ddtree-mlx) extends DFlash by building a tree of multiple candidate paths from the draft, verified in one forward pass. Adds ~10-15% on top of DFlash for code and structured content. Falls back to DFlash for prose. Set `DDTREE=0` to disable.

### Prefix cache (best-of-N / critique-refine speedup)

For sequential calls that share an identical system prompt (or leading conversation history) and only vary in the last user message, the server reuses the KV prefill from the previous call. Measured on the `small` tier with an 834-token system prompt: warm calls run ~5× faster (0.34s vs 1.7s).

Responses include `usage.prompt_tokens` and `usage.cached_tokens` so clients can see what was reused:

```json
{
  "usage": {
    "prompt_tokens": 834,
    "cached_tokens": 818,
    "completion_tokens": 7,
    "_generation_time_sec": 0.34
  }
}
```

Per-model hit/miss stats appear under `/stats` → `models.<tier>.prompt_cache` and in the live dashboard.

The cache is disabled automatically under speculative decoding and when the last message isn't a user turn.

### Per-token logprobs

Pass `return_logprobs: true` (optionally with `top_logprobs: N`) to get OpenAI-format per-token logprobs attached to `choices[0].logprobs.content`:

```json
{
  "model": "mini",
  "messages": [{"role": "user", "content": "Complete: The capital of France is"}],
  "max_tokens": 5,
  "temperature": 0,
  "return_logprobs": true,
  "top_logprobs": 3
}
```

Default-path cost is unchanged — when `return_logprobs` is false (default), no extra work is done. Not yet supported on the batch endpoint.

### Multi-process mode + routing proxy

For workloads that benefit from true GPU parallelism between models, run each tier as its own process behind a routing proxy. Two ways to launch.

**One command, tmux session, live output per pane (recommended):**

```bash
./start_tmux.sh                                     # default: mini + small
MODELS="mini:8810,small:8811,medium:8812" ./start_tmux.sh
PROXY_PORT=9000 ./start_tmux.sh                     # different proxy port
./start_tmux.sh --detach                            # don't auto-attach
./start_tmux.sh --kill                              # tear down
```

Creates a tmux session named `mlx-multi` with one vertical pane per model + one for the routing proxy. Detach with `Ctrl-b d`, reattach with `tmux attach -t mlx-multi`. You see each backend's stdout in its own pane.

**One command, all logs to files (no tmux):**

```bash
./start_multi.sh                                    # foreground; stdout merged
MODELS="mini:8810,small:8811,medium:8812" ./start_multi.sh
```

Logs land in `/tmp/mlx_multi/{tier}_{port}.log` and `/tmp/mlx_multi/proxy_{port}.log`. Use this for headless / supervised runs.

**What both launch:**

- One `serve.py` process per `tier:port` pair, with the model preloaded
- One `route_proxy.py` on `PROXY_PORT` (default 8800) that:
  - Routes `POST /v1/chat/completions` by `model` field (resolves aliases like `gpt-4o-mini`)
  - Aggregates `GET /v1/models`, `/v1/models/catalog`, `/health`, `/stats`
  - Forwards everything else to the first backend

**When this beats single-process**: mixed `mini`+`small` traffic at high concurrency, or any setup where multiple models need to generate concurrently. ~25% aggregate throughput win on 20 mixed requests at concurrency 4. For medium/large tier workloads where a single request saturates the GPU, single-process (`./start.sh`) is simpler.

### Integrating with [Hermes Agent](https://hermes-agent.nousresearch.com/)

Hermes is a self-improving agent framework (Nous Research) that consumes OpenAI-compatible APIs. The server speaks Hermes's protocol natively — streaming SSE chunks, tool_calls deltas, and the auxiliary endpoints it probes (`/v1/models/{id}` etc.).

**Setup** (one-time, after `./setup.sh`):
```bash
# Install Hermes (if not already)
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash

# Copy the bundled template (yours becomes ~/.hermes/config.yaml — edit freely)
mkdir -p ~/.hermes
cp hermes-config.yaml.example ~/.hermes/config.yaml

# Tell Hermes the endpoint (no real API key needed for local)
cat > ~/.hermes/.env <<'EOF'
OPENAI_API_KEY=local
OPENAI_BASE_URL=http://localhost:8800/v1
EOF
```

The bundled `hermes-config.yaml.example` routes Hermes's roles to local tiers:
- **Main agent** → `large` (Qwen3.5-27B) — best output quality without placeholder hallucination
- **Fallback** → `huge` (Qwen3.6-35B-A3B) — top benchmarks for hardest cases
- **Subagent delegation** → `small` (4B) — fast for narrow scoped subtasks
- **Auxiliary** (titles, compression, vision, web extract, session search) → `mini` or `small`

**Run**:
```bash
./start.sh                  # auto-preloads `large` for fast Hermes startup
hermes                      # in another terminal
```

**Caveats** (learned the hard way):
- **Smaller models hallucinate placeholders** in long structured outputs (e.g. tables with `[Title]`/`[Authors]`). Stick to `large`/`huge` for synthesis-heavy work.
- **Per-turn cost is dominated by prefill of accumulated tool outputs** — browser snapshots can be 8K+ chars each. Setting `agent.max_turns: 200` plus aggressive tool calls compounds context fast.
- **Streaming is required** — Hermes sends `stream: true` for chat. Non-streaming JSON gets parsed as a malformed SSE chunk and treated as empty.

## Architecture

```
                    ┌──────────────────────────────────────┐
                    │     FastAPI Server (:8800)            │
                    │     OpenAI-compatible API             │
                    ├──────────────────────────────────────┤
                    │     ModelManager (dynamic loading)    │
                    │     ┌─ LRU eviction (15min idle)     │
                    │     ├─ Ref counting (in-flight)      │
                    │     ├─ Memory tracking (52 GB)       │
                    │     └─ Background evictor task        │
                    ├──────────────────────────────────────┤
                    │  DFlash Block Diffusion Spec Decode    │
                    │  ├─ z-lab/*-DFlash draft models       │
                    │  ├─ 16-token parallel draft + verify  │
                    │  ├─ 2-4x lossless speedup             │
                    │  └─ DDTree tree decode (+10-15%)      │
                    ├──────────────────────────────────────┤
                    │  Models load/unload on demand:        │
                    │  mini(0.5) small(2.5) medium(5.5)    │
                    │  large(14)  huge(20)                  │
                    │  Qwen3.5 + Qwen3.6 · 262K ctx        │
                    ├──────────────────────────────────────┤
                    │  4-bit KV Cache · Prefix KV Cache     │
                    │  Thermal Governor · Gen Timeout        │
                    ├──────────────────────────────────────┤
                    │  Apple M1 Max · 64 GB · 400 GB/s     │
                    └──────────────────────────────────────┘
```
