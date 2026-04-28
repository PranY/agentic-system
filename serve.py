#!/usr/bin/env python3
"""
MLX inference server with dynamic model loading.

Models load on-demand when first requested and auto-evict after idle timeout.
Five tiers: mini, small, medium, large, huge — each mapped to optimal
Qwen3.5 MLX models for Apple Silicon.

Endpoints:
  POST /v1/chat/completions     — OpenAI-compatible (auto-loads model)
  POST /v1/chat/completions/batch — batched inference (multiple prompts, one GPU pass)
  GET  /v1/models               — list loaded models with status
  GET  /v1/models/catalog       — full catalog with all tiers
  POST /v1/models/load          — pre-load a model (non-blocking)
  GET  /v1/models/load/{id}     — poll load status
  POST /v1/models/unload        — explicitly unload a model
  GET  /v1/memory               — memory breakdown
  GET  /health                  — status check
  GET  /stats                   — per-model metrics (JSON)
  GET  /stats/live              — auto-refreshing dashboard
"""

import json
import os
import re
import sys
import time
import asyncio
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

# NOTE: We don't set HF_HUB_OFFLINE globally because huggingface_hub reads
# it once at import time. Instead, model_manager.py toggles
# huggingface_hub.constants.HF_HUB_OFFLINE directly per-load:
# - cached models: force offline (instant load, no slow update check)
# - new models:    force online (allow first download)

# Force IPv4 for HuggingFace Hub. CloudFront's IPv6 endpoints can hang for
# 60+ seconds on SYN_SENT before falling back. IPv4 connects instantly.
# Set FORCE_IPV4=0 to disable.
if os.environ.get("FORCE_IPV4", "1") == "1":
    import socket as _socket
    _orig_getaddrinfo = _socket.getaddrinfo
    def _ipv4_only_getaddrinfo(*args, **kwargs):
        return [r for r in _orig_getaddrinfo(*args, **kwargs) if r[0] == _socket.AF_INET]
    _socket.getaddrinfo = _ipv4_only_getaddrinfo

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from model_manager import ModelManager, IDLE_TIMEOUT
from select_models import MODEL_CATALOG, TIER_ORDER

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "system_config.json")

# KV cache quantization bits (0=disabled, 4=recommended, 8=high quality)
KV_CACHE_BITS = int(os.environ.get("KV_BITS", "4"))

# DFlash speculative decoding — block diffusion drafts, 2-4x lossless speedup.
# Disabled by default until verified stable end-to-end. Enable with DFLASH=1.
DFLASH_ENABLED = os.environ.get("DFLASH", "0") != "0"

# DDTree — tree-based extension of dflash, ~10-15% faster on code/structured.
# Disabled by default. Enable with DDTREE=1 (requires DFLASH=1).
DDTREE_ENABLED = os.environ.get("DDTREE", "0") != "0"

# Legacy: kept for backward compat but ignored when dflash is active.
SPEC_DECODE_ENABLED = os.environ.get("SPEC_DECODE", "1") != "0"
SPEC_DECODE_NUM_DRAFT = int(os.environ.get("SPEC_DECODE_DRAFT_TOKENS", "4"))

# Per-request generation timeout (seconds). 0 = disabled.
MAX_GENERATION_TIME = int(os.environ.get("MAX_GENERATION_TIME", "120"))


# Thermal governor
THERMAL_GOVERNOR_ENABLED = os.environ.get("THERMAL_GOVERNOR", "1") != "0"
THERMAL_COOLDOWNS = {0: 0.0, 1: 0.5, 2: 3.0, 3: 8.0}

_thermal_info = None


def get_thermal_state() -> int:
    global _thermal_info
    try:
        if _thermal_info is None:
            from Foundation import NSProcessInfo
            _thermal_info = NSProcessInfo.processInfo()
        return _thermal_info.thermalState()
    except Exception:
        return 0


def get_thermal_label(state: int) -> str:
    return {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}.get(state, "unknown")


# Max concurrent cross-model generations. mx.eval is serialized internally
# via EVAL_LOCK, so CPU graph building overlaps with GPU eval from other models.
# Set to 1 to revert to fully serial behavior.
MAX_CONCURRENT_GENERATIONS = int(os.environ.get("MAX_CONCURRENT_GENERATIONS", "4"))

# Per-model LRU prompt cache. Avoids re-prefilling shared prefixes across
# best-of-N / critique-refine / repeated-system-prompt calls.
# Set to 0 to disable.
PROMPT_CACHE_SIZE = int(os.environ.get("PROMPT_CACHE_SIZE", "4"))
PROMPT_CACHE_MAX_MB = int(os.environ.get("PROMPT_CACHE_MAX_MB", "2048"))
PROMPT_CACHE_MIN_TOKENS = int(os.environ.get("PROMPT_CACHE_MIN_TOKENS", "64"))

# ═══════════════════════════════════════════════════
#  Global state
# ═══════════════════════════════════════════════════

GPU_SEM: asyncio.Semaphore | None = None    # limits concurrent generations
manager: ModelManager | None = None
SERVER_START_TIME = 0.0

# ═══════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════

@dataclass
class ModelMetrics:
    total_requests: int = 0
    total_tokens: int = 0
    total_errors: int = 0
    total_generation_sec: float = 0.0
    in_flight: int = 0
    queued: int = 0
    last_request_time: float = 0.0
    recent_latencies: list = field(default_factory=list)
    recent_tps: list = field(default_factory=list)
    _max_window: int = 100

    def record_request(self, generation_sec: float, tokens: int, tok_per_sec: float):
        self.total_requests += 1
        self.total_tokens += tokens
        self.total_generation_sec += generation_sec
        self.last_request_time = time.time()
        self.recent_latencies.append(generation_sec)
        self.recent_tps.append(tok_per_sec)
        if len(self.recent_latencies) > self._max_window:
            self.recent_latencies.pop(0)
            self.recent_tps.pop(0)

    def record_error(self):
        self.total_errors += 1

    def snapshot(self) -> dict:
        avg_latency = (self.total_generation_sec / self.total_requests) if self.total_requests > 0 else 0
        avg_tps = (self.total_tokens / self.total_generation_sec) if self.total_generation_sec > 0 else 0
        p50 = p95 = p99 = 0.0
        recent_tps_avg = 0.0
        if self.recent_latencies:
            s = sorted(self.recent_latencies)
            n = len(s)
            p50 = s[n // 2]
            p95 = s[min(int(n * 0.95), n - 1)]
            p99 = s[min(int(n * 0.99), n - 1)]
            recent_tps_avg = sum(self.recent_tps) / len(self.recent_tps)
        return {
            "total_requests": self.total_requests,
            "total_tokens_generated": self.total_tokens,
            "total_errors": self.total_errors,
            "in_flight": self.in_flight,
            "queued": self.queued,
            "avg_latency_sec": round(avg_latency, 3),
            "avg_tok_per_sec": round(avg_tps, 1),
            "recent": {
                "window_size": len(self.recent_latencies),
                "p50_latency_sec": round(p50, 3),
                "p95_latency_sec": round(p95, 3),
                "p99_latency_sec": round(p99, 3),
                "avg_tok_per_sec": round(recent_tps_avg, 1),
            },
            "last_request_ago_sec": round(time.time() - self.last_request_time, 1) if self.last_request_time > 0 else None,
        }


METRICS: dict[str, ModelMetrics] = {}


def get_metrics(alias: str) -> ModelMetrics:
    if alias not in METRICS:
        METRICS[alias] = ModelMetrics()
    return METRICS[alias]


# ═══════════════════════════════════════════════════
#  Pydantic models
# ═══════════════════════════════════════════════════

class ChatMessage(BaseModel):
    role: str
    content: str
    tool_calls: list | None = None
    tool_call_id: str | None = None


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    stream: bool = False
    enable_thinking: bool = False
    strip_thinking: bool = False
    response_format: dict | None = None
    # OpenAI-compatible tool calling. When provided, tools are passed into
    # the chat template; <tool_call>...</tool_call> blocks in the output are
    # parsed back into OpenAI tool_calls format.
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    # When true, attach per-token logprobs to the response under
    # choices[0].logprobs.content. top_logprobs > 0 additionally returns the
    # top-k alternative tokens and their logprobs for each sampled token.
    return_logprobs: bool = False
    top_logprobs: int = 0


class BatchItemRequest(BaseModel):
    messages: list[ChatMessage]
    max_tokens: int = 2048


class BatchRequest(BaseModel):
    model: str
    requests: list[BatchItemRequest]
    temperature: float = 0.7
    top_p: float = 0.9
    enable_thinking: bool = False
    response_format: dict | None = None


class ChatChoice(BaseModel):
    index: int = 0
    message: dict
    finish_reason: str = "stop"
    logprobs: dict | None = None


class ChatResponse(BaseModel):
    id: str = "local"
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: list[ChatChoice]
    usage: dict = {}


# ═══════════════════════════════════════════════════
#  MLX imports (cached)
# ═══════════════════════════════════════════════════

_mlx_lm_generate = None
_mlx_lm_stream_generate = None
_mlx_lm_batch_generate = None
_mlx_lm_make_sampler = None
_mlx_lm_make_prompt_cache = None

# DFlash / DDTree (lazy, optional)
_dflash_stream_generate = None
_dflash_available = False
_ddtree_generate = None
_ddtree_available = False


def _import_mlx_lm():
    global _mlx_lm_generate, _mlx_lm_stream_generate, _mlx_lm_batch_generate
    global _mlx_lm_make_sampler, _mlx_lm_make_prompt_cache
    if _mlx_lm_generate is None:
        from mlx_lm import generate, stream_generate, batch_generate
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.models.cache import make_prompt_cache
        _mlx_lm_generate = generate
        _mlx_lm_stream_generate = stream_generate
        _mlx_lm_batch_generate = batch_generate
        _mlx_lm_make_sampler = make_sampler
        _mlx_lm_make_prompt_cache = make_prompt_cache


def _import_dflash():
    global _dflash_stream_generate, _dflash_available
    if _dflash_stream_generate is not None or not DFLASH_ENABLED:
        return
    try:
        from dflash_mlx.runtime import stream_dflash_generate
        _dflash_stream_generate = stream_dflash_generate
        _dflash_available = True
    except ImportError:
        _dflash_available = False


def _import_ddtree():
    global _ddtree_generate, _ddtree_available
    if _ddtree_generate is not None or not DDTREE_ENABLED:
        return
    try:
        from ddtree_mlx.runtime import generate_ddtree_once
        _ddtree_generate = generate_ddtree_once
        _ddtree_available = True
    except ImportError:
        _ddtree_available = False


class PrefixCache:
    """Exact-prefix LRU cache of pre-prefilled KV states, keyed on token tuples.

    Designed for best-of-N / critique-refine patterns where multiple sequential
    calls share an identical system prompt (and optional conversation history).
    Unlike mlx_lm.LRUPromptCache, this doesn't rely on trim_prompt_cache, so it
    works for models with non-trimmable (rotating) KV caches like Qwen3.5.
    """
    def __init__(self, max_size: int, max_bytes: int):
        from collections import OrderedDict
        self.max_size = max_size
        self.max_bytes = max_bytes
        self._store: OrderedDict = OrderedDict()  # key -> (cache_copy, nbytes)
        self._total_bytes = 0
        self.hits = 0
        self.misses = 0
        self.tokens_saved = 0

    def get(self, key):
        """Return a deep-copy of the cached KV state, or None on miss."""
        import copy
        entry = self._store.get(key)
        if entry is None:
            return None
        self._store.move_to_end(key)
        return copy.deepcopy(entry[0])

    def put(self, key, cache_copy):
        """Store a pre-made (already deep-copied) cache under key, evicting LRU if needed."""
        nbytes = sum(getattr(c, "nbytes", 0) for c in cache_copy)
        if key in self._store:
            self._total_bytes -= self._store[key][1]
        self._store[key] = (cache_copy, nbytes)
        self._store.move_to_end(key)
        self._total_bytes += nbytes
        while (len(self._store) > self.max_size
               or self._total_bytes > self.max_bytes) and self._store:
            _, (_, old_nbytes) = self._store.popitem(last=False)
            self._total_bytes -= old_nbytes

    def snapshot(self):
        return {
            "entries": len(self._store),
            "bytes": self._total_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "tokens_saved": self.tokens_saved,
        }


def _get_prompt_cache(loaded_model):
    """Lazily attach a PrefixCache to a LoadedModel. Returns None if disabled."""
    if PROMPT_CACHE_SIZE <= 0:
        return None
    if loaded_model.prompt_cache is None:
        loaded_model.prompt_cache = PrefixCache(
            max_size=PROMPT_CACHE_SIZE,
            max_bytes=PROMPT_CACHE_MAX_MB * 1024 * 1024,
        )
    return loaded_model.prompt_cache


def _prefill_cache(model, tokens, cache):
    """Run the model forward on `tokens` to populate `cache` in place.
    Equivalent to a max_tokens=0 prefill — no generation, just KV state."""
    import mlx.core as mx
    if not tokens:
        return
    arr = mx.array(tokens)
    step_size = 2048
    offset = 0
    while offset < len(arr):
        chunk = arr[offset : offset + step_size][None]  # add batch dim
        model(chunk, cache=cache)
        mx.eval([c.state for c in cache])
        offset += step_size


_PREFIX_DUMMY_CONTENT = "__MLX_PREFIX_CACHE_MARKER_Q7Z9P2__"


def _compute_prefix_and_full_tokens(tokenizer, messages, enable_thinking, response_format):
    """Return (prefix_tokens, full_tokens) where prefix_tokens is the largest
    token-level prefix of full_tokens that corresponds to everything before the
    final user message's content (i.e. excludes the turn that varies across
    best-of-N / critique-refine calls).

    Strategy: render the chat template twice — once with the real last user
    message, and once with its content replaced by a distinctive marker. The
    longest common token prefix is the shared (template-agnostic) prefix.

    Returns (None, full_tokens) when caching isn't worthwhile (e.g. last message
    isn't a user turn, prefix shorter than PROMPT_CACHE_MIN_TOKENS).
    """
    full_text = format_chat_prompt(tokenizer, messages, enable_thinking, response_format=response_format)
    full_tokens = tokenizer.encode(full_text)
    if len(messages) < 2 or messages[-1].role != "user":
        return None, full_tokens
    # Build a dummy-content variant of the last user message
    dummy_messages = list(messages[:-1])
    dummy_last = ChatMessage(role="user", content=_PREFIX_DUMMY_CONTENT)
    dummy_messages.append(dummy_last)
    try:
        dummy_text = format_chat_prompt(
            tokenizer, dummy_messages, enable_thinking, response_format=response_format
        )
    except Exception:
        return None, full_tokens
    dummy_tokens = tokenizer.encode(dummy_text)
    # Longest common token prefix
    common = 0
    for a, b in zip(full_tokens, dummy_tokens):
        if a != b:
            break
        common += 1
    if common < PROMPT_CACHE_MIN_TOKENS:
        return None, full_tokens
    return full_tokens[:common], full_tokens


# ═══════════════════════════════════════════════════
#  Core inference
# ═══════════════════════════════════════════════════

def _normalize_tool_calls_for_template(tool_calls):
    """Convert OpenAI tool_calls (arguments as JSON string) to the form chat
    templates expect: arguments parsed back into a dict. Some templates also
    flatten {function: {name, arguments}} into top-level {name, arguments}."""
    import json as _json
    normalized = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            normalized.append(tc)
            continue
        # OpenAI shape: {id, type, function: {name, arguments}}
        if "function" in tc and isinstance(tc["function"], dict):
            fn = tc["function"]
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = _json.loads(args)
                except Exception:
                    args = {"_raw": args}
            normalized.append({
                "id": tc.get("id"),
                "type": tc.get("type", "function"),
                "function": {"name": fn.get("name", ""), "arguments": args},
            })
        # Already-flat shape: {name, arguments}
        elif "name" in tc:
            args = tc.get("arguments")
            if isinstance(args, str):
                try:
                    args = _json.loads(args)
                except Exception:
                    args = {"_raw": args}
            normalized.append({"name": tc["name"], "arguments": args})
        else:
            normalized.append(tc)
    return normalized


def format_chat_prompt(tokenizer, messages, enable_thinking=False, response_format=None, tools=None):
    msg_dicts = []
    for m in messages:
        d = {"role": m.role, "content": m.content or ""}
        if m.tool_calls is not None:
            d["tool_calls"] = _normalize_tool_calls_for_template(m.tool_calls)
        if m.tool_call_id is not None:
            d["tool_call_id"] = m.tool_call_id
        msg_dicts.append(d)

    if response_format and response_format.get("type") == "json_object":
        json_instruction = "You must respond with valid JSON only. No markdown, no code fences, no explanation — just the JSON object."
        if msg_dicts and msg_dicts[0]["role"] == "system":
            msg_dicts[0]["content"] = msg_dicts[0]["content"] + "\n\n" + json_instruction
        else:
            msg_dicts.insert(0, {"role": "system", "content": json_instruction})

    kwargs = {"tokenize": False, "add_generation_prompt": True}
    kwargs["enable_thinking"] = enable_thinking
    if tools:
        kwargs["tools"] = tools

    try:
        return tokenizer.apply_chat_template(msg_dicts, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        try:
            return tokenizer.apply_chat_template(msg_dicts, **kwargs)
        except TypeError:
            kwargs.pop("tools", None)
            return tokenizer.apply_chat_template(msg_dicts, **kwargs)


def _generate_dflash(loaded_model, tokenizer, prompt_tokens, max_tokens,
                     stop_token_ids=None):
    """Generate using DFlash block diffusion speculative decoding.

    Returns (response_text, output_token_count, acceptance_ratio).
    """
    _import_dflash()
    stream_gen = _dflash_stream_generate

    if stop_token_ids is None:
        from dflash_mlx.generate import get_stop_token_ids
        stop_token_ids = get_stop_token_ids(tokenizer)

    response = ""
    output_tokens = 0
    acceptance_ratio = 0.0
    for event in stream_gen(
        target_model=loaded_model.model,
        draft_model=loaded_model.dflash_draft,
        tokenizer=tokenizer,
        prompt=None,
        prompt_tokens_override=prompt_tokens,
        max_new_tokens=max_tokens,
        use_chat_template=False,  # already applied
        stop_token_ids=stop_token_ids,
    ):
        etype = event.get("event")
        if etype == "token":
            response += tokenizer.decode([event["token_id"]])
        elif etype == "summary":
            output_tokens = event.get("generation_tokens", 0)
            acceptance_ratio = event.get("acceptance_ratio", 0.0)

    if output_tokens == 0:
        output_tokens = len(tokenizer.encode(response))

    return response, output_tokens, acceptance_ratio


def _generate_ddtree(loaded_model, tokenizer, prompt_tokens, max_tokens):
    """Generate using DDTree tree-based speculative decoding.

    Returns (response_text, output_token_count) or None if ddtree fails.
    """
    _import_ddtree()
    if not _ddtree_available or _ddtree_generate is None:
        return None

    try:
        result = _ddtree_generate(
            target_model=loaded_model.model,
            draft_model=loaded_model.dflash_draft,
            tokenizer=tokenizer,
            prompt_tokens=list(prompt_tokens),
            max_new_tokens=max_tokens,
            tree_budget=4,
        )
        text = result.get("text", "")
        tokens = result.get("generation_tokens", len(tokenizer.encode(text)))
        return text, tokens
    except Exception:
        return None


def generate_response(loaded_model, messages, temperature, max_tokens, top_p,
                      enable_thinking, alias=None, response_format=None,
                      return_logprobs=False, top_logprobs=0):
    _import_mlx_lm()
    _import_dflash()
    _import_ddtree()
    generate = _mlx_lm_generate
    make_sampler = _mlx_lm_make_sampler

    model = loaded_model.model
    tokenizer = loaded_model.tokenizer

    prompt_text = format_chat_prompt(tokenizer, messages, enable_thinking, response_format=response_format)
    sampler = make_sampler(temp=temperature, top_p=top_p)

    kwargs = {
        "max_tokens": max_tokens,
        "sampler": sampler,
        "verbose": False,
    }
    if KV_CACHE_BITS > 0:
        kwargs["kv_bits"] = KV_CACHE_BITS

    # DFlash / DDTree speculative decoding — block diffusion draft models.
    # Gives 2-4x lossless speedup. Takes priority over the old mlx-lm spec decode.
    # Disabled when: logprobs requested, batch mode, or no draft model loaded.
    use_dflash = (
        _dflash_available
        and loaded_model.dflash_draft is not None
        and not return_logprobs
        and temperature <= 0.01  # dflash is greedy / near-greedy
    )

    # Legacy mlx-lm speculative decoding — only if dflash is unavailable.
    spec_decode_active = False
    if not use_dflash and (alias in ("large", "huge") and SPEC_DECODE_ENABLED
            and temperature == 0 and manager):
        draft = manager.get_draft_model()
        if draft is not None and draft is not model:
            kwargs["draft_model"] = draft
            kwargs["num_draft_tokens"] = SPEC_DECODE_NUM_DRAFT
            kwargs.pop("kv_bits", None)
            spec_decode_active = True

    # Exact-prefix KV cache — reuse prefill across best-of-N / critique-refine
    # calls where messages[:-1] is identical. Disabled under speculative
    # decoding (both dflash and legacy — draft model needs its own cache).
    skip_prefix_cache = spec_decode_active or use_dflash
    pcache = None if skip_prefix_cache else _get_prompt_cache(loaded_model)
    prefix_tokens = None
    full_tokens = None
    cached_tokens = 0
    prompt_cache_obj = None
    prefill_sec = 0.0
    if pcache is not None:
        prefix_tokens, full_tokens = _compute_prefix_and_full_tokens(
            tokenizer, messages, enable_thinking, response_format
        )
    start = time.time()

    # DFlash / DDTree path — block diffusion speculative decoding
    if use_dflash:
        full_tokens = tokenizer.encode(prompt_text)

        # Try DDTree first (tree-based, ~10-15% faster on code)
        ddtree_result = None
        if _ddtree_available and DDTREE_ENABLED:
            ddtree_result = _generate_ddtree(loaded_model, tokenizer, full_tokens, max_tokens)

        if ddtree_result is not None:
            response, output_tokens = ddtree_result
            decode_method = "ddtree"
        else:
            response, output_tokens, _ = _generate_dflash(
                loaded_model, tokenizer, full_tokens, max_tokens
            )
            decode_method = "dflash"

        elapsed = time.time() - start
        tps = output_tokens / elapsed if elapsed > 0 else 0
        return response, {
            "generation_time_sec": round(elapsed, 3),
            "prefill_time_sec": 0.0,
            "approx_output_tokens": output_tokens,
            "approx_tok_per_sec": round(tps, 1),
            "prompt_tokens": len(full_tokens),
            "cached_tokens": 0,
            "logprobs": None,
            "decode_method": decode_method,
        }

    # Standard mlx-lm path (with optional prefix cache and logprobs)
    if pcache is not None and prefix_tokens is not None:
        import copy as _copy
        key = tuple(prefix_tokens)
        with loaded_model.prompt_cache_lock:
            prompt_cache_obj = pcache.get(key)
            is_hit = prompt_cache_obj is not None
        if is_hit:
            cached_tokens = len(prefix_tokens)
            pcache.hits += 1
            pcache.tokens_saved += cached_tokens
            loaded_model.prompt_cache_hits += 1
            loaded_model.prompt_cache_tokens_saved += cached_tokens
        else:
            # MISS: fresh cache, run pure prefill on the prefix, store a copy,
            # then continue using this same cache for the actual generation.
            prompt_cache_obj = _mlx_lm_make_prompt_cache(model)
            prefill_start = time.time()
            _prefill_cache(model, prefix_tokens, prompt_cache_obj)
            prefill_sec = time.time() - prefill_start
            with loaded_model.prompt_cache_lock:
                pcache.put(key, _copy.deepcopy(prompt_cache_obj))
            pcache.misses += 1
            loaded_model.prompt_cache_misses += 1
        # kv_bits conflicts with a pre-built unquantized cache — drop it.
        kwargs.pop("kv_bits", None)
        kwargs["prompt_cache"] = prompt_cache_obj
        gen_prompt = full_tokens[len(prefix_tokens):]
    else:
        gen_prompt = prompt_text
        full_tokens = None  # computed lazily below if needed

    logprobs_data = None
    if return_logprobs:
        # Stream generation to capture per-token logprob distributions.
        import mlx.core as mx
        stream_generate = _mlx_lm_stream_generate
        stream_kwargs = {k: v for k, v in kwargs.items() if k != "verbose"}
        response = ""
        logprobs_data = []
        output_tokens = 0
        for step in stream_generate(model, tokenizer, prompt=gen_prompt, **stream_kwargs):
            response += step.text
            output_tokens = step.generation_tokens
            # step.logprobs is the full log-softmax distribution for this step.
            token_id = int(step.token)
            logp = float(step.logprobs[token_id])
            token_str = tokenizer.decode([token_id])
            entry = {
                "token": token_str,
                "logprob": logp,
                "bytes": list(token_str.encode("utf-8")),
            }
            if top_logprobs > 0:
                k = min(top_logprobs, step.logprobs.shape[-1])
                # Top-k by logprob magnitude — argpartition is O(n).
                top_idx = mx.argpartition(-step.logprobs, k - 1)[:k]
                top_idx_py = [int(i) for i in top_idx.tolist()]
                # Sort the k picks by descending logprob for readability.
                top_idx_py.sort(key=lambda i: -float(step.logprobs[i]))
                entry["top_logprobs"] = [
                    {
                        "token": tokenizer.decode([i]),
                        "logprob": float(step.logprobs[i]),
                        "bytes": list(tokenizer.decode([i]).encode("utf-8")),
                    }
                    for i in top_idx_py
                ]
            logprobs_data.append(entry)
    else:
        response = generate(model, tokenizer, prompt=gen_prompt, **kwargs)
        output_tokens = len(tokenizer.encode(response))

    elapsed = time.time() - start
    tps = output_tokens / elapsed if elapsed > 0 else 0

    prompt_token_count = (
        len(full_tokens) if full_tokens is not None
        else len(tokenizer.encode(prompt_text))
    )

    return response, {
        "generation_time_sec": round(elapsed, 3),
        "prefill_time_sec": round(prefill_sec, 3),
        "approx_output_tokens": output_tokens,
        "approx_tok_per_sec": round(tps, 1),
        "prompt_tokens": prompt_token_count,
        "cached_tokens": cached_tokens,
        "logprobs": logprobs_data,
        "decode_method": "baseline",
    }


def batch_generate_response(loaded_model, batch_messages, temperature, max_tokens,
                            top_p, enable_thinking, response_format=None):
    """Generate responses for multiple prompts in a single batched GPU pass.

    Applies the per-model PrefixCache to each batch item independently, so
    batches that share a system prompt across items reuse prefill state.
    """
    _import_mlx_lm()
    batch_generate = _mlx_lm_batch_generate

    model = loaded_model.model
    tokenizer = loaded_model.tokenizer

    pcache = _get_prompt_cache(loaded_model)

    # Per-item: (suffix_tokens_to_prefill, cache_or_None, cached_tokens, full_tokens_len)
    per_item = []
    total_cached = 0
    total_prompt_tokens = 0
    prefill_sec = 0.0
    import copy as _copy

    start = time.time()
    for msgs in batch_messages:
        prefix_tokens = None
        full_tokens = None
        if pcache is not None:
            prefix_tokens, full_tokens = _compute_prefix_and_full_tokens(
                tokenizer, msgs, enable_thinking, response_format
            )
        if pcache is None or prefix_tokens is None:
            # Fall back: no caching for this item
            full_text = format_chat_prompt(
                tokenizer, msgs, enable_thinking, response_format=response_format
            )
            tokens = tokenizer.encode(full_text)
            per_item.append((tokens, None, 0, len(tokens)))
            total_prompt_tokens += len(tokens)
            continue

        key = tuple(prefix_tokens)
        with loaded_model.prompt_cache_lock:
            cache_obj = pcache.get(key)
            is_hit = cache_obj is not None
        if is_hit:
            cached = len(prefix_tokens)
            pcache.hits += 1
            pcache.tokens_saved += cached
            loaded_model.prompt_cache_hits += 1
            loaded_model.prompt_cache_tokens_saved += cached
        else:
            cache_obj = _mlx_lm_make_prompt_cache(model)
            prefill_start = time.time()
            _prefill_cache(model, prefix_tokens, cache_obj)
            prefill_sec += time.time() - prefill_start
            with loaded_model.prompt_cache_lock:
                pcache.put(key, _copy.deepcopy(cache_obj))
            pcache.misses += 1
            loaded_model.prompt_cache_misses += 1
            cached = 0
        suffix = full_tokens[len(prefix_tokens):]
        per_item.append((suffix, cache_obj, cached, len(full_tokens)))
        total_cached += cached
        total_prompt_tokens += len(full_tokens)

    prompts_tokenized = [item[0] for item in per_item]
    prompt_caches = [item[1] for item in per_item]
    # If every item fell back to no caching, pass None so batch_generate
    # constructs its own default caches.
    if all(c is None for c in prompt_caches):
        prompt_caches = None

    result = batch_generate(
        model, tokenizer,
        prompts=prompts_tokenized,
        prompt_caches=prompt_caches,
        max_tokens=max_tokens,
        verbose=False,
    )
    elapsed = time.time() - start

    texts = result.texts
    total_tokens = sum(len(tokenizer.encode(t)) for t in texts)
    tps = total_tokens / elapsed if elapsed > 0 else 0

    return texts, {
        "generation_time_sec": round(elapsed, 3),
        "prefill_time_sec": round(prefill_sec, 3),
        "total_output_tokens": total_tokens,
        "approx_tok_per_sec": round(tps, 1),
        "batch_size": len(batch_messages),
        "prompt_tokens": total_prompt_tokens,
        "cached_tokens": total_cached,
    }


# ═══════════════════════════════════════════════════
#  App lifecycle
# ═══════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager, GPU_SEM, SERVER_START_TIME

    # Load system config for memory budget
    memory_budget = 60.0  # default
    bandwidth = 400
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        memory_budget = cfg["memory"]["available_for_models_gb"]
        bandwidth = cfg.get("bandwidth_gbps", 400)

    manager = ModelManager(memory_budget, bandwidth, dflash_enabled=DFLASH_ENABLED)
    GPU_SEM = asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)
    SERVER_START_TIME = time.time()

    # Patch mx.eval/mx.async_eval with a threading lock to serialize Metal
    # command submission. CPU-side graph building still runs in parallel
    # across threads — only the GPU eval step is serialized. Empirically
    # gives ~1.69x speedup on mixed concurrent workloads.
    import mlx.core as mx
    _eval_lock = threading.Lock()
    _original_eval = mx.eval
    _original_async_eval = mx.async_eval

    def _locked_eval(*args, **kwargs):
        with _eval_lock:
            return _original_eval(*args, **kwargs)

    def _locked_async_eval(*args, **kwargs):
        with _eval_lock:
            return _original_async_eval(*args, **kwargs)

    mx.eval = _locked_eval
    mx.async_eval = _locked_async_eval

    # Start idle evictor
    await manager.start_evictor()

    # Preload models if requested via --preload
    if PRELOAD_MODELS:
        await manager.preload(PRELOAD_MODELS)

    yield

    await manager.stop_evictor()
    # Unload all
    for alias in list(manager.loaded.keys()):
        manager._unload_sync(alias)


app = FastAPI(
    title="MLX Local Inference",
    description="Dynamic model loading MLX inference server for Apple Silicon",
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════════
#  Chat completions (auto-loads model)
# ═══════════════════════════════════════════════════

# Map common external model names to our tiers. Hermes and other clients
# may send hardcoded model names like "gpt-4o-mini" for auxiliary tasks.
MODEL_ALIASES = {
    "gpt-4o-mini": "mini",
    "gpt-4o": "small",
    "gpt-4": "medium",
    "gpt-4-turbo": "medium",
    "gpt-3.5-turbo": "mini",
    "claude-3-haiku": "mini",
    "claude-3-sonnet": "medium",
    "claude-3-opus": "large",
}


def _resolve_alias(model: str) -> str:
    """Resolve external model names to local tier names."""
    return MODEL_ALIASES.get(model, model)


async def _stream_chat(request: ChatRequest, alias: str, loaded_model):
    """Generate Server-Sent Events stream in OpenAI chat.completion.chunk format."""
    import json as _json
    _import_mlx_lm()

    completion_id = f"mlx-{int(time.time())}"
    created = int(time.time())
    model_name = loaded_model.info["name"]

    def make_chunk(delta: dict, finish_reason=None):
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }],
        }

    metrics = get_metrics(alias)
    manager.acquire(alias)
    metrics.queued += 1

    async def event_source():
        nonlocal metrics
        try:
            async with GPU_SEM:
                metrics.queued -= 1
                metrics.in_flight = 1
                try:
                    # Initial chunk: assistant role
                    yield f"data: {_json.dumps(make_chunk({'role': 'assistant'}))}\n\n"

                    # Build prompt synchronously
                    tokenizer = loaded_model.tokenizer
                    model = loaded_model.model
                    sampler = _mlx_lm_make_sampler(temp=request.temperature, top_p=request.top_p)
                    prompt_text = format_chat_prompt(
                        tokenizer, request.messages, request.enable_thinking,
                        response_format=request.response_format,
                        tools=request.tools,
                    )
                    kwargs = {"max_tokens": request.max_tokens, "sampler": sampler}
                    if KV_CACHE_BITS > 0:
                        kwargs["kv_bits"] = KV_CACHE_BITS

                    # Cache only the SYSTEM-PROMPT prefix. This portion is stable
                    # across every turn of an agent loop (Hermes ships the same
                    # 13K-token system prompt every call), so it gives reliable
                    # cache hits. Caching beyond the system prompt would require
                    # matching the model's generated assistant messages against
                    # their re-rendering in subsequent turns — which doesn't work
                    # because re-rendering produces different tokenization.
                    new_tokens = tokenizer.encode(prompt_text)
                    pcache = _get_prompt_cache(loaded_model)
                    import copy as _copy

                    # Compute the longest STABLE prefix shared across any future
                    # turn. We do this by rendering [system, dummy_user] twice
                    # with two distinct markers and taking the LCP. The result
                    # is the system portion + chat template user header — stable
                    # across all turns regardless of conversation history.
                    sys_token_count = 0
                    has_system = (request.messages and request.messages[0].role == "system")
                    if has_system and pcache is not None:
                        try:
                            sys_msg = request.messages[0]
                            marker_a = ChatMessage(role="user", content="__MARKER_A_X9P2__")
                            marker_b = ChatMessage(role="user", content="__MARKER_B_Z7Q4__")
                            text_a = format_chat_prompt(
                                tokenizer, [sys_msg, marker_a],
                                request.enable_thinking,
                                response_format=request.response_format,
                                tools=request.tools,
                            )
                            text_b = format_chat_prompt(
                                tokenizer, [sys_msg, marker_b],
                                request.enable_thinking,
                                response_format=request.response_format,
                                tools=request.tools,
                            )
                            tokens_a = tokenizer.encode(text_a)
                            tokens_b = tokenizer.encode(text_b)
                            # LCP across (a, b, new) — using all three guarantees
                            # the prefix is stable and matches the actual prompt.
                            # BPE merges at the user-content boundary can blur a
                            # "two-marker" LCP, so we cross-check against new_tokens.
                            common = 0
                            n = min(len(tokens_a), len(tokens_b), len(new_tokens))
                            while (common < n
                                    and tokens_a[common] == tokens_b[common]
                                    and tokens_a[common] == new_tokens[common]):
                                common += 1
                            if common >= PROMPT_CACHE_MIN_TOKENS and common < len(new_tokens):
                                sys_token_count = common
                            elif os.environ.get("DEBUG_REQUESTS") == "1":
                                print(f"[CACHE-DBG] common={common} min_tokens={PROMPT_CACHE_MIN_TOKENS} "
                                      f"len_new={len(new_tokens)} len_a={len(tokens_a)} len_b={len(tokens_b)}",
                                      flush=True)
                        except Exception as e:
                            if os.environ.get("DEBUG_REQUESTS") == "1":
                                print(f"[CACHE-DBG] exception: {e}", flush=True)

                    # Look up cache entry keyed on (model_alias, sys_tokens_tuple)
                    cache_key = tuple(new_tokens[:sys_token_count]) if sys_token_count else None
                    stream_cache = None
                    if cache_key is not None:
                        with loaded_model.prompt_cache_lock:
                            stream_cache = pcache.get(cache_key)
                        if stream_cache is not None:
                            pcache.hits += 1
                            pcache.tokens_saved += sys_token_count
                            loaded_model.prompt_cache_hits += 1
                            loaded_model.prompt_cache_tokens_saved += sys_token_count

                    if stream_cache is None:
                        # Build cache from scratch by prefilling the system tokens
                        stream_cache = _mlx_lm_make_prompt_cache(model)
                        if cache_key is not None:
                            # Prefill just the system portion, store snapshot in LRU
                            _prefill_cache(model, list(cache_key), stream_cache)
                            with loaded_model.prompt_cache_lock:
                                pcache.put(cache_key, _copy.deepcopy(stream_cache))
                            if pcache is not None:
                                pcache.misses += 1
                                loaded_model.prompt_cache_misses += 1

                    kwargs["prompt_cache"] = stream_cache
                    kwargs.pop("kv_bits", None)  # incompatible with prebuilt cache

                    # Pass only the suffix (everything after the system prompt)
                    suffix_tokens = new_tokens[sys_token_count:]
                    gen_prompt = suffix_tokens if sys_token_count > 0 else prompt_text
                    lcp = sys_token_count

                    if os.environ.get("DEBUG_REQUESTS") == "1":
                        total = len(new_tokens)
                        print(f"[CACHE] alias={alias} sys_cached={sys_token_count}/{total} "
                              f"({100*sys_token_count//max(total,1)}% reused), "
                              f"prefilling {len(suffix_tokens)} new tokens",
                              flush=True)

                    # Tool call markers (from tokenizer when tools are passed)
                    has_tools = bool(request.tools) and getattr(tokenizer, "has_tool_calling", False)
                    tc_start = getattr(tokenizer, "tool_call_start", "<tool_call>") if has_tools else None
                    tc_end = getattr(tokenizer, "tool_call_end", "</tool_call>") if has_tools else None
                    tool_parser = getattr(tokenizer, "tool_parser", None) if has_tools else None

                    # Run generation in a thread, push tokens via a queue
                    import queue as _q
                    q: _q.Queue = _q.Queue()
                    DONE = object()

                    generated_token_ids: list[int] = []

                    def _run():
                        try:
                            for step in _mlx_lm_stream_generate(
                                model, tokenizer, prompt=gen_prompt, **kwargs
                            ):
                                generated_token_ids.append(int(step.token))
                                q.put(step.text)
                            q.put(DONE)
                        except Exception as e:
                            q.put(e)

                    start = time.time()
                    threading.Thread(target=_run, daemon=True).start()

                    # Accumulate full output to handle <think> and <tool_call> blocks.
                    full_text = ""
                    sent_any_content = False
                    strip_think = (
                        request.strip_thinking
                        or os.environ.get("STRIP_THINK", "1") == "1"
                    )
                    output_tokens = 0
                    streamed_upto = 0  # index in full_text already streamed as content

                    def emit_content(text: str):
                        return f"data: {_json.dumps(make_chunk({'content': text}))}\n\n"

                    # Markers we hide from streamed content (think blocks + tool calls).
                    # Each marker has (start, end). When found, we skip the whole block.
                    hide_markers = []
                    if strip_think:
                        hide_markers.append(("<think>", "</think>"))
                    if tc_start and tc_end:
                        hide_markers.append((tc_start, tc_end))
                    # Longest possible partial-marker tail to hold back at end of stream
                    max_marker_len = max((len(s) for s, _ in hide_markers), default=0)

                    def safe_emit_index(buf: str) -> int:
                        """Return the largest index up to which `buf` can be safely emitted
                        without breaking inside any marker."""
                        i = 0
                        n = len(buf)
                        while i < n:
                            # Are we at the start of any marker?
                            in_marker = False
                            for start, end in hide_markers:
                                if buf.startswith(start, i):
                                    # Look for the closing end
                                    close = buf.find(end, i + len(start))
                                    if close == -1:
                                        # Open block not yet closed — stop here
                                        return i
                                    i = close + len(end)
                                    in_marker = True
                                    break
                            if in_marker:
                                continue
                            # Check if we're partway into a possible marker start
                            partial = False
                            for start, _ in hide_markers:
                                # Could the rest be the prefix of `start`?
                                rest = buf[i:]
                                if len(rest) < len(start) and start.startswith(rest):
                                    partial = True
                                    break
                            if partial:
                                return i
                            i += 1
                        return n

                    while True:
                        item = await asyncio.to_thread(q.get)
                        if item is DONE:
                            break
                        if isinstance(item, Exception):
                            raise item
                        output_tokens += 1
                        full_text += item

                        # Compute how much of full_text we can safely emit now.
                        idx = safe_emit_index(full_text)
                        if idx > streamed_upto:
                            # Build the visible text by removing complete hide-blocks
                            # from this safe-to-emit slice.
                            slice_text = full_text[streamed_upto:idx]
                            for s_start, s_end in hide_markers:
                                slice_text = re.sub(
                                    re.escape(s_start) + r".*?" + re.escape(s_end),
                                    "", slice_text, flags=re.DOTALL,
                                )
                            if slice_text:
                                sent_any_content = True
                                yield emit_content(slice_text)
                            streamed_upto = idx

                    # Flush remaining (treat all blocks as closed at end of stream)
                    if streamed_upto < len(full_text):
                        tail = full_text[streamed_upto:]
                        for s_start, s_end in hide_markers:
                            tail = re.sub(
                                re.escape(s_start) + r".*?" + re.escape(s_end),
                                "", tail, flags=re.DOTALL,
                            )
                        if tail:
                            sent_any_content = True
                            yield emit_content(tail)

                    # Parse tool_calls from full_text
                    tool_calls_out = []
                    if tc_start and tc_end and tc_start in full_text:
                        pattern = re.escape(tc_start) + r"(.*?)" + re.escape(tc_end)
                        for m in re.finditer(pattern, full_text, flags=re.DOTALL):
                            inner = m.group(1).strip()
                            try:
                                if tool_parser:
                                    parsed = tool_parser(inner, request.tools)
                                    parsed_list = parsed if isinstance(parsed, list) else [parsed]
                                else:
                                    parsed_list = [_json.loads(inner)]
                                for tc in parsed_list:
                                    args = tc.get("arguments", {})
                                    if not isinstance(args, str):
                                        args = _json.dumps(args, ensure_ascii=False)
                                    tool_calls_out.append({
                                        "id": tc.get("id") or f"call_{len(tool_calls_out)}_{int(time.time()*1000)%100000}",
                                        "type": "function",
                                        "function": {
                                            "name": tc.get("name", ""),
                                            "arguments": args,
                                        },
                                        "index": len(tool_calls_out),
                                    })
                            except Exception as ex:
                                if os.environ.get("DEBUG_REQUESTS") == "1":
                                    print(f"[STREAM] tool_call parse error: {ex} | inner={inner[:100]}", flush=True)

                    if tool_calls_out:
                        # Emit a single delta containing the tool_calls
                        yield f"data: {_json.dumps(make_chunk({'tool_calls': tool_calls_out}))}\n\n"

                    # If we never emitted anything (model produced only think blocks
                    # or empty output), send a fallback to avoid empty-response errors.
                    if not sent_any_content and not tool_calls_out:
                        yield emit_content("(no content — try increasing max_tokens)")

                    elapsed = time.time() - start
                    tps = output_tokens / elapsed if elapsed > 0 else 0
                    metrics.record_request(elapsed, output_tokens, tps)
                    if os.environ.get("DEBUG_REQUESTS") == "1":
                        print(f"[STREAM] alias={alias} tokens={output_tokens} elapsed={elapsed:.2f}s "
                              f"prefill_skipped={lcp}", flush=True)

                    # Final chunk
                    finish = "tool_calls" if tool_calls_out else "stop"
                    yield f"data: {_json.dumps(make_chunk({}, finish_reason=finish))}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    metrics.in_flight = 0
        except Exception as e:
            import traceback
            metrics.record_error()
            tb = traceback.format_exc()
            print(f"[STREAM ERROR] alias={alias}: {e}\n{tb}", flush=True)
            # Send a valid chunk with error message as content so OpenAI clients
            # can display it instead of failing to parse.
            err_msg = f"[server error: {type(e).__name__}: {e}]"
            yield f"data: {_json.dumps(make_chunk({'content': err_msg}))}\n\n"
            yield f"data: {_json.dumps(make_chunk({}, finish_reason='stop'))}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            manager.release(alias)

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    alias = _resolve_alias(request.model)

    # Optional debug: dump full request and response. Enable with DEBUG_REQUESTS=1.
    if os.environ.get("DEBUG_REQUESTS") == "1":
        try:
            import json as _json
            dump = {
                "model": request.model,
                "alias": alias,
                "stream": request.stream,
                "n_messages": len(request.messages),
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "enable_thinking": request.enable_thinking,
                "strip_thinking": request.strip_thinking,
                "response_format": request.response_format,
                "msg_roles": [m.role for m in request.messages],
                "msg_lengths": [len(m.content or "") for m in request.messages],
            }
            print(f"[REQ] {_json.dumps(dump)}", flush=True)
        except Exception as e:
            print(f"[REQ] dump error: {e}", flush=True)

    # Auto-load model if not loaded
    try:
        loaded_model = await manager.ensure_loaded(alias)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except TimeoutError as e:
        raise HTTPException(status_code=503, detail=str(e),
                            headers={"Retry-After": "30"})
    except MemoryError as e:
        raise HTTPException(status_code=503, detail=str(e),
                            headers={"Retry-After": "10"})
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Streaming path: return SSE chunks instead of single JSON.
    # Required for clients like Hermes that send stream=true and parse chunks.
    if request.stream:
        return await _stream_chat(request, alias, loaded_model)

    metrics = get_metrics(alias)

    # Thermal governor
    if THERMAL_GOVERNOR_ENABLED:
        thermal_state = get_thermal_state()
        thermal_delay = THERMAL_COOLDOWNS.get(thermal_state, 0.0)
        if thermal_delay > 0:
            pass  # thermal cooldown
            await asyncio.sleep(thermal_delay)

    # Ref count protects from eviction during generation
    manager.acquire(alias)
    metrics.queued += 1
    try:
        async with GPU_SEM:
            metrics.queued -= 1
            metrics.in_flight = 1
            try:
                gen_coro = asyncio.to_thread(
                    generate_response,
                    loaded_model,
                    request.messages,
                    request.temperature,
                    request.max_tokens,
                    request.top_p,
                    request.enable_thinking,
                    alias=alias,
                    response_format=request.response_format,
                    return_logprobs=request.return_logprobs,
                    top_logprobs=request.top_logprobs,
                )
                if MAX_GENERATION_TIME > 0:
                    try:
                        response_text, stats = await asyncio.wait_for(
                            gen_coro, timeout=MAX_GENERATION_TIME
                        )
                    except asyncio.TimeoutError:
                        metrics.record_error()
                        metrics.in_flight = 0
                        raise HTTPException(
                            status_code=504,
                            detail=f"Generation timed out after {MAX_GENERATION_TIME}s.",
                        )
                else:
                    response_text, stats = await gen_coro
            except HTTPException:
                raise
            except Exception:
                metrics.record_error()
                metrics.in_flight = 0
                raise
            metrics.in_flight = 0
    except HTTPException:
        raise
    except Exception:
        metrics.queued = max(0, metrics.queued - 1)
        raise
    finally:
        manager.release(alias)

    metrics.record_request(
        stats["generation_time_sec"],
        stats["approx_output_tokens"],
        stats["approx_tok_per_sec"],
    )

    # Strip thinking blocks. Default ON because OpenAI-compatible clients
    # (like Hermes) treat content with raw <think> tags as malformed/empty.
    # Set STRIP_THINK=0 to keep thinking blocks in the response.
    strip = request.strip_thinking or os.environ.get("STRIP_THINK", "1") == "1"
    if strip:
        # Qwen3.5 uses <think>...</think>, Gemma 4 uses <|channel>thought...
        response_text = re.sub(r"<think>.*?</think>\s*", "", response_text, flags=re.DOTALL)
        response_text = re.sub(r"<\|channel>thought.*?(?=<\|channel>|$)", "", response_text, flags=re.DOTALL).strip()
        if response_text.lstrip().startswith("Thinking Process"):
            found = False
            for marker in [r'\n\n?(?:Answer|Response|Output|Result|Final Answer):?\s*',
                           r'\n\n?\*\*(?:Answer|Response|Output|Result)\*\*:?\s*']:
                m = re.search(marker, response_text, re.IGNORECASE)
                if m:
                    response_text = response_text[m.end():]
                    found = True
                    break
            if not found:
                json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(1)
                else:
                    parts = response_text.strip().split("\n\n")
                    if len(parts) > 1:
                        response_text = parts[-1]
        response_text = response_text.strip()

    # JSON extraction for response_format
    if (request.response_format and request.response_format.get("type") == "json_object"
            and not response_text.lstrip().startswith(("{", "["))):
        json_match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', response_text, re.DOTALL)
        if not json_match:
            json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1).strip()

    if os.environ.get("DEBUG_REQUESTS") == "1":
        print(f"[RESP] alias={alias} content_len={len(response_text)} "
              f"completion_tokens={stats.get('approx_output_tokens', 0)} "
              f"first_200={repr(response_text[:200])}", flush=True)

    choice_kwargs = {
        "message": {"role": "assistant", "content": response_text},
        "finish_reason": "stop",
    }
    if stats.get("logprobs") is not None:
        choice_kwargs["logprobs"] = {"content": stats["logprobs"]}

    return ChatResponse(
        id=f"mlx-{int(time.time())}",
        created=int(time.time()),
        model=loaded_model.info["name"],
        choices=[ChatChoice(**choice_kwargs)],
        usage={
            "prompt_tokens": stats.get("prompt_tokens", 0),
            "completion_tokens": stats["approx_output_tokens"],
            "total_tokens": stats.get("prompt_tokens", 0) + stats["approx_output_tokens"],
            "cached_tokens": stats.get("cached_tokens", 0),
            "_generation_time_sec": stats["generation_time_sec"],
            "_tok_per_sec": stats["approx_tok_per_sec"],
            "_decode_method": stats.get("decode_method", "baseline"),
        },
    )


# ═══════════════════════════════════════════════════
#  Batch completions
# ═══════════════════════════════════════════════════

@app.post("/v1/chat/completions/batch")
async def batch_completions(request: BatchRequest):
    """Process multiple prompts in a single batched GPU pass for higher throughput."""
    alias = _resolve_alias(request.model)

    if not request.requests:
        raise HTTPException(status_code=400, detail="Empty requests list")

    try:
        loaded_model = await manager.ensure_loaded(alias)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except TimeoutError as e:
        raise HTTPException(status_code=503, detail=str(e),
                            headers={"Retry-After": "30"})
    except MemoryError as e:
        raise HTTPException(status_code=503, detail=str(e),
                            headers={"Retry-After": "10"})
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    metrics = get_metrics(alias)

    # Use the max of all per-request max_tokens
    max_tokens = max(r.max_tokens for r in request.requests)

    manager.acquire(alias)
    metrics.queued += 1
    try:
        async with GPU_SEM:
            metrics.queued -= 1
            metrics.in_flight = len(request.requests)
            try:
                gen_coro = asyncio.to_thread(
                    batch_generate_response,
                    loaded_model,
                    [r.messages for r in request.requests],
                    request.temperature,
                    max_tokens,
                    request.top_p,
                    request.enable_thinking,
                    response_format=request.response_format,
                )
                if MAX_GENERATION_TIME > 0:
                    batch_timeout = MAX_GENERATION_TIME * len(request.requests)
                    try:
                        texts, stats = await asyncio.wait_for(
                            gen_coro, timeout=batch_timeout
                        )
                    except asyncio.TimeoutError:
                        metrics.record_error()
                        metrics.in_flight = 0
                        raise HTTPException(
                            status_code=504,
                            detail=f"Batch generation timed out after {batch_timeout}s.",
                        )
                else:
                    texts, stats = await gen_coro
            except HTTPException:
                raise
            except Exception:
                metrics.record_error()
                metrics.in_flight = 0
                raise
            metrics.in_flight = 0
    except HTTPException:
        raise
    except Exception:
        metrics.queued = max(0, metrics.queued - 1)
        raise
    finally:
        manager.release(alias)

    # Record metrics (one entry per batch item)
    per_item_sec = stats["generation_time_sec"] / len(texts) if texts else 0
    per_item_tps = stats["approx_tok_per_sec"]
    for _ in texts:
        per_tokens = stats["total_output_tokens"] // len(texts) if texts else 0
        metrics.record_request(per_item_sec, per_tokens, per_item_tps)

    # Build response
    responses = []
    for i, text in enumerate(texts):
        responses.append({
            "index": i,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        })

    return {
        "id": f"mlx-batch-{int(time.time())}",
        "object": "chat.completion.batch",
        "created": int(time.time()),
        "model": loaded_model.info["name"],
        "responses": responses,
        "usage": {
            "prompt_tokens": stats.get("prompt_tokens", 0),
            "cached_tokens": stats.get("cached_tokens", 0),
            "total_output_tokens": stats["total_output_tokens"],
            "batch_size": stats["batch_size"],
            "_generation_time_sec": stats["generation_time_sec"],
            "_prefill_time_sec": stats.get("prefill_time_sec", 0),
            "_tok_per_sec": stats["approx_tok_per_sec"],
        },
    }


# ═══════════════════════════════════════════════════
#  Model management endpoints
# ═══════════════════════════════════════════════════

class LoadRequest(BaseModel):
    model: str  # tier name: mini, small, medium, large, huge


class UnloadRequest(BaseModel):
    model: str


@app.post("/v1/models/load")
async def load_model(request: LoadRequest):
    """Pre-load a model. Returns immediately if already loaded, otherwise starts loading."""
    alias = _resolve_alias(request.model)

    if alias in manager.loaded:
        return {
            "status": "ready",
            "model": alias,
            "model_name": manager.loaded[alias].info["name"],
            "mem_gb": manager.loaded[alias].mem_gb,
        }

    if alias in manager.loading_aliases:
        load_id = manager.loading_aliases[alias]
        return {
            "status": "loading",
            "model": alias,
            "load_id": load_id,
        }

    # Start loading in background
    from select_models import get_default_model
    model_info = get_default_model(alias)
    if model_info is None:
        raise HTTPException(status_code=404, detail=f"Unknown tier '{alias}'. Available: {TIER_ORDER}")

    # Fire-and-forget load
    async def _bg_load():
        try:
            await manager.ensure_loaded(alias, timeout=300)
        except Exception as e:
            pass

    asyncio.create_task(_bg_load())
    # Wait briefly for loading_aliases to populate
    await asyncio.sleep(0.05)

    load_id = manager.loading_aliases.get(alias)
    return {
        "status": "loading",
        "model": alias,
        "model_name": model_info["name"],
        "mem_gb": model_info["mem_gb"],
        "load_id": load_id,
        "estimated_seconds": int(model_info["mem_gb"] * 2),  # rough estimate
    }


@app.get("/v1/models/load/{load_id}")
async def get_load_status(load_id: str):
    """Poll load status by load ID."""
    status = manager.get_load_status(load_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Load ID '{load_id}' not found")
    return {
        "load_id": status.load_id,
        "model": status.alias,
        "status": status.status,
        "error": status.error,
        "elapsed_sec": round(time.time() - status.started_at, 1),
    }


@app.post("/v1/models/unload")
async def unload_model(request: UnloadRequest):
    """Explicitly unload a model to free memory."""
    alias = _resolve_alias(request.model)
    if alias not in manager.loaded:
        raise HTTPException(status_code=404, detail=f"Model '{alias}' is not loaded")
    try:
        freed = await manager.unload(alias)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {
        "status": "unloaded",
        "model": alias,
        "freed_gb": freed,
        "free_memory_gb": round(manager.free_memory_gb, 1),
    }


@app.get("/v1/models")
async def list_models():
    """List loaded models with capabilities."""
    models = []
    for alias, lm in manager.loaded.items():
        models.append({
            "id": alias,
            "object": "model",
            "owned_by": "local-mlx",
            "full_name": lm.info["name"],
            "mem_gb": lm.mem_gb,
            "status": "loaded",
            "capabilities": lm.info.get("capabilities", {}),
            "idle_sec": round(time.time() - lm.last_used_at, 1),
            "ref_count": lm.ref_count,
        })
    return {"object": "list", "data": models}


@app.get("/v1/models/catalog")
async def model_catalog():
    """Full catalog with load status per tier."""
    return {
        "tiers": manager.catalog_status(),
        "memory": manager.memory_status(),
    }


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """OpenAI-compatible: retrieve a single model by ID (tier name)."""
    model_id = _resolve_alias(model_id)
    # If loaded, return full info
    if model_id in manager.loaded:
        lm = manager.loaded[model_id]
        return {
            "id": model_id,
            "object": "model",
            "created": int(lm.loaded_at),
            "owned_by": "local-mlx",
            "full_name": lm.info["name"],
            "mem_gb": lm.mem_gb,
            "status": "loaded",
        }
    # If known tier but not loaded, return available status
    from select_models import get_default_model
    model_info = get_default_model(model_id)
    if model_info is not None:
        return {
            "id": model_id,
            "object": "model",
            "created": 0,
            "owned_by": "local-mlx",
            "full_name": model_info["name"],
            "mem_gb": model_info["mem_gb"],
            "status": "available",
        }
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found. Available: {TIER_ORDER}")


@app.get("/v1/memory")
async def memory_endpoint():
    """Detailed memory breakdown."""
    return manager.memory_status()


# ═══════════════════════════════════════════════════
#  Health / Stats
# ═══════════════════════════════════════════════════

@app.get("/health")
async def health():
    mem = manager.memory_status()
    thermal_state = get_thermal_state() if THERMAL_GOVERNOR_ENABLED else None
    return {
        "status": "ok",
        "models_loaded": len(manager.loaded),
        "models": list(manager.loaded.keys()),
        "memory_used_gb": mem["used_gb"],
        "memory_free_gb": mem["free_gb"],
        "idle_timeout_sec": IDLE_TIMEOUT,
        "thermal": {
            "state": get_thermal_label(thermal_state) if thermal_state is not None else "unknown",
            "level": thermal_state,
            "governor": "enabled" if THERMAL_GOVERNOR_ENABLED else "disabled",
            "cooldown_sec": THERMAL_COOLDOWNS.get(thermal_state, 0) if thermal_state is not None else 0,
        },
    }


@app.get("/stats")
async def stats_json():
    uptime = time.time() - SERVER_START_TIME if SERVER_START_TIME else 0
    total_reqs = sum(m.total_requests for m in METRICS.values())
    total_errs = sum(m.total_errors for m in METRICS.values())
    total_tokens = sum(m.total_tokens for m in METRICS.values())
    total_gen_sec = sum(m.total_generation_sec for m in METRICS.values())
    gpu_utilization = (total_gen_sec / uptime * 100) if uptime > 0 else 0

    thermal_state = get_thermal_state() if THERMAL_GOVERNOR_ENABLED else 0
    mem = manager.memory_status()

    # Parallel generation stats
    total_in_flight = sum(m.in_flight for m in METRICS.values())
    total_queued = sum(m.queued for m in METRICS.values())
    sem_available = GPU_SEM._value if GPU_SEM else MAX_CONCURRENT_GENERATIONS

    return {
        "uptime_sec": round(uptime, 1),
        "gpu_busy_pct": round(min(gpu_utilization, 100), 1),
        "thermal_state": get_thermal_label(thermal_state),
        "thermal_cooldown_sec": THERMAL_COOLDOWNS.get(thermal_state, 0),
        "parallel": {
            "max": MAX_CONCURRENT_GENERATIONS,
            "active": total_in_flight,
            "queued": total_queued,
            "slots_free": sem_available,
        },
        "memory": mem,
        "total_requests": total_reqs,
        "total_errors": total_errs,
        "total_tokens_generated": total_tokens,
        "requests_per_minute": round(total_reqs / (uptime / 60), 1) if uptime > 60 else total_reqs,
        "models": {
            alias: {
                **metrics.snapshot(),
                "loaded": alias in manager.loaded,
                "prompt_cache": _prompt_cache_snapshot(alias),
                "dflash_draft": (
                    manager.loaded[alias].dflash_draft_ref
                    if alias in manager.loaded and manager.loaded[alias].dflash_draft is not None
                    else None
                ),
            }
            for alias, metrics in METRICS.items()
        },
        "prompt_cache_config": {
            "max_size": PROMPT_CACHE_SIZE,
            "max_mb": PROMPT_CACHE_MAX_MB,
            "min_tokens": PROMPT_CACHE_MIN_TOKENS,
        },
        "dflash": {
            "enabled": DFLASH_ENABLED,
            "available": _dflash_available,
        },
        "ddtree": {
            "enabled": DDTREE_ENABLED,
            "available": _ddtree_available,
        },
    }


def _prompt_cache_snapshot(alias: str) -> dict:
    """Return a JSON-friendly view of a model's prompt cache, or an empty dict."""
    if manager is None or alias not in manager.loaded:
        return {}
    lm = manager.loaded[alias]
    pc = lm.prompt_cache
    if pc is None:
        return {
            "entries": 0,
            "bytes": 0,
            "hits": lm.prompt_cache_hits,
            "misses": lm.prompt_cache_misses,
            "tokens_saved": lm.prompt_cache_tokens_saved,
        }
    snap = pc.snapshot()
    snap["hit_rate"] = (
        round(snap["hits"] / (snap["hits"] + snap["misses"]) * 100, 1)
        if (snap["hits"] + snap["misses"]) > 0 else 0.0
    )
    return snap


@app.get("/stats/live", response_class=HTMLResponse)
async def stats_dashboard():
    return """<!DOCTYPE html>
<html>
<head>
<title>MLX Inference — Live Stats</title>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'SF Mono', 'Menlo', 'Consolas', monospace; background: #0d1117; color: #c9d1d9; padding: 20px; }
  h1 { color: #58a6ff; margin-bottom: 4px; font-size: 18px; }
  .subtitle { color: #8b949e; font-size: 12px; margin-bottom: 20px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; margin-bottom: 20px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  .card h2 { font-size: 14px; color: #58a6ff; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; }
  .card h2 .badge { font-size: 11px; padding: 2px 8px; border-radius: 10px; }
  .badge.idle { background: #1f2a1f; color: #3fb950; }
  .badge.active { background: #2a1f1f; color: #f97316; animation: pulse 1s infinite; }
  .badge.queued { background: #2a2a1f; color: #e3b341; }
  .badge.unloaded { background: #1f1f2a; color: #8b949e; }
  .badge.thermal-nominal { background: #1f2a1f; color: #3fb950; }
  .badge.thermal-fair { background: #2a2a1f; color: #e3b341; }
  .badge.thermal-serious { background: #2a1f1f; color: #f97316; animation: pulse 1.5s infinite; }
  .badge.thermal-critical { background: #3a1010; color: #f85149; animation: pulse 0.5s infinite; }
  .stat-row { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #21262d; font-size: 13px; }
  .stat-row:last-child { border-bottom: none; }
  .stat-label { color: #8b949e; }
  .stat-value { color: #f0f6fc; font-weight: 600; }
  .stat-value.highlight { color: #58a6ff; }
  .stat-value.warn { color: #e3b341; }
  .stat-value.good { color: #3fb950; }
  .global { background: #0d1117; border: 1px solid #58a6ff33; }
  .bar-container { height: 6px; background: #21262d; border-radius: 3px; margin-top: 4px; }
  .bar { height: 100%; border-radius: 3px; transition: width 0.5s ease; }
  .bar.blue { background: #58a6ff; }
  .bar.green { background: #3fb950; }
  .bar.orange { background: #f97316; }
  .catalog-row { display: flex; justify-content: space-between; padding: 3px 0; font-size: 12px; }
  .catalog-row .tier { color: #58a6ff; width: 60px; }
  .catalog-row .mem { color: #8b949e; }
  .catalog-row .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; }
  .dot-loaded { background: #3fb950; }
  .dot-loading { background: #e3b341; animation: pulse 1s infinite; }
  .dot-available { background: #30363d; }
  .btn { background: #21262d; border: 1px solid #30363d; color: #c9d1d9; padding: 4px 12px; border-radius: 6px; cursor: pointer; font-size: 11px; font-family: inherit; }
  .btn:hover { background: #30363d; }
  .btn.load { border-color: #3fb950; color: #3fb950; }
  .btn.unload { border-color: #f97316; color: #f97316; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
  .refresh { color: #484f58; font-size: 11px; text-align: right; margin-top: 8px; }
</style>
</head>
<body>
<h1>MLX Inference Server</h1>
<p class="subtitle">Dynamic model loading — auto-refreshes every 2s</p>

<div class="grid" id="dashboard">
  <div class="card global"><h2>Loading...</h2></div>
</div>
<div class="refresh" id="refresh-time"></div>

<script>
function fmtTime(sec) {
  if (sec < 60) return sec.toFixed(1) + 's';
  if (sec < 3600) return (sec/60).toFixed(1) + 'm';
  return (sec/3600).toFixed(1) + 'h';
}
function fmtNum(n) {
  if (n >= 1e6) return (n/1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n/1e3).toFixed(1) + 'K';
  return n.toString();
}
function statusBadge(m) {
  if (!m.loaded) return '<span class="badge unloaded">UNLOADED</span>';
  if (m.in_flight > 0) return '<span class="badge active">GENERATING</span>';
  if (m.queued > 0) return '<span class="badge queued">QUEUED ' + m.queued + '</span>';
  return '<span class="badge idle">IDLE</span>';
}
function gpuBar(pct) {
  const cls = pct > 80 ? 'orange' : pct > 40 ? 'blue' : 'green';
  return '<div class="bar-container"><div class="bar ' + cls + '" style="width:' + Math.min(pct,100) + '%"></div></div>';
}

async function loadModel(tier) {
  await fetch('/v1/models/load', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({model:tier})});
}
async function unloadModel(tier) {
  await fetch('/v1/models/unload', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({model:tier})});
}

async function refresh() {
  try {
    const [sr, cr] = await Promise.all([fetch('/stats'), fetch('/v1/models/catalog')]);
    const d = await sr.json();
    const cat = await cr.json();
    let html = '';

    // Server overview
    const ts = d.thermal_state || 'nominal';
    const tc = d.thermal_cooldown_sec || 0;
    const mem = d.memory || {};
    const thermalBadge = '<span class="badge thermal-' + ts + '">' + ts.toUpperCase() + (tc > 0 ? ' (' + tc + 's)' : '') + '</span>';
    html += '<div class="card global"><h2>Server Overview</h2>';
    html += '<div class="stat-row"><span class="stat-label">Uptime</span><span class="stat-value">' + fmtTime(d.uptime_sec) + '</span></div>';
    html += '<div class="stat-row"><span class="stat-label">Thermal</span>' + thermalBadge + '</div>';
    html += '<div class="stat-row"><span class="stat-label">Memory</span><span class="stat-value">' + (mem.used_gb||0) + ' / ' + (mem.usable_for_weights_gb||0) + ' GB</span></div>';
    html += '<div class="stat-row"><span class="stat-label">Free</span><span class="stat-value good">' + (mem.free_gb||0) + ' GB</span></div>';
    html += '<div class="stat-row"><span class="stat-label">Total requests</span><span class="stat-value highlight">' + fmtNum(d.total_requests) + '</span></div>';
    html += '<div class="stat-row"><span class="stat-label">Errors</span><span class="stat-value ' + (d.total_errors > 0 ? 'warn' : 'good') + '">' + d.total_errors + '</span></div>';
    html += '<div class="stat-row"><span class="stat-label">GPU busy</span><span class="stat-value">' + d.gpu_busy_pct + '%</span></div>';
    html += gpuBar(d.gpu_busy_pct);

    // Parallel generation status
    const p = d.parallel || {max:1, active:0, queued:0, slots_free:1};
    const parColor = p.active > 1 ? 'highlight' : (p.active > 0 ? 'good' : '');
    html += '<div class="stat-row"><span class="stat-label">Generating</span><span class="stat-value ' + parColor + '">' + p.active + ' / ' + p.max + '</span></div>';
    if (p.queued > 0) html += '<div class="stat-row"><span class="stat-label">Queued</span><span class="stat-value warn">' + p.queued + '</span></div>';
    html += '</div>';

    // Catalog card with load/unload buttons
    html += '<div class="card"><h2>Model Catalog</h2>';
    for (const t of (cat.tiers || [])) {
      const dotCls = t.status === 'loaded' ? 'dot-loaded' : t.status === 'loading' ? 'dot-loading' : 'dot-available';
      const btn = t.status === 'loaded'
        ? '<button class="btn unload" onclick="unloadModel(\\''+t.tier+'\\')">unload</button>'
        : '<button class="btn load" onclick="loadModel(\\''+t.tier+'\\')">load</button>';
      html += '<div class="catalog-row"><span class="tier"><span class="status-dot ' + dotCls + '"></span>' + t.tier + '</span><span class="mem">' + (t.mem_gb||'?') + ' GB</span>' + btn + '</div>';
    }
    html += '</div>';

    // Per-model cards (only models with metrics)
    for (const [alias, m] of Object.entries(d.models)) {
      const r = m.recent;
      html += '<div class="card"><h2>' + alias + ' ' + statusBadge(m) + '</h2>';
      html += '<div class="stat-row"><span class="stat-label">Requests</span><span class="stat-value highlight">' + fmtNum(m.total_requests) + '</span></div>';
      html += '<div class="stat-row"><span class="stat-label">Tokens</span><span class="stat-value">' + fmtNum(m.total_tokens_generated) + '</span></div>';
      html += '<div class="stat-row"><span class="stat-label">Errors</span><span class="stat-value ' + (m.total_errors > 0 ? 'warn' : 'good') + '">' + m.total_errors + '</span></div>';
      html += '<div class="stat-row"><span class="stat-label">Avg latency</span><span class="stat-value">' + m.avg_latency_sec.toFixed(2) + 's</span></div>';
      html += '<div class="stat-row"><span class="stat-label">Avg tok/s</span><span class="stat-value good">' + m.avg_tok_per_sec + '</span></div>';
      html += '<div class="stat-row"><span class="stat-label">p50</span><span class="stat-value">' + r.p50_latency_sec.toFixed(2) + 's</span></div>';
      html += '<div class="stat-row"><span class="stat-label">p95</span><span class="stat-value ' + (r.p95_latency_sec > 5 ? 'warn' : '') + '">' + r.p95_latency_sec.toFixed(2) + 's</span></div>';
      html += '<div class="stat-row"><span class="stat-label">p99</span><span class="stat-value">' + r.p99_latency_sec.toFixed(2) + 's</span></div>';
      html += '<div class="stat-row"><span class="stat-label">Last request</span><span class="stat-value">' + (m.last_request_ago_sec !== null ? fmtTime(m.last_request_ago_sec) + ' ago' : 'never') + '</span></div>';
      const pc = m.prompt_cache || {};
      if (pc.hits !== undefined && (pc.hits + pc.misses) > 0) {
        const mb = (pc.bytes / (1024*1024)).toFixed(0);
        html += '<div class="stat-row"><span class="stat-label">Prefix cache</span><span class="stat-value good">' + pc.hits + ' hit / ' + pc.misses + ' miss (' + (pc.hit_rate||0) + '%)</span></div>';
        html += '<div class="stat-row"><span class="stat-label">&nbsp;&nbsp;entries</span><span class="stat-value">' + (pc.entries||0) + ' · ' + mb + ' MB · ' + fmtNum(pc.tokens_saved||0) + ' tok saved</span></div>';
      }
      html += '</div>';
    }

    document.getElementById('dashboard').innerHTML = html;
    document.getElementById('refresh-time').textContent = 'Updated: ' + new Date().toLocaleTimeString();
  } catch(e) {
    document.getElementById('refresh-time').textContent = 'Error: ' + e.message;
  }
}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════

PRELOAD_MODELS: list[str] = []

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MLX Dynamic Inference Server")
    parser.add_argument("--preload", type=str, default=None,
                        help="Comma-separated models to pre-load at startup "
                             "(e.g. 'small,medium'). Default: none (on-demand).")
    parser.add_argument("--port", type=int, default=None,
                        help="Server port (overrides MLX_PORT env var)")
    parser.add_argument("--strip-think", action="store_true", default=False,
                        help="Strip <think> blocks from responses server-wide")
    args = parser.parse_args()

    if args.preload:
        PRELOAD_MODELS = [m.strip() for m in args.preload.split(",")]
    if args.strip_think:
        os.environ["STRIP_THINK"] = "1"

    port = args.port or int(os.environ.get("MLX_PORT", "8800"))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
