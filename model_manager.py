"""
Dynamic model lifecycle manager for MLX inference server.

Handles on-demand loading, LRU eviction, memory tracking, ref counting,
idle timeout unloading, and speculative decoding draft model resolution.

Thread safety:
  - LOAD_LOCK serializes load/unload decisions (asyncio.Lock)
  - EVAL_LOCK (threading.Lock) serializes mx.eval() calls (serve.py monkey-patches mx.eval)
  - GPU_SEM (asyncio.Semaphore) limits concurrent generations (serve.py)
  - ref_count prevents eviction of models with in-flight requests
"""

import asyncio
import gc
import os
import threading
import time
import uuid
from dataclasses import dataclass, field

from select_models import MODEL_CATALOG, TIER_ORDER, get_default_model, DFLASH_DRAFT_REGISTRY

# Memory reserved for KV cache and system overhead (GB)
KV_CACHE_RESERVE = float(os.environ.get("KV_CACHE_RESERVE_GB", "8"))

# Idle timeout: unload models not used for this many seconds
IDLE_TIMEOUT = int(os.environ.get("IDLE_TIMEOUT_SEC", "900"))  # 15 minutes

# Eviction check interval
EVICTOR_INTERVAL = int(os.environ.get("EVICTOR_INTERVAL_SEC", "60"))


@dataclass
class LoadedModel:
    alias: str                  # tier name: mini, small, medium, large, huge
    model: object               # mlx model object
    tokenizer: object           # tokenizer
    info: dict                  # catalog entry metadata
    mem_gb: float
    loaded_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    ref_count: int = 0          # in-flight requests using this model
    # Per-model LRU prompt cache (PrefixCache). Populated lazily on first use
    # in serve.py. GC'd with LoadedModel on unload.
    prompt_cache: object | None = None
    # Protects prompt_cache trie/LRU updates when MAX_CONCURRENT_GENERATIONS > 1.
    prompt_cache_lock: threading.Lock = field(default_factory=threading.Lock)
    # Hit/miss counters for observability.
    prompt_cache_hits: int = 0
    prompt_cache_misses: int = 0
    prompt_cache_tokens_saved: int = 0
    # DFlash block diffusion draft model (loaded alongside target if available).
    dflash_draft: object | None = None
    dflash_draft_ref: str | None = None     # HF model ID of the draft
    dflash_draft_mem_gb: float = 0.0
    # Streaming chat uses the same per-model PrefixCache as non-streaming
    # (loaded_model.prompt_cache) — system-prompt prefill is detected via
    # dummy-marker LCP and stored in the LRU for reuse across turns.


@dataclass
class LoadStatus:
    load_id: str
    alias: str
    status: str = "loading"     # loading, ready, failed
    error: str | None = None
    started_at: float = field(default_factory=time.time)


class ModelManager:
    def __init__(self, memory_budget_gb: float, bandwidth_gbps: float = 400,
                 dflash_enabled: bool = True):
        self.memory_budget = memory_budget_gb
        self.usable_for_weights = memory_budget_gb - KV_CACHE_RESERVE
        self.bandwidth_gbps = bandwidth_gbps
        self._dflash_enabled = dflash_enabled

        # State
        self.loaded: dict[str, LoadedModel] = {}        # alias -> LoadedModel
        self.load_statuses: dict[str, LoadStatus] = {}   # load_id -> LoadStatus
        self.loading_aliases: dict[str, str] = {}         # alias -> load_id (if currently loading)
        self.loading_events: dict[str, asyncio.Event] = {}  # alias -> Event (signaled when done)

        # Locks
        self.load_lock = asyncio.Lock()

        # MLX imports (lazy)
        self._mlx_lm_load = None
        self._mlx_lm_generate = None

        # Background task handle
        self._evictor_task: asyncio.Task | None = None

    def _import_mlx_lm(self):
        if self._mlx_lm_load is None:
            from mlx_lm import load, generate
            self._mlx_lm_load = load
            self._mlx_lm_generate = generate

    # ═══════════════════════════════════════════════════
    #  Memory accounting
    # ═══════════════════════════════════════════════════

    @property
    def used_memory_gb(self) -> float:
        return sum(m.mem_gb for m in self.loaded.values())

    @property
    def free_memory_gb(self) -> float:
        return self.usable_for_weights - self.used_memory_gb

    def memory_status(self) -> dict:
        models = {}
        now = time.time()
        for alias, lm in self.loaded.items():
            entry = {
                "mem_gb": lm.mem_gb,
                "model_name": lm.info["name"],
                "loaded_ago_sec": round(now - lm.loaded_at, 1),
                "last_used_ago_sec": round(now - lm.last_used_at, 1),
                "ref_count": lm.ref_count,
                "idle": lm.ref_count == 0,
            }
            if lm.dflash_draft is not None:
                entry["dflash_draft"] = lm.dflash_draft_ref
                entry["dflash_draft_mem_gb"] = lm.dflash_draft_mem_gb
            models[alias] = entry
        return {
            "total_budget_gb": self.memory_budget,
            "kv_cache_reserve_gb": KV_CACHE_RESERVE,
            "usable_for_weights_gb": self.usable_for_weights,
            "used_gb": round(self.used_memory_gb, 1),
            "free_gb": round(self.free_memory_gb, 1),
            "models": models,
        }

    # ═══════════════════════════════════════════════════
    #  Core: ensure_loaded (the main entry point)
    # ═══════════════════════════════════════════════════

    async def ensure_loaded(self, alias: str, timeout: float = 120) -> LoadedModel:
        """
        Ensure a model is loaded and return it. Loads on-demand if needed.
        Waits up to `timeout` seconds for loading to complete.
        Raises ValueError if alias is unknown, TimeoutError if loading times out,
        MemoryError if not enough memory after eviction.
        """
        # Fast path: already loaded
        if alias in self.loaded:
            self.loaded[alias].last_used_at = time.time()
            return self.loaded[alias]

        # Check if it's a known tier
        model_info = get_default_model(alias)
        if model_info is None:
            raise ValueError(
                f"Unknown model tier '{alias}'. Available: {TIER_ORDER}"
            )

        # Check if already loading — wait for it
        if alias in self.loading_aliases:
            event = self.loading_events[alias]
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Model '{alias}' is loading but timed out after {timeout}s. "
                    f"Load ID: {self.loading_aliases.get(alias)}"
                )
            if alias in self.loaded:
                self.loaded[alias].last_used_at = time.time()
                return self.loaded[alias]
            raise RuntimeError(f"Model '{alias}' failed to load")

        # Need to load — acquire lock to prevent concurrent loads of same model
        async with self.load_lock:
            # Double-check after acquiring lock
            if alias in self.loaded:
                self.loaded[alias].last_used_at = time.time()
                return self.loaded[alias]

            # Set up loading state
            load_id = f"load-{uuid.uuid4().hex[:8]}"
            event = asyncio.Event()
            self.loading_aliases[alias] = load_id
            self.loading_events[alias] = event
            self.load_statuses[load_id] = LoadStatus(
                load_id=load_id, alias=alias
            )

            # Evict if needed
            needed = model_info["mem_gb"]
            if needed > self.free_memory_gb:
                freed = await self._evict_for(needed - self.free_memory_gb)
                if not freed:
                    self._cleanup_loading(alias, load_id, "not enough memory")
                    raise MemoryError(
                        f"Cannot load '{alias}' ({needed:.1f} GB): only {self.free_memory_gb:.1f} GB free "
                        f"after eviction. Loaded: {list(self.loaded.keys())}"
                    )

        # Load outside the lock (loading is slow, don't block other operations)
        try:
            loaded_model = await asyncio.to_thread(
                self._load_model_sync, alias, model_info
            )
            self.loaded[alias] = loaded_model
            self.load_statuses[load_id].status = "ready"
        except Exception as e:
            self._cleanup_loading(alias, load_id, str(e))
            raise RuntimeError(f"Failed to load '{alias}': {e}") from e
        finally:
            # Signal waiters
            self.loading_aliases.pop(alias, None)
            event.set()
            self.loading_events.pop(alias, None)

        return self.loaded[alias]

    def _cleanup_loading(self, alias: str, load_id: str, error: str):
        self.load_statuses[load_id].status = "failed"
        self.load_statuses[load_id].error = error
        self.loading_aliases.pop(alias, None)
        event = self.loading_events.pop(alias, None)
        if event:
            event.set()

    def _load_model_sync(self, alias: str, model_info: dict) -> LoadedModel:
        """Synchronous model loading (runs in thread)."""
        self._import_mlx_lm()
        load = self._mlx_lm_load

        name = model_info["name"]

        # Detect if model is cached. Force offline mode for cached models to
        # skip slow HF update checks (the root cause of multi-minute load
        # times). For new models, ensure online mode to allow download.
        # Modify the constant directly because huggingface_hub reads the
        # env var only once at import time.
        from huggingface_hub import try_to_load_from_cache
        import huggingface_hub.constants as hf_const
        is_cached = try_to_load_from_cache(name, "config.json") is not None
        prev_offline = hf_const.HF_HUB_OFFLINE
        hf_const.HF_HUB_OFFLINE = bool(is_cached)

        start = time.time()
        try:
            model, tokenizer = load(name)
        finally:
            hf_const.HF_HUB_OFFLINE = prev_offline
        elapsed = time.time() - start

        # Warm-up: single token generation to compile Metal shaders
        try:
            from mlx_lm import generate
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": "hi"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            generate(model, tokenizer, prompt=prompt, max_tokens=1, verbose=False)
        except Exception:
            pass

        # Load DFlash block diffusion draft model if available
        dflash_draft = None
        dflash_draft_ref = None
        dflash_draft_mem_gb = 0.0
        draft_ref = DFLASH_DRAFT_REGISTRY.get(name)
        if draft_ref and self._dflash_enabled:
            try:
                from dflash_mlx.runtime import load_draft_bundle
                draft_model, draft_meta = load_draft_bundle(draft_ref)
                dflash_draft = draft_model
                dflash_draft_ref = draft_ref
                # Draft models are ~1B, estimate ~0.5 GB at 4-bit
                dflash_draft_mem_gb = draft_meta.get("mem_gb", 0.5)
            except Exception:
                pass  # Fall back to baseline generation

        return LoadedModel(
            alias=alias,
            model=model,
            tokenizer=tokenizer,
            info=model_info,
            mem_gb=model_info["mem_gb"] + dflash_draft_mem_gb,
            dflash_draft=dflash_draft,
            dflash_draft_ref=dflash_draft_ref,
            dflash_draft_mem_gb=dflash_draft_mem_gb,
        )

    # ═══════════════════════════════════════════════════
    #  Eviction
    # ═══════════════════════════════════════════════════

    async def _evict_for(self, needed_gb: float) -> bool:
        """
        Evict least-recently-used idle models to free needed_gb.
        Must be called with load_lock held. Returns True if enough freed.
        """
        # Sort by last_used_at ascending (oldest first)
        candidates = sorted(
            [(a, m) for a, m in self.loaded.items() if m.ref_count == 0],
            key=lambda x: x[1].last_used_at,
        )

        freed = 0.0
        for alias, lm in candidates:
            if freed >= needed_gb:
                break
            self._unload_sync(alias)
            freed += lm.mem_gb

        return freed >= needed_gb

    async def unload(self, alias: str) -> float:
        """
        Explicitly unload a model. Returns GB freed, or 0 if not loaded.
        Raises RuntimeError if model has in-flight requests.
        """
        async with self.load_lock:
            if alias not in self.loaded:
                return 0.0
            lm = self.loaded[alias]
            if lm.ref_count > 0:
                raise RuntimeError(
                    f"Cannot unload '{alias}': {lm.ref_count} request(s) in flight"
                )
            mem = lm.mem_gb
            self._unload_sync(alias)
            return mem

    def _unload_sync(self, alias: str):
        """Remove model from memory. Caller must ensure ref_count == 0."""
        lm = self.loaded.pop(alias, None)
        if lm:
            del lm.model
            del lm.tokenizer
            if lm.dflash_draft is not None:
                del lm.dflash_draft
            gc.collect()
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:
                pass

    # ═══════════════════════════════════════════════════
    #  Idle evictor (background task)
    # ═══════════════════════════════════════════════════

    async def start_evictor(self):
        """Start the background idle evictor task."""
        self._evictor_task = asyncio.create_task(self._evictor_loop())

    async def stop_evictor(self):
        if self._evictor_task:
            self._evictor_task.cancel()
            try:
                await self._evictor_task
            except asyncio.CancelledError:
                pass

    async def _evictor_loop(self):
        """Periodically check for idle models and evict them."""
        while True:
            await asyncio.sleep(EVICTOR_INTERVAL)
            now = time.time()
            to_evict = []
            for alias, lm in self.loaded.items():
                idle_sec = now - lm.last_used_at
                if lm.ref_count == 0 and idle_sec > IDLE_TIMEOUT:
                    to_evict.append((alias, idle_sec))

            for alias, idle_sec in to_evict:
                async with self.load_lock:
                    # Re-check under lock
                    if alias in self.loaded and self.loaded[alias].ref_count == 0:
                        lm = self.loaded[alias]
                        if (now - lm.last_used_at) > IDLE_TIMEOUT:
                            self._unload_sync(alias)

    # ═══════════════════════════════════════════════════
    #  Ref counting (for in-flight request protection)
    # ═══════════════════════════════════════════════════

    def acquire(self, alias: str):
        """Increment ref count before generation. Prevents eviction."""
        if alias in self.loaded:
            self.loaded[alias].ref_count += 1
            self.loaded[alias].last_used_at = time.time()

    def release(self, alias: str):
        """Decrement ref count after generation."""
        if alias in self.loaded:
            self.loaded[alias].ref_count = max(0, self.loaded[alias].ref_count - 1)

    # ═══════════════════════════════════════════════════
    #  Speculative decoding
    # ═══════════════════════════════════════════════════

    def get_draft_model(self):
        """
        Return the smallest loaded model's mlx object for speculative decoding.
        Returns None if no suitable draft model is loaded.
        """
        for tier in TIER_ORDER:
            if tier in self.loaded:
                return self.loaded[tier].model
        return None

    # ═══════════════════════════════════════════════════
    #  Catalog info
    # ═══════════════════════════════════════════════════

    def catalog_status(self) -> list[dict]:
        """Return full catalog with load status for each tier."""
        result = []
        for tier in TIER_ORDER:
            candidates = MODEL_CATALOG.get(tier, [])
            default = candidates[0] if candidates else None
            status = "loaded" if tier in self.loaded else (
                "loading" if tier in self.loading_aliases else "available"
            )
            result.append({
                "tier": tier,
                "status": status,
                "default_model": default["name"] if default else None,
                "mem_gb": default["mem_gb"] if default else None,
                "params_b": default["params_b"] if default else None,
                "candidates": len(candidates),
                "capabilities": default.get("capabilities", {}) if default else {},
            })
        return result

    def get_load_status(self, load_id: str) -> LoadStatus | None:
        return self.load_statuses.get(load_id)

    # ═══════════════════════════════════════════════════
    #  Preload (for startup)
    # ═══════════════════════════════════════════════════

    async def preload(self, aliases: list[str]):
        """Load multiple models at startup."""
        for alias in aliases:
            try:
                await self.ensure_loaded(alias, timeout=300)
            except Exception as e:
                pass
