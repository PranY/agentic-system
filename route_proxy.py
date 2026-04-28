#!/usr/bin/env python3
"""Routing proxy for multi-instance MLX serving.

Listens on a single port (default 8800) and forwards each request to the
appropriate backend based on the `model` field. Used with start_multi.sh
so clients like Hermes see a unified endpoint while models run as separate
processes for true GPU parallelism.

Routes:
  POST /v1/chat/completions       — route by model name
  POST /v1/chat/completions/batch — route by model name
  POST /v1/models/load            — route by model name
  POST /v1/models/unload          — route by model name
  GET  /v1/models/{model_id}      — route by model_id
  GET  /v1/models                 — aggregated from all backends
  GET  /v1/models/catalog         — aggregated from all backends
  GET  /health                    — aggregate
  GET  /stats                     — aggregate
  GET  /stats/live                — pick first backend
  Other GETs                      — pick first backend
"""

import asyncio
import json
import os
import sys

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

# Same model alias mapping as serve.py
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


def parse_routes() -> dict[str, str]:
    """Parse ROUTES env var: 'mini:8810,small:8811,medium:8812' -> dict."""
    routes_str = os.environ.get("ROUTES", "mini:8810,small:8811")
    routes = {}
    for pair in routes_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        tier, port = pair.split(":")
        routes[tier.strip()] = f"http://localhost:{port.strip()}"
    return routes


ROUTES = parse_routes()
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8800"))


def resolve_alias(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def backend_for(model: str) -> str:
    alias = resolve_alias(model)
    if alias in ROUTES:
        return ROUTES[alias]
    # Fallback: first available backend
    return next(iter(ROUTES.values()))


app = FastAPI(title="MLX Routing Proxy")
client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def startup():
    global client
    client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=5.0))
    print(f"Proxy on :{PROXY_PORT} routing to: {ROUTES}", flush=True)


@app.on_event("shutdown")
async def shutdown():
    if client:
        await client.aclose()


async def _forward_streaming(backend: str, path: str, body: bytes, headers: dict):
    """Forward a request and stream the response back."""
    url = f"{backend}{path}"
    req = client.build_request("POST", url, content=body, headers=headers)
    resp = await client.send(req, stream=True)

    async def gen():
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            await resp.aclose()

    response_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in ("content-length", "content-encoding", "transfer-encoding")
    }
    return StreamingResponse(
        gen(),
        status_code=resp.status_code,
        headers=response_headers,
        media_type=resp.headers.get("content-type", "application/json"),
    )


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.body()
    try:
        payload = json.loads(body)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    model = payload.get("model", "")
    backend = backend_for(model)
    n_msgs = len(payload.get("messages", []))
    has_tools = bool(payload.get("tools"))
    stream = payload.get("stream", False)
    print(f"[ROUTE] model={model} → {backend} | msgs={n_msgs} tools={has_tools} stream={stream}",
          flush=True)
    headers = {"content-type": "application/json"}
    return await _forward_streaming(backend, "/v1/chat/completions", body, headers)


@app.post("/v1/chat/completions/batch")
async def batch(request: Request):
    body = await request.body()
    payload = json.loads(body)
    backend = backend_for(payload.get("model", ""))
    headers = {"content-type": "application/json"}
    return await _forward_streaming(backend, "/v1/chat/completions/batch", body, headers)


@app.post("/v1/models/load")
async def load_model(request: Request):
    body = await request.body()
    payload = json.loads(body)
    backend = backend_for(payload.get("model", ""))
    r = await client.post(f"{backend}/v1/models/load", content=body,
                           headers={"content-type": "application/json"})
    return JSONResponse(content=r.json(), status_code=r.status_code)


@app.post("/v1/models/unload")
async def unload_model(request: Request):
    body = await request.body()
    payload = json.loads(body)
    backend = backend_for(payload.get("model", ""))
    r = await client.post(f"{backend}/v1/models/unload", content=body,
                           headers={"content-type": "application/json"})
    return JSONResponse(content=r.json(), status_code=r.status_code)


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    if model_id == "catalog":
        return await model_catalog()
    backend = backend_for(model_id)
    r = await client.get(f"{backend}/v1/models/{resolve_alias(model_id)}")
    return JSONResponse(content=r.json(), status_code=r.status_code)


@app.get("/v1/models")
async def list_models():
    """Aggregate /v1/models across all backends."""
    all_models = []
    seen = set()
    for tier, backend in ROUTES.items():
        try:
            r = await client.get(f"{backend}/v1/models", timeout=5.0)
            data = r.json().get("data", [])
            for m in data:
                if m["id"] not in seen:
                    all_models.append(m)
                    seen.add(m["id"])
        except Exception:
            continue
    # Also expose all configured tiers as available even if not loaded
    for tier in ROUTES:
        if tier not in seen:
            all_models.append({
                "id": tier,
                "object": "model",
                "owned_by": "local-mlx",
                "status": "available",
            })
    return {"object": "list", "data": all_models}


@app.get("/v1/models/catalog")
async def model_catalog():
    """Aggregate catalog from all backends."""
    tiers = []
    memory = {"used_gb": 0.0, "free_gb": 0.0, "models": {}}
    seen_tiers = set()
    for tier, backend in ROUTES.items():
        try:
            r = await client.get(f"{backend}/v1/models/catalog", timeout=5.0)
            data = r.json()
            for t in data.get("tiers", []):
                if t["tier"] not in seen_tiers:
                    tiers.append(t)
                    seen_tiers.add(t["tier"])
            mem = data.get("memory", {})
            memory["used_gb"] += mem.get("used_gb", 0.0)
            memory["free_gb"] = max(memory["free_gb"], mem.get("free_gb", 0.0))
            for k, v in mem.get("models", {}).items():
                memory["models"][k] = v
        except Exception:
            continue
    return {"tiers": tiers, "memory": memory}


@app.get("/v1/memory")
async def memory():
    """Aggregate memory across backends."""
    used = 0.0
    free = 60.0
    models = {}
    for tier, backend in ROUTES.items():
        try:
            r = await client.get(f"{backend}/v1/memory", timeout=5.0)
            data = r.json()
            used += data.get("used_gb", 0.0)
            free = min(free, data.get("free_gb", 60.0))
            for k, v in data.get("models", {}).items():
                models[k] = v
        except Exception:
            continue
    return {"used_gb": used, "free_gb": free, "models": models}


@app.get("/health")
async def health():
    statuses = {}
    all_ok = True
    for tier, backend in ROUTES.items():
        try:
            r = await client.get(f"{backend}/health", timeout=3.0)
            statuses[tier] = r.json()
        except Exception as e:
            statuses[tier] = {"status": "down", "error": str(e)}
            all_ok = False
    return {"status": "ok" if all_ok else "degraded", "backends": statuses}


@app.get("/stats")
async def stats():
    """Aggregate stats across backends."""
    backends = {}
    for tier, backend in ROUTES.items():
        try:
            r = await client.get(f"{backend}/stats", timeout=3.0)
            backends[tier] = r.json()
        except Exception as e:
            backends[tier] = {"error": str(e)}
    return {"backends": backends, "routes": ROUTES}


@app.get("/stats/live", response_class=HTMLResponse)
async def stats_live():
    """Forward to first backend's live dashboard."""
    backend = next(iter(ROUTES.values()))
    r = await client.get(f"{backend}/stats/live", timeout=10.0)
    return HTMLResponse(content=r.text, status_code=r.status_code)


# Generic GET fallback for any other endpoint (forwards to first backend)
@app.get("/{full_path:path}")
async def generic_get(full_path: str, request: Request):
    backend = next(iter(ROUTES.values()))
    url = f"{backend}/{full_path}"
    if request.url.query:
        url += f"?{request.url.query}"
    try:
        r = await client.get(url, timeout=10.0)
        try:
            return JSONResponse(content=r.json(), status_code=r.status_code)
        except Exception:
            return HTMLResponse(content=r.text, status_code=r.status_code)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT, log_level="info")
