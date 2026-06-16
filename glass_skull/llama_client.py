from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class LlamaServerStatus:
    url: str
    online: bool
    latency_ms: float | None
    models: list[str]
    glass_available: bool
    glass_info: dict[str, Any]
    error: str | None = None


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.strip().rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    return base_url


def _join_url(base_url: str, path: str) -> str:
    return normalize_base_url(base_url) + "/" + path.lstrip("/")


def _request_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 20.0) -> dict[str, Any]:
    data = None
    headers = {
        "Accept": "application/json",
        "User-Agent": "glass-skull/0.6",
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            if not body.strip():
                return {}
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:500]}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from {url}: {exc}") from exc


def check_server(base_url: str, timeout: float = 3.0) -> LlamaServerStatus:
    base_url = normalize_base_url(base_url)
    start = time.perf_counter()
    models: list[str] = []
    glass_available = False
    glass_info: dict[str, Any] = {}

    try:
        model_payload = _request_json("GET", _join_url(base_url, "/v1/models"), timeout=timeout)
        latency_ms = (time.perf_counter() - start) * 1000
        for item in model_payload.get("data", []):
            model_id = item.get("id") or item.get("model")
            if model_id:
                models.append(str(model_id))
    except Exception as exc:
        return LlamaServerStatus(
            url=base_url,
            online=False,
            latency_ms=None,
            models=[],
            glass_available=False,
            glass_info={},
            error=str(exc),
        )

    try:
        glass_info = _request_json("GET", _join_url(base_url, "/glass-skull/info"), timeout=timeout)
        glass_available = True
    except Exception:
        glass_info = {}
        glass_available = False

    return LlamaServerStatus(
        url=base_url,
        online=True,
        latency_ms=latency_ms,
        models=models,
        glass_available=glass_available,
        glass_info=glass_info,
        error=None,
    )


def chat_completion(
    base_url: str,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    system_prompt: str | None = None,
    timeout: float = 120.0,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "messages": messages,
        "max_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "stream": False,
    }
    result = _request_json("POST", _join_url(base_url, "/v1/chat/completions"), payload=payload, timeout=timeout)
    choices = result.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content")
    if content is not None:
        return str(content)
    text = choices[0].get("text")
    return str(text or "")


def trace_summary(
    base_url: str,
    prompt: str,
    layers: list[int] | None = None,
    streams: list[str] | None = None,
    top_k: int = 32,
    timeout: float = 120.0,
) -> dict[str, Any]:
    payload = {
        "prompt": prompt,
        "layers": layers or [],
        "streams": streams or ["resid_post"],
        "top_k": int(top_k),
    }
    return _request_json("POST", _join_url(base_url, "/glass-skull/trace-summary"), payload=payload, timeout=timeout)
