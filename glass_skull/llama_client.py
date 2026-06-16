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


def _request_text(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> str:
    data = None
    headers = {
        "Accept": "application/json, text/event-stream",
        "User-Agent": "glass-skull/0.7",
    }

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:1000]}") from exc


def _request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    body = _request_text(method, url, payload=payload, timeout=timeout)
    if not body.strip():
        return {}

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        # llama.cpp should respect stream=false, but if something upstream returns SSE,
        # don't make the whole app faceplant with a useless JSON error.
        if "data:" in body:
            return {"_sse_text": body}
        raise RuntimeError(f"Invalid JSON from {url}: {exc}\nBody preview:\n{body[:1000]}") from exc


def _content_from_message_content(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    # OpenAI-ish multimodal/content-part style:
    # [{"type":"text","text":"hello"}]
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
        return "".join(parts)

    return str(content)


def _extract_content(obj: Any) -> str:
    """Extract generated text from OpenAI, llama.cpp, legacy, and SSE-like shapes."""

    if obj is None:
        return ""

    if isinstance(obj, str):
        return obj

    if not isinstance(obj, dict):
        return str(obj)

    # Non-stream JSON response variants.
    for key in ("response", "content", "text", "generated_text", "completion"):
        value = obj.get(key)
        if isinstance(value, str) and value:
            return value

    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        collected = []

        for choice in choices:
            if not isinstance(choice, dict):
                continue

            message = choice.get("message")
            if isinstance(message, dict):
                content = _content_from_message_content(message.get("content"))
                if content:
                    collected.append(content)

            delta = choice.get("delta")
            if isinstance(delta, dict):
                content = _content_from_message_content(delta.get("content"))
                if content:
                    collected.append(content)

            for key in ("content", "text"):
                content = _content_from_message_content(choice.get(key))
                if content:
                    collected.append(content)

        if collected:
            return "".join(collected)

    # SSE fallback packed into _sse_text by _request_json.
    sse_text = obj.get("_sse_text")
    if isinstance(sse_text, str):
        return _extract_sse_content(sse_text)

    # Some APIs nest the real payload.
    for key in ("data", "result", "output"):
        nested = obj.get(key)
        if nested:
            content = _extract_content(nested)
            if content:
                return content

    err = obj.get("error")
    if err:
        raise RuntimeError(f"llama.cpp returned an error payload: {err}")

    return ""


def _extract_sse_content(text: str) -> str:
    chunks = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue

        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue

        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue

        content = _extract_content(obj)
        if content:
            chunks.append(content)

    return "".join(chunks)


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


def _first_model_id(base_url: str) -> str | None:
    try:
        payload = _request_json("GET", _join_url(base_url, "/v1/models"), timeout=5.0)
        data = payload.get("data", [])
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                model_id = first.get("id") or first.get("model")
                if model_id:
                    return str(model_id)
    except Exception:
        return None
    return None


def chat_completion(
    base_url: str,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    system_prompt: str | None = None,
    timeout: float = 300.0,
) -> str:
    base_url = normalize_base_url(base_url)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "messages": messages,
        "max_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "stream": False,
    }

    # llama.cpp usually does not require model, but some OpenAI-compatible servers care.
    model_id = _first_model_id(base_url)
    if model_id:
        payload["model"] = model_id

    # Primary OpenAI-compatible chat endpoint.
    result = _request_json(
        "POST",
        _join_url(base_url, "/v1/chat/completions"),
        payload=payload,
        timeout=timeout,
    )
    content = _extract_content(result)

    # Fallback to legacy llama.cpp /completion if chat returns empty.
    if not content.strip():
        legacy_payload = {
            "prompt": prompt,
            "n_predict": int(max_new_tokens),
            "temperature": float(temperature),
            "stream": False,
        }
        legacy_result = _request_json(
            "POST",
            _join_url(base_url, "/completion"),
            payload=legacy_payload,
            timeout=timeout,
        )
        content = _extract_content(legacy_result)

    if not content.strip():
        raise RuntimeError(
            "llama.cpp returned no assistant text. "
            "The server generated something in the terminal, but the HTTP response body did not contain a recognized text field."
        )

    return content


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
