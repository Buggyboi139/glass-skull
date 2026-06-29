from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode


@dataclass
class LlamaServerStatus:
    url: str
    online: bool
    latency_ms: float | None
    models: list[str]
    glass_available: bool
    glass_info: dict[str, Any]
    steering_supported: bool = False
    direct_activation_steering_status: dict[str, str] | None = None
    activation_patch_supported: bool = False
    error: str | None = None


REASONING_OFF_PARAMS: dict[str, Any] = {
    "reasoning_budget": 0,
    "enable_thinking": False,
    "chat_template_kwargs": {"enable_thinking": False},
}
LOCAL_DIRECT_SYSTEM_PROMPT = (
    "Answer directly and concisely. Do not include reasoning, chain-of-thought, "
    "scratchpad, analysis, explanation sections, or thinking tags."
)


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.strip().rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    return base_url


def _join_url(base_url: str, path: str) -> str:
    return normalize_base_url(base_url) + "/" + path.lstrip("/")


def _append_query(url: str, params: dict[str, str]) -> str:
    clean = {key: value for key, value in params.items() if value}
    if not clean:
        return url
    separator = "&" if "?" in url else "?"
    return url + separator + urlencode(clean)


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
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc.reason}") from exc


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


def _strip_reasoning(text: str) -> str:
    text = re.sub(r"(?is)<think\b[^>]*>.*?(?:</think>|$)\s*", "", text)
    text = re.sub(r"(?is)<thinking\b[^>]*>.*?(?:</thinking>|$)\s*", "", text)
    text = re.sub(
        r"(?ims)\n+\s*(reasoning|analysis|chain[- ]of[- ]thought|thought process)\s*:\s*.*\Z",
        "",
        text,
    )
    return text.strip()


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


def _json_shape(value: Any) -> str:
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


def _request_json_object(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 120.0,
    context: str = "llama.cpp request",
) -> dict[str, Any]:
    try:
        result = _request_json(method, url, payload=payload, timeout=timeout)
    except RuntimeError as exc:
        raise RuntimeError(f"{context} failed: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"{context} failed: {exc}") from exc

    if not isinstance(result, dict):
        raise RuntimeError(f"{context} returned JSON {_json_shape(result)}, expected object")

    if result.get("error"):
        raise RuntimeError(f"{context} returned error: {result['error']}")

    return result


def per_request_steering_supported(glass_info: dict[str, Any] | None) -> bool:
    if not isinstance(glass_info, dict):
        return False
    capabilities = glass_info.get("capabilities")
    if not isinstance(capabilities, dict):
        return False
    steering = capabilities.get("steering")
    if not isinstance(steering, dict):
        return False
    per_request = steering.get("per_request")
    return isinstance(per_request, dict) and per_request.get("supported") is True


def _contract_mentions_direct_activation(section: dict[str, Any]) -> bool:
    text = json.dumps(section, sort_keys=True).lower()
    return "glass_skull.direct_activation_steering" in text or "direct_activation_steering" in text


def direct_activation_steering_status(glass_info: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(glass_info, dict) or not glass_info:
        return {
            "status": "unsupported",
            "label": "Unsupported",
            "message": "This llama.cpp server does not advertise Glass Skull direct activation steering.",
        }

    capabilities = glass_info.get("capabilities")
    if not isinstance(capabilities, dict):
        return {
            "status": "unsupported",
            "label": "Unsupported",
            "message": "This llama.cpp server exposes Glass Skull info without capability metadata.",
        }

    candidates: list[tuple[dict[str, Any], bool]] = []
    for key in ("direct_activation_steering", "activation_steering", "direct_activation"):
        section = capabilities.get(key)
        if isinstance(section, dict):
            candidates.append((section, True))

    steering = capabilities.get("steering")
    if isinstance(steering, dict):
        for key in ("direct_activation_steering", "direct_activation", "activation_steering"):
            section = steering.get(key)
            if isinstance(section, dict):
                candidates.append((section, True))
        per_request = steering.get("per_request")
        if isinstance(per_request, dict):
            candidates.append((per_request, False))

    for section, _direct_named in candidates:
        if section.get("supported") is True and _contract_mentions_direct_activation(section):
            return {
                "status": "supported",
                "label": "Supported",
                "message": "This llama.cpp server advertises direct activation steering.",
            }

    for section, direct_named in candidates:
        if direct_named or _contract_mentions_direct_activation(section):
            reason = str(section.get("reason") or "Direct activation steering is advertised but not fully supported.")
            return {"status": "partial", "label": "Partial", "message": reason}

    if per_request_steering_supported(glass_info):
        return {
            "status": "partial",
            "label": "Partial",
            "message": "This llama.cpp server advertises only the legacy steering contract, not direct activation steering.",
        }

    return {
        "status": "unsupported",
        "label": "Unsupported",
        "message": "This llama.cpp server does not advertise direct activation steering support.",
    }


def direct_activation_steering_supported(glass_info: dict[str, Any] | None) -> bool:
    return direct_activation_steering_status(glass_info)["status"] == "supported"


def activation_patch_supported(glass_info: dict[str, Any] | None) -> bool:
    if not isinstance(glass_info, dict):
        return False
    capabilities = glass_info.get("capabilities")
    if not isinstance(capabilities, dict):
        return False
    for key in ("activation_patch", "activation_patching", "patching"):
        patching = capabilities.get(key)
        if not isinstance(patching, dict):
            continue
        if patching.get("supported") is True:
            return True
        per_request = patching.get("per_request")
        if isinstance(per_request, dict) and per_request.get("supported") is True:
            return True
    return False


def trace_vectors_supported(glass_info: dict[str, Any] | None) -> bool:
    if not isinstance(glass_info, dict):
        return False
    capabilities = glass_info.get("capabilities")
    if not isinstance(capabilities, dict):
        return False
    trace = capabilities.get("trace")
    if not isinstance(trace, dict):
        return False
    for key in ("layer_inputs", "activations"):
        section = trace.get(key)
        if isinstance(section, dict) and section.get("supported") is True and section.get("vectors") is True:
            return True
    return False


def activation_patch_diagnostic(glass_info: dict[str, Any] | None) -> str:
    if activation_patch_supported(glass_info):
        return "This llama.cpp server advertises per-request activation patch support."
    if not isinstance(glass_info, dict) or not glass_info:
        return "This llama.cpp server does not advertise Glass Skull activation patch capabilities."
    capabilities = glass_info.get("capabilities") if isinstance(glass_info.get("capabilities"), dict) else {}
    for key in ("activation_patch", "activation_patching", "patching"):
        patching = capabilities.get(key)
        if isinstance(patching, dict):
            reason = patching.get("reason")
            per_request = patching.get("per_request")
            if isinstance(per_request, dict):
                reason = per_request.get("reason") or reason
            return str(reason or "Activation patch capability is present but not marked supported.")
    return "This llama.cpp server does not advertise activation patching; recipe save/load and baseline comparison are available only."

def _coerce_layers(layers: list[int] | None) -> list[int] | None:
    if layers is None:
        return None
    if isinstance(layers, (str, bytes)):
        raise ValueError("layers must be a list of integers")
    return [int(layer) for layer in layers]


def _coerce_streams(streams: list[str] | None) -> list[str] | None:
    if streams is None:
        return None
    if isinstance(streams, (str, bytes)):
        raise ValueError("streams must be a list of stream names")
    return [str(stream).strip() for stream in streams if str(stream).strip()]


def _glass_trace_payload(
    prompt: str,
    model_alias: str | None = None,
    layers: list[int] | None = None,
    streams: list[str] | None = None,
    max_new_tokens: int | None = None,
    top_k: int | None = None,
    with_pieces: bool | None = None,
    include_vectors: bool | None = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"prompt": prompt}

    if model_alias:
        payload["model"] = str(model_alias).strip()
        payload["model_alias"] = str(model_alias).strip()
    if layers is not None:
        payload["layers"] = _coerce_layers(layers)
    if streams is not None:
        payload["streams"] = _coerce_streams(streams)
    if max_new_tokens is not None:
        payload["max_new_tokens"] = int(max_new_tokens)
        payload["max_tokens"] = int(max_new_tokens)
    if top_k is not None:
        payload["top_k"] = int(top_k)
    if with_pieces is not None:
        payload["with_pieces"] = bool(with_pieces)
    if include_vectors is not None:
        payload["include_vectors"] = bool(include_vectors)
    payload["capture"] = {
        "prompt_tokens": True,
        "layer_inputs": bool(layers is not None or streams is not None),
        "next_token_logits": False,
    }

    return payload


def _validate_trace_response(result: dict[str, Any], context: str) -> None:
    list_fields = ("tokens", "trace_layers")
    for field in list_fields:
        if field in result and not isinstance(result[field], list):
            raise RuntimeError(
                f"{context} returned invalid shape: field {field!r} is "
                f"{_json_shape(result[field])}, expected array"
            )

    text_fields = ("content", "completion", "generated_text", "text")
    for field in text_fields:
        if field in result and result[field] is not None and not isinstance(result[field], str):
            raise RuntimeError(
                f"{context} returned invalid shape: field {field!r} is "
                f"{_json_shape(result[field])}, expected string"
            )

    activations = result.get("activations")
    if activations is not None and not isinstance(activations, (dict, list)):
        raise RuntimeError(
            f"{context} returned invalid shape: field 'activations' is "
            f"{_json_shape(activations)}, expected object or array"
        )

    layer_norms = result.get("layer_norms")
    if layer_norms is not None and not isinstance(layer_norms, list):
        raise RuntimeError(
            f"{context} returned invalid shape: field 'layer_norms' is "
            f"{_json_shape(layer_norms)}, expected array"
        )

    layer_inputs = result.get("layer_inputs")
    if layer_inputs is not None and not isinstance(layer_inputs, list):
        raise RuntimeError(
            f"{context} returned invalid shape: field 'layer_inputs' is "
            f"{_json_shape(layer_inputs)}, expected array"
        )

    logits = result.get("logits")
    if logits is not None and not isinstance(logits, (dict, list)):
        raise RuntimeError(
            f"{context} returned invalid shape: field 'logits' is "
            f"{_json_shape(logits)}, expected object or array"
        )

    prompt = result.get("prompt")
    if prompt is not None:
        if not isinstance(prompt, dict):
            raise RuntimeError(
                f"{context} returned invalid shape: field 'prompt' is "
                f"{_json_shape(prompt)}, expected object"
            )
        traces = prompt.get("traces")
        if traces is not None:
            if not isinstance(traces, list):
                raise RuntimeError(
                    f"{context} returned invalid shape: field 'prompt.traces' is "
                    f"{_json_shape(traces)}, expected array"
                )
            for index, trace in enumerate(traces):
                if not isinstance(trace, dict):
                    raise RuntimeError(
                        f"{context} returned invalid shape: field 'prompt.traces[{index}]' is "
                        f"{_json_shape(trace)}, expected object"
                    )
                tokens = trace.get("tokens")
                if tokens is not None and not isinstance(tokens, list):
                    raise RuntimeError(
                        f"{context} returned invalid shape: field 'prompt.traces[{index}].tokens' is "
                        f"{_json_shape(tokens)}, expected array"
                    )
                pieces = trace.get("pieces")
                if pieces is not None and not isinstance(pieces, list):
                    raise RuntimeError(
                        f"{context} returned invalid shape: field 'prompt.traces[{index}].pieces' is "
                        f"{_json_shape(pieces)}, expected array"
                    )


def check_server(base_url: str, timeout: float = 3.0, model_alias: str | None = None) -> LlamaServerStatus:
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
            steering_supported=False,
            direct_activation_steering_status=direct_activation_steering_status({}),
            activation_patch_supported=False,
            error=str(exc),
        )

    try:
        glass_model = _resolve_model_id(base_url, model_alias)
        glass_info = get_glass_info(base_url, model_alias=glass_model, timeout=timeout)
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
        steering_supported=direct_activation_steering_supported(glass_info),
        direct_activation_steering_status=direct_activation_steering_status(glass_info),
        activation_patch_supported=activation_patch_supported(glass_info),
        error=None,
    )


def get_glass_info(base_url: str, model_alias: str | None = None, timeout: float = 5.0) -> dict[str, Any]:
    """Read Glass Skull model metadata from a patched llama.cpp server."""

    url = _join_url(base_url, "/glass-skull/info")
    if model_alias:
        url = _append_query(url, {"model": str(model_alias).strip()})
    return _request_json_object(
        "GET",
        url,
        timeout=timeout,
        context="Glass Skull info request",
    )


def trace_glass_prompt(
    base_url: str,
    prompt: str,
    model_alias: str | None = None,
    layers: list[int] | None = None,
    streams: list[str] | None = None,
    max_new_tokens: int | None = None,
    top_k: int | None = None,
    with_pieces: bool | None = None,
    include_vectors: bool | None = True,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Request a local llama.cpp trace payload.

    The endpoint returns local llama.cpp trace data when the managed Glass Skull
    patch is present. It may return token-only diagnostics when activation
    capture is unavailable.
    """

    payload = _glass_trace_payload(
        prompt=prompt,
        model_alias=model_alias,
        layers=layers,
        streams=streams,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        with_pieces=with_pieces,
        include_vectors=include_vectors,
    )
    result = _request_json_object(
        "POST",
        _join_url(base_url, "/glass-skull/trace"),
        payload=payload,
        timeout=timeout,
        context="Glass Skull trace request",
    )
    if result.get("supported") is False:
        reason = result.get("reason") or result.get("error") or "server reported trace unsupported"
        raise RuntimeError(str(reason))
    _validate_trace_response(result, "Glass Skull trace request")
    return result


def _model_items(base_url: str) -> list[dict[str, Any]]:
    try:
        payload = _request_json("GET", _join_url(base_url, "/v1/models"), timeout=5.0)
        data = payload.get("data", [])
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    except Exception:
        pass
    return []


def _model_id(item: dict[str, Any]) -> str | None:
    model_id = item.get("id") or item.get("model")
    return str(model_id) if model_id else None


def _model_status_value(item: dict[str, Any]) -> str:
    status = item.get("status")
    if isinstance(status, dict):
        return str(status.get("value", "")).lower()
    return str(status or "").lower()


def _first_model_id(base_url: str) -> str | None:
    items = _model_items(base_url)
    for item in items:
        if _model_status_value(item) == "loaded":
            model_id = _model_id(item)
            if model_id:
                return model_id
    for item in items:
        model_id = _model_id(item)
        if model_id:
            return model_id
    return None


def _resolve_model_id(base_url: str, model_alias: str | None) -> str | None:
    alias = (model_alias or "").strip()
    if not alias:
        return _first_model_id(base_url)

    # llama-server-router commonly exposes a "default" preset even when it is
    # failed/unloaded. In that case, prefer the actual loaded model to avoid a
    # 400/500 router error while still honoring explicit non-default aliases.
    if alias == "default":
        items = _model_items(base_url)
        for item in items:
            if _model_id(item) == alias and _model_status_value(item) == "loaded":
                return alias
        for item in items:
            if _model_status_value(item) == "loaded":
                model_id = _model_id(item)
                if model_id:
                    return model_id

    return alias


def chat_completion(
    base_url: str,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    system_prompt: str | None = None,
    messages: list[dict[str, str]] | None = None,
    model_alias: str | None = None,
    direct_activation_steering: dict[str, Any] | None = None,
    direct_activation_steering_supported: bool = False,
    activation_patch: dict[str, Any] | None = None,
    activation_patch_supported: bool = False,
    timeout: float = 300.0,
) -> str:
    base_url = normalize_base_url(base_url)

    if direct_activation_steering is not None and not direct_activation_steering_supported:
        raise RuntimeError("This llama.cpp server does not advertise direct activation steering support.")
    if activation_patch is not None and not activation_patch_supported:
        raise RuntimeError("This llama.cpp server does not advertise per-request activation patch support.")

    system_parts = [LOCAL_DIRECT_SYSTEM_PROMPT]
    if system_prompt:
        system_parts.append(system_prompt.strip())
    chat_messages = [{"role": "system", "content": "\n\n".join(part for part in system_parts if part)}]
    for message in messages or []:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            chat_messages.append({"role": role, "content": content})
    if not chat_messages or chat_messages[-1].get("role") != "user" or chat_messages[-1].get("content") != prompt:
        chat_messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "messages": chat_messages,
        "max_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "stream": False,
        **REASONING_OFF_PARAMS,
    }

    # llama.cpp direct servers often work without "model". Routers usually need
    # an explicit model/alias, so prefer the configured local alias when present.
    model_id = _resolve_model_id(base_url, model_alias)
    if model_id:
        payload["model"] = model_id
    if direct_activation_steering is not None:
        payload.setdefault("glass_skull", {})["direct_activation_steering"] = direct_activation_steering
        payload.setdefault("metadata", {}).setdefault("glass_skull", {})["direct_activation_steering"] = direct_activation_steering
    if activation_patch is not None:
        payload.setdefault("glass_skull", {})["activation_patch"] = activation_patch
        payload.setdefault("metadata", {}).setdefault("glass_skull", {})["activation_patch"] = activation_patch

    # Primary OpenAI-compatible chat endpoint.
    result = _request_json(
        "POST",
        _join_url(base_url, "/v1/chat/completions"),
        payload=payload,
        timeout=timeout,
    )
    content = _strip_reasoning(_extract_content(result))

    # Fallback to legacy llama.cpp /completion if chat returns empty.
    if not content.strip():
        history_lines = []
        for message in chat_messages:
            role = message["role"].title()
            history_lines.append(f"{role}: {message['content']}")
        legacy_prompt = "\n\n".join(history_lines) + "\n\nAssistant:"
        legacy_payload = {
            "prompt": legacy_prompt,
            "n_predict": int(max_new_tokens),
            "temperature": float(temperature),
            "stream": False,
            **REASONING_OFF_PARAMS,
        }
        if model_id:
            legacy_payload["model"] = model_id
        if direct_activation_steering is not None:
            legacy_payload.setdefault("glass_skull", {})["direct_activation_steering"] = direct_activation_steering
            legacy_payload.setdefault("metadata", {}).setdefault("glass_skull", {})["direct_activation_steering"] = direct_activation_steering
        if activation_patch is not None:
            legacy_payload.setdefault("glass_skull", {})["activation_patch"] = activation_patch
        legacy_result = _request_json(
            "POST",
            _join_url(base_url, "/completion"),
            payload=legacy_payload,
            timeout=timeout,
        )
        content = _strip_reasoning(_extract_content(legacy_result))

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
    return trace_glass_prompt(
        base_url,
        prompt,
        layers=layers,
        streams=streams,
        top_k=top_k,
        timeout=timeout,
    )
