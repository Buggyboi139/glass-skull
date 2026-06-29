from __future__ import annotations

import re
from typing import Any, Iterable


NODE_ID_RE = re.compile(r"^L(?P<layer>\d+)-N(?P<node>\d+)$")
TOKEN_SCOPES = {"all", "prompt", "generated"}
TOKEN_SCOPE_ALIASES = {
    "all tokens": "all",
    "prompt tokens": "prompt",
    "generated tokens": "generated",
}


def _node_validation_error(value: str) -> ValueError:
    return ValueError(f"Malformed activation node ID {value!r}. Expected L<layer>-N<node>, for example L36-N175.")


def _parse_node_id(value: Any) -> dict[str, Any]:
    node_id = str(value or "").strip()
    match = NODE_ID_RE.match(node_id)
    if not match:
        raise _node_validation_error(node_id)
    layer = int(match.group("layer"))
    node = int(match.group("node"))
    canonical = f"L{layer}-N{node}"
    return {
        "node_id": canonical,
        "layer": layer,
        "node": node,
        "node_range": [node, node],
    }


def _split_targets(raw: str | Iterable[Any]) -> list[str]:
    if isinstance(raw, str):
        return [part.strip() for part in re.split(r"[,\n\r]+", raw) if part.strip()]
    return [str(part).strip() for part in raw if str(part).strip()]


def parse_activation_node_ids(raw: str | Iterable[Any]) -> list[dict[str, Any]]:
    parts = _split_targets(raw)
    if not parts:
        raise ValueError("At least one activation node ID is required.")

    targets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for part in parts:
        target = _parse_node_id(part)
        if target["node_id"] in seen:
            continue
        seen.add(target["node_id"])
        targets.append(target)
    return targets


def _coerce_targets(targets: str | Iterable[Any]) -> list[dict[str, Any]]:
    if isinstance(targets, str):
        return parse_activation_node_ids(targets)

    resolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in targets:
        if isinstance(item, dict):
            base = _parse_node_id(item.get("node_id") or item.get("nodeId") or item.get("groupId"))
            merged = {**base}
            node_range = item.get("node_range", item.get("nodeRange"))
            if isinstance(node_range, (list, tuple)) and len(node_range) == 2:
                try:
                    start, end = int(node_range[0]), int(node_range[1])
                    if start <= end:
                        merged["node_range"] = [start, end]
                except (TypeError, ValueError):
                    pass
            if "activation_value" in item:
                merged["activation_value"] = item.get("activation_value")
            elif "activationValue" in item:
                merged["activation_value"] = item.get("activationValue")
        else:
            merged = _parse_node_id(item)
        if merged["node_id"] in seen:
            continue
        seen.add(merged["node_id"])
        resolved.append(merged)
    if not resolved:
        raise ValueError("At least one activation node ID is required.")
    return resolved


def normalize_direction(direction: str) -> str:
    value = str(direction or "").strip().lower()
    if value in {"toward", "positive", "+"}:
        return "toward"
    if value in {"away", "negative", "-"}:
        return "away"
    raise ValueError("Direction must be Toward or Away.")


def normalize_token_scope(token_scope: str) -> str:
    value = str(token_scope or "").strip().lower().replace("_", " ")
    value = TOKEN_SCOPE_ALIASES.get(value, value)
    if value not in TOKEN_SCOPES:
        raise ValueError("Token scope must be one of: all, prompt, generated.")
    return value


def build_direct_activation_steering_payload(
    enabled: bool,
    targets: str | Iterable[Any],
    direction: str,
    strength: float,
    token_scope: str,
) -> dict[str, Any] | None:
    if not enabled:
        return None

    resolved_direction = normalize_direction(direction)
    magnitude = abs(float(strength))
    signed_strength = magnitude if resolved_direction == "toward" else -magnitude
    return {
        "enabled": True,
        "targets": _coerce_targets(targets),
        "direction": resolved_direction,
        "strength": signed_strength,
        "token_scope": normalize_token_scope(token_scope),
    }


def selected_activation_target_from_group(group: dict[str, Any]) -> dict[str, Any]:
    target = _parse_node_id(group.get("nodeId") or group.get("groupId"))
    layer_value = group.get("layer")
    if layer_value is None:
        layer_value = str(group.get("layerId") or "").replace("L", "")
    try:
        layer = int(layer_value)
        if layer != target["layer"]:
            target["layer"] = layer
    except (TypeError, ValueError):
        pass

    node_range = group.get("nodeRange") or group.get("node_range")
    if isinstance(node_range, (list, tuple)) and len(node_range) == 2:
        try:
            start, end = int(node_range[0]), int(node_range[1])
            if start <= end:
                target["node_range"] = [start, end]
        except (TypeError, ValueError):
            pass

    target["channel"] = target["node"]
    target["activation_value"] = float(group.get("activationValue", group.get("activation", 0.0)) or 0.0)
    return target
