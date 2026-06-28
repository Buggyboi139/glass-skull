from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from math import ceil, sqrt
from pathlib import Path
from typing import Any

import pandas as pd

from .experiment_store import latest_run_artifacts, load_run_artifact
from .node_annotations import compact_annotation_match, get_annotations_for_node, load_annotations, model_fingerprint


VISUALIZATION_MODES = {"single_prompt", "batch_overlay", "aggregate_heatmap", "compare_prompts"}
DATA_MODES = {"real_vectors", "top_dims_approx", "scalar_layer_summary", "aggregated", "unavailable"}
DEFAULT_TOP_K = 8
PROMPT_PATH_PALETTE = (
    {"name": "cyan", "color": "#62E4FF", "rgb": [98, 228, 255]},
    {"name": "amber", "color": "#FFC857", "rgb": [255, 200, 87]},
    {"name": "violet", "color": "#B7A4FF", "rgb": [183, 164, 255]},
    {"name": "mint", "color": "#72F2B6", "rgb": [114, 242, 182]},
    {"name": "rose", "color": "#FF8DB3", "rgb": [255, 141, 179]},
    {"name": "sky", "color": "#7EB6FF", "rgb": [126, 182, 255]},
    {"name": "lime", "color": "#D4F75F", "rgb": [212, 247, 95]},
    {"name": "coral", "color": "#FF9B73", "rgb": [255, 155, 115]},
    {"name": "indigo", "color": "#8EA0FF", "rgb": [142, 160, 255]},
    {"name": "gold", "color": "#FFE27A", "rgb": [255, 226, 122]},
)
PROMPT_LEGEND_LIMIT = 10
PROMPT_REUSE_OPACITIES = (1.0, 0.82, 0.68, 0.56)
PROMPT_REUSE_DASHES = ([], [12, 6], [4, 5], [14, 4, 4, 4])


@dataclass
class ActivationNode:
    run_id: str
    batch_id: str | None
    prompt_id: Any
    token_id: int | None
    layer: int
    node_id: str
    cluster_id: int | str | None
    activation: float
    normalized_activation: float
    node_range: list[int] | None
    token_text: str
    source_fields: list[str]
    confidence: float
    mode: str
    y: float = 0.5
    prompt_label: str = ""
    prompt_text: str = ""
    prompt_preview_list: list[str] = field(default_factory=list)
    prompt_observations: list[dict[str, Any]] = field(default_factory=list)
    source_row_index: int | None = None
    vector: list[float] | None = None
    real: bool = True


@dataclass
class ActivationEdge:
    run_id: str
    batch_id: str | None
    prompt_id: Any
    token_id: int | None
    from_layer: int
    to_layer: int
    from_node_id: str
    to_node_id: str
    weight: float
    method: str
    confidence: float
    prompt_text: str = ""


@dataclass
class ActivationPath:
    run_id: str
    batch_id: str | None
    prompt_id: Any
    token_id: int | None
    nodes: list[ActivationNode]
    edges: list[ActivationEdge]
    path_confidence: float
    prompt_text: str = ""
    branches: list[ActivationNode] = field(default_factory=list)


@dataclass
class ActivationMap:
    mode: str
    run_id: str
    model: str
    layer_count: int
    prompt_count: int
    batch_count: int
    selected_prompt_id: Any
    selected_batch_id: str | None
    selected_token_id: int | None
    paths: list[ActivationPath]
    heatmap: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    nodes: list[ActivationNode] = field(default_factory=list)


def _int_or_none(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def _float_or_zero(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _preview(value: Any, limit: int = 72) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _prompt_preview(value: Any, limit: int = 260) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if not text:
        return ""
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _string_id(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _prompt_sort_key(value: Any) -> tuple[int, int | str]:
    text = _string_id(value)
    if text.lstrip("-").isdigit():
        return (0, int(text))
    return (1, text)


def _prompt_style(prompt_id: Any, index: int) -> dict[str, Any]:
    palette_index = index % len(PROMPT_PATH_PALETTE)
    cycle = index // len(PROMPT_PATH_PALETTE)
    palette_item = PROMPT_PATH_PALETTE[palette_index]
    return {
        "promptColor": palette_item["color"],
        "promptRgb": palette_item["rgb"],
        "promptColorName": palette_item["name"],
        "promptColorIndex": palette_index,
        "promptColorCycle": cycle,
        "promptOpacity": PROMPT_REUSE_OPACITIES[min(cycle, len(PROMPT_REUSE_OPACITIES) - 1)],
        "promptDash": PROMPT_REUSE_DASHES[min(cycle, len(PROMPT_REUSE_DASHES) - 1)],
        "promptKey": _string_id(prompt_id),
    }


def _build_prompt_color_plan(mode: str, prompt_ids: list[Any]) -> dict[str, Any]:
    unique_by_key = {
        _string_id(prompt_id): prompt_id
        for prompt_id in prompt_ids
        if prompt_id is not None and _string_id(prompt_id) != ""
    }
    ordered_prompt_ids = [unique_by_key[key] for key in sorted(unique_by_key, key=_prompt_sort_key)]
    if mode == "aggregate_heatmap":
        color_mode = "aggregate"
        color_count = 0
        ordered_prompt_ids = []
    elif mode in {"batch_overlay", "compare_prompts"} and len(ordered_prompt_ids) > 1:
        color_mode = "prompt_palette"
        color_count = len(ordered_prompt_ids)
    else:
        color_mode = "single"
        color_count = len(ordered_prompt_ids)
    styles_by_prompt = {
        _string_id(prompt_id): _prompt_style(prompt_id, index)
        for index, prompt_id in enumerate(ordered_prompt_ids)
    }
    entries = [
        {
            "promptId": prompt_id,
            **styles_by_prompt[_string_id(prompt_id)],
        }
        for prompt_id in ordered_prompt_ids
    ]
    return {
        "mode": color_mode,
        "count": color_count,
        "paletteSize": len(PROMPT_PATH_PALETTE),
        "colorsReused": color_mode == "prompt_palette" and color_count > len(PROMPT_PATH_PALETTE),
        "entries": entries,
        "stylesByPrompt": styles_by_prompt,
    }


def _style_for_prompt(prompt_styles: dict[str, dict[str, Any]], prompt_id: Any) -> dict[str, Any]:
    return dict(prompt_styles.get(_string_id(prompt_id), _prompt_style(prompt_id, 0)))


def _legend_entries(
    entries: list[dict[str, Any]],
    activation_paths: list[dict[str, Any]],
    batches: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    prompt_text: dict[str, str] = {}
    for item in [*batches, *activation_paths]:
        key = _string_id(item.get("promptId"))
        if not key:
            continue
        text = _prompt_preview(item.get("promptText") or item.get("promptPreview"))
        if text and key not in prompt_text:
            prompt_text[key] = text
    legend = []
    for entry in entries[:PROMPT_LEGEND_LIMIT]:
        key = entry["promptKey"]
        label = prompt_text.get(key) or f"prompt {entry['promptId']}"
        legend.append({
            **entry,
            "label": _preview(label, 38),
        })
    return legend, max(0, len(entries) - PROMPT_LEGEND_LIMIT)


def build_model_meta(summary: dict, local_model_context: dict | None, backend: str) -> dict:
    local_model_context = local_model_context or {}
    is_local = bool(local_model_context)
    metadata = local_model_context.get("metadata") or {}
    layer_count = (
        _int_or_none(local_model_context.get("block_count"))
        if is_local
        else _int_or_none(summary.get("layers"))
    )
    hidden_size = (
        _int_or_none(local_model_context.get("embedding_length"))
        if is_local
        else _int_or_none(summary.get("d_model"))
    )
    heads = (
        _int_or_none(local_model_context.get("head_count"))
        if is_local
        else _int_or_none(summary.get("heads"))
    )
    context_length = (
        _int_or_none(local_model_context.get("context_length"))
        if is_local
        else _int_or_none(summary.get("context_length") or summary.get("n_ctx"))
    )
    meta = {
        "source": local_model_context.get("source") if is_local else "Local GGUF",
        "backend": backend,
        "modelName": local_model_context.get("display_name") if is_local else str(summary.get("model_name", "")),
        "modelPath": local_model_context.get("model_path") if is_local else str(summary.get("model_path", "")),
        "architecture": local_model_context.get("architecture") if is_local else str(summary.get("architecture", "local")),
        "layerCount": max(layer_count or 0, 0),
        "hiddenSize": max(hidden_size or 0, 0),
        "attentionHeads": heads,
        "kvHeads": _int_or_none(local_model_context.get("head_count_kv")) if is_local else _int_or_none(summary.get("kv_heads")),
        "mlpIntermediateSize": _int_or_none(local_model_context.get("d_mlp")) if is_local else _int_or_none(summary.get("d_mlp")),
        "vocabSize": _int_or_none(metadata.get("tokenizer.ggml.tokens")) or _int_or_none(summary.get("vocab_size")),
        "contextLength": context_length,
        "quantization": str(metadata.get("general.file_type") or metadata.get("quantization") or ""),
        "parameterCount": _int_or_none(summary.get("parameters")),
        "tensorCount": _int_or_none(local_model_context.get("tensor_count")) if is_local else None,
        "metadataSource": "gguf" if is_local else "local",
        "metadataErrors": list(local_model_context.get("errors") or []),
        "visualizationMode": "unavailable",
    }
    meta["modelFingerprint"] = model_fingerprint(meta)
    return meta


def _batch_rows(artifact: dict) -> list[dict[str, Any]]:
    rows = []
    for index, prompt in enumerate(artifact.get("prompts", [])):
        prompt_id = prompt.get("prompt_id", index)
        rows.append({
            "batchId": str(prompt.get("batch_id") or f"batch-{prompt_id}"),
            "promptId": prompt_id,
            "label": prompt.get("label") or "unlabeled",
            "promptText": str(prompt.get("prompt") or ""),
            "promptPreview": _preview(prompt.get("prompt")),
            "tokenRange": "",
            "outputToken": _preview(prompt.get("output"), 32),
            "traceAvailable": any(
                isinstance(row, dict)
                and row.get("trace_available", True) is not False
                and _int_or_none(row.get("layer")) is not None
                for row in prompt.get("trace_rows", [])
            ),
        })
    return rows


def _artifact_trace_rows(artifact: dict) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_id = str(artifact.get("run_id") or "")
    for prompt_index, prompt in enumerate(artifact.get("prompts", [])):
        prompt_id = prompt.get("prompt_id", prompt_index)
        for row_index, row in enumerate(prompt.get("trace_rows", [])):
            if not isinstance(row, dict):
                continue
            merged = dict(row)
            if not merged.get("run_id"):
                merged["run_id"] = run_id
            merged.setdefault("prompt_id", prompt_id)
            merged.setdefault("prompt_index", row.get("prompt_index", prompt_index))
            if not merged.get("batch_id"):
                merged["batch_id"] = prompt.get("batch_id") or row.get("batch_id") or (f"batch-{prompt_id}" if prompt_id is not None else None)
            merged.setdefault("label", prompt.get("label") or "unlabeled")
            merged.setdefault("prompt_text", prompt.get("prompt") or "")
            merged.setdefault("_source_prompt_index", prompt_index)
            merged.setdefault("_source_row_index", row_index)
            rows.append(merged)
    return rows


def _available_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    available = []
    for row in rows:
        if row.get("trace_available", True) is False:
            continue
        if _int_or_none(row.get("layer")) is None:
            continue
        has_activation = row.get("activation_norm") is not None or row.get("norm") is not None or row.get("l2_norm") is not None
        if not has_activation and not row.get("top_dims") and not row.get("vector") and not row.get("nodes"):
            continue
        available.append(row)
    return available


def _fields_present(rows: list[dict[str, Any]]) -> list[str]:
    fields: set[str] = set()
    for row in rows:
        fields.update(key for key in row if not key.startswith("_"))
    return sorted(fields)


def _top_dims(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("top_dims") or []
    return [item for item in value if isinstance(item, dict)]


def _vector(row: dict[str, Any]) -> list[float]:
    value = row.get("vector") or row.get("activation_vector")
    if not isinstance(value, list):
        return []
    return [_float_or_zero(item) for item in value if isinstance(item, (int, float))]


def _hidden_size(rows: list[dict[str, Any]], model_meta: dict[str, Any]) -> int:
    hidden_size = _int_or_none(model_meta.get("hiddenSize")) or 0
    for row in rows:
        n_embd = _int_or_none(row.get("n_embd"))
        if n_embd:
            hidden_size = max(hidden_size, n_embd)
        vector = _vector(row)
        if vector:
            hidden_size = max(hidden_size, len(vector))
        for item in _top_dims(row):
            dim = _int_or_none(item.get("dimension"))
            if dim is not None:
                hidden_size = max(hidden_size, dim + 1)
    return hidden_size


def _layer_count(rows: list[dict[str, Any]], model_meta: dict[str, Any]) -> int:
    count = _int_or_none(model_meta.get("layerCount")) or 0
    layer_values = [_int_or_none(row.get("layer")) for row in rows]
    layer_values = [layer for layer in layer_values if layer is not None]
    if layer_values:
        count = max(count, max(layer_values) + 1)
    return count


def _unavailable_reason(rows: list[dict[str, Any]]) -> str:
    reasons: list[str] = []
    for row in rows:
        reason = str(row.get("unavailable_reason") or "").strip()
        if reason and reason not in reasons:
            reasons.append(reason)
    return " | ".join(reasons)


def _is_aggregated(row: dict[str, Any]) -> bool:
    if row.get("aggregation") or row.get("aggregated") is True:
        return True
    return row.get("prompt_id") is None


def _data_mode(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "unavailable"
    if all(_is_aggregated(row) for row in rows):
        return "aggregated"
    if any(_vector(row) for row in rows):
        return "real_vectors"
    if any(_top_dims(row) or row.get("nodes") for row in rows):
        return "top_dims_approx"
    return "scalar_layer_summary"


def _normalise_nodes(nodes: list[ActivationNode]) -> None:
    peak = max((node.activation for node in nodes), default=0.0)
    for node in nodes:
        node.normalized_activation = node.activation / peak if peak > 0 else 0.0


def _row_activation(row: dict[str, Any]) -> float:
    return _float_or_zero(row.get("activation_norm", row.get("norm", row.get("l2_norm"))))


def _node_y(cluster_id: int | str | None, hidden_size: int) -> float:
    if isinstance(cluster_id, int) and hidden_size > 1:
        return max(0.04, min(0.96, cluster_id / (hidden_size - 1)))
    return 0.5


def _node_from_dimension(
    row: dict[str, Any],
    *,
    dimension: int,
    activation: float,
    source_fields: list[str],
    confidence: float,
    mode: str,
    hidden_size: int,
    vector: list[float] | None = None,
) -> ActivationNode:
    layer = int(row["layer"])
    prompt_id = row.get("prompt_id")
    token_id = _int_or_none(row.get("token_index", row.get("token_id")))
    node_id = f"L{layer}-N{dimension}"
    return ActivationNode(
        run_id=str(row.get("run_id") or ""),
        batch_id=row.get("batch_id"),
        prompt_id=prompt_id,
        token_id=token_id,
        layer=layer,
        node_id=node_id,
        cluster_id=dimension,
        activation=abs(float(activation)),
        normalized_activation=0.0,
        node_range=[dimension, dimension],
        token_text=str(row.get("token") or ""),
        source_fields=source_fields,
        confidence=confidence,
        mode=mode,
        y=_node_y(dimension, hidden_size),
        prompt_label=str(row.get("label") or ""),
        prompt_text=str(row.get("prompt_text") or ""),
        prompt_preview_list=[_prompt_preview(row.get("prompt_text"))] if _prompt_preview(row.get("prompt_text")) else [],
        source_row_index=_int_or_none(row.get("_source_row_index")),
        vector=vector,
        real=mode == "real_vectors",
    )


def _row_nodes(row: dict[str, Any], data_mode: str, hidden_size: int, top_k: int) -> list[ActivationNode]:
    layer = _int_or_none(row.get("layer"))
    if layer is None:
        return []
    vector = _vector(row)
    if data_mode == "real_vectors" and vector:
        if vector:
            ranked = sorted(range(len(vector)), key=lambda i: abs(vector[i]), reverse=True)[: max(1, top_k)]
            return [
                _node_from_dimension(
                    row,
                    dimension=dimension,
                    activation=vector[dimension],
                    source_fields=["vector", "activation_norm"],
                    confidence=0.90,
                    mode="real_vectors",
                    hidden_size=max(hidden_size, len(vector)),
                    vector=vector,
                )
                for dimension in ranked
            ]
    dims = _top_dims(row)
    if dims:
        ranked_dims = sorted(
            dims,
            key=lambda item: abs(_float_or_zero(item.get("abs_activation", item.get("activation")))),
            reverse=True,
        )[: max(1, top_k)]
        nodes = []
        for item in ranked_dims:
            dimension = _int_or_none(item.get("dimension"))
            if dimension is None:
                continue
            nodes.append(_node_from_dimension(
                row,
                dimension=dimension,
                activation=_float_or_zero(item.get("abs_activation", item.get("activation"))),
                source_fields=["top_dims", "activation_norm"],
                confidence=0.48,
                mode="top_dims_approx",
                hidden_size=hidden_size,
            ))
        return nodes
    return []


def _aggregate_nodes(rows: list[dict[str, Any]], data_mode: str, hidden_size: int, top_k: int) -> list[ActivationNode]:
    buckets: dict[tuple[int, str], ActivationNode] = {}
    members: dict[tuple[int, str], set[str]] = {}
    prompts: dict[tuple[int, str], dict[str, str]] = {}
    observations: dict[tuple[int, str], list[ActivationNode]] = {}
    for row in rows:
        for node in _row_nodes(row, data_mode, hidden_size, top_k):
            key = (node.layer, node.node_id)
            if key not in buckets or node.activation > buckets[key].activation:
                buckets[key] = node
            members.setdefault(key, set()).add(f"{node.prompt_id}:{node.token_id}")
            if node.prompt_text:
                prompts.setdefault(key, {})[f"{node.batch_id}:{node.prompt_id}"] = node.prompt_text
            observations.setdefault(key, []).append(node)
    nodes = list(buckets.values())
    _normalise_nodes(nodes)
    observation_peak = max((node.activation for bucket in observations.values() for node in bucket), default=0.0)
    for node in nodes:
        key = (node.layer, node.node_id)
        observation_rows = sorted(observations.get(key, []), key=lambda item: item.activation, reverse=True)
        node.source_fields = sorted(set(node.source_fields + ["member_count"]))
        node.prompt_label = f"{len(members.get(key, set()))} observations"
        node.prompt_preview_list = [_prompt_preview(text) for text in prompts.get(key, {}).values() if _prompt_preview(text)]
        if len(node.prompt_preview_list) == 1:
            node.prompt_text = node.prompt_preview_list[0]
        node.prompt_observations = [
            {
                "promptId": item.prompt_id,
                "batchId": item.batch_id,
                "tokenIndex": item.token_id,
                "tokenId": item.token_id,
                "token": item.token_text,
                "activationValue": item.activation,
                "activation": item.activation,
                "normalizedActivation": item.activation / observation_peak if observation_peak > 0 else 0.0,
                "promptText": item.prompt_text,
                "promptPreview": _prompt_preview(item.prompt_text),
                "layerId": f"L{item.layer}",
                "nodeId": item.node_id,
                "clusterId": item.cluster_id,
            }
            for item in observation_rows
        ]
    return nodes


def _cosine(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right:
        return 0.0
    width = min(len(left), len(right))
    if width <= 0:
        return 0.0
    dot = sum(left[i] * right[i] for i in range(width))
    ln = sqrt(sum(left[i] * left[i] for i in range(width)))
    rn = sqrt(sum(right[i] * right[i] for i in range(width)))
    if ln <= 0 or rn <= 0:
        return 0.0
    return max(0.0, dot / (ln * rn))


def _edge_between(left: ActivationNode, right: ActivationNode, data_mode: str) -> ActivationEdge:
    if data_mode == "real_vectors" and left.vector and right.vector:
        method = "cosine_similarity"
        weight = _cosine(left.vector, right.vector)
        confidence = 0.88
    else:
        method = "nearest_projected_position"
        distance = abs(float(left.y) - float(right.y))
        weight = max(0.0, 1.0 - distance)
        confidence = 0.42
    return ActivationEdge(
        run_id=left.run_id,
        batch_id=left.batch_id,
        prompt_id=left.prompt_id,
        token_id=left.token_id,
        from_layer=left.layer,
        to_layer=right.layer,
        from_node_id=left.node_id,
        to_node_id=right.node_id,
        weight=weight,
        method=method,
        confidence=confidence,
        prompt_text=left.prompt_text,
    )


def _rows_by_identity(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if _is_aggregated(row):
            continue
        token_id = _int_or_none(row.get("token_index", row.get("token_id")))
        key = (_string_id(row.get("batch_id")), _string_id(row.get("prompt_id")), _string_id(token_id))
        grouped.setdefault(key, []).append(row)
    return grouped


def _build_paths(rows: list[dict[str, Any]], data_mode: str, hidden_size: int, top_k: int) -> list[ActivationPath]:
    if data_mode not in {"real_vectors", "top_dims_approx"}:
        return []
    paths: list[ActivationPath] = []
    for (_batch_id, _prompt_id, _token_id), identity_rows in sorted(_rows_by_identity(rows).items(), key=lambda item: item[0]):
        rows_by_layer: dict[int, list[dict[str, Any]]] = {}
        for row in identity_rows:
            layer = _int_or_none(row.get("layer"))
            if layer is not None:
                rows_by_layer.setdefault(layer, []).append(row)
        nodes: list[ActivationNode] = []
        branches: list[ActivationNode] = []
        for layer in sorted(rows_by_layer):
            candidates: list[ActivationNode] = []
            for row in rows_by_layer[layer]:
                candidates.extend(_row_nodes(row, data_mode, hidden_size, top_k))
            if not candidates:
                continue
            candidates.sort(key=lambda node: node.activation, reverse=True)
            nodes.append(candidates[0])
            branches.extend(candidates[1:])
        if not nodes:
            continue
        _normalise_nodes(nodes)
        _normalise_nodes(branches)
        edges = [_edge_between(left, right, data_mode) for left, right in zip(nodes, nodes[1:]) if right.layer == left.layer + 1]
        path_confidence = min((node.confidence for node in nodes), default=0.0)
        paths.append(ActivationPath(
            run_id=nodes[0].run_id,
            batch_id=nodes[0].batch_id,
            prompt_id=nodes[0].prompt_id,
            token_id=nodes[0].token_id,
            nodes=nodes,
            edges=edges,
            path_confidence=path_confidence,
            prompt_text=nodes[0].prompt_text,
            branches=branches,
        ))
    return paths


def _build_heatmap(rows: list[dict[str, Any]], hidden_size: int) -> list[dict[str, Any]]:
    heat: dict[tuple[int, int | str], dict[str, Any]] = {}
    for row in rows:
        layer = _int_or_none(row.get("layer"))
        if layer is None:
            continue
        dims = _top_dims(row)
        vector = _vector(row)
        if dims:
            items: list[tuple[int | str, float]] = [
                (_int_or_none(item.get("dimension")) or 0, abs(_float_or_zero(item.get("abs_activation", item.get("activation")))))
                for item in dims
            ]
        elif vector:
            ranked = sorted(range(len(vector)), key=lambda i: abs(vector[i]), reverse=True)[:DEFAULT_TOP_K]
            items = [(idx, abs(vector[idx])) for idx in ranked]
        else:
            items = [("scalar", _row_activation(row))]
        for cluster_id, activation in items:
            key = (layer, cluster_id)
            cell = heat.setdefault(key, {
                "layer": layer,
                "clusterId": cluster_id,
                "nodeId": f"L{layer}-N{cluster_id}" if isinstance(cluster_id, int) else f"L{layer}-S0",
                "activationSum": 0.0,
                "activationMax": 0.0,
                "count": 0,
                "promptIds": set(),
                "promptTexts": {},
                "y": _node_y(cluster_id, hidden_size),
            })
            cell["activationSum"] += activation
            cell["activationMax"] = max(cell["activationMax"], activation)
            cell["count"] += 1
            if row.get("prompt_id") is not None:
                cell["promptIds"].add(row.get("prompt_id"))
            prompt_text = _prompt_preview(row.get("prompt_text"))
            if prompt_text:
                cell["promptTexts"][f"{row.get('batch_id')}:{row.get('prompt_id')}"] = prompt_text
    peak = max((cell["activationMax"] for cell in heat.values()), default=0.0)
    result = []
    for cell in sorted(heat.values(), key=lambda item: (item["layer"], str(item["clusterId"]))):
        prompt_previews = list(cell["promptTexts"].values())
        result.append({
            **{key: value for key, value in cell.items() if key not in {"promptIds", "promptTexts"}},
            "activationMean": cell["activationSum"] / cell["count"] if cell["count"] else 0.0,
            "normalizedActivation": cell["activationMax"] / peak if peak > 0 else 0.0,
            "promptCount": len(cell["promptIds"]),
            "promptIds": sorted(cell["promptIds"], key=lambda value: str(value)),
            "promptPreviewList": prompt_previews,
            "promptPreviewTop": prompt_previews[:5],
            "promptPreviewMoreCount": max(0, len(prompt_previews) - 5),
        })
    return result


def _prompt_count(artifact: dict, rows: list[dict[str, Any]]) -> int:
    summary_count = _int_or_none((artifact.get("summary") or {}).get("prompt_count"))
    prompt_ids = {row.get("prompt_id") for row in rows if row.get("prompt_id") is not None}
    if prompt_ids:
        return len(prompt_ids)
    return summary_count or len(artifact.get("prompts", []))


def _batch_count(rows: list[dict[str, Any]]) -> int:
    return len({row.get("batch_id") for row in rows if row.get("batch_id") is not None})


def _token_count(rows: list[dict[str, Any]]) -> int:
    return len({
        (row.get("prompt_id"), _int_or_none(row.get("token_index", row.get("token_id"))))
        for row in rows
        if row.get("prompt_id") is not None and _int_or_none(row.get("token_index", row.get("token_id"))) is not None
    })


def _rows_per_layer(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        layer = _int_or_none(row.get("layer"))
        if layer is not None:
            key = str(layer)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _default_visualization_mode(data_mode: str, prompt_count: int, requested: str | None) -> str:
    if requested in VISUALIZATION_MODES:
        return requested
    if data_mode in {"aggregated", "scalar_layer_summary", "unavailable"}:
        return "aggregate_heatmap"
    if prompt_count <= 1:
        return "single_prompt"
    return "batch_overlay"


def _filter_rows_for_renderer(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    selected_prompt: Any,
    selected_token: int | None,
) -> list[dict[str, Any]]:
    if mode in {"aggregate_heatmap", "compare_prompts"}:
        return rows
    filtered = rows
    if mode == "single_prompt" and selected_prompt is not None:
        filtered = [row for row in filtered if _string_id(row.get("prompt_id")) == _string_id(selected_prompt)]
    if selected_token is not None and mode in {"single_prompt", "batch_overlay"}:
        filtered = [
            row
            for row in filtered
            if _int_or_none(row.get("token_index", row.get("token_id"))) == selected_token
        ]
    return filtered


def inspect_trace_artifact(artifact: dict, summary: dict | None = None, local_model_context: dict | None = None) -> dict[str, Any]:
    summary = summary or artifact.get("summary") or {}
    backend = str(artifact.get("backend") or summary.get("backend") or "llama.cpp")
    model_meta = build_model_meta(summary, local_model_context, backend)
    all_rows = _artifact_trace_rows(artifact)
    available = _available_rows(all_rows)
    data_mode = _data_mode(available)
    prompt_count = _prompt_count(artifact, available)
    batch_count = _batch_count(available)
    token_count = _token_count(available)
    layer_count = _layer_count(available, model_meta)
    renderer_mode = _default_visualization_mode(data_mode, prompt_count, None)
    return {
        "run_id": artifact.get("run_id"),
        "model": artifact.get("model") or model_meta.get("modelName"),
        "batch_count": batch_count,
        "prompt_count": prompt_count,
        "token_count": token_count,
        "layer_count": layer_count,
        "rows": len(available),
        "rows_per_layer": _rows_per_layer(available),
        "fields_present": _fields_present(all_rows),
        "vectors_present": any(_vector(row) for row in available),
        "include_vectors_response": any(row.get("include_vectors_response") is True for row in available) or any(_vector(row) for row in available),
        "vector_availability": "available" if any(_vector(row) for row in available) else "unavailable",
        "node_availability": "available" if any(_top_dims(row) or _vector(row) or row.get("nodes") for row in available) else "unavailable",
        "token_ranges_present": any(row.get("token_range") or row.get("tokenRange") for row in available),
        "node_ranges_present": any(row.get("node_range") or row.get("nodeRange") for row in available),
        "top_dims_present": any(_top_dims(row) for row in available),
        "activation_magnitudes_present": any(row.get("activation_norm") is not None or row.get("norm") is not None or row.get("l2_norm") is not None for row in available),
        "path_edge_data_exists": any(row.get("edges") or row.get("paths") for row in available),
        "data_granularity": "aggregated" if data_mode == "aggregated" else "raw" if data_mode in {"real_vectors", "top_dims_approx", "scalar_layer_summary"} else "unavailable",
        "data_mode": data_mode,
        "renderer_mode": renderer_mode,
        "mode": renderer_mode,
    }


def build_activation_map(
    artifact: dict,
    summary: dict,
    local_model_context: dict | None = None,
    *,
    visualization_mode: str | None = None,
    selected_prompt: Any = None,
    selected_token: int | None = None,
    compare_prompt: Any = None,
    top_k: int = DEFAULT_TOP_K,
) -> ActivationMap:
    backend = str(artifact.get("backend") or summary.get("backend") or "llama.cpp")
    model_meta = build_model_meta(summary, local_model_context, backend)
    rows = _available_rows(_artifact_trace_rows(artifact))
    data_mode = _data_mode(rows)
    hidden_size = _hidden_size(rows, model_meta)
    if hidden_size > model_meta.get("hiddenSize", 0):
        model_meta["hiddenSize"] = hidden_size
    layer_count = _layer_count(rows, model_meta)
    prompt_count = _prompt_count(artifact, rows)
    batch_count = _batch_count(rows)
    mode = _default_visualization_mode(data_mode, prompt_count, visualization_mode)

    warnings: list[str] = []
    if data_mode == "aggregated":
        warnings.append("Backend trace is already aggregated. Per-prompt paths unavailable.")
        mode = "aggregate_heatmap" if mode != "compare_prompts" else mode
    elif data_mode == "scalar_layer_summary":
        warnings.append("Trace contains scalar layer rows without nodes or vectors. Per-prompt paths unavailable.")
        mode = "aggregate_heatmap" if mode in {"single_prompt", "batch_overlay"} else mode
    elif data_mode == "top_dims_approx":
        warnings.append("Real activation vectors were not returned by the backend. Showing top-k dimension approximation only.")
    elif data_mode == "unavailable":
        warnings.append(_unavailable_reason(rows) or "activation trace data is unavailable")
        mode = "aggregate_heatmap"

    paths = _build_paths(rows, data_mode, hidden_size, top_k)
    if selected_prompt is not None and mode == "single_prompt":
        paths = [path for path in paths if _string_id(path.prompt_id) == _string_id(selected_prompt)]
    if selected_token is not None and mode in {"single_prompt", "batch_overlay"}:
        paths = [path for path in paths if path.token_id == selected_token]
    elif mode == "batch_overlay" and paths:
        token_ids = sorted({path.token_id for path in paths if path.token_id is not None})
        if token_ids:
            selected_token = token_ids[0]
            paths = [path for path in paths if path.token_id == selected_token]
    if mode == "single_prompt" and paths:
        if selected_prompt is None:
            first_prompt = paths[0].prompt_id
            paths = [path for path in paths if _string_id(path.prompt_id) == _string_id(first_prompt)]
        if selected_token is None:
            first_token = paths[0].token_id
            paths = [path for path in paths if path.token_id == first_token]
    if mode in {"aggregate_heatmap", "compare_prompts"}:
        paths = []

    selected_prompt_id = selected_prompt if selected_prompt is not None else (paths[0].prompt_id if paths else None)
    selected_token_id = selected_token if selected_token is not None else (paths[0].token_id if paths else None)
    selected_batch_id = paths[0].batch_id if paths else None
    renderer_rows = _filter_rows_for_renderer(
        rows,
        mode=mode,
        selected_prompt=selected_prompt_id,
        selected_token=selected_token_id,
    )
    aggregate_nodes = _aggregate_nodes(renderer_rows, data_mode, hidden_size, top_k) if data_mode in {"real_vectors", "top_dims_approx"} else []
    heatmap_rows = rows if mode == "aggregate_heatmap" else renderer_rows
    heatmap = _build_heatmap(heatmap_rows, hidden_size)

    diagnostics = inspect_trace_artifact(artifact, summary, local_model_context)
    diagnostics.update({
        "warnings": warnings,
        "dataMode": data_mode,
        "renderer_mode": mode,
        "rendererMode": mode,
        "visualizationMode": mode,
        "unavailableReason": _unavailable_reason(rows),
        "edgeCount": sum(len(path.edges) for path in paths),
        "nodeCount": len(aggregate_nodes),
        "pathCount": len(paths),
        "topK": top_k,
        "selectedPromptId": selected_prompt_id,
        "selectedTokenId": selected_token_id,
        "comparePromptId": compare_prompt,
        "modelMeta": model_meta,
    })

    return ActivationMap(
        mode=mode,
        run_id=str(artifact.get("run_id") or ""),
        model=str(artifact.get("model") or model_meta.get("modelName") or ""),
        layer_count=layer_count,
        prompt_count=prompt_count,
        batch_count=batch_count,
        selected_prompt_id=selected_prompt_id,
        selected_batch_id=selected_batch_id,
        selected_token_id=selected_token_id,
        paths=paths,
        heatmap=heatmap,
        diagnostics=diagnostics,
        nodes=aggregate_nodes,
    )


def _layer_rows(layer_count: int, nodes: list[ActivationNode], data_mode: str) -> list[dict[str, Any]]:
    by_layer: dict[int, list[ActivationNode]] = {}
    for node in nodes:
        by_layer.setdefault(node.layer, []).append(node)
    rows = []
    for layer_index in range(max(layer_count, 0)):
        layer_nodes = by_layer.get(layer_index, [])
        active = [node for node in layer_nodes if node.activation > 0]
        rows.append({
            "layerId": f"L{layer_index}",
            "index": layer_index,
            "name": f"L{layer_index}",
            "layerType": "llm_block",
            "nodeCount": len(layer_nodes),
            "groupCount": len(layer_nodes),
            "activationDensity": (len(active) / len(layer_nodes)) if layer_nodes else 0.0,
            "topActiveGroups": [node.node_id for node in sorted(active, key=lambda node: node.activation, reverse=True)[:3]],
            "visualizationMode": data_mode if layer_nodes else "unavailable",
            "selected": layer_index == 0,
        })
    return rows


def _node_group_rows(nodes: list[ActivationNode], prompt_styles: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    prompt_styles = prompt_styles or {}
    rows = []
    for node in nodes:
        observations = node.prompt_observations or [{
            "promptId": node.prompt_id,
            "batchId": node.batch_id,
            "tokenIndex": node.token_id,
            "tokenId": node.token_id,
            "token": node.token_text,
            "activationValue": node.activation,
            "activation": node.activation,
            "normalizedActivation": node.normalized_activation,
            "promptText": node.prompt_text,
            "promptPreview": _prompt_preview(node.prompt_text),
            "layerId": f"L{node.layer}",
            "nodeId": node.node_id,
            "clusterId": node.cluster_id,
        }]
        styled_observations = [
            {
                **item,
                **_style_for_prompt(prompt_styles, item.get("promptId")),
            }
            for item in observations
        ]
        prompt_ids = sorted(
            {item.get("promptId") for item in styled_observations if item.get("promptId") is not None},
            key=_prompt_sort_key,
        )
        batch_ids = sorted({str(item.get("batchId")) for item in styled_observations if item.get("batchId")})
        token_ids = sorted(
            {item.get("tokenIndex") for item in styled_observations if item.get("tokenIndex") is not None},
            key=lambda value: int(value) if str(value).lstrip("-").isdigit() else str(value),
        )
        prompt_previews = []
        for item in styled_observations:
            preview = _prompt_preview(item.get("promptText") or item.get("promptPreview"))
            if preview and preview not in prompt_previews:
                prompt_previews.append(preview)
        prompt_overlap_count = len(prompt_ids)
        color_strategy = "multi_prompt_overlap" if prompt_overlap_count > 1 else "prompt_color"
        rows.append({
            "groupId": node.node_id,
            "nodeId": node.node_id,
            "layerId": f"L{node.layer}",
            "layer": node.layer,
            "name": str(node.cluster_id),
            "label": str(node.cluster_id),
            "clusterId": node.cluster_id,
            "tokenRange": [node.token_id, node.token_id] if node.token_id is not None else None,
            "nodeRange": node.node_range,
            "sourceIndices": list(range(node.node_range[0], node.node_range[1] + 1)) if node.node_range else [],
            "groupingMethod": "vector_dominant_dimension" if node.mode == "real_vectors" else "top_dim_bucket",
            "activationValue": node.activation,
            "activation": node.activation,
            "normalizedActivation": node.normalized_activation,
            "attributionScore": node.normalized_activation,
            "batchParticipation": len(batch_ids) or 1,
            "observationCount": len(styled_observations),
            "promptCount": prompt_overlap_count,
            "promptOverlapCount": prompt_overlap_count,
            "promptIds": prompt_ids,
            "batchIds": batch_ids,
            "tokenIds": token_ids,
            "promptActivations": styled_observations,
            "promptColorStrategy": color_strategy,
            "box2Radius": 2.8 + sqrt(max(0.0, min(1.0, node.normalized_activation))) * 7.4,
            "box2Opacity": 0.16 + max(0.0, min(1.0, node.normalized_activation)) * 0.72,
            "confidence": node.confidence,
            "sourceFields": node.source_fields,
            "approximationReason": "" if node.real else "top-k dimension node approximates a cluster because full vectors were unavailable",
            "visualizationMode": node.mode,
            "yPosition": node.y,
            "promptId": node.prompt_id,
            "batchId": node.batch_id,
            "tokenIndex": node.token_id,
            "token": node.token_text,
            "promptText": node.prompt_text,
            "promptPreview": _prompt_preview(node.prompt_text),
            "promptPreviewList": prompt_previews,
            "promptPreviewTop": prompt_previews[:5],
            "promptPreviewMoreCount": max(0, len(prompt_previews) - 5),
            "real": node.real,
            **_style_for_prompt(prompt_styles, node.prompt_id),
        })
    return rows


def _box2_diagnostics(
    selected_layer_row: dict[str, Any] | None,
    node_groups: list[dict[str, Any]],
    *,
    mode: str,
    prompt_color_mode: str,
) -> dict[str, Any]:
    selected_layer_id = (selected_layer_row or {}).get("layerId")
    visible_groups = [group for group in node_groups if group.get("layerId") == selected_layer_id] if selected_layer_id else []
    has_normalized = any(
        group.get("normalizedActivation") is not None and pd.notna(group.get("normalizedActivation"))
        for group in visible_groups
    )
    has_raw = any(
        group.get("activationValue") is not None and pd.notna(group.get("activationValue"))
        for group in visible_groups
    )
    if has_normalized:
        scaling = "normalized"
    elif has_raw:
        scaling = "layer_normalized"
    else:
        scaling = "unavailable"
    return {
        "selectedLayer": selected_layer_id,
        "visibleNodeCount": len(visible_groups),
        "activeNodeCount": sum(1 for group in visible_groups if float(group.get("activationValue") or 0.0) > 0.0),
        "activationScaling": scaling,
        "promptColorMode": prompt_color_mode,
        "overlappingPromptNodes": sum(1 for group in visible_groups if int(group.get("promptOverlapCount") or 0) > 1),
        "aggregateMode": mode == "aggregate_heatmap",
    }


def _empty_annotation_fields() -> dict[str, Any]:
    return {
        "annotationIds": [],
        "annotationMatchType": "none",
        "annotationMatchConfidence": "none",
        "annotationTags": [],
        "annotationNote": "",
    }


def _apply_annotation_matches(
    model_meta: dict[str, Any],
    node_groups: list[dict[str, Any]],
    heatmap: list[dict[str, Any]],
    annotations: dict[str, Any],
) -> None:
    for group in node_groups:
        match = get_annotations_for_node(
            model_meta,
            int(group.get("layer") or 0),
            group.get("clusterId"),
            group.get("nodeId") or group.get("groupId"),
            group.get("nodeRange"),
            annotations=annotations,
        )
        group.update(compact_annotation_match(match) if match.get("annotations") else _empty_annotation_fields())
    for cell in heatmap:
        match = get_annotations_for_node(
            model_meta,
            int(cell.get("layer") or 0),
            cell.get("clusterId"),
            cell.get("nodeId"),
            cell.get("nodeRange"),
            annotations=annotations,
        )
        cell.update(compact_annotation_match(match) if match.get("annotations") else _empty_annotation_fields())


def _edge_rows(paths: list[ActivationPath], prompt_styles: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    prompt_styles = prompt_styles or {}
    rows: list[dict[str, Any]] = []
    for path in paths:
        for edge in path.edges:
            rows.append({
                "edgeId": f"{path.prompt_id}:{path.token_id}:{edge.from_node_id}->{edge.to_node_id}",
                "fromLayer": edge.from_layer,
                "fromLayerId": f"L{edge.from_layer}",
                "fromNodeId": edge.from_node_id,
                "fromGroupId": edge.from_node_id,
                "toLayer": edge.to_layer,
                "toLayerId": f"L{edge.to_layer}",
                "toNodeId": edge.to_node_id,
                "toGroupId": edge.to_node_id,
                "promptId": edge.prompt_id,
                "batchId": edge.batch_id,
                "tokenId": edge.token_id,
                "tokenIndex": edge.token_id,
                "promptText": edge.prompt_text,
                "promptPreview": _prompt_preview(edge.prompt_text),
                "promptPreviewList": [_prompt_preview(edge.prompt_text)] if _prompt_preview(edge.prompt_text) else [],
                "weight": edge.weight,
                "confidence": edge.confidence,
                "method": edge.method,
                "approximationReason": "" if edge.method == "cosine_similarity" else "edge is approximated from nearest projected node position; prompt/token identity is preserved",
                "visualizationMode": edge.method,
                **_style_for_prompt(prompt_styles, edge.prompt_id),
            })
    return rows


def _path_rows(paths: list[ActivationPath], layer_count: int, prompt_styles: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    prompt_styles = prompt_styles or {}
    path_rows = []
    max_layer = max(layer_count - 1, 1)
    for path in paths:
        path_style = _style_for_prompt(prompt_styles, path.prompt_id)
        points = [
            {
                "layerId": f"L{node.layer}",
                "layerIndex": node.layer,
                "groupId": node.node_id,
                "nodeId": node.node_id,
                "groupIndex": node.cluster_id if isinstance(node.cluster_id, int) else 0,
                "x": node.layer / max_layer,
                "y": node.y,
                "activationValue": node.activation,
                "normalizedActivation": node.normalized_activation,
                "visualizationMode": node.mode,
                "promptId": node.prompt_id,
                "batchId": node.batch_id,
                "promptText": node.prompt_text,
                "promptPreview": _prompt_preview(node.prompt_text),
                "token": node.token_text,
                "tokenText": node.token_text,
                "tokenIndex": node.token_id,
                "tokenId": node.token_id,
                "clusterId": node.cluster_id,
                "nodeRange": node.node_range,
                "sourceFields": node.source_fields,
                "confidence": node.confidence,
                "real": node.real,
                **_style_for_prompt(prompt_styles, node.prompt_id),
            }
            for node in path.nodes
        ]
        branches = [
            {
                "layerId": f"L{node.layer}",
                "layerIndex": node.layer,
                "groupId": node.node_id,
                "nodeId": node.node_id,
                "x": node.layer / max_layer,
                "y": node.y,
                "activationValue": node.activation,
                "normalizedActivation": node.normalized_activation,
                "promptId": node.prompt_id,
                "batchId": node.batch_id,
                "promptText": node.prompt_text,
                "promptPreview": _prompt_preview(node.prompt_text),
                "tokenIndex": node.token_id,
                "token": node.token_text,
                "confidence": node.confidence,
                **_style_for_prompt(prompt_styles, node.prompt_id),
            }
            for node in path.branches
        ]
        path_rows.append({
            "pathId": f"{path.batch_id}-{path.prompt_id}-{path.token_id}-path",
            "batchId": path.batch_id,
            "promptId": path.prompt_id,
            "promptText": path.prompt_text,
            "promptPreview": _prompt_preview(path.prompt_text),
            "promptPreviewList": [_prompt_preview(path.prompt_text)] if _prompt_preview(path.prompt_text) else [],
            "tokenId": path.token_id,
            "tokenIndex": path.token_id,
            "points": points,
            "nodes": [asdict(node) for node in path.nodes],
            "edges": [
                {
                    "runId": edge.run_id,
                    "batchId": edge.batch_id,
                    "promptId": edge.prompt_id,
                    "tokenId": edge.token_id,
                    "tokenIndex": edge.token_id,
                    "promptText": edge.prompt_text,
                    "promptPreview": _prompt_preview(edge.prompt_text),
                    "fromLayer": edge.from_layer,
                    "toLayer": edge.to_layer,
                    "fromNodeId": edge.from_node_id,
                    "toNodeId": edge.to_node_id,
                    "weight": edge.weight,
                    "method": edge.method,
                    "confidence": edge.confidence,
                    **_style_for_prompt(prompt_styles, edge.prompt_id),
                }
                for edge in path.edges
            ],
            "branches": branches,
            "strength": sum(point["activationValue"] for point in points) / len(points) if points else 0.0,
            "frequency": len(points),
            "tokenRange": f"{path.token_id}..{path.token_id}" if path.token_id is not None else "",
            "outputToken": points[0]["token"] if points else "",
            "activationSummary": f"{len(points)} traced layers for prompt {path.prompt_id}, token {path.token_id}",
            "attributionScore": max((point["activationValue"] for point in points), default=0.0),
            "confidence": path.path_confidence,
            "pathMethod": "top_activation_per_layer",
            "visualizationMode": points[0]["visualizationMode"] if points else "unavailable",
            "approximationReason": "" if points and points[0]["real"] else "dominant path is approximated from top-k dimensions; prompt/token identity is preserved",
            **path_style,
        })
    return path_rows


def _compare_prompts(rows: list[dict[str, Any]], selected_prompt: Any, compare_prompt: Any) -> dict[str, Any]:
    if selected_prompt is None or compare_prompt is None:
        prompt_ids = sorted({row.get("prompt_id") for row in rows if row.get("prompt_id") is not None}, key=lambda value: str(value))
        if len(prompt_ids) >= 2:
            selected_prompt = prompt_ids[0]
            compare_prompt = prompt_ids[1]
    left: dict[tuple[int, int | None], float] = {}
    right: dict[tuple[int, int | None], float] = {}
    for row in rows:
        key = (_int_or_none(row.get("layer")) or 0, _int_or_none(row.get("token_index", row.get("token_id"))))
        if _string_id(row.get("prompt_id")) == _string_id(selected_prompt):
            left[key] = max(left.get(key, 0.0), _row_activation(row))
        if _string_id(row.get("prompt_id")) == _string_id(compare_prompt):
            right[key] = max(right.get(key, 0.0), _row_activation(row))
    deltas = [
        {
            "layer": layer,
            "tokenId": token_id,
            "selectedPromptId": selected_prompt,
            "comparePromptId": compare_prompt,
            "selectedActivation": left.get((layer, token_id), 0.0),
            "compareActivation": right.get((layer, token_id), 0.0),
            "deltaActivation": left.get((layer, token_id), 0.0) - right.get((layer, token_id), 0.0),
        }
        for layer, token_id in sorted(set(left) | set(right))
    ]
    return {"selectedPromptId": selected_prompt, "comparePromptId": compare_prompt, "deltas": deltas}


def build_activation_map_payload(
    artifact: dict,
    summary: dict,
    local_model_context: dict | None = None,
    selected_layer: int | None = None,
    selected_group: str | None = None,
    selected_batch: str | None = None,
    *,
    visualization_mode: str | None = None,
    selected_prompt: Any = None,
    selected_token: int | None = None,
    compare_prompt: Any = None,
    top_k: int = DEFAULT_TOP_K,
    background_opacity: float = 0.24,
    edge_threshold: float = 0.0,
    show_aggregate_heatmap: bool = False,
    show_secondary_branches: bool = True,
    developer_diagnostics: bool = False,
    annotations_path: str | Path | None = None,
) -> dict:
    backend = str(artifact.get("backend") or summary.get("backend") or "llama.cpp")
    model_meta = build_model_meta(summary, local_model_context, backend)
    activation_map = build_activation_map(
        artifact,
        summary,
        local_model_context,
        visualization_mode=visualization_mode,
        selected_prompt=selected_prompt,
        selected_token=selected_token,
        compare_prompt=compare_prompt,
        top_k=top_k,
    )
    data_mode = activation_map.diagnostics.get("dataMode", "unavailable")
    model_meta["layerCount"] = activation_map.layer_count
    model_meta["visualizationMode"] = activation_map.mode
    model_meta["dataMode"] = data_mode

    batches = _batch_rows(artifact)
    layers = _layer_rows(activation_map.layer_count, activation_map.nodes, data_mode)
    if layers:
        valid_selected_layer = selected_layer if selected_layer is not None and 0 <= selected_layer < len(layers) else 0
        for layer in layers:
            layer["selected"] = layer["index"] == valid_selected_layer
    prompt_color_plan = _build_prompt_color_plan(activation_map.mode, [path.prompt_id for path in activation_map.paths])
    prompt_styles = prompt_color_plan["stylesByPrompt"]
    node_groups = _node_group_rows(activation_map.nodes, prompt_styles)
    annotations = load_annotations(annotations_path)
    _apply_annotation_matches(model_meta, node_groups, activation_map.heatmap, annotations)
    edge_rows = [edge for edge in _edge_rows(activation_map.paths, prompt_styles) if edge.get("weight", 0.0) >= edge_threshold]
    activation_paths = _path_rows(activation_map.paths, activation_map.layer_count, prompt_styles)

    selected_layer_row = next((layer for layer in layers if layer.get("selected")), layers[0] if layers else None)
    selected_group_row = next((group for group in node_groups if group["groupId"] == selected_group), None)
    if selected_group_row is None and selected_layer_row:
        selected_group_row = next((group for group in node_groups if group["layerId"] == selected_layer_row["layerId"]), None)
    if selected_group_row is None and node_groups:
        selected_group_row = node_groups[0]
    batch_lookup = {batch["batchId"]: batch for batch in batches}
    selected_batch_id = selected_batch if selected_batch in batch_lookup else (activation_map.selected_batch_id or (batches[0]["batchId"] if batches else None))
    selected_batch_row = batch_lookup.get(selected_batch_id) if selected_batch_id else None
    box2_diagnostics = _box2_diagnostics(
        selected_layer_row,
        node_groups,
        mode=activation_map.mode,
        prompt_color_mode=prompt_color_plan["mode"],
    )

    heatmap_stats = {
        "layers": [
            {
                "layerId": f"L{cell['layer']}",
                "activationCount": cell["count"],
                "maxActivation": cell["activationMax"],
                "meanActivation": cell["activationMean"],
                "density": cell["normalizedActivation"],
                "batchParticipation": cell["promptCount"],
            }
            for cell in activation_map.heatmap
        ],
        "groups": [
            {
                "groupId": cell["nodeId"],
                "layerId": f"L{cell['layer']}",
                "activationCount": cell["count"],
                "maxActivation": cell["activationMax"],
                "meanActivation": cell["activationMean"],
                "density": cell["normalizedActivation"],
                "batchParticipation": cell["promptCount"],
            }
            for cell in activation_map.heatmap
        ],
    }

    compare_data = _compare_prompts(_available_rows(_artifact_trace_rows(artifact)), selected_prompt, compare_prompt) if activation_map.mode == "compare_prompts" else {"deltas": []}
    if activation_map.mode == "compare_prompts":
        compare_prompt_ids = [
            compare_data.get("selectedPromptId"),
            compare_data.get("comparePromptId"),
        ]
        prompt_color_plan = _build_prompt_color_plan(activation_map.mode, compare_prompt_ids)
        compare_styles = prompt_color_plan["stylesByPrompt"]
        selected_style = _style_for_prompt(compare_styles, compare_data.get("selectedPromptId"))
        compare_style = _style_for_prompt(compare_styles, compare_data.get("comparePromptId"))
        compare_data = {
            **compare_data,
            "selectedPromptColor": selected_style["promptColor"],
            "comparePromptColor": compare_style["promptColor"],
            "promptColors": {
                selected_style["promptKey"]: selected_style,
                compare_style["promptKey"]: compare_style,
            },
        }
    legend_entries, legend_more_count = _legend_entries(prompt_color_plan["entries"], activation_paths, batches)

    diagnostics = {
        **activation_map.diagnostics,
        "selectedBatch": selected_batch_row,
        "selectedLayer": selected_layer_row,
        "selectedGroup": selected_group_row,
        "activationValue": (selected_group_row or {}).get("activationValue", 0.0),
        "attributionScore": (selected_group_row or {}).get("attributionScore", 0.0),
        "confidence": (selected_group_row or {}).get("confidence", 0.0),
        "visualizationMode": activation_map.mode,
        "mode": activation_map.mode,
        "dataMode": data_mode,
        "sourceToken": "",
        "destinationToken": (selected_batch_row or {}).get("outputToken", ""),
        "topContributingHeads": [],
        "topContributingFeatures": [],
        "modelMeta": model_meta,
        "captureTimestamp": artifact.get("created_at") or datetime.now(timezone.utc).isoformat(),
        "runId": activation_map.run_id,
        "model": activation_map.model,
        "layerCount": activation_map.layer_count,
        "edgeCount": len(edge_rows),
        "nodeCount": len(node_groups),
        "promptColorMode": prompt_color_plan["mode"],
        "promptColorCount": prompt_color_plan["count"],
        "paletteSize": prompt_color_plan["paletteSize"],
        "colorsReused": prompt_color_plan["colorsReused"],
        "box2": box2_diagnostics,
    }
    if prompt_color_plan["colorsReused"]:
        reuse_warning = f"Prompt path palette reused after {prompt_color_plan['paletteSize']} colors; tooltips identify exact prompts."
        if reuse_warning not in diagnostics.get("warnings", []):
            diagnostics["warnings"] = [*diagnostics.get("warnings", []), reuse_warning]
    diagnostics.update({
        "rows": diagnostics.get("rows", 0),
        "maxLayer": max((int(layer) for layer in diagnostics.get("rows_per_layer", {}) if str(layer).isdigit()), default=None),
        "fieldsPresent": diagnostics.get("fields_present", []),
        "vectorsPresent": diagnostics.get("vectors_present", False),
        "includeVectorsResponse": diagnostics.get("include_vectors_response", False),
        "tokenRangesPresent": diagnostics.get("token_ranges_present", False),
        "nodeRangesPresent": diagnostics.get("node_ranges_present", False),
        "activationMagnitudesPresent": diagnostics.get("activation_magnitudes_present", False),
        "pathEdgeDataExists": diagnostics.get("path_edge_data_exists", False),
        "rendererMode": activation_map.mode,
    })
    patch_meta = artifact.get("activation_patch") or artifact.get("activationPatch")
    if isinstance(patch_meta, dict):
        diagnostics["activationPatch"] = patch_meta

    payload = {
        "mode": activation_map.mode,
        "dataMode": data_mode,
        "modelMeta": model_meta,
        "batches": batches,
        "layers": layers,
        "nodeGroups": node_groups,
        "activationEdges": edge_rows,
        "activationPaths": activation_paths,
        "heatmap": activation_map.heatmap,
        "heatmapStats": heatmap_stats,
        "comparePrompts": compare_data,
        "promptColorLegend": legend_entries,
        "promptColorLegendMoreCount": legend_more_count,
        "promptLegendPanel": {
            "title": "Prompt paths",
            "placement": "external_below_graph",
            "entries": legend_entries,
            "moreCount": legend_more_count,
            "colorsReused": prompt_color_plan["colorsReused"],
            "selectedPromptId": activation_map.selected_prompt_id,
            "supportsClickSelection": False,
            "clickSelectionReason": "Streamlit iframe renderer does not expose prompt click events to session state.",
        },
        "diagnostics": diagnostics,
        "rendererOptions": {
            "visualizationMode": activation_map.mode,
            "selectedPromptId": activation_map.selected_prompt_id,
            "selectedBatchId": activation_map.selected_batch_id,
            "selectedTokenId": activation_map.selected_token_id,
            "topK": top_k,
            "backgroundOpacity": background_opacity,
            "edgeThreshold": edge_threshold,
            "showAggregateHeatmap": show_aggregate_heatmap,
            "showSecondaryBranches": show_secondary_branches,
            "developerDiagnostics": developer_diagnostics,
        },
        "visualizationMode": activation_map.mode,
        "unavailableReason": diagnostics.get("unavailableReason", ""),
    }
    if isinstance(patch_meta, dict):
        payload["activationPatch"] = patch_meta
    return payload


def _inspect_latest() -> None:
    artifacts = latest_run_artifacts(limit=25)
    if not artifacts:
        print(json.dumps({"error": "no run artifacts found"}, indent=2))
        return
    artifact = None
    for candidate in artifacts:
        loaded = load_run_artifact(candidate["artifact_path"])
        if _available_rows(_artifact_trace_rows(loaded)):
            artifact = loaded
            break
    if artifact is None:
        artifact = load_run_artifact(artifacts[0]["artifact_path"])
    payload = build_activation_map_payload(artifact, artifact.get("summary", {}))
    report = inspect_trace_artifact(artifact, artifact.get("summary", {}))
    report["node_count"] = len(payload.get("nodeGroups", []))
    report["edge_count"] = len(payload.get("activationEdges", []))
    report["path_count"] = len(payload.get("activationPaths", []))
    report["renderer_mode"] = payload.get("visualizationMode")
    report["warnings"] = payload.get("diagnostics", {}).get("warnings", [])
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Glass Skull activation map artifacts.")
    parser.add_argument("--inspect-latest", action="store_true", help="print schema and map diagnostics for the latest run artifact")
    args = parser.parse_args()
    if args.inspect_latest:
        _inspect_latest()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
