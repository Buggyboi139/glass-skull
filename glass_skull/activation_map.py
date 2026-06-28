from __future__ import annotations

from datetime import datetime, timezone
from math import ceil
from typing import Any

import pandas as pd

from .run_artifacts import activation_path_df

VISUALIZATION_MODES = {"exact", "sampled", "clustered", "approximate", "unavailable"}


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
    return {
        "source": local_model_context.get("source") if is_local else "TransformerLens",
        "backend": backend,
        "modelName": local_model_context.get("display_name") if is_local else str(summary.get("model_name", "")),
        "architecture": local_model_context.get("architecture") if is_local else str(summary.get("architecture", "transformer")),
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
        "metadataSource": "gguf" if is_local else "transformerlens",
        "metadataErrors": list(local_model_context.get("errors") or []),
        "visualizationMode": "clustered" if layer_count else "unavailable",
    }


def _group_count(hidden_size: int, available_rows: int) -> int:
    if hidden_size <= 0:
        return 0
    target = max(50, min(150, available_rows or 96))
    return max(1, min(target, hidden_size))


def _source_indices(group_index: int, group_count: int, hidden_size: int) -> list[int]:
    if hidden_size <= 0 or group_count <= 0:
        return []
    chunk = max(1, ceil(hidden_size / group_count))
    start = group_index * chunk
    end = min(hidden_size, start + chunk)
    return list(range(start, end))


def _batch_rows(artifact: dict) -> list[dict]:
    rows = []
    for prompt in artifact.get("prompts", []):
        prompt_id = prompt.get("prompt_id", len(rows))
        rows.append({
            "batchId": f"batch-{prompt_id}",
            "promptId": prompt_id,
            "label": prompt.get("label") or "unlabeled",
            "promptPreview": _preview(prompt.get("prompt")),
            "tokenRange": "",
            "outputToken": _preview(prompt.get("output"), 32),
            "traceAvailable": any(row.get("trace_available", True) for row in prompt.get("trace_rows", [])),
        })
    return rows


def _available_path_df(artifact: dict) -> pd.DataFrame:
    df = activation_path_df(artifact)
    if df.empty:
        return df
    df = df[df["trace_available"] != False].copy()
    df["activation_norm"] = pd.to_numeric(df["activation_norm"], errors="coerce")
    return df.dropna(subset=["layer", "activation_norm"])


def _artifact_trace_rows(artifact: dict) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prompt in artifact.get("prompts", []):
        for row in prompt.get("trace_rows", []):
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _unavailable_reason(rows: list[dict[str, Any]]) -> str:
    reasons: list[str] = []
    for row in rows:
        reason = str(row.get("unavailable_reason") or "").strip()
        if reason and reason not in reasons:
            reasons.append(reason)
    return " | ".join(reasons)


def _trace_hidden_size(df: pd.DataFrame) -> int:
    max_dim = -1
    for dims in df.get("top_dims", pd.Series(dtype=object)):
        if not isinstance(dims, list):
            continue
        for item in dims:
            if not isinstance(item, dict):
                continue
            dim = _int_or_none(item.get("dimension"))
            if dim is not None:
                max_dim = max(max_dim, dim)
    return max_dim + 1 if max_dim >= 0 else 0


def _group_index_for_dimension(dimension: int | None, group_count: int, hidden_size: int) -> int:
    if dimension is None or hidden_size <= 0 or group_count <= 1:
        return 0
    chunk = max(1, ceil(hidden_size / group_count))
    return max(0, min(group_count - 1, dimension // chunk))


def _point_y(group_index: int, group_count: int) -> float:
    if group_count <= 0:
        return 0.5
    return (group_index + 0.5) / group_count


def _path_mode(df: pd.DataFrame, trace_rows: list[dict[str, Any]], node_groups: list[dict]) -> str:
    if not df.empty:
        return "sampled"
    if trace_rows:
        return "unavailable"
    if node_groups:
        return "clustered"
    return "unavailable"


def build_activation_map_payload(
    artifact: dict,
    summary: dict,
    local_model_context: dict | None = None,
    selected_layer: int | None = None,
    selected_group: str | None = None,
    selected_batch: str | None = None,
) -> dict:
    backend = str(artifact.get("backend") or summary.get("backend") or "TransformerLens")
    model_meta = build_model_meta(summary, local_model_context, backend)
    batches = _batch_rows(artifact)
    trace_rows = _artifact_trace_rows(artifact)
    available_df = _available_path_df(artifact)
    unavailable_reason = _unavailable_reason(trace_rows)
    trace_hidden_size = _trace_hidden_size(available_df) if not available_df.empty else 0
    hidden_size = max(model_meta["hiddenSize"], trace_hidden_size)
    layer_count = model_meta["layerCount"]

    if layer_count <= 0 and not available_df.empty:
        layer_count = int(pd.to_numeric(available_df["layer"], errors="coerce").dropna().max()) + 1
        model_meta["layerCount"] = layer_count
    if model_meta["hiddenSize"] <= 0 and hidden_size > 0:
        model_meta["hiddenSize"] = hidden_size

    layers: list[dict[str, Any]] = []
    node_groups: list[dict[str, Any]] = []
    group_lookup: dict[tuple[int, int], dict[str, Any]] = {}
    group_heatmap_values: dict[str, list[float]] = {}
    top_dims_lookup: dict[tuple[int, str], list[dict[str, Any]]] = {}

    for layer_index in range(max(layer_count, 0)):
        layer_rows = available_df[available_df["layer"] == layer_index].copy() if not available_df.empty else pd.DataFrame()
        group_count = _group_count(hidden_size, len(layer_rows))
        active_groups: set[int] = set()
        group_values = {group_index: 0.0 for group_index in range(group_count)}
        group_attr = {group_index: 0.0 for group_index in range(group_count)}
        group_batches = {group_index: set() for group_index in range(group_count)}

        if not layer_rows.empty:
            layer_rows["batch_id"] = layer_rows["prompt_id"].apply(lambda value: f"batch-{value}")
            strongest_rows = layer_rows.sort_values("activation_norm", ascending=False)
            for row in strongest_rows.to_dict("records"):
                batch_id = str(row.get("batch_id") or f"batch-{row.get('prompt_id')}")
                activation_value = _float_or_zero(row.get("activation_norm"))
                dims = row.get("top_dims")
                if isinstance(dims, list) and dims:
                    key = (layer_index, batch_id)
                    top_dims_lookup.setdefault(key, [])
                    top_dims_lookup[key].extend([item for item in dims if isinstance(item, dict)])
                    row_group_ids: set[str] = set()
                    for item in dims:
                        if not isinstance(item, dict):
                            continue
                        dim = _int_or_none(item.get("dimension"))
                        group_index = _group_index_for_dimension(dim, group_count, hidden_size)
                        group_id = f"L{layer_index}-G{group_index}"
                        row_group_ids.add(group_id)
                        group_values[group_index] = max(group_values[group_index], activation_value)
                        group_attr[group_index] = max(
                            group_attr[group_index],
                            abs(_float_or_zero(item.get("abs_activation", item.get("activation")))),
                        )
                        group_batches[group_index].add(batch_id)
                        active_groups.add(group_index)
                    for group_id in row_group_ids:
                        group_heatmap_values.setdefault(group_id, []).append(activation_value)
                elif group_count:
                    group_id = f"L{layer_index}-G0"
                    group_values[0] = max(group_values[0], activation_value)
                    group_attr[0] = max(group_attr[0], activation_value)
                    group_batches[0].add(batch_id)
                    active_groups.add(0)
                    group_heatmap_values.setdefault(group_id, []).append(activation_value)

        for group_index in range(group_count):
            group = {
                "groupId": f"L{layer_index}-G{group_index}",
                "layerId": f"L{layer_index}",
                "name": f"G{group_index}",
                "sourceIndices": _source_indices(group_index, group_count, hidden_size),
                "groupingMethod": "index_chunk",
                "activationValue": group_values[group_index],
                "attributionScore": group_attr[group_index],
                "batchParticipation": len(group_batches[group_index]),
                "visualizationMode": "sampled" if group_index in active_groups else "clustered",
            }
            node_groups.append(group)
            group_lookup[(layer_index, group_index)] = group
            group_heatmap_values.setdefault(group["groupId"], [])

        top_active_groups = [
            group_lookup[(layer_index, group_index)]["groupId"]
            for group_index, _ in sorted(group_values.items(), key=lambda item: item[1], reverse=True)
            if group_values[group_index] > 0
        ][:3]

        layers.append({
            "layerId": f"L{layer_index}",
            "index": layer_index,
            "name": f"L{layer_index}",
            "layerType": "transformer_block",
            "nodeCount": hidden_size,
            "groupCount": group_count,
            "activationDensity": (len(active_groups) / group_count) if group_count else 0.0,
            "topActiveGroups": top_active_groups,
            "visualizationMode": "sampled" if not layer_rows.empty else ("clustered" if group_count else "unavailable"),
            "selected": False,
        })

    if layers:
        valid_selected_layer = selected_layer if selected_layer is not None and 0 <= selected_layer < len(layers) else 0
        layers[valid_selected_layer]["selected"] = True
        selected_layer = valid_selected_layer
    else:
        selected_layer = None

    activation_paths: list[dict[str, Any]] = []
    batch_lookup = {batch["batchId"]: batch for batch in batches}
    batch_order = [batch["batchId"] for batch in batches]

    for batch_id in batch_order:
        batch = batch_lookup[batch_id]
        prompt_id = batch["promptId"]
        prompt_rows = available_df[available_df["prompt_id"] == prompt_id].copy() if not available_df.empty else pd.DataFrame()
        if prompt_rows.empty:
            continue

        points = []
        selected_rows = prompt_rows.sort_values(["layer", "activation_norm"], ascending=[True, False])
        for layer_index, layer_rows in selected_rows.groupby("layer", sort=True):
            row = layer_rows.iloc[0].to_dict()
            layer_index = int(layer_index)
            layer_group_count = layers[layer_index]["groupCount"] if 0 <= layer_index < len(layers) else 0
            dims = top_dims_lookup.get((layer_index, batch_id), [])
            dim = _int_or_none(dims[0].get("dimension")) if dims else None
            group_index = _group_index_for_dimension(dim, layer_group_count, hidden_size) if layer_group_count else 0
            group = group_lookup.get((layer_index, group_index))
            points.append({
                "layerId": f"L{layer_index}",
                "layerIndex": layer_index,
                "groupId": group["groupId"] if group else f"L{layer_index}-G0",
                "groupIndex": group_index,
                "x": layer_index / max(len(layers) - 1, 1) if layers else 0.0,
                "y": _point_y(group_index, layer_group_count),
                "activationValue": _float_or_zero(row.get("activation_norm")),
                "stream": str(row.get("stream") or row.get("component") or ""),
                "token": str(row.get("token") or ""),
                "tokenIndex": _int_or_none(row.get("token_index")),
                "visualizationMode": "sampled",
            })

        if not points:
            continue

        strengths = [point["activationValue"] for point in points]
        top_dims = top_dims_lookup.get((points[0]["layerIndex"], batch_id), [])
        activation_paths.append({
            "pathId": f"{batch_id}-path",
            "batchId": batch_id,
            "promptId": prompt_id,
            "points": points,
            "strength": sum(strengths) / len(strengths),
            "frequency": len(points),
            "tokenRange": batch.get("tokenRange", ""),
            "outputToken": batch.get("outputToken", ""),
            "activationSummary": f"{len(points)} traced layers",
            "attributionScore": max((abs(_float_or_zero(item.get("abs_activation", item.get("activation")))) for item in top_dims), default=0.0),
            "confidence": 0.75,
            "visualizationMode": "sampled",
        })

    selected_batch_id = selected_batch if selected_batch in batch_lookup else (activation_paths[0]["batchId"] if activation_paths else (batches[0]["batchId"] if batches else None))
    selected_batch_row = batch_lookup.get(selected_batch_id) if selected_batch_id else None

    selected_group_row = None
    if selected_group:
        selected_group_row = next((group for group in node_groups if group["groupId"] == selected_group), None)
    if selected_group_row is None and selected_layer is not None:
        selected_group_row = next((group for group in node_groups if group["layerId"] == f"L{selected_layer}"), None)
    if selected_group_row is None and node_groups:
        selected_group_row = node_groups[0]

    selected_layer_row = None
    if selected_layer is not None:
        selected_layer_row = next((layer for layer in layers if layer["index"] == selected_layer), None)
    if selected_layer_row is None and layers:
        selected_layer_row = layers[0]

    selected_path = next((path for path in activation_paths if path["batchId"] == selected_batch_id), None)
    selected_point = None
    if selected_path and selected_layer_row:
        selected_point = next((point for point in selected_path["points"] if point["layerId"] == selected_layer_row["layerId"]), None)

    heatmap_stats = {
        "layers": [],
        "groups": [],
    }
    for layer in layers:
        layer_rows = available_df[available_df["layer"] == layer["index"]] if not available_df.empty else pd.DataFrame()
        heatmap_stats["layers"].append({
            "layerId": layer["layerId"],
            "activationCount": int(len(layer_rows)),
            "maxActivation": _float_or_zero(layer_rows["activation_norm"].max()) if not layer_rows.empty else 0.0,
            "meanActivation": _float_or_zero(layer_rows["activation_norm"].mean()) if not layer_rows.empty else 0.0,
            "density": layer["activationDensity"],
            "batchParticipation": int(layer_rows["prompt_id"].nunique()) if not layer_rows.empty else 0,
        })
    for group in node_groups:
        group_values = group_heatmap_values.get(group["groupId"], [])
        heatmap_stats["groups"].append({
            "groupId": group["groupId"],
            "layerId": group["layerId"],
            "activationCount": len(group_values),
            "maxActivation": max(group_values) if group_values else 0.0,
            "meanActivation": (sum(group_values) / len(group_values)) if group_values else 0.0,
            "density": 1.0 if group_values else 0.0,
            "batchParticipation": group["batchParticipation"],
        })

    visualization_mode = _path_mode(available_df, trace_rows, node_groups)
    diagnostics = {
        "selectedBatch": selected_batch_row,
        "selectedLayer": selected_layer_row,
        "selectedGroup": selected_group_row,
        "activationValue": selected_point["activationValue"] if selected_point else (selected_group_row or {}).get("activationValue", 0.0),
        "attributionScore": selected_path["attributionScore"] if selected_path else (selected_group_row or {}).get("attributionScore", 0.0),
        "confidence": selected_path["confidence"] if selected_path else 0.0,
        "visualizationMode": visualization_mode,
        "unavailableReason": unavailable_reason,
        "sourceToken": selected_point["token"] if selected_point else "",
        "destinationToken": (selected_batch_row or {}).get("outputToken", ""),
        "topContributingHeads": [],
        "topContributingFeatures": top_dims_lookup.get(
            (selected_layer_row["index"], selected_batch_id),
            [],
        )[:5] if selected_layer_row and selected_batch_id else [],
        "modelMeta": model_meta,
        "captureTimestamp": artifact.get("created_at") or datetime.now(timezone.utc).isoformat(),
    }

    return {
        "modelMeta": model_meta,
        "batches": batches,
        "layers": layers,
        "nodeGroups": node_groups,
        "activationPaths": activation_paths,
        "heatmapStats": heatmap_stats,
        "diagnostics": diagnostics,
        "visualizationMode": visualization_mode,
        "unavailableReason": unavailable_reason,
    }
