from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .gguf import read_gguf_metadata, read_gguf_tensor_index


def _metadata_value(metadata: dict[str, Any], architecture: str, *suffixes: str) -> Any:
    prefixes = [architecture] if architecture and architecture != "unknown" else []
    for suffix in suffixes:
        candidates = [suffix, f"general.{suffix}"]
        for prefix in prefixes:
            candidates.extend([f"{prefix}.{suffix}", f"{prefix}.attention.{suffix}"])
        for key in candidates:
            if key in metadata:
                return metadata[key]
    return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "unknown"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _expert_label(used: int | None, total: int | None) -> str:
    if used is None and total is None:
        return "unknown"
    if used is None:
        return f"{total:,}"
    if total is None:
        return f"{used:,} active"
    return f"{used:,} / {total:,}"


def local_gguf_context(model_path: str | Path, model_alias: str = "", backend: str = "") -> dict[str, Any]:
    """Return display-safe local GGUF summary data.

    Values come from GGUF metadata and tensor headers only. The tensor element
    total is a tensor-index summary, not a live runtime parameter inspection.
    """
    path = Path(model_path)
    metadata: dict[str, Any] = {}
    tensor_df = pd.DataFrame()
    errors: list[str] = []

    if not path.exists():
        errors.append(f"GGUF model path does not exist: {path}")
    else:
        try:
            metadata = read_gguf_metadata(path)
        except Exception as exc:
            errors.append(f"Could not read GGUF metadata: {exc}")
        try:
            tensor_df = pd.DataFrame(read_gguf_tensor_index(path))
        except Exception as exc:
            errors.append(f"Could not read GGUF tensor index: {exc}")

    architecture = str(metadata.get("general.architecture") or metadata.get("architecture") or "unknown")
    block_count = _int_or_none(_metadata_value(metadata, architecture, "block_count"))
    embedding_length = _int_or_none(_metadata_value(metadata, architecture, "embedding_length"))
    head_count = _int_or_none(_metadata_value(metadata, architecture, "head_count", "attention.head_count"))
    head_count_kv = _int_or_none(_metadata_value(metadata, architecture, "head_count_kv", "attention.head_count_kv"))
    d_head = _int_or_none(_metadata_value(metadata, architecture, "key_length", "attention.key_length"))
    if d_head is None and embedding_length and head_count:
        d_head = int(embedding_length / head_count)
    d_mlp = _int_or_none(_metadata_value(metadata, architecture, "expert_feed_forward_length", "feed_forward_length"))
    expert_count = _int_or_none(_metadata_value(metadata, architecture, "expert_count"))
    expert_used_count = _int_or_none(_metadata_value(metadata, architecture, "expert_used_count"))
    context_length = _int_or_none(_metadata_value(metadata, architecture, "context_length"))
    nextn_predict_layers = _int_or_none(_metadata_value(metadata, architecture, "nextn_predict_layers"))
    trace_layer_count = block_count
    if block_count is not None and nextn_predict_layers is not None and 0 < nextn_predict_layers < block_count:
        trace_layer_count = block_count - nextn_predict_layers

    tensor_count = _int_or_none(metadata.get("_tensor_count"))
    if tensor_count is None and not tensor_df.empty:
        tensor_count = len(tensor_df)
    tensor_elements = None
    if not tensor_df.empty and "elements" in tensor_df.columns:
        tensor_elements = int(tensor_df["elements"].sum())

    display_name = model_alias.strip() or path.name or str(path)
    return {
        "source": "Local GGUF",
        "backend": backend,
        "display_name": display_name,
        "model_path": str(path),
        "model_alias": model_alias.strip(),
        "metadata": metadata,
        "tensors_df": tensor_df,
        "errors": errors,
        "architecture": architecture,
        "block_count": block_count,
        "trace_layer_count": trace_layer_count,
        "nextn_predict_layers": nextn_predict_layers,
        "embedding_length": embedding_length,
        "head_count": head_count,
        "head_count_kv": head_count_kv,
        "d_head": d_head,
        "d_mlp": d_mlp,
        "expert_count": expert_count,
        "expert_used_count": expert_used_count,
        "context_length": context_length,
        "tensor_count": tensor_count,
        "tensor_elements": tensor_elements,
        "metadata_kv_count": _int_or_none(metadata.get("_metadata_kv_count")),
        "gguf_version": _int_or_none(metadata.get("_gguf_version")),
        "hud_stats": [
            ("Blocks", _fmt(block_count)),
            ("d_model", _fmt(embedding_length)),
            ("Heads", _fmt(head_count)),
            ("Experts", _expert_label(expert_used_count, expert_count)),
            ("Ctx", _fmt(context_length)),
        ],
        "property_rows": [
            ("source", "Local GGUF"),
            ("active_backend", backend or "unknown"),
            ("model_alias", model_alias.strip() or "unset"),
            ("model_path", str(path)),
            ("architecture", architecture),
            ("block_count", _fmt(block_count)),
            ("trace_layer_count", _fmt(trace_layer_count)),
            ("nextn_predict_layers", _fmt(nextn_predict_layers)),
            ("embedding_length", _fmt(embedding_length)),
            ("head_count", _fmt(head_count)),
            ("head_count_kv", _fmt(head_count_kv)),
            ("d_head", _fmt(d_head)),
            ("expert_feed_forward_length", _fmt(d_mlp)),
            ("expert_used_count", _fmt(expert_used_count)),
            ("expert_count", _fmt(expert_count)),
            ("context_length", _fmt(context_length)),
            ("tensor_entries", _fmt(tensor_count)),
            ("tensor_elements", _fmt(tensor_elements)),
            ("metadata_kv_count", _fmt(_int_or_none(metadata.get("_metadata_kv_count")))),
            ("gguf_version", _fmt(_int_or_none(metadata.get("_gguf_version")))),
        ],
    }


def local_trace_layer_norms(trace_payload: dict[str, Any]) -> pd.DataFrame:
    rows = trace_payload.get("layer_norms", [])
    if not isinstance(rows, list):
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    required = {"layer", "stream", "token_index", "norm"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()
    return df
