from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pandas as pd


SCHEMA_VERSION = 1
TRACE_COLUMNS = [
    "run_id",
    "prompt_id",
    "label",
    "layer",
    "stream",
    "component",
    "token_index",
    "token",
    "activation_norm",
    "trace_available",
    "trace_source",
    "unavailable_reason",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _int_or_none(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def _token_from_trace(tokens: list[Any], token_index: int | None, fallback: Any = None) -> str:
    if fallback is not None and str(fallback) != "":
        return str(fallback)
    if token_index is not None and 0 <= token_index < len(tokens):
        token = tokens[token_index]
        if isinstance(token, dict):
            return str(token.get("piece") or token.get("token") or token.get("id") or token_index)
        return str(token)
    return "" if token_index is None else str(token_index)


def _llama_tokens(payload: dict[str, Any]) -> list[Any]:
    prompt = payload.get("prompt")
    if isinstance(prompt, dict):
        traces = prompt.get("traces")
        if isinstance(traces, list) and traces:
            first = traces[0] if isinstance(traces[0], dict) else {}
            pieces = first.get("pieces")
            if isinstance(pieces, list) and pieces:
                return pieces
            tokens = first.get("tokens")
            if isinstance(tokens, list):
                return tokens
    tokens = payload.get("tokens")
    return tokens if isinstance(tokens, list) else []


def _unavailable_reason(payload: dict[str, Any]) -> str:
    for key in ("activations", "layer_norms", "trace"):
        value = payload.get(key)
        if isinstance(value, dict):
            reason = value.get("reason") or value.get("error")
            if reason:
                return str(reason)
    return "activation summaries were not returned by this backend"


def normalize_transformerlens_trace(
    trace: Any,
    prompt_id: Any,
    label: str | None,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    metadata = metadata or {}
    run_id = str(metadata.get("run_id") or "")
    layer_norms = getattr(trace, "layer_norms", pd.DataFrame())
    if layer_norms is None or getattr(layer_norms, "empty", True):
        return []
    tokens = list(getattr(trace, "tokens", []) or [])
    rows: list[dict[str, Any]] = []
    for row in layer_norms.to_dict("records"):
        token_index = _int_or_none(row.get("token_index"))
        stream = str(row.get("stream") or row.get("component") or "resid_post")
        rows.append({
            "run_id": run_id,
            "prompt_id": prompt_id,
            "label": label or "unlabeled",
            "layer": _int_or_none(row.get("layer")),
            "stream": stream,
            "component": stream,
            "token_index": token_index,
            "token": _token_from_trace(tokens, token_index, row.get("token")),
            "activation_norm": _finite_float(row.get("activation_norm", row.get("norm"))),
            "trace_available": True,
            "trace_source": "transformerlens",
            "unavailable_reason": "",
            "top_dims": row.get("top_dims", []),
        })
    return rows


def normalize_llama_trace(
    payload: dict[str, Any],
    prompt_id: Any,
    label: str | None,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    metadata = metadata or {}
    run_id = str(metadata.get("run_id") or "")
    layer_norms = payload.get("layer_norms")
    if not isinstance(layer_norms, list) or not layer_norms:
        return []

    tokens = _llama_tokens(payload)
    rows: list[dict[str, Any]] = []
    for row in layer_norms:
        if not isinstance(row, dict):
            continue
        token_index = _int_or_none(row.get("token_index"))
        stream = str(row.get("stream") or row.get("component") or "resid_post")
        rows.append({
            "run_id": run_id,
            "prompt_id": prompt_id,
            "label": label or "unlabeled",
            "layer": _int_or_none(row.get("layer")),
            "stream": stream,
            "component": stream,
            "token_index": token_index,
            "token": _token_from_trace(tokens, token_index, row.get("token")),
            "activation_norm": _finite_float(row.get("activation_norm", row.get("norm"))),
            "trace_available": True,
            "trace_source": "llama.cpp",
            "unavailable_reason": "",
            "top_dims": row.get("top_dims", []),
        })
    return rows


def trace_unavailable_row(
    run_id: str,
    prompt_id: Any,
    label: str | None,
    source: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "prompt_id": prompt_id,
        "label": label or "unlabeled",
        "layer": None,
        "stream": "",
        "component": "",
        "token_index": None,
        "token": "",
        "activation_norm": None,
        "trace_available": False,
        "trace_source": source,
        "unavailable_reason": reason,
        "top_dims": [],
    }


def llama_trace_unavailable_reason(payload: dict[str, Any]) -> str:
    return _unavailable_reason(payload)


def build_run_artifact(
    *,
    run_id: str | None,
    mode: str,
    backend: str,
    model: str | None,
    prompts: list[dict[str, Any]],
    summary: dict[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    clean_run_id = run_id or uuid4().hex[:12]
    prompt_runs = []
    error_count = 0
    latencies = []
    trace_rows_count = 0
    unavailable_count = 0

    for idx, prompt in enumerate(prompts):
        trace_rows = list(prompt.get("trace_rows") or [])
        trace_rows_count += sum(1 for row in trace_rows if row.get("trace_available", True))
        unavailable_count += sum(1 for row in trace_rows if row.get("trace_available") is False)
        if prompt.get("error"):
            error_count += 1
        if prompt.get("elapsed_ms") is not None:
            latencies.append(_finite_float(prompt.get("elapsed_ms")))
        prompt_runs.append({
            "prompt_id": prompt.get("prompt_id", idx),
            "label": prompt.get("label") or "unlabeled",
            "prompt": prompt.get("prompt", ""),
            "output": prompt.get("output", ""),
            "error": prompt.get("error"),
            "elapsed_ms": prompt.get("elapsed_ms"),
            "metadata": prompt.get("metadata") or {},
            "trace_rows": trace_rows,
        })

    base_summary = {
        "prompt_count": len(prompt_runs),
        "error_count": error_count,
        "latency_ms_mean": sum(latencies) / len(latencies) if latencies else None,
        "trace_row_count": trace_rows_count,
        "trace_unavailable_count": unavailable_count,
        "trace_supported": trace_rows_count > 0,
    }
    if summary:
        base_summary.update(summary)

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": clean_run_id,
        "mode": mode,
        "backend": backend,
        "model": model or "",
        "prompts": prompt_runs,
        "summary": base_summary,
        "created_at": created_at or _now_iso(),
    }


def activation_path_df(artifact: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for prompt in artifact.get("prompts", []):
        for row in prompt.get("trace_rows", []):
            merged = {col: row.get(col) for col in TRACE_COLUMNS}
            merged["run_id"] = merged.get("run_id") or artifact.get("run_id")
            merged["prompt_id"] = merged.get("prompt_id", prompt.get("prompt_id"))
            merged["label"] = merged.get("label") or prompt.get("label") or "unlabeled"
            rows.append(merged)
    if not rows:
        return pd.DataFrame(columns=TRACE_COLUMNS)
    df = pd.DataFrame(rows)
    for col in TRACE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[TRACE_COLUMNS]


def _available_path_df(artifact: dict[str, Any]) -> pd.DataFrame:
    df = activation_path_df(artifact)
    if df.empty:
        return df
    df = df[df["trace_available"] != False].copy()
    if "activation_norm" in df:
        df["activation_norm"] = pd.to_numeric(df["activation_norm"], errors="coerce")
    return df.dropna(subset=["layer", "stream", "activation_norm"])


def batch_heatmap_df(artifact: dict[str, Any], group_by: str = "prompt") -> pd.DataFrame:
    df = _available_path_df(artifact)
    if df.empty:
        return pd.DataFrame(columns=["run_id", "group", "layer", "stream", "activation_norm", "count"])

    if group_by == "label":
        df["group"] = df["label"].fillna("unlabeled")
    elif group_by == "all":
        df["group"] = "all prompts"
    else:
        df["group"] = df["prompt_id"].astype(str)

    return df.groupby(["run_id", "group", "layer", "stream"], as_index=False).agg(
        activation_norm=("activation_norm", "mean"),
        count=("activation_norm", "count"),
    )


def label_heatmap_df(artifact: dict[str, Any]) -> pd.DataFrame:
    return batch_heatmap_df(artifact, group_by="label")


def dimension_frequency_df(artifact: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for prompt in artifact.get("prompts", []):
        for trace_row in prompt.get("trace_rows", []):
            for item in trace_row.get("top_dims", []) or []:
                if not isinstance(item, dict):
                    continue
                rows.append({
                    "run_id": trace_row.get("run_id") or artifact.get("run_id"),
                    "prompt_id": trace_row.get("prompt_id", prompt.get("prompt_id")),
                    "label": trace_row.get("label") or prompt.get("label") or "unlabeled",
                    "layer": trace_row.get("layer"),
                    "stream": trace_row.get("stream"),
                    "dimension": item.get("dimension"),
                    "count": 1,
                    "mean_abs_activation": abs(_finite_float(item.get("abs_activation", item.get("activation")))),
                })
    if not rows:
        return pd.DataFrame(columns=["run_id", "prompt_id", "label", "layer", "stream", "dimension", "count", "mean_abs_activation"])
    df = pd.DataFrame(rows)
    return df.groupby(["run_id", "label", "layer", "stream", "dimension"], as_index=False).agg(
        count=("count", "sum"),
        mean_abs_activation=("mean_abs_activation", "mean"),
    )
