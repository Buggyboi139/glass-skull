from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd


def prompt_layer_heatmap(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in records:
        prompt_id = rec.get("prompt_id")
        label = rec.get("label", "unlabeled")
        for layer in rec.get("trace_layers", []):
            rows.append({
                "prompt_id": prompt_id,
                "label": label,
                "layer": layer.get("layer"),
                "stream": layer.get("stream", "resid_post"),
                "norm": layer.get("norm"),
            })
    return pd.DataFrame(rows)


def label_layer_heatmap(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = prompt_layer_heatmap(records)
    if df.empty:
        return df
    return df.groupby(["label", "layer", "stream"], as_index=False)["norm"].mean()


def dimension_frequency(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in records:
        label = rec.get("label", "unlabeled")
        for layer in rec.get("trace_layers", []):
            for item in layer.get("top_dims", []):
                rows.append({
                    "label": label,
                    "layer": layer.get("layer"),
                    "stream": layer.get("stream", "resid_post"),
                    "dimension": item.get("dimension"),
                    "count": 1,
                    "mean_abs_activation": abs(float(item.get("activation", 0.0))),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.groupby(["label", "layer", "stream", "dimension"], as_index=False).agg(
        count=("count", "sum"),
        mean_abs_activation=("mean_abs_activation", "mean"),
    )


def top_recurring_dimensions(records: list[dict[str, Any]], limit: int = 50) -> pd.DataFrame:
    df = dimension_frequency(records)
    if df.empty:
        return df
    return df.sort_values(["count", "mean_abs_activation"], ascending=[False, False]).head(limit)


def _finite_or_zero(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def label_separation_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Rank label pairs by activation-norm separation for each layer/stream.

    This is not a final causal proof. It is a cheap locator beacon for where to try a contrast feature.
    """
    df = prompt_layer_heatmap(records)
    if df.empty or "label" not in df:
        return pd.DataFrame()

    labels = sorted([x for x in df["label"].dropna().unique().tolist() if x != "unlabeled"])
    if len(labels) < 2:
        return pd.DataFrame()

    rows = []
    for a, b in combinations(labels, 2):
        sub = df[df["label"].isin([a, b])]
        grouped = sub.groupby(["label", "layer", "stream"], as_index=False).agg(
            mean_norm=("norm", "mean"),
            std_norm=("norm", "std"),
            count=("norm", "count"),
        )
        for (layer, stream), chunk in grouped.groupby(["layer", "stream"]):
            if set(chunk["label"]) != {a, b}:
                continue
            row_a = chunk[chunk["label"] == a].iloc[0]
            row_b = chunk[chunk["label"] == b].iloc[0]
            mean_a = _finite_or_zero(row_a["mean_norm"])
            mean_b = _finite_or_zero(row_b["mean_norm"])
            diff = mean_a - mean_b
            std_a = _finite_or_zero(row_a["std_norm"])
            std_b = _finite_or_zero(row_b["std_norm"])
            pooled = (std_a + std_b) / 2.0
            score = abs(diff) / (pooled + 1e-6)
            rows.append({
                "label_a": a,
                "label_b": b,
                "layer": int(layer),
                "stream": stream,
                "mean_a": mean_a,
                "mean_b": mean_b,
                "difference": diff,
                "abs_difference": abs(diff),
                "separation_score": score,
                "count_a": int(row_a["count"]),
                "count_b": int(row_b["count"]),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["separation_score", "abs_difference"], ascending=[False, False]).reset_index(drop=True)
