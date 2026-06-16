from __future__ import annotations

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
