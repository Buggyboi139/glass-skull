from __future__ import annotations

from typing import Any

import pandas as pd


def rank_activation_paths(
    path_df: pd.DataFrame,
    *,
    positive_label: str,
    negative_label: str,
    min_count: int = 1,
) -> pd.DataFrame:
    if path_df is None or path_df.empty:
        return pd.DataFrame(columns=[
            "layer",
            "stream",
            "positive_label",
            "negative_label",
            "positive_mean",
            "negative_mean",
            "delta",
            "score",
            "positive_count",
            "negative_count",
        ])
    required = {"label", "layer", "stream", "activation_norm"}
    if not required.issubset(path_df.columns):
        return pd.DataFrame()

    df = path_df[path_df.get("trace_available", True) != False].copy()
    df["activation_norm"] = pd.to_numeric(df["activation_norm"], errors="coerce")
    df = df.dropna(subset=["label", "layer", "stream", "activation_norm"])
    df = df[df["label"].isin([positive_label, negative_label])]
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby(["label", "layer", "stream"], as_index=False).agg(
        mean_activation=("activation_norm", "mean"),
        std_activation=("activation_norm", "std"),
        count=("activation_norm", "count"),
    )
    pos = grouped[grouped["label"] == positive_label].rename(columns={
        "mean_activation": "positive_mean",
        "std_activation": "positive_std",
        "count": "positive_count",
    })
    neg = grouped[grouped["label"] == negative_label].rename(columns={
        "mean_activation": "negative_mean",
        "std_activation": "negative_std",
        "count": "negative_count",
    })
    merged = pos.merge(neg, on=["layer", "stream"], how="inner", suffixes=("", "_neg"))
    if merged.empty:
        return pd.DataFrame()

    merged = merged[(merged["positive_count"] >= min_count) & (merged["negative_count"] >= min_count)].copy()
    if merged.empty:
        return pd.DataFrame()

    merged["delta"] = merged["positive_mean"] - merged["negative_mean"]
    pooled_std = (
        pd.to_numeric(merged["positive_std"], errors="coerce").fillna(0)
        + pd.to_numeric(merged["negative_std"], errors="coerce").fillna(0)
    ) / 2
    merged["score"] = merged["delta"].abs() / pooled_std.where(pooled_std > 1e-9, 1.0)
    merged["positive_label"] = positive_label
    merged["negative_label"] = negative_label
    merged = merged.sort_values(["score", "delta"], ascending=[False, False])
    return merged[[
        "layer",
        "stream",
        "positive_label",
        "negative_label",
        "positive_mean",
        "negative_mean",
        "delta",
        "score",
        "positive_count",
        "negative_count",
    ]].reset_index(drop=True)


def recommended_steering_targets(ranked_paths: pd.DataFrame, *, limit: int = 5) -> list[dict[str, Any]]:
    if ranked_paths is None or ranked_paths.empty:
        return []
    rows = []
    for row in ranked_paths.head(limit).to_dict("records"):
        rows.append({
            "layer": int(row["layer"]),
            "stream": str(row["stream"]),
            "score": float(row["score"]),
            "delta": float(row["delta"]),
            "direction": "increase" if float(row["delta"]) >= 0 else "decrease",
            "positive_label": row.get("positive_label"),
            "negative_label": row.get("negative_label"),
        })
    return rows
