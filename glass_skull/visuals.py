from __future__ import annotations

import re

import pandas as pd
import plotly.express as px


SEQUENTIAL_SCALE = "Viridis"
DIVERGING_SCALE = "RdBu_r"
QUALITATIVE_COLORS = px.colors.qualitative.Safe


def activation_path_graph(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    required = {"layer", "stream", "activation_norm"}
    if not required.issubset(df.columns):
        return None
    plot_df = df[df.get("trace_available", True) != False].copy()
    plot_df["activation_norm"] = pd.to_numeric(plot_df["activation_norm"], errors="coerce")
    plot_df = plot_df.dropna(subset=["layer", "stream", "activation_norm"])
    if plot_df.empty:
        return None
    max_norm = max(float(plot_df["activation_norm"].max()), 1e-6)
    plot_df["marker_size"] = 8 + 18 * (plot_df["activation_norm"] / max_norm)
    hover_cols = [col for col in ["prompt_id", "label", "token_index", "token", "activation_norm"] if col in plot_df.columns]
    fig = px.scatter(
        plot_df,
        x="layer",
        y="stream",
        size="marker_size",
        color="activation_norm",
        color_continuous_scale=SEQUENTIAL_SCALE,
        hover_data=hover_cols,
        title="Activation path",
        labels={"layer": "Layer", "stream": "Stream/component", "activation_norm": "activation norm"},
    )
    fig.update_traces(marker=dict(sizemode="diameter", sizeref=1))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def batch_activation_heatmap(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    required = {"group", "layer", "stream", "activation_norm"}
    if not required.issubset(df.columns):
        return None
    heat = df.copy()
    heat["path"] = heat["layer"].astype(str) + ":" + heat["stream"].astype(str)
    pivot = heat.pivot_table(index="group", columns="path", values="activation_norm", aggfunc="mean")
    if pivot.empty:
        return None
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale=SEQUENTIAL_SCALE,
        labels={"x": "layer:stream", "y": "group", "color": "avg activation norm"},
        title="Batch activation heatmap",
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def label_activation_heatmap(df: pd.DataFrame):
    fig = batch_activation_heatmap(df)
    if fig is not None:
        fig.update_layout(title="Label activation heatmap")
    return fig


def behavior_score_timeline_fig(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    required = {"run_order", "run_id", "label", "score"}
    if not required.issubset(df.columns):
        return None
    plot_df = df.copy()
    plot_df["score"] = pd.to_numeric(plot_df["score"], errors="coerce")
    plot_df = plot_df.dropna(subset=["score"])
    if plot_df.empty:
        return None
    plot_df["run_label"] = plot_df["run_order"].astype(str) + " - " + plot_df["run_id"].astype(str)
    fig = px.line(
        plot_df.sort_values(["run_order", "label"]),
        x="run_label",
        y="score",
        color="label",
        markers=True,
        color_discrete_sequence=QUALITATIVE_COLORS,
        title="Behavior score over runs",
        labels={"run_label": "Run", "score": "Mean behavior score", "label": "Prompt label"},
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=20), xaxis_tickangle=-20)
    return fig


def behavior_delta_bar_fig(df: pd.DataFrame, *, baseline_run_id: str, comparison_run_id: str):
    if df is None or df.empty:
        return None
    required = {"run_id", "label", "score"}
    if not required.issubset(df.columns):
        return None
    plot_df = df.copy()
    plot_df["score"] = pd.to_numeric(plot_df["score"], errors="coerce")
    pivot = plot_df.pivot_table(index="label", columns="run_id", values="score", aggfunc="mean")
    if baseline_run_id not in pivot or comparison_run_id not in pivot:
        return None
    delta = (pivot[comparison_run_id] - pivot[baseline_run_id]).dropna().reset_index(name="delta")
    if delta.empty:
        return None
    fig = px.bar(
        delta.sort_values("delta"),
        x="delta",
        y="label",
        orientation="h",
        color="delta",
        color_continuous_scale=DIVERGING_SCALE,
        color_continuous_midpoint=0,
        title="Behavior score delta",
        labels={"delta": f"{comparison_run_id} minus {baseline_run_id}", "label": "Prompt label"},
    )
    fig.add_vline(x=0, line_width=1, line_color="rgba(255,255,255,0.5)")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def path_rank_bar_fig(df: pd.DataFrame, limit: int = 12):
    if df is None or df.empty:
        return None
    required = {"layer", "stream", "score", "delta"}
    if not required.issubset(df.columns):
        return None
    plot_df = df.head(limit).copy()
    plot_df["score"] = pd.to_numeric(plot_df["score"], errors="coerce")
    plot_df["delta"] = pd.to_numeric(plot_df["delta"], errors="coerce")
    plot_df = plot_df.dropna(subset=["score", "delta"])
    if plot_df.empty:
        return None
    plot_df["path"] = "L" + plot_df["layer"].astype(int).astype(str) + " " + plot_df["stream"].astype(str)
    fig = px.bar(
        plot_df.sort_values("score"),
        x="score",
        y="path",
        orientation="h",
        color="delta",
        color_continuous_scale=DIVERGING_SCALE,
        color_continuous_midpoint=0,
        title="Ranked activation paths",
        labels={"score": "Separation score", "path": "Layer and stream", "delta": "Positive - negative"},
    )
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def dim_frequency_fig(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    required = {"dimension", "count", "mean_abs_activation"}
    if not required.issubset(df.columns):
        return None
    plot_df = df.copy()
    plot_df["dimension"] = plot_df["dimension"].astype(str)
    fig = px.bar(
        plot_df.sort_values(["count", "mean_abs_activation"], ascending=[False, False]).head(40),
        x="dimension",
        y="count",
        color="mean_abs_activation",
        color_continuous_scale=SEQUENTIAL_SCALE,
        title="Recurring active dimensions",
        labels={"dimension": "Dimension", "count": "Count", "mean_abs_activation": "mean abs activation"},
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def _tensor_component(name: str) -> str:
    if ".attn_" in name or ".attention." in name:
        return "attention"
    if ".ffn_" in name or ".mlp." in name or "feed_forward" in name:
        return "mlp"
    if "token_embd" in name or "embed" in name:
        return "embedding"
    if "output" in name:
        return "output"
    if "norm" in name:
        return "norm"
    return "other"


def _tensor_layer(name: str) -> int:
    match = re.search(r"(?:blk|layers?)\.(\d+)", name)
    return int(match.group(1)) if match else -1


def gguf_tensor_shape_scatter_fig(tensors: pd.DataFrame):
    if tensors is None or tensors.empty:
        return None
    required = {"name", "elements"}
    if not required.issubset(tensors.columns):
        return None
    df = tensors.copy()
    df["elements"] = pd.to_numeric(df["elements"], errors="coerce").fillna(0)
    df = df[df["elements"] > 0]
    if df.empty:
        return None
    df["layer"] = df["name"].astype(str).map(_tensor_layer)
    df["component"] = df["name"].astype(str).map(_tensor_component)
    fig = px.scatter(
        df,
        x="layer",
        y="elements",
        color="component",
        size="elements",
        hover_data=[col for col in ["name", "shape", "dtype", "offset"] if col in df.columns],
        title="GGUF tensor shapes",
        labels={"layer": "block index (-1 = global)", "elements": "tensor elements", "component": "component"},
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=45, b=10))
    return fig
