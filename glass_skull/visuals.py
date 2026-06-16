from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def activation_heatmap(layer_norms: pd.DataFrame):
    if layer_norms.empty:
        return None
    heat = layer_norms.copy()
    heat["layer_stream"] = heat["layer"].astype(str) + ":" + heat["stream"]
    pivot = heat.pivot_table(index="layer_stream", columns="token_index", values="norm", aggfunc="mean")
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="Turbo",
        labels={"x": "token", "y": "layer:stream", "color": "activation"},
        title="Activation heatmap",
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def activation_pulse(layer_norms: pd.DataFrame):
    """Create a more visceral layer-stack visual from activation norms."""
    if layer_norms.empty:
        return None

    df = layer_norms.groupby(["layer", "stream"], as_index=False)["norm"].mean()
    streams = ["resid_pre", "attn_out", "mlp_out", "resid_post"]
    stream_y = {name: i for i, name in enumerate(streams)}
    df["x"] = df["layer"].astype(float)
    df["y"] = df["stream"].map(stream_y).fillna(0).astype(float)

    fig = go.Figure()
    max_norm = max(float(df["norm"].max()), 1e-6)
    for stream in streams:
        sub = df[df["stream"] == stream]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["x"],
            y=sub["y"],
            mode="lines+markers",
            name=stream,
            marker=dict(size=10 + 18 * (sub["norm"] / max_norm)),
            line=dict(width=2),
            text=[f"layer {r.layer}<br>{r.stream}<br>norm {r.norm:.3f}" for r in sub.itertuples()],
            hoverinfo="text",
        ))

    fig.update_layout(
        title="Activation pulse through the model",
        xaxis_title="Layer",
        yaxis=dict(
            tickmode="array",
            tickvals=list(stream_y.values()),
            ticktext=list(stream_y.keys()),
            title="Stream",
        ),
        height=320,
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return fig


def edge_constellation(edges: pd.DataFrame):
    if edges.empty:
        return None
    fig = px.scatter(
        edges,
        x="from_dim",
        y="to_dim",
        size="abs_contribution",
        color="contribution",
        color_continuous_scale="RdBu",
        hover_data=["rank", "source_hook", "input_activation", "weight", "contribution"],
        title="Active contribution constellation",
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def fuzz_prompt_layer_fig(df: pd.DataFrame):
    if df.empty:
        return None
    pivot = df.pivot_table(index="prompt_id", columns="layer", values="norm", aggfunc="mean")
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="Turbo",
        labels={"x": "layer", "y": "prompt", "color": "norm"},
        title="Fuzz heatmap: prompt by layer",
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def fuzz_label_layer_fig(df: pd.DataFrame):
    if df.empty:
        return None
    pivot = df.pivot_table(index="label", columns="layer", values="norm", aggfunc="mean")
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="Turbo",
        labels={"x": "layer", "y": "label", "color": "avg norm"},
        title="Fuzz heatmap: label by layer",
    )
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def dim_frequency_fig(df: pd.DataFrame, limit: int = 50):
    if df.empty:
        return None
    top = df.sort_values(["count", "mean_abs_activation"], ascending=[False, False]).head(limit).copy()
    top["layer_dim"] = "L" + top["layer"].astype(str) + ":" + top["dimension"].astype(str)
    fig = px.bar(
        top,
        x="count",
        y="layer_dim",
        color="mean_abs_activation",
        orientation="h",
        title="Most recurring active dimensions",
        labels={"count": "times in top-k", "layer_dim": "layer:dimension"},
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=45, b=10), yaxis=dict(autorange="reversed"))
    return fig
