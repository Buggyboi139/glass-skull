from __future__ import annotations

import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def activation_path_graph(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    required = {"layer", "stream", "activation_norm"}
    if not required.issubset(df.columns):
        return None
    plot_df = df[df.get("trace_available", True) != False].copy()
    if plot_df.empty:
        return None
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
        color_continuous_scale="Turbo",
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
        color_continuous_scale="Turbo",
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


def attention_heatmap(attn: pd.DataFrame):
    if attn.empty:
        return None
    required = {"dest_index", "src_index", "dest_token", "src_token", "attention"}
    if not required.issubset(attn.columns):
        return None

    row_labels = attn.sort_values("dest_index").drop_duplicates("dest_index")["dest_token"].tolist()
    col_labels = attn.sort_values("src_index").drop_duplicates("src_index")["src_token"].tolist()
    pivot = attn.pivot_table(index="dest_token", columns="src_token", values="attention", aggfunc="mean")
    pivot = pivot.reindex(index=row_labels, columns=col_labels)
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="Turbo",
        labels={"x": "source token", "y": "destination token", "color": "attention"},
        title="Attention head token map",
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def logit_lens_probability_fig(df: pd.DataFrame):
    if df.empty or "rank" not in df.columns:
        return None
    top = df[df["rank"] == 1].copy()
    if top.empty:
        return None
    fig = px.bar(
        top,
        x="layer",
        y="probability",
        hover_data=["token", "token_id", "logit"],
        title="Logit Lens top prediction confidence by layer",
        labels={"probability": "top-token probability"},
    )
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def logit_lens_token_heatmap(df: pd.DataFrame):
    if df.empty:
        return None
    pivot = df.pivot_table(index="layer", columns="token_index", values="probability", aggfunc="max")
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="Turbo",
        labels={"x": "input token", "y": "layer", "color": "top-token probability"},
        title="Logit Lens certainty by layer and token",
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def comparison_delta_heatmap(df: pd.DataFrame):
    if df.empty or "delta" not in df.columns:
        return None
    table = df.copy()
    table["layer_stream"] = table["layer"].astype(str) + ":" + table["stream"]
    pivot = table.pivot_table(index="layer_stream", columns="token_index", values="delta", aggfunc="mean")
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="RdBu",
        labels={"x": "token", "y": "layer:stream", "color": "steered - normal"},
        title="Normal vs steered activation delta",
    )
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10))
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


# ---------------------------------------------------------------------------
# Architecture graphs (derived directly from the loaded model parameters)
# ---------------------------------------------------------------------------
def _layer_of(name: str) -> int:
    match = re.search(r"blocks\.(\d+)\.", name)
    return int(match.group(1)) if match else -1


def _component_of(name: str) -> str:
    if "blocks." not in name:
        if "embed" in name and "unembed" not in name:
            return "embedding"
        if "unembed" in name:
            return "unembedding"
        if "ln_final" in name or name.startswith("ln"):
            return "final layer norm"
        return "other"
    if ".attn." in name:
        return "attention"
    if ".mlp." in name:
        return "mlp"
    if ".ln" in name:
        return "layer norm"
    return "other"


def parameters_per_layer_fig(params: pd.DataFrame):
    if params is None or params.empty or "name" not in params.columns:
        return None
    df = params.copy()
    df["layer"] = df["name"].map(_layer_of)
    block = df[df["layer"] >= 0].groupby("layer", as_index=False)["parameters"].sum()
    if block.empty:
        return None
    fig = px.bar(
        block,
        x="layer",
        y="parameters",
        title="Parameters per transformer block",
        labels={"layer": "block index", "parameters": "parameter count"},
        color="parameters",
        color_continuous_scale="Tealgrn",
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10), coloraxis_showscale=False)
    return fig


def parameters_by_component_fig(params: pd.DataFrame):
    if params is None or params.empty or "name" not in params.columns:
        return None
    df = params.copy()
    df["component"] = df["name"].map(_component_of)
    agg = df.groupby("component", as_index=False)["parameters"].sum().sort_values("parameters")
    if agg.empty:
        return None
    fig = px.bar(
        agg,
        x="parameters",
        y="component",
        orientation="h",
        title="Parameters by component type",
        labels={"parameters": "parameter count", "component": ""},
        color="parameters",
        color_continuous_scale="Purpor",
    )
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=45, b=10), coloraxis_showscale=False)
    return fig


def parameter_shape_scatter_fig(params: pd.DataFrame):
    if params is None or params.empty or "name" not in params.columns:
        return None
    df = params.copy()
    df["layer"] = df["name"].map(_layer_of)
    df["component"] = df["name"].map(_component_of)
    df = df[df["parameters"] > 0]
    if df.empty:
        return None
    fig = px.scatter(
        df,
        x="layer",
        y="parameters",
        color="component",
        size="parameters",
        hover_data=["name", "shape", "dtype"],
        title="Parameter tensors across depth",
        labels={"layer": "block index (-1 = global)", "parameters": "parameter count"},
    )
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10))
    return fig


# ---------------------------------------------------------------------------
# GGUF model graphs (derived from local tensor index, no model load required)
# ---------------------------------------------------------------------------
def _gguf_layer_of(name: str) -> int:
    match = re.search(r"blk\.(\d+)\.", name)
    return int(match.group(1)) if match else -1


def _gguf_component_of(name: str) -> str:
    if name.startswith("token_embd"):
        return "embedding"
    if name.startswith("output."):
        return "output"
    if "norm" in name:
        return "norm"
    if ".attn" in name:
        return "attention"
    if ".ffn" in name:
        return "ffn / moe"
    if ".ssm" in name:
        return "ssm / mtp"
    return "other"


def gguf_tensors_per_layer_fig(tensors: pd.DataFrame):
    if tensors is None or tensors.empty or "name" not in tensors.columns:
        return None
    df = tensors.copy()
    df["layer"] = df["name"].map(_gguf_layer_of)
    block = df[df["layer"] >= 0].groupby("layer", as_index=False).agg(
        tensors=("name", "count"),
        elements=("elements", "sum"),
    )
    if block.empty:
        return None
    fig = px.bar(
        block,
        x="layer",
        y="elements",
        hover_data=["tensors"],
        title="GGUF tensor elements per local model block",
        labels={"layer": "block index", "elements": "tensor elements"},
        color="elements",
        color_continuous_scale="Tealgrn",
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10), coloraxis_showscale=False)
    return fig


def gguf_tensors_by_component_fig(tensors: pd.DataFrame):
    if tensors is None or tensors.empty or "name" not in tensors.columns:
        return None
    df = tensors.copy()
    df["component"] = df["name"].map(_gguf_component_of)
    agg = df.groupby("component", as_index=False).agg(
        tensors=("name", "count"),
        elements=("elements", "sum"),
    ).sort_values("elements")
    if agg.empty:
        return None
    fig = px.bar(
        agg,
        x="elements",
        y="component",
        orientation="h",
        hover_data=["tensors"],
        title="GGUF tensor elements by component",
        labels={"elements": "tensor elements", "component": ""},
        color="elements",
        color_continuous_scale="Purpor",
    )
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=45, b=10), coloraxis_showscale=False)
    return fig


def gguf_tensor_dtype_fig(tensors: pd.DataFrame):
    if tensors is None or tensors.empty or "dtype" not in tensors.columns:
        return None
    agg = tensors.groupby("dtype", as_index=False).agg(
        tensors=("name", "count"),
        elements=("elements", "sum"),
    ).sort_values("elements")
    if agg.empty:
        return None
    fig = px.bar(
        agg,
        x="elements",
        y="dtype",
        orientation="h",
        hover_data=["tensors"],
        title="GGUF quantization types",
        labels={"elements": "tensor elements", "dtype": ""},
        color="elements",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=45, b=10), coloraxis_showscale=False)
    return fig


def gguf_tensor_shape_scatter_fig(tensors: pd.DataFrame):
    if tensors is None or tensors.empty or "name" not in tensors.columns:
        return None
    df = tensors.copy()
    df["layer"] = df["name"].map(_gguf_layer_of)
    df["component"] = df["name"].map(_gguf_component_of)
    df = df[df["elements"] > 0]
    if df.empty:
        return None
    fig = px.scatter(
        df,
        x="layer",
        y="elements",
        color="component",
        size="elements",
        hover_data=["name", "shape", "dtype"],
        title="GGUF tensors across local model depth",
        labels={"layer": "block index (-1 = global)", "elements": "tensor elements"},
    )
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10))
    return fig


# ---------------------------------------------------------------------------
# Trace-derived graphs
# ---------------------------------------------------------------------------
def mean_norm_by_layer_fig(layer_norms: pd.DataFrame):
    if layer_norms is None or layer_norms.empty:
        return None
    df = layer_norms.groupby(["layer", "stream"], as_index=False)["norm"].mean()
    fig = px.line(
        df,
        x="layer",
        y="norm",
        color="stream",
        markers=True,
        title="Mean activation norm by layer",
        labels={"layer": "layer", "norm": "mean L2 norm"},
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def norm_growth_fig(layer_norms: pd.DataFrame):
    if layer_norms is None or layer_norms.empty:
        return None
    resid = layer_norms[layer_norms["stream"] == "resid_post"]
    if resid.empty:
        resid = layer_norms
    df = resid.groupby("layer", as_index=False)["norm"].mean()
    fig = px.area(
        df,
        x="layer",
        y="norm",
        title="Residual stream magnitude through depth",
        labels={"layer": "layer", "norm": "mean resid_post norm"},
    )
    fig.update_traces(line_color="#38bdf8", fillcolor="rgba(56,189,248,0.18)")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def next_token_bar_fig(df: pd.DataFrame):
    if df is None or df.empty or "token" not in df.columns or "probability" not in df.columns:
        return None
    top = df.sort_values("probability", ascending=True).copy()
    top["label"] = top["token"].astype(str) + "  (#" + top["token_id"].astype(str) + ")"
    fig = px.bar(
        top,
        x="probability",
        y="label",
        orientation="h",
        hover_data=["token_id", "logit"],
        title="Top next-token probabilities",
        labels={"probability": "probability", "label": ""},
        color="probability",
        color_continuous_scale="Tealgrn",
    )
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=45, b=10), coloraxis_showscale=False)
    return fig
