from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from glass_skull.anatomy import config_table, expected_block_table, global_hook_table, hook_table, parameter_table
from glass_skull.config import DEFAULT_MODEL, MODEL_PRESETS, ensure_dirs, normalize_model_name
from glass_skull.model_loader import load_hooked_model, model_summary
from glass_skull.tracer import trace_prompt, top_active_dimensions, next_token_table
from glass_skull.contribution import top_contribution_edges
from glass_skull.steering import build_contrast_vector, generate_normal, generate_steered, vector_summary
from glass_skull.feature_store import save_feature, load_feature, list_features
from glass_skull.logger import log_run, log_observations, log_edges, recent_runs


st.set_page_config(page_title="Operation Glass Skull", layout="wide")
ensure_dirs()

st.title("Operation Glass Skull v0.4")
st.caption("Local transformer activation tracing, feature mapping, and first-pass steering.")

if "trace" not in st.session_state:
    st.session_state.trace = None
if "model_name" not in st.session_state:
    st.session_state.model_name = DEFAULT_MODEL

with st.sidebar:
    st.header("Model")
    preset_options = ["custom"] + MODEL_PRESETS
    preset_index = preset_options.index(st.session_state.model_name) if st.session_state.model_name in preset_options else 0
    preset = st.selectbox("Preset", preset_options, index=preset_index)
    if preset != "custom":
        model_name = preset
    else:
        model_name = st.text_input("TransformerLens model", value=st.session_state.model_name)
    model_name = normalize_model_name(model_name)

    device_choice = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
    if st.button("Load model", type="primary"):
        st.session_state.model_name = model_name
        st.session_state.trace = None
        load_hooked_model.clear()

model = load_hooked_model(model_name, device_choice=device_choice)
summary = model_summary(model)

with st.sidebar:
    st.subheader("Loaded")
    st.write(f"Model: `{summary['model_name']}`")
    st.write(f"Device: `{summary['device']}`")
    st.write(f"Dtype: `{summary['dtype']}`")
    st.write(f"Layers: `{summary['layers']}`")
    st.write(f"d_model: `{summary['d_model']}`")
    st.write(f"Params: `{summary['parameters']:,}`")

st.sidebar.divider()
page = st.sidebar.radio(
    "Mode",
    ["Anatomy", "Trace", "Active Edges", "Map Feature", "Steer", "Logs"],
)


if page == "Anatomy":
    st.header("0. Model anatomy")
    st.write("This view is grounded in the loaded model config, parameter tensors, and TransformerLens hook points.")

    st.subheader("Config")
    st.dataframe(config_table(model), use_container_width=True)

    st.subheader("Global hooks")
    st.dataframe(global_hook_table(model), use_container_width=True)

    st.subheader("Expected block components")
    block_df = expected_block_table(model)
    st.dataframe(block_df, use_container_width=True)

    if not block_df.empty:
        present_counts = block_df.groupby("component")["present"].mean().reset_index()
        present_counts["present_fraction"] = present_counts["present"]
        fig = px.bar(
            present_counts,
            x="component",
            y="present_fraction",
            title="Component hook availability across layers",
            labels={"present_fraction": "fraction present"},
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("All discovered hook points")
    st.dataframe(hook_table(model), use_container_width=True)

    st.subheader("Parameter tensors")
    params = parameter_table(model)
    st.dataframe(params, use_container_width=True)
    if not params.empty:
        st.write(f"Parameter tensors: `{len(params)}`")
        st.write(f"Total parameters from table: `{int(params['parameters'].sum()):,}`")


elif page == "Trace":
    st.header("1. Trace prompt")
    prompt = st.text_area("Prompt", value="The cat sat on the", height=120)
    top_k = st.slider("Top active dimensions", min_value=5, max_value=100, value=30, step=5)

    if st.button("Run trace", type="primary"):
        with st.spinner("Running model and caching activations..."):
            trace = trace_prompt(model, prompt)
            st.session_state.trace = trace
            run_id = log_run(
                model_name=model_name,
                mode="trace",
                prompt=prompt,
                metadata={"tokens": trace.tokens, "summary": summary},
            )
            st.session_state.last_run_id = run_id

    trace = st.session_state.trace
    if trace is not None:
        st.subheader("Tokens")
        st.code(" | ".join([f"{i}:{tok}" for i, tok in enumerate(trace.tokens)]))

        st.subheader("Layer/token activation norm heatmap")
        heat = trace.layer_norms.copy()
        if heat.empty:
            st.warning("No activation rows were captured. Try a different model or hook stream.")
        else:
            heat["layer_stream"] = heat["layer"].astype(str) + ":" + heat["stream"]
            pivot = heat.pivot_table(index="layer_stream", columns="token_index", values="norm", aggfunc="mean")
            fig = px.imshow(
                pivot,
                aspect="auto",
                color_continuous_scale="Viridis",
                labels={"x": "token index", "y": "layer:stream", "color": "norm"},
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Inspect one layer/token")
        c1, c2, c3 = st.columns(3)
        with c1:
            layer = st.number_input("Layer", min_value=0, max_value=summary["layers"] - 1, value=0)
        with c2:
            stream = st.selectbox("Stream", ["resid_pre", "attn_out", "mlp_out", "resid_post"], index=3)
        with c3:
            token_index = st.number_input("Token index", min_value=0, max_value=max(len(trace.tokens) - 1, 0), value=max(len(trace.tokens) - 1, 0))

        dims = top_active_dimensions(trace.cache, int(layer), stream, int(token_index), top_k=top_k)
        st.dataframe(dims, use_container_width=True)

        if st.button("Log displayed top dimensions"):
            run_id = st.session_state.get("last_run_id") or log_run(model_name, "trace_dims", prompt)
            rows = dims.assign(layer=int(layer), stream=stream, token_index=int(token_index), token=trace.tokens[int(token_index)]).to_dict("records")
            log_observations(run_id, rows)
            st.success(f"Logged {len(rows)} dimension rows to run {run_id}")

        st.subheader("Next-token probabilities")
        st.dataframe(next_token_table(model, trace.logits, top_k=20), use_container_width=True)


elif page == "Active Edges":
    st.header("2. Active contribution edges")
    trace = st.session_state.trace
    if trace is None:
        st.warning("Run a trace first. No cached activations are available yet.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            layer = st.number_input("Layer", min_value=0, max_value=summary["layers"] - 1, value=0, key="edges_layer")
        with c2:
            module = st.selectbox("Matrix", ["mlp.W_in", "mlp.W_out"])
        with c3:
            token_index = st.number_input("Token index", min_value=0, max_value=max(len(trace.tokens) - 1, 0), value=max(len(trace.tokens) - 1, 0), key="edges_token")
        with c4:
            top_k = st.slider("Top edges", min_value=5, max_value=200, value=50, step=5)

        try:
            edges = top_contribution_edges(model, trace.cache, int(layer), module, int(token_index), top_k=top_k)
            st.dataframe(edges, use_container_width=True)

            st.subheader("Contribution graph data")
            fig = px.scatter(
                edges,
                x="from_dim",
                y="to_dim",
                size="abs_contribution",
                color="contribution",
                hover_data=["rank", "source_hook", "input_activation", "weight", "contribution"],
                title="Top active contribution edges",
            )
            st.plotly_chart(fig, use_container_width=True)

            if st.button("Log displayed edges"):
                run_id = st.session_state.get("last_run_id") or log_run(model_name, "active_edges", trace.prompt)
                log_edges(run_id, edges.to_dict("records"))
                st.success(f"Logged {len(edges)} edge rows to run {run_id}")
        except Exception as exc:
            st.error(str(exc))


elif page == "Map Feature":
    st.header("3. Build contrast feature vector")
    st.write("Positive prompts minus negative prompts. This makes a coordinate direction in activation space.")

    c1, c2 = st.columns(2)
    with c1:
        positive_text = st.text_area(
            "Positive prompts, one per line",
            value="Oh great, another meeting.\nFantastic, the server broke again.\nWonderful, I get to debug this all night.",
            height=180,
        )
    with c2:
        negative_text = st.text_area(
            "Negative prompts, one per line",
            value="The meeting started.\nThe server stopped responding.\nI need to debug this program.",
            height=180,
        )

    c3, c4, c5 = st.columns(3)
    with c3:
        layer = st.number_input("Layer", min_value=0, max_value=summary["layers"] - 1, value=min(3, summary["layers"] - 1), key="map_layer")
    with c4:
        stream = st.selectbox("Stream", ["resid_pre", "attn_out", "mlp_out", "resid_post"], index=3, key="map_stream")
    with c5:
        feature_name = st.text_input("Feature name", value="sarcasm-ish")

    if st.button("Build and save feature", type="primary"):
        positive = [p.strip() for p in positive_text.splitlines() if p.strip()]
        negative = [p.strip() for p in negative_text.splitlines() if p.strip()]
        with st.spinner("Computing contrast vector..."):
            vec = build_contrast_vector(model, positive, negative, int(layer), stream=stream)
            tensor_path, meta_path = save_feature(
                feature_name,
                vec,
                {
                    "model_name": model_name,
                    "layer": int(layer),
                    "stream": stream,
                    "positive_count": len(positive),
                    "negative_count": len(negative),
                    "positive_prompts": positive,
                    "negative_prompts": negative,
                },
            )
            log_run(model_name, "build_feature", feature_name, metadata={"layer": int(layer), "stream": stream})
        st.success(f"Saved {tensor_path.name} and {meta_path.name}")
        st.dataframe(pd.DataFrame(vector_summary(vec, top_k=40)), use_container_width=True)

    st.subheader("Saved features")
    st.dataframe(pd.DataFrame(list_features()), use_container_width=True)


elif page == "Steer":
    st.header("4. Steering mode")
    features = list_features()
    if not features:
        st.warning("No saved features yet. Build one in Map Feature first.")
    else:
        names = [f["name"] for f in features]
        feature_name = st.selectbox("Feature", names)
        vector, meta = load_feature(feature_name)
        default_layer = int(meta.get("layer", 0))
        default_stream = meta.get("stream", "resid_post")

        c1, c2, c3 = st.columns(3)
        with c1:
            layer = st.number_input("Layer", min_value=0, max_value=summary["layers"] - 1, value=min(default_layer, summary["layers"] - 1), key="steer_layer")
        with c2:
            stream_options = ["resid_pre", "attn_out", "mlp_out", "resid_post"]
            stream = st.selectbox("Stream", stream_options, index=stream_options.index(default_stream) if default_stream in stream_options else 3, key="steer_stream")
        with c3:
            strength = st.slider("Strength", min_value=-5.0, max_value=5.0, value=1.5, step=0.25)

        prompt = st.text_area("Prompt", value="Explain what a mammal is.", height=120, key="steer_prompt")
        max_new_tokens = st.slider("Max new tokens", min_value=10, max_value=200, value=60, step=10)
        temperature = st.slider("Temperature", min_value=0.01, max_value=1.5, value=0.8, step=0.05)

        st.subheader("Feature vector top dimensions")
        st.dataframe(pd.DataFrame(vector_summary(vector, top_k=40)), use_container_width=True)

        if st.button("Generate normal vs steered", type="primary"):
            normal = ""
            steered = ""
            normal_error = None
            steered_error = None
            with st.spinner("Generating normal output..."):
                try:
                    normal = generate_normal(model, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
                except Exception as exc:
                    normal_error = str(exc)
            with st.spinner("Generating steered output..."):
                try:
                    steered = generate_steered(
                        model,
                        prompt,
                        vector,
                        int(layer),
                        stream=stream,
                        strength=float(strength),
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                except Exception as exc:
                    steered_error = str(exc)

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Normal")
                if normal_error:
                    st.error(normal_error)
                else:
                    st.write(normal)
            with c2:
                st.subheader("Steered")
                if steered_error:
                    st.error(steered_error)
                    st.info("If this fails on your TransformerLens version, tracing and feature mapping still work. The generation wrapper will need a local compatibility patch.")
                else:
                    st.write(steered)

            log_run(
                model_name,
                "steer",
                prompt,
                output=steered or normal,
                metadata={
                    "feature": feature_name,
                    "layer": int(layer),
                    "stream": stream,
                    "strength": float(strength),
                    "normal_error": normal_error,
                    "steered_error": steered_error,
                    "normal": normal,
                    "steered": steered,
                },
            )


elif page == "Logs":
    st.header("Logs")
    rows = recent_runs(limit=100)
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
