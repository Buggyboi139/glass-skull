from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from glass_skull.anatomy import config_table, expected_block_table, global_hook_table, hook_table, parameter_table
from glass_skull.config import DEFAULT_MODEL, MODEL_PRESETS, ensure_dirs, normalize_model_name
from glass_skull.model_loader import load_hooked_model, model_summary
from glass_skull.tracer import next_token_table, top_active_dimensions, trace_prompt
from glass_skull.contribution import top_contribution_edges
from glass_skull.steering import build_contrast_vector, generate_normal, generate_steered, vector_summary
from glass_skull.feature_store import list_features, load_feature, save_feature
from glass_skull.logger import log_edges, log_observations, log_run, recent_runs


st.set_page_config(page_title="Operation Glass Skull", layout="wide")
ensure_dirs()

st.title("Operation Glass Skull v0.5")
st.caption("Chat with a small transformer while watching, mapping, and poking its activations.")

HELP = {
    "preset": "Pick a small model that TransformerLens knows how to open. Start small before making your CPU suffer for art.",
    "device": "CPU always works. CUDA only works if your PyTorch install can see an NVIDIA GPU.",
    "temperature": "Higher means more random. Lower means more predictable.",
    "max_new_tokens": "How many new tokens the model is allowed to write after your prompt.",
    "trace": "Runs the prompt through the model and saves the internal activation values.",
    "stream": "A checkpoint inside each transformer block. resid_post is the block's final working state.",
    "layer": "Which transformer block to inspect. Early layers catch surface patterns. Later layers tend to be more abstract.",
    "token": "Which prompt token to inspect. The last token is usually best for next-token prediction.",
    "top_dims": "Shows the strongest hidden dimensions for the selected layer and token.",
    "edges": "Shows the strongest current contribution paths through a selected MLP matrix. These are computed from real activations and real weights.",
    "feature": "A saved direction in activation space, usually made by subtracting one prompt group from another.",
    "strength": "How hard to push the selected feature direction during generation. Negative values suppress it.",
    "positive": "Examples that contain the behavior or concept you want to map.",
    "negative": "Plain or opposite examples used as the comparison baseline.",
}

if "trace" not in st.session_state:
    st.session_state.trace = None
if "model_name" not in st.session_state:
    st.session_state.model_name = DEFAULT_MODEL
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "last_output" not in st.session_state:
    st.session_state.last_output = ""
if "last_run_id" not in st.session_state:
    st.session_state.last_run_id = None

with st.sidebar:
    st.header("Model")
    preset_options = ["custom"] + MODEL_PRESETS
    preset_index = preset_options.index(st.session_state.model_name) if st.session_state.model_name in preset_options else 0
    preset = st.selectbox("Preset", preset_options, index=preset_index, help=HELP["preset"])
    if preset != "custom":
        model_name = preset
    else:
        model_name = st.text_input("TransformerLens model", value=st.session_state.model_name, help="Use a TransformerLens-supported model name.")
    model_name = normalize_model_name(model_name)

    device_choice = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0, help=HELP["device"])
    if st.button("Load model", type="primary", help="Reloads the selected model and clears the current trace."):
        st.session_state.model_name = model_name
        st.session_state.trace = None
        st.session_state.last_output = ""
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
    st.divider()
    if st.button("Clear chat", help="Clears only the visible chat history. It does not delete saved logs."):
        st.session_state.chat_messages = []
        st.session_state.last_output = ""
    if st.button("Clear trace", help="Clears the currently cached activations."):
        st.session_state.trace = None

features = list_features()
feature_names = [f["name"] for f in features]

chat_col, trace_col, poke_col, anatomy_col = st.columns([1, 1, 1, 1], gap="large")

with chat_col:
    st.subheader("Chat")
    st.caption("Talk to the model here. Each send can also trace the model's internals.")

    max_new_tokens = st.slider("Max new tokens", 10, 200, 60, 10, help=HELP["max_new_tokens"], key="chat_max_new")
    temperature = st.slider("Temperature", 0.01, 1.5, 0.8, 0.05, help=HELP["temperature"], key="chat_temp")
    auto_trace = st.toggle("Trace every message", value=True, help="When enabled, every message also captures activations for the trace panel.")
    use_steering = st.toggle("Use steering", value=False, help="When enabled, the response is generated with the selected feature vector from the Poke panel.")

    chat_box = st.container(height=420, border=True)
    with chat_box:
        if not st.session_state.chat_messages:
            st.info("Send a message to start. The tiny model will not be brilliant. It is here for dissection, not poetry.")
        for msg in st.session_state.chat_messages[-12:]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    prompt = st.text_area(
        "Message",
        value="The cat sat on the",
        height=110,
        help="This text is sent to the model. If tracing is on, this exact text is also used for the activation trace.",
        key="chat_prompt",
    )

    send = st.button("Send", type="primary", use_container_width=True, help="Generate a reply and optionally trace the prompt.")

    if send and prompt.strip():
        prompt = prompt.strip()
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        if auto_trace:
            with st.spinner("Tracing prompt..."):
                trace = trace_prompt(model, prompt)
                st.session_state.trace = trace
                run_id = log_run(
                    model_name=model_name,
                    mode="chat_trace",
                    prompt=prompt,
                    metadata={"tokens": trace.tokens, "summary": summary},
                )
                st.session_state.last_run_id = run_id

        output = ""
        error = None
        with st.spinner("Generating reply..."):
            try:
                if use_steering and feature_names:
                    selected_feature = st.session_state.get("selected_feature", feature_names[0])
                    vector, meta = load_feature(selected_feature)
                    layer = int(st.session_state.get("poke_layer", int(meta.get("layer", 0))))
                    stream = st.session_state.get("poke_stream", meta.get("stream", "resid_post"))
                    strength = float(st.session_state.get("poke_strength", 1.5))
                    output = generate_steered(
                        model,
                        prompt,
                        vector,
                        layer,
                        stream=stream,
                        strength=strength,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                else:
                    output = generate_normal(model, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
            except Exception as exc:
                error = str(exc)
                output = f"Generation error: {error}"

        st.session_state.last_output = output
        st.session_state.chat_messages.append({"role": "assistant", "content": output})
        log_run(
            model_name=model_name,
            mode="chat_generate",
            prompt=prompt,
            output=output,
            metadata={"used_steering": bool(use_steering), "error": error},
        )
        st.rerun()

with trace_col:
    st.subheader("Trace")
    st.caption("This panel shows what lit up during the latest traced prompt.")

    trace = st.session_state.trace
    if trace is None:
        st.info("No trace yet. Send a chat message with tracing enabled.")
    else:
        st.markdown("**Tokens**")
        st.code(" | ".join([f"{i}:{tok}" for i, tok in enumerate(trace.tokens)]))

        heat = trace.layer_norms.copy()
        if heat.empty:
            st.warning("No activation rows were captured for this trace.")
        else:
            heat["layer_stream"] = heat["layer"].astype(str) + ":" + heat["stream"]
            pivot = heat.pivot_table(index="layer_stream", columns="token_index", values="norm", aggfunc="mean")
            fig = px.imshow(
                pivot,
                aspect="auto",
                color_continuous_scale="Viridis",
                labels={"x": "token", "y": "layer:stream", "color": "strength"},
                title="Activation strength",
            )
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Inspect a point**")
        layer = st.number_input("Layer", 0, summary["layers"] - 1, 0, help=HELP["layer"], key="trace_layer")
        stream = st.selectbox("Stream", ["resid_pre", "attn_out", "mlp_out", "resid_post"], index=3, help=HELP["stream"], key="trace_stream")
        token_index = st.number_input("Token", 0, max(len(trace.tokens) - 1, 0), max(len(trace.tokens) - 1, 0), help=HELP["token"], key="trace_token")
        top_k = st.slider("Top dims", 5, 100, 30, 5, help=HELP["top_dims"], key="trace_top_dims")

        try:
            dims = top_active_dimensions(trace.cache, int(layer), stream, int(token_index), top_k=top_k)
            st.dataframe(dims, use_container_width=True, height=260)
            if st.button("Log dims", help="Save the displayed activation dimensions to SQLite."):
                run_id = st.session_state.get("last_run_id") or log_run(model_name, "trace_dims", trace.prompt)
                rows = dims.assign(layer=int(layer), stream=stream, token_index=int(token_index), token=trace.tokens[int(token_index)]).to_dict("records")
                log_observations(run_id, rows)
                st.success(f"Logged {len(rows)} rows")
        except Exception as exc:
            st.error(str(exc))

        with st.expander("Next-token probabilities", expanded=False):
            st.dataframe(next_token_table(model, trace.logits, top_k=20), use_container_width=True)

with poke_col:
    st.subheader("Poke")
    st.caption("Build or load a feature vector, then push the model while it writes.")

    tab_steer, tab_map, tab_edges = st.tabs(["Steer", "Map", "Edges"])

    with tab_steer:
        if not feature_names:
            st.info("No saved features yet. Use the Map tab to make one.")
        else:
            selected_feature = st.selectbox("Feature", feature_names, help=HELP["feature"], key="selected_feature")
            vector, meta = load_feature(selected_feature)
            default_layer = int(meta.get("layer", 0))
            default_stream = meta.get("stream", "resid_post")
            stream_options = ["resid_pre", "attn_out", "mlp_out", "resid_post"]

            st.session_state.poke_layer = st.number_input(
                "Layer",
                0,
                summary["layers"] - 1,
                min(default_layer, summary["layers"] - 1),
                help=HELP["layer"],
                key="poke_layer_widget",
            )
            st.session_state.poke_stream = st.selectbox(
                "Stream",
                stream_options,
                index=stream_options.index(default_stream) if default_stream in stream_options else 3,
                help=HELP["stream"],
                key="poke_stream_widget",
            )
            st.session_state.poke_strength = st.slider(
                "Strength",
                -5.0,
                5.0,
                1.5,
                0.25,
                help=HELP["strength"],
                key="poke_strength_widget",
            )

            st.markdown("**Feature shape**")
            st.dataframe(pd.DataFrame(vector_summary(vector, top_k=25)), use_container_width=True, height=260)
            st.caption("Turn on 'Use steering' in Chat to apply this feature to replies.")

    with tab_map:
        positive_text = st.text_area(
            "Positive examples",
            value="Oh great, another meeting.\nFantastic, the server broke again.\nWonderful, I get to debug this all night.",
            height=130,
            help=HELP["positive"],
            key="map_positive",
        )
        negative_text = st.text_area(
            "Negative examples",
            value="The meeting started.\nThe server stopped responding.\nI need to debug this program.",
            height=130,
            help=HELP["negative"],
            key="map_negative",
        )
        map_layer = st.number_input("Map layer", 0, summary["layers"] - 1, min(3, summary["layers"] - 1), help=HELP["layer"], key="map_layer")
        map_stream = st.selectbox("Map stream", ["resid_pre", "attn_out", "mlp_out", "resid_post"], index=3, help=HELP["stream"], key="map_stream")
        feature_name = st.text_input("Feature name", value="sarcasm-ish", help="Name for the saved activation direction.")

        if st.button("Build feature", use_container_width=True, help="Average positive examples, subtract negative examples, and save the result."):
            positive = [p.strip() for p in positive_text.splitlines() if p.strip()]
            negative = [p.strip() for p in negative_text.splitlines() if p.strip()]
            try:
                with st.spinner("Computing feature vector..."):
                    vec = build_contrast_vector(model, positive, negative, int(map_layer), stream=map_stream)
                    tensor_path, meta_path = save_feature(
                        feature_name,
                        vec,
                        {
                            "model_name": model_name,
                            "layer": int(map_layer),
                            "stream": map_stream,
                            "positive_count": len(positive),
                            "negative_count": len(negative),
                            "positive_prompts": positive,
                            "negative_prompts": negative,
                        },
                    )
                    log_run(model_name, "build_feature", feature_name, metadata={"layer": int(map_layer), "stream": map_stream})
                st.success(f"Saved {tensor_path.name}")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    with tab_edges:
        trace = st.session_state.trace
        if trace is None:
            st.info("Trace a message first, then active edges can be computed from that trace.")
        else:
            edge_layer = st.number_input("Edge layer", 0, summary["layers"] - 1, 0, help=HELP["layer"], key="edge_layer")
            module = st.selectbox("Matrix", ["mlp.W_in", "mlp.W_out"], help=HELP["edges"], key="edge_module")
            edge_token = st.number_input("Edge token", 0, max(len(trace.tokens) - 1, 0), max(len(trace.tokens) - 1, 0), help=HELP["token"], key="edge_token")
            edge_k = st.slider("Top edges", 5, 200, 50, 5, help="How many strongest contribution paths to show.", key="edge_k")
            try:
                edges = top_contribution_edges(model, trace.cache, int(edge_layer), module, int(edge_token), top_k=edge_k)
                fig = px.scatter(
                    edges,
                    x="from_dim",
                    y="to_dim",
                    size="abs_contribution",
                    color="contribution",
                    hover_data=["rank", "source_hook", "input_activation", "weight", "contribution"],
                    title="Active contribution edges",
                )
                fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(edges, use_container_width=True, height=220)
                if st.button("Log edges", help="Save the displayed edge rows to SQLite."):
                    run_id = st.session_state.get("last_run_id") or log_run(model_name, "active_edges", trace.prompt)
                    log_edges(run_id, edges.to_dict("records"))
                    st.success(f"Logged {len(edges)} rows")
            except Exception as exc:
                st.error(str(exc))

with anatomy_col:
    st.subheader("Anatomy / Logs")
    st.caption("Ground truth about the loaded model and recent experiments.")

    panel = st.radio("Panel", ["Anatomy", "Hooks", "Parameters", "Logs"], horizontal=True, help="Switch this quarter of the screen between model structure and saved run history.")

    if panel == "Anatomy":
        st.markdown("**Config**")
        st.dataframe(config_table(model), use_container_width=True, height=230)
        st.markdown("**Block components**")
        block_df = expected_block_table(model)
        st.dataframe(block_df, use_container_width=True, height=330)
    elif panel == "Hooks":
        st.markdown("**Global hooks**")
        st.dataframe(global_hook_table(model), use_container_width=True, height=180)
        st.markdown("**All hook points**")
        st.dataframe(hook_table(model), use_container_width=True, height=430)
    elif panel == "Parameters":
        params = parameter_table(model)
        st.write(f"Tensor count: `{len(params)}`")
        if not params.empty:
            st.write(f"Total parameters: `{int(params['parameters'].sum()):,}`")
        st.dataframe(params, use_container_width=True, height=560)
    elif panel == "Logs":
        rows = recent_runs(limit=100)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=590)
