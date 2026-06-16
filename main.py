from __future__ import annotations

import pandas as pd
import streamlit as st

from glass_skull.anatomy import config_table, expected_block_table, global_hook_table, hook_table, parameter_table
from glass_skull.config import DEFAULT_MODEL, MODEL_PRESETS, ensure_dirs, normalize_model_name
from glass_skull.model_loader import load_hooked_model, model_summary
from glass_skull.tracer import next_token_table, top_active_dimensions, trace_prompt
from glass_skull.contribution import top_contribution_edges
from glass_skull.steering import build_contrast_vector, generate_normal, generate_steered, vector_summary
from glass_skull.feature_store import list_features, load_feature, save_feature
from glass_skull.logger import log_edges, log_observations, log_run, recent_runs
from glass_skull.llama_client import chat_completion, check_server
from glass_skull.prompt_loader import load_prompt_file_bytes
from glass_skull.fuzzing import run_fuzz_experiment
from glass_skull.experiment_store import list_experiments
from glass_skull.visuals import (
    activation_heatmap,
    activation_pulse,
    dim_frequency_fig,
    edge_constellation,
    fuzz_label_layer_fig,
    fuzz_prompt_layer_fig,
)


st.set_page_config(page_title="Operation Glass Skull", layout="wide")
ensure_dirs()

st.title("Operation Glass Skull v0.6")
st.caption("Chat with llama.cpp or TransformerLens, trace activations, fuzz prompt files, and build maps worth staring at.")

HELP = {
    "preset": "Pick the local TransformerLens model used for tracing, feature mapping, and optional steering.",
    "device": "CPU always works. CUDA only works if PyTorch can see an NVIDIA GPU. AMD Vulkan belongs to llama.cpp, not PyTorch.",
    "temperature": "Higher means more random. Lower means more predictable.",
    "max_new_tokens": "How many new tokens the model is allowed to write after your prompt.",
    "stream": "A checkpoint inside each transformer block. resid_post is the block's final working state.",
    "layer": "Which transformer block to inspect. Early layers catch surface patterns. Later layers tend to be more abstract.",
    "token": "Which prompt token to inspect. The last token is usually best for next-token prediction.",
    "top_dims": "Shows the strongest hidden dimensions for the selected layer and token.",
    "edges": "Shows strongest current contribution paths through a selected MLP matrix. Computed from real activations and weights.",
    "feature": "A saved direction in activation space, usually made by subtracting one prompt group from another.",
    "strength": "How hard to push the selected feature direction during generation. Negative values suppress it.",
    "positive": "Examples that contain the behavior or concept you want to map.",
    "negative": "Plain or opposite examples used as the comparison baseline.",
    "llama_url": "Base URL for a running llama.cpp server. Example: http://127.0.0.1:8080",
    "backend": "Where chat responses come from. Tracing still uses TransformerLens unless a patched llama.cpp trace endpoint exists later.",
    "fuzz": "Rapid-fire a file of prompts through a backend, optionally trace them, and build heatmaps across prompts, labels, layers, and dimensions.",
}


def init_state() -> None:
    defaults = {
        "trace": None,
        "model_name": DEFAULT_MODEL,
        "chat_messages": [],
        "last_output": "",
        "last_run_id": None,
        "llama_url": "http://127.0.0.1:8080",
        "llama_glass_url": "http://127.0.0.1:8088",
        "llama_status": None,
        "llama_glass_status": None,
        "last_fuzz_result": None,
        "selected_feature": None,
        "poke_layer": 0,
        "poke_stream": "resid_post",
        "poke_strength": 1.5,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def parse_layer_list(raw: str, max_layer: int) -> list[int]:
    raw = raw.strip().lower()
    if not raw or raw == "all":
        return list(range(max_layer + 1))

    layers: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s.strip())
            end = int(end_s.strip())
            if end < start:
                start, end = end, start
            for layer in range(start, end + 1):
                if 0 <= layer <= max_layer:
                    layers.add(layer)
        else:
            layer = int(part)
            if 0 <= layer <= max_layer:
                layers.add(layer)
    return sorted(layers)


def render_status(label: str, status) -> None:
    if status is None:
        st.caption(f"{label}: unchecked")
        return
    if status.online:
        glass = "yes" if status.glass_available else "no"
        model_text = ", ".join(status.models) if status.models else "unknown"
        latency = f"{status.latency_ms:.0f} ms" if status.latency_ms is not None else "unknown latency"
        st.success(f"{label}: online, {latency}, glass: {glass}")
        st.caption(f"Model: {model_text}")
    else:
        st.error(f"{label}: offline")
        st.caption(status.error or "no details")


def plot_if_present(fig) -> None:
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)


init_state()

with st.sidebar:
    st.header("Trace model")
    preset_options = ["custom"] + MODEL_PRESETS
    preset_index = preset_options.index(st.session_state.model_name) if st.session_state.model_name in preset_options else 0
    preset = st.selectbox("Preset", preset_options, index=preset_index, help=HELP["preset"])
    if preset != "custom":
        model_name = preset
    else:
        model_name = st.text_input("TransformerLens model", value=st.session_state.model_name, help="Use a TransformerLens-supported model name.")
    model_name = normalize_model_name(model_name)

    device_choice = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0, help=HELP["device"])
    if st.button("Load trace model", type="primary", help="Reloads the selected TransformerLens model and clears the current trace."):
        st.session_state.model_name = model_name
        st.session_state.trace = None
        st.session_state.last_output = ""
        load_hooked_model.clear()

model = load_hooked_model(model_name, device_choice=device_choice)
summary = model_summary(model)

with st.sidebar:
    st.subheader("Loaded trace model")
    st.write(f"Model: `{summary['model_name']}`")
    st.write(f"Device: `{summary['device']}`")
    st.write(f"Layers: `{summary['layers']}`")
    st.write(f"d_model: `{summary['d_model']}`")
    st.write(f"Params: `{summary['parameters']:,}`")

    st.divider()
    st.header("llama.cpp")
    st.session_state.llama_url = st.text_input("Normal server URL", value=st.session_state.llama_url, help=HELP["llama_url"])
    st.session_state.llama_glass_url = st.text_input("Glass server URL", value=st.session_state.llama_glass_url, help="Future patched llama.cpp lab server. Use a nonstandard port like 8088.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Check normal", help="Checks /v1/models and optional /glass-skull/info."):
            st.session_state.llama_status = check_server(st.session_state.llama_url)
    with c2:
        if st.button("Check glass", help="Checks the lab llama.cpp server on its separate port."):
            st.session_state.llama_glass_status = check_server(st.session_state.llama_glass_url)
    render_status("Normal", st.session_state.llama_status)
    render_status("Glass", st.session_state.llama_glass_status)

    st.divider()
    if st.button("Clear chat", help="Clears only visible chat history. Logs stay saved."):
        st.session_state.chat_messages = []
        st.session_state.last_output = ""
    if st.button("Clear trace", help="Clears the currently cached activations."):
        st.session_state.trace = None

features = list_features()
feature_names = [f["name"] for f in features]

chat_col, trace_col, poke_col, anatomy_col = st.columns([1, 1, 1, 1], gap="large")

with chat_col:
    st.subheader("Chat")
    st.caption("Use llama.cpp for real chat, while Glass Skull traces locally or later through a patched server.")

    chat_backend = st.selectbox(
        "Chat backend",
        ["TransformerLens", "llama.cpp normal", "llama.cpp glass"],
        help=HELP["backend"],
    )
    max_new_tokens = st.slider("Max new tokens", 10, 300, 80, 10, help=HELP["max_new_tokens"], key="chat_max_new")
    temperature = st.slider("Temperature", 0.01, 1.5, 0.8, 0.05, help=HELP["temperature"], key="chat_temp")
    auto_trace = st.toggle("Trace every message", value=True, help="When enabled, every message also captures activations with the trace model.")
    use_steering = st.toggle("Use steering", value=False, help="Only applies when the chat backend is TransformerLens. Stock llama.cpp cannot be activation-steered yet.")

    chat_box = st.container(height=420, border=True)
    with chat_box:
        if not st.session_state.chat_messages:
            st.info("Send a message to start. llama.cpp can answer; TransformerLens can dissect. Finally, a division of labor that makes sense.")
        for msg in st.session_state.chat_messages[-12:]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    prompt = st.text_area("Message", value="The cat sat on the", height=110, help="The text sent to the selected chat backend.", key="chat_prompt")
    send = st.button("Send", type="primary", use_container_width=True, help="Generate a reply and optionally trace the prompt.")

    if send and prompt.strip():
        prompt = prompt.strip()
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        if auto_trace:
            with st.spinner("Tracing prompt locally..."):
                trace = trace_prompt(model, prompt)
                st.session_state.trace = trace
                run_id = log_run(model_name=model_name, mode="chat_trace", prompt=prompt, metadata={"tokens": trace.tokens, "summary": summary})
                st.session_state.last_run_id = run_id

        output = ""
        error = None
        with st.spinner("Generating reply..."):
            try:
                if chat_backend == "llama.cpp normal":
                    output = chat_completion(st.session_state.llama_url, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
                elif chat_backend == "llama.cpp glass":
                    output = chat_completion(st.session_state.llama_glass_url, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
                elif use_steering and feature_names:
                    selected_feature = st.session_state.get("selected_feature") or feature_names[0]
                    vector, meta = load_feature(selected_feature)
                    layer = int(st.session_state.get("poke_layer", int(meta.get("layer", 0))))
                    stream = st.session_state.get("poke_stream", meta.get("stream", "resid_post"))
                    strength = float(st.session_state.get("poke_strength", 1.5))
                    output = generate_steered(model, prompt, vector, layer, stream=stream, strength=strength, max_new_tokens=max_new_tokens, temperature=temperature)
                else:
                    output = generate_normal(model, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
            except Exception as exc:
                error = str(exc)
                output = f"Generation error: {error}"

        st.session_state.last_output = output
        st.session_state.chat_messages.append({"role": "assistant", "content": output})
        log_run(model_name=model_name, mode="chat_generate", prompt=prompt, output=output, metadata={"backend": chat_backend, "used_steering": bool(use_steering), "error": error})
        st.rerun()

with trace_col:
    st.subheader("Trace")
    st.caption("Latest traced prompt, shown as heatmap plus pulse view.")

    trace = st.session_state.trace
    if trace is None:
        st.info("No trace yet. Send a chat message with tracing enabled.")
    else:
        st.markdown("**Tokens**")
        st.code(" | ".join([f"{i}:{tok}" for i, tok in enumerate(trace.tokens)]))

        visual_mode = st.radio("Visual", ["Pulse", "Heatmap"], horizontal=True, help="Pulse is prettier. Heatmap is denser. Humanity demands both.")
        fig = activation_pulse(trace.layer_norms) if visual_mode == "Pulse" else activation_heatmap(trace.layer_norms)
        if fig is None:
            st.warning("No activation rows were captured for this trace.")
        else:
            plot_if_present(fig)

        st.markdown("**Inspect a point**")
        layer = st.number_input("Layer", 0, summary["layers"] - 1, 0, help=HELP["layer"], key="trace_layer")
        stream = st.selectbox("Stream", ["resid_pre", "attn_out", "mlp_out", "resid_post"], index=3, help=HELP["stream"], key="trace_stream")
        token_index = st.number_input("Token", 0, max(len(trace.tokens) - 1, 0), max(len(trace.tokens) - 1, 0), help=HELP["token"], key="trace_token")
        top_k = st.slider("Top dims", 5, 100, 30, 5, help=HELP["top_dims"], key="trace_top_dims")

        try:
            dims = top_active_dimensions(trace.cache, int(layer), stream, int(token_index), top_k=top_k)
            st.dataframe(dims, use_container_width=True, height=220)
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
    st.subheader("Poke / Fuzz")
    st.caption("Map features, inspect active edges, or rapid-fire prompt files.")

    tab_steer, tab_map, tab_edges, tab_fuzz = st.tabs(["Steer", "Map", "Edges", "Fuzz"])

    with tab_steer:
        if not feature_names:
            st.info("No saved features yet. Use the Map tab to make one.")
        else:
            selected_feature = st.selectbox("Feature", feature_names, help=HELP["feature"], key="selected_feature")
            vector, meta = load_feature(selected_feature)
            default_layer = int(meta.get("layer", 0))
            default_stream = meta.get("stream", "resid_post")
            stream_options = ["resid_pre", "attn_out", "mlp_out", "resid_post"]
            st.session_state.poke_layer = st.number_input("Layer", 0, summary["layers"] - 1, min(default_layer, summary["layers"] - 1), help=HELP["layer"], key="poke_layer_widget")
            st.session_state.poke_stream = st.selectbox("Stream", stream_options, index=stream_options.index(default_stream) if default_stream in stream_options else 3, help=HELP["stream"], key="poke_stream_widget")
            st.session_state.poke_strength = st.slider("Strength", -5.0, 5.0, 1.5, 0.25, help=HELP["strength"], key="poke_strength_widget")
            st.dataframe(pd.DataFrame(vector_summary(vector, top_k=25)), use_container_width=True, height=260)
            st.caption("Turn on 'Use steering' in Chat to apply this feature to TransformerLens replies.")

    with tab_map:
        positive_text = st.text_area("Positive examples", value="Oh great, another meeting.\nFantastic, the server broke again.\nWonderful, I get to debug this all night.", height=120, help=HELP["positive"], key="map_positive")
        negative_text = st.text_area("Negative examples", value="The meeting started.\nThe server stopped responding.\nI need to debug this program.", height=120, help=HELP["negative"], key="map_negative")
        map_layer = st.number_input("Map layer", 0, summary["layers"] - 1, min(3, summary["layers"] - 1), help=HELP["layer"], key="map_layer")
        map_stream = st.selectbox("Map stream", ["resid_pre", "attn_out", "mlp_out", "resid_post"], index=3, help=HELP["stream"], key="map_stream")
        feature_name = st.text_input("Feature name", value="sarcasm-ish", help="Name for the saved activation direction.")
        if st.button("Build feature", use_container_width=True, help="Average positive examples, subtract negative examples, and save the result."):
            positive = [p.strip() for p in positive_text.splitlines() if p.strip()]
            negative = [p.strip() for p in negative_text.splitlines() if p.strip()]
            try:
                with st.spinner("Computing feature vector..."):
                    vec = build_contrast_vector(model, positive, negative, int(map_layer), stream=map_stream)
                    tensor_path, meta_path = save_feature(feature_name, vec, {"model_name": model_name, "layer": int(map_layer), "stream": map_stream, "positive_count": len(positive), "negative_count": len(negative), "positive_prompts": positive, "negative_prompts": negative})
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
                plot_if_present(edge_constellation(edges))
                st.dataframe(edges, use_container_width=True, height=220)
                if st.button("Log edges", help="Save the displayed edge rows to SQLite."):
                    run_id = st.session_state.get("last_run_id") or log_run(model_name, "active_edges", trace.prompt)
                    log_edges(run_id, edges.to_dict("records"))
                    st.success(f"Logged {len(edges)} rows")
            except Exception as exc:
                st.error(str(exc))

    with tab_fuzz:
        st.caption(HELP["fuzz"])
        fuzz_name = st.text_input("Experiment name", value="glass_probe", help="Used to name the saved folder under data/experiments.")
        uploaded = st.file_uploader("Prompt file", type=["txt", "jsonl", "csv"], help="TXT: one prompt per line. JSONL/CSV: use prompt and optional label fields.")
        fuzz_backend = st.selectbox("Fuzz chat backend", ["TransformerLens", "llama.cpp normal", "llama.cpp glass"], help="Where generated outputs come from.")
        trace_fuzz = st.toggle("Trace fuzz prompts", value=True, help="Captures activation summaries with the TransformerLens trace model for each prompt.")
        layer_raw = st.text_input("Layers", value="all", help="Use 'all', comma lists like 0,4,8, or ranges like 0-8.")
        streams = st.multiselect("Streams", ["resid_pre", "attn_out", "mlp_out", "resid_post"], default=["resid_post"], help="Which activation streams to aggregate.")
        fuzz_top_k = st.slider("Top dims per point", 5, 100, 32, 1, help="How many active dimensions to store per selected layer/stream.")
        fuzz_limit = st.slider("Prompt limit", 1, 1000, 200, 1, help="Caps the number of prompts from the file for this run.")

        if uploaded is not None:
            try:
                prompt_items = load_prompt_file_bytes(uploaded.name, uploaded.getvalue())[:fuzz_limit]
                st.write(f"Loaded `{len(prompt_items)}` prompts")
                st.dataframe(pd.DataFrame([{"id": p.prompt_id, "label": p.label, "prompt": p.prompt} for p in prompt_items[:20]]), use_container_width=True, height=160)
            except Exception as exc:
                prompt_items = []
                st.error(str(exc))
        else:
            prompt_items = []

        if st.button("Run fuzz", type="primary", use_container_width=True, disabled=not prompt_items):
            try:
                layers = parse_layer_list(layer_raw, summary["layers"] - 1)
                if not layers:
                    raise ValueError("No valid layers selected. Use 'all', a comma list like 0,4,8, or a range like 0-8.")
            except Exception as exc:
                st.error(f"Invalid layer list: {exc}")
            else:
                backend_key = "transformerlens" if fuzz_backend == "TransformerLens" else "llama.cpp"
                llama_url = st.session_state.llama_url if fuzz_backend == "llama.cpp normal" else st.session_state.llama_glass_url
                progress = st.progress(0)
                status = st.empty()

                def cb(i: int, total: int, current_prompt: str) -> None:
                    progress.progress(i / max(total, 1))
                    status.caption(f"{i}/{total}: {current_prompt[:120]}")

                try:
                    with st.spinner("Running fuzz experiment..."):
                        result = run_fuzz_experiment(
                            name=fuzz_name,
                            prompts=prompt_items,
                            chat_backend=backend_key,
                            trace_enabled=trace_fuzz,
                            model=model,
                            llama_url=llama_url,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            layers=layers,
                            streams=streams or ["resid_post"],
                            top_k=fuzz_top_k,
                            progress_callback=cb,
                        )
                        st.session_state.last_fuzz_result = result
                    st.success(f"Saved experiment: {result['summary']['experiment_path']}")
                except Exception as exc:
                    st.error(str(exc))

        result = st.session_state.last_fuzz_result
        if result:
            st.markdown("**Last fuzz visuals**")
            plot_if_present(fuzz_prompt_layer_fig(result.get("prompt_layer_df", pd.DataFrame())))
            plot_if_present(fuzz_label_layer_fig(result.get("label_layer_df", pd.DataFrame())))
            plot_if_present(dim_frequency_fig(result.get("top_dims_df", pd.DataFrame())))

with anatomy_col:
    st.subheader("Anatomy / Logs")
    st.caption("Ground truth about the loaded trace model, backend status, and saved experiments.")

    panel = st.radio("Panel", ["Anatomy", "Hooks", "Parameters", "Experiments", "Logs"], horizontal=True, help="Switch this quarter of the screen between model structure and saved run history.")

    if panel == "Anatomy":
        st.markdown("**Config**")
        st.dataframe(config_table(model), use_container_width=True, height=230)
        st.markdown("**Block components**")
        st.dataframe(expected_block_table(model), use_container_width=True, height=330)
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
    elif panel == "Experiments":
        st.dataframe(pd.DataFrame(list_experiments()), use_container_width=True, height=590)
    elif panel == "Logs":
        st.dataframe(pd.DataFrame(recent_runs(limit=100)), use_container_width=True, height=590)
