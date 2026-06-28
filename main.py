from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from glass_skull import ui_theme as ui
from glass_skull.activation_map import build_activation_map_payload
from glass_skull.activation_map_view import render_activation_map
from glass_skull.behavior_profiles import get_behavior_profile, list_behavior_profiles
from glass_skull.behavior_scoring import behavior_timeline_df, score_run_artifact
from glass_skull.chat_store import list_chats, load_chat, save_chat
from glass_skull.config import ensure_dirs
from glass_skull.experiment_store import list_experiments
from glass_skull.fuzzing import run_fuzz_experiment
from glass_skull.llama_client import (
    build_steering_metadata,
    chat_completion,
    check_server,
    per_request_steering_supported,
    trace_glass_prompt,
)
from glass_skull.llama_control import (
    DEFAULT_CVECTOR_GENERATOR,
    DEFAULT_LLAMA_SERVER,
    ControlVectorRunError,
    build_cvector_command,
    build_llama_server_command,
    generate_control_vector,
    list_control_sets,
    list_control_vectors,
    preflight_control_vector_run,
    shell_join,
    write_control_set,
)
from glass_skull.logger import log_run, recent_runs
from glass_skull.model_context import local_gguf_context
from glass_skull.path_mapping import rank_activation_paths, recommended_steering_targets
from glass_skull.prompt_loader import load_prompt_file_bytes
from glass_skull.run_artifacts import (
    activation_path_df,
    batch_heatmap_df,
    build_run_artifact,
    dimension_frequency_df,
    label_heatmap_df,
    normalize_llama_trace,
    trace_unavailable_row,
)
from glass_skull.ui_local import LOCAL_TABS, batch_items_from_inputs, dashboard_context, new_run_id
from glass_skull.visuals import (
    activation_path_graph,
    batch_activation_heatmap,
    behavior_delta_bar_fig,
    behavior_score_timeline_fig,
    dim_frequency_fig,
    gguf_tensor_shape_scatter_fig,
    label_activation_heatmap,
    path_rank_bar_fig,
)


st.set_page_config(page_title="Operation Glass Skull", layout="wide", initial_sidebar_state="collapsed")
ensure_dirs()
ui.inject_theme()

CHAT_BACKEND_NORMAL = "Local GGUF normal (llama.cpp)"
CHAT_BACKEND_STEERED = "Local GGUF steered (llama.cpp)"
CHAT_BACKENDS = [CHAT_BACKEND_NORMAL, CHAT_BACKEND_STEERED]


def normalize_chat_backend(label: str) -> str:
    if label in {"llama.cpp normal", CHAT_BACKEND_NORMAL}:
        return "llama.cpp normal"
    if label in {"llama.cpp glass", CHAT_BACKEND_STEERED}:
        return "llama.cpp glass"
    return "llama.cpp normal"


def chat_backend_display(canonical: str) -> str:
    return CHAT_BACKEND_STEERED if canonical == "llama.cpp glass" else CHAT_BACKEND_NORMAL


def init_state() -> None:
    defaults = {
        "chat_messages": [],
        "chat_backend_label": CHAT_BACKEND_NORMAL,
        "last_output": "",
        "last_run_id": None,
        "active_run_id": None,
        "active_run_mode": "Single message",
        "llama_url": "http://127.0.0.1:8080",
        "llama_glass_url": "http://127.0.0.1:8088",
        "llama_status": None,
        "llama_glass_status": None,
        "llama_model_alias": "local",
        "llama_model_path": "",
        "llama_cvector_generator": str(DEFAULT_CVECTOR_GENERATOR),
        "llama_server_bin": str(DEFAULT_LLAMA_SERVER),
        "llama_control_set": "",
        "llama_control_vector": "",
        "llama_control_strength": 1.25,
        "llama_control_layer_start": 1,
        "llama_control_layer_end": 32,
        "llama_control_port": 8088,
        "llama_control_extra_args": "--jinja --flash-attn auto",
        "llama_last_preflight": None,
        "llama_last_cvector_failure": None,
        "local_dashboard_trace": None,
        "local_dashboard_trace_meta": {},
        "local_dashboard_trace_counter": 0,
        "last_batch_result": None,
        "last_fuzz_result": None,
        "last_behavior_artifact": None,
        "last_behavior_scores": pd.DataFrame(),
        "behavior_run_history": [],
        "behavior_profile": "concise_helpfulness",
        "batch_running": False,
        "batch_status": "",
        "chat_cancel_requested": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def plot_if_present(fig, key_hint: str = "plot") -> None:
    if fig is not None:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=ui.TEXT, family="Inter, sans-serif"),
            title_font=dict(color=ui.TEXT, size=15),
        )
        st.plotly_chart(fig, width="stretch", key=f"{key_hint}_{id(fig)}")


def active_model_label() -> str:
    return st.session_state.get("llama_model_alias") or Path(st.session_state.get("llama_model_path", "")).name or "local"


def local_summary(local_context: dict | None) -> dict:
    local_context = local_context or {}
    return {
        "model_name": local_context.get("display_name") or active_model_label(),
        "backend": "llama.cpp",
        "device": "local",
        "layers": int(local_context.get("block_count") or 1),
        "heads": int(local_context.get("head_count") or 1),
        "d_model": int(local_context.get("embedding_length") or 0),
        "d_head": int(local_context.get("d_head") or 0),
        "d_mlp": int(local_context.get("d_mlp") or 0),
        "vocab_size": 0,
        "parameters": int(local_context.get("tensor_elements") or 0),
        "dtype": "gguf",
    }


def set_local_dashboard_trace(trace_payload: dict, prompt: str, backend: str, model_label: str) -> None:
    st.session_state.local_dashboard_trace = trace_payload
    st.session_state.local_dashboard_trace_counter = int(st.session_state.get("local_dashboard_trace_counter", 0)) + 1
    prompt_info = trace_payload.get("prompt", {}) if isinstance(trace_payload.get("prompt"), dict) else {}
    st.session_state.local_dashboard_trace_meta = {
        "prompt": prompt,
        "backend": backend,
        "trace_model": model_label,
        "updated_at": datetime.now().strftime("%H:%M:%S"),
        "token_count": prompt_info.get("n_tokens_total", "-"),
        "run": st.session_state.local_dashboard_trace_counter,
        "run_id": st.session_state.get("active_run_id"),
    }


def single_trace_artifact_from_llama(trace_payload: dict, dash_meta: dict, output: str = "", error: str | None = None) -> dict:
    run_id = str(st.session_state.get("active_run_id") or dash_meta.get("run_id") or "single")
    trace_rows = normalize_llama_trace(
        trace_payload,
        prompt_id=0,
        label="single",
        metadata={"run_id": run_id, "mode": "Single message"},
    )
    if not trace_rows:
        trace_rows = [trace_unavailable_row(run_id, 0, "single", "llama.cpp", "activation summaries were not returned by this backend")]
    return build_run_artifact(
        run_id=run_id,
        mode="Single message",
        backend=str(dash_meta.get("backend", "llama.cpp")),
        model=str(dash_meta.get("trace_model", active_model_label())),
        prompts=[{
            "prompt_id": 0,
            "label": "single",
            "prompt": str(dash_meta.get("prompt", "")),
            "output": output,
            "error": error,
            "elapsed_ms": None,
            "trace_rows": trace_rows,
            "metadata": {"run_id": run_id, "mode": "Single message"},
        }],
    )


def store_behavior_artifact(artifact: dict) -> pd.DataFrame:
    profile = get_behavior_profile(st.session_state.get("behavior_profile", "concise_helpfulness"))
    scores = score_run_artifact(artifact, profile=profile)
    st.session_state.last_behavior_artifact = artifact
    st.session_state.last_behavior_scores = scores
    history = list(st.session_state.get("behavior_run_history", []))
    run_id = str(artifact.get("run_id") or "")
    history = [item for item in history if str(item.get("run_id") or "") != run_id]
    history.append({"run_id": run_id, "artifact": artifact, "scores": scores})
    st.session_state.behavior_run_history = history[-12:]
    return scores


def latest_behavior_artifact() -> dict:
    artifact = st.session_state.get("last_behavior_artifact")
    if isinstance(artifact, dict) and artifact:
        return artifact
    result = st.session_state.get("last_batch_result") or st.session_state.get("last_fuzz_result")
    if isinstance(result, dict) and isinstance(result.get("artifact"), dict):
        return result["artifact"]
    return {}


def selected_control_vector_payload() -> tuple[dict | None, str | None]:
    vector_name = str(st.session_state.get("llama_control_vector") or "").strip()
    if not vector_name:
        return None, "No control vector is selected."
    try:
        return build_steering_metadata(
            vector_name,
            float(st.session_state.get("llama_control_strength", 1.25)),
            int(st.session_state.get("llama_control_layer_start", 1)),
            int(st.session_state.get("llama_control_layer_end", 32)),
        ), None
    except Exception as exc:
        return None, str(exc)


def chat_history_for_send(prompt: str) -> list[dict[str, str]]:
    history = []
    for msg in st.session_state.chat_messages[-10:]:
        role = str(msg.get("role", ""))
        content = str(msg.get("content", ""))
        if role in {"user", "assistant"} and content:
            history.append({"role": role, "content": content})
    if not history or history[-1].get("content") != prompt:
        history.append({"role": "user", "content": prompt})
    return history


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
            start, end = int(start_s), int(end_s)
            if end < start:
                start, end = end, start
            layers.update(layer for layer in range(start, end + 1) if 0 <= layer <= max_layer)
        else:
            layer = int(part)
            if 0 <= layer <= max_layer:
                layers.add(layer)
    return sorted(layers)


def render_control_panel(local_context: dict | None) -> None:
    ui.section_header("Local control vectors", "Generate and launch llama.cpp control-vector runs.")
    control_sets = list_control_sets()
    control_vectors = list_control_vectors()

    c1, c2 = st.columns(2)
    with c1:
        st.session_state.llama_control_set = st.selectbox(
            "Control set",
            [""] + [str(item.get("name", "")) for item in control_sets],
            index=0,
            help="Positive and negative prompts stored under data/control_sets.",
        )
    with c2:
        st.session_state.llama_control_vector = st.selectbox(
            "Control vector",
            [""] + [str(item.get("name", "")) for item in control_vectors],
            index=0,
            help="Generated GGUF vectors stored under data/control_vectors.",
        )

    with st.expander("Create control set", expanded=False):
        set_name = st.text_input("Set name", value="local_behavior")
        pos = st.text_area("Positive prompts", height=120)
        neg = st.text_area("Negative prompts", height=120)
        if st.button("Save control set", width="stretch"):
            positive = [line.strip() for line in pos.splitlines() if line.strip()]
            negative = [line.strip() for line in neg.splitlines() if line.strip()]
            if positive and negative:
                write_control_set(set_name, "\n".join(positive), "\n".join(negative))
                st.success("Control set saved.")
                st.rerun()
            else:
                st.error("Both positive and negative prompts are required.")

    p1, p2, p3 = st.columns(3)
    with p1:
        st.session_state.llama_control_strength = st.number_input("Strength", value=float(st.session_state.llama_control_strength), step=0.25)
    with p2:
        st.session_state.llama_control_layer_start = st.number_input("Layer start", min_value=1, value=int(st.session_state.llama_control_layer_start))
    with p3:
        st.session_state.llama_control_layer_end = st.number_input("Layer end", min_value=1, value=int(st.session_state.llama_control_layer_end))

    preflight = preflight_control_vector_run(
        st.session_state.llama_model_path,
        None,
        None,
        st.session_state.llama_cvector_generator,
        st.session_state.llama_server_bin,
    )
    st.session_state.llama_last_preflight = preflight
    if preflight.errors:
        for error in preflight.errors:
            st.error(error)
    if preflight.warnings:
        for warning in preflight.warnings:
            st.warning(warning)

    if st.session_state.llama_control_set:
        selected = next((item for item in control_sets if item.get("name") == st.session_state.llama_control_set), None)
        if selected:
            vector_name = st.text_input("Vector name", value=selected.name)
            command = build_cvector_command(
                st.session_state.llama_model_path,
                selected["positive_path"],
                selected["negative_path"],
                f"data/control_vectors/{vector_name}.gguf",
                st.session_state.llama_cvector_generator,
            )
            st.code(shell_join(command), language="bash")
            if st.button("Generate vector", type="primary", width="stretch"):
                try:
                    meta = generate_control_vector(
                        vector_name,
                        st.session_state.llama_model_path,
                        selected["positive_path"],
                        selected["negative_path"],
                        st.session_state.llama_cvector_generator,
                    )
                    st.success(f"Generated {Path(meta.vector_path).name}")
                    st.rerun()
                except ControlVectorRunError as exc:
                    st.session_state.llama_last_cvector_failure = exc.metadata.failure.__dict__
                    st.error(str(exc))

    if st.session_state.llama_control_vector:
        cmd = build_llama_server_command(
            st.session_state.llama_model_path,
            st.session_state.llama_control_vector,
            float(st.session_state.llama_control_strength),
            int(st.session_state.llama_control_layer_start),
            int(st.session_state.llama_control_layer_end),
            st.session_state.llama_server_bin,
            port=int(st.session_state.llama_control_port),
            extra_args=st.session_state.llama_control_extra_args,
            alias=st.session_state.llama_model_alias,
        )
        st.caption("Steered server command")
        st.code(shell_join(cmd), language="bash")


def run_app() -> None:
    init_state()
    active_backend = normalize_chat_backend(st.session_state.get("chat_backend_label", CHAT_BACKEND_NORMAL))
    st.session_state.chat_backend_label = chat_backend_display(active_backend)
    local_context = local_gguf_context(
        st.session_state.get("llama_model_path", ""),
        st.session_state.get("llama_model_alias", ""),
        active_backend,
    )
    summary = local_summary(local_context)

    trace_active = st.session_state.get("local_dashboard_trace") is not None
    ui.hud(
        title="Operation Glass Skull",
        subtitle="Local-only llama.cpp cockpit for GGUF chat, control vectors, and activation-path visualization.",
        stats=local_context["hud_stats"],
        pills_html="".join([
            ui.pill(f"Model: {active_model_label()}", ui.GREEN),
            ui.pill(f"Backend: {active_backend}", ui.TEAL),
            ui.pill("Tracing active" if trace_active else "Trace idle", ui.AMBER if trace_active else ui.SLATE, pulse=trace_active),
            ui.pill("Normal server", ui.server_status_color(st.session_state.llama_status)),
            ui.pill("Glass server", ui.server_status_color(st.session_state.llama_glass_status)),
        ]),
    )

    tabs = dict(zip(LOCAL_TABS, st.tabs(LOCAL_TABS)))

    with tabs["Run"]:
        ui.section_header("Run", "Chat with local llama.cpp and capture Glass Skull trace payloads when the server supports them.")
        mode = st.segmented_control("Mode", ["Single message", "Batch run"], default=st.session_state.active_run_mode)
        st.session_state.active_run_mode = mode
        c1, c2, c3 = st.columns(3)
        with c1:
            backend_label = st.selectbox("Backend", CHAT_BACKENDS, key="chat_backend_label")
            chat_backend = normalize_chat_backend(backend_label)
        with c2:
            max_new_tokens = st.slider("Max new tokens", 8, 512, 80, 8)
        with c3:
            temperature = st.slider("Temperature", 0.01, 1.5, 0.8, 0.05)

        auto_trace = st.toggle("Trace prompt", value=True, help="Calls the patched /glass-skull/trace endpoint before generation.")
        local_steering_supported = per_request_steering_supported(
            st.session_state.llama_status.glass_info if chat_backend == "llama.cpp normal" and st.session_state.llama_status else
            st.session_state.llama_glass_status.glass_info if st.session_state.llama_glass_status else {}
        )
        startup_steered = chat_backend == "llama.cpp glass"
        use_steering = False if startup_steered else st.toggle("Use per-request steering", value=False, disabled=not local_steering_supported)
        steering_payload, steering_error = selected_control_vector_payload() if use_steering else (None, None)
        if use_steering and steering_error:
            st.warning(steering_error)

        if mode == "Batch run":
            uploaded = st.file_uploader("Batch prompt file", type=["txt", "jsonl", "csv"])
            uploaded_items = []
            if uploaded is not None:
                uploaded_items = load_prompt_file_bytes(uploaded.name, uploaded.getvalue())
                st.caption(f"Loaded {len(uploaded_items)} prompts.")
            pasted = st.text_area("Pasted prompts", height=120)
            repeat = st.text_input("Repeat current prompt", value="")
            repeat_count = st.number_input("Repeat count", 1, 1000, 1)
        else:
            uploaded_items = []
            pasted = ""
            repeat = ""
            repeat_count = 1

        chat_box = st.container(height=420, border=True)
        with chat_box:
            if not st.session_state.chat_messages:
                ui.empty_state("No messages yet", "Send a prompt to start a local run.")
            for msg in st.session_state.chat_messages[-12:]:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    if msg.get("ts"):
                        st.caption(msg["ts"])

        prompt = st.chat_input("Send a local prompt")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("New chat", width="stretch"):
                save_chat(st.session_state.chat_messages)
                st.session_state.chat_messages = []
                st.rerun()
        with col_b:
            if st.button("Load latest chat", width="stretch"):
                loaded = load_chat()
                if loaded:
                    st.session_state.chat_messages = loaded
                    st.rerun()
                st.warning("No saved chats found.")

        if prompt and mode == "Single message":
            prompt = prompt.strip()
            run_id = new_run_id("single")
            st.session_state.active_run_id = run_id
            st.session_state.chat_messages.append({"role": "user", "content": prompt, "ts": datetime.now().strftime("%H:%M:%S")})
            output = ""
            error = None
            trace_payload = None
            trace_url = st.session_state.llama_glass_url if chat_backend == "llama.cpp glass" else st.session_state.llama_url
            if auto_trace:
                try:
                    layers = list(range(max(int(summary["layers"]), 1)))
                    trace_payload = trace_glass_prompt(
                        trace_url,
                        prompt,
                        model_alias=st.session_state.llama_model_alias,
                        layers=layers,
                        streams=["resid_pre"],
                        max_new_tokens=max_new_tokens,
                        top_k=32,
                        with_pieces=True,
                    )
                    set_local_dashboard_trace(trace_payload, prompt, chat_backend, active_model_label())
                except Exception as exc:
                    st.warning(f"Local trace unavailable: {exc}")
            try:
                if use_steering and steering_payload is None:
                    raise ValueError(steering_error or "No steering payload is available.")
                output = chat_completion(
                    st.session_state.llama_glass_url if chat_backend == "llama.cpp glass" else st.session_state.llama_url,
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    messages=chat_history_for_send(prompt),
                    model_alias=st.session_state.llama_model_alias,
                    steering=steering_payload,
                    steering_supported=local_steering_supported,
                )
            except Exception as exc:
                error = str(exc)
                output = f"Generation error: {error}"
            st.session_state.chat_messages.append({"role": "assistant", "content": output, "ts": datetime.now().strftime("%H:%M:%S")})
            save_chat(st.session_state.chat_messages)
            dash_meta = st.session_state.get("local_dashboard_trace_meta", {}) or {"prompt": prompt, "backend": chat_backend, "trace_model": active_model_label(), "run_id": run_id}
            artifact = single_trace_artifact_from_llama(trace_payload or {}, dash_meta, output=output, error=error)
            store_behavior_artifact(artifact)
            log_run(active_model_label(), "chat_generate", prompt, output=output, metadata={"backend": chat_backend, "run_id": run_id, "error": error})
            st.rerun()

        if prompt and mode == "Batch run":
            prompt_items = batch_items_from_inputs(
                pasted_payload=pasted,
                repeat_prompt=prompt or repeat,
                repeat_count=int(repeat_count),
                uploaded_items=uploaded_items,
            )
            if not prompt_items:
                st.warning("Batch run needs at least one prompt.")
            else:
                run_id = new_run_id("batch")
                st.session_state.active_run_id = run_id
                progress = st.progress(0)
                status = st.empty()

                def cb(i: int, total: int, current_prompt: str) -> None:
                    progress.progress(i / max(total, 1))
                    status.info(f"{i}/{total}: {current_prompt[:120]}")

                try:
                    result = run_fuzz_experiment(
                        name=run_id,
                        prompts=prompt_items,
                        chat_backend="llama.cpp",
                        trace_enabled=bool(auto_trace),
                        llama_url=st.session_state.llama_glass_url if chat_backend == "llama.cpp glass" else st.session_state.llama_url,
                        llama_model_alias=st.session_state.llama_model_alias,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        layers=list(range(max(int(summary["layers"]), 1))),
                        streams=["resid_pre"],
                        top_k=32,
                        run_id=run_id,
                        mode="Batch run",
                        progress_callback=cb,
                    )
                    st.session_state.last_batch_result = result
                    st.session_state.last_fuzz_result = result
                    store_behavior_artifact(result["artifact"])
                    st.success(f"Batch complete: {len(prompt_items)} prompts")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

        context = dashboard_context(str(mode), st.session_state.get("active_run_id"))
        st.caption(f"mode {context['mode']} | run {context['run_id']}")

    with tabs["Map"]:
        ui.section_header("Map", "Visual representation of the latest local run artifact.")
        st.session_state.behavior_profile = st.selectbox("Behavior profile", list_behavior_profiles(), index=list_behavior_profiles().index(st.session_state.behavior_profile))
        artifact = latest_behavior_artifact()
        if not artifact:
            ui.empty_state("No run artifact yet", "Run a single message or batch to populate the map.")
        else:
            payload = build_activation_map_payload(artifact, summary, local_model_context=local_context)
            render_activation_map(payload, key=f"activation_map_{artifact.get('run_id', 'latest')}", height=920)
            scores_df = score_run_artifact(artifact, profile=get_behavior_profile(st.session_state.behavior_profile))
            path_df = activation_path_df(artifact)
            with st.expander("Tables and diagnostics", expanded=False):
                if not scores_df.empty:
                    st.dataframe(scores_df, width="stretch", height=180)
                available = path_df[path_df["trace_available"] != False] if not path_df.empty else pd.DataFrame()
                if available.empty:
                    ui.empty_state("Activation map unavailable", "No activation summaries were captured for this run.")
                else:
                    group = st.segmented_control("Aggregate", ["prompt", "label", "all"], default="label")
                    heat_df = batch_heatmap_df(artifact, group_by=group)
                    plot_if_present(label_activation_heatmap(heat_df) if group == "label" else batch_activation_heatmap(heat_df), key_hint="map_heat")
                    labels = sorted(available["label"].dropna().astype(str).unique().tolist())
                    if len(labels) >= 2:
                        c1, c2 = st.columns(2)
                        with c1:
                            pos_label = st.selectbox("Positive label", labels)
                        with c2:
                            neg_label = st.selectbox("Negative label", labels, index=1 if len(labels) > 1 else 0)
                        if pos_label != neg_label:
                            ranked = rank_activation_paths(available, positive_label=pos_label, negative_label=neg_label)
                            plot_if_present(path_rank_bar_fig(ranked), key_hint="ranked_paths")
                            targets = recommended_steering_targets(ranked, limit=5)
                            if targets:
                                st.dataframe(pd.DataFrame(targets), width="stretch", hide_index=True)
                    plot_if_present(dim_frequency_fig(dimension_frequency_df(artifact)), key_hint="dims")

    with tabs["Steer"]:
        render_control_panel(local_context)

    with tabs["Timeline"]:
        ui.section_header("Timeline", "Behavior scores and saved local run history.")
        history = list(st.session_state.get("behavior_run_history", []))
        score_runs = [item["scores"].assign(run_id=str(item.get("run_id") or "")) for item in history if isinstance(item.get("scores"), pd.DataFrame) and not item["scores"].empty]
        timeline = behavior_timeline_df(score_runs)
        if timeline.empty:
            ui.empty_state("No behavior timeline yet", "Complete local runs to score behavior over time.")
        else:
            plot_if_present(behavior_score_timeline_fig(timeline), key_hint="behavior_timeline")
            run_ids = timeline.sort_values("run_order")["run_id"].dropna().astype(str).unique().tolist()
            if len(run_ids) >= 2:
                c1, c2 = st.columns(2)
                with c1:
                    baseline = st.selectbox("Baseline run", run_ids)
                with c2:
                    comparison = st.selectbox("Comparison run", run_ids, index=len(run_ids) - 1)
                if baseline != comparison:
                    plot_if_present(behavior_delta_bar_fig(timeline, baseline_run_id=baseline, comparison_run_id=comparison), key_hint="behavior_delta")
            st.dataframe(timeline, width="stretch", height=180, hide_index=True)
        st.markdown("---")
        st.dataframe(pd.DataFrame(list_experiments()), width="stretch", height=220)
        st.dataframe(pd.DataFrame(recent_runs()), width="stretch", height=220)

    with tabs["Model"]:
        ui.section_header("Model", "Local GGUF metadata and server capability checks.")
        if st.button("Check servers", type="primary"):
            st.session_state.llama_status = check_server(st.session_state.llama_url, model_alias=st.session_state.llama_model_alias)
            st.session_state.llama_glass_status = check_server(st.session_state.llama_glass_url, model_alias=st.session_state.llama_model_alias)
        ui.property_list(local_context["property_rows"])
        for err in local_context["errors"]:
            st.warning(err)
        tensors = local_context.get("tensors_df", pd.DataFrame())
        plot_if_present(gguf_tensor_shape_scatter_fig(tensors), key_hint="tensor_shape")
        if not tensors.empty:
            st.dataframe(tensors, width="stretch", height=260)

    with tabs["Settings"]:
        ui.section_header("Settings", "Local llama.cpp paths and URLs.")
        st.session_state.llama_model_alias = st.text_input("Model alias", value=st.session_state.llama_model_alias).strip()
        st.session_state.llama_model_path = st.text_input("GGUF model path", value=st.session_state.llama_model_path)
        st.session_state.llama_url = st.text_input("Normal server URL", value=st.session_state.llama_url)
        st.session_state.llama_glass_url = st.text_input("Steered/server trace URL", value=st.session_state.llama_glass_url)
        st.session_state.llama_server_bin = st.text_input("llama-server binary", value=st.session_state.llama_server_bin)
        st.session_state.llama_cvector_generator = st.text_input("llama-cvector-generator binary", value=st.session_state.llama_cvector_generator)
        st.session_state.llama_control_port = st.number_input("Steered server port", min_value=1, max_value=65535, value=int(st.session_state.llama_control_port))
        st.session_state.llama_control_extra_args = st.text_input("Extra llama-server args", value=st.session_state.llama_control_extra_args)


run_app()
