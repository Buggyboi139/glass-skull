from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from glass_skull import ui_theme as ui
from glass_skull.anatomy import config_table, expected_block_table, global_hook_table, hook_table, parameter_table
from glass_skull.attention_view import attention_pattern_table, top_attention_links
from glass_skull.comparison import compare_normal_vs_steered
from glass_skull.config import DEFAULT_MODEL, MODEL_PRESETS, ensure_dirs, normalize_model_name
from glass_skull.contribution import top_contribution_edges
from glass_skull.experiment_store import list_experiments
from glass_skull.feature_store import compatible_features, list_features, load_feature, save_feature
from glass_skull.hf_registry import capabilities_for_backend, families, model_state, registry_as_dicts, visible_models
from glass_skull.hf_loader import build_hf_load_plan
from glass_skull.hf_access import access_badge_text, check_model_access, validate_token
from glass_skull.fuzzing import run_fuzz_experiment
from glass_skull.lens import logit_lens_table, logit_lens_top_token_heatmap
from glass_skull.llama_client import chat_completion, check_server
from glass_skull.logger import log_edges, log_observations, log_run, recent_runs
from glass_skull.model_loader import load_hooked_model, model_summary
from glass_skull.prompt_loader import load_prompt_file_bytes
from glass_skull.steering import build_contrast_vector, generate_normal, generate_steered, vector_summary
from glass_skull.tracer import next_token_table, top_active_dimensions, trace_prompt
from glass_skull.visuals import (
    activation_heatmap,
    activation_pulse,
    attention_heatmap,
    comparison_delta_heatmap,
    dim_frequency_fig,
    edge_constellation,
    fuzz_label_layer_fig,
    fuzz_prompt_layer_fig,
    logit_lens_probability_fig,
    logit_lens_token_heatmap,
    mean_norm_by_layer_fig,
    next_token_bar_fig,
    norm_growth_fig,
    parameter_shape_scatter_fig,
    parameters_by_component_fig,
    parameters_per_layer_fig,
)


st.set_page_config(page_title="Operation Glass Skull", layout="wide", initial_sidebar_state="expanded")
ensure_dirs()
ui.inject_theme()

PLOT_COUNTER = 0

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
    "feature": "A saved direction in activation space. Features only work with models that have the same hidden width.",
    "strength": "How hard to push the selected feature direction during generation. Negative values suppress it.",
    "positive": "Examples that contain the behavior or concept you want to map.",
    "negative": "Plain or opposite examples used as the comparison baseline.",
    "llama_url": "Base URL for a running llama.cpp server. Example: http://127.0.0.1:8080",
    "backend": "Where chat responses come from. Tracing still uses TransformerLens unless a patched llama.cpp trace endpoint exists later.",
    "fuzz": "Rapid-fire a file of prompts through a backend, optionally trace them, and build heatmaps across prompts, labels, layers, and dimensions.",
    "lens": "Projects each layer's internal state through the output vocabulary to estimate what the model is leaning toward at that point.",
    "attention": "Shows which prompt tokens a selected attention head is looking at.",
    "compare": "Runs the same prompt normally and with steering, then shows text and activation differences.",
    "hf_token": "Optional Hugging Face read token. Required for gated models after you accept the model license/access terms on Hugging Face.",
    "hf_catalog": "Official HF model catalog. Visible does not mean loadable; loadability depends on token, access, hardware, and adapter support.",
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
        "last_comparison": None,
        "dashboard_trace": None,
        "dashboard_trace_meta": {},
        "dashboard_trace_counter": 0,
        "selected_feature": None,
        "poke_layer": 0,
        "poke_stream": "resid_post",
        "poke_strength": 1.5,
        "hf_token": "",
        "hf_token_status": None,
        "hf_model_access_cache": {},
        "hf_selected_family": "All",
        "hf_recommended_only": False,
        "hf_selected_repo": "",
        "hf_last_load_plan": None,
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


def plot_if_present(fig, key_hint: str = "plot") -> None:
    global PLOT_COUNTER
    if fig is not None:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=ui.TEXT, family="Inter, sans-serif"),
            title_font=dict(color=ui.TEXT, size=15),
        )
        PLOT_COUNTER += 1
        st.plotly_chart(fig, width="stretch", key=f"{key_hint}_{PLOT_COUNTER}")


def active_chat_model_label(chat_backend: str, summary: dict) -> str:
    if chat_backend == "TransformerLens":
        return f"TransformerLens: {summary['model_name']}"
    if chat_backend == "llama.cpp normal":
        status = st.session_state.llama_status
        model = ", ".join(status.models) if status and status.online and status.models else st.session_state.llama_url
        return f"llama.cpp normal: {model}"
    status = st.session_state.llama_glass_status
    model = ", ".join(status.models) if status and status.online and status.models else st.session_state.llama_glass_url
    return f"llama.cpp glass: {model}"


def feature_names_from_rows(rows: list[dict]) -> list[str]:
    return [str(f["name"]) for f in rows if f.get("name")]



def set_dashboard_trace(trace, prompt: str, backend: str, trace_model: str) -> None:
    st.session_state.dashboard_trace = trace
    st.session_state.dashboard_trace_counter = int(st.session_state.get("dashboard_trace_counter", 0)) + 1
    st.session_state.dashboard_trace_meta = {
        "prompt": prompt,
        "backend": backend,
        "trace_model": trace_model,
        "updated_at": datetime.now().strftime("%H:%M:%S"),
        "token_count": len(getattr(trace, "tokens", []) or []),
        "run": st.session_state.dashboard_trace_counter,
    }



def render_capability_warning(chat_backend: str) -> dict[str, bool]:
    caps = capabilities_for_backend(chat_backend)
    if "llama.cpp" in chat_backend.lower():
        st.warning(
            "llama.cpp backend is chat/output-only right now. Activation Trace, Logit Lens, Attention, Map, Steer, and activation Compare are disabled until llama.cpp-glass exposes real hooks."
        )
    return caps



def render_hf_catalog_panel() -> None:
    st.markdown("### Hugging Face Access")
    token = st.text_input(
        "HF token",
        value=st.session_state.get("hf_token", ""),
        type="password",
        help=HELP["hf_token"],
    )
    st.session_state.hf_token = token

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Validate token", width="stretch"):
            st.session_state.hf_token_status = validate_token(token)
    with c2:
        if st.button("Clear token", width="stretch"):
            st.session_state.hf_token = ""
            st.session_state.hf_token_status = None
            st.session_state.hf_model_access_cache = {}
            st.session_state.hf_last_load_plan = None

    token_status = st.session_state.get("hf_token_status")
    token_valid = bool(token_status and token_status.valid)
    if token_status is None:
        st.caption("HF token: not checked")
    elif token_status.valid:
        st.success(token_status.label())
    else:
        st.error(token_status.label())

    st.markdown("### Official model catalog")
    st.caption(HELP["hf_catalog"])
    fam_options = ["All"] + families()
    fam = st.selectbox(
        "Family",
        fam_options,
        index=fam_options.index(st.session_state.get("hf_selected_family", "All")) if st.session_state.get("hf_selected_family", "All") in fam_options else 0,
    )
    st.session_state.hf_selected_family = fam
    recommended_only = st.toggle("Recommended practical first", value=bool(st.session_state.get("hf_recommended_only", False)))
    st.session_state.hf_recommended_only = recommended_only

    rows = visible_models(fam, recommended_only=recommended_only)
    if not rows:
        st.info("No models match the current filters.")
        return

    labels = [f"{m.family} · {m.display_name} · {m.repo_id}" for m in rows]
    selected_label = st.selectbox("Model", labels)
    selected = rows[labels.index(selected_label)]
    st.session_state.hf_selected_repo = selected.repo_id

    cache = st.session_state.setdefault("hf_model_access_cache", {})
    access = cache.get(selected.repo_id)
    access_status = access.get("status") if isinstance(access, dict) else None
    state, reason, enabled = model_state(selected, token_valid, access_status)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Params", "?" if selected.params_b is None else f"{selected.params_b:g}B")
    with col_b:
        st.metric("Trace", selected.trace_level)
    with col_c:
        st.metric("Access", state)

    st.code(selected.repo_id)
    st.caption(reason)
    st.caption(selected.notes)

    a1, a2 = st.columns(2)
    with a1:
        if st.button("Check model access", width="stretch"):
            checked = check_model_access(selected.repo_id, token=token if token_valid else None)
            cache[selected.repo_id] = checked.__dict__
            st.session_state.hf_model_access_cache = cache
            st.rerun()
    with a2:
        if st.button("Plan HF load", width="stretch", disabled=not enabled):
            plan = build_hf_load_plan(selected.repo_id, token=token if token_valid else None)
            st.session_state.hf_last_load_plan = plan.__dict__

    if access:
        st.info(f"Hub access: {access_badge_text(type('A', (), access)())}")
    if st.session_state.get("hf_last_load_plan"):
        st.json(st.session_state.hf_last_load_plan)

    st.dataframe(pd.DataFrame(registry_as_dicts()), width="stretch", height=220, hide_index=True)


init_state()

# ===========================================================================
# SIDEBAR — collapsible control panel: Model / Servers / Session
# ===========================================================================
with st.sidebar:
    st.markdown(
        '<div class="gs-sidebar-head">'
        '<div class="gs-sidebar-mark"><span style="font-weight:800;font-size:14px;color:#04121c;">GS</span></div>'
        '<div><div class="gs-sidebar-name">Glass Skull</div>'
        '<div class="gs-sidebar-ver">control deck · v0.7</div></div></div>',
        unsafe_allow_html=True,
    )

    with st.expander("Model", expanded=True):
        preset_options = ["custom"] + MODEL_PRESETS
        preset_index = preset_options.index(st.session_state.model_name) if st.session_state.model_name in preset_options else 0
        preset = st.selectbox("Preset", preset_options, index=preset_index, help=HELP["preset"])
        if preset != "custom":
            model_name = preset
        else:
            model_name = st.text_input("TransformerLens model", value=st.session_state.model_name, help="Use a TransformerLens-supported model name.")
        model_name = normalize_model_name(model_name)

        device_choice = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0, help=HELP["device"])
        load_clicked = st.button(
            "Load trace model",
            type="primary",
            width="stretch",
            help="Reloads the selected TransformerLens model and clears the current trace.",
        )
        if load_clicked:
            st.session_state.model_name = model_name
            st.session_state.trace = None
            st.session_state.last_output = ""
            st.session_state.last_comparison = None
            st.session_state.dashboard_trace = None
            st.session_state.dashboard_trace_meta = {}
            load_hooked_model.clear()

model = load_hooked_model(model_name, device_choice=device_choice)
summary = model_summary(model)
expected_dim = int(summary["d_model"])

with st.sidebar:
    ui.sec_label("Loaded trace model")
    ui.property_list(
        [
            ("model", str(summary["model_name"])),
            ("device", str(summary["device"])),
            ("layers", str(summary["layers"])),
            ("d_model", str(summary["d_model"])),
            ("params", f"{summary['parameters']:,}"),
        ]
    )

    with st.expander("Servers", expanded=True):
        st.session_state.llama_url = st.text_input("Normal server URL", value=st.session_state.llama_url, help=HELP["llama_url"])
        st.session_state.llama_glass_url = st.text_input("Glass server URL", value=st.session_state.llama_glass_url, help="Future patched llama.cpp lab server. Use a nonstandard port like 8088.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Check normal", width="stretch", help="Checks /v1/models and optional /glass-skull/info."):
                st.session_state.llama_status = check_server(st.session_state.llama_url)
        with c2:
            if st.button("Check glass", width="stretch", help="Checks the lab llama.cpp server on its separate port."):
                st.session_state.llama_glass_status = check_server(st.session_state.llama_glass_url)

        st.markdown(
            ui.server_health_inline("Normal", st.session_state.llama_url, st.session_state.llama_status)
            + ui.server_health_inline("Glass", st.session_state.llama_glass_url, st.session_state.llama_glass_status),
            unsafe_allow_html=True,
        )
        normal_status = st.session_state.llama_status
        if normal_status is not None and normal_status.online:
            glass = "yes" if normal_status.glass_available else "no"
            models_text = ", ".join(normal_status.models) if normal_status.models else "unknown"
            st.caption(f"Normal models: {models_text} · glass endpoint: {glass}")
        elif normal_status is not None:
            st.caption(f"Normal error: {normal_status.error or 'no details'}")
        glass_status = st.session_state.llama_glass_status
        if glass_status is not None and glass_status.online:
            glass = "yes" if glass_status.glass_available else "no"
            models_text = ", ".join(glass_status.models) if glass_status.models else "unknown"
            st.caption(f"Glass models: {models_text} · glass endpoint: {glass}")
        elif glass_status is not None:
            st.caption(f"Glass error: {glass_status.error or 'no details'}")

    with st.expander("Session", expanded=False):

    with st.expander("Hugging Face", expanded=False):
        render_hf_catalog_panel()

        if st.button("Clear chat", width="stretch", help="Clears only visible chat history. Logs stay saved."):
            st.session_state.chat_messages = []
            st.session_state.last_output = ""
        if st.button("Clear trace", width="stretch", help="Clears the currently cached activations."):
            st.session_state.trace = None
            st.session_state.dashboard_trace = None
            st.session_state.dashboard_trace_meta = {}
        if st.button("Clear comparison", width="stretch", help="Clears the last normal-vs-steered comparison."):
            st.session_state.last_comparison = None

all_features = list_features(include_missing=False)
compatible_feature_rows = compatible_features(expected_dim)
all_feature_names = feature_names_from_rows(all_features)
compatible_feature_names = feature_names_from_rows(compatible_feature_rows)
params_df = parameter_table(model)

# ===========================================================================
# TOP HUD — prominent system state at a glance
# ===========================================================================
trace_active = st.session_state.trace is not None
pills = [
    ui.pill(f"Model: {summary['model_name']}", ui.GREEN),
    ui.pill("Tracing active" if trace_active else "Trace idle", ui.AMBER if trace_active else ui.SLATE, pulse=trace_active),
    ui.pill("Normal server", ui.server_status_color(st.session_state.llama_status)),
    ui.pill("Glass server", ui.server_status_color(st.session_state.llama_glass_status)),
    ui.pill("HF token", ui.GREEN if (st.session_state.get("hf_token_status") and st.session_state.hf_token_status.valid) else ui.SLATE),
]
ui.hud(
    title="Operation Glass Skull",
    subtitle="Probe, trace, and compare transformer behavior — no magic crystals.",
    stats=[
        ("Layers", str(summary["layers"])),
        ("d_model", str(summary["d_model"])),
        ("Heads", str(summary["heads"])),
        ("Params", f"{summary['parameters'] / 1e6:.0f}M"),
        ("Device", str(summary["device"])),
    ],
    pills_html="".join(pills),
)

# ===========================================================================
# MAIN PANELS
# ===========================================================================
tab_dash, tab_chat, tab_trace, tab_poke, tab_anatomy = st.tabs(
    ["Dashboard", "Chat", "Trace / Lens", "Poke / Compare / Fuzz", "Anatomy / Logs"]
)

# ------------------------------------------------------------- Dashboard ----
with tab_dash:
    ui.section_header("Model architecture", "Structure and parameter distribution read directly from the loaded model.")
    ui.render_network(summary)

    ui.property_list(
        [
            ("model_name", str(summary["model_name"])),
            ("device", str(summary["device"])),
            ("dtype", str(summary.get("dtype", ""))),
            ("n_layers", str(summary["layers"])),
            ("d_model", str(summary["d_model"])),
            ("n_heads", str(summary["heads"])),
            ("d_head", str(summary["d_head"])),
            ("d_mlp", str(summary["d_mlp"])),
            ("vocab_size", f"{int(summary['vocab_size']):,}"),
            ("total_parameters", f"{int(summary['parameters']):,}"),
        ]
    )

    g1, g2 = st.columns(2)
    with g1:
        plot_if_present(parameters_per_layer_fig(params_df))
    with g2:
        plot_if_present(parameters_by_component_fig(params_df))
    plot_if_present(parameter_shape_scatter_fig(params_df))

    st.markdown("---")
    ui.section_header("Live activations", "Derived from the most recent traced prompt. Scroll for the full picture.")
    trace = st.session_state.get("dashboard_trace") or st.session_state.get("trace")
    dash_meta = st.session_state.get("dashboard_trace_meta", {}) or {}
    if trace is None:
        ui.empty_state("No trace captured yet", "Send a message in the Chat tab with tracing enabled to populate these graphs.")
    else:
        ui.property_list([
            ("updated", str(dash_meta.get("updated_at", "current run"))),
            ("backend", str(dash_meta.get("backend", "unknown"))),
            ("trace_model", str(dash_meta.get("trace_model", summary["model_name"]))),
            ("tokens", str(dash_meta.get("token_count", len(trace.tokens)))),
            ("run", str(dash_meta.get("run", "-"))),
        ])
        st.code(trace.prompt)
        ui.timeline([f"{i}·{tok}" for i, tok in enumerate(trace.tokens)], current=len(trace.tokens) - 1)

        d1, d2 = st.columns(2)
        with d1:
            plot_if_present(mean_norm_by_layer_fig(trace.layer_norms))
        with d2:
            plot_if_present(norm_growth_fig(trace.layer_norms))

        plot_if_present(activation_pulse(trace.layer_norms))
        plot_if_present(activation_heatmap(trace.layer_norms))

        last_token = max(len(trace.tokens) - 1, 0)
        try:
            lens_df = logit_lens_table(model, trace.cache, token_index=last_token, stream="resid_post", top_k=5)
            plot_if_present(logit_lens_probability_fig(lens_df))
        except Exception as exc:
            st.caption(f"Logit lens unavailable: {exc}")

        try:
            plot_if_present(next_token_bar_fig(next_token_table(model, trace.logits, top_k=15)))
        except Exception as exc:
            st.caption(f"Next-token graph unavailable: {exc}")

        try:
            attn_df = attention_pattern_table(trace.cache, 0, 0, trace.tokens)
            plot_if_present(attention_heatmap(attn_df))
            st.caption("Attention shown for layer 0, head 0. Use the Trace / Lens tab to explore other heads.")
        except Exception as exc:
            st.caption(f"Attention graph unavailable: {exc}")

# ------------------------------------------------------------------ Chat ----
with tab_chat:
    ui.section_header("Chat", "Talk to the model and capture activations as you go.")

    cfg1, cfg2, cfg3 = st.columns([2, 1, 1])
    with cfg1:
        chat_backend = st.selectbox(
            "Chat backend",
            ["TransformerLens", "llama.cpp normal", "llama.cpp glass"],
            help=HELP["backend"],
        )
    with cfg2:
        max_new_tokens = st.slider("Max new tokens", 10, 300, 80, 10, help=HELP["max_new_tokens"], key="chat_max_new")
    with cfg3:
        temperature = st.slider("Temperature", 0.01, 1.5, 0.8, 0.05, help=HELP["temperature"], key="chat_temp")

    tog1, tog2 = st.columns(2)

    caps = render_capability_warning(chat_backend)

    with tog1:
        auto_trace = st.toggle("Trace every message", value=True, help="When enabled, every message also captures activations with the trace model.")
    with tog2:
        use_steering = st.toggle("Use steering", value=False, disabled=not caps["activation_steering"], help="Only applies when the chat backend is TransformerLens. Stock llama.cpp cannot be activation-steered yet.")

    # Persistent HUD line above the conversation
    config_badges = "".join(
        [
            ui.badge(active_chat_model_label(chat_backend, summary), ui.ACCENT),
            ui.badge(f"temp {temperature:.2f}", ui.TEAL),
            ui.badge(f"{max_new_tokens} tokens", ui.TEAL),
            ui.badge("tracing", ui.AMBER, active=auto_trace),
            ui.badge("steering", ui.PURPLE, active=use_steering),
        ]
    )
    st.markdown(
        f'<div class="gs-card" style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;">{config_badges}</div>',
        unsafe_allow_html=True,
    )

    if use_steering and chat_backend != "TransformerLens":
        st.warning("Steering only applies to TransformerLens chat right now. llama.cpp steering waits for the future C++ cave expedition.")
    if use_steering and not compatible_feature_names:
        st.warning(f"No compatible features for d_model {expected_dim}. Rebuild a feature with the currently loaded trace model.")

    chat_box = st.container(height=430, border=True)
    with chat_box:
        if not st.session_state.chat_messages:
            ui.empty_state("No messages yet", "Send a prompt below to start the conversation and tracing.")
        for msg in st.session_state.chat_messages[-12:]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                ts = msg.get("ts")
                if ts:
                    st.caption(ts)

    with st.form("chat_form", clear_on_submit=False, border=False):
        prompt = st.text_area("Message", value="The cat sat on the", height=110, help="The text sent to the selected chat backend. Press Ctrl+Enter to send.", key="chat_prompt")
        send = st.form_submit_button("Send message", type="primary", width="stretch")

    if send and prompt.strip():
        prompt = prompt.strip()
        now = datetime.now().strftime("%H:%M:%S")
        st.session_state.chat_messages.append({"role": "user", "content": prompt, "ts": now})

        if auto_trace:
            with st.spinner("Tracing prompt locally..."):
                trace = trace_prompt(model, prompt)
                st.session_state.trace = trace
                set_dashboard_trace(trace, prompt, chat_backend, str(summary["model_name"]))
                run_id = log_run(model_name=model_name, mode="chat_trace", prompt=prompt, metadata={"tokens": trace.tokens, "summary": summary, "chat_backend": chat_backend})
                st.session_state.last_run_id = run_id

        output = ""
        error = None
        with st.spinner("Generating reply..."):
            try:
                if chat_backend == "llama.cpp normal":
                    output = chat_completion(st.session_state.llama_url, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
                elif chat_backend == "llama.cpp glass":
                    output = chat_completion(st.session_state.llama_glass_url, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
                elif use_steering:
                    if not compatible_feature_names:
                        raise ValueError(f"No compatible steering features for current trace model d_model {expected_dim}.")
                    selected_feature = st.session_state.get("selected_feature") or compatible_feature_names[0]
                    if selected_feature not in compatible_feature_names:
                        selected_feature = compatible_feature_names[0]
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
        st.session_state.chat_messages.append({"role": "assistant", "content": output, "ts": datetime.now().strftime("%H:%M:%S")})
        log_run(model_name=model_name, mode="chat_generate", prompt=prompt, output=output, metadata={"backend": chat_backend, "used_steering": bool(use_steering), "error": error})
        st.rerun()

# ----------------------------------------------------------- Trace / Lens ----
with tab_trace:
    ui.section_header("Trace / Lens", "Visualize model internals captured from the latest traced prompt.")
    if 'chat_backend' in locals() and "llama.cpp" in chat_backend.lower():
        ui.empty_state("Trace source is not llama.cpp", "The current llama.cpp backend can chat, but it does not expose activations. Switch to TransformerLens/HF trace mode for Logit Lens, Attention, and steering.")

    trace = st.session_state.trace
    if trace is None:
        ui.empty_state("No trace active", "Start a chat with tracing enabled to begin tracing.")
    else:
        ui.timeline([f"{i}·{tok}" for i, tok in enumerate(trace.tokens)], current=len(trace.tokens) - 1)
        st.code(" | ".join([f"{i}:{tok}" for i, tok in enumerate(trace.tokens)]))
        trace_mode = st.radio("View", ["Pulse", "Heatmap", "Logit Lens", "Attention"], horizontal=True)

        if trace_mode == "Pulse":
            plot_if_present(activation_pulse(trace.layer_norms))
        elif trace_mode == "Heatmap":
            plot_if_present(activation_heatmap(trace.layer_norms))
        elif trace_mode == "Logit Lens":
            ui.purpose(HELP["lens"])
            lens_stream = st.selectbox("Lens stream", ["resid_pre", "resid_mid", "resid_post"], index=2, key="lens_stream")
            lens_token = st.number_input("Lens token", 0, max(len(trace.tokens) - 1, 0), max(len(trace.tokens) - 1, 0), key="lens_token")
            lens_k = st.slider("Top predictions", 1, 20, 5, 1, key="lens_k")
            try:
                lens_df = logit_lens_table(model, trace.cache, token_index=int(lens_token), stream=lens_stream, top_k=lens_k)
                plot_if_present(logit_lens_probability_fig(lens_df))
                st.dataframe(lens_df, width="stretch", height=220)
                with st.expander("Layer/token certainty heatmap", expanded=False):
                    token_df = logit_lens_top_token_heatmap(model, trace.cache, trace.tokens, stream=lens_stream)
                    plot_if_present(logit_lens_token_heatmap(token_df))
                    st.dataframe(token_df, width="stretch", height=220)
            except Exception as exc:
                st.error(str(exc))
        elif trace_mode == "Attention":
            ui.purpose(HELP["attention"])
            attn_layer = st.number_input("Attention layer", 0, summary["layers"] - 1, 0, key="attn_layer")
            attn_head = st.number_input("Head", 0, max(summary["heads"] - 1, 0), 0, key="attn_head")
            try:
                attn_df = attention_pattern_table(trace.cache, int(attn_layer), int(attn_head), trace.tokens)
                plot_if_present(attention_heatmap(attn_df))
                st.dataframe(top_attention_links(trace.cache, int(attn_layer), int(attn_head), trace.tokens, top_k=30), width="stretch", height=220)
            except Exception as exc:
                st.error(str(exc))

        st.markdown("---")
        ui.section_header("Inspect a point", "Drill into a specific layer, stream, and token.")
        ip1, ip2, ip3, ip4 = st.columns(4)
        with ip1:
            layer = st.number_input("Layer", 0, summary["layers"] - 1, 0, help=HELP["layer"], key="trace_layer")
        with ip2:
            stream = st.selectbox("Stream", ["resid_pre", "attn_out", "mlp_out", "resid_post"], index=3, help=HELP["stream"], key="trace_stream")
        with ip3:
            token_index = st.number_input("Token", 0, max(len(trace.tokens) - 1, 0), max(len(trace.tokens) - 1, 0), help=HELP["token"], key="trace_token")
        with ip4:
            top_k = st.slider("Top dims", 5, 100, 30, 5, help=HELP["top_dims"], key="trace_top_dims")
        try:
            dims = top_active_dimensions(trace.cache, int(layer), stream, int(token_index), top_k=top_k)
            st.dataframe(dims, width="stretch", height=200)
            if st.button("Log dims", help="Save the displayed activation dimensions to SQLite."):
                run_id = st.session_state.get("last_run_id") or log_run(model_name, "trace_dims", trace.prompt)
                rows = dims.assign(layer=int(layer), stream=stream, token_index=int(token_index), token=trace.tokens[int(token_index)]).to_dict("records")
                log_observations(run_id, rows)
                st.success(f"Logged {len(rows)} rows")
        except Exception as exc:
            st.error(str(exc))

        with st.expander("Final next-token probabilities", expanded=False):
            st.dataframe(next_token_table(model, trace.logits, top_k=20), width="stretch")

# ------------------------------------------------ Poke / Compare / Fuzz ----
with tab_poke:
    ui.section_header("Poke / Compare / Fuzz", "Probe and stress-test model behavior.")
    if 'chat_backend' in locals() and "llama.cpp" in chat_backend.lower():
        st.info("llama.cpp selected: chat and fuzz output are available, but activation Map/Steer/Compare controls target TransformerLens only until llama.cpp-glass exposes hooks.")

    tab_steer, tab_map, tab_compare, tab_edges, tab_fuzz = st.tabs(
        ["Steer", "Map", "Compare", "Edges", "Fuzz"]
    )

    with tab_steer:
        ui.purpose("Apply a saved activation direction to TransformerLens replies. Pick a feature, layer, stream, and strength.")
        if all_feature_names and not compatible_feature_names:
            st.warning(f"Saved features exist, but none match current d_model {expected_dim}. Rebuild one with this trace model.")
            st.dataframe(pd.DataFrame(all_features), width="stretch", height=160)
        elif not compatible_feature_names:
            st.info("No compatible saved features yet. Use the Map tab to make one.")
        else:
            selected_feature = st.selectbox("Feature", compatible_feature_names, help=HELP["feature"], key="selected_feature")
            vector, meta = load_feature(selected_feature)
            default_layer = int(meta.get("layer", 0))
            default_stream = meta.get("stream", "resid_post")
            stream_options = ["resid_pre", "attn_out", "mlp_out", "resid_post"]
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.session_state.poke_layer = st.number_input("Layer", 0, summary["layers"] - 1, min(default_layer, summary["layers"] - 1), help=HELP["layer"], key="poke_layer_widget")
            with sc2:
                st.session_state.poke_stream = st.selectbox("Stream", stream_options, index=stream_options.index(default_stream) if default_stream in stream_options else 3, help=HELP["stream"], key="poke_stream_widget")
            with sc3:
                st.session_state.poke_strength = st.slider("Strength", -5.0, 5.0, 1.5, 0.25, help=HELP["strength"], key="poke_strength_widget")
            st.caption(f"Feature vector dim: `{meta.get('vector_dim')}` | model d_model: `{expected_dim}`")
            st.dataframe(pd.DataFrame(vector_summary(vector, top_k=25)), width="stretch", height=240)
            st.info("Turn on 'Use steering' in Chat to apply this feature to TransformerLens replies.")

    with tab_map:
        ui.purpose("Build a feature: average positive examples, subtract negative examples, and save the resulting direction.")
        positive_text = st.text_area("Positive examples", value="Oh great, another meeting.\nFantastic, the server broke again.\nWonderful, I get to debug this all night.", height=110, help=HELP["positive"], key="map_positive")
        negative_text = st.text_area("Negative examples", value="The meeting started.\nThe server stopped responding.\nI need to debug this program.", height=110, help=HELP["negative"], key="map_negative")
        mc1, mc2 = st.columns(2)
        with mc1:
            map_layer = st.number_input("Map layer", 0, summary["layers"] - 1, min(3, summary["layers"] - 1), help=HELP["layer"], key="map_layer")
        with mc2:
            map_stream = st.selectbox("Map stream", ["resid_pre", "attn_out", "mlp_out", "resid_post"], index=3, help=HELP["stream"], key="map_stream")
        feature_name = st.text_input("Feature name", value="sarcasm-ish", help="Name for the saved activation direction.")
        if st.button("Build feature", type="primary", width="stretch", help="Average positive examples, subtract negative examples, and save the result."):
            positive = [p.strip() for p in positive_text.splitlines() if p.strip()]
            negative = [p.strip() for p in negative_text.splitlines() if p.strip()]
            try:
                with st.spinner("Computing feature vector..."):
                    vec = build_contrast_vector(model, positive, negative, int(map_layer), stream=map_stream)
                    tensor_path, _ = save_feature(feature_name, vec, {"model_name": model_name, "d_model": expected_dim, "layer": int(map_layer), "stream": map_stream, "positive_count": len(positive), "negative_count": len(negative), "positive_prompts": positive, "negative_prompts": negative})
                    log_run(model_name, "build_feature", feature_name, metadata={"layer": int(map_layer), "stream": map_stream})
                st.success(f"Saved {tensor_path.name}")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    with tab_compare:
        ui.purpose(HELP["compare"])
        if not compatible_feature_names:
            st.info("Build a compatible feature before comparing normal vs steered runs.")
        else:
            cmp_feature = st.selectbox("Compare feature", compatible_feature_names, key="cmp_feature")
            cmp_prompt = st.text_area("Compare prompt", value=st.session_state.get("chat_prompt", "Explain what a mammal is."), height=90, key="cmp_prompt")
            vector, meta = load_feature(cmp_feature)
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                cmp_layer = st.number_input("Compare layer", 0, summary["layers"] - 1, min(int(meta.get("layer", 0)), summary["layers"] - 1), key="cmp_layer")
            with cc2:
                cmp_stream = st.selectbox("Compare stream", ["resid_pre", "attn_out", "mlp_out", "resid_post"], index=3, key="cmp_stream")
            with cc3:
                cmp_strength = st.slider("Compare strength", -5.0, 5.0, 1.5, 0.25, key="cmp_strength")
            if st.button("Run comparison", type="primary", width="stretch"):
                try:
                    with st.spinner("Running normal vs steered comparison..."):
                        st.session_state.last_comparison = compare_normal_vs_steered(
                            model,
                            cmp_prompt,
                            vector,
                            int(cmp_layer),
                            cmp_stream,
                            float(cmp_strength),
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                        )
                except Exception as exc:
                    st.error(str(exc))
            cmp_result = st.session_state.last_comparison
            if cmp_result:
                left, right = st.columns(2)
                with left:
                    st.markdown(ui.badge("NORMAL", ui.SLATE), unsafe_allow_html=True)
                    st.write(cmp_result.normal_output)
                with right:
                    st.markdown(ui.badge("STEERED", ui.PURPLE), unsafe_allow_html=True)
                    st.write(cmp_result.steered_output)
                plot_if_present(comparison_delta_heatmap(cmp_result.norm_diff))
                st.dataframe(cmp_result.norm_diff.sort_values("abs_delta", ascending=False).head(50), width="stretch", height=220)

    with tab_edges:
        ui.purpose(HELP["edges"])
        trace = st.session_state.trace
        if trace is None:
            ui.empty_state("No trace to walk", "Trace a message first, then active edges can be computed from that trace.")
        else:
            ec1, ec2, ec3, ec4 = st.columns(4)
            with ec1:
                edge_layer = st.number_input("Edge layer", 0, summary["layers"] - 1, 0, help=HELP["layer"], key="edge_layer")
            with ec2:
                module = st.selectbox("Matrix", ["mlp.W_in", "mlp.W_out"], help=HELP["edges"], key="edge_module")
            with ec3:
                edge_token = st.number_input("Edge token", 0, max(len(trace.tokens) - 1, 0), max(len(trace.tokens) - 1, 0), help=HELP["token"], key="edge_token")
            with ec4:
                edge_k = st.slider("Top edges", 5, 200, 50, 5, help="How many strongest contribution paths to show.", key="edge_k")
            try:
                edges = top_contribution_edges(model, trace.cache, int(edge_layer), module, int(edge_token), top_k=edge_k)
                plot_if_present(edge_constellation(edges))
                st.dataframe(edges, width="stretch", height=220)
                if st.button("Log edges", help="Save the displayed edge rows to SQLite."):
                    run_id = st.session_state.get("last_run_id") or log_run(model_name, "active_edges", trace.prompt)
                    log_edges(run_id, edges.to_dict("records"))
                    st.success(f"Logged {len(edges)} rows")
            except Exception as exc:
                st.error(str(exc))

    with tab_fuzz:
        ui.purpose(HELP["fuzz"])
        fz1, fz2 = st.columns(2)
        with fz1:
            fuzz_name = st.text_input("Experiment name", value="glass_probe", help="Used to name the saved folder under data/experiments.")
        with fz2:
            fuzz_backend = st.selectbox("Fuzz chat backend", ["TransformerLens", "llama.cpp normal", "llama.cpp glass"], help="Where generated outputs come from.")
        uploaded = st.file_uploader("Prompt file", type=["txt", "jsonl", "csv"], help="TXT: one prompt per line. JSONL/CSV: use prompt and optional label fields.")
        trace_fuzz = st.toggle("Trace fuzz prompts", value=True, help="Captures activation summaries with the TransformerLens trace model for each prompt.")
        fzc1, fzc2 = st.columns(2)
        with fzc1:
            layer_raw = st.text_input("Layers", value="all", help="Use 'all', comma lists like 0,4,8, or ranges like 0-8.")
        with fzc2:
            streams = st.multiselect("Streams", ["resid_pre", "attn_out", "mlp_out", "resid_post"], default=["resid_post"], help="Which activation streams to aggregate.")
        fzc3, fzc4 = st.columns(2)
        with fzc3:
            fuzz_top_k = st.slider("Top dims per point", 5, 100, 32, 1, help="How many active dimensions to store per selected layer/stream.")
        with fzc4:
            fuzz_limit = st.slider("Prompt limit", 1, 1000, 200, 1, help="Caps the number of prompts from the file for this run.")

        if uploaded is not None:
            try:
                prompt_items = load_prompt_file_bytes(uploaded.name, uploaded.getvalue())[:fuzz_limit]
                st.write(f"Loaded `{len(prompt_items)}` prompts")
                st.dataframe(pd.DataFrame([{"id": p.prompt_id, "label": p.label, "prompt": p.prompt} for p in prompt_items[:20]]), width="stretch", height=140)
            except Exception as exc:
                prompt_items = []
                st.error(str(exc))
        else:
            prompt_items = []

        if st.button("Run fuzz", type="primary", width="stretch", disabled=not prompt_items):
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
            sep_df = result.get("separation_df", pd.DataFrame())
            if not sep_df.empty:
                st.markdown("**Suggested map locations**")
                st.dataframe(sep_df.head(30), width="stretch", height=180)
                labels = sorted(set([r.get("label", "unlabeled") for r in result.get("records", []) if r.get("label") != "unlabeled"]))
                if len(labels) >= 2:
                    a = st.selectbox("Positive label", labels, key="fuzz_pos_label")
                    b = st.selectbox("Negative label", labels, index=1 if len(labels) > 1 else 0, key="fuzz_neg_label")
                    best = sep_df.iloc[0]
                    f_layer = st.number_input("Feature layer from fuzz", 0, summary["layers"] - 1, int(best["layer"]), key="fuzz_feature_layer")
                    f_stream = st.selectbox("Feature stream from fuzz", ["resid_pre", "attn_out", "mlp_out", "resid_post"], index=["resid_pre", "attn_out", "mlp_out", "resid_post"].index(str(best["stream"])) if str(best["stream"]) in ["resid_pre", "attn_out", "mlp_out", "resid_post"] else 3, key="fuzz_feature_stream")
                    f_name = st.text_input("Fuzz feature name", value=f"{a}_minus_{b}_L{int(f_layer)}", key="fuzz_feature_name")
                    if st.button("Save feature from fuzz labels", width="stretch"):
                        pos = [r["prompt"] for r in result["records"] if r.get("label") == a]
                        neg = [r["prompt"] for r in result["records"] if r.get("label") == b]
                        if not pos or not neg:
                            st.error("Both labels need at least one prompt.")
                        else:
                            try:
                                vec = build_contrast_vector(model, pos, neg, int(f_layer), stream=f_stream)
                                tensor_path, _ = save_feature(f_name, vec, {"model_name": model_name, "d_model": expected_dim, "layer": int(f_layer), "stream": f_stream, "positive_label": a, "negative_label": b, "positive_count": len(pos), "negative_count": len(neg), "source": "fuzz"})
                                st.success(f"Saved {tensor_path.name}")
                                st.rerun()
                            except Exception as exc:
                                st.error(str(exc))

# ----------------------------------------------------- Anatomy / Logs ----
with tab_anatomy:
    ui.section_header("Anatomy / Logs", "Inspect model architecture and saved run history.")
    panel = st.radio(
        "Panel",
        ["Anatomy", "Hooks", "Parameters", "Experiments", "Features", "HF Catalog", "Logs"],
        horizontal=True,
        help="Switch between model structure and saved run history.",
    )

    if panel == "Anatomy":
        cfg_df = config_table(model)
        ui.sec_label("Configuration")
        ui.property_list([(str(row["field"]), str(row["value"])) for _, row in cfg_df.iterrows()])
        ui.sec_label("Block components")
        block_df = expected_block_table(model)
        st.dataframe(
            block_df,
            width="stretch",
            height=420,
            hide_index=True,
            column_config={
                "layer": st.column_config.NumberColumn("Layer", width="small"),
                "component": st.column_config.TextColumn("Component", width="medium"),
                "hook_point": st.column_config.TextColumn("Hook point", width="large"),
                "present": st.column_config.CheckboxColumn("Present", width="small"),
            },
        )
    elif panel == "Hooks":
        ui.sec_label("Global hooks")
        st.dataframe(global_hook_table(model), width="stretch", height=180, hide_index=True)
        ui.sec_label("All hook points")
        st.dataframe(hook_table(model), width="stretch", height=430, hide_index=True)
    elif panel == "Parameters":
        params = params_df
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Tensor count", f"{len(params)}")
        with m2:
            if not params.empty:
                st.metric("Total parameters", f"{int(params['parameters'].sum()):,}")
        st.dataframe(params, width="stretch", height=520, hide_index=True)
    elif panel == "Experiments":
        st.dataframe(pd.DataFrame(list_experiments()), width="stretch", height=590, hide_index=True)
    elif panel == "Features":
        st.caption(f"Compatible with current d_model {expected_dim}: {len(compatible_feature_names)} / {len(all_feature_names)}")
        st.dataframe(pd.DataFrame(all_features), width="stretch", height=590, hide_index=True)
    elif panel == "HF Catalog":
        st.dataframe(pd.DataFrame(registry_as_dicts()), width="stretch", height=590, hide_index=True)
    elif panel == "Logs":
        runs = recent_runs(limit=100)
        log_lines = []
        for r in runs:
            ts = str(r.get("created_at", ""))
            ts_short = ts[11:19] if len(ts) >= 19 else ts
            tag = f"[{r.get('mode', '?')}]"
            prompt_text = (r.get("prompt") or "").replace("\n", " ")
            out_text = (r.get("output") or "").replace("\n", " ")
            msg = f"{r.get('model_name', '?')} · {prompt_text[:80]}"
            if out_text:
                msg += f"  ->  {out_text[:80]}"
            log_lines.append((ts_short, tag, msg))
        ui.terminal("glass_skull.runs — recent activity", log_lines)