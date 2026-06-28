from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from glass_skull import ui_theme as ui
from glass_skull.anatomy import config_table, expected_block_table, global_hook_table, hook_table, parameter_table
from glass_skull.attention_view import attention_pattern_table, top_attention_links
from glass_skull.chat_store import list_chats, load_chat, save_chat
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
from glass_skull.llama_control import (
    ControlVectorRunError,
    DEFAULT_CVECTOR_GENERATOR,
    DEFAULT_LLAMA_SERVER,
    build_cvector_command,
    build_llama_server_command,
    generate_control_vector,
    list_control_sets,
    list_control_vectors,
    preflight_control_vector_run,
    read_gguf_tensor_index,
    shell_join,
    write_control_set,
)
from glass_skull.llama_client import build_steering_metadata, chat_completion, check_server, per_request_steering_supported, trace_glass_prompt
from glass_skull.logger import log_edges, log_observations, log_run, recent_runs
from glass_skull.model_context import local_gguf_context
from glass_skull.model_loader import load_hooked_model, model_summary
from glass_skull.prompt_loader import load_prompt_file_bytes
from glass_skull.run_artifacts import (
    activation_path_df,
    batch_heatmap_df,
    build_run_artifact,
    normalize_llama_trace,
    normalize_transformerlens_trace,
    trace_unavailable_row,
)
from glass_skull.steering import build_contrast_vector, generate_normal, generate_steered, vector_summary
from glass_skull.tracer import next_token_table, top_active_dimensions, trace_prompt
from glass_skull.ui_lab import hf_enabled as source_hf_enabled, lab_enabled as source_lab_enabled, trace_enabled as source_trace_enabled
from glass_skull.ui_local import LOCAL_TABS, batch_items_from_inputs, dashboard_context, new_run_id
from glass_skull.visuals import (
    activation_path_graph,
    batch_activation_heatmap,
    activation_heatmap,
    activation_pulse,
    attention_heatmap,
    comparison_delta_heatmap,
    dim_frequency_fig,
    edge_constellation,
    fuzz_label_layer_fig,
    fuzz_prompt_layer_fig,
    label_activation_heatmap,
    gguf_tensor_dtype_fig,
    gguf_tensor_shape_scatter_fig,
    gguf_tensors_by_component_fig,
    gguf_tensors_per_layer_fig,
    logit_lens_probability_fig,
    logit_lens_token_heatmap,
    mean_norm_by_layer_fig,
    next_token_bar_fig,
    norm_growth_fig,
    parameter_shape_scatter_fig,
    parameters_by_component_fig,
    parameters_per_layer_fig,
)


st.set_page_config(page_title="Operation Glass Skull", layout="wide", initial_sidebar_state="collapsed")
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
    "backend": "Where chat responses come from. Local GGUF options use the normal and steered llama.cpp server URLs and model alias from Settings. Tracing uses the TransformerLens trace model.",
    "llama_control": "llama.cpp control vectors are loaded at server startup. Generate a vector, then launch a steered server with the shown flags.",
    "fuzz": "Rapid-fire a file of prompts through a backend, optionally trace them, and build heatmaps across prompts, labels, layers, and dimensions.",
    "lens": "Projects each layer's internal state through the output vocabulary to estimate what the model is leaning toward at that point.",
    "attention": "Shows which prompt tokens a selected attention head is looking at.",
    "compare": "Runs the same prompt normally and with steering, then shows text and activation differences.",
    "hf_token": "Optional Hugging Face read token. Required for gated models after you accept the model license/access terms on Hugging Face.",
    "hf_catalog": "Official HF model catalog. Visible does not mean loadable; loadability depends on token, access, hardware, and adapter support.",
}

CHAT_BACKEND_TRACE = "Trace model (TransformerLens)"
CHAT_BACKEND_LOCAL_NORMAL = "Local GGUF normal (llama.cpp)"
CHAT_BACKEND_LOCAL_STEERED = "Local GGUF steered (llama.cpp)"
CHAT_BACKEND_OPTIONS = [CHAT_BACKEND_LOCAL_NORMAL, CHAT_BACKEND_LOCAL_STEERED, CHAT_BACKEND_TRACE]
MODEL_SOURCE_TRACE = "Trace model"
MODEL_SOURCE_LOCAL = "Local GGUF"
MODEL_SOURCE_HF = "Hugging Face"
MODEL_SOURCE_OPTIONS = [MODEL_SOURCE_LOCAL, MODEL_SOURCE_HF, MODEL_SOURCE_TRACE]


def normalize_chat_backend(label: str) -> str:
    if label in {"TransformerLens", CHAT_BACKEND_TRACE}:
        return "TransformerLens"
    if label in {"llama.cpp normal", CHAT_BACKEND_LOCAL_NORMAL}:
        return "llama.cpp normal"
    if label in {"llama.cpp glass", CHAT_BACKEND_LOCAL_STEERED}:
        return "llama.cpp glass"
    return label


def chat_backend_display(canonical: str) -> str:
    if canonical == "TransformerLens":
        return CHAT_BACKEND_TRACE
    if canonical == "llama.cpp normal":
        return CHAT_BACKEND_LOCAL_NORMAL
    if canonical == "llama.cpp glass":
        return CHAT_BACKEND_LOCAL_STEERED
    return canonical


def init_state() -> None:
    defaults = {
        "trace": None,
        "model_name": DEFAULT_MODEL,
        "chat_messages": [],
        "chat_backend_label": CHAT_BACKEND_LOCAL_NORMAL,
        "last_output": "",
        "last_run_id": None,
        "llama_url": "http://127.0.0.1:8080",
        "llama_glass_url": "http://127.0.0.1:8088",
        "llama_status": None,
        "llama_glass_status": None,
        "device_choice": "auto",
        "llama_model_alias": "qwen3.6-35b-mtp-q4-ks-vision",
        "llama_model_path": "/home/dsmason321/models/Best/Qwen3.6-35B-MTP-Q4_KS.gguf",
        "llama_cvector_generator": str(DEFAULT_CVECTOR_GENERATOR),
        "llama_server_bin": str(DEFAULT_LLAMA_SERVER),
        "llama_control_set": "",
        "llama_control_vector": "",
        "llama_control_strength": 1.25,
        "llama_control_layer_start": 20,
        "llama_control_layer_end": 60,
        "llama_control_port": 8088,
        "llama_control_extra_args": "--jinja --flash-attn auto --cache-type-k q4_0 --cache-type-v q4_0 --no-mmap",
        "llama_cvector_explicit_ngl": False,
        "llama_cvector_ngl": 0,
        "llama_cvector_fit": "auto",
        "llama_cvector_ctx_size": 0,
        "llama_cvector_pca_batch": 0,
        "llama_cvector_pca_iter": 0,
        "llama_last_preflight": None,
        "llama_last_cvector_failure": None,
        "last_fuzz_result": None,
        "last_comparison": None,
        "dashboard_trace": None,
        "dashboard_trace_meta": {},
        "dashboard_trace_counter": 0,
        "local_dashboard_trace": None,
        "local_dashboard_trace_meta": {},
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
        "workflow_setup_complete": False,
        "workflow_setup_sources": [MODEL_SOURCE_LOCAL],
        "workflow_setup_last_status": {},
        "workflow_hf_requires_token": False,
        "chat_cancel_requested": False,
        "last_loaded_chat": "",
        "active_run_id": None,
        "active_run_mode": "Single message",
        "last_batch_result": None,
        "batch_running": False,
        "batch_status": "",
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
    chat_backend = normalize_chat_backend(chat_backend)
    if chat_backend == "TransformerLens":
        return f"Trace model: {summary['model_name']}"
    alias = str(st.session_state.get("llama_model_alias", "")).strip()
    if chat_backend == "llama.cpp normal":
        status = st.session_state.llama_status
        model = alias or (", ".join(status.models) if status and status.online and status.models else st.session_state.llama_url)
        return f"Local GGUF normal: {model}"
    status = st.session_state.llama_glass_status
    model = alias or (", ".join(status.models) if status and status.online and status.models else st.session_state.llama_glass_url)
    return f"Local GGUF steered: {model}"


def local_steering_capability(chat_backend: str) -> tuple[bool, str]:
    chat_backend = normalize_chat_backend(chat_backend)
    if chat_backend == "llama.cpp normal":
        status = st.session_state.get("llama_status")
    elif chat_backend == "llama.cpp glass":
        status = st.session_state.get("llama_glass_status")
    else:
        return False, "TransformerLens steering uses activation hooks."

    if not status or not getattr(status, "online", False):
        return False, "Check the selected llama.cpp server first."
    if not getattr(status, "glass_available", False):
        return False, "Server does not expose /glass-skull/info."
    if getattr(status, "steering_supported", False) or per_request_steering_supported(getattr(status, "glass_info", {})):
        return True, "Per-request control-vector steering is advertised."
    return False, "Server does not advertise capabilities.steering.per_request.supported."


def selected_control_vector_payload() -> tuple[dict | None, str | None]:
    vectors = list_control_vectors()
    selected_name = st.session_state.get("llama_control_vector", "")
    selected = next((row for row in vectors if row.get("name") == selected_name and row.get("vector_exists")), None)
    if not selected:
        return None, "Choose a generated control vector in Local Alter."
    try:
        payload = build_steering_metadata(
            str(selected["vector_path"]),
            float(st.session_state.get("llama_control_strength", 1.25)),
            int(st.session_state.get("llama_control_layer_start", 1)),
            int(st.session_state.get("llama_control_layer_end", 1)),
        )
    except Exception as exc:
        return None, str(exc)
    return payload, None


def feature_names_from_rows(rows: list[dict]) -> list[str]:
    return [str(f["name"]) for f in rows if f.get("name")]


def chat_history_for_send(new_prompt: str, limit: int = 24) -> list[dict[str, str]]:
    history = []
    for message in st.session_state.get("chat_messages", [])[-limit:]:
        role = str(message.get("role", ""))
        content = str(message.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            history.append({"role": role, "content": content})
    history.append({"role": "user", "content": new_prompt})
    return history


def transcript_prompt(messages: list[dict[str, str]]) -> str:
    lines = []
    for message in messages:
        role = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{role}: {message['content']}")
    return "\n\n".join(lines) + "\n\nAssistant:"



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


def set_local_dashboard_trace(trace_payload: dict, prompt: str, backend: str, model_name: str) -> None:
    st.session_state.local_dashboard_trace = trace_payload
    st.session_state.dashboard_trace_counter = int(st.session_state.get("dashboard_trace_counter", 0)) + 1
    prompt_info = trace_payload.get("prompt", {}) if isinstance(trace_payload.get("prompt"), dict) else {}
    token_count = int(prompt_info.get("n_tokens_total", 0) or len(trace_payload.get("tokens", []) or []))
    st.session_state.local_dashboard_trace_meta = {
        "prompt": prompt,
        "backend": backend,
        "trace_model": model_name,
        "updated_at": datetime.now().strftime("%H:%M:%S"),
        "token_count": token_count,
        "run": st.session_state.dashboard_trace_counter,
    }


def single_trace_artifact_from_transformerlens(trace, dash_meta: dict) -> dict:
    run_id = str(st.session_state.get("active_run_id") or dash_meta.get("run") or "single")
    trace_rows = normalize_transformerlens_trace(
        trace,
        prompt_id=0,
        label="single",
        metadata={"run_id": run_id, "mode": "Single message"},
    )
    if not trace_rows:
        trace_rows = [trace_unavailable_row(run_id, 0, "single", "transformerlens", "activation summaries were not returned")]
    return build_run_artifact(
        run_id=run_id,
        mode="Single message",
        backend=str(dash_meta.get("backend", "TransformerLens")),
        model=str(dash_meta.get("trace_model", "")),
        prompts=[{
            "prompt_id": 0,
            "label": "single",
            "prompt": getattr(trace, "prompt", str(dash_meta.get("prompt", ""))),
            "output": "",
            "error": None,
            "elapsed_ms": None,
            "trace_rows": trace_rows,
        }],
    )


def single_trace_artifact_from_llama(trace_payload: dict, dash_meta: dict) -> dict:
    run_id = str(st.session_state.get("active_run_id") or dash_meta.get("run") or "single")
    trace_rows = normalize_llama_trace(
        trace_payload,
        prompt_id=0,
        label="single",
        metadata={"run_id": run_id, "mode": "Single message"},
    )
    if not trace_rows:
        activations = trace_payload.get("activations", {})
        reason = "activation summaries were not returned by this llama.cpp server"
        if isinstance(activations, dict) and activations.get("reason"):
            reason = str(activations["reason"])
        trace_rows = [trace_unavailable_row(run_id, 0, "single", "llama.cpp", reason)]
    return build_run_artifact(
        run_id=run_id,
        mode="Single message",
        backend=str(dash_meta.get("backend", "llama.cpp")),
        model=str(dash_meta.get("trace_model", "")),
        prompts=[{
            "prompt_id": 0,
            "label": "single",
            "prompt": str(dash_meta.get("prompt", "")),
            "output": "",
            "error": None,
            "elapsed_ms": None,
            "trace_rows": trace_rows,
        }],
    )


def workflow_source_status(sources: list[str] | None = None) -> dict[str, dict[str, str | bool]]:
    sources = sources or st.session_state.get("workflow_setup_sources", [])
    status: dict[str, dict[str, str | bool]] = {}

    if MODEL_SOURCE_LOCAL in sources:
        model_path = Path(st.session_state.get("llama_model_path", ""))
        server_path = Path(st.session_state.get("llama_server_bin", ""))
        generator_path = Path(st.session_state.get("llama_cvector_generator", ""))
        alias = str(st.session_state.get("llama_model_alias", "")).strip()
        missing = []
        optional_missing = []
        if not alias:
            missing.append("router alias")
        if not model_path.exists():
            missing.append("GGUF model")
        if not server_path.exists():
            optional_missing.append("llama-server")
        if not generator_path.exists():
            optional_missing.append("llama-cvector-generator")
        state = "ready" if not missing else "missing " + ", ".join(missing)
        if not missing and optional_missing:
            state = "ready; optional tools missing " + ", ".join(optional_missing)
        status[MODEL_SOURCE_LOCAL] = {
            "ok": not missing,
            "state": state,
            "detail": f"{alias or 'no alias'} | {model_path}",
            "requirement": "Requires a router model alias and an existing GGUF file. llama-server and llama-cvector-generator are optional setup-time tools for Local Controls.",
        }

    if MODEL_SOURCE_HF in sources:
        token_status = st.session_state.get("hf_token_status")
        requires_token = bool(st.session_state.get("workflow_hf_requires_token", False))
        has_token = bool(token_status and token_status.valid)
        ok = has_token if requires_token else True
        if requires_token:
            state = "ready" if has_token else "needs valid token"
            detail = token_status.label() if token_status else "No validated token"
        else:
            state = "catalog ready"
            detail = token_status.label() if token_status else "Public catalog available; gated/private models need a token."
        status[MODEL_SOURCE_HF] = {
            "ok": ok,
            "state": state,
            "detail": detail,
            "requirement": "Requires transformers/accelerate for local HF loading. Gated or private Hugging Face repos require an approved read token.",
        }

    if MODEL_SOURCE_TRACE in sources:
        trace_ok = bool(st.session_state.get("model_name"))
        status[MODEL_SOURCE_TRACE] = {
            "ok": trace_ok,
            "state": "ready" if trace_ok else "needs model",
            "detail": f"Trace model: {st.session_state.get('model_name') or 'not selected'}",
            "requirement": "Requires a TransformerLens-supported model and local PyTorch runtime. First load may need network or cached weights.",
        }

    return status


def workflow_status_rows(status: dict[str, dict[str, str | bool]]) -> list[dict[str, str]]:
    return [
        {
            "source": source,
            "state": str(row.get("state", "")),
            "detail": str(row.get("detail", "")),
            "requirement": str(row.get("requirement", "")),
        }
        for source, row in status.items()
    ]


def workflow_is_ready(status: dict[str, dict[str, str | bool]]) -> bool:
    return bool(status) and all(bool(row.get("ok")) for row in status.values())


@st.dialog("Workflow setup")
def render_workflow_setup_dialog() -> None:
    st.caption("Choose the model sources this session should use. You can reopen this setup from Settings or Models.")

    local_enabled = st.checkbox(
        "Local GGUF (llama.cpp)",
        value=MODEL_SOURCE_LOCAL in st.session_state.workflow_setup_sources,
        help="Requires a model alias used in OpenAI-compatible requests, an existing .gguf model path, and llama-server. Control-vector generation requires llama-cvector-generator from a local llama.cpp build.",
    )
    hf_enabled = st.checkbox(
        "Hugging Face models",
        value=MODEL_SOURCE_HF in st.session_state.workflow_setup_sources,
        help="Requires transformers/accelerate for local loading. Public repos can be browsed without a token; gated or private repos require an approved Hugging Face read token.",
    )
    trace_enabled = st.checkbox(
        "Trace model (TransformerLens)",
        value=MODEL_SOURCE_TRACE in st.session_state.workflow_setup_sources,
        help="Requires a TransformerLens-supported model, PyTorch, and either cached model weights or network access for first load. Enables Trace, Lens, Attention, Map, activation Steer, and tensor Compare.",
    )

    selected_sources = []
    if local_enabled:
        selected_sources.append(MODEL_SOURCE_LOCAL)
    if hf_enabled:
        selected_sources.append(MODEL_SOURCE_HF)
    if trace_enabled:
        selected_sources.append(MODEL_SOURCE_TRACE)

    if local_enabled:
        st.session_state.llama_model_alias = st.text_input(
            "Local model alias / router model",
            value=st.session_state.llama_model_alias,
            help="Sent as the OpenAI-compatible `model` field for router-backed local chat. Match this to the alias exposed by your router or llama.cpp launch.",
            key="setup_llama_model_alias",
        ).strip()
        st.session_state.llama_model_path = st.text_input(
            "Local GGUF path",
            value=st.session_state.llama_model_path,
            help="Must point to an existing GGUF model file. This is also the model used by Local Alter graphs.",
            key="setup_llama_model_path",
        )
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.llama_server_bin = st.text_input(
                "llama-server",
                value=st.session_state.llama_server_bin,
                help="Required for Local GGUF normal and steered chat.",
                key="setup_llama_server_bin",
            )
        with c2:
            st.session_state.llama_cvector_generator = st.text_input(
                "llama-cvector-generator",
                value=st.session_state.llama_cvector_generator,
                help="Required only when generating control vectors in Local Alter.",
                key="setup_cvector_generator",
            )

    if hf_enabled:
        st.session_state.workflow_hf_requires_token = st.toggle(
            "Require gated/private HF access",
            value=bool(st.session_state.workflow_hf_requires_token),
            help="Turn this on when the workflow depends on gated or private Hugging Face repos. A valid approved token is then required before setup completes.",
        )
        st.session_state.hf_token = st.text_input(
            "HF token",
            value=st.session_state.get("hf_token", ""),
            type="password",
            help=HELP["hf_token"],
            key="setup_hf_token",
        )
        if st.button("Validate HF token", width="stretch"):
            st.session_state.hf_token_status = validate_token(st.session_state.hf_token)

    if trace_enabled:
        preset_options = ["custom"] + MODEL_PRESETS
        current = st.session_state.get("model_name", DEFAULT_MODEL)
        preset_index = preset_options.index(current) if current in preset_options else 0
        setup_preset = st.selectbox("Trace model preset", preset_options, index=preset_index, help=HELP["preset"], key="setup_trace_preset")
        if setup_preset == "custom":
            st.session_state.model_name = normalize_model_name(
                st.text_input("Trace model name", value=current, help="Use a TransformerLens-supported model name.", key="setup_trace_custom")
            )
        else:
            st.session_state.model_name = normalize_model_name(setup_preset)

    st.session_state.workflow_setup_sources = selected_sources
    status = workflow_source_status(selected_sources)
    st.session_state.workflow_setup_last_status = status
    if status:
        st.dataframe(pd.DataFrame(workflow_status_rows(status)), width="stretch", hide_index=True)
    else:
        st.error("Select at least one model source.")

    ready = workflow_is_ready(status)
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Save setup", type="primary", width="stretch", disabled=not ready):
            st.session_state.workflow_setup_complete = True
            st.session_state.workflow_setup_last_status = status
            st.rerun()
    with b2:
        if st.button("Skip for this session", width="stretch"):
            st.session_state.workflow_setup_complete = True
            st.session_state.workflow_setup_last_status = status
            st.rerun()


def render_models_tab() -> None:
    ui.section_header("Models", "Configured model sources for this session.")
    status = workflow_source_status()
    st.session_state.workflow_setup_last_status = status
    m1, m2 = st.columns([1, 1])
    with m1:
        if st.button("Open workflow setup", type="primary", width="stretch"):
            st.session_state.workflow_setup_complete = False
            st.rerun()
    with m2:
        st.metric("Configured sources", f"{sum(1 for row in status.values() if row.get('ok'))} / {len(status)}")

    if not status:
        ui.empty_state("No model sources selected", "Open workflow setup and select at least one source.")
        return

    st.dataframe(pd.DataFrame(workflow_status_rows(status)), width="stretch", height=220, hide_index=True)
    model_tab_names = []
    if MODEL_SOURCE_LOCAL in status:
        model_tab_names.append("Local GGUF")
    if MODEL_SOURCE_HF in status:
        model_tab_names.append("Hugging Face")
    if MODEL_SOURCE_TRACE in status:
        model_tab_names.append("Trace model")
    model_tabs = dict(zip(model_tab_names, st.tabs(model_tab_names)))

    if MODEL_SOURCE_LOCAL in status:
        with model_tabs["Local GGUF"]:
            preflight = preflight_control_vector_run(
                st.session_state.llama_model_path,
                None,
                None,
                st.session_state.llama_cvector_generator,
                st.session_state.llama_server_bin,
            )
            ui.property_list(
                [
                    ("alias", st.session_state.get("llama_model_alias", "")),
                    ("model_path", st.session_state.llama_model_path),
                    ("architecture", preflight.model_architecture or "unknown"),
                    ("normal_server_url", st.session_state.llama_url),
                    ("steered_server_url", st.session_state.llama_glass_url),
                ]
            )
            if preflight.warnings:
                for warning in preflight.warnings:
                    st.warning(warning)
            if preflight.errors:
                for error in preflight.errors:
                    st.error(error)

    if MODEL_SOURCE_HF in status:
        with model_tabs["Hugging Face"]:
            token_status = st.session_state.get("hf_token_status")
            st.caption(token_status.label() if token_status else "HF token has not been validated.")
            st.dataframe(pd.DataFrame(registry_as_dicts()), width="stretch", height=360, hide_index=True)

    if MODEL_SOURCE_TRACE in status:
        with model_tabs["Trace model"]:
            ui.property_list(
                [
                    ("model", str(summary["model_name"]) if "summary" in globals() else str(st.session_state.model_name)),
                    ("device", str(summary["device"]) if "summary" in globals() else "pending load"),
                    ("layers", str(summary["layers"]) if "summary" in globals() else "pending load"),
                    ("d_model", str(summary["d_model"]) if "summary" in globals() else "pending load"),
                ]
            )


def render_settings_tab(summary: dict) -> None:
    ui.section_header("Settings", "Master configuration for model sources and the workflows built from them.")

    settings_tab_names = ["Local"]
    if hf_source_enabled:
        settings_tab_names.append("HF")
    if trace_source_enabled:
        settings_tab_names.append("Trace")
    settings_tab_names.append("Session")
    settings_tabs = dict(zip(settings_tab_names, st.tabs(settings_tab_names)))

    with settings_tabs["Local"]:
        st.markdown("#### Local model")
        st.session_state.llama_model_alias = st.text_input(
            "Alias / router model name",
            value=st.session_state.llama_model_alias,
            help="Sent as the OpenAI-compatible `model` field for Local GGUF chat and fuzzing. Set this to the model name your router expects.",
            key="settings_llama_model_alias",
        ).strip()
        st.session_state.llama_model_path = st.text_input(
            "GGUF model path",
            value=st.session_state.llama_model_path,
            help="Used for Local Alter preflight, tensor graphs, and llama-server launch commands.",
            key="settings_llama_model_path",
        )

        l1, l2 = st.columns(2)
        with l1:
            st.session_state.llama_url = st.text_input("Normal server URL", value=st.session_state.llama_url, help=HELP["llama_url"], key="settings_llama_url")
        with l2:
            st.session_state.llama_glass_url = st.text_input(
                "Steered server URL",
                value=st.session_state.llama_glass_url,
                help="Use this for a llama.cpp server launched with control-vector flags.",
                key="settings_llama_glass_url",
            )

        b1, b2 = st.columns(2)
        with b1:
            st.session_state.llama_server_bin = st.text_input(
                "llama-server",
                value=st.session_state.llama_server_bin,
                help="Required for normal and steered Local GGUF chat.",
                key="settings_llama_server_bin",
            )
        with b2:
            st.session_state.llama_cvector_generator = st.text_input(
                "llama-cvector-generator",
                value=st.session_state.llama_cvector_generator,
                help="Required for generating control vectors in Local Alter.",
                key="settings_llama_cvector_generator",
            )

        lc1, lc2 = st.columns(2)
        with lc1:
            if st.button("Check normal server", width="stretch", help="Checks /v1/models and optional /glass-skull/info.", key="settings_check_normal_server"):
                st.session_state.llama_status = check_server(
                    st.session_state.llama_url,
                    model_alias=st.session_state.get("llama_model_alias", ""),
                )
        with lc2:
            if st.button("Check steered server", width="stretch", help="Checks the steered llama.cpp server on its separate port.", key="settings_check_steered_server"):
                st.session_state.llama_glass_status = check_server(
                    st.session_state.llama_glass_url,
                    model_alias=st.session_state.get("llama_model_alias", ""),
                )

        st.markdown(
            ui.server_health_inline("Normal", st.session_state.llama_url, st.session_state.llama_status)
            + ui.server_health_inline("Steered", st.session_state.llama_glass_url, st.session_state.llama_glass_status),
            unsafe_allow_html=True,
        )

        local_preflight = preflight_control_vector_run(
            st.session_state.llama_model_path,
            None,
            None,
            st.session_state.llama_cvector_generator,
            st.session_state.llama_server_bin,
        )
        ui.property_list(
            [
                ("alias", st.session_state.llama_model_alias or "missing"),
                ("architecture", local_preflight.model_architecture or "unknown"),
                ("model_exists", str(Path(st.session_state.llama_model_path).exists())),
                ("llama_server_bin", str(Path(st.session_state.llama_server_bin).exists())),
                ("cvector_generator", str(Path(st.session_state.llama_cvector_generator).exists())),
            ]
        )
        if local_preflight.warnings:
            for warning in local_preflight.warnings:
                st.warning(warning)
        if local_preflight.errors:
            for error in local_preflight.errors:
                st.error(error)

    if hf_source_enabled:
        with settings_tabs["HF"]:
            render_hf_catalog_panel()

    if trace_source_enabled:
        with settings_tabs["Trace"]:
            st.markdown("#### Trace model")
            preset_options = ["custom"] + MODEL_PRESETS
            current_model = st.session_state.get("model_name", DEFAULT_MODEL)
            preset_index = preset_options.index(current_model) if current_model in preset_options else 0
            s1, s2 = st.columns([2, 1])
            with s1:
                preset = st.selectbox("Preset", preset_options, index=preset_index, help=HELP["preset"], key="settings_model_preset")
                if preset != "custom":
                    pending_model_name = normalize_model_name(preset)
                else:
                    pending_model_name = normalize_model_name(
                        st.text_input("TransformerLens model", value=current_model, help="Use a TransformerLens-supported model name.", key="settings_model_custom")
                    )
            with s2:
                pending_device_choice = st.selectbox(
                    "Device",
                    ["auto", "cpu", "cuda"],
                    index=["auto", "cpu", "cuda"].index(st.session_state.get("device_choice", "auto")),
                    help=HELP["device"],
                    key="settings_device_choice",
                )

            if st.button("Load trace model", type="primary", width="stretch", help="Reloads the selected TransformerLens model and clears the current trace.", key="settings_load_trace_model"):
                st.session_state.model_name = pending_model_name
                st.session_state.device_choice = pending_device_choice
                st.session_state.trace = None
                st.session_state.last_output = ""
                st.session_state.last_comparison = None
                st.session_state.dashboard_trace = None
                st.session_state.dashboard_trace_meta = {}
                load_hooked_model.clear()
                st.rerun()

            ui.property_list(
                [
                    ("loaded_model", str(summary["model_name"])),
                    ("device", str(summary["device"])),
                    ("layers", str(summary["layers"])),
                    ("d_model", str(summary["d_model"])),
                    ("params", f"{summary['parameters']:,}"),
                ]
            )

    with settings_tabs["Session"]:
        st.markdown("#### Session")
        ss1, ss2, ss3, ss4 = st.columns(4)
        with ss1:
            if st.button("New chat", width="stretch", help="Archives the current chat transcript and clears visible chat history.", key="settings_new_chat"):
                save_chat(st.session_state.chat_messages)
                st.session_state.chat_messages = []
                st.session_state.last_output = ""
                st.session_state.last_loaded_chat = ""
                st.rerun()
        with ss2:
            chats = list_chats()
            chat_labels = [f"{row['created_at'][:19]} · {row['label']}" for row in chats] if chats else []
            selected_chat_label = st.selectbox("Saved chats", chat_labels, disabled=not chats, key="settings_saved_chat")
            if st.button("Load selected chat", width="stretch", disabled=not chats, key="settings_load_selected_chat"):
                selected_id = chats[chat_labels.index(selected_chat_label)]["id"]
                st.session_state.chat_messages = load_chat(selected_id)
                st.session_state.last_loaded_chat = selected_id
                st.rerun()
        with ss3:
            if st.button("Clear trace", width="stretch", help="Clears the currently cached activations.", key="settings_clear_trace"):
                st.session_state.trace = None
                st.session_state.dashboard_trace = None
                st.session_state.dashboard_trace_meta = {}
                st.session_state.local_dashboard_trace = None
                st.session_state.local_dashboard_trace_meta = {}
                st.rerun()
        with ss4:
            if st.button("Clear comparison", width="stretch", help="Clears the last normal-vs-steered comparison.", key="settings_clear_comparison"):
                st.session_state.last_comparison = None
                st.rerun()
        if st.button("Open workflow setup", type="primary", width="stretch", key="settings_open_workflow_setup"):
            st.session_state.workflow_setup_complete = False
            st.rerun()



def render_capability_warning(chat_backend: str) -> dict[str, bool]:
    chat_backend = normalize_chat_backend(chat_backend)
    caps = capabilities_for_backend(chat_backend)
    if "llama.cpp" in chat_backend.lower():
        st.info(
            "Local GGUF chat uses the alias and server URLs from Settings > Local. Per-request control-vector steering is enabled only when /glass-skull/info advertises it; otherwise use the startup control-vector server fallback from Local Alter. Logit Lens, Attention, Map, and full activation hooks remain TransformerLens-only."
        )
    return caps


def render_llama_control_panel() -> None:
    ui.section_header("Local Alter", "Learn a behavior vector from local GGUF prompts, then launch normal and steered llama.cpp servers.")
    st.caption(HELP["llama_control"])

    st.markdown("#### 1. Local GGUF and tools")
    ui.property_list([("alias", st.session_state.get("llama_model_alias", "")), ("configured_in", "Settings > Local")])
    m1, m2 = st.columns([2, 1])
    with m1:
        st.session_state.llama_model_path = st.text_input(
            "GGUF model path",
            value=st.session_state.llama_model_path,
            placeholder="/path/to/qwen3.6-35b.gguf",
            help="Your local GGUF model. This is passed to -m.",
        )
    with m2:
        st.session_state.llama_control_port = st.number_input("Steered server port", 1, 65535, int(st.session_state.llama_control_port), 1)

    b1, b2 = st.columns(2)
    with b1:
        st.session_state.llama_cvector_generator = st.text_input(
            "llama-cvector-generator",
            value=st.session_state.llama_cvector_generator,
            help="Path to llama-cvector-generator.",
        )
    with b2:
        st.session_state.llama_server_bin = st.text_input(
            "llama-server",
            value=st.session_state.llama_server_bin,
            help="Path to llama-server.",
        )

    st.markdown("#### 2. Positive and negative prompt sets")
    with st.expander("Create or update a prompt set", expanded=False):
        set_name = st.text_input("Control set name", value="qwen_behavior_vector")
        positive_text = st.text_area(
            "Positive prompts",
            value="Answer with crisp, direct technical reasoning.\nPrefer concrete implementation details over generic advice.",
            height=100,
            key="llama_positive_text",
        )
        negative_text = st.text_area(
            "Negative prompts",
            value="Answer vaguely with broad motivational statements.\nAvoid implementation details.",
            height=100,
            key="llama_negative_text",
        )
        if st.button("Save control set", width="stretch"):
            try:
                paths = write_control_set(set_name, positive_text, negative_text)
                st.session_state.llama_control_set = paths.name
                st.success(f"Saved `{paths.name}`")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    control_sets = list_control_sets()
    if control_sets:
        labels = [row["name"] for row in control_sets]
        selected_index = labels.index(st.session_state.llama_control_set) if st.session_state.llama_control_set in labels else 0
        selected_label = st.selectbox("Control set", labels, index=selected_index)
        st.session_state.llama_control_set = selected_label
        selected_set = control_sets[labels.index(selected_label)]
        ui.property_list(
            [
                ("positive", str(selected_set["positive_path"])),
                ("negative", str(selected_set["negative_path"])),
                ("pairs", f"{selected_set.get('positive_count', '?')} / {selected_set.get('negative_count', '?')}"),
            ]
        )
    else:
        selected_set = None
        st.info("No GGUF control sets yet. Save positive/negative prompts above.")

    st.markdown("#### 3. Preflight")
    preflight = preflight_control_vector_run(
        st.session_state.llama_model_path,
        (selected_set or {}).get("positive_path"),
        (selected_set or {}).get("negative_path"),
        st.session_state.llama_cvector_generator,
        st.session_state.llama_server_bin,
    )
    st.session_state.llama_last_preflight = {
        "errors": preflight.errors,
        "warnings": preflight.warnings,
        "model_architecture": preflight.model_architecture,
    }
    checks_df = pd.DataFrame([{"check": c.name, "status": c.status, "detail": c.detail} for c in preflight.checks])
    if not checks_df.empty:
        st.dataframe(checks_df, width="stretch", height=180, hide_index=True)
    if preflight.errors:
        for err in preflight.errors:
            st.error(err)
    if preflight.warnings:
        for warning in preflight.warnings:
            st.warning(warning)

    metadata = preflight.model_metadata or {}
    arch = preflight.model_architecture or "unknown"
    layer_key = f"{arch}.block_count" if arch != "unknown" else ""
    if metadata:
        meta_rows = [
            ("architecture", arch),
            ("block_count", str(metadata.get(layer_key, metadata.get("block_count", "unknown")))),
            ("tensor_count", str(metadata.get("_tensor_count", "unknown"))),
            ("metadata_kv_count", str(metadata.get("_metadata_kv_count", "unknown"))),
        ]
        ui.property_list(meta_rows)

    st.markdown("#### 4. Local model graphs")
    if preflight.errors and any("Model path does not exist" in err for err in preflight.errors):
        ui.empty_state("No local model graph", "Choose an existing GGUF model path to render local tensor graphs.")
    else:
        try:
            local_tensors_df = pd.DataFrame(read_gguf_tensor_index(st.session_state.llama_model_path))
            if local_tensors_df.empty:
                ui.empty_state("No tensor index", "The GGUF file did not expose tensor entries.")
            else:
                total_elements = int(local_tensors_df["elements"].sum())
                ui.property_list(
                    [
                        ("graph_model", st.session_state.llama_model_path),
                        ("tensor_entries", f"{len(local_tensors_df):,}"),
                        ("tensor_elements", f"{total_elements:,}"),
                    ]
                )
                lg1, lg2 = st.columns(2)
                with lg1:
                    plot_if_present(gguf_tensors_per_layer_fig(local_tensors_df), key_hint="gguf_layer")
                with lg2:
                    plot_if_present(gguf_tensors_by_component_fig(local_tensors_df), key_hint="gguf_component")
                lg3, lg4 = st.columns(2)
                with lg3:
                    plot_if_present(gguf_tensor_dtype_fig(local_tensors_df), key_hint="gguf_dtype")
                with lg4:
                    plot_if_present(gguf_tensor_shape_scatter_fig(local_tensors_df), key_hint="gguf_shape")
                with st.expander("Local GGUF tensor index", expanded=False):
                    st.dataframe(
                        local_tensors_df[["index", "name", "shape", "dtype", "elements", "offset"]],
                        width="stretch",
                        height=260,
                        hide_index=True,
                    )
        except Exception as exc:
            st.warning(f"Could not render local GGUF graphs: {exc}")

    st.markdown("#### 5. Generate control vector")
    gen_name = st.text_input("Vector name", value=st.session_state.llama_control_set or "qwen_behavior_vector")
    gc1, gc2 = st.columns(2)
    with gc1:
        cvec_method = st.selectbox("Method", ["mean"], help="llama.cpp cvector-generator method.")
    with gc2:
        st.session_state.llama_cvector_explicit_ngl = st.toggle(
            "Set generator GPU layers",
            value=bool(st.session_state.llama_cvector_explicit_ngl),
            help="Off uses llama.cpp auto fit behavior and omits -ngl.",
        )
    advanced = st.expander("Advanced generator options", expanded=False)
    with advanced:
        if st.session_state.llama_cvector_explicit_ngl:
            st.session_state.llama_cvector_ngl = st.number_input("Generator -ngl", 0, 999, int(st.session_state.llama_cvector_ngl), 1)
        st.session_state.llama_cvector_fit = st.selectbox(
            "Fit",
            ["auto", "on", "off"],
            index=["auto", "on", "off"].index(st.session_state.llama_cvector_fit) if st.session_state.llama_cvector_fit in ["auto", "on", "off"] else 0,
            help="auto omits --fit; off adds --fit off.",
        )
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            st.session_state.llama_cvector_ctx_size = st.number_input("Context size", 0, 262144, int(st.session_state.llama_cvector_ctx_size), 1024, help="0 omits -c.")
        with ac2:
            st.session_state.llama_cvector_pca_batch = st.number_input("--pca-batch", 0, 1000000, int(st.session_state.llama_cvector_pca_batch), 10, help="0 omits --pca-batch.")
        with ac3:
            st.session_state.llama_cvector_pca_iter = st.number_input("--pca-iter", 0, 1000000, int(st.session_state.llama_cvector_pca_iter), 100, help="0 omits --pca-iter.")

    cvec_ngl = int(st.session_state.llama_cvector_ngl) if st.session_state.llama_cvector_explicit_ngl else None
    cvec_fit = None if st.session_state.llama_cvector_fit == "auto" else st.session_state.llama_cvector_fit
    cvec_ctx = int(st.session_state.llama_cvector_ctx_size) if int(st.session_state.llama_cvector_ctx_size) > 0 else None
    cvec_pca_batch = int(st.session_state.llama_cvector_pca_batch) if int(st.session_state.llama_cvector_pca_batch) > 0 else None
    cvec_pca_iter = int(st.session_state.llama_cvector_pca_iter) if int(st.session_state.llama_cvector_pca_iter) > 0 else None

    if selected_set and st.session_state.llama_model_path:
        preview_cmd = build_cvector_command(
            st.session_state.llama_model_path,
            selected_set["positive_path"],
            selected_set["negative_path"],
            f"data/control_vectors/{gen_name}.gguf",
            st.session_state.llama_cvector_generator,
            cvec_method,
            ngl=cvec_ngl,
            fit=cvec_fit,
            ctx_size=cvec_ctx,
            pca_batch=cvec_pca_batch,
            pca_iter=cvec_pca_iter,
        )
        st.code(shell_join(preview_cmd), language="bash")
        if "-ngl" not in preview_cmd:
            st.caption("Generator GPU layers: auto. The preview intentionally omits `-ngl`.")

    can_generate = bool(selected_set and st.session_state.llama_model_path and not preflight.errors)
    if st.button("Generate control vector", type="primary", width="stretch", disabled=not can_generate):
        try:
            with st.spinner("Running llama-cvector-generator..."):
                meta = generate_control_vector(
                    gen_name,
                    st.session_state.llama_model_path,
                    selected_set["positive_path"],
                    selected_set["negative_path"],
                    st.session_state.llama_cvector_generator,
                    cvec_method,
                    ngl=cvec_ngl,
                    fit=cvec_fit,
                    ctx_size=cvec_ctx,
                    pca_batch=cvec_pca_batch,
                    pca_iter=cvec_pca_iter,
                    compatibility_warnings=preflight.warnings,
                    model_architecture=preflight.model_architecture,
                )
            st.session_state.llama_control_vector = meta.name
            st.session_state.llama_last_cvector_failure = None
            st.success(f"Generated `{meta.vector_path}`")
            st.rerun()
        except ControlVectorRunError as exc:
            failure_payload = asdict(exc.metadata) if hasattr(exc.metadata, "__dataclass_fields__") else {}
            if failure_payload.get("vector_path") is not None:
                failure_payload["vector_path"] = str(failure_payload["vector_path"])
            st.session_state.llama_last_cvector_failure = failure_payload
            st.error(exc.failure.cause)
            st.warning(exc.failure.recommendation)
            for warning in exc.failure.warnings:
                st.warning(warning)
            with st.expander("Full generator stdout/stderr", expanded=True):
                st.code(exc.metadata.stdout or "(no stdout)", language="text")
                st.code(exc.metadata.stderr or "(no stderr)", language="text")
        except Exception as exc:
            st.error(str(exc))

    failure_payload = st.session_state.get("llama_last_cvector_failure")
    if failure_payload:
        with st.expander("Last failed attempt metadata", expanded=False):
            st.json(failure_payload)

    st.markdown("#### 6. Configure steering and launch")
    vectors = list_control_vectors()
    usable_vectors = [row for row in vectors if row.get("vector_exists") and row.get("returncode") in (None, 0)]
    failed_vectors = [row for row in vectors if not row.get("vector_exists") or row.get("returncode") not in (None, 0)]
    if usable_vectors:
        vector_labels = [row["name"] for row in usable_vectors]
        vector_index = vector_labels.index(st.session_state.llama_control_vector) if st.session_state.llama_control_vector in vector_labels else 0
        vector_label = st.selectbox("Control vector", vector_labels, index=vector_index)
        st.session_state.llama_control_vector = vector_label
        selected_vector = usable_vectors[vector_labels.index(vector_label)]
        st.caption(f"Vector: `{selected_vector['vector_path']}`")
    else:
        selected_vector = None
        st.info("No generated `.gguf` control vectors yet.")
    if failed_vectors:
        with st.expander("Failed vector attempts", expanded=False):
            st.dataframe(pd.DataFrame(failed_vectors), width="stretch", height=160, hide_index=True)

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.session_state.llama_control_strength = st.slider("Strength", -5.0, 5.0, float(st.session_state.llama_control_strength), 0.05)
    with sc2:
        st.session_state.llama_control_layer_start = st.number_input("Layer start", 0, 200, int(st.session_state.llama_control_layer_start), 1)
    with sc3:
        st.session_state.llama_control_layer_end = st.number_input("Layer end", 0, 200, int(st.session_state.llama_control_layer_end), 1)
    pc1, pc2 = st.columns(2)
    with pc1:
        normal_port = st.number_input("Normal server port", 1, 65535, 8080, 1)
    with pc2:
        server_ngl = st.number_input("Server GPU layers", 0, 999, 999, 1, key="llama_server_ngl")
    ctx_size = st.number_input("Context size", 0, 262144, 32768, 1024, help="0 omits -c.")
    st.session_state.llama_control_extra_args = st.text_input(
        "Extra llama-server args",
        value=st.session_state.llama_control_extra_args,
        help="Parsed with shell-like quoting and appended to the command.",
    )

    model_for_server = st.session_state.llama_model_path or (selected_vector or {}).get("model_path", "")
    vector_path = (selected_vector or {}).get("vector_path")
    if model_for_server:
        normal_cmd = build_llama_server_command(
            model_for_server,
            None,
            server_path=st.session_state.llama_server_bin,
            port=int(normal_port),
            ngl=int(server_ngl),
            ctx_size=int(ctx_size) if int(ctx_size) > 0 else None,
            extra_args=st.session_state.llama_control_extra_args,
            alias=st.session_state.get("llama_model_alias", ""),
        )
        server_cmd = build_llama_server_command(
            model_for_server,
            vector_path,
            st.session_state.llama_control_strength,
            st.session_state.llama_control_layer_start,
            st.session_state.llama_control_layer_end,
            st.session_state.llama_server_bin,
            port=int(st.session_state.llama_control_port),
            ngl=int(server_ngl),
            ctx_size=int(ctx_size) if int(ctx_size) > 0 else None,
            extra_args=st.session_state.llama_control_extra_args,
            alias=st.session_state.get("llama_model_alias", ""),
        )
        st.caption("Normal server")
        st.code(shell_join(normal_cmd), language="bash")
        st.caption("Steered server")
        st.code(shell_join(server_cmd), language="bash")
        st.caption("Start both servers, set Settings > Local URLs to the matching ports, then compare `Local GGUF normal` and `Local GGUF steered` in Chat.")

    with st.expander("Experimental llama.cpp compatibility patch", expanded=False):
        st.warning("This path is disabled by default. Use it only if stock llama-cvector-generator still fails on Qwen3.6 MoE/MTP.")
        st.write("Target file: `/home/dsmason321/llama.cpp/tools/cvector-generator/cvector-generator.cpp`")
        st.write("Patch plan: add debug output around expected layer count versus captured `l_out` tensors, then adapt the generator to use the captured layer count for Qwen3.6 MoE/MTP instead of asserting `n_layers - 1`.")
        st.code(
            "cd /home/dsmason321/llama.cpp\n"
            "cmake --build build --target llama-cvector-generator llama-server",
            language="bash",
        )



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
if not st.session_state.workflow_setup_complete:
    render_workflow_setup_dialog()
    st.stop()

model_name = st.session_state.model_name
device_choice = st.session_state.device_choice
enabled_sources = list(st.session_state.get("workflow_setup_sources", [MODEL_SOURCE_LOCAL]))
trace_source_enabled = source_trace_enabled(enabled_sources)
hf_source_enabled = source_hf_enabled(enabled_sources)
lab_source_enabled = source_lab_enabled(enabled_sources)
model = None
summary = {
    "model_name": model_name,
    "device": "not loaded",
    "layers": 1,
    "heads": 1,
    "d_model": 0,
    "d_head": 0,
    "d_mlp": 0,
    "vocab_size": 0,
    "parameters": 0,
    "dtype": "",
}
if trace_source_enabled:
    model = load_hooked_model(model_name, device_choice=device_choice)
    summary = model_summary(model)
expected_dim = int(summary["d_model"])
stored_chat_backend_label = chat_backend_display(
    normalize_chat_backend(st.session_state.get("chat_backend_label", CHAT_BACKEND_LOCAL_NORMAL))
)
visible_chat_backend_options = [CHAT_BACKEND_LOCAL_NORMAL, CHAT_BACKEND_LOCAL_STEERED]
if trace_source_enabled:
    visible_chat_backend_options.append(CHAT_BACKEND_TRACE)
if stored_chat_backend_label not in visible_chat_backend_options:
    stored_chat_backend_label = CHAT_BACKEND_LOCAL_NORMAL
st.session_state.chat_backend_label = stored_chat_backend_label
active_chat_backend = normalize_chat_backend(stored_chat_backend_label)
local_model_context = local_gguf_context(
    st.session_state.get("llama_model_path", ""),
    st.session_state.get("llama_model_alias", ""),
    active_chat_backend,
) if "llama.cpp" in active_chat_backend.lower() else None

all_features = list_features(include_missing=False)
compatible_feature_rows = compatible_features(expected_dim) if trace_source_enabled and expected_dim else []
all_feature_names = feature_names_from_rows(all_features)
compatible_feature_names = feature_names_from_rows(compatible_feature_rows)
params_df = parameter_table(model) if model is not None else pd.DataFrame()

# ===========================================================================
# TOP HUD — prominent system state at a glance
# ===========================================================================
trace_active = st.session_state.get("local_dashboard_trace") is not None if local_model_context else st.session_state.trace is not None
active_model_name = (
    f"Local: {local_model_context['display_name']}"
    if local_model_context
    else str(summary["model_name"])
)
hud_stats = (
    local_model_context["hud_stats"]
    if local_model_context
    else [
        ("Layers", str(summary["layers"])),
        ("d_model", str(summary["d_model"])),
        ("Heads", str(summary["heads"])),
        ("Params", f"{summary['parameters'] / 1e6:.0f}M"),
        ("Device", str(summary["device"])),
    ]
)
hud_subtitle = (
    "Local GGUF metadata, tensor index, and llama.cpp trace endpoint status."
    if local_model_context
    else "Probe, trace, and compare transformer behavior — no magic crystals."
)
pills = [
    ui.pill(f"Model: {active_model_name}", ui.GREEN),
    ui.pill(f"Backend: {active_chat_backend}", ui.TEAL if local_model_context else ui.PURPLE),
    ui.pill("Tracing active" if trace_active else "Trace idle", ui.AMBER if trace_active else ui.SLATE, pulse=trace_active),
    ui.pill("Normal server", ui.server_status_color(st.session_state.llama_status)),
    ui.pill("Glass server", ui.server_status_color(st.session_state.llama_glass_status)),
    ui.pill("HF token", ui.GREEN if (st.session_state.get("hf_token_status") and st.session_state.hf_token_status.valid) else ui.SLATE),
]
ui.hud(
    title="Operation Glass Skull",
    subtitle=hud_subtitle,
    stats=hud_stats,
    pills_html="".join(pills),
)

# ===========================================================================
# MAIN PANELS
# ===========================================================================
tab_names = LOCAL_TABS + (["Lab"] if lab_source_enabled else [])
tab_objects = st.tabs(tab_names)
tabs = dict(zip(tab_names, tab_objects))
tab_chat = tabs["Chat"]
tab_dash = tabs["Dashboard"]
tab_local_alter = tabs["Local Controls"]
tab_anatomy = tabs["Anatomy / Logs"]
tab_models = tabs["Models"]
tab_settings = tabs["Settings"]
tab_lab = tabs.get("Lab")

# ------------------------------------------------------------- Dashboard ----
with tab_dash:
    if local_model_context:
        ui.section_header("Local GGUF model", "Architecture metadata and tensor-index summary read from the configured GGUF file.")
        ui.property_list(local_model_context["property_rows"])
        for err in local_model_context["errors"]:
            st.warning(err)
        local_tensors_df = local_model_context["tensors_df"]
        if local_tensors_df.empty:
            ui.empty_state("No local tensor graph", "The configured GGUF file did not expose a readable tensor index.")
        else:
            g1, g2 = st.columns(2)
            with g1:
                plot_if_present(gguf_tensors_per_layer_fig(local_tensors_df), key_hint="dash_gguf_layer")
            with g2:
                plot_if_present(gguf_tensors_by_component_fig(local_tensors_df), key_hint="dash_gguf_component")
            g3, g4 = st.columns(2)
            with g3:
                plot_if_present(gguf_tensor_dtype_fig(local_tensors_df), key_hint="dash_gguf_dtype")
            with g4:
                plot_if_present(gguf_tensor_shape_scatter_fig(local_tensors_df), key_hint="dash_gguf_shape")
            with st.expander("Local GGUF tensor index", expanded=False):
                columns = [col for col in ["index", "name", "shape", "dtype", "elements", "offset"] if col in local_tensors_df.columns]
                st.dataframe(local_tensors_df[columns], width="stretch", height=260, hide_index=True)
    else:
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
    if local_model_context:
        ui.section_header("Live local trace", "Derived from the patched llama.cpp Glass Skull endpoint when available.")
        local_trace = st.session_state.get("local_dashboard_trace")
        dash_meta = st.session_state.get("local_dashboard_trace_meta", {}) or {}
        if not local_trace:
            ui.empty_state(
                "No local trace captured yet",
                "Send a message in the Chat tab with tracing enabled. If the server lacks /glass-skull/trace, chat will still work and this panel will stay empty.",
            )
        else:
            prompt_info = local_trace.get("prompt", {}) if isinstance(local_trace.get("prompt"), dict) else {}
            traces = prompt_info.get("traces", []) if isinstance(prompt_info.get("traces", []), list) else []
            ui.property_list([
                ("mode", str(st.session_state.get("active_run_mode", "Single message"))),
                ("run_id", str(st.session_state.get("active_run_id") or "-")),
                ("updated", str(dash_meta.get("updated_at", "current run"))),
                ("backend", str(dash_meta.get("backend", "unknown"))),
                ("trace_model", str(dash_meta.get("trace_model", local_model_context["display_name"]))),
                ("tokens", str(dash_meta.get("token_count", prompt_info.get("n_tokens_total", "-")))),
                ("positions", str(prompt_info.get("n_positions_total", "-"))),
                ("run", str(dash_meta.get("run", "-"))),
            ])
            st.code(str(dash_meta.get("prompt", "")))
            if traces:
                trace_rows = []
                piece_rows = []
                for prompt_index, prompt_trace in enumerate(traces):
                    token_ids = prompt_trace.get("tokens", []) or []
                    trace_rows.append({
                        "prompt_index": prompt_index,
                        "tokens": len(token_ids),
                        "positions": prompt_trace.get("n_positions"),
                        "contains_media": prompt_trace.get("contains_media"),
                    })
                    for token_index, token in enumerate(token_ids):
                        piece = ""
                        pieces = prompt_trace.get("pieces", [])
                        if isinstance(pieces, list) and token_index < len(pieces) and isinstance(pieces[token_index], dict):
                            piece = pieces[token_index].get("piece", "")
                        piece_rows.append({
                            "prompt_index": prompt_index,
                            "token_index": token_index,
                            "token_id": token,
                            "piece": piece,
                        })
                st.dataframe(pd.DataFrame(trace_rows), width="stretch", hide_index=True)
                if piece_rows:
                    with st.expander("Prompt token pieces", expanded=False):
                        st.dataframe(pd.DataFrame(piece_rows), width="stretch", height=260, hide_index=True)

            local_artifact = single_trace_artifact_from_llama(local_trace, dash_meta)
            local_path_df = activation_path_df(local_artifact)
            available_local = local_path_df[local_path_df["trace_available"] != False] if not local_path_df.empty else pd.DataFrame()
            if not available_local.empty:
                plot_if_present(activation_path_graph(local_path_df), key_hint="local_activation_path")
            else:
                unavailable = local_path_df["unavailable_reason"].dropna().astype(str).iloc[0] if not local_path_df.empty else "activation summaries were not returned by this llama.cpp server"
                ui.empty_state("Activation path unavailable", unavailable)
                activations = local_trace.get("activations", {})
                if isinstance(activations, dict) and activations.get("reason"):
                    st.caption(f"Activations unavailable: {activations['reason']}")
                next_tokens = local_trace.get("next_tokens", {})
                if isinstance(next_tokens, dict) and next_tokens.get("reason"):
                    st.caption(f"Next-token logits unavailable: {next_tokens['reason']}")
    else:
        ui.section_header("Live activations", "Derived from the most recent traced prompt. Scroll for the full picture.")
        trace = st.session_state.get("dashboard_trace") or st.session_state.get("trace")
        dash_meta = st.session_state.get("dashboard_trace_meta", {}) or {}
        if trace is None:
            ui.empty_state("No trace captured yet", "Send a message in the Chat tab with tracing enabled to populate these graphs.")
        else:
            ui.property_list([
                ("mode", str(st.session_state.get("active_run_mode", "Single message"))),
                ("run_id", str(st.session_state.get("active_run_id") or "-")),
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

            trace_artifact = single_trace_artifact_from_transformerlens(trace, dash_meta)
            trace_path_df = activation_path_df(trace_artifact)
            if trace_path_df.empty or trace_path_df[trace_path_df["trace_available"] != False].empty:
                ui.empty_state("Activation path unavailable", "The trace did not include activation summaries.")
            else:
                plot_if_present(activation_path_graph(trace_path_df), key_hint="trace_activation_path")

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

# ---------------------------------------------------------- Local Alter ----
with tab_local_alter:
    render_llama_control_panel()

# ------------------------------------------------------------------ Chat ----
with tab_chat:
    ui.section_header("Chat", "Talk to the model and capture activations as you go.")

    chat_mode = st.segmented_control(
        "Mode",
        ["Single message", "Batch run"],
        default=st.session_state.get("active_run_mode", "Single message"),
        help="Single message sends one prompt to chat. Batch run sends pasted, repeated, or uploaded prompts as one run.",
        key="chat_mode",
    )
    st.session_state.active_run_mode = chat_mode

    with st.form("chat_form", clear_on_submit=False, border=False):
        prompt = st.text_area("Message", value="The cat sat on the", height=110, help="Text sent to the selected backend. In Batch run, this can also be repeated N times.", key="chat_prompt")
        batch_payload = ""
        batch_repeat_count = 1
        if chat_mode == "Batch run":
            batch_payload = st.text_area(
                "Pasted payloads",
                value="",
                height=120,
                help="One prompt per non-empty line. These prompts are combined with repeats and optional file upload into a single batch run.",
                key="chat_batch_payload",
            )
            batch_repeat_count = st.number_input(
                "Repeat current message",
                min_value=0,
                max_value=1000,
                value=1,
                step=1,
                help="Adds the current Message to this batch N times. Set to 0 to use only pasted or uploaded prompts.",
                key="chat_batch_repeat_count",
            )
        send_col, cancel_col, new_col, load_col = st.columns([1.25, 1, 1, 1])
        with send_col:
            send = st.form_submit_button("Send message" if chat_mode == "Single message" else "Run batch", type="primary", width="stretch")
        with cancel_col:
            cancel_chat = st.form_submit_button("Cancel chat", width="stretch")
        with new_col:
            new_chat = st.form_submit_button("New chat", width="stretch")
        with load_col:
            load_saved_chat = st.form_submit_button("Load chat", width="stretch")

    cfg1, cfg2, cfg3 = st.columns([2, 1, 1])
    with cfg1:
        chat_backend_label = st.selectbox(
            "Chat backend",
            visible_chat_backend_options,
            help=HELP["backend"],
            key="chat_backend_label",
        )
        chat_backend = normalize_chat_backend(chat_backend_label)
    with cfg2:
        max_new_tokens = st.slider("Max new tokens", 10, 300, 80, 10, help=HELP["max_new_tokens"], key="chat_max_new")
    with cfg3:
        temperature = st.slider("Temperature", 0.01, 1.5, 0.8, 0.05, help=HELP["temperature"], key="chat_temp")

    tog1, tog2 = st.columns(2)

    caps = render_capability_warning(chat_backend)
    local_steering_supported, local_steering_reason = local_steering_capability(chat_backend)
    local_steering_payload, local_steering_payload_error = selected_control_vector_payload() if "llama.cpp" in chat_backend.lower() else (None, None)
    startup_steered_backend = chat_backend == "llama.cpp glass"
    steering_available = caps["activation_steering"] or local_steering_supported

    with tog1:
        auto_trace = st.toggle(
            "Trace every message",
            value=True,
            help="Local GGUF backends call /glass-skull/trace when available. TransformerLens captures full activation traces.",
        )
    with tog2:
        if startup_steered_backend:
            use_steering = True
            st.info("Steering active: this backend points at the steered llama.cpp server.")
        elif steering_available:
            use_steering = st.toggle(
                "Use steering",
                value=False,
                help="TransformerLens uses activation hooks. Patched llama.cpp-glass servers can also accept per-request GGUF control-vector metadata when advertised.",
            )
        else:
            use_steering = False
            st.caption("Steering is unavailable for this backend until a compatible trace model or per-request-capable llama.cpp-glass server is selected.")

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

    if "llama.cpp" in chat_backend.lower():
        st.caption(f"Per-request steering: {local_steering_reason}")
        if use_steering and not startup_steered_backend and local_steering_payload:
            with st.expander("Local steering payload preview", expanded=False):
                st.json({"glass_skull": {"steering": local_steering_payload}, "metadata": {"glass_skull": {"steering": local_steering_payload}}})
        elif use_steering and not startup_steered_backend and local_steering_payload_error:
            st.warning(local_steering_payload_error)
    if use_steering and chat_backend == "TransformerLens" and not compatible_feature_names:
        st.warning(f"No compatible features for d_model {expected_dim}. Rebuild a feature with the currently loaded trace model.")

    batch_uploaded_items = []
    if chat_mode == "Batch run":
        uploaded_batch = st.file_uploader(
            "Batch prompt file",
            type=["txt", "jsonl", "csv"],
            help="Optional file input for the current batch. TXT uses one prompt per line; JSONL/CSV use prompt and optional label fields.",
            key="chat_batch_file",
        )
        if uploaded_batch is not None:
            try:
                batch_uploaded_items = load_prompt_file_bytes(uploaded_batch.name, uploaded_batch.getvalue())
                st.caption(f"Loaded {len(batch_uploaded_items)} file prompts.")
            except Exception as exc:
                st.error(str(exc))

    context = dashboard_context(str(chat_mode), st.session_state.get("active_run_id"))
    if st.session_state.get("batch_running"):
        st.warning(st.session_state.get("batch_status") or "Batch run is in progress.")
    elif st.session_state.get("last_batch_result"):
        batch_summary = st.session_state.last_batch_result.get("summary", {})
        st.success(
            f"Last batch finished: {batch_summary.get('prompt_count', 0)} prompts, "
            f"{batch_summary.get('error_count', 0)} errors, run {batch_summary.get('run_id', '-')}"
        )
    st.markdown(
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 12px 0;">'
        f'{ui.badge("mode " + context["mode"], ui.TEAL, active=True)}'
        f'{ui.badge("run " + context["run_id"], ui.AMBER if context["run_id"] != "none" else ui.SLATE, active=context["run_id"] != "none")}'
        f'</div>',
        unsafe_allow_html=True,
    )

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

    if cancel_chat:
        st.session_state.chat_cancel_requested = True
        st.info("Canceled pending chat action. Running generations cannot be interrupted until the backend request returns.")

    if new_chat:
        save_chat(st.session_state.chat_messages)
        st.session_state.chat_messages = []
        st.session_state.last_output = ""
        st.session_state.last_loaded_chat = ""
        st.rerun()

    if load_saved_chat:
        loaded = load_chat()
        if loaded:
            st.session_state.chat_messages = loaded
            st.session_state.last_loaded_chat = "latest"
            st.rerun()
        else:
            st.warning("No saved chats found.")

    if send and chat_mode == "Batch run" and not st.session_state.get("chat_cancel_requested"):
        prompt_items = batch_items_from_inputs(
            pasted_payload=batch_payload,
            repeat_prompt=prompt,
            repeat_count=int(batch_repeat_count),
            uploaded_items=batch_uploaded_items,
        )
        if not prompt_items:
            st.warning("Batch run needs at least one pasted, repeated, or uploaded prompt.")
        else:
            run_id = new_run_id("batch")
            st.session_state.active_run_id = run_id
            st.session_state.last_run_id = log_run(
                model_name=st.session_state.get("llama_model_alias", "") or model_name,
                mode="batch_start",
                prompt=f"{len(prompt_items)} prompts",
                metadata={
                    "run_id": run_id,
                    "mode": "Batch run",
                    "prompt_count": len(prompt_items),
                    "repeat_count": int(batch_repeat_count),
                    "pasted_count": len([line for line in batch_payload.splitlines() if line.strip()]),
                    "uploaded_count": len(batch_uploaded_items),
                },
            )
            backend_key = "transformerlens" if chat_backend == "TransformerLens" else "llama.cpp"
            llama_url = st.session_state.llama_glass_url if chat_backend == "llama.cpp glass" else st.session_state.llama_url
            try:
                layers = list(range(int(summary["layers"]))) if model is not None else [0]
                st.session_state.batch_running = True
                st.session_state.batch_status = f"Running batch {run_id}: 0/{len(prompt_items)} prompts"
                batch_progress = st.progress(0)
                batch_status = st.empty()

                def batch_cb(i: int, total: int, current_prompt: str) -> None:
                    st.session_state.batch_status = f"Running batch {run_id}: {i}/{total} prompts"
                    batch_progress.progress(i / max(total, 1))
                    batch_status.info(f"{st.session_state.batch_status} - {current_prompt[:120]}")

                with st.spinner("Running batch..."):
                    result = run_fuzz_experiment(
                        name=run_id,
                        prompts=prompt_items,
                        chat_backend=backend_key,
                        trace_enabled=bool(auto_trace),
                        model=model,
                        llama_url=llama_url,
                        llama_model_alias=st.session_state.get("llama_model_alias", ""),
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        layers=layers,
                        streams=["resid_post"],
                        top_k=32,
                        run_id=run_id,
                        mode="Batch run",
                        progress_callback=batch_cb,
                    )
                result["summary"]["run_id"] = run_id
                result["summary"]["mode"] = "Batch run"
                st.session_state.last_batch_result = result
                st.session_state.last_fuzz_result = result
                st.session_state.batch_running = False
                st.session_state.batch_status = f"Batch {run_id} finished"
                st.success(f"Batch complete: {len(prompt_items)} prompts")
                st.rerun()
            except Exception as exc:
                st.session_state.batch_running = False
                st.session_state.batch_status = f"Batch {run_id} failed: {exc}"
                st.error(str(exc))

    if send and chat_mode == "Single message" and not st.session_state.get("chat_cancel_requested") and prompt.strip():
        save_chat(st.session_state.chat_messages)
        prompt = prompt.strip()
        outbound_messages = chat_history_for_send(prompt)
        outbound_prompt = transcript_prompt(outbound_messages)
        run_session_id = new_run_id("single")
        st.session_state.active_run_id = run_session_id
        now = datetime.now().strftime("%H:%M:%S")
        st.session_state.chat_messages.append({"role": "user", "content": prompt, "ts": now})

        if auto_trace:
            if "llama.cpp" in chat_backend.lower():
                trace_url = st.session_state.llama_glass_url if chat_backend == "llama.cpp glass" else st.session_state.llama_url
                with st.spinner("Tracing prompt through llama.cpp..."):
                    try:
                        trace_payload = trace_glass_prompt(
                            trace_url,
                            prompt,
                            model_alias=st.session_state.get("llama_model_alias", ""),
                            with_pieces=True,
                        )
                        set_local_dashboard_trace(trace_payload, prompt, chat_backend, str(local_model_context["display_name"] if local_model_context else st.session_state.get("llama_model_alias", "")))
                        run_id = log_run(
                            model_name=st.session_state.get("llama_model_alias", ""),
                            mode="local_chat_trace",
                            prompt=prompt,
                            metadata={"trace": trace_payload, "chat_backend": chat_backend, "run_id": run_session_id, "mode": "Single message"},
                        )
                        st.session_state.last_run_id = run_id
                    except Exception as exc:
                        st.warning(f"Local trace unavailable: {exc}")
            else:
                with st.spinner("Tracing prompt locally..."):
                    trace = trace_prompt(model, prompt)
                    st.session_state.trace = trace
                    set_dashboard_trace(trace, prompt, chat_backend, str(summary["model_name"]))
                    run_id = log_run(model_name=model_name, mode="chat_trace", prompt=prompt, metadata={"tokens": trace.tokens, "summary": summary, "chat_backend": chat_backend, "run_id": run_session_id, "mode": "Single message"})
                    st.session_state.last_run_id = run_id

        output = ""
        error = None
        with st.spinner("Generating reply..."):
            try:
                if chat_backend == "llama.cpp normal":
                    steering_payload = local_steering_payload if use_steering else None
                    if use_steering and steering_payload is None:
                        raise ValueError(local_steering_payload_error or "No local control-vector payload is available.")
                    output = chat_completion(
                        st.session_state.llama_url,
                        prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        messages=outbound_messages,
                        model_alias=st.session_state.get("llama_model_alias", ""),
                        steering=steering_payload,
                        steering_supported=local_steering_supported,
                    )
                elif chat_backend == "llama.cpp glass":
                    output = chat_completion(
                        st.session_state.llama_glass_url,
                        prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        messages=outbound_messages,
                        model_alias=st.session_state.get("llama_model_alias", ""),
                        steering=None,
                        steering_supported=local_steering_supported,
                    )
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
                    output = generate_steered(model, outbound_prompt, vector, layer, stream=stream, strength=strength, max_new_tokens=max_new_tokens, temperature=temperature)
                else:
                    output = generate_normal(model, outbound_prompt, max_new_tokens=max_new_tokens, temperature=temperature)
            except Exception as exc:
                error = str(exc)
                output = f"Generation error: {error}"

        st.session_state.last_output = output
        st.session_state.chat_messages.append({"role": "assistant", "content": output, "ts": datetime.now().strftime("%H:%M:%S")})
        save_chat(st.session_state.chat_messages)
        log_run(model_name=model_name, mode="chat_generate", prompt=prompt, output=output, metadata={"backend": chat_backend, "used_steering": bool(use_steering), "error": error, "run_id": run_session_id, "mode": "Single message"})
        st.rerun()
    st.session_state.chat_cancel_requested = False

# ----------------------------------------------------------- Lab ----
if tab_lab is not None:
    with tab_lab:
        lab_trace, lab_poke = st.tabs(["Trace / Lens", "Map / Steer / Compare / Batch"])

    with lab_trace:
        ui.section_header("Trace / Lens", "Visualize model internals captured from the latest traced prompt.")
        if 'chat_backend' in locals() and "llama.cpp" in chat_backend.lower():
            ui.empty_state("Trace source is the TransformerLens model", "Local GGUF chat can generate replies, but this app does not receive llama.cpp activations yet. Use the trace model for Logit Lens, Attention, and activation steering.")

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

# ------------------------------------------------ Map / Compare / Batch ----
    with lab_poke:
        ui.section_header("Poke / Compare / Fuzz", "Probe and stress-test model behavior.")
        if not trace_source_enabled:
            if hf_source_enabled:
                render_hf_catalog_panel()
            else:
                ui.empty_state("Lab is not configured", "Open workflow setup and select Hugging Face or Trace model.")
        else:
            if 'chat_backend' in locals() and "llama.cpp" in chat_backend.lower():
                st.info("Local GGUF selected: chat and fuzz output are available through llama.cpp, but activation Map/Steer/Compare controls target the TransformerLens trace model until llama.cpp exposes trace hooks.")

            tab_steer, tab_map, tab_compare, tab_edges, tab_fuzz = st.tabs(
                ["Steer", "Map", "Compare", "Edges", "Fuzz"]
            )

        if trace_source_enabled:
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
                    fuzz_backend_label = st.selectbox("Fuzz chat backend", visible_chat_backend_options, help="Where generated outputs come from.")
                    fuzz_backend = normalize_chat_backend(fuzz_backend_label)
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
                                    llama_model_alias=st.session_state.get("llama_model_alias", ""),
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
                    st.markdown("**Last run activation views**")
                    artifact = result.get("artifact") or {}
                    run_summary = result.get("summary", {})
                    ui.property_list([
                        ("run_id", str(run_summary.get("run_id", artifact.get("run_id", "-")))),
                        ("mode", str(run_summary.get("mode", artifact.get("mode", "-")))),
                        ("backend", str(run_summary.get("chat_backend", artifact.get("backend", "-")))),
                        ("model", str(artifact.get("model") or run_summary.get("llama_model_alias", "-"))),
                        ("prompts", str(run_summary.get("prompt_count", artifact.get("summary", {}).get("prompt_count", "-")))),
                        ("errors", str(run_summary.get("error_count", artifact.get("summary", {}).get("error_count", "-")))),
                        ("trace_supported", str(run_summary.get("trace_supported", artifact.get("summary", {}).get("trace_supported", False)))),
                    ])
                    path_df = result.get("activation_path_df", pd.DataFrame())
                    if path_df.empty and artifact:
                        path_df = activation_path_df(artifact)
                    available_path = path_df[path_df["trace_available"] != False] if not path_df.empty else pd.DataFrame()
                    if available_path.empty:
                        reason = "No activation summaries were captured for this run."
                        if not path_df.empty and "unavailable_reason" in path_df:
                            reasons = [r for r in path_df["unavailable_reason"].dropna().astype(str).unique().tolist() if r]
                            if reasons:
                                reason = "; ".join(reasons[:3])
                        ui.empty_state("Activation graphs unavailable", reason)
                    else:
                        group_mode = st.segmented_control("Aggregate", ["prompt", "label", "all"], default="prompt", key="fuzz_artifact_group")
                        heat_df = batch_heatmap_df(artifact, group_by=group_mode) if artifact else result.get("batch_prompt_heatmap_df", pd.DataFrame())
                        if group_mode == "label":
                            plot_if_present(label_activation_heatmap(heat_df), key_hint="fuzz_label_activation")
                        else:
                            plot_if_present(batch_activation_heatmap(heat_df), key_hint="fuzz_batch_activation")
                        prompt_ids = sorted(available_path["prompt_id"].dropna().astype(str).unique().tolist())
                        selected_prompt = st.selectbox("Prompt path", prompt_ids, key="fuzz_prompt_path") if prompt_ids else ""
                        if selected_prompt:
                            plot_if_present(activation_path_graph(available_path[available_path["prompt_id"].astype(str) == selected_prompt]), key_hint="fuzz_prompt_path_graph")
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
    anatomy_panels = ["Anatomy", "Experiments", "Logs"]
    if trace_source_enabled:
        anatomy_panels[1:1] = ["Hooks", "Parameters", "Features"]
    if hf_source_enabled:
        anatomy_panels.insert(-1, "HF Catalog")
    panel = st.radio(
        "Panel",
        anatomy_panels,
        horizontal=True,
        help="Switch between model structure and saved run history.",
    )

    if panel == "Anatomy":
        if model is None:
            ui.sec_label("Local GGUF")
            if local_model_context:
                ui.property_list(local_model_context["property_rows"])
                for err in local_model_context["errors"]:
                    st.warning(err)
                local_tensors_df = local_model_context.get("tensors_df", pd.DataFrame())
                if not local_tensors_df.empty:
                    columns = [col for col in ["index", "name", "shape", "dtype", "elements", "offset"] if col in local_tensors_df.columns]
                    st.dataframe(local_tensors_df[columns], width="stretch", height=420, hide_index=True)
                else:
                    ui.empty_state("No tensor index", "The configured GGUF file did not expose a readable tensor index.")
            else:
                ui.empty_state("No local model context", "Choose a local backend in Chat or configure Local GGUF in Models.")
        else:
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

# -------------------------------------------------------------- Models ----
with tab_models:
    render_models_tab()

# ------------------------------------------------------------- Settings ----
with tab_settings:
    render_settings_tab(summary)
