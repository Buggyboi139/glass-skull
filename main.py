from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from glass_skull import ui_theme as ui
from glass_skull.activation_map import build_activation_map_payload, managed_backend_process_model_path
from glass_skull.activation_map_view import render_activation_map
from glass_skull.activation_patch import (
    build_activation_patch_backend_payload,
    build_activation_patch_recipe,
    compare_patch_outputs,
    validate_activation_patch_recipe,
)
from glass_skull.behavior_profiles import get_behavior_profile, list_behavior_profiles
from glass_skull.behavior_scoring import behavior_timeline_df, score_run_artifact
from glass_skull.chat_store import list_chats, load_chat, save_chat
from glass_skull.config import DEFAULT_BATCH_MESSAGES, DEFAULT_GGUF_MODEL_PATH, ensure_dirs, seed_batch_prompt_default, seed_missing_defaults
from glass_skull.experiment_store import (
    create_experiment_dir,
    latest_run_artifacts,
    list_activation_patch_recipes,
    list_experiments,
    load_activation_patch_recipe,
    load_run_artifact,
    save_activation_patch_recipe,
    write_json,
    write_run_artifact,
)
from glass_skull.fuzzing import run_fuzz_experiment
from glass_skull.llama_client import (
    build_steering_metadata,
    chat_completion,
    check_server,
    get_glass_info,
    activation_patch_diagnostic,
    activation_patch_supported,
    per_request_steering_supported,
    trace_glass_prompt,
    trace_vectors_supported,
)
from glass_skull.node_annotations import add_tag, delete_annotation, delete_note, delete_tag, update_note, upsert_annotation
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
from glass_skull.run_artifacts import (
    activation_path_df,
    batch_heatmap_df,
    build_run_artifact,
    dimension_frequency_df,
    label_heatmap_df,
    llama_trace_unavailable_reason,
    normalize_llama_trace,
    trace_unavailable_row,
)
from glass_skull.ui_local import LOCAL_TABS, batch_items_from_inputs, dashboard_context, new_run_id
from glass_skull.tooltip_generator import STEER_CONTROL_METADATA, ensure_tooltips, tooltip_text
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
from glass_skull.workspaces import (
    apply_tab_state,
    apply_workspace_state,
    clear_tab_state,
    clear_workspace,
    collect_tab_state,
    list_tab_states,
    list_workspaces,
    load_tab_state,
    load_workspace,
    save_tab_state,
    save_tab_state_as,
    save_workspace,
    save_workspace_as,
)


st.set_page_config(page_title="Operation Glass Skull", layout="wide", initial_sidebar_state="collapsed")
ensure_dirs()
ui.inject_theme()

CHAT_BACKEND_NORMAL = "Local GGUF normal (llama.cpp)"
CHAT_BACKEND_STEERED = "Local GGUF steered (llama.cpp)"
CHAT_BACKENDS = [CHAT_BACKEND_NORMAL, CHAT_BACKEND_STEERED]
DEFAULT_CHAT_INPUT = "hi"


def normalize_chat_backend(label: str) -> str:
    if label in {"llama.cpp normal", CHAT_BACKEND_NORMAL}:
        return "llama.cpp normal"
    if label in {"llama.cpp glass", CHAT_BACKEND_STEERED}:
        return "llama.cpp glass"
    return "llama.cpp normal"


def chat_backend_display(canonical: str) -> str:
    return CHAT_BACKEND_STEERED if canonical == "llama.cpp glass" else CHAT_BACKEND_NORMAL


def seeded_chat_input() -> str | None:
    # Streamlit 1.58.0 st.chat_input has no value/default parameter. A form
    # text input preserves a visible default, trading off sticky chat styling.
    with st.form("chat_prompt_form", clear_on_submit=True):
        prompt = st.text_input("Send a local prompt", value=DEFAULT_CHAT_INPUT, key="chat_prompt_text")
        submitted = st.form_submit_button("Send", width="stretch")
    return prompt if submitted else None


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
        "llama_model_path": str(DEFAULT_GGUF_MODEL_PATH),
        "activeModelFingerprint": "",
        "modelArtifactsInvalidated": False,
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
        "batch_pasted_prompts": DEFAULT_BATCH_MESSAGES,
        "batch_pasted_prompts_user_set": False,
        "batch_running": False,
        "batch_status": "",
        "chat_cancel_requested": False,
        "workspace_name": "default",
        "workspace_save_as_name": "",
        "workspace_error": "",
        "workspace_warning": "",
        "tab_state": {},
        "tab_save_as_names": {},
        "tab_workspace_errors": {},
        "map_visualization_mode": "",
        "map_selected_prompt": None,
        "map_selected_batch": None,
        "map_selected_token": None,
        "map_top_k": 8,
        "map_background_opacity": 0.24,
        "map_edge_threshold": 0.0,
        "map_show_aggregate_heatmap": False,
        "map_show_secondary_branches": True,
        "map_annotation_selected_group": "",
        "map_annotation_new_tag": "",
        "map_annotation_note": "",
        "map_annotation_status": "",
    }
    seed_missing_defaults(st.session_state, defaults)
    seed_batch_prompt_default(st.session_state)


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


def _annotation_prompt_excerpt(group: dict) -> str:
    previews = group.get("promptPreviewList") if isinstance(group.get("promptPreviewList"), list) else []
    text = previews[0] if previews else group.get("promptPreview") or group.get("promptText") or ""
    return str(text).replace("\n", " ").strip()[:220]


def _split_tag_input(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def render_node_annotation_inspector(payload: dict) -> None:
    groups = payload.get("nodeGroups") if isinstance(payload.get("nodeGroups"), list) else []
    if not groups:
        return
    st.subheader("Node annotation inspector")
    labels = {
        str(group.get("groupId")): f"{group.get('layerId')} | {group.get('groupId')} | activation {float(group.get('activationValue') or 0):.2f}"
        for group in groups
        if group.get("groupId")
    }
    if not labels:
        return
    selected_key = st.session_state.get("map_annotation_selected_group") or str((payload.get("diagnostics", {}).get("selectedGroup") or {}).get("groupId") or "")
    if selected_key not in labels:
        selected_key = next(iter(labels))
    selected_group_id = st.selectbox(
        "Node/cluster",
        list(labels),
        format_func=lambda value: labels.get(value, str(value)),
        index=list(labels).index(selected_key),
        key="map_annotation_selected_group",
    )
    group = next((item for item in groups if str(item.get("groupId")) == str(selected_group_id)), groups[0])
    annotation_ids = [item for item in group.get("annotationIds", []) if item]
    match_type = str(group.get("annotationMatchType") or "none")
    tags = list(group.get("annotationTags") or [])
    note = str(group.get("annotationNote") or "")

    st.caption(f"annotation match: {match_type}")
    if tags:
        st.write("Tags: " + ", ".join(tags))
    else:
        st.caption("Tags: none")
    if note:
        st.text_area("Saved note", value=note, height=100, disabled=True, key=f"annotation_saved_note_{selected_group_id}")
    else:
        st.caption("Note: none")

    annotation_id = annotation_ids[0] if annotation_ids else ""
    if annotation_id and match_type == "approximate":
        st.warning("Editing this annotation updates an approximate match.")

    edit_cols = st.columns([1, 2])
    with edit_cols[0]:
        new_tag = st.text_input("Add tag", key="map_annotation_new_tag", placeholder="comedy")
        if st.button("Save tag", key="map_annotation_save_tag", width="stretch"):
            try:
                if annotation_id:
                    add_tag(annotation_id, new_tag)
                else:
                    upsert_annotation(
                        payload.get("modelMeta"),
                        layer=int(group.get("layer") or str(group.get("layerId") or "L0").replace("L", "") or 0),
                        cluster_id=group.get("clusterId"),
                        node_id=group.get("nodeId") or group.get("groupId"),
                        node_range=group.get("nodeRange"),
                        tags=_split_tag_input(new_tag),
                        note=note,
                        created_from={
                            "run_id": payload.get("diagnostics", {}).get("runId"),
                            "prompt_id": group.get("promptId"),
                            "batch_id": group.get("batchId"),
                            "token_id": group.get("tokenIndex"),
                            "prompt_excerpt": _annotation_prompt_excerpt(group),
                        },
                    )
                st.session_state.map_annotation_status = "Saved tag."
                st.rerun()
            except Exception as exc:
                st.error(f"Could not save tag: {exc}")
        if tags:
            tag_to_delete = st.selectbox("Delete tag", tags, key="map_annotation_delete_tag_choice")
            if st.button("Delete tag", key="map_annotation_delete_tag", width="stretch", disabled=not annotation_id):
                try:
                    delete_tag(annotation_id, tag_to_delete)
                    st.session_state.map_annotation_status = "Deleted tag."
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not delete tag: {exc}")
    with edit_cols[1]:
        note_value = st.text_area("Edit note", value=note, height=128, key=f"map_annotation_note_{selected_group_id}")
        note_cols = st.columns(3)
        with note_cols[0]:
            if st.button("Save note", key="map_annotation_save_note", width="stretch"):
                try:
                    if annotation_id:
                        update_note(annotation_id, note_value)
                    else:
                        upsert_annotation(
                            payload.get("modelMeta"),
                            layer=int(group.get("layer") or str(group.get("layerId") or "L0").replace("L", "") or 0),
                            cluster_id=group.get("clusterId"),
                            node_id=group.get("nodeId") or group.get("groupId"),
                            node_range=group.get("nodeRange"),
                            tags=tags,
                            note=note_value,
                            created_from={
                                "run_id": payload.get("diagnostics", {}).get("runId"),
                                "prompt_id": group.get("promptId"),
                                "batch_id": group.get("batchId"),
                                "token_id": group.get("tokenIndex"),
                                "prompt_excerpt": _annotation_prompt_excerpt(group),
                            },
                        )
                    st.session_state.map_annotation_status = "Saved note."
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not save note: {exc}")
        with note_cols[1]:
            if st.button("Delete note", key="map_annotation_delete_note", width="stretch", disabled=not annotation_id):
                try:
                    delete_note(annotation_id)
                    st.session_state.map_annotation_status = "Deleted note."
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not delete note: {exc}")
        with note_cols[2]:
            if st.button("Delete annotation", key="map_annotation_delete_annotation", width="stretch", disabled=not annotation_id):
                try:
                    delete_annotation(annotation_id)
                    st.session_state.map_annotation_status = "Deleted annotation."
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not delete annotation: {exc}")
    if st.session_state.get("map_annotation_status"):
        st.success(st.session_state.map_annotation_status)


def _tab_state(tab_name: str) -> dict:
    tabs = st.session_state.setdefault("tab_state", {})
    if not isinstance(tabs, dict):
        tabs = {}
        st.session_state.tab_state = tabs
    current = tabs.setdefault(tab_name, {})
    if not isinstance(current, dict):
        current = {}
        tabs[tab_name] = current
    return current


def _set_status(result, *, scope: str) -> None:
    if result.error:
        st.session_state[f"{scope}_error"] = result.error
        st.session_state[f"{scope}_warning"] = ""
    elif result.warning:
        st.session_state[f"{scope}_warning"] = result.warning
        st.session_state[f"{scope}_error"] = ""
    else:
        st.session_state[f"{scope}_error"] = ""
        st.session_state[f"{scope}_warning"] = ""


def steer_help(control_id: str, tooltips: dict | None = None) -> str | None:
    entries = tooltips if isinstance(tooltips, dict) else ensure_tooltips("steer", STEER_CONTROL_METADATA)
    return tooltip_text(entries.get(control_id))


def mark_batch_prompts_user_set() -> None:
    st.session_state.batch_pasted_prompts_user_set = True


def render_global_workspace_controls() -> None:
    st.caption("Workspace")
    workspaces = list_workspaces()
    selected = st.selectbox(
        "Workspace",
        workspaces or [st.session_state.workspace_name],
        index=0,
        label_visibility="collapsed",
        key="workspace_load_name",
    )
    cols = st.columns([1, 1, 1, 1, 1.4])
    with cols[0]:
        if st.button("Save", key="workspace_save", width="stretch"):
            result = save_workspace(st.session_state)
            _set_status(result, scope="workspace")
            st.rerun()
    with cols[1]:
        if st.button("Save As", key="workspace_save_as", width="stretch"):
            name = st.session_state.get("workspace_save_as_name", "").strip()
            if not name:
                st.session_state.workspace_error = "Save As requires a workspace name."
            else:
                result = save_workspace_as(st.session_state, name)
                _set_status(result, scope="workspace")
                st.rerun()
    with cols[2]:
        if st.button("Load", key="workspace_load", width="stretch"):
            result = load_workspace(str(selected))
            _set_status(result, scope="workspace")
            if not result.error and not result.warning:
                apply_workspace_state(st.session_state, result.state)
                st.session_state.workspace_name = result.name
                st.rerun()
    with cols[3]:
        if st.button("Clear", key="workspace_clear", width="stretch"):
            clear_workspace(st.session_state)
            st.rerun()
    with cols[4]:
        st.text_input("Save As name", key="workspace_save_as_name", label_visibility="collapsed", placeholder="workspace name")
    if st.session_state.get("workspace_error"):
        st.error(st.session_state.workspace_error)
    if st.session_state.get("workspace_warning"):
        st.warning(st.session_state.workspace_warning)


def render_tab_workspace_controls(tab_name: str) -> None:
    state = _tab_state(tab_name)
    tab_key = safe_tab_key(tab_name)
    saved = list_tab_states(tab_name)
    load_key = f"{tab_key}_tab_load_name"
    save_as_key = f"{tab_key}_tab_save_as_name"
    error_key = f"{tab_key}_tab_error"
    help_for = (lambda control_id: steer_help(control_id)) if tab_name == "Steer" else (lambda control_id: None)
    st.caption(f"{tab_name} state")
    cols = st.columns([1, 1, 1, 1, 1.4])
    with cols[0]:
        if st.button("Save", key=f"{tab_key}_tab_save", width="stretch", help=help_for(f"{tab_key}_tab_save")):
            result = save_tab_state(tab_name, collect_tab_state(tab_name, st.session_state, state), name=state.get("workspace_name"))
            state["workspace_name"] = result.name
            st.session_state[error_key] = result.error or result.warning or ""
            st.rerun()
    with cols[1]:
        if st.button("Save As", key=f"{tab_key}_tab_save_as", width="stretch", help=help_for(f"{tab_key}_tab_save_as")):
            name = str(st.session_state.get(save_as_key, "")).strip()
            if not name:
                st.session_state[error_key] = "Save As requires a tab state name."
            else:
                result = save_tab_state_as(tab_name, collect_tab_state(tab_name, st.session_state, state), name)
                state["workspace_name"] = result.name
                st.session_state[error_key] = result.error or result.warning or ""
                st.rerun()
    with cols[2]:
        selected = st.selectbox(
            "Load tab state",
            saved or [state.get("workspace_name") or "default"],
            key=load_key,
            label_visibility="collapsed",
            help=help_for(f"{tab_key}_tab_load_name"),
        )
        if st.button("Load", key=f"{tab_key}_tab_load", width="stretch", help=help_for(f"{tab_key}_tab_load")):
            result = load_tab_state(tab_name, str(selected))
            st.session_state[error_key] = result.error or result.warning or ""
            if not result.error and not result.warning:
                model_backend_before = model_backend_snapshot(st.session_state)
                apply_tab_state(st.session_state, tab_name, result.state)
                st.session_state.tab_state[tab_name]["workspace_name"] = result.name
                st.session_state[f"{tab_key}_tab_pending_state"] = result.state
                invalidate_if_model_backend_changed(st.session_state, model_backend_before)
                st.rerun()
    with cols[3]:
        if st.button("Clear", key=f"{tab_key}_tab_clear", width="stretch", help=help_for(f"{tab_key}_tab_clear")):
            model_backend_before = model_backend_snapshot(st.session_state)
            clear_tab_state(st.session_state, tab_name)
            invalidate_if_model_backend_changed(st.session_state, model_backend_before)
            st.rerun()
    with cols[4]:
        st.text_input("Save As tab name", key=save_as_key, label_visibility="collapsed", placeholder="tab state name", help=help_for(f"{tab_key}_tab_save_as_name"))
    if st.session_state.get(error_key):
        st.warning(st.session_state[error_key])


def consume_loaded_tab_state(tab_name: str) -> dict | None:
    tab_key = safe_tab_key(tab_name)
    pending_key = f"{tab_key}_tab_pending_state"
    pending = st.session_state.get(pending_key)
    if isinstance(pending, dict):
        del st.session_state[pending_key]
        return pending
    return None


def apply_known_tab_state(tab_name: str, state: dict | None) -> None:
    if not state:
        return
    if tab_name == "Run" and state.get("active_run_mode") in {"Single message", "Batch run"}:
        st.session_state.active_run_mode = state["active_run_mode"]
    if tab_name == "Run" and "batch_pasted_prompts" in state:
        st.session_state.batch_pasted_prompts = state["batch_pasted_prompts"]
    if tab_name == "Map":
        mapping = {
            "visualization_mode": "map_visualization_mode",
            "selected_prompt": "map_selected_prompt",
            "selected_batch": "map_selected_batch",
            "selected_token": "map_selected_token",
            "top_k": "map_top_k",
            "background_opacity": "map_background_opacity",
            "edge_threshold": "map_edge_threshold",
            "show_aggregate_heatmap": "map_show_aggregate_heatmap",
            "show_secondary_branches": "map_show_secondary_branches",
            "annotation_selected_group": "map_annotation_selected_group",
        }
        for source, dest in mapping.items():
            if source in state:
                st.session_state[dest] = state[source]


def safe_tab_key(tab_name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in tab_name).strip("_") or "tab"


def local_summary(local_context: dict | None) -> dict:
    local_context = local_context or {}
    return {
        "model_name": local_context.get("display_name") or active_model_label(),
        "backend": "llama.cpp",
        "device": "local",
        "layers": int(local_context.get("trace_layer_count") or local_context.get("block_count") or 1),
        "gguf_block_count": int(local_context.get("block_count") or 0),
        "gguf_trace_layer_count": int(local_context.get("trace_layer_count") or local_context.get("block_count") or 0),
        "heads": int(local_context.get("head_count") or 1),
        "d_model": int(local_context.get("embedding_length") or 0),
        "d_head": int(local_context.get("d_head") or 0),
        "d_mlp": int(local_context.get("d_mlp") or 0),
        "vocab_size": 0,
        "parameters": int(local_context.get("tensor_elements") or 0),
        "dtype": "gguf",
    }


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _trace_layer_count_from_glass_info(info: dict | None) -> int | None:
    if not isinstance(info, dict):
        return None

    count = _positive_int(info.get("layers"))
    if count is not None:
        return count

    layers = info.get("layers")
    if isinstance(layers, list) and layers:
        return len(layers)

    meta = info.get("meta")
    if isinstance(meta, dict):
        return _positive_int(meta.get("n_layer"))

    return None


def model_identity_diagnostics(
    ui_selected_model_path: str | None,
    backend_info: dict | None,
    backend_process_model_path: str | None = None,
    ui_selected_model_name: str | None = None,
) -> dict:
    def normal_path(value: object) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        from pathlib import Path as _Path
        try:
            return str(_Path(text).expanduser().resolve(strict=False))
        except Exception:
            return text

    def backend_path(info: dict | None) -> str:
        if not isinstance(info, dict):
            return ""
        model = info.get("model") if isinstance(info.get("model"), dict) else {}
        for source in (model, info):
            for key in ("path", "model_path", "modelPath"):
                value = source.get(key)
                if value:
                    return str(value)
        return ""

    def backend_name(info: dict | None) -> str:
        if not isinstance(info, dict):
            return ""
        model = info.get("model") if isinstance(info.get("model"), dict) else {}
        for source in (model, info):
            for key in ("id", "name", "model", "model_name", "modelName"):
                value = source.get(key)
                if value:
                    return str(value)
        return ""

    ui_path = str(ui_selected_model_path or "")
    ui_name = str(ui_selected_model_name or "").strip()
    info_path = backend_path(backend_info)
    info_name = backend_name(backend_info)
    process_path = str(backend_process_model_path or "")
    mismatches = []
    if ui_path and info_path and normal_path(ui_path) != normal_path(info_path):
        mismatches.append("UI selected model path differs from backend info model path")
    if ui_path and process_path and normal_path(ui_path) != normal_path(process_path):
        mismatches.append("UI selected model path differs from backend process model path")
    if info_path and process_path and normal_path(info_path) != normal_path(process_path):
        mismatches.append("backend info model path differs from backend process model path")
    if not info_path and ui_name and info_name and ui_name.lower() != "local" and ui_name != info_name:
        mismatches.append("UI selected model name differs from backend info model name")
    return {
        "uiSelectedModelPath": ui_path,
        "uiSelectedModelName": ui_name,
        "backendInfoModelPath": info_path,
        "backendInfoModelName": info_name,
        "backendProcessModelPath": process_path,
        "modelIdentityMismatch": bool(mismatches),
        "mismatchSummary": "; ".join(mismatches),
    }


def resolve_trace_plan(
    base_url: str,
    model_alias: str | None,
    summary: dict | None,
    *,
    ui_model_path: str | None = None,
    backend_process_model_path: str | None = None,
    explicit_layers: list[int] | None = None,
) -> dict:
    info = None
    count = None
    info_error = ""
    try:
        info = get_glass_info(base_url, model_alias=model_alias)
        count = _trace_layer_count_from_glass_info(info)
    except Exception as exc:
        info_error = str(exc)
        info = None
        count = None

    if explicit_layers is not None:
        layers = sorted({max(0, int(layer)) for layer in explicit_layers})
        source = "explicit_user_override"
    else:
        if count is None and isinstance(summary, dict):
            count = _positive_int(summary.get("layers"))
        if count is None:
            layers = [0]
            source = "fallback"
        else:
            max_layer = count - 1
            layers = [min(max(layer, 0), max_layer) for layer in range(count)] or [0]
            source = "backend_info" if info is not None else "gguf_metadata"

    identity = model_identity_diagnostics(
        ui_model_path,
        info,
        backend_process_model_path=backend_process_model_path,
        ui_selected_model_name=model_alias,
    )
    return {
        "layers": layers,
        "source": source,
        "backend_info": info,
        "backend_info_error": info_error,
        **identity,
        "traceRequestedMinLayer": min(layers) if layers else None,
        "traceRequestedMaxLayer": max(layers) if layers else None,
        "traceRequestedLayerCount": len(layers),
    }


def resolve_trace_layers(base_url: str, model_alias: str | None, summary: dict | None) -> list[int]:
    return resolve_trace_plan(base_url, model_alias, summary)["layers"]


def clear_model_dependent_state(state) -> None:
    clear_values = {
        "llama_status": None,
        "llama_glass_status": None,
        "local_dashboard_trace": None,
        "local_dashboard_trace_meta": {},
        "last_batch_result": None,
        "last_fuzz_result": None,
        "last_behavior_artifact": None,
        "last_behavior_scores": pd.DataFrame,
        "map_selected_prompt": None,
        "map_selected_batch": None,
        "map_selected_token": None,
        "map_annotation_selected_group": "",
    }
    for key, value in clear_values.items():
        state[key] = value() if callable(value) else value
    state["modelArtifactsInvalidated"] = True


def model_backend_snapshot(state) -> tuple[str, str, str, str]:
    keys = (
        "llama_model_path",
        "llama_model_alias",
        "llama_url",
        "llama_glass_url",
    )
    return tuple(str(state.get(key) or "").strip() for key in keys)


def invalidate_model_dependent_state(state, model_path: str | None, model_alias: str | None) -> bool:
    fingerprint = f"{str(model_path or '').strip()}|{str(model_alias or '').strip()}"
    if state.get("activeModelFingerprint") == fingerprint:
        return False
    state["activeModelFingerprint"] = fingerprint
    clear_model_dependent_state(state)
    return True


def invalidate_if_model_backend_changed(state, before: tuple[str, str, str, str]) -> bool:
    after = model_backend_snapshot(state)
    if after == before:
        return False
    changed = invalidate_model_dependent_state(
        state,
        state.get("llama_model_path", ""),
        state.get("llama_model_alias", ""),
    )
    if not changed:
        clear_model_dependent_state(state)
    return True


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


def single_trace_artifact_from_llama(
    trace_payload: dict,
    dash_meta: dict,
    output: str = "",
    error: str | None = None,
    trace_error: str | None = None,
) -> dict:
    run_id = str(st.session_state.get("active_run_id") or dash_meta.get("run_id") or "single")
    trace_rows = normalize_llama_trace(
        trace_payload,
        prompt_id=0,
        label="single",
        metadata={"run_id": run_id, "mode": "Single message"},
    )
    if not trace_rows:
        reason = trace_error or llama_trace_unavailable_reason(trace_payload)
        trace_rows = [trace_unavailable_row(run_id, 0, "single", "llama.cpp", reason)]
    summary = {}
    if isinstance(dash_meta.get("backend_info"), dict):
        summary["backend_info"] = dash_meta.get("backend_info")
    if isinstance(dash_meta.get("trace_layers"), list):
        summary["layers"] = list(dash_meta.get("trace_layers") or [])
    if isinstance(dash_meta.get("trace_plan"), dict):
        summary["trace_plan"] = dash_meta.get("trace_plan")
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
            "metadata": {"run_id": run_id, "mode": "Single message", "trace_error": trace_error},
        }],
        summary=summary or None,
    )


def store_behavior_artifact(artifact: dict) -> pd.DataFrame:
    profile = get_behavior_profile(st.session_state.get("behavior_profile", "concise_helpfulness"))
    scores = score_run_artifact(artifact, profile=profile)
    st.session_state.last_behavior_artifact = artifact
    st.session_state.last_behavior_scores = scores
    st.session_state.modelArtifactsInvalidated = False
    history = list(st.session_state.get("behavior_run_history", []))
    run_id = str(artifact.get("run_id") or "")
    history = [item for item in history if str(item.get("run_id") or "") != run_id]
    history.append({"run_id": run_id, "artifact": artifact, "scores": scores})
    st.session_state.behavior_run_history = history[-12:]
    return scores


def persist_single_run_artifact(artifact: dict) -> None:
    run_id = str(artifact.get("run_id") or "single")
    exp_dir = create_experiment_dir(run_id)
    write_run_artifact(exp_dir, artifact)
    write_json(exp_dir / "summary.json", {
        "run_id": run_id,
        "mode": artifact.get("mode", "Single message"),
        "artifact_path": str(exp_dir / "artifact.json"),
        **(artifact.get("summary", {}) if isinstance(artifact.get("summary"), dict) else {}),
    })


def latest_saved_behavior_artifact() -> dict:
    for row in latest_run_artifacts(limit=50):
        if row.get("mode") not in {"Single message", "Batch run"}:
            continue
        artifact_path = row.get("artifact_path")
        if not artifact_path:
            continue
        try:
            return load_run_artifact(artifact_path)
        except Exception:
            continue
    return {}


def saved_behavior_artifact_options() -> list[dict]:
    rows = [
        row for row in latest_run_artifacts(limit=50)
        if row.get("mode") in {"Single message", "Batch run"} and row.get("artifact_path")
    ]
    current = st.session_state.get("last_behavior_artifact")
    if isinstance(current, dict) and current:
        run_id = str(current.get("run_id") or "")
        if run_id and not any(str(row.get("run_id") or "") == run_id for row in rows):
            rows.insert(0, {
                "name": f"current_{run_id}",
                "path": "",
                "artifact_path": "",
                "run_id": run_id,
                "mode": current.get("mode"),
                "backend": current.get("backend"),
                "model": current.get("model"),
                "created_at": current.get("created_at"),
                "summary": current.get("summary", {}),
                "artifact": current,
            })
    return rows


def artifact_from_option(row: dict) -> dict:
    if isinstance(row.get("artifact"), dict):
        return row["artifact"]
    return load_run_artifact(str(row["artifact_path"]))


def latest_behavior_artifact() -> dict:
    if st.session_state.get("modelArtifactsInvalidated"):
        return {}
    artifact = st.session_state.get("last_behavior_artifact")
    if isinstance(artifact, dict) and artifact:
        return artifact
    result = st.session_state.get("last_batch_result") or st.session_state.get("last_fuzz_result")
    if isinstance(result, dict) and isinstance(result.get("artifact"), dict):
        return result["artifact"]
    return latest_saved_behavior_artifact()


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


def render_control_panel(local_context: dict | None, steer_tooltips: dict | None = None) -> None:
    ui.section_header("Local control vectors", "Generate and launch llama.cpp control-vector runs.")
    control_sets = list_control_sets()
    control_vectors = list_control_vectors()

    c1, c2 = st.columns(2)
    with c1:
        control_set_names = [""] + [str(item.get("name", "")) for item in control_sets]
        current_control_set = st.session_state.get("llama_control_set", "")
        st.session_state.llama_control_set = st.selectbox(
            "Control set",
            control_set_names,
            index=control_set_names.index(current_control_set) if current_control_set in control_set_names else 0,
            help=steer_help("control_set", steer_tooltips),
        )
    with c2:
        control_vector_names = [""] + [str(item.get("name", "")) for item in control_vectors]
        current_control_vector = st.session_state.get("llama_control_vector", "")
        st.session_state.llama_control_vector = st.selectbox(
            "Control vector",
            control_vector_names,
            index=control_vector_names.index(current_control_vector) if current_control_vector in control_vector_names else 0,
            help=steer_help("control_vector", steer_tooltips),
        )

    with st.expander("Create control set", expanded=False):
        set_name = st.text_input("Set name", value="local_behavior", help=steer_help("set_name", steer_tooltips))
        pos = st.text_area("Positive prompts", height=120, help=steer_help("positive_prompts", steer_tooltips))
        neg = st.text_area("Negative prompts", height=120, help=steer_help("negative_prompts", steer_tooltips))
        if st.button("Save control set", width="stretch", help=steer_help("save_control_set", steer_tooltips)):
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
        st.session_state.llama_control_strength = st.number_input("Strength", value=float(st.session_state.llama_control_strength), step=0.25, help=steer_help("strength", steer_tooltips))
    with p2:
        st.session_state.llama_control_layer_start = st.number_input("Layer start", min_value=1, value=int(st.session_state.llama_control_layer_start), help=steer_help("layer_start", steer_tooltips))
    with p3:
        st.session_state.llama_control_layer_end = st.number_input("Layer end", min_value=1, value=int(st.session_state.llama_control_layer_end), help=steer_help("layer_end", steer_tooltips))

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
            vector_name = st.text_input("Vector name", value=selected.name, help=steer_help("vector_name", steer_tooltips))
            command = build_cvector_command(
                st.session_state.llama_model_path,
                selected["positive_path"],
                selected["negative_path"],
                f"data/control_vectors/{vector_name}.gguf",
                st.session_state.llama_cvector_generator,
            )
            st.code(shell_join(command), language="bash")
            if st.button("Generate vector", type="primary", width="stretch", help=steer_help("generate_vector", steer_tooltips)):
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


def render_activation_patch_panel(chat_backend: str, max_new_tokens: int, temperature: float, steer_tooltips: dict | None = None) -> None:
    ui.section_header("Activation Patch", "Capture local trace activations, save recipes, and compare patched llama.cpp runs.")
    artifact_options = saved_behavior_artifact_options()
    if not artifact_options:
        ui.empty_state("No source runs", "Run a traced prompt or batch before creating an activation patch.")
        return

    option_labels = [
        f"{row.get('run_id') or row.get('name')} | {row.get('mode')} | {Path(str(row.get('artifact_path'))).parent.name}"
        for row in artifact_options
    ]
    c1, c2 = st.columns(2)
    with c1:
        source_label = st.selectbox("Source run", option_labels, key="patch_source_run", help=steer_help("patch_source_run", steer_tooltips))
    with c2:
        target_label = st.selectbox("Target/current run", option_labels, index=0, key="patch_target_run", help=steer_help("patch_target_run", steer_tooltips))

    source_row = artifact_options[option_labels.index(source_label)]
    target_row = artifact_options[option_labels.index(target_label)]
    source_artifact = artifact_from_option(source_row)
    target_artifact = artifact_from_option(target_row)

    available = activation_path_df(source_artifact)
    available = available[available["trace_available"] != False] if not available.empty else pd.DataFrame()
    layer_values = sorted(pd.to_numeric(available.get("layer", pd.Series(dtype=int)), errors="coerce").dropna().astype(int).unique().tolist())
    if not layer_values:
        ui.empty_state("No captured activation rows", "The selected source run has no activation summaries.")
        return

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        layer = st.selectbox("Layer", layer_values, key="patch_layer", help=steer_help("patch_layer", steer_tooltips))
    with p2:
        token_start = st.number_input("Token start", min_value=0, value=0, step=1, key="patch_token_start", help=steer_help("patch_token_start", steer_tooltips))
    with p3:
        token_end = st.number_input("Token end", min_value=0, value=max(int(token_start), 0), step=1, key="patch_token_end", help=steer_help("patch_token_end", steer_tooltips))
    with p4:
        mode = st.selectbox("Patch mode", ["replace", "add_delta", "scale", "zero", "blend"], index=4, key="patch_mode", help=steer_help("patch_mode", steer_tooltips))

    n1, n2, n3 = st.columns(3)
    with n1:
        node_start_raw = st.text_input("Node/channel start", value="", key="patch_node_start", help=steer_help("patch_node_start", steer_tooltips))
    with n2:
        node_end_raw = st.text_input("Node/channel end", value="", key="patch_node_end", help=steer_help("patch_node_end", steer_tooltips))
    with n3:
        strength = st.slider("Strength", 0.0, 2.0, 0.35, 0.05, key="patch_strength", help=steer_help("patch_strength", steer_tooltips))

    prompt_default = ""
    for prompt_run in target_artifact.get("prompts", []):
        if prompt_run.get("prompt"):
            prompt_default = str(prompt_run["prompt"])
            break
    target_prompt = st.text_area("Target prompt", value=prompt_default or DEFAULT_CHAT_INPUT, height=90, help=steer_help("patch_target_prompt", steer_tooltips))
    recipe_name = st.text_input("Recipe name", value=f"patch-{source_artifact.get('run_id', 'source')}-layer-{layer}", help=steer_help("patch_recipe_name", steer_tooltips))

    def optional_int(raw: str) -> int | None:
        raw = raw.strip()
        return int(raw) if raw else None

    try:
        recipe = build_activation_patch_recipe(
            name=recipe_name,
            backend="llama.cpp",
            model=active_model_label(),
            source_run_id=str(source_artifact.get("run_id") or ""),
            target_run_id=str(target_artifact.get("run_id") or ""),
            layer=int(layer),
            token_start=int(token_start),
            token_end=int(token_end),
            node_start=optional_int(node_start_raw),
            node_end=optional_int(node_end_raw),
            mode=str(mode),
            strength=float(strength),
        )
        recipe = validate_activation_patch_recipe(recipe)
        validation_errors = []
    except Exception as exc:
        recipe = {}
        validation_errors = [str(exc)]

    if validation_errors:
        for issue in validation_errors:
            st.warning(issue)
    else:
        st.json(recipe, expanded=False)

    recipe_cols = st.columns(3)
    with recipe_cols[0]:
        if st.button("Save patch recipe", width="stretch", disabled=bool(validation_errors), help=steer_help("save_patch_recipe", steer_tooltips)):
            path = save_activation_patch_recipe(recipe, recipe_name)
            st.success(f"Saved {path.name}")
    with recipe_cols[1]:
        recipes = list_activation_patch_recipes()
        recipe_labels = [str(item.get("name")) for item in recipes]
        selected_recipe = st.selectbox("Load recipe", [""] + recipe_labels, key="patch_recipe_load", help=steer_help("patch_recipe_load", steer_tooltips))
        if selected_recipe:
            recipe_row = recipes[recipe_labels.index(selected_recipe)]
            st.session_state.loaded_activation_patch_recipe = load_activation_patch_recipe(str(recipe_row["path"]))
    with recipe_cols[2]:
        if st.session_state.get("loaded_activation_patch_recipe"):
            st.json(st.session_state.loaded_activation_patch_recipe, expanded=False)

    active_recipe = recipe
    loaded_recipe = st.session_state.get("loaded_activation_patch_recipe")
    if isinstance(loaded_recipe, dict):
        try:
            active_recipe = validate_activation_patch_recipe(loaded_recipe)
            st.caption(f"Loaded recipe active: {active_recipe.get('name', 'activation patch')}")
        except Exception as exc:
            st.warning(f"Loaded recipe is invalid: {exc}")
            active_recipe = recipe

    trace_url = st.session_state.llama_glass_url if chat_backend == "llama.cpp glass" else st.session_state.llama_url
    server_info = {}
    try:
        server_info = get_glass_info(trace_url, model_alias=st.session_state.llama_model_alias)
    except Exception:
        server_info = {}
    patch_supported = activation_patch_supported(server_info)
    st.caption(activation_patch_diagnostic(server_info))

    action_cols = st.columns(2)
    with action_cols[0]:
        run_patched = st.button("Run patched generation", type="primary", width="stretch", disabled=bool(validation_errors), help=steer_help("run_patched_generation", steer_tooltips))
    with action_cols[1]:
        run_compare = st.button("Compare patched vs baseline", width="stretch", disabled=bool(validation_errors), help=steer_help("compare_patched_vs_baseline", steer_tooltips))

    if run_patched or run_compare:
        baseline = ""
        patched = ""
        comparison_error = None
        try:
            if run_compare:
                baseline = chat_completion(
                    trace_url,
                    target_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    model_alias=st.session_state.llama_model_alias,
                )
            backend_patch = build_activation_patch_backend_payload(active_recipe, source_artifact)
            patched = chat_completion(
                trace_url,
                target_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                model_alias=st.session_state.llama_model_alias,
                activation_patch=backend_patch,
                activation_patch_supported=patch_supported,
            )
        except Exception as exc:
            comparison_error = str(exc)
        comparison = compare_patch_outputs(
            baseline_run_id=str(target_artifact.get("run_id") or ""),
            patched_run_id=f"{target_artifact.get('run_id') or 'target'}-patched",
            baseline_output=baseline,
            patched_output=patched,
        )
        comparison["error"] = comparison_error
        comparison["recipe"] = active_recipe
        comparison["action"] = "compare" if run_compare else "patched"
        st.session_state.last_activation_patch_comparison = comparison

    comparison = st.session_state.get("last_activation_patch_comparison")
    if isinstance(comparison, dict) and comparison:
        st.caption(f"changed {comparison.get('changed')} | length delta {comparison.get('length_delta')}")
        if comparison.get("error"):
            st.warning(comparison["error"])
        c1, c2 = st.columns(2)
        with c1:
            st.text_area("Baseline output", value=str(comparison.get("baseline_output") or ""), height=140)
        with c2:
            st.text_area("Patched output", value=str(comparison.get("patched_output") or ""), height=140)


def run_app() -> None:
    init_state()
    invalidate_model_dependent_state(
        st.session_state,
        st.session_state.get("llama_model_path", ""),
        st.session_state.get("llama_model_alias", ""),
    )
    active_backend = normalize_chat_backend(st.session_state.get("chat_backend_label", CHAT_BACKEND_NORMAL))
    st.session_state.chat_backend_label = chat_backend_display(active_backend)
    local_context = local_gguf_context(
        st.session_state.get("llama_model_path", ""),
        st.session_state.get("llama_model_alias", ""),
        active_backend,
    )
    summary = local_summary(local_context)
    steer_tooltips = ensure_tooltips("steer", STEER_CONTROL_METADATA)

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
        render_tab_workspace_controls("Run")
        apply_known_tab_state("Run", consume_loaded_tab_state("Run"))
        mode = st.segmented_control("Mode", ["Single message", "Batch run"], default=st.session_state.active_run_mode)
        st.session_state.active_run_mode = mode
        _tab_state("Run")["active_run_mode"] = mode
        c1, c2, c3 = st.columns(3)
        with c1:
            backend_label = st.selectbox("Backend", CHAT_BACKENDS, key="chat_backend_label")
            chat_backend = normalize_chat_backend(backend_label)
        with c2:
            max_new_tokens = st.slider("Max new tokens", 8, 512, 80, 8)
        with c3:
            temperature = st.slider("Temperature", 0.01, 1.5, 0.8, 0.05)

        auto_trace = st.toggle("Trace prompt", value=True, help="Calls the patched /glass-skull/trace endpoint before generation.")
        local_glass_info = (
            st.session_state.llama_status.glass_info if chat_backend == "llama.cpp normal" and st.session_state.llama_status else
            st.session_state.llama_glass_status.glass_info if st.session_state.llama_glass_status else {}
        )
        local_steering_supported = per_request_steering_supported(
            local_glass_info
        )
        include_trace_vectors = trace_vectors_supported(local_glass_info)
        startup_steered = chat_backend == "llama.cpp glass"
        use_steering = False if startup_steered else st.toggle("Use per-request steering", value=False, disabled=not local_steering_supported)
        steering_payload, steering_error = selected_control_vector_payload() if use_steering else (None, None)
        if use_steering and steering_error:
            st.warning(steering_error)

        if mode == "Batch run":
            pasted = st.text_area("Pasted prompts", height=120, key="batch_pasted_prompts", on_change=mark_batch_prompts_user_set)
            run_batch = st.button("Run batch", type="primary", width="stretch")
            _tab_state("Run")["batch_pasted_prompts"] = st.session_state.batch_pasted_prompts
        else:
            pasted = ""
            run_batch = False

        chat_box = st.container(height=420, border=True)
        with chat_box:
            if not st.session_state.chat_messages:
                ui.empty_state("No messages yet", "Send a prompt to start a local run.")
            for msg in st.session_state.chat_messages[-12:]:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    if msg.get("ts"):
                        st.caption(msg["ts"])

        prompt = seeded_chat_input() if mode == "Single message" else None
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
            trace_error = None
            trace_payload = None
            trace_plan = None
            trace_url = st.session_state.llama_glass_url if chat_backend == "llama.cpp glass" else st.session_state.llama_url
            if auto_trace:
                try:
                    trace_plan = resolve_trace_plan(
                        trace_url,
                        st.session_state.llama_model_alias,
                        summary,
                        ui_model_path=st.session_state.llama_model_path,
                        backend_process_model_path=managed_backend_process_model_path(),
                    )
                    if trace_plan.get("modelIdentityMismatch"):
                        raise RuntimeError(
                            "UI selected model does not match the running llama.cpp backend. "
                            f"{trace_plan.get('mismatchSummary')}. Restart llama.cpp with the selected GGUF before tracing."
                        )
                    trace_layers = trace_plan["layers"]
                    trace_payload = trace_glass_prompt(
                        trace_url,
                        prompt,
                        model_alias=st.session_state.llama_model_alias,
                        layers=trace_layers,
                        streams=["resid_pre"],
                        max_new_tokens=max_new_tokens,
                        top_k=32,
                        with_pieces=True,
                        include_vectors=include_trace_vectors,
                    )
                    set_local_dashboard_trace(trace_payload, prompt, chat_backend, active_model_label())
                except Exception as exc:
                    trace_error = str(exc)
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
            if isinstance(trace_plan, dict):
                dash_meta["trace_layers"] = trace_plan.get("layers")
                dash_meta["trace_plan"] = trace_plan
                dash_meta["backend_info"] = trace_plan.get("backend_info") if isinstance(trace_plan.get("backend_info"), dict) else local_glass_info
            else:
                dash_meta["backend_info"] = local_glass_info
            artifact = single_trace_artifact_from_llama(trace_payload or {}, dash_meta, output=output, error=error, trace_error=trace_error)
            store_behavior_artifact(artifact)
            persist_single_run_artifact(artifact)
            log_run(active_model_label(), "chat_generate", prompt, output=output, metadata={"backend": chat_backend, "run_id": run_id, "error": error})
            st.rerun()

        if run_batch and mode == "Batch run":
            prompt_items = batch_items_from_inputs(
                pasted_payload=pasted,
                repeat_prompt="",
                repeat_count=1,
                uploaded_items=[],
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
                    trace_url = st.session_state.llama_glass_url if chat_backend == "llama.cpp glass" else st.session_state.llama_url
                    trace_plan = resolve_trace_plan(
                        trace_url,
                        st.session_state.llama_model_alias,
                        summary,
                        ui_model_path=st.session_state.llama_model_path,
                        backend_process_model_path=managed_backend_process_model_path(),
                    ) if auto_trace else {"layers": [0], "backend_info": local_glass_info}
                    if auto_trace and trace_plan.get("modelIdentityMismatch"):
                        raise RuntimeError(
                            "UI selected model does not match the running llama.cpp backend. "
                            f"{trace_plan.get('mismatchSummary')}. Restart llama.cpp with the selected GGUF before tracing."
                        )
                    trace_layers = trace_plan["layers"]
                    result = run_fuzz_experiment(
                        name=run_id,
                        prompts=prompt_items,
                        chat_backend="llama.cpp",
                        trace_enabled=bool(auto_trace),
                        llama_url=trace_url,
                        llama_model_alias=st.session_state.llama_model_alias,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        layers=trace_layers,
                        streams=["resid_pre"],
                        top_k=32,
                        include_vectors=include_trace_vectors,
                        backend_info=trace_plan.get("backend_info") if isinstance(trace_plan.get("backend_info"), dict) else local_glass_info,
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
        render_tab_workspace_controls("Map")
        apply_known_tab_state("Map", consume_loaded_tab_state("Map"))
        st.session_state.behavior_profile = st.selectbox("Behavior profile", list_behavior_profiles(), index=list_behavior_profiles().index(st.session_state.behavior_profile))
        artifact = latest_behavior_artifact()
        if not artifact:
            ui.empty_state("No run artifact yet", "Run a single message or batch to populate the map.")
        else:
            prompt_options = [
                prompt.get("prompt_id")
                for prompt in artifact.get("prompts", [])
                if prompt.get("prompt_id") is not None
            ]
            batch_options = [
                str(prompt.get("batch_id") or f"batch-{prompt.get('prompt_id', index)}")
                for index, prompt in enumerate(artifact.get("prompts", []))
            ]
            token_options = sorted({
                row.get("token_index", row.get("token_id"))
                for prompt in artifact.get("prompts", [])
                for row in prompt.get("trace_rows", [])
                if row.get("token_index", row.get("token_id")) is not None
            })
            map_cols = st.columns([1.2, 1, 1, 1, 1])
            with map_cols[0]:
                mode_options = ["auto", "single_prompt", "batch_overlay", "aggregate_heatmap", "compare_prompts"]
                current_mode = st.session_state.get("map_visualization_mode") or "auto"
                st.session_state.map_visualization_mode = st.selectbox(
                    "Map mode",
                    mode_options,
                    index=mode_options.index(current_mode) if current_mode in mode_options else 0,
                )
            with map_cols[1]:
                prompt_choices = ["auto"] + [str(item) for item in prompt_options]
                current_prompt = "auto" if st.session_state.get("map_selected_prompt") is None else str(st.session_state.map_selected_prompt)
                selected_prompt_text = st.selectbox(
                    "Prompt",
                    prompt_choices,
                    index=prompt_choices.index(current_prompt) if current_prompt in prompt_choices else 0,
                    key="map_selected_prompt_choice",
                )
                st.session_state.map_selected_prompt = None if selected_prompt_text == "auto" else selected_prompt_text
            with map_cols[2]:
                batch_choices = ["auto"] + batch_options
                current_batch = st.session_state.get("map_selected_batch") or "auto"
                st.session_state.map_selected_batch = st.selectbox(
                    "Batch",
                    batch_choices,
                    index=batch_choices.index(current_batch) if current_batch in batch_choices else 0,
                )
                if st.session_state.map_selected_batch == "auto":
                    st.session_state.map_selected_batch = None
            with map_cols[3]:
                token_choices = ["auto"] + [str(item) for item in token_options]
                current_token = "auto" if st.session_state.get("map_selected_token") is None else str(st.session_state.map_selected_token)
                selected_token_text = st.selectbox(
                    "Token",
                    token_choices,
                    index=token_choices.index(current_token) if current_token in token_choices else 0,
                )
                st.session_state.map_selected_token = None if selected_token_text == "auto" else int(selected_token_text)
            with map_cols[4]:
                st.session_state.map_top_k = st.slider("Top K", 1, 32, int(st.session_state.get("map_top_k") or 8), 1)
            settings_cols = st.columns([1, 1, 1, 1])
            with settings_cols[0]:
                st.session_state.map_background_opacity = st.slider("Background", 0.0, 1.0, float(st.session_state.map_background_opacity), 0.05)
            with settings_cols[1]:
                st.session_state.map_edge_threshold = st.slider("Edge threshold", 0.0, 1.0, float(st.session_state.map_edge_threshold), 0.05)
            with settings_cols[2]:
                st.session_state.map_show_aggregate_heatmap = st.toggle("Aggregate heatmap", value=bool(st.session_state.map_show_aggregate_heatmap))
            with settings_cols[3]:
                st.session_state.map_show_secondary_branches = st.toggle("Secondary branches", value=bool(st.session_state.map_show_secondary_branches))
            _tab_state("Map").update({
                "visualization_mode": st.session_state.map_visualization_mode,
                "selected_prompt": st.session_state.map_selected_prompt,
                "selected_batch": st.session_state.map_selected_batch,
                "selected_token": st.session_state.map_selected_token,
                "top_k": st.session_state.map_top_k,
                "background_opacity": st.session_state.map_background_opacity,
                "edge_threshold": st.session_state.map_edge_threshold,
                "show_aggregate_heatmap": st.session_state.map_show_aggregate_heatmap,
                "show_secondary_branches": st.session_state.map_show_secondary_branches,
                "annotation_selected_group": st.session_state.map_annotation_selected_group,
            })
            artifact_summary = artifact.get("summary", {}) if isinstance(artifact.get("summary"), dict) else {}
            st.caption(
                " | ".join([
                    f"run {artifact.get('run_id', 'latest')}",
                    f"trace rows {artifact_summary.get('trace_row_count', 0)}",
                    f"unavailable {artifact_summary.get('trace_unavailable_count', 0)}",
                    f"trace supported {artifact_summary.get('trace_supported', False)}",
                ])
            )
            payload = build_activation_map_payload(
                artifact,
                summary,
                local_model_context=local_context,
                backend_info=local_glass_info,
                ui_selected_model_path=st.session_state.llama_model_path,
                ui_selected_model_name=active_model_label(),
                backend_process_model_path=managed_backend_process_model_path(),
                active_model_fingerprint=st.session_state.get("activeModelFingerprint", ""),
                visualization_mode=None if st.session_state.map_visualization_mode == "auto" else st.session_state.map_visualization_mode,
                selected_prompt=st.session_state.map_selected_prompt,
                selected_token=int(st.session_state.map_selected_token) if st.session_state.map_selected_token is not None else None,
                selected_batch=st.session_state.map_selected_batch,
                top_k=int(st.session_state.map_top_k),
                background_opacity=float(st.session_state.map_background_opacity),
                edge_threshold=float(st.session_state.map_edge_threshold),
                show_aggregate_heatmap=bool(st.session_state.map_show_aggregate_heatmap),
                show_secondary_branches=bool(st.session_state.map_show_secondary_branches),
            )
            patch_comparison = st.session_state.get("last_activation_patch_comparison")
            if isinstance(patch_comparison, dict) and patch_comparison.get("recipe"):
                payload["activationPatch"] = patch_comparison["recipe"]
                payload["patchComparison"] = {
                    "baselineRunId": patch_comparison.get("baseline_run_id"),
                    "patchedRunId": patch_comparison.get("patched_run_id"),
                    "changed": patch_comparison.get("changed"),
                    "lengthDelta": patch_comparison.get("length_delta"),
                    "error": patch_comparison.get("error"),
                }
            render_activation_map(payload, key=f"activation_map_{artifact.get('run_id', 'latest')}", height=920)
            render_node_annotation_inspector(payload)
            _tab_state("Map")["annotation_selected_group"] = st.session_state.map_annotation_selected_group
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
        render_tab_workspace_controls("Steer")
        render_control_panel(local_context, steer_tooltips)
        st.markdown("---")
        render_activation_patch_panel(chat_backend, max_new_tokens, temperature, steer_tooltips)

    with tabs["Timeline"]:
        ui.section_header("Timeline", "Behavior scores and saved local run history.")
        render_tab_workspace_controls("Timeline")
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
        render_tab_workspace_controls("Model")
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
        render_tab_workspace_controls("Settings")
        render_global_workspace_controls()
        next_model_alias = st.text_input("Model alias", value=st.session_state.llama_model_alias).strip()
        next_model_path = st.text_input("GGUF model path", value=st.session_state.llama_model_path)
        if next_model_alias != st.session_state.llama_model_alias or next_model_path != st.session_state.llama_model_path:
            st.session_state.llama_model_alias = next_model_alias
            st.session_state.llama_model_path = next_model_path
            invalidate_model_dependent_state(st.session_state, next_model_path, next_model_alias)
            st.warning("Model selection changed. Cached backend info and trace artifacts were cleared; restart llama.cpp with this GGUF before tracing.")
            st.rerun()
        st.session_state.llama_url = st.text_input("Normal server URL", value=st.session_state.llama_url)
        st.session_state.llama_glass_url = st.text_input("Steered/server trace URL", value=st.session_state.llama_glass_url)
        st.session_state.llama_server_bin = st.text_input("llama-server binary", value=st.session_state.llama_server_bin)
        st.session_state.llama_cvector_generator = st.text_input("llama-cvector-generator binary", value=st.session_state.llama_cvector_generator)
        st.session_state.llama_control_port = st.number_input("Steered server port", min_value=1, max_value=65535, value=int(st.session_state.llama_control_port))
        st.session_state.llama_control_extra_args = st.text_input("Extra llama-server args", value=st.session_state.llama_control_extra_args)


run_app()
