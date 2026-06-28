from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, MutableMapping

from .config import GLOBAL_WORKSPACE_DIR, TAB_WORKSPACE_DIR
from .experiment_store import safe_slug
from .node_annotations import annotation_file_metadata


GLOBAL_STATE_KEYS = [
    "active_run_id",
    "active_run_mode",
    "chat_backend_label",
    "llama_model_alias",
    "llama_model_path",
    "llama_url",
    "llama_glass_url",
    "llama_server_bin",
    "llama_cvector_generator",
    "behavior_profile",
    "map_visualization_mode",
    "map_selected_prompt",
    "map_selected_batch",
    "map_selected_token",
    "map_top_k",
    "map_background_opacity",
    "map_edge_threshold",
    "map_show_aggregate_heatmap",
    "map_show_secondary_branches",
    "llama_control_set",
    "llama_control_vector",
    "llama_control_strength",
    "llama_control_layer_start",
    "llama_control_layer_end",
    "llama_control_port",
    "llama_control_extra_args",
    "loaded_activation_patch_recipe",
    "last_activation_patch_comparison",
    "tab_state",
]

GLOBAL_CLEAR_DEFAULTS: dict[str, Any] = {
    "active_run_id": None,
    "last_run_id": None,
    "last_output": "",
    "local_dashboard_trace": None,
    "local_dashboard_trace_meta": {},
    "local_dashboard_trace_counter": 0,
    "last_batch_result": None,
    "last_fuzz_result": None,
    "last_behavior_artifact": None,
    "last_behavior_scores": None,
    "behavior_run_history": [],
    "loaded_activation_patch_recipe": None,
    "last_activation_patch_comparison": None,
    "tab_state": {},
    "map_selected_prompt": None,
    "map_selected_batch": None,
    "map_selected_token": None,
}


@dataclass
class WorkspaceResult:
    name: str
    path: Path
    state: dict[str, Any]
    error: str | None = None
    warning: str | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        try:
            return _jsonable(value.to_dict(orient="records"))
        except TypeError:
            try:
                return _jsonable(value.to_dict())
            except Exception:
                pass
    return str(value)


def _workspace_name(state: MutableMapping[str, Any] | None, explicit: str | None, key: str) -> str:
    raw = explicit or (state.get(key) if state is not None else None) or "default"
    return safe_slug(str(raw))


def _global_path(name: str) -> Path:
    return GLOBAL_WORKSPACE_DIR / f"{safe_slug(name)}.json"


def _tab_dir(tab_name: str) -> Path:
    return TAB_WORKSPACE_DIR / safe_slug(tab_name)


def _tab_path(tab_name: str, name: str) -> Path:
    return _tab_dir(tab_name) / f"{safe_slug(name)}.json"


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def list_workspaces() -> list[str]:
    GLOBAL_WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(path.stem for path in GLOBAL_WORKSPACE_DIR.glob("*.json"))


def list_tab_states(tab_name: str) -> list[str]:
    path = _tab_dir(tab_name)
    path.mkdir(parents=True, exist_ok=True)
    return sorted(item.stem for item in path.glob("*.json"))


def collect_workspace_state(state: MutableMapping[str, Any]) -> dict[str, Any]:
    selected = {key: _jsonable(state.get(key)) for key in GLOBAL_STATE_KEYS if key in state}
    selected.setdefault("active_run_id", None)
    selected.setdefault("active_run_mode", "Single message")
    selected.setdefault("tab_state", {})
    selected["model"] = {
        "alias": _jsonable(state.get("llama_model_alias", "")),
        "path": _jsonable(state.get("llama_model_path", "")),
        "normal_url": _jsonable(state.get("llama_url", "")),
        "glass_url": _jsonable(state.get("llama_glass_url", "")),
        "server_bin": _jsonable(state.get("llama_server_bin", "")),
        "cvector_generator": _jsonable(state.get("llama_cvector_generator", "")),
        "backend": _jsonable(state.get("chat_backend_label", "")),
    }
    selected["map"] = {
        "visualization_mode": _jsonable(state.get("map_visualization_mode", "")),
        "selected_prompt": _jsonable(state.get("map_selected_prompt")),
        "selected_batch": _jsonable(state.get("map_selected_batch")),
        "selected_token": _jsonable(state.get("map_selected_token")),
        "top_k": _jsonable(state.get("map_top_k", 8)),
        "background_opacity": _jsonable(state.get("map_background_opacity", 0.24)),
        "edge_threshold": _jsonable(state.get("map_edge_threshold", 0.0)),
        "show_aggregate_heatmap": _jsonable(state.get("map_show_aggregate_heatmap", False)),
        "show_secondary_branches": _jsonable(state.get("map_show_secondary_branches", True)),
    }
    selected["steering"] = {
        "control_set": _jsonable(state.get("llama_control_set", "")),
        "control_vector": _jsonable(state.get("llama_control_vector", "")),
        "strength": _jsonable(state.get("llama_control_strength", 1.25)),
        "layer_start": _jsonable(state.get("llama_control_layer_start", 1)),
        "layer_end": _jsonable(state.get("llama_control_layer_end", 32)),
        "port": _jsonable(state.get("llama_control_port", 8088)),
        "extra_args": _jsonable(state.get("llama_control_extra_args", "")),
        "patch_recipe": _jsonable(state.get("loaded_activation_patch_recipe")),
    }
    selected["annotations"] = annotation_file_metadata()
    return selected


def save_workspace(state: MutableMapping[str, Any], name: str | None = None) -> WorkspaceResult:
    workspace_name = _workspace_name(state, name, "workspace_name")
    workspace_state = collect_workspace_state(state)
    path = _global_path(workspace_name)
    _write_payload(path, {
        "schema_version": 1,
        "scope": "global",
        "name": workspace_name,
        "saved_at": _now_iso(),
        "state": workspace_state,
    })
    state["workspace_name"] = workspace_name
    return WorkspaceResult(workspace_name, path, workspace_state)


def save_workspace_as(state: MutableMapping[str, Any], name: str) -> WorkspaceResult:
    return save_workspace(state, name=name)


def load_workspace(name: str) -> WorkspaceResult:
    workspace_name = safe_slug(name)
    path = _global_path(workspace_name)
    if not path.exists():
        return WorkspaceResult(workspace_name, path, {}, warning=f"Workspace not found: {workspace_name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return WorkspaceResult(workspace_name, path, {}, error=f"Invalid workspace JSON: {exc}")
    except OSError as exc:
        return WorkspaceResult(workspace_name, path, {}, error=f"Could not load workspace: {exc}")
    state = payload.get("state")
    if not isinstance(state, dict):
        return WorkspaceResult(workspace_name, path, {}, error="Invalid workspace JSON: missing state object")
    return WorkspaceResult(str(payload.get("name") or workspace_name), path, state)


def apply_workspace_state(target: MutableMapping[str, Any], state: dict[str, Any]) -> None:
    for key, value in state.items():
        if key in {"model", "map", "steering"}:
            continue
        target[key] = value
    model = state.get("model") if isinstance(state.get("model"), dict) else {}
    map_state = state.get("map") if isinstance(state.get("map"), dict) else {}
    steering = state.get("steering") if isinstance(state.get("steering"), dict) else {}
    key_map = {
        "alias": "llama_model_alias",
        "path": "llama_model_path",
        "normal_url": "llama_url",
        "glass_url": "llama_glass_url",
        "server_bin": "llama_server_bin",
        "cvector_generator": "llama_cvector_generator",
        "backend": "chat_backend_label",
    }
    for source, dest in key_map.items():
        if source in model:
            target[dest] = model[source]
    for source, dest in {
        "visualization_mode": "map_visualization_mode",
        "selected_prompt": "map_selected_prompt",
        "selected_batch": "map_selected_batch",
        "selected_token": "map_selected_token",
        "top_k": "map_top_k",
        "background_opacity": "map_background_opacity",
        "edge_threshold": "map_edge_threshold",
        "show_aggregate_heatmap": "map_show_aggregate_heatmap",
        "show_secondary_branches": "map_show_secondary_branches",
    }.items():
        if source in map_state:
            target[dest] = map_state[source]
    for source, dest in {
        "control_set": "llama_control_set",
        "control_vector": "llama_control_vector",
        "strength": "llama_control_strength",
        "layer_start": "llama_control_layer_start",
        "layer_end": "llama_control_layer_end",
        "port": "llama_control_port",
        "extra_args": "llama_control_extra_args",
        "patch_recipe": "loaded_activation_patch_recipe",
    }.items():
        if source in steering:
            target[dest] = steering[source]


def clear_workspace(state: MutableMapping[str, Any]) -> None:
    for key, value in GLOBAL_CLEAR_DEFAULTS.items():
        state[key] = _jsonable(value)


def save_tab_state(tab_name: str, state: MutableMapping[str, Any] | dict[str, Any] | None = None, name: str | None = None) -> WorkspaceResult:
    tab_state = dict(state or {})
    current_names = tab_state.get("current_tab_workspace_names") if isinstance(tab_state.get("current_tab_workspace_names"), dict) else {}
    explicit_name = name or current_names.get(tab_name) if isinstance(current_names, dict) else name
    tab_workspace_name = safe_slug(str(explicit_name or "default"))
    path = _tab_path(tab_name, tab_workspace_name)
    clean_state = _jsonable(tab_state)
    payload = {
        "schema_version": 1,
        "scope": "tab",
        "tab_name": tab_name,
        "name": tab_workspace_name,
        "saved_at": _now_iso(),
        "state": clean_state,
    }
    _write_payload(path, payload)
    return WorkspaceResult(tab_workspace_name, path, clean_state)


def save_tab_state_as(tab_name: str, state: MutableMapping[str, Any] | dict[str, Any], name: str) -> WorkspaceResult:
    return save_tab_state(tab_name, state, name=name)


def load_tab_state(tab_name: str, name: str) -> WorkspaceResult:
    tab_workspace_name = safe_slug(name)
    path = _tab_path(tab_name, tab_workspace_name)
    if not path.exists():
        return WorkspaceResult(tab_workspace_name, path, {}, warning=f"Tab state not found: {tab_name}/{tab_workspace_name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return WorkspaceResult(tab_workspace_name, path, {}, error=f"Invalid workspace JSON: {exc}")
    except OSError as exc:
        return WorkspaceResult(tab_workspace_name, path, {}, error=f"Could not load tab state: {exc}")
    state = payload.get("state")
    if not isinstance(state, dict):
        return WorkspaceResult(tab_workspace_name, path, {}, error="Invalid workspace JSON: missing state object")
    return WorkspaceResult(str(payload.get("name") or tab_workspace_name), path, state)


def clear_tab_state(state: MutableMapping[str, Any], tab_name: str) -> None:
    tabs = state.get("tab_state")
    if not isinstance(tabs, dict):
        tabs = {}
        state["tab_state"] = tabs
    tabs[tab_name] = {}
