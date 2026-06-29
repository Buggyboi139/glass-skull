from __future__ import annotations

import ast
import importlib
import json
import inspect
from pathlib import Path
import types
from typing import Any

import pandas as pd

from glass_skull.activation_map import (
    build_activation_map_payload,
    build_model_meta,
    inspect_trace_artifact,
    resolve_render_layers,
    trace_layer_diagnostics,
)
from glass_skull.activation_map_view import activation_map_html
from glass_skull.behavior_profiles import get_behavior_profile, list_behavior_profiles
from glass_skull.behavior_scoring import behavior_timeline_df, score_behavior_output, score_run_artifact
from glass_skull.chat_store import load_chat, save_chat
from glass_skull.config import (
    CONTROL_SET_DIR,
    CONTROL_VECTOR_DIR,
    DEFAULT_BATCH_MESSAGES,
    DEFAULT_GGUF_MODEL_PATH,
    TOOLTIP_DIR,
    ensure_dirs,
    seed_batch_prompt_default,
    seed_missing_defaults,
)
from glass_skull.experiment_store import safe_slug
from glass_skull.fuzzing import run_fuzz_experiment
from glass_skull import activation_map_view
from glass_skull.activation_map_view import _legend_panel_html
from glass_skull.node_annotations import (
    add_tag,
    delete_annotation,
    delete_note,
    delete_tag,
    get_annotations_for_node,
    load_annotations,
    save_annotations,
    upsert_annotation,
    update_note,
)
from glass_skull.llama_client import (
    build_steering_metadata,
    chat_completion,
    normalize_base_url,
    per_request_steering_supported,
    trace_glass_prompt,
)
from glass_skull.llama_control import (
    DEFAULT_CVECTOR_GENERATOR,
    DEFAULT_LLAMA_SERVER,
    build_cvector_command,
    build_llama_server_command,
    classify_cvector_failure,
    preflight_control_vector_run,
    read_gguf_tensor_index,
    shell_join,
)
from glass_skull.path_mapping import rank_activation_paths, recommended_steering_targets
from glass_skull.prompt_loader import load_jsonl, load_txt
from glass_skull.run_artifacts import (
    TRACE_COLUMNS,
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
    gguf_tensor_shape_scatter_fig,
    label_activation_heatmap,
    path_rank_bar_fig,
)
from glass_skull.workspaces import (
    apply_workspace_state,
    apply_tab_state,
    clear_tab_state,
    clear_workspace,
    collect_tab_state,
    load_tab_state,
    load_workspace,
    save_tab_state,
    save_tab_state_as,
    save_workspace,
    save_workspace_as,
)
from glass_skull.tooltip_generator import ensure_tooltips, tooltip_text


def _write_fake_gguf(path: Path) -> None:
    def write_u32(f, value: int) -> None:
        f.write(int(value).to_bytes(4, "little", signed=False))

    def write_u64(f, value: int) -> None:
        f.write(int(value).to_bytes(8, "little", signed=False))

    def write_string(f, value: str) -> None:
        data = value.encode("utf-8")
        write_u64(f, len(data))
        f.write(data)

    with path.open("wb") as f:
        f.write(b"GGUF")
        write_u32(f, 3)
        write_u64(f, 1)
        write_u64(f, 2)

        write_string(f, "general.architecture")
        write_u32(f, 8)
        write_string(f, "llama")

        write_string(f, "llama.block_count")
        write_u32(f, 4)
        write_u32(f, 2)

        write_string(f, "blk.0.attn_q.weight")
        write_u32(f, 2)
        write_u64(f, 16)
        write_u64(f, 16)
        write_u32(f, 0)
        write_u64(f, 0)


def _fake_llama_request(captured: list[dict[str, Any]]):
    def request_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 120.0) -> dict[str, Any]:
        captured.append({"method": method, "url": url, "payload": payload, "timeout": timeout})
        if url.endswith("/v1/chat/completions"):
            return {"choices": [{"message": {"content": "local reply"}}]}
        if url.endswith("/completion"):
            return {"content": "legacy reply"}
        if url.endswith("/glass-skull/trace"):
            return {
                "model": "local",
                "prompt": {"traces": [{"tokens": [10, 20], "pieces": [{"piece": "hi"}, {"piece": " there"}]}]},
                "layer_inputs": [
                    {"layer": 0, "stream": "resid_pre", "token_index": 0, "l2_norm": 1.25, "top_dims": [
                        {"dimension": 3, "activation": 0.5},
                        {"dimension": 7, "activation": -0.4},
                        {"dimension": 12, "activation": 0.3},
                    ]},
                    {"layer": 1, "stream": "resid_pre", "token_index": 1, "l2_norm": 2.5, "top_dims": [
                        {"dimension": 4, "activation": -1.0},
                        {"dimension": 8, "activation": 0.7},
                        {"dimension": 13, "activation": -0.2},
                    ]},
                ],
                "logits": {"supported": False, "reason": "not requested"},
            }
        if url.endswith("/v1/models"):
            return {"data": [{"id": "local", "status": "loaded"}]}
        if "/glass-skull/info" in url:
            return {
                "layers": 40,
                "meta": {"n_layer": 40},
                "capabilities": {
                    "steering": {"per_request": {"supported": True}},
                    "trace": {"layer_inputs": {"supported": True}},
                    "activation_patch": {"supported": True},
                },
            }
        raise AssertionError(url)

    return request_json


class _SessionState(dict):
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _load_main_functions(names: set[str], namespace: dict[str, Any]) -> dict[str, Any]:
    source = Path("main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    found = {node.name for node in body}
    missing = names - found
    assert not missing, f"missing main.py functions: {sorted(missing)}"
    ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(ast.Module(body=body, type_ignores=[]), "main.py", "exec"), namespace)
    return {name: namespace[name] for name in names}


def _activation_patch_api() -> dict[str, Any]:
    activation_patch = importlib.import_module("glass_skull.activation_patch")
    experiment_store = importlib.import_module("glass_skull.experiment_store")
    names = [
        "PATCH_MODES",
        "build_activation_patch_backend_payload",
        "build_activation_patch_recipe",
        "compare_patch_outputs",
        "validate_activation_patch_recipe",
    ]
    store_names = [
        "list_activation_patch_recipes",
        "load_activation_patch_recipe",
        "save_activation_patch_recipe",
    ]
    api = {name: getattr(activation_patch, name) for name in names}
    api.update({name: getattr(experiment_store, name) for name in store_names})
    return api


def _assert_raises(exc_type: type[BaseException], func, *args, **kwargs) -> None:
    try:
        func(*args, **kwargs)
    except exc_type:
        return
    raise AssertionError(f"{getattr(func, '__name__', func)!r} did not raise {exc_type.__name__}")


def _assert_activation_patch_recipe_round_trip() -> None:
    api = _activation_patch_api()
    assert set(api["PATCH_MODES"]) == {"replace", "add_delta", "scale", "zero", "blend"}
    recipe = api["build_activation_patch_recipe"](
        name="smoke patch",
        mode="blend",
        layer=7,
        token_start=2,
        token_end=5,
        node_start=1,
        node_end=3,
        strength=0.6,
        source_run_id="baseline",
        target_run_id="patched",
    )
    validated = api["validate_activation_patch_recipe"](recipe)
    assert validated["mode"] == "blend"
    patch = recipe["patches"][0]
    assert patch["mode"] == "blend"
    assert patch["layer"] == 7
    assert patch["token_range"] == [2, 5]
    assert patch["node_range"] == [1, 3]
    assert patch["strength"] == 0.6
    vector_rows = normalize_llama_trace(
        {
            "layer_inputs": [{
                "layer": 7,
                "stream": "resid_pre",
                "token_index": 2,
                "token": "hi",
                "activation_norm": 1.0,
                "vector": [0.1, 0.2, 0.3, 0.4],
            }]
        },
        prompt_id=0,
        label="source",
        metadata={"run_id": "baseline"},
    )
    assert vector_rows[0]["vector"] == [0.1, 0.2, 0.3, 0.4]
    source_artifact = build_run_artifact(
        run_id="baseline",
        mode="Single message",
        backend="llama.cpp",
        model="local",
        prompts=[{
            "prompt_id": 0,
            "label": "source",
            "prompt": "hi",
            "output": "plain answer",
            "trace_rows": vector_rows,
        }],
    )
    backend_payload = api["build_activation_patch_backend_payload"](recipe, source_artifact)
    assert backend_payload["recipe"]["patches"][0]["mode"] == "blend"
    assert backend_payload["source_vectors"][0]["vector"] == [0.1, 0.2, 0.3, 0.4]

    path = api["save_activation_patch_recipe"](recipe, "smoke patch")
    try:
        loaded = api["load_activation_patch_recipe"](path)
        assert loaded == recipe
        assert any(row.get("name") == "smoke_patch" for row in api["list_activation_patch_recipes"]())
    finally:
        Path(path).unlink(missing_ok=True)

    def invalid_patch(**updates: Any) -> dict[str, Any]:
        bad = json.loads(json.dumps(recipe))
        bad["patches"][0].update(updates)
        return bad

    _assert_raises(ValueError, api["validate_activation_patch_recipe"], invalid_patch(mode="invalid"))
    _assert_raises(ValueError, api["validate_activation_patch_recipe"], invalid_patch(layer=-1))
    _assert_raises(ValueError, api["validate_activation_patch_recipe"], invalid_patch(token_range=[6, 5]))
    _assert_raises(ValueError, api["validate_activation_patch_recipe"], invalid_patch(strength="hard"))

    comparison = api["compare_patch_outputs"]("plain answer", "clear concise safe action")
    assert comparison["baseline_output"] == "plain answer"
    assert comparison["patched_output"] == "clear concise safe action"
    assert comparison["changed"] is True


def _assert_main_layer_resolver_and_trace_mock(summary: dict, local_meta: dict) -> None:
    import glass_skull.llama_client as llama_client

    captured: list[dict[str, Any]] = []
    original_request_json = llama_client._request_json
    namespace = {
        "get_glass_info": llama_client.get_glass_info,
    }
    funcs = _load_main_functions(
        {
            "_positive_int",
            "_trace_layer_count_from_glass_info",
            "model_identity_diagnostics",
            "resolve_trace_plan",
            "resolve_trace_layers",
        },
        namespace,
    )
    resolve_trace_layers = funcs["resolve_trace_layers"]
    try:
        llama_client._request_json = _fake_llama_request(captured)
        resolved = resolve_trace_layers("http://local", "local", {"layers": 41})
        assert resolved == list(range(40))
        trace = trace_glass_prompt(
            "http://local",
            "hello",
            model_alias="local",
            layers=resolved,
            streams=["resid_pre"],
            max_new_tokens=8,
            with_pieces=True,
        )
    finally:
        llama_client._request_json = original_request_json

    info_requests = [r for r in captured if "/glass-skull/info" in r["url"]]
    assert info_requests
    trace_request = [r for r in captured if r["url"].endswith("/glass-skull/trace")][0]
    assert 40 not in trace_request["payload"]["layers"]
    assert trace_request["payload"]["layers"][-1] == 39
    normalized = normalize_llama_trace(trace, prompt_id=0, label="single", metadata={"run_id": "mock40"})
    assert len(normalized) > 0
    mock_artifact = build_run_artifact(
        run_id="mock40",
        mode="Single message",
        backend="llama.cpp",
        model="local",
        prompts=[{"prompt_id": 0, "label": "single", "prompt": "hello", "output": "ok", "trace_rows": normalized}],
    )
    mock_payload = build_activation_map_payload(mock_artifact, summary, local_model_context={**local_meta, "block_count": 41})
    assert mock_payload["activationPaths"]

    namespace["get_glass_info"] = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("missing server"))
    funcs = _load_main_functions(
        {
            "_positive_int",
            "_trace_layer_count_from_glass_info",
            "model_identity_diagnostics",
            "resolve_trace_plan",
            "resolve_trace_layers",
        },
        namespace,
    )
    assert funcs["resolve_trace_layers"]("http://missing", "local", {"layers": 41}) == list(range(41))
    assert funcs["resolve_trace_layers"]("http://missing", "local", {"layers": 0}) == [0]
    namespace["get_glass_info"] = lambda *args, **kwargs: {"layers": "bad", "meta": {"n_layer": -3}}
    funcs = _load_main_functions(
        {
            "_positive_int",
            "_trace_layer_count_from_glass_info",
            "model_identity_diagnostics",
            "resolve_trace_plan",
            "resolve_trace_layers",
        },
        namespace,
    )
    assert funcs["resolve_trace_layers"]("http://bad", "local", {"layers": "invalid"}) == [0]


def _assert_model_layer_contracts(summary: dict, local_meta: dict) -> None:
    namespace = {
        "pd": pd,
        "get_glass_info": lambda *args, **kwargs: {
            "layers": 60,
            "meta": {"n_layer": 60},
            "model": {
                "path": "/models/gemma-4-31b.gguf",
                "id": "gemma-live",
                "architecture": "gemma",
            },
        },
    }
    funcs = _load_main_functions(
        {
            "_positive_int",
            "_trace_layer_count_from_glass_info",
            "resolve_trace_layers",
            "resolve_trace_plan",
            "model_identity_diagnostics",
            "clear_model_dependent_state",
            "invalidate_model_dependent_state",
            "invalidate_if_model_backend_changed",
            "model_backend_snapshot",
        },
        namespace,
    )

    stale_qwen_summary = {**summary, "layers": 40}
    assert funcs["resolve_trace_layers"]("http://local", "gemma-live", stale_qwen_summary) == list(range(60))
    trace_plan = funcs["resolve_trace_plan"](
        "http://local",
        "gemma-live",
        stale_qwen_summary,
        ui_model_path="/models/gemma-4-31b.gguf",
    )
    assert trace_plan["layers"] == list(range(60))
    assert trace_plan["traceRequestedMinLayer"] == 0
    assert trace_plan["traceRequestedMaxLayer"] == 59
    assert trace_plan["traceRequestedLayerCount"] == 60
    assert trace_plan["modelIdentityMismatch"] is False

    mismatch = funcs["model_identity_diagnostics"](
        "/models/gemma-4-31b.gguf",
        {"model": {"path": "/models/qwen.gguf", "id": "qwen-live"}, "layers": 40},
        backend_process_model_path="/models/qwen.gguf",
    )
    assert mismatch["modelIdentityMismatch"] is True
    assert "UI selected model path differs from backend info model path" in mismatch["mismatchSummary"]
    assert "backend process model path" in mismatch["mismatchSummary"]
    name_mismatch = funcs["model_identity_diagnostics"](
        "",
        {"model": {"id": "qwen-live"}, "layers": 40},
        ui_selected_model_name="gemma-live",
    )
    assert name_mismatch["modelIdentityMismatch"] is True
    assert "UI selected model name differs from backend info model name" in name_mismatch["mismatchSummary"]

    state = _SessionState(
        activeModelFingerprint="old",
        llama_status={"old": True},
        llama_glass_status={"old": True},
        local_dashboard_trace={"old": True},
        local_dashboard_trace_meta={"old": True},
        last_batch_result={"old": True},
        last_fuzz_result={"old": True},
        last_behavior_artifact={"run_id": "old"},
        last_behavior_scores="old",
        map_selected_prompt="0",
        map_selected_batch="batch-0",
        map_selected_token=3,
        map_annotation_selected_group="L39-N1",
    )
    changed = funcs["invalidate_model_dependent_state"](state, "/models/gemma-4-31b.gguf", "gemma-live")
    assert changed is True
    assert state["llama_status"] is None
    assert state["llama_glass_status"] is None
    assert state["last_behavior_artifact"] is None
    assert state["last_batch_result"] is None
    assert state["map_selected_token"] is None
    assert state["modelArtifactsInvalidated"] is True
    same_model_new_url_state = _SessionState(
        activeModelFingerprint="/models/gemma-4-31b.gguf|gemma-live",
        llama_model_path="/models/gemma-4-31b.gguf",
        llama_model_alias="gemma-live",
        llama_url="http://old",
        llama_glass_url="http://old-glass",
        last_behavior_artifact={"run_id": "stale"},
        last_behavior_scores="old",
        local_dashboard_trace={"old": True},
        map_selected_batch="batch-0",
    )
    before = funcs["model_backend_snapshot"](same_model_new_url_state)
    same_model_new_url_state["llama_url"] = "http://new"
    assert funcs["invalidate_if_model_backend_changed"](same_model_new_url_state, before) is True
    assert same_model_new_url_state["activeModelFingerprint"] == "/models/gemma-4-31b.gguf|gemma-live"
    assert same_model_new_url_state["last_behavior_artifact"] is None
    assert same_model_new_url_state["local_dashboard_trace"] is None
    assert same_model_new_url_state["map_selected_batch"] is None
    namespace["st"] = types.SimpleNamespace(session_state=_SessionState(modelArtifactsInvalidated=True))
    namespace["latest_saved_behavior_artifact"] = lambda: {"run_id": "old_saved_model"}
    funcs = _load_main_functions({"latest_behavior_artifact"}, namespace)
    assert funcs["latest_behavior_artifact"]() == {}

    sixty_artifact = _layer_count_fixture_artifact(list(range(60)))
    sixty_payload = build_activation_map_payload(
        sixty_artifact,
        {**summary, "layers": 60},
        local_model_context={**local_meta, "block_count": 60},
        backend_info={"layers": 60},
    )
    assert len(sixty_payload["layers"]) == 60
    assert sixty_payload["layers"][-1]["layerId"] == "L59"
    assert sixty_payload["diagnostics"]["rendererLayerCount"] == 60
    assert sixty_payload["diagnostics"]["traceReturnedLayerCount"] == 60
    assert not sixty_payload["diagnostics"]["layerMismatchWarning"]

    partial_artifact = _layer_count_fixture_artifact(list(range(40)))
    partial_artifact["summary"]["layers"] = list(range(60))
    partial_payload = build_activation_map_payload(
        partial_artifact,
        {**summary, "layers": 60},
        local_model_context={**local_meta, "block_count": 60},
        backend_info={"layers": 60},
    )
    assert len(partial_payload["layers"]) == 40
    assert partial_payload["diagnostics"]["traceRequestedLayerCount"] == 60
    assert partial_payload["diagnostics"]["traceReturnedLayerCount"] == 40
    assert partial_payload["diagnostics"]["staleCacheSuspected"] is True
    assert "Trace returned fewer layers than requested" in partial_payload["diagnostics"]["layerMismatchWarning"]

    saved_backend_artifact = _layer_count_fixture_artifact(list(range(60)))
    saved_backend_artifact["summary"]["backend_info"] = {"layers": 60, "model": {"path": "/models/gemma.gguf"}}
    saved_backend_diag = trace_layer_diagnostics(
        saved_backend_artifact,
        saved_backend_artifact.get("summary", {}),
        backend_info={},
        local_model_context={**local_meta, "block_count": 60},
    )
    assert saved_backend_diag["backendInfoLayerCount"] == 60

    artifact_summary_precedence = _layer_count_fixture_artifact(list(range(40)))
    artifact_summary_precedence["summary"]["layers"] = list(range(40))
    current_summary = {**summary, "layers": 60}
    current_summary_diag = trace_layer_diagnostics(
        artifact_summary_precedence,
        current_summary,
        backend_info={"layers": 60},
        local_model_context={**local_meta, "block_count": 60},
    )
    assert current_summary_diag["traceRequestedLayerCount"] == 40
    assert current_summary_diag["traceRequestedMaxLayer"] == 39

    diag = trace_layer_diagnostics(
        partial_artifact,
        partial_artifact.get("summary", {}),
        backend_info={"layers": 60},
        local_model_context={**local_meta, "block_count": 60},
    )
    assert diag["traceRequestedMaxLayer"] == 59
    assert diag["traceReturnedMaxLayer"] == 39
    assert diag["rendererMaxLayer"] == 39


def _assert_single_trace_failure_preserves_reason() -> None:
    fake_st = types.SimpleNamespace(session_state=_SessionState(active_run_id="single_fail"))
    namespace = {
        "st": fake_st,
        "normalize_llama_trace": normalize_llama_trace,
        "trace_unavailable_row": trace_unavailable_row,
        "llama_trace_unavailable_reason": lambda payload: "generic fallback",
        "build_run_artifact": build_run_artifact,
        "active_model_label": lambda: "local",
    }
    funcs = _load_main_functions({"single_trace_artifact_from_llama"}, namespace)
    artifact = funcs["single_trace_artifact_from_llama"](
        {},
        {"prompt": "hello", "backend": "llama.cpp", "trace_model": "local", "run_id": "single_fail"},
        output="local reply",
        trace_error="trace socket closed",
    )
    row = artifact["prompts"][0]["trace_rows"][0]
    assert row["trace_available"] is False
    assert row["unavailable_reason"] == "trace socket closed"


def _assert_patched_baseline_visible_to_app(rows: list[dict[str, Any]]) -> None:
    fake_st = types.SimpleNamespace(
        session_state=_SessionState(
            behavior_profile="concise_helpfulness",
            behavior_run_history=[],
            last_behavior_artifact=None,
            last_behavior_scores=pd.DataFrame(),
            last_batch_result=None,
            last_fuzz_result=None,
        )
    )
    namespace = {
        "st": fake_st,
        "pd": pd,
        "get_behavior_profile": get_behavior_profile,
        "score_run_artifact": score_run_artifact,
        "latest_saved_behavior_artifact": lambda: {},
    }
    funcs = _load_main_functions({"store_behavior_artifact", "latest_behavior_artifact"}, namespace)
    baseline = build_run_artifact(
        run_id="baseline",
        mode="Single message",
        backend="llama.cpp",
        model="local",
        prompts=[{"prompt_id": 0, "label": "single", "prompt": "hi", "output": "plain answer", "trace_rows": rows}],
    )
    patched = build_run_artifact(
        run_id="patched",
        mode="Single message",
        backend="llama.cpp",
        model="local",
        prompts=[{"prompt_id": 0, "label": "single", "prompt": "hi", "output": "clear concise safe action", "trace_rows": rows}],
    )
    funcs["store_behavior_artifact"](baseline)
    funcs["store_behavior_artifact"](patched)
    history = fake_st.session_state.behavior_run_history
    assert [item["run_id"] for item in history] == ["baseline", "patched"]
    assert funcs["latest_behavior_artifact"]()["run_id"] == "patched"
    timeline = behavior_timeline_df([item["scores"].assign(run_id=item["run_id"]) for item in history])
    assert behavior_delta_bar_fig(timeline, baseline_run_id="baseline", comparison_run_id="patched") is not None


def _assert_managed_llama_defaults() -> None:
    launcher_source = Path("run_glass_skull.sh").read_text()
    managed_server_ref = 'managed/llama.cpp-glass/build/bin/llama-server'
    managed_cvector_ref = 'managed/llama.cpp-glass/build/bin/llama-cvector-generator'
    assert DEFAULT_LLAMA_SERVER == Path.cwd() / managed_server_ref
    assert DEFAULT_CVECTOR_GENERATOR == Path.cwd() / managed_cvector_ref
    assert "/home/dsmason321/llama.cpp" not in launcher_source
    assert "LLAMA_SERVER_BIN override" in launcher_source
    assert "Managed llama.cpp binary:" in launcher_source
    assert managed_server_ref in launcher_source
    assert "verify_glass_trace_activation_support" in launcher_source


def _assert_workspace_round_trip() -> None:
    state = {
        "active_run_id": "run-a",
        "chat_backend_label": "Local GGUF normal (llama.cpp)",
        "llama_model_alias": "local",
        "llama_model_path": str(DEFAULT_GGUF_MODEL_PATH),
        "active_run_mode": "Batch run",
        "batch_pasted_prompts": "edited\nbatch\nprompts",
        "behavior_profile": "concise_helpfulness",
        "map_visualization_mode": "batch_overlay",
        "map_selected_prompt": 2,
        "map_selected_batch": "batch-2",
        "map_selected_token": 0,
        "map_top_k": 8,
        "map_background_opacity": 0.24,
        "map_edge_threshold": 0.0,
        "llama_control_vector": "vec-a",
        "llama_control_strength": 1.25,
        "loaded_activation_patch_recipe": {"name": "patch-a"},
        "tab_state": {"Map": {"selected_group": "L0-N1"}, "Run": {"draft": "hello"}},
    }
    saved = save_workspace(state, name="smoke_global")
    assert saved.path.name == "smoke_global.json"
    assert saved.path.exists()
    data = json.loads(saved.path.read_text(encoding="utf-8"))
    assert data["scope"] == "global"
    assert data["state"]["active_run_id"] == "run-a"
    assert data["state"]["batch_pasted_prompts"] == "edited\nbatch\nprompts"
    assert data["state"]["model"]["alias"] == "local"
    assert data["state"]["map"]["selected_prompt"] == 2
    assert data["state"]["tab_state"]["Map"]["selected_group"] == "L0-N1"
    assert data["state"]["annotations"]["path"].endswith("data/node_annotations/annotations.json")
    assert "annotations" not in data["state"]["annotations"]

    state["active_run_id"] = "run-b"
    renamed = save_workspace_as(state, "smoke_global_as")
    assert renamed.name == "smoke_global_as"
    loaded = load_workspace("smoke_global")
    assert loaded.state["active_run_id"] == "run-a"
    assert loaded.state["batch_pasted_prompts"] == "edited\nbatch\nprompts"
    assert loaded.error is None
    assert load_workspace("missing-workspace").warning

    corrupt_path = saved.path.parent / "smoke_corrupt.json"
    corrupt_path.write_text("{not-json", encoding="utf-8")
    corrupt = load_workspace("smoke_corrupt")
    assert corrupt.error and "Invalid workspace JSON" in corrupt.error

    clear_workspace(state)
    assert state["active_run_id"] is None
    assert state["active_run_mode"] == "Single message"
    assert state["batch_pasted_prompts"] == DEFAULT_BATCH_MESSAGES
    assert state["llama_model_path"] == str(DEFAULT_GGUF_MODEL_PATH)
    assert state["last_behavior_artifact"] is None
    assert state["tab_state"] == {}
    assert saved.path.exists()

    tab_state = {"global_marker": "keep", "tab_state": {"Map": {"selected_group": "old"}, "Run": {"draft": "keep"}}}
    tab_saved = save_tab_state("Map", {"selected_group": "L1-N2", "zoom": 2}, name="smoke_map")
    assert "tabs/Map" in tab_saved.path.as_posix()
    tab_data = json.loads(tab_saved.path.read_text(encoding="utf-8"))
    assert tab_data["scope"] == "tab"
    assert tab_data["tab_name"] == "Map"
    assert tab_data["state"] == {"selected_group": "L1-N2", "zoom": 2}
    run_tab_saved = save_tab_state_as("Run", {"draft": "run-only", "batch_pasted_prompts": ""}, "smoke_run")
    loaded_tab = load_tab_state("Map", "smoke_map")
    assert loaded_tab.state == {"selected_group": "L1-N2", "zoom": 2}
    loaded_run_tab = load_tab_state("Run", "smoke_run")
    assert loaded_run_tab.state["batch_pasted_prompts"] == ""
    tab_state["tab_state"]["Map"] = loaded_tab.state
    assert tab_state["tab_state"]["Run"] == {"draft": "keep"}
    clear_tab_state(tab_state, "Map")
    assert tab_state["global_marker"] == "keep"
    assert tab_state["tab_state"]["Map"] == {}
    assert tab_state["tab_state"]["Run"] == {"draft": "keep"}

    app_state = {
        "llama_control_strength": 0.5,
        "llama_control_layer_start": 2,
        "llama_control_layer_end": 6,
        "llama_control_port": 8099,
        "llama_control_extra_args": "--jinja",
        "tab_state": {"Steer": {"workspace_name": "old"}},
    }
    steer_state = collect_tab_state("Steer", app_state)
    assert steer_state["llama_control_strength"] == 0.5
    apply_tab_state(app_state, "Steer", {"llama_control_strength": 1.75, "llama_control_layer_end": 12})
    assert app_state["llama_control_strength"] == 1.75
    assert app_state["llama_control_layer_end"] == 12
    clear_tab_state(app_state, "Steer")
    assert app_state["llama_control_strength"] == 1.25
    assert app_state["llama_control_layer_start"] == 1
    assert app_state["tab_state"]["Steer"] == {}
    settings_state = {
        "llama_model_alias": "changed",
        "llama_model_path": "/tmp/changed.gguf",
        "llama_url": "http://127.0.0.1:9999",
        "llama_glass_url": "http://127.0.0.1:9998",
        "llama_control_strength": 9.0,
        "tab_state": {"Settings": {"workspace_name": "old"}},
    }
    clear_tab_state(settings_state, "Settings")
    assert settings_state["llama_model_alias"] == "local"
    assert settings_state["llama_model_path"] == str(DEFAULT_GGUF_MODEL_PATH)
    assert settings_state["llama_control_strength"] == 9.0
    assert settings_state["tab_state"]["Settings"] == {}
    for path in [saved.path, renamed.path, corrupt_path, tab_saved.path, run_tab_saved.path]:
        path.unlink(missing_ok=True)


def _assert_steer_tooltips() -> None:
    metadata = [
        {
            "control_id": "smoke_control",
            "title": "Smoke control",
            "control_type": "slider",
            "typical_values": "1 to 3",
            "use": "Use this in smoke tests.",
            "downside": "Bad values make the smoke test fail.",
        }
    ]
    path = TOOLTIP_DIR / "smoke_tooltips.json"
    if path.exists():
        path.unlink()
    tooltips = ensure_tooltips("smoke", metadata)
    assert path.exists()
    assert "smoke_control" in tooltips
    first_text = path.read_text(encoding="utf-8")
    tooltips["smoke_control"]["explanation"] = "Approved local edit."
    path.write_text(json.dumps(tooltips, indent=2), encoding="utf-8")
    reused = ensure_tooltips("smoke", metadata)
    assert reused["smoke_control"]["explanation"] == "Approved local edit."
    assert path.read_text(encoding="utf-8") != first_text
    rendered = tooltip_text(reused["smoke_control"])
    assert "Smoke control" in rendered
    assert "Approved local edit." in rendered
    path.unlink(missing_ok=True)


def _assert_node_annotation_round_trip() -> None:
    path = Path("data/node_annotations/smoke_annotations.json")
    path.unlink(missing_ok=True)
    model_meta = {
        "modelFingerprint": "model-fp-a",
        "modelName": "local-a",
        "backend": "llama.cpp",
    }
    other_model_meta = {
        "modelFingerprint": "model-fp-b",
        "modelName": "local-b",
        "backend": "llama.cpp",
    }

    empty = load_annotations(path)
    assert empty["version"] == 1
    assert empty["annotations"] == []

    created = upsert_annotation(
        model_meta,
        layer=5,
        cluster_id="L5-C12",
        node_id="L5-N1294",
        node_range=[1024, 1199],
        tags=[" Comedy ", "sarcasm", "comedy"],
        note="Comedy prompts consistently light this cluster.",
        created_from={"run_id": "run-a", "prompt_id": "p1", "prompt_excerpt": "tell me a joke"},
        path=path,
    )
    assert created["id"]
    assert created["tags"] == ["comedy", "sarcasm"]
    assert created["note"] == "Comedy prompts consistently light this cluster."
    assert path.exists()

    saved_payload = load_annotations(path)
    assert len(saved_payload["annotations"]) == 1
    updated = update_note(created["id"], "Sarcastic jokes are strongest.", path=path)
    assert updated["note"] == "Sarcastic jokes are strongest."
    tagged = add_tag(created["id"], " Sarcasm ", path=path)
    assert tagged["tags"] == ["comedy", "sarcasm"]
    tagged = add_tag(created["id"], "refusal", path=path)
    assert tagged["tags"] == ["comedy", "sarcasm", "refusal"]
    after_delete_tag = delete_tag(created["id"], "sarcasm", path=path)
    assert after_delete_tag["tags"] == ["comedy", "refusal"]
    after_delete_note = delete_note(created["id"], path=path)
    assert after_delete_note["note"] == ""

    exact = get_annotations_for_node(
        model_meta,
        layer=5,
        cluster_id="L5-C12",
        node_id="L5-N1294",
        node_range=[1024, 1199],
        annotations=load_annotations(path),
    )
    assert exact["match_type"] == "exact"
    assert exact["annotations"][0]["tags"] == ["comedy", "refusal"]

    approximate = get_annotations_for_node(
        model_meta,
        layer=5,
        cluster_id="L5-C99",
        node_id="L5-N1400",
        node_range=[1024, 1199],
        annotations=load_annotations(path),
    )
    assert approximate["match_type"] == "approximate"
    assert approximate["annotations"][0]["id"] == created["id"]

    wrong_model = get_annotations_for_node(
        other_model_meta,
        layer=5,
        cluster_id="L5-C12",
        node_id="L5-N1294",
        node_range=[1024, 1199],
        annotations=load_annotations(path),
    )
    assert wrong_model["match_type"] == "none"
    assert wrong_model["annotations"] == []

    deleted = delete_annotation(created["id"], path=path)
    assert deleted is True
    assert load_annotations(path)["annotations"] == []
    path.unlink(missing_ok=True)


def _path_fixture_artifact(prompt_count: int, *, vector: bool = False, scalar_only: bool = False) -> dict[str, Any]:
    prompts = []
    for prompt_id in range(prompt_count):
        trace_rows = []
        for layer in range(3):
            base_dim = (prompt_id * 7 + layer * 2) % 24
            row: dict[str, Any] = {
                "layer": layer,
                "stream": "resid_pre",
                "token_index": 0,
                "token": f"tok-{prompt_id}",
                "activation_norm": 1.0 + prompt_id * 0.03 + layer * 0.1,
            }
            if vector:
                width = 24
                values = [0.0] * width
                values[base_dim] = 1.0 + layer * 0.1
                row["vector"] = values
            elif not scalar_only:
                row["top_dims"] = [{"dimension": base_dim, "activation": 1.0 + layer * 0.1}]
            trace_rows.append(row)
        prompts.append({
            "prompt_id": prompt_id,
            "label": f"p{prompt_id}",
            "prompt": f"prompt {prompt_id}",
            "output": f"output {prompt_id}",
            "trace_rows": normalize_llama_trace(
                {
                    "activations": {"supported": True, "vectors": vector, "n_embd": 24},
                    "layer_inputs": trace_rows,
                },
                prompt_id=prompt_id,
                label=f"p{prompt_id}",
                metadata={"run_id": "path_fixture"},
            ),
        })
    return build_run_artifact(
        run_id="path_fixture",
        mode="Batch run" if prompt_count > 1 else "Single message",
        backend="llama.cpp",
        model="local",
        prompts=prompts,
    )


def _layer_count_fixture_artifact(layer_ids: list[int], *, prompt_id: int = 0, vector: bool = False) -> dict[str, Any]:
    trace_rows = []
    for index, layer in enumerate(layer_ids):
        row: dict[str, Any] = {
            "layer": layer,
            "stream": "resid_pre",
            "token_index": 0,
            "token": "tok",
            "activation_norm": 1.0 + index * 0.05,
        }
        if vector:
            row["vector"] = [1.0, 0.0, 0.0, 0.0]
        else:
            row["top_dims"] = [{"dimension": layer % 24, "activation": 1.0}]
        trace_rows.append(row)
    return build_run_artifact(
        run_id=f"layers_{len(layer_ids)}",
        mode="local",
        backend="llama.cpp",
        model="local",
        prompts=[{
            "prompt_id": prompt_id,
            "label": "layers",
            "prompt": "trace layers",
            "output": "ok",
            "trace_rows": normalize_llama_trace(
                {"layer_inputs": trace_rows, "activations": {"vectors": vector, "n_embd": 4 if vector else 24}},
                prompt_id=prompt_id,
                label="layers",
                metadata={"run_id": f"layers_{len(layer_ids)}"},
            ),
        }],
    )


def _overlap_path_fixture_artifact() -> dict[str, Any]:
    prompts = []
    for prompt_id, prompt_text in [(0, "shared alpha prompt"), (1, "shared beta prompt")]:
        trace_rows = []
        for layer in range(3):
            shared_dim = 5 if layer == 0 else 8 + prompt_id + layer
            trace_rows.append({
                "layer": layer,
                "stream": "resid_pre",
                "token_index": 0,
                "token": f"shared-{prompt_id}",
                "activation_norm": 1.0 + prompt_id * 0.2 + layer * 0.1,
                "top_dims": [{"dimension": shared_dim, "activation": 1.0 + prompt_id * 0.2 + layer * 0.1}],
            })
        prompts.append({
            "prompt_id": prompt_id,
            "label": f"shared-{prompt_id}",
            "prompt": prompt_text,
            "output": f"shared output {prompt_id}",
            "trace_rows": normalize_llama_trace(
                {
                    "activations": {"supported": True, "vectors": False, "n_embd": 24},
                    "layer_inputs": trace_rows,
                },
                prompt_id=prompt_id,
                label=f"shared-{prompt_id}",
                metadata={"run_id": "overlap_fixture"},
            ),
        })
    return build_run_artifact(
        run_id="overlap_fixture",
        mode="Batch run",
        backend="llama.cpp",
        model="local",
        prompts=prompts,
    )


def _aggregate_only_artifact() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": "aggregate_only",
        "mode": "Batch run",
        "backend": "llama.cpp",
        "model": "local",
        "created_at": "2026-06-28T00:00:00+00:00",
        "summary": {"prompt_count": 15, "trace_row_count": 3, "layers": 3, "d_model": 24},
        "prompts": [{
            "prompt_id": None,
            "label": "aggregate",
            "prompt": "",
            "output": "",
            "trace_rows": [
                {
                    "run_id": "aggregate_only",
                    "prompt_id": None,
                    "batch_id": None,
                    "layer": layer,
                    "stream": "resid_pre",
                    "token_index": None,
                    "token": "",
                    "activation_norm": 1.0 + layer,
                    "trace_available": True,
                    "trace_source": "llama.cpp",
                    "aggregation": "all_prompts_layer_mean",
                    "top_dims": [{"dimension": layer * 3, "activation": 1.0}],
                }
                for layer in range(3)
            ],
        }],
    }


def _tokenless_prompt_artifact() -> dict[str, Any]:
    return build_run_artifact(
        run_id="tokenless_prompt",
        mode="Batch run",
        backend="llama.cpp",
        model="local",
        prompts=[
            {
                "prompt_id": 0,
                "batch_id": "batch-0",
                "label": "p0",
                "prompt": "tokenless prompt",
                "output": "ok",
                "trace_rows": [
                    {
                        "run_id": "tokenless_prompt",
                        "batch_id": "batch-0",
                        "prompt_id": 0,
                        "layer": 0,
                        "stream": "resid_pre",
                        "token_index": None,
                        "token_id": None,
                        "token": "",
                        "activation_norm": 1.0,
                        "trace_available": True,
                        "trace_source": "llama.cpp",
                        "top_dims": [{"dimension": 4, "activation": 1.0}],
                    }
                ],
            }
        ],
    )


def main() -> None:
    ensure_dirs()
    assert LOCAL_TABS == ["Run", "Map", "Steer", "Timeline", "Model", "Settings"]
    _assert_managed_llama_defaults()
    assert DEFAULT_GGUF_MODEL_PATH == Path("/home/dsmason321/models/Best/Qwen3.6-35B-heretic-MTP-Q4_K_S.gguf")
    main_source = Path("main.py").read_text()
    default_batch_lines = DEFAULT_BATCH_MESSAGES.splitlines()
    assert len(default_batch_lines) == 15
    assert default_batch_lines == [
        "What is the capital of Mongolia?",
        "Name one mammal that can glide without powered flight.",
        "Why do leaves change color in autumn?",
        "Convert 98.6°F to Celsius.",
        "What is the primary purpose of DNS?",
        "Who painted the ceiling of the Sistine Chapel?",
        "What does the acronym SQL stand for?",
        "How many moons does Mars have?",
        "What is the largest organ in the human body?",
        'Define the term "opportunity cost" in one sentence.',
        "Which programming language introduced the `async` and `await` keywords first?",
        "What causes a rainbow to appear?",
        "Name one advantage of using a hash table.",
        "What year did the first human land on the Moon?",
        "Explain the difference between RAM and SSD in one sentence.",
    ]
    assert "98.6°F" in DEFAULT_BATCH_MESSAGES
    assert "`async` and `await`" in DEFAULT_BATCH_MESSAGES
    assert '"opportunity cost"' in DEFAULT_BATCH_MESSAGES
    assert '"batch_pasted_prompts": DEFAULT_BATCH_MESSAGES' in main_source
    assert 'key="batch_pasted_prompts_widget"' in main_source
    assert "sync_batch_prompt_widget_state()" in main_source
    assert "update_batch_prompts_from_widget" in main_source
    assert '"batch_pasted_prompts"' in Path("glass_skull/workspaces.py").read_text()
    assert "with tabs[\"Settings\"]" in main_source
    run_app_source = main_source.split("def run_app() -> None:", 1)[1]
    settings_section = run_app_source.split('with tabs["Settings"]:', 1)[1]
    before_settings = run_app_source.split('with tabs["Settings"]:', 1)[0]
    assert "render_global_workspace_controls()" in settings_section
    assert "render_global_workspace_controls()" not in before_settings
    assert "apply_tab_state(st.session_state, tab_name, result.state)" in main_source
    assert "collect_tab_state(tab_name, st.session_state, state)" in main_source
    assert "repeat_prompt=prompt" not in main_source
    assert "repeat_prompt=prompt or repeat" not in main_source
    assert "ensure_tooltips(\"steer\", STEER_CONTROL_METADATA)" in main_source
    assert "help=steer_help(" in main_source
    assert "seed_missing_defaults(st.session_state, defaults)" in main_source
    assert "persist_single_run_artifact(artifact)" in main_source
    assert "latest_saved_behavior_artifact()" in main_source
    assert 'DEFAULT_CHAT_INPUT = "hi"' in main_source
    assert 'st.text_input("Send a local prompt", value=DEFAULT_CHAT_INPUT' in main_source
    assert 'st.form("chat_prompt_form", clear_on_submit=True)' in main_source
    assert '{"Single message", "Batch run"}' in main_source
    assert "def resolve_trace_layers(" in main_source
    assert "trace_plan = resolve_trace_plan(" in main_source
    assert "layers=trace_layers" in main_source
    assert "active_recipe = validate_activation_patch_recipe(loaded_recipe)" in main_source
    assert "build_activation_patch_backend_payload(active_recipe" in main_source
    assert "render_activation_map(payload, key=f\"activation_map_{artifact.get('run_id', 'latest')}\", height=1840)" in main_source
    render_source = inspect.getsource(activation_map_view.render_activation_map)
    assert "st.iframe" in render_source
    assert "components.html" not in render_source
    assert inspect.signature(activation_map_view.activation_map_html).parameters["height"].default == 1920
    assert inspect.signature(activation_map_view.render_activation_map).parameters["height"].default == 1920
    assert normalize_base_url("http://127.0.0.1:8080/v1") == "http://127.0.0.1:8080"
    assert safe_slug("../bad name") == "bad_name"

    prompts = load_txt("one\n# skip\ntwo")
    assert [p.prompt for p in prompts] == ["one", "two"]
    labeled = load_jsonl('{"prompt":"A","label":"x"}\n{"prompt":"B"}')
    assert [p.label for p in labeled] == ["x", "unlabeled"]
    batch = batch_items_from_inputs(pasted_payload="pasted", repeat_prompt="again", repeat_count=2, uploaded_items=labeled)
    assert len(batch) == 5
    pasted_only_batch = batch_items_from_inputs(pasted_payload="pasted", repeat_prompt="", repeat_count=1, uploaded_items=[])
    assert [item.prompt for item in pasted_only_batch] == ["pasted"]
    user_batch_state = {"batch_pasted_prompts": "custom\ncontent"}
    original_user_batch = dict(user_batch_state)
    seed_missing_defaults(user_batch_state, {"batch_pasted_prompts": DEFAULT_BATCH_MESSAGES})
    assert user_batch_state == original_user_batch
    stale_empty_batch_state = {"batch_pasted_prompts": ""}
    seed_missing_defaults(stale_empty_batch_state, {"batch_pasted_prompts": DEFAULT_BATCH_MESSAGES})
    seed_batch_prompt_default(stale_empty_batch_state)
    assert stale_empty_batch_state["batch_pasted_prompts"] == DEFAULT_BATCH_MESSAGES
    stale_empty_with_old_marker = {"batch_pasted_prompts": "", "batch_pasted_prompts_user_set": True}
    seed_batch_prompt_default(stale_empty_with_old_marker)
    assert stale_empty_with_old_marker["batch_pasted_prompts"] == DEFAULT_BATCH_MESSAGES
    user_cleared_batch_state = {"batch_pasted_prompts": "", "batch_pasted_prompts_source": "user"}
    seed_missing_defaults(user_cleared_batch_state, {"batch_pasted_prompts": DEFAULT_BATCH_MESSAGES})
    seed_batch_prompt_default(user_cleared_batch_state)
    assert user_cleared_batch_state["batch_pasted_prompts"] == ""
    saved_batch_state = {"batch_pasted_prompts": "loaded workspace prompt"}
    target_batch_state = {"batch_pasted_prompts": DEFAULT_BATCH_MESSAGES}
    apply_workspace_state(target_batch_state, saved_batch_state)
    assert target_batch_state["batch_pasted_prompts"] == "loaded workspace prompt"
    loaded_empty_batch_state = {"batch_pasted_prompts": DEFAULT_BATCH_MESSAGES}
    apply_workspace_state(loaded_empty_batch_state, {"batch_pasted_prompts": ""})
    seed_batch_prompt_default(loaded_empty_batch_state)
    assert loaded_empty_batch_state["batch_pasted_prompts"] == ""
    assert loaded_empty_batch_state["batch_pasted_prompts_user_set"] is True
    assert loaded_empty_batch_state["batch_pasted_prompts_source"] == "loaded"
    loaded_empty_tab_state = {"batch_pasted_prompts": DEFAULT_BATCH_MESSAGES, "tab_state": {}}
    apply_tab_state(loaded_empty_tab_state, "Run", {"batch_pasted_prompts": ""})
    seed_batch_prompt_default(loaded_empty_tab_state)
    assert loaded_empty_tab_state["batch_pasted_prompts"] == ""
    assert loaded_empty_tab_state["batch_pasted_prompts_user_set"] is True
    assert loaded_empty_tab_state["batch_pasted_prompts_source"] == "loaded"
    widget_state = _SessionState(
        batch_pasted_prompts="",
        batch_pasted_prompts_user_set=True,
        batch_pasted_prompts_widget="",
    )
    widget_namespace = {
        "DEFAULT_BATCH_MESSAGES": DEFAULT_BATCH_MESSAGES,
        "seed_batch_prompt_default": seed_batch_prompt_default,
        "st": types.SimpleNamespace(session_state=widget_state),
    }
    widget_funcs = _load_main_functions(
        {"sync_batch_prompt_widget_state", "update_batch_prompts_from_widget"},
        widget_namespace,
    )
    widget_funcs["sync_batch_prompt_widget_state"]()
    assert widget_state["batch_pasted_prompts"] == DEFAULT_BATCH_MESSAGES
    assert widget_state["batch_pasted_prompts_widget"] == DEFAULT_BATCH_MESSAGES
    widget_state["batch_pasted_prompts_widget"] = ""
    widget_funcs["update_batch_prompts_from_widget"]()
    assert widget_state["batch_pasted_prompts"] == ""
    assert widget_state["batch_pasted_prompts_source"] == "user"
    assert dashboard_context("Batch run", "abc") == {"mode": "Batch run", "run_id": "abc"}
    assert new_run_id("unit").startswith("unit_")
    _assert_workspace_round_trip()
    _assert_steer_tooltips()
    _assert_node_annotation_round_trip()

    dummy_backend_patch = {"recipe": {"name": "smoke patch", "target_run_id": "run1", "patches": []}, "source_vectors": []}
    captured: list[dict[str, Any]] = []
    import glass_skull.llama_client as llama_client
    original_request_json = llama_client._request_json
    try:
        llama_client._request_json = _fake_llama_request(captured)
        output = chat_completion("http://local/v1", "hello", model_alias="local")
        assert output == "local reply"
        patched_output = chat_completion(
            "http://local/v1",
            "hello",
            model_alias="local",
            activation_patch=dummy_backend_patch,
            activation_patch_supported=True,
        )
        assert patched_output == "local reply"
        trace = trace_glass_prompt(
            "http://local",
            "hello",
            model_alias="local",
            layers=[0, 1],
            streams=["resid_pre"],
            max_new_tokens=8,
            with_pieces=True,
        )
    finally:
        llama_client._request_json = original_request_json

    trace_request = [r for r in captured if r["url"].endswith("/glass-skull/trace")][0]
    patched_request = [r for r in captured if r["url"].endswith("/v1/chat/completions")][-1]
    assert patched_request["payload"]["glass_skull"]["activation_patch"]["recipe"]["name"] == "smoke patch"
    assert patched_request["payload"]["metadata"]["glass_skull"]["activation_patch"]["recipe"]["target_run_id"] == "run1"
    assert trace_request["payload"]["capture"]["prompt_tokens"] is True
    assert trace_request["payload"]["capture"]["layer_inputs"] is True
    assert trace_request["payload"]["max_tokens"] == 8
    assert trace_request["payload"]["include_vectors"] is True
    assert per_request_steering_supported({"capabilities": {"steering": {"per_request": {"supported": True}}}}) is True
    assert build_steering_metadata("vec.gguf", 1.25, 1, 4)["layer_end"] == 4

    rows = normalize_llama_trace(trace, prompt_id=0, label="single", metadata={"run_id": "run1"})
    assert rows[0]["token"] == "hi"
    assert rows[0]["batch_id"] is None
    assert rows[0]["token_id"] == rows[0]["token_index"]
    assert rows[1]["activation_norm"] == 2.5
    token_id_rows = normalize_llama_trace(
        {"layer_inputs": [{"layer": 0, "stream": "resid_pre", "token_id": 3, "activation_norm": 1.0, "top_dims": [{"dimension": 2, "activation": 1.0}]}]},
        prompt_id=7,
        label="token-id",
        metadata={"run_id": "token_id_run"},
    )
    assert token_id_rows[0]["token_index"] == 3
    assert token_id_rows[0]["token_id"] == 3
    vector_rows = normalize_llama_trace(
        {
            "layer_inputs": [{
                "layer": 0,
                "stream": "resid_pre",
                "token_index": 0,
                "activation_norm": 1.0,
                "vector": [0.1, 0.2, 0.3, 0.4],
            }]
        },
        prompt_id=0,
        label="source",
        metadata={"run_id": "source1"},
    )
    assert vector_rows[0]["vector"] == [0.1, 0.2, 0.3, 0.4]
    unavailable = trace_unavailable_row("run1", 2, "x", "llama.cpp", "not exposed")
    stale_unavailable = trace_unavailable_row("run1", 0, "single", "llama.cpp", "old unavailable trace")
    artifact = build_run_artifact(
        run_id="run1",
        mode="local",
        backend="llama.cpp",
        model="local",
        prompts=[
            {"prompt_id": 0, "label": "single", "prompt": "hello", "output": "clear concise action", "trace_rows": [stale_unavailable, *rows]},
            {"prompt_id": 2, "label": "x", "prompt": "missing", "output": "", "trace_rows": [unavailable]},
        ],
    )
    path_df = activation_path_df(artifact)
    assert len(path_df) == 3
    assert "batch_id" in TRACE_COLUMNS
    assert "token_id" in TRACE_COLUMNS
    assert "batch_id" in path_df.columns
    assert "token_id" in path_df.columns
    assert "top_dims" in path_df.columns
    assert path_df[path_df["trace_available"] != False].iloc[0]["top_dims"]
    assert artifact["prompts"][0]["trace_rows"] == rows
    assert artifact["prompts"][0]["batch_id"] == "batch-0"
    assert artifact["summary"]["trace_row_count"] == 2
    assert artifact["summary"]["trace_unavailable_count"] == 1
    assert batch_heatmap_df(artifact, group_by="label").iloc[0]["activation_norm"] > 0
    assert not label_heatmap_df(artifact).empty
    assert dimension_frequency_df(artifact).iloc[0]["dimension"] in {3, 7}

    local_meta = {
        "source": "Local GGUF",
        "display_name": "local",
        "architecture": "llama",
        "block_count": 2,
        "embedding_length": 16,
        "head_count": 2,
        "head_count_kv": 2,
        "d_mlp": 64,
        "context_length": 4096,
        "tensor_count": 1,
        "metadata": {},
        "errors": [],
    }
    summary = {"model_name": "local", "layers": 2, "d_model": 16, "heads": 2, "vocab_size": 0, "parameters": 0}
    meta = build_model_meta(summary, local_meta, "llama.cpp")
    assert meta["source"] == "Local GGUF"
    payload = build_activation_map_payload(artifact, summary, local_model_context=local_meta)
    assert payload["modelMeta"]["metadataSource"] == "gguf"
    assert payload["activationPaths"]
    assert payload["visualizationMode"] == "single_prompt"
    assert payload["dataMode"] == "top_dims_approx"
    assert payload["diagnostics"]["vectorsPresent"] is False
    assert payload["diagnostics"]["includeVectorsResponse"] is False
    assert payload["diagnostics"]["warnings"]
    assert payload["diagnostics"]["renderLayerSource"] == "trace_rows"
    assert payload["diagnostics"]["renderLayerCount"] == 2
    assert payload["diagnostics"]["renderFirstLayer"] == 0
    assert payload["diagnostics"]["renderLastLayer"] == 1
    assert payload["diagnostics"]["traceLayerCount"] == 2
    assert payload["diagnostics"]["traceMinLayer"] == 0
    assert payload["diagnostics"]["traceMaxLayer"] == 1
    assert payload["diagnostics"]["backendInfoLayerCount"] is None
    assert payload["diagnostics"]["modelMetaLayerCount"] == 2
    layer0_groups = [group for group in payload["nodeGroups"] if group["layerId"] == "L0" and group["activationValue"] > 0]
    assert len(layer0_groups) > 1
    assert all(edge["method"] != "cosine_similarity" for edge in payload["activationEdges"])
    assert all(edge["tokenIndex"] == payload["activationPaths"][0]["tokenIndex"] for edge in payload["activationEdges"])
    html = activation_map_html(payload)
    assert 'class="gs-map-shell" style="height:1920px"' in html
    assert "h = Math.max(480, rect.height * 0.32)" in html
    assert "w = rect.width - 124, h = 300" in html
    assert "canvas" in html and "selectedDiagnostics" in html and "Activation path unavailable" not in html
    assert "function clamp(" in html
    assert "function withPanelClip(" in html
    assert "clampedPoint(" in html
    assert "function nearestValidLayerId" in html
    assert "function layerOrdinal" in html
    assert "layerX(`L${cell.layer}`" in html
    assert "groupIndexOffset" in html
    assert "drawSelectedTarget(x + w / 2, y + h / 2 + 6, selected.groupId, drilldownBounds);" in html
    assert "drawSelectedTarget(x + w / 2, y + h / 2 + 6, selected.groupId);" not in html
    assert "Real activation vectors were not returned" in html
    assert "identities.has" in html
    assert "rendererOptions.visualizationMode === 'aggregate_heatmap' ? null : pathByBatchId" in html
    assert "visualization-mode" in html
    assert "selected-prompt" in html
    assert "selected-token" in html
    assert "top-k" in html
    assert "background-opacity" in html
    assert "edge-threshold" in html
    assert "show-aggregate" in html
    assert "show-secondary" in html
    assert "developer-diagnostics" in html
    assert "function groupActivation" in html
    assert "function box2Radius" in html
    assert "function box2Alpha" in html
    assert "selectedGroupOutline" in html
    assert "box2NodeRenderDiagnostics" in html
    assert "const selectedEdge = edgeById(selected.edgeId) || null;" in html

    vector_artifact = build_run_artifact(
        run_id="vector1",
        mode="local",
        backend="llama.cpp",
        model="local",
        prompts=[{
            "prompt_id": 0,
            "label": "single",
            "prompt": "hello",
            "output": "ok",
            "trace_rows": normalize_llama_trace(
                {
                    "activations": {"supported": True, "vectors": True, "n_embd": 4},
                    "layer_inputs": [
                        {"layer": 0, "stream": "resid_pre", "token_index": 0, "activation_norm": 1.0, "vector": [1.0, 0.0, 0.0, 0.0]},
                        {"layer": 0, "stream": "resid_pre", "token_index": 1, "activation_norm": 0.8, "vector": [0.0, 1.0, 0.0, 0.0]},
                        {"layer": 1, "stream": "resid_pre", "token_index": 0, "activation_norm": 1.2, "vector": [0.9, 0.1, 0.0, 0.0]},
                        {"layer": 1, "stream": "resid_pre", "token_index": 1, "activation_norm": 0.9, "vector": [0.1, 0.9, 0.0, 0.0]},
                    ],
                },
                prompt_id=0,
                label="single",
                metadata={"run_id": "vector1"},
            ),
        }],
    )
    vector_payload = build_activation_map_payload(vector_artifact, summary, local_model_context=local_meta)
    assert vector_payload["visualizationMode"] == "single_prompt"
    assert vector_payload["dataMode"] == "real_vectors"
    assert vector_payload["diagnostics"]["vectorsPresent"] is True
    assert len([group for group in vector_payload["nodeGroups"] if group["layerId"] == "L0"]) >= 2
    assert vector_payload["activationEdges"]
    assert {edge["method"] for edge in vector_payload["activationEdges"]} == {"cosine_similarity"}

    qwen_layers = list(range(40))
    qwen_artifact = _layer_count_fixture_artifact(qwen_layers)
    qwen_meta = {**local_meta, "block_count": 40}
    qwen_summary = {**summary, "layers": 40}
    qwen_bounds = resolve_render_layers(qwen_artifact, {"layers": 40}, build_model_meta(qwen_summary, qwen_meta, "llama.cpp"))
    assert qwen_bounds.layer_ids == qwen_layers
    qwen_payload = build_activation_map_payload(qwen_artifact, qwen_summary, local_model_context=qwen_meta, backend_info={"layers": 40})
    assert len(qwen_payload["layers"]) == 40
    assert qwen_payload["layers"][0]["layerId"] == "L0"
    assert qwen_payload["layers"][-1]["layerId"] == "L39"
    assert qwen_payload["diagnostics"]["renderLayerSource"] == "trace_rows"
    assert qwen_payload["diagnostics"]["renderLayerCount"] == 40
    assert qwen_payload["diagnostics"]["traceMaxLayer"] == 39

    gemma_layers = list(range(31))
    gemma_artifact = _layer_count_fixture_artifact(gemma_layers)
    gemma_meta = {**local_meta, "display_name": "gemma-4-31b", "architecture": "gemma", "block_count": 63}
    gemma_summary = {**summary, "model_name": "gemma-4-31b", "layers": 63}
    gemma_payload = build_activation_map_payload(gemma_artifact, gemma_summary, local_model_context=gemma_meta, backend_info={"layers": 63})
    assert len(gemma_payload["layers"]) == 31
    assert [layer["index"] for layer in gemma_payload["layers"]] == gemma_layers
    assert gemma_payload["layers"][-1]["layerId"] == "L30"
    assert all(group["layer"] <= 30 for group in gemma_payload["nodeGroups"])
    assert all(point["layerIndex"] <= 30 and point["x"] <= 1.0 for path in gemma_payload["activationPaths"] for point in path["points"])
    assert gemma_payload["activationPaths"][0]["points"][-1]["layerId"] == "L30"
    assert gemma_payload["diagnostics"]["renderLayerCount"] == 31
    assert gemma_payload["diagnostics"]["backendInfoLayerCount"] == 63
    assert gemma_payload["diagnostics"]["modelMetaLayerCount"] == 63
    assert gemma_payload["diagnostics"]["layerMismatchWarning"]
    assert "trace layers 31" in gemma_payload["diagnostics"]["layerMismatchWarning"]

    incomplete_artifact = _layer_count_fixture_artifact([0, 1, 2])
    incomplete_artifact["prompts"][0]["trace_rows"] = [
        row for row in incomplete_artifact["prompts"][0]["trace_rows"] if row.get("layer") != 2
    ] + [{
        "run_id": incomplete_artifact["run_id"],
        "prompt_id": 99,
        "batch_id": "batch-other",
        "label": "other",
        "prompt_text": "other",
        "layer": 2,
        "stream": "resid_pre",
        "token_index": 0,
        "token": "tok",
        "activation_norm": 1.0,
        "trace_available": True,
        "trace_source": "llama.cpp",
        "top_dims": [{"dimension": 3, "activation": 1.0}],
    }]
    incomplete_payload = build_activation_map_payload(incomplete_artifact, summary, local_model_context={**local_meta, "block_count": 3}, selected_prompt=0)
    assert incomplete_payload["diagnostics"]["renderLastLayer"] == 2
    assert incomplete_payload["diagnostics"]["pathCompletenessWarning"]

    sparse_artifact = _layer_count_fixture_artifact([0, 2, 4])
    sparse_payload = build_activation_map_payload(sparse_artifact, {**summary, "layers": 9}, local_model_context={**local_meta, "block_count": 9})
    assert [layer["index"] for layer in sparse_payload["layers"]] == [0, 2, 4]
    assert sparse_payload["diagnostics"]["renderLayerCount"] == 3
    assert sparse_payload["diagnostics"]["layerMismatchWarning"]

    selected_clamp_payload = build_activation_map_payload(gemma_artifact, gemma_summary, local_model_context=gemma_meta, selected_layer=62)
    assert selected_clamp_payload["diagnostics"]["selectedLayer"]["layerId"] == "L30"

    batch_path_artifact = build_run_artifact(
        run_id="batch_paths",
        mode="Batch run",
        backend="llama.cpp",
        model="local",
        prompts=[
            {
                "prompt_id": 0,
                "label": "a",
                "prompt": "alpha",
                "output": "ok",
                "trace_rows": normalize_llama_trace(
                    {"layer_inputs": [
                        {"layer": 0, "stream": "resid_pre", "token_index": 0, "activation_norm": 1.0, "top_dims": [{"dimension": 1, "activation": 1.0}]},
                        {"layer": 1, "stream": "resid_pre", "token_index": 0, "activation_norm": 1.0, "top_dims": [{"dimension": 2, "activation": 1.0}]},
                    ]},
                    prompt_id=0,
                    label="a",
                    metadata={"run_id": "batch_paths"},
                ),
            },
            {
                "prompt_id": 1,
                "label": "b",
                "prompt": "beta",
                "output": "ok",
                "trace_rows": normalize_llama_trace(
                    {"layer_inputs": [
                        {"layer": 0, "stream": "resid_pre", "token_index": 0, "activation_norm": 1.0, "top_dims": [{"dimension": 14, "activation": 1.0}]},
                        {"layer": 1, "stream": "resid_pre", "token_index": 0, "activation_norm": 1.0, "top_dims": [{"dimension": 3, "activation": 1.0}]},
                    ]},
                    prompt_id=1,
                    label="b",
                    metadata={"run_id": "batch_paths"},
                ),
            },
        ],
    )
    batch_path_payload = build_activation_map_payload(batch_path_artifact, summary, local_model_context=local_meta)
    batch_paths = batch_path_payload["activationPaths"]
    assert batch_path_payload["visualizationMode"] == "batch_overlay"
    assert batch_path_payload["diagnostics"]["rendererMode"] == "batch_overlay"
    assert batch_path_payload["diagnostics"]["promptColorMode"] == "prompt_palette"
    assert batch_path_payload["diagnostics"]["promptColorCount"] == 2
    assert batch_path_payload["diagnostics"]["colorsReused"] is False
    assert len(batch_paths) == 2
    assert {path["pathMethod"] for path in batch_paths} == {"top_activation_per_layer"}
    assert [point["groupId"] for point in batch_paths[0]["points"]] != [point["groupId"] for point in batch_paths[1]["points"]]
    assert all(point["promptId"] == path["promptId"] for path in batch_paths for point in path["points"])
    assert all(point["tokenIndex"] == 0 for path in batch_paths for point in path["points"])
    assert all(edge["promptId"] == path["promptId"] for path in batch_paths for edge in path["edges"])
    assert len({path["promptColor"] for path in batch_paths}) == 2
    assert all(point["promptColor"] == path["promptColor"] for path in batch_paths for point in path["points"])
    assert all(edge["promptColor"] == path["promptColor"] for path in batch_paths for edge in path["edges"])
    assert [path["promptText"] for path in batch_paths] == ["alpha", "beta"]
    assert all(point["promptText"] == path["promptText"] for path in batch_paths for point in path["points"])
    assert all(edge["promptText"] == path["promptText"] for path in batch_paths for edge in path["edges"])
    rerendered_batch_payload = build_activation_map_payload(batch_path_artifact, summary, local_model_context=local_meta)
    assert {
        path["promptId"]: path["promptColor"] for path in rerendered_batch_payload["activationPaths"]
    } == {path["promptId"]: path["promptColor"] for path in batch_paths}
    batch_html = activation_map_html(batch_path_payload)
    legend_html = _legend_panel_html(batch_path_payload)
    assert "gs-prompt-legend-panel" in legend_html
    assert "width: 10px;" in legend_html
    long_prompt = (
        "alpha prompt with enough extra detail to reach the far edge of the prompt paths panel before truncation "
        * 4
    ).strip()
    long_prompt_artifact = build_run_artifact(
        run_id="long_prompt_paths",
        mode="Batch run",
        backend="llama.cpp",
        model="local",
        prompts=[
            {
                "prompt_id": 0,
                "label": "a",
                "prompt": long_prompt,
                "output": "ok",
                "trace_rows": normalize_llama_trace(
                    {"layer_inputs": [
                        {"layer": 0, "stream": "resid_pre", "token_index": 0, "activation_norm": 1.0, "top_dims": [{"dimension": 1, "activation": 1.0}]},
                        {"layer": 1, "stream": "resid_pre", "token_index": 0, "activation_norm": 1.0, "top_dims": [{"dimension": 2, "activation": 1.0}]},
                    ]},
                    prompt_id=0,
                    label="a",
                    metadata={"run_id": "long_prompt_paths"},
                ),
            },
            {
                "prompt_id": 1,
                "label": "b",
                "prompt": "short comparison prompt",
                "output": "ok",
                "trace_rows": normalize_llama_trace(
                    {"layer_inputs": [
                        {"layer": 0, "stream": "resid_pre", "token_index": 0, "activation_norm": 1.0, "top_dims": [{"dimension": 3, "activation": 1.0}]},
                        {"layer": 1, "stream": "resid_pre", "token_index": 0, "activation_norm": 1.0, "top_dims": [{"dimension": 4, "activation": 1.0}]},
                    ]},
                    prompt_id=1,
                    label="b",
                    metadata={"run_id": "long_prompt_paths"},
                ),
            },
        ],
    )
    long_prompt_payload = build_activation_map_payload(long_prompt_artifact, summary, local_model_context=local_meta)
    assert long_prompt_payload["promptLegendPanel"]["entries"][0]["label"] == long_prompt
    assert long_prompt in _legend_panel_html(long_prompt_payload)
    legend_source = inspect.getsource(activation_map_view._legend_panel_html)
    assert '""" %' not in legend_source
    assert "% (" not in legend_source
    assert "drawPromptLegend" not in batch_html
    assert "drawPromptLegend(rect)" not in batch_html
    assert batch_path_payload["promptLegendPanel"]["placement"] == "external_below_graph"
    assert batch_path_payload["promptLegendPanel"]["entries"]
    assert batch_path_payload["promptLegendPanel"]["entries"][0]["promptId"] == 0
    assert "promptStyle(path)" in batch_html
    assert "promptRows(path)" in batch_html
    assert "promptPreviewList" in batch_html
    assert "Prompt path" in batch_html
    assert "box2ColorStyle(group" in batch_html
    assert "drawPromptOverlapMarkers" in batch_html
    assert "promptActivations" in batch_html
    assert "prompt_id(s)" in batch_html
    assert "batch_id(s)" in batch_html
    assert "token_id(s)" in batch_html
    box2 = batch_path_payload["diagnostics"]["box2"]
    assert box2["selectedLayer"] == "L0"
    assert box2["visibleNodeCount"] >= 2
    assert box2["activeNodeCount"] >= 2
    assert box2["activationScaling"] == "normalized"
    assert box2["promptColorMode"] == "prompt_palette"
    assert box2["overlappingPromptNodes"] == 0
    assert box2["aggregateMode"] is False
    box2_groups = [group for group in batch_path_payload["nodeGroups"] if group["layerId"] == "L0"]
    assert len(box2_groups) >= 2
    assert any(group["normalizedActivation"] > 0 for group in box2_groups)
    assert all(group.get("promptActivations") for group in box2_groups)
    assert all(group.get("box2Opacity", 0) > 0 for group in box2_groups)
    assert all(group["box2Radius"] > 2.8 for group in box2_groups)
    assert max(group["box2Radius"] for group in layer0_groups) > min(group["box2Radius"] for group in layer0_groups)

    overlap_payload = build_activation_map_payload(_overlap_path_fixture_artifact(), summary, local_model_context=local_meta)
    overlap_group = next(group for group in overlap_payload["nodeGroups"] if group["groupId"] == "L0-N5")
    assert overlap_payload["diagnostics"]["box2"]["overlappingPromptNodes"] == 1
    assert overlap_group["promptOverlapCount"] == 2
    assert overlap_group["promptColorStrategy"] == "multi_prompt_overlap"
    assert {item["promptId"] for item in overlap_group["promptActivations"]} == {0, 1}
    assert {item["promptText"] for item in overlap_group["promptActivations"]} == {"shared alpha prompt", "shared beta prompt"}
    assert set(overlap_group["promptIds"]) == {0, 1}
    assert set(overlap_group["batchIds"]) == {"batch-0", "batch-1"}
    assert overlap_group["promptPreviewMoreCount"] == 0

    annotation_payload_path = Path("data/node_annotations/smoke_payload_annotations.json")
    annotation_payload_path.unlink(missing_ok=True)
    annotation_model_meta = build_model_meta(summary, local_meta, "llama.cpp")
    upsert_annotation(
        annotation_model_meta,
        layer=0,
        cluster_id=1,
        node_id="L0-N1",
        node_range=[1, 1],
        tags=["comedy"],
        note="Comedy prompt marker.",
        created_from={"run_id": "batch_paths", "prompt_id": 0, "batch_id": "batch-0", "prompt_excerpt": "alpha"},
        path=annotation_payload_path,
    )
    annotated_payload = build_activation_map_payload(
        batch_path_artifact,
        summary,
        local_model_context=local_meta,
        annotations_path=annotation_payload_path,
    )
    annotated_group = next(group for group in annotated_payload["nodeGroups"] if group["groupId"] == "L0-N1")
    assert annotated_group["annotationMatchType"] == "exact"
    assert annotated_group["annotationTags"] == ["comedy"]
    assert annotated_group["annotationNote"] == "Comedy prompt marker."
    annotated_html = activation_map_html(annotated_payload)
    assert "annotationRows(group)" in annotated_html
    assert "annotation match" in annotated_html
    assert "Comedy prompt marker." in annotated_html
    annotation_payload_path.unlink(missing_ok=True)

    single_fixture_payload = build_activation_map_payload(_path_fixture_artifact(1), summary, local_model_context=local_meta)
    assert single_fixture_payload["visualizationMode"] == "single_prompt"
    assert single_fixture_payload["diagnostics"]["promptColorMode"] == "single"
    assert single_fixture_payload["diagnostics"]["promptColorCount"] == 1
    assert single_fixture_payload["activationPaths"][0]["promptColor"] == "#62E4FF"
    assert len(single_fixture_payload["activationPaths"]) == 1
    assert single_fixture_payload["activationPaths"][0]["promptId"] == 0

    fifteen_payload = build_activation_map_payload(_path_fixture_artifact(15), summary, local_model_context=local_meta)
    assert fifteen_payload["visualizationMode"] == "batch_overlay"
    assert fifteen_payload["diagnostics"]["promptColorMode"] == "prompt_palette"
    assert fifteen_payload["diagnostics"]["promptColorCount"] == 15
    assert fifteen_payload["diagnostics"]["paletteSize"] == 10
    assert fifteen_payload["diagnostics"]["colorsReused"] is True
    assert len(fifteen_payload["activationPaths"]) == 15
    assert len({path["promptId"] for path in fifteen_payload["activationPaths"]}) == 15
    sorted_palette_paths = sorted(fifteen_payload["activationPaths"], key=lambda item: int(item["promptId"]))
    assert len({path["promptColor"] for path in sorted_palette_paths[:10]}) == 10
    assert sorted_palette_paths[10]["promptColor"] == sorted_palette_paths[0]["promptColor"]
    assert sorted_palette_paths[10]["promptColorCycle"] == 1
    assert sorted_palette_paths[10]["promptOpacity"] < sorted_palette_paths[0]["promptOpacity"]
    assert fifteen_payload["promptColorLegend"][0]["promptId"] == 0
    assert fifteen_payload["promptColorLegendMoreCount"] == 5
    assert fifteen_payload["promptLegendPanel"]["colorsReused"] is True
    assert fifteen_payload["promptLegendPanel"]["moreCount"] == 5
    assert {path["batchId"]: path["promptText"] for path in fifteen_payload["activationPaths"]} == {
        f"batch-{index}": f"prompt {index}" for index in range(15)
    }
    assert all(point["promptText"] == f"prompt {point['promptId']}" for path in fifteen_payload["activationPaths"] for point in path["points"])
    assert "aggregate_heatmap" not in {fifteen_payload["visualizationMode"], fifteen_payload["diagnostics"]["rendererMode"]}

    scalar_multi_payload = build_activation_map_payload(_path_fixture_artifact(7, scalar_only=True), summary, local_model_context=local_meta)
    heatmap_prompt_list = next(cell["promptPreviewList"] for cell in scalar_multi_payload["heatmap"] if len(cell.get("promptPreviewList", [])) > 5)
    assert len(heatmap_prompt_list) == 7
    assert heatmap_prompt_list[:5] == ["prompt 0", "prompt 1", "prompt 2", "prompt 3", "prompt 4"]

    aggregate_payload = build_activation_map_payload(_aggregate_only_artifact(), summary, local_model_context=local_meta)
    assert aggregate_payload["visualizationMode"] == "aggregate_heatmap"
    assert aggregate_payload["diagnostics"]["promptColorMode"] == "aggregate"
    assert aggregate_payload["diagnostics"]["promptColorCount"] == 0
    assert aggregate_payload["promptColorLegend"] == []
    assert aggregate_payload["promptLegendPanel"]["entries"] == []
    assert aggregate_payload["diagnostics"]["box2"]["aggregateMode"] is True
    assert aggregate_payload["diagnostics"]["box2"]["promptColorMode"] == "aggregate"
    assert aggregate_payload["activationPaths"] == []
    assert aggregate_payload["heatmap"]
    assert "Backend trace is already aggregated. Per-prompt paths unavailable." in aggregate_payload["diagnostics"]["warnings"]
    aggregate_report = inspect_trace_artifact(_aggregate_only_artifact(), summary, local_model_context=local_meta)
    assert aggregate_report["prompt_count"] == 15
    assert aggregate_report["batch_count"] == 0
    assert aggregate_report["data_granularity"] == "aggregated"

    scalar_only_payload = build_activation_map_payload(_path_fixture_artifact(1, scalar_only=True), summary, local_model_context=local_meta)
    assert scalar_only_payload["dataMode"] == "scalar_layer_summary"
    assert scalar_only_payload["activationPaths"] == []
    assert scalar_only_payload["visualizationMode"] == "aggregate_heatmap"
    assert scalar_only_payload["heatmap"]

    tokenless_payload = build_activation_map_payload(_tokenless_prompt_artifact(), summary, local_model_context=local_meta)
    assert tokenless_payload["dataMode"] == "top_dims_approx"
    assert tokenless_payload["diagnostics"]["data_granularity"] == "raw"
    assert tokenless_payload["visualizationMode"] == "single_prompt"

    compare_payload = build_activation_map_payload(
        _path_fixture_artifact(2),
        summary,
        local_model_context=local_meta,
        visualization_mode="compare_prompts",
        selected_prompt=0,
        compare_prompt=1,
    )
    assert compare_payload["visualizationMode"] == "compare_prompts"
    assert compare_payload["activationPaths"] == []
    assert compare_payload["comparePrompts"]["deltas"]
    assert compare_payload["diagnostics"]["promptColorMode"] == "prompt_palette"
    assert compare_payload["comparePrompts"]["selectedPromptColor"] != compare_payload["comparePrompts"]["comparePromptColor"]
    _assert_main_layer_resolver_and_trace_mock(summary, local_meta)
    _assert_model_layer_contracts(summary, local_meta)
    _assert_single_trace_failure_preserves_reason()
    _assert_patched_baseline_visible_to_app(rows)
    _assert_activation_patch_recipe_round_trip()

    profile = get_behavior_profile("concise_helpfulness")
    assert profile.name in list_behavior_profiles()
    score = score_behavior_output("clear concise safe action", profile=profile, run_id="run1")
    assert score["score"] > 0
    scored = score_run_artifact(artifact, profile=profile)
    timeline = behavior_timeline_df([scored.assign(run_id="run1"), scored.assign(run_id="run2", score=scored["score"] + 1)])
    assert not timeline.empty
    ranked = rank_activation_paths(path_df, positive_label="single", negative_label="x")
    assert ranked.empty
    assert recommended_steering_targets(pd.DataFrame()) == []

    assert activation_path_graph(path_df) is not None
    assert batch_activation_heatmap(batch_heatmap_df(artifact)) is not None
    assert label_activation_heatmap(label_heatmap_df(artifact)) is not None
    assert behavior_score_timeline_fig(timeline) is not None
    assert behavior_delta_bar_fig(timeline, baseline_run_id="run1", comparison_run_id="run2") is not None
    assert path_rank_bar_fig(pd.DataFrame([{"layer": 0, "stream": "resid_pre", "score": 1.0, "delta": 0.5, "positive_mean": 1, "negative_mean": 0, "positive_count": 1, "negative_count": 1}])) is not None

    fake_gguf = CONTROL_VECTOR_DIR / "smoke_fake.gguf"
    _write_fake_gguf(fake_gguf)
    tensors = pd.DataFrame(read_gguf_tensor_index(fake_gguf))
    assert not tensors.empty
    assert gguf_tensor_shape_scatter_fig(tensors) is not None
    preflight = preflight_control_vector_run(fake_gguf, None, None, "/no/generator", "/no/server")
    assert any(check.name == "GGUF model" and check.status == "ok" for check in preflight.checks)
    cmd = build_cvector_command(fake_gguf, "pos.txt", "neg.txt", "vec.gguf", "/bin/gen")
    assert "--positive-file" in cmd and "--negative-file" in cmd
    server_cmd = build_llama_server_command(fake_gguf, "vec.gguf", 1.25, 1, 2, "/bin/server", port=8088, alias="local")
    assert "--control-vector-scaled" in server_cmd
    assert "llama-server" not in shell_join(["/custom/server"])
    failure = classify_cvector_failure("invalid output tensor count", "")
    assert failure.cause

    save_chat([{"role": "user", "content": "smoke"}], label="smoke")
    assert isinstance(load_chat(), list)

    import glass_skull.fuzzing as fuzzing
    original_chat = fuzzing.chat_completion
    original_trace = fuzzing.trace_glass_prompt
    try:
        fuzzing.chat_completion = lambda *args, **kwargs: "clear concise action"
        fuzzing.trace_glass_prompt = lambda *args, **kwargs: trace
        result = run_fuzz_experiment(
            name="smoke_local",
            prompts=prompts[:1],
            chat_backend="llama.cpp",
            trace_enabled=True,
            llama_url="http://local",
            llama_model_alias="local",
            layers=[0, 1],
            streams=["resid_pre"],
            run_id="batch1",
        )
    finally:
        fuzzing.chat_completion = original_chat
        fuzzing.trace_glass_prompt = original_trace
    assert result["summary"]["trace_supported"] is True
    assert result["summary"]["trace_row_count"] > 0
    assert result["summary"]["trace_unavailable_count"] == 0
    assert result["artifact"]["summary"]["trace_row_count"] > 0
    assert result["artifact"]["summary"]["trace_unavailable_count"] == 0

    try:
        fuzzing.chat_completion = lambda *args, **kwargs: "clear concise action"
        fuzzing.trace_glass_prompt = lambda *args, **kwargs: {"layer_inputs": {"reason": "batch trace disabled"}}
        unavailable_result = run_fuzz_experiment(
            name="smoke_unavailable",
            prompts=prompts[:1],
            chat_backend="llama.cpp",
            trace_enabled=True,
            llama_url="http://local",
            llama_model_alias="local",
            layers=[0, 1],
            streams=["resid_pre"],
            run_id="batch2",
        )
    finally:
        fuzzing.chat_completion = original_chat
        fuzzing.trace_glass_prompt = original_trace
    unavailable_row = unavailable_result["artifact"]["prompts"][0]["trace_rows"][0]
    assert unavailable_row["trace_available"] is False
    assert unavailable_row["unavailable_reason"] == "batch trace disabled"
    assert unavailable_result["summary"]["trace_unavailable_count"] == 1

    print("Glass Skull local-only smoke check passed.")


if __name__ == "__main__":
    main()
