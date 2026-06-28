from __future__ import annotations

import ast
import importlib
import json
import inspect
from pathlib import Path
import types
from typing import Any

import pandas as pd

from glass_skull.activation_map import build_activation_map_payload, build_model_meta, inspect_trace_artifact
from glass_skull.activation_map_view import activation_map_html
from glass_skull.behavior_profiles import get_behavior_profile, list_behavior_profiles
from glass_skull.behavior_scoring import behavior_timeline_df, score_behavior_output, score_run_artifact
from glass_skull.chat_store import load_chat, save_chat
from glass_skull.config import CONTROL_SET_DIR, CONTROL_VECTOR_DIR, DEFAULT_GGUF_MODEL_PATH, ensure_dirs
from glass_skull.experiment_store import safe_slug
from glass_skull.fuzzing import run_fuzz_experiment
from glass_skull import activation_map_view
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
        {"_positive_int", "_trace_layer_count_from_glass_info", "resolve_trace_layers"},
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
        {"_positive_int", "_trace_layer_count_from_glass_info", "resolve_trace_layers"},
        namespace,
    )
    assert funcs["resolve_trace_layers"]("http://missing", "local", {"layers": 41}) == list(range(41))
    assert funcs["resolve_trace_layers"]("http://missing", "local", {"layers": 0}) == [0]
    namespace["get_glass_info"] = lambda *args, **kwargs: {"layers": "bad", "meta": {"n_layer": -3}}
    funcs = _load_main_functions(
        {"_positive_int", "_trace_layer_count_from_glass_info", "resolve_trace_layers"},
        namespace,
    )
    assert funcs["resolve_trace_layers"]("http://bad", "local", {"layers": "invalid"}) == [0]


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
    assert 'key == "llama_model_path" and not st.session_state.get(key)' in main_source
    assert "persist_single_run_artifact(artifact)" in main_source
    assert "latest_saved_behavior_artifact()" in main_source
    assert 'DEFAULT_CHAT_INPUT = "hi"' in main_source
    assert 'st.text_input("Send a local prompt", value=DEFAULT_CHAT_INPUT' in main_source
    assert 'st.form("chat_prompt_form", clear_on_submit=True)' in main_source
    assert '{"Single message", "Batch run"}' in main_source
    assert "def resolve_trace_layers(" in main_source
    assert "trace_layers = resolve_trace_layers(" in main_source
    assert "layers=trace_layers" in main_source
    assert "active_recipe = validate_activation_patch_recipe(loaded_recipe)" in main_source
    assert "build_activation_patch_backend_payload(active_recipe" in main_source
    render_source = inspect.getsource(activation_map_view.render_activation_map)
    assert "st.iframe" in render_source
    assert "components.html" not in render_source
    assert normalize_base_url("http://127.0.0.1:8080/v1") == "http://127.0.0.1:8080"
    assert safe_slug("../bad name") == "bad_name"

    prompts = load_txt("one\n# skip\ntwo")
    assert [p.prompt for p in prompts] == ["one", "two"]
    labeled = load_jsonl('{"prompt":"A","label":"x"}\n{"prompt":"B"}')
    assert [p.label for p in labeled] == ["x", "unlabeled"]
    batch = batch_items_from_inputs(pasted_payload="pasted", repeat_prompt="again", repeat_count=2, uploaded_items=labeled)
    assert len(batch) == 5
    assert dashboard_context("Batch run", "abc") == {"mode": "Batch run", "run_id": "abc"}
    assert new_run_id("unit").startswith("unit_")

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
    layer0_groups = [group for group in payload["nodeGroups"] if group["layerId"] == "L0" and group["activationValue"] > 0]
    assert len(layer0_groups) > 1
    assert all(edge["method"] != "cosine_similarity" for edge in payload["activationEdges"])
    assert all(edge["tokenIndex"] == payload["activationPaths"][0]["tokenIndex"] for edge in payload["activationEdges"])
    html = activation_map_html(payload)
    assert "canvas" in html and "selectedDiagnostics" in html and "Activation path unavailable" not in html
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
    assert len(batch_paths) == 2
    assert {path["pathMethod"] for path in batch_paths} == {"top_activation_per_layer"}
    assert [point["groupId"] for point in batch_paths[0]["points"]] != [point["groupId"] for point in batch_paths[1]["points"]]
    assert all(point["promptId"] == path["promptId"] for path in batch_paths for point in path["points"])
    assert all(point["tokenIndex"] == 0 for path in batch_paths for point in path["points"])
    assert all(edge["promptId"] == path["promptId"] for path in batch_paths for edge in path["edges"])
    assert "Prompt path" in activation_map_html(batch_path_payload)

    single_fixture_payload = build_activation_map_payload(_path_fixture_artifact(1), summary, local_model_context=local_meta)
    assert single_fixture_payload["visualizationMode"] == "single_prompt"
    assert len(single_fixture_payload["activationPaths"]) == 1
    assert single_fixture_payload["activationPaths"][0]["promptId"] == 0

    fifteen_payload = build_activation_map_payload(_path_fixture_artifact(15), summary, local_model_context=local_meta)
    assert fifteen_payload["visualizationMode"] == "batch_overlay"
    assert len(fifteen_payload["activationPaths"]) == 15
    assert len({path["promptId"] for path in fifteen_payload["activationPaths"]}) == 15
    assert "aggregate_heatmap" not in {fifteen_payload["visualizationMode"], fifteen_payload["diagnostics"]["rendererMode"]}

    aggregate_payload = build_activation_map_payload(_aggregate_only_artifact(), summary, local_model_context=local_meta)
    assert aggregate_payload["visualizationMode"] == "aggregate_heatmap"
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
    _assert_main_layer_resolver_and_trace_mock(summary, local_meta)
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
