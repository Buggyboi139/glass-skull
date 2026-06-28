from __future__ import annotations

import json
import struct
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from glass_skull.aggregation import label_separation_table
from glass_skull.activation_map import build_activation_map_payload, build_model_meta
from glass_skull.attention_view import indexed_tokens
from glass_skull.chat_store import load_chat, save_chat
from glass_skull.config import CONTROL_SET_DIR, CONTROL_VECTOR_DIR, DEFAULT_MODEL, MODEL_PRESETS, ensure_dirs
from glass_skull.experiment_store import safe_slug
from glass_skull.feature_store import compatible_features
from glass_skull.hf_access import HFTokenStatus
from glass_skull.hf_loader import build_hf_load_plan
from glass_skull.hf_registry import capabilities_for_backend, families, get_model, registry_as_dicts, visible_models
import glass_skull.fuzzing as fuzzing
from glass_skull.run_artifacts import (
    activation_path_df,
    batch_heatmap_df,
    build_run_artifact,
    dimension_frequency_df,
    label_heatmap_df,
    normalize_llama_trace,
    normalize_transformerlens_trace,
    trace_unavailable_row,
)
from glass_skull.lens import logit_lens_table
from glass_skull.llama_control import build_cvector_command, build_llama_server_command, classify_cvector_failure, preflight_control_vector_run, read_gguf_tensor_index, shell_join
from glass_skull import llama_client
from glass_skull.llama_client import build_steering_metadata, chat_completion, get_glass_info, normalize_base_url, per_request_steering_supported, trace_glass_prompt
from glass_skull.prompt_loader import load_jsonl, load_txt
from glass_skull.ui_lab import hf_enabled, lab_enabled, trace_enabled
from glass_skull.ui_local import batch_items_from_inputs, dashboard_context, new_run_id, parse_pasted_payload, repeated_prompt_items
from glass_skull.visuals import comparison_delta_heatmap, gguf_tensor_dtype_fig, gguf_tensor_shape_scatter_fig, gguf_tensors_by_component_fig, gguf_tensors_per_layer_fig


def _write_gguf_string(f, value: str) -> None:
    data = value.encode("utf-8")
    f.write(struct.pack("<Q", len(data)))
    f.write(data)


def _write_minimal_gguf(path: Path) -> None:
    with path.open("wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 3))
        f.write(struct.pack("<Q", 1))
        _write_gguf_string(f, "general.architecture")
        f.write(struct.pack("<I", 8))
        _write_gguf_string(f, "qwen35moe")
        for index, name in enumerate(["token_embd.weight", "blk.0.attn_qkv.weight", "blk.1.ffn_up_exps.weight"]):
            _write_gguf_string(f, name)
            f.write(struct.pack("<I", 2))
            f.write(struct.pack("<Q", 4 + index))
            f.write(struct.pack("<Q", 8))
            f.write(struct.pack("<I", 8))
            f.write(struct.pack("<Q", index * 128))


def main() -> None:
    ensure_dirs()

    assert DEFAULT_MODEL in MODEL_PRESETS, "DEFAULT_MODEL should be one of MODEL_PRESETS"
    assert normalize_base_url("http://127.0.0.1:8080/v1") == "http://127.0.0.1:8080"
    assert safe_slug("Glass Probe!!!") == "Glass_Probe"
    assert CONTROL_SET_DIR.exists()
    assert CONTROL_VECTOR_DIR.exists()
    assert indexed_tokens(["The", " cat"]) == ["0:The", "1: cat"]
    assert isinstance(compatible_features(512), list)

    captured_requests = []
    original_request_json = llama_client._request_json

    def fake_request_json(method, url, payload=None, timeout=120.0):
        captured_requests.append({"method": method, "url": url, "payload": payload or {}})
        if url.endswith("/v1/chat/completions"):
            return {"choices": [{"message": {"content": ""}}]}
        if url.endswith("/completion"):
            return {"content": "fallback ok"}
        return {"data": []}

    llama_client._request_json = fake_request_json
    try:
        assert chat_completion("http://router.local", "hello", model_alias="qwen-local") == "fallback ok"
        assert chat_completion(
            "http://router.local",
            "second",
            model_alias="qwen-local",
            messages=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "second"},
            ],
        ) == "fallback ok"
    finally:
        llama_client._request_json = original_request_json
    assert captured_requests[0]["payload"]["model"] == "qwen-local"
    assert captured_requests[1]["payload"]["model"] == "qwen-local"
    assert captured_requests[0]["payload"]["enable_thinking"] is False
    assert captured_requests[0]["payload"]["reasoning_budget"] == 0
    assert captured_requests[0]["payload"]["chat_template_kwargs"]["enable_thinking"] is False
    assert captured_requests[1]["payload"]["enable_thinking"] is False
    assert captured_requests[1]["payload"]["reasoning_budget"] == 0
    assert captured_requests[0]["payload"]["messages"][0]["role"] == "system"
    assert "Do not include reasoning" in captured_requests[0]["payload"]["messages"][0]["content"]
    assert captured_requests[2]["payload"]["messages"][1]["content"] == "first"
    assert captured_requests[2]["payload"]["messages"][2]["role"] == "assistant"
    assert captured_requests[2]["payload"]["messages"][3]["content"] == "second"
    assert captured_requests[1]["payload"]["prompt"].startswith("System: Answer directly")

    steering_payload = build_steering_metadata("/tmp/vector.gguf", 1.25, 2, 8)
    try:
        chat_completion("http://router.local", "hello", steering=steering_payload, steering_supported=False)
        raise AssertionError("Expected unsupported steering to fail closed before request")
    except RuntimeError as exc:
        assert "does not advertise" in str(exc)

    captured_requests = []

    def fake_steering_request_json(method, url, payload=None, timeout=120.0):
        captured_requests.append({"method": method, "url": url, "payload": payload or {}})
        if url.endswith("/v1/chat/completions"):
            return {"choices": [{"message": {"content": "steered ok"}}]}
        return {}

    llama_client._request_json = fake_steering_request_json
    try:
        assert chat_completion(
            "http://router.local",
            "hello",
            model_alias="qwen-local",
            steering=steering_payload,
            steering_supported=True,
        ) == "steered ok"
    finally:
        llama_client._request_json = original_request_json
    assert captured_requests[0]["payload"]["glass_skull"]["steering"] == steering_payload
    assert captured_requests[0]["payload"]["metadata"]["glass_skull"]["steering"] == steering_payload

    captured_requests = []

    def fake_glass_request_json(method, url, payload=None, timeout=120.0):
        captured_requests.append({"method": method, "url": url, "payload": payload or {}, "timeout": timeout})
        if "/glass-skull/info" in url:
            return {
                "object": "glass_skull.info",
                "model": {
                    "id": "qwen-local",
                    "aliases": ["qwen-local"],
                    "tags": [],
                    "path": "/models/qwen.gguf",
                    "architecture": "qwen35moe",
                    "description": "Qwen local",
                },
                "context_length": 4096,
                "training_context_length": 32768,
                "embedding_width": 2048,
                "layers": 40,
                "heads": 16,
                "kv_heads": 2,
                "params": 8_000_000_000,
                "size": 4_000_000_000,
                "capabilities": {
                    "completion": True,
                    "embedding": False,
                    "rerank": False,
                    "multimodal": False,
                    "modalities": {"image": False, "audio": False, "video": False},
                    "trace": {
                        "prompt_tokens": {"supported": True},
                        "next_token_logits": {"supported": False, "reason": "not generated"},
                        "activations": {"supported": False, "reason": "not exposed"},
                    },
                    "steering": {
                        "per_request": {
                            "supported": True,
                            "guards": {
                                "exclusive_execution": True,
                                "prompt_cache_reuse": False,
                            },
                        },
                    },
                },
                "meta": {
                    "vocab_type": 1,
                    "n_vocab": 151936,
                    "n_ctx": 4096,
                    "n_ctx_train": 32768,
                    "n_embd": 2048,
                    "n_layer": 40,
                    "n_head": 16,
                    "n_head_kv": 2,
                    "n_params": 8_000_000_000,
                    "size": 4_000_000_000,
                },
            }
        if url.endswith("/glass-skull/trace"):
            return {
                "object": "glass_skull.trace",
                "model": "qwen-local",
                "supported": True,
                "prompt": {
                    "traces": [
                        {
                            "tokens": [101, 202],
                            "n_tokens": 2,
                            "n_positions": 2,
                            "contains_media": False,
                            "pieces": [{"id": 101, "piece": "hello"}, {"id": 202, "piece": " world"}],
                        }
                    ],
                    "n_tokens_total": 2,
                    "n_positions_total": 2,
                    "contains_media": False,
                },
                "next_tokens": {"supported": False, "reason": "not generated by this endpoint"},
                "activations": {"supported": False, "reason": "not exposed"},
            }
        return {}

    llama_client._request_json = fake_glass_request_json
    try:
        glass_info = get_glass_info("http://glass.local/v1/", model_alias="qwen-local")
        glass_trace = trace_glass_prompt(
            "http://glass.local/v1/",
            "hello",
            model_alias="qwen-local",
            layers=[0, 1],
            streams=["resid_post"],
            max_new_tokens=8,
            with_pieces=True,
        )
    finally:
        llama_client._request_json = original_request_json
    assert glass_info["model"]["architecture"] == "qwen35moe"
    assert glass_info["context_length"] == 4096
    assert glass_info["capabilities"]["trace"]["prompt_tokens"]["supported"] is True
    assert per_request_steering_supported(glass_info) is True
    assert glass_info["meta"]["n_layer"] == 40
    assert glass_trace["prompt"]["n_tokens_total"] == 2
    assert glass_trace["prompt"]["traces"][0]["tokens"] == [101, 202]
    assert glass_trace["prompt"]["traces"][0]["pieces"][1]["piece"] == " world"
    assert captured_requests[0]["method"] == "GET"
    assert captured_requests[0]["url"] == "http://glass.local/glass-skull/info?model=qwen-local"
    assert captured_requests[1]["method"] == "POST"
    assert captured_requests[1]["url"] == "http://glass.local/glass-skull/trace"
    assert captured_requests[1]["payload"] == {
        "prompt": "hello",
        "model": "qwen-local",
        "model_alias": "qwen-local",
        "layers": [0, 1],
        "streams": ["resid_post"],
        "max_new_tokens": 8,
        "with_pieces": True,
    }

    def fake_bad_glass_request_json(method, url, payload=None, timeout=120.0):
        if url.endswith("/glass-skull/info"):
            return []
        if url.endswith("/glass-skull/trace"):
            return {"prompt": {"traces": "bad"}}
        return {}

    llama_client._request_json = fake_bad_glass_request_json
    try:
        try:
            get_glass_info("http://glass.local")
            raise AssertionError("Expected get_glass_info to reject non-object JSON")
        except RuntimeError as exc:
            assert "expected object" in str(exc)

        try:
            trace_glass_prompt("http://glass.local", "hello")
            raise AssertionError("Expected trace_glass_prompt to reject malformed prompt.traces")
        except RuntimeError as exc:
            assert "prompt.traces" in str(exc)
    finally:
        llama_client._request_json = original_request_json

    captured_requests = []

    def fake_router_request_json(method, url, payload=None, timeout=120.0):
        captured_requests.append({"method": method, "url": url, "payload": payload or {}})
        if url.endswith("/v1/models"):
            return {
                "data": [
                    {"id": "default", "status": {"value": "unloaded", "failed": True}},
                    {"id": "qwen3.6-35b-mtp-q4-ks-vision", "status": {"value": "loaded"}},
                ]
            }
        if url.endswith("/v1/chat/completions"):
            return {"choices": [{"message": {"content": "<think>hidden</think>\nrouter ok\n\nReasoning:\nThis should never be shown."}}]}
        return {}

    llama_client._request_json = fake_router_request_json
    try:
        assert chat_completion("http://router.local", "hello", model_alias="default") == "router ok"
    finally:
        llama_client._request_json = original_request_json
    chat_payloads = [req["payload"] for req in captured_requests if req["url"].endswith("/v1/chat/completions")]
    assert chat_payloads[0]["model"] == "qwen3.6-35b-mtp-q4-ks-vision"
    assert chat_payloads[0]["enable_thinking"] is False

    chat_path = save_chat([{"role": "user", "content": "smoke chat"}], label="smoke")
    assert chat_path is not None and chat_path.exists()
    assert load_chat(chat_path.stem)[0]["content"] == "smoke chat"
    chat_path.unlink(missing_ok=True)

    txt_items = load_txt("The cat sat on the\n# comment\nThe car drove away\n")
    assert len(txt_items) == 2
    assert txt_items[0].prompt == "The cat sat on the"
    pasted_items = parse_pasted_payload(" first prompt\n\nsecond prompt\n")
    assert [item.prompt for item in pasted_items] == ["first prompt", "second prompt"]
    assert pasted_items[0].metadata["source"] == "pasted_payload"
    repeated_items = repeated_prompt_items("Repeat me", 3)
    assert len(repeated_items) == 3
    assert repeated_items[2].metadata["repeat_count"] == 3
    batch_items = batch_items_from_inputs(
        pasted_payload="alpha\nbeta",
        repeat_prompt="gamma",
        repeat_count=2,
        uploaded_items=txt_items[:1],
    )
    assert len(batch_items) == 5
    assert [item.prompt_id for item in batch_items] == list(range(5))
    run_id = new_run_id("smoke")
    assert run_id.startswith("smoke_")
    assert dashboard_context("Batch run", run_id)["run_id"] == run_id
    assert lab_enabled(["Local GGUF"]) is False
    assert trace_enabled(["Local GGUF"]) is False
    assert lab_enabled(["Local GGUF", "Trace model"]) is True
    assert trace_enabled(["Trace model"]) is True
    assert hf_enabled(["Hugging Face"]) is True
    original_fuzz_chat = fuzzing.chat_completion
    original_fuzz_trace_glass = fuzzing.trace_glass_prompt
    original_fuzz_tl_trace = fuzzing.trace_layers_for_prompt
    fuzzing.chat_completion = lambda *args, **kwargs: "batch ok"
    try:
        batch_result = fuzzing.run_fuzz_experiment(
            name="smoke_batch",
            prompts=batch_items[:2],
            chat_backend="llama.cpp",
            trace_enabled=False,
            llama_url="http://local",
            llama_model_alias="qwen-local",
            run_id=run_id,
            mode="Batch run",
        )
    finally:
        fuzzing.chat_completion = original_fuzz_chat
        fuzzing.trace_glass_prompt = original_fuzz_trace_glass
        fuzzing.trace_layers_for_prompt = original_fuzz_tl_trace
    assert batch_result["summary"]["run_id"] == run_id
    assert batch_result["summary"]["mode"] == "Batch run"
    assert batch_result["records"][0]["metadata"]["run_id"] == run_id
    assert batch_result["records"][0]["metadata"]["mode"] == "Batch run"
    assert Path(batch_result["summary"]["experiment_path"], "artifact.json").exists()
    assert Path(batch_result["summary"]["experiment_path"], "outputs.jsonl").exists()

    llama_trace_calls = []

    def fake_fuzz_trace_glass(*args, **kwargs):
        llama_trace_calls.append({"args": args, "kwargs": kwargs})
        return {
            "prompt": {"traces": [{"pieces": [{"piece": "alpha"}]}]},
            "layer_norms": [{"layer": 0, "stream": "resid_post", "token_index": 0, "norm": 1.5}],
        }

    def fail_tl_trace(*args, **kwargs):
        raise AssertionError("llama.cpp runs must use /glass-skull/trace, not TransformerLens tracing")

    fuzzing.chat_completion = lambda *args, **kwargs: "batch ok"
    fuzzing.trace_glass_prompt = fake_fuzz_trace_glass
    fuzzing.trace_layers_for_prompt = fail_tl_trace
    try:
        traced_llama_batch = fuzzing.run_fuzz_experiment(
            name="smoke_llama_trace_batch",
            prompts=batch_items[:1],
            chat_backend="llama.cpp",
            trace_enabled=True,
            model=object(),
            llama_url="http://local",
            llama_model_alias="qwen-local",
            layers=[0],
            streams=["resid_post"],
            run_id=run_id,
            mode="Batch run",
        )
    finally:
        fuzzing.chat_completion = original_fuzz_chat
        fuzzing.trace_glass_prompt = original_fuzz_trace_glass
        fuzzing.trace_layers_for_prompt = original_fuzz_tl_trace
    assert len(llama_trace_calls) == 1
    assert traced_llama_batch["activation_path_df"].iloc[0]["trace_source"] == "llama.cpp"

    tl_trace = type("TinyTrace", (), {
        "tokens": ["The", " cat"],
        "layer_norms": pd.DataFrame([
            {"layer": 0, "stream": "resid_post", "token_index": 0, "token": "The", "norm": 1.0},
            {"layer": 1, "stream": "mlp_out", "token_index": 1, "token": " cat", "norm": 2.5},
        ]),
    })()
    tl_rows = normalize_transformerlens_trace(tl_trace, prompt_id=7, label="animal", metadata={"run_id": run_id})
    assert tl_rows[0]["run_id"] == run_id
    assert tl_rows[1]["component"] == "mlp_out"
    assert tl_rows[1]["activation_norm"] == 2.5

    llama_rows = normalize_llama_trace(
        {
            "model": "qwen-local",
            "prompt": {"traces": [{"pieces": [{"piece": "A"}, {"piece": " B"}]}]},
            "layer_norms": [
                {"layer": 0, "stream": "resid_post", "token_index": 0, "norm": 3.0},
                {"layer": 0, "component": "mlp_out", "token_index": 1, "token": " B", "activation_norm": 4.0},
            ],
        },
        prompt_id=8,
        label="letter",
        metadata={"run_id": run_id},
    )
    assert [row["token"] for row in llama_rows] == ["A", " B"]
    assert llama_rows[0]["trace_available"] is True
    tokens_only = normalize_llama_trace(
        {"prompt": {"traces": [{"tokens": [1, 2], "pieces": [{"piece": "x"}, {"piece": "y"}]}]}, "activations": {"supported": False, "reason": "not exposed"}},
        prompt_id=9,
        label="none",
        metadata={"run_id": run_id},
    )
    assert tokens_only == []

    artifact = build_run_artifact(
        run_id=run_id,
        mode="Batch run",
        backend="llama.cpp",
        model="qwen-local",
        prompts=[
            {"prompt_id": 8, "label": "letter", "prompt": "A B", "output": "ok", "error": None, "elapsed_ms": 12.0, "trace_rows": llama_rows},
            {"prompt_id": 7, "label": "animal", "prompt": "The cat", "output": "ok", "error": None, "elapsed_ms": 8.0, "trace_rows": tl_rows},
        ],
    )
    path_df = activation_path_df(artifact)
    assert {"run_id", "prompt_id", "label", "layer", "stream", "token_index", "activation_norm"}.issubset(path_df.columns)
    assert set(path_df["run_id"]) == {run_id}
    prompt_heat = batch_heatmap_df(artifact, group_by="prompt")
    assert {"group", "layer", "stream", "activation_norm", "run_id"}.issubset(prompt_heat.columns)
    label_heat = label_heatmap_df(artifact)
    assert set(label_heat["group"]) == {"animal", "letter"}
    dim_freq = dimension_frequency_df({"prompts": [{"trace_rows": [{"run_id": run_id, "prompt_id": 1, "label": "x", "layer": 0, "stream": "resid_post", "top_dims": [{"dimension": 4, "activation": -2.0}]}]}]})
    assert dim_freq.iloc[0]["dimension"] == 4
    map_summary = {
        "model_name": "tiny-trace",
        "device": "cpu",
        "layers": 3,
        "heads": 2,
        "d_model": 12,
        "d_head": 6,
        "d_mlp": 48,
        "vocab_size": 100,
        "parameters": 1234,
        "dtype": "torch.float32",
    }
    meta = build_model_meta(map_summary, None, "TransformerLens")
    assert meta["layerCount"] == 3
    assert meta["hiddenSize"] == 12
    assert meta["attentionHeads"] == 2
    assert meta["visualizationMode"] == "clustered"

    activation_payload = build_activation_map_payload(
        artifact,
        map_summary,
        local_model_context=None,
        selected_layer=1,
        selected_group="L1-G0",
        selected_batch="batch-8",
    )
    assert activation_payload["visualizationMode"] in {"sampled", "clustered", "approximate", "unavailable"}
    assert activation_payload["modelMeta"]["layerCount"] == 3
    assert [layer["name"] for layer in activation_payload["layers"]] == ["L0", "L1", "L2"]
    assert activation_payload["layers"][1]["selected"] is True
    assert activation_payload["nodeGroups"], "node groups should derive from hidden size or trace dimensions"
    assert activation_payload["activationPaths"], "available trace rows should produce activation paths"
    assert activation_payload["diagnostics"]["selectedLayer"]["name"] == "L1"
    assert activation_payload["diagnostics"]["selectedGroup"]["groupId"] == "L1-G0"
    assert activation_payload["diagnostics"]["modelMeta"]["hiddenSize"] == 12
    first_path = activation_payload["activationPaths"][0]
    assert {"batchId", "promptId", "points", "strength", "visualizationMode"}.issubset(first_path)
    assert first_path["points"], "paths should include per-layer Canvas points"
    layer_zero_group = next(group for group in activation_payload["heatmapStats"]["groups"] if group["groupId"] == "L0-G0")
    assert layer_zero_group["activationCount"] == 3
    assert layer_zero_group["maxActivation"] == 4.0
    assert abs(layer_zero_group["meanActivation"] - (8.0 / 3.0)) < 1e-9

    unavailable_artifact = build_run_artifact(
        run_id=f"{run_id}-unavailable",
        mode="Batch run",
        backend="llama.cpp",
        model="qwen-local",
        prompts=[
            {
                "prompt_id": 11,
                "label": "missing",
                "prompt": "No trace",
                "output": "n/a",
                "error": None,
                "elapsed_ms": 1.0,
                "trace_rows": [
                    {
                        "run_id": f"{run_id}-unavailable",
                        "prompt_id": 11,
                        "label": "missing",
                        "layer": None,
                        "stream": "resid_post",
                        "component": "resid_post",
                        "token_index": 0,
                        "token": "No",
                        "activation_norm": None,
                        "trace_available": True,
                        "trace_source": "llama.cpp",
                        "unavailable_reason": "",
                        "top_dims": [],
                    },
                    trace_unavailable_row(f"{run_id}-unavailable", 11, "missing", "llama.cpp", "not exposed"),
                ],
            }
        ],
    )
    unavailable_payload = build_activation_map_payload(unavailable_artifact, map_summary, local_model_context=None)
    assert unavailable_payload["visualizationMode"] == "unavailable"
    assert unavailable_payload["batches"][0]["traceAvailable"] is False
    assert unavailable_payload["unavailableReason"] == "not exposed"
    assert unavailable_payload["diagnostics"]["visualizationMode"] == "unavailable"
    assert unavailable_payload["diagnostics"]["unavailableReason"] == "not exposed"
    assert unavailable_payload["activationPaths"] == []

    jsonl = '\n'.join([
        json.dumps({"label": "animal", "prompt": "Explain what a mouse is."}),
        json.dumps({"label": "vehicle", "prompt": "Explain what a car is."}),
    ])
    jsonl_items = load_jsonl(jsonl)
    assert len(jsonl_items) == 2
    assert jsonl_items[0].label == "animal"

    records = [
        {"prompt_id": 0, "label": "a", "trace_layers": [{"layer": 0, "stream": "resid_post", "norm": 1.0}]},
        {"prompt_id": 1, "label": "b", "trace_layers": [{"layer": 0, "stream": "resid_post", "norm": 2.0}]},
    ]
    sep = label_separation_table(records)
    assert not sep.empty

    registry = registry_as_dicts()
    assert registry, "HF registry should not be empty"
    assert "Gemma" in families()
    assert get_model("google/gemma-4-12B-it") is not None
    assert visible_models("Qwen"), "Qwen family should be present"
    llama_caps = capabilities_for_backend("llama.cpp glass")
    assert llama_caps["activation_steering"] is False
    assert llama_caps["control_vector_steering"] is True
    assert capabilities_for_backend("TransformerLens")["activation_steering"] is True
    assert HFTokenStatus(configured=False, valid=False).label() == "No token configured"
    plan = build_hf_load_plan("Qwen/Qwen3-4B", token=None)
    assert plan.repo_id == "Qwen/Qwen3-4B"

    # Import-only checks so new cockpit modules fail fast without loading a model.
    assert callable(logit_lens_table)
    assert callable(comparison_delta_heatmap)

    cvec_cmd = build_cvector_command("/models/qwen.gguf", "pos.txt", "neg.txt", "vec.gguf", "/bin/cvec", "mean")
    assert "--positive-file" in cvec_cmd
    assert "-ngl" not in cvec_cmd
    explicit_cvec_cmd = build_cvector_command("/models/qwen.gguf", "pos.txt", "neg.txt", "vec.gguf", "/bin/cvec", "mean", ngl=12, fit="off", ctx_size=4096, pca_batch=100, pca_iter=2000)
    assert explicit_cvec_cmd[3:5] == ["-ngl", "12"]
    assert "--fit" in explicit_cvec_cmd
    assert "off" in explicit_cvec_cmd
    assert "-c" in explicit_cvec_cmd
    assert "--pca-batch" in explicit_cvec_cmd
    assert "--pca-iter" in explicit_cvec_cmd
    ngl_failure = classify_cvector_failure("failed to fit params: n_gpu_layers already set by user to 999, abort")
    assert "GPU layer auto-fit" in ngl_failure.cause
    diff_failure = classify_cvector_failure("GGML_ASSERT((int) diff_filtered.size() == n_layers - 1) failed")
    assert "layer-output count" in diff_failure.cause

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        model = root / "missing.gguf"
        cvec = root / "missing-cvec"
        server = root / "missing-server"
        pos = root / "pos.txt"
        neg = root / "neg.txt"
        pos.write_text("a\nb\n", encoding="utf-8")
        neg.write_text("c\n", encoding="utf-8")
        preflight = preflight_control_vector_run(model, pos, neg, cvec, server)
        assert preflight.errors
        assert any("Model path does not exist" in err for err in preflight.errors)
        assert any("equal non-empty line counts" in err for err in preflight.errors)

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        model = root / "mini.gguf"
        _write_minimal_gguf(model)
        pos = root / "pos.txt"
        neg = root / "neg.txt"
        cvec = root / "llama-cvector-generator"
        server = root / "llama-server"
        pos.write_text("positive\n", encoding="utf-8")
        neg.write_text("negative\n", encoding="utf-8")
        cvec.write_text("", encoding="utf-8")
        server.write_text("", encoding="utf-8")
        preflight = preflight_control_vector_run(model, pos, neg, cvec, server)
        assert not preflight.errors
        assert preflight.model_architecture == "qwen35moe"
        tensors = pd.DataFrame(read_gguf_tensor_index(model))
        assert len(tensors) == 3
        assert callable(gguf_tensors_per_layer_fig)
        assert gguf_tensors_per_layer_fig(tensors) is not None
        assert gguf_tensors_by_component_fig(tensors) is not None
        assert gguf_tensor_dtype_fig(tensors) is not None
        assert gguf_tensor_shape_scatter_fig(tensors) is not None

    server_cmd = build_llama_server_command("/models/qwen.gguf", "vec.gguf", 1.25, 20, 60, "/bin/llama-server", port=8088, alias="qwen-local")
    assert "--control-vector-scaled" in server_cmd
    assert "vec.gguf:1.25" in server_cmd
    assert "--control-vector-layer-range" in server_cmd
    assert "--alias" in server_cmd
    assert "qwen-local" in server_cmd
    assert "llama-server" in shell_join(server_cmd)

    print("Glass Skull smoke check passed.")


if __name__ == "__main__":
    main()
