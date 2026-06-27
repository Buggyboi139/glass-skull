from __future__ import annotations

import json
import struct
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from glass_skull.aggregation import label_separation_table
from glass_skull.attention_view import indexed_tokens
from glass_skull.chat_store import load_chat, save_chat
from glass_skull.config import CONTROL_SET_DIR, CONTROL_VECTOR_DIR, DEFAULT_MODEL, MODEL_PRESETS, ensure_dirs
from glass_skull.experiment_store import safe_slug
from glass_skull.feature_store import compatible_features
from glass_skull.hf_access import HFTokenStatus
from glass_skull.hf_loader import build_hf_load_plan
from glass_skull.hf_registry import capabilities_for_backend, families, get_model, registry_as_dicts, visible_models
from glass_skull.lens import logit_lens_table
from glass_skull.llama_control import build_cvector_command, build_llama_server_command, classify_cvector_failure, preflight_control_vector_run, read_gguf_tensor_index, shell_join
from glass_skull import llama_client
from glass_skull.llama_client import chat_completion, normalize_base_url
from glass_skull.prompt_loader import load_jsonl, load_txt
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
    assert captured_requests[1]["payload"]["prompt"].startswith("System: Answer directly")

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
