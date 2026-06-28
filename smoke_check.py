from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from glass_skull.activation_map import build_activation_map_payload, build_model_meta
from glass_skull.activation_map_view import activation_map_html
from glass_skull.behavior_profiles import get_behavior_profile, list_behavior_profiles
from glass_skull.behavior_scoring import behavior_timeline_df, score_behavior_output, score_run_artifact
from glass_skull.chat_store import load_chat, save_chat
from glass_skull.config import CONTROL_SET_DIR, CONTROL_VECTOR_DIR, ensure_dirs
from glass_skull.experiment_store import safe_slug
from glass_skull.fuzzing import run_fuzz_experiment
from glass_skull.llama_client import (
    build_steering_metadata,
    chat_completion,
    normalize_base_url,
    per_request_steering_supported,
    trace_glass_prompt,
)
from glass_skull.llama_control import (
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
                    {"layer": 0, "stream": "resid_pre", "token_index": 0, "l2_norm": 1.25, "top_dims": [{"dimension": 3, "activation": 0.5}]},
                    {"layer": 1, "stream": "resid_pre", "token_index": 1, "l2_norm": 2.5, "top_dims": [{"dimension": 7, "activation": -1.0}]},
                ],
                "logits": {"supported": False, "reason": "not requested"},
            }
        if url.endswith("/v1/models"):
            return {"data": [{"id": "local", "status": "loaded"}]}
        if "/glass-skull/info" in url:
            return {"capabilities": {"steering": {"per_request": {"supported": True}}}}
        raise AssertionError(url)

    return request_json


def main() -> None:
    ensure_dirs()
    assert LOCAL_TABS == ["Run", "Map", "Steer", "Timeline", "Model", "Settings"]
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

    captured: list[dict[str, Any]] = []
    import glass_skull.llama_client as llama_client
    original_request_json = llama_client._request_json
    try:
        llama_client._request_json = _fake_llama_request(captured)
        output = chat_completion("http://local/v1", "hello", model_alias="local")
        assert output == "local reply"
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
    assert trace_request["payload"]["capture"]["prompt_tokens"] is True
    assert trace_request["payload"]["capture"]["layer_inputs"] is True
    assert trace_request["payload"]["max_tokens"] == 8
    assert per_request_steering_supported({"capabilities": {"steering": {"per_request": {"supported": True}}}}) is True
    assert build_steering_metadata("vec.gguf", 1.25, 1, 4)["layer_end"] == 4

    rows = normalize_llama_trace(trace, prompt_id=0, label="single", metadata={"run_id": "run1"})
    assert rows[0]["token"] == "hi"
    assert rows[1]["activation_norm"] == 2.5
    unavailable = trace_unavailable_row("run1", 2, "x", "llama.cpp", "not exposed")
    artifact = build_run_artifact(
        run_id="run1",
        mode="local",
        backend="llama.cpp",
        model="local",
        prompts=[
            {"prompt_id": 0, "label": "single", "prompt": "hello", "output": "clear concise action", "trace_rows": rows},
            {"prompt_id": 2, "label": "x", "prompt": "missing", "output": "", "trace_rows": [unavailable]},
        ],
    )
    path_df = activation_path_df(artifact)
    assert len(path_df) == 3
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
    html = activation_map_html(payload)
    assert "canvas" in html and "selectedDiagnostics" in html and "Activation path unavailable" not in html

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

    print("Glass Skull local-only smoke check passed.")


if __name__ == "__main__":
    main()
