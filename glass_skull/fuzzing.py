from __future__ import annotations

import time
from typing import Any, Callable

import pandas as pd

from .experiment_store import append_jsonl, create_experiment_dir, write_dataframe, write_json, write_run_artifact
from .llama_client import chat_completion, trace_glass_prompt
from .prompt_loader import PromptItem, prompt_items_to_records
from .run_artifacts import (
    activation_path_df,
    batch_heatmap_df,
    build_run_artifact,
    dimension_frequency_df,
    label_heatmap_df,
    llama_trace_unavailable_reason,
    normalize_llama_trace,
    trace_unavailable_row,
)


def run_fuzz_experiment(
    name: str,
    prompts: list[PromptItem],
    chat_backend: str,
    trace_enabled: bool,
    model: Any | None = None,
    llama_url: str | None = None,
    llama_model_alias: str | None = None,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    layers: list[int] | None = None,
    streams: list[str] | None = None,
    top_k: int = 32,
    include_vectors: bool = True,
    run_id: str | None = None,
    mode: str = "fuzz",
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    if chat_backend != "llama.cpp":
        raise ValueError("Glass Skull is local-only; batch runs require chat_backend='llama.cpp'")
    if not llama_url:
        raise ValueError("llama.cpp backend requires a server URL")

    layers = layers or [0]
    streams = streams or ["resid_pre"]

    exp_dir = create_experiment_dir(name)
    config = {
        "name": name,
        "chat_backend": "llama.cpp",
        "llama_url": llama_url,
        "llama_model_alias": llama_model_alias,
        "trace_enabled": trace_enabled,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "layers": layers,
        "streams": streams,
        "top_k": top_k,
        "include_vectors": include_vectors,
        "prompt_count": len(prompts),
        "run_id": run_id,
        "mode": mode,
    }
    write_json(exp_dir / "config.json", config)
    write_json(exp_dir / "prompts.json", prompt_items_to_records(prompts))

    records: list[dict[str, Any]] = []
    artifact_prompts: list[dict[str, Any]] = []
    outputs_path = exp_dir / "outputs.jsonl"

    for idx, item in enumerate(prompts, start=1):
        if progress_callback:
            progress_callback(idx, len(prompts), item.prompt)

        started = time.perf_counter()
        error = None
        output = ""
        trace_rows: list[dict[str, Any]] = []

        try:
            output = chat_completion(
                llama_url,
                item.prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                model_alias=llama_model_alias,
            )
        except Exception as exc:
            error = str(exc)

        if trace_enabled:
            try:
                payload = trace_glass_prompt(
                    llama_url,
                    item.prompt,
                    model_alias=llama_model_alias,
                    layers=layers,
                    streams=streams,
                    max_new_tokens=max_new_tokens,
                    top_k=top_k,
                    with_pieces=True,
                    include_vectors=include_vectors,
                )
                trace_rows = normalize_llama_trace(
                    payload,
                    prompt_id=item.prompt_id,
                    label=item.label,
                    metadata={"run_id": run_id, "mode": mode},
                )
                if not trace_rows:
                    trace_rows = [
                        trace_unavailable_row(
                            str(run_id or ""),
                            item.prompt_id,
                            item.label,
                            "llama.cpp",
                            llama_trace_unavailable_reason(payload),
                        )
                    ]
            except Exception as exc:
                error = f"{error} | trace: {exc}" if error else f"trace: {exc}"
                trace_rows = [trace_unavailable_row(str(run_id or ""), item.prompt_id, item.label, "llama.cpp", str(exc))]

        elapsed_ms = (time.perf_counter() - started) * 1000
        rec = {
            "prompt_id": item.prompt_id,
            "label": item.label,
            "prompt": item.prompt,
            "output": output,
            "error": error,
            "elapsed_ms": elapsed_ms,
            "trace_layers": _trace_layers_from_rows(trace_rows),
            "metadata": {**(item.metadata or {}), "run_id": run_id, "mode": mode},
        }
        records.append(rec)
        artifact_prompts.append({
            "prompt_id": item.prompt_id,
            "label": item.label,
            "prompt": item.prompt,
            "output": output,
            "error": error,
            "elapsed_ms": elapsed_ms,
            "metadata": rec["metadata"],
            "trace_rows": trace_rows,
        })
        append_jsonl(outputs_path, rec)

    artifact = build_run_artifact(
        run_id=run_id,
        mode=mode,
        backend="llama.cpp",
        model=llama_model_alias,
        prompts=artifact_prompts,
        summary={
            "experiment_path": str(exp_dir),
            "trace_enabled": trace_enabled,
            "layers": layers,
            "streams": streams,
        },
    )
    write_run_artifact(exp_dir, artifact)

    activation_df = activation_path_df(artifact)
    batch_prompt_df = batch_heatmap_df(artifact, group_by="prompt")
    batch_label_df = label_heatmap_df(artifact)
    batch_all_df = batch_heatmap_df(artifact, group_by="all")
    recurring_dims_df = dimension_frequency_df(artifact)

    if activation_df.empty:
        top_dims_df = pd.DataFrame()
        separation_df = pd.DataFrame()
    else:
        top_dims_df = recurring_dims_df
        separation_df = pd.DataFrame()

    for filename, df in [
        ("activation_path.csv", activation_df),
        ("batch_prompt_heatmap.csv", batch_prompt_df),
        ("batch_label_heatmap.csv", batch_label_df),
        ("batch_all_heatmap.csv", batch_all_df),
        ("recurring_dimensions.csv", recurring_dims_df),
    ]:
        if not df.empty:
            write_dataframe(exp_dir / filename, df)

    summary = {
        "experiment_path": str(exp_dir),
        "prompt_count": len(prompts),
        "error_count": sum(1 for r in records if r.get("error")),
        "chat_backend": "llama.cpp",
        "trace_enabled": trace_enabled,
        "run_id": run_id,
        "mode": mode,
        "artifact_path": str(exp_dir / "artifact.json"),
        "trace_row_count": artifact["summary"].get("trace_row_count", 0),
        "trace_supported": artifact["summary"].get("trace_supported", False),
        "trace_unavailable_count": artifact["summary"].get("trace_unavailable_count", 0),
    }
    write_json(exp_dir / "summary.json", summary)
    return {
        "summary": summary,
        "records": records,
        "top_dims_df": top_dims_df,
        "separation_df": separation_df,
        "artifact": artifact,
        "activation_path_df": activation_df,
        "batch_prompt_heatmap_df": batch_prompt_df,
        "batch_label_heatmap_df": batch_label_df,
        "batch_all_heatmap_df": batch_all_df,
        "recurring_dimensions_df": recurring_dims_df,
    }


def _trace_layers_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame([row for row in rows if row.get("trace_available", True)])
    if df.empty:
        return []
    grouped = df.groupby(["layer", "stream"], as_index=False)["activation_norm"].mean()
    return [
        {
            "layer": int(row["layer"]),
            "stream": str(row["stream"]),
            "norm": float(row["activation_norm"]),
            "top_dims": [],
        }
        for row in grouped.to_dict("records")
        if row.get("layer") is not None
    ]
