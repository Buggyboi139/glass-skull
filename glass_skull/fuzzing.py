from __future__ import annotations

import time
from typing import Any, Callable

import pandas as pd
from transformer_lens import HookedTransformer

from .experiment_store import append_jsonl, create_experiment_dir, write_dataframe, write_json, write_run_artifact
from .llama_client import chat_completion, trace_glass_prompt
from .prompt_loader import PromptItem, prompt_items_to_records
from .tracer import top_active_dimensions, trace_prompt
from .aggregation import label_layer_heatmap, label_separation_table, prompt_layer_heatmap, top_recurring_dimensions
from .run_artifacts import (
    activation_path_df,
    batch_heatmap_df,
    build_run_artifact,
    dimension_frequency_df,
    label_heatmap_df,
    llama_trace_unavailable_reason,
    normalize_llama_trace,
    normalize_transformerlens_trace,
    trace_unavailable_row,
)


def trace_layers_for_prompt(
    model: HookedTransformer,
    prompt: str,
    layers: list[int],
    streams: list[str],
    top_k: int,
) -> tuple[list[dict[str, Any]], Any]:
    trace = trace_prompt(model, prompt)
    rows: list[dict[str, Any]] = []
    for layer in layers:
        for stream in streams:
            try:
                dims = top_active_dimensions(trace.cache, int(layer), stream, token_index=len(trace.tokens) - 1, top_k=top_k)
            except Exception:
                continue
            layer_norm_rows = trace.layer_norms[
                (trace.layer_norms["layer"] == int(layer)) &
                (trace.layer_norms["stream"] == stream)
            ]
            norm = float(layer_norm_rows["norm"].mean()) if not layer_norm_rows.empty else 0.0
            rows.append({
                "layer": int(layer),
                "stream": stream,
                "norm": norm,
                "top_dims": dims.to_dict("records"),
            })
    return rows, trace


def run_fuzz_experiment(
    name: str,
    prompts: list[PromptItem],
    chat_backend: str,
    trace_enabled: bool,
    model: HookedTransformer | None = None,
    llama_url: str | None = None,
    llama_model_alias: str | None = None,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    layers: list[int] | None = None,
    streams: list[str] | None = None,
    top_k: int = 32,
    run_id: str | None = None,
    mode: str = "fuzz",
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    if chat_backend == "transformerlens" and model is None:
        raise ValueError("TransformerLens backend requires a loaded model")
    if chat_backend == "llama.cpp" and not llama_url:
        raise ValueError("llama.cpp backend requires a server URL")

    layers = layers or ([0] if model is None else list(range(model.cfg.n_layers)))
    streams = streams or ["resid_post"]

    exp_dir = create_experiment_dir(name)
    config = {
        "name": name,
        "chat_backend": chat_backend,
        "llama_url": llama_url,
        "llama_model_alias": llama_model_alias,
        "trace_enabled": trace_enabled,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "layers": layers,
        "streams": streams,
        "top_k": top_k,
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
        trace_layers: list[dict[str, Any]] = []
        trace_rows: list[dict[str, Any]] = []

        try:
            if chat_backend == "llama.cpp":
                output = chat_completion(
                    llama_url or "",
                    item.prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    model_alias=llama_model_alias,
                )
            else:
                output = model.generate(
                    item.prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    verbose=False,
                )
        except Exception as exc:
            error = str(exc)

        if trace_enabled and chat_backend == "llama.cpp":
            try:
                payload = trace_glass_prompt(
                    llama_url or "",
                    item.prompt,
                    model_alias=llama_model_alias,
                    layers=layers,
                    streams=streams,
                    max_new_tokens=max_new_tokens,
                    top_k=top_k,
                    with_pieces=True,
                )
                trace_rows = normalize_llama_trace(
                    payload,
                    prompt_id=item.prompt_id,
                    label=item.label,
                    metadata={"run_id": run_id, "mode": mode},
                )
                if trace_rows:
                    trace_layers = _trace_layers_from_rows(trace_rows)
                else:
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
                if error:
                    error = error + " | trace: " + str(exc)
                else:
                    error = "trace: " + str(exc)
                trace_rows = [trace_unavailable_row(str(run_id or ""), item.prompt_id, item.label, "llama.cpp", str(exc))]
        elif trace_enabled and model is not None:
            try:
                trace_layers, trace = trace_layers_for_prompt(model, item.prompt, layers, streams, top_k)
                trace_rows = normalize_transformerlens_trace(
                    trace,
                    prompt_id=item.prompt_id,
                    label=item.label,
                    metadata={"run_id": run_id, "mode": mode},
                )
            except Exception as exc:
                if error:
                    error = error + " | trace: " + str(exc)
                else:
                    error = "trace: " + str(exc)
                trace_rows = [trace_unavailable_row(str(run_id or ""), item.prompt_id, item.label, "transformerlens", str(exc))]

        elapsed_ms = (time.perf_counter() - started) * 1000
        rec = {
            "prompt_id": item.prompt_id,
            "label": item.label,
            "prompt": item.prompt,
            "output": output,
            "error": error,
            "elapsed_ms": elapsed_ms,
            "trace_layers": trace_layers,
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
        backend=chat_backend,
        model=llama_model_alias if chat_backend == "llama.cpp" else getattr(getattr(model, "cfg", None), "model_name", None),
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

    prompt_df = prompt_layer_heatmap(records)
    label_df = label_layer_heatmap(records)
    top_dims_df = top_recurring_dimensions(records)
    separation_df = label_separation_table(records)

    if not prompt_df.empty:
        write_dataframe(exp_dir / "prompt_layer_heatmap.csv", prompt_df)
    if not label_df.empty:
        write_dataframe(exp_dir / "label_layer_heatmap.csv", label_df)
    if not top_dims_df.empty:
        write_dataframe(exp_dir / "top_recurring_dimensions.csv", top_dims_df)
    if not separation_df.empty:
        write_dataframe(exp_dir / "label_separation.csv", separation_df)
    if not activation_df.empty:
        write_dataframe(exp_dir / "activation_path.csv", activation_df)
    if not batch_prompt_df.empty:
        write_dataframe(exp_dir / "batch_prompt_heatmap.csv", batch_prompt_df)
    if not batch_label_df.empty:
        write_dataframe(exp_dir / "batch_label_heatmap.csv", batch_label_df)
    if not batch_all_df.empty:
        write_dataframe(exp_dir / "batch_all_heatmap.csv", batch_all_df)
    if not recurring_dims_df.empty:
        write_dataframe(exp_dir / "recurring_dimensions.csv", recurring_dims_df)

    summary = {
        "experiment_path": str(exp_dir),
        "prompt_count": len(prompts),
        "error_count": sum(1 for r in records if r.get("error")),
        "chat_backend": chat_backend,
        "trace_enabled": trace_enabled,
        "run_id": run_id,
        "mode": mode,
        "artifact_path": str(exp_dir / "artifact.json"),
        "trace_supported": artifact["summary"].get("trace_supported", False),
        "trace_unavailable_count": artifact["summary"].get("trace_unavailable_count", 0),
    }
    write_json(exp_dir / "summary.json", summary)
    return {
        "summary": summary,
        "records": records,
        "prompt_layer_df": prompt_df,
        "label_layer_df": label_df,
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
