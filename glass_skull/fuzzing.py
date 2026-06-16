from __future__ import annotations

import time
from typing import Any, Callable

from transformer_lens import HookedTransformer

from .experiment_store import append_jsonl, create_experiment_dir, write_dataframe, write_json
from .llama_client import chat_completion
from .prompt_loader import PromptItem, prompt_items_to_records
from .tracer import top_active_dimensions, trace_prompt
from .aggregation import label_layer_heatmap, prompt_layer_heatmap, top_recurring_dimensions


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
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    layers: list[int] | None = None,
    streams: list[str] | None = None,
    top_k: int = 32,
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
        "trace_enabled": trace_enabled,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "layers": layers,
        "streams": streams,
        "top_k": top_k,
        "prompt_count": len(prompts),
    }
    write_json(exp_dir / "config.json", config)
    write_json(exp_dir / "prompts.json", prompt_items_to_records(prompts))

    records: list[dict[str, Any]] = []
    outputs_path = exp_dir / "outputs.jsonl"

    for idx, item in enumerate(prompts, start=1):
        if progress_callback:
            progress_callback(idx, len(prompts), item.prompt)

        started = time.perf_counter()
        error = None
        output = ""
        trace_layers: list[dict[str, Any]] = []

        try:
            if chat_backend == "llama.cpp":
                output = chat_completion(
                    llama_url or "",
                    item.prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
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

        if trace_enabled and model is not None:
            try:
                trace_layers, _ = trace_layers_for_prompt(model, item.prompt, layers, streams, top_k)
            except Exception as exc:
                if error:
                    error = error + " | trace: " + str(exc)
                else:
                    error = "trace: " + str(exc)

        elapsed_ms = (time.perf_counter() - started) * 1000
        rec = {
            "prompt_id": item.prompt_id,
            "label": item.label,
            "prompt": item.prompt,
            "output": output,
            "error": error,
            "elapsed_ms": elapsed_ms,
            "trace_layers": trace_layers,
            "metadata": item.metadata or {},
        }
        records.append(rec)
        append_jsonl(outputs_path, rec)

    prompt_df = prompt_layer_heatmap(records)
    label_df = label_layer_heatmap(records)
    top_dims_df = top_recurring_dimensions(records)

    if not prompt_df.empty:
        write_dataframe(exp_dir / "prompt_layer_heatmap.csv", prompt_df)
    if not label_df.empty:
        write_dataframe(exp_dir / "label_layer_heatmap.csv", label_df)
    if not top_dims_df.empty:
        write_dataframe(exp_dir / "top_recurring_dimensions.csv", top_dims_df)

    summary = {
        "experiment_path": str(exp_dir),
        "prompt_count": len(prompts),
        "error_count": sum(1 for r in records if r.get("error")),
        "chat_backend": chat_backend,
        "trace_enabled": trace_enabled,
    }
    write_json(exp_dir / "summary.json", summary)
    return {
        "summary": summary,
        "records": records,
        "prompt_layer_df": prompt_df,
        "label_layer_df": label_df,
        "top_dims_df": top_dims_df,
    }
