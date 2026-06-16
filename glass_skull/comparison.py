from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from transformer_lens import HookedTransformer

from .steering import generate_normal, generate_steered, make_steering_hook, validate_steering_vector
from .tracer import TraceResult, activation_norm_table, hook_point, trace_prompt


DIFF_COLUMNS = [
    "layer",
    "stream",
    "token_index",
    "token",
    "normal_norm",
    "steered_norm",
    "delta",
    "abs_delta",
]


@dataclass
class ComparisonResult:
    prompt: str
    normal_output: str
    steered_output: str
    normal_trace: TraceResult
    steered_trace: TraceResult
    norm_diff: pd.DataFrame


def empty_diff() -> pd.DataFrame:
    return pd.DataFrame(columns=DIFF_COLUMNS)


def trace_prompt_steered(
    model: HookedTransformer,
    prompt: str,
    vector: torch.Tensor,
    layer: int,
    stream: str,
    strength: float,
    token_position: int = -1,
) -> TraceResult:
    validate_steering_vector(vector, int(model.cfg.d_model), context="feature")
    hp = hook_point(layer, stream)
    hook = make_steering_hook(vector, strength=strength, token_position=token_position)
    tokens = model.to_str_tokens(prompt)
    with model.hooks(fwd_hooks=[(hp, hook)]):
        logits, cache = model.run_with_cache(prompt, remove_batch_dim=False)
    layer_norms = activation_norm_table(model, cache, tokens)
    return TraceResult(prompt=prompt, tokens=tokens, logits=logits.detach().cpu(), cache=cache, layer_norms=layer_norms)


def compare_normal_vs_steered(
    model: HookedTransformer,
    prompt: str,
    vector: torch.Tensor,
    layer: int,
    stream: str,
    strength: float,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
) -> ComparisonResult:
    normal_trace = trace_prompt(model, prompt)
    steered_trace = trace_prompt_steered(model, prompt, vector, layer, stream, strength)
    normal_output = generate_normal(model, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    steered_output = generate_steered(
        model,
        prompt,
        vector,
        layer,
        stream=stream,
        strength=strength,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    diff = activation_norm_diff(normal_trace.layer_norms, steered_trace.layer_norms)
    return ComparisonResult(
        prompt=prompt,
        normal_output=normal_output,
        steered_output=steered_output,
        normal_trace=normal_trace,
        steered_trace=steered_trace,
        norm_diff=diff,
    )


def activation_norm_diff(normal: pd.DataFrame, steered: pd.DataFrame) -> pd.DataFrame:
    if normal.empty or steered.empty:
        return empty_diff()
    required = {"layer", "stream", "token_index", "norm"}
    if not required.issubset(normal.columns) or not required.issubset(steered.columns):
        return empty_diff()

    n = normal.rename(columns={"norm": "normal_norm"})
    s = steered.rename(columns={"norm": "steered_norm"})
    merged = n.merge(
        s[["layer", "stream", "token_index", "steered_norm"]],
        on=["layer", "stream", "token_index"],
        how="inner",
    )
    if merged.empty:
        return empty_diff()
    merged["delta"] = merged["steered_norm"] - merged["normal_norm"]
    merged["abs_delta"] = merged["delta"].abs()
    for col in DIFF_COLUMNS:
        if col not in merged.columns:
            merged[col] = None
    return merged[DIFF_COLUMNS]
