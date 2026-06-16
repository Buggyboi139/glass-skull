from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import pandas as pd
from transformer_lens import HookedTransformer


@dataclass
class TraceResult:
    prompt: str
    tokens: list[str]
    logits: torch.Tensor
    cache: Any
    layer_norms: pd.DataFrame


def hook_point(layer: int, name: str) -> str:
    allowed = {
        "resid_pre": f"blocks.{layer}.hook_resid_pre",
        "resid_mid": f"blocks.{layer}.hook_resid_mid",
        "resid_post": f"blocks.{layer}.hook_resid_post",
        "attn_out": f"blocks.{layer}.hook_attn_out",
        "mlp_out": f"blocks.{layer}.hook_mlp_out",
    }
    if name not in allowed:
        raise ValueError(f"Unsupported hook name: {name}. Choose one of {sorted(allowed)}")
    return allowed[name]


def trace_prompt(model: HookedTransformer, prompt: str) -> TraceResult:
    tokens = model.to_str_tokens(prompt)
    logits, cache = model.run_with_cache(prompt, remove_batch_dim=False)
    layer_norms = activation_norm_table(model, cache, tokens)
    return TraceResult(prompt=prompt, tokens=tokens, logits=logits.detach().cpu(), cache=cache, layer_norms=layer_norms)


def activation_norm_table(model: HookedTransformer, cache: Any, tokens: list[str]) -> pd.DataFrame:
    rows = []
    for layer in range(model.cfg.n_layers):
        for stream in ["resid_pre", "attn_out", "mlp_out", "resid_post"]:
            name = hook_point(layer, stream)
            if name not in cache:
                continue
            act = cache[name][0].detach().float().cpu()  # [tokens, d_model]
            norms = act.norm(dim=-1)
            for pos, value in enumerate(norms.tolist()):
                rows.append({
                    "layer": layer,
                    "stream": stream,
                    "token_index": pos,
                    "token": tokens[pos] if pos < len(tokens) else str(pos),
                    "norm": value,
                })
    return pd.DataFrame(rows)


def top_active_dimensions(cache: Any, layer: int, stream: str, token_index: int, top_k: int = 30) -> pd.DataFrame:
    name = hook_point(layer, stream)
    if name not in cache:
        raise KeyError(f"Hook point not found in cache: {name}")

    act = cache[name][0, token_index].detach().float().cpu()
    k = min(top_k, act.numel())
    vals, idxs = torch.topk(act.abs(), k=k)

    rows = []
    for rank, (abs_value, idx) in enumerate(zip(vals.tolist(), idxs.tolist()), start=1):
        raw = act[idx].item()
        rows.append({
            "rank": rank,
            "dimension": idx,
            "activation": raw,
            "abs_activation": abs_value,
        })
    return pd.DataFrame(rows)


def next_token_table(model: HookedTransformer, logits: torch.Tensor, top_k: int = 20) -> pd.DataFrame:
    last_logits = logits[0, -1].detach().float()
    probs = torch.softmax(last_logits, dim=-1)
    vals, idxs = torch.topk(probs, k=top_k)
    rows = []
    for rank, (prob, tok_id) in enumerate(zip(vals.tolist(), idxs.tolist()), start=1):
        rows.append({
            "rank": rank,
            "token_id": tok_id,
            "token": model.to_string(tok_id),
            "probability": prob,
            "logit": last_logits[tok_id].item(),
        })
    return pd.DataFrame(rows)
