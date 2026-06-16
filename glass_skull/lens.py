from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from transformer_lens import HookedTransformer

from .tracer import cache_get, hook_point


def _apply_final_norm(model: HookedTransformer, resid: torch.Tensor) -> torch.Tensor:
    """Apply the model's final norm when available before unembedding.

    This is a simple Logit Lens, not a Tuned Lens. It is intentionally blunt but useful.
    """
    if hasattr(model, "ln_final") and model.ln_final is not None:
        return model.ln_final(resid)
    return resid


def logits_from_resid(model: HookedTransformer, resid: torch.Tensor) -> torch.Tensor:
    resid = resid.to(next(model.parameters()).device)
    resid = _apply_final_norm(model, resid)
    return model.unembed(resid)


def logit_lens_table(
    model: HookedTransformer,
    cache: Any,
    token_index: int = -1,
    stream: str = "resid_post",
    top_k: int = 5,
) -> pd.DataFrame:
    rows = []
    token_index = int(token_index)

    for layer in range(model.cfg.n_layers):
        hp = hook_point(layer, stream)
        try:
            resid = cache_get(cache, hp)[:, token_index:token_index + 1, :]
            logits = logits_from_resid(model, resid)[0, 0].detach().float().cpu()
        except Exception:
            continue

        probs = torch.softmax(logits, dim=-1)
        vals, idxs = torch.topk(probs, k=min(top_k, probs.numel()))
        for rank, (prob, tok_id) in enumerate(zip(vals.tolist(), idxs.tolist()), start=1):
            tok_id = int(tok_id)
            try:
                token = model.to_single_str_token(tok_id)
            except Exception:
                token = model.to_string(tok_id)
            rows.append({
                "layer": layer,
                "stream": stream,
                "rank": rank,
                "token_id": tok_id,
                "token": token,
                "probability": float(prob),
                "logit": float(logits[tok_id].item()),
            })
    return pd.DataFrame(rows)


def logit_lens_top_token_heatmap(
    model: HookedTransformer,
    cache: Any,
    tokens: list[str],
    stream: str = "resid_post",
) -> pd.DataFrame:
    rows = []
    for layer in range(model.cfg.n_layers):
        hp = hook_point(layer, stream)
        try:
            resid = cache_get(cache, hp)
            logits = logits_from_resid(model, resid)[0].detach().float().cpu()
        except Exception:
            continue

        probs = torch.softmax(logits, dim=-1)
        top_probs, top_ids = torch.max(probs, dim=-1)
        for pos, (prob, tok_id) in enumerate(zip(top_probs.tolist(), top_ids.tolist())):
            tok_id = int(tok_id)
            try:
                token = model.to_single_str_token(tok_id)
            except Exception:
                token = model.to_string(tok_id)
            rows.append({
                "layer": layer,
                "token_index": pos,
                "input_token": tokens[pos] if pos < len(tokens) else str(pos),
                "predicted_token": token,
                "probability": float(prob),
                "token_id": tok_id,
            })
    return pd.DataFrame(rows)
