from __future__ import annotations

from typing import Any

import pandas as pd

from .tracer import cache_get


def attention_pattern_table(cache: Any, layer: int, head: int, tokens: list[str]) -> pd.DataFrame:
    hp = f"blocks.{int(layer)}.attn.hook_pattern"
    pattern = cache_get(cache, hp).detach().float().cpu()
    if pattern.ndim != 4:
        raise ValueError(f"Expected attention pattern [batch, head, dest, src], got {tuple(pattern.shape)}")

    n_heads = pattern.shape[1]
    if head < 0 or head >= n_heads:
        raise ValueError(f"Head {head} out of range. Model/cache has {n_heads} heads.")

    mat = pattern[0, int(head)]
    rows = []
    for dest in range(mat.shape[0]):
        for src in range(mat.shape[1]):
            rows.append({
                "dest_index": dest,
                "src_index": src,
                "dest_token": tokens[dest] if dest < len(tokens) else str(dest),
                "src_token": tokens[src] if src < len(tokens) else str(src),
                "attention": float(mat[dest, src].item()),
            })
    return pd.DataFrame(rows)


def top_attention_links(cache: Any, layer: int, head: int, tokens: list[str], top_k: int = 25) -> pd.DataFrame:
    table = attention_pattern_table(cache, layer, head, tokens)
    if table.empty:
        return table
    return table.sort_values("attention", ascending=False).head(top_k).reset_index(drop=True)
