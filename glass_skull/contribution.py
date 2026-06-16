from __future__ import annotations

import pandas as pd
import torch
from transformer_lens import HookedTransformer

from .tracer import cache_get, cache_has, hook_point


def get_weight_matrix(model: HookedTransformer, layer: int, module: str) -> torch.Tensor:
    """Return selected matrix in a consistent [input_dim, output_dim] orientation where possible.

    v0.4 starts with MLP contribution views because they are easier to ground accurately than attention.
    """
    block = model.blocks[layer]

    if module == "mlp.W_in":
        return block.mlp.W_in.detach().float().cpu()
    if module == "mlp.W_out":
        return block.mlp.W_out.detach().float().cpu()

    raise ValueError("Unsupported module for v0.4. Use one of: mlp.W_in, mlp.W_out")


def source_hook_for_module(layer: int, module: str, cache) -> str:
    if module == "mlp.W_in":
        return hook_point(layer, "resid_mid")

    if module == "mlp.W_out":
        source_hp = f"blocks.{layer}.mlp.hook_post"
        if not cache_has(cache, source_hp):
            raise KeyError(
                "MLP post-activation hook was not found in the cache. "
                "Cannot compute mlp.W_out contribution edges accurately for this model/cache. "
                "Use mlp.W_in for now, or add hook_post support for this architecture."
            )
        return source_hp

    raise ValueError("Unsupported module for v0.4. Use one of: mlp.W_in, mlp.W_out")


def top_contribution_edges(
    model: HookedTransformer,
    cache,
    layer: int,
    module: str,
    token_index: int,
    top_k: int = 50,
    max_inputs: int = 256,
    max_outputs: int = 256,
) -> pd.DataFrame:
    """Compute top active contribution edges for one token/layer/module.

    For y = x @ W, an edge contribution is x[i] * W[i, j].
    We draw only the highest absolute contributions.
    """
    source_hp = source_hook_for_module(layer, module, cache)
    x = cache_get(cache, source_hp)[0, token_index].detach().float().cpu().flatten()
    W = get_weight_matrix(model, layer, module)

    if W.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {tuple(W.shape)}")

    if x.numel() == W.shape[0]:
        W_use = W
    elif x.numel() == W.shape[1]:
        W_use = W.T
    else:
        raise ValueError(
            f"Source activation dim {x.numel()} does not match matrix shape {tuple(W.shape)}"
        )

    source_k = min(max_inputs, x.numel())
    _, source_idxs = torch.topk(x.abs(), k=source_k)
    x_small = x[source_idxs]
    W_small = W_use[source_idxs, :]

    out_strength = (x_small[:, None] * W_small).abs().sum(dim=0)
    output_k = min(max_outputs, out_strength.numel())
    _, output_idxs = torch.topk(out_strength, k=output_k)

    contrib = x_small[:, None] * W_small[:, output_idxs]
    flat = contrib.abs().flatten()
    k = min(top_k, flat.numel())
    _, flat_idxs = torch.topk(flat, k=k)

    rows = []
    out_count = int(output_idxs.numel())
    for rank, flat_idx in enumerate(flat_idxs.tolist(), start=1):
        src_local = flat_idx // out_count
        out_local = flat_idx % out_count
        from_dim = int(source_idxs[src_local])
        to_dim = int(output_idxs[out_local])
        input_activation = float(x[from_dim])
        weight = float(W_use[from_dim, to_dim])
        contribution = input_activation * weight
        rows.append({
            "rank": rank,
            "layer": layer,
            "module": module,
            "source_hook": source_hp,
            "token_index": token_index,
            "from_dim": from_dim,
            "to_dim": to_dim,
            "input_activation": input_activation,
            "weight": weight,
            "contribution": contribution,
            "abs_contribution": abs(contribution),
        })
    return pd.DataFrame(rows)
