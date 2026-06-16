from __future__ import annotations

import torch
import pandas as pd
from transformer_lens import HookedTransformer

from .tracer import hook_point


def get_weight_matrix(model: HookedTransformer, layer: int, module: str) -> torch.Tensor:
    """Return selected matrix in a consistent [input_dim, output_dim]-ish orientation where possible.

    v0.4 starts with MLP contribution views because they are much easier to explain than attention.
    """
    block = model.blocks[layer]

    if module == "mlp.W_in":
        return block.mlp.W_in.detach().float().cpu()
    if module == "mlp.W_out":
        return block.mlp.W_out.detach().float().cpu()

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
    We do not draw every edge. We draw the highest absolute contributions.
    """
    if module == "mlp.W_in":
        source_hp = hook_point(layer, "resid_mid")
    elif module == "mlp.W_out":
        # TransformerLens exposes post-activation MLP hidden values as hook_post for most supported models.
        source_hp = f"blocks.{layer}.mlp.hook_post"
        if source_hp not in cache:
            # Fallback: this is less direct, but keeps the UI from exploding while we improve architecture coverage.
            source_hp = hook_point(layer, "mlp_out")
    else:
        raise ValueError("Unsupported module for v0.4. Use one of: mlp.W_in, mlp.W_out")

    if source_hp not in cache:
        raise KeyError(f"Source activation not found in cache: {source_hp}")

    x = cache[source_hp][0, token_index].detach().float().cpu().flatten()
    W = get_weight_matrix(model, layer, module)

    # Ensure W is 2D.
    if W.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {tuple(W.shape)}")

    # Match source dimension to matrix orientation.
    if x.numel() == W.shape[0]:
        W_use = W
    elif x.numel() == W.shape[1]:
        W_use = W.T
    else:
        raise ValueError(
            f"Source activation dim {x.numel()} does not match matrix shape {tuple(W.shape)}"
        )

    # Limit computation by first selecting the strongest source activations and strongest output columns.
    source_k = min(max_inputs, x.numel())
    source_vals, source_idxs = torch.topk(x.abs(), k=source_k)
    x_small = x[source_idxs]
    W_small = W_use[source_idxs, :]

    out_strength = (x_small[:, None] * W_small).abs().sum(dim=0)
    output_k = min(max_outputs, out_strength.numel())
    _, output_idxs = torch.topk(out_strength, k=output_k)

    contrib = x_small[:, None] * W_small[:, output_idxs]
    flat = contrib.abs().flatten()
    k = min(top_k, flat.numel())
    vals, flat_idxs = torch.topk(flat, k=k)

    rows = []
    out_count = output_idxs.numel()
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
            "token_index": token_index,
            "from_dim": from_dim,
            "to_dim": to_dim,
            "input_activation": input_activation,
            "weight": weight,
            "contribution": contribution,
            "abs_contribution": abs(contribution),
        })
    return pd.DataFrame(rows)
