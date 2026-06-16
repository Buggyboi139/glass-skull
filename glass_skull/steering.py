from __future__ import annotations

from typing import Callable

import torch
from transformer_lens import HookedTransformer

from .tracer import cache_get, hook_point


def build_contrast_vector(
    model: HookedTransformer,
    positive_prompts: list[str],
    negative_prompts: list[str],
    layer: int,
    stream: str = "resid_post",
    token_position: int = -1,
    normalize: bool = True,
) -> torch.Tensor:
    if not positive_prompts:
        raise ValueError("positive_prompts cannot be empty")
    if not negative_prompts:
        raise ValueError("negative_prompts cannot be empty")

    hp = hook_point(layer, stream)

    def mean_activation(prompts: list[str]) -> torch.Tensor:
        acts = []
        for prompt in prompts:
            _, cache = model.run_with_cache(prompt, remove_batch_dim=False)
            act = cache_get(cache, hp)[0, token_position].detach().float().cpu()
            acts.append(act)
        return torch.stack(acts, dim=0).mean(dim=0)

    pos = mean_activation(positive_prompts)
    neg = mean_activation(negative_prompts)
    vec = pos - neg

    if normalize:
        vec = vec / (vec.norm() + 1e-8)
    return vec


def make_steering_hook(vector: torch.Tensor, strength: float, token_position: int = -1) -> Callable:
    def hook_fn(activation: torch.Tensor, hook):
        steer = vector.to(device=activation.device, dtype=activation.dtype)
        if activation.ndim != 3:
            raise ValueError(f"Expected activation shape [batch, pos, d_model], got {tuple(activation.shape)}")
        activation[:, token_position, :] = activation[:, token_position, :] + strength * steer
        return activation

    return hook_fn


def generate_normal(
    model: HookedTransformer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
) -> str:
    return model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        verbose=False,
    )


def generate_steered(
    model: HookedTransformer,
    prompt: str,
    vector: torch.Tensor,
    layer: int,
    stream: str = "resid_post",
    strength: float = 1.0,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    token_position: int = -1,
) -> str:
    hp = hook_point(layer, stream)
    hook = make_steering_hook(vector, strength=strength, token_position=token_position)

    with model.hooks(fwd_hooks=[(hp, hook)]):
        return model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            verbose=False,
        )


def vector_summary(vector: torch.Tensor, top_k: int = 30) -> list[dict]:
    v = vector.detach().float().cpu()
    vals, idxs = torch.topk(v.abs(), k=min(top_k, v.numel()))
    rows = []
    for rank, (abs_value, idx) in enumerate(zip(vals.tolist(), idxs.tolist()), start=1):
        idx = int(idx)
        rows.append({
            "rank": rank,
            "dimension": idx,
            "value": float(v[idx]),
            "abs_value": float(abs_value),
        })
    return rows
