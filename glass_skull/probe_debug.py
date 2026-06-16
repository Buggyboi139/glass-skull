from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch
from transformer_lens import HookedTransformer

from .steering import generate_normal, generate_steered, validate_steering_vector
from .tracer import cache_get, hook_point


@dataclass
class ProbeDebugResult:
    backend: str
    trace_model: str
    layer: int
    stream: str
    strength: float
    vector_norm: float
    vector_max_abs: float
    vector_width: int
    expected_width: int
    hook_point: str
    prompt: str
    normal_output: str
    steered_output: str
    output_changed: bool
    output_prefix_same_chars: int
    activation_delta_norm: float
    activation_delta_max_abs: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def activation_delta_at_hook(
    model: HookedTransformer,
    prompt: str,
    vector: torch.Tensor,
    layer: int,
    stream: str,
    strength: float,
    token_position: int = -1,
) -> tuple[float, float]:
    expected = int(model.cfg.d_model)
    v = validate_steering_vector(vector, expected, context="probe debug").detach().float().cpu()
    hp = hook_point(layer, stream)

    _, normal_cache = model.run_with_cache(prompt, remove_batch_dim=False)
    normal = cache_get(normal_cache, hp).detach().float().cpu()

    steered = normal.clone()
    steered[:, token_position, :] += strength * v.to(dtype=steered.dtype)
    delta = steered - normal
    token_delta = delta[:, token_position, :]
    return float(token_delta.norm().item()), float(token_delta.abs().max().item())


def probe_debug_run(
    model: HookedTransformer,
    prompt: str,
    vector: torch.Tensor,
    layer: int,
    stream: str,
    strength: float,
    max_new_tokens: int = 80,
    temperature: float = 0.2,
    backend: str = "TransformerLens",
    token_position: int = -1,
) -> ProbeDebugResult:
    expected = int(model.cfg.d_model)
    v = validate_steering_vector(vector, expected, context="probe debug").detach().float().cpu().flatten()
    hp = hook_point(layer, stream)

    normal_output = generate_normal(model, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    steered_output = generate_steered(
        model,
        prompt,
        v,
        layer=layer,
        stream=stream,
        strength=strength,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        token_position=token_position,
    )
    delta_norm, delta_max_abs = activation_delta_at_hook(
        model,
        prompt,
        v,
        layer=layer,
        stream=stream,
        strength=strength,
        token_position=token_position,
    )

    changed = normal_output != steered_output
    prefix = common_prefix_len(normal_output, steered_output)
    if backend != "TransformerLens":
        message = "Probe vectors only affect TransformerLens generation. llama.cpp is chat/output-only until glass hooks exist."
    elif not changed and abs(strength) < 5:
        message = "Hook is live, but output did not change. Increase strength, use resid_post, or test a stronger contrast vector."
    elif not changed:
        message = "Hook applied a nonzero activation delta, but decoded text stayed identical. Try a different layer/stream or a sharper prompt."
    else:
        message = "Probe changed decoded output. Electrode is live."

    return ProbeDebugResult(
        backend=backend,
        trace_model=str(model.cfg.model_name),
        layer=int(layer),
        stream=str(stream),
        strength=float(strength),
        vector_norm=float(v.norm().item()),
        vector_max_abs=float(v.abs().max().item()) if v.numel() else 0.0,
        vector_width=int(v.numel()),
        expected_width=expected,
        hook_point=hp,
        prompt=prompt,
        normal_output=normal_output,
        steered_output=steered_output,
        output_changed=changed,
        output_prefix_same_chars=prefix,
        activation_delta_norm=delta_norm,
        activation_delta_max_abs=delta_max_abs,
        message=message,
    )
