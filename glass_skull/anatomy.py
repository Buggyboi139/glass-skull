from __future__ import annotations

import pandas as pd
from transformer_lens import HookedTransformer


CORE_STREAMS = [
    "hook_embed",
    "hook_pos_embed",
    "blocks.{layer}.hook_resid_pre",
    "blocks.{layer}.ln1.hook_scale",
    "blocks.{layer}.attn.hook_q",
    "blocks.{layer}.attn.hook_k",
    "blocks.{layer}.attn.hook_v",
    "blocks.{layer}.attn.hook_attn_scores",
    "blocks.{layer}.attn.hook_pattern",
    "blocks.{layer}.attn.hook_z",
    "blocks.{layer}.hook_attn_out",
    "blocks.{layer}.hook_resid_mid",
    "blocks.{layer}.ln2.hook_scale",
    "blocks.{layer}.mlp.hook_pre",
    "blocks.{layer}.mlp.hook_post",
    "blocks.{layer}.hook_mlp_out",
    "blocks.{layer}.hook_resid_post",
    "ln_final.hook_scale",
]


def config_table(model: HookedTransformer) -> pd.DataFrame:
    cfg = model.cfg
    fields = [
        "model_name",
        "n_layers",
        "d_model",
        "n_heads",
        "d_head",
        "d_mlp",
        "d_vocab",
        "n_ctx",
        "act_fn",
        "normalization_type",
        "positional_embedding_type",
        "attention_dir",
    ]
    rows = []
    for field in fields:
        rows.append({"field": field, "value": str(getattr(cfg, field, None))})
    return pd.DataFrame(rows)


def parameter_table(model: HookedTransformer) -> pd.DataFrame:
    rows = []
    for name, param in model.named_parameters():
        rows.append({
            "name": name,
            "shape": tuple(param.shape),
            "parameters": int(param.numel()),
            "dtype": str(param.dtype),
            "device": str(param.device),
        })
    return pd.DataFrame(rows)


def hook_table(model: HookedTransformer) -> pd.DataFrame:
    rows = []
    hook_dict = getattr(model, "hook_dict", {})
    for name in sorted(hook_dict.keys()):
        rows.append({"hook_point": name})
    return pd.DataFrame(rows)


def expected_block_table(model: HookedTransformer) -> pd.DataFrame:
    rows = []
    hook_names = set(getattr(model, "hook_dict", {}).keys())
    for layer in range(model.cfg.n_layers):
        for template in CORE_STREAMS:
            if "{layer}" not in template:
                continue
            hook = template.format(layer=layer)
            rows.append({
                "layer": layer,
                "component": component_from_hook(hook),
                "hook_point": hook,
                "present": hook in hook_names,
            })
    return pd.DataFrame(rows)


def global_hook_table(model: HookedTransformer) -> pd.DataFrame:
    hook_names = set(getattr(model, "hook_dict", {}).keys())
    rows = []
    for hook in ["hook_embed", "hook_pos_embed", "ln_final.hook_scale"]:
        rows.append({
            "component": component_from_hook(hook),
            "hook_point": hook,
            "present": hook in hook_names,
        })
    return pd.DataFrame(rows)


def component_from_hook(hook: str) -> str:
    if "hook_embed" in hook:
        return "token embedding"
    if "hook_pos_embed" in hook:
        return "positional embedding"
    if "hook_resid_pre" in hook:
        return "residual stream before block"
    if ".ln1." in hook:
        return "attention layer norm"
    if ".attn.hook_q" in hook:
        return "attention query"
    if ".attn.hook_k" in hook:
        return "attention key"
    if ".attn.hook_v" in hook:
        return "attention value"
    if ".attn.hook_attn_scores" in hook:
        return "attention scores"
    if ".attn.hook_pattern" in hook:
        return "attention pattern"
    if ".attn.hook_z" in hook:
        return "attention head result"
    if "hook_attn_out" in hook:
        return "attention output"
    if "hook_resid_mid" in hook:
        return "residual stream after attention"
    if ".ln2." in hook:
        return "MLP layer norm"
    if ".mlp.hook_pre" in hook:
        return "MLP pre-activation"
    if ".mlp.hook_post" in hook:
        return "MLP post-activation"
    if "hook_mlp_out" in hook:
        return "MLP output"
    if "hook_resid_post" in hook:
        return "residual stream after block"
    if "ln_final" in hook:
        return "final layer norm"
    return "other"
