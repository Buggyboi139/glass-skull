from __future__ import annotations

import torch
import streamlit as st
from transformer_lens import HookedTransformer

from .config import normalize_model_name


@st.cache_resource(show_spinner=True)
def load_hooked_model(model_name: str, device_choice: str = "auto") -> HookedTransformer:
    """Load a TransformerLens HookedTransformer once per Streamlit session."""
    model_name = normalize_model_name(model_name)

    if device_choice == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_choice

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected, but torch.cuda.is_available() is False. Use cpu or fix PyTorch GPU support.")

    dtype = torch.float16 if device == "cuda" else torch.float32

    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=dtype,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    model.eval()
    return model


def model_summary(model: HookedTransformer) -> dict:
    cfg = model.cfg
    param_count = sum(p.numel() for p in model.parameters())
    first_param = next(model.parameters())
    return {
        "model_name": cfg.model_name,
        "device": str(first_param.device),
        "dtype": str(first_param.dtype),
        "parameters": param_count,
        "layers": cfg.n_layers,
        "d_model": cfg.d_model,
        "heads": cfg.n_heads,
        "d_head": cfg.d_head,
        "d_mlp": cfg.d_mlp,
        "vocab_size": cfg.d_vocab,
    }
