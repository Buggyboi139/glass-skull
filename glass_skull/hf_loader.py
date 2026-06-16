from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class HFLoadPlan:
    repo_id: str
    token_supplied: bool
    dtype: str
    device_map: str
    trust_remote_code: bool
    status: str
    notes: str


def build_hf_load_plan(
    repo_id: str,
    token: str | None = None,
    dtype: str = "auto",
    device_map: str = "auto",
    trust_remote_code: bool = False,
) -> HFLoadPlan:
    return HFLoadPlan(
        repo_id=repo_id,
        token_supplied=bool((token or "").strip()),
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        status="planned",
        notes="Generic HF loading scaffold. Full hook adapter is not wired into the cockpit yet.",
    )


def load_hf_causal_lm(
    repo_id: str,
    token: str | None = None,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    trust_remote_code: bool = False,
) -> tuple[Any, Any]:
    """Load a generic HF causal LM.

    This is intentionally separate from TransformerLens. Generic HF models can chat/generate,
    but they do not automatically expose the same Glass Skull hook map until family adapters exist.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - depends on optional deps
        raise RuntimeError("Generic HF loading requires transformers and torch to be installed") from exc

    dtype_obj: Any = torch_dtype
    if torch_dtype == "bfloat16":
        dtype_obj = torch.bfloat16
    elif torch_dtype == "float16":
        dtype_obj = torch.float16
    elif torch_dtype == "float32":
        dtype_obj = torch.float32

    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
    }
    if token:
        kwargs["token"] = token

    tokenizer = AutoTokenizer.from_pretrained(repo_id, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        torch_dtype=dtype_obj,
        device_map=device_map,
        **kwargs,
    )
    return tokenizer, model
