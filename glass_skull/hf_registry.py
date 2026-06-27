from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


ModelSource = Literal["transformerlens", "hf_transformers", "llama_cpp"]
AccessState = Literal["public", "gated", "pending_validation"]
TraceLevel = Literal["full", "partial", "none"]


@dataclass(frozen=True)
class HFModelSpec:
    display_name: str
    repo_id: str
    family: str
    organization: str
    params_b: float | None
    active_params_b: float | None
    context: str
    access: AccessState
    official: bool
    source: ModelSource
    trace_level: TraceLevel
    supports_chat: bool
    supports_attention: bool
    supports_logit_lens: bool
    supports_steering: bool
    supports_fuzz_trace: bool
    min_vram_gb: int | None
    recommended: bool
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


MODEL_REGISTRY: list[HFModelSpec] = [
    # TransformerLens-native baseline models. Ugly little lab rats, but they actually dissect cleanly.
    HFModelSpec("Pythia 70M", "pythia-70m-deduped", "Pythia", "EleutherAI", 0.07, None, "2k", "public", True, "transformerlens", "full", True, True, True, True, True, 1, True, "Fast smoke-test trace model."),
    HFModelSpec("Pythia 160M", "pythia-160m-deduped", "Pythia", "EleutherAI", 0.16, None, "2k", "public", True, "transformerlens", "full", True, True, True, True, True, 2, True, "Small full-trace model."),
    HFModelSpec("Pythia 410M", "pythia-410m-deduped", "Pythia", "EleutherAI", 0.41, None, "2k", "public", True, "transformerlens", "full", True, True, True, True, True, 4, True, "Better trace model, still realistic on CPU."),
    HFModelSpec("Pythia 1B", "pythia-1b-deduped", "Pythia", "EleutherAI", 1.0, None, "2k", "public", True, "transformerlens", "full", True, True, True, True, True, 8, False, "Useful but slower."),

    # Gemma 4 official Google family, <=70B total/active target.
    HFModelSpec("Gemma 4 E2B", "google/gemma-4-E2B", "Gemma", "google", 2.0, None, "128k", "public", True, "hf_transformers", "partial", True, False, False, False, False, 8, True, "Official Google Gemma 4 small model. Generic HF hooks only until adapter work."),
    HFModelSpec("Gemma 4 E2B IT", "google/gemma-4-E2B-it", "Gemma", "google", 2.0, None, "128k", "public", True, "hf_transformers", "partial", True, False, False, False, False, 8, True, "Instruction tuned official Gemma 4 E2B."),
    HFModelSpec("Gemma 4 E4B", "google/gemma-4-E4B", "Gemma", "google", 4.0, None, "128k", "public", True, "hf_transformers", "partial", True, False, False, False, False, 12, True, "Official Google Gemma 4 mid-small model."),
    HFModelSpec("Gemma 4 E4B IT", "google/gemma-4-E4B-it", "Gemma", "google", 4.0, None, "128k", "public", True, "hf_transformers", "partial", True, False, False, False, False, 12, True, "Instruction tuned official Gemma 4 E4B."),
    HFModelSpec("Gemma 4 12B", "google/gemma-4-12B", "Gemma", "google", 12.0, None, "256k", "public", True, "hf_transformers", "partial", True, False, False, False, False, 28, True, "Official Google Gemma 4 12B."),
    HFModelSpec("Gemma 4 12B IT", "google/gemma-4-12B-it", "Gemma", "google", 12.0, None, "256k", "public", True, "hf_transformers", "partial", True, False, False, False, False, 28, True, "Instruction tuned official Gemma 4 12B."),
    HFModelSpec("Gemma 4 26B A4B IT", "google/gemma-4-26B-A4B-it", "Gemma", "google", 26.0, 4.0, "256k", "public", True, "hf_transformers", "partial", True, False, False, False, False, 48, False, "Official Gemma 4 MoE-ish model; large local HF load."),
    HFModelSpec("Gemma 4 31B IT", "google/gemma-4-31B-it", "Gemma", "google", 31.0, None, "256k", "public", True, "hf_transformers", "partial", True, False, False, False, False, 64, False, "Official Gemma 4 31B; likely not fun on this box through HF."),

    # Qwen official Qwen3 family.
    HFModelSpec("Qwen3 0.6B", "Qwen/Qwen3-0.6B", "Qwen", "Qwen", 0.6, None, "varies", "public", True, "hf_transformers", "partial", True, False, False, False, False, 2, True, "Official Qwen3 small model."),
    HFModelSpec("Qwen3 1.7B", "Qwen/Qwen3-1.7B", "Qwen", "Qwen", 1.7, None, "varies", "public", True, "hf_transformers", "partial", True, False, False, False, False, 6, True, "Official Qwen3 small model."),
    HFModelSpec("Qwen3 4B", "Qwen/Qwen3-4B", "Qwen", "Qwen", 4.0, None, "varies", "public", True, "hf_transformers", "partial", True, False, False, False, False, 12, True, "Official Qwen3 4B."),
    HFModelSpec("Qwen3 8B", "Qwen/Qwen3-8B", "Qwen", "Qwen", 8.0, None, "varies", "public", True, "hf_transformers", "partial", True, False, False, False, False, 20, True, "Official Qwen3 8B."),
    HFModelSpec("Qwen3 14B", "Qwen/Qwen3-14B", "Qwen", "Qwen", 14.0, None, "varies", "public", True, "hf_transformers", "partial", True, False, False, False, False, 32, False, "Official Qwen3 14B."),
    HFModelSpec("Qwen3 30B A3B", "Qwen/Qwen3-30B-A3B", "Qwen", "Qwen", 30.0, 3.0, "varies", "public", True, "hf_transformers", "partial", True, False, False, False, False, 48, False, "Official Qwen3 MoE-style model; active params are low, total is not tiny."),
    HFModelSpec("Qwen3 32B", "Qwen/Qwen3-32B", "Qwen", "Qwen", 32.0, None, "varies", "public", True, "hf_transformers", "partial", True, False, False, False, False, 64, False, "Official Qwen3 32B."),
    HFModelSpec("Qwen3.6 27B", "Qwen/Qwen3.6-27B", "Qwen", "Qwen", 27.0, None, "unknown", "pending_validation", True, "hf_transformers", "partial", True, False, False, False, False, 56, False, "Visible only as pending until Hub access validates the official repo."),
    HFModelSpec("Qwen3.6 35B A3B", "Qwen/Qwen3.6-35B-A3B", "Qwen", "Qwen", 35.0, 3.0, "unknown", "pending_validation", True, "hf_transformers", "partial", True, False, False, False, False, 64, False, "Visible only as pending until Hub access validates the official repo."),

    # Meta Llama official, gated.
    HFModelSpec("Llama 3.2 1B Instruct", "meta-llama/Llama-3.2-1B-Instruct", "Llama", "meta-llama", 1.0, None, "128k", "gated", True, "hf_transformers", "partial", True, False, False, False, False, 4, True, "Official Meta Llama; requires accepted license/access."),
    HFModelSpec("Llama 3.2 3B Instruct", "meta-llama/Llama-3.2-3B-Instruct", "Llama", "meta-llama", 3.0, None, "128k", "gated", True, "hf_transformers", "partial", True, False, False, False, False, 10, True, "Official Meta Llama; requires accepted license/access."),
    HFModelSpec("Llama 3.1 8B Instruct", "meta-llama/Llama-3.1-8B-Instruct", "Llama", "meta-llama", 8.0, None, "128k", "gated", True, "hf_transformers", "partial", True, False, False, False, False, 20, True, "Official Meta Llama; requires accepted license/access."),
    HFModelSpec("Llama 3.1 70B Instruct", "meta-llama/Llama-3.1-70B-Instruct", "Llama", "meta-llama", 70.0, None, "128k", "gated", True, "hf_transformers", "partial", True, False, False, False, False, 150, False, "Official Meta Llama 70B; cloud/big GPU territory."),
    HFModelSpec("Llama 3.3 70B Instruct", "meta-llama/Llama-3.3-70B-Instruct", "Llama", "meta-llama", 70.0, None, "128k", "gated", True, "hf_transformers", "partial", True, False, False, False, False, 150, False, "Official Meta Llama 3.3 70B; gated and heavy."),

    # Mistral official.
    HFModelSpec("Mistral 7B Instruct v0.3", "mistralai/Mistral-7B-Instruct-v0.3", "Mistral", "mistralai", 7.0, None, "32k", "public", True, "hf_transformers", "partial", True, False, False, False, False, 18, True, "Official Mistral 7B instruct."),
    HFModelSpec("Mistral Small 24B 2501", "mistralai/Mistral-Small-24B-Instruct-2501", "Mistral", "mistralai", 24.0, None, "varies", "public", True, "hf_transformers", "partial", True, False, False, False, False, 48, False, "Official Mistral Small 24B."),
    HFModelSpec("Mistral Small 3.1 24B 2503", "mistralai/Mistral-Small-3.1-24B-Instruct-2503", "Mistral", "mistralai", 24.0, None, "varies", "public", True, "hf_transformers", "partial", True, False, False, False, False, 48, False, "Official Mistral Small 3.1."),
    HFModelSpec("Mistral Small 3.2 24B 2506", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "Mistral", "mistralai", 24.0, None, "varies", "public", True, "hf_transformers", "partial", True, False, False, False, False, 48, False, "Official Mistral Small 3.2."),

    # Microsoft Phi official.
    HFModelSpec("Phi 3.5 Mini Instruct", "microsoft/Phi-3.5-mini-instruct", "Phi", "microsoft", 3.8, None, "128k", "public", True, "hf_transformers", "partial", True, False, False, False, False, 10, True, "Official Microsoft Phi mini."),
    HFModelSpec("Phi 3.5 MoE Instruct", "microsoft/Phi-3.5-MoE-instruct", "Phi", "microsoft", 42.0, 6.6, "128k", "public", True, "hf_transformers", "partial", True, False, False, False, False, 64, False, "Official Phi MoE; not small despite active params."),
    HFModelSpec("Phi 4", "microsoft/Phi-4", "Phi", "microsoft", 14.0, None, "16k", "public", True, "hf_transformers", "partial", True, False, False, False, False, 32, True, "Official Microsoft Phi 4."),
    HFModelSpec("Phi 4 Mini Instruct", "microsoft/Phi-4-mini-instruct", "Phi", "microsoft", 3.8, None, "128k", "public", True, "hf_transformers", "partial", True, False, False, False, False, 10, True, "Official Phi 4 mini instruct."),
    HFModelSpec("Phi 4 Reasoning", "microsoft/Phi-4-reasoning", "Phi", "microsoft", 14.0, None, "varies", "public", True, "hf_transformers", "partial", True, False, False, False, False, 32, False, "Official Phi 4 reasoning."),
    HFModelSpec("Phi 4 Mini Reasoning", "microsoft/Phi-4-mini-reasoning", "Phi", "microsoft", 3.8, None, "varies", "public", True, "hf_transformers", "partial", True, False, False, False, False, 10, True, "Official Phi 4 mini reasoning."),

    # DeepSeek official distills / coder, 70B and below.
    HFModelSpec("DeepSeek Coder 33B Instruct", "deepseek-ai/deepseek-coder-33b-instruct", "DeepSeek", "deepseek-ai", 33.0, None, "16k", "public", True, "hf_transformers", "partial", True, False, False, False, False, 64, False, "Official DeepSeek coder model."),
    HFModelSpec("DeepSeek R1 Distill Qwen 1.5B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek", "deepseek-ai", 1.5, None, "varies", "public", True, "hf_transformers", "partial", True, False, False, False, False, 6, True, "Official DeepSeek R1 distill."),
    HFModelSpec("DeepSeek R1 Distill Qwen 7B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "DeepSeek", "deepseek-ai", 7.0, None, "varies", "public", True, "hf_transformers", "partial", True, False, False, False, False, 18, True, "Official DeepSeek R1 distill."),
    HFModelSpec("DeepSeek R1 Distill Qwen 14B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "DeepSeek", "deepseek-ai", 14.0, None, "varies", "public", True, "hf_transformers", "partial", True, False, False, False, False, 32, False, "Official DeepSeek R1 distill."),
    HFModelSpec("DeepSeek R1 Distill Qwen 32B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "DeepSeek", "deepseek-ai", 32.0, None, "varies", "public", True, "hf_transformers", "partial", True, False, False, False, False, 64, False, "Official DeepSeek R1 distill."),
    HFModelSpec("DeepSeek R1 Distill Llama 8B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "DeepSeek", "deepseek-ai", 8.0, None, "varies", "public", True, "hf_transformers", "partial", True, False, False, False, False, 20, True, "Official DeepSeek R1 distill."),
    HFModelSpec("DeepSeek R1 Distill Llama 70B", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "DeepSeek", "deepseek-ai", 70.0, None, "varies", "public", True, "hf_transformers", "partial", True, False, False, False, False, 150, False, "Official DeepSeek R1 distill; huge."),
]


def registry_as_dicts() -> list[dict]:
    return [m.to_dict() for m in MODEL_REGISTRY]


def families() -> list[str]:
    return sorted({m.family for m in MODEL_REGISTRY})


def get_model(repo_id: str) -> HFModelSpec | None:
    for model in MODEL_REGISTRY:
        if model.repo_id == repo_id:
            return model
    return None


def visible_models(family: str = "All", recommended_only: bool = False) -> list[HFModelSpec]:
    rows = MODEL_REGISTRY
    if family != "All":
        rows = [m for m in rows if m.family == family]
    if recommended_only:
        rows = [m for m in rows if m.recommended]
    return rows


def capabilities_for_backend(backend: str, trace_source: str = "TransformerLens") -> dict[str, bool]:
    backend = backend.lower()
    is_llama = "llama.cpp" in backend
    return {
        "chat": True,
        "server_metadata": is_llama,
        "fuzz_outputs": True,
        "control_vector_steering": is_llama,
        "activation_trace": not is_llama and trace_source == "TransformerLens",
        "logit_lens": not is_llama and trace_source == "TransformerLens",
        "attention": not is_llama and trace_source == "TransformerLens",
        "activation_steering": not is_llama and trace_source == "TransformerLens",
        "activation_compare": not is_llama and trace_source == "TransformerLens",
        "feature_mapping": not is_llama and trace_source == "TransformerLens",
    }


def model_state(model: HFModelSpec, hf_token_valid: bool, model_access: str | None = None) -> tuple[str, str, bool]:
    """Return (state, reason, load_enabled)."""
    if model.source == "transformerlens":
        return "available", "TransformerLens-native full trace model.", True

    if model.access == "pending_validation":
        if model_access == "ok":
            return "available", "Official repo validated through the Hub.", True
        return "pending", "Pending Hub validation. Visible, but disabled until the repo validates.", False

    if model.access == "gated":
        if not hf_token_valid:
            return "needs_token", "Requires a valid Hugging Face token and accepted model license.", False
        if model_access == "ok":
            return "available", "Token valid and model access confirmed.", True
        if model_access in {"missing", "forbidden", "unauthorized"}:
            return "needs_access", "Token is valid, but this account does not have access to the selected model.", False
        return "needs_access_check", "Token valid. Check model access before loading.", False

    if model.access == "public":
        if model_access in {"missing", "forbidden", "unauthorized"}:
            return "unavailable", "Hub access check failed for this repo.", False
        return "available", "Public official model. Token optional unless your environment needs one.", True

    return "unknown", "Unknown registry state.", False
