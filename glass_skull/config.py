from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"
FEATURE_DIR = DATA_DIR / "features"
LOG_DIR = DATA_DIR / "logs"
PROMPT_SET_DIR = DATA_DIR / "prompt_sets"
CONTROL_SET_DIR = DATA_DIR / "control_sets"
CONTROL_VECTOR_DIR = DATA_DIR / "control_vectors"
CHAT_DIR = DATA_DIR / "chats"
DB_PATH = LOG_DIR / "glass_skull.db"


# TransformerLens generally prefers short canonical model names for supported models.
DEFAULT_MODEL = "pythia-70m-deduped"
DEFAULT_DEVICE = "auto"

MODEL_PRESETS = [
    "pythia-70m-deduped",
    "pythia-160m-deduped",
    "pythia-410m-deduped",
    "pythia-1b-deduped",
]

MODEL_ALIASES = {
    "EleutherAI/pythia-70m-deduped": "pythia-70m-deduped",
    "EleutherAI/pythia-160m-deduped": "pythia-160m-deduped",
    "EleutherAI/pythia-410m-deduped": "pythia-410m-deduped",
    "EleutherAI/pythia-1b-deduped": "pythia-1b-deduped",
}


@dataclass(frozen=True)
class TraceConfig:
    model_name: str = DEFAULT_MODEL
    prompt: str = "The cat sat on the"
    max_new_tokens: int = 50
    temperature: float = 0.8
    top_k: int = 30
    layer: int = 0
    hook_name: str = "resid_post"


def normalize_model_name(model_name: str) -> str:
    model_name = model_name.strip()
    return MODEL_ALIASES.get(model_name, model_name)


def ensure_dirs() -> None:
    for path in [DATA_DIR, FEATURE_DIR, LOG_DIR, PROMPT_SET_DIR, CONTROL_SET_DIR, CONTROL_VECTOR_DIR, CHAT_DIR]:
        path.mkdir(parents=True, exist_ok=True)
