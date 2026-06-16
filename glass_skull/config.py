from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"
FEATURE_DIR = DATA_DIR / "features"
LOG_DIR = DATA_DIR / "logs"
PROMPT_SET_DIR = DATA_DIR / "prompt_sets"
DB_PATH = LOG_DIR / "glass_skull.db"


DEFAULT_MODEL = "EleutherAI/pythia-70m-deduped"
DEFAULT_DEVICE = "auto"


@dataclass(frozen=True)
class TraceConfig:
    model_name: str = DEFAULT_MODEL
    prompt: str = "The cat sat on the"
    max_new_tokens: int = 50
    temperature: float = 0.8
    top_k: int = 30
    layer: int = 0
    hook_name: str = "resid_post"


def ensure_dirs() -> None:
    for path in [DATA_DIR, FEATURE_DIR, LOG_DIR, PROMPT_SET_DIR]:
        path.mkdir(parents=True, exist_ok=True)
