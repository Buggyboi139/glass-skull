from __future__ import annotations

from pathlib import Path


APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"
LOG_DIR = DATA_DIR / "logs"
PROMPT_SET_DIR = DATA_DIR / "prompt_sets"
CONTROL_SET_DIR = DATA_DIR / "control_sets"
CONTROL_VECTOR_DIR = DATA_DIR / "control_vectors"
CHAT_DIR = DATA_DIR / "chats"
EXPERIMENT_DIR = DATA_DIR / "experiments"
DB_PATH = LOG_DIR / "glass_skull.db"


def ensure_dirs() -> None:
    for path in [DATA_DIR, LOG_DIR, PROMPT_SET_DIR, CONTROL_SET_DIR, CONTROL_VECTOR_DIR, CHAT_DIR, EXPERIMENT_DIR]:
        path.mkdir(parents=True, exist_ok=True)
