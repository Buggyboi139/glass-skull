from __future__ import annotations

from pathlib import Path
from typing import MutableMapping, Any


APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"
LOG_DIR = DATA_DIR / "logs"
PROMPT_SET_DIR = DATA_DIR / "prompt_sets"
CONTROL_SET_DIR = DATA_DIR / "control_sets"
CONTROL_VECTOR_DIR = DATA_DIR / "control_vectors"
CHAT_DIR = DATA_DIR / "chats"
EXPERIMENT_DIR = DATA_DIR / "experiments"
ACTIVATION_PATCH_RECIPE_DIR = DATA_DIR / "activation_patch_recipes"
WORKSPACE_DIR = DATA_DIR / "workspaces"
GLOBAL_WORKSPACE_DIR = WORKSPACE_DIR / "global"
TAB_WORKSPACE_DIR = WORKSPACE_DIR / "tabs"
NODE_ANNOTATION_DIR = DATA_DIR / "node_annotations"
NODE_ANNOTATION_PATH = NODE_ANNOTATION_DIR / "annotations.json"
DB_PATH = LOG_DIR / "glass_skull.db"
DEFAULT_GGUF_MODEL_PATH = Path("/home/dsmason321/models/Best/Qwen3.6-35B-heretic-MTP-Q4_K_S.gguf")
DEFAULT_BATCH_MESSAGES = "\n".join([
    "What is the capital of Mongolia?",
    "Name one mammal that can glide without powered flight.",
    "Why do leaves change color in autumn?",
    "Convert 98.6°F to Celsius.",
    "What is the primary purpose of DNS?",
    "Who painted the ceiling of the Sistine Chapel?",
    "What does the acronym SQL stand for?",
    "How many moons does Mars have?",
    "What is the largest organ in the human body?",
    'Define the term "opportunity cost" in one sentence.',
    "Which programming language introduced the `async` and `await` keywords first?",
    "What causes a rainbow to appear?",
    "Name one advantage of using a hash table.",
    "What year did the first human land on the Moon?",
    "Explain the difference between RAM and SSD in one sentence.",
])


def seed_missing_defaults(state: MutableMapping[str, Any], defaults: dict[str, Any]) -> None:
    for key, value in defaults.items():
        if key not in state or (key == "llama_model_path" and not state.get(key)):
            state[key] = value


def ensure_dirs() -> None:
    for path in [
        DATA_DIR,
        LOG_DIR,
        PROMPT_SET_DIR,
        CONTROL_SET_DIR,
        CONTROL_VECTOR_DIR,
        CHAT_DIR,
        EXPERIMENT_DIR,
        ACTIVATION_PATCH_RECIPE_DIR,
        GLOBAL_WORKSPACE_DIR,
        TAB_WORKSPACE_DIR,
        NODE_ANNOTATION_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
