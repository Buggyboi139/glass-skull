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
ACTIVATION_PATCH_RECIPE_DIR = DATA_DIR / "activation_patch_recipes"
WORKSPACE_DIR = DATA_DIR / "workspaces"
GLOBAL_WORKSPACE_DIR = WORKSPACE_DIR / "global"
TAB_WORKSPACE_DIR = WORKSPACE_DIR / "tabs"
NODE_ANNOTATION_DIR = DATA_DIR / "node_annotations"
NODE_ANNOTATION_PATH = NODE_ANNOTATION_DIR / "annotations.json"
DB_PATH = LOG_DIR / "glass_skull.db"
DEFAULT_GGUF_MODEL_PATH = Path("/home/dsmason321/models/Best/Qwen3.6-35B-heretic-MTP-Q4_K_S.gguf")


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
