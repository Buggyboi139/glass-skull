from __future__ import annotations

import os
from pathlib import Path


GLASS_SKULL_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_LLAMA_CPP_COMMIT = "73618f27a801c0b8614ceaf3547d3c2a99baae14"
MANAGED_LLAMA_CPP_DIR = Path(
    os.environ.get("GLASS_SKULL_LLAMA_CPP_DIR", GLASS_SKULL_ROOT / "managed" / "llama.cpp-glass")
).expanduser()
MANAGED_LLAMA_BUILD_DIR = Path(
    os.environ.get("GLASS_SKULL_LLAMA_BUILD_DIR", MANAGED_LLAMA_CPP_DIR / "build")
).expanduser()

DEFAULT_CVECTOR_GENERATOR = Path(
    os.environ.get("GLASS_SKULL_CVECTOR_GENERATOR", MANAGED_LLAMA_BUILD_DIR / "bin" / "llama-cvector-generator")
).expanduser()
DEFAULT_LLAMA_SERVER = Path(
    os.environ.get("GLASS_SKULL_LLAMA_SERVER", MANAGED_LLAMA_BUILD_DIR / "bin" / "llama-server")
).expanduser()
