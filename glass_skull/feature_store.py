from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .config import FEATURE_DIR


def _safe_name(name: str) -> str:
    cleaned = name.strip().replace(" ", "_")
    keep = []
    for ch in cleaned:
        if ch.isalnum() or ch in {"_", "-", "."}:
            keep.append(ch)
    safe = "".join(keep).strip("._-")
    if not safe:
        raise ValueError("Feature name cannot be empty")
    return safe


def feature_paths(name: str) -> tuple[Path, Path]:
    safe = _safe_name(name)
    return FEATURE_DIR / f"{safe}.pt", FEATURE_DIR / f"{safe}.json"


def save_feature(name: str, vector: torch.Tensor, metadata: dict[str, Any]) -> tuple[Path, Path]:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    tensor_path, meta_path = feature_paths(name)
    torch.save(vector.detach().cpu(), tensor_path)
    meta = dict(metadata)
    meta["name"] = name
    meta["tensor_file"] = tensor_path.name
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return tensor_path, meta_path


def load_feature(name: str) -> tuple[torch.Tensor, dict[str, Any]]:
    tensor_path, meta_path = feature_paths(name)
    if not tensor_path.exists():
        raise FileNotFoundError(tensor_path)
    vector = torch.load(tensor_path, map_location="cpu")
    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return vector, metadata


def list_features() -> list[dict[str, Any]]:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for meta_path in sorted(FEATURE_DIR.glob("*.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {"name": meta_path.stem, "error": "failed to read metadata"}
        tensor_path = FEATURE_DIR / meta.get("tensor_file", f"{meta_path.stem}.pt")
        meta["exists"] = tensor_path.exists()
        rows.append(meta)
    return rows
