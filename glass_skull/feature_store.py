from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .config import FEATURE_DIR


def _safe_name(name: str | None) -> str:
    if name is None:
        raise ValueError("Feature name cannot be None")
    cleaned = str(name).strip().replace(" ", "_")
    keep = []
    for ch in cleaned:
        if ch.isalnum() or ch in {"_", "-", "."}:
            keep.append(ch)
    safe = "".join(keep).strip("._-")
    if not safe:
        raise ValueError("Feature name cannot be empty")
    return safe


def feature_paths(name: str | None) -> tuple[Path, Path]:
    safe = _safe_name(name)
    return FEATURE_DIR / f"{safe}.pt", FEATURE_DIR / f"{safe}.json"


def tensor_vector_dim(tensor_path: Path) -> int | None:
    if not tensor_path.exists():
        return None
    try:
        tensor = torch.load(tensor_path, map_location="cpu")
        return int(tensor.detach().flatten().numel())
    except Exception:
        return None


def save_feature(name: str, vector: torch.Tensor, metadata: dict[str, Any]) -> tuple[Path, Path]:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    vector = vector.detach().cpu().flatten()
    tensor_path, meta_path = feature_paths(name)
    torch.save(vector, tensor_path)
    meta = dict(metadata)
    meta["name"] = name
    meta["tensor_file"] = tensor_path.name
    meta["vector_dim"] = int(vector.numel())
    meta["vector_shape"] = list(vector.shape)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return tensor_path, meta_path


def _first_existing_feature_name() -> str:
    features = list_features(include_missing=False)
    for item in features:
        if item.get("exists") and item.get("name"):
            return str(item["name"])
    raise FileNotFoundError("No loadable feature vectors were found in data/features")


def load_feature(name: str | None) -> tuple[torch.Tensor, dict[str, Any]]:
    # Streamlit can preserve a stale None in session_state for selectboxes.
    # If the UI says features exist but the selected value is None, load the first valid feature.
    if name is None:
        name = _first_existing_feature_name()

    tensor_path, meta_path = feature_paths(name)
    if not tensor_path.exists():
        raise FileNotFoundError(tensor_path)
    vector = torch.load(tensor_path, map_location="cpu").detach().flatten()
    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if "name" not in metadata:
        metadata["name"] = name
    metadata["vector_dim"] = int(vector.numel())
    metadata["vector_shape"] = list(vector.shape)
    return vector, metadata


def list_features(include_missing: bool = False) -> list[dict[str, Any]]:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for meta_path in sorted(FEATURE_DIR.glob("*.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                meta = {"name": meta_path.stem, "error": "metadata was not an object"}
        except Exception:
            meta = {"name": meta_path.stem, "error": "failed to read metadata"}

        name = meta.get("name") or meta_path.stem
        try:
            safe = _safe_name(str(name))
        except ValueError:
            safe = meta_path.stem
            name = meta_path.stem

        tensor_name = meta.get("tensor_file") or f"{safe}.pt"
        tensor_path = FEATURE_DIR / str(tensor_name)
        exists = tensor_path.exists()
        meta["name"] = str(name)
        meta["exists"] = exists
        if exists:
            meta["vector_dim"] = meta.get("vector_dim") or tensor_vector_dim(tensor_path)
        if include_missing or exists:
            rows.append(meta)
    return rows


def compatible_features(expected_dim: int) -> list[dict[str, Any]]:
    rows = []
    for item in list_features(include_missing=False):
        if item.get("vector_dim") == int(expected_dim):
            rows.append(item)
    return rows
