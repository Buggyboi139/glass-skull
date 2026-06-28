from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ACTIVATION_PATCH_RECIPE_DIR, DATA_DIR


EXPERIMENT_DIR = DATA_DIR / "experiments"


def safe_slug(name: str) -> str:
    keep = []
    for ch in name.strip().replace(" ", "_"):
        if ch.isalnum() or ch in {"_", "-", "."}:
            keep.append(ch)
    slug = "".join(keep).strip("._-")
    return slug or "experiment"


def create_experiment_dir(name: str) -> Path:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = EXPERIMENT_DIR / f"{stamp}_{safe_slug(name)}"
    path = base
    suffix = 1
    while path.exists():
        suffix += 1
        path = Path(f"{base}_{suffix}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_run_artifact(path: Path, artifact: dict[str, Any]) -> None:
    target = path if path.name == "artifact.json" else path / "artifact.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, artifact)


def load_run_artifact(path: Path | str) -> dict[str, Any]:
    source = Path(path)
    if source.is_dir():
        source = source / "artifact.json"
    return json.loads(source.read_text(encoding="utf-8"))


def save_activation_patch_recipe(recipe: dict[str, Any] | str, name: str | dict[str, Any] | None = None) -> Path:
    ACTIVATION_PATCH_RECIPE_DIR.mkdir(parents=True, exist_ok=True)
    if isinstance(recipe, str) and isinstance(name, dict):
        recipe, name = name, recipe
    if not isinstance(recipe, dict):
        raise TypeError("recipe must be a dict")
    recipe_name = safe_slug(str(name) if isinstance(name, str) else str(recipe.get("name") or "activation_patch"))
    path = ACTIVATION_PATCH_RECIPE_DIR / f"{recipe_name}.json"
    write_json(path, recipe)
    return path


def load_activation_patch_recipe(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def list_activation_patch_recipes() -> list[dict[str, Any]]:
    ACTIVATION_PATCH_RECIPE_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for path in sorted(ACTIVATION_PATCH_RECIPE_DIR.glob("*.json")):
        try:
            payload = load_activation_patch_recipe(path)
            rows.append({
                "name": safe_slug(path.stem),
                "path": str(path),
                "recipe_name": payload.get("name", path.stem),
                "source_run_id": payload.get("source_run_id"),
                "target_run_id": payload.get("target_run_id"),
            })
        except Exception as exc:
            rows.append({"name": safe_slug(path.stem), "path": str(path), "error": str(exc)})
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_dataframe(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def list_experiments() -> list[dict[str, Any]]:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(EXPERIMENT_DIR.iterdir(), reverse=True):
        if not path.is_dir():
            continue
        summary_path = path / "summary.json"
        summary = {}
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary = {"error": "failed to read summary"}
        rows.append({"name": path.name, "path": str(path), **summary})
    return rows


def latest_run_artifacts(limit: int = 25) -> list[dict[str, Any]]:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for path in sorted(EXPERIMENT_DIR.iterdir(), reverse=True):
        if not path.is_dir():
            continue
        artifact_path = path / "artifact.json"
        if not artifact_path.exists():
            continue
        try:
            artifact = load_run_artifact(artifact_path)
            rows.append({
                "name": path.name,
                "path": str(path),
                "artifact_path": str(artifact_path),
                "run_id": artifact.get("run_id"),
                "mode": artifact.get("mode"),
                "backend": artifact.get("backend"),
                "model": artifact.get("model"),
                "created_at": artifact.get("created_at"),
                "summary": artifact.get("summary", {}),
            })
        except Exception as exc:
            rows.append({"name": path.name, "path": str(path), "error": str(exc)})
        if len(rows) >= limit:
            break
    return rows
