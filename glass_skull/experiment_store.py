from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DATA_DIR


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
