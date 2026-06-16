from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class PromptItem:
    prompt_id: int
    prompt: str
    label: str = "unlabeled"
    metadata: dict | None = None


def load_prompt_file_bytes(filename: str, data: bytes) -> list[PromptItem]:
    suffix = Path(filename).suffix.lower()
    text = data.decode("utf-8", errors="replace")
    if suffix == ".jsonl":
        return load_jsonl(text)
    if suffix == ".csv":
        return load_csv(text)
    return load_txt(text)


def load_txt(text: str) -> list[PromptItem]:
    rows = []
    for idx, line in enumerate(text.splitlines()):
        prompt = line.strip()
        if not prompt or prompt.startswith("#"):
            continue
        rows.append(PromptItem(prompt_id=len(rows), prompt=prompt))
    return rows


def load_jsonl(text: str) -> list[PromptItem]:
    rows = []
    for raw_idx, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        prompt = str(obj.get("prompt", "")).strip()
        if not prompt:
            continue
        label = str(obj.get("label", "unlabeled"))
        metadata = {k: v for k, v in obj.items() if k not in {"prompt", "label"}}
        rows.append(PromptItem(prompt_id=len(rows), prompt=prompt, label=label, metadata=metadata))
    return rows


def load_csv(text: str) -> list[PromptItem]:
    rows = []
    reader = csv.DictReader(io.StringIO(text))
    for obj in reader:
        prompt = str(obj.get("prompt", "")).strip()
        if not prompt:
            continue
        label = str(obj.get("label", "unlabeled") or "unlabeled")
        metadata = {k: v for k, v in obj.items() if k not in {"prompt", "label"}}
        rows.append(PromptItem(prompt_id=len(rows), prompt=prompt, label=label, metadata=metadata))
    return rows


def prompt_items_to_records(items: Iterable[PromptItem]) -> list[dict]:
    return [
        {
            "prompt_id": item.prompt_id,
            "label": item.label,
            "prompt": item.prompt,
            "metadata": item.metadata or {},
        }
        for item in items
    ]
