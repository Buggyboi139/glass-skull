from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from .prompt_loader import PromptItem


LOCAL_TABS = ["Run", "Map", "Steer", "Timeline", "Model", "Settings"]


def new_run_id(prefix: str = "run") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}_{uuid4().hex[:8]}"


def parse_pasted_payload(text: str) -> list[PromptItem]:
    prompts = [line.strip() for line in text.splitlines() if line.strip()]
    return [
        PromptItem(
            prompt_id=index,
            prompt=prompt,
            metadata={"source": "pasted_payload", "batch_index": index},
        )
        for index, prompt in enumerate(prompts)
    ]


def repeated_prompt_items(prompt: str, repeat_count: int) -> list[PromptItem]:
    prompt = prompt.strip()
    if not prompt or repeat_count <= 0:
        return []
    return [
        PromptItem(
            prompt_id=index,
            prompt=prompt,
            metadata={"source": "repeat_current_message", "repeat_index": index, "repeat_count": repeat_count},
        )
        for index in range(repeat_count)
    ]


def batch_items_from_inputs(
    *,
    pasted_payload: str = "",
    repeat_prompt: str = "",
    repeat_count: int = 1,
    uploaded_items: list[PromptItem] | None = None,
) -> list[PromptItem]:
    items: list[PromptItem] = []
    for item in parse_pasted_payload(pasted_payload):
        items.append(item)
    offset = len(items)
    for item in repeated_prompt_items(repeat_prompt, repeat_count):
        item.prompt_id = offset + len(items) - offset
        items.append(item)
    offset = len(items)
    for index, item in enumerate(uploaded_items or []):
        metadata = dict(item.metadata or {})
        metadata.setdefault("source", "uploaded_file")
        items.append(PromptItem(prompt_id=offset + index, prompt=item.prompt, label=item.label, metadata=metadata))
    return items


def dashboard_context(mode: str, run_id: str | None) -> dict[str, str]:
    return {
        "mode": mode,
        "run_id": run_id or "none",
    }
