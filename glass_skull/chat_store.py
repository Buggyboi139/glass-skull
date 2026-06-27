from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import CHAT_DIR
from .experiment_store import safe_slug


def _chat_path(chat_id: str) -> Path:
    return CHAT_DIR / f"{safe_slug(chat_id)}.json"


def save_chat(messages: list[dict[str, Any]], label: str = "chat") -> Path | None:
    if not messages:
        return None
    CHAT_DIR.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    first_user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
    stem = safe_slug(first_user[:48] or label) or "chat"
    chat_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{stem}"
    path = _chat_path(chat_id)
    payload = {
        "id": path.stem,
        "label": first_user[:120] or label,
        "created_at": created_at,
        "message_count": len(messages),
        "messages": messages,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def list_chats() -> list[dict[str, Any]]:
    CHAT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for path in sorted(CHAT_DIR.glob("*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append(
            {
                "id": payload.get("id") or path.stem,
                "label": payload.get("label") or path.stem,
                "created_at": payload.get("created_at", ""),
                "message_count": payload.get("message_count", len(payload.get("messages", []))),
                "path": str(path),
            }
        )
    return rows


def load_chat(chat_id: str | None = None) -> list[dict[str, Any]]:
    chats = list_chats()
    if not chats:
        return []
    selected = chats[0] if chat_id is None else next((row for row in chats if row["id"] == chat_id), chats[0])
    payload = json.loads(Path(selected["path"]).read_text(encoding="utf-8"))
    messages = payload.get("messages", [])
    return messages if isinstance(messages, list) else []
