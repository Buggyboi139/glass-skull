from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

HF_API = "https://huggingface.co/api"


@dataclass
class HFTokenStatus:
    configured: bool
    valid: bool
    username: str | None = None
    error: str | None = None

    def label(self) -> str:
        if not self.configured:
            return "No token configured"
        if self.valid:
            return f"Valid token: {self.username or 'unknown user'}"
        return f"Invalid token: {self.error or 'unknown error'}"


@dataclass
class HFModelAccess:
    repo_id: str
    status: str
    ok: bool
    gated: bool | None = None
    private: bool | None = None
    error: str | None = None
    card_data: dict[str, Any] | None = None


def _headers(token: str | None = None) -> dict[str, str]:
    h = {
        "Accept": "application/json",
        "User-Agent": "glass-skull/0.7",
    }
    if token:
        h["Authorization"] = f"Bearer {token.strip()}"
    return h


def _get_json(url: str, token: str | None = None, timeout: float = 20.0) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(url, headers=_headers(token), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return int(resp.status), json.loads(body) if body.strip() else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(body) if body.strip() else {}
        except Exception:
            data = {"error": body[:500]}
        return int(exc.code), data


def validate_token(token: str | None) -> HFTokenStatus:
    token = (token or "").strip()
    if not token:
        return HFTokenStatus(configured=False, valid=False)

    status, data = _get_json(f"{HF_API}/whoami-v2", token=token, timeout=15.0)
    if status == 200:
        username = data.get("name") or data.get("fullname") or data.get("email")
        return HFTokenStatus(configured=True, valid=True, username=str(username) if username else None)
    err = data.get("error") or data.get("message") or f"HTTP {status}"
    return HFTokenStatus(configured=True, valid=False, error=str(err))


def check_model_access(repo_id: str, token: str | None = None) -> HFModelAccess:
    quoted = urllib.parse.quote(repo_id, safe="")
    status, data = _get_json(f"{HF_API}/models/{quoted}", token=(token or "").strip() or None, timeout=20.0)

    if status == 200:
        card_data = data.get("cardData") if isinstance(data.get("cardData"), dict) else {}
        gated = bool(data.get("gated")) if data.get("gated") is not None else bool(card_data.get("gated"))
        private = bool(data.get("private")) if data.get("private") is not None else False
        return HFModelAccess(repo_id=repo_id, status="ok", ok=True, gated=gated, private=private, card_data=card_data)

    if status == 401:
        return HFModelAccess(repo_id=repo_id, status="unauthorized", ok=False, error="Unauthorized. Token missing, invalid, or insufficient.")
    if status == 403:
        return HFModelAccess(repo_id=repo_id, status="forbidden", ok=False, error="Forbidden. Token is valid but model access is not approved.")
    if status == 404:
        return HFModelAccess(repo_id=repo_id, status="missing", ok=False, error="Model repo not found, private, or not visible to this token.")

    err = data.get("error") or data.get("message") or f"HTTP {status}"
    return HFModelAccess(repo_id=repo_id, status="error", ok=False, error=str(err))


def access_badge_text(access: HFModelAccess | None) -> str:
    if access is None:
        return "access not checked"
    if access.ok:
        flags = []
        if access.gated:
            flags.append("gated")
        if access.private:
            flags.append("private")
        suffix = f" ({', '.join(flags)})" if flags else ""
        return f"access ok{suffix}"
    return access.status.replace("_", " ")
