from __future__ import annotations

import re
from typing import Any, Iterable

import pandas as pd

from glass_skull.behavior_profiles import BehaviorProfile, get_behavior_profile


def _term_hits(text: str, terms: Iterable[str]) -> list[str]:
    lowered = text.lower()
    hits = []
    for term in terms:
        clean = term.strip().lower()
        if clean and clean in lowered:
            hits.append(term)
    return hits


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def score_behavior_output(
    output: str,
    *,
    profile: BehaviorProfile | None = None,
    prompt_id: Any = None,
    label: str | None = None,
    prompt: str = "",
    run_id: str | None = None,
) -> dict[str, Any]:
    active_profile = profile or get_behavior_profile()
    text = output or ""
    positive_hits = _term_hits(text, active_profile.positive_terms)
    negative_hits = _term_hits(text, active_profile.negative_terms)
    words = _word_count(text)

    length_bonus = 0.0
    if active_profile.length_target:
        distance = abs(words - active_profile.length_target)
        length_bonus = max(0.0, 1.0 - (distance / max(active_profile.length_target, 1)))

    score = len(positive_hits) - len(negative_hits) + length_bonus
    return {
        "run_id": run_id or "",
        "prompt_id": prompt_id,
        "label": label or "unlabeled",
        "prompt": prompt,
        "profile": active_profile.name,
        "score": float(score),
        "positive_hits": len(positive_hits),
        "negative_hits": len(negative_hits),
        "word_count": words,
        "positive_terms": positive_hits,
        "negative_terms": negative_hits,
    }


def score_run_artifact(
    artifact: dict[str, Any],
    *,
    profile: BehaviorProfile | None = None,
) -> pd.DataFrame:
    active_profile = profile or get_behavior_profile()
    rows = []
    for prompt in artifact.get("prompts", []) or []:
        rows.append(score_behavior_output(
            str(prompt.get("output") or ""),
            profile=active_profile,
            prompt_id=prompt.get("prompt_id"),
            label=prompt.get("label"),
            prompt=str(prompt.get("prompt") or ""),
            run_id=str(prompt.get("metadata", {}).get("run_id") or artifact.get("run_id") or ""),
        ))
    return pd.DataFrame(rows, columns=[
        "run_id",
        "prompt_id",
        "label",
        "prompt",
        "profile",
        "score",
        "positive_hits",
        "negative_hits",
        "word_count",
        "positive_terms",
        "negative_terms",
    ])


def behavior_timeline_df(runs: Iterable[pd.DataFrame | dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for index, run in enumerate(runs):
        if isinstance(run, dict):
            scored = score_run_artifact(run)
            run_id = str(run.get("run_id") or f"run_{index}")
        else:
            scored = run.copy()
            run_id = str(scored["run_id"].dropna().iloc[0]) if "run_id" in scored and not scored.empty else f"run_{index}"
        if scored.empty or "score" not in scored:
            continue
        scored["score"] = pd.to_numeric(scored["score"], errors="coerce")
        scored = scored.dropna(subset=["score"])
        if scored.empty:
            continue
        group_cols = ["label"] if "label" in scored.columns else []
        grouped = scored.groupby(group_cols, dropna=False) if group_cols else [((), scored)]
        for key, sub in grouped:
            label = key if isinstance(key, str) else (key[0] if isinstance(key, tuple) and key else "all")
            rows.append({
                "run_order": index,
                "run_id": run_id,
                "label": label or "unlabeled",
                "score": float(sub["score"].mean()),
                "prompt_count": int(len(sub)),
                "positive_hits": float(pd.to_numeric(sub.get("positive_hits", 0), errors="coerce").fillna(0).mean()),
                "negative_hits": float(pd.to_numeric(sub.get("negative_hits", 0), errors="coerce").fillna(0).mean()),
            })
    return pd.DataFrame(rows, columns=[
        "run_order",
        "run_id",
        "label",
        "score",
        "prompt_count",
        "positive_hits",
        "negative_hits",
    ])
