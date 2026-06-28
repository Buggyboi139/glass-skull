from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BehaviorProfile:
    name: str
    label: str
    positive_terms: tuple[str, ...]
    negative_terms: tuple[str, ...]
    length_target: int | None = None


_PROFILES: dict[str, BehaviorProfile] = {
    "concise_helpfulness": BehaviorProfile(
        name="concise_helpfulness",
        label="Concise helpfulness",
        positive_terms=("clear", "concise", "safe", "action", "steps", "specific"),
        negative_terms=("cannot", "sorry", "unable", "maybe", "vague", "unsafe"),
        length_target=120,
    ),
    "refusal_reduction": BehaviorProfile(
        name="refusal_reduction",
        label="Refusal reduction",
        positive_terms=("answer", "steps", "example", "specific", "directly"),
        negative_terms=("cannot", "sorry", "unable", "policy", "refuse", "instead"),
        length_target=160,
    ),
    "direct_answering": BehaviorProfile(
        name="direct_answering",
        label="Direct answering",
        positive_terms=("answer", "because", "therefore", "first", "next"),
        negative_terms=("reasoning:", "analysis:", "<think", "as an ai", "cannot"),
        length_target=100,
    ),
}


def list_behavior_profiles() -> list[str]:
    return sorted(_PROFILES)


def get_behavior_profile(name: str | None = None) -> BehaviorProfile:
    key = (name or "concise_helpfulness").strip()
    if key not in _PROFILES:
        available = ", ".join(list_behavior_profiles())
        raise KeyError(f"Unknown behavior profile {key!r}. Available profiles: {available}")
    return _PROFILES[key]
