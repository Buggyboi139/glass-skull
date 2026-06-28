from __future__ import annotations


LAB_SOURCES = {"Hugging Face", "Trace model"}
LAB_CAPTION = "Optional Hugging Face and TransformerLens tools."


def lab_enabled(sources: list[str] | tuple[str, ...] | set[str]) -> bool:
    return any(source in LAB_SOURCES for source in sources)


def trace_enabled(sources: list[str] | tuple[str, ...] | set[str]) -> bool:
    return "Trace model" in sources


def hf_enabled(sources: list[str] | tuple[str, ...] | set[str]) -> bool:
    return "Hugging Face" in sources
