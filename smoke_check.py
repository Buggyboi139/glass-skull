from __future__ import annotations

import json

from glass_skull.config import DEFAULT_MODEL, MODEL_PRESETS, ensure_dirs
from glass_skull.experiment_store import safe_slug
from glass_skull.llama_client import normalize_base_url
from glass_skull.prompt_loader import load_jsonl, load_txt


def main() -> None:
    ensure_dirs()

    assert DEFAULT_MODEL in MODEL_PRESETS, "DEFAULT_MODEL should be one of MODEL_PRESETS"
    assert normalize_base_url("http://127.0.0.1:8080/v1") == "http://127.0.0.1:8080"
    assert safe_slug("Glass Probe!!!") == "Glass_Probe"

    txt_items = load_txt("The cat sat on the\n# comment\nThe car drove away\n")
    assert len(txt_items) == 2
    assert txt_items[0].prompt == "The cat sat on the"

    jsonl = '\n'.join([
        json.dumps({"label": "animal", "prompt": "Explain what a mouse is."}),
        json.dumps({"label": "vehicle", "prompt": "Explain what a car is."}),
    ])
    jsonl_items = load_jsonl(jsonl)
    assert len(jsonl_items) == 2
    assert jsonl_items[0].label == "animal"

    print("Glass Skull smoke check passed.")


if __name__ == "__main__":
    main()
