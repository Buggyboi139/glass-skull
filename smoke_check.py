from __future__ import annotations

import json

from glass_skull.aggregation import label_separation_table
from glass_skull.attention_view import indexed_tokens
from glass_skull.config import DEFAULT_MODEL, MODEL_PRESETS, ensure_dirs
from glass_skull.experiment_store import safe_slug
from glass_skull.feature_store import compatible_features
from glass_skull.lens import logit_lens_table
from glass_skull.llama_client import normalize_base_url
from glass_skull.prompt_loader import load_jsonl, load_txt
from glass_skull.visuals import comparison_delta_heatmap


def main() -> None:
    ensure_dirs()

    assert DEFAULT_MODEL in MODEL_PRESETS, "DEFAULT_MODEL should be one of MODEL_PRESETS"
    assert normalize_base_url("http://127.0.0.1:8080/v1") == "http://127.0.0.1:8080"
    assert safe_slug("Glass Probe!!!") == "Glass_Probe"
    assert indexed_tokens(["The", " cat"]) == ["0:The", "1: cat"]
    assert isinstance(compatible_features(512), list)

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

    records = [
        {"prompt_id": 0, "label": "a", "trace_layers": [{"layer": 0, "stream": "resid_post", "norm": 1.0}]},
        {"prompt_id": 1, "label": "b", "trace_layers": [{"layer": 0, "stream": "resid_post", "norm": 2.0}]},
    ]
    sep = label_separation_table(records)
    assert not sep.empty

    # Import-only checks so new cockpit modules fail fast without loading a model.
    assert callable(logit_lens_table)
    assert callable(comparison_delta_heatmap)

    print("Glass Skull smoke check passed.")


if __name__ == "__main__":
    main()
