from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import TOOLTIP_DIR


STEER_CONTROL_METADATA: list[dict[str, str]] = [
    {
        "control_id": "steer_tab_load_name",
        "title": "Load Steer state selector",
        "control_type": "dropdown",
        "typical_values": "Any saved Steer tab state.",
        "use": "Use it before pressing Load to choose which local Steer settings to restore.",
        "downside": "Loading replaces only Steer tab settings, so unsaved Steer edits are lost.",
    },
    {
        "control_id": "steer_tab_save",
        "title": "Save Steer state",
        "control_type": "button",
        "typical_values": "Current Steer tab state name.",
        "use": "Use it to overwrite the selected Steer tab state with the current controls.",
        "downside": "It overwrites that tab state; use Save As when you need a new copy.",
    },
    {
        "control_id": "steer_tab_save_as",
        "title": "Save Steer state as",
        "control_type": "button",
        "typical_values": "A short local name.",
        "use": "Use it to create a separate saved Steer tab state.",
        "downside": "Reusing a name replaces the existing saved Steer state with that name.",
    },
    {
        "control_id": "steer_tab_load",
        "title": "Load Steer state",
        "control_type": "button",
        "typical_values": "Any saved Steer tab state.",
        "use": "Use it to restore only Steer controls from the selected tab state.",
        "downside": "Other tabs and the global workspace are not changed.",
    },
    {
        "control_id": "steer_tab_clear",
        "title": "Clear Steer state",
        "control_type": "button",
        "typical_values": "Built-in Steer defaults.",
        "use": "Use it to reset only the Steer tab controls to defaults.",
        "downside": "It does not delete saved tab states, but unsaved Steer edits are discarded.",
    },
    {
        "control_id": "steer_tab_save_as_name",
        "title": "Steer Save As name",
        "control_type": "text input",
        "typical_values": "Names such as baseline_steer or high_strength_test.",
        "use": "Use it to name the next Steer Save As target.",
        "downside": "Names are normalized for filesystem safety, so punctuation may become underscores.",
    },
    {
        "control_id": "control_set",
        "title": "Control set",
        "control_type": "dropdown",
        "typical_values": "Blank, or a saved positive/negative prompt set.",
        "use": "Use it to choose prompts for generating a new control vector.",
        "downside": "A weak or mismatched control set can produce a vector that steers unrelated behavior.",
    },
    {
        "control_id": "control_vector",
        "title": "Control vector",
        "control_type": "dropdown",
        "typical_values": "Blank, or a generated GGUF control vector.",
        "use": "Use it to select the vector applied to a steered llama.cpp run.",
        "downside": "Vectors from another model or layer layout may fail or steer unpredictably.",
    },
    {
        "control_id": "set_name",
        "title": "Set name",
        "control_type": "text input",
        "typical_values": "Short names such as concise_helpfulness.",
        "use": "Use it to name the saved positive/negative prompt set.",
        "downside": "Reusing a name can replace the previous control set files.",
    },
    {
        "control_id": "positive_prompts",
        "title": "Positive prompts",
        "control_type": "text area",
        "typical_values": "Several examples of the behavior you want more of.",
        "use": "Use it to define the direction the control vector should encourage.",
        "downside": "Too few or inconsistent examples make the generated vector noisy.",
    },
    {
        "control_id": "negative_prompts",
        "title": "Negative prompts",
        "control_type": "text area",
        "typical_values": "Several examples of the behavior you want less of.",
        "use": "Use it to define the contrast direction for the control vector.",
        "downside": "If negatives differ in topic instead of behavior, the vector can steer topic rather than style.",
    },
    {
        "control_id": "save_control_set",
        "title": "Save control set",
        "control_type": "button",
        "typical_values": "Positive and negative prompt lists.",
        "use": "Use it to write the control set under data/control_sets.",
        "downside": "Both prompt lists are required; empty lists are rejected.",
    },
    {
        "control_id": "strength",
        "title": "Strength",
        "control_type": "number input",
        "typical_values": "Around 0.5 to 2.0; default 1.25.",
        "use": "Increase it for stronger steering or reduce it for subtler behavior changes.",
        "downside": "High strength can degrade coherence or overpower the base model.",
    },
    {
        "control_id": "layer_start",
        "title": "Layer start",
        "control_type": "number input",
        "typical_values": "Early positive layer index, default 1.",
        "use": "Use it to choose the first transformer layer receiving the control vector.",
        "downside": "Starting too early can create broad, hard-to-debug changes.",
    },
    {
        "control_id": "layer_end",
        "title": "Layer end",
        "control_type": "number input",
        "typical_values": "A later layer index such as 16 to 32.",
        "use": "Use it to choose the last transformer layer receiving the control vector.",
        "downside": "A range outside the loaded model's layers will fail in llama.cpp.",
    },
    {
        "control_id": "vector_name",
        "title": "Vector name",
        "control_type": "text input",
        "typical_values": "A short name matching the control set.",
        "use": "Use it to name the generated GGUF control vector file.",
        "downside": "Reusing a name can replace the previous vector metadata and file.",
    },
    {
        "control_id": "generate_vector",
        "title": "Generate vector",
        "control_type": "button",
        "typical_values": "The selected control set and current GGUF model.",
        "use": "Use it to run the repo-managed cvector generator and create a new vector.",
        "downside": "Generation can take time and fails if the managed llama.cpp tools are missing.",
    },
    {
        "control_id": "patch_source_run",
        "title": "Source run",
        "control_type": "dropdown",
        "typical_values": "A saved single-message or batch run with trace rows.",
        "use": "Use it to choose where activation values are copied or transformed from.",
        "downside": "Runs without captured activations cannot produce useful patch targets.",
    },
    {
        "control_id": "patch_target_run",
        "title": "Target/current run",
        "control_type": "dropdown",
        "typical_values": "A saved run you want to compare against or patch.",
        "use": "Use it to provide the target prompt and baseline context for patch comparison.",
        "downside": "Using an unrelated target run can make comparisons misleading.",
    },
    {
        "control_id": "patch_layer",
        "title": "Patch layer",
        "control_type": "dropdown",
        "typical_values": "Any layer captured in the source trace.",
        "use": "Use it to choose which layer's activation slice the patch modifies.",
        "downside": "Layer effects vary by model; a layer that works for one behavior may do little for another.",
    },
    {
        "control_id": "patch_token_start",
        "title": "Token start",
        "control_type": "number input",
        "typical_values": "Zero-based token index.",
        "use": "Use it to choose the first target token in the activation patch span.",
        "downside": "Choosing the wrong token span can patch irrelevant context.",
    },
    {
        "control_id": "patch_token_end",
        "title": "Token end",
        "control_type": "number input",
        "typical_values": "A token index at or after Token start.",
        "use": "Use it to choose the final token in the activation patch span.",
        "downside": "Large spans can make the intervention harder to attribute.",
    },
    {
        "control_id": "patch_mode",
        "title": "Patch mode",
        "control_type": "dropdown",
        "typical_values": "blend, replace, add_delta, scale, or zero.",
        "use": "Use it to choose how source activations modify target activations.",
        "downside": "Replace and zero are strong interventions and can produce brittle outputs.",
    },
    {
        "control_id": "patch_node_start",
        "title": "Node/channel start",
        "control_type": "text input",
        "typical_values": "Blank for all channels, or a zero-based channel index.",
        "use": "Use it to limit the patch to a specific activation channel range.",
        "downside": "Narrow ranges are easier to inspect but may miss distributed behavior.",
    },
    {
        "control_id": "patch_node_end",
        "title": "Node/channel end",
        "control_type": "text input",
        "typical_values": "Blank for all channels, or an index after the start.",
        "use": "Use it to close the channel range for a targeted patch.",
        "downside": "Invalid ranges are rejected and overly wide ranges can obscure causality.",
    },
    {
        "control_id": "patch_strength",
        "title": "Patch strength",
        "control_type": "slider",
        "typical_values": "0.0 to 2.0; default 0.35.",
        "use": "Use it to scale the patch effect for blend, delta, and scale modes.",
        "downside": "Large values can destabilize generation or make comparisons less interpretable.",
    },
    {
        "control_id": "patch_target_prompt",
        "title": "Target prompt",
        "control_type": "text area",
        "typical_values": "The prompt to run with the patch.",
        "use": "Use it to edit the generation prompt for patched and comparison runs.",
        "downside": "Changing it after choosing runs can make the baseline comparison less direct.",
    },
    {
        "control_id": "patch_recipe_name",
        "title": "Recipe name",
        "control_type": "text input",
        "typical_values": "Names that include source run and layer.",
        "use": "Use it to name the saved activation patch recipe.",
        "downside": "Reusing a name can replace an earlier recipe.",
    },
    {
        "control_id": "save_patch_recipe",
        "title": "Save patch recipe",
        "control_type": "button",
        "typical_values": "The currently valid patch settings.",
        "use": "Use it to store the patch recipe under data/activation_patch_recipes.",
        "downside": "Invalid recipes cannot be saved and saved recipes do not include source trace files.",
    },
    {
        "control_id": "patch_recipe_load",
        "title": "Load recipe",
        "control_type": "dropdown",
        "typical_values": "Any saved activation patch recipe.",
        "use": "Use it to activate a saved patch recipe for the next patched generation.",
        "downside": "A recipe built from old runs may not match the current source artifact.",
    },
    {
        "control_id": "run_patched_generation",
        "title": "Run patched generation",
        "control_type": "button",
        "typical_values": "Current target prompt and active patch recipe.",
        "use": "Use it to send one patched request to the local llama.cpp server.",
        "downside": "It requires server-side activation patch support.",
    },
    {
        "control_id": "compare_patched_vs_baseline",
        "title": "Compare patched vs baseline",
        "control_type": "button",
        "typical_values": "Current target prompt and active patch recipe.",
        "use": "Use it to run a baseline and patched output for side-by-side comparison.",
        "downside": "It performs two generations, so runtime and model variance both increase.",
    },
]


def _path(scope: str) -> Path:
    return TOOLTIP_DIR / f"{scope}_tooltips.json"


def _read_tooltips(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): value for key, value in payload.items() if isinstance(value, dict)}


def _generate_tooltip(item: dict[str, str]) -> dict[str, str]:
    title = item.get("title") or item.get("control_id") or "Control"
    control_type = item.get("control_type", "control")
    typical = item.get("typical_values", "Use the defaults unless a run requires a change.")
    practical_use = item.get("use", "Use it when that part of the Steer workflow needs adjustment.")
    warning = item.get("downside", "Changing it can affect the current run.")
    return {
        "control_id": str(item.get("control_id", "")),
        "title": title,
        "explanation": f"{title} is a {control_type} in the Steer workflow. Changing it updates how local steering or activation patching is configured.",
        "practical_use": f"Typical values: {typical} {practical_use}",
        "warning_if_any": warning,
    }


def ensure_tooltips(scope: str, metadata: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    path = _path(scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    tooltips = _read_tooltips(path)
    changed = False
    for item in metadata:
        control_id = str(item.get("control_id", "")).strip()
        if not control_id or control_id in tooltips:
            continue
        tooltips[control_id] = _generate_tooltip(item)
        changed = True
    if changed or not path.exists():
        path.write_text(json.dumps(tooltips, indent=2, sort_keys=True), encoding="utf-8")
    return tooltips


def tooltip_text(entry: dict[str, Any] | None) -> str | None:
    if not isinstance(entry, dict):
        return None
    parts = [
        str(entry.get("title", "")).strip(),
        str(entry.get("explanation", "")).strip(),
        str(entry.get("practical_use", "")).strip(),
        str(entry.get("warning_if_any", "")).strip(),
    ]
    text = " ".join(part for part in parts if part)
    return text or None
