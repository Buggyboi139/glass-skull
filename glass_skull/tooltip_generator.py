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
        "control_id": "direct_steering_enabled",
        "title": "Enable Direct Steering",
        "control_type": "toggle",
        "typical_values": "Off until targets validate and the backend advertises support.",
        "use": "Use it to include direct activation steering in the next local chat request.",
        "downside": "Unsupported backends reject the request rather than falling back.",
    },
    {
        "control_id": "direct_steering_targets",
        "title": "Target Node IDs",
        "control_type": "text area",
        "typical_values": "L36-N175, L40-N18, or one ID per line.",
        "use": "Use it to choose Activation Map nodes as steering targets.",
        "downside": "Malformed IDs are rejected before a request is sent.",
    },
    {
        "control_id": "direct_steering_direction",
        "title": "Direction",
        "control_type": "dropdown",
        "typical_values": "Toward or Away.",
        "use": "Use Toward for positive steering and Away for negative steering.",
        "downside": "Away sends a negative strength, so verify direction before running.",
    },
    {
        "control_id": "direct_steering_strength",
        "title": "Strength",
        "control_type": "slider",
        "typical_values": "0.0 to 2.0; default 0.4.",
        "use": "Increase it for stronger direct activation steering or reduce it for subtler changes.",
        "downside": "High strength can degrade coherence or overpower the base model.",
    },
    {
        "control_id": "direct_steering_token_scope",
        "title": "Token Scope",
        "control_type": "dropdown",
        "typical_values": "All tokens, Prompt tokens, or Generated tokens.",
        "use": "Use it to choose where the direct activation steering applies.",
        "downside": "A scope that misses the relevant tokens may produce little visible change.",
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
