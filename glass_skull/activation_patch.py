from __future__ import annotations

from typing import Any


PATCH_MODES = ("replace", "add_delta", "scale", "zero", "blend")
SOURCE_VECTOR_MODES = {"replace", "add_delta", "blend"}


def _range_or_none(start: int | None, end: int | None) -> list[int] | None:
    if start is None or end is None:
        return None
    start_i = int(start)
    end_i = int(end)
    if end_i < start_i:
        start_i, end_i = end_i, start_i
    return [start_i, end_i]


def build_activation_patch_recipe(
    *,
    name: str = "activation-patch",
    source_run_id: str | None = None,
    target_run_id: str | None = None,
    layer: int,
    token_start: int,
    token_end: int,
    mode: str,
    strength: float,
    node_start: int | None = None,
    node_end: int | None = None,
    backend: str = "llama.cpp",
    model: str = "local-model-name",
    baseline_run_id: str | None = None,
    patched_run_id: str | None = None,
) -> dict[str, Any]:
    token_range = _range_or_none(token_start, token_end) or [0, 0]
    source_run_id = source_run_id or baseline_run_id or ""
    target_run_id = target_run_id or patched_run_id or ""
    recipe = {
        "name": str(name).strip() or "activation-patch",
        "backend": backend,
        "model": model,
        "source_run_id": str(source_run_id),
        "target_run_id": str(target_run_id),
        "baseline_run_id": baseline_run_id or source_run_id,
        "patched_run_id": patched_run_id or target_run_id,
        "mode": str(mode),
        "layer": int(layer),
        "token_start": int(token_range[0]),
        "token_end": int(token_range[1]),
        "node_start": node_start,
        "node_end": node_end,
        "strength": float(strength),
        "patches": [
            {
                "mode": str(mode),
                "layer": int(layer),
                "token_range": token_range,
                "source_token_range": list(token_range),
                "node_range": _range_or_none(node_start, node_end),
                "strength": float(strength),
            }
        ],
    }
    return recipe


def validate_activation_patch_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    if isinstance(recipe, dict) and "patches" not in recipe:
        return _validate_flat_recipe(recipe)

    errors: list[str] = []
    if not isinstance(recipe, dict):
        raise ValueError("recipe must be an object")
    if not str(recipe.get("name") or "").strip():
        errors.append("name is required")
    if recipe.get("backend") != "llama.cpp":
        errors.append("backend must be llama.cpp")
    for key in ("source_run_id", "target_run_id"):
        if not str(recipe.get(key) or "").strip():
            errors.append(f"{key} is required")
    patches = recipe.get("patches")
    if not isinstance(patches, list) or not patches:
        errors.append("patches must contain at least one patch")
        raise ValueError("; ".join(errors))
    for index, patch in enumerate(patches):
        if not isinstance(patch, dict):
            errors.append(f"patches[{index}] must be an object")
            continue
        if patch.get("mode") not in PATCH_MODES:
            errors.append(f"patches[{index}].mode must be one of {', '.join(PATCH_MODES)}")
        try:
            if int(patch.get("layer")) < 0:
                errors.append(f"patches[{index}].layer must be nonnegative")
        except Exception:
            errors.append(f"patches[{index}].layer must be an integer")
        token_range = patch.get("token_range")
        if not _valid_range(token_range):
            errors.append(f"patches[{index}].token_range must be [start, end]")
        source_token_range = patch.get("source_token_range")
        if source_token_range is not None and not _valid_range(source_token_range):
            errors.append(f"patches[{index}].source_token_range must be [start, end]")
        node_range = patch.get("node_range")
        if node_range is not None and not _valid_range(node_range):
            errors.append(f"patches[{index}].node_range must be [start, end] or null")
        try:
            float(patch.get("strength"))
        except Exception:
            errors.append(f"patches[{index}].strength must be numeric")
    if errors:
        raise ValueError("; ".join(errors))
    first = recipe["patches"][0]
    token_range = first.get("token_range") or [recipe.get("token_start"), recipe.get("token_end")]
    node_range = first.get("node_range")
    return {
        **recipe,
        "mode": first.get("mode"),
        "layer": int(first.get("layer")),
        "token_start": int(token_range[0]),
        "token_end": int(token_range[1]),
        "node_start": int(node_range[0]) if isinstance(node_range, list) and len(node_range) == 2 else recipe.get("node_start"),
        "node_end": int(node_range[1]) if isinstance(node_range, list) and len(node_range) == 2 else recipe.get("node_end"),
        "strength": float(first.get("strength")),
    }


def _validate_flat_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    mode = recipe.get("mode")
    if mode not in PATCH_MODES:
        raise ValueError(f"mode must be one of {', '.join(PATCH_MODES)}")
    try:
        layer = int(recipe.get("layer"))
    except Exception as exc:
        raise ValueError("layer must be an integer") from exc
    if layer < 0:
        raise ValueError("layer must be nonnegative")
    try:
        token_start = int(recipe.get("token_start"))
        token_end = int(recipe.get("token_end"))
    except Exception as exc:
        raise ValueError("token range must be numeric") from exc
    if token_start < 0 or token_end < token_start:
        raise ValueError("token_start and token_end must be a valid range")
    try:
        strength = float(recipe.get("strength"))
    except Exception as exc:
        raise ValueError("strength must be numeric") from exc
    return {
        **recipe,
        "mode": mode,
        "layer": layer,
        "token_start": token_start,
        "token_end": token_end,
        "strength": strength,
    }


def _valid_range(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    try:
        start, end = int(value[0]), int(value[1])
    except Exception:
        return False
    return start >= 0 and end >= start


def build_activation_patch_backend_payload(recipe: dict[str, Any], source_artifact: dict[str, Any] | None = None) -> dict[str, Any]:
    validation = validate_activation_patch_recipe(recipe)
    if "patches" not in recipe:
        validated = validation
        return {
            "mode": validated["mode"],
            "layer": validated["layer"],
            "token_range": [validated["token_start"], validated["token_end"]],
            "node_range": _range_or_none(validated.get("node_start"), validated.get("node_end")),
            "strength": validated["strength"],
            "baseline_run_id": validated.get("baseline_run_id"),
            "patched_run_id": validated.get("patched_run_id"),
        }
    source_vectors = _source_vectors_for_recipe(recipe, source_artifact or {})
    if source_artifact is not None and validation["mode"] in SOURCE_VECTOR_MODES and not source_vectors:
        raise RuntimeError(
            "Activation patch recipe needs source activation vectors for replace, add_delta, or blend modes. "
            "Capture the source trace with include_vectors=True before running this patched generation."
        )
    return {
        "recipe": recipe,
        "source_vectors": source_vectors,
        "mode": validation["mode"],
        "layer": validation["layer"],
        "token_range": [validation["token_start"], validation["token_end"]],
        "node_range": _range_or_none(validation.get("node_start"), validation.get("node_end")),
        "strength": validation["strength"],
        "diagnostics": {
            "backend": "llama.cpp",
            "source_run_id": recipe.get("source_run_id"),
            "target_run_id": recipe.get("target_run_id"),
        },
    }


def _source_vectors_for_recipe(recipe: dict[str, Any], artifact: dict[str, Any]) -> list[dict[str, Any]]:
    vectors: list[dict[str, Any]] = []
    wanted_layers = {int(patch["layer"]) for patch in recipe.get("patches", []) if isinstance(patch, dict) and "layer" in patch}
    for prompt in artifact.get("prompts", []):
        for row in prompt.get("trace_rows", []):
            if not isinstance(row, dict):
                continue
            if row.get("trace_available") is False:
                continue
            vector = row.get("vector") or row.get("activation_vector")
            if not isinstance(vector, list):
                continue
            try:
                layer = int(row.get("layer"))
            except Exception:
                continue
            if wanted_layers and layer not in wanted_layers:
                continue
            vectors.append({
                "run_id": row.get("run_id") or artifact.get("run_id"),
                "prompt_id": row.get("prompt_id", prompt.get("prompt_id")),
                "layer": layer,
                "token_index": row.get("token_index"),
                "stream": row.get("stream") or row.get("component") or "resid_pre",
                "vector": vector,
            })
    return vectors


def compare_patch_outputs(
    baseline: str | None = None,
    patched: str | None = None,
    *,
    baseline_run_id: str | None = None,
    patched_run_id: str | None = None,
    baseline_output: str | None = None,
    patched_output: str | None = None,
) -> dict[str, Any]:
    baseline_text = baseline_output if baseline_output is not None else (baseline or "")
    patched_text = patched_output if patched_output is not None else (patched or "")
    return {
        "baseline_run_id": baseline_run_id,
        "patched_run_id": patched_run_id,
        "baseline_output": baseline_text,
        "patched_output": patched_text,
        "changed": baseline_text != patched_text,
        "length_delta": len(patched_text) - len(baseline_text),
    }
