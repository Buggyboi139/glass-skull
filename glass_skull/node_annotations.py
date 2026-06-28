from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import NODE_ANNOTATION_PATH


SCHEMA_VERSION = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _annotation_path(path: str | Path | None = None) -> Path:
    return Path(path) if path is not None else NODE_ANNOTATION_PATH


def _blank_payload() -> dict[str, Any]:
    return {"version": SCHEMA_VERSION, "annotations": []}


def _normalise_tags(tags: list[str] | tuple[str, ...] | set[str] | None) -> list[str]:
    normalised: list[str] = []
    for tag in tags or []:
        text = str(tag).strip().lower()
        if text and text not in normalised:
            normalised.append(text)
    return normalised


def _normalise_range(node_range: Any) -> list[int] | None:
    if not isinstance(node_range, (list, tuple)) or len(node_range) < 2:
        return None
    try:
        return [int(node_range[0]), int(node_range[1])]
    except (TypeError, ValueError):
        return None


def model_fingerprint(model_meta: dict[str, Any] | None) -> str:
    meta = model_meta or {}
    for key in ("modelFingerprint", "model_fingerprint", "fingerprint", "sha256", "modelSha256"):
        value = str(meta.get(key) or "").strip()
        if value:
            return value
    values = [
        meta.get("backend"),
        meta.get("modelName") or meta.get("model_name"),
        meta.get("modelPath") or meta.get("model_path"),
        meta.get("architecture"),
        meta.get("layerCount"),
        meta.get("hiddenSize"),
        meta.get("vocabSize"),
        meta.get("quantization"),
    ]
    compact = "|".join(str(value or "") for value in values).strip("|")
    return compact


def _model_name(model_meta: dict[str, Any] | None) -> str:
    meta = model_meta or {}
    return str(meta.get("modelName") or meta.get("model_name") or meta.get("model") or "").strip()


def _backend(model_meta: dict[str, Any] | None) -> str:
    return str((model_meta or {}).get("backend") or "").strip()


def _same_model(annotation: dict[str, Any], model_meta: dict[str, Any] | None) -> tuple[bool, str]:
    requested_fingerprint = model_fingerprint(model_meta)
    annotation_fingerprint = str(annotation.get("model_fingerprint") or "").strip()
    if requested_fingerprint and annotation_fingerprint:
        return requested_fingerprint == annotation_fingerprint, "fingerprint"
    if annotation_fingerprint and not requested_fingerprint:
        return False, "fingerprint"
    same_name = str(annotation.get("model_name") or "").strip() == _model_name(model_meta)
    same_backend = str(annotation.get("backend") or "").strip() == _backend(model_meta)
    return bool(same_name and same_backend), "model_backend"


def load_annotations(path: str | Path | None = None) -> dict[str, Any]:
    target = _annotation_path(path)
    if not target.exists():
        return _blank_payload()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _blank_payload()
    annotations = payload.get("annotations")
    if not isinstance(annotations, list):
        annotations = []
    return {"version": int(payload.get("version") or SCHEMA_VERSION), "annotations": [item for item in annotations if isinstance(item, dict)]}


def save_annotations(payload: dict[str, Any], path: str | Path | None = None) -> dict[str, Any]:
    target = _annotation_path(path)
    annotations = payload.get("annotations") if isinstance(payload, dict) else []
    clean = {"version": SCHEMA_VERSION, "annotations": [item for item in annotations or [] if isinstance(item, dict)]}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(clean, indent=2), encoding="utf-8")
    return clean


def _annotation_payload(
    model_meta: dict[str, Any] | None,
    *,
    layer: int,
    cluster_id: Any,
    node_id: str | None,
    node_range: Any,
    tags: list[str] | None,
    note: str | None,
    created_from: dict[str, Any] | None,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = _now_iso()
    previous = dict(existing or {})
    return {
        "id": previous.get("id") or str(uuid.uuid4()),
        "model_fingerprint": model_fingerprint(model_meta),
        "model_name": _model_name(model_meta),
        "backend": _backend(model_meta),
        "layer": int(layer),
        "cluster_id": cluster_id,
        "node_id": str(node_id or ""),
        "node_range": _normalise_range(node_range),
        "tags": _normalise_tags(tags),
        "note": str(note or ""),
        "created_at": previous.get("created_at") or now,
        "updated_at": now,
        "created_from": dict(created_from or previous.get("created_from") or {}),
    }


def _find_annotation(
    payload: dict[str, Any],
    model_meta: dict[str, Any] | None,
    *,
    layer: int,
    cluster_id: Any,
    node_id: str | None,
    node_range: Any,
) -> dict[str, Any] | None:
    match = get_annotations_for_node(model_meta, layer, cluster_id, node_id, node_range, annotations=payload)
    if match["annotations"]:
        return match["annotations"][0]
    return None


def upsert_annotation(
    model_meta: dict[str, Any] | None,
    *,
    layer: int,
    cluster_id: Any,
    node_id: str | None,
    node_range: Any,
    tags: list[str] | None = None,
    note: str | None = "",
    created_from: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    payload = load_annotations(path)
    existing = _find_annotation(payload, model_meta, layer=layer, cluster_id=cluster_id, node_id=node_id, node_range=node_range)
    annotation = _annotation_payload(
        model_meta,
        layer=layer,
        cluster_id=cluster_id,
        node_id=node_id,
        node_range=node_range,
        tags=tags,
        note=note,
        created_from=created_from,
        existing=existing,
    )
    annotations = payload["annotations"]
    if existing:
        annotations[annotations.index(existing)] = annotation
    else:
        annotations.append(annotation)
    save_annotations(payload, path)
    return annotation


def _update_annotation(annotation_id: str, path: str | Path | None, updater) -> dict[str, Any]:
    payload = load_annotations(path)
    for annotation in payload["annotations"]:
        if annotation.get("id") == annotation_id:
            updater(annotation)
            annotation["updated_at"] = _now_iso()
            save_annotations(payload, path)
            return annotation
    raise KeyError(f"Annotation not found: {annotation_id}")


def add_tag(annotation_id: str, tag: str, path: str | Path | None = None) -> dict[str, Any]:
    def updater(annotation: dict[str, Any]) -> None:
        annotation["tags"] = _normalise_tags(list(annotation.get("tags") or []) + [tag])

    return _update_annotation(annotation_id, path, updater)


def delete_tag(annotation_id: str, tag: str, path: str | Path | None = None) -> dict[str, Any]:
    normalised = _normalise_tags([tag])
    target = normalised[0] if normalised else ""

    def updater(annotation: dict[str, Any]) -> None:
        annotation["tags"] = [item for item in _normalise_tags(annotation.get("tags") or []) if item != target]

    return _update_annotation(annotation_id, path, updater)


def update_note(annotation_id: str, note: str, path: str | Path | None = None) -> dict[str, Any]:
    def updater(annotation: dict[str, Any]) -> None:
        annotation["note"] = str(note or "")

    return _update_annotation(annotation_id, path, updater)


def delete_note(annotation_id: str, path: str | Path | None = None) -> dict[str, Any]:
    return update_note(annotation_id, "", path=path)


def delete_annotation(annotation_id: str, path: str | Path | None = None) -> bool:
    payload = load_annotations(path)
    before = len(payload["annotations"])
    payload["annotations"] = [item for item in payload["annotations"] if item.get("id") != annotation_id]
    save_annotations(payload, path)
    return len(payload["annotations"]) != before


def get_annotations_for_node(
    model_meta: dict[str, Any] | None,
    layer: int,
    cluster_id: Any,
    node_id: str | None,
    node_range: Any,
    annotations: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    payload = annotations if annotations is not None else load_annotations(path)
    requested_range = _normalise_range(node_range)
    exact: list[dict[str, Any]] = []
    approximate: list[dict[str, Any]] = []
    lower_confidence_exact: list[dict[str, Any]] = []
    for annotation in payload.get("annotations", []):
        if int(annotation.get("layer", -1)) != int(layer):
            continue
        same_model, model_match_basis = _same_model(annotation, model_meta)
        if not same_model:
            continue
        if str(annotation.get("cluster_id")) == str(cluster_id) or (node_id and str(annotation.get("node_id")) == str(node_id)):
            if model_match_basis == "fingerprint":
                exact.append(annotation)
            else:
                lower_confidence_exact.append(annotation)
            continue
        annotation_range = _normalise_range(annotation.get("node_range"))
        if requested_range and annotation_range == requested_range:
            approximate.append(annotation)
    if exact:
        return {"match_type": "exact", "confidence": "high", "annotations": exact}
    if lower_confidence_exact:
        return {"match_type": "approximate", "confidence": "low", "annotations": lower_confidence_exact}
    if approximate:
        return {"match_type": "approximate", "confidence": "medium", "annotations": approximate}
    return {"match_type": "none", "confidence": "none", "annotations": []}


def compact_annotation_match(match: dict[str, Any]) -> dict[str, Any]:
    annotations = match.get("annotations") or []
    tags: list[str] = []
    notes: list[str] = []
    ids: list[str] = []
    for annotation in annotations:
        ids.append(str(annotation.get("id") or ""))
        for tag in _normalise_tags(annotation.get("tags") or []):
            if tag not in tags:
                tags.append(tag)
        note = str(annotation.get("note") or "").strip()
        if note:
            notes.append(note)
    return {
        "annotationIds": [item for item in ids if item],
        "annotationMatchType": match.get("match_type", "none"),
        "annotationMatchConfidence": match.get("confidence", "none"),
        "annotationTags": tags,
        "annotationNote": "\n\n".join(notes),
    }


def annotation_file_metadata(path: str | Path | None = None) -> dict[str, Any]:
    target = _annotation_path(path)
    payload = load_annotations(target)
    updated_at = ""
    for annotation in payload["annotations"]:
        updated_at = max(updated_at, str(annotation.get("updated_at") or ""))
    return {
        "path": str(target),
        "version": payload["version"],
        "count": len(payload["annotations"]),
        "updated_at": updated_at,
    }
