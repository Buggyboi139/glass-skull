from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import struct
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import CONTROL_SET_DIR, CONTROL_VECTOR_DIR
from .experiment_store import safe_slug
from .llama_paths import DEFAULT_CVECTOR_GENERATOR, DEFAULT_LLAMA_SERVER, MANAGED_LLAMA_CPP_DIR


DEFAULT_LLAMA_CPP_DIR = MANAGED_LLAMA_CPP_DIR

GGML_TYPE_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
}


@dataclass(frozen=True)
class ControlSetPaths:
    name: str
    positive_path: Path
    negative_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class ControlVectorMetadata:
    name: str
    vector_path: Path
    model_path: str
    model_sha256: str | None
    positive_path: str
    negative_path: str
    method: str
    ngl: int | None
    gpu_layer_mode: str
    fit: str | None
    ctx_size: int | None
    pca_batch: int | None
    pca_iter: int | None
    model_architecture: str | None
    compatibility_warnings: list[str]
    created_at: str
    generator_path: str
    command: list[str]
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    failure_cause: str = ""
    recommended_action: str = ""


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class ControlVectorPreflight:
    checks: list[PreflightCheck]
    warnings: list[str]
    errors: list[str]
    model_architecture: str | None = None
    model_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class CVectorFailure:
    cause: str
    recommendation: str
    warnings: list[str]


class ControlVectorRunError(RuntimeError):
    def __init__(self, metadata: ControlVectorMetadata, failure: CVectorFailure):
        self.metadata = metadata
        self.failure = failure
        super().__init__(
            f"llama-cvector-generator failed with exit code {metadata.returncode}: "
            f"{failure.cause or 'see stderr'}"
        )


def shell_join(args: list[str | Path]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def _nonempty_lines(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_lines(path: Path, text: str) -> int:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def write_control_set(name: str, positive_text: str, negative_text: str) -> ControlSetPaths:
    slug = safe_slug(name)
    base = CONTROL_SET_DIR / slug
    base.mkdir(parents=True, exist_ok=True)
    positive_path = base / "positive.txt"
    negative_path = base / "negative.txt"
    metadata_path = base / "metadata.json"
    positive_count = _write_lines(positive_path, positive_text)
    negative_count = _write_lines(negative_path, negative_text)
    metadata = {
        "name": slug,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "positive_path": str(positive_path),
        "negative_path": str(negative_path),
        "positive_count": positive_count,
        "negative_count": negative_count,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return ControlSetPaths(slug, positive_path, negative_path, metadata_path)


def list_control_sets() -> list[dict[str, Any]]:
    CONTROL_SET_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for path in sorted(CONTROL_SET_DIR.iterdir()):
        if not path.is_dir():
            continue
        positive_path = path / "positive.txt"
        negative_path = path / "negative.txt"
        metadata_path = path / "metadata.json"
        if not positive_path.exists() or not negative_path.exists():
            continue
        meta: dict[str, Any] = {}
        if metadata_path.exists():
            try:
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        rows.append(
            {
                "name": meta.get("name") or path.name,
                "positive_path": str(positive_path),
                "negative_path": str(negative_path),
                "positive_count": meta.get("positive_count"),
                "negative_count": meta.get("negative_count"),
                "created_at": meta.get("created_at", ""),
            }
        )
    return rows


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_gguf_string(f) -> str:
    size_raw = f.read(8)
    if len(size_raw) != 8:
        raise ValueError("short GGUF string length")
    size = struct.unpack("<Q", size_raw)[0]
    data = f.read(size)
    if len(data) != size:
        raise ValueError("short GGUF string data")
    return data.decode("utf-8", errors="replace")


def _skip_gguf_value(f, value_type: int) -> None:
    fixed_sizes = {
        0: 1,  # uint8
        1: 1,  # int8
        2: 2,  # uint16
        3: 2,  # int16
        4: 4,  # uint32
        5: 4,  # int32
        6: 4,  # float32
        7: 1,  # bool
        10: 8,  # uint64
        11: 8,  # int64
        12: 8,  # float64
    }
    if value_type == 8:
        _read_gguf_string(f)
        return
    if value_type == 9:
        subtype = struct.unpack("<I", f.read(4))[0]
        length = struct.unpack("<Q", f.read(8))[0]
        _skip_gguf_array_items(f, subtype, length)
        return
    if value_type not in fixed_sizes:
        raise ValueError(f"unsupported GGUF value type {value_type}")
    f.seek(fixed_sizes[value_type], 1)


def _skip_gguf_array_items(f, subtype: int, length: int) -> None:
    fixed_sizes = {
        0: 1,
        1: 1,
        2: 2,
        3: 2,
        4: 4,
        5: 4,
        6: 4,
        7: 1,
        10: 8,
        11: 8,
        12: 8,
    }
    if subtype == 8:
        for _ in range(length):
            _read_gguf_string(f)
        return
    if subtype not in fixed_sizes:
        raise ValueError(f"unsupported GGUF array subtype {subtype}")
    f.seek(fixed_sizes[subtype] * length, 1)


def _read_gguf_value(f, value_type: int) -> Any:
    formats = {
        0: "<B",
        1: "<b",
        2: "<H",
        3: "<h",
        4: "<I",
        5: "<i",
        6: "<f",
        7: "<?",
        10: "<Q",
        11: "<q",
        12: "<d",
    }
    if value_type == 8:
        return _read_gguf_string(f)
    if value_type == 9:
        subtype = struct.unpack("<I", f.read(4))[0]
        length = struct.unpack("<Q", f.read(8))[0]
        if length > 64:
            _skip_gguf_array_items(f, subtype, length)
            return f"<array type={subtype} length={length}>"
        if subtype == 8:
            return [_read_gguf_string(f) for _ in range(length)]
        if subtype not in formats:
            raise ValueError(f"unsupported GGUF array subtype {subtype}")
        size = struct.calcsize(formats[subtype])
        return [struct.unpack(formats[subtype], f.read(size))[0] for _ in range(length)]
    if value_type not in formats:
        raise ValueError(f"unsupported GGUF value type {value_type}")
    size = struct.calcsize(formats[value_type])
    return struct.unpack(formats[value_type], f.read(size))[0]


def read_gguf_metadata(path: str | Path, max_items: int = 256) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    with Path(path).open("rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        metadata["_gguf_version"] = struct.unpack("<I", f.read(4))[0]
        metadata["_tensor_count"] = struct.unpack("<Q", f.read(8))[0]
        kv_count = struct.unpack("<Q", f.read(8))[0]
        metadata["_metadata_kv_count"] = kv_count
        for index in range(kv_count):
            key = _read_gguf_string(f)
            value_type = struct.unpack("<I", f.read(4))[0]
            if index < max_items:
                metadata[key] = _read_gguf_value(f, value_type)
            else:
                _skip_gguf_value(f, value_type)
    return metadata


def read_gguf_tensor_index(path: str | Path) -> list[dict[str, Any]]:
    tensors: list[dict[str, Any]] = []
    with Path(path).open("rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version = struct.unpack("<I", f.read(4))[0]
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        kv_count = struct.unpack("<Q", f.read(8))[0]
        for _ in range(kv_count):
            _read_gguf_string(f)
            value_type = struct.unpack("<I", f.read(4))[0]
            _skip_gguf_value(f, value_type)
        for index in range(tensor_count):
            name = _read_gguf_string(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            shape = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            type_id = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            elements = 1
            for dim in shape:
                elements *= int(dim)
            tensors.append(
                {
                    "index": index,
                    "name": name,
                    "shape": "x".join(str(dim) for dim in shape),
                    "n_dims": n_dims,
                    "type_id": type_id,
                    "dtype": GGML_TYPE_NAMES.get(type_id, f"type_{type_id}"),
                    "elements": elements,
                    "offset": offset,
                    "gguf_version": version,
                }
            )
    return tensors


def model_compatibility_warnings(model_architecture: str | None, metadata: dict[str, Any] | None, model_path: str | Path) -> list[str]:
    metadata = metadata or {}
    arch = (model_architecture or "").lower()
    keys = " ".join(str(key).lower() for key in metadata.keys())
    path_text = str(model_path).lower()
    warnings: list[str] = []
    if "qwen35moe" in arch or "qwen35moe" in keys:
        warnings.append("Model architecture is qwen35moe; stock cvector generation may hit the layer-count assertion on MoE/MTP graphs.")
    elif "moe" in arch or ".expert_" in keys or "expert_count" in keys:
        warnings.append("Model appears to be MoE; stock cvector generation may not capture one output tensor per transformer layer.")
    if "mtp" in path_text or "nextn" in keys or "mtp" in keys:
        warnings.append("Model appears to include MTP/next-token-prediction heads; cvector-generator compatibility may require a llama.cpp patch.")
    return warnings


def preflight_control_vector_run(
    model_path: str | Path,
    positive_path: str | Path | None,
    negative_path: str | Path | None,
    generator_path: str | Path = DEFAULT_CVECTOR_GENERATOR,
    server_path: str | Path = DEFAULT_LLAMA_SERVER,
) -> ControlVectorPreflight:
    checks: list[PreflightCheck] = []
    warnings: list[str] = []
    errors: list[str] = []
    model_metadata: dict[str, Any] = {}
    model_architecture: str | None = None

    model = Path(model_path)
    if model.exists() and model.is_file():
        checks.append(PreflightCheck("GGUF model", "ok", str(model)))
        try:
            model_metadata = read_gguf_metadata(model)
            model_architecture = str(model_metadata.get("general.architecture") or "") or None
            if model_architecture:
                checks.append(PreflightCheck("GGUF architecture", "ok", model_architecture))
            else:
                warnings.append("GGUF metadata did not expose general.architecture.")
        except Exception as exc:
            warnings.append(f"Could not parse GGUF metadata: {exc}")
    else:
        errors.append(f"Model path does not exist: {model}")
        checks.append(PreflightCheck("GGUF model", "error", str(model)))

    for label, binary in [("llama-cvector-generator", generator_path), ("llama-server", server_path)]:
        path = Path(binary)
        if path.exists() and path.is_file():
            checks.append(PreflightCheck(label, "ok", str(path)))
        else:
            errors.append(f"{label} binary does not exist: {path}")
            checks.append(PreflightCheck(label, "error", str(path)))

    if positive_path and negative_path:
        pos = Path(positive_path)
        neg = Path(negative_path)
        if not pos.exists():
            errors.append(f"Positive prompt file does not exist: {pos}")
            checks.append(PreflightCheck("Positive prompts", "error", str(pos)))
        if not neg.exists():
            errors.append(f"Negative prompt file does not exist: {neg}")
            checks.append(PreflightCheck("Negative prompts", "error", str(neg)))
        if pos.exists() and neg.exists():
            pos_lines = _nonempty_lines(pos)
            neg_lines = _nonempty_lines(neg)
            if not pos_lines or not neg_lines:
                errors.append("Positive and negative prompt files must both contain at least one non-empty line.")
                checks.append(PreflightCheck("Prompt counts", "error", f"{len(pos_lines)} positive, {len(neg_lines)} negative"))
            elif len(pos_lines) != len(neg_lines):
                errors.append(f"Prompt files must have equal non-empty line counts: {len(pos_lines)} positive, {len(neg_lines)} negative.")
                checks.append(PreflightCheck("Prompt counts", "error", f"{len(pos_lines)} positive, {len(neg_lines)} negative"))
            else:
                checks.append(PreflightCheck("Prompt counts", "ok", f"{len(pos_lines)} matched pairs"))
    else:
        warnings.append("No prompt set selected yet.")

    warnings.extend(model_compatibility_warnings(model_architecture, model_metadata, model_path))
    return ControlVectorPreflight(checks, warnings, errors, model_architecture, model_metadata)


def classify_cvector_failure(stderr: str, stdout: str = "") -> CVectorFailure:
    combined = f"{stderr}\n{stdout}"
    warnings: list[str] = []
    if "radv is not a conformant Vulkan implementation" in combined:
        warnings.append("Vulkan radv warning was present; it is probably not the primary failure.")
    if "n_gpu_layers already set by user to 999" in combined:
        return CVectorFailure(
            "GPU layer auto-fit could not run because -ngl was explicitly set to 999.",
            "Leave generator GPU layers on auto, or set a lower explicit -ngl value.",
            warnings,
        )
    if "diff_filtered.size() == n_layers - 1" in combined:
        return CVectorFailure(
            "Captured layer-output count did not match llama-cvector-generator's expected layer count.",
            "Treat this as likely Qwen3.6 MoE/MTP cvector incompatibility; use the compatibility patch path or try a non-MoE GGUF.",
            warnings,
        )
    if "PCA iterations must" in combined:
        return CVectorFailure(
            "Invalid PCA settings.",
            "Make --pca-iter a multiple of --pca-batch, or switch method to mean.",
            warnings,
        )
    return CVectorFailure(
        "llama-cvector-generator exited unsuccessfully.",
        "Review stderr/stdout, then try auto GPU layers, --fit off, smaller context, or a different model.",
        warnings,
    )


def build_cvector_command(
    model_path: str | Path,
    positive_path: str | Path,
    negative_path: str | Path,
    output_path: str | Path,
    generator_path: str | Path = DEFAULT_CVECTOR_GENERATOR,
    method: str = "mean",
    ngl: int | None = None,
    fit: str | None = None,
    ctx_size: int | None = None,
    pca_batch: int | None = None,
    pca_iter: int | None = None,
) -> list[str]:
    command = [
        str(generator_path),
        "-m",
        str(model_path),
        "--positive-file",
        str(positive_path),
        "--negative-file",
        str(negative_path),
        "--method",
        method,
        "-o",
        str(output_path),
    ]
    if ngl is not None:
        command[3:3] = ["-ngl", str(int(ngl))]
    if fit:
        command.extend(["--fit", fit])
    if ctx_size:
        command.extend(["-c", str(int(ctx_size))])
    if pca_batch:
        command.extend(["--pca-batch", str(int(pca_batch))])
    if pca_iter:
        command.extend(["--pca-iter", str(int(pca_iter))])
    return command


def generate_control_vector(
    name: str,
    model_path: str | Path,
    positive_path: str | Path,
    negative_path: str | Path,
    generator_path: str | Path = DEFAULT_CVECTOR_GENERATOR,
    method: str = "mean",
    ngl: int | None = None,
    fit: str | None = None,
    ctx_size: int | None = None,
    pca_batch: int | None = None,
    pca_iter: int | None = None,
    compatibility_warnings: list[str] | None = None,
    model_architecture: str | None = None,
    timeout: float | None = None,
) -> ControlVectorMetadata:
    slug = safe_slug(name)
    CONTROL_VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    vector_path = CONTROL_VECTOR_DIR / f"{slug}.gguf"
    metadata_path = CONTROL_VECTOR_DIR / f"{slug}.json"
    command = build_cvector_command(
        model_path,
        positive_path,
        negative_path,
        vector_path,
        generator_path,
        method,
        ngl=ngl,
        fit=fit,
        ctx_size=ctx_size,
        pca_batch=pca_batch,
        pca_iter=pca_iter,
    )

    started = datetime.now(timezone.utc).isoformat()
    model_hash = file_sha256(model_path) if Path(model_path).exists() else None
    proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    failure = classify_cvector_failure(proc.stderr, proc.stdout) if proc.returncode != 0 else CVectorFailure("", "", [])
    metadata = ControlVectorMetadata(
        name=slug,
        vector_path=vector_path,
        model_path=str(model_path),
        model_sha256=model_hash,
        positive_path=str(positive_path),
        negative_path=str(negative_path),
        method=method,
        ngl=int(ngl) if ngl is not None else None,
        gpu_layer_mode="explicit" if ngl is not None else "auto",
        fit=fit,
        ctx_size=int(ctx_size) if ctx_size else None,
        pca_batch=int(pca_batch) if pca_batch else None,
        pca_iter=int(pca_iter) if pca_iter else None,
        model_architecture=model_architecture,
        compatibility_warnings=compatibility_warnings or [],
        created_at=started,
        generator_path=str(generator_path),
        command=command,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        failure_cause=failure.cause,
        recommended_action=failure.recommendation,
    )
    payload = asdict(metadata)
    payload["vector_path"] = str(metadata.vector_path)
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if proc.returncode != 0:
        raise ControlVectorRunError(metadata, failure)
    return metadata


def list_control_vectors() -> list[dict[str, Any]]:
    CONTROL_VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    names = {path.stem for path in CONTROL_VECTOR_DIR.glob("*.gguf")} | {path.stem for path in CONTROL_VECTOR_DIR.glob("*.json")}
    for name in sorted(names):
        vector_path = CONTROL_VECTOR_DIR / f"{name}.gguf"
        metadata_path = CONTROL_VECTOR_DIR / f"{name}.json"
        meta: dict[str, Any] = {}
        if metadata_path.exists():
            try:
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        rows.append(
            {
                "name": meta.get("name") or vector_path.stem,
                "vector_path": str(vector_path),
                "vector_exists": vector_path.exists(),
                "model_path": meta.get("model_path", ""),
                "method": meta.get("method", ""),
                "ngl": meta.get("ngl", ""),
                "gpu_layer_mode": meta.get("gpu_layer_mode", ""),
                "model_architecture": meta.get("model_architecture", ""),
                "created_at": meta.get("created_at", ""),
                "returncode": meta.get("returncode"),
                "failure_cause": meta.get("failure_cause", ""),
                "recommended_action": meta.get("recommended_action", ""),
            }
        )
    return rows


def build_llama_server_command(
    model_path: str | Path,
    vector_path: str | Path | None = None,
    strength: float = 1.25,
    layer_start: int | None = None,
    layer_end: int | None = None,
    server_path: str | Path = DEFAULT_LLAMA_SERVER,
    host: str = "127.0.0.1",
    port: int = 8088,
    ngl: int = 999,
    ctx_size: int | None = None,
    extra_args: str = "",
    alias: str | None = None,
) -> list[str]:
    command = [
        str(server_path),
        "-m",
        str(model_path),
        "--host",
        host,
        "--port",
        str(int(port)),
        "-ngl",
        str(int(ngl)),
    ]
    if ctx_size:
        command.extend(["-c", str(int(ctx_size))])
    if alias and alias.strip():
        command.extend(["--alias", alias.strip()])
    if vector_path:
        command.extend(["--control-vector-scaled", f"{vector_path}:{float(strength):g}"])
    if vector_path and layer_start is not None and layer_end is not None:
        command.extend(["--control-vector-layer-range", str(int(layer_start)), str(int(layer_end))])
    if extra_args.strip():
        command.extend(shlex.split(extra_args))
    return command
