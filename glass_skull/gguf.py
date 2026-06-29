from __future__ import annotations

import struct
from pathlib import Path
from typing import Any


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


def _read_gguf_string(f) -> str:
    size_raw = f.read(8)
    if len(size_raw) != 8:
        raise ValueError("short GGUF string length")
    size = struct.unpack("<Q", size_raw)[0]
    data = f.read(size)
    if len(data) != size:
        raise ValueError("short GGUF string data")
    return data.decode("utf-8", errors="replace")


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


def _skip_gguf_value(f, value_type: int) -> None:
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
