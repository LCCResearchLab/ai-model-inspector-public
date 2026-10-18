# Authors: LCC Research Lab
# Project: From Document to Evidence
"""Read a safetensors header without loading tensor bytes."""

from __future__ import annotations

import json
import struct
from pathlib import Path

HEADER_LIMIT = 100_000_000

DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U64": 8,
    "U32": 4,
    "U16": 2,
    "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
}

DTYPE_LABELS = {
    "F64": "FP64",
    "F32": "FP32",
    "F16": "FP16",
    "BF16": "BF16",
    "I64": "INT64",
    "I32": "INT32",
    "I16": "INT16",
    "I8": "INT8",
    "U64": "UINT64",
    "U32": "UINT32",
    "U16": "UINT16",
    "U8": "UINT8",
    "BOOL": "BOOL",
    "F8_E4M3": "FP8_E4M3",
    "F8_E5M2": "FP8_E5M2",
    "F8_E8M0": "FP8_E8M0",
}


class MalformedSafetensors(Exception):
    pass


def read_header(path: Path) -> tuple[dict, int]:
    """Return the JSON header and its byte length. Tensor bytes are not read."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) < 8:
            raise MalformedSafetensors("The file is shorter than a safetensors header.")
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len <= 0 or header_len > HEADER_LIMIT or header_len > size - 8:
            raise MalformedSafetensors("The safetensors header length is not valid.")
        raw = handle.read(header_len)
        if len(raw) != header_len:
            raise MalformedSafetensors("The safetensors header is truncated.")
    try:
        header = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MalformedSafetensors("The safetensors header is not valid JSON.") from exc
    if not isinstance(header, dict):
        raise MalformedSafetensors("The safetensors header is not an object.")
    return header, header_len


def tensor_records(header: dict, data_size: int) -> tuple[list[dict], dict[str, str]]:
    metadata_raw = header.get("__metadata__", {})
    if metadata_raw is None:
        metadata_raw = {}
    if not isinstance(metadata_raw, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in metadata_raw.items()
    ):
        raise MalformedSafetensors("The safetensors metadata is not a map of strings.")

    records = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(info, dict):
            raise MalformedSafetensors(f"Tensor {name!r} is not described by an object.")
        shape = info.get("shape")
        dtype = info.get("dtype")
        offsets = info.get("data_offsets")
        if not isinstance(shape, list) or not all(isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0 for dim in shape):
            raise MalformedSafetensors(f"Tensor {name} has no valid shape.")
        if not isinstance(dtype, str):
            raise MalformedSafetensors(f"Tensor {name} has no data type.")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in offsets)
            or offsets[1] < offsets[0]
        ):
            raise MalformedSafetensors(f"Tensor {name} has no valid data offsets.")
        count = 1
        for dim in shape:
            count *= dim
        byte_len = offsets[1] - offsets[0]
        if offsets[1] > data_size:
            raise MalformedSafetensors(f"Tensor {name} extends past the file.")
        item = DTYPE_BYTES.get(dtype)
        if item is not None and byte_len != count * item:
            raise MalformedSafetensors(
                f"Tensor {name} has a size that does not match its shape and data type."
            )
        records.append(
            {
                "name": name,
                "shape": shape,
                "dtype": DTYPE_LABELS.get(dtype, dtype),
                "safetensors_dtype": dtype,
                "parameter_count": count,
            }
        )
    records.sort(key=lambda row: row["name"])
    return records, {str(key): str(value) for key, value in metadata_raw.items()}
