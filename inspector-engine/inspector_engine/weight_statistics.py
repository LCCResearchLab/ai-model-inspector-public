# Authors: LCC Research Lab
# Project: From Document to Evidence
"""Read-only statistics of stored checkpoint values.

A and B are measured as separate populations. This instrument does not
compute the scaled product BA and does not interpret a norm as learning.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

from inspector_engine.inspect_checkpoint import InspectionError, _sha256
from inspector_engine.roots import PathRejected, present_path, resolve_checkpoint
from inspector_engine.safetensors_meta import MalformedSafetensors, read_header, tensor_records

LIMITS = (
    "Minimum, maximum, mean, and population standard deviation use finite values only. "
    "L1 is the sum of absolute finite values. L2 is the square root of the sum of their squares. "
    "A zero is an exact 0. NaN and infinity are counted separately and left out of the numeric summaries. "
    "LoRA A and LoRA B are separate populations. "
    "These statistics are not the scaled product BA, and they are not a measurement of learning."
)


class _Accumulator:
    def __init__(self) -> None:
        self.parameter_count = 0
        self.finite_count = 0
        self.zero_count = 0
        self.nan_count = 0
        self.inf_count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.total_abs = 0.0
        self.minimum: float | None = None
        self.maximum: float | None = None

    def add(self, value: float) -> None:
        self.parameter_count += 1
        if math.isnan(value):
            self.nan_count += 1
            return
        if math.isinf(value):
            self.inf_count += 1
            return
        self.finite_count += 1
        self.total += value
        self.total_sq += value * value
        self.total_abs += abs(value)
        if value == 0.0:
            self.zero_count += 1
        if self.minimum is None or value < self.minimum:
            self.minimum = value
        if self.maximum is None or value > self.maximum:
            self.maximum = value

    def public(self) -> dict:
        finite = self.finite_count
        mean = self.total / finite if finite else None
        if finite:
            variance = self.total_sq / finite - (mean or 0.0) ** 2
            if variance < 0:
                variance = 0.0
            standard_deviation = math.sqrt(variance)
            l2_norm = math.sqrt(self.total_sq)
        else:
            standard_deviation = None
            l2_norm = None
        return {
            "parameter_count": self.parameter_count,
            "finite_count": finite,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": mean,
            "standard_deviation": standard_deviation,
            "l1_norm": self.total_abs if finite else None,
            "l2_norm": l2_norm,
            "zero_count": self.zero_count,
            "nonzero_count": finite - self.zero_count,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
        }


def measure_weight_statistics(path_text: str, roots: list[Path] | None = None) -> dict:
    try:
        path = resolve_checkpoint(path_text, roots)
    except PathRejected as exc:
        raise InspectionError(exc.code, exc.message, exc.status) from exc
    try:
        measured = _measure(path)
    except MalformedSafetensors as exc:
        raise InspectionError("malformed_safetensors", str(exc), 400) from exc
    except OSError as exc:
        raise InspectionError("missing_file", "The checkpoint file could not be read.", 404) from exc
    measured["ok"] = True
    measured["instrument"] = "weight_statistics"
    measured["limits"] = LIMITS
    measured["checkpoint"] = {
        "name": path.name,
        "path": str(path),
        **present_path(path),
        "sha256": _sha256(path),
    }
    return measured


def _measure(path: Path) -> dict:
    header, header_len = read_header(path)
    data_size = path.stat().st_size - 8 - header_len
    records, _metadata = tensor_records(header, data_size)
    with path.open("rb") as handle:
        handle.seek(8 + header_len)
        blob = handle.read(data_size)
    groups = {"A": _Accumulator(), "B": _Accumulator(), "other": _Accumulator()}
    tensors = []
    described = {row["name"]: row for row in records}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        role = _role(name)
        values = _values(name, info, blob)
        summary = _Accumulator()
        group = groups[role]
        for value in values:
            summary.add(value)
            group.add(value)
        tensors.append({"name": name, "role": role, "dtype": described[name]["dtype"], **summary.public()})
    tensors.sort(key=lambda row: row["name"])
    has_lora = any(row["role"] in {"A", "B"} for row in tensors)
    groups_out = []
    for role in ("A", "B", "other"):
        members = [row for row in tensors if row["role"] == role]
        if role == "other" and not members:
            continue
        if role in {"A", "B"} and not has_lora:
            continue
        groups_out.append({"role": role, "tensor_count": len(members), **groups[role].public()})
    return {"groups": groups_out, "tensors": tensors}


def _role(name: str) -> str:
    if name.endswith("lora_A.weight"):
        return "A"
    if name.endswith("lora_B.weight"):
        return "B"
    return "other"


def _values(name: str, info: object, blob: bytes) -> list[float]:
    if not isinstance(info, dict):
        raise MalformedSafetensors(f"Tensor {name} is not described by an object.")
    dtype = info.get("dtype")
    offsets = info.get("data_offsets")
    if not isinstance(dtype, str) or not isinstance(offsets, list) or len(offsets) != 2:
        raise MalformedSafetensors(f"Tensor {name} cannot be measured.")
    start, end = offsets
    raw = blob[start:end]
    if dtype == "F32":
        count = len(raw) // 4
        return list(struct.unpack(f"<{count}f", raw))
    if dtype == "F64":
        count = len(raw) // 8
        return list(struct.unpack(f"<{count}d", raw))
    if dtype == "F16":
        count = len(raw) // 2
        return [_f16(item) for item in struct.unpack(f"<{count}H", raw)]
    if dtype == "BF16":
        count = len(raw) // 2
        return [_bf16(item) for item in struct.unpack(f"<{count}H", raw)]
    raise InspectionError(
        "unsupported_dtype",
        f"Weight statistics cannot read {dtype} values in {name}.",
        400,
    )


def _f16(value: int) -> float:
    sign = (value >> 15) & 1
    exponent = (value >> 10) & 0x1F
    fraction = value & 0x3FF
    if exponent == 0:
        magnitude = (fraction / 1024.0) * 2.0**-14
    elif exponent == 31:
        if fraction == 0:
            return math.copysign(math.inf, -1.0 if sign else 1.0)
        return math.nan
    else:
        magnitude = (1.0 + fraction / 1024.0) * 2.0 ** (exponent - 15)
    return -magnitude if sign else magnitude


def _bf16(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value << 16))[0]
