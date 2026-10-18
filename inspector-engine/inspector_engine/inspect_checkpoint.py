# Authors: LCC Research Lab
# Project: From Document to Evidence
"""Read-only inventory of a local safetensors checkpoint."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from inspector_engine import ENGINE_VERSION
from inspector_engine.architecture import (
    VALIDATION_LIMITS,
    apply_layer_classification,
    classify_observed,
    load_declarations,
    merge_warnings,
    validation_warnings,
)
from inspector_engine.roots import PathRejected, configured_roots, present_path, resolve_checkpoint
from inspector_engine.safetensors_meta import MalformedSafetensors, read_header, tensor_records

ADAPTER_METADATA_KEYS = {
    "peft_type",
    "base_model_name_or_path",
    "lora_alpha",
    "target_modules",
}
BASE_METADATA_KEYS = {"model_type", "architectures"}


class InspectionError(Exception):
    def __init__(self, code: str, message: str, status: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def inspect_checkpoint(path_text: str, roots: list[Path] | None = None) -> dict:
    allowed = configured_roots() if roots is None else roots
    try:
        path = resolve_checkpoint(path_text, allowed)
    except PathRejected as exc:
        raise InspectionError(exc.code, exc.message, exc.status) from exc
    try:
        header, header_len = read_header(path)
        file_size = path.stat().st_size
        records, metadata = tensor_records(header, file_size - 8 - header_len)
    except MalformedSafetensors as exc:
        raise InspectionError("malformed_safetensors", str(exc), 400) from exc
    except OSError as exc:
        raise InspectionError("missing_file", "The checkpoint file could not be read.", 404) from exc

    checkpoint_type, basis = _classify(metadata)
    declared, adapter, referenced, read_warnings = load_declarations(path, allowed)
    observed = apply_layer_classification(classify_observed(records, metadata, adapter), declared)
    warnings = merge_warnings(read_warnings, validation_warnings(records, declared, adapter, referenced, observed))
    total = sum(row["parameter_count"] for row in records)
    adapter_evidence = checkpoint_type == "lora_adapter" or observed["classification"] == "adapter"
    scope = (
        "This file stores selected adapter tensors. The total is not the parameter count of the underlying base model."
        if adapter_evidence
        else "This total is the number of elements stored in this checkpoint."
    )
    return {
        "ok": True,
        "engine_version": ENGINE_VERSION,
        "inspected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checkpoint": {
            "name": path.name,
            "path": str(path),
            **present_path(path),
            "format": "safetensors",
            "size_bytes": file_size,
            "sha256": _sha256(path),
            "checkpoint_type": checkpoint_type,
            "checkpoint_type_basis": basis,
        },
        "configuration": {
            "source": "safetensors_metadata" if metadata else "unavailable",
            "metadata": metadata or None,
            "limits": (
                "These metadata fields are stored inside the checkpoint. "
                "They are separate from config.json and from the measured tensor inventory."
            ),
        },
        "declared_architecture": declared,
        "declared_adapter": adapter,
        "referenced_base_architecture": referenced,
        "observed_checkpoint": observed,
        "validation": {"warnings": warnings, "limits": VALIDATION_LIMITS},
        "tensors": records,
        "tensor_count": len(records),
        "total_stored_parameters": total,
        "parameter_scope": scope,
    }


def _classify(metadata: dict[str, str]) -> tuple[str, str]:
    keys = {key.lower() for key in metadata}
    if keys & {name.lower() for name in ADAPTER_METADATA_KEYS}:
        return "lora_adapter", "safetensors_metadata"
    if keys & {name.lower() for name in BASE_METADATA_KEYS}:
        return "base", "safetensors_metadata"
    return "unknown", "unavailable"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
