# Authors: LCC Research Lab
# Project: From Document to Evidence
"""Read-only map from Book Three Freshman subjects to saved artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from inspector_engine.effective_lora import measure_effective_lora
from inspector_engine.inspect_checkpoint import inspect_checkpoint
from inspector_engine.model_library import format_bytes
from inspector_engine.roots import present_path
from inspector_engine.weight_statistics import measure_weight_statistics

ENGINE_DIR = Path(__file__).resolve().parent.parent
REGISTRY_FILE = ENGINE_DIR / "research" / "book_three_freshman.json"
ROOT_FILE = ENGINE_DIR / "research_root.txt"

ARTIFACT_LABELS = {
    "base_weights": "Starting model weights",
    "lora_adapter": "LoRA adapter",
    "full_checkpoint": "Full checkpoint",
    "unknown": "Unknown",
}
STUDENT_NOTES = {
    "not_saved": "Not saved",
    "not_applicable": "This is the starting model, not a trained student",
    "present": "A full student checkpoint is recorded",
}

_HASH_CACHE: dict[tuple[str, int, int], str] = {}


class RegistryError(Exception):
    def __init__(self, code: str, message: str, status: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def list_freshman(*, registry: Path | None = None, root: Path | None = None, configured: bool | None = None) -> dict:
    document = _load_registry(registry)
    resolved_root, root_status = _root(root, configured)
    subjects = [
        _public_subject(row, resolved_root, root_status, document)
        for row in document["subjects"]
        if isinstance(row, dict)
    ]
    return {
        "ok": True,
        "study": document.get("study"),
        "base_model": document.get("base_model"),
        "record": document.get("record"),
        "limits": document.get("limits"),
        "research_root": str(resolved_root) if resolved_root is not None else None,
        "root_status": root_status,
        "subjects": subjects,
    }


def registered_subject_id(path_text: str) -> str | None:
    """Match a checkpoint path to a research subject without hashing the file."""
    text = (path_text or "").strip()
    if not text:
        return None
    try:
        resolved = Path(text).expanduser().resolve()
    except OSError:
        return None
    try:
        root, status = _root(None, None)
        document = _load_registry(None)
    except RegistryError:
        return None
    for row in document.get("subjects", []):
        if not isinstance(row, dict):
            continue
        location, _size = _locate(root, status, str(row.get("relative_path") or ""))
        subject_id = row.get("id")
        if location is not None and subject_id and location.resolve() == resolved:
            return str(subject_id)
    return None


def open_freshman(subject_id: str, *, registry: Path | None = None, root: Path | None = None, configured: bool | None = None) -> dict:
    listed = list_freshman(registry=registry, root=root, configured=configured)
    subject = next((row for row in listed["subjects"] if row["id"] == subject_id), None)
    if subject is None:
        raise RegistryError("unknown_subject", "That research subject is not in the Freshman registry.", 404)
    if subject["hash_status"] != "verified_match" or not subject["location"]:
        raise RegistryError("artifact_not_verified", subject["status"], 409)
    report = inspect_checkpoint(subject["location"], [Path(subject["location"]).parent])
    report["research"] = {
        "subject_id": subject["id"],
        "hash_status": subject["hash_status"],
        "expected_sha256": subject["expected_sha256"],
        "observed_sha256": subject["observed_sha256"],
        "artifact_type": subject["artifact_type"],
        "artifact_label": subject["artifact_label"],
        "full_student_checkpoint": subject["full_student_checkpoint"],
    }
    report["research"]["provenance"] = subject.get("provenance")
    return report


def measure_freshman(subject_id: str, *, registry: Path | None = None, root: Path | None = None, configured: bool | None = None) -> dict:
    listed = list_freshman(registry=registry, root=root, configured=configured)
    subject = next((row for row in listed["subjects"] if row["id"] == subject_id), None)
    if subject is None:
        raise RegistryError("unknown_subject", "That research subject is not in the Freshman registry.", 404)
    if subject["hash_status"] != "verified_match" or not subject["location"]:
        raise RegistryError("artifact_not_verified", subject["status"], 409)
    report = measure_weight_statistics(subject["location"], [Path(subject["location"]).parent])
    report["research"] = {
        "subject_id": subject["id"],
        "hash_status": subject["hash_status"],
        "artifact_type": subject["artifact_type"],
        "full_student_checkpoint": subject["full_student_checkpoint"],
        "provenance": subject.get("provenance"),
    }
    return report


def measure_freshman_effective(subject_id: str, *, registry: Path | None = None, root: Path | None = None, configured: bool | None = None) -> dict:
    subject, adapter, research_config, packaged_config, base = _verified_freshman_adapter(
        subject_id, registry=registry, root=root, configured=configured
    )
    report = measure_effective_lora(adapter, research_config, packaged_config, base)
    report["research"] = {
        "subject_id": subject["id"],
        "hash_status": subject["hash_status"],
        "artifact_type": subject["artifact_type"],
        "full_student_checkpoint": subject["full_student_checkpoint"],
        "provenance": subject.get("provenance"),
    }
    return report


def freshman_update_factors(subject_id: str, *, registry: Path | None = None, root: Path | None = None, configured: bool | None = None) -> dict:
    """Factors for a verified research adapter. Used to compare update direction."""
    from inspector_engine.effective_lora import load_update_factors

    subject, adapter, research_config, packaged_config, _base = _verified_freshman_adapter(
        subject_id, registry=registry, root=root, configured=configured
    )
    if subject["artifact_type"] != "lora_adapter":
        raise RegistryError("not_an_adapter", f"{subject_id} is not a LoRA adapter. Direction is defined for an effective update.", 400)
    loaded = load_update_factors(adapter, research_config, packaged_config)
    loaded["subject_id"] = subject["id"]
    loaded["name"] = adapter.name
    loaded["path"] = str(adapter)
    return loaded


def _verified_freshman_adapter(
    subject_id: str, *, registry: Path | None, root: Path | None, configured: bool | None
) -> tuple[dict, Path, Path, Path, Path | None]:
    listed = list_freshman(registry=registry, root=root, configured=configured)
    subject = next((row for row in listed["subjects"] if row["id"] == subject_id), None)
    if subject is None:
        raise RegistryError("unknown_subject", "That research subject is not in the Freshman registry.", 404)
    if subject["hash_status"] != "verified_match" or not subject["location"]:
        raise RegistryError("artifact_not_verified", subject["status"], 409)
    provenance = subject.get("provenance") or {}
    original = (provenance.get("original_configuration") or {}).get("sha256")
    adapter = Path(subject["location"])
    research_config = adapter.parent / "adapter_config.research.json"
    recorded = _sha256(research_config)
    if not original or recorded != original:
        raise RegistryError(
            "config_provenance",
            "The research configuration file does not match the recorded original SHA-256.",
            409,
        )
    base_info = provenance.get("base_weights") or {}
    base = Path(base_info["location"]) if base_info.get("status") == "verified_match" and base_info.get("location") else None
    return subject, adapter, research_config, adapter.parent / "adapter_config.json", base


def _load_registry(registry: Path | None) -> dict:
    path = registry or REGISTRY_FILE
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError("registry_unreadable", "The Freshman research registry cannot be read.", 500) from exc
    if not isinstance(document, dict) or not isinstance(document.get("subjects"), list):
        raise RegistryError("registry_unreadable", "The Freshman research registry is not a subject list.", 500)
    return document


def _root(root: Path | None, configured: bool | None) -> tuple[Path | None, str]:
    if root is not None:
        return (root if root.is_dir() else None, "configured" if configured is not False else "not_configured")
    if configured is False:
        return None, "not_configured"
    env = os.environ.get("INSPECTOR_RESEARCH_ROOT", "").strip()
    text = env or _root_file()
    if not text:
        return None, "not_configured"
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = ENGINE_DIR / path
    if not path.is_dir():
        return None, "root_missing"
    return path.resolve(), "configured"


def _root_file() -> str:
    if not ROOT_FILE.is_file():
        return ""
    for line in ROOT_FILE.read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if text:
            return text
    return ""


def _location_fields(location: Path | None) -> dict[str, str | None]:
    shown = present_path(location)
    return {
        "location_label": shown["path_label"],
        "location_display": shown["path_display"],
        "location_scope": shown["path_scope"],
    }


def _public_subject(row: dict, root: Path | None, root_status: str, document: dict) -> dict:
    artifact_type = str(row.get("artifact_type") or "unknown")
    student = str(row.get("full_student_checkpoint") or "not_saved")
    expected = _hex_or_none(row.get("expected_sha256"))
    relative = str(row.get("relative_path") or "")
    location, size = _locate(root, root_status, relative)
    observed = _sha256(location) if location is not None else None
    hash_status = _hash_status(root_status, location, expected, observed)
    return {
        "id": row.get("id"),
        "experiment": row.get("experiment"),
        "experiment_role": row.get("experiment_role"),
        "unit_id": row.get("unit_id"),
        "seed": row.get("seed"),
        "epoch": row.get("epoch"),
        "base_model": "SmolLM2-1.7B-Instruct",
        "artifact_type": artifact_type,
        "artifact_label": ARTIFACT_LABELS.get(artifact_type, ARTIFACT_LABELS["unknown"]),
        "full_student_checkpoint": student,
        "full_student_label": STUDENT_NOTES.get(student, student),
        "relative_path": relative,
        "location": str(location) if location is not None else None,
        **_location_fields(location),
        "file_name": Path(relative).name if relative else None,
        "size_bytes": size,
        "size_human": format_bytes(size),
        "expected_sha256": expected,
        "observed_sha256": observed,
        "hash_status": hash_status,
        "inspectable": hash_status == "verified_match",
        "status": _status_text(hash_status),
        "provenance": _provenance(row, location, root, root_status, document),
    }


def _provenance(row: dict, location: Path | None, root: Path | None, root_status: str, document: dict) -> dict:
    expected = _hex_or_none(row.get("expected_sha256"))
    observed = _sha256(location) if location is not None else None
    weight = {
        "identity": "stored weights",
        "expected_sha256": expected,
        "observed_sha256": observed,
        "status": _hash_status(root_status, location, expected, observed),
        "note": "This hash identifies the weight file only.",
    }
    if str(row.get("artifact_type") or "") != "lora_adapter":
        return {
            "weight": weight,
            "original_configuration": None,
            "packaged_configuration": None,
            "base_weights": None,
        }
    original = _hex_or_none(row.get("original_config_sha256"))
    packaged_path = location.parent / "adapter_config.json" if location is not None else None
    packaged = _sha256(packaged_path) if packaged_path is not None and packaged_path.is_file() else None
    matches = packaged == original if packaged is not None and original is not None else None
    return {
        "weight": weight,
        "original_configuration": {
            "identity": "research provenance",
            "sha256": original,
            "note": "SHA-256 of the original research configuration. This is not the hash of the packaged file.",
        },
        "packaged_configuration": {
            "identity": "distribution provenance",
            "sha256": packaged,
            "matches_original": matches,
            "note": (
                "SHA-256 of the configuration shipped in this package. "
                "A difference from the research record means this copy is not byte-identical to the original."
            ),
        },
        "base_weights": _base_weights(packaged_path, root, root_status, document),
    }


def _base_weights(config_path: Path | None, root: Path | None, root_status: str, document: dict) -> dict:
    note = "The base-model path in the configuration is not evidence. This status comes from hashing the file."
    base_row = next((item for item in document.get("subjects", []) if isinstance(item, dict) and item.get("id") == "BASE"), None)
    expected = _hex_or_none(base_row.get("expected_sha256")) if isinstance(base_row, dict) else None
    if config_path is None or not config_path.is_file() or root is None or root_status != "configured":
        return {"identity": "starting model file", "status": "not_verified", "expected_sha256": expected, "observed_sha256": None, "location": None, "note": note}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"identity": "starting model file", "status": "not_verified", "expected_sha256": expected, "observed_sha256": None, "location": None, "note": note}
    raw = data.get("base_model_name_or_path") if isinstance(data, dict) else None
    if not isinstance(raw, str) or not raw.strip():
        return {"identity": "starting model file", "status": "not_verified", "expected_sha256": expected, "observed_sha256": None, "location": None, "note": note}
    candidate = Path(raw.strip()).expanduser()
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    try:
        resolved = candidate.resolve()
    except OSError:
        resolved = candidate
    weight = resolved / "model.safetensors" if resolved.is_dir() or not resolved.suffix else resolved
    try:
        weight.relative_to(root.resolve())
    except ValueError:
        return {"identity": "starting model file", "status": "not_in_package", "expected_sha256": expected, "observed_sha256": None, "location": None, "note": note}
    if not weight.is_file():
        return {"identity": "starting model file", "status": "missing", "expected_sha256": expected, "observed_sha256": None, "location": None, "note": note}
    observed = _sha256(weight)
    status = "verified_match" if expected is not None and observed == expected else "mismatch" if expected is not None else "unreferenced"
    return {
        "identity": "starting model file",
        "status": status,
        "expected_sha256": expected,
        "observed_sha256": observed,
        "location": str(weight),
        **_location_fields(weight),
        "note": note,
    }


def _locate(root: Path | None, root_status: str, relative: str) -> tuple[Path | None, int | None]:
    if root is None or root_status != "configured" or not relative:
        return None, None
    try:
        relative_path = Path(relative)
    except (TypeError, ValueError):
        return None, None
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return None, None
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None, None
    if not candidate.is_file():
        return None, None
    return candidate, candidate.stat().st_size


def _sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    cached = _HASH_CACHE.get(key)
    if cached:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _HASH_CACHE[key] = value
    return value


def _hash_status(root_status: str, location: Path | None, expected: str | None, observed: str | None) -> str:
    if root_status == "not_configured":
        return "not_configured"
    if root_status == "root_missing" or location is None:
        return "missing"
    if expected is None:
        return "unreferenced"
    if observed == expected:
        return "verified_match"
    return "mismatch"


def _status_text(status: str) -> str:
    if status == "verified_match":
        return "Available. The SHA-256 matches the research record."
    if status == "mismatch":
        return "A file is present, and its SHA-256 does not match the research record."
    if status == "unreferenced":
        return "A file is present. Its SHA-256 was calculated, and no trusted reference is recorded."
    if status == "not_configured":
        return "The research directory is not configured."
    return "Not on disk."


def _hex_or_none(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if len(text) == 64 and all(character in "0123456789abcdef" for character in text):
        return text
    return None
