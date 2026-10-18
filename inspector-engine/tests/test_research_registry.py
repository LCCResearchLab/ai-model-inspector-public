# Authors: LCC Research Lab
# Project: From Document to Evidence

import hashlib
import json
import struct
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from inspector_engine.research_registry import RegistryError, list_freshman, open_freshman, registered_subject_id
from server import app

EXPECTED = "a" * 64


def write_safetensors(path: Path, raw: bytes = b"\x00" * 24) -> None:
    header = {"layer.weight": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, len(raw)]}}
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    pad = (8 - (len(raw_header) % 8)) % 8
    raw_header += b" " * pad
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + raw)


def registry(tmp_path: Path, subjects: list[dict]) -> Path:
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "study": "book-three-freshman",
                "base_model": "SmolLM2-1.7B-Instruct",
                "limits": "Does not reconstruct a student.",
                "subjects": subjects,
            }
        ),
        encoding="utf-8",
    )
    return path


def subject(relative: str, digest: str | None = EXPECTED, artifact: str = "lora_adapter") -> dict:
    return {
        "id": "F5-1",
        "experiment": "Book Two, Chapter 5",
        "experiment_role": "First Method training run",
        "unit_id": "CH5|Freshman|method|22001001|3",
        "seed": 22001001,
        "epoch": 3,
        "artifact_type": artifact,
        "relative_path": relative,
        "expected_sha256": digest,
        "full_student_checkpoint": "not_saved",
    }


def test_matching_file_is_a_verified_lora_adapter(tmp_path: Path) -> None:
    weight = tmp_path / "adapters" / "adapter_model.safetensors"
    weight.parent.mkdir()
    write_safetensors(weight)
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    before = weight.stat().st_mtime_ns
    listed = list_freshman(registry=registry(tmp_path, [subject("adapters/adapter_model.safetensors", digest)]), root=tmp_path)
    row = listed["subjects"][0]
    assert row["hash_status"] == "verified_match"
    assert row["inspectable"] is True
    assert row["artifact_label"] == "LoRA adapter"
    assert row["full_student_label"] == "Not saved"
    opened = open_freshman("F5-1", registry=registry(tmp_path, [subject("adapters/adapter_model.safetensors", digest)]), root=tmp_path)
    assert opened["ok"] is True
    assert opened["research"]["hash_status"] == "verified_match"
    assert opened["research"]["observed_sha256"] == digest
    assert weight.stat().st_mtime_ns == before


def test_mismatch_missing_and_unreferenced_are_not_inspectable(tmp_path: Path) -> None:
    weight = tmp_path / "adapter_model.safetensors"
    write_safetensors(weight, b"\x01" * 24)
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    rows = list_freshman(
        registry=registry(
            tmp_path,
            [
                subject("adapter_model.safetensors", "b" * 64),
                {**subject("absent.safetensors", digest), "id": "F5-2"},
                {**subject("adapter_model.safetensors", None), "id": "F6"},
                {**subject("../outside.safetensors", digest), "id": "F5-3"},
            ],
        ),
        root=tmp_path,
    )["subjects"]
    by_id = {row["id"]: row for row in rows}
    assert by_id["F5-1"]["hash_status"] == "mismatch"
    assert by_id["F5-1"]["inspectable"] is False
    assert by_id["F5-1"]["observed_sha256"] == digest
    assert by_id["F5-2"]["hash_status"] == "missing"
    assert by_id["F6"]["hash_status"] == "unreferenced"
    assert by_id["F6"]["observed_sha256"] == digest
    assert by_id["F5-3"]["location"] is None
    with pytest.raises(RegistryError) as caught:
        open_freshman("F5-1", registry=registry(tmp_path, [subject("adapter_model.safetensors", "b" * 64)]), root=tmp_path)
    assert caught.value.status == 409


def test_unconfigured_root_keeps_the_identity_and_disables_inspection(tmp_path: Path) -> None:
    listed = list_freshman(registry=registry(tmp_path, [subject("adapter_model.safetensors")]), root=tmp_path, configured=False)
    row = listed["subjects"][0]
    assert listed["root_status"] == "not_configured"
    assert row["hash_status"] == "not_configured"
    assert row["inspectable"] is False
    assert row["expected_sha256"] == EXPECTED


def test_api_reports_a_verified_subject_and_refuses_a_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weight = tmp_path / "model.safetensors"
    write_safetensors(weight)
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    path = registry(
        tmp_path,
        [
            subject("model.safetensors", digest, "base_weights") | {"id": "BASE", "full_student_checkpoint": "not_applicable"},
            subject("model.safetensors", "c" * 64) | {"id": "F5-1"},
        ],
    )
    monkeypatch.setattr("inspector_engine.research_registry.REGISTRY_FILE", path)
    monkeypatch.setattr("inspector_engine.research_registry._root", lambda root=None, configured=None: (tmp_path, "configured"))
    client = TestClient(app)
    listed = client.get("/research/book-three/freshman")
    assert listed.status_code == 200
    body = listed.json()
    assert [row["id"] for row in body["subjects"]] == ["BASE", "F5-1"]
    assert body["subjects"][0]["artifact_label"] == "Starting model weights"
    assert body["subjects"][0]["inspectable"] is True
    assert body["subjects"][1]["inspectable"] is False
    opened = client.post("/research/book-three/freshman/BASE/inspect")
    assert opened.status_code == 200
    assert opened.json()["research"]["observed_sha256"] == digest
    refused = client.post("/research/book-three/freshman/F5-1/inspect")
    assert refused.status_code == 409
    assert refused.json()["ok"] is False


def test_recorded_freshman_subjects_match_files_on_disk() -> None:
    listed = list_freshman()
    if listed["root_status"] != "configured":
        pytest.skip("research directory is not configured")
    subjects = listed["subjects"]
    assert [row["id"] for row in subjects] == ["BASE", "F5-1", "F5-2", "F5-3", "F6"]
    if any(row["hash_status"] == "missing" for row in subjects):
        pytest.skip("research artifacts are not on this disk")
    assert [row["hash_status"] for row in subjects] == ["verified_match"] * 5
    distribution = Path(__file__).resolve().parents[1] / "distribution" / "book-three" / "freshman"
    assert listed["research_root"] == str(distribution)
    assert all(str(row["location"]).startswith(str(distribution)) for row in subjects)
    assert subjects[0]["artifact_type"] == "base_weights"
    assert [row["artifact_type"] for row in subjects[1:]] == ["lora_adapter"] * 4
    assert [row["full_student_checkpoint"] for row in subjects[1:]] == ["not_saved"] * 4
    opened = open_freshman("F5-1")
    assert opened["checkpoint"]["name"] == "adapter_model.safetensors"
    assert opened["research"]["expected_sha256"].startswith("001ff8f0")
    provenance = subjects[1]["provenance"]
    assert provenance["weight"]["status"] == "verified_match"
    assert provenance["original_configuration"]["sha256"].startswith("4c0458bd")
    assert provenance["packaged_configuration"]["matches_original"] is False
    assert provenance["packaged_configuration"]["sha256"] != provenance["original_configuration"]["sha256"]
    assert provenance["base_weights"]["status"] == "verified_match"
    location = provenance["base_weights"]["location"] or ""
    assert Path(location).name == "model.safetensors"


def test_a_base_path_string_does_not_count_as_the_base_file(tmp_path: Path) -> None:
    weight = tmp_path / "F5-1" / "adapter_model.safetensors"
    weight.parent.mkdir()
    write_safetensors(weight)
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    (weight.parent / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "../BASE"}),
        encoding="utf-8",
    )
    outside = weight.parent / "outside.json"
    outside.write_text(json.dumps({"base_model_name_or_path": "/tmp/not-the-package"}), encoding="utf-8")
    listed = list_freshman(
        registry=registry(tmp_path, [subject("F5-1/adapter_model.safetensors", digest) | {"original_config_sha256": "ab" * 32}]),
        root=tmp_path,
    )
    base = listed["subjects"][0]["provenance"]["base_weights"]
    assert base["status"] == "missing"
    assert base["observed_sha256"] is None
    assert listed["subjects"][0]["provenance"]["packaged_configuration"]["matches_original"] is False
    (weight.parent / "adapter_config.json").write_text(outside.read_text(encoding="utf-8"), encoding="utf-8")
    escaped = list_freshman(
        registry=registry(tmp_path, [subject("F5-1/adapter_model.safetensors", digest)]),
        root=tmp_path,
    )["subjects"][0]["provenance"]["base_weights"]
    assert escaped["status"] == "not_in_package"
    assert escaped["location"] is None


def test_any_checkpoint_path_can_identify_a_registered_subject(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weight = tmp_path / "adapter_model.safetensors"
    write_safetensors(weight)
    monkeypatch.setattr(
        "inspector_engine.research_registry.REGISTRY_FILE",
        registry(tmp_path, [subject("adapter_model.safetensors", "ab" * 32)]),
    )
    monkeypatch.setattr("inspector_engine.research_registry._root", lambda root=None, configured=None: (tmp_path, "configured"))
    assert registered_subject_id(str(weight)) == "F5-1"
    assert registered_subject_id(str(tmp_path / "missing.safetensors")) is None
