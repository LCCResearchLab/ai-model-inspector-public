# Authors: LCC Research Lab
# Project: From Document to Evidence

import json
import math
import struct
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from inspector_engine.research_registry import measure_freshman
from inspector_engine.weight_statistics import measure_weight_statistics
from server import app


def write_safetensors(path: Path, tensors: list[tuple[str, str, list[int], bytes]]) -> None:
    header: dict = {}
    offset = 0
    blobs = []
    for name, dtype, shape, raw in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    pad = (8 - (len(raw_header) % 8)) % 8
    raw_header += b" " * pad
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + b"".join(blobs))


def test_a_and_b_are_measured_separately_and_ba_is_not_computed(tmp_path: Path) -> None:
    path = tmp_path / "adapter_model.safetensors"
    write_safetensors(
        path,
        [
            ("layer.lora_A.weight", "F32", [2, 2], struct.pack("<4f", 1.0, -3.0, 0.0, 2.0)),
            ("layer.lora_B.weight", "F32", [2], struct.pack("<2f", 0.0, 0.0)),
            ("other.weight", "F32", [3], struct.pack("<3f", math.nan, math.inf, 1.0)),
        ],
    )
    report = measure_weight_statistics(str(path), [tmp_path])
    assert "scaled_update" not in report
    assert report["limits"].find("not the scaled product BA") != -1
    groups = {row["role"]: row for row in report["groups"]}
    a_values = groups["A"]
    assert a_values["tensor_count"] == 1
    assert a_values["parameter_count"] == 4
    assert a_values["minimum"] == -3.0
    assert a_values["maximum"] == 2.0
    assert a_values["mean"] == 0.0
    assert a_values["l1_norm"] == 6.0
    assert a_values["l2_norm"] == math.sqrt(14.0)
    assert a_values["standard_deviation"] == math.sqrt(3.5)
    assert a_values["zero_count"] == 1
    assert a_values["nonzero_count"] == 3
    assert groups["B"]["l2_norm"] == 0.0
    assert groups["B"]["zero_count"] == 2
    assert groups["B"]["nonzero_count"] == 0
    assert groups["other"]["nan_count"] == 1
    assert groups["other"]["inf_count"] == 1
    assert groups["other"]["finite_count"] == 1
    assert groups["other"]["mean"] == 1.0
    roles = {row["name"]: row["role"] for row in report["tensors"]}
    assert roles["layer.lora_A.weight"] == "A"
    assert roles["layer.lora_B.weight"] == "B"


def test_api_measures_a_verified_subject_and_refuses_a_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "adapter_model.safetensors"
    write_safetensors(path, [("layer.lora_A.weight", "F32", [1], struct.pack("<f", 2.0))])
    digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "study": "book-three-freshman",
                "limits": "separate",
                "subjects": [
                    {
                        "id": "F5-1",
                        "artifact_type": "lora_adapter",
                        "relative_path": "adapter_model.safetensors",
                        "expected_sha256": digest,
                        "full_student_checkpoint": "not_saved",
                    },
                    {
                        "id": "F5-2",
                        "artifact_type": "lora_adapter",
                        "relative_path": "adapter_model.safetensors",
                        "expected_sha256": "c" * 64,
                        "full_student_checkpoint": "not_saved",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("inspector_engine.research_registry.REGISTRY_FILE", registry)
    monkeypatch.setattr("inspector_engine.research_registry._root", lambda root=None, configured=None: (tmp_path, "configured"))
    measured = measure_freshman("F5-1")
    assert measured["groups"][0]["role"] == "A"
    assert measured["groups"][0]["mean"] == 2.0
    client = TestClient(app)
    refused = client.post("/research/book-three/freshman/F5-2/weight-statistics")
    assert refused.status_code == 409


def test_recorded_adapters_keep_a_and_b_separate() -> None:
    reports = []
    for subject_id in ("F5-1", "F5-2", "F5-3", "F6"):
        try:
            reports.append(measure_freshman(subject_id))
        except Exception as exc:
            if getattr(exc, "code", "") in {"artifact_not_verified"}:
                pytest.skip("research artifacts are not on this disk")
            raise
    for report in reports:
        groups = {row["role"]: row for row in report["groups"]}
        assert set(groups) == {"A", "B"}
        assert groups["A"]["tensor_count"] == 96
        assert groups["B"]["tensor_count"] == 96
        assert groups["A"]["parameter_count"] == 96 * 16 * 2048
        assert groups["B"]["parameter_count"] == 96 * 2048 * 16
        assert groups["A"]["nan_count"] == 0
        assert groups["B"]["nan_count"] == 0
        assert groups["A"]["inf_count"] == 0
        assert groups["B"]["inf_count"] == 0
        assert "scaled_update" not in report
