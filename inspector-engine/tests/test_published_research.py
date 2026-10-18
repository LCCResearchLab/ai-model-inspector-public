# Authors: LCC Research Lab
# Project: From Document to Evidence

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from inspector_engine.research_registry import RegistryError, VALIDATION_FILE, list_freshman, open_freshman, published_research_results
from server import app

RECORD = json.loads(VALIDATION_FILE.read_text(encoding="utf-8"))
SUBJECT_KEYS = (
    "id",
    "weight_sha256",
    "rank",
    "lora_alpha",
    "scale",
    "pair_count",
    "effective_update_norm",
    "relative_to_base",
)


def _expected_subjects() -> list[dict]:
    return [{key: row[key] for key in SUBJECT_KEYS} for row in RECORD["subjects"]]


def test_published_results_match_the_validation_record() -> None:
    published = published_research_results()
    assert published["label"] == "Recorded Research Results"
    assert published["record"] == RECORD["record"]
    assert published["measured_on"] == RECORD["measured_on"]
    assert published["subjects"] == _expected_subjects()
    assert published["chapter_five_cosines"] == RECORD["chapter_five_cosines"]
    blob = json.dumps(published)
    assert "/Users/" not in blob
    assert "/Volumes/" not in blob


def test_recorded_results_stay_available_when_checkpoints_are_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty = tmp_path / "absent"
    empty.mkdir()
    monkeypatch.setenv("INSPECTOR_RESEARCH_ROOT", str(empty))
    listed = list_freshman()
    assert listed["root_status"] == "configured"
    assert [row["id"] for row in listed["subjects"]] == ["BASE", "F5-1", "F5-2", "F5-3", "F6"]
    assert all(row["hash_status"] == "missing" for row in listed["subjects"])
    assert all(row["location"] is None for row in listed["subjects"])
    assert all(row["inspectable"] is False for row in listed["subjects"])
    assert all(row["status"] == "Not on disk." for row in listed["subjects"])
    assert listed["published"]["subjects"] == _expected_subjects()
    assert listed["published"]["chapter_five_cosines"] == RECORD["chapter_five_cosines"]
    fresh = next(row for row in listed["published"]["subjects"] if row["id"] == "F5-1")
    assert fresh["rank"] == 16
    assert fresh["lora_alpha"] == 32
    assert fresh["scale"] == 2
    assert fresh["effective_update_norm"] == RECORD["subjects"][0]["effective_update_norm"]
    assert fresh["relative_to_base"] == RECORD["subjects"][0]["relative_to_base"]
    assert fresh["weight_sha256"] == RECORD["subjects"][0]["weight_sha256"]
    with pytest.raises(RegistryError) as caught:
        open_freshman("F5-1")
    assert caught.value.status == 409


def test_recorded_results_stay_available_when_the_research_root_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INSPECTOR_RESEARCH_ROOT", str(tmp_path / "not-created"))
    listed = list_freshman()
    assert listed["root_status"] == "root_missing"
    assert all(row["hash_status"] == "missing" for row in listed["subjects"])
    assert listed["published"]["label"] == "Recorded Research Results"
    assert [row["id"] for row in listed["published"]["subjects"]] == ["F5-1", "F5-2", "F5-3", "F6"]


def test_api_returns_recorded_results_without_checkpoint_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty = tmp_path / "absent"
    empty.mkdir()
    monkeypatch.setenv("INSPECTOR_RESEARCH_ROOT", str(empty))
    client = TestClient(app)
    listed = client.get("/research/book-three/freshman")
    assert listed.status_code == 200
    body = listed.json()
    assert body["published"]["subjects"] == _expected_subjects()
    assert body["published"]["chapter_five_cosines"] == RECORD["chapter_five_cosines"]
    assert body["subjects"][1]["status"] == "Not on disk."
    refused = client.post("/research/book-three/freshman/F5-1/inspect")
    assert refused.status_code == 409
    assert refused.json()["ok"] is False
