# Authors: LCC Research Lab
# Project: From Document to Evidence

import json
from pathlib import Path

import pytest

RECORD_PATH = Path(__file__).resolve().parents[1] / "research" / "freshman_validation.json"
RECORD = json.loads(RECORD_PATH.read_text(encoding="utf-8"))


def test_the_validation_record_can_be_published_without_private_paths() -> None:
    text = RECORD_PATH.read_text(encoding="utf-8")
    assert "/Users/" not in text
    assert "/Volumes/" not in text
    assert [row["id"] for row in RECORD["subjects"]] == ["F5-1", "F5-2", "F5-3", "F6"]
    assert len(RECORD["chapter_five_cosines"]) == 3
    assert RECORD["tolerances"]["norm_absolute"] == 5e-7


def test_published_freshman_measurements_stay_inside_the_recorded_tolerances() -> None:
    from inspector_engine.effective_lora import align_update_factors
    from inspector_engine.research_registry import freshman_update_factors, list_freshman, measure_freshman_effective

    listed = list_freshman()
    if listed["root_status"] != "configured" or any(row["hash_status"] == "missing" for row in listed["subjects"]):
        pytest.skip("research artifacts are not on this disk")
    by_id = {row["id"]: row for row in listed["subjects"]}
    tolerances = RECORD["tolerances"]
    try:
        measured = {row["id"]: measure_freshman_effective(row["id"]) for row in RECORD["subjects"]}
        loaded = [freshman_update_factors(row["id"]) for row in RECORD["subjects"][:3]]
    except Exception as exc:
        if getattr(exc, "code", "") in {"artifact_not_verified", "config_provenance"}:
            pytest.skip("research artifacts are not on this disk")
        raise
    for row in RECORD["subjects"]:
        report = measured[row["id"]]
        agreement = report["configuration_agreement"]
        assert by_id[row["id"]]["observed_sha256"] == row["weight_sha256"]
        assert by_id[row["id"]]["hash_status"] == "verified_match"
        assert agreement["r"] == row["rank"]
        assert agreement["lora_alpha"] == row["lora_alpha"]
        assert agreement["scale"] == pytest.approx(row["scale"])
        assert report["pair_count"] == row["pair_count"]
        assert report["total_frobenius_norm"] == pytest.approx(row["effective_update_norm"], abs=tolerances["norm_absolute"])
        assert report["total_relative_frobenius"] == pytest.approx(row["relative_to_base"], abs=tolerances["relative_absolute"])
    aligned = align_update_factors(loaded)
    comparisons = {(row["left"], row["right"]): row for row in aligned["comparisons"]}
    for expected in RECORD["chapter_five_cosines"]:
        found = comparisons[(expected["left"], expected["right"])]
        assert found["cosine"] == pytest.approx(expected["cosine"], abs=tolerances["cosine_absolute"])
        assert found["mean_pair_cosine"] == pytest.approx(expected["mean_pair_cosine"], abs=tolerances["cosine_absolute"])
