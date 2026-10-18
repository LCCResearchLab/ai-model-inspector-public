# Authors: LCC Research Lab
# Project: From Document to Evidence

import json
import math
import struct
from pathlib import Path

import pytest

import numpy as np

from inspector_engine.effective_lora import align_update_factors, factors_for_checkpoint, measure_effective_lora, measure_effective_update
from inspector_engine.inspect_checkpoint import InspectionError
from inspector_engine.research_registry import measure_freshman_effective


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


def config(rank: int = 2, alpha: int = 4, rslora: bool = False) -> dict:
    return {
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": alpha,
        "use_rslora": rslora,
        "use_dora": False,
        "fan_in_fan_out": False,
        "bias": "none",
        "rank_pattern": {},
        "alpha_pattern": {},
        "target_modules": ["q_proj"],
        "base_model_name_or_path": "/original/place",
    }


def test_synthetic_product_matches_an_independent_calculation(tmp_path: Path) -> None:
    # A is 2×3 and B is 3×2. BA is written out by hand below.
    left = struct.pack("<6f", 1, 0, 1, 0, 1, 0)
    right = struct.pack("<6f", 1, 0, 0, 1, 1, 1)
    base = struct.pack("<9f", 1, 0, 0, 0, 1, 0, 0, 0, 1)
    adapter = tmp_path / "adapter_model.safetensors"
    weights = tmp_path / "model.safetensors"
    write_safetensors(adapter, [
        ("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", "F32", [2, 3], left),
        ("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight", "F32", [3, 2], right),
    ])
    write_safetensors(weights, [("model.layers.0.self_attn.q_proj.weight", "F32", [3, 3], base)])
    research = config()
    packaged = dict(research)
    packaged["base_model_name_or_path"] = "../BASE"
    (tmp_path / "adapter_config.research.json").write_text(json.dumps(research), encoding="utf-8")
    (tmp_path / "adapter_config.json").write_text(json.dumps(packaged), encoding="utf-8")
    before = adapter.stat().st_mtime_ns
    report = measure_effective_lora(
        adapter,
        tmp_path / "adapter_config.research.json",
        tmp_path / "adapter_config.json",
        weights,
    )
    assert adapter.stat().st_mtime_ns == before
    assert report["configuration_agreement"]["agreed"] is True
    assert report["configuration_agreement"]["scale"] == 2
    assert report["configuration_agreement"]["scale_rule"] == "lora_alpha / r"
    pair = report["pairs"][0]
    assert pair["layer"] == 0
    assert pair["projection"] == "q_proj"
    assert pair["frobenius_norm"] == pytest.approx(2 * math.sqrt(6))
    assert pair["maximum_absolute"] == pytest.approx(2)
    assert pair["mean_absolute"] == pytest.approx(4 / 3)
    assert pair["base_frobenius_norm"] == pytest.approx(math.sqrt(3))
    assert pair["relative_frobenius"] == pytest.approx(2 * math.sqrt(2))
    assert report["total_frobenius_norm"] == pytest.approx(pair["frobenius_norm"])
    assert "matrix" not in pair


def test_a_changed_rank_refuses_the_measurement(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter_model.safetensors"
    write_safetensors(adapter, [("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", "F32", [1, 1], struct.pack("<f", 1))])
    research = config(rank=1, alpha=1)
    packaged = dict(research)
    packaged["r"] = 4
    (tmp_path / "adapter_config.research.json").write_text(json.dumps(research), encoding="utf-8")
    (tmp_path / "adapter_config.json").write_text(json.dumps(packaged), encoding="utf-8")
    with pytest.raises(InspectionError) as caught:
        measure_effective_lora(adapter, tmp_path / "adapter_config.research.json", tmp_path / "adapter_config.json")
    assert caught.value.code == "config_disagreement"
    assert "r" in caught.value.message


def test_rslora_uses_the_square_root_rule(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter_model.safetensors"
    write_safetensors(
        adapter,
        [
            ("base_model.model.model.layers.1.self_attn.v_proj.lora_A.weight", "F32", [4, 1], struct.pack("<4f", 1, 0, 0, 0)),
            ("base_model.model.model.layers.1.self_attn.v_proj.lora_B.weight", "F32", [1, 4], struct.pack("<4f", 1, 0, 0, 0)),
        ],
    )
    research = config(rank=4, alpha=4, rslora=True)
    research["target_modules"] = ["v_proj"]
    (tmp_path / "adapter_config.research.json").write_text(json.dumps(research), encoding="utf-8")
    (tmp_path / "adapter_config.json").write_text(json.dumps(research), encoding="utf-8")
    report = measure_effective_lora(adapter, tmp_path / "adapter_config.research.json", tmp_path / "adapter_config.json")
    assert report["configuration_agreement"]["scale"] == pytest.approx(4 / math.sqrt(4))
    assert report["configuration_agreement"]["scale_rule"] == "lora_alpha / sqrt(r)"
    assert report["pairs"][0]["frobenius_norm"] == pytest.approx(2)


def test_a_checkpoint_with_one_configuration_uses_the_same_instrument(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from server import app

    adapter = tmp_path / "adapter_model.safetensors"
    left = struct.pack("<6f", 1, 0, 1, 0, 1, 0)
    right = struct.pack("<6f", 1, 0, 0, 1, 1, 1)
    write_safetensors(adapter, [
        ("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", "F32", [2, 3], left),
        ("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight", "F32", [3, 2], right),
    ])
    (tmp_path / "adapter_config.json").write_text(json.dumps(config()), encoding="utf-8")
    direct = measure_effective_update(str(adapter), [tmp_path])
    assert direct["configuration_agreement"]["configuration_file_count"] == 1
    assert direct["pair_count"] == 1
    assert direct["total_frobenius_norm"] == pytest.approx(2 * math.sqrt(6))
    assert direct["total_relative_frobenius"] is None
    monkeypatch.setenv("INSPECTOR_CHECKPOINT_DIRS", str(tmp_path))
    monkeypatch.setattr("inspector_engine.roots.ROOTS_FILE", tmp_path / "no-roots.txt")
    response = TestClient(app).post("/inspect/effective-update", json={"path": str(adapter)})
    assert response.status_code == 200
    body = response.json()
    assert body["instrument"] == "effective_lora_update"
    assert body["pair_count"] == 1
    assert body["total_frobenius_norm"] == pytest.approx(direct["total_frobenius_norm"])


def test_recorded_adapters_have_agreed_settings_and_ninety_six_pairs() -> None:
    summaries = []
    for subject_id in ("F5-1", "F5-2", "F5-3", "F6"):
        try:
            report = measure_freshman_effective(subject_id)
        except Exception as exc:
            if getattr(exc, "code", "") in {"artifact_not_verified", "config_provenance"}:
                pytest.skip("research artifacts are not on this disk")
            raise
        agreement = report["configuration_agreement"]
        assert agreement["agreed"] is True
        assert agreement["differing_fields"] == []
        assert agreement["scale"] == 2
        assert agreement["r"] == 16
        assert agreement["lora_alpha"] == 32
        assert report["pair_count"] == 96
        assert {row["projection"] for row in report["pairs"]} == {"q_proj", "k_proj", "v_proj", "o_proj"}
        assert {row["layer"] for row in report["pairs"]} == set(range(24))
        assert all(row["relative_frobenius"] is not None for row in report["pairs"])
        assert report["research"]["hash_status"] == "verified_match"
        summaries.append(report["total_frobenius_norm"])
    assert all(math.isfinite(value) and value > 0 for value in summaries)


def _adapter(directory: Path, tensors: list[tuple[str, list[int], list[float]]], alpha: int = 1, rank: int = 1) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    encoded = []
    for tensor_name, shape, values in tensors:
        encoded.append((tensor_name, "F32", shape, struct.pack(f"<{len(values)}f", *values)))
    path = directory / "adapter_model.safetensors"
    write_safetensors(path, encoded)
    research = config(rank=rank, alpha=alpha)
    research["target_modules"] = ["q_proj", "v_proj"]
    (directory / "adapter_config.json").write_text(json.dumps(research), encoding="utf-8")
    return path


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sum(left * right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def test_direction_matches_the_full_update_and_ignores_shared_scale(tmp_path: Path) -> None:
    left_a = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    left_b = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    same = _adapter(
        tmp_path / "same",
        [
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", [2, 3], left_a),
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight", [3, 2], left_b),
        ],
        alpha=2,
        rank=2,
    )
    larger = _adapter(
        tmp_path / "larger",
        [
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", [2, 3], left_a),
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight", [3, 2], left_b),
        ],
        alpha=8,
        rank=2,
    )
    before = same.stat().st_mtime_ns
    report = align_update_factors([factors_for_checkpoint(str(same), [tmp_path]), factors_for_checkpoint(str(larger), [tmp_path])])
    assert same.stat().st_mtime_ns == before
    comparison = report["comparisons"][0]
    assert comparison["cosine"] == pytest.approx(1)
    assert comparison["mean_pair_cosine"] == pytest.approx(1)
    assert report["baseline"]["entry_count"] == 9
    assert report["baseline"]["dense_isotropic_scale"] == pytest.approx(1 / 3)
    assert report["baseline"]["seed"] == 22001001
    repeat = align_update_factors([factors_for_checkpoint(str(same), [tmp_path]), factors_for_checkpoint(str(larger), [tmp_path])])
    assert repeat["baseline"]["random_factor_median_absolute"] == pytest.approx(report["baseline"]["random_factor_median_absolute"])
    assert report["subjects"][0]["total_frobenius_norm"] * 4 == pytest.approx(report["subjects"][1]["total_frobenius_norm"])
    matrix = 2 * (np.array(left_b, dtype=np.float64).reshape(3, 2) @ np.array(left_a, dtype=np.float64).reshape(2, 3))
    assert _cosine(matrix, matrix) == pytest.approx(comparison["cosine"])


def test_opposite_and_orthogonal_updates_keep_their_directions(tmp_path: Path) -> None:
    first = _adapter(
        tmp_path / "first",
        [
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", [1, 2], [1, 0]),
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight", [2, 1], [1, 0]),
        ],
    )
    opposite = _adapter(
        tmp_path / "opposite",
        [
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", [1, 2], [1, 0]),
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight", [2, 1], [-1, 0]),
        ],
    )
    orthogonal = _adapter(
        tmp_path / "orthogonal",
        [
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", [1, 2], [0, 1]),
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight", [2, 1], [0, 1]),
        ],
    )
    roots = [tmp_path]
    opposite_report = align_update_factors([factors_for_checkpoint(str(first), roots), factors_for_checkpoint(str(opposite), roots)])
    orthogonal_report = align_update_factors([factors_for_checkpoint(str(first), roots), factors_for_checkpoint(str(orthogonal), roots)])
    assert opposite_report["comparisons"][0]["cosine"] == pytest.approx(-1)
    assert orthogonal_report["comparisons"][0]["cosine"] == pytest.approx(0)


def test_overall_cosine_weights_larger_layers_and_the_mean_does_not(tmp_path: Path) -> None:
    def tensors(second_b: float) -> list[tuple[str, list[int], list[float]]]:
        return [
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", [1, 2], [1, 0]),
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight", [2, 1], [3, 0]),
            ("base_model.model.model.layers.1.self_attn.v_proj.lora_A.weight", [1, 2], [1, 0]),
            ("base_model.model.model.layers.1.self_attn.v_proj.lora_B.weight", [2, 1], [second_b, 0]),
        ]

    left = _adapter(tmp_path / "left", tensors(1))
    right = _adapter(tmp_path / "right", tensors(-1))
    report = align_update_factors([factors_for_checkpoint(str(left), [tmp_path]), factors_for_checkpoint(str(right), [tmp_path])])
    comparison = report["comparisons"][0]
    assert comparison["cosine"] == pytest.approx(0.8)
    assert comparison["mean_pair_cosine"] == pytest.approx(0)
    assert comparison["by_layer"][0]["cosine"] == pytest.approx(1)
    assert comparison["by_layer"][1]["cosine"] == pytest.approx(-1)


def test_different_projection_coverage_is_not_compared(tmp_path: Path) -> None:
    left = _adapter(
        tmp_path / "covered",
        [
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", [1, 2], [1, 0]),
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight", [2, 1], [1, 0]),
        ],
    )
    right = _adapter(
        tmp_path / "shifted",
        [
            ("base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight", [1, 2], [1, 0]),
            ("base_model.model.model.layers.1.self_attn.q_proj.lora_B.weight", [2, 1], [1, 0]),
        ],
    )
    with pytest.raises(InspectionError) as caught:
        align_update_factors([factors_for_checkpoint(str(left), [tmp_path]), factors_for_checkpoint(str(right), [tmp_path])])
    assert caught.value.code == "pair_mismatch"


def test_alpha_comes_from_the_configuration_and_an_outside_base_is_not_measured(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    outside.mkdir()
    base = outside / "model.safetensors"
    write_safetensors(base, [("model.layers.0.self_attn.q_proj.weight", "F32", [2, 2], struct.pack("<4f", 3, 0, 0, 3))])
    path = _adapter(
        allowed / "adapter",
        [
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", [4, 2], [1, 0, 0, 1, 1, 0, 0, 1]),
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight", [2, 4], [1, 0, 0, 0, 0, 1, 0, 0]),
        ],
        alpha=8,
        rank=4,
    )
    packaged = json.loads((path.parent / "adapter_config.json").read_text(encoding="utf-8"))
    packaged["base_model_name_or_path"] = str(outside)
    (path.parent / "adapter_config.json").write_text(json.dumps(packaged), encoding="utf-8")
    report = measure_effective_update(str(path), [allowed])
    assert report["configuration_agreement"]["lora_alpha"] == 8
    assert report["configuration_agreement"]["scale"] == pytest.approx(2)
    assert "adapter_config.json" in report["sources"]["lora_alpha"]
    assert "not stored in the A or B tensors" in report["sources"]["lora_alpha"]
    assert report["relative_measured"] is False
    assert report["total_relative_frobenius"] is None
    assert report["pair_count"] == 1
    assert report["total_frobenius_norm"] > 0
    assert "No relative magnitude" in report["relative_note"]


def test_mlp_and_named_adapter_factors_use_the_same_product(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter_model.safetensors"
    write_safetensors(
        adapter,
        [
            ("base_model.model.model.layers.0.mlp.down_proj.lora_A.default.weight", "F32", [2, 3], struct.pack("<6f", 1, 0, 1, 0, 1, 0)),
            ("base_model.model.model.layers.0.mlp.down_proj.lora_B.default.weight", "F32", [3, 2], struct.pack("<6f", 1, 0, 0, 1, 1, 1)),
        ],
    )
    research = config()
    research["target_modules"] = ["down_proj"]
    (tmp_path / "adapter_config.json").write_text(json.dumps(research), encoding="utf-8")
    report = measure_effective_update(str(adapter), [tmp_path])
    pair = report["pairs"][0]
    assert pair["module"] == "model.layers.0.mlp.down_proj"
    assert pair["projection"] == "down_proj"
    assert report["configuration_agreement"]["scale"] == 2
    assert pair["frobenius_norm"] == pytest.approx(2 * math.sqrt(6))
    assert report["relative_measured"] is False


def test_a_downloaded_repository_is_the_base_when_it_is_in_the_library(tmp_path: Path) -> None:
    library = tmp_path / "lab__tiny"
    library.mkdir()
    base = library / "model.safetensors"
    write_safetensors(base, [("model.layers.0.self_attn.q_proj.weight", "F32", [3, 3], struct.pack("<9f", 1, 0, 0, 0, 1, 0, 0, 0, 1))])
    adapter_dir = tmp_path / "adapter"
    path = _adapter(
        adapter_dir,
        [
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", [2, 3], [1, 0, 1, 0, 1, 0]),
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight", [3, 2], [1, 0, 0, 1, 1, 1]),
        ],
        alpha=4,
        rank=2,
    )
    packaged = json.loads((path.parent / "adapter_config.json").read_text(encoding="utf-8"))
    packaged["base_model_name_or_path"] = "lab/tiny"
    (path.parent / "adapter_config.json").write_text(json.dumps(packaged), encoding="utf-8")
    report = measure_effective_update(str(path), [tmp_path])
    assert report["relative_measured"] is True
    assert report["base_checkpoint"]["name"] == "model.safetensors"
    assert report["pairs"][0]["relative_frobenius"] == pytest.approx(2 * math.sqrt(2))
    assert report["total_relative_frobenius"] == pytest.approx(report["pairs"][0]["relative_frobenius"])


def test_a_named_repository_that_was_not_downloaded_explains_the_omission(tmp_path: Path) -> None:
    path = _adapter(
        tmp_path / "adapter",
        [
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", [1, 2], [1, 0]),
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight", [2, 1], [1, 0]),
        ],
    )
    packaged = json.loads((path.parent / "adapter_config.json").read_text(encoding="utf-8"))
    packaged["base_model_name_or_path"] = "lab/not-downloaded"
    (path.parent / "adapter_config.json").write_text(json.dumps(packaged), encoding="utf-8")
    report = measure_effective_update(str(path), [tmp_path])
    assert report["relative_measured"] is False
    assert report["total_relative_frobenius"] is None
    assert "lab/not-downloaded" in report["relative_note"]
    assert "not in this library" in report["relative_note"]


def test_an_incompatible_base_shape_is_not_given_a_relative_magnitude(tmp_path: Path) -> None:
    base = tmp_path / "model.safetensors"
    write_safetensors(base, [("model.layers.0.self_attn.q_proj.weight", "F32", [2, 2], struct.pack("<4f", 1, 0, 0, 1))])
    path = _adapter(
        tmp_path / "adapter",
        [
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", [1, 3], [1, 0, 0]),
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight", [4, 1], [1, 0, 0, 0]),
        ],
        rank=1,
        alpha=1,
    )
    with pytest.raises(InspectionError) as caught:
        measure_effective_lora(path, path.parent / "adapter_config.json", path.parent / "adapter_config.json", base)
    assert caught.value.code == "shape_mismatch"
    assert "does not match" in caught.value.message


def test_qalora_and_an_unrecognized_lora_tensor_are_explained(tmp_path: Path) -> None:
    path = _adapter(
        tmp_path / "adapter",
        [
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", [1, 2], [1, 0]),
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight", [2, 1], [1, 0]),
        ],
    )
    packaged = json.loads((path.parent / "adapter_config.json").read_text(encoding="utf-8"))
    packaged["use_qalora"] = True
    (path.parent / "adapter_config.json").write_text(json.dumps(packaged), encoding="utf-8")
    with pytest.raises(InspectionError) as caught:
        measure_effective_update(str(path), [tmp_path])
    assert caught.value.code == "unsupported_adapter"
    assert "QA-LoRA" in caught.value.message
    packaged["use_qalora"] = False
    (path.parent / "adapter_config.json").write_text(json.dumps(packaged), encoding="utf-8")
    write_safetensors(
        path,
        [
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", "F32", [1, 2], struct.pack("<2f", 1, 0)),
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight", "F32", [2, 1], struct.pack("<2f", 1, 0)),
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_magnitude_vector", "F32", [2], struct.pack("<2f", 1, 1)),
        ],
    )
    with pytest.raises(InspectionError) as refused:
        measure_effective_update(str(path), [tmp_path])
    assert refused.value.code == "unsupported_adapter"
    assert "lora_magnitude_vector" in refused.value.message


def test_missing_alpha_does_not_produce_an_effective_update(tmp_path: Path) -> None:
    path = _adapter(
        tmp_path / "adapter",
        [
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", [1, 2], [1, 0]),
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight", [2, 1], [1, 0]),
        ],
    )
    packaged = json.loads((path.parent / "adapter_config.json").read_text(encoding="utf-8"))
    del packaged["lora_alpha"]
    (path.parent / "adapter_config.json").write_text(json.dumps(packaged), encoding="utf-8")
    with pytest.raises(InspectionError) as caught:
        measure_effective_update(str(path), [tmp_path])
    assert caught.value.code == "unsupported_adapter"
    assert "alpha" in caught.value.message


def test_recorded_chapter_five_directions_use_the_same_update_as_the_magnitude() -> None:
    from inspector_engine.research_registry import freshman_update_factors
    from inspector_engine.effective_lora import align_update_factors as align

    try:
        loaded = [freshman_update_factors(subject_id) for subject_id in ("F5-1", "F5-2", "F5-3")]
        magnitudes = [measure_freshman_effective(subject_id)["total_frobenius_norm"] for subject_id in ("F5-1", "F5-2", "F5-3")]
    except Exception as exc:
        if getattr(exc, "code", "") in {"artifact_not_verified", "config_provenance"}:
            pytest.skip("research artifacts are not on this disk")
        raise
    report = align(loaded)
    assert [row["subject_id"] for row in report["subjects"]] == ["F5-1", "F5-2", "F5-3"]
    assert [row["total_frobenius_norm"] for row in report["subjects"]] == pytest.approx(magnitudes)
    assert len(report["comparisons"]) == 3
    for comparison in report["comparisons"]:
        assert comparison["pair_count"] == 96
        assert comparison["cosine"] is not None
        assert -1 <= comparison["cosine"] <= 1
        assert comparison["mean_pair_cosine"] is not None
        assert -1 <= comparison["mean_pair_cosine"] <= 1
    assert "matrix" not in report
    assert report["baseline"]["entry_count"] == 96 * 2048 * 2048
    assert report["baseline"]["dense_isotropic_scale"] == pytest.approx(1 / math.sqrt(report["baseline"]["entry_count"]))
