# Authors: LCC Research Lab
# Project: From Document to Evidence

import hashlib
import json
import os
import shutil
import struct
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from inspector_engine.inspect_checkpoint import InspectionError, inspect_checkpoint
from inspector_engine.roots import PathRejected, configured_roots, present_path, project_root, resolve_checkpoint
from server import app


def write_safetensors(path: Path, tensors: list[tuple[str, str, list[int], bytes]], metadata: dict | None = None) -> None:
    header: dict = {}
    offset = 0
    blobs = []
    for name, dtype, shape, raw in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    if metadata is not None:
        header["__metadata__"] = metadata
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    pad = (8 - (len(raw_header) % 8)) % 8
    raw_header += b" " * pad
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + b"".join(blobs))


def test_inventory_reports_shape_dtype_and_count(tmp_path: Path) -> None:
    path = tmp_path / "student.safetensors"
    write_safetensors(path, [("layer.weight", "F32", [2, 3], b"\x00" * 24)])
    report = inspect_checkpoint(str(path), [tmp_path])
    assert report["checkpoint"]["format"] == "safetensors"
    assert report["checkpoint"]["checkpoint_type"] == "unknown"
    assert report["checkpoint"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["tensors"] == [
        {
            "name": "layer.weight",
            "shape": [2, 3],
            "dtype": "FP32",
            "safetensors_dtype": "F32",
            "parameter_count": 6,
        }
    ]
    assert report["total_stored_parameters"] == 6
    assert report["configuration"]["source"] == "unavailable"
    assert report["engine_version"] == "1.0.0"


def test_lora_metadata_is_not_described_as_a_full_model(tmp_path: Path) -> None:
    path = tmp_path / "adapter.safetensors"
    write_safetensors(
        path,
        [("lora_A", "F16", [4], b"\x00" * 8)],
        {"peft_type": "LORA", "base_model_name_or_path": "base"},
    )
    report = inspect_checkpoint(str(path), [tmp_path])
    assert report["checkpoint"]["checkpoint_type"] == "lora_adapter"
    assert report["checkpoint"]["checkpoint_type_basis"] == "safetensors_metadata"
    assert report["configuration"]["source"] == "safetensors_metadata"
    assert report["configuration"]["metadata"]["peft_type"] == "LORA"
    assert report["total_stored_parameters"] == 4
    assert "not the parameter count of the underlying base model" in report["parameter_scope"]


def test_base_metadata(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    write_safetensors(path, [("w", "BF16", [1, 1], b"\x00\x00")], {"model_type": "qwen2"})
    report = inspect_checkpoint(str(path), [tmp_path])
    assert report["checkpoint"]["checkpoint_type"] == "base"
    assert report["tensors"][0]["dtype"] == "BF16"


def test_rejects_paths_outside_configured_directories(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    path = outside / "model.safetensors"
    write_safetensors(path, [("w", "F32", [1], b"\x00\x00\x00\x00")])
    with pytest.raises(InspectionError) as caught:
        inspect_checkpoint(str(path), [allowed])
    assert caught.value.code == "path_not_allowed"


def test_rejects_missing_malformed_and_unsupported_files(tmp_path: Path) -> None:
    missing = tmp_path / "gone.safetensors"
    with pytest.raises(InspectionError) as missing_error:
        inspect_checkpoint(str(missing), [tmp_path])
    assert missing_error.value.code == "missing_file"

    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(b"not-a-header")
    with pytest.raises(InspectionError) as broken_error:
        inspect_checkpoint(str(broken), [tmp_path])
    assert broken_error.value.code == "malformed_safetensors"

    other = tmp_path / "weights.bin"
    other.write_bytes(b"abc")
    with pytest.raises(InspectionError) as format_error:
        inspect_checkpoint(str(other), [tmp_path])
    assert format_error.value.code == "unsupported_format"


def test_api_returns_the_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "model.safetensors"
    write_safetensors(path, [("w", "F16", [2], b"\x00\x00\x00\x00")])
    monkeypatch.setenv("INSPECTOR_CHECKPOINT_DIRS", str(tmp_path))
    monkeypatch.setattr("inspector_engine.roots.ROOTS_FILE", tmp_path / "no-roots.txt")
    client = TestClient(app)
    response = client.post("/inspect/checkpoint", json={"path": str(path)})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["total_stored_parameters"] == 2
    assert body["tensors"][0]["dtype"] == "FP16"


def test_health_reports_the_default_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["pytest"])
    monkeypatch.delenv("INSPECTOR_PORT", raising=False)
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["service"] == "ai-model-inspector"
    assert body["port"] == 8810


def test_health_reports_uvicorn_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["python", "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", "9000"])
    monkeypatch.setenv("INSPECTOR_PORT", "8810")
    client = TestClient(app)
    assert client.get("/health").json()["port"] == 9000


def test_health_reports_inspector_port_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["pytest"])
    monkeypatch.setenv("INSPECTOR_PORT", "9000")
    client = TestClient(app)
    assert client.get("/health").json()["port"] == 9000


def test_full_model_declaration_matches_observed_layers(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    config = {
        "model_type": "toy",
        "architectures": ["ToyForCausalLM"],
        "num_hidden_layers": 2,
        "hidden_size": 4,
        "num_attention_heads": 2,
        "vocab_size": 6,
        "intermediate_size": 8,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    write_safetensors(
        path,
        [
            ("model.embed_tokens.weight", "F32", [6, 4], b"\x00" * 96),
            ("model.layers.0.mlp.down_proj.weight", "F32", [4, 4], b"\x00" * 64),
            ("model.layers.1.mlp.down_proj.weight", "F32", [4, 4], b"\x00" * 64),
        ],
    )
    before = {item.name: item.read_bytes() for item in tmp_path.iterdir()}
    report = inspect_checkpoint(str(path), [tmp_path])
    assert {item.name: item.read_bytes() for item in tmp_path.iterdir()} == before
    declared = report["declared_architecture"]
    assert declared["model_type"] == "toy"
    assert declared["architecture_name"] == ["ToyForCausalLM"]
    assert declared["num_hidden_layers"] == 2
    assert declared["hidden_size"] == 4
    assert declared["num_attention_heads"] == 2
    assert declared["vocab_size"] == 6
    assert declared["other_properties"]["intermediate_size"] == 8
    assert declared["unknown_fields"] == []
    assert report["observed_checkpoint"]["classification"] == "full_model"
    assert report["validation"]["warnings"] == []
    assert report["ok"] is True


def test_lora_adapter_is_a_warning_not_a_failure(tmp_path: Path) -> None:
    path = tmp_path / "adapter_model.safetensors"
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "toy",
                "architectures": ["ToyForCausalLM"],
                "num_hidden_layers": 4,
                "hidden_size": 8,
                "num_attention_heads": 2,
                "vocab_size": 10,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": "org/toy",
                "r": 2,
                "lora_alpha": 4,
                "target_modules": ["q_proj", "v_proj"],
            }
        ),
        encoding="utf-8",
    )
    write_safetensors(
        path,
        [
            ("model.layers.0.self_attn.q_proj.lora_A.weight", "F32", [2, 8], b"\x00" * 64),
            ("model.layers.0.self_attn.q_proj.lora_B.weight", "F32", [8, 2], b"\x00" * 64),
        ],
    )
    before = {item.name: hashlib.sha256(item.read_bytes()).hexdigest() for item in tmp_path.iterdir()}
    report = inspect_checkpoint(str(path), [tmp_path])
    after = {item.name: hashlib.sha256(item.read_bytes()).hexdigest() for item in tmp_path.iterdir()}
    assert after == before
    assert report["ok"] is True
    assert report["observed_checkpoint"]["classification"] == "adapter"
    assert report["declared_adapter"]["peft_type"] == "LORA"
    assert report["declared_adapter"]["r"] == 2
    assert report["declared_architecture"]["num_hidden_layers"] == 4
    codes = [warning["code"] for warning in report["validation"]["warnings"]]
    assert "adapter_subset" in codes
    assert "declared_targets_absent" in codes
    assert report["total_stored_parameters"] == 32


def test_partial_layer_stack_is_an_unknown_subset(tmp_path: Path) -> None:
    path = tmp_path / "partial.safetensors"
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "toy", "num_hidden_layers": 2, "hidden_size": 4}),
        encoding="utf-8",
    )
    write_safetensors(path, [("model.layers.0.mlp.down_proj.weight", "F32", [4, 4], b"\x00" * 64)])
    report = inspect_checkpoint(str(path), [tmp_path])
    assert report["ok"] is True
    assert report["observed_checkpoint"]["classification"] == "unknown_subset"
    assert report["declared_architecture"]["num_attention_heads"] is None
    assert "num_attention_heads" in report["declared_architecture"]["unknown_fields"]
    assert any(warning["code"] == "layer_coverage" for warning in report["validation"]["warnings"])


def test_lora_rank_must_match_a_declared_dimension(tmp_path: Path) -> None:
    path = tmp_path / "adapter_model.safetensors"
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 8, "target_modules": ["q_proj"]}),
        encoding="utf-8",
    )
    write_safetensors(path, [("model.layers.0.self_attn.q_proj.lora_A.weight", "F32", [2, 4], b"\x00" * 32)])
    report = inspect_checkpoint(str(path), [tmp_path])
    assert report["ok"] is True
    assert any(warning["code"] == "lora_rank_mismatch" for warning in report["validation"]["warnings"])
    assert report["declared_architecture"]["available"] is False


def test_local_base_config_is_read_and_a_hub_id_is_not(tmp_path: Path) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    (base / "config.json").write_text(
        json.dumps({"model_type": "toy", "num_hidden_layers": 3, "hidden_size": 4, "vocab_size": 6}),
        encoding="utf-8",
    )
    (adapter / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": str(base), "r": 1, "target_modules": ["q_proj"]}),
        encoding="utf-8",
    )
    path = adapter / "adapter_model.safetensors"
    write_safetensors(path, [("model.layers.0.self_attn.q_proj.lora_A.weight", "F16", [1, 4], b"\x00\x00\x00\x00\x00\x00\x00\x00")])
    report = inspect_checkpoint(str(path), [tmp_path])
    assert report["declared_architecture"]["origin"] == "base_model_name_or_path"
    assert report["declared_architecture"]["model_type"] == "toy"
    assert report["declared_architecture"]["num_hidden_layers"] == 3

    hub = tmp_path / "hub"
    hub.mkdir()
    (hub / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": "org/toy", "r": 1}),
        encoding="utf-8",
    )
    hub_path = hub / "adapter_model.safetensors"
    write_safetensors(hub_path, [("lora_A.weight", "F32", [1, 1], b"\x00\x00\x00\x00")])
    hub_report = inspect_checkpoint(str(hub_path), [tmp_path])
    assert hub_report["declared_architecture"]["available"] is False
    assert not (hub / "org").exists()
    assert any(warning["code"] == "base_architecture_unavailable" for warning in hub_report["validation"]["warnings"])
    downloaded = tmp_path / "org__toy"
    downloaded.mkdir()
    (downloaded / "config.json").write_text(
        json.dumps({"model_type": "llama", "num_hidden_layers": 2, "hidden_size": 16}),
        encoding="utf-8",
    )
    found = inspect_checkpoint(str(hub_path), [tmp_path])
    assert found["declared_architecture"]["model_type"] == "llama"
    assert found["declared_architecture"]["num_hidden_layers"] == 2
    assert not any(warning["code"] == "base_architecture_unavailable" for warning in found["validation"]["warnings"])


def test_mlx_adapter_rank_is_read_from_its_declared_key(tmp_path: Path) -> None:
    path = tmp_path / "adapters.safetensors"
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "fine_tune_type": "lora",
                "num_layers": 1,
                "model": "/missing/base/train",
                "lora_parameters": {"rank": 8, "scale": 20.0, "dropout": 0.0},
            }
        ),
        encoding="utf-8",
    )
    write_safetensors(
        path,
        [("model.layers.16.self_attn.q_proj.lora_a", "F32", [4, 8], b"\x00" * 128)],
    )
    report = inspect_checkpoint(str(path), [tmp_path])
    adapter = report["declared_adapter"]
    assert adapter["peft_type"] is None
    assert adapter["fine_tune_type"] == "lora"
    assert adapter["r"] == 8
    assert adapter["r_key"] == "lora_parameters.rank"
    assert adapter["adapter_num_layers"] == 1
    assert adapter["base_model_path"] == "/missing/base/train"
    assert adapter["base_model_path_key"] == "model"
    assert report["declared_architecture"]["num_hidden_layers"] is None
    assert report["observed_checkpoint"]["classification"] == "adapter"
    assert report["observed_checkpoint"]["layer_indices"] == [16]
    codes = [warning["code"] for warning in report["validation"]["warnings"]]
    assert "base_architecture_unavailable" in codes
    assert "lora_rank_mismatch" not in codes
    assert "adapter_layer_selection" not in codes


def test_unreadable_config_does_not_hide_the_tensor_inventory(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    (tmp_path / "config.json").write_text("{", encoding="utf-8")
    write_safetensors(path, [("model.layers.0.weight", "F32", [1], b"\x00\x00\x00\x00")])
    report = inspect_checkpoint(str(path), [tmp_path])
    assert report["ok"] is True
    assert report["tensors"][0]["name"] == "model.layers.0.weight"
    assert report["declared_architecture"]["model_type"] is None
    assert any(warning["code"] == "config_unreadable" for warning in report["validation"]["warnings"])


def test_gpt2_style_depth_comes_from_the_checkpoint_file(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gpt2",
                "architectures": ["GPT2LMHeadModel"],
                "n_layer": 2,
                "n_embd": 8,
                "n_head": 2,
                "vocab_size": 16,
            }
        ),
        encoding="utf-8",
    )
    raw = b"\x00" * (8 * 24 * 4)
    write_safetensors(
        path,
        [
            ("transformer.h.0.attn.c_attn.weight", "F32", [8, 24], raw),
            ("transformer.h.1.attn.c_attn.weight", "F32", [8, 24], raw),
        ],
    )
    report = inspect_checkpoint(str(path), [tmp_path])
    architecture = report["declared_architecture"]
    assert architecture["model_type"] == "gpt2"
    assert architecture["architecture_name"] == ["GPT2LMHeadModel"]
    assert architecture["num_hidden_layers"] == 2
    assert architecture["hidden_size"] == 8
    assert architecture["num_attention_heads"] == 2
    assert report["observed_checkpoint"]["classification"] == "full_model"
    assert report["observed_checkpoint"]["layer_indices"] == [0, 1]
    assert report["tensor_count"] == 2
    assert report["total_stored_parameters"] == 384
    assert {row["dtype"] for row in report["tensors"]} == {"FP32"}
    assert {tuple(row["shape"]) for row in report["tensors"]} == {(8, 24)}
    assert report["checkpoint"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_base_config_outside_checkpoint_directories_is_not_read(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    adapter = allowed / "adapter"
    adapter.mkdir(parents=True)
    outside.mkdir()
    (outside / "config.json").write_text(
        json.dumps({"model_type": "not-this-model", "num_hidden_layers": 24, "hidden_size": 2048}),
        encoding="utf-8",
    )
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": str(outside),
                "r": 4,
                "target_modules": ["c_attn"],
            }
        ),
        encoding="utf-8",
    )
    path = adapter / "adapter_model.safetensors"
    write_safetensors(path, [("transformer.h.0.attn.c_attn.lora_A.weight", "F16", [4, 8], b"\x00" * 64)])
    report = inspect_checkpoint(str(path), [allowed])
    encoded = json.dumps(report)
    assert "not-this-model" not in encoded
    assert report["declared_architecture"]["available"] is False
    assert report["declared_architecture"]["num_hidden_layers"] is None
    assert report["declared_adapter"]["r"] == 4
    assert report["tensors"][0]["shape"] == [4, 8]
    assert report["tensors"][0]["dtype"] == "FP16"
    assert any(warning["code"] == "base_architecture_outside_roots" for warning in report["validation"]["warnings"])
    assert not any(warning["code"] == "base_architecture_unavailable" for warning in report["validation"]["warnings"])


def test_a_project_relative_path_survives_moving_the_project(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    first = tmp_path / "first-install"
    library = first / "inspector-engine" / "library"
    library.mkdir(parents=True)
    weight = library / "model.safetensors"
    write_safetensors(weight, [("layer.weight", "F32", [2], b"\x00\x00\x80\x3f\x00\x00\x00\x40")])
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    relative = "inspector-engine/library/model.safetensors"
    monkeypatch.setattr("inspector_engine.roots.project_root", lambda: first)

    shown = present_path(weight)
    assert shown["path_label"] == "Project-relative path"
    assert shown["path_display"] == relative
    resolved = resolve_checkpoint(relative, [library])
    report = inspect_checkpoint(relative, [library])
    assert resolved == weight.resolve()
    assert report["checkpoint"]["path"] == str(weight.resolve())
    assert report["checkpoint"]["sha256"] == digest
    assert report["checkpoint"]["path_display"] == relative

    second = tmp_path / "second-install"
    shutil.move(first, second)
    moved_library = second / "inspector-engine" / "library"
    moved = moved_library / "model.safetensors"
    monkeypatch.setattr("inspector_engine.roots.project_root", lambda: second)
    shown_again = present_path(moved)
    report_again = inspect_checkpoint(relative, [moved_library])
    assert shown_again["path_display"] == relative
    assert report_again["checkpoint"]["path_display"] == relative
    assert report_again["checkpoint"]["sha256"] == digest
    assert hashlib.sha256(moved.read_bytes()).hexdigest() == digest


def test_an_imported_file_outside_the_project_is_labeled_external(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "imported-weights"
    outside.mkdir()
    weight = outside / "model.safetensors"
    write_safetensors(weight, [("layer.weight", "F32", [1], b"\x00\x00\x80\x3f")])
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    monkeypatch.setattr("inspector_engine.roots.project_root", lambda: project)

    report = inspect_checkpoint(str(weight), [outside])
    assert report["checkpoint"]["path"] == str(weight.resolve())
    assert report["checkpoint"]["sha256"] == digest
    assert report["checkpoint"]["path_label"] == "External file"
    assert report["checkpoint"]["path_scope"] == "external"
    assert report["checkpoint"]["path_display"].startswith("Outside this project")
    assert not str(report["checkpoint"]["path_display"]).startswith("/")
    opened = inspect_checkpoint("../imported-weights/model.safetensors", [outside])
    assert opened["checkpoint"]["path"] == str(weight.resolve())
    assert opened["checkpoint"]["sha256"] == digest
    assert opened["checkpoint"]["path_label"] == "External file"
    with pytest.raises(InspectionError) as caught:
        inspect_checkpoint("inspector-engine/library/../../imported-weights/model.safetensors", [project / "inspector-engine" / "library"])
    assert caught.value.code == "path_not_allowed"


def test_the_project_root_comes_from_the_application_layout() -> None:
    from inspector_engine.roots import ENGINE_DIR

    assert project_root() == ENGINE_DIR.parent
    qwen = project_root() / "inspector-engine/library/Qwen__Qwen3-Coder-30B-A3B-Instruct/model-00016-of-00016.safetensors"
    if qwen.is_file():
        shown = present_path(qwen)
        assert shown == {
            "path_scope": "project",
            "path_label": "Project-relative path",
            "path_display": "inspector-engine/library/Qwen__Qwen3-Coder-30B-A3B-Instruct/model-00016-of-00016.safetensors",
        }


def test_api_error_for_a_relative_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INSPECTOR_CHECKPOINT_DIRS", str(tmp_path))
    monkeypatch.setattr("inspector_engine.roots.ROOTS_FILE", tmp_path / "no-roots.txt")
    client = TestClient(app)
    response = client.post("/inspect/checkpoint", json={"path": "model.safetensors"})
    assert response.status_code == 400
    assert response.json()["error"] == "path_not_allowed"


def test_a_null_byte_cannot_select_a_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(PathRejected) as caught:
        resolve_checkpoint("model.safetensors\x00", [tmp_path])
    assert caught.value.code == "path_not_allowed"


def test_a_symlink_cannot_leave_the_configured_directory(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    target = outside / "model.safetensors"
    write_safetensors(target, [("layer.weight", "F32", [1], b"\x00\x00\x80\x3f")])
    link = allowed / "model.safetensors"
    link.symlink_to(target)
    with pytest.raises(PathRejected) as caught:
        resolve_checkpoint(str(link), [allowed])
    assert caught.value.code == "path_not_allowed"


def test_the_filesystem_root_is_not_an_inspection_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INSPECTOR_CHECKPOINT_DIRS", os.pathsep.join(["/", str(Path.home())]))
    monkeypatch.setattr("inspector_engine.roots.ROOTS_FILE", tmp_path / "no-roots.txt")
    roots = configured_roots()
    assert Path("/").resolve() not in roots
    assert Path.home().resolve() not in roots
