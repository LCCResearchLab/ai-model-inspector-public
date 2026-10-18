# Authors: LCC Research Lab
# Project: From Document to Evidence
"""Effective LoRA updates, one A/B pair at a time.

The stored update is (lora_alpha / r) BA when use_rslora is false, and
(lora_alpha / sqrt(r)) BA when use_rslora is true. The full matrices are not
kept. Magnitude is not a measurement of acquired knowledge.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from inspector_engine.inspect_checkpoint import InspectionError
from inspector_engine.roots import PathRejected, configured_roots, present_path, resolve_checkpoint
from inspector_engine.safetensors_meta import MalformedSafetensors, read_header

LIMITS = (
    "Each value is the effective update, scale times B times A, for one adapted linear projection. "
    "The scale is lora_alpha / r, or lora_alpha / sqrt(r) when use_rslora is true. "
    "Frobenius norms, maximum absolute values, and mean absolute values describe that product. "
    "The total norm is the square root of the sum of squared Frobenius norms. "
    "Relative magnitude divides the update norm by the Frobenius norm of the corresponding base matrix. "
    "It is omitted when no compatible base checkpoint inside the configured directories is available. "
    "A repository id in the configuration is used only when that repository has been downloaded into this library. "
    "Alpha and rank are read from adapter_config.json. Alpha is not stored in the A or B tensors. "
    "Rank is confirmed when those tensor shapes contain it. "
    "These magnitudes are not a measurement of acquired knowledge."
)
IGNORED_CONFIG_FIELD = "base_model_name_or_path"
FACTOR_NAME = re.compile(
    r"^(?P<prefix>.+)\.lora_(?P<side>[AB])(?:\.(?P<adapter>[A-Za-z0-9_-]+))?\.weight$"
)
LAYER_IN_NAME = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
REPO_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
CHUNK = 128
NO_BASE = "No relative magnitude was calculated. No base checkpoint inside the configured directories was available."


def differing_fields(research: dict, packaged: dict) -> list[str]:
    keys = set(research) | set(packaged)
    return sorted(key for key in keys if key != IGNORED_CONFIG_FIELD and research.get(key) != packaged.get(key))


def lora_scale(config: dict) -> tuple[float, str]:
    if config.get("peft_type") != "LORA":
        raise InspectionError("unsupported_adapter", "Effective updates are defined here for peft_type LORA.", 400)
    if config.get("use_dora") or config.get("fan_in_fan_out") or config.get("use_qalora"):
        raise InspectionError(
            "unsupported_adapter",
            "DoRA, QA-LoRA, and fan_in_fan_out change the update and are not measured here.",
            400,
        )
    if config.get("bias") not in (None, "none") or config.get("lora_bias"):
        raise InspectionError("unsupported_adapter", "A LoRA bias is present. This measurement is the matrix product only.", 400)
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise InspectionError("unsupported_adapter", "Per-module rank or alpha patterns are present. One scale would not describe them.", 400)
    rank = config.get("r")
    alpha = config.get("lora_alpha")
    if isinstance(rank, bool) or not isinstance(rank, (int, float)) or rank <= 0:
        raise InspectionError("unsupported_adapter", "The configuration has no positive LoRA rank.", 400)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise InspectionError("unsupported_adapter", "The configuration has no LoRA alpha.", 400)
    if config.get("use_rslora"):
        return float(alpha) / math.sqrt(float(rank)), "lora_alpha / sqrt(r)"
    return float(alpha) / float(rank), "lora_alpha / r"


def measure_effective_update(path_text: str, roots: list[Path] | None = None) -> dict:
    """Measure one checkpoint. A research configuration is used when it sits beside the file."""
    try:
        adapter = resolve_checkpoint(path_text, roots)
    except PathRejected as exc:
        raise InspectionError(exc.code, exc.message, exc.status) from exc
    packaged = adapter.parent / "adapter_config.json"
    if not packaged.is_file():
        raise InspectionError("config_unreadable", "No adapter_config.json is beside this checkpoint.", 400)
    research = adapter.parent / "adapter_config.research.json"
    research_path = research if research.is_file() else packaged
    base, missing_reason = _optional_base(adapter, packaged, roots)
    report = measure_effective_lora(adapter, research_path, packaged, base)
    if not report["relative_measured"] and missing_reason:
        report["relative_note"] = missing_reason
    report["configuration_agreement"]["configuration_file_count"] = 1 if research_path == packaged else 2
    report["checkpoint"] = {"name": adapter.name, "path": str(adapter), **present_path(adapter)}
    return report


def _optional_base(adapter: Path, packaged: Path, roots: list[Path] | None) -> tuple[Path | None, str]:
    """Resolve a base checkpoint. A Hub repository id counts only after it is in this library."""
    try:
        data = json.loads(packaged.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, NO_BASE
    raw = data.get("base_model_name_or_path") if isinstance(data, dict) else None
    if not isinstance(raw, str) or not raw.strip():
        return None, "No relative magnitude was calculated. The configuration does not name a base model."
    text = raw.strip()
    local = _local_base_candidate(adapter, text)
    if local is not None and local.is_file():
        try:
            return resolve_checkpoint(str(local), roots), ""
        except PathRejected:
            if _repository_id(text) is None:
                return None, NO_BASE
    repo = _repository_id(text)
    if repo is None:
        return None, NO_BASE
    found, kind = _library_base(repo, roots)
    if kind == "file" and found is not None:
        try:
            return resolve_checkpoint(str(found), roots), ""
        except PathRejected:
            return None, NO_BASE
    if kind == "sharded":
        return None, (
            "No relative magnitude was calculated. "
            f"The base repository {repo} is in the library as more than one shard."
        )
    return None, f"No relative magnitude was calculated. The base repository {repo} is not in this library."


def _repository_id(text: str) -> str | None:
    if text.count("/") != 1 or "\\" in text or ".." in text.split("/"):
        return None
    owner, name = text.split("/")
    if REPO_PART.fullmatch(owner) is None or REPO_PART.fullmatch(name) is None or Path(name).suffix:
        return None
    return text


def _local_base_candidate(adapter: Path, text: str) -> Path | None:
    if _repository_id(text) is not None and not Path(text).is_absolute():
        return None
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = adapter.parent / candidate
    if candidate.is_dir() or not candidate.suffix:
        candidate = candidate / "model.safetensors"
    return candidate


def _library_base(repo: str, roots: list[Path] | None) -> tuple[Path | None, str]:
    allowed = configured_roots() if roots is None else roots
    folder = repo.replace("/", "__")
    saw_shard = False
    for root in allowed:
        directory = (root / folder).resolve()
        try:
            directory.relative_to(root.resolve())
        except ValueError:
            continue
        if not directory.is_dir():
            continue
        single = directory / "model.safetensors"
        if single.is_file():
            return single, "file"
        if (directory / "model.safetensors.index.json").is_file() or any(directory.glob("*.safetensors")):
            saw_shard = True
    return None, "sharded" if saw_shard else "missing"


def _base_tensor_name(prefix: str) -> str:
    stem = prefix[len("base_model.model.") :] if prefix.startswith("base_model.model.") else prefix
    return stem + ".weight"


def measure_effective_lora(
    adapter_path: Path,
    research_config: Path,
    packaged_config: Path,
    base_path: Path | None = None,
) -> dict:
    research, packaged = _read_config(research_config), _read_config(packaged_config)
    diffs = differing_fields(research, packaged)
    if diffs:
        raise InspectionError(
            "config_disagreement",
            "The packaged configuration differs from the research configuration in: " + ", ".join(diffs) + ".",
            409,
        )
    scale, rule = lora_scale(research)
    rank = int(research["r"])
    try:
        pairs, data_offset = _pairs(adapter_path)
        base_index = _base_index(base_path) if base_path is not None else None
    except MalformedSafetensors as exc:
        raise InspectionError("malformed_safetensors", str(exc), 400) from exc
    measured = []
    sum_squares = 0.0
    with adapter_path.open("rb") as adapter_file, _optional(base_path) as base_file:
        for pair in pairs:
            left = _load(adapter_file, data_offset, pair["A"])
            right = _load(adapter_file, data_offset, pair["B"])
            _check_shapes(pair, left, right, rank)
            frobenius, max_abs, mean_abs = _update_stats(left, right, scale)
            base_name = pair["base_name"]
            relative = None
            base_norm = None
            if base_index is not None and base_file is not None and base_name in base_index["tensors"]:
                base = _load(base_file, base_index["offset"], base_index["tensors"][base_name])
                if base.shape != (right.shape[0], left.shape[1]):
                    raise InspectionError(
                        "shape_mismatch",
                        f"{base_name} has shape {list(base.shape)}, which does not match the LoRA product.",
                        400,
                    )
                base_norm = float(np.linalg.norm(base))
                relative = frobenius / base_norm if base_norm else None
                del base
            del left, right
            sum_squares += frobenius ** 2
            measured.append(
                {
                    "layer": pair["layer"],
                    "module": pair["module"],
                    "projection": pair["projection"],
                    "frobenius_norm": frobenius,
                    "maximum_absolute": max_abs,
                    "mean_absolute": mean_abs,
                    "base_frobenius_norm": base_norm,
                    "relative_frobenius": relative,
                }
            )
    measured.sort(key=lambda row: (row["layer"], row["module"]))
    base_complete = bool(measured) and all(row["base_frobenius_norm"] is not None for row in measured)
    if base_complete:
        base_total = math.sqrt(sum(row["base_frobenius_norm"] ** 2 for row in measured))
        relative_total = math.sqrt(sum_squares) / base_total if base_total else None
        relative_note = "The relative magnitude uses the Frobenius norms of the matching projections in the resolved base checkpoint."
    elif base_path is not None:
        base_total = None
        relative_total = None
        relative_note = (
            "The base checkpoint does not contain every adapted projection, "
            "so no total relative magnitude was calculated."
        )
    else:
        base_total = None
        relative_total = None
        relative_note = NO_BASE
    report = {
        "ok": True,
        "instrument": "effective_lora_update",
        "limits": LIMITS,
        "configuration_agreement": {
            "agreed": True,
            "ignored_field": IGNORED_CONFIG_FIELD,
            "differing_fields": [],
            "scale": scale,
            "scale_rule": rule,
            "r": rank,
            "lora_alpha": research.get("lora_alpha"),
            "use_rslora": bool(research.get("use_rslora")),
            "target_modules": research.get("target_modules"),
        },
        "pair_count": len(measured),
        "total_frobenius_norm": math.sqrt(sum_squares),
        "total_base_frobenius_norm": base_total,
        "total_relative_frobenius": relative_total,
        "relative_measured": base_complete,
        "relative_note": relative_note,
        "base_checkpoint": (
            {"name": base_path.name, "path": str(base_path), **present_path(base_path)} if base_path is not None else None
        ),
        "sources": {
            "lora_alpha": "adapter_config.json. Alpha is not stored in the A or B tensors.",
            "r": "adapter_config.json, confirmed by the A and B shapes.",
            "scale": f"Computed from the declared alpha and rank by {rule}.",
        },
        "by_projection": _aggregate(measured, "projection"),
        "by_layer": _aggregate(measured, "layer"),
        "pairs": measured,
    }
    return report


def _read_config(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InspectionError("config_unreadable", f"The configuration {path.name} cannot be read.", 400) from exc
    if not isinstance(data, dict):
        raise InspectionError("config_unreadable", f"The configuration {path.name} is not an object.", 400)
    return data


def _pairs(path: Path) -> tuple[list[dict], int]:
    header, header_len = read_header(path)
    grouped: dict[tuple[str, str], dict] = {}
    excluded: list[str] = []
    adapters: set[str] = set()
    for name, info in header.items():
        if name == "__metadata__" or not isinstance(info, dict):
            continue
        match = FACTOR_NAME.fullmatch(name)
        if match is None:
            if "lora_" in Path(name).name:
                excluded.append(name)
            continue
        prefix = match.group("prefix")
        side = match.group("side")
        adapter_name = match.group("adapter") or ""
        layer_match = None
        for candidate in LAYER_IN_NAME.finditer(prefix):
            layer_match = candidate
        if layer_match is None:
            excluded.append(name)
            continue
        adapters.add(adapter_name)
        key = (prefix, adapter_name)
        slot = grouped.setdefault(
            key,
            {
                "prefix": prefix,
                "layer": int(layer_match.group(1)),
                "module": _base_tensor_name(prefix)[: -len(".weight")],
            },
        )
        if side in slot:
            raise InspectionError("unpaired_factor", f"{prefix} has more than one LoRA {side} factor.", 400)
        slot[side] = info
    if len(adapters) > 1:
        raise InspectionError(
            "unsupported_adapter",
            "This file contains more than one named LoRA adapter, so one effective update is not defined.",
            400,
        )
    if excluded:
        shown = ", ".join(excluded[:6])
        extra = "" if len(excluded) <= 6 else f", and {len(excluded) - 6} more"
        raise InspectionError(
            "unsupported_adapter",
            "These LoRA tensors are not matrix pairs in a transformer layer, so the effective update was not calculated: "
            + shown
            + extra
            + ".",
            400,
        )
    pairs = []
    for key in sorted(grouped):
        slot = grouped[key]
        module = slot["module"]
        projection = module.split(".")[-1] if module else ""
        if "A" not in slot or "B" not in slot or not projection:
            raise InspectionError("unpaired_factor", f"{module or slot['prefix']} does not have both A and B.", 400)
        slot["projection"] = projection
        slot["base_name"] = _base_tensor_name(slot["prefix"])
        pairs.append(slot)
    if not pairs:
        raise InspectionError("unpaired_factor", "No LoRA A/B pairs were found.", 400)
    pairs.sort(key=lambda row: (row["layer"], row["module"]))
    return pairs, 8 + header_len


def _base_index(path: Path) -> dict:
    header, header_len = read_header(path)
    tensors = {name: info for name, info in header.items() if name != "__metadata__" and isinstance(info, dict)}
    return {"offset": 8 + header_len, "tensors": tensors}


def _load(handle, data_offset: int, info: dict) -> np.ndarray:
    start, end = info["data_offsets"]
    handle.seek(data_offset + start)
    raw = handle.read(end - start)
    dtype = info["dtype"]
    shape = tuple(info["shape"])
    if dtype == "F32":
        values = np.frombuffer(raw, dtype="<f4")
    elif dtype == "F64":
        values = np.frombuffer(raw, dtype="<f8")
    elif dtype == "BF16":
        wide = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        values = wide.view(np.float32)
    elif dtype == "F16":
        values = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    else:
        raise InspectionError("unsupported_dtype", f"Effective updates cannot read {dtype}.", 400)
    return np.array(values, dtype=np.float64).reshape(shape)


def _check_shapes(pair: dict, left: np.ndarray, right: np.ndarray, rank: int) -> None:
    label = pair.get("module") or f"layer {pair['layer']} {pair['projection']}"
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != rank or right.shape[1] != rank:
        raise InspectionError(
            "shape_mismatch",
            f"{label} has A shape {list(left.shape)} and B shape {list(right.shape)} for rank {rank}.",
            400,
        )


def _update_stats(left: np.ndarray, right: np.ndarray, scale: float) -> tuple[float, float, float]:
    sum_squares = 0.0
    sum_absolute = 0.0
    maximum = 0.0
    count = 0
    for start in range(0, right.shape[0], CHUNK):
        block = (right[start : start + CHUNK] @ left) * scale
        sum_squares += float(np.square(block).sum())
        sum_absolute += float(np.abs(block).sum())
        maximum = max(maximum, float(np.abs(block).max()))
        count += int(block.size)
        del block
    frobenius = math.sqrt(sum_squares)
    gram = abs(scale) * math.sqrt(float(np.sum((right.T @ right) * (left @ left.T))))
    if not math.isclose(frobenius, gram, rel_tol=1e-8, abs_tol=1e-8):
        raise InspectionError("update_check_failed", "The product norm did not match the Gram-matrix check.", 500)
    return frobenius, maximum, sum_absolute / count


BASELINE_DRAWS = 12
BASELINE_SEED = 22001001
BASELINE_LIMITS = (
    "The dense scale is 1 divided by the square root of the number of entries in the concatenated effective update. "
    "The random-factor values are absolute cosines between independent normal A and B factors with the same shapes. "
    "The seed is fixed, so the same shapes produce the same baseline. "
    "This is a chance alignment for an update of this size and rank. "
    "It is not a model of training, and it is not a measurement of acquired knowledge. "
    "These draws are an exploratory comparison. They do not establish statistical significance, "
    "and they are not the distribution of independently trained adapters."
)
ALIGNMENT_LIMITS = (
    "Cosine similarity compares the direction of the effective update, scale times B times A. "
    "The overall cosine treats every corresponding matrix entry as one coordinate, so a larger update contributes more. "
    "The mean pair cosine gives each layer and projection an equal vote. "
    "A value of 1 is the same direction, 0 means the updates are orthogonal, and -1 is the opposite direction. "
    "The comparison does not use the size of the update by itself. "
    "It is not a measurement of acquired knowledge."
)


def load_update_factors(adapter_path: Path, research_config: Path, packaged_config: Path) -> dict:
    """Load the A and B factors used by the effective update. The checkpoint is not modified."""
    research, packaged = _read_config(research_config), _read_config(packaged_config)
    diffs = differing_fields(research, packaged)
    if diffs:
        raise InspectionError(
            "config_disagreement",
            "The packaged configuration differs from the research configuration in: " + ", ".join(diffs) + ".",
            409,
        )
    scale, rule = lora_scale(research)
    rank = int(research["r"])
    try:
        pairs, data_offset = _pairs(adapter_path)
    except MalformedSafetensors as exc:
        raise InspectionError("malformed_safetensors", str(exc), 400) from exc
    loaded = []
    with adapter_path.open("rb") as handle:
        for pair in pairs:
            left = _load(handle, data_offset, pair["A"])
            right = _load(handle, data_offset, pair["B"])
            _check_shapes(pair, left, right, rank)
            frobenius, _max_abs, _mean_abs = _update_stats(left, right, scale)
            loaded.append(
                {
                    "layer": pair["layer"],
                    "module": pair["module"],
                    "projection": pair["projection"],
                    "frobenius_norm": frobenius,
                    "A": left,
                    "B": right,
                }
            )
    loaded.sort(key=lambda row: (row["layer"], row["projection"]))
    return {"scale": scale, "scale_rule": rule, "r": rank, "pairs": loaded}


def factors_for_checkpoint(path_text: str, roots: list[Path] | None = None) -> dict:
    try:
        adapter = resolve_checkpoint(path_text, roots)
    except PathRejected as exc:
        raise InspectionError(exc.code, exc.message, exc.status) from exc
    packaged = adapter.parent / "adapter_config.json"
    if not packaged.is_file():
        raise InspectionError("config_unreadable", "No adapter_config.json is beside this checkpoint.", 400)
    research = adapter.parent / "adapter_config.research.json"
    research_path = research if research.is_file() else packaged
    loaded = load_update_factors(adapter, research_path, packaged)
    loaded["subject_id"] = None
    loaded["name"] = adapter.name
    loaded["path"] = str(adapter)
    return loaded


def align_update_factors(loaded: list[dict]) -> dict:
    """Compare directions of effective updates that cover the same layers and projections."""
    if len(loaded) < 2:
        raise InspectionError("alignment_subjects", "Direction comparison needs at least two effective updates.", 400)
    if len(loaded) > 6:
        raise InspectionError("alignment_subjects", "Direction comparison accepts at most six effective updates.", 400)
    signatures = [tuple((pair["layer"], pair.get("module") or pair["projection"]) for pair in item["pairs"]) for item in loaded]
    if any(signature != signatures[0] for signature in signatures):
        raise InspectionError(
            "pair_mismatch",
            "The effective updates do not cover the same layers and projections, so their directions are not compared.",
            400,
        )
    comparisons = []
    for left_index, left in enumerate(loaded):
        for right in loaded[left_index + 1 :]:
            comparisons.append(_align_pair(left, right))
    subjects = []
    for item in loaded:
        total = math.sqrt(sum(pair["frobenius_norm"] ** 2 for pair in item["pairs"]))
        subjects.append(
            {
                "subject_id": item.get("subject_id"),
                "name": item.get("name"),
                "total_frobenius_norm": total,
            }
        )
    return {
        "ok": True,
        "instrument": "effective_update_alignment",
        "limits": ALIGNMENT_LIMITS,
        "subjects": subjects,
        "comparisons": comparisons,
        "baseline": alignment_baseline(loaded[0]),
    }


def alignment_baseline(subject: dict, draws: int = BASELINE_DRAWS, seed: int = BASELINE_SEED) -> dict:
    """Chance alignment for effective updates with these shapes. The seed makes the draws repeatable."""
    shapes = [(pair["A"].shape, pair["B"].shape) for pair in subject["pairs"]]
    entries = 0
    for a_shape, b_shape in shapes:
        entries += int(b_shape[0]) * int(a_shape[1])
    rng = np.random.default_rng(seed)
    samples = [_random_factor_cosine(rng, shapes) for _ in range(draws)]
    absolute = [abs(value) for value in samples]
    return {
        "seed": seed,
        "draws": draws,
        "entry_count": entries,
        "dense_isotropic_scale": 1 / math.sqrt(entries) if entries else None,
        "random_factor_median_absolute": float(np.median(absolute)) if absolute else None,
        "random_factor_maximum_absolute": float(max(absolute)) if absolute else None,
        "limits": BASELINE_LIMITS,
    }


def _random_factor_cosine(rng: np.random.Generator, shapes: list[tuple[tuple[int, ...], tuple[int, ...]]]) -> float:
    dot = 0.0
    left_square = 0.0
    right_square = 0.0
    for a_shape, b_shape in shapes:
        left_a = rng.normal(size=a_shape)
        left_b = rng.normal(size=b_shape)
        right_a = rng.normal(size=a_shape)
        right_b = rng.normal(size=b_shape)
        dot += float(np.sum((left_b.T @ right_b) * (left_a @ right_a.T)))
        left_square += float(np.sum((left_b.T @ left_b) * (left_a @ left_a.T)))
        right_square += float(np.sum((right_b.T @ right_b) * (right_a @ right_a.T)))
    if not left_square or not right_square:
        return 0.0
    return dot / math.sqrt(left_square * right_square)


def _alignment_label(item: dict) -> str:
    if item.get("subject_id"):
        return str(item["subject_id"])
    return str(item.get("path") or item.get("name") or "update")


def _align_pair(left: dict, right: dict) -> dict:
    rows = []
    for left_pair, right_pair in zip(left["pairs"], right["pairs"]):
        dot = _update_dot(left_pair, right_pair, left["scale"], right["scale"])
        left_norm = left_pair["frobenius_norm"]
        right_norm = right_pair["frobenius_norm"]
        cosine = dot / (left_norm * right_norm) if left_norm and right_norm else None
        rows.append(
            {
                "layer": left_pair["layer"],
                "projection": left_pair["projection"],
                "dot": dot,
                "left_norm": left_norm,
                "right_norm": right_norm,
                "cosine": cosine,
            }
        )
    dot_total = sum(row["dot"] for row in rows)
    left_total = math.sqrt(sum(row["left_norm"] ** 2 for row in rows))
    right_total = math.sqrt(sum(row["right_norm"] ** 2 for row in rows))
    defined = [row["cosine"] for row in rows if row["cosine"] is not None]
    return {
        "left": _alignment_label(left),
        "right": _alignment_label(right),
        "cosine": dot_total / (left_total * right_total) if left_total and right_total else None,
        "mean_pair_cosine": sum(defined) / len(defined) if defined else None,
        "pair_count": len(rows),
        "by_projection": _cosine_aggregate(rows, "projection"),
        "by_layer": _cosine_aggregate(rows, "layer"),
    }


def _update_dot(left_pair: dict, right_pair: dict, scale_left: float, scale_right: float) -> float:
    """Frobenius inner product of scale×BA without building the full update matrices."""
    cross_b = left_pair["B"].T @ right_pair["B"]
    cross_a = left_pair["A"] @ right_pair["A"].T
    return float(scale_left * scale_right * np.sum(cross_b * cross_a))


def _cosine_aggregate(rows: list[dict], key: str) -> list[dict]:
    grouped: dict[object, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row[key], []).append(row)
    output = []
    for label in sorted(grouped):
        members = grouped[label]
        dot = sum(item["dot"] for item in members)
        left_norm = math.sqrt(sum(item["left_norm"] ** 2 for item in members))
        right_norm = math.sqrt(sum(item["right_norm"] ** 2 for item in members))
        output.append(
            {
                "label": label,
                "pair_count": len(members),
                "cosine": dot / (left_norm * right_norm) if left_norm and right_norm else None,
            }
        )
    return output


def _aggregate(rows: list[dict], key: str) -> list[dict]:
    grouped: dict[object, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row[key], []).append(row)
    output = []
    for label in sorted(grouped):
        members = grouped[label]
        total = math.sqrt(sum(item["frobenius_norm"] ** 2 for item in members))
        output.append({"label": label, "pair_count": len(members), "frobenius_norm": total})
    return output


class _optional:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.handle = None

    def __enter__(self):
        if self.path is not None:
            self.handle = self.path.open("rb")
        return self.handle

    def __exit__(self, *_args) -> None:
        if self.handle is not None:
            self.handle.close()
