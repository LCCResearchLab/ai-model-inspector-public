# Authors: LCC Research Lab
# Project: From Document to Evidence
"""Read declared architecture files without writing, and compare them with observed tensors."""

from __future__ import annotations

import json
import re
from pathlib import Path

from inspector_engine.roots import _inside_any

CONFIG_LIMIT = 2_000_000

PRIMARY_FIELDS = (
    ("model_type", ("model_type",)),
    ("architecture_name", ("architectures", "architecture")),
    ("num_hidden_layers", ("num_hidden_layers", "n_layer", "num_layers", "n_layers")),
    ("hidden_size", ("hidden_size", "n_embd", "d_model")),
    ("num_attention_heads", ("num_attention_heads", "n_head", "num_heads")),
    ("vocab_size", ("vocab_size",)),
)

OPTIONAL_KEYS = (
    "num_key_value_heads",
    "intermediate_size",
    "max_position_embeddings",
    "hidden_act",
    "rms_norm_eps",
    "rope_theta",
    "tie_word_embeddings",
    "num_local_experts",
    "num_experts",
    "num_experts_per_tok",
    "torch_dtype",
    "initializer_range",
    "attention_dropout",
)

ADAPTER_FIELDS = (
    ("peft_type", ("peft_type",)),
    ("fine_tune_type", ("fine_tune_type",)),
    ("base_model_name_or_path", ("base_model_name_or_path",)),
    ("r", ("r", "lora_r")),
    ("lora_alpha", ("lora_alpha",)),
    ("lora_dropout", ("lora_dropout",)),
    ("target_modules", ("target_modules",)),
    ("adapter_num_layers", ("num_layers",)),
    ("task_type", ("task_type",)),
    ("bias", ("bias",)),
)

LAYER_INDEX = re.compile(r"(?:^|\.)(?:layers|h|blocks|block|layer)\.(\d+)(?:\.|$)")
ADAPTER_TENSOR = re.compile(r"(?:^|\.)lora_(?:A|B|embedding_A|embedding_B)(?:\.|$)", re.IGNORECASE)
ADAPTER_MODULE = re.compile(
    r"(?:^|\.)([A-Za-z0-9_]+)\.lora_(?:A|B|embedding_A|embedding_B)(?:\.|$)",
    re.IGNORECASE,
)
EMBEDDING_SUFFIXES = (
    "embed_tokens.weight",
    "wte.weight",
    "embed_in.weight",
    "embeddings.word_embeddings.weight",
)

ARCHITECTURE_LIMITS = (
    "These values are copied from a configuration file. "
    "They describe an intended architecture, not the tensors stored in the checkpoint."
)
ADAPTER_LIMITS = (
    "These values are copied from adapter_config.json. "
    "They describe the adapter that was declared, not the base model and not the stored tensor bytes. "
    "num_layers in this file is the adapter's declared layer selection, not the base model's depth."
)
VALIDATION_LIMITS = (
    "Warnings compare declared files with observed tensor names and shapes. "
    "A warning is not an inspection failure. A LoRA adapter normally stores only a subset of the model."
)


def load_declarations(checkpoint: Path, roots: list[Path] | None = None) -> tuple[dict, dict, dict | None, list[dict]]:
    """Return declared architecture, adapter declaration, referenced base, and read warnings."""
    warnings: list[dict] = []
    directory = checkpoint.parent
    packaged, packaged_warning = _read_json(directory / "config.json")
    adapter_raw, adapter_warning = _read_json(directory / "adapter_config.json")
    if packaged_warning:
        warnings.append(_warning("config_unreadable", packaged_warning))
    if adapter_warning:
        warnings.append(_warning("adapter_config_unreadable", adapter_warning))

    base_reference, base_reference_key = _base_model_reference(adapter_raw)
    referenced_path = _local_config_path(directory, base_reference)
    if referenced_path is None:
        referenced_path = _library_config_path(base_reference, roots)
    referenced_outside = False
    if referenced_path is not None and roots is not None and not _inside_any(referenced_path, roots):
        referenced_outside = True
        referenced_path = None
        warnings.append(
            _warning(
                "base_architecture_outside_roots",
                "adapter_config.json names a base model outside the configured checkpoint directories. "
                "That configuration was not read. The base architecture is unknown.",
            )
        )
    referenced_origin = "base_model_name_or_path" if base_reference_key == "base_model_name_or_path" else "adapter_model_path"
    referenced_raw = None
    referenced_warning = None
    if referenced_path is not None and (packaged is None or referenced_path != (directory / "config.json").resolve()):
        referenced_raw, referenced_warning = _read_json(referenced_path)
        if referenced_warning:
            warnings.append(_warning("config_unreadable", referenced_warning))

    if packaged is not None:
        declared = _architecture_report(packaged, directory / "config.json", "checkpoint_directory")
    elif referenced_raw is not None and referenced_path is not None:
        declared = _architecture_report(referenced_raw, referenced_path, referenced_origin)
    else:
        declared = _empty_architecture()

    referenced = None
    if packaged is not None and referenced_raw is not None and referenced_path is not None:
        referenced = _architecture_report(referenced_raw, referenced_path, referenced_origin)

    adapter = _adapter_report(adapter_raw, directory / "adapter_config.json" if adapter_raw is not None else None)
    if not declared["available"]:
        if referenced_outside:
            pass
        elif adapter["available"] and adapter.get("base_model_path"):
            warnings.append(
                _warning(
                    "base_architecture_unavailable",
                    "adapter_config.json names a base model at "
                    f"{adapter['base_model_path']}. No local config.json could be read from that path. "
                    "The base architecture is unknown.",
                )
            )
        elif adapter_raw is None and packaged is None and not packaged_warning:
            warnings.append(
                _warning(
                    "declared_architecture_unavailable",
                    "No config.json was found beside this checkpoint. Declared architecture is unknown.",
                )
            )
    return declared, adapter, referenced, warnings


def classify_observed(records: list[dict], metadata: dict[str, str], adapter: dict) -> dict:
    names = [str(row["name"]) for row in records]
    layer_indices = sorted({int(match) for name in names for match in LAYER_INDEX.findall(name)})
    modules = sorted({match for name in names for match in ADAPTER_MODULE.findall(name)})
    evidence: list[str] = []
    declared_kind = str(adapter.get("peft_type") or adapter.get("fine_tune_type") or "").lower()
    if adapter.get("available") and (
        declared_kind in {"lora", "dora", "adalora"} or adapter.get("target_modules") or adapter.get("r") is not None
    ):
        evidence.append("adapter_config.json")
    metadata_keys = {key.lower() for key in metadata}
    if metadata_keys & {"peft_type", "base_model_name_or_path", "lora_alpha", "target_modules"}:
        evidence.append("safetensors metadata")
    if any(ADAPTER_TENSOR.search(name) for name in names):
        evidence.append("tensor names")
    if evidence:
        return {
            "classification": "adapter",
            "classification_basis": "Adapter evidence from " + ", ".join(evidence) + ".",
            "layer_indices": layer_indices,
            "adapter_modules": modules,
        }
    return {
        "classification": "unknown",
        "classification_basis": "No adapter evidence was found. Full-model coverage is decided only when a declared layer count can be compared.",
        "layer_indices": layer_indices,
        "adapter_modules": modules,
    }


def apply_layer_classification(observed: dict, declared: dict) -> dict:
    """Classify a non-adapter checkpoint only when declared layer indices can be checked."""
    if observed["classification"] == "adapter":
        return observed
    declared_layers = _as_int(declared.get("num_hidden_layers"))
    found = list(observed["layer_indices"])
    if declared_layers is None:
        observed["classification"] = "unknown"
        observed["classification_basis"] = (
            "The checkpoint has no adapter evidence, and no declared layer count was available for comparison."
        )
        return observed
    expected = set(range(declared_layers))
    if found and set(found) == expected:
        observed["classification"] = "full_model"
        observed["classification_basis"] = (
            f"Tensor names include every layer index declared in the configuration, 0 through {declared_layers - 1}."
        )
        return observed
    if found and set(found) != expected:
        observed["classification"] = "unknown_subset"
        observed["classification_basis"] = (
            "Tensor names include some declared layer indices and omit others. "
            "This is not adapter evidence, so the checkpoint is an unknown subset."
        )
        return observed
    observed["classification"] = "unknown"
    observed["classification_basis"] = (
        "The configuration declares a layer count, and the tensor names do not include recognizable layer indices."
    )
    return observed


def validation_warnings(
    records: list[dict],
    declared: dict,
    adapter: dict,
    referenced: dict | None,
    observed: dict,
) -> list[dict]:
    warnings: list[dict] = []
    declared_layers = _as_int(declared.get("num_hidden_layers"))
    found = list(observed["layer_indices"])
    if observed["classification"] == "adapter" and declared["available"]:
        layer_text = (
            f" The configuration declares {declared_layers} layers. Observed adapter tensor names mention layers {found}."
            if declared_layers is not None
            else ""
        )
        warnings.append(
            _warning(
                "adapter_subset",
                "The declared architecture describes an intended model. "
                "The checkpoint stores adapter tensors, which are normally only a subset of that model."
                + layer_text,
            )
        )
    elif observed["classification"] == "unknown_subset" and declared_layers is not None:
        warnings.append(
            _warning(
                "layer_coverage",
                f"The configuration declares {declared_layers} layers. "
                f"Observed tensor names include layer indices {found}. "
                "The difference is reported as a warning, not an inspection failure.",
            )
        )
    elif (
        observed["classification"] != "adapter"
        and declared_layers is not None
        and not found
        and declared["available"]
    ):
        warnings.append(
            _warning(
                "layer_indices_unrecognized",
                f"The configuration declares {declared_layers} layers. "
                "Tensor names do not include recognizable layer indices, so layer coverage was not compared.",
            )
        )

    if adapter.get("available"):
        warnings.extend(_target_warnings(adapter, observed["adapter_modules"]))
        warnings.extend(_rank_warnings(records, adapter))
        warnings.extend(_adapter_layer_warnings(adapter, observed))
    if observed["classification"] == "full_model":
        warnings.extend(_embedding_warnings(records, declared))
    if referenced is not None:
        warnings.extend(_declaration_differences(declared, referenced))
    return _unique(warnings)


def _architecture_report(data: dict, path: Path, origin: str) -> dict:
    fields, unknown, used = _take_fields(data, PRIMARY_FIELDS)
    other = {key: data[key] for key in OPTIONAL_KEYS if key in data and key not in used and _plain(data[key]) is not None}
    other = {key: _plain(value) for key, value in other.items()}
    return {
        "available": True,
        "source": "config.json",
        "path": str(path),
        "origin": origin,
        **fields,
        "unknown_fields": unknown,
        "other_properties": other,
        "limits": ARCHITECTURE_LIMITS,
    }


def _empty_architecture() -> dict:
    fields = {name: None for name, _aliases in PRIMARY_FIELDS}
    return {
        "available": False,
        "source": "unavailable",
        "path": None,
        "origin": "unavailable",
        **fields,
        "unknown_fields": [name for name, _aliases in PRIMARY_FIELDS],
        "other_properties": {},
        "limits": ARCHITECTURE_LIMITS,
    }


def _adapter_report(data: dict | None, path: Path | None) -> dict:
    if data is None or path is None:
        fields = {name: None for name, _aliases in ADAPTER_FIELDS}
        keys = {f"{name}_key": None for name, _aliases in ADAPTER_FIELDS}
        return {
            "available": False,
            "source": "unavailable",
            "path": None,
            **fields,
            **keys,
            "base_model_path": None,
            "base_model_path_key": None,
            "unknown_fields": [name for name, _aliases in ADAPTER_FIELDS],
            "other_properties": {},
            "limits": ADAPTER_LIMITS,
        }
    rank_key = None
    prepared = dict(data)
    if "r" not in prepared and "lora_r" not in prepared:
        params = prepared.get("lora_parameters")
        if isinstance(params, dict) and _plain(params.get("rank")) is not None:
            prepared["r"] = params.get("rank")
            rank_key = "lora_parameters.rank"
    fields, unknown, _used = _take_fields(prepared, ADAPTER_FIELDS)
    if rank_key and fields.get("r") is not None:
        fields["r_key"] = rank_key
    base_path, base_key = _base_model_reference(data)
    other: dict = {}
    params = data.get("lora_parameters")
    if isinstance(params, dict):
        for key in ("scale", "dropout"):
            plain = _plain(params.get(key))
            if plain is not None:
                other[f"lora_parameters.{key}"] = plain
    return {
        "available": True,
        "source": "adapter_config.json",
        "path": str(path),
        **fields,
        "base_model_path": base_path,
        "base_model_path_key": base_key,
        "unknown_fields": unknown,
        "other_properties": other,
        "limits": ADAPTER_LIMITS,
    }


def _take_fields(data: dict, spec: tuple[tuple[str, tuple[str, ...]], ...]) -> tuple[dict, list[str], set[str]]:
    values: dict = {}
    unknown: list[str] = []
    used: set[str] = set()
    for name, aliases in spec:
        chosen = None
        chosen_key = None
        for alias in aliases:
            if alias in data and _plain(data[alias]) is not None:
                chosen = _plain(data[alias])
                chosen_key = alias
                used.add(alias)
                break
        values[name] = chosen
        if chosen_key:
            values[f"{name}_key"] = chosen_key
        else:
            values[f"{name}_key"] = None
            unknown.append(name)
    return values, unknown, used


def _read_json(path: Path) -> tuple[dict | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        size = path.stat().st_size
    except OSError:
        return None, f"{path.name} could not be read."
    if size > CONFIG_LIMIT:
        return None, f"{path.name} is larger than the configuration reader accepts."
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, f"{path.name} is not valid JSON. Declared values from that file are unknown."
    if not isinstance(data, dict):
        return None, f"{path.name} is not a JSON object. Declared values from that file are unknown."
    return data, None


def _base_model_reference(data: dict | None) -> tuple[object, str | None]:
    if not data:
        return None, None
    named = _plain(data.get("base_model_name_or_path"))
    if isinstance(named, str) and named.strip():
        return named, "base_model_name_or_path"
    model = _plain(data.get("model"))
    if isinstance(model, str) and model.strip():
        return model, "model"
    return None, None


def _adapter_layer_warnings(adapter: dict, observed: dict) -> list[dict]:
    tuned = _as_int(adapter.get("adapter_num_layers"))
    if tuned is None:
        return []
    found = list(observed["layer_indices"])
    if not found:
        return [
            _warning(
                "adapter_layer_selection",
                f"adapter_config.json declares num_layers {tuned}. "
                "That field is the adapter file's own layer selection, not a base-model depth. "
                "Observed tensor names do not include recognizable layer indices.",
            )
        ]
    if len(found) != tuned:
        return [
            _warning(
                "adapter_layer_selection",
                f"adapter_config.json declares num_layers {tuned}. "
                "That field is the adapter file's own layer selection, not a base-model depth. "
                f"Observed tensor names mention {len(found)} layer indices: {found}.",
            )
        ]
    return []


def _library_config_path(raw: object, roots: list[Path] | None) -> Path | None:
    """A downloaded library copy of a Hub repository, not a request to the Hub."""
    if not isinstance(raw, str) or roots is None:
        return None
    text = raw.strip()
    if text.count("/") != 1 or "\\" in text or ".." in text.split("/"):
        return None
    owner, name = text.split("/")
    if not owner or not name or Path(name).suffix:
        return None
    folder = f"{owner}__{name}"
    for root in roots:
        candidate = (root / folder / "config.json").resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    return None


def _local_config_path(directory: Path, raw: object) -> Path | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    candidate = Path(raw.strip()).expanduser()
    if not candidate.is_absolute():
        candidate = directory / candidate
    try:
        resolved = candidate.resolve(strict=False)
    except OSError:
        return None
    if resolved.is_dir():
        config = resolved / "config.json"
        return config.resolve() if config.is_file() else None
    if resolved.is_file() and resolved.name == "config.json":
        return resolved
    return None


def _field_value(data: dict | None, key: str) -> object:
    if not data or key not in data:
        return None
    return _plain(data[key])


def _plain(value: object) -> object | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, list) and all(isinstance(item, (str, int, float)) and not isinstance(item, bool) for item in value):
        return value
    return None


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _target_warnings(adapter: dict, observed_modules: list[str]) -> list[dict]:
    declared = adapter.get("target_modules")
    if isinstance(declared, str):
        declared_set = {declared}
    elif isinstance(declared, list) and all(isinstance(item, str) for item in declared):
        declared_set = set(declared)
    else:
        return []
    found = set(observed_modules)
    if not found and not declared_set:
        return []
    warnings = []
    missing = sorted(declared_set - found)
    extra = sorted(found - declared_set)
    if missing:
        warnings.append(
            _warning(
                "declared_targets_absent",
                "adapter_config.json declares target modules "
                + ", ".join(sorted(declared_set))
                + ". Observed adapter tensor names do not include "
                + ", ".join(missing)
                + ".",
            )
        )
    if extra:
        warnings.append(
            _warning(
                "observed_targets_not_declared",
                "Observed adapter tensor names include "
                + ", ".join(extra)
                + ", which adapter_config.json does not list as target modules.",
            )
        )
    return warnings


def _rank_warnings(records: list[dict], adapter: dict) -> list[dict]:
    rank = _as_int(adapter.get("r"))
    if rank is None:
        return []
    mismatched = []
    for row in records:
        if not ADAPTER_TENSOR.search(str(row["name"])):
            continue
        shape = [int(item) for item in row["shape"]]
        if rank not in shape:
            mismatched.append(str(row["name"]))
    if not mismatched:
        return []
    shown = ", ".join(mismatched[:8])
    if len(mismatched) > 8:
        shown += f", and {len(mismatched) - 8} more"
    return [
        _warning(
            "lora_rank_mismatch",
            f"adapter_config.json declares LoRA rank {rank}. These tensors do not have a dimension of that rank: {shown}.",
        )
    ]


def _embedding_warnings(records: list[dict], declared: dict) -> list[dict]:
    vocab = _as_int(declared.get("vocab_size"))
    hidden = _as_int(declared.get("hidden_size"))
    embeddings = [row for row in records if str(row["name"]).endswith(EMBEDDING_SUFFIXES)]
    if not embeddings:
        if vocab is not None:
            return [
                _warning(
                    "embedding_not_observed",
                    f"The configuration declares vocab size {vocab}. No embedding tensor name was observed.",
                )
            ]
        return []
    if vocab is None or hidden is None:
        return []
    warnings = []
    for row in embeddings:
        shape = [int(item) for item in row["shape"]]
        if len(shape) != 2:
            continue
        orientations = {(shape[0], shape[1]), (shape[1], shape[0])}
        if (vocab, hidden) not in orientations:
            warnings.append(
                _warning(
                    "embedding_shape",
                    f"{row['name']} has shape {shape}. "
                    f"The configuration declares vocab size {vocab} and hidden size {hidden}.",
                )
            )
    return warnings


def _declaration_differences(left: dict, right: dict) -> list[dict]:
    compared = ("model_type", "num_hidden_layers", "hidden_size", "num_attention_heads", "vocab_size")
    diffs = []
    for key in compared:
        if left.get(key) is None or right.get(key) is None:
            continue
        if left.get(key) != right.get(key):
            diffs.append(f"{key} is {left.get(key)} beside the checkpoint and {right.get(key)} at the referenced base model")
    if not diffs:
        return []
    return [
        _warning(
            "architecture_declarations_differ",
            "The config.json beside the checkpoint and the referenced base config.json disagree: " + "; ".join(diffs) + ".",
        )
    ]


def merge_warnings(*groups: list[dict]) -> list[dict]:
    return _unique([item for group in groups for item in group])


def _warning(code: str, message: str) -> dict:
    return {"code": code, "message": message}


def _unique(warnings: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    result = []
    for warning in warnings:
        key = (warning["code"], warning["message"])
        if key in seen:
            continue
        seen.add(key)
        result.append(warning)
    return result
