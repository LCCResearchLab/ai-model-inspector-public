# Authors: LCC Research Lab
# Project: From Document to Evidence
"""Checkpoint directories the API is allowed to read."""

from __future__ import annotations

import os
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent.parent
ROOTS_FILE = ENGINE_DIR / "checkpoint_roots.txt"


class PathRejected(Exception):
    def __init__(self, code: str, message: str, status: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def project_root() -> Path:
    """The AI Model Inspector project directory.

    This is the parent of ``inspector-engine``, taken from this file's location.
    It is not a path written into the program for one machine.
    """
    return Path(__file__).resolve().parents[2]


def present_path(path: Path | str | None) -> dict[str, str | None]:
    """Describe a path for the screen. This does not grant access to the file."""
    if path is None or not str(path).strip():
        return {"path_scope": "unavailable", "path_label": "Project-relative path", "path_display": None}
    candidate = Path(str(path)).expanduser()
    try:
        resolved = candidate.resolve()
    except OSError:
        resolved = candidate
    try:
        relative = resolved.relative_to(project_root().resolve()).as_posix()
    except ValueError:
        return {
            "path_scope": "external",
            "path_label": "External file",
            "path_display": _external_description(resolved),
        }
    return {"path_scope": "project", "path_label": "Project-relative path", "path_display": relative}


def _external_description(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(Path.home().resolve()).as_posix()
    except (OSError, ValueError):
        folder = path.parent.name or "another folder"
        return f"Outside this project, in {folder}."
    return f"Outside this project, under the home directory ({relative})."


def configured_roots() -> list[Path]:
    raw = os.environ.get("INSPECTOR_CHECKPOINT_DIRS", "")
    entries = [part.strip() for part in raw.split(os.pathsep) if part.strip()] if raw else []
    if ROOTS_FILE.is_file():
        for line in ROOTS_FILE.read_text(encoding="utf-8").splitlines():
            text = line.split("#", 1)[0].strip()
            if text:
                entries.append(text)
    roots: list[Path] = []
    for entry in entries:
        path = Path(entry).expanduser()
        if not path.is_absolute():
            path = ENGINE_DIR / path
        if _broad_directory(path):
            continue
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved.is_dir() and not _broad_directory(resolved):
            roots.append(resolved)
    library = (ENGINE_DIR / "library").resolve()
    if library.is_dir() and library not in roots:
        roots.append(library)
    bundled = (ENGINE_DIR / "bundled").resolve()
    if bundled.is_dir() and bundled not in roots:
        roots.append(bundled)
    return roots


def resolve_checkpoint(path_text: str, roots: list[Path] | None = None) -> Path:
    allowed = configured_roots() if roots is None else roots
    if not allowed:
        raise PathRejected(
            "not_configured",
            "No checkpoint directories are configured.",
            400,
        )
    text = (path_text or "").strip()
    if not text:
        raise PathRejected("missing_path", "A checkpoint path is required.", 400)
    if "\x00" in text:
        raise PathRejected("path_not_allowed", "The checkpoint path is not allowed.", 400)
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        # A project-relative path is resolved first. The directory check below
        # still decides whether the resolved file may be read.
        candidate = project_root() / candidate
    try:
        resolved = candidate.resolve(strict=False)
    except OSError as exc:
        raise PathRejected("path_not_allowed", "The checkpoint path cannot be resolved.", 400) from exc
    if resolved.suffix.lower() != ".safetensors":
        raise PathRejected(
            "unsupported_format",
            "This instrument reads .safetensors files.",
            400,
        )
    if not _inside_any(resolved, allowed):
        raise PathRejected(
            "path_not_allowed",
            "The checkpoint is outside the configured directories.",
            400,
        )
    if not resolved.exists():
        raise PathRejected("missing_file", "The checkpoint file does not exist.", 404)
    if not resolved.is_file():
        raise PathRejected("missing_file", "The checkpoint path is not a file.", 400)
    return resolved


def _broad_directory(path: Path) -> bool:
    """A directory that would let one path read unrelated files on this computer."""
    try:
        resolved = path.resolve()
    except OSError:
        return True
    blocked = []
    for raw in ("/", "/Users", "/Volumes", "/private", "/home", "/tmp"):
        try:
            blocked.append(Path(raw).resolve())
        except OSError:
            continue
    try:
        blocked.append(Path.home().resolve())
    except OSError:
        pass
    return resolved in blocked


def _inside_any(path: Path, roots: list[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        return True
    return False
