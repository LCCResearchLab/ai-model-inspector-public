# Authors: LCC Research Lab
# Project: From Document to Evidence
"""Discover Hugging Face models and download selected files into the Inspector library.

Discovery never downloads weights. Credentials are read on this computer and are
never returned, logged, or taken from another application.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from inspector_engine.roots import present_path

ENGINE_DIR = Path(__file__).resolve().parent.parent
LIBRARY_DIR = ENGINE_DIR / "library"
BUNDLED_DIR = ENGINE_DIR / "bundled"
ALLOWED_HOSTS = ("huggingface.co", "hf.co")
STORAGE_MARGIN = 1.1
README_LIMIT = 100_000
CHUNK = 64 * 1024

_REPO_PART = r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94})"
_REPO_RE = re.compile(rf"^{_REPO_PART}/{_REPO_PART}$")
_REV_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.RLock()
_INSPECTING: set[str] = set()
_INSPECT_LOCK = threading.Lock()
_SHARD_RE = re.compile(r"-(\d+)-of-(\d+)\.safetensors$", re.IGNORECASE)


def inspection_capabilities(access: str) -> dict:
    """What this access method can support. An inference API is not a weight measurement."""
    local = access == "local_weights"
    remote = access == "remote_files"
    api = access == "inference_api"
    return {
        "model_information": local or remote or api,
        "architecture_details": local or remote,
        "tensor_inspection": local,
        "weight_magnitude": local,
        "weight_direction": local,
        "behavioral_evaluation": False,
    }


class LibraryError(Exception):
    def __init__(self, code: str, message: str, status: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


class HubClient(Protocol):
    def model_info(self, repo: str, token: str) -> dict: ...

    def file_list(self, repo: str, revision: str, token: str) -> list[dict]: ...

    def readme(self, repo: str, revision: str, token: str) -> str | None: ...

    def open_file(self, repo: str, revision: str, filename: str, token: str) -> tuple[int, Iterable[bytes]]: ...

    def search_models(
        self,
        query: str,
        token: str,
        cursor: str,
        *,
        sort: str = "relevance",
        pipeline: str = "",
        author: str = "",
    ) -> tuple[list, str]: ...

    def account_authors(self, token: str) -> list[str]: ...


def read_token() -> str:
    """Token used for Hugging Face requests.

    INSPECTOR_HF_TOKEN overrides a token saved in the macOS Keychain.
    A plaintext token file is not read, and another application's credentials are not read.
    """
    env = os.environ.get("INSPECTOR_HF_TOKEN", "").strip()
    if env:
        return env
    from inspector_engine.credentials import keychain_token

    return keychain_token("huggingface")


def token_configured() -> bool:
    return bool(read_token())


def auth_view() -> dict:
    from inspector_engine.credentials import huggingface_auth

    return huggingface_auth()


def hub_token() -> str:
    """Token attached to a Hub request. A rejected token is omitted so public models still load."""
    token = read_token()
    if not token or auth_view().get("status") == "invalid":
        return ""
    return token


NO_TOKEN_ACCESS = (
    "This repository requires authentication. Public model search, information, and downloads stay available. "
    "Open HF Token to add one. A token does not grant access to every private or gated repository."
)
TOKEN_REJECTED_ACCESS = (
    "The saved HF Token was rejected and is still stored. This repository requires authentication. "
    "Public model search and downloads stay available without it. Open HF Token to replace it."
)
ACCESS_DENIED = (
    "Hugging Face accepted the HF Token, but access to this repository was not granted. "
    "A valid token does not grant access to every private or gated repository."
)
PUBLIC_WITH_REJECTED_TOKEN = (
    "The saved HF Token was rejected, so this public model was read without it. The token is still stored."
)
PRIVATE_TOKEN_REQUIRED = (
    "My Private Models needs a validated HF Token. Public model search stays available."
)
PRIVATE_TOKEN_REJECTED = (
    "The saved HF Token was rejected and is still stored. "
    "My Private Models stays unavailable until the token is replaced. Public model search stays available."
)
PRIVATE_PERMISSION = (
    "Hugging Face accepted the HF Token, but this account cannot list private repositories. "
    "That is a permission limit, not an invalid token, and it is not a missing repository."
)
COMBINED_TOKEN_REJECTED = (
    "The saved HF Token was rejected and is still stored. Public models are shown. "
    "Private repositories stay hidden until the token is replaced."
)
COMBINED_PERMISSION = (
    "Hugging Face accepted the HF Token, but private repositories could not be listed. "
    "Public models are shown. That is a permission limit, not an invalid token."
)


def parse_repo(text: str) -> tuple[str, str]:
    raw = (text or "").strip()
    if not raw:
        raise LibraryError("malformed_repo", "Enter a Hugging Face repository id or URL.", 400)
    revision = "main"
    if "://" in raw or raw.startswith("huggingface.co/") or raw.startswith("hf.co/"):
        if "://" not in raw:
            raw = "https://" + raw
        parsed = urlparse(raw)
        host = (parsed.hostname or "").lower()
        if host not in ALLOWED_HOSTS:
            raise LibraryError("malformed_repo", "The URL is not a Hugging Face repository.", 400)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            raise LibraryError("malformed_repo", "The URL does not include an owner and a repository name.", 400)
        repo = f"{parts[0]}/{parts[1]}"
        if len(parts) >= 4 and parts[2] in {"tree", "blob", "resolve"} and _REV_RE.match(parts[3]):
            revision = parts[3]
        elif len(parts) > 2 and parts[2] not in {"tree", "blob", "resolve"}:
            raise LibraryError("malformed_repo", "The URL is not a Hugging Face repository.", 400)
    else:
        repo = raw.split("?", 1)[0].strip("/")
    if not _REPO_RE.match(repo):
        raise LibraryError(
            "malformed_repo",
            "A repository id looks like owner/name. It cannot contain spaces or paths.",
            400,
        )
    if not _REV_RE.match(revision):
        raise LibraryError("malformed_repo", "The revision is not a valid Hugging Face revision.", 400)
    return repo, revision


def compatible_file(name: str) -> bool:
    base = Path(name).name.lower()
    return base.endswith(".safetensors") or base in {"config.json", "adapter_config.json"}


def downloadable_file(name: str) -> bool:
    """Files the library may store. An index is metadata, not an inspectable checkpoint."""
    return compatible_file(name) or Path(name).name.lower().endswith(".safetensors.index.json")


INDEX_LIMIT = 8_000_000
_UNIDENTIFIED = "This repository has no Safetensors index, so a complete checkpoint cannot be identified."


def _unidentified(reason: str) -> dict:
    return {
        "identifiable": False,
        "reason": reason,
        "shards": [],
        "metadata": [],
        "files": [],
        "shard_count": 0,
        "total_bytes": None,
        "total_label": None,
    }


def _shards_from_index(payload: dict) -> list[str]:
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise LibraryError("incomplete_index", "The Safetensors index does not list weight shards.", 400)
    shards: list[str] = []
    for target in weight_map.values():
        if not isinstance(target, str) or not target.lower().endswith(".safetensors"):
            raise LibraryError("incomplete_index", "The Safetensors index names a file that is not a weight shard.", 400)
        text = safe_relative(target).as_posix()
        if text not in shards:
            shards.append(text)
    if not shards:
        raise LibraryError("incomplete_index", "The Safetensors index does not list weight shards.", 400)
    return sorted(shards)


def _read_json_file(client: HubClient, repo: str, revision: str, name: str, token: str) -> dict:
    remote_size, chunks = client.open_file(repo, revision, name, token)
    if isinstance(remote_size, int) and not isinstance(remote_size, bool) and remote_size > INDEX_LIMIT:
        raise LibraryError("incomplete_index", "The Safetensors index is too large to read.", 400)
    blob = bytearray()
    for chunk in chunks:
        blob.extend(chunk)
        if len(blob) > INDEX_LIMIT:
            raise LibraryError("incomplete_index", "The Safetensors index is too large to read.", 400)
    try:
        payload = json.loads(bytes(blob).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LibraryError("incomplete_index", "The Safetensors index could not be read.", 400) from exc
    if not isinstance(payload, dict):
        raise LibraryError("incomplete_index", "The Safetensors index could not be read.", 400)
    return payload


def _checkpoint_plan(client: HubClient, repo: str, revision: str, token: str, rows: list[dict]) -> dict:
    """Identify a complete checkpoint from a Safetensors index, or one weight file."""
    names = [str(row.get("name") or "") for row in rows if row.get("name")]
    present = set(names)
    indexes = sorted(name for name in names if name.lower().endswith(".safetensors.index.json"))
    metadata: list[str] = []
    for name in ("config.json", "adapter_config.json"):
        if name in present:
            metadata.append(name)
    if indexes:
        shards: list[str] = []
        for index_name in indexes:
            for shard in _shards_from_index(_read_json_file(client, repo, revision, index_name, token)):
                if shard not in shards:
                    shards.append(shard)
        shards.sort()
        missing = [shard for shard in shards if shard not in present]
        if missing:
            return _unidentified(
                "The Safetensors index names files that are not in the repository. A complete checkpoint cannot be identified."
            )
        metadata.extend(indexes)
        files = shards + [name for name in metadata if name not in shards]
        return _plan_record(rows, shards, metadata, files, True, None)
    weights = sorted(name for name in names if Path(name).name.lower().endswith(".safetensors"))
    if len(weights) == 1:
        files = weights + [name for name in metadata if name not in weights]
        return _plan_record(rows, weights, metadata, files, True, None)
    return _unidentified(_UNIDENTIFIED)


def _plan_record(rows: list[dict], shards: list[str], metadata: list[str], files: list[str], identifiable: bool, reason: str | None) -> dict:
    sizes = {}
    known = True
    total = 0
    for row in rows:
        size = row.get("size_bytes")
        if row.get("name"):
            sizes[str(row["name"])] = size
    for name in files:
        size = sizes.get(name)
        if isinstance(size, int) and not isinstance(size, bool) and size >= 0:
            total += size
        else:
            known = False
    return {
        "identifiable": identifiable,
        "reason": reason,
        "shards": shards,
        "metadata": metadata,
        "files": files,
        "shard_count": len(shards),
        "total_bytes": total if known else None,
        "total_label": format_bytes(total) if known else None,
    }


def _assess_checkpoint(required_shards: list[str], required_metadata: list[str], have: set[str]) -> dict:
    if not required_shards:
        return {
            "checkpoint_status": "partial",
            "checkpoint_label": "Partial — individual shard inspection only",
            "weight_shards_present": 0,
            "weight_shards_required": 0,
        }
    present = [name for name in required_shards if name in have]
    metadata_ready = all(name in have for name in required_metadata)
    complete = len(present) == len(required_shards) and metadata_ready
    return {
        "checkpoint_status": "complete" if complete else "partial",
        "checkpoint_label": (
            "Complete checkpoint"
            if complete
            else f"Partial checkpoint — {len(present)} of {len(required_shards)} shards downloaded."
        ),
        "weight_shards_present": len(present),
        "weight_shards_required": len(required_shards),
    }


def safe_relative(name: str) -> Path:
    text = (name or "").strip()
    if not text or text.startswith(("/", "\\")) or "\\" in text:
        raise LibraryError("path_rejected", "That file name is not allowed.", 400)
    rel = Path(text)
    if rel.is_absolute() or ".." in rel.parts:
        raise LibraryError("path_rejected", "That file name is not allowed.", 400)
    return rel


def format_bytes(size: int | None) -> str | None:
    if size is None:
        return None
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def free_bytes(path: Path) -> int:
    target = path if path.exists() else path.parent
    return shutil.disk_usage(target).free


def library_space(library: Path | None = None) -> dict:
    """Free space on the volume that holds downloaded models."""
    amount = free_bytes(library or LIBRARY_DIR)
    return {"free_bytes": amount, "free_label": format_bytes(amount)}


def _require_disk_space(root: Path, needed: int) -> None:
    """Refuse before any weight bytes are written. The reserve matches the download screen."""
    if needed > 0 and free_bytes(root) < int(needed * STORAGE_MARGIN):
        raise LibraryError(
            "insufficient_disk",
            "There is not enough free disk space for the selected files. Nothing was downloaded.",
            400,
        )


def _manifest_index(dest_root: Path) -> tuple[str | None, dict[str, dict]]:
    if dest_root.is_symlink() or not dest_root.is_dir():
        return None, {}
    path = dest_root / "manifest.json"
    if path.is_symlink() or not path.is_file():
        return None, {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, {}
    if not isinstance(loaded, dict):
        return None, {}
    revision = loaded.get("revision") if isinstance(loaded.get("revision"), str) else None
    rows: dict[str, dict] = {}
    for row in loaded.get("files") or []:
        if isinstance(row, dict) and isinstance(row.get("name"), str):
            rows[row["name"]] = row
    return revision, rows


def _reuse_record(
    local: Path | None,
    size: int | None,
    checksum: str | None,
    revision: str,
    row: dict | None,
    manifest_revision: str | None,
    *,
    verify_checksum: bool,
) -> dict | None:
    """Keep a local file only for this repository, revision, and expected size.

    The path is already confined to the repository directory. A temporary
    ``.part`` file is not a completed download. A recorded checksum mismatch
    is not reusable. When a checksum is published and this file is not already
    verified against it, the bytes are hashed only if ``verify_checksum`` is set.
    """
    if local is None or size is None or local.is_symlink() or not local.is_file() or local.name.endswith(".part"):
        return None
    try:
        actual = local.stat().st_size
    except OSError:
        return None
    if actual != size:
        return None
    recorded = None
    if isinstance(row, dict) and isinstance(row.get("revision"), str):
        recorded = row["revision"]
    elif isinstance(manifest_revision, str):
        recorded = manifest_revision
    if recorded != revision:
        return None
    if not checksum:
        return {"expected_sha256": None, "observed_sha256": None, "integrity": "unavailable"}
    integrity = row.get("integrity") if isinstance(row, dict) else None
    expected = row.get("expected_sha256") if isinstance(row, dict) else None
    if integrity == "mismatch":
        return None
    if integrity == "verified" and expected == checksum:
        observed = row.get("observed_sha256") if isinstance(row, dict) and isinstance(row.get("observed_sha256"), str) else None
        return {"expected_sha256": checksum, "observed_sha256": observed, "integrity": "verified"}
    if not verify_checksum:
        return None
    checked = _file_integrity(local, checksum)
    if checked["integrity"] != "verified":
        return None
    return checked


def _space_for_download(
    dest_root: Path,
    planned: list[tuple[str, int | None, str | None]],
    revision: str,
    *,
    verify_checksum: bool,
) -> tuple[int, list[dict | None]]:
    """Bytes that still have to be written. A reusable file contributes nothing."""
    manifest_revision, rows = _manifest_index(dest_root)
    present = dest_root.is_dir() and not dest_root.is_symlink()
    needed = 0
    reuse: list[dict | None] = []
    for name, size, checksum in planned:
        local = _library_file(dest_root, name) if present else None
        record = _reuse_record(
            local,
            size,
            checksum,
            revision,
            rows.get(name),
            manifest_revision,
            verify_checksum=verify_checksum,
        )
        reuse.append(record)
        if record is None and size is not None:
            needed += size
    return needed, reuse


SEARCH_LIMIT = 20
_OTHER_PAGE_LIMIT = 4
_PRIVATE_PAGE_LIMIT = 8
_PRIVATE_AUTHOR_LIMIT = 30
_SORTS = {"relevance", "downloads", "updated"}
_TASKS = {"all", "text", "image", "other"}
_PIPELINE_FOR_TASK = {"text": "text-generation", "image": "text-to-image"}
_TEXT_TASKS = {"text-generation", "text2text-generation", "conversational"}
_IMAGE_TASKS = {"text-to-image", "image-to-image", "unconditional-image-generation"}
_TASK_LABELS = {
    "text-generation": "Text Generation",
    "text2text-generation": "Text-to-Text",
    "conversational": "Conversational",
    "text-to-image": "Text-to-Image",
    "image-to-image": "Image-to-Image",
    "unconditional-image-generation": "Image Generation",
    "image-to-text": "Image-to-Text",
    "image-text-to-text": "Image-Text-to-Text",
    "text-to-speech": "Text-to-Speech",
    "text-to-audio": "Text-to-Audio",
    "text-to-video": "Text-to-Video",
    "image-to-video": "Image-to-Video",
    "automatic-speech-recognition": "Speech Recognition",
    "token-classification": "Token Classification",
    "question-answering": "Question Answering",
    "fill-mask": "Fill-Mask",
    "feature-extraction": "Feature Extraction",
    "sentence-similarity": "Sentence Similarity",
    "translation": "Translation",
    "summarization": "Summarization",
    "zero-shot-classification": "Zero-Shot Classification",
}
_PUBLISHER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}\Z")


def search_query(text: str) -> str:
    """Search text, or the repository id when the user pastes a Hugging Face URL."""
    raw = (text or "").strip()
    if not raw or any(char in raw for char in ("\n", "\r", "\x00")):
        raise LibraryError("empty_search", "Enter a model name, keyword, repository id, or Hugging Face URL.", 400)
    if len(raw) > 200:
        raise LibraryError("empty_search", "Search is too long. Enter a shorter model name or repository ID.", 400)
    if "://" in raw or raw.startswith("huggingface.co/") or raw.startswith("hf.co/"):
        repo, _revision = parse_repo(raw)
        return repo
    return raw


def search_models(
    text: str,
    *,
    cursor: str = "",
    sort: str = "relevance",
    task: str = "all",
    scope: str = "public",
    hub: HubClient | None = None,
) -> dict:
    """Search the Hub catalog. This does not download a model.

    Repositories whose publisher name matches the search are listed first.
    Community models remain in the results. Sort and task narrow the same catalog.
    A combined search also lists private repositories the saved token can access.
    """
    if sort not in _SORTS:
        raise LibraryError("rejected", "Choose Relevance, Most Downloads, or Recently Updated.", 400)
    if task not in _TASKS:
        raise LibraryError("rejected", "Choose All, Text Generation, Image Generation, or Other.", 400)
    if scope not in {"public", "private", "combined"}:
        raise LibraryError("rejected", "That search scope is not valid.", 400)
    if scope == "private":
        return _search_private(text, cursor=cursor, sort=sort, task=task, hub=hub)
    private_offset = 0
    public_cursor = cursor
    if scope == "combined":
        private_offset, public_cursor = _unpack_combined_cursor(cursor)
    if scope == "combined" and private_offset > 0 and not public_cursor:
        report = _empty_catalog(text, sort, task)
    else:
        report = _search_public(text, cursor=public_cursor, sort=sort, task=task, hub=hub)
    if scope != "combined":
        return report
    return _add_private_results(report, text, private_offset=private_offset, sort=sort, task=task, hub=hub)


def _search_public(
    text: str,
    *,
    cursor: str,
    sort: str,
    task: str,
    hub: HubClient | None,
) -> dict:
    query = search_query(text)
    page = _search_cursor(cursor)
    client = hub or UrllibHub()
    pipeline = _PIPELINE_FOR_TASK.get(task, "")
    publisher = _publisher_token(query)

    def fetch(token: str, page_cursor: str, author: str) -> tuple[list, str]:
        return client.search_models(query, token, page_cursor, sort=sort, pipeline=pipeline, author=author)

    def load(token: str) -> tuple[list, list, str]:
        official: list = []
        rows, next_cursor = fetch(token, page, "")
        if publisher and not page:
            official = _official_rows(fetch, token, rows, publisher)
        if task == "other":
            hops = 0
            while next_cursor and hops < _OTHER_PAGE_LIMIT and _other_count(official) + _other_count(rows) < SEARCH_LIMIT:
                more, next_cursor = fetch(token, next_cursor, "")
                rows.extend(more if isinstance(more, list) else [])
                hops += 1
        return official, rows, next_cursor

    official_rows, rows, next_cursor = _with_hub_auth(load)[0]
    secret = read_token()
    if secret and next_cursor and secret in next_cursor:
        next_cursor = ""
    results = [
        row for row in _merge_hits(official_rows, rows, secret, publisher, task) if row.get("private") is not True
    ]
    auth = auth_view()
    return {
        "ok": True,
        "query": "" if secret and secret in query else query,
        "scope": "public",
        "sort": sort,
        "task": task,
        "results": results,
        "next_cursor": next_cursor or None,
        "direct_repo": _direct_repository(text, secret),
        "hf_auth": auth["status"],
        "hf_auth_label": auth["label"],
        "hf_validated": bool(auth.get("validated")),
    }


def _empty_catalog(text: str, sort: str, task: str) -> dict:
    secret = read_token()
    query = search_query(text)
    auth = auth_view()
    return {
        "ok": True,
        "query": "" if secret and secret in query else query,
        "scope": "public",
        "sort": sort,
        "task": task,
        "results": [],
        "next_cursor": None,
        "direct_repo": _direct_repository(text, secret),
        "hf_auth": auth["status"],
        "hf_auth_label": auth["label"],
        "hf_validated": bool(auth.get("validated")),
    }


def _add_private_results(
    report: dict,
    text: str,
    *,
    private_offset: int,
    sort: str,
    task: str,
    hub: HubClient | None,
) -> dict:
    """Add accessible private repositories without hiding the public page.

    The public catalog does not list private repositories, so they come from a
    second authenticated request. A failure there leaves the public rows in place.
    """
    if auth_view().get("status") == "invalid" or not hub_token():
        if auth_view().get("status") == "invalid":
            report["permission_note"] = COMBINED_TOKEN_REJECTED
            report["scope"] = "combined"
        return report
    try:
        hits, note = _matching_private_hits(text, sort=sort, task=task, hub=hub)
    except LibraryError as exc:
        if exc.code not in {"hub_unauthorized", "hub_token_rejected", "hub_access_denied"}:
            raise
        report["scope"] = "combined"
        report["permission_note"] = COMBINED_TOKEN_REJECTED if exc.code == "hub_token_rejected" else COMBINED_PERMISSION
        return report
    report["scope"] = "combined"
    if note:
        report["permission_note"] = note
    page = hits[private_offset : private_offset + SEARCH_LIMIT]
    report["results"] = _place_private(report["results"], page, sort)
    next_private = private_offset + SEARCH_LIMIT
    more_private = next_private < len(hits)
    hub_cursor = report.get("next_cursor") or ""
    if more_private or hub_cursor:
        report["next_cursor"] = _pack_cursor(next_private if more_private else len(hits), hub_cursor)
    else:
        report["next_cursor"] = None
    return report


def _place_private(public_rows: list[dict], private_rows: list[dict], sort: str) -> list[dict]:
    private_by_repo = {row["repo"]: row for row in private_rows}
    replaced = []
    for row in public_rows:
        replacement = private_by_repo.get(row["repo"])
        replaced.append(replacement if replacement and replacement.get("private") else row)
    seen = {row["repo"] for row in replaced}
    extra = [row for row in private_rows if row["repo"] not in seen]
    if sort == "downloads":
        combined = replaced + extra
        combined.sort(key=lambda row: (-int(row["downloads"] if row.get("downloads") is not None else -1), row["repo"]))
        return combined
    official = [row for row in replaced if row.get("official")]
    rest = [row for row in replaced if not row.get("official")]
    return official + extra + rest


def _search_private(
    text: str,
    *,
    cursor: str,
    sort: str,
    task: str,
    hub: HubClient | None,
) -> dict:
    """List private repositories the token can access, then filter them locally.

    This is not the public catalog search. Nothing from this list is written to disk.
    """
    query = search_query(text)
    hits, note = _matching_private_hits(text, sort=sort, task=task, hub=hub)
    secret = read_token()
    offset = _private_offset(cursor)
    page = hits[offset : offset + SEARCH_LIMIT]
    next_offset = offset + SEARCH_LIMIT
    auth = auth_view()
    return {
        "ok": True,
        "query": "" if secret and secret in query else query,
        "scope": "private",
        "sort": sort,
        "task": task,
        "results": page,
        "next_cursor": str(next_offset) if next_offset < len(hits) else None,
        "direct_repo": _direct_repository(text, secret),
        "permission_note": note,
        "hf_auth": auth["status"],
        "hf_auth_label": auth["label"],
        "hf_validated": True,
    }


def _matching_private_hits(
    text: str,
    *,
    sort: str,
    task: str,
    hub: HubClient | None,
) -> tuple[list[dict], str | None]:
    query = search_query(text)
    token = _private_token()
    client = hub or UrllibHub()
    try:
        authors = client.account_authors(token)
    except LibraryError as exc:
        _raise_private_auth(exc, token)
        raise
    from inspector_engine.credentials import mark_huggingface_validated

    mark_huggingface_validated()
    secret = token
    collected: list[dict] = []
    denied: list[str] = []
    for author in authors[:_PRIVATE_AUTHOR_LIMIT]:
        if secret and secret in author:
            continue
        try:
            collected.extend(_author_models(client, token, author))
        except LibraryError as exc:
            if exc.code in {"hub_unauthorized", "hub_forbidden"}:
                denied.append(author)
                continue
            raise
    if not collected and denied and len(denied) == min(len(authors), _PRIVATE_AUTHOR_LIMIT):
        raise LibraryError("hub_access_denied", PRIVATE_PERMISSION, 403)
    hits = []
    seen: set[str] = set()
    folded = query.casefold()
    for row in collected:
        if not isinstance(row, dict) or row.get("private") is not True:
            continue
        viewed = _search_hit(row, secret)
        if not viewed or viewed["repo"] in seen:
            continue
        if folded not in viewed["repo"].casefold():
            continue
        if task != "all" and _task_group(viewed.get("pipeline")) != task:
            continue
        viewed["official"] = False
        viewed["_downloads"] = viewed.get("downloads") if viewed.get("downloads") is not None else -1
        viewed["_updated"] = str(row.get("lastModified") or "")
        seen.add(viewed["repo"])
        hits.append(viewed)
    if sort == "downloads":
        hits.sort(key=lambda row: (-int(row["_downloads"]), row["repo"]))
    elif sort == "updated":
        hits.sort(key=lambda row: (row["_updated"], row["repo"]), reverse=True)
    else:
        hits.sort(key=lambda row: (not row["repo"].casefold().startswith(folded), row["repo"]))
    for row in hits:
        row.pop("_downloads", None)
        row.pop("_updated", None)
    note = None
    if denied:
        shown = ", ".join(name for name in denied if not secret or secret not in name)
        note = (
            "Hugging Face accepted the token, but these accounts could not be listed: "
            f"{shown}. That is a permission limit, not an invalid token, and it is not a missing repository."
        )
    return hits, note


def _pack_cursor(private_offset: int, hub_cursor: str) -> str:
    payload = json.dumps([private_offset, hub_cursor], separators=(",", ":")).encode()
    return "c." + base64.urlsafe_b64encode(payload).decode().rstrip("=")


def _unpack_combined_cursor(cursor: str) -> tuple[int, str]:
    text = _search_cursor(cursor)
    if not text:
        return 0, ""
    if not text.startswith("c."):
        return 0, text
    padded = text[2:] + "=" * (-len(text[2:]) % 4)
    try:
        offset, hub_cursor = json.loads(base64.urlsafe_b64decode(padded))
    except (ValueError, json.JSONDecodeError):
        raise LibraryError("rejected", "That search page is not valid.", 400) from None
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0 or offset > 100_000:
        raise LibraryError("rejected", "That search page is not valid.", 400)
    if not isinstance(hub_cursor, str) or len(hub_cursor) > 2000:
        raise LibraryError("rejected", "That search page is not valid.", 400)
    return offset, hub_cursor


def _private_token() -> str:
    auth = auth_view()
    if auth.get("status") == "invalid":
        raise LibraryError("hub_token_rejected", PRIVATE_TOKEN_REJECTED, 401)
    token = hub_token()
    if not token:
        raise LibraryError("hub_unauthorized", PRIVATE_TOKEN_REQUIRED, 401)
    return token


def _raise_private_auth(exc: LibraryError, token: str) -> None:
    if exc.code not in {"hub_unauthorized", "hub_forbidden"}:
        return
    from inspector_engine.credentials import note_hub_refusal

    kind = note_hub_refusal(token)
    if kind == "valid":
        raise LibraryError("hub_access_denied", PRIVATE_PERMISSION, 403) from None
    if kind == "rejected":
        raise LibraryError("hub_token_rejected", PRIVATE_TOKEN_REJECTED, 401) from None
    raise exc


def _author_models(client: HubClient, token: str, author: str) -> list:
    rows: list = []
    cursor = ""
    for _page in range(_PRIVATE_PAGE_LIMIT):
        found, cursor = client.search_models("", token, cursor, author=author)
        if isinstance(found, list):
            rows.extend(found)
        if not cursor:
            break
    return rows


def _direct_repository(text: str, secret: str) -> str | None:
    try:
        repo, _revision = parse_repo(text)
    except LibraryError:
        return None
    if secret and secret in repo:
        return None
    return repo


def _private_offset(cursor: str) -> int:
    text = (cursor or "").strip()
    if not text:
        return 0
    if not text.isdigit() or len(text) > 6:
        raise LibraryError("rejected", "That search page is not valid.", 400)
    return int(text)


def discover(text: str, *, hub: HubClient | None = None) -> dict:
    repo, revision = parse_repo(text)
    client = hub or UrllibHub()

    def load(token: str) -> dict:
        return _model_card(client, repo, revision, token)

    card, used = _with_hub_auth(load)
    auth = auth_view()
    card["hf_auth"] = auth["status"]
    card["hf_auth_label"] = auth["label"]
    card["hf_token_configured"] = bool(auth["present"] or read_token())
    if card.get("private") or card.get("gated"):
        card["auth_note"] = (
            "This repository is private or gated. A valid HF Token does not grant access to every such repository."
        )
    elif auth["status"] == "invalid" and not used:
        card["auth_note"] = PUBLIC_WITH_REJECTED_TOKEN
    return card


def _model_card(client: HubClient, repo: str, revision: str, token: str) -> dict:
    info = client.model_info(repo, token)
    files = client.file_list(repo, revision, token)
    description = _description(client.readme(repo, revision, token))
    card = info.get("cardData") if isinstance(info.get("cardData"), dict) else {}
    config = info.get("config") if isinstance(info.get("config"), dict) else {}
    secret = read_token()
    try:
        stored_root = _destination(LIBRARY_DIR, repo)
    except LibraryError:
        stored_root = LIBRARY_DIR / "_rejected"
    manifest_revision, stored_rows = _manifest_index(stored_root)
    listed = []
    for row in files:
        name = str(row.get("name") or "")
        if not name or name.endswith("/") or (secret and secret in name):
            continue
        try:
            safe_relative(name)
        except LibraryError:
            continue
        size = row.get("size_bytes")
        size_int = int(size) if isinstance(size, int) and not isinstance(size, bool) and size >= 0 else None
        checksum = row.get("expected_sha256") if isinstance(row.get("expected_sha256"), str) else _expected_sha256(row)
        local = _library_file(stored_root, name) if stored_root.is_dir() and not stored_root.is_symlink() else None
        reused = _reuse_record(
            local,
            size_int,
            checksum,
            revision,
            stored_rows.get(name),
            manifest_revision,
            verify_checksum=False,
        )
        listed.append(
            {
                "name": name,
                "size_bytes": size_int,
                "size_human": format_bytes(size_int),
                "compatible": compatible_file(name),
                "local_bytes": size_int if reused is not None and size_int is not None else 0,
            }
        )
    listed.sort(key=lambda row: (not row["compatible"], row["name"]))
    try:
        checkpoint = _checkpoint_plan(client, repo, revision, token, listed)
    except LibraryError:
        checkpoint = _unidentified("The Safetensors index could not be read. A complete checkpoint cannot be identified.")
    return {
        "ok": True,
        "repo": repo,
        "revision": revision,
        "card_url": f"https://huggingface.co/{repo}",
        "hf_token_configured": bool(read_token()),
        "private": bool(info.get("private")),
        "gated": _is_gated(info.get("gated")),
        "description": _hide(description, secret),
        "license": _hide(_text(card.get("license") or info.get("license")), secret),
        "pipeline": _hide(_text(info.get("pipeline_tag") or card.get("pipeline_tag")), secret),
        "library": _hide(_text(info.get("library_name")), secret),
        "model_type": _hide(_text(config.get("model_type") or card.get("model_type")), secret),
        "architectures": _hide_names(_names(config.get("architectures") or card.get("architectures")), secret),
        "base_model": _hide(_first_name(card.get("base_model") or card.get("base_models")), secret),
        "tags": (_hide_names(_names(info.get("tags")), secret) or [])[:16],
        "files": listed,
        "checkpoint": checkpoint,
        "storage": library_space(),
        "provider": "huggingface",
        "access": "remote_files",
        "download_status": "not_downloaded",
        "capabilities": inspection_capabilities("remote_files"),
        "limits": (
            "This is the Hub model card and file list. "
            "Looking up a model does not download its weights. "
            "Tensor inspection starts after a compatible file is on this computer."
        ),
    }


@contextmanager
def hold_inspection(path: Path):
    """Remember a file while a read is in progress so deletion can wait."""
    key = str(path.resolve())
    with _INSPECT_LOCK:
        _INSPECTING.add(key)
    try:
        yield
    finally:
        with _INSPECT_LOCK:
            _INSPECTING.discard(key)


def inspection_busy(path: Path) -> bool:
    try:
        key = str(path.resolve())
    except OSError:
        return False
    with _INSPECT_LOCK:
        return key in _INSPECTING


def list_local_models(*, library: Path | None = None) -> dict:
    root = library or LIBRARY_DIR
    models = []
    if root.is_dir():
        for manifest_path in sorted(root.glob("*/manifest.json")):
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and payload.get("repo"):
                public = _public_manifest(payload, manifest_path.parent)
                if public:
                    models.append(public)
    return {
        "ok": True,
        "library_dir": str(root),
        "models": models,
        "bundled": list_bundled_models(),
        "hf_token_configured": token_configured(),
        "hf_auth": auth_view()["status"],
        "hf_auth_label": auth_view()["label"],
        "hf_validated": bool(auth_view().get("validated")),
    }


def list_bundled_models() -> list[dict]:
    models = []
    if not BUNDLED_DIR.is_dir():
        return models
    for manifest_path in sorted(BUNDLED_DIR.glob("*/manifest.json")):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("repo"):
            public = _public_manifest(payload, manifest_path.parent)
            if public:
                public["bundled"] = True
                models.append(public)
    return models


def start_download(
    repo_text: str,
    filenames: list[str],
    *,
    revision: str | None = None,
    mode: str = "selected",
    hub: HubClient | None = None,
    library: Path | None = None,
    wait: bool = False,
) -> dict:
    repo, parsed_revision = parse_repo(repo_text)
    revision = revision or parsed_revision
    if not _REV_RE.match(revision):
        raise LibraryError("malformed_repo", "The revision is not a valid Hugging Face revision.", 400)
    if mode not in {"selected", "complete"}:
        raise LibraryError("rejected", "Choose Download Complete Model or Download Selected Files.", 400)
    client = hub or UrllibHub()

    def listed(token: str) -> tuple[dict, dict]:
        rows = list(client.file_list(repo, revision, token))
        available = {row["name"]: row for row in rows}
        plan = _checkpoint_plan(client, repo, revision, token, rows)
        return available, plan

    loaded, token = _with_hub_auth(listed)
    available, plan = loaded
    if mode == "complete":
        if not plan.get("identifiable"):
            raise LibraryError(
                "incomplete_checkpoint",
                plan.get("reason") or "A complete checkpoint cannot be identified. Nothing was downloaded.",
                400,
            )
        chosen = list(plan["files"])
    else:
        chosen = _validate_selection(filenames)
    missing = [name for name in chosen if name not in available]
    if missing:
        raise LibraryError(
            "missing_file",
            "These files are not in the repository file list: " + ", ".join(missing) + ".",
            404,
        )
    incompatible = [name for name in chosen if not downloadable_file(name)]
    if incompatible:
        raise LibraryError(
            "incompatible_file",
            "Only .safetensors, config.json, adapter_config.json, and a Safetensors index can be downloaded.",
            400,
        )
    root = (library or LIBRARY_DIR).resolve()
    root.mkdir(parents=True, exist_ok=True)
    planned: list[tuple[str, int | None, str | None]] = []
    dest_root = _destination(root, repo)
    for name in chosen:
        size = available[name].get("size_bytes")
        size_int = int(size) if isinstance(size, int) and not isinstance(size, bool) and size >= 0 else None
        expected_sha = available[name].get("expected_sha256")
        checksum = expected_sha if isinstance(expected_sha, str) else None
        planned.append((name, size_int, checksum))
    needed, _shown = _space_for_download(dest_root, planned, revision, verify_checksum=False)
    _require_disk_space(root, needed)
    total = sum(size or 0 for _, size, _checksum in planned)
    job_id = uuid.uuid4().hex[:12]
    job = {
        "ok": True,
        "id": job_id,
        "status": "running",
        "percent": 0,
        "bytes_got": 0,
        "bytes_total": total,
        "message": f"Preparing to download {len(chosen)} file(s) from {repo}.",
        "error": None,
        "repo": repo,
        "revision": revision,
        "cancel_requested": False,
        "manifest": None,
        "checkpoint_plan": plan if plan.get("identifiable") else None,
    }
    with _JOBS_LOCK:
        _JOBS[job_id] = job
    if wait:
        _run_download(job_id, repo, revision, planned, client, token, root, plan)
    else:
        threading.Thread(
            target=_run_download,
            args=(job_id, repo, revision, planned, client, token, root, plan),
            daemon=True,
        ).start()
    return _public_job(job_id)


def job_status(job_id: str) -> dict:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise LibraryError("missing_job", "That download is not on this server.", 404)
        return _public_job(job_id)


def cancel_job(job_id: str) -> dict:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise LibraryError("missing_job", "That download is not on this server.", 404)
        job["cancel_requested"] = True
        if job["status"] == "running":
            job["message"] = "Cancelling the download."
    return _public_job(job_id)


def _run_download(
    job_id: str,
    repo: str,
    revision: str,
    planned: list[tuple[str, int | None, str | None]],
    hub: HubClient,
    token: str,
    library: Path,
    plan: dict | None = None,
) -> None:
    dest_root = _destination(library, repo)
    completed: list[dict] = []
    got = 0
    total = sum(size or 0 for _, size, _checksum in planned) or 1
    needed, reuse = _space_for_download(dest_root, planned, revision, verify_checksum=True)
    try:
        _require_disk_space(library, needed)
    except LibraryError as exc:
        _fail(job_id, exc.message, exc.code)
        return
    try:
        for index, (name, size, expected_sha) in enumerate(planned):
            if _cancelled(job_id):
                _finish_cancel(job_id, dest_root, completed, repo, revision, plan)
                return
            dest = dest_root / safe_relative(name)
            kept = reuse[index]
            if kept is not None:
                got += size or 0
                completed.append({"name": name, "size_bytes": size, "path": str(dest), **kept})
                _progress(job_id, got, total, f"{name} is already in the library.")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_suffix(dest.suffix + ".part")
            remote_size, chunks = _open_for_download(hub, repo, revision, name, token)
            expected = size if size is not None else (remote_size or None)
            written = 0
            try:
                with part.open("wb") as handle:
                    for chunk in chunks:
                        if _cancelled(job_id):
                            raise LibraryError("cancelled", "Download cancelled.", 400)
                        handle.write(chunk)
                        written += len(chunk)
                        _progress(
                            job_id,
                            got + written,
                            total if size is not None else max(total, got + written),
                            f"Downloading {name}: {format_bytes(written)}"
                            + (f" / {format_bytes(expected)}" if expected else ""),
                        )
            except LibraryError:
                part.unlink(missing_ok=True)
                raise
            except Exception:
                part.unlink(missing_ok=True)
                raise
            if expected is not None and written != expected:
                part.unlink(missing_ok=True)
                raise LibraryError(
                    "incomplete_download",
                    f"{name} stopped at {written} of {expected} bytes. The partial file was removed.",
                    400,
                )
            part.replace(dest)
            final_size = dest.stat().st_size
            got += final_size
            if expected_sha:
                _progress(job_id, got, total, f"Checking the published checksum for {name}.")
            completed.append(
                {"name": name, "size_bytes": final_size, "path": str(dest), **_file_integrity(dest, expected_sha)}
            )
        manifest = _write_manifest(dest_root, repo, revision, completed, plan)
        public = _public_manifest(manifest)
        with _JOBS_LOCK:
            job = _JOBS[job_id]
            job["status"] = "done"
            job["percent"] = 100
            job["bytes_got"] = got
            if public and public.get("checkpoint_status") == "complete":
                job["message"] = (
                    f"Downloaded the complete checkpoint ({public['weight_shards_required']} weight shards) from {repo}."
                )
            else:
                detail = (public or {}).get("checkpoint_label") or "Partial checkpoint — individual shard inspection only."
                job["message"] = f"Downloaded {len(completed)} file(s) from {repo}. {detail}"
            job["manifest"] = public
    except LibraryError as exc:
        if exc.code == "cancelled" or _cancelled(job_id):
            _finish_cancel(job_id, dest_root, completed, repo, revision, plan)
            return
        _fail(job_id, exc.message, exc.code)
    except Exception:
        _fail(job_id, "The download failed. No token was written to the library.")


def _finish_cancel(job_id: str, dest_root: Path, completed: list[dict], repo: str, revision: str, plan: dict | None = None) -> None:
    for part in dest_root.rglob("*.part"):
        part.unlink(missing_ok=True)
    manifest = _write_manifest(dest_root, repo, revision, completed, plan) if completed else None
    with _JOBS_LOCK:
        job = _JOBS[job_id]
        job["status"] = "cancelled"
        job["message"] = "The download was cancelled. Partial files were removed."
        job["manifest"] = _public_manifest(manifest) if manifest else None


def _fail(job_id: str, message: str, code: str | None = None) -> None:
    with _JOBS_LOCK:
        job = _JOBS[job_id]
        if job["status"] == "cancelled":
            return
        job["status"] = "error"
        job["error"] = message
        job["error_code"] = code
        job["message"] = message


def _with_hub_auth(action):
    token = hub_token()
    try:
        return action(token), token
    except LibraryError as exc:
        if exc.code not in {"hub_unauthorized", "hub_forbidden"}:
            raise
        if not token:
            raise LibraryError("hub_unauthorized", NO_TOKEN_ACCESS, 401) from None
        from inspector_engine.credentials import note_hub_refusal

        kind = note_hub_refusal(token)
        if kind == "valid":
            raise LibraryError("hub_access_denied", ACCESS_DENIED, 403) from None
        if kind != "rejected":
            raise
        try:
            return action(""), ""
        except LibraryError as retry:
            if retry.code in {"hub_unauthorized", "hub_forbidden"}:
                raise LibraryError("hub_token_rejected", TOKEN_REJECTED_ACCESS, 401) from None
            raise


def _open_for_download(hub: HubClient, repo: str, revision: str, name: str, token: str):
    try:
        return hub.open_file(repo, revision, name, token)
    except LibraryError as exc:
        if exc.code not in {"hub_unauthorized", "hub_forbidden"}:
            raise
        if not token:
            raise LibraryError("hub_unauthorized", NO_TOKEN_ACCESS, 401) from None
        from inspector_engine.credentials import note_hub_refusal

        kind = note_hub_refusal(token)
        if kind == "valid":
            raise LibraryError("hub_access_denied", ACCESS_DENIED, 403) from None
        if kind != "rejected":
            raise
        try:
            return hub.open_file(repo, revision, name, "")
        except LibraryError as retry:
            if retry.code in {"hub_unauthorized", "hub_forbidden"}:
                raise LibraryError("hub_token_rejected", TOKEN_REJECTED_ACCESS, 401) from None
            raise


def _cancelled(job_id: str) -> bool:
    with _JOBS_LOCK:
        return bool(_JOBS.get(job_id, {}).get("cancel_requested"))


def _progress(job_id: str, got: int, total: int, message: str) -> None:
    percent = 100 if total <= 0 else min(99, int(got * 100 / total))
    with _JOBS_LOCK:
        job = _JOBS[job_id]
        if job["status"] != "running":
            return
        job["bytes_got"] = got
        job["bytes_total"] = total
        job["percent"] = percent
        job["message"] = message


def _validate_selection(filenames: list[str]) -> list[str]:
    if not filenames:
        raise LibraryError("missing_file", "Select at least one file. Nothing was downloaded.", 400)
    chosen: list[str] = []
    for name in filenames:
        rel = safe_relative(name)
        text = rel.as_posix()
        if text not in chosen:
            chosen.append(text)
    return chosen


def _destination(library: Path, repo: str) -> Path:
    dest = (library / repo.replace("/", "__")).resolve()
    library_resolved = library.resolve()
    if library_resolved != dest and library_resolved not in dest.parents:
        raise LibraryError("path_rejected", "The download destination is outside the Inspector library.", 400)
    return dest


def _write_manifest(dest_root: Path, repo: str, revision: str, files: list[dict], plan: dict | None = None) -> dict:
    dest_root.mkdir(parents=True, exist_ok=True)
    path = dest_root / "manifest.json"
    existing: dict = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except (OSError, json.JSONDecodeError):
            existing = {}
    by_name = {row["name"]: row for row in existing.get("files") or [] if isinstance(row, dict) and row.get("name")}
    for row in files:
        stored = dict(row)
        stored["revision"] = revision
        by_name[row["name"]] = stored
    required_shards = list(existing.get("required_shards") or [])
    required_metadata = list(existing.get("required_metadata") or [])
    if plan and plan.get("identifiable"):
        required_shards = list(plan.get("shards") or [])
        required_metadata = list(plan.get("metadata") or [])
    assessment = _assess_checkpoint(required_shards, required_metadata, set(by_name))
    manifest = {
        "repo": repo,
        "revision": revision,
        "card_url": f"https://huggingface.co/{repo}",
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": [by_name[name] for name in sorted(by_name)],
        "required_shards": required_shards,
        "required_metadata": required_metadata,
        **assessment,
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _public_manifest(manifest: dict | None, directory: Path | None = None) -> dict | None:
    if not manifest:
        return None
    files = []
    for row in manifest.get("files") or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "")
        path = row.get("path")
        size = row.get("size_bytes")
        local = _library_file(directory, name) if directory is not None and name else None
        if local is not None:
            path = str(local)
            size = local.stat().st_size
        elif directory is not None and name and not path:
            file_path = directory / safe_relative(name)
            path = str(file_path.resolve())
            if file_path.is_file() and not file_path.is_symlink():
                size = file_path.stat().st_size
        shown = present_path(path if isinstance(path, str) else None)
        files.append(
            {
                "name": name,
                "size_bytes": size if isinstance(size, int) and not isinstance(size, bool) else None,
                "size_human": format_bytes(size) if isinstance(size, int) and not isinstance(size, bool) else None,
                "path": path,
                "path_label": shown["path_label"],
                "path_display": shown["path_display"],
                "path_scope": shown["path_scope"],
                "inspectable": name.lower().endswith(".safetensors"),
                "integrity": row.get("integrity") if row.get("integrity") in {"verified", "mismatch", "unavailable"} else "unavailable",
                "expected_sha256": row.get("expected_sha256") if isinstance(row.get("expected_sha256"), str) else None,
                "observed_sha256": row.get("observed_sha256") if isinstance(row.get("observed_sha256"), str) else None,
                "inspection": row.get("inspection") if row.get("inspection") in {"inspected", "unreadable", "not_inspected"} else "not_inspected",
            }
        )
    return {
        "title": manifest.get("title") or manifest.get("repo"),
        "repo": manifest.get("repo"),
        "revision": manifest.get("revision"),
        "card_url": manifest.get("card_url"),
        "downloaded_at": manifest.get("downloaded_at"),
        "bundled": bool(manifest.get("bundled")),
        "provider": "huggingface" if str(manifest.get("repo") or "").count("/") == 1 and not str(manifest.get("repo")).startswith("book-three/") else "local",
        "access": "local_weights",
        "download_status": "packaged" if manifest.get("bundled") else "downloaded",
        "capabilities": inspection_capabilities("local_weights"),
        "files": files,
        **_displayed_checkpoint(manifest, [row["name"] for row in files]),
        **_evidence(manifest, files),
        **_storage(directory, manifest),
    }


def _displayed_checkpoint(manifest: dict, names: list[str]) -> dict:
    """Describe what is on disk. A numbered shard is not called a complete model."""
    repo = str(manifest.get("repo") or "")
    if manifest.get("bundled") or repo.startswith("book-three/"):
        return {
            "checkpoint_status": manifest.get("checkpoint_status"),
            "checkpoint_label": manifest.get("checkpoint_label"),
            "weight_shards_present": manifest.get("weight_shards_present"),
            "weight_shards_required": manifest.get("weight_shards_required"),
        }
    required = [str(name) for name in manifest.get("required_shards") or [] if isinstance(name, str)]
    if required:
        metadata = [str(name) for name in manifest.get("required_metadata") or [] if isinstance(name, str)]
        return _assess_checkpoint(required, metadata, set(names))
    weights = [name for name in names if name.lower().endswith(".safetensors")]
    totals = {int(match.group(2)) for name in weights if (match := _SHARD_RE.search(Path(name).name))}
    if len(totals) == 1:
        required_count = totals.pop()
        present = len(weights)
        complete = present == required_count and "config.json" in names
        return {
            "checkpoint_status": "complete" if complete else "partial",
            "checkpoint_label": (
                "Complete checkpoint"
                if complete
                else f"Partial checkpoint — {present} of {required_count} shards downloaded."
            ),
            "weight_shards_present": present,
            "weight_shards_required": required_count,
        }
    if len(weights) == 1:
        return {
            "checkpoint_status": "complete",
            "checkpoint_label": "Complete checkpoint",
            "weight_shards_present": 1,
            "weight_shards_required": 1,
        }
    if len(weights) > 1:
        return {
            "checkpoint_status": "partial",
            "checkpoint_label": "Partial checkpoint — a complete set of shards is not on this computer.",
            "weight_shards_present": len(weights),
            "weight_shards_required": None,
        }
    return {
        "checkpoint_status": manifest.get("checkpoint_status"),
        "checkpoint_label": manifest.get("checkpoint_label"),
        "weight_shards_present": manifest.get("weight_shards_present"),
        "weight_shards_required": manifest.get("weight_shards_required"),
    }


def _evidence(manifest: dict, files: list[dict]) -> dict:
    """Checksum agreement and a successful parse are not the same claim as a complete file set."""
    repo = str(manifest.get("repo") or "")
    if manifest.get("bundled") or repo.startswith("book-three/"):
        return {
            "integrity_status": None,
            "integrity_label": None,
            "inspection_status": None,
            "inspection_label": None,
        }
    states = [row.get("integrity") for row in files]
    if any(state == "mismatch" for state in states):
        integrity_status = "mismatch"
        integrity_label = "Integrity check failed — a file does not match its published checksum."
    elif any(state == "verified" for state in states):
        integrity_status = "verified"
        if any(state != "verified" for state in states):
            integrity_label = "Integrity verified — published checksums match. Files without a published checksum were not hashed."
        else:
            integrity_label = "Integrity verified — published checksums match."
    else:
        integrity_status = "unavailable"
        integrity_label = "Integrity not verified — no published checksum was available."
    weights = [row for row in files if str(row.get("name") or "").lower().endswith(".safetensors")]
    inspections = [row.get("inspection") for row in weights]
    if any(state == "inspected" for state in inspections):
        inspection_status = "inspected"
        inspection_label = "Inspectable — the checkpoint was parsed and its tensors can be examined."
    elif any(state == "unreadable" for state in inspections):
        inspection_status = "unreadable"
        inspection_label = "Not inspectable — the file could not be parsed."
    else:
        inspection_status = "not_inspected"
        inspection_label = "Not yet inspected."
    return {
        "integrity_status": integrity_status,
        "integrity_label": integrity_label,
        "inspection_status": inspection_status,
        "inspection_label": inspection_label,
    }


def note_library_inspection(path: Path, *, tensor_count: int | None, readable: bool) -> None:
    """Record a parse result on a library manifest. Measurement results are left unchanged."""
    try:
        resolved = path.resolve()
        library = LIBRARY_DIR.resolve()
    except OSError:
        return
    if library != resolved and library not in resolved.parents:
        return
    folder = resolved.parent
    manifest_path: Path | None = None
    while True:
        candidate = folder / "manifest.json"
        if candidate.is_file() and not candidate.is_symlink():
            manifest_path = candidate
            break
        if folder == library or library not in folder.parents:
            return
        folder = folder.parent
    if manifest_path is None:
        return
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict):
        return
    changed = False
    for row in payload.get("files") or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "")
        if not name or Path(name).name != resolved.name:
            continue
        local = _library_file(manifest_path.parent, name)
        if local is None or local.resolve() != resolved:
            continue
        row["inspection"] = "inspected" if readable else "unreadable"
        if readable and isinstance(tensor_count, int):
            row["tensor_count"] = tensor_count
        changed = True
    if changed:
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _storage(directory: Path | None, manifest: dict) -> dict:
    if directory is None or manifest.get("bundled") or str(manifest.get("repo") or "").startswith("book-three/"):
        return {"storage_bytes": None, "storage_label": None}
    try:
        total = _stored_bytes(directory)
    except OSError:
        return {"storage_bytes": None, "storage_label": None}
    return {"storage_bytes": total, "storage_label": format_bytes(total)}


def _stored_bytes(dest: Path) -> int:
    if dest.is_symlink() or not dest.is_dir():
        return 0
    total = 0
    root = dest.resolve()
    for child in dest.rglob("*"):
        if child.is_symlink() or not child.is_file():
            continue
        resolved = child.resolve()
        if root != resolved and root not in resolved.parents:
            continue
        total += child.stat().st_size
    return total


def _library_file(directory: Path, name: str) -> Path | None:
    try:
        rel = safe_relative(name)
    except LibraryError:
        return None
    return _file_inside(directory, rel)


def _file_inside(directory: Path, rel: Path) -> Path | None:
    if directory.is_symlink() or not directory.is_dir():
        return None
    current = directory
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            return None
    if not current.is_file():
        return None
    root = directory.resolve()
    resolved = current.resolve()
    if root != resolved and root not in resolved.parents:
        return None
    return resolved


def _repo_download_active(repo: str) -> bool:
    with _JOBS_LOCK:
        return any(job.get("repo") == repo and job.get("status") == "running" for job in _JOBS.values())


def delete_downloaded(repo_text: str, *, filename: str | None = None, scope: str = "file", library: Path | None = None) -> dict:
    """Remove a library file or one downloaded model. Remote repositories are not changed."""
    if scope not in {"file", "model"}:
        raise LibraryError("rejected", "Choose a file or the whole model. Nothing was deleted.", 400)
    repo, _revision = parse_repo(repo_text)
    if repo.startswith("book-three/"):
        raise LibraryError("protected", "Books research files stay in place. Nothing was deleted.", 400)
    root = (library or LIBRARY_DIR).resolve()
    folder = root / repo.replace("/", "__")
    if folder.is_symlink():
        raise LibraryError("path_rejected", "The model directory is a symbolic link. Nothing was deleted.", 400)
    dest = _destination(root, repo)
    if not dest.is_dir() or dest.is_symlink():
        raise LibraryError("missing_file", "That model is not in the Inspector library. Nothing was deleted.", 404)
    if _repo_download_active(repo):
        raise LibraryError(
            "download_active",
            "A download for this model is still running. Cancel it before deleting. Nothing was deleted.",
            409,
        )
    if scope == "model":
        return _delete_model_directory(dest, repo)
    if not filename:
        raise LibraryError("missing_file", "Choose a file. Nothing was deleted.", 400)
    return _delete_one_file(dest, repo, filename)


def _delete_one_file(dest: Path, repo: str, filename: str) -> dict:
    rel = safe_relative(filename)
    target = _file_inside(dest, rel)
    if target is None:
        raise LibraryError("path_rejected", "That file is not in the Inspector library. Nothing was deleted.", 400)
    if inspection_busy(target):
        raise LibraryError(
            "inspection_active",
            "This file is being inspected. Wait until that finishes. Nothing was deleted.",
            409,
        )
    freed = target.stat().st_size
    target.unlink()
    _drop_manifest_name(dest, repo, rel.as_posix())
    return {
        "ok": True,
        "repo": repo,
        "scope": "file",
        "removed": [rel.as_posix()],
        "freed_bytes": freed,
        "freed_label": format_bytes(freed),
    }


def _delete_model_directory(dest: Path, repo: str) -> dict:
    files = _collect_library_files(dest)
    for path in files:
        if inspection_busy(path):
            raise LibraryError(
                "inspection_active",
                f"{path.name} is being inspected. Wait until that finishes. Nothing was deleted.",
                409,
            )
    freed = 0
    removed: list[str] = []
    for path in sorted(files, key=lambda item: len(item.parts), reverse=True):
        freed += path.stat().st_size
        removed.append(path.relative_to(dest).as_posix())
        path.unlink()
    _remove_empty_dirs(dest)
    return {
        "ok": True,
        "repo": repo,
        "scope": "model",
        "removed": removed,
        "freed_bytes": freed,
        "freed_label": format_bytes(freed),
    }


def _collect_library_files(dest: Path) -> list[Path]:
    files: list[Path] = []
    root = dest.resolve()
    stack = [dest]
    while stack:
        current = stack.pop()
        if current.is_symlink():
            raise LibraryError("path_rejected", "Symbolic links are not deleted. Nothing was deleted.", 400)
        for child in current.iterdir():
            if child.is_symlink():
                raise LibraryError("path_rejected", "Symbolic links are not deleted. Nothing was deleted.", 400)
            resolved = child.resolve()
            if root != resolved and root not in resolved.parents:
                raise LibraryError("path_rejected", "A file is outside the Inspector library. Nothing was deleted.", 400)
            if child.is_dir():
                stack.append(child)
            elif child.is_file():
                files.append(child)
            else:
                raise LibraryError("path_rejected", "This model contains an unexpected entry. Nothing was deleted.", 400)
    return files


def _remove_empty_dirs(dest: Path) -> None:
    directories = [path for path in dest.rglob("*") if path.is_dir() and not path.is_symlink()]
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        if not any(path.iterdir()):
            path.rmdir()
    if dest.is_dir() and not dest.is_symlink() and not any(dest.iterdir()):
        dest.rmdir()


def _drop_manifest_name(dest: Path, repo: str, filename: str) -> None:
    path = dest / "manifest.json"
    if not path.is_file() or path.is_symlink():
        _remove_empty_dirs(dest)
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    files = [row for row in payload.get("files") or [] if isinstance(row, dict) and row.get("name") != filename]
    if not files:
        path.unlink(missing_ok=True)
        _remove_empty_dirs(dest)
        return
    payload["files"] = files
    payload["repo"] = payload.get("repo") or repo
    required = [str(name) for name in payload.get("required_shards") or [] if isinstance(name, str)]
    metadata = [str(name) for name in payload.get("required_metadata") or [] if isinstance(name, str)]
    names = [str(row.get("name")) for row in files if row.get("name")]
    if required:
        payload.update(_assess_checkpoint(required, metadata, set(names)))
    else:
        payload.update(_displayed_checkpoint(payload, names))
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _public_job(job_id: str) -> dict:
    with _JOBS_LOCK:
        job = dict(_JOBS[job_id])
    job.pop("cancel_requested", None)
    job.pop("checkpoint_plan", None)
    return job


def _description(readme: str | None) -> str | None:
    if not readme:
        return None
    text = readme
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4 :]
    paragraph: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("<!--"):
            continue
        if not stripped:
            if paragraph:
                break
            continue
        paragraph.append(stripped)
        if sum(len(part) for part in paragraph) > 500:
            break
    excerpt = " ".join(paragraph).strip()
    if not excerpt:
        return None
    return excerpt[:500]


def _hide(text: str | None, token: str) -> str | None:
    if text is None:
        return None
    if token and token in text:
        return None
    return text


def _hide_names(names: list[str] | None, token: str) -> list[str] | None:
    if not names:
        return None
    hidden = [name for name in names if not token or token not in name]
    return hidden or None


def _text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _names(value: object) -> list[str] | None:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        names = [str(item) for item in value if isinstance(item, str) and item.strip()]
        return names or None
    return None


def _first_name(value: object) -> str | None:
    names = _names(value)
    return names[0] if names else None


def _is_gated(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def _publisher_token(query: str) -> str:
    if "/" in query or "\\" in query:
        return ""
    token = query.split()[0]
    if not _PUBLISHER_RE.fullmatch(token):
        return ""
    return token


def _owner_name(row: dict) -> str:
    repo = str(row.get("id") or row.get("modelId") or "")
    return repo.split("/", 1)[0] if "/" in repo else ""


def _official_rows(fetch, token: str, rows: list, publisher: str) -> list:
    spelling = ""
    for row in rows:
        if isinstance(row, dict) and _owner_name(row).casefold() == publisher.casefold():
            spelling = _owner_name(row)
            break
    candidates = [spelling or publisher]
    titled = publisher[:1].upper() + publisher[1:]
    if titled not in candidates:
        candidates.append(titled)
    for author in candidates:
        try:
            found, _ignored = fetch(token, "", author)
        except LibraryError as exc:
            if exc.code in {"hub_unauthorized", "hub_forbidden"}:
                raise
            continue
        if found:
            return found
    return []


def _task_group(pipeline: str | None) -> str:
    if pipeline in _TEXT_TASKS:
        return "text"
    if pipeline in _IMAGE_TASKS:
        return "image"
    return "other"


def _task_label(pipeline: str | None) -> str | None:
    if not pipeline:
        return None
    known = _TASK_LABELS.get(pipeline)
    if known:
        return known
    label = ""
    for piece in pipeline.split("-"):
        if not piece:
            continue
        word = "to" if piece.lower() == "to" else piece[:1].upper() + piece[1:]
        if word == "to" and label:
            label += "-to-"
        elif label.endswith("-to-"):
            label += word
        elif label:
            label += " " + word
        else:
            label += word
    return label or None


def _architecture_label(model_type: str | None, architectures: list[str] | None) -> str | None:
    if model_type:
        folded = model_type.lower().replace("_", "-")
        if folded not in _TASK_LABELS and folded not in _TEXT_TASKS and folded not in _IMAGE_TASKS:
            return model_type[:1].upper() + model_type[1:]
    if architectures:
        return ", ".join(architectures)
    return None


def _other_count(rows: list) -> int:
    count = 0
    for row in rows:
        if isinstance(row, dict) and _task_group(_text(row.get("pipeline_tag"))) == "other":
            count += 1
    return count


def _merge_hits(official_rows: list, rows: list, secret: str, publisher: str, task: str) -> list[dict]:
    seen: set[str] = set()
    results = []
    for row in list(official_rows) + list(rows):
        if not isinstance(row, dict):
            continue
        viewed = _search_hit(row, secret)
        if not viewed or viewed["repo"] in seen:
            continue
        if task != "all" and _task_group(viewed.get("pipeline")) != task:
            continue
        owner = viewed["repo"].split("/", 1)[0]
        viewed["official"] = bool(publisher) and owner.casefold() == publisher.casefold()
        seen.add(viewed["repo"])
        results.append(viewed)
    return results


def _search_cursor(cursor: str) -> str:
    text = (cursor or "").strip()
    if not text:
        return ""
    if len(text) > 2000 or any(char in text for char in ("\n", "\r", "\x00", " ", "&")):
        raise LibraryError("rejected", "That search page is not valid.", 400)
    return text


def _search_hit(row: dict, secret: str) -> dict | None:
    repo = str(row.get("id") or row.get("modelId") or "").strip()
    if not _REPO_RE.match(repo) or (secret and secret in repo):
        return None
    config = row.get("config") if isinstance(row.get("config"), dict) else {}
    model_type = _text(config.get("model_type"))
    architectures = _names(config.get("architectures"))
    pipeline = _text(row.get("pipeline_tag"))
    library = _text(row.get("library_name"))
    parameters = _whole_number((row.get("safetensors") or {}).get("total") if isinstance(row.get("safetensors"), dict) else None)
    downloads = _whole_number(row.get("downloads"))
    private = row.get("private")
    gated = row.get("gated")
    if private is True:
        access = "private"
    elif _is_gated(gated):
        access = "gated"
    elif private is False and gated is False:
        access = "public"
    else:
        access = None
    return {
        "repo": repo,
        "pipeline": _hide(pipeline, secret),
        "task_label": _hide(_task_label(pipeline), secret),
        "library": _hide(library, secret),
        "model_type": _hide(model_type, secret),
        "architectures": _hide_names(architectures, secret),
        "architecture": _hide(_architecture_label(model_type, architectures), secret),
        "official": False,
        "parameters": parameters,
        "parameters_label": format_count(parameters) if parameters is not None else None,
        "downloads": downloads,
        "downloads_label": f"{downloads:,}" if downloads is not None else None,
        "license": _hide(_license_name(row), secret),
        "private": private is True,
        "gated": _is_gated(gated),
        "access": access,
    }


def _license_name(row: dict) -> str | None:
    card = row.get("cardData") if isinstance(row.get("cardData"), dict) else {}
    named = _text(card.get("license"))
    if named:
        return named
    for tag in row.get("tags") or []:
        if isinstance(tag, str) and tag.startswith("license:"):
            return _text(tag.split(":", 1)[1])
    return None


def _whole_number(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _expected_sha256(item: dict) -> str | None:
    """Use a published LFS SHA-256. A git object id is not a checksum of the file bytes."""
    lfs = item.get("lfs") if isinstance(item.get("lfs"), dict) else {}
    oid = lfs.get("oid")
    if isinstance(oid, str):
        text = oid.strip().lower()
        if text.startswith("sha256:"):
            text = text.split(":", 1)[1]
        if re.fullmatch(r"[0-9a-f]{64}", text):
            return text
    direct = item.get("expected_sha256")
    if isinstance(direct, str) and re.fullmatch(r"[0-9a-f]{64}", direct.strip().lower()):
        return direct.strip().lower()
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_integrity(path: Path, expected: str | None) -> dict:
    """Size is recorded elsewhere. A checksum is a separate claim, and only when one was published."""
    if not expected:
        return {"expected_sha256": None, "observed_sha256": None, "integrity": "unavailable"}
    observed = _sha256_file(path)
    return {
        "expected_sha256": expected,
        "observed_sha256": observed,
        "integrity": "verified" if observed == expected else "mismatch",
    }


def format_count(count: int) -> str:
    if count >= 1_000_000_000:
        scaled, unit = count / 1_000_000_000, "B"
    elif count >= 1_000_000:
        scaled, unit = count / 1_000_000, "M"
    elif count >= 1_000:
        scaled, unit = count / 1_000, "K"
    else:
        return str(count)
    return f"{scaled:.1f}".rstrip("0").rstrip(".") + unit


class UrllibHub:
    def model_info(self, repo: str, token: str) -> dict:
        data = _get_json(f"https://huggingface.co/api/models/{quote(repo, safe='/')}", token)
        if not isinstance(data, dict):
            raise LibraryError("hub_error", "Hugging Face did not return a model card.", 502)
        return data

    def file_list(self, repo: str, revision: str, token: str) -> list[dict]:
        quoted = quote(repo, safe="/")
        rows: list[dict] = []
        cursor = ""
        for _page in range(20):
            url = f"https://huggingface.co/api/models/{quoted}/tree/{quote(revision, safe='')}?recursive=1"
            if cursor:
                url += "&cursor=" + quote(cursor, safe="")
            payload, next_cursor = _get_json_page(url, token)
            items = payload if isinstance(payload, list) else []
            for item in items:
                if not isinstance(item, dict) or item.get("type") not in {None, "file"}:
                    continue
                name = str(item.get("path") or item.get("rfilename") or "")
                if not name or item.get("type") == "directory":
                    continue
                size = item.get("size")
                rows.append(
                    {
                        "name": name,
                        "size_bytes": int(size) if isinstance(size, int) and not isinstance(size, bool) else None,
                        "expected_sha256": _expected_sha256(item),
                    }
                )
            if not next_cursor:
                break
            cursor = next_cursor
        return rows

    def search_models(
        self,
        query: str,
        token: str,
        cursor: str,
        *,
        sort: str = "relevance",
        pipeline: str = "",
        author: str = "",
    ) -> tuple[list, str]:
        # expand=safetensors and expand=gated each omit the rest of the catalog record, so they are merged by id.
        if author and not query:
            base = f"https://huggingface.co/api/models?author={quote(author)}&limit={SEARCH_LIMIT}"
        else:
            base = "https://huggingface.co/api/models?search=" + quote(query) + f"&limit={SEARCH_LIMIT}"
            if author:
                base += "&author=" + quote(author)
        if pipeline:
            base += "&pipeline_tag=" + quote(pipeline)
        if sort == "downloads":
            base += "&sort=downloads&direction=-1"
        elif sort == "updated":
            base += "&sort=lastModified&direction=-1"
        if cursor:
            base += "&cursor=" + quote(cursor, safe="")
        payload, next_cursor = _get_json_page(base + "&config=true", token)
        if not isinstance(payload, list):
            raise LibraryError("hub_error", "Hugging Face did not return a model list.", 502)
        totals: dict[str, dict] = {}
        try:
            expanded, _ignored = _get_json_page(base + "&expand=safetensors", token)
        except LibraryError:
            expanded = []
        if isinstance(expanded, list):
            for row in expanded:
                if isinstance(row, dict) and isinstance(row.get("safetensors"), dict):
                    totals[str(row.get("id") or "")] = row["safetensors"]
        gated: dict[str, object] = {}
        try:
            access_rows, _ignored = _get_json_page(base + "&expand=gated", token)
        except LibraryError:
            access_rows = []
        if isinstance(access_rows, list):
            for row in access_rows:
                if isinstance(row, dict) and "gated" in row:
                    gated[str(row.get("id") or "")] = row.get("gated")
        for row in payload:
            if isinstance(row, dict):
                repo = str(row.get("id") or "")
                found = totals.get(repo)
                if found:
                    row["safetensors"] = found
                if repo in gated:
                    row["gated"] = gated[repo]
        return payload, unquote(next_cursor) if next_cursor else ""

    def account_authors(self, token: str) -> list[str]:
        """Names of the user and organizations this token can see. Email is not returned."""
        from inspector_engine.credentials import WHOAMI_URL

        if not token:
            raise LibraryError("hub_unauthorized", PRIVATE_TOKEN_REQUIRED, 401)
        data = _get_json(WHOAMI_URL, token)
        if not isinstance(data, dict):
            raise LibraryError("hub_error", "Hugging Face did not identify this token.", 502)
        authors: list[str] = []
        name = data.get("name")
        if isinstance(name, str) and name.strip():
            authors.append(name.strip())
        for org in data.get("orgs") or []:
            if not isinstance(org, dict):
                continue
            org_name = org.get("name")
            if isinstance(org_name, str) and org_name.strip() and org_name.strip() not in authors:
                authors.append(org_name.strip())
        if not authors:
            raise LibraryError("hub_error", "Hugging Face did not name an account for this token.", 502)
        return authors

    def readme(self, repo: str, revision: str, token: str) -> str | None:
        url = (
            f"https://huggingface.co/{quote(repo, safe='/')}/resolve/{quote(revision, safe='')}/README.md"
        )
        try:
            return _get_text(url, token, README_LIMIT)
        except LibraryError as exc:
            if exc.status in {401, 403, 404}:
                return None
            return None

    def open_file(self, repo: str, revision: str, filename: str, token: str) -> tuple[int, Iterable[bytes]]:
        safe_relative(filename)
        url = (
            f"https://huggingface.co/{quote(repo, safe='/')}/resolve/"
            f"{quote(revision, safe='')}/{quote(filename, safe='/')}"
        )
        return _open_stream(url, token)


class _StayOnHub(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _allowed_redirect(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open(request: Request):
    return build_opener(_StayOnHub).open(request, timeout=30)


def _headers(token: str) -> dict[str, str]:
    headers = {"User-Agent": "ai-model-inspector", "Accept": "*/*"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _allowed_redirect(url: str) -> None:
    host = (urlparse(url).hostname or "").lower()
    if host in ALLOWED_HOSTS or host.endswith(".huggingface.co") or host.endswith(".hf.co"):
        return
    raise LibraryError("hub_error", "Hugging Face redirected the download away from its own network.", 502)


def _get_json(url: str, token: str) -> dict | list:
    payload, _cursor = _get_json_page(url, token)
    return payload


def _get_json_page(url: str, token: str) -> tuple[dict | list, str]:
    request = Request(url, headers={**_headers(token), "Accept": "application/json"}, method="GET")
    try:
        with _open(request) as response:
            _allowed_redirect(response.geturl())
            raw = response.read().decode("utf-8")
            cursor = response.headers.get("X-Next-Cursor") or response.headers.get("Link") or ""
            if 'rel="next"' in cursor and "cursor=" in cursor:
                match = re.search(r"cursor=([^&>]+)", cursor)
                cursor = match.group(1) if match else ""
            elif cursor and "http" in cursor:
                cursor = ""
    except HTTPError as exc:
        _raise_http(exc)
    except URLError as exc:
        raise LibraryError("hub_unreachable", "The Hugging Face Hub did not respond.", 502) from exc
    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise LibraryError("hub_error", "Hugging Face returned a response that was not JSON.", 502) from exc
    return data, cursor


def _get_text(url: str, token: str, limit: int) -> str:
    request = Request(url, headers=_headers(token), method="GET")
    try:
        with _open(request) as response:
            _allowed_redirect(response.geturl())
            raw = response.read(limit + 1)
    except HTTPError as exc:
        _raise_http(exc)
    except URLError as exc:
        raise LibraryError("hub_unreachable", "The Hugging Face Hub did not respond.", 502) from exc
    return raw[:limit].decode("utf-8", errors="replace")


def _open_stream(url: str, token: str) -> tuple[int, Iterable[bytes]]:
    request = Request(url, headers=_headers(token), method="GET")
    try:
        response = build_opener(_StayOnHub).open(request, timeout=60)
    except HTTPError as exc:
        _raise_http(exc)
    except URLError as exc:
        raise LibraryError("hub_unreachable", "The Hugging Face Hub did not respond.", 502) from exc
    try:
        _allowed_redirect(response.geturl())
    except LibraryError:
        response.close()
        raise
    size = int(response.headers.get("Content-Length") or 0)

    def chunks() -> Iterable[bytes]:
        try:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                yield block
        finally:
            response.close()

    return size, chunks()


def _raise_http(exc: HTTPError) -> None:
    if exc.code == 401:
        raise LibraryError("hub_unauthorized", "Hugging Face refused this request.", 401) from exc
    if exc.code == 403:
        raise LibraryError("hub_forbidden", "Hugging Face refused this request.", 403) from exc
    if exc.code == 404:
        raise LibraryError("missing_file", "The repository or file was not found on Hugging Face.", 404) from exc
    raise LibraryError("hub_error", f"Hugging Face returned HTTP {exc.code}.", 502) from exc
