# Authors: LCC Research Lab
# Project: From Document to Evidence

import hashlib
import io
import json
import threading
import time
from pathlib import Path
from urllib.error import HTTPError

import pytest
from fastapi.testclient import TestClient

from inspector_engine.model_library import (
    LibraryError,
    _raise_http,
    cancel_job,
    _expected_sha256,
    delete_downloaded,
    discover,
    format_count,
    hold_inspection,
    job_status,
    list_local_models,
    note_library_inspection,
    parse_repo,
    search_models,
    search_query,
    start_download,
)
import inspector_engine.model_library as model_library
from server import app

SECRET = "hf_secret_SHOULD_NOT_LEAK_9911"


class FakeHub:
    def __init__(self, *, files=None, readme="A small language model for tests.", slow=False, missing=False):
        self.opened = False
        self.files = files or [
            {"name": "config.json", "size_bytes": 12},
            {"name": "model.safetensors", "size_bytes": 4},
            {"name": "README.md", "size_bytes": 20},
        ]
        self.readme_text = readme
        self.slow = slow
        self.missing = missing
        self.started = threading.Event()
        self.token_seen = ""

    def model_info(self, repo: str, token: str) -> dict:
        self.token_seen = token
        if repo == "gated/model" and not token:
            raise LibraryError(
                "hub_unauthorized",
                "Hugging Face refused access to this repository. A token on this Mac may be required, "
                "or the repository license may still need to be accepted on the Hub.",
                401,
            )
        return {
            "private": False,
            "gated": False,
            "pipeline_tag": "text-generation",
            "library_name": "transformers",
            "license": SECRET if token == SECRET else "apache-2.0",
            "cardData": {
                "license": SECRET if token == SECRET else "apache-2.0",
                "base_model": "org/base",
            },
            "config": {"model_type": "toy", "architectures": ["ToyForCausalLM"]},
            "tags": ["text-generation"],
        }

    def file_list(self, repo: str, revision: str, token: str) -> list[dict]:
        return list(self.files)

    def readme(self, repo: str, revision: str, token: str) -> str | None:
        return self.readme_text

    def open_file(self, repo: str, revision: str, filename: str, token: str):
        self.opened = True
        self.token_seen = token
        if self.missing or filename == "absent.safetensors":
            raise LibraryError("missing_file", "The repository or file was not found on Hugging Face.", 404)
        payload = b"data" if filename.endswith(".safetensors") else b'{"ok":true}\n'
        if self.slow:
            def chunks():
                self.started.set()
                for _index in range(40):
                    time.sleep(0.05)
                    yield b"x" * 1000
            return 40_000, chunks()
        return len(payload), iter([payload])


class SearchHub:
    def __init__(self) -> None:
        self.tokens: list[str] = []
        self.calls: list[dict] = []

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
        self.tokens.append(token)
        self.calls.append({"query": query, "cursor": cursor, "sort": sort, "pipeline": pipeline, "author": author})
        if token == "rejected-token":
            raise LibraryError("hub_unauthorized", "refused", 401)
        if author:
            return [self._official()], ""
        if cursor == "page-2":
            return [{"id": "org/more", "downloads": 3, "private": False, "gated": False, "pipeline_tag": "text-generation"}], ""
        return [
            self._community(),
            self._official(),
            self._image(),
            {"id": "org/sparse", "private": False},
            self._gated(),
        ], "page-2"

    @staticmethod
    def _official() -> dict:
        return {
            "id": "Qwen/Qwen2.5-0.5B-Instruct",
            "downloads": 12,
            "private": False,
            "gated": False,
            "pipeline_tag": "text-generation",
            "library_name": "transformers",
            "tags": ["license:apache-2.0"],
            "config": {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]},
            "safetensors": {"total": 494_032_768},
        }

    @staticmethod
    def _community() -> dict:
        return {
            "id": "community/a-very-long-qwen-community-finetune-name",
            "downloads": 99,
            "private": False,
            "gated": False,
            "pipeline_tag": "text-generation",
        }

    @staticmethod
    def _image() -> dict:
        return {
            "id": "community/qwen-image",
            "downloads": 4,
            "private": False,
            "gated": False,
            "pipeline_tag": "text-to-image",
            "config": {"model_type": "text-to-image"},
        }

    @staticmethod
    def _gated() -> dict:
        return {
            "id": "org/gated-qwen",
            "downloads": 1,
            "private": False,
            "gated": "manual",
            "pipeline_tag": "image-text-to-text",
            "config": {"architectures": ["Qwen2VLForConditionalGeneration"]},
        }


def test_search_uses_provided_metadata_and_does_not_download(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("inspector_engine.model_library.read_token", lambda: "")
    monkeypatch.setattr("inspector_engine.model_library.hub_token", lambda: "")
    hub = SearchHub()
    report = search_models("Qwen", hub=hub)
    assert hub.calls[0]["author"] == ""
    assert hub.calls[1]["author"] == "Qwen"
    assert hub.tokens == ["", ""]
    assert report["query"] == "Qwen"
    assert report["sort"] == "relevance"
    assert [row["repo"] for row in report["results"]] == [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "community/a-very-long-qwen-community-finetune-name",
        "community/qwen-image",
        "org/sparse",
        "org/gated-qwen",
    ]
    first = report["results"][0]
    assert first["official"] is True
    assert report["results"][1]["official"] is False
    assert first["parameters"] == 494_032_768
    assert first["parameters_label"] == "494M"
    assert first["downloads_label"] == "12"
    assert first["license"] == "apache-2.0"
    assert first["model_type"] == "qwen2"
    assert first["task_label"] == "Text Generation"
    assert first["architecture"] == "Qwen2"
    assert first["access"] == "public"
    image = report["results"][2]
    assert image["task_label"] == "Text-to-Image"
    assert image["architecture"] is None
    sparse = report["results"][3]
    assert sparse["parameters"] is None
    assert sparse["parameters_label"] is None
    assert sparse["license"] is None
    assert sparse["downloads"] is None
    assert sparse["model_type"] is None
    assert sparse["task_label"] is None
    assert sparse["architecture"] is None
    assert sparse["access"] is None
    assert report["results"][4]["access"] == "gated"
    assert report["results"][4]["task_label"] == "Image-Text-to-Text"
    assert report["results"][4]["architecture"] == "Qwen2VLForConditionalGeneration"
    assert report["next_cursor"] == "page-2"
    more = search_models("Qwen", cursor="page-2", hub=hub)
    assert more["results"][0]["repo"] == "org/more"
    assert all(call["author"] == "" or call["cursor"] == "" for call in hub.calls)
    assert search_query("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/main") == "Qwen/Qwen2.5-0.5B-Instruct"
    assert format_count(1_500_000_000) == "1.5B"
    with pytest.raises(LibraryError) as caught:
        search_models("   ", hub=hub)
    assert caught.value.code == "empty_search"
    text_only = search_models("Qwen", task="text", hub=hub)
    assert [row["repo"] for row in text_only["results"]] == [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "community/a-very-long-qwen-community-finetune-name",
    ]
    assert search_models("Qwen", task="image", hub=hub)["results"][0]["repo"] == "community/qwen-image"
    assert [row["repo"] for row in search_models("Qwen", task="other", hub=hub)["results"]] == ["org/sparse", "org/gated-qwen"]
    downloads = search_models("Qwen", sort="downloads", hub=hub)
    assert any(call["sort"] == "downloads" for call in hub.calls)
    assert downloads["results"][0]["official"] is True
    with pytest.raises(LibraryError) as rejected:
        search_models("Qwen", sort="stars", hub=hub)
    assert rejected.value.code == "rejected"


class PrivateHub:
    def __init__(self, *, deny_lab: bool = False, whoami: str = "ok") -> None:
        self.deny_lab = deny_lab
        self.whoami = whoami
        self.authors: list[str] = []

    def account_authors(self, token: str) -> list[str]:
        if self.whoami == "rejected":
            raise LibraryError("hub_unauthorized", "refused", 401)
        if self.whoami == "forbidden":
            raise LibraryError("hub_forbidden", "refused", 403)
        if token != "good-token":
            raise LibraryError("hub_unauthorized", "refused", 401)
        return ["researcher", "lab"]

    def search_models(self, query: str, token: str, cursor: str, *, sort: str = "relevance", pipeline: str = "", author: str = "") -> tuple[list, str]:
        self.authors.append(author)
        if author == "lab" and self.deny_lab:
            raise LibraryError("hub_forbidden", "no", 403)
        if author == "researcher":
            return [
                {"id": "researcher/secret-lora", "private": True, "gated": False, "pipeline_tag": "text-generation", "downloads": 1, "tags": ["license:apache-2.0"]},
                {"id": "researcher/public-base", "private": False, "gated": False, "downloads": 9, "pipeline_tag": "text-generation"},
                {"id": "researcher/other-adapter", "private": True, "gated": False, "pipeline_tag": "text-to-image", "downloads": 4},
            ], ""
        if author == "lab":
            return [{"id": "lab/shared-private", "private": True, "gated": False, "downloads": 3, "pipeline_tag": "text-generation", "lastModified": "2026-01-02"}], ""
        return [], ""


def test_public_search_omits_private_repositories(monkeypatch: pytest.MonkeyPatch) -> None:
    class Hub(SearchHub):
        def search_models(self, query, token, cursor, *, sort="relevance", pipeline="", author=""):
            rows, next_cursor = super().search_models(query, token, cursor, sort=sort, pipeline=pipeline, author=author)
            if not author and cursor != "page-2":
                rows = list(rows) + [{"id": "researcher/secret-lora", "private": True, "gated": False, "pipeline_tag": "text-generation"}]
            return rows, next_cursor

    monkeypatch.setattr("inspector_engine.model_library.read_token", lambda: "")
    monkeypatch.setattr("inspector_engine.model_library.hub_token", lambda: "")
    report = search_models("Qwen", hub=Hub())
    assert "researcher/secret-lora" not in [row["repo"] for row in report["results"]]
    assert report["scope"] == "public"
    assert report["direct_repo"] is None
    exact = search_models("researcher/secret-lora", hub=Hub())
    assert "researcher/secret-lora" not in [row["repo"] for row in exact["results"]]
    assert exact["direct_repo"] == "researcher/secret-lora"


def test_private_search_lists_only_accessible_private_repositories(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("inspector_engine.model_library.hub_token", lambda: "good-token")
    monkeypatch.setattr("inspector_engine.model_library.read_token", lambda: "good-token")
    monkeypatch.setattr(
        "inspector_engine.model_library.auth_view",
        lambda: {"status": "configured", "label": "Configured", "present": True, "usable": True, "validated": True},
    )
    marked: list[str] = []
    monkeypatch.setattr("inspector_engine.credentials.mark_huggingface_validated", lambda: marked.append("validated"))
    hub = PrivateHub()
    report = search_models("private", scope="private", hub=hub)
    assert marked == ["validated"]
    assert hub.authors == ["researcher", "lab"]
    assert [row["repo"] for row in report["results"]] == ["lab/shared-private"]
    assert all(row["private"] is True for row in report["results"])
    assert "good-token" not in json.dumps(report)
    assert "email" not in json.dumps(report)
    lora = search_models("lora", scope="private", hub=hub)
    assert [row["repo"] for row in lora["results"]] == ["researcher/secret-lora"]
    assert lora["direct_repo"] is None
    direct = search_models("https://huggingface.co/lab/not-in-the-list", scope="private", hub=hub)
    assert direct["results"] == []
    assert direct["direct_repo"] == "lab/not-in-the-list"
    denied = search_models("private", scope="private", hub=PrivateHub(deny_lab=True))
    assert denied["permission_note"]
    assert "permission limit" in denied["permission_note"]
    assert [row["repo"] for row in denied["results"]] == []


def test_private_search_distinguishes_missing_token_rejection_and_permission(monkeypatch: pytest.MonkeyPatch) -> None:
    hub = PrivateHub()

    def no_token() -> str:
        return ""

    monkeypatch.setattr("inspector_engine.model_library.hub_token", no_token)
    monkeypatch.setattr(
        "inspector_engine.model_library.auth_view",
        lambda: {"status": "none", "label": "No token", "present": False, "usable": False, "validated": False},
    )
    with pytest.raises(LibraryError) as missing:
        search_models("lora", scope="private", hub=hub)
    assert missing.value.code == "hub_unauthorized"
    assert "validated HF Token" in missing.value.message

    monkeypatch.setattr(
        "inspector_engine.model_library.auth_view",
        lambda: {"status": "invalid", "label": "Invalid", "present": True, "usable": False, "validated": False},
    )
    with pytest.raises(LibraryError) as rejected:
        search_models("lora", scope="private", hub=hub)
    assert rejected.value.code == "hub_token_rejected"
    assert "rejected" in rejected.value.message

    monkeypatch.setattr("inspector_engine.model_library.hub_token", lambda: "good-token")
    monkeypatch.setattr(
        "inspector_engine.model_library.auth_view",
        lambda: {"status": "configured", "label": "Configured", "present": True, "usable": True, "validated": True},
    )
    monkeypatch.setattr("inspector_engine.credentials.note_hub_refusal", lambda token: "valid")
    with pytest.raises(LibraryError) as forbidden:
        search_models("lora", scope="private", hub=PrivateHub(whoami="forbidden"))
    assert forbidden.value.code == "hub_access_denied"
    assert "permission limit" in forbidden.value.message
    monkeypatch.setattr("inspector_engine.credentials.note_hub_refusal", lambda token: "rejected")
    with pytest.raises(LibraryError) as invalid:
        search_models("lora", scope="private", hub=PrivateHub(whoami="rejected"))
    assert invalid.value.code == "hub_token_rejected"


def test_combined_search_adds_private_repositories_and_keeps_public_results(monkeypatch: pytest.MonkeyPatch) -> None:
    class Hub(PrivateHub):
        def search_models(self, query, token, cursor, *, sort="relevance", pipeline="", author=""):
            if not author:
                self.authors.append("")
                if cursor == "page-2":
                    return [{"id": "org/lora-next", "private": False, "gated": False, "downloads": 2, "pipeline_tag": "text-generation"}], ""
                return [
                    {"id": "org/lora-public", "private": False, "gated": False, "downloads": 8, "pipeline_tag": "text-generation"},
                    {"id": "researcher/secret-lora", "private": True, "gated": False, "pipeline_tag": "text-generation"},
                ], "page-2"
            return super().search_models(query, token, cursor, sort=sort, pipeline=pipeline, author=author)

    monkeypatch.setattr("inspector_engine.model_library.hub_token", lambda: "")
    monkeypatch.setattr("inspector_engine.model_library.read_token", lambda: "")
    monkeypatch.setattr(
        "inspector_engine.model_library.auth_view",
        lambda: {"status": "none", "label": "No token", "present": False, "usable": False, "validated": False},
    )
    anonymous = Hub()
    hidden = search_models("lora", scope="combined", hub=anonymous)
    assert hidden["scope"] == "public"
    assert [row["repo"] for row in hidden["results"]] == ["org/lora-public"]
    assert "researcher" not in anonymous.authors
    assert "lab" not in anonymous.authors

    monkeypatch.setattr("inspector_engine.model_library.hub_token", lambda: "good-token")
    monkeypatch.setattr("inspector_engine.model_library.read_token", lambda: "good-token")
    monkeypatch.setattr(
        "inspector_engine.model_library.auth_view",
        lambda: {"status": "configured", "label": "Configured", "present": True, "usable": True, "validated": True},
    )
    monkeypatch.setattr("inspector_engine.credentials.mark_huggingface_validated", lambda: None)
    opened = Hub()
    report = search_models("lora", scope="combined", hub=opened)
    assert report["scope"] == "combined"
    assert [row["repo"] for row in report["results"]] == ["researcher/secret-lora", "org/lora-public"]
    assert report["results"][0]["private"] is True
    assert report["results"][0]["access"] == "private"
    assert report["results"][0]["repo"] not in [row["repo"] for row in report["results"][1:]]
    assert "good-token" not in json.dumps(report)
    assert report["next_cursor"].startswith("c.")
    more = search_models("lora", cursor=report["next_cursor"], scope="combined", hub=opened)
    assert [row["repo"] for row in more["results"]] == ["org/lora-next"]

    monkeypatch.setattr("inspector_engine.credentials.note_hub_refusal", lambda token: "rejected")
    failed = search_models("lora", scope="combined", hub=Hub(whoami="rejected"))
    assert [row["repo"] for row in failed["results"]] == ["org/lora-public"]
    assert "Public models are shown" in failed["permission_note"]
    assert "good-token" not in json.dumps(failed)

    monkeypatch.setattr(
        "inspector_engine.model_library.auth_view",
        lambda: {"status": "invalid", "label": "Invalid", "present": True, "usable": False, "validated": False},
    )
    skipped = Hub()
    rejected = search_models("lora", scope="combined", hub=skipped)
    assert [row["repo"] for row in rejected["results"]] == ["org/lora-public"]
    assert "rejected" in rejected["permission_note"]
    assert "researcher" not in skipped.authors


def test_search_retries_without_a_rejected_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("inspector_engine.model_library.hub_token", lambda: "rejected-token")
    monkeypatch.setattr("inspector_engine.model_library.read_token", lambda: "rejected-token")
    monkeypatch.setattr("inspector_engine.credentials.note_hub_refusal", lambda token: "rejected")
    hub = SearchHub()
    report = search_models("SmolLM", hub=hub)
    assert hub.tokens[0] == "rejected-token"
    assert hub.tokens[1:] == ["", ""]
    assert report["results"][0]["repo"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert "rejected-token" not in json.dumps(report)


def test_an_exact_repository_distinguishes_absence_authentication_and_denial(monkeypatch: pytest.MonkeyPatch) -> None:
    class LookupHub:
        def __init__(self) -> None:
            self.tokens: list[str] = []

        def model_info(self, repo: str, token: str) -> dict:
            self.tokens.append(token)
            if repo == "missing/model":
                raise LibraryError("missing_file", "The repository or file was not found on Hugging Face.", 404)
            if repo != "owner/private-model":
                raise LibraryError("missing_file", "The repository or file was not found on Hugging Face.", 404)
            if not token:
                raise LibraryError("hub_unauthorized", "Hugging Face refused this request.", 401)
            if token == "limited-token":
                raise LibraryError("hub_forbidden", "Hugging Face refused this request.", 403)
            return {
                "private": True,
                "gated": False,
                "pipeline_tag": "text-generation",
                "library_name": "transformers",
                "license": "other",
                "cardData": {},
                "config": {"model_type": "toy"},
                "tags": [],
            }

        def file_list(self, repo: str, revision: str, token: str) -> list[dict]:
            return [{"name": "config.json", "size_bytes": 2}]

        def readme(self, repo: str, revision: str, token: str) -> str | None:
            return "Private weights for the account that can open this repository."

    monkeypatch.setattr("inspector_engine.model_library.read_token", lambda: "")
    monkeypatch.setattr("inspector_engine.model_library.hub_token", lambda: "")
    hub = LookupHub()
    with pytest.raises(LibraryError) as missing:
        discover("missing/model", hub=hub)
    assert missing.value.code == "missing_file"
    assert "not found" in missing.value.message
    with pytest.raises(LibraryError) as unauthenticated:
        discover("https://huggingface.co/owner/private-model", hub=hub)
    assert unauthenticated.value.code == "hub_unauthorized"
    assert "authentication" in unauthenticated.value.message
    assert "not found" not in unauthenticated.value.message

    monkeypatch.setattr("inspector_engine.model_library.read_token", lambda: "limited-token")
    monkeypatch.setattr("inspector_engine.model_library.hub_token", lambda: "limited-token")
    monkeypatch.setattr("inspector_engine.credentials.note_hub_refusal", lambda token: "valid")
    with pytest.raises(LibraryError) as denied:
        discover("owner/private-model", hub=LookupHub())
    assert denied.value.code == "hub_access_denied"
    assert "not granted" in denied.value.message
    assert "limited-token" not in denied.value.message
    assert "not found" not in denied.value.message

    monkeypatch.setattr("inspector_engine.model_library.read_token", lambda: "authorized-token")
    monkeypatch.setattr("inspector_engine.model_library.hub_token", lambda: "authorized-token")
    opened = LookupHub()
    report = discover("owner/private-model", hub=opened)
    assert opened.tokens[0] == "authorized-token"
    assert report["repo"] == "owner/private-model"
    assert report["private"] is True
    assert "authorized-token" not in json.dumps(report)
    name_search = search_models("Qwen", hub=SearchHub())
    assert name_search["results"]
    assert name_search["direct_repo"] is None


def test_discover_reads_the_card_and_does_not_download(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("inspector_engine.model_library.read_token", lambda: "")
    hub = FakeHub()
    report = discover("https://huggingface.co/org/toy/tree/main", hub=hub)
    assert report["repo"] == "org/toy"
    assert report["revision"] == "main"
    assert report["license"] == "apache-2.0"
    assert report["model_type"] == "toy"
    assert report["architectures"] == ["ToyForCausalLM"]
    assert report["description"].startswith("A small language model")
    assert [row["name"] for row in report["files"] if row["compatible"]] == ["config.json", "model.safetensors"]
    assert report["provider"] == "huggingface"
    assert report["access"] == "remote_files"
    assert report["capabilities"]["tensor_inspection"] is False
    assert report["capabilities"]["model_information"] is True
    assert hub.opened is False
    assert SECRET not in json.dumps(report)


def test_discover_counts_only_a_reusable_local_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_library, "LIBRARY_DIR", tmp_path)
    monkeypatch.setattr(model_library, "read_token", lambda: "")
    digest = hashlib.sha256(b"data").hexdigest()
    dest = tmp_path / "org__toy"
    dest.mkdir()
    (dest / "model.safetensors").write_bytes(b"data")
    (dest / "config.json").write_bytes(b"x")
    (dest / "model.safetensors.part").write_bytes(b"data")
    (dest / "README.md").symlink_to(dest / "model.safetensors")
    (dest / "manifest.json").write_text(
        json.dumps(
            {
                "repo": "org/toy",
                "revision": "main",
                "files": [
                    {
                        "name": "model.safetensors",
                        "size_bytes": 4,
                        "revision": "main",
                        "integrity": "verified",
                        "expected_sha256": digest,
                        "observed_sha256": digest,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    other = tmp_path / "org__other"
    other.mkdir()
    (other / "model.safetensors").write_bytes(b"data")
    report = discover(
        "org/toy",
        hub=FakeHub(
            files=[
                {"name": "config.json", "size_bytes": 12},
                {"name": "model.safetensors", "size_bytes": 4, "expected_sha256": digest},
                {"name": "README.md", "size_bytes": 20},
            ]
        ),
    )
    assert report["storage"]["free_bytes"] == model_library.free_bytes(tmp_path)
    by_name = {row["name"]: row for row in report["files"]}
    assert by_name["model.safetensors"]["local_bytes"] == 4
    assert by_name["config.json"]["local_bytes"] == 0
    assert by_name["README.md"]["local_bytes"] == 0
    (dest / "manifest.json").write_text(
        json.dumps(
            {
                "repo": "org/toy",
                "revision": "other",
                "files": [
                    {
                        "name": "model.safetensors",
                        "size_bytes": 4,
                        "integrity": "mismatch",
                        "expected_sha256": digest,
                        "observed_sha256": "ab" * 32,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    rejected = discover(
        "org/toy",
        hub=FakeHub(files=[{"name": "model.safetensors", "size_bytes": 4, "expected_sha256": digest}]),
    )
    assert rejected["files"][0]["local_bytes"] == 0


def test_bundled_freshman_is_a_local_checkpoint() -> None:
    from inspector_engine.model_library import list_bundled_models

    models = list_bundled_models()
    freshman = next(model for model in models if model["repo"] == "book-three/freshman")
    qwen = next(model for model in models if model["repo"] == "book-three/qwen")
    assert [model["title"] for model in models if model["repo"].startswith("book-three/")] == ["Freshman", "Qwen"]
    assert freshman["bundled"] is True and qwen["bundled"] is True
    assert qwen["access"] == "local_weights"
    assert qwen["capabilities"]["tensor_inspection"] is True
    assert qwen["capabilities"]["behavioral_evaluation"] is False
    freshman_weights = next(row for row in freshman["files"] if row["name"] == "adapters.safetensors")
    freshman_manifest = json.loads((Path(__file__).resolve().parents[1] / "bundled" / "freshman" / "manifest.json").read_text(encoding="utf-8"))
    assert freshman_manifest["redistribution"] == "not_included"
    if Path(freshman_weights["path"]).is_file():
        assert freshman_weights["inspectable"] is True
    else:
        assert freshman_weights["size_bytes"] in {None, 0}
    qwen_weights = next(row for row in qwen["files"] if row["name"] == "adapter_model.safetensors")
    qwen_manifest = json.loads((Path(__file__).resolve().parents[1] / "bundled" / "qwen" / "manifest.json").read_text(encoding="utf-8"))
    assert qwen_manifest["redistribution"] == "withheld"
    if Path(qwen_weights["path"]).is_file():
        assert qwen_weights["inspectable"] is True
    else:
        assert qwen_weights["size_bytes"] in {None, 0}
    assert "hf_" not in json.dumps(models)


def test_malformed_repository_ids() -> None:
    for text in ("", "noslash", "org/name/extra", "https://example.com/org/name", "../etc/passwd"):
        with pytest.raises(LibraryError) as caught:
            parse_repo(text)
        assert caught.value.code == "malformed_repo"


def test_download_writes_a_manifest_without_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("inspector_engine.model_library.read_token", lambda: SECRET)
    hub = FakeHub()
    job = start_download("org/toy", ["model.safetensors", "config.json"], hub=hub, library=tmp_path, wait=True)
    assert job["status"] == "done"
    manifest_path = tmp_path / "org__toy" / "manifest.json"
    text = manifest_path.read_text(encoding="utf-8")
    assert SECRET not in text
    assert SECRET not in json.dumps(job)
    manifest = json.loads(text)
    assert manifest["repo"] == "org/toy"
    names = {row["name"] for row in manifest["files"]}
    assert names == {"config.json", "model.safetensors"}
    assert (tmp_path / "org__toy" / "model.safetensors").read_bytes() == b"data"


def test_cancel_removes_a_partial_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("inspector_engine.model_library.read_token", lambda: "")
    hub = FakeHub(files=[{"name": "model.safetensors", "size_bytes": 40_000}], slow=True)
    job = start_download("org/toy", ["model.safetensors"], hub=hub, library=tmp_path)
    assert hub.started.wait(2)
    cancelled = cancel_job(job["id"])
    deadline = time.time() + 3
    status = cancelled
    while time.time() < deadline and status["status"] == "running":
        time.sleep(0.05)
        status = job_status(job["id"])
    assert status["status"] == "cancelled"
    assert not (tmp_path / "org__toy" / "model.safetensors").exists()
    assert list((tmp_path / "org__toy").glob("*.part")) == []


def test_missing_file_is_rejected_before_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("inspector_engine.model_library.read_token", lambda: "")
    hub = FakeHub()
    with pytest.raises(LibraryError) as caught:
        start_download("org/toy", ["absent.safetensors"], hub=hub, library=tmp_path, wait=True)
    assert caught.value.code == "missing_file"
    assert hub.opened is False


def test_hub_reports_a_listed_file_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("inspector_engine.model_library.read_token", lambda: "")
    hub = FakeHub(missing=True)
    job = start_download("org/toy", ["model.safetensors"], hub=hub, library=tmp_path, wait=True)
    assert job["status"] == "error"
    assert "not found" in job["message"].lower()
    assert not (tmp_path / "org__toy" / "model.safetensors").exists()


class ShardedHub(FakeHub):
    def __init__(self) -> None:
        self.index = {
            "metadata": {"total_size": 8},
            "weight_map": {
                "blk.0.weight": "model-00001-of-00002.safetensors",
                "blk.1.weight": "model-00002-of-00002.safetensors",
            },
        }
        self.index_bytes = json.dumps(self.index).encode()
        super().__init__(
            files=[
                {"name": "config.json", "size_bytes": 12},
                {"name": "model.safetensors.index.json", "size_bytes": len(self.index_bytes)},
                {"name": "model-00001-of-00002.safetensors", "size_bytes": 4},
                {"name": "model-00002-of-00002.safetensors", "size_bytes": 4},
                {"name": "README.md", "size_bytes": 20},
            ]
        )
        self.opened_names: list[str] = []

    def open_file(self, repo: str, revision: str, filename: str, token: str):
        self.opened_names.append(filename)
        if filename.endswith(".safetensors.index.json"):
            self.opened = True
            return len(self.index_bytes), iter([self.index_bytes])
        return super().open_file(repo, revision, filename, token)


def test_complete_download_uses_the_safetensors_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("inspector_engine.model_library.read_token", lambda: "")
    hub = ShardedHub()
    report = discover("org/sharded", hub=hub)
    assert report["checkpoint"]["identifiable"] is True
    assert report["checkpoint"]["shards"] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert report["checkpoint"]["metadata"] == ["config.json", "model.safetensors.index.json"]
    assert report["checkpoint"]["total_bytes"] == 12 + len(hub.index_bytes) + 8
    job = start_download("org/sharded", [], mode="complete", hub=hub, library=tmp_path, wait=True)
    assert job["status"] == "done"
    assert "complete checkpoint" in job["message"]
    manifest = json.loads((tmp_path / "org__sharded" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["checkpoint_status"] == "complete"
    assert manifest["weight_shards_present"] == 2
    assert manifest["weight_shards_required"] == 2
    assert {row["name"] for row in manifest["files"]} == {
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    assert "README.md" not in {row["name"] for row in manifest["files"]}
    partial_root = tmp_path / "partial"
    partial = start_download(
        "org/sharded",
        ["model-00001-of-00002.safetensors"],
        mode="selected",
        hub=ShardedHub(),
        library=partial_root,
        wait=True,
    )
    assert partial["status"] == "done"
    assert "Partial checkpoint" in partial["message"]
    partial_manifest = json.loads((partial_root / "org__sharded" / "manifest.json").read_text(encoding="utf-8"))
    assert partial_manifest["checkpoint_status"] == "partial"
    assert partial_manifest["weight_shards_present"] == 1
    assert partial_manifest["weight_shards_required"] == 2
    assert [row["name"] for row in partial_manifest["files"]] == ["model-00001-of-00002.safetensors"]


def test_complete_download_stops_before_any_weight_when_disk_is_short(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("inspector_engine.model_library.read_token", lambda: "")
    monkeypatch.setattr("inspector_engine.model_library.free_bytes", lambda _path: 1)
    hub = ShardedHub()
    with pytest.raises(LibraryError) as caught:
        start_download("org/sharded", [], mode="complete", hub=hub, library=tmp_path, wait=True)
    assert caught.value.code == "insufficient_disk"
    assert not any(name.endswith(".safetensors") for name in hub.opened_names)
    assert not (tmp_path / "org__sharded").exists()


def test_two_weight_files_without_an_index_are_not_a_complete_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("inspector_engine.model_library.read_token", lambda: "")
    hub = FakeHub(
        files=[
            {"name": "config.json", "size_bytes": 12},
            {"name": "encoder.safetensors", "size_bytes": 4},
            {"name": "decoder.safetensors", "size_bytes": 4},
        ]
    )
    with pytest.raises(LibraryError) as caught:
        start_download("org/toy", [], mode="complete", hub=hub, library=tmp_path, wait=True)
    assert caught.value.code == "incomplete_checkpoint"
    assert hub.opened is False
    assert not (tmp_path / "org__toy").exists()


def test_disk_space_is_measured_again_immediately_before_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("inspector_engine.model_library.read_token", lambda: "")
    answers = iter([10**12, 1])
    monkeypatch.setattr("inspector_engine.model_library.free_bytes", lambda _path: next(answers))
    hub = FakeHub()
    job = start_download("org/toy", ["model.safetensors"], hub=hub, library=tmp_path, wait=True)
    assert job["status"] == "error"
    assert job["error_code"] == "insufficient_disk"
    assert hub.opened is False
    stored = tmp_path / "org__toy"
    assert not stored.exists() or not any(stored.rglob("*.safetensors"))
    assert not stored.exists() or not any(stored.rglob("*.part"))


def _store_weight(dest: Path, *, revision: str, integrity: str, digest: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "model.safetensors").write_bytes(b"data")
    (dest / "manifest.json").write_text(
        json.dumps(
            {
                "repo": "org/toy",
                "revision": revision,
                "files": [
                    {
                        "name": "model.safetensors",
                        "size_bytes": 4,
                        "revision": revision,
                        "integrity": integrity,
                        "expected_sha256": digest,
                        "observed_sha256": digest if integrity == "verified" else "ab" * 32,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_a_local_file_is_reused_only_when_revision_size_and_checksum_agree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("inspector_engine.model_library.read_token", lambda: "")
    digest = hashlib.sha256(b"data").hexdigest()
    files = [{"name": "model.safetensors", "size_bytes": 4, "expected_sha256": digest}]
    kept = tmp_path / "kept"
    _store_weight(kept / "org__toy", revision="main", integrity="verified", digest=digest)
    (kept / "org__toy" / "model.safetensors.part").write_bytes(b"xxxx")
    kept_hub = FakeHub(files=files)
    kept_job = start_download("org/toy", ["model.safetensors"], hub=kept_hub, library=kept, wait=True)
    assert kept_job["status"] == "done"
    assert kept_hub.opened is False
    assert (kept / "org__toy" / "model.safetensors").read_bytes() == b"data"
    assert (kept / "org__toy" / "model.safetensors.part").read_bytes() == b"xxxx"

    mismatched = tmp_path / "mismatched"
    _store_weight(mismatched / "org__toy", revision="main", integrity="mismatch", digest=digest)
    mismatch_hub = FakeHub(files=files)
    mismatch_job = start_download("org/toy", ["model.safetensors"], hub=mismatch_hub, library=mismatched, wait=True)
    assert mismatch_job["status"] == "done"
    assert mismatch_hub.opened is True

    other_revision = tmp_path / "revision"
    _store_weight(other_revision / "org__toy", revision="old", integrity="verified", digest=digest)
    revision_hub = FakeHub(files=files)
    revision_job = start_download("org/toy", ["model.safetensors"], hub=revision_hub, library=other_revision, wait=True)
    assert revision_job["status"] == "done"
    assert revision_hub.opened is True

    partial = tmp_path / "partial"
    partial_dest = partial / "org__toy"
    partial_dest.mkdir(parents=True)
    (partial_dest / "model.safetensors.part").write_bytes(b"data")
    partial_hub = FakeHub(files=files)
    partial_job = start_download("org/toy", ["model.safetensors"], hub=partial_hub, library=partial, wait=True)
    assert partial_job["status"] == "done"
    assert partial_hub.opened is True
    assert (partial_dest / "model.safetensors").read_bytes() == b"data"

    neighbor = tmp_path / "neighbor"
    (neighbor / "org__other").mkdir(parents=True)
    (neighbor / "org__other" / "model.safetensors").write_bytes(b"data")
    neighbor_hub = FakeHub(files=files)
    neighbor_job = start_download("org/toy", ["model.safetensors"], hub=neighbor_hub, library=neighbor, wait=True)
    assert neighbor_job["status"] == "done"
    assert neighbor_hub.opened is True
    assert (neighbor / "org__other" / "model.safetensors").read_bytes() == b"data"


def test_an_unverified_checksum_is_hashed_before_a_file_is_reused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("inspector_engine.model_library.read_token", lambda: "")
    digest = hashlib.sha256(b"data").hexdigest()
    dest = tmp_path / "org__toy"
    dest.mkdir()
    (dest / "model.safetensors").write_bytes(b"data")
    (dest / "manifest.json").write_text(
        json.dumps({"repo": "org/toy", "revision": "main", "files": [{"name": "model.safetensors", "size_bytes": 4, "revision": "main"}]}),
        encoding="utf-8",
    )
    hub = FakeHub(files=[{"name": "model.safetensors", "size_bytes": 4, "expected_sha256": digest}])
    job = start_download("org/toy", ["model.safetensors"], hub=hub, library=tmp_path, wait=True)
    assert job["status"] == "done"
    assert hub.opened is False
    stored = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    row = stored["files"][0]
    assert row["integrity"] == "verified"
    assert row["revision"] == "main"
    assert row["observed_sha256"] == digest


def test_insufficient_disk_downloads_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("inspector_engine.model_library.read_token", lambda: "")
    monkeypatch.setattr("inspector_engine.model_library.free_bytes", lambda _path: 1)
    hub = FakeHub()
    with pytest.raises(LibraryError) as caught:
        start_download("org/toy", ["model.safetensors"], hub=hub, library=tmp_path, wait=True)
    assert caught.value.code == "insufficient_disk"
    assert hub.opened is False
    assert not (tmp_path / "org__toy").exists()


def test_authentication_error_does_not_reveal_the_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("inspector_engine.model_library.read_token", lambda: "")
    with pytest.raises(LibraryError) as caught:
        discover("gated/model", hub=FakeHub())
    assert caught.value.code == "hub_unauthorized"
    assert "Public model search" in caught.value.message
    assert SECRET not in caught.value.message
    monkeypatch.setattr(
        "inspector_engine.model_library.auth_view",
        lambda: {"status": "none", "label": "No token", "present": False, "usable": False, "validated": False},
    )
    assert discover("org/toy", hub=FakeHub())["hf_auth_label"] == "No token"

    monkeypatch.setattr("inspector_engine.model_library.read_token", lambda: SECRET)
    report = discover("org/toy", hub=FakeHub())
    assert report["hf_token_configured"] is True
    assert report["license"] is None
    assert SECRET not in json.dumps(report)

    leaked = HTTPError("https://huggingface.co/org/toy", 401, "no", hdrs=None, fp=io.BytesIO(SECRET.encode()))
    with pytest.raises(LibraryError) as http_error:
        _raise_http(leaked)
    assert SECRET not in http_error.value.message


def test_download_route_accepts_complete_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def fake(repo: str, files: list[str], revision: str | None = None, mode: str = "selected") -> dict:
        seen["repo"] = repo
        seen["files"] = files
        seen["mode"] = mode
        return {"ok": True, "id": "job", "status": "running", "message": "Download started."}

    monkeypatch.setattr("server.start_download", fake)
    client = TestClient(app)
    response = client.post("/models/download", json={"repo": "org/sharded", "files": [], "revision": "main", "mode": "complete"})
    assert response.status_code == 200
    assert seen == {"repo": "org/sharded", "files": [], "mode": "complete"}


def test_search_route_returns_the_catalog_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "server.search_models",
        lambda q, cursor="", sort="relevance", task="all", scope="public": {
            "ok": True,
            "query": q,
            "results": [],
            "next_cursor": cursor or None,
            "sort": sort,
            "task": task,
            "scope": scope,
        },
    )
    client = TestClient(app)
    response = client.get("/models/search", params={"q": "Gemma", "cursor": "page-2", "sort": "downloads", "task": "text", "scope": "private"})
    assert response.status_code == 200
    assert response.json()["query"] == "Gemma"
    assert response.json()["next_cursor"] == "page-2"
    assert response.json()["sort"] == "downloads"
    assert response.json()["task"] == "text"
    assert response.json()["scope"] == "private"


def test_api_discovery_and_local_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("inspector_engine.model_library.read_token", lambda: SECRET)
    monkeypatch.setattr("inspector_engine.model_library.LIBRARY_DIR", tmp_path)
    hub = FakeHub()
    monkeypatch.setattr("inspector_engine.model_library.UrllibHub", lambda: hub)
    client = TestClient(app)
    response = client.get("/models/discover", params={"q": "org/toy"})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert SECRET not in response.text
    started = start_download("org/toy", ["config.json"], hub=hub, library=tmp_path, wait=True)
    assert started["status"] == "done"
    listed = client.get("/models/local")
    assert listed.status_code == 200
    assert listed.json()["models"][0]["repo"] == "org/toy"
    assert SECRET not in listed.text


def _write_model(root: Path, repo: str, files: dict[str, bytes], required: list[str] | None = None) -> Path:
    dest = root / repo.replace("/", "__")
    dest.mkdir(parents=True)
    rows = []
    for name, blob in files.items():
        path = dest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        rows.append({"name": name, "size_bytes": len(blob), "path": str(path)})
    payload = {"repo": repo, "revision": "main", "files": rows}
    if required is not None:
        payload["required_shards"] = required
        payload["required_metadata"] = ["config.json"] if "config.json" in files else []
    (dest / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return dest


def test_downloaded_list_names_a_partial_shard_without_calling_it_complete(tmp_path: Path) -> None:
    _write_model(tmp_path, "Qwen/Qwen3-Coder-30B-A3B-Instruct", {"model-00016-of-00016.safetensors": b"shard"})
    _write_model(tmp_path, "hf-internal-testing/tiny-random-gpt2", {"config.json": b"{}", "model.safetensors": b"weights"})
    listed = list_local_models(library=tmp_path)
    by_repo = {row["repo"]: row for row in listed["models"]}
    partial = by_repo["Qwen/Qwen3-Coder-30B-A3B-Instruct"]
    assert partial["checkpoint_label"] == "Partial checkpoint — 1 of 16 shards downloaded."
    assert partial["checkpoint_status"] == "partial"
    assert partial["weight_shards_present"] == 1
    assert partial["weight_shards_required"] == 16
    complete = by_repo["hf-internal-testing/tiny-random-gpt2"]
    assert complete["checkpoint_label"] == "Complete checkpoint"
    assert complete["checkpoint_status"] == "complete"


def test_delete_one_shard_leaves_the_other_file(tmp_path: Path) -> None:
    _write_model(
        tmp_path,
        "org/sharded",
        {
            "model-00001-of-00002.safetensors": b"aaaa",
            "model-00002-of-00002.safetensors": b"bbbbbb",
            "config.json": b"{}",
        },
        required=["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
    )
    before = (tmp_path / "org__sharded" / "model-00002-of-00002.safetensors").read_bytes()
    result = delete_downloaded("org/sharded", filename="model-00001-of-00002.safetensors", scope="file", library=tmp_path)
    assert result["freed_bytes"] == 4
    assert result["freed_label"]
    dest = tmp_path / "org__sharded"
    assert not (dest / "model-00001-of-00002.safetensors").exists()
    assert (dest / "model-00002-of-00002.safetensors").read_bytes() == before
    assert (dest / "config.json").read_bytes() == b"{}"
    listed = list_local_models(library=tmp_path)["models"]
    assert listed[0]["checkpoint_label"] == "Partial checkpoint — 1 of 2 shards downloaded."
    assert [row["name"] for row in listed[0]["files"]] == ["model-00002-of-00002.safetensors", "config.json"]


def test_delete_entire_model_removes_only_that_library_directory(tmp_path: Path) -> None:
    dest = _write_model(tmp_path, "org/small", {"model.safetensors": b"tiny", "config.json": b"{}"})
    _write_model(tmp_path, "org/keep", {"model.safetensors": b"stay"})
    expected = sum(path.stat().st_size for path in dest.rglob("*") if path.is_file())
    result = delete_downloaded("org/small", scope="model", library=tmp_path)
    assert result["freed_bytes"] == expected
    assert not dest.exists()
    assert (tmp_path / "org__keep" / "model.safetensors").read_bytes() == b"stay"
    assert [row["repo"] for row in list_local_models(library=tmp_path)["models"]] == ["org/keep"]


def test_delete_refuses_paths_outside_the_library(tmp_path: Path) -> None:
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"secret")
    _write_model(tmp_path / "library", "org/toy", {"model.safetensors": b"keep"})
    library = tmp_path / "library"
    for filename in ("/etc/passwd", "../../outside.safetensors", "model.safetensors/../../outside.safetensors"):
        with pytest.raises(LibraryError) as caught:
            delete_downloaded("org/toy", filename=filename, scope="file", library=library)
        assert caught.value.code == "path_rejected"
    assert outside.read_bytes() == b"secret"
    assert (library / "org__toy" / "model.safetensors").read_bytes() == b"keep"
    with pytest.raises(LibraryError):
        delete_downloaded("book-three/freshman", scope="model", library=library)
    assert outside.read_bytes() == b"secret"


def test_delete_refuses_a_symbolic_link_and_an_active_read(tmp_path: Path) -> None:
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"secret")
    dest = _write_model(tmp_path, "org/toy", {"model.safetensors": b"keep"})
    link = dest / "escape.safetensors"
    link.symlink_to(outside)
    with pytest.raises(LibraryError) as linked:
        delete_downloaded("org/toy", filename="escape.safetensors", scope="file", library=tmp_path)
    assert linked.value.code == "path_rejected"
    assert outside.read_bytes() == b"secret"
    with pytest.raises(LibraryError) as whole:
        delete_downloaded("org/toy", scope="model", library=tmp_path)
    assert whole.value.code == "path_rejected"
    assert (dest / "model.safetensors").read_bytes() == b"keep"
    assert outside.read_bytes() == b"secret"
    link.unlink()
    model_library._JOBS["active"] = {"repo": "org/toy", "status": "running"}
    try:
        with pytest.raises(LibraryError) as busy:
            delete_downloaded("org/toy", filename="model.safetensors", scope="file", library=tmp_path)
        assert busy.value.code == "download_active"
    finally:
        model_library._JOBS.pop("active", None)
    assert (dest / "model.safetensors").read_bytes() == b"keep"
    release = threading.Event()
    started = threading.Event()

    def hold() -> None:
        with hold_inspection(dest / "model.safetensors"):
            started.set()
            release.wait(2)

    thread = threading.Thread(target=hold)
    thread.start()
    assert started.wait(2)
    try:
        with pytest.raises(LibraryError) as inspected:
            delete_downloaded("org/toy", filename="model.safetensors", scope="file", library=tmp_path)
        assert inspected.value.code == "inspection_active"
    finally:
        release.set()
        thread.join(2)
    assert (dest / "model.safetensors").read_bytes() == b"keep"


def test_a_published_lfs_oid_is_a_file_checksum() -> None:
    digest = "ab" * 32
    assert _expected_sha256({"lfs": {"oid": digest}}) == digest
    assert _expected_sha256({"lfs": {"oid": "sha256:" + digest}}) == digest
    assert _expected_sha256({"oid": digest}) is None
    assert _expected_sha256({"lfs": {"oid": "not-a-hash"}}) is None


def test_checksum_agreement_does_not_decide_completeness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("inspector_engine.model_library.read_token", lambda: "")
    digest = hashlib.sha256(b"data").hexdigest()
    hub = FakeHub(
        files=[
            {"name": "config.json", "size_bytes": 12},
            {"name": "model.safetensors", "size_bytes": 4, "expected_sha256": digest},
        ]
    )
    job = start_download("org/toy", ["config.json", "model.safetensors"], hub=hub, library=tmp_path, wait=True)
    assert job["status"] == "done"
    listed = list_local_models(library=tmp_path)["models"][0]
    assert listed["checkpoint_status"] == "complete"
    assert listed["integrity_status"] == "verified"
    assert "published checksums match" in listed["integrity_label"]
    assert listed["inspection_status"] == "not_inspected"
    weight = next(row for row in listed["files"] if row["name"] == "model.safetensors")
    assert weight["observed_sha256"] == digest
    assert weight["integrity"] == "verified"
    bad_root = tmp_path / "bad"
    bad = FakeHub(
        files=[
            {"name": "config.json", "size_bytes": 12},
            {"name": "model.safetensors", "size_bytes": 4, "expected_sha256": "cd" * 32},
        ]
    )
    mismatch = start_download(
        "org/toy",
        ["config.json", "model.safetensors"],
        hub=bad,
        library=bad_root,
        wait=True,
    )
    assert mismatch["status"] == "done"
    assert (bad_root / "org__toy" / "model.safetensors").read_bytes() == b"data"
    bad_listed = list_local_models(library=bad_root)["models"][0]
    assert bad_listed["checkpoint_status"] == "complete"
    assert bad_listed["integrity_status"] == "mismatch"


def test_parsing_a_file_does_not_verify_its_checksum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("inspector_engine.model_library.LIBRARY_DIR", tmp_path)
    dest = _write_model(tmp_path, "org/toy", {"model.safetensors": b"data", "config.json": b"{}"})
    note_library_inspection(dest / "model.safetensors", tensor_count=4, readable=True)
    listed = list_local_models(library=tmp_path)["models"][0]
    assert listed["inspection_status"] == "inspected"
    assert listed["integrity_status"] == "unavailable"
    assert listed["checkpoint_status"] == "complete"
