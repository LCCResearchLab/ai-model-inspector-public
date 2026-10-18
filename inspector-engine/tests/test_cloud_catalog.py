# Authors: LCC Research Lab
# Project: From Document to Evidence

import io
import json

import pytest
from urllib.error import HTTPError

from inspector_engine import cloud_catalog, credentials
from inspector_engine.cloud_catalog import _require_safe_url, fetch_catalog, probe_cloud
from inspector_engine.credentials import CredentialError, list_credentials, save_credential

OPENAI_KEY = "openai-secret-SHOULD_NOT_LEAK"
ANTHROPIC_KEY = "anthropic-secret-SHOULD_NOT_LEAK"
GOOGLE_KEY = "google-secret-SHOULD_NOT_LEAK"


class MemoryStore:
    def __init__(self) -> None:
        self.items: dict[str, str] = {}

    def exists(self, provider: str) -> bool:
        return provider in self.items

    def get(self, provider: str) -> str | None:
        return self.items.get(provider)

    def put(self, provider: str, secret: str) -> None:
        self.items[provider] = secret

    def delete(self, provider: str) -> None:
        self.items.pop(provider, None)


@pytest.fixture
def store(tmp_path, monkeypatch: pytest.MonkeyPatch) -> MemoryStore:
    memory = MemoryStore()
    monkeypatch.setattr(credentials, "STATUS_PATH", tmp_path / "credential-status.json")
    monkeypatch.setattr(credentials, "CONTROLLER_TOKEN", tmp_path / "controller-token")
    monkeypatch.delenv("INSPECTOR_HF_TOKEN", raising=False)
    credentials.set_store(memory)
    yield memory
    credentials.set_store(None)


def _body(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


def test_provider_requests_use_headers_and_keep_the_key_out_of_the_url(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, str, str]] = []

    class Opener:
        def open(self, request, timeout):
            url = request.full_url
            if url.startswith("https://api.openai.com/"):
                provider = "openai"
                header = request.get_header("Authorization")
            elif url.startswith("https://api.anthropic.com/"):
                provider = "anthropic"
                header = request.get_header("X-api-key")
                assert request.get_header("Anthropic-version") == "2023-06-01"
            else:
                provider = "google"
                header = request.get_header("X-goog-api-key")
                assert request.get_header("Authorization") is None
            seen.append((provider, url, header))
            assert "key=" not in request.full_url
            assert OPENAI_KEY not in request.full_url
            assert ANTHROPIC_KEY not in request.full_url
            assert GOOGLE_KEY not in request.full_url
            raise HTTPError(request.full_url, 401, "no", hdrs=None, fp=io.BytesIO(b'{"error":"invalid api key"}'))

    monkeypatch.setattr(cloud_catalog, "build_opener", lambda *args: Opener())
    assert probe_cloud("openai", OPENAI_KEY) == "rejected"
    assert probe_cloud("anthropic", ANTHROPIC_KEY) == "rejected"
    assert probe_cloud("google", GOOGLE_KEY) == "rejected"
    assert seen[0][1] == "https://api.openai.com/v1/models"
    assert seen[0][2] == f"Bearer {OPENAI_KEY}"
    assert seen[1][1] == "https://api.anthropic.com/v1/models?limit=100"
    assert seen[1][2] == ANTHROPIC_KEY
    assert "pageToken" not in seen[2][1]
    assert "key=" not in seen[2][1]
    assert seen[2][2] == GOOGLE_KEY


def test_a_redirect_is_refused_without_removing_or_echoing_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    class Opener:
        def open(self, request, timeout):
            handler = args_handler
            return handler.redirect_request(request, None, 302, "see", {}, "https://evil.example/?key=" + OPENAI_KEY)

    opener = Opener()
    args_handler = None

    def build(*args):
        nonlocal args_handler
        args_handler = args[0]
        return opener

    monkeypatch.setattr(cloud_catalog, "build_opener", build)
    with pytest.raises(CredentialError) as caught:
        probe_cloud("openai", OPENAI_KEY)
    assert caught.value.code == "connection_failed"
    assert OPENAI_KEY not in caught.value.message
    assert "evil.example" not in caught.value.message


def test_permission_rate_limit_and_network_failures_stay_on_one_provider(store: MemoryStore) -> None:
    save_credential("openai", OPENAI_KEY)
    save_credential("anthropic", ANTHROPIC_KEY)
    save_credential("google", GOOGLE_KEY)

    def transport(provider: str, secret: str, url: str):
        assert secret not in url
        assert "key=" not in url
        if provider == "openai":
            return 403, b'{"error":{"status":"PERMISSION_DENIED"}}'
        if provider == "anthropic":
            return 429, b"{}"
        raise OSError(f"down {secret}")

    with pytest.raises(CredentialError) as denied:
        fetch_catalog("openai", OPENAI_KEY, transport=transport)
    assert denied.value.code == "permission_denied"
    assert OPENAI_KEY not in denied.value.message

    with pytest.raises(CredentialError) as limited:
        fetch_catalog("anthropic", ANTHROPIC_KEY, transport=transport)
    assert limited.value.code == "rate_limited"
    assert ANTHROPIC_KEY not in limited.value.message

    with pytest.raises(CredentialError) as offline:
        probe_cloud("google", GOOGLE_KEY, transport=transport)
    assert offline.value.code == "connection_failed"
    assert GOOGLE_KEY not in offline.value.message

    listed = {row["id"]: row["status"] for row in list_credentials()["providers"]}
    assert listed["openai"] == "configured"
    assert listed["anthropic"] == "configured"
    assert listed["google"] == "configured"
    assert store.get("openai") == OPENAI_KEY
    assert store.get("anthropic") == ANTHROPIC_KEY
    assert store.get("google") == GOOGLE_KEY


def test_catalogs_keep_reported_metadata_and_follow_only_safe_pages() -> None:
    calls: list[str] = []

    def transport(provider: str, secret: str, url: str):
        calls.append(url)
        assert secret not in url
        assert "key=" not in url.lower()
        if provider == "google" and "pageToken" not in url:
            return 200, _body(
                {
                    "models": [
                        {
                            "name": "models/gemini-test",
                            "displayName": "Gemini Test",
                            "supportedGenerationMethods": ["generateContent", "countTokens"],
                        }
                    ],
                    "nextPageToken": "next-page",
                }
            )
        if provider == "google":
            return 200, _body({"models": [{"name": "models/gemini-next", "displayName": "Next"}]})
        if provider == "anthropic":
            return 200, _body(
                {
                    "data": [{"id": "claude-test", "display_name": "Claude Test", "created_at": "2026-01-01T00:00:00Z"}],
                    "has_more": False,
                }
            )
        return 200, _body({"data": [{"id": "gpt-test", "owned_by": "example-org", "created": 1}]})

    google = fetch_catalog("google", GOOGLE_KEY, transport=transport)
    assert [model["id"] for model in google["models"]] == ["gemini-test", "gemini-next"]
    assert google["models"][0]["detail"] == "Provider-reported methods: generateContent, countTokens"
    assert calls[1].startswith("https://generativelanguage.googleapis.com/v1beta/models?")
    assert "pageToken=next-page" in calls[1]
    assert "key=" not in calls[1]

    anthropic = fetch_catalog("anthropic", ANTHROPIC_KEY, transport=transport)
    assert anthropic["models"][0]["display_name"] == "Claude Test"
    assert "Provider-reported created" in anthropic["models"][0]["detail"]

    openai = fetch_catalog("openai", OPENAI_KEY, transport=transport)
    assert openai["models"] == [
        {
            "id": "gpt-test",
            "display_name": None,
            "detail": "Owned by example-org",
            "facts": [{"name": "Owner", "value": "example-org"}],
        }
    ]
    assert openai["truncated"] is False


def test_a_google_page_token_is_not_a_credential_parameter() -> None:
    allowed = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=50&pageToken=next-page"
    _require_safe_url("google", GOOGLE_KEY, allowed)
    with pytest.raises(CredentialError) as token_param:
        _require_safe_url("google", GOOGLE_KEY, "https://generativelanguage.googleapis.com/v1beta/models?token=next-page")
    assert "stopped before it was sent" in token_param.value.message
    assert GOOGLE_KEY not in token_param.value.message
    with pytest.raises(CredentialError):
        _require_safe_url("google", GOOGLE_KEY, "https://generativelanguage.googleapis.com/v1beta/models?key=next-page")


def test_a_model_id_that_contains_the_key_is_omitted() -> None:
    def transport(provider: str, secret: str, url: str):
        return 200, _body({"data": [{"id": secret, "owned_by": "example"}, {"id": "gpt-public", "owned_by": secret}]})

    catalog = fetch_catalog("openai", OPENAI_KEY, transport=transport)
    assert catalog["models"] == [{"id": "gpt-public", "display_name": None, "detail": None, "facts": []}]
    assert OPENAI_KEY not in json.dumps(catalog)


def test_model_details_keep_reported_fields_and_omit_unreported_measurements() -> None:
    def transport(provider: str, secret: str, url: str):
        if provider == "openai":
            return 200, _body(
                {
                    "data": [
                        {
                            "id": "gpt-test",
                            "owned_by": "system",
                            "created": 1_700_000_000,
                            "shutdown_date": "2026-12-01",
                            "context_window": 128000,
                            "parameters": "1.8T",
                        }
                    ]
                }
            )
        return 200, _body(
            {
                "models": [
                    {
                        "name": "models/gemini-test",
                        "displayName": "Gemini Test",
                        "inputTokenLimit": 1048576,
                        "outputTokenLimit": 8192,
                        "supportedGenerationMethods": ["generateContent"],
                    }
                ]
            }
        )

    openai = fetch_catalog("openai", OPENAI_KEY, transport=transport)
    names = [fact["name"] for fact in openai["models"][0]["facts"]]
    assert names == ["Owner", "Created", "Shutdown date"]
    assert openai["models"][0]["facts"][1]["value"] == "2023-11-14 22:13:20 UTC"
    assert "128000" not in json.dumps(openai)
    assert "1.8T" not in json.dumps(openai)

    google = fetch_catalog("google", GOOGLE_KEY, transport=transport)
    assert google["models"][0]["facts"] == [
        {"name": "Input token limit", "value": "1048576"},
        {"name": "Output token limit", "value": "8192"},
        {"name": "Supported methods", "value": "generateContent"},
    ]


def test_a_rejected_catalog_marks_only_that_provider(store: MemoryStore, monkeypatch: pytest.MonkeyPatch) -> None:
    from inspector_engine.credentials import list_cloud_models

    save_credential("openai", OPENAI_KEY)
    save_credential("anthropic", ANTHROPIC_KEY)
    save_credential("google", GOOGLE_KEY)

    def catalog(provider: str, secret: str, *, transport=None) -> dict:
        if provider == "openai":
            raise CredentialError("rejected", "OpenAI did not accept this key. Nothing was removed.", 401)
        assert secret == ANTHROPIC_KEY
        return {"models": [{"id": "claude-test", "display_name": "Claude Test", "detail": None}], "truncated": False}

    monkeypatch.setattr(cloud_catalog, "fetch_catalog", catalog)
    with pytest.raises(CredentialError) as caught:
        list_cloud_models("openai")
    assert caught.value.code == "rejected"
    assert OPENAI_KEY not in caught.value.message
    listed = list_cloud_models("anthropic")
    assert listed["models"][0]["id"] == "claude-test"
    assert OPENAI_KEY not in json.dumps(listed)
    assert ANTHROPIC_KEY not in json.dumps(listed)
    status = {row["id"]: row["status"] for row in list_credentials()["providers"]}
    assert status["openai"] == "invalid"
    assert status["anthropic"] == "validated"
    assert status["google"] == "configured"


def test_an_invalid_google_key_is_rejected_from_the_error_body() -> None:
    def transport(provider: str, secret: str, url: str):
        return 400, _body({"error": {"status": "INVALID_ARGUMENT", "message": f"API key not valid {secret}"}})

    assert probe_cloud("google", GOOGLE_KEY, transport=transport) == "rejected"
