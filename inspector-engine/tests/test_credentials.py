# Authors: LCC Research Lab
# Project: From Document to Evidence

import ctypes
import io
import json
import sys
import tempfile
from ctypes import POINTER, byref, c_char_p, c_int32, c_uint32, c_void_p
from pathlib import Path
from urllib.error import HTTPError

import pytest
from fastapi.testclient import TestClient

from inspector_engine import credentials
from inspector_engine.credentials import (
    CredentialError,
    KeychainStore,
    list_credentials,
    probe_huggingface,
    remove_credential,
    save_credential,
    check_credential,
)
from inspector_engine.model_library import LibraryError, discover, read_token
from server import app

SECRET = "hf_secret_SHOULD_NOT_LEAK_9911"
CONTROLLER = "controller-test-token"


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


class Hub:
    def __init__(self) -> None:
        self.token_seen = ""

    def model_info(self, repo: str, token: str) -> dict:
        self.token_seen = token
        return {"license": SECRET if token == SECRET else "apache-2.0", "cardData": {}, "config": {"model_type": "toy"}}

    def file_list(self, repo: str, revision: str, token: str) -> list[dict]:
        return [{"name": "config.json", "size_bytes": 4}]

    def readme(self, repo: str, revision: str, token: str) -> str | None:
        return "A public card."

    def open_file(self, repo: str, revision: str, filename: str, token: str):
        raise AssertionError("lookup must not download")


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryStore:
    memory = MemoryStore()
    monkeypatch.setattr(credentials, "STATUS_PATH", tmp_path / "credential-status.json")
    monkeypatch.setattr(credentials, "CONTROLLER_TOKEN", tmp_path / "controller-token")
    (tmp_path / "controller-token").write_text(CONTROLLER + "\n", encoding="utf-8")
    monkeypatch.delenv("INSPECTOR_HF_TOKEN", raising=False)
    monkeypatch.delenv("INSPECTOR_HOST", raising=False)
    credentials.set_store(memory)
    yield memory
    credentials.set_store(None)


def test_environment_token_overrides_the_keychain(store: MemoryStore, monkeypatch: pytest.MonkeyPatch) -> None:
    store.put("huggingface", "keychain-value-should-stay-hidden")
    monkeypatch.setenv("INSPECTOR_HF_TOKEN", SECRET)
    assert read_token() == SECRET
    listed = list_credentials()
    assert SECRET not in json.dumps(listed)
    assert "keychain-value-should-stay-hidden" not in json.dumps(listed)
    huggingface = next(row for row in listed["providers"] if row["id"] == "huggingface")
    assert huggingface["status_label"] == "Configured"
    assert huggingface["source"] == "environment"
    assert [row["id"] for row in listed["providers"]] == ["huggingface", "openai", "anthropic", "google"]
    assert listed["providers"][1]["available"] is True
    assert listed["providers"][1]["source"] is None


def test_a_plaintext_token_file_is_not_a_credential_source(store: MemoryStore, tmp_path: Path) -> None:
    source = Path(credentials.__file__).resolve().parent.joinpath("model_library.py").read_text(encoding="utf-8")
    assert "hf-token" not in source
    leftover = tmp_path / "hf-token"
    leftover.write_text(SECRET, encoding="utf-8")
    assert read_token() == ""
    listed = json.dumps(list_credentials())
    assert SECRET not in listed


def test_save_replace_remove_and_a_failed_test_keeps_the_token(store: MemoryStore, tmp_path: Path) -> None:
    saved = save_credential("huggingface", SECRET)
    assert saved["status_label"] == "Configured"
    assert saved["saved"] is True
    assert SECRET not in json.dumps(saved)
    assert store.get("huggingface") == SECRET
    assert SECRET not in (tmp_path / "credential-status.json").read_text(encoding="utf-8")

    replaced = save_credential("huggingface", "replacement-token-value")
    assert replaced["status_label"] == "Configured"
    assert store.get("huggingface") == "replacement-token-value"
    store.put("huggingface", SECRET)

    def reject(token: str) -> str:
        assert token == SECRET
        return "rejected"

    tested = check_credential("huggingface", probe=reject)
    assert tested["status_label"] == "Invalid"
    assert "Nothing was removed" in tested["message"]
    assert store.get("huggingface") == SECRET
    assert SECRET not in json.dumps(tested)

    def explode(token: str) -> str:
        raise RuntimeError(f"hub said {token}")

    with pytest.raises(CredentialError) as caught:
        check_credential("huggingface", probe=explode)
    assert caught.value.status == 502
    assert SECRET not in caught.value.message
    assert store.get("huggingface") == SECRET
    assert list_credentials()["providers"][0]["status_label"] == "Invalid"

    removed = remove_credential("huggingface")
    assert removed["status_label"] == "Not configured"
    assert removed["saved"] is False
    assert store.get("huggingface") is None


class _AuthHub:
    def __init__(self, *, refuse_token: bool = False, refuse_public: bool = False, private: bool = False):
        self.tokens: list[str] = []
        self.refuse_token = refuse_token
        self.refuse_public = refuse_public
        self.private = private

    def model_info(self, repo: str, token: str) -> dict:
        self.tokens.append(token)
        if token and self.refuse_token:
            raise LibraryError("hub_unauthorized", "refused", 401)
        if not token and self.refuse_public:
            raise LibraryError("hub_unauthorized", "refused", 401)
        return {
            "private": self.private,
            "gated": False,
            "license": "apache-2.0",
            "cardData": {},
            "config": {"model_type": "toy"},
            "tags": [],
        }

    def file_list(self, repo: str, revision: str, token: str) -> list[dict]:
        self.tokens.append(token)
        return [{"name": "config.json", "size_bytes": 4}]

    def readme(self, repo: str, revision: str, token: str) -> str | None:
        return "Public model."

    def open_file(self, repo: str, revision: str, filename: str, token: str):
        raise AssertionError("lookup must not download")


def test_missing_token_keeps_public_lookup_available(store: MemoryStore) -> None:
    from inspector_engine.credentials import huggingface_auth

    public = _AuthHub()
    report = discover("org/toy", hub=public)
    assert public.tokens and set(public.tokens) == {""}
    assert report["hf_auth"] == "none"
    assert report["hf_auth_label"] == "No token"
    assert huggingface_auth()["present"] is False
    gated = _AuthHub(refuse_public=True)
    with pytest.raises(LibraryError) as caught:
        discover("org/gated", hub=gated)
    assert caught.value.code == "hub_unauthorized"
    assert "Open HF Token" in caught.value.message
    assert SECRET not in caught.value.message
    again = discover("org/toy", hub=_AuthHub())
    assert again["ok"] is True
    assert again["hf_auth_label"] == "No token"


def test_a_successful_token_check_is_what_unlocks_private_search(store: MemoryStore) -> None:
    from inspector_engine.credentials import _remember, huggingface_auth

    store.put("huggingface", SECRET)
    _remember("huggingface", "keychain", "configured")
    assert huggingface_auth()["validated"] is False
    _remember("huggingface", "keychain", "validated")
    auth = huggingface_auth()
    assert auth["status"] == "configured"
    assert auth["validated"] is True
    assert SECRET not in json.dumps(auth)


def test_valid_token_does_not_grant_every_private_repository(store: MemoryStore, monkeypatch: pytest.MonkeyPatch) -> None:
    from inspector_engine.credentials import _remember

    store.put("huggingface", SECRET)
    _remember("huggingface", "keychain", "configured")
    monkeypatch.setattr(credentials, "probe_huggingface", lambda token: "ok")
    opened = _AuthHub(private=True)
    report = discover("org/private", hub=opened)
    assert opened.tokens[0] == SECRET
    assert report["hf_auth"] == "configured"
    assert report["private"] is True
    assert "does not grant access" in report["auth_note"]
    assert SECRET not in json.dumps(report)

    denied = _AuthHub(refuse_token=True, refuse_public=True)
    with pytest.raises(LibraryError) as caught:
        discover("org/closed", hub=denied)
    assert caught.value.code == "hub_access_denied"
    assert "not granted" in caught.value.message
    assert store.get("huggingface") == SECRET
    assert SECRET not in caught.value.message


def test_invalid_token_is_kept_and_omitted_from_public_requests(store: MemoryStore) -> None:
    from inspector_engine.credentials import _remember

    store.put("huggingface", SECRET)
    _remember("huggingface", "keychain", "invalid")
    hub = _AuthHub()
    report = discover("org/toy", hub=hub)
    assert hub.tokens and set(hub.tokens) == {""}
    assert report["hf_auth"] == "invalid"
    assert report["hf_auth_label"] == "Invalid"
    assert "still stored" in report["auth_note"]
    assert store.get("huggingface") == SECRET
    assert SECRET not in json.dumps(report)


def test_revoked_token_stays_stored_and_public_lookup_retries(store: MemoryStore, monkeypatch: pytest.MonkeyPatch) -> None:
    from inspector_engine.credentials import _remember, huggingface_auth

    store.put("huggingface", SECRET)
    _remember("huggingface", "keychain", "configured")
    monkeypatch.setattr(credentials, "probe_huggingface", lambda token: "rejected")
    hub = _AuthHub(refuse_token=True)
    report = discover("org/toy", hub=hub)
    assert hub.tokens[0] == SECRET
    assert "" in hub.tokens
    assert report["hf_auth"] == "invalid"
    assert store.get("huggingface") == SECRET
    assert huggingface_auth()["usable"] is False
    assert SECRET not in json.dumps(report)
    later = _AuthHub()
    again = discover("org/toy", hub=later)
    assert later.tokens and set(later.tokens) == {""}
    assert again["hf_auth_label"] == "Invalid"


def test_lookup_uses_the_saved_token_without_returning_it(store: MemoryStore) -> None:
    store.put("huggingface", SECRET)
    hub = Hub()
    report = discover("org/toy", hub=hub)
    assert hub.token_seen == SECRET
    assert SECRET not in json.dumps(report)
    assert report["hf_token_configured"] is True


def test_connection_probe_talks_only_to_huggingface(monkeypatch: pytest.MonkeyPatch) -> None:
    class Opener:
        def open(self, request, timeout):
            assert request.full_url == "https://huggingface.co/api/whoami-v2"
            assert request.get_header("Authorization") == f"Bearer {SECRET}"
            raise HTTPError(request.full_url, 401, "no", hdrs=None, fp=io.BytesIO(SECRET.encode()))

    monkeypatch.setattr(credentials, "build_opener", lambda *args: Opener())
    assert probe_huggingface(SECRET) == "rejected"


def test_cloud_keys_are_stored_without_appearing_in_the_response(store: MemoryStore, tmp_path: Path) -> None:
    saved = save_credential("openai", "openai-posted-secret")
    assert saved["status"] == "configured"
    assert saved["saved"] is True
    assert "openai-posted-secret" not in json.dumps(saved)
    assert store.get("openai") == "openai-posted-secret"
    assert "openai-posted-secret" not in (tmp_path / "credential-status.json").read_text(encoding="utf-8")
    assert store.get("huggingface") is None


def test_provider_accounts_stay_independent(store: MemoryStore) -> None:
    save_credential("huggingface", SECRET)
    save_credential("openai", "openai-account-separate")
    save_credential("anthropic", "anthropic-account-separate")
    save_credential("google", "google-account-separate")
    accounts = [credentials._names(provider)[1] for provider in ("huggingface", "openai", "anthropic", "google")]
    assert len(set(accounts)) == 4
    assert store.get("openai") != store.get("huggingface")

    def reject_openai(token: str) -> str:
        assert token == "openai-account-separate"
        return "rejected"

    checked = check_credential("openai", probe=reject_openai)
    assert checked["status"] == "invalid"
    assert "openai-account-separate" not in json.dumps(checked)
    assert SECRET not in json.dumps(checked)
    listed = {row["id"]: row["status"] for row in list_credentials()["providers"]}
    assert listed["openai"] == "invalid"
    assert listed["anthropic"] == "configured"
    assert listed["google"] == "configured"
    assert listed["huggingface"] == "configured"
    remove_credential("openai")
    assert store.get("openai") is None
    assert store.get("huggingface") == SECRET
    assert store.get("anthropic") == "anthropic-account-separate"
    assert store.get("google") == "google-account-separate"


def test_a_hugging_face_environment_token_is_not_a_cloud_key(store: MemoryStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INSPECTOR_HF_TOKEN", SECRET)
    seen: list[str] = []

    def capture(token: str) -> str:
        seen.append(token)
        return "ok"

    with pytest.raises(CredentialError) as caught:
        check_credential("openai", probe=capture)
    assert caught.value.code == "not_configured"
    assert seen == []
    assert SECRET not in caught.value.message
    save_credential("google", "google-account-separate")
    checked = check_credential("google", probe=capture)
    assert seen == ["google-account-separate"]
    assert checked["status"] == "validated"
    assert SECRET not in json.dumps(checked)


def test_cloud_routes_store_a_posted_key_without_echoing_it(store: MemoryStore, monkeypatch: pytest.MonkeyPatch) -> None:
    store.put("huggingface", SECRET)
    client = TestClient(app)

    def accept(provider: str, secret: str, *, transport=None) -> str:
        assert secret == f"{provider}-posted-secret"
        return "ok"

    def catalog(provider: str, secret: str) -> dict:
        assert secret == f"{provider}-posted-secret"
        return {"models": [{"id": f"{provider}-model", "display_name": None, "detail": "Owned by example"}], "truncated": False}

    monkeypatch.setattr("inspector_engine.cloud_catalog.probe_cloud", accept)
    monkeypatch.setattr("inspector_engine.cloud_catalog.fetch_catalog", catalog)
    for provider in ("openai", "anthropic", "google"):
        posted = f"{provider}-posted-secret"
        response = client.post(f"/credentials/{provider}", headers=_auth(), json={"token": posted})
        assert response.status_code == 200
        assert response.json()["status"] == "configured"
        assert posted not in response.text
        assert SECRET not in response.text
        assert store.get(provider) == posted
        tested = client.post(f"/credentials/{provider}/test", headers=_auth())
        assert tested.status_code == 200
        assert tested.json()["status"] == "validated"
        assert posted not in tested.text
        listed = client.get(f"/credentials/{provider}/models", headers=_auth())
        assert listed.status_code == 200
        assert listed.json()["models"][0]["id"] == f"{provider}-model"
        assert posted not in listed.text
        assert "does not verify inference" in listed.json()["note"]
    assert store.get("huggingface") == SECRET
    blocked = client.get("/credentials/huggingface/models", headers=_auth())
    assert blocked.status_code == 404
    assert SECRET not in blocked.text
    leaked = client.get("/credentials/openai/models?key=openai-query-secret", headers=_auth())
    assert leaked.status_code == 400
    assert "openai-query-secret" not in leaked.text
    removed = client.post("/credentials/anthropic/remove", headers=_auth())
    assert removed.status_code == 200
    assert store.get("anthropic") is None
    assert store.get("openai") == "openai-posted-secret"
    assert store.get("google") == "google-posted-secret"


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {CONTROLLER}", "Origin": "http://127.0.0.1:8810"}


def test_credential_routes_require_the_local_controller(store: MemoryStore, monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(app)
    denied = client.post("/credentials/huggingface", json={"token": SECRET})
    assert denied.status_code == 401
    assert SECRET not in denied.text
    assert store.items == {}

    foreign = client.post(
        "/credentials/huggingface",
        json={"token": SECRET},
        headers={"Authorization": f"Bearer {CONTROLLER}", "Origin": "https://evil.example"},
    )
    assert foreign.status_code == 403
    assert "access-control-allow-origin" not in {key.lower() for key in foreign.headers}
    assert store.items == {}

    leaked = client.post(f"/credentials/huggingface?token={SECRET}", headers=_auth(), json={"token": "other-value"})
    assert leaked.status_code == 400
    assert SECRET not in leaked.text
    assert store.items == {}

    saved = client.post("/credentials/huggingface", headers=_auth(), json={"token": SECRET})
    assert saved.status_code == 200
    assert saved.json()["status_label"] == "Configured"
    assert SECRET not in saved.text
    assert store.get("huggingface") == SECRET

    listed = client.get("/credentials", headers=_auth())
    assert listed.status_code == 200
    assert SECRET not in listed.text
    assert listed.json()["providers"][0]["status_label"] == "Configured"

    monkeypatch.setattr(credentials, "probe_huggingface", lambda token: "rejected" if token == SECRET else "ok")
    tested = client.post("/credentials/huggingface/test", headers=_auth())
    assert tested.status_code == 200
    assert tested.json()["status_label"] == "Invalid"
    assert store.get("huggingface") == SECRET
    assert SECRET not in tested.text

    removed = client.post("/credentials/huggingface/remove", headers=_auth())
    assert removed.status_code == 200
    assert removed.json()["status_label"] == "Not configured"
    assert store.get("huggingface") is None


def test_a_foreign_origin_cannot_read_the_controller_credential(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib

    from inspector_engine import console_host

    real = "controller-secret-SHOULD_NOT_LEAK_4411"
    token_path = tmp_path / "controller-token"
    token_path.write_text(real + "\n", encoding="utf-8")
    monkeypatch.setattr(console_host, "TOKEN_PATH", token_path)
    real_hash = hashlib.sha256(real.encode()).hexdigest()
    client = TestClient(app)
    foreign = client.get(
        "/controller-credential",
        headers={"Host": "127.0.0.1", "Origin": "https://evil.example"},
    )
    assert foreign.status_code == 403
    assert "access-control-allow-origin" not in {key.lower() for key in foreign.headers}
    assert hashlib.sha256(foreign.text.strip().encode()).hexdigest() != real_hash

    local = client.get(
        "/controller-credential",
        headers={"Host": "127.0.0.1", "Origin": "http://127.0.0.1:8810"},
    )
    assert local.status_code == 200
    assert hashlib.sha256(local.text.strip().encode()).hexdigest() == real_hash


def test_removing_the_keychain_token_leaves_the_environment_value(store: MemoryStore, monkeypatch: pytest.MonkeyPatch) -> None:
    store.put("huggingface", SECRET)
    monkeypatch.setenv("INSPECTOR_HF_TOKEN", "env-value-should-stay-hidden")
    removed = remove_credential("huggingface")
    assert store.get("huggingface") is None
    assert removed["source"] == "environment"
    assert removed["status_label"] == "Configured"
    assert "INSPECTOR_HF_TOKEN is still set" in removed["message"]
    assert SECRET not in json.dumps(removed)
    assert "env-value-should-stay-hidden" not in json.dumps(removed)
    assert read_token() == "env-value-should-stay-hidden"


def test_a_public_bind_refuses_credential_management(store: MemoryStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INSPECTOR_HOST", "0.0.0.0")
    client = TestClient(app)
    refused = client.post("/credentials/huggingface", headers=_auth(), json={"token": SECRET})
    assert refused.status_code == 403
    assert store.items == {}


def test_other_operating_systems_do_not_open_the_mac_keychain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(credentials.sys, "platform", "linux")
    credentials.set_store(None)
    try:
        assert isinstance(credentials.get_store(), credentials.UnavailableStore)
    finally:
        credentials.set_store(None)


def test_macos_keychain_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if sys.platform != "darwin":
        pytest.skip("macOS Keychain")
    security = ctypes.CDLL("/System/Library/Frameworks/Security.framework/Security")
    foundation = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    security.SecKeychainCreate.argtypes = [c_char_p, c_uint32, c_void_p, ctypes.c_bool, c_void_p, POINTER(c_void_p)]
    security.SecKeychainCreate.restype = c_int32
    security.SecKeychainDelete.argtypes = [c_void_p]
    security.SecKeychainDelete.restype = c_int32
    foundation.CFRelease.argtypes = [c_void_p]
    path = Path(tempfile.mktemp(prefix="aimi-test-", suffix=".keychain"))
    ref = c_void_p()
    password = b"aimi-test-keychain"
    created = security.SecKeychainCreate(str(path).encode(), len(password), password, False, None, byref(ref))
    assert created == 0 and ref.value
    keychain = KeychainStore(path, "aimi-test-keychain")
    monkeypatch.setattr(credentials, "STATUS_PATH", tmp_path / "credential-status.json")
    monkeypatch.delenv("INSPECTOR_HF_TOKEN", raising=False)
    credentials.set_store(keychain)
    try:
        save_credential("huggingface", SECRET)
        assert keychain.get("huggingface") == SECRET
        save_credential("huggingface", "replacement-token-value")
        assert keychain.get("huggingface") == "replacement-token-value"
        remove_credential("huggingface")
        assert keychain.get("huggingface") is None
        assert SECRET not in (tmp_path / "credential-status.json").read_text(encoding="utf-8")
    finally:
        keychain.close()
        security.SecKeychainDelete(ref)
        credentials.set_store(None)
