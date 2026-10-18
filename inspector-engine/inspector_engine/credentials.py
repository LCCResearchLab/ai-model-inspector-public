# Authors: LCC Research Lab
# Project: From Document to Evidence
"""Local API credentials for Model Library.

Hugging Face, OpenAI, Anthropic, and Google can be saved. A saved secret goes
to the macOS Keychain and is not returned to the browser. INSPECTOR_HF_TOKEN
overrides only the Hugging Face token. A cloud key is never taken from that
variable or from another provider's Keychain item.

This module does not read another application's files or keychain items.
"""

from __future__ import annotations

import ctypes
import hmac
import json
import os
import sys
from ctypes import POINTER, byref, c_char_p, c_int32, c_uint32, c_void_p
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from inspector_engine.server_controller import local_browser_origin

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_TOKEN = PROJECT_ROOT / ".inspector-runtime" / "controller-token"
STATUS_PATH = PROJECT_ROOT / ".inspector-runtime" / "credential-status.json"

SERVICE = "AI Model Inspector"
TOKEN_LIMIT = 512
WHOAMI_URL = "https://huggingface.co/api/whoami-v2"
NOT_FOUND = -25300
DUPLICATE = -25299

PROVIDERS: tuple[dict, ...] = (
    {"id": "huggingface", "label": "Hugging Face", "available": True},
    {"id": "openai", "label": "OpenAI", "available": True},
    {"id": "anthropic", "label": "Anthropic", "available": True},
    {"id": "google", "label": "Google Gemini", "available": True},
)

CLOUD_PROVIDERS = {"openai", "anthropic", "google"}

STATUS_LABELS = {
    "configured": "Configured",
    "validated": "Configured",
    "not_configured": "Not configured",
    "invalid": "Invalid",
}

_STORE: "CredentialStore | None" = None


class CredentialError(Exception):
    def __init__(self, code: str, message: str, status: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


class CredentialStore:
    def exists(self, provider: str) -> bool:
        raise NotImplementedError

    def get(self, provider: str) -> str | None:
        raise NotImplementedError

    def put(self, provider: str, secret: str) -> None:
        raise NotImplementedError

    def delete(self, provider: str) -> None:
        raise NotImplementedError


def get_store() -> CredentialStore:
    global _STORE
    if _STORE is None:
        _STORE = KeychainStore() if sys.platform == "darwin" else UnavailableStore()
    return _STORE


def set_store(store: CredentialStore | None) -> None:
    global _STORE
    _STORE = store


def mark_huggingface_validated() -> None:
    """Record that Hugging Face accepted the current token. The secret is not written here."""
    source = "environment" if _environment_token() else "keychain"
    _remember("huggingface", source, "validated")


def keychain_token(provider: str) -> str:
    """Saved token for Hub requests. Empty when none is saved or Keychain cannot be read."""
    try:
        secret = get_store().get(provider)
    except CredentialError:
        return ""
    return secret or ""


def huggingface_auth() -> dict:
    """Whether a Hugging Face token is absent, stored, or rejected. The secret is not included."""
    env = _environment_token()
    try:
        saved = bool(get_store().exists("huggingface"))
    except CredentialError:
        saved = False
    record = _load_status().get("huggingface")
    record = record if isinstance(record, dict) else {}
    if env:
        state = record.get("environment") if record.get("environment") in STATUS_LABELS else "configured"
        present = True
    elif saved:
        state = record.get("keychain") if record.get("keychain") in STATUS_LABELS else "configured"
        present = True
    else:
        state = "not_configured"
        present = False
    if not present or state == "not_configured":
        status = "none"
    elif state == "invalid":
        status = "invalid"
    else:
        status = "configured"
    labels = {"none": "No token", "configured": "Configured", "invalid": "Invalid"}
    return {
        "status": status,
        "label": labels[status],
        "present": present,
        "usable": status == "configured",
        "validated": state == "validated",
    }


def note_hub_refusal(token: str) -> str:
    """Classify a Hub refusal. A rejected token is marked Invalid and is not deleted."""
    if not token:
        return "missing"
    try:
        outcome = probe_huggingface(token)
    except CredentialError:
        return "unknown"
    active = _environment_token() or _saved_token("huggingface")
    if token != active:
        return "rejected" if outcome == "rejected" else "valid" if outcome == "ok" else "unknown"
    source = "environment" if _environment_token() else "keychain"
    if outcome == "rejected":
        _remember("huggingface", source, "invalid")
        return "rejected"
    if outcome == "ok":
        _remember("huggingface", source, "validated")
        return "valid"
    return "unknown"


def list_credentials() -> dict:
    return {
        "ok": True,
        "precedence": (
            "INSPECTOR_HF_TOKEN, when set for the API process, is used before a token saved in the macOS Keychain. "
            "A public model can be looked up without either one."
        ),
        "providers": [_public_provider(row) for row in PROVIDERS],
    }


def save_credential(provider: str, token: object) -> dict:
    row = _provider(provider)
    if not row["available"]:
        raise CredentialError(
            "provider_unavailable",
            f"{row['label']} can be added later. No key was stored.",
            400,
        )
    secret = _accept_token(token)
    get_store().put(provider, secret)
    status = _load_status()
    entry = status.setdefault(provider, {})
    entry["keychain"] = "configured"
    _save_status(status)
    saved = _public_provider(row)
    saved["ok"] = True
    return saved


def remove_credential(provider: str) -> dict:
    row = _provider(provider)
    if not row["available"]:
        raise CredentialError(
            "provider_unavailable",
            f"{row['label']} is not connected. Nothing was removed.",
            400,
        )
    get_store().delete(provider)
    status = _load_status()
    entry = status.get(provider)
    if isinstance(entry, dict):
        entry.pop("keychain", None)
        if not entry:
            status.pop(provider, None)
    _save_status(status)
    removed = _public_provider(row)
    removed["ok"] = True
    if provider == "huggingface" and _environment_token():
        removed["message"] = (
            "The Keychain token was removed. INSPECTOR_HF_TOKEN is still set for this API process, "
            "so Hugging Face requests continue to use that environment value. It is not shown."
        )
    return removed


def check_credential(provider: str, *, probe=None) -> dict:
    row = _provider(provider)
    if not row["available"]:
        raise CredentialError(
            "provider_unavailable",
            f"{row['label']} is not connected. Nothing was tested.",
            400,
        )
    huggingface = provider == "huggingface"
    env = _environment_token() if huggingface else ""
    saved = _saved_token(provider)
    secret = env or saved
    if not secret:
        missing = (
            "No Hugging Face token is configured. Nothing was tested."
            if huggingface
            else f"No {row['label']} API key is configured. Nothing was tested."
        )
        raise CredentialError("not_configured", missing, 400)
    source = "environment" if env else "keychain"
    check = probe or _probe_for(provider)
    try:
        outcome = check(secret)
    except CredentialError:
        raise
    except Exception as exc:
        message = scrub(str(exc), secret) or "The connection test did not finish."
        kept = "The saved token was not removed." if huggingface else "The key was not removed."
        raise CredentialError("connection_failed", f"{message} {kept}", 502) from None
    if outcome == "ok":
        _remember(provider, source, "validated")
        accepted = (
            "Hugging Face accepted the token. The token is not shown."
            if huggingface
            else f"{row['label']} accepted the key. The key is not shown."
        )
        viewed = _public_provider(row, message=accepted, secret=secret)
        viewed["ok"] = True
        return viewed
    if outcome == "rejected":
        _remember(provider, source, "invalid")
        rejected = (
            "Hugging Face did not accept this token. Nothing was removed."
            if huggingface
            else f"{row['label']} did not accept this key. Nothing was removed."
        )
        viewed = _public_provider(row, message=rejected, secret=secret)
        viewed["ok"] = True
        return viewed
    unfinished = (
        "The connection test did not finish. The saved token was not removed."
        if huggingface
        else "The connection test did not finish. The key was not removed."
    )
    raise CredentialError("connection_failed", unfinished, 502)


def list_cloud_models(provider: str) -> dict:
    """Read-only catalog for one cloud account. The key is not included."""
    row = _provider(provider)
    if provider not in CLOUD_PROVIDERS:
        raise CredentialError(
            "unknown_provider",
            "Model listing for that provider is not part of Cloud Models.",
            404,
        )
    secret = _saved_token(provider)
    if not secret:
        raise CredentialError(
            "not_configured",
            f"No {row['label']} API key is configured. Nothing was requested.",
            400,
        )
    from inspector_engine.cloud_catalog import fetch_catalog

    try:
        catalog = fetch_catalog(provider, secret)
    except CredentialError as exc:
        if exc.code == "rejected":
            _remember(provider, "keychain", "invalid")
        raise
    _remember(provider, "keychain", "validated")
    body = {
        "ok": True,
        "id": provider,
        "label": row["label"],
        "models": catalog["models"],
        "truncated": bool(catalog["truncated"]),
        "note": (
            "These models are exposed by the provider for this account. "
            "A listing does not verify inference, chat, training, or access to internal weights."
        ),
    }
    if secret in json.dumps(body):
        raise CredentialError(
            "connection_failed",
            f"{row['label']} returned a response that could not be shown. The key was not removed.",
            502,
        )
    return body


def _probe_for(provider: str):
    if provider == "huggingface":
        return probe_huggingface
    if provider in CLOUD_PROVIDERS:
        from inspector_engine.cloud_catalog import probe_cloud

        def cloud_probe(secret: str) -> str:
            return probe_cloud(provider, secret)

        return cloud_probe
    raise CredentialError("provider_unavailable", "That provider is not connected. Nothing was tested.", 400)


def probe_huggingface(token: str) -> str:
    """One authenticated request to Hugging Face. Returns ok or rejected."""
    request = Request(
        WHOAMI_URL,
        headers={
            "User-Agent": "ai-model-inspector",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="GET",
    )
    try:
        with build_opener(_RefuseRedirect()).open(request, timeout=20) as response:
            response.read(256)
            code = getattr(response, "status", None) or response.getcode()
    except HTTPError as exc:
        code = exc.code
        try:
            exc.close()
        except Exception:
            pass
    except URLError as exc:
        raise CredentialError(
            "connection_failed",
            "Hugging Face did not respond. The saved token was not removed.",
            502,
        ) from exc
    except CredentialError:
        raise
    except Exception:
        raise CredentialError(
            "connection_failed",
            "The connection test did not finish. The saved token was not removed.",
            502,
        ) from None
    if code == 200:
        return "ok"
    if code in {401, 403}:
        return "rejected"
    raise CredentialError(
        "connection_failed",
        f"Hugging Face returned HTTP {code}. The saved token was not removed.",
        502,
    )


def authorize(request) -> None:
    """Local controller credential, local browser, and a loopback bind."""
    if not loopback_process():
        raise CredentialError(
            "not_local",
            "Credential management is available only on this computer.",
            403,
        )
    client = request.client.host if request.client else ""
    host = (request.headers.get("host") or "").split(":")[0].lower()
    if client not in {"127.0.0.1", "::1", "testclient"} or host not in {"127.0.0.1", "localhost", "testserver"}:
        raise CredentialError(
            "not_local",
            "Credential management is available only on this computer.",
            403,
        )
    origin = request.headers.get("origin")
    if origin and not local_browser_origin(origin):
        raise CredentialError(
            "not_local",
            "Credential management is available only on this computer.",
            403,
        )
    query = (request.url.query or "").lower()
    if any(part in query for part in ("token=", "secret=", "password=", "key=")):
        raise CredentialError("rejected", "A credential cannot be sent in a URL.", 400)
    expected = _controller_token()
    header = request.headers.get("authorization") or ""
    if not expected or not _bearer(header, expected):
        raise CredentialError("unauthorized", "The local controller credential was rejected.", 401)


def loopback_process() -> bool:
    host = os.environ.get("INSPECTOR_HOST", "127.0.0.1").strip().lower()
    return host in {"", "127.0.0.1", "localhost", "::1"}


def scrub(text: str, secret: str) -> str:
    if not text:
        return ""
    if secret and secret in text:
        text = text.replace(secret, "")
    return " ".join(text.split())


def _public_provider(row: dict, message: str | None = None, secret: str = "") -> dict:
    provider = row["id"]
    env = _environment_token() if provider == "huggingface" else ""
    try:
        saved = bool(row["available"] and get_store().exists(provider))
    except CredentialError:
        saved = False
    record = _load_status().get(provider)
    record = record if isinstance(record, dict) else {}
    if provider == "huggingface" and env:
        source = "environment"
        state = record.get("environment") if record.get("environment") in STATUS_LABELS else "configured"
    elif saved:
        source = "keychain"
        state = record.get("keychain") if record.get("keychain") in STATUS_LABELS else "configured"
    else:
        source = None
        state = "not_configured"
    if state == "not_configured":
        source = None
    body = {
        "id": provider,
        "label": row["label"],
        "available": bool(row["available"]),
        "status": state,
        "status_label": STATUS_LABELS[state],
        "source": source,
        "saved": saved,
    }
    if message:
        body["message"] = scrub(message, secret)
    return body


def _provider(provider: str) -> dict:
    for row in PROVIDERS:
        if row["id"] == provider:
            return row
    raise CredentialError("unknown_provider", "That provider is not part of Model Library.", 404)


def _accept_token(token: object) -> str:
    if not isinstance(token, str):
        raise CredentialError("rejected", "The credential was not saved.", 400)
    secret = token.strip()
    if not secret or any(char in secret for char in ("\n", "\r", "\x00")):
        raise CredentialError("rejected", "The credential was not saved.", 400)
    if len(secret) > TOKEN_LIMIT:
        raise CredentialError("rejected", "The credential is too long. Nothing was saved.", 400)
    return secret


def _environment_token() -> str:
    return os.environ.get("INSPECTOR_HF_TOKEN", "").strip()


def _saved_token(provider: str) -> str:
    try:
        return get_store().get(provider) or ""
    except CredentialError:
        return ""


def _remember(provider: str, source: str, state: str) -> None:
    status = _load_status()
    entry = status.setdefault(provider, {})
    entry[source] = state
    _save_status(status)


def _load_status() -> dict:
    try:
        payload = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    clean: dict = {}
    for provider, entry in payload.items():
        if provider not in {row["id"] for row in PROVIDERS} or not isinstance(entry, dict):
            continue
        kept = {
            source: state
            for source, state in entry.items()
            if source in {"keychain", "environment"} and state in {"configured", "validated", "invalid"}
        }
        if kept:
            clean[provider] = kept
    return clean


def _save_status(status: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(STATUS_PATH)
    os.chmod(STATUS_PATH, 0o600)


def _controller_token() -> str:
    try:
        token = CONTROLLER_TOKEN.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if not token or len(token) > 200 or "\n" in token:
        return ""
    return token


def _bearer(header: str, expected: str) -> bool:
    if not header.startswith("Bearer ") or len(header) > 300:
        return False
    given = header[7:].strip()
    if not given or len(given) > 200:
        return False
    return hmac.compare_digest(given.encode("utf-8"), expected.encode("utf-8"))


class _RefuseRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise CredentialError(
            "connection_failed",
            "Hugging Face redirected the connection test. The saved token was not removed.",
            502,
        )


class UnavailableStore(CredentialStore):
    """Non-macOS computers keep using INSPECTOR_HF_TOKEN until a platform store exists."""

    def exists(self, provider: str) -> bool:
        raise self._unavailable()

    def get(self, provider: str) -> str | None:
        raise self._unavailable()

    def put(self, provider: str, secret: str) -> None:
        raise self._unavailable()

    def delete(self, provider: str) -> None:
        raise self._unavailable()

    def _unavailable(self) -> CredentialError:
        return CredentialError(
            "store_unavailable",
            "Saving a token needs the operating system credential store. "
            "This computer does not have one yet. Set INSPECTOR_HF_TOKEN for the API process instead. Nothing was stored.",
            500,
        )


class KeychainStore(CredentialStore):
    """Generic-password items in a macOS keychain. The default is the login keychain."""

    def __init__(self, path: Path | None = None, password: str | None = None) -> None:
        self._security, self._foundation = _frameworks()
        self._keychain: c_void_p | None = None
        if path is None:
            return
        ref = c_void_p()
        status = self._security.SecKeychainOpen(str(path).encode("utf-8"), byref(ref))
        if status != 0 or not ref.value:
            raise CredentialError("keychain_unavailable", "The macOS Keychain could not be opened.", 500)
        if password is not None:
            encoded = password.encode("utf-8")
            unlocked = self._security.SecKeychainUnlock(ref, len(encoded), encoded, True)
            if unlocked != 0:
                self._foundation.CFRelease(ref)
                raise CredentialError("keychain_unavailable", "The macOS Keychain could not be opened.", 500)
        self._keychain = ref

    def close(self) -> None:
        if self._keychain is not None and self._keychain.value:
            self._foundation.CFRelease(self._keychain)
        self._keychain = None

    def exists(self, provider: str) -> bool:
        return self.get(provider) is not None

    def get(self, provider: str) -> str | None:
        length = c_uint32()
        data = c_void_p()
        item = c_void_p()
        status = self._find(provider, length, data, item)
        try:
            if status == NOT_FOUND:
                return None
            if status != 0 or not data.value:
                raise CredentialError("keychain_unavailable", "The macOS Keychain could not be read.", 500)
            raw = ctypes.string_at(data.value, length.value)
            return raw.decode("utf-8")
        finally:
            if data.value:
                self._security.SecKeychainItemFreeContent(None, data)
            if item.value:
                self._foundation.CFRelease(item)

    def put(self, provider: str, secret: str) -> None:
        encoded = secret.encode("utf-8")
        service, account = _names(provider)
        item = c_void_p()
        status = self._security.SecKeychainAddGenericPassword(
            self._keychain,
            len(service),
            service,
            len(account),
            account,
            len(encoded),
            encoded,
            byref(item),
        )
        if status == 0:
            if item.value:
                self._foundation.CFRelease(item)
            return
        if status != DUPLICATE:
            raise CredentialError("keychain_unavailable", "The macOS Keychain did not store the credential.", 500)
        current = self.get(provider)
        if current is None:
            raise CredentialError("keychain_unavailable", "The macOS Keychain did not store the credential.", 500)
        length = c_uint32()
        data = c_void_p()
        found = c_void_p()
        found_status = self._find(provider, length, data, found)
        if data.value:
            self._security.SecKeychainItemFreeContent(None, data)
        if found_status != 0 or not found.value:
            raise CredentialError("keychain_unavailable", "The macOS Keychain did not store the credential.", 500)
        try:
            modified = self._security.SecKeychainItemModifyAttributesAndData(found, None, len(encoded), encoded)
        finally:
            self._foundation.CFRelease(found)
        if modified != 0:
            raise CredentialError("keychain_unavailable", "The macOS Keychain did not store the credential.", 500)

    def delete(self, provider: str) -> None:
        length = c_uint32()
        data = c_void_p()
        item = c_void_p()
        status = self._find(provider, length, data, item)
        if data.value:
            self._security.SecKeychainItemFreeContent(None, data)
        if status == NOT_FOUND:
            return
        if status != 0 or not item.value:
            raise CredentialError("keychain_unavailable", "The macOS Keychain could not remove the credential.", 500)
        try:
            removed = self._security.SecKeychainItemDelete(item)
        finally:
            self._foundation.CFRelease(item)
        if removed not in {0, NOT_FOUND}:
            raise CredentialError("keychain_unavailable", "The macOS Keychain could not remove the credential.", 500)

    def _find(self, provider: str, length: c_uint32, data: c_void_p, item: c_void_p) -> int:
        service, account = _names(provider)
        return self._security.SecKeychainFindGenericPassword(
            self._keychain,
            len(service),
            service,
            len(account),
            account,
            byref(length),
            byref(data),
            byref(item),
        )


def _names(provider: str) -> tuple[bytes, bytes]:
    if not provider.isascii() or not provider.replace("-", "").isalnum():
        raise CredentialError("unknown_provider", "That provider is not part of Model Library.", 404)
    return SERVICE.encode("utf-8"), provider.encode("utf-8")


def _frameworks():
    foundation = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    security = ctypes.CDLL("/System/Library/Frameworks/Security.framework/Security")
    foundation.CFRelease.argtypes = [c_void_p]
    foundation.CFRelease.restype = None
    security.SecKeychainOpen.argtypes = [c_char_p, POINTER(c_void_p)]
    security.SecKeychainOpen.restype = c_int32
    security.SecKeychainUnlock.argtypes = [c_void_p, c_uint32, c_void_p, ctypes.c_bool]
    security.SecKeychainUnlock.restype = c_int32
    security.SecKeychainAddGenericPassword.argtypes = [
        c_void_p,
        c_uint32,
        c_char_p,
        c_uint32,
        c_char_p,
        c_uint32,
        c_void_p,
        POINTER(c_void_p),
    ]
    security.SecKeychainAddGenericPassword.restype = c_int32
    security.SecKeychainFindGenericPassword.argtypes = [
        c_void_p,
        c_uint32,
        c_char_p,
        c_uint32,
        c_char_p,
        POINTER(c_uint32),
        POINTER(c_void_p),
        POINTER(c_void_p),
    ]
    security.SecKeychainFindGenericPassword.restype = c_int32
    security.SecKeychainItemModifyAttributesAndData.argtypes = [c_void_p, c_void_p, c_uint32, c_void_p]
    security.SecKeychainItemModifyAttributesAndData.restype = c_int32
    security.SecKeychainItemDelete.argtypes = [c_void_p]
    security.SecKeychainItemDelete.restype = c_int32
    security.SecKeychainItemFreeContent.argtypes = [c_void_p, c_void_p]
    security.SecKeychainItemFreeContent.restype = c_int32
    return security, foundation
