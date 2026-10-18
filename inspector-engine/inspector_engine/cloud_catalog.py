# Authors: LCC Research Lab
# Project: From Document to Evidence
"""Read-only model catalogs for OpenAI, Anthropic, and Google.

Each request sends the key in a header. The key is never placed in a URL.
This module does not call inference, chat, or training endpoints.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from inspector_engine.credentials import CredentialError, scrub

PAGE_LIMIT = 4
MODEL_LIMIT = 200
READ_LIMIT = 1_500_000
ANTHROPIC_VERSION = "2023-06-01"

_HOSTS = {
    "openai": "api.openai.com",
    "anthropic": "api.anthropic.com",
    "google": "generativelanguage.googleapis.com",
}

_LABELS = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google Gemini",
}


def probe_cloud(provider: str, secret: str, *, transport=None) -> str:
    """One authenticated catalog request. Returns ok or rejected."""
    outcome, _body = _exchange(provider, secret, _first_url(provider), transport)
    if outcome == "ok":
        return "ok"
    if outcome == "rejected":
        return "rejected"
    raise CredentialError("connection_failed", f"{_LABELS[provider]} did not respond. The key was not removed.", 502)


def fetch_catalog(provider: str, secret: str, *, transport=None) -> dict:
    """Models exposed by this account. Listing does not verify later operations."""
    if provider not in _HOSTS:
        raise CredentialError("unknown_provider", "That provider is not part of Cloud Models.", 404)
    models: list[dict] = []
    truncated = False
    url = _first_url(provider)
    for _page in range(PAGE_LIMIT):
        outcome, body = _exchange(provider, secret, url, transport)
        if outcome == "rejected":
            raise CredentialError(
                "rejected",
                f"{_LABELS[provider]} did not accept this key. Nothing was removed.",
                401,
            )
        if outcome != "ok" or body is None:
            raise CredentialError(
                "connection_failed",
                f"{_LABELS[provider]} returned a response that could not be read. The key was not removed.",
                502,
            )
        payload = _json_object(provider, body, secret)
        page, nxt = _parse_page(provider, payload, secret)
        for item in page:
            if len(models) >= MODEL_LIMIT:
                truncated = True
                break
            models.append(item)
        if truncated or not nxt:
            if nxt and len(models) >= MODEL_LIMIT:
                truncated = True
            break
        url = nxt
    else:
        truncated = True
    return {"models": models, "truncated": truncated}


def _first_url(provider: str) -> str:
    if provider == "openai":
        return "https://api.openai.com/v1/models"
    if provider == "anthropic":
        return "https://api.anthropic.com/v1/models?limit=100"
    if provider == "google":
        return "https://generativelanguage.googleapis.com/v1beta/models?pageSize=50"
    raise CredentialError("unknown_provider", "That provider is not part of Cloud Models.", 404)


def _exchange(provider: str, secret: str, url: str, transport) -> tuple[str, bytes | None]:
    label = _LABELS[provider]
    try:
        if transport is not None:
            code, body = transport(provider, secret, url)
        else:
            code, body = _http_get(provider, secret, url)
    except CredentialError:
        raise
    except Exception as exc:
        message = scrub(str(exc), secret) or f"{label} did not respond."
        raise CredentialError("connection_failed", f"{message} The key was not removed.", 502) from None
    if not isinstance(code, int):
        raise CredentialError("connection_failed", f"{label} did not respond. The key was not removed.", 502)
    raw = body if isinstance(body, bytes) else b""
    return _outcome(provider, code, raw, secret), raw


def _outcome(provider: str, code: int, body: bytes, secret: str) -> str:
    label = _LABELS[provider]
    if code == 200:
        return "ok"
    if code == 401 or (code in {400, 403} and _key_rejected(body, secret)):
        return "rejected"
    if code == 403:
        raise CredentialError(
            "permission_denied",
            f"{label} recognized the key but did not allow a model list. The key was not removed.",
            403,
        )
    if code == 429:
        raise CredentialError(
            "rate_limited",
            f"{label} limited this request. Wait and try again. The key was not removed.",
            429,
        )
    raise CredentialError(
        "connection_failed",
        f"{label} returned HTTP {code}. The key was not removed.",
        502,
    )


def _key_rejected(body: bytes, secret: str) -> bool:
    text = scrub(body.decode("utf-8", errors="replace")[:2000], secret).lower()
    markers = (
        "api key not valid",
        "api_key_invalid",
        "invalid_api_key",
        "invalid api key",
        "incorrect api key",
        "invalid x-api-key",
    )
    return any(marker in text for marker in markers)


def _http_get(provider: str, secret: str, url: str) -> tuple[int, bytes]:
    label = _LABELS[provider]
    _require_safe_url(provider, secret, url)
    request = Request(url, headers=_headers(provider, secret), method="GET")
    try:
        with build_opener(_RefuseRedirect(label)).open(request, timeout=20) as response:
            body = _read_limited(response, label)
            code = getattr(response, "status", None) or response.getcode()
    except HTTPError as exc:
        code = exc.code
        try:
            body = exc.read(8192)
        except Exception:
            body = b""
        try:
            exc.close()
        except Exception:
            pass
    except URLError as exc:
        message = scrub(str(exc.reason), secret) or f"{label} did not respond."
        raise CredentialError("connection_failed", f"{message} The key was not removed.", 502) from None
    except CredentialError:
        raise
    except Exception as exc:
        message = scrub(str(exc), secret) or f"{label} did not respond."
        raise CredentialError("connection_failed", f"{message} The key was not removed.", 502) from None
    return int(code), body


def _headers(provider: str, secret: str) -> dict[str, str]:
    headers = {"User-Agent": "ai-model-inspector", "Accept": "application/json"}
    if provider == "openai":
        headers["Authorization"] = f"Bearer {secret}"
    elif provider == "anthropic":
        headers["x-api-key"] = secret
        headers["anthropic-version"] = ANTHROPIC_VERSION
    elif provider == "google":
        headers["x-goog-api-key"] = secret
    return headers


def _require_safe_url(provider: str, secret: str, url: str) -> None:
    label = _LABELS[provider]
    parsed = urlsplit(url)
    names = {name.lower() for name, _value in parse_qsl(parsed.query, keep_blank_values=True)}
    if (
        parsed.scheme != "https"
        or parsed.hostname != _HOSTS[provider]
        or parsed.username
        or parsed.password
        or secret and secret in url
        or names & {"key", "token", "secret"}
    ):
        raise CredentialError(
            "connection_failed",
            f"{label} model list was stopped before it was sent. The key was not removed.",
            502,
        )


def _read_limited(response, label: str) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while total < READ_LIMIT:
        block = response.read(min(65536, READ_LIMIT - total))
        if not block:
            return b"".join(chunks)
        chunks.append(block)
        total += len(block)
    if response.read(1):
        raise CredentialError(
            "connection_failed",
            f"{label} returned a response that was too large. The key was not removed.",
            502,
        )
    return b"".join(chunks)


def _json_object(provider: str, body: bytes, secret: str) -> dict:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise CredentialError(
            "connection_failed",
            f"{_LABELS[provider]} returned a response that could not be read. The key was not removed.",
            502,
        ) from None
    if not isinstance(payload, dict):
        raise CredentialError(
            "connection_failed",
            f"{_LABELS[provider]} returned a response that could not be read. The key was not removed.",
            502,
        )
    if secret and secret in json.dumps(payload):
        try:
            payload = json.loads(scrub(json.dumps(payload), secret) or "{}")
        except json.JSONDecodeError:
            raise CredentialError(
                "connection_failed",
                f"{_LABELS[provider]} returned a response that could not be shown. The key was not removed.",
                502,
            ) from None
        if not isinstance(payload, dict):
            payload = {}
    return payload


def _parse_page(provider: str, payload: dict, secret: str) -> tuple[list[dict], str | None]:
    if provider == "openai":
        rows = _rows(provider, payload, "data")
        return [model for model in (_openai_model(row, secret) for row in rows) if model], None
    if provider == "anthropic":
        rows = _rows(provider, payload, "data")
        nxt = _anthropic_next(payload.get("last_id"), secret) if payload.get("has_more") is True else None
        return [model for model in (_anthropic_model(row, secret) for row in rows) if model], nxt
    rows = _rows(provider, payload, "models")
    token = _page_token(payload.get("nextPageToken"), secret)
    nxt = None
    if token:
        nxt = (
            "https://generativelanguage.googleapis.com/v1beta/models"
            f"?pageSize=50&pageToken={quote(token, safe='')}"
        )
    return [model for model in (_google_model(row, secret) for row in rows) if model], nxt


def _rows(provider: str, payload: dict, key: str) -> list:
    rows = payload.get(key)
    if not isinstance(rows, list):
        raise CredentialError(
            "connection_failed",
            f"{_LABELS[provider]} returned a response that could not be read. The key was not removed.",
            502,
        )
    return rows


def _openai_model(row: object, secret: str) -> dict | None:
    if not isinstance(row, dict):
        return None
    model_id = _text(row.get("id"), secret, 200)
    if not model_id:
        return None
    owned = _text(row.get("owned_by"), secret, 80)
    detail = f"Owned by {owned}" if owned else None
    facts = _facts(
        ("Owner", owned),
        ("Created", _utc_stamp(row.get("created"))),
        ("Shutdown date", _text(row.get("shutdown_date"), secret, 40)),
    )
    return {"id": model_id, "display_name": None, "detail": detail, "facts": facts}


def _anthropic_model(row: object, secret: str) -> dict | None:
    if not isinstance(row, dict):
        return None
    model_id = _text(row.get("id"), secret, 200)
    if not model_id:
        return None
    created = _text(row.get("created_at"), secret, 40)
    kind = _text(row.get("type"), secret, 40)
    return {
        "id": model_id,
        "display_name": _text(row.get("display_name"), secret, 120),
        "detail": f"Provider-reported created {created}" if created else None,
        "facts": _facts(("Created", created), ("Type", kind)),
    }


def _google_model(row: object, secret: str) -> dict | None:
    if not isinstance(row, dict):
        return None
    name = _text(row.get("name"), secret, 200)
    if not name:
        return None
    model_id = name[7:] if name.startswith("models/") else name
    if not model_id or (secret and secret in model_id):
        return None
    methods = row.get("supportedGenerationMethods")
    reported: list[str] = []
    if isinstance(methods, list):
        for method in methods:
            text = _text(method, secret, 40)
            if text:
                reported.append(text)
            if len(reported) == 6:
                break
    detail = f"Provider-reported methods: {', '.join(reported)}" if reported else None
    return {
        "id": model_id,
        "display_name": _text(row.get("displayName"), secret, 120),
        "detail": detail,
        "facts": _facts(
            ("Input token limit", _count(row.get("inputTokenLimit"))),
            ("Output token limit", _count(row.get("outputTokenLimit"))),
            ("Supported methods", ", ".join(reported) if reported else None),
        ),
    }


def _anthropic_next(last_id: object, secret: str) -> str | None:
    token = _page_token(last_id, secret)
    if not token:
        return None
    return f"https://api.anthropic.com/v1/models?limit=100&after_id={quote(token, safe='')}"


def _page_token(value: object, secret: str) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip()
    if not token or len(token) > 500 or any(char in token for char in ("\n", "\r", "\x00", " ")):
        return None
    if secret and secret in token:
        return None
    return token


def _facts(*pairs: tuple[str, str | None]) -> list[dict[str, str]]:
    return [{"name": name, "value": value} for name, value in pairs if value]


def _count(value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 1 or value > 100_000_000:
        return None
    return str(value)


def _utc_stamp(value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 946_684_800 or value > 4_102_444_800:
        return None
    return datetime.fromtimestamp(value, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _text(value: object, secret: str, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if not text or len(text) > limit or any(char in text for char in ("\n", "\r", "\x00")):
        return None
    if secret and secret in text:
        return None
    return text


class _RefuseRedirect(HTTPRedirectHandler):
    def __init__(self, label: str) -> None:
        super().__init__()
        self._label = label

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise CredentialError(
            "connection_failed",
            f"{self._label} redirected the request. The key was not removed.",
            502,
        )
