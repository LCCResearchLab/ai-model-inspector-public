# Authors: LCC Research Lab
# Project: From Document to Evidence

import logging

from inspector_engine.access_log import CredentialQueryFilter, install_access_redaction, redact_credential_query
from server import refuse_public_bind


def test_a_rejected_credential_query_is_redacted() -> None:
    secret = "sk-test-value-SHOULD_NOT_LEAK"
    raw = f'127.0.0.1:1 - "GET /credentials/openai/models?key={secret} HTTP/1.1" 400'
    assert secret not in redact_credential_query(raw)
    assert "key=redacted" in redact_credential_query(raw)
    record = logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:1", "GET", f"/credentials/openai/models?key={secret}", "1.1", 400),
        exc_info=None,
    )
    assert CredentialQueryFilter().filter(record) is True
    assert secret not in record.getMessage()
    assert "key=redacted" in record.getMessage()


def test_page_tokens_and_ordinary_paths_stay_in_the_log() -> None:
    text = "/models?pageSize=50&pageToken=next-page"
    assert redact_credential_query(text) == text


def test_the_access_filter_is_installed_once() -> None:
    install_access_redaction()
    install_access_redaction()
    logger = logging.getLogger("uvicorn.access")
    assert sum(isinstance(item, CredentialQueryFilter) for item in logger.filters) == 1


def test_a_public_uvicorn_host_is_refused() -> None:
    try:
        refuse_public_bind(["server:app", "--host", "0.0.0.0", "--port", "8810"])
    except SystemExit as exc:
        assert "127.0.0.1" in str(exc)
    else:
        raise AssertionError("a public host was accepted")
    refuse_public_bind(["server:app", "--host", "127.0.0.1", "--port", "8810"])
    refuse_public_bind(["pytest"])
