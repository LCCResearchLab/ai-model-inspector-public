# Authors: LCC Research Lab
# Project: From Document to Evidence
"""Keep credential values out of the API access log."""

from __future__ import annotations

import logging
import re

_QUERY_SECRET = re.compile(
    r"([?&](?:key|token|secret|password|access_token|api_key)=)[^&\s\"]*",
    re.IGNORECASE,
)


def redact_credential_query(text: str) -> str:
    """Replace credential query values. The parameter name stays so the refusal is visible."""
    return _QUERY_SECRET.sub(r"\1redacted", text)


class CredentialQueryFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple) and len(record.args) >= 3 and isinstance(record.args[2], str):
            args = list(record.args)
            args[2] = redact_credential_query(args[2])
            record.args = tuple(args)
        if isinstance(record.msg, str):
            record.msg = redact_credential_query(record.msg)
        return True


def install_access_redaction() -> None:
    logger = logging.getLogger("uvicorn.access")
    if any(isinstance(item, CredentialQueryFilter) for item in logger.filters):
        return
    logger.addFilter(CredentialQueryFilter())
