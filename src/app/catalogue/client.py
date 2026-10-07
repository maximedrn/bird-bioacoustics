"""Catalogue client services."""

from __future__ import annotations

from dataclasses import dataclass, field
from http import HTTPStatus
from http.client import IncompleteRead
from json import loads
from os import environ
from pathlib import Path
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.catalogue.transport import is_retryable_status
from app.domain.constants import (
    API_KEY_VARIABLE,
    USER_AGENT,
    XENO_CANTO_API_URL,
    FileName,
    MetadataKey,
    ResultColumn,
)
from app.domain.messages import ErrorMessage
from app.domain.settings import CorpusSettings


@dataclass(frozen=True, slots=True)
class XenoCantoClient:
    """Read API v3 pages without including the secret in repr or error
    messages.
    """

    api_key: str = field(repr=False)
    settings: CorpusSettings = field(default_factory=CorpusSettings)

    @staticmethod
    def local_key(root: Path) -> str:
        """Read an environment variable or one assignment in an ignored .env
        file.

        :param root: Project directory containing the optional .env file.
        :type root: Path
        :return: API key, without logging or persisting its value.
        :rtype: str
        """
        configured: str = environ.get(API_KEY_VARIABLE, "").strip()
        if configured:
            return configured
        path: Path = root / FileName.ENVIRONMENT
        if path.exists():
            for line_value in path.read_text().splitlines():
                line: str = line_value.strip()
                name, separator, value = line.partition("=")
                if separator and name.strip() == API_KEY_VARIABLE:
                    configured = value.strip().strip("\"'")
                    if configured:
                        return configured
        raise RuntimeError(ErrorMessage.MISSING_API_KEY)

    def page(self, query: str, page: int) -> dict[str, object]:
        """Retrieve one metadata page with retries and sanitized failures.

        :param query: Explicit tagged catalogue query.
        :type query: str
        :param page: One-based result page.
        :type page: int
        :return: Decoded API response.
        :rtype: dict[str, object]
        """
        parameters: dict[str, object] = {
            MetadataKey.QUERY: query,
            "page": page,
            "per_page": self.settings.per_page,
            "key": self.api_key,
        }
        request: Request = Request(
            XENO_CANTO_API_URL + "?" + urlencode(parameters),
            headers={"User-Agent": USER_AGENT},
        )
        for attempt in range(self.settings.request_attempts):
            try:
                with urlopen(request, timeout=90) as response:
                    payload: dict[str, object] = loads(response.read())
                if ResultColumn.RECORDINGS not in payload or not isinstance(
                    payload[ResultColumn.RECORDINGS], list
                ):
                    raise RuntimeError(ErrorMessage.INVALID_API_PAGE)
                sleep(self.settings.request_interval_seconds)
                return payload
            except HTTPError as error:
                if error.code in {
                    HTTPStatus.UNAUTHORIZED,
                    HTTPStatus.FORBIDDEN,
                }:
                    raise RuntimeError(
                        ErrorMessage.API_ACCESS_REFUSED.format(code=error.code)
                    ) from None
                if not is_retryable_status(error.code):
                    raise RuntimeError(
                        ErrorMessage.API_REQUEST_FAILED.format(code=error.code)
                    ) from None
            except (URLError, TimeoutError, OSError, IncompleteRead):
                pass
            if attempt + 1 < self.settings.request_attempts:
                sleep(min(2**attempt, 30))
        raise RuntimeError(
            ErrorMessage.API_RETRIES_EXHAUSTED.format(page=page)
        ) from None
