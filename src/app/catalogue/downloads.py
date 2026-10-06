"""Download one recording with bounded retries and atomic local replacement."""

from __future__ import annotations

from http.client import HTTPResponse, IncompleteRead
from pathlib import Path
from shutil import copyfileobj
from time import sleep
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.catalogue.transport import is_retryable_status
from app.domain.constants import USER_AGENT
from app.domain.errors import (
    DownloadTransientError,
    DownloadUnavailableError,
)
from app.domain.messages import ErrorMessage
from app.domain.models import AudioSource
from app.domain.settings import CorpusSettings, ProjectPaths


class ExperimentRepository:
    """Manage current-recording downloads without keeping a local corpus."""

    def __init__(
        self, paths: ProjectPaths, settings: CorpusSettings | None = None
    ) -> None:
        """Configure paths and network retry limits.

        :param paths: Project directories.
        :type paths: ProjectPaths
        :param settings: Catalogue and download retry settings.
        :type settings: CorpusSettings | None
        :return: None.
        :rtype: None
        """
        self._paths: ProjectPaths = paths
        self._settings: CorpusSettings = settings or CorpusSettings()

    def download(self, source: AudioSource) -> Path:
        """Retry transient failures without exposing URLs or partial audio
        files.

        :param source: Validated recording source.
        :type source: AudioSource
        :return: Complete local original audio.
        :rtype: Path
        """
        attempt: int
        error: HTTPError
        destination: Path = self._paths.data / source.filename
        if destination.is_file():
            return destination
        for attempt in range(self._settings.request_attempts):
            try:
                return self._download(source, destination)
            except HTTPError as error:
                if not is_retryable_status(error.code):
                    raise DownloadUnavailableError(
                        ErrorMessage.AUDIO_DOWNLOAD_UNAVAILABLE.format(
                            recording_id=source.recording_id, code=error.code
                        )
                    ) from None
            except (
                URLError,
                TimeoutError,
                IncompleteRead,
                DownloadTransientError,
            ):
                pass
            if attempt + 1 < self._settings.request_attempts:
                sleep(min(2**attempt, 30))
        raise DownloadTransientError(
            ErrorMessage.DOWNLOAD_RETRIES_EXHAUSTED.format(
                recording_id=source.recording_id
            )
        ) from None

    @staticmethod
    def _download(source: AudioSource, destination: Path) -> Path:
        """Write a complete response atomically and verify its declared byte
        count.

        :param source: Recording source and credits.
        :type source: AudioSource
        :param destination: Original audio destination.
        :type destination: Path
        :return: Completed original file.
        :rtype: Path
        """
        response: HTTPResponse
        stream: BinaryIO
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path = destination.with_suffix(destination.suffix + ".part")
        request: Request = Request(
            source.url, headers={"User-Agent": USER_AGENT}
        )
        try:
            with urlopen(request, timeout=90) as response:
                content_type: str = response.headers.get_content_type()
                if (
                    not content_type.startswith("audio/")
                    and content_type != "application/octet-stream"
                ):
                    raise DownloadUnavailableError(
                        ErrorMessage.UNEXPECTED_AUDIO_CONTENT.format(
                            recording_id=source.recording_id,
                            content_type=content_type,
                        )
                    )
                with temporary.open("wb") as stream:
                    copyfileobj(response, stream)
                size: int = temporary.stat().st_size
                expected: str | None = response.headers.get("Content-Length")
                if size == 0 or (
                    expected is not None and size != int(expected)
                ):
                    raise DownloadTransientError(
                        ErrorMessage.INCOMPLETE_AUDIO_RESPONSE.format(
                            recording_id=source.recording_id
                        )
                    )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    @staticmethod
    def portable_path(path: Path) -> str:
        """Render a project-relative path when possible.

        :param path: Filesystem path.
        :type path: Path
        :return: Relative path or basename.
        :rtype: str
        """
        try:
            return str(path.resolve().relative_to(Path.cwd().resolve()))
        except ValueError:
            return path.name
