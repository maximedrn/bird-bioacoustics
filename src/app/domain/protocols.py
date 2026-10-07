"""Domain protocols services."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self

from app.domain.settings import CorpusSettings


class CsvWritable(Protocol):
    """Describe a prediction object that can be exported to CSV."""

    def to_csv(self, path: str, *, silent: bool = True) -> object:
        """Export predictions to a CSV file.

        :param path: Destination CSV path.
        :type path: str
        :param silent: Suppress per-file CSV progress output.
        :type silent: bool
        :return: Backend-specific export result.
        :rtype: object
        """


class PredictionSession(Protocol):
    """Describe the session operations used by recording analysis."""

    def run(self, paths: tuple[str, ...]) -> CsvWritable:
        """Analyze a group of paths and return exportable predictions.

        :param paths: Complete decoded recordings or generated variants.
        :type paths: tuple[str, ...]
        :return: Predictions exposing CSV export.
        :rtype: CsvWritable
        """
        raise NotImplementedError

    def cancel(self) -> None:
        """Cancel active queues before a failed session is closed.

        :return: None.
        :rtype: None
        """

    def __enter__(self) -> Self:
        """Enter the reusable inference session.

        :return: Active session.
        :rtype: Self
        """
        raise NotImplementedError

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close worker resources after success, failure or interruption.

        :param exception_type: Raised exception type.
        :type exception_type: type[BaseException] | None
        :param exception: Raised exception instance.
        :type exception: BaseException | None
        :param traceback: Exception traceback.
        :type traceback: TracebackType | None
        :return: None.
        :rtype: None
        """


class CatalogueClient(Protocol):
    """Describe metadata pagination independently of its HTTP client."""

    @property
    def settings(self) -> CorpusSettings:
        """Read the immutable query and transport configuration.

        :return: Catalogue configuration.
        :rtype: CorpusSettings
        """
        raise NotImplementedError

    def page(self, query: str, page: int) -> dict[str, object]:
        """Retrieve one complete catalogue page.

        :param query: Frozen catalogue query.
        :type query: str
        :param page: One-based page number.
        :type page: int
        :return: Metadata response.
        :rtype: dict[str, object]
        """
        raise NotImplementedError


class BirdNetModelProtocol(Protocol):
    """Describe the BirdNET model methods used by this notebook."""

    @property
    def species_list(self) -> Collection[str]:
        """Read the model's supported species labels.

        :return: Model species labels.
        :rtype: Collection[str]
        """
        raise NotImplementedError

    @property
    def model_path(self) -> Path:
        """Locate the cached model weights downloaded by BirdNET.

        :return: Local model file.
        :rtype: Path
        """
        raise NotImplementedError

    def predict(
        self,
        audio_paths: str | tuple[str, ...],
        *,
        top_k: int | None = None,
        n_workers: int = 1,
        default_confidence_threshold: float = 0.1,
    ) -> CsvWritable:
        """Run BirdNET inference on one or more audio files.

        :param audio_paths: Input audio paths.
        :type audio_paths: str | tuple[str, ...]
        :param top_k: Maximum species per window, or no limit.
        :type top_k: int | None
        :param n_workers: Number of inference workers.
        :type n_workers: int
        :param default_confidence_threshold: Minimum exported confidence.
        :type default_confidence_threshold: float
        :return: Prediction object exposing CSV export.
        :rtype: CsvWritable
        """
        raise NotImplementedError

    def predict_session(
        self,
        *,
        top_k: int | None,
        n_workers: int,
        n_producers: int,
        device: str,
        batch_size: int,
        max_n_files: int,
        default_confidence_threshold: float,
    ) -> PredictionSession:
        """Create reusable inference workers for the sequential corpus.

        :param top_k: Species limit per window, or None.
        :type top_k: int | None
        :param n_workers: Inference worker count.
        :type n_workers: int
        :param n_producers: Audio preparation process count.
        :type n_producers: int
        :param device: CPU or GPU execution target.
        :type device: str
        :param batch_size: Audio windows per inference batch.
        :type batch_size: int
        :param max_n_files: Largest temporary file batch.
        :type max_n_files: int
        :param default_confidence_threshold: Minimum exported score.
        :type default_confidence_threshold: float
        :return: Context-managed reusable BirdNET session.
        :rtype: PredictionSession
        """
        raise NotImplementedError
