"""Domain settings services."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from numpy import isfinite

from app.domain.constants import FileName, InferenceDevice
from app.domain.messages import ErrorMessage


@dataclass(frozen=True, slots=True)
class CorpusSettings:
    """Configure one bounded recording batch and its catalogue requests."""

    query: str = "grp:birds"
    per_page: int = 500
    request_interval_seconds: float = 1.0
    request_attempts: int = 4
    batch_size: int = 100
    progress_interval: int = 100

    def __post_init__(self) -> None:
        """Reject invalid pagination, timing and batch size.

        :return: None.
        :rtype: None
        """
        if not self.query.strip() or not 50 <= self.per_page <= 500:
            raise ValueError(ErrorMessage.INVALID_CATALOGUE_SETTINGS)
        if self.request_interval_seconds < 0 or self.request_attempts < 1:
            raise ValueError(ErrorMessage.INVALID_REQUEST_SETTINGS)
        if self.batch_size < 1:
            raise ValueError(ErrorMessage.INVALID_BATCH_SIZE)
        if self.progress_interval < 1:
            raise ValueError(ErrorMessage.INVALID_PROGRESS_INTERVAL)


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Store immutable project paths."""

    root: Path
    data: Path
    results: Path
    generated_audio: Path

    @classmethod
    def from_working_directory(cls) -> Self:
        """Build project paths from the current working directory.

        :return: Project path configuration.
        :rtype: ProjectPaths
        """
        candidate: Path
        root: Path = Path.cwd().resolve()
        for candidate in (root, *root.parents):
            if (candidate / "pyproject.toml").is_file():
                return cls.from_root(candidate)
        raise FileNotFoundError(ErrorMessage.MISSING_PROJECT_ROOT)

    @classmethod
    def from_root(cls, root: Path) -> Self:
        """Resolve project paths using the configured database filename.

        :param root: Project directory independent of the working directory.
        :type root: Path
        :return: Immutable absolute project paths.
        :rtype: ProjectPaths
        """
        root = root.expanduser().resolve()
        data: Path = root / "data"
        return cls(
            root=root,
            data=data,
            results=data / "results",
            generated_audio=data / "generated_audio",
        )

    @property
    def checkpoint(self) -> Path:
        """Locate the durable experiment database.

        :return: Checkpoint filename in the configured data directory.
        :rtype: Path
        """
        return self.data / FileName.CHECKPOINT

    @property
    def annotations(self) -> Path:
        """Locate optional complete manual annotations.

        :return: Annotation CSV path.
        :rtype: Path
        """
        return self.root / "data" / FileName.ANNOTATIONS

    def create_directories(self) -> None:
        """Create all writable project directories.

        :return: None.
        :rtype: None
        """
        directory_value: Path
        directories: tuple[Path, ...] = (
            self.data,
            self.results,
            self.generated_audio,
        )
        for directory_value in directories:
            directory: Path = directory_value
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """Preserve the historical fingerprint of the BirdNET model.

    Override hardware controls through InferenceSettings when resuming.
    """

    family: Literal["acoustic"] = "acoustic"
    version: Literal["2.4", "3.0"] = "3.0"
    backend: Literal["tf", "pb", "pt", "onnx"] = "onnx"
    minimum_confidence: float = 0.1
    n_workers: int = 1
    batch_size: int = 16


@dataclass(frozen=True, slots=True)
class InferenceSettings:
    """Configure concurrent recordings outside the saved scientific protocol.

    Each recording owns one inference process and n_producers audio
    preparation processes. The batch size counts windows per model call.
    """

    device: InferenceDevice = InferenceDevice.CPU
    n_workers: int = 1
    n_producers: int = 1
    batch_size: int = 16

    def __post_init__(self) -> None:
        """Reject invalid worker counts and unsupported execution targets.

        :return: None.
        :rtype: None
        """
        counts: tuple[int, ...] = (
            self.n_workers,
            self.n_producers,
            self.batch_size,
        )
        if any(not self._positive_integer(count) for count in counts):
            raise ValueError(ErrorMessage.INVALID_INFERENCE_SETTINGS)
        if self.device not in InferenceDevice:
            raise ValueError(ErrorMessage.INVALID_INFERENCE_DEVICE)

    @staticmethod
    def _positive_integer(value: object) -> bool:
        """Validate integer counts at the untyped CLI or notebook boundary.

        :param value: Requested resource count.
        :type value: object
        :return: Whether the value is a positive integer other than a bool.
        :rtype: bool
        """
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
        )


@dataclass(frozen=True, slots=True)
class MixRatio:
    """Store one immutable pair of amplitude coefficients."""

    first: float
    second: float

    def __post_init__(self) -> None:
        """Validate amplitude coefficients.

        :return: None.
        :rtype: None
        """
        if self.first < 0.0 or self.second < 0.0:
            raise ValueError(ErrorMessage.NEGATIVE_MIX_WEIGHTS)
        if self.first + self.second == 0.0:
            raise ValueError(ErrorMessage.ZERO_MIX_WEIGHTS)

    @property
    def label(self) -> str:
        """Return a percentage-style label for the mix ratio.

        :return: Mix ratio label.
        :rtype: str
        """
        first_percent: int = round(self.first * 100.0)
        second_percent: int = round(self.second * 100.0)
        return f"{first_percent}/{second_percent}"


@dataclass(frozen=True, slots=True)
class ExperimentSettings:
    """Store immutable experiment parameters."""

    random_seed: int = 42
    noise_repetitions: int = 5
    snr_values_db: tuple[float, ...] = (20.0, 10.0, 5.0, 0.0)
    mix_ratios: tuple[MixRatio, ...] = (
        MixRatio(0.50, 0.50),
        MixRatio(0.75, 0.25),
        MixRatio(0.90, 0.10),
    )
    confidence_thresholds: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75)

    def __post_init__(self) -> None:
        """Validate experiment parameters.

        :return: None.
        :rtype: None
        """
        if self.random_seed < 0:
            raise ValueError(ErrorMessage.NEGATIVE_RANDOM_SEED)
        if self.noise_repetitions < 2:
            raise ValueError(ErrorMessage.INSUFFICIENT_NOISE_TRIALS)
        if not self.snr_values_db or not all(
            isfinite(value) for value in self.snr_values_db
        ):
            raise ValueError(ErrorMessage.INVALID_SNR_VALUES)
        if len(set(self.snr_values_db)) != len(self.snr_values_db):
            raise ValueError(ErrorMessage.DUPLICATE_SNR_VALUES)
        if not self.mix_ratios or not self.confidence_thresholds:
            raise ValueError(ErrorMessage.EMPTY_EXPERIMENT_CONDITIONS)
        invalid_thresholds: tuple[float, ...] = tuple(
            threshold_value
            for threshold_value in self.confidence_thresholds
            if not ModelSettings().minimum_confidence <= threshold_value <= 1.0
        )
        if invalid_thresholds:
            raise ValueError(ErrorMessage.INVALID_CONFIDENCE_THRESHOLDS)
