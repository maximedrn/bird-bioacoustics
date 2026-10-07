"""Domain models services."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Self

from pandas import DataFrame
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from app.domain.constants import Artifact
from app.domain.errors import AudioUnavailableError, RecordingError
from app.domain.messages import ErrorMessage
from app.domain.types import CorpusStatus, FloatArray


@dataclass(frozen=True, slots=True)
class AudioSource:
    """Store the immutable source audio description."""

    url: str
    filename: str
    recording_id: str
    scientific_name: str
    recordist: str
    country: str
    license_url: str
    background: str = "none"

    @classmethod
    def from_metadata(
        cls, identifier: str, metadata: dict[str, object]
    ) -> Self:
        """Validate external catalogue fields and reject unavailable download
        URLs.

        :param identifier: Stable XC recording identifier.
        :type identifier: str
        :param metadata: Cached Xeno-canto API fields.
        :type metadata: dict[str, object]
        :return: Safe source description compatible with stored catalogue rows.
        :rtype: Self
        """
        if not identifier.startswith("XC") or not identifier[2:].isdigit():
            raise RecordingError(ErrorMessage.INVALID_RECORDING_IDENTIFIER)
        try:
            recording: CatalogueRecording = CatalogueRecording.model_validate(
                metadata
            )
        except ValidationError:
            raise RecordingError(
                ErrorMessage.INVALID_CATALOGUE_METADATA.format(
                    identifier=identifier
                )
            ) from None
        url: str = (recording.file_url or "").strip()
        if url.startswith("//"):
            url = "https:" + url
        if not url.startswith("https://"):
            raise AudioUnavailableError(
                ErrorMessage.MISSING_AUDIO_URL.format(identifier=identifier)
            )
        extension: str = (
            Path(recording.file_name or "").suffix.lower() or ".audio"
        )
        return cls(
            url=url,
            filename=identifier + extension,
            recording_id=identifier,
            scientific_name=recording.scientific_name,
            recordist=recording.recordist,
            country=recording.country,
            license_url=recording.license_url,
            background=", ".join(recording.background),
        )


class FrozenValidationModel(BaseModel):
    """Provide strict immutable validation for experiment records."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
    )


class CatalogueRecording(FrozenValidationModel):
    """Validate the external metadata fields consumed by recording analysis."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="ignore"
    )
    genus: str = Field(alias="gen")
    species: str = Field(alias="sp")
    status: str = ""
    file_url: str | None = Field(default=None, alias="file")
    file_name: str | None = Field(default=None, alias="file-name")
    recordist: str = Field(default="", alias="rec")
    country: str = Field(default="", alias="cnt")
    license_url: str = Field(default="", alias="lic")
    background: list[str] = Field(default_factory=list, alias="also")

    @property
    def scientific_name(self) -> str:
        """Build the Latin name used to look up the BirdNET label.

        :return: Genus and species name.
        :rtype: str
        """
        return f"{self.genus} {self.species}".strip()


class EnvironmentEntry(FrozenValidationModel):
    """Represent one software component and its version."""

    component: str = Field(min_length=1)
    version: str = Field(min_length=1)


class DetectionCandidate(FrozenValidationModel):
    """Represent one validated BirdNET detection candidate."""

    species_name: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        """Validate the candidate time interval.

        :return: Validated detection candidate.
        :rtype: DetectionCandidate
        """
        if self.end_seconds <= self.start_seconds:
            raise ValueError(ErrorMessage.INVALID_DETECTION_INTERVAL)
        return self


class NoiseResult(FrozenValidationModel):
    """Represent one result from the noise robustness experiment."""

    recording_id: str = Field(min_length=1)
    trial: int = Field(ge=0)
    seed: int | None = Field(default=None, ge=0)
    condition: str = Field(min_length=1)
    snr_db: float
    target_species: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    rank: int | None = Field(default=None, ge=1)


class OverlapResult(FrozenValidationModel):
    """Represent one result from the overlap experiment."""

    pair_id: str = Field(min_length=1)
    recording_a: str = Field(min_length=1)
    recording_b: str = Field(min_length=1)
    mix: str = Field(min_length=1)
    weight_a: float = Field(ge=0.0)
    weight_b: float = Field(ge=0.0)
    species_a: str = Field(min_length=1)
    confidence_a: float = Field(ge=0.0, le=1.0)
    species_b: str = Field(min_length=1)
    confidence_b: float = Field(ge=0.0, le=1.0)


class ThresholdResult(FrozenValidationModel):
    """Represent one result from the confidence-threshold experiment."""

    threshold: float = Field(ge=0.0, le=1.0)
    detections: int = Field(ge=0)
    distinct_species: int = Field(ge=0)
    recordings: int = Field(ge=1)
    target_detection_rate: float = Field(ge=0.0, le=1.0)
    annotated_recordings: int = Field(ge=0)
    true_positives: int | None = Field(default=None, ge=0)
    false_positives: int | None = Field(default=None, ge=0)
    false_negatives: int | None = Field(default=None, ge=0)
    precision: float | None = Field(default=None, ge=0.0, le=1.0)
    recall: float | None = Field(default=None, ge=0.0, le=1.0)
    f1_score: float | None = Field(default=None, ge=0.0, le=1.0)


@dataclass(frozen=True, slots=True)
class NoiseExperimentOutput:
    """Store outputs required by later notebook sections."""

    recording_id: str
    target: DetectionCandidate
    sample_rate: int
    target_segment: FloatArray
    target_segment_path: Path
    results: DataFrame


@dataclass(frozen=True, slots=True)
class OverlapExperimentOutput:
    """Store outputs from the overlap experiment."""

    species_a: DetectionCandidate
    species_b: DetectionCandidate
    segment_a: FloatArray
    segment_b: FloatArray
    segment_a_path: Path
    segment_b_path: Path
    results: DataFrame


@dataclass(frozen=True, slots=True)
class RecordingInput:
    """Associate a downloaded recording with its metadata target."""

    source: AudioSource
    path: Path
    target_species: str


class ReferenceAnnotation(FrozenValidationModel):
    """Represent a species label from a fully annotated recording."""

    recording_id: str = Field(min_length=1)
    species_name: str
    complete: bool

    @model_validator(mode="after")
    def validate_coverage(self) -> Self:
        """Reject partial labels as a source of precision and recall.

        :return: Validated reference annotation.
        :rtype: ReferenceAnnotation
        """
        if not self.complete:
            raise ValueError(ErrorMessage.INCOMPLETE_REFERENCE)
        return self


@dataclass(frozen=True, slots=True)
class RecordingBatch:
    """Describe one durable recording interval, using a zero-based processing
    cursor.
    """

    batch_id: int
    start_cursor: int
    end_cursor: int

    @property
    def size(self) -> int:
        """Return the number of recording attempts planned for this batch.

        :return: Batch size, including a possible shorter final batch.
        :rtype: int
        """
        return self.end_cursor - self.start_cursor


@dataclass(frozen=True, slots=True)
class ResultDetails:
    """Keep compact diagnostic tables from the same cumulative snapshot."""

    noise_distribution: DataFrame
    species_noise: DataFrame
    noise_pairs: DataFrame
    overlap_outcomes: DataFrame
    species_coverage: DataFrame
    duration_confidence: DataFrame
    batch_history: DataFrame

    def tables(self) -> dict[Artifact, DataFrame]:
        """Map diagnostic figures to their exported measurements.

        :return: Stable artifact names and their corresponding tables.
        :rtype: dict[Artifact, DataFrame]
        """
        return {
            Artifact.NOISE_DISTRIBUTION: self.noise_distribution,
            Artifact.SPECIES_NOISE: self.species_noise,
            Artifact.NOISE_PAIRS: self.noise_pairs,
            Artifact.OVERLAP_OUTCOMES: self.overlap_outcomes,
            Artifact.SPECIES_COVERAGE: self.species_coverage,
            Artifact.DURATION_CONFIDENCE: self.duration_confidence,
            Artifact.BATCH_HISTORY: self.batch_history,
        }


@dataclass(frozen=True, slots=True)
class ResultSnapshot:
    """Keep summaries, coverage and preview from one consistent database
    snapshot.
    """

    noise: DataFrame
    overlap: DataFrame
    thresholds: DataFrame
    status: CorpusStatus
    skipped_noise: int
    preview: NoiseExperimentOutput | None
    details: ResultDetails
