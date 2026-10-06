"""Analyze one recording independently of batch cursors and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import file_digest
from pathlib import Path
from time import sleep
from typing import BinaryIO

from pandas import DataFrame
from soundfile import info as audio_info

from app.audio.prediction import BirdNetPredictor
from app.audio.processing import AudioProcessor
from app.catalogue.downloads import ExperimentRepository
from app.domain.constants import Artifact, CatalogueStatus
from app.domain.errors import AudioDecodingError
from app.domain.messages import ErrorMessage, ExperimentMessage
from app.domain.models import (
    AudioSource,
    CatalogueRecording,
    NoiseExperimentOutput,
)
from app.domain.settings import CorpusSettings
from app.experiments.noise import NoiseRobustnessExperiment


@dataclass(frozen=True, slots=True)
class RecordingMeasurements:
    """Commit a recording's measurements in one storage transaction."""

    baseline: DataFrame
    target_species: str | None
    noise: NoiseExperimentOutput | None
    noise_skip_reason: str | None
    download_bytes: int
    duration_seconds: float
    fingerprint: str


class RecordingAnalysisService:
    """Download, decode and measure one original while keeping bounded disk
    use.
    """

    def __init__(
        self,
        repository: ExperimentRepository,
        processor: AudioProcessor,
        predictor: BirdNetPredictor,
        noise: NoiseRobustnessExperiment,
        species_lookup: dict[str, str],
        workspace: Path,
        settings: CorpusSettings,
    ) -> None:
        """Configure recording analysis without owning the checkpoint.

        :param repository: Download service.
        :type repository: ExperimentRepository
        :param processor: Bounded audio processor.
        :type processor: AudioProcessor
        :param predictor: Active reusable inference session.
        :type predictor: BirdNetPredictor
        :param noise: Scientific noise experiment.
        :type noise: NoiseRobustnessExperiment
        :param species_lookup: Latin names mapped to supported model labels.
        :type species_lookup: dict[str, str]
        :param workspace: Temporary decoded-audio directory.
        :type workspace: Path
        :param settings: Retry and pacing settings.
        :type settings: CorpusSettings
        :return: None.
        :rtype: None
        """
        self._repository: ExperimentRepository = repository
        self._processor: AudioProcessor = processor
        self._predictor: BirdNetPredictor = predictor
        self._noise: NoiseRobustnessExperiment = noise
        self._species: dict[str, str] = species_lookup
        self._workspace: Path = workspace
        self._settings: CorpusSettings = settings

    def analyze(
        self, identifier: str, metadata: dict[str, object]
    ) -> RecordingMeasurements:
        """Measure one original and delete it even when inference is
        interrupted.

        :param identifier: XC recording identifier.
        :type identifier: str
        :param metadata: Cached catalogue fields.
        :type metadata: dict[str, object]
        :return: Complete recording measurements ready to commit.
        :rtype: RecordingMeasurements
        """
        stream: BinaryIO
        source: AudioSource = AudioSource.from_metadata(identifier, metadata)
        downloaded: Path | None = None
        decoded: Path = self._workspace / f"{identifier}.wav"
        try:
            downloaded = self._download_decoded(source, decoded)
            baseline: DataFrame = self._predictor.predict(
                decoded, Artifact.BASELINE
            )
            recording: CatalogueRecording = CatalogueRecording.model_validate(
                metadata
            )
            target: str | None = (
                self._species.get(source.scientific_name)
                if recording.status == CatalogueStatus.IDENTIFIED
                else None
            )
            output: NoiseExperimentOutput | None = None
            reason: str | None = None
            if target is None:
                reason = ExperimentMessage.UNSUPPORTED_TARGET
            else:
                output = self._noise.run(baseline, decoded, target, identifier)
                if output is None:
                    reason = ExperimentMessage.UNDETECTED_TARGET
            with downloaded.open("rb") as stream:
                digest: str = file_digest(stream, "sha256").hexdigest()
            return RecordingMeasurements(
                baseline,
                target,
                output,
                reason,
                downloaded.stat().st_size,
                float(audio_info(decoded).duration),
                digest,
            )
        finally:
            if downloaded is not None:
                downloaded.unlink(missing_ok=True)
            decoded.unlink(missing_ok=True)

    def _download_decoded(self, source: AudioSource, decoded: Path) -> Path:
        """Retry decoding failures after discarding their downloaded originals.

        :param source: Catalogue audio source.
        :type source: AudioSource
        :param decoded: Temporary inference WAV destination.
        :type decoded: Path
        :return: Valid original retained until measurement completes.
        :rtype: Path
        """
        attempt: int
        for attempt in range(self._settings.request_attempts):
            try:
                return self._decode_attempt(source, decoded)
            except AudioDecodingError:
                if attempt + 1 == self._settings.request_attempts:
                    raise
                sleep(min(2**attempt, 30))
        raise RuntimeError(ErrorMessage.MISSING_DOWNLOADED_AUDIO)

    def _decode_attempt(self, source: AudioSource, decoded: Path) -> Path:
        """Decode one download and discard both files if the attempt fails.

        :param source: Requested original recording.
        :type source: AudioSource
        :param decoded: Temporary WAV destination.
        :type decoded: Path
        :return: Original file for fingerprinting and transfer accounting.
        :rtype: Path
        """
        downloaded: Path = self._repository.download(source)
        try:
            self._processor.decode_for_inference(downloaded, decoded)
        except BaseException:
            downloaded.unlink(missing_ok=True)
            decoded.unlink(missing_ok=True)
            raise
        return downloaded
