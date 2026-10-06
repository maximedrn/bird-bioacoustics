"""Coordinate one durable recording batch and its scientific measurements."""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
from importlib.metadata import version
from json import dumps
from logging import Logger, getLogger
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep

from sqlalchemy import select
from sqlalchemy.sql.functions import count as sql_count

from app.audio.backend import load_model
from app.audio.prediction import BirdNetPredictor
from app.audio.processing import AudioProcessor
from app.catalogue.client import XenoCantoClient
from app.catalogue.downloads import ExperimentRepository
from app.domain.constants import (
    BIRDNET_DISTRIBUTION,
    PROTOCOL_VERSION,
    MetadataFlag,
    MetadataKey,
    OverlapStatus,
    ProcessingStatus,
)
from app.domain.errors import BirdNetInferenceError, RecordingError
from app.domain.messages import ErrorMessage, LogMessage
from app.domain.models import NoiseExperimentOutput, RecordingBatch
from app.domain.protocols import BirdNetModelProtocol, PredictionSession
from app.domain.settings import (
    CorpusSettings,
    ExperimentSettings,
    ModelSettings,
    ProjectPaths,
)
from app.domain.types import CorpusStatus
from app.experiments.noise import NoiseRobustnessExperiment
from app.experiments.overlap import OverlapExperiment
from app.experiments.progress import ProgressObserver, TqdmBatchProgress
from app.experiments.recording import (
    RecordingAnalysisService,
    RecordingMeasurements,
)
from app.storage.checkpoint import ExperimentCheckpoint
from app.storage.schema import (
    PendingSegmentRow,
    PreviewSegmentRow,
    RecordingRow,
)

logger: Logger = getLogger(__name__)


def protocol_configuration(
    model: ModelSettings,
    experiment: ExperimentSettings,
    corpus: CorpusSettings,
) -> dict[str, object]:
    """Build the same scientific fingerprint inputs used by the original
    notebook.

    :param model: Acoustic model settings.
    :type model: ModelSettings
    :param experiment: Scientific perturbation settings.
    :type experiment: ExperimentSettings
    :param corpus: Catalogue scope and batch ordering.
    :type corpus: CorpusSettings
    :return: JSON-compatible existing checkpoint configuration.
    :rtype: dict[str, object]
    """
    return {
        "model": asdict(model),
        "experiment": asdict(experiment),
        MetadataKey.QUERY: corpus.query,
        BIRDNET_DISTRIBUTION: version(BIRDNET_DISTRIBUTION),
        "protocol_version": PROTOCOL_VERSION,
        MetadataKey.BATCH_SIZE: corpus.batch_size,
    }


class XenoCantoExperiment:
    """Own one batch worker while delegating recording analysis and storage."""

    def __init__(
        self,
        paths: ProjectPaths,
        model_settings: ModelSettings,
        settings: ExperimentSettings,
        corpus_settings: CorpusSettings,
    ) -> None:
        """Open compatible writable storage without loading the inference
        model.

        :param paths: Project directories.
        :type paths: ProjectPaths
        :param model_settings: Model configuration.
        :type model_settings: ModelSettings
        :param settings: Scientific protocol.
        :type settings: ExperimentSettings
        :param corpus_settings: Catalogue and batch settings.
        :type corpus_settings: CorpusSettings
        :return: None.
        :rtype: None
        """
        self._paths: ProjectPaths = paths
        self._model_settings: ModelSettings = model_settings
        self._settings: ExperimentSettings = settings
        self._corpus_settings: CorpusSettings = corpus_settings
        paths.create_directories()
        configuration: dict[str, object] = protocol_configuration(
            model_settings, settings, corpus_settings
        )
        signature: str = sha256(
            dumps(configuration, sort_keys=True).encode()
        ).hexdigest()
        self.checkpoint: ExperimentCheckpoint = ExperimentCheckpoint(
            paths.checkpoint, signature
        )
        self.checkpoint.set_metadata(
            MetadataKey.CONFIGURATION, dumps(configuration, sort_keys=True)
        )
        self.checkpoint.connection.commit()

    def run(self, progress: ProgressObserver | None = None) -> CorpusStatus:
        """Process exactly one batch, updating progress after committed
        outcomes.

        :param progress: Optional progress observer; defaults to tqdm.
        :type progress: ProgressObserver | None
        :return: Cumulative checkpoint coverage.
        :rtype: CorpusStatus
        """
        with self.checkpoint.exclusive_run():
            recovered: int = (
                self.checkpoint.batches.recover_interrupted_inference()
            )
            if recovered:
                logger.info(LogMessage.INFERENCE_RECOVERED, recovered)
            batch: RecordingBatch | None = (
                self.checkpoint.batches.active_batch()
            )
            if batch is None:
                client: XenoCantoClient | None = None
                if (
                    self.checkpoint.get_metadata(
                        MetadataKey.CATALOGUE_COMPLETE
                    )
                    != MetadataFlag.TRUE
                ):
                    client = XenoCantoClient(
                        XenoCantoClient.local_key(self._paths.root),
                        self._corpus_settings,
                    )
                batch = self.checkpoint.batches.prepare_batch(
                    client,
                    self._settings.random_seed,
                    self._corpus_settings.batch_size,
                )
            if batch is None:
                return self.checkpoint.report()
            counts: dict[str, int] = dict(
                self.checkpoint.connection.execute(
                    select(RecordingRow.status, sql_count())
                    .where(RecordingRow.batch_id == batch.batch_id)
                    .group_by(RecordingRow.status)
                ).all()
            )
            self.checkpoint.connection.commit()
            if not counts.get(ProcessingStatus.PENDING, 0):
                self.checkpoint.batches.finish_batch(batch)
                return self.checkpoint.report()
            observer: ProgressObserver = (
                progress if progress is not None else TqdmBatchProgress()
            )
            cursor: int = int(
                self.checkpoint.get_metadata(MetadataKey.CURSOR)
                or batch.start_cursor
            )
            self.checkpoint.connection.commit()
            try:
                observer.start(batch, cursor, counts)
                self._process_batch(batch, observer)
            finally:
                observer.close()
            self.checkpoint.batches.finish_batch(batch)
            return self.checkpoint.report()

    @property
    def paths(self) -> ProjectPaths:
        """Expose immutable paths for independent reporting.

        :return: Project directory configuration.
        :rtype: ProjectPaths
        """
        return self._paths

    def _process_batch(
        self, batch: RecordingBatch, progress: ProgressObserver
    ) -> None:
        """Own temporary workspace and load the model for exactly one batch.

        :param batch: Durable recording interval.
        :type batch: RecordingBatch
        :param progress: Progress observer receiving committed outcomes.
        :type progress: ProgressObserver
        :return: None.
        :rtype: None
        """
        model: BirdNetModelProtocol = load_model(self._model_settings)
        directory: str
        with TemporaryDirectory(
            prefix="worker_", dir=self._paths.generated_audio
        ) as directory:
            self._process_session(batch, progress, model, Path(directory))

    def _process_session(
        self,
        batch: RecordingBatch,
        progress: ProgressObserver,
        model: BirdNetModelProtocol,
        workspace: Path,
    ) -> None:
        """Reuse one healthy session and release it after completion or
        interruption.

        :param batch: Durable recording interval.
        :type batch: RecordingBatch
        :param progress: Committed progress observer.
        :type progress: ProgressObserver
        :param model: Loaded acoustic model.
        :type model: BirdNetModelProtocol
        :param workspace: Managed scratch directory.
        :type workspace: Path
        :return: None.
        :rtype: None
        """
        species: dict[str, str] = {
            label.split("_", 1)[0]: label for label in model.species_list
        }
        repository: ExperimentRepository = ExperimentRepository(
            self._paths, self._corpus_settings
        )
        processor: AudioProcessor = AudioProcessor()
        scratch_paths: ProjectPaths = replace(
            self._paths, results=workspace, generated_audio=workspace
        )
        session: PredictionSession
        with model.predict_session(
            top_k=None,
            n_workers=self._model_settings.n_workers,
            batch_size=self._model_settings.batch_size,
            max_n_files=max(
                1
                + len(self._settings.snr_values_db)
                * self._settings.noise_repetitions,
                len(self._settings.mix_ratios),
            ),
            default_confidence_threshold=self._model_settings.minimum_confidence,
        ) as session:
            predictor: BirdNetPredictor = BirdNetPredictor(
                model, scratch_paths, self._model_settings, session
            )
            noise: NoiseRobustnessExperiment = NoiseRobustnessExperiment(
                predictor, processor, scratch_paths, self._settings
            )
            overlap: OverlapExperiment = OverlapExperiment(
                predictor, processor, scratch_paths, self._settings
            )
            analysis: RecordingAnalysisService = RecordingAnalysisService(
                repository,
                processor,
                predictor,
                noise,
                species,
                workspace,
                self._corpus_settings,
            )
            self._process_records(
                batch, progress, analysis, overlap, workspace
            )

    def _process_records(
        self,
        batch: RecordingBatch,
        progress: ProgressObserver,
        analysis: RecordingAnalysisService,
        overlap: OverlapExperiment,
        workspace: Path,
    ) -> None:
        """Advance progress after committed outcomes and roll back unfinished
        work.

        :param batch: Durable processing interval.
        :type batch: RecordingBatch
        :param progress: Progress observer.
        :type progress: ProgressObserver
        :param analysis: Single-recording measurement service.
        :type analysis: RecordingAnalysisService
        :param overlap: Scientific mixture service.
        :type overlap: OverlapExperiment
        :param workspace: Temporary working directory.
        :type workspace: Path
        :return: None.
        :rtype: None
        """
        identifier: str
        metadata: dict[str, object]
        try:
            for (
                identifier,
                metadata,
            ) in self.checkpoint.batches.pending_sources(batch.batch_id):
                status: ProcessingStatus = self._process_recording(
                    identifier, metadata, analysis, overlap, workspace
                )
                cursor: int = int(
                    self.checkpoint.get_metadata(MetadataKey.CURSOR) or "0"
                )
                self.checkpoint.connection.commit()
                progress.update(cursor, identifier, status)
                sleep(self._corpus_settings.request_interval_seconds)
        finally:
            self.checkpoint.connection.rollback()

    def _process_recording(
        self,
        identifier: str,
        metadata: dict[str, object],
        analysis: RecordingAnalysisService,
        overlap: OverlapExperiment,
        workspace: Path,
    ) -> ProcessingStatus:
        """Commit scientific measures and cursor together, leaving
        interruptions pending.

        :param identifier: Current XC recording identifier.
        :type identifier: str
        :param metadata: Cached catalogue metadata.
        :type metadata: dict[str, object]
        :param analysis: Single-recording measurement service.
        :type analysis: RecordingAnalysisService
        :param overlap: Scientific mixture service.
        :type overlap: OverlapExperiment
        :param workspace: Managed temporary directory.
        :type workspace: Path
        :return: Persisted successful or permanent-failure outcome.
        :rtype: ProcessingStatus
        """
        error: BirdNetInferenceError | RecordingError
        try:
            measurements: RecordingMeasurements = analysis.analyze(
                identifier, metadata
            )
            self.checkpoint.measurements.store(
                identifier,
                measurements.baseline,
                measurements.target_species,
                measurements.noise,
                measurements.noise_skip_reason,
                measurements.download_bytes,
                measurements.duration_seconds,
                measurements.fingerprint,
                self._settings,
            )
            self._store_overlap(
                identifier, measurements.noise, overlap, workspace
            )
            self.checkpoint.batches.advance_cursor(identifier)
            self.checkpoint.connection.commit()
            return ProcessingStatus.DONE
        except BirdNetInferenceError as error:
            self.checkpoint.connection.rollback()
            raise BirdNetInferenceError(
                ErrorMessage.INFERENCE_INTERRUPTED.format(
                    identifier=identifier, error=error
                )
            ) from error
        except RecordingError as error:
            self.checkpoint.connection.rollback()
            self.checkpoint.batches.fail(
                identifier,
                LogMessage.FAILURE_DETAIL.format(
                    kind=type(error).__name__, error=error
                ),
            )
            logger.debug(LogMessage.RECORDING_FAILED, identifier, error)
            return ProcessingStatus.FAILED
        finally:
            self._clean_workspace(workspace)

    @staticmethod
    def _clean_workspace(workspace: Path) -> None:
        """Remove generated files before processing another recording.

        :param workspace: Temporary batch directory.
        :type workspace: Path
        :return: None.
        :rtype: None
        """
        temporary_value: Path
        for temporary_value in workspace.iterdir():
            temporary: Path = temporary_value
            if temporary.is_file():
                temporary.unlink()

    def _store_overlap(
        self,
        identifier: str,
        output: NoiseExperimentOutput | None,
        overlap: OverlapExperiment,
        workspace: Path,
    ) -> None:
        """Stage disjoint pairs while preserving the previous batch's partner.

        :param identifier: Current recording identifier.
        :type identifier: str
        :param output: Detectable target segment, if any.
        :type output: NoiseExperimentOutput | None
        :param overlap: Scientific mixture service.
        :type overlap: OverlapExperiment
        :param workspace: Temporary segment directory.
        :type workspace: Path
        :return: None.
        :rtype: None
        """
        if output is None:
            return
        pending: NoiseExperimentOutput | None = self.checkpoint.segment(
            PendingSegmentRow, workspace
        )
        if pending is None:
            self.checkpoint.measurements.save_segment(
                PendingSegmentRow, output
            )
            self.checkpoint.measurements.overlap_status(
                (identifier,), ProcessingStatus.PENDING
            )
        elif pending.target.species_name == output.target.species_name:
            self.checkpoint.measurements.overlap_status(
                (identifier,), OverlapStatus.SAME_SPECIES_AS_PENDING
            )
        else:
            pair_id: str = f"{pending.recording_id}_{identifier}"
            self.checkpoint.measurements.save_overlap(
                overlap.run(pending, output, pair_id)
            )
        self.checkpoint.measurements.save_segment(PreviewSegmentRow, output)


def run_batch(
    paths: ProjectPaths | None = None,
    *,
    corpus: CorpusSettings | None = None,
    model: ModelSettings | None = None,
    experiment: ExperimentSettings | None = None,
    progress: ProgressObserver | None = None,
) -> CorpusStatus:
    """Run one batch and always release its database connection and worker
    lock.

    :param paths: Explicit project paths, or paths discovered from the working
        directory.
    :type paths: ProjectPaths | None
    :param corpus: Batch and network settings.
    :type corpus: CorpusSettings | None
    :param model: Acoustic model settings.
    :type model: ModelSettings | None
    :param experiment: Scientific perturbation settings.
    :type experiment: ExperimentSettings | None
    :param progress: Optional progress observer.
    :type progress: ProgressObserver | None
    :return: Cumulative processing status after one batch.
    :rtype: CorpusStatus
    """
    worker: XenoCantoExperiment = XenoCantoExperiment(
        paths or ProjectPaths.from_working_directory(),
        model or ModelSettings(),
        experiment or ExperimentSettings(),
        corpus or CorpusSettings(),
    )
    try:
        return worker.run(progress)
    finally:
        worker.checkpoint.close()
