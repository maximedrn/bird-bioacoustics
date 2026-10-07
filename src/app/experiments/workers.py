"""Give concurrent recordings independent inference and temporary resources."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

from app.audio.prediction import BirdNetPredictor
from app.audio.processing import AudioProcessor
from app.catalogue.downloads import ExperimentRepository
from app.domain.constants import RecordingRuntime
from app.domain.protocols import BirdNetModelProtocol, PredictionSession
from app.domain.settings import (
    CorpusSettings,
    ExperimentSettings,
    InferenceSettings,
    ModelSettings,
    ProjectPaths,
)
from app.experiments.noise import NoiseRobustnessExperiment
from app.experiments.overlap import OverlapExperiment
from app.experiments.recording import RecordingAnalysisService


@dataclass(frozen=True, slots=True)
class RecordingWorker:
    """Own one recording slot without accessing experiment storage."""

    analysis: RecordingAnalysisService
    overlap: OverlapExperiment
    workspace: Path
    session: PredictionSession

    def clean(self) -> None:
        """Delete scratch files after the coordinator stores this recording.

        :return: None.
        :rtype: None
        """
        for temporary in self.workspace.rglob("*"):
            if temporary.is_file():
                temporary.unlink()


class RecordingWorkerFactory:
    """Open reusable sessions before starting recording dispatch threads."""

    def __init__(
        self,
        paths: ProjectPaths,
        model: BirdNetModelProtocol,
        model_settings: ModelSettings,
        experiment: ExperimentSettings,
        corpus: CorpusSettings,
        inference: InferenceSettings,
    ) -> None:
        """Share immutable configuration while isolating mutable services.

        :param paths: Project directories.
        :type paths: ProjectPaths
        :param model: Cached model descriptor, with no native session.
        :type model: BirdNetModelProtocol
        :param model_settings: Scientific model configuration.
        :type model_settings: ModelSettings
        :param experiment: Scientific perturbation configuration.
        :type experiment: ExperimentSettings
        :param corpus: Download and retry configuration.
        :type corpus: CorpusSettings
        :param inference: Hardware and per-recording producer settings.
        :type inference: InferenceSettings
        :return: None.
        :rtype: None
        """
        self._paths: ProjectPaths = paths
        self._model: BirdNetModelProtocol = model
        self._model_settings: ModelSettings = model_settings
        self._experiment: ExperimentSettings = experiment
        self._corpus: CorpusSettings = corpus
        self._inference: InferenceSettings = inference

    def create(self, resources: ExitStack) -> RecordingWorker:
        """Register a workspace and one inference process for a recording slot.

        :param resources: Owner of all batch sessions and scratch directories.
        :type resources: ExitStack
        :return: Independent analysis and mixture services.
        :rtype: RecordingWorker
        """
        directory: str = resources.enter_context(
            TemporaryDirectory(
                prefix=RecordingRuntime.WORKSPACE_PREFIX,
                dir=self._paths.generated_audio,
            )
        )
        workspace: Path = Path(directory)
        scratch: ProjectPaths = replace(
            self._paths,
            data=workspace / RecordingRuntime.DOWNLOAD_DIRECTORY,
            results=workspace,
            generated_audio=workspace,
        )
        scratch.create_directories()
        session: PredictionSession = resources.enter_context(
            self._model.predict_session(
                top_k=None,
                n_workers=RecordingRuntime.INFERENCE_WORKERS,
                n_producers=self._inference.n_producers,
                device=self._inference.device,
                batch_size=self._inference.batch_size,
                max_n_files=max(
                    1
                    + len(self._experiment.snr_values_db)
                    * self._experiment.noise_repetitions,
                    len(self._experiment.mix_ratios),
                ),
                default_confidence_threshold=(
                    self._model_settings.minimum_confidence
                ),
            )
        )
        return self._services(scratch, session)

    def _services(
        self, scratch: ProjectPaths, session: PredictionSession
    ) -> RecordingWorker:
        """Build services that only touch their recording slot's files.

        :param scratch: Isolated download and generated-audio paths.
        :type scratch: ProjectPaths
        :param session: Entered reusable BirdNET session.
        :type session: PredictionSession
        :return: Analysis worker without a checkpoint connection.
        :rtype: RecordingWorker
        """
        processor: AudioProcessor = AudioProcessor()
        predictor: BirdNetPredictor = BirdNetPredictor(
            self._model, scratch, self._model_settings, session
        )
        noise: NoiseRobustnessExperiment = NoiseRobustnessExperiment(
            predictor, processor, scratch, self._experiment
        )
        species: dict[str, str] = {}
        label: str
        for label in self._model.species_list:
            species[label.split("_", 1)[0]] = label
        analysis: RecordingAnalysisService = RecordingAnalysisService(
            ExperimentRepository(scratch, self._corpus),
            processor,
            predictor,
            noise,
            species,
            scratch.generated_audio,
            self._corpus,
        )
        return RecordingWorker(
            analysis,
            OverlapExperiment(predictor, processor, scratch, self._experiment),
            scratch.generated_audio,
            session,
        )
