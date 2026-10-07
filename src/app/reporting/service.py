"""Expose cumulative reports for the CLI and results notebook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from IPython.display import Image, Markdown, display
from pandas import DataFrame

from app.domain.constants import Artifact, CoverageKey, MetadataKey
from app.domain.messages import ErrorMessage
from app.domain.models import NoiseExperimentOutput, ResultSnapshot
from app.domain.settings import ProjectPaths
from app.domain.types import CorpusStatus
from app.reporting.diagnostics import DIAGNOSTIC_OBSERVATIONS, DIAGNOSTIC_PLOTS
from app.reporting.messages import ReportMessage
from app.reporting.observations import NotebookReporter
from app.reporting.plots import ExperimentPlotter
from app.reporting.results import ExperimentResults


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    """Present one consistent cumulative snapshot after closing its reader."""

    paths: ProjectPaths
    snapshot: ResultSnapshot

    def coverage_text(self) -> str:
        """Summarize the observed database coverage in English Markdown.

        :return: Counts and the committed cursor.
        :rtype: str
        """
        status: CorpusStatus = self.snapshot.status
        return ReportMessage.COVERAGE.format(
            processed=status[CoverageKey.PROCESSED_RECORDINGS.value],
            failed=status[CoverageKey.FAILED_RECORDINGS.value],
            batches=status[CoverageKey.COMPLETED_BATCHES.value],
            cursor=status[MetadataKey.CURSOR.value],
        )

    def figure(self, artifact: Artifact) -> Path | None:
        """Save an available figure using only persisted measurements.

        :param artifact: Public figure to produce.
        :type artifact: Artifact
        :return: Saved PNG, or None when scientific measurements are
            unavailable.
        :rtype: Path | None
        """

        destination: Path = artifact.path(self.paths.results)
        snapshot: ResultSnapshot = self.snapshot
        if artifact in DIAGNOSTIC_PLOTS:
            table: DataFrame = snapshot.details.tables()[artifact]
            if table.empty:
                return None
            DIAGNOSTIC_PLOTS[artifact](table, destination)
            return destination
        if artifact == Artifact.PROGRESS:
            ExperimentPlotter.plot_progress(snapshot.status, destination)
        elif artifact == Artifact.NOISE and not snapshot.noise.empty:
            ExperimentPlotter.plot_noise(snapshot.noise, destination)
        elif artifact == Artifact.OVERLAP and not snapshot.overlap.empty:
            ExperimentPlotter.plot_overlap(snapshot.overlap, destination)
        elif artifact == Artifact.THRESHOLD and int(
            snapshot.status[CoverageKey.PROCESSED_RECORDINGS.value]
        ):
            ExperimentPlotter.plot_threshold(snapshot.thresholds, destination)
        elif artifact == Artifact.SPECTROGRAM and snapshot.preview is not None:
            preview: NoiseExperimentOutput = snapshot.preview
            ExperimentPlotter.plot_spectrogram(
                preview.target_segment, preview.sample_rate, destination
            )
        elif artifact in (
            Artifact.NOISE,
            Artifact.OVERLAP,
            Artifact.THRESHOLD,
            Artifact.SPECTROGRAM,
        ):
            return None
        else:
            raise ValueError(
                ErrorMessage.INVALID_PLOT_ARTIFACT.format(artifact=artifact)
            )
        return destination

    def show(self, artifact: Artifact) -> None:
        """Display an available PNG inline and describe its saved measurements.

        :param artifact: Plot artifact to display.
        :type artifact: Artifact
        :return: None.
        :rtype: None
        """

        destination: Path | None = self.figure(artifact)
        if destination is None:
            display(Markdown(ReportMessage.MISSING_MEASUREMENTS))
            return
        display(Image(filename=str(destination)))
        text: str | None = self.observation(artifact)
        if text is not None:
            display(Markdown(text))

    def observation(self, artifact: Artifact) -> str | None:
        """Describe measurements for inline and saved notebook presentation.

        :param artifact: Experiment whose measurements should be described.
        :type artifact: Artifact
        :return: Current observation, or None for a figure without one.
        :rtype: str | None
        """
        if artifact == Artifact.NOISE:
            return NotebookReporter.noise(
                self.snapshot.noise, self.snapshot.skipped_noise
            )
        if artifact == Artifact.OVERLAP:
            return NotebookReporter.overlap(self.snapshot.overlap)
        if artifact == Artifact.THRESHOLD:
            return NotebookReporter.threshold(self.snapshot.thresholds)
        return DIAGNOSTIC_OBSERVATIONS.get(artifact)

    def save_figures(self) -> tuple[Path, ...]:
        """Save available figures for notebook publication and direct viewing.

        :return: Generated plot destinations.
        :rtype: tuple[Path, ...]
        """
        artifact_value: Artifact
        paths: list[Path] = []
        for artifact_value in (
            Artifact.PROGRESS,
            Artifact.NOISE,
            Artifact.OVERLAP,
            Artifact.THRESHOLD,
            Artifact.SPECTROGRAM,
            *DIAGNOSTIC_PLOTS,
        ):
            artifact: Artifact = artifact_value
            path: Path | None = self.figure(artifact)
            if path is not None:
                paths.append(path)
        return tuple(paths)


def load_report(paths: ProjectPaths | None = None) -> ExperimentReport:
    """Read and export a consistent report without API access or inference.

    :param paths: Explicit paths, or paths discovered from the current
        directory.
    :type paths: ProjectPaths | None
    :return: Cumulative reporting snapshot detached from its database reader.
    :rtype: ExperimentReport
    """
    selected: ProjectPaths = paths or ProjectPaths.from_working_directory()
    with ExperimentResults(selected.checkpoint) as reader:
        snapshot: ResultSnapshot = reader.export(
            selected.results, selected.annotations
        )
    return ExperimentReport(selected, snapshot)
