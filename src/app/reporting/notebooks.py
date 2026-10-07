"""Publish cumulative report outputs in the existing presentation notebook."""

from __future__ import annotations

from base64 import b64encode
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, cast

from nbformat import NotebookNode, read, validate, write
from nbformat.v4 import new_output

from app.domain.constants import Artifact, FileName
from app.reporting.messages import ReportMessage
from app.reporting.service import ExperimentReport
from app.reporting.writing import atomic_output


class NotebookKey(StrEnum):
    """Name notebook fields and the presentation section marker."""

    CELLS = "cells"
    CELL_TYPE = "cell_type"
    METADATA = "metadata"
    OUTPUTS = "outputs"
    EXECUTION_COUNT = "execution_count"
    EXECUTION = "execution"
    REPORT_SECTION = "app_report"
    ALTERNATIVE_TEXT = "alt"


class NotebookContent:
    """Declare cell, output and MIME types supported by the report."""

    CODE: Final[str] = "code"
    DISPLAY_DATA: Final[str] = "display_data"
    PNG: Final[str] = "image/png"
    MARKDOWN: Final[str] = "text/markdown"
    TEXT: Final[str] = "text/plain"


class NotebookSection(StrEnum):
    """Identify report cells independently of their order or source code."""

    COVERAGE = "coverage"
    PROGRESS = Artifact.PROGRESS.value
    NOISE = Artifact.NOISE.value
    OVERLAP = Artifact.OVERLAP.value
    THRESHOLD = Artifact.THRESHOLD.value
    SPECTROGRAM = Artifact.SPECTROGRAM.value
    NOISE_DISTRIBUTION = Artifact.NOISE_DISTRIBUTION.value
    SPECIES_NOISE = Artifact.SPECIES_NOISE.value
    NOISE_PAIRS = Artifact.NOISE_PAIRS.value
    OVERLAP_OUTCOMES = Artifact.OVERLAP_OUTCOMES.value
    SPECIES_COVERAGE = Artifact.SPECIES_COVERAGE.value
    DURATION_CONFIDENCE = Artifact.DURATION_CONFIDENCE.value
    BATCH_HISTORY = Artifact.BATCH_HISTORY.value


@dataclass(frozen=True, slots=True)
class NotebookPublisher:
    """Save notebook outputs from the same snapshot as the exported figures."""

    report: ExperimentReport
    figures: tuple[Path, ...]

    def refresh(self) -> Path:
        """Replace saved outputs while preserving cells and their source code.

        :return: Atomically updated presentation notebook.
        :rtype: Path
        """
        destination: Path = self.report.paths.root / FileName.NOTEBOOK
        notebook: NotebookNode = read(str(destination), as_version=4)
        validate(notebook)
        cells: dict[NotebookSection, NotebookNode] = self._sections(notebook)
        for section, cell in cells.items():
            cell[NotebookKey.OUTPUTS] = self._outputs(section)
            # Outputs were published from a snapshot, without executing cells.
            cell[NotebookKey.EXECUTION_COUNT] = None
            metadata: dict[str, object] = cast(
                dict[str, object], cell[NotebookKey.METADATA]
            )
            metadata.pop(NotebookKey.EXECUTION, None)
        validate(notebook)
        with atomic_output(destination) as temporary:
            write(notebook, str(temporary))
        return destination

    @staticmethod
    def _sections(
        notebook: NotebookNode,
    ) -> dict[NotebookSection, NotebookNode]:
        """Validate stable markers before replacing any notebook outputs.

        :param notebook: Schema-validated presentation notebook.
        :type notebook: NotebookNode
        :return: Exactly one code cell for each report section.
        :rtype: dict[NotebookSection, NotebookNode]
        """
        cells: list[NotebookNode] = cast(
            list[NotebookNode], notebook[NotebookKey.CELLS]
        )
        sections: dict[NotebookSection, NotebookNode] = {}
        for cell in cells:
            if cell[NotebookKey.CELL_TYPE] != NotebookContent.CODE:
                continue
            metadata: dict[str, object] = cast(
                dict[str, object], cell[NotebookKey.METADATA]
            )
            marker: object = metadata.get(NotebookKey.REPORT_SECTION)
            if marker is None:
                continue
            if not isinstance(marker, str):
                raise ValueError(ReportMessage.INVALID_NOTEBOOK_SECTIONS)
            try:
                section: NotebookSection = NotebookSection(marker)
            except ValueError:
                raise ValueError(
                    ReportMessage.INVALID_NOTEBOOK_SECTIONS
                ) from None
            if section in sections:
                raise ValueError(ReportMessage.INVALID_NOTEBOOK_SECTIONS)
            sections[section] = cell
        if set(sections) != set(NotebookSection):
            raise ValueError(ReportMessage.INVALID_NOTEBOOK_SECTIONS)
        return sections

    def _outputs(self, section: NotebookSection) -> list[NotebookNode]:
        """Build rich outputs using only figures from the current snapshot.

        :param section: Validated coverage or figure section.
        :type section: NotebookSection
        :return: Embedded image and optional observation, or coverage text.
        :rtype: list[NotebookNode]
        """
        if section == NotebookSection.COVERAGE:
            return [self._markdown(self.report.coverage_text())]
        artifact: Artifact = Artifact(section.value)
        path: Path = artifact.path(self.report.paths.results)
        if path not in self.figures:
            return [self._markdown(ReportMessage.MISSING_MEASUREMENTS)]
        output: NotebookNode = new_output(
            output_type=NotebookContent.DISPLAY_DATA,
            data={NotebookContent.PNG: b64encode(path.read_bytes()).decode()},
        )
        output[NotebookKey.METADATA] = {
            NotebookContent.PNG: {
                NotebookKey.ALTERNATIVE_TEXT: (
                    ReportMessage.FIGURE_ALTERNATIVE_TEXT.format(
                        section=artifact.value.replace("_", " ")
                    )
                )
            }
        }
        outputs: list[NotebookNode] = [output]
        text: str | None = self.report.observation(artifact)
        if text is not None:
            outputs.append(self._markdown(text))
        return outputs

    @staticmethod
    def _markdown(text: str) -> NotebookNode:
        """Encode Markdown with a plain-text fallback for notebook viewers.

        :param text: Current coverage or scientific observation.
        :type text: str
        :return: Valid notebook display output.
        :rtype: NotebookNode
        """
        return new_output(
            output_type=NotebookContent.DISPLAY_DATA,
            data={NotebookContent.MARKDOWN: text, NotebookContent.TEXT: text},
        )
