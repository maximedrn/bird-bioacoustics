"""Experiments annotations services."""

from __future__ import annotations

from pathlib import Path

from pandas import DataFrame, read_csv

from app.domain.constants import (
    ExperimentPhase,
    PredictionColumn,
    ResultColumn,
)
from app.domain.messages import ErrorMessage
from app.domain.models import ReferenceAnnotation
from app.domain.tables import table_records
from app.domain.types import ReferenceLabels


class AnnotationRepository:
    """Load optional complete recording-level annotations independently of XC
    metadata.
    """

    @staticmethod
    def load(
        path: Path, recording_ids: set[str] | None
    ) -> ReferenceLabels | None:
        """Read manually verified labels, rejecting partial or unknown
        recordings.

        :param path: CSV with recording_id, species_name, and complete columns.
        :type path: Path
        :param recording_ids: Identifiers available in the current corpus.
        :type recording_ids: set[str] | None
        :return: Complete label sets, or None when the optional file is absent.
        :rtype: ReferenceLabels | None
        """
        record_value: dict[str, object]
        if not path.exists():
            return None
        dataframe: DataFrame = read_csv(path, keep_default_na=False)
        required: set[str] = {
            ResultColumn.RECORDING_ID,
            PredictionColumn.SPECIES_NAME,
            ExperimentPhase.COMPLETE,
        }
        if not required.issubset(dataframe.columns):
            raise ValueError(ErrorMessage.INVALID_ANNOTATION_COLUMNS)
        if dataframe.empty:
            raise ValueError(ErrorMessage.EMPTY_ANNOTATIONS)
        labels: dict[str, set[str]] = {}
        for record_value in table_records(dataframe[list(required)]):
            record: dict[str, object] = record_value
            annotation: ReferenceAnnotation = (
                ReferenceAnnotation.model_validate(record)
            )
            if (
                recording_ids is not None
                and annotation.recording_id not in recording_ids
            ):
                raise ValueError(
                    ErrorMessage.UNKNOWN_ANNOTATION.format(
                        recording_id=annotation.recording_id
                    )
                )
            labels.setdefault(annotation.recording_id, set())
            if annotation.species_name.strip():
                labels[annotation.recording_id].add(
                    annotation.species_name.strip()
                )
        references: ReferenceLabels = {
            recording_id: frozenset(species_values)
            for recording_id, species_values in labels.items()
        }
        return references
