"""Storage measurements services."""

from __future__ import annotations

from pandas import DataFrame, Series
from sqlalchemy import delete, insert, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.domain.constants import (
    OverlapStatus,
    PredictionColumn,
    ProcessingStatus,
    ResultColumn,
)
from app.domain.models import (
    NoiseExperimentOutput,
    OverlapExperimentOutput,
)
from app.domain.settings import ExperimentSettings
from app.domain.tables import table_records
from app.experiments.noise import ExperimentSummary
from app.storage.database import CheckpointAccess
from app.storage.schema import (
    NoiseRow,
    NoiseSummaryRow,
    OverlapRow,
    PendingSegmentRow,
    PreviewSegmentRow,
    RecordingRow,
    SpeciesScoreRow,
    ThresholdRow,
)


class MeasurementStore:
    """Persist measurements using the shared checkpoint transaction."""

    def __init__(self, database: CheckpointAccess) -> None:
        """Share the checkpoint transaction without opening another connection.

        :param database: Writable checkpoint access.
        :type database: CheckpointAccess
        :return: None.
        :rtype: None
        """
        self._database: CheckpointAccess = database

    def save_segment(
        self,
        model: type[PendingSegmentRow | PreviewSegmentRow],
        output: NoiseExperimentOutput,
    ) -> None:
        """Stage one short target segment in the current recording transaction.

        :param model: Pending-pair or spectrogram segment model.
        :type model: type[PendingSegmentRow | PreviewSegmentRow]
        :param output: Current target and float32 samples.
        :type output: NoiseExperimentOutput
        :return: None.
        :rtype: None
        """
        self._database.connection.execute(
            sqlite_insert(model)
            .values(
                singleton=1,
                recording_id=output.recording_id,
                species_name=output.target.species_name,
                confidence=output.target.confidence,
                start_seconds=output.target.start_seconds,
                end_seconds=output.target.end_seconds,
                sample_rate=output.sample_rate,
                samples=output.target_segment.tobytes(),
            )
            .on_conflict_do_nothing(index_elements=[model.singleton])
        )

    def overlap_status(
        self, identifiers: tuple[str, ...], status: str
    ) -> None:
        """Stage pairing outcomes for the current recording transaction.

        :param identifiers: Affected recording identifiers.
        :type identifiers: tuple[str, ...]
        :param status: Explicit overlap outcome.
        :type status: str
        :return: None.
        :rtype: None
        """
        self._database.connection.execute(
            update(RecordingRow)
            .where(RecordingRow.recording_id.in_(identifiers))
            .values(overlap_status=status)
        )

    def save_overlap(self, output: OverlapExperimentOutput) -> None:
        """Store all mixture measurements and consume their pending partner.

        :param output: Validated overlap output for one disjoint pair.
        :type output: OverlapExperimentOutput
        :return: None.
        :rtype: None
        """
        rows: list[dict[str, object]] = table_records(output.results)
        self._database.connection.execute(insert(OverlapRow), rows)
        self._database.connection.execute(delete(PendingSegmentRow))
        self.overlap_status(
            (str(rows[0]["recording_a"]), str(rows[0]["recording_b"])),
            OverlapStatus.PAIRED,
        )

    def store(
        self,
        identifier: str,
        baseline: DataFrame,
        target: str | None,
        output: NoiseExperimentOutput | None,
        skip_reason: str | None,
        downloaded_bytes: int,
        duration: float,
        digest: str,
        settings: ExperimentSettings,
    ) -> None:
        """Stage all scientific measures before the recording transaction is
        committed.

        :param identifier: Recording identifier.
        :type identifier: str
        :param baseline: Full-file window predictions.
        :type baseline: DataFrame
        :param target: Supported identified metadata target, or None.
        :type target: str | None
        :param output: Optional targeted noise measurements.
        :type output: NoiseExperimentOutput | None
        :param skip_reason: Explicit targeted-test exclusion.
        :type skip_reason: str | None
        :param downloaded_bytes: Original file size.
        :type downloaded_bytes: int
        :param duration: Original audio duration in seconds.
        :type duration: float
        :param digest: Original audio SHA-256.
        :type digest: str
        :param settings: Scientific thresholds.
        :type settings: ExperimentSettings
        :return: None.
        :rtype: None
        """
        maxima: Series[float] = baseline.groupby(
            PredictionColumn.SPECIES_NAME
        )[PredictionColumn.CONFIDENCE].max()
        scores: list[dict[str, object]] = [
            {
                ResultColumn.RECORDING_ID: identifier,
                PredictionColumn.SPECIES_NAME: str(species),
                PredictionColumn.CONFIDENCE: float(score),
            }
            for species, score in maxima.items()
        ]
        if scores:
            self._database.connection.execute(insert(SpeciesScoreRow), scores)
        self._database.connection.execute(
            insert(ThresholdRow),
            [
                {
                    ResultColumn.RECORDING_ID: identifier,
                    ResultColumn.THRESHOLD: threshold,
                    ResultColumn.DETECTIONS: int(
                        (
                            baseline[PredictionColumn.CONFIDENCE] >= threshold
                        ).sum()
                    ),
                    "target_detected": int(
                        target is not None
                        and float(maxima.get(target, 0.0)) >= threshold
                    ),
                }
                for threshold in settings.confidence_thresholds
            ],
        )
        if output is not None:
            self._database.connection.execute(
                insert(NoiseRow), table_records(output.results)
            )
            self._database.connection.execute(
                insert(NoiseSummaryRow),
                table_records(ExperimentSummary.noise(output.results)[0]),
            )
        self._database.connection.execute(
            update(RecordingRow)
            .where(RecordingRow.recording_id == identifier)
            .values(
                status=ProcessingStatus.DONE,
                target_species=target,
                noise_skip_reason=skip_reason,
                download_bytes=downloaded_bytes,
                duration_seconds=duration,
                sha256=digest,
            )
        )
