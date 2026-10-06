"""Storage database services."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Protocol, Self

from numpy import float32, frombuffer
from pandas import DataFrame, read_sql_query
from sqlalchemy import Engine, func, select
from sqlalchemy.engine import Connection as SqlConnection
from sqlalchemy.engine import Row, RowMapping
from sqlalchemy.sql.functions import count as sql_count

from app.domain.constants import (
    CoverageKey,
    ExperimentPhase,
    MetadataFlag,
    MetadataKey,
    PredictionColumn,
    ProcessingStatus,
    ResultColumn,
)
from app.domain.messages import ErrorMessage
from app.domain.models import DetectionCandidate, NoiseExperimentOutput
from app.domain.types import CorpusStatus
from app.storage.schema import (
    BatchRow,
    DatabaseSchema,
    MetadataRow,
    NoiseRow,
    NoiseSummaryRow,
    OverlapRow,
    PendingSegmentRow,
    PreviewSegmentRow,
    RecordingRow,
    SpeciesScoreRow,
    ThresholdRow,
)


class ExperimentDatabase:
    """Read bounded checkpoint data using SQLAlchemy expressions."""

    path: Path
    engine: Engine
    connection: SqlConnection

    def get_metadata(self, name: str) -> str | None:
        """Read one non-secret configuration attribute.

        :param name: Attribute name.
        :type name: str
        :return: Stored value or None.
        :rtype: str | None
        """
        return self.connection.scalar(
            select(MetadataRow.value).where(MetadataRow.name == name)
        )

    def report(self) -> CorpusStatus:
        """Report observed coverage and volume from the persisted recording
        outcomes.

        :return: Successful, failed and pending recording counts and volume.
        :rtype: CorpusStatus
        """
        counts: dict[str, int] = dict(
            self.connection.execute(
                select(RecordingRow.status, sql_count()).group_by(
                    RecordingRow.status
                )
            ).all()
        )
        volumes: Row[int | None, float | None] = self.connection.execute(
            select(
                func.coalesce(func.sum(RecordingRow.download_bytes), 0),
                func.coalesce(func.sum(RecordingRow.duration_seconds), 0),
            ).where(RecordingRow.status == ProcessingStatus.DONE)
        ).one()
        complete: bool = (
            self.get_metadata(MetadataKey.CATALOGUE_COMPLETE)
            == MetadataFlag.TRUE
        )
        return {
            CoverageKey.PHASE.value: ExperimentPhase.PROCESSING
            if counts.get(ProcessingStatus.PENDING, 0)
            else ExperimentPhase.READY_FOR_NEXT_BATCH
            if not complete and counts
            else ExperimentPhase.CATALOGUE
            if not complete
            else ExperimentPhase.COMPLETED_WITH_ERRORS
            if counts.get(ProcessingStatus.FAILED, 0)
            else ExperimentPhase.COMPLETE,
            MetadataKey.QUERY.value: self.get_metadata(MetadataKey.QUERY),
            CoverageKey.SNAPSHOT_AT.value: self.get_metadata(
                MetadataKey.CREATED_AT
            ),
            MetadataKey.EXPECTED_RECORDINGS.value: int(
                self.get_metadata(MetadataKey.EXPECTED_RECORDINGS) or "0"
            ),
            CoverageKey.CATALOGUED_RECORDINGS.value: sum(counts.values()),
            MetadataKey.CATALOGUE_COMPLETE.value: complete,
            CoverageKey.PROCESSED_RECORDINGS.value: counts.get(
                ProcessingStatus.DONE, 0
            ),
            CoverageKey.FAILED_RECORDINGS.value: counts.get(
                ProcessingStatus.FAILED, 0
            ),
            CoverageKey.PENDING_RECORDINGS.value: counts.get(
                ProcessingStatus.PENDING, 0
            ),
            CoverageKey.ALL_RECORDINGS_ATTEMPTED.value: complete
            and not counts.get(ProcessingStatus.PENDING, 0),
            MetadataKey.CURSOR.value: int(
                self.get_metadata(MetadataKey.CURSOR) or "0"
            ),
            MetadataKey.NEXT_PAGE.value: int(
                self.get_metadata(MetadataKey.NEXT_PAGE) or "1"
            ),
            MetadataKey.BATCH_SIZE.value: int(
                self.get_metadata(MetadataKey.BATCH_SIZE) or "0"
            ),
            MetadataKey.CURRENT_BATCH.value: int(
                self.get_metadata(MetadataKey.CURRENT_BATCH) or "0"
            ),
            CoverageKey.COMPLETED_BATCHES.value: int(
                self.connection.scalar(
                    select(sql_count())
                    .select_from(BatchRow)
                    .where(BatchRow.status == ProcessingStatus.DONE)
                )
                or 0
            ),
            CoverageKey.DOWNLOADED_BYTES.value: int(volumes[0] or 0),
            CoverageKey.PROCESSED_AUDIO_SECONDS.value: float(
                volumes[1] or 0.0
            ),
        }

    def segment(
        self,
        model: type[PendingSegmentRow | PreviewSegmentRow],
        directory: Path,
    ) -> NoiseExperimentOutput | None:
        """Restore a saved short segment without downloading its original
        audio.

        :param model: Pending-pair or spectrogram segment model.
        :type model: type[PendingSegmentRow | PreviewSegmentRow]
        :param directory: Working directory for a potential generated variant.
        :type directory: Path
        :return: Stored target segment, or None.
        :rtype: NoiseExperimentOutput | None
        """
        row: RowMapping | None = (
            self.connection.execute(select(model).where(model.singleton == 1))
            .mappings()
            .first()
        )
        if row is None:
            return None
        return NoiseExperimentOutput(
            recording_id=row[ResultColumn.RECORDING_ID],
            target=DetectionCandidate(
                species_name=row[PredictionColumn.SPECIES_NAME],
                confidence=row[PredictionColumn.CONFIDENCE],
                start_seconds=row["start_seconds"],
                end_seconds=row["end_seconds"],
            ),
            sample_rate=row["sample_rate"],
            target_segment=frombuffer(row["samples"], dtype=float32).copy(),
            target_segment_path=directory
            / f"{row[ResultColumn.RECORDING_ID]}_target_original.wav",
            results=DataFrame(),
        )

    def table(self, name: str, limit: int = 20) -> DataFrame:
        """Read a bounded preview without retaining a catalogue-sized
        dataframe.

        :param name: Public measurement table name.
        :type name: str
        :param limit: Positive maximum number of rows.
        :type limit: int
        :return: Measurement preview.
        :rtype: DataFrame
        """
        models: dict[str, type[DatabaseSchema]] = {
            model.__tablename__: model
            for model in (
                BatchRow,
                RecordingRow,
                NoiseRow,
                NoiseSummaryRow,
                OverlapRow,
                SpeciesScoreRow,
                ThresholdRow,
            )
        }
        if name not in models or limit < 1:
            raise ValueError(ErrorMessage.INVALID_TABLE_PREVIEW)
        return read_sql_query(
            select(models[name]).limit(limit), self.connection
        )

    def close(self) -> None:
        """Release the connection and its pooled database resources.

        :return: None.
        :rtype: None
        """
        self.connection.close()
        self.engine.dispose()

    def __enter__(self) -> Self:
        """Enter a resource-managed reader.

        :return: Current database reader.
        :rtype: Self
        """
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release database resources after a read succeeds or fails.

        :param exception_type: Raised exception class.
        :type exception_type: type[BaseException] | None
        :param exception: Raised exception instance.
        :type exception: BaseException | None
        :param traceback: Associated traceback.
        :type traceback: TracebackType | None
        :return: None.
        :rtype: None
        """
        self.close()


class CheckpointAccess(Protocol):
    """Describe shared transactional access used by writable storage
    components.
    """

    connection: SqlConnection

    def get_metadata(self, name: str) -> str | None:
        """Read one checkpoint attribute.

        :param name: Persisted key.
        :type name: str
        :return: Stored value or None.
        :rtype: str | None
        """

    def set_metadata(self, name: str, value: str) -> None:
        """Stage one checkpoint attribute in the shared transaction.

        :param name: Persisted key.
        :type name: str
        :param value: Non-secret attribute value.
        :type value: str
        :return: None.
        :rtype: None
        """
