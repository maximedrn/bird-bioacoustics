"""Storage batches services."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import UTC, datetime

from sqlalchemy import Select, bindparam, func, insert, select, update
from sqlalchemy.engine import Row
from sqlalchemy.sql.functions import count as sql_count

from app.domain.constants import (
    MetadataFlag,
    MetadataKey,
    ProcessingStatus,
)
from app.domain.messages import ErrorMessage, ExperimentMessage
from app.domain.models import RecordingBatch
from app.domain.protocols import CatalogueClient
from app.storage.catalogue import CatalogueStore
from app.storage.database import CheckpointAccess
from app.storage.schema import BatchRow, RecordingRow


class BatchStore:
    """Persist batches using the shared checkpoint transaction."""

    def __init__(
        self, database: CheckpointAccess, catalogue: CatalogueStore
    ) -> None:
        """Share the checkpoint transaction without opening another connection.

        :param database: Writable checkpoint access.
        :type database: CheckpointAccess
        :return: None.
        :rtype: None
        """
        self._database: CheckpointAccess = database
        self._catalogue: CatalogueStore = catalogue

    def active_batch(self) -> RecordingBatch | None:
        """Read an unfinished batch before fetching any further catalogue
        pages.

        :return: Interrupted/running interval, or None.
        :rtype: RecordingBatch | None
        """
        row: Row[int, int, int] | None = self._database.connection.execute(
            select(
                BatchRow.batch_id, BatchRow.start_cursor, BatchRow.end_cursor
            )
            .where(BatchRow.status == ProcessingStatus.RUNNING)
            .order_by(BatchRow.batch_id)
            .limit(1)
        ).first()
        return RecordingBatch(*row) if row is not None else None

    def prepare_batch(
        self, client: CatalogueClient | None, random_seed: int, batch_size: int
    ) -> RecordingBatch | None:
        """Resume a saved batch or collect just enough metadata for the next
        interval.

        :param client: Catalogue client; None is sufficient for an existing
            batch.
        :type client: CatalogueClient | None
        :param random_seed: Seed for ordering records inside each batch.
        :type random_seed: int
        :param batch_size: Maximum number of new recordings in one batch.
        :type batch_size: int
        :return: Durable batch interval, or None when the corpus is exhausted.
        :rtype: RecordingBatch | None
        """
        if batch_size < 1:
            raise ValueError(ErrorMessage.INVALID_BATCH_SIZE)
        batch: RecordingBatch | None = self.active_batch()
        if batch is not None:
            return batch
        available: Select[int] = (
            select(sql_count())
            .select_from(RecordingRow)
            .where(
                RecordingRow.status == ProcessingStatus.PENDING,
                RecordingRow.batch_id.is_(None),
            )
        )
        while (
            int(self._database.connection.scalar(available) or 0) < batch_size
        ):
            if (
                self._database.get_metadata(MetadataKey.CATALOGUE_COMPLETE)
                == MetadataFlag.TRUE
            ):
                break
            if client is None:
                raise RuntimeError(ErrorMessage.MISSING_CATALOGUE_CLIENT)
            self._catalogue.cache_page(client, random_seed)
        identifiers: list[str] = list(
            self._database.connection.scalars(
                select(RecordingRow.recording_id)
                .where(
                    RecordingRow.status == ProcessingStatus.PENDING,
                    RecordingRow.batch_id.is_(None),
                )
                .order_by(RecordingRow.sort_key)
                .limit(batch_size)
            )
        )
        if not identifiers:
            self._database.connection.commit()
            return None
        start: int = int(
            self._database.get_metadata(MetadataKey.CURSOR) or "0"
        )
        batch_id: int = (
            int(
                self._database.connection.scalar(
                    select(func.max(BatchRow.batch_id))
                )
                or 0
            )
            + 1
        )
        batch = RecordingBatch(batch_id, start, start + len(identifiers))
        self._database.connection.execute(
            insert(BatchRow).values(
                batch_id=batch.batch_id,
                start_cursor=batch.start_cursor,
                end_cursor=batch.end_cursor,
                started_at=datetime.now(UTC).isoformat(),
            )
        )
        self._database.connection.execute(
            update(RecordingRow)
            .where(RecordingRow.recording_id == bindparam("selected_id"))
            .values(
                batch_id=batch.batch_id,
                position=bindparam("assigned_position"),
            ),
            [
                {
                    "selected_id": identifier,
                    "assigned_position": start + offset,
                }
                for offset, identifier in enumerate(identifiers)
            ],
        )
        self._database.set_metadata(MetadataKey.CURSOR, str(start))
        self._database.set_metadata(MetadataKey.BATCH_SIZE, str(batch_size))
        self._database.set_metadata(
            MetadataKey.CURRENT_BATCH, str(batch.batch_id)
        )
        self._database.connection.commit()
        return batch

    def pending_sources(
        self, batch_id: int
    ) -> Iterator[tuple[str, dict[str, object]]]:
        """Yield one batch in cursor order with short SQLite read transactions.

        :param batch_id: Durable batch to resume or process.
        :type batch_id: int
        :return: Pending recording identifiers and their metadata.
        :rtype: Iterator[tuple[str, dict[str, object]]]
        """
        identifier: str
        position: int | None
        source: dict[str, object]
        last_position: int = (
            int(self._database.get_metadata(MetadataKey.CURSOR) or "0") - 1
        )
        while True:
            rows: Sequence[Row[str, int | None, dict[str, object]]] = (
                self._database.connection.execute(
                    select(
                        RecordingRow.recording_id,
                        RecordingRow.position,
                        RecordingRow.source,
                    )
                    .where(
                        RecordingRow.batch_id == batch_id,
                        RecordingRow.status == ProcessingStatus.PENDING,
                        RecordingRow.position > last_position,
                    )
                    .order_by(RecordingRow.position)
                    .limit(100)
                ).all()
            )
            self._database.connection.commit()
            if not rows:
                return
            for identifier, position, source in rows:
                if position is None:
                    raise ValueError(ErrorMessage.MISSING_RECORDING_POSITION)
                last_position = position
                yield identifier, source

    def recover_interrupted_inference(self) -> int:
        """Requeue legacy worker failures while preserving completed
        measurements.

        :return: Number of recordings incorrectly failed by a cancelled
            session.
        :rtype: int
        """
        batch: RecordingBatch | None = self.active_batch()
        if batch is None:
            return 0
        rows: Sequence[Row[str, int | None]] = (
            self._database.connection.execute(
                select(RecordingRow.recording_id, RecordingRow.position).where(
                    RecordingRow.batch_id == batch.batch_id,
                    RecordingRow.status == ProcessingStatus.FAILED,
                    RecordingRow.error.startswith(
                        ExperimentMessage.LEGACY_ERROR_PREFIX
                    ),
                    RecordingRow.error.contains(
                        ExperimentMessage.LEGACY_SESSION_CANCELLED
                    ),
                    RecordingRow.download_bytes.is_(None),
                    RecordingRow.sha256.is_(None),
                )
            ).all()
        )
        if not rows:
            return 0
        identifiers: tuple[str, ...] = tuple(row.recording_id for row in rows)
        self._database.connection.execute(
            update(RecordingRow)
            .where(RecordingRow.recording_id.in_(identifiers))
            .values(status=ProcessingStatus.PENDING, error=None)
        )
        cursor: int = int(
            self._database.get_metadata(MetadataKey.CURSOR) or "0"
        )
        self._database.set_metadata(
            MetadataKey.CURSOR,
            str(min(cursor, min(row.position for row in rows))),
        )
        self._database.connection.commit()
        return len(rows)

    def advance_cursor(self, identifier: str) -> None:
        """Stage cursor advancement atomically with a successful or failed
        recording.

        :param identifier: Recording whose outcome has just been staged.
        :type identifier: str
        :return: None.
        :rtype: None
        """
        row: Row[int | None, str, int | None] = (
            self._database.connection.execute(
                select(
                    RecordingRow.position,
                    RecordingRow.status,
                    RecordingRow.batch_id,
                ).where(RecordingRow.recording_id == identifier)
            ).one()
        )
        cursor: int = int(
            self._database.get_metadata(MetadataKey.CURSOR) or "0"
        )
        if row.position != cursor or row.status == ProcessingStatus.PENDING:
            raise ValueError(ErrorMessage.INVALID_CURSOR_ADVANCEMENT)
        # Recovery can leave genuine file failures ahead of the cursor.
        # Keep their outcomes and resume at the next unprocessed recording.
        next_position: int | None = self._database.connection.scalar(
            select(func.min(RecordingRow.position)).where(
                RecordingRow.batch_id == row.batch_id,
                RecordingRow.status == ProcessingStatus.PENDING,
                RecordingRow.position > cursor,
            )
        )
        end_cursor: int = self._database.connection.execute(
            select(BatchRow.end_cursor).where(
                BatchRow.batch_id == row.batch_id
            )
        ).scalar_one()
        self._database.set_metadata(
            MetadataKey.CURSOR,
            str(end_cursor if next_position is None else next_position),
        )

    def finish_batch(self, batch: RecordingBatch) -> None:
        """Commit batch completion only after every planned recording was
        attempted.

        :param batch: Durable processing interval.
        :type batch: RecordingBatch
        :return: None.
        :rtype: None
        """
        pending: int = int(
            self._database.connection.scalar(
                select(sql_count())
                .select_from(RecordingRow)
                .where(
                    RecordingRow.batch_id == batch.batch_id,
                    RecordingRow.status == ProcessingStatus.PENDING,
                )
            )
            or 0
        )
        if (
            pending
            or int(self._database.get_metadata(MetadataKey.CURSOR) or "0")
            != batch.end_cursor
        ):
            raise ValueError(ErrorMessage.UNFINISHED_BATCH)
        self._database.connection.execute(
            update(BatchRow)
            .where(BatchRow.batch_id == batch.batch_id)
            .values(
                status=ProcessingStatus.DONE,
                completed_at=datetime.now(UTC).isoformat(),
            )
        )
        self._database.connection.commit()

    def fail(self, identifier: str, reason: str) -> None:
        """Commit one failed attempt after its scientific transaction was
        rolled back.

        :param identifier: Recording identifier.
        :type identifier: str
        :param reason: Sanitized failure description.
        :type reason: str
        :return: None.
        :rtype: None
        """
        self._database.connection.execute(
            update(RecordingRow)
            .where(RecordingRow.recording_id == identifier)
            .values(status=ProcessingStatus.FAILED, error=reason)
        )
        self.advance_cursor(identifier)
        self._database.connection.commit()
