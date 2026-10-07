"""Storage checkpoint services."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from hashlib import sha256
from json import JSONDecodeError, dumps, loads
from pathlib import Path
from sqlite3 import SQLITE_DBCONFIG_ENABLE_FKEY
from sqlite3 import Connection as SqliteConnection
from types import TracebackType
from typing import Self, cast

from filelock import FileLock
from filelock import Timeout as FileLockTimeout
from sqlalchemy import URL, Engine, create_engine, event, inspect
from sqlalchemy.dialects.sqlite import Insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Connection as SqlConnection
from sqlalchemy.pool import ConnectionPoolEntry

from app.domain.constants import MetadataKey
from app.domain.errors import ExperimentAlreadyRunningError
from app.domain.messages import ErrorMessage
from app.storage.batches import BatchStore
from app.storage.catalogue import CatalogueStore
from app.storage.database import ExperimentDatabase
from app.storage.measurements import MeasurementStore
from app.storage.schema import DatabaseSchema, MetadataRow


class ExperimentCheckpoint(ExperimentDatabase):
    """Own a compatible checkpoint and its writable storage components."""

    def __init__(self, path: Path, signature: str) -> None:
        """Open compatible storage while preserving existing catalogue pages.

        :param path: Durable SQLite checkpoint.
        :type path: Path
        :param signature: Hash of the scientific protocol.
        :type signature: str
        :return: None.
        :rtype: None
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path: Path = path
        self._lock: FileLock = FileLock(
            str(path.with_suffix(".lock")), timeout=0
        )
        try:
            self._lock.acquire()
        except FileLockTimeout:
            raise ExperimentAlreadyRunningError(
                ErrorMessage.ACTIVE_WORKER
            ) from None
        try:
            self._open(signature)
        except BaseException:
            if hasattr(self, "connection"):
                self.connection.close()
            if hasattr(self, "engine"):
                self.engine.dispose()
            self._lock.release()
            raise

    def _open(self, signature: str) -> None:
        """Initialize compatible storage only after acquiring the worker lock.

        :param signature: Scientific protocol fingerprint.
        :type signature: str
        :return: None.
        :rtype: None
        """
        self.engine: Engine = create_engine(
            URL.create("sqlite", database=str(self.path)),
            connect_args={"autocommit": False},
        )
        event.listen(self.engine, "connect", self.configure_connection)
        self.connection: SqlConnection = self.engine.connect()
        existing: str | None = (
            self.get_metadata(MetadataKey.SIGNATURE)
            if inspect(self.engine).has_table(MetadataRow.__tablename__)
            else None
        )
        self._validate_signature(existing, signature)
        DatabaseSchema.metadata.create_all(self.connection)
        self.catalogue: CatalogueStore = CatalogueStore(self)
        self.batches: BatchStore = BatchStore(self, self.catalogue)
        self.measurements: MeasurementStore = MeasurementStore(self)
        self.set_metadata(MetadataKey.SIGNATURE, signature)
        self.connection.commit()

    def _validate_signature(
        self, existing: str | None, signature: str
    ) -> None:
        """Migrate a verified legacy signature without recording batch size.

        :param existing: Saved signature, or None for a new checkpoint.
        :type existing: str | None
        :param signature: Requested scientific protocol fingerprint.
        :type signature: str
        :return: None.
        :rtype: None
        """
        if existing is None or existing == signature:
            return
        saved: str | None = self.get_metadata(MetadataKey.CONFIGURATION)
        if saved is None:
            raise ValueError(ErrorMessage.INCOMPATIBLE_CHECKPOINT)
        try:
            payload: object = loads(saved)
        except JSONDecodeError:
            raise ValueError(ErrorMessage.INCOMPATIBLE_CHECKPOINT) from None
        if not isinstance(payload, dict):
            raise ValueError(ErrorMessage.INCOMPATIBLE_CHECKPOINT)
        configuration: dict[str, object] = cast(dict[str, object], payload)
        legacy_signature: str = sha256(
            dumps(configuration, sort_keys=True).encode()
        ).hexdigest()
        if legacy_signature != existing:
            raise ValueError(ErrorMessage.INCOMPATIBLE_CHECKPOINT)
        configuration.pop(MetadataKey.BATCH_SIZE, None)
        normalized: str = dumps(configuration, sort_keys=True)
        if sha256(normalized.encode()).hexdigest() != signature:
            raise ValueError(ErrorMessage.INCOMPATIBLE_CHECKPOINT)
        self.set_metadata(MetadataKey.CONFIGURATION, normalized)

    @staticmethod
    def configure_connection(
        connection: SqliteConnection, _record: ConnectionPoolEntry
    ) -> None:
        """Enable SQLite's concurrent readers and referential integrity.

        :param connection: SQLite DBAPI connection provided by SQLAlchemy.
        :type connection: SqliteConnection
        :param record: SQLAlchemy pool entry associated with the connection.
        :type record: ConnectionPoolEntry
        :return: None.
        :rtype: None
        """
        connection.autocommit = True
        # Set WAL at database level; data queries use SQLAlchemy expressions.
        connection.execute("PRAGMA journal_mode=WAL").close()
        connection.setconfig(SQLITE_DBCONFIG_ENABLE_FKEY, True)
        connection.autocommit = False

    @contextmanager
    def exclusive_run(self) -> Generator[None, None, None]:
        """Permit one experiment worker while allowing independent result
        readers.

        :return: Exclusive file-lock context, released on exit or process
            termination.
        :rtype: Generator[None, None, None]
        """
        with self._lock:
            yield

    def close(self) -> None:
        """Release storage and the lifetime worker lock even after an
        exception.

        :return: None.
        :rtype: None
        """
        try:
            super().close()
        finally:
            self._lock.release()

    def __enter__(self) -> Self:
        """Enter a resource-managed writable checkpoint.

        :return: Current checkpoint.
        :rtype: Self
        """
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the checkpoint on success, failure or interruption.

        :param exception_type: Raised exception class, if any.
        :type exception_type: type[BaseException] | None
        :param exception: Raised exception instance.
        :type exception: BaseException | None
        :param traceback: Associated traceback.
        :type traceback: TracebackType | None
        :return: None.
        :rtype: None
        """
        self.close()

    def set_metadata(self, name: str, value: str) -> None:
        """Stage a non-secret attribute in the active transaction.

        :param name: Attribute name.
        :type name: str
        :param value: Serializable text.
        :type value: str
        :return: None.
        :rtype: None
        """
        statement: Insert = sqlite_insert(MetadataRow).values(
            name=name, value=value
        )
        self.connection.execute(
            statement.on_conflict_do_update(
                index_elements=[MetadataRow.name],
                set_={"value": statement.excluded.value},
            )
        )
