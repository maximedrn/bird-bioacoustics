"""Open read-only checkpoints with portable SQLite journal initialization."""

from enum import StrEnum
from pathlib import Path
from typing import Final

from sqlalchemy import URL, Engine, create_engine, select
from sqlalchemy.engine import Connection as SqlConnection

from app.domain.constants import MetadataFlag
from app.storage.schema import MetadataRow


class SqliteAccess(StrEnum):
    """Select an existing database's access mode."""

    READ_ONLY = "ro"
    READ_WRITE = "rw"


class SqliteStatement:
    """Keep connection settings separate from measurement queries."""

    QUERY_ONLY: Final[str] = "PRAGMA query_only=ON"


def database_url(path: Path, access: SqliteAccess) -> URL:
    """Encode special filename characters in an existing database URI.

    :param path: Existing SQLite database.
    :type path: Path
    :param access: File access mode.
    :type access: SqliteAccess
    :return: SQLAlchemy URL with SQLite URI options.
    :rtype: URL
    """
    return URL.create(
        "sqlite",
        database=path.resolve().as_uri(),
        query={"mode": access.value, "uri": MetadataFlag.TRUE},
    )


def open_read_connection(engine: Engine) -> SqlConnection:
    """Read the schema immediately and release an unusable connection.

    :param engine: Read-only checkpoint engine.
    :type engine: Engine
    :return: Validated connection without an active transaction.
    :rtype: SqlConnection
    """
    connection: SqlConnection = engine.connect()
    try:
        connection.execute(select(MetadataRow.value).limit(0)).close()
        connection.rollback()
    except BaseException:
        connection.close()
        raise
    return connection


def initialize_read_journal(path: Path) -> None:
    """Recreate missing WAL sidecars while prohibiting measurement writes.

    :param path: Existing renamed or copied checkpoint.
    :type path: Path
    :return: None.
    :rtype: None
    """
    engine: Engine = create_engine(
        database_url(path, SqliteAccess.READ_WRITE),
        connect_args={"autocommit": True},
    )
    connection: SqlConnection
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql(SqliteStatement.QUERY_ONLY).close()
            connection.execute(select(MetadataRow.value).limit(0)).close()
    finally:
        engine.dispose()
