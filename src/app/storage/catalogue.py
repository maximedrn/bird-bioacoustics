"""Storage catalogue services."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from typing import cast

from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import Insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.sql.functions import count as sql_count

from app.domain.constants import (
    MetadataFlag,
    MetadataKey,
    ProcessingStatus,
    ResultColumn,
)
from app.domain.messages import ErrorMessage, LogMessage
from app.domain.protocols import CatalogueClient
from app.storage.database import CheckpointAccess
from app.storage.schema import RecordingRow


class CatalogueStore:
    """Persist catalogue using the shared checkpoint transaction."""

    def __init__(self, database: CheckpointAccess) -> None:
        """Share the checkpoint transaction without opening another connection.

        :param database: Writable checkpoint access.
        :type database: CheckpointAccess
        :return: None.
        :rtype: None
        """
        self._database: CheckpointAccess = database

    def freeze_catalogue(self, client: CatalogueClient) -> str:
        """Store the upper XC identifier bound before incrementally paging
        metadata.

        :param client: Authenticated catalogue client.
        :type client: CatalogueClient
        :return: Frozen catalogue query.
        :rtype: str
        """
        page: int
        query: str | None = self._database.get_metadata(MetadataKey.QUERY)
        if query is not None:
            return query
        latest: dict[str, object] = client.page(
            client.settings.query + " since:1", 1
        )
        recent: list[dict[str, object]] = cast(
            list[dict[str, object]], latest[ResultColumn.RECORDINGS]
        )
        if not recent:
            raise RuntimeError(ErrorMessage.MISSING_RECENT_UPLOADS)
        maximum_id: int = max(int(str(item["id"])) for item in recent)
        for page in range(2, int(str(latest["numPages"])) + 1):
            payload: dict[str, object] = client.page(
                client.settings.query + " since:1", page
            )
            maximum_id = max(
                maximum_id,
                *(
                    int(str(item["id"]))
                    for item in cast(
                        list[dict[str, object]],
                        payload[ResultColumn.RECORDINGS],
                    )
                ),
            )
        query = f"{client.settings.query} nr:1-{maximum_id}"
        self._database.set_metadata(MetadataKey.QUERY, query)
        self._database.set_metadata(
            MetadataKey.CREATED_AT, datetime.now(UTC).isoformat()
        )
        self._database.connection.commit()
        return query

    def cache_page(self, client: CatalogueClient, random_seed: int) -> None:
        """Commit one metadata page and its next-page cursor without fetching
        audio.

        :param client: Authenticated catalogue client.
        :type client: CatalogueClient
        :param random_seed: Seed for deterministic ordering inside a batch.
        :type random_seed: int
        :return: None.
        :rtype: None
        """
        item: dict[str, object]
        query: str = self.freeze_catalogue(client)
        next_page: int = int(
            self._database.get_metadata(MetadataKey.NEXT_PAGE) or "1"
        )
        generation: int = int(
            self._database.get_metadata(MetadataKey.CATALOGUE_GENERATION)
            or "1"
        )
        payload: dict[str, object] = client.page(query, next_page)
        expected: int = int(str(payload["numRecordings"]))
        pages: int = int(str(payload["numPages"]))
        rows: list[dict[str, object]] = []
        for item in cast(
            list[dict[str, object]], payload[ResultColumn.RECORDINGS]
        ):
            identifier: str = "XC" + str(int(str(item["id"])))
            source: dict[str, object] = {
                name: item.get(name, "")
                for name in (
                    "id",
                    "gen",
                    "sp",
                    "grp",
                    "status",
                    "rec",
                    "cnt",
                    "url",
                    "file",
                    "file-name",
                    "lic",
                    "q",
                    "length",
                    "also",
                    "smp",
                    "uploaded",
                )
            }
            rows.append(
                {
                    ResultColumn.RECORDING_ID: identifier,
                    "sort_key": sha256(
                        f"{random_seed}:{identifier}".encode()
                    ).hexdigest(),
                    "source": source,
                    "seen_generation": generation,
                }
            )
        if rows:
            statement: Insert = sqlite_insert(RecordingRow)
            self._database.connection.execute(
                statement.on_conflict_do_update(
                    index_elements=[RecordingRow.recording_id],
                    set_={
                        "source": statement.excluded.source,
                        "seen_generation": statement.excluded.seen_generation,
                    },
                ),
                rows,
            )
        self._database.set_metadata(MetadataKey.NEXT_PAGE, str(next_page + 1))
        self._database.set_metadata(MetadataKey.PAGES, str(pages))
        self._database.set_metadata(
            MetadataKey.EXPECTED_RECORDINGS, str(expected)
        )
        self._database.set_metadata(
            MetadataKey.EXPECTED_SPECIES, str(payload["numSpecies"])
        )
        if next_page >= pages:
            cached: int = int(
                self._database.connection.scalar(
                    select(sql_count())
                    .select_from(RecordingRow)
                    .where(RecordingRow.seen_generation == generation)
                )
                or 0
            )
            if cached != expected:
                self._database.set_metadata(MetadataKey.NEXT_PAGE, "1")
                self._database.set_metadata(
                    MetadataKey.CATALOGUE_GENERATION, str(generation + 1)
                )
                self._database.connection.commit()
                raise RuntimeError(
                    ErrorMessage.CATALOGUE_CHANGED.format(
                        cached=cached, expected=expected
                    )
                )
            self._database.connection.execute(
                delete(RecordingRow).where(
                    RecordingRow.seen_generation != generation,
                    RecordingRow.status == ProcessingStatus.PENDING,
                    RecordingRow.batch_id.is_(None),
                )
            )
            self._database.set_metadata(
                MetadataKey.CATALOGUE_COMPLETE, MetadataFlag.TRUE
            )
        # Commit the last page and its completion flag together.
        self._database.connection.commit()
        print(
            LogMessage.CATALOGUE_CACHED.format(page=next_page, pages=pages),
            flush=True,
        )
