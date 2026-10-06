"""Convert dataframe records at the typed storage and validation boundary."""

from __future__ import annotations

from collections.abc import Hashable

from pandas import DataFrame


def table_records(table: DataFrame) -> list[dict[str, object]]:
    """Make column names explicit strings before validating or storing records.

    :param table: Scientific or annotation dataframe.
    :type table: DataFrame
    :return: Records with string keys and unchanged cell values.
    :rtype: list[dict[str, object]]
    """
    records: list[dict[Hashable, object]] = table.to_dict(orient="records")
    return [
        {str(key): value for key, value in record.items()}
        for record in records
    ]
