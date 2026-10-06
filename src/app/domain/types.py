"""Domain types services."""

from __future__ import annotations

from typing import TypedDict

from numpy import float32, float64
from numpy.typing import NDArray

type FloatArray = NDArray[float32]


type Float64Array = NDArray[float64]


type ReferenceLabels = dict[str, frozenset[str]]


class CorpusStatus(TypedDict):
    """Type every field of the stable, JSON-compatible coverage report."""

    phase: str
    query: str | None
    snapshot_at: str | None
    expected_recordings: int
    catalogued_recordings: int
    catalogue_complete: bool
    processed_recordings: int
    failed_recordings: int
    pending_recordings: int
    all_recordings_attempted: bool
    cursor: int
    next_page: int
    batch_size: int
    current_batch: int
    completed_batches: int
    downloaded_bytes: int
    processed_audio_seconds: float
