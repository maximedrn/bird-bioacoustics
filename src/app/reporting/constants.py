"""Name diagnostic columns and bound cumulative report presentation."""

from enum import StrEnum
from typing import Final


class DetailColumn(StrEnum):
    """Identify diagnostic values independently of display labels."""

    MINIMUM = "minimum"
    Q1 = "q1"
    MEDIAN = "median"
    Q3 = "q3"
    MAXIMUM = "maximum"
    ORIGINAL_CONFIDENCE = "original_confidence"
    BOTH = "both"
    ONLY_A = "only_a"
    ONLY_B = "only_b"
    NEITHER = "neither"
    DURATION_SECONDS = "duration_seconds"
    TARGET_CONFIDENCE = "target_confidence"
    BATCH_ID = "batch_id"
    BATCH_STATUS = "batch_status"
    CONFIDENCE_TOTAL = "confidence_total"
    DETECTION_TOTAL = "detection_total"
    POSITION = "position"
    SAMPLE_RANK = "sample_rank"
    TOTAL_RECORDINGS = "total_recordings"


class ChartSetting:
    """Bound readable plots while retaining full-corpus aggregate measures."""

    TOP_SPECIES: Final[int] = 20
    SCATTER_RECORDINGS: Final[int] = 5_000
    STREAM_ROWS: Final[int] = 1_000
    BATCH_TICKS: Final[int] = 10
    QUARTILES: Final[tuple[float, ...]] = (0.0, 0.25, 0.5, 0.75, 1.0)
