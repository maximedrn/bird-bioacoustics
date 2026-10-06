"""Storage schema services."""

from __future__ import annotations

from sqlalchemy import JSON, ForeignKey, Index, LargeBinary
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.domain.constants import ProcessingStatus, ResultColumn


class DatabaseSchema(DeclarativeBase):
    """Declare the checkpoint schema with typed SQLAlchemy models."""


class MetadataRow(DatabaseSchema):
    """Store non-secret corpus and scientific configuration attributes."""

    __tablename__: str = "metadata"
    name: Mapped[str] = mapped_column(primary_key=True)
    value: Mapped[str]


class BatchRow(DatabaseSchema):
    """Track the processing interval and completion of each batch."""

    __tablename__: str = "batches"
    batch_id: Mapped[int] = mapped_column(primary_key=True)
    start_cursor: Mapped[int]
    end_cursor: Mapped[int]
    status: Mapped[str] = mapped_column(default=ProcessingStatus.RUNNING)
    started_at: Mapped[str]
    completed_at: Mapped[str | None]


class RecordingRow(DatabaseSchema):
    """Keep source attribution and the outcome of each recording attempt."""

    __tablename__: str = ResultColumn.RECORDINGS
    __table_args__: tuple[Index, ...] = (
        Index("recording_order", "batch_id", "status", "position"),
    )
    recording_id: Mapped[str] = mapped_column(primary_key=True)
    sort_key: Mapped[str]
    batch_id: Mapped[int | None] = mapped_column(
        ForeignKey("batches.batch_id")
    )
    position: Mapped[int | None] = mapped_column(unique=True)
    source: Mapped[dict[str, object]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(default=ProcessingStatus.PENDING)
    error: Mapped[str | None]
    target_species: Mapped[str | None]
    noise_skip_reason: Mapped[str | None]
    overlap_status: Mapped[str | None]
    download_bytes: Mapped[int | None]
    duration_seconds: Mapped[float | None]
    sha256: Mapped[str | None]
    seen_generation: Mapped[int] = mapped_column(default=1)


class SpeciesScoreRow(DatabaseSchema):
    """Retain one maximum confidence per recording and predicted species."""

    __tablename__: str = "species_scores"
    recording_id: Mapped[str] = mapped_column(
        ForeignKey("recordings.recording_id"), primary_key=True
    )
    species_name: Mapped[str] = mapped_column(primary_key=True)
    confidence: Mapped[float]


class ThresholdRow(DatabaseSchema):
    """Persist full-file detection counts and metadata-target recovery."""

    __tablename__: str = "thresholds"
    recording_id: Mapped[str] = mapped_column(
        ForeignKey("recordings.recording_id"), primary_key=True
    )
    threshold: Mapped[float] = mapped_column(primary_key=True)
    detections: Mapped[int]
    target_detected: Mapped[int]


class NoiseRow(DatabaseSchema):
    """Store individual seeded noise trials and their target predictions."""

    __tablename__: str = "noise"
    recording_id: Mapped[str] = mapped_column(
        ForeignKey("recordings.recording_id"), primary_key=True
    )
    trial: Mapped[int] = mapped_column(primary_key=True)
    seed: Mapped[int | None]
    condition: Mapped[str] = mapped_column(primary_key=True)
    snr_db: Mapped[float]
    target_species: Mapped[str]
    confidence: Mapped[float]
    rank: Mapped[int | None]


class NoiseSummaryRow(DatabaseSchema):
    """Keep per-recording moments for equally weighted corpus aggregation."""

    __tablename__: str = "noise_summaries"
    recording_id: Mapped[str] = mapped_column(
        ForeignKey("recordings.recording_id"), primary_key=True
    )
    condition: Mapped[str] = mapped_column(primary_key=True)
    snr_db: Mapped[float]
    confidence_mean: Mapped[float]
    confidence_std: Mapped[float]
    detection_rate: Mapped[float]
    trials: Mapped[int]


class OverlapRow(DatabaseSchema):
    """Store both target roles for each independent recording pair."""

    __tablename__: str = "overlap"
    pair_id: Mapped[str] = mapped_column(primary_key=True)
    recording_a: Mapped[str]
    recording_b: Mapped[str]
    mix: Mapped[str] = mapped_column(primary_key=True)
    weight_a: Mapped[float]
    weight_b: Mapped[float]
    species_a: Mapped[str]
    confidence_a: Mapped[float]
    species_b: Mapped[str]
    confidence_b: Mapped[float]


class SegmentColumns:
    """Describe a short float32 segment, without keeping the original audio."""

    singleton: Mapped[int] = mapped_column(primary_key=True)
    recording_id: Mapped[str]
    species_name: Mapped[str]
    confidence: Mapped[float]
    start_seconds: Mapped[float]
    end_seconds: Mapped[float]
    sample_rate: Mapped[int]
    samples: Mapped[bytes] = mapped_column(LargeBinary)


class PendingSegmentRow(SegmentColumns, DatabaseSchema):
    """Keep at most one segment waiting for an overlap partner."""

    __tablename__: str = "pending_segment"


class PreviewSegmentRow(SegmentColumns, DatabaseSchema):
    """Keep the first target segment for independent spectrogram rendering."""

    __tablename__: str = "preview_segment"
