"""Domain errors services."""

from __future__ import annotations


class BirdNetInferenceError(RuntimeError):
    """Report a failed inference session separately from an unavailable audio
    file.
    """


class ExperimentAlreadyRunningError(RuntimeError):
    """Signal that another worker already owns this experiment checkpoint."""


class RecordingError(ValueError):
    """Describe a permanent recording failure that can safely advance its
    cursor.
    """


class AudioUnavailableError(RecordingError):
    """Report catalogue metadata without a downloadable audio resource."""


class AudioDecodingError(RecordingError):
    """Report audio that cannot be decoded using either supported decoder."""


class DownloadUnavailableError(RecordingError):
    """Report a permanently refused or unavailable recording download."""


class DownloadTransientError(RuntimeError):
    """Stop a batch on exhausted network retries while keeping the audio
    pending.
    """
