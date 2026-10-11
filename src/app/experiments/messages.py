"""Centralize validation and progress messages for annotated experiments."""

from typing import Final


class StatisticalError:
    """Describe invalid inputs at the statistical computation boundary."""

    INVALID_OBSERVATIONS: Final[str] = (
        "Require finite observations and >=100 resamples"
    )
    INVALID_GROUPS: Final[str] = "Each observation needs one group"
    INSUFFICIENT_UNITS: Final[str] = (
        "Require at least two independent sampling units"
    )
    INVALID_MATRICES: Final[str] = (
        "Require equal nonempty binary clip/class matrices"
    )
    INVALID_CLIP_GROUPS: Final[str] = (
        "Require one group per clip and >=100 resamples"
    )
    INSUFFICIENT_SOURCES: Final[str] = (
        "Require at least two independent source files"
    )


class SoundscapeMessage:
    """Describe downloads, cache checks and completed model comparisons."""

    DOWNLOADING: Final[str] = "Downloading {filename}"
    CHANGED_SELECTION: Final[str] = "Fixed benchmark selection has changed"
    MISSING_ARCHIVE_CONTENTS: Final[str] = "Missing archive contents: {name}"
    MISSING_AUDIO: Final[str] = "Missing selected audio: {names}"
    INVALID_ARCHIVE_FINGERPRINT: Final[str] = (
        "Missing or invalid archive fingerprint: {path}"
    )
    CHANGED_ARCHIVE: Final[str] = (
        "Downloaded archive differs from its saved fingerprint: {path}"
    )
    DIFFERENT_COVERAGE: Final[str] = (
        "{dataset}: coverage differs for {excluded}; retain as reference classes"
    )
    SELECTED_CLIPS: Final[str] = (
        "{dataset}: {clips} clips, {classes} common classes"
    )
    RUNNING_INFERENCE: Final[str] = (
        "Running bound inference: {dataset} {version}"
    )
    MISSING_PREDICTIONS: Final[str] = (
        "One or more selected files lack inference output"
    )
    INVALID_PREDICTION_ORDER: Final[str] = (
        "Benchmark prediction order does not match reference"
    )
    INCOMPLETE_PREDICTIONS: Final[str] = "Incomplete benchmark predictions"
    COMPLETED: Final[str] = "Completed {dataset} BirdNET {version}"


class ValidationMessage:
    """Describe invalid source partitions and calibration predictions."""

    INSUFFICIENT_SOURCES: Final[str] = (
        "Source calibration requires at least four recordings"
    )
    INVALID_SCORES: Final[str] = (
        "Calibration requires aligned finite scores within [0, 1]"
    )
    MISALIGNED_SCORES: Final[str] = (
        "Calibration scores do not match the selected clips"
    )


class ControlledMessage:
    """Describe invalid perturbations and bound inference progress."""

    INSUFFICIENT_FOREGROUNDS: Final[str] = (
        "{dataset}: insufficient held-out positive sources"
    )
    INSUFFICIENT_BACKGROUNDS: Final[str] = (
        "Insufficient independent annotated background sources"
    )
    INSUFFICIENT_PAIRS: Final[str] = (
        "Controlled overlap requires at least two disjoint pairs"
    )
    INCOMPLETE_PREDICTIONS: Final[str] = (
        "Controlled observations lack complete inference output"
    )
    INVALID_SCORES: Final[str] = (
        "Controlled score order or numeric values are invalid"
    )
    CHANGED_MODEL: Final[str] = (
        "Controlled model differs from recorded benchmark weights"
    )
    RUNNING_INFERENCE: Final[str] = (
        "Controlled {kind}: {dataset}, {count} observations"
    )
    INVALID_SHAPES: Final[str] = (
        "Foreground and background require equal nonempty shapes"
    )
    INVALID_NOISE: Final[str] = (
        "Controlled noise requires finite SNR and nonzero audio"
    )
    INVALID_MIXTURE: Final[str] = (
        "Controlled mixing requires two nonempty weighted signals"
    )
    ZERO_POWER: Final[str] = "Controlled mixing requires nonzero audio power"
    SHORT_BACKGROUND: Final[str] = (
        "Recorded background is shorter than its foreground"
    )
