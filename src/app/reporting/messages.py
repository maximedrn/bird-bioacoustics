"""Presentation text for plots, notebook observations and coverage."""

from typing import Final


class PlotLabel:
    """Centralize graph labels independently of scientific measurements."""

    RECORDINGS_IN_THE_CATALOGUE_SNAPSHOT: Final[str] = (
        "Recordings in the catalogue snapshot"
    )
    ANALYSIS_OF_CACHED_RECORDINGS: Final[str] = "Analysis of cached recordings"
    CATALOGUED_RECORDINGS: Final[str] = "Catalogued recordings"
    TARGET_SPECIES_CONFIDENCE: Final[str] = "Target-species confidence"
    NOISE_TARGET_CONFIDENCE: Final[str] = "Noise: target confidence"
    TARGET_DETECTED_IN_EXPORTED_PREDICTIONS: Final[str] = (
        "Target detected in exported predictions"
    )
    NOISE_TARGET_DETECTION_RATE: Final[str] = "Noise: target detection rate"
    A_B_MIX_AMPLITUDE_COEFFICIENTS: Final[str] = (
        "A/B mix (amplitude coefficients)"
    )
    BIRDNET_CONFIDENCE: Final[str] = "BirdNET confidence"
    EFFECT_OF_OVERLAPPING_BIRD_VOCALIZATIONS: Final[str] = (
        "Effect of overlapping bird vocalizations"
    )
    COUNT_ACROSS_RECORDINGS: Final[str] = "Count across recordings"
    RETAINED_PREDICTIONS: Final[str] = "Retained predictions"
    RECORDING_LEVEL_RATE: Final[str] = "Recording-level rate"
    TARGET_RETENTION_OPTIONAL_REFERENCE_METRICS: Final[str] = (
        "Target retention / optional reference metrics"
    )
    TIME_S: Final[str] = "Time (s)"
    FREQUENCY_HZ: Final[str] = "Frequency (Hz)"
    TARGET_SEGMENT_SPECTROGRAM: Final[str] = "Target-segment spectrogram"
    CACHED: Final[str] = "Cached"
    STILL_TO_COLLECT: Final[str] = "Still to collect"
    CATALOGUE_COVERAGE_TOTAL_UNKNOWN: Final[str] = (
        "Catalogue coverage (total unknown)"
    )
    ANALYSED: Final[str] = "Analysed"
    FAILED: Final[str] = "Failed"
    PENDING: Final[str] = "Pending"
    RECORDING_MEANS_1_SD: Final[str] = "Recording means ± 1 SD"
    ORIGINAL_SEGMENTS_MEAN: Final[str] = "Original segments (mean)"
    MEAN_DETECTION_RATE: Final[str] = "Mean detection rate"
    ORIGINAL_SEGMENTS: Final[str] = "Original segments"
    SNR_DB_LOWER_VALUES_MEAN_STRONGER_NOISE: Final[str] = (
        "SNR (dB) — lower values mean stronger noise"
    )
    TARGET_A_MEAN_1_SD: Final[str] = "Target A (mean ± 1 SD)"
    TARGET_B_MEAN_1_SD: Final[str] = "Target B (mean ± 1 SD)"
    DETECTIONS: Final[str] = "Detections"
    DISTINCT_SPECIES: Final[str] = "Distinct species"
    METADATA_TARGET_RECOVERED: Final[str] = "Metadata target recovered"
    CONFIDENCE_THRESHOLD: Final[str] = "Confidence threshold"
    POWER_DB: Final[str] = "Power (dB)"
    CATALOGUE_COVERAGE: Final[str] = "Catalogue coverage ({ratio:.1%})"


class ObservationMessage:
    """Describe cumulative results using explicit template fields."""

    HEADING: Final[str] = "### Observation"
    NOISE_BASELINE: Final[str] = (
        "Segments from **{count} recordings** were tested. Mean target "
        "confidence without added noise is "
        "**{baseline_confidence:.3f}**."
    )
    NOISE_LEVEL: Final[str] = (
        "- **{snr_db:g} dB**: mean confidence "
        "**{confidence_mean:.3f}**, standard deviation across recordings "
        "**{confidence_std:.3f}**, target detected in "
        "**{detection_rate:.1%}** of trials."
    )
    NOISE_EXPLANATION: Final[str] = (
        "Error bars show one standard deviation of recording means. "
        "A value of 0 means "
        "the target is absent from predictions exported at the minimum "
        "confidence threshold of 0.10."
    )
    NOISE_SKIPPED: Final[str] = (
        "**{skipped} recordings** without an initially detected target "
        "were excluded from this test."
    )
    PARAGRAPH_BREAK: Final[str] = "\n\n"
    LINE_BREAK: Final[str] = "\n"
    THRESHOLD_COUNTS: Final[str] = (
        "### Observation\n\nBetween thresholds **{first_threshold:.2f}** "
        "and **{last_threshold:.2f}**, retained detections change from "
        "**{first_detections}** to **{last_detections}**, and predicted "
        "species from **{first_species}** to **{last_species}**.\n\n"
    )
    TARGET_RECOVERY: Final[str] = (
        "Xeno-canto's primary species is recovered in "
        "**{first_rate:.1%}** then **{last_rate:.1%}** of recordings whose "
        "target is identified and supported by BirdNET. This rate "
        "measures recovery of the known target; recall across all species "
        "requires complete annotations."
    )
    NO_TARGET: Final[str] = (
        "No identified target supported by BirdNET is available in the "
        "processed recordings."
    )
    COMPLETE_ANNOTATIONS: Final[str] = (
        "\n\nPrecision, recall and micro F1 use only the "
        "**{recordings} fully annotated recordings**, counting each "
        "recording/species pair once."
    )
    NO_ANNOTATIONS: Final[str] = (
        "\n\nPrecision, recall and F1 are not calculated without complete "
        "annotations."
    )
    OVERLAP: Final[str] = (
        "### Observation\n\nMeans cover **{pair_count} pairs** of distinct "
        "recordings. Between mixes {first_mix} and {last_mix}, mean target "
        "A confidence changes from **{first_a:.3f}** to "
        "**{last_a:.3f}**, and target B confidence from **{first_b:.3f}** "
        "to **{last_b:.3f}**.\n\nA and B identify the roles within each "
        "pair. Error bars show one standard deviation across "
        "pairs. Coefficients apply to amplitude, without prior level "
        "equalization."
    )


class ReportMessage:
    """Present saved corpus coverage and unavailable figures."""

    COVERAGE: Final[str] = (
        "**{processed:,} recordings analysed** · **{failed:,} failures** · "
        "**{batches} completed batches** · **cursor {cursor:,}**"
    )
    MISSING_MEASUREMENTS: Final[str] = (
        "There are not enough measurements to plot this chart yet."
    )
    INVALID_NOTEBOOK_SECTIONS: Final[str] = (
        "The result notebook must contain one marked code cell for each "
        "report section."
    )
    NOTEBOOK_PDF_FAILED: Final[str] = "Notebook PDF rendering failed: {error}"
    EMPTY_NOTEBOOK_PDF: Final[str] = "The notebook PDF contains no pages."
    NOTEBOOK_PREVIEW_TITLE: Final[str] = (
        "Cumulative Xeno-canto experiment results"
    )
