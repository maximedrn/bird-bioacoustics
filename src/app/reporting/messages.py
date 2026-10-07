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
    NOISE_DISTRIBUTION: Final[str] = "Noise: distribution across recordings"
    NOISE_BOX_RANGE: Final[str] = (
        "Boxes: median and interquartile range; whiskers: minimum / maximum"
    )
    CONDITION_COUNT: Final[str] = "{condition}\nn={count:,}"
    ORIGINAL: Final[str] = "Original"
    SPECIES_NOISE: Final[str] = "Noise: target recovery by species"
    SPECIES_RATE_COUNT: Final[str] = "{rate:.0%}\nn={count:,}"
    RATE: Final[str] = "Mean target detection rate"
    ORIGINAL_CONFIDENCE: Final[str] = "Original segment confidence"
    NOISY_CONFIDENCE: Final[str] = "Noisy segment mean confidence"
    PAIRED_NOISE: Final[str] = "Original vs noisy target confidence"
    SAMPLE_COUNT: Final[str] = "{label} · showing {shown:,} / {total:,}"
    UNCHANGED: Final[str] = "Unchanged confidence"
    BOTH: Final[str] = "Both targets"
    ONLY_A: Final[str] = "Only target A"
    ONLY_B: Final[str] = "Only target B"
    NEITHER: Final[str] = "Neither target"
    OVERLAP_OUTCOMES: Final[str] = (
        "Overlap: joint target recovery at confidence ≥ {threshold:g}"
    )
    PAIR_PROPORTION: Final[str] = "Proportion of recording pairs"
    SPECIES_COVERAGE: Final[str] = (
        "Species representation: top {shown} of {total:,} supported targets"
    )
    ANALYSED_RECORDINGS: Final[str] = "Successfully analysed recordings"
    DURATION_CONFIDENCE: Final[str] = "Recording duration vs target confidence"
    RECORDING_DURATION: Final[str] = (
        "Original recording duration (s, log scale)"
    )
    TARGET_MAXIMUM: Final[str] = "Maximum full-file target confidence"
    BATCH_HISTORY: Final[str] = "Cumulative estimates after each batch"
    BATCH_NUMBER: Final[str] = "Batch number (last batch may be partial)"
    CUMULATIVE_COUNT: Final[str] = "{condition} (latest n={count:,})"
    CUMULATIVE_CONFIDENCE: Final[str] = "Cumulative mean target confidence"
    CUMULATIVE_RATE: Final[str] = "Cumulative mean target detection rate"


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
    NOISE_DISTRIBUTION: Final[str] = (
        "Each observation is one recording's mean across noise trials. "
        "Boxes show the median and middle 50%; whiskers span the observed "
        "minimum and maximum. Only recordings with a selected target "
        "segment enter these noise charts."
    )
    SPECIES_NOISE: Final[str] = (
        "The chart shows up to 20 species with the most noise-eligible "
        "recordings. Rates give each recording equal weight; n is the "
        "number of recordings in that cell. Small groups are descriptive "
        "and need more recordings for a stable estimate."
    )
    NOISE_PAIRS: Final[str] = (
        "Each point pairs a recording's original segment with its mean "
        "noisy confidence. Points below the diagonal indicate a decrease. "
        "Up to 5,000 recordings per condition are selected deterministically "
        "across processing order. Confidence is not a measure of accuracy."
    )
    OVERLAP_OUTCOMES: Final[str] = (
        "The four outcomes use the minimum confidence threshold saved for "
        "the experiment. Each mixture counts a pair once; n is its pair "
        "count. A and B describe roles within each pair."
    )
    SPECIES_COVERAGE: Final[str] = (
        "Counts cover successfully analysed recordings with an identified "
        "primary species supported by BirdNET, including undetected targets. "
        "The chart shows the 20 most represented species; the CSV includes "
        "every supported target species."
    )
    DURATION_CONFIDENCE: Final[str] = (
        "Each point relates the original recording's duration to the "
        "maximum exported confidence for its supported primary species. "
        "Zero indicates that the target was absent from exported predictions. "
        "Up to 5,000 recordings are selected across processing order. "
        "Longer recordings offer more windows for a maximum; this plot "
        "describes an association rather than a causal effect."
    )
    BATCH_HISTORY: Final[str] = (
        "Curves pool all measured recordings up to each batch, giving each "
        "recording equal weight. The latest batch includes its committed "
        "recordings even when it is incomplete. Counts refer to "
        "noise-eligible recordings. Stability does not establish that the "
        "processed subset represents the complete catalogue."
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
    FIGURE_ALTERNATIVE_TEXT: Final[str] = (
        "Cumulative Xeno-canto results: {section}."
    )
