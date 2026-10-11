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
    """State a short observation for each diagnostic result."""

    PARTIAL_COVERAGE: Final[str] = (
        "\n\n**Observation.** Collection is still partial; the results"
        " describe recordings already processed."
    )
    COMPLETE_COVERAGE: Final[str] = (
        "\n\n**Observation.** Every recording in the catalogue has been"
        " attempted."
    )
    PROGRESS: Final[str] = (
        "**Observation.** Much of the catalogue remains to be processed;"
        " the analysed corpus does not yet cover all collected recordings."
    )
    SPECTROGRAM: Final[str] = (
        "**Observation.** The spectrogram shows the time-frequency"
        " distribution of a segment; it does not validate its species."
    )
    NOISE: Final[str] = (
        "**Observation.** Mean confidence decreases from {original:.3f} to"
        " {noisy:.3f} under the strongest noise. This measurement covers"
        " initially recognised targets; {skipped} recordings are excluded"
        " from the noise experiment."
    )
    OVERLAP: Final[str] = (
        "**Observation.** As the contribution of B decreases from 50 to"
        " 10%, its mean confidence changes from {first:.3f} to {last:.3f}."
        " The weaker target is more strongly masked."
    )
    THRESHOLD: Final[str] = (
        "**Observation.** Raising the threshold from {first_threshold:.2f}"
        " to {last_threshold:.2f} reduces retained predictions from"
        " {first_detections:,} to {last_detections:,}. These predictions"
        " are not a validated species inventory."
    )

    NOISE_DISTRIBUTION: Final[str] = (
        "**Observation.** Noise shifts scores towards lower values, with"
        " substantial differences between recordings."
    )
    SPECIES_NOISE: Final[str] = (
        "**Observation.** The represented species respond differently to"
        " noise; their recording counts also differ."
    )
    NOISE_PAIRS: Final[str] = (
        "**Observation.** Many points fall below the diagonal as noise"
        " increases: their score becomes lower than the original."
    )
    OVERLAP_OUTCOMES: Final[str] = (
        "**Observation.** In unbalanced mixtures, recovering only A"
        " becomes more frequent than recovering both targets."
    )
    SPECIES_COVERAGE: Final[str] = (
        "**Observation.** A few species account for a large share of"
        " recordings, while many are sparsely represented."
    )
    DURATION_CONFIDENCE: Final[str] = (
        "**Observation.** Longer recordings tend to have a higher maximum"
        " score; they also provide more analysis windows."
    )
    BATCH_HISTORY: Final[str] = (
        "**Observation.** Estimates change as recordings are added, and"
        " the 0 dB condition remains the most difficult."
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
        "Bird acoustic recognition experiments"
    )
    FIGURE_ALTERNATIVE_TEXT: Final[str] = (
        "Cumulative Xeno-canto results: {section}."
    )
    UNKNOWN_EXPERIMENT: Final[str] = "Unknown experiment result: {section}"
    EXPERIMENT_ALTERNATIVE_TEXT: Final[str] = "Experimental results: {section}"


class AnalysisError:
    """Reject inconsistent saved measurements before publishing statistics."""

    INVALID_NOISE_COVERAGE: Final[str] = (
        "Noise conditions must cover the same unique recordings."
    )
    INVALID_REPETITIONS: Final[str] = (
        "Noise trial counts do not match the saved experiment."
    )
    INVALID_EXCLUSIONS: Final[str] = (
        "Noise eligibility and exclusions do not cover supported targets."
    )
    INVALID_CONFIDENCE: Final[str] = "Noise confidences must be within [0, 1]."
    INCONSISTENT_MEANS: Final[str] = (
        "Saved recording means do not match their noise trials."
    )
    INVALID_PAIRS: Final[str] = (
        "Overlap mixtures must use the same disjoint recording pairs."
    )


class ExperimentLabel:
    """Centralize experiment table headings and graph labels."""

    CONDITION: Final[str] = "Condition"
    SOURCE_SPLIT: Final[str] = "Sources (calibration / test)"
    CONTROL_GROUPS: Final[str] = "Groups"
    CONTROL_UNITS: Final[str] = "Units"
    PAIRED_POINTS: Final[str] = "Paired difference (points) [CI]"
    ISOLATED_RECOVERY: Final[str] = "Isolated controls"
    RECORDINGS: Final[str] = "Recordings"
    SPECIES: Final[str] = "Species"
    SCORE_INTERVAL: Final[str] = "Score [CI]"
    RECOVERY_INTERVAL: Final[str] = "Recovery [CI]"
    PAIRED_INTERVAL: Final[str] = "Paired difference [CI]"
    SPECIES_SCORE_INTERVAL: Final[str] = "Species score [CI]"
    SPECIES_RECOVERY_INTERVAL: Final[str] = "Species recovery [CI]"
    SPECIES_CLUSTER_INTERVAL: Final[str] = "Species-cluster difference [CI]"
    THRESHOLD: Final[str] = "Threshold"
    FILE_INTERVAL: Final[str] = "Per recording [CI]"
    SPECIES_INTERVAL: Final[str] = "Per species [CI]"
    VERSION: Final[str] = "Version"
    PRECISION: Final[str] = "Precision"
    RECALL: Final[str] = "Recall"
    F1_INTERVAL: Final[str] = "Micro F1 [CI]"
    F1_MACRO: Final[str] = "Macro F1"
    TP: Final[str] = "TP"
    FP: Final[str] = "FP"
    FN: Final[str] = "FN"
    NEGATIVE_FALSE_ALARMS: Final[str] = "False alarms on negative clips"
    CLIPS: Final[str] = "Clips"
    SOURCES: Final[str] = "Sources"
    POSITIVE_CLASSES: Final[str] = "Classes with positives"
    NEGATIVE_CLIPS: Final[str] = "Negative clips"
    DATASET: Final[str] = "Dataset"
    F1_DIFFERENCE: Final[str] = "F1 difference (3.0 - 2.4)"
    INTERVAL: Final[str] = "95% CI"
    COUNTRY: Final[str] = "Country"
    QUALITY_XENO_CANTO: Final[str] = "Xeno-canto quality"
    FILE: Final[str] = "Per recording"
    SPECIES_WEIGHTED: Final[str] = "Per species"
    MIX: Final[str] = "Mixture"
    BOTH_INTERVAL: Final[str] = "Both targets [95% CI]"
    CLUSTERS_INTERVAL: Final[str] = "Species clusters [CI]"
    QUALITY: Final[str] = "Quality"
    RECOVERY_95_INTERVAL: Final[str] = "Recovery [95% CI]"
    PREDICTIONS: Final[str] = "Predictions"
    DISTINCT_LABELS: Final[str] = "Distinct labels"
    TARGET_RECOVERED: Final[str] = "Target recovered"
    MINIMUM: Final[str] = "Minimum"
    Q1: Final[str] = "Q1"
    MEDIAN: Final[str] = "Median"
    Q3: Final[str] = "Q3"
    MAXIMUM: Final[str] = "Maximum"
    BOTH: Final[str] = "Both targets"
    ONLY_A: Final[str] = "Only A"
    ONLY_B: Final[str] = "Only B"
    NEITHER: Final[str] = "Neither"
    F1_95_INTERVAL: Final[str] = "Micro F1 [95% CI]"
    COMMON_FALSE_ALARMS: Final[str] = "False alarms (15 clips)"
    WITH_FALSE_ALARM: Final[str] = "With false alarm"
    F1_MICRO: Final[str] = "Micro F1"
    F1_DIFFERENCE_INTERVAL: Final[str] = "F1 difference (3.0 - 2.4) [95% CI]"
    CONFIDENCE_INTERVAL: Final[str] = "Confidence [95% CI]"
    PAIRED_DIFFERENCE: Final[str] = "Paired difference"
    MEAN_CONFIDENCE: Final[str] = "Mean confidence"
    ORIGINAL_DIFFERENCE: Final[str] = "Difference from original"
    ADDED_SNR: Final[str] = "Added SNR (dB)"
    CONFIDENCE_THRESHOLD: Final[str] = "Confidence threshold"
    NOISE_FILE_TITLE: Final[str] = "Noise: results per recording, 95% CI"
    NOISE_SPECIES_TITLE: Final[str] = (
        "Noise: equal species weights and clusters, 95% CI"
    )
    THRESHOLD_TITLE: Final[str] = "Thresholds: target recovery, 95% CI"
    PAIRED_TITLE: Final[str] = "Paired model comparison, 95% CI"
    GROUP_TITLE: Final[str] = (
        "Recovery at threshold 0.10, groups with at least 30 recordings"
    )
    METRICS_TITLE: Final[str] = "{dataset}: precision, recall and F1, {scope}"
    COUNTS_TITLE: Final[str] = (
        "{dataset}: clip/species counts and false alarms"
    )
    CONTEXT_TITLE: Final[str] = "{dataset}: evaluated vocabulary counts"
    MODEL_LEGEND: Final[str] = "BirdNET {version}"


class ExperimentObservation:
    """Describe the saved additional results in short notebook observations."""

    PREFIX: Final[str] = "**Observation.** "
    MISSING: Final[str] = "This experiment has not been run yet."
    NOISE_FIGURE: Final[str] = (
        "The score decrease under noise is visible with the 95% intervals."
    )
    COMPARISON: Final[str] = (
        "The model ranking depends on the dataset and decision threshold."
    )
    NOISE_INTERVALS: Final[str] = (
        "Degradation persists when species have equal weights; cluster"
        " intervals are wider."
    )
    THRESHOLDS: Final[str] = (
        "Filtering reduces recovery; the species average remains below the"
        " recording average."
    )
    COMMON_SPECIES: Final[str] = (
        "The advantage of 3.0 on NBP remains visible for the 50 common"
        " classes. The 15 negative clips here are negative only within this"
        " reduced vocabulary."
    )
    SOUNDSCAPES: Final[str] = (
        "3.0 achieves a higher F1 on NBP; on HSN, the result depends on"
        " the threshold."
    )
    PAIRED: Final[str] = (
        "The difference favours 3.0 on NBP and changes sign with the"
        " threshold on HSN."
    )
    COUNTRIES: Final[str] = (
        "Rates vary between countries, but species and conditions also differ."
    )
    QUALITY: Final[str] = (
        "Quality categories have different rates; this comparison remains"
        " descriptive."
    )
