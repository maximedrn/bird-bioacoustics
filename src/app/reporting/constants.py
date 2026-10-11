"""Name diagnostic columns and bound cumulative report presentation."""

from enum import StrEnum
from pathlib import Path
from typing import Final

from app.domain.constants import ResultDirectory
from app.reporting.messages import ExperimentLabel


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


class ExperimentFile(StrEnum):
    """Name cached inputs and relative experiment artifact paths.

    LaTeX fragments belong under report_tables; other outputs use results.
    """

    COUNTRY_RECOVERY_CSV = f"{ResultDirectory.CORPUS}/country_recovery.csv"
    COUNTRY_TABLE_TEX = f"{ResultDirectory.CORPUS}/country_table.tex"
    QUALITY_RECOVERY_CSV = f"{ResultDirectory.CORPUS}/quality_recovery.csv"
    QUALITY_TABLE_TEX = f"{ResultDirectory.CORPUS}/quality_table.tex"
    CORPUS_PROVENANCE_JSON = f"{ResultDirectory.CORPUS}/provenance.json"
    NUMBERS_TEX = f"{ResultDirectory.CORPUS}/numbers.tex"

    ALL_THRESHOLDS_TABLE_TEX = (
        f"{ResultDirectory.BENCHMARK}/all_thresholds_table.tex"
    )
    AUDIO_TAR_GZ = "audio.tar.gz"
    BENCHMARK_F1_PDF = f"{ResultDirectory.BENCHMARK}/benchmark_f1.pdf"
    BENCHMARK_F1_PNG = f"{ResultDirectory.BENCHMARK}/benchmark_f1.png"
    BENCHMARK_TABLE_TEX = f"{ResultDirectory.BENCHMARK}/benchmark_table.tex"
    BENCHMARK_PROVENANCE_JSON = f"{ResultDirectory.BENCHMARK}/provenance.json"
    COMMON_SPECIES_COUNTS_TABLE_TEX = (
        f"{ResultDirectory.BENCHMARK}/common_species_counts_table.tex"
    )
    COMMON_SPECIES_TABLE_TEX = (
        f"{ResultDirectory.BENCHMARK}/common_species_table.tex"
    )
    EBIRD_2021_CSV = "ebird_2021.csv"
    METADATA_PARQUET = "metadata.parquet"
    METRICS_CSV = f"{ResultDirectory.BENCHMARK}/metrics.csv"
    NEGATIVE_CLIPS_TABLE_TEX = (
        f"{ResultDirectory.BENCHMARK}/negative_clips_table.tex"
    )
    NOISE_DISTRIBUTION_TABLE_TEX = (
        f"{ResultDirectory.NOISE}/noise_distribution_table.tex"
    )
    NOISE_INTERVALS_CSV = f"{ResultDirectory.NOISE}/noise_intervals.csv"
    NOISE_INTERVALS_PDF = f"{ResultDirectory.NOISE}/noise_intervals.pdf"
    NOISE_INTERVALS_PNG = f"{ResultDirectory.NOISE}/noise_intervals.png"
    NOISE_MACRO_TABLE_TEX = f"{ResultDirectory.NOISE}/noise_macro_table.tex"
    NOISE_TABLE_TEX = f"{ResultDirectory.NOISE}/noise_table.tex"
    NOISE_UNCERTAINTY_TABLE_TEX = (
        f"{ResultDirectory.NOISE}/noise_uncertainty_table.tex"
    )
    OVERLAP_INTERVALS_CSV = f"{ResultDirectory.OVERLAP}/overlap_intervals.csv"
    OVERLAP_OUTCOMES_TABLE_TEX = (
        f"{ResultDirectory.OVERLAP}/overlap_outcomes_table.tex"
    )
    OVERLAP_TABLE_TEX = f"{ResultDirectory.OVERLAP}/overlap_table.tex"
    PAIRED_DIFFERENCE_TABLE_TEX = (
        f"{ResultDirectory.BENCHMARK}/paired_difference_table.tex"
    )
    PAIRED_MODEL_DIFFERENCES_CSV = (
        f"{ResultDirectory.BENCHMARK}/paired_model_differences.csv"
    )
    THRESHOLD_COUNTS_TABLE_TEX = (
        f"{ResultDirectory.THRESHOLD}/threshold_counts_table.tex"
    )
    THRESHOLD_INTERVALS_CSV = (
        f"{ResultDirectory.THRESHOLD}/threshold_intervals.csv"
    )
    THRESHOLD_TABLE_TEX = f"{ResultDirectory.THRESHOLD}/threshold_table.tex"


class ExperimentSection(StrEnum):
    """Identify additional result cells independently of their order."""

    NOISE_INTERVALS = "noise_intervals"
    NOISE_INTERVAL_FIGURE = "noise_interval_figure"
    THRESHOLD_INTERVALS = "threshold_intervals"
    SOUNDSCAPE_METRICS = "soundscape_metrics"
    COMMON_SPECIES_METRICS = "common_species_metrics"
    SOUNDSCAPE_COMPARISON = "soundscape_comparison"
    PAIRED_MODELS = "paired_models"
    COUNTRIES = "countries"
    QUALITY = "quality"
    SOURCE_VALIDATION = "source_validation"
    CONTROLLED_NOISE = "controlled_noise"
    CONTROLLED_NOISE_FIGURE = "controlled_noise_figure"
    CONTROLLED_OVERLAP = "controlled_overlap"
    CONTROLLED_OVERLAP_FIGURE = "controlled_overlap_figure"


class AnalysisSetting:
    """Bound descriptive groups and preserve stored measurement tolerances."""

    PASSED: Final[str] = "passed"
    MINIMUM_GROUP_RECORDINGS: Final[int] = 30
    CONFIDENCE_RTOL: Final[float] = 1e-10
    CONFIDENCE_ATOL: Final[float] = 1e-12
    FIGURE_DPI: Final[int] = 180


class NotebookKey(StrEnum):
    """Name notebook fields and the stable presentation section marker."""

    CELLS = "cells"
    CELL_TYPE = "cell_type"
    METADATA = "metadata"
    OUTPUTS = "outputs"
    EXECUTION_COUNT = "execution_count"
    EXECUTION = "execution"
    SCROLLED = "scrolled"
    JUPYTER = "jupyter"
    OUTPUTS_HIDDEN = "outputs_hidden"
    REPORT_SECTION = "app_report"
    ALTERNATIVE_TEXT = "alt"
    FILENAME = "filename"


class NotebookContent:
    """Declare cell, output and MIME types supported by report exporters."""

    CODE: Final[str] = "code"
    DISPLAY_DATA: Final[str] = "display_data"
    PNG: Final[str] = "image/png"
    HTML: Final[str] = "text/html"
    MARKDOWN: Final[str] = "text/markdown"
    TEXT: Final[str] = "text/plain"


class TableFormat:
    """Share stable output formatting without changing measured precision."""

    DECIMAL: Final[str] = ".3f"
    SIGNED: Final[str] = "+.3f"
    PERCENT: Final[str] = ".1%"
    LOW_SUFFIX: Final[str] = "_low"
    HIGH_SUFFIX: Final[str] = "_high"


EXPERIMENT_FIGURES: Final[dict[str, Path]] = {
    ExperimentSection.CONTROLLED_NOISE_FIGURE: Path(
        "controlled/noise/comparison.png"
    ),
    ExperimentSection.CONTROLLED_OVERLAP_FIGURE: Path(
        "controlled/overlap/comparison.png"
    ),
    ExperimentSection.NOISE_INTERVAL_FIGURE: Path(
        ExperimentFile.NOISE_INTERVALS_PNG
    ),
    ExperimentSection.SOUNDSCAPE_COMPARISON: Path(
        ExperimentFile.BENCHMARK_F1_PNG
    ),
}

EXPERIMENT_TABLES: Final[dict[str, Path]] = {
    ExperimentSection.SOURCE_VALIDATION: Path(
        "benchmark/validation/metrics.csv"
    ),
    ExperimentSection.CONTROLLED_NOISE: Path("controlled/noise/summary.csv"),
    ExperimentSection.CONTROLLED_OVERLAP: Path(
        "controlled/overlap/summary.csv"
    ),
    ExperimentSection.NOISE_INTERVALS: Path(
        ExperimentFile.NOISE_INTERVALS_CSV
    ),
    ExperimentSection.THRESHOLD_INTERVALS: Path(
        ExperimentFile.THRESHOLD_INTERVALS_CSV
    ),
    ExperimentSection.SOUNDSCAPE_METRICS: Path(ExperimentFile.METRICS_CSV),
    ExperimentSection.COMMON_SPECIES_METRICS: Path(ExperimentFile.METRICS_CSV),
    ExperimentSection.PAIRED_MODELS: Path(
        ExperimentFile.PAIRED_MODEL_DIFFERENCES_CSV
    ),
    ExperimentSection.COUNTRIES: Path(ExperimentFile.COUNTRY_RECOVERY_CSV),
    ExperimentSection.QUALITY: Path(ExperimentFile.QUALITY_RECOVERY_CSV),
}


# Keep translations in the LaTeX document and notebook labels in English.
LATEX_TABLE_HEADINGS: Final[dict[str, str]] = {
    ExperimentLabel.SOURCE_SPLIT: r"\ReportSourceSplit",
    ExperimentLabel.CONTROL_GROUPS: r"\ReportControlGroups",
    ExperimentLabel.CONTROL_UNITS: r"\ReportControlUnits",
    ExperimentLabel.PAIRED_POINTS: r"\ReportPairedPoints",
    ExperimentLabel.ISOLATED_RECOVERY: r"\ReportIsolatedRecovery",
    ExperimentLabel.BOTH: r"\ReportBoth",
    ExperimentLabel.BOTH_INTERVAL: r"\ReportBothInterval",
    ExperimentLabel.CLUSTERS_INTERVAL: r"\ReportClustersInterval",
    ExperimentLabel.COMMON_FALSE_ALARMS: r"\ReportCommonFalseAlarms",
    ExperimentLabel.CONDITION: r"\ReportCondition",
    ExperimentLabel.CONFIDENCE_INTERVAL: r"\ReportConfidenceInterval",
    ExperimentLabel.COUNTRY: r"\ReportCountry",
    ExperimentLabel.DATASET: r"\ReportDataset",
    ExperimentLabel.DISTINCT_LABELS: r"\ReportDistinctLabels",
    ExperimentLabel.F1_95_INTERVAL: r"\ReportFOneNineFiveInterval",
    ExperimentLabel.F1_DIFFERENCE_INTERVAL: r"\ReportFOneDifferenceInterval",
    ExperimentLabel.F1_MACRO: r"\ReportFOneMacro",
    ExperimentLabel.F1_MICRO: r"\ReportFOneMicro",
    ExperimentLabel.FILE: r"\ReportFile",
    ExperimentLabel.FN: r"\ReportFn",
    ExperimentLabel.FP: r"\ReportFp",
    ExperimentLabel.MAXIMUM: r"\ReportMaximum",
    ExperimentLabel.MEDIAN: r"\ReportMedian",
    ExperimentLabel.MINIMUM: r"\ReportMinimum",
    ExperimentLabel.MIX: r"\ReportMix",
    ExperimentLabel.NEGATIVE_CLIPS: r"\ReportNegativeClips",
    ExperimentLabel.NEITHER: r"\ReportNeither",
    ExperimentLabel.ONLY_A: r"\ReportOnlyA",
    ExperimentLabel.ONLY_B: r"\ReportOnlyB",
    ExperimentLabel.PAIRED_DIFFERENCE: r"\ReportPairedDifference",
    ExperimentLabel.PAIRED_INTERVAL: r"\ReportPairedInterval",
    ExperimentLabel.PRECISION: r"\ReportPrecision",
    ExperimentLabel.PREDICTIONS: r"\ReportPredictions",
    ExperimentLabel.Q1: r"\ReportQOne",
    ExperimentLabel.Q3: r"\ReportQThree",
    ExperimentLabel.QUALITY: r"\ReportQuality",
    ExperimentLabel.RECALL: r"\ReportRecall",
    ExperimentLabel.RECORDINGS: r"\ReportRecordings",
    ExperimentLabel.RECOVERY_95_INTERVAL: r"\ReportRecoveryNineFiveInterval",
    ExperimentLabel.RECOVERY_INTERVAL: r"\ReportRecoveryInterval",
    ExperimentLabel.SPECIES: r"\ReportSpecies",
    ExperimentLabel.SPECIES_RECOVERY_INTERVAL: r"\ReportSpeciesRecoveryInterval",
    ExperimentLabel.SPECIES_SCORE_INTERVAL: r"\ReportSpeciesScoreInterval",
    ExperimentLabel.SPECIES_WEIGHTED: r"\ReportSpeciesWeighted",
    ExperimentLabel.TARGET_RECOVERED: r"\ReportTargetRecovered",
    ExperimentLabel.THRESHOLD: r"\ReportThreshold",
    ExperimentLabel.TP: r"\ReportTp",
    ExperimentLabel.VERSION: r"\ReportVersion",
    ExperimentLabel.WITH_FALSE_ALARM: r"\ReportWithFalseAlarm",
}
