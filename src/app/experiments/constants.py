"""Name the fixed annotated experiment and statistical sampling settings."""

from enum import StrEnum
from typing import Final

from app.domain.constants import ResultDirectory
from app.domain.settings import ExperimentSettings


class StatisticalSetting:
    """Preserve the sampling parameters of the saved measurements."""

    REPETITIONS: Final[int] = 2_000
    RANDOM_SEED: Final[int] = ExperimentSettings().random_seed
    MINIMUM_REPETITIONS: Final[int] = 100
    CHUNK_SIZE: Final[int] = 100
    INTERVAL_QUANTILES: Final[tuple[float, float]] = (0.025, 0.975)


class SoundscapeSetting:
    """Identify the immutable dataset revision and inference protocol."""

    DIRECTORY: Final[str] = ResultDirectory.BENCHMARK
    REVISION: Final[str] = "806ed2cda4ddcbe6efa194ccafff930aa0e557ce"
    BASE_URL: Final[str] = (
        "https://huggingface.co/datasets/DBD-research-group/BirdSet/resolve/"
        + REVISION
    )
    TAXONOMY_URL: Final[str] = (
        "https://api.ebird.org/v2/ref/taxonomy/ebird?fmt=csv&version=2021"
    )
    RANDOM_SEED: Final[int] = 20261008
    THRESHOLDS: Final[tuple[float, ...]] = (
        ExperimentSettings().confidence_thresholds
    )
    SOURCE_WINDOWS: Final[int] = 6
    PROTOCOL_VERSION: Final[int] = 2
    DOWNLOAD_TIMEOUT_SECONDS: Final[int] = 60
    ARCHIVE_FINGERPRINT_SUFFIX: Final[str] = ".sha256"
    ARCHIVE_FINGERPRINT_PATTERN: Final[str] = r"[0-9a-f]{64}"
    INFERENCE_WORKERS: Final[int] = 1
    AUDIO_PRODUCERS: Final[int] = 1
    INFERENCE_BATCH_SIZE: Final[int] = 16
    EXPORT_THRESHOLD: Final[float] = 0.0
    AGGREGATION: Final[str] = (
        "max over model windows and outputs of reference concept"
    )
    PACKAGES: Final[tuple[str, ...]] = (
        "birdnet",
        "onnxruntime",
        "ai-edge-litert",
        "numpy",
        "pandas",
        "soundfile",
        "pyarrow",
    )


class SoundscapeDataset(StrEnum):
    """Name the two fixed public test subsets."""

    NBP = "NBP"
    HSN = "HSN"


class SoundscapeVersion(StrEnum):
    """Name the compared model versions in their original evaluation order."""

    V2_4 = "2.4"
    V3_0 = "3.0"


class EvaluationScope(StrEnum):
    """Distinguish the full reference vocabulary from shared model classes."""

    ALL_REFERENCE_CLASSES = "all_reference_classes"
    COMMON_CLASSES = "common_classes"


class MetricColumn(StrEnum):
    """Share metric identifiers between computation and presentation."""

    TP = "tp"
    FP = "fp"
    FN = "fn"
    PRECISION_MICRO = "precision_micro"
    RECALL_MICRO = "recall_micro"
    F1_MICRO = "f1_micro"
    F1_MACRO = "f1_macro"
    F1_MICRO_LOW = "f1_micro_low"
    F1_MICRO_HIGH = "f1_micro_high"
    SUPPORTED_CLASSES = "supported_classes"
    NEGATIVE_CLIPS = "negative_clips"
    NEGATIVE_CLIP_FALSE_ALARM_RATE = "negative_clip_false_alarm_rate"
    F1_DELTA_3_MINUS_2 = "f1_delta_3_minus_2"


# eBird 2021 reference concepts may contain later split model classes.
TAXON_CONCEPTS: Final[dict[str, tuple[str, ...]]] = {
    "amepip": ("Anthus rubescens", "Anthus japonicus"),
    "foxspa": (
        "Passerella iliaca",
        "Passerella unalaschcensis",
        "Passerella megarhyncha",
        "Passerella schistacea",
    ),
    "yelwar": ("Setophaga petechia", "Setophaga aestiva"),
    "yerwar": (
        "Setophaga coronata",
        "Setophaga auduboni",
        "Setophaga goldmani",
    ),
    "norfli": ("Colaptes auratus", "Colaptes mexicanoides"),
    "comcha": (
        "Fringilla coelebs",
        "Fringilla canariensis",
        "Fringilla maderensis",
        "Fringilla moreletti",
        "Fringilla spodiogenys",
    ),
    "eurgol": ("Carduelis carduelis", "Carduelis caniceps"),
    "eurnut2": ("Sitta europaea", "Sitta arctica"),
    "spofly1": ("Muscicapa striata", "Muscicapa tyrrhenica"),
}


class ExperimentColumn(StrEnum):
    """Name stored benchmark columns and measurement provenance fields."""

    AGGREGATION = "aggregation"
    ANALYSIS_AT = "analysis_at"
    ARCHIVE_SHA256 = "archive_sha256"
    AUDIO_HOURS = "audio_hours"
    AUDIO_SHA256 = "audio_sha256"
    BACKEND = "backend"
    BINDING = "binding"
    BOOTSTRAP_REPETITIONS = "bootstrap_repetitions"
    BOOTSTRAP_SEED = "bootstrap_seed"
    CLASSES = "classes"
    CLIPS = "clips"
    COMPLETED_AT = "completed_at"
    COUNTRIES = "countries"
    COUNTRY = "country"
    DATASET = "dataset"
    DATASET_REVISION = "dataset_revision"
    DATASETS = "datasets"
    DEFAULT_CONFIDENCE_THRESHOLD = "default_confidence_threshold"
    DELTA_A = "delta_a"
    DELTA_B = "delta_b"
    EBIRD_CODE_MULTILABEL = "ebird_code_multilabel"
    ENVIRONMENT = "environment"
    ESTIMATE = "estimate"
    EXCLUSIONS = "exclusions"
    FILE = "file"
    FILEPATH = "filepath"
    HIGH = "high"
    INFERENCE = "inference"
    INTEGRITY_CHECKS = "integrity_checks"
    JOINT_RECOVERY = "joint_recovery"
    LOW = "low"
    MANIFEST_SHA256 = "manifest_sha256"
    MAPPING = "mapping"
    METADATA_SHA256 = "metadata_sha256"
    MODELS = "models"
    N_PRODUCERS = "n_producers"
    N_WORKERS = "n_workers"
    NOISE_ELIGIBLE = "noise_eligible"
    NOISE_SKIP_REASON = "noise_skip_reason"
    NOISE_SPECIES = "noise_species"
    NONCOMMON_CLASSES = "noncommon_classes"
    OVERLAP_PAIRS = "overlap_pairs"
    PACKAGES = "packages"
    PAIRED_DELTA = "paired_delta"
    PLATFORM = "platform"
    PREDICTION_SETTINGS = "prediction_settings"
    PROTOCOL = "protocol"
    PYTHON = "python"
    QUALITY = "quality"
    QUANTITY = "quantity"
    RECORDING_A = "recording_a"
    RECORDING_B = "recording_b"
    RECORDIST = "recordist"
    RECORDISTS = "recordists"
    REFERENCE_CLASSES = "reference_classes"
    SCOPE = "scope"
    SCORES_SHA256 = "scores_sha256"
    SHA256 = "sha256"
    SOURCE = "source"
    SOURCE_RECORDING = "source_recording"
    SOURCE_RECORDINGS = "source_recordings"
    SPECIES = "species"
    SPECIES_LT5 = "species_lt5"
    SPECIES_SINGLETON = "species_singleton"
    STARTED_AT = "started_at"
    STATUS = "status"
    SUPPORTED_RECORDINGS = "supported_recordings"
    SUPPORTED_SPECIES = "supported_species"
    TARGET_DETECTED = "target_detected"
    TAXONOMY_SHA256 = "taxonomy_sha256"
    TOP_K = "top_k"
    TRAINING_INDEPENDENCE_VERIFIED = "training_independence_verified"
    UNDETECTED_SUPPORTED = "undetected_supported"
    UNSUPPORTED_OR_UNIDENTIFIED = "unsupported_or_unidentified"
    WINDOWS_SHA256 = "windows_sha256"
    REFERENCE_EBIRD_CODES = "reference_ebird_codes"
    THRESHOLDS = "thresholds"


class TaxonomyField(StrEnum):
    """Identify fields in the pinned eBird reference taxonomy."""

    SPECIES_CODE = "SPECIES_CODE"
    SCIENTIFIC_NAME = "SCIENTIFIC_NAME"
    COMMON_NAME = "COMMON_NAME"


class SoundscapeFile:
    """Share URLs and result-root paths between inference and publication."""

    METADATA_URL: Final[str] = (
        "{base}/{dataset}/{dataset}_metadata_test_5s.parquet"
    )
    AUDIO_URL: Final[str] = (
        "{base}/{dataset}/{dataset}_test5s_shard_0001.tar.gz"
    )
    MANIFEST: Final[str] = (
        f"{ResultDirectory.BENCHMARK}/{{dataset}}/manifest.csv"
    )
    REFERENCE: Final[str] = (
        f"{ResultDirectory.BENCHMARK}/{{dataset}}/reference.csv"
    )
    SCORES: Final[str] = (
        f"{ResultDirectory.BENCHMARK}/{{dataset}}/{{version}}/scores.csv"
    )
    WINDOWS: Final[str] = (
        f"{ResultDirectory.BENCHMARK}/{{dataset}}/{{version}}/windows.csv"
    )
    INFERENCE: Final[str] = (
        f"{ResultDirectory.BENCHMARK}/{{dataset}}/{{version}}/inference.json"
    )
    SCORE_FINGERPRINT: Final[str] = "scores_{version}_sha256"


class ValidationSetting:
    """Freeze source calibration independently of model outputs."""

    CALIBRATION_FRACTION: Final[float] = 0.25
    THRESHOLDS: Final[tuple[float, ...]] = tuple(i / 20 for i in range(1, 20))
    DIRECTORY: Final[str] = "benchmark/validation"
    SPLIT: Final[str] = "split.csv"
    METRICS: Final[str] = "metrics.csv"
    PAIRED: Final[str] = "paired.csv"
    PROVENANCE: Final[str] = "provenance.json"
    CALIBRATION: Final[str] = "calibration"
    TEST: Final[str] = "test"
    PARTITION: Final[str] = "partition"
    CALIBRATION_F1: Final[str] = "calibration_f1"
    CALIBRATION_SOURCES: Final[str] = "calibration_sources"


class ControlSetting:
    """Define annotation-based perturbations before examining predictions."""

    DIRECTORY: Final[str] = "controlled"
    PROTOCOL_VERSION: Final[int] = 2
    POSITIVE_SOURCES: Final[int] = 40
    BACKGROUND_SOURCES: Final[int] = 20
    MINIMUM_SECONDS: Final[float] = 3.0
    BACKGROUND_SECONDS: Final[float] = 5.0
    SAMPLE_RATE: Final[int] = 48_000
    RMS: Final[float] = 0.1
    REPETITIONS: Final[int] = ExperimentSettings().noise_repetitions
    SNR_VALUES: Final[tuple[float, ...]] = ExperimentSettings().snr_values_db
    MIX_RATIOS: Final[tuple[tuple[float, float], ...]] = (
        (0.5, 0.5),
        (0.75, 0.25),
        (0.9, 0.1),
    )
    NOISE: Final[str] = "noise"
    OVERLAP: Final[str] = "overlap"
    CLEAN: Final[str] = "clean"
    GAUSSIAN: Final[str] = "gaussian"
    NATURAL: Final[str] = "natural"
    FIRST: Final[str] = "first_alone"
    SECOND: Final[str] = "second_alone"
    MIXED: Final[str] = "mixed"
    MANIFEST: Final[str] = "manifest.csv"
    SCORES: Final[str] = "scores.csv"
    SUMMARY: Final[str] = "summary.csv"
    MEASUREMENTS: Final[str] = "measurements.csv"
    PROVENANCE: Final[str] = "provenance.json"


class ControlColumn(StrEnum):
    """Bind every controlled score to its inputs and repeated unit."""

    CONDITION = "condition"
    SNR = "snr_db"
    TRIAL = "trial"
    UNIT = "unit"
    GROUP = "group"
    FIRST_FILE = "first_file"
    SECOND_FILE = "second_file"
    FIRST_SOURCE = "first_source"
    SECOND_SOURCE = "second_source"
    FIRST_TARGET = "first_target"
    SECOND_TARGET = "second_target"
    BACKGROUND = "background"
    BACKGROUND_SOURCE = "background_source"
    RATIO = "ratio"
    RECOVERY = "recovery"
    CLEAN_RECOVERY = "clean_recovery"
    DELTA = "delta"
    CONFIDENCE = "confidence"
    ESTIMATE = "estimate"
    LOW = "low"
    HIGH = "high"
    UNITS = "units"
    GROUPS = "groups"
