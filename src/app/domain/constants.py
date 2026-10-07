"""Stable identifiers shared by persistence, processing and presentation."""

from enum import StrEnum
from pathlib import Path
from typing import Final

API_KEY_VARIABLE: Final[str] = "XENO_CANTO_API_KEY"
XENO_CANTO_API_URL: Final[str] = "https://xeno-canto.org/api/3/recordings"
USER_AGENT: Final[str] = "bird-bioacoustics/0.1 (academic audio experiments)"
BIRDNET_DISTRIBUTION: Final[str] = "birdnet"
PROTOCOL_VERSION: Final[int] = 4


class ProcessingStatus(StrEnum):
    """Define stable processingstatus values without scattered literals."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class OverlapStatus(StrEnum):
    """Define stable overlapstatus values without scattered literals."""

    PAIRED = "paired"
    SAME_SPECIES_AS_PENDING = "same_species_as_pending"


class MetadataKey(StrEnum):
    """Define stable metadatakey values without scattered literals."""

    SIGNATURE = "signature"
    CONFIGURATION = "configuration"
    QUERY = "query"
    CATALOGUE_COMPLETE = "catalogue_complete"
    CREATED_AT = "created_at"
    NEXT_PAGE = "next_page"
    CATALOGUE_GENERATION = "catalogue_generation"
    PAGES = "pages"
    EXPECTED_RECORDINGS = "expected_recordings"
    EXPECTED_SPECIES = "expected_species"
    CURSOR = "cursor"
    BATCH_SIZE = "batch_size"
    CURRENT_BATCH = "current_batch"
    MODEL = "model"
    EXPERIMENT = "experiment"
    PROTOCOL_VERSION = "protocol_version"


class CoverageKey(StrEnum):
    """Define stable coveragekey values without scattered literals."""

    PHASE = "phase"
    SNAPSHOT_AT = "snapshot_at"
    CATALOGUED_RECORDINGS = "catalogued_recordings"
    PROCESSED_RECORDINGS = "processed_recordings"
    FAILED_RECORDINGS = "failed_recordings"
    PENDING_RECORDINGS = "pending_recordings"
    ALL_RECORDINGS_ATTEMPTED = "all_recordings_attempted"
    COMPLETED_BATCHES = "completed_batches"
    DOWNLOADED_BYTES = "downloaded_bytes"
    PROCESSED_AUDIO_SECONDS = "processed_audio_seconds"


class ExperimentPhase(StrEnum):
    """Define stable experimentphase values without scattered literals."""

    PROCESSING = "processing"
    READY_FOR_NEXT_BATCH = "ready_for_next_batch"
    CATALOGUE = "catalogue"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    COMPLETE = "complete"


class PredictionColumn(StrEnum):
    """Define stable predictioncolumn values without scattered literals."""

    INPUT = "input"
    START_TIME = "start_time"
    END_TIME = "end_time"
    SPECIES_NAME = "species_name"
    CONFIDENCE = "confidence"


class NoiseCondition(StrEnum):
    """Define stable noisecondition values without scattered literals."""

    ORIGINAL = "original"


class CatalogueStatus(StrEnum):
    """Define stable cataloguestatus values without scattered literals."""

    IDENTIFIED = "identified"


class FileName(StrEnum):
    """Define stable filename values without scattered literals."""

    CHECKPOINT = "database.sqlite3"
    ENVIRONMENT = ".env"
    ANNOTATIONS = "annotations.csv"
    NOTEBOOK = "notebook.ipynb"


class Artifact(StrEnum):
    """Name public result artifacts consistently across exports and plots."""

    NOISE = "experiment_noise"
    OVERLAP = "experiment_overlap"
    THRESHOLD = "experiment_threshold"
    PROGRESS = "corpus_progress"
    STATUS = "corpus_status"
    SPECTROGRAM = "target_spectrogram"
    BASELINE = "baseline_predictions"
    NOTEBOOK = "notebook"

    def filename(self, suffix: str) -> str:
        """Build one artifact filename.

        :param suffix: File extension without a leading dot.
        :type suffix: str
        :return: Stable output filename.
        :rtype: str
        """
        return f"{self.value}.{suffix}"

    def path(self, directory: Path, suffix: str = "png") -> Path:
        """Locate an artifact in an explicitly supplied results directory.

        :param directory: Output directory.
        :type directory: Path
        :param suffix: File extension without a leading dot.
        :type suffix: str
        :return: Artifact destination.
        :rtype: Path
        """
        return directory / self.filename(suffix)


class ResultColumn(StrEnum):
    """Share the scientific table columns between measurements and
    presentation.
    """

    RECORDING_ID = "recording_id"
    TRIAL = "trial"
    SEED = "seed"
    RANK = "rank"
    CONDITION = "condition"
    SNR_DB = "snr_db"
    TARGET_SPECIES = "target_species"
    CONFIDENCE_MEAN = "confidence_mean"
    CONFIDENCE_STD = "confidence_std"
    NOISE_STD_MEAN = "noise_std_mean"
    DETECTION_RATE = "detection_rate"
    RECORDINGS = "recordings"
    TRIALS = "trials"
    MIX = "mix"
    WEIGHT_A = "weight_a"
    WEIGHT_B = "weight_b"
    SPECIES_A = "species_a"
    SPECIES_B = "species_b"
    CONFIDENCE_A = "confidence_a"
    CONFIDENCE_B = "confidence_b"
    CONFIDENCE_A_MEAN = "confidence_a_mean"
    CONFIDENCE_B_MEAN = "confidence_b_mean"
    CONFIDENCE_A_STD = "confidence_a_std"
    CONFIDENCE_B_STD = "confidence_b_std"
    PAIRS = "pairs"
    THRESHOLD = "threshold"
    DETECTIONS = "detections"
    DISTINCT_SPECIES = "distinct_species"
    TARGET_RECORDINGS = "target_recordings"
    TARGET_DETECTION_RATE = "target_detection_rate"
    ANNOTATED_RECORDINGS = "annotated_recordings"
    TRUE_POSITIVES = "true_positives"
    FALSE_POSITIVES = "false_positives"
    FALSE_NEGATIVES = "false_negatives"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"


class MetadataFlag(StrEnum):
    """Preserve the checkpoint boolean encoding used by existing databases."""

    TRUE = "true"


class RuntimeModule(StrEnum):
    """Identify runtime boundaries without scattering module-name strings."""

    BIRDNET = "birdnet"
    ONNX = "onnxruntime"
    SYSTEM = "sys"


class InferenceDevice(StrEnum):
    """Name supported CPU and single-GPU execution targets."""

    CPU = "CPU"
    GPU = "GPU:0"


class ExecutionProvider(StrEnum):
    """Name ONNX providers used to verify the requested hardware."""

    CPU = "CPUExecutionProvider"
    CUDA = "CUDAExecutionProvider"


class OnnxOption:
    """Share the CUDA provider configuration with preparation checks."""

    DEVICE_ID: Final[str] = "device_id"
    FIRST_GPU: Final[str] = "0"
