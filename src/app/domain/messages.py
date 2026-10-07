"""Shared errors and operational messages with explicit template fields."""

from typing import Final


class ErrorMessage:
    """Centralize validation and processing failures without changing their
    wording.
    """

    MISSING_RECORDING_POSITION: Final[str] = (
        "A batched recording is missing its position."
    )

    ACTIVE_WORKER: Final[str] = (
        "Another batch is running for this checkpoint. Reporting remains "
        "available."
    )
    ANNOTATION_OUTSIDE_CATALOGUE: Final[str] = (
        "Annotated recording is outside this catalogue: {identifier}"
    )
    API_ACCESS_REFUSED: Final[str] = (
        "Xeno-canto API access refused (HTTP {code})."
    )
    API_REQUEST_FAILED: Final[str] = (
        "Xeno-canto API request failed (HTTP {code})."
    )
    API_RETRIES_EXHAUSTED: Final[str] = (
        "Xeno-canto API page {page} failed after retries."
    )
    AUDIO_DOWNLOAD_UNAVAILABLE: Final[str] = (
        "Recording {recording_id} is unavailable (HTTP {code})."
    )
    CATALOGUE_CHANGED: Final[str] = (
        "Catalogue changed while paging ({cached}/{expected} IDs); rerun "
        "to reconcile it. Completed batches remain stored."
    )
    CHANGING_AUDIO_LAYOUT: Final[str] = (
        "The channel layout or sample rate changes within the recording."
    )
    DAMAGED_AUDIO_PACKETS: Final[str] = (
        "Damaged packets occur before the end of the recording."
    )
    DECODING_FAILED: Final[str] = (
        "Unable to decode {name} with either decoder."
    )
    DOWNLOAD_RETRIES_EXHAUSTED: Final[str] = (
        "Download of {recording_id} failed after retries. Rerun the batch "
        "to resume."
    )
    DUPLICATE_AUDIO_FILENAMES: Final[str] = (
        "Audio filenames must be distinct within a batch."
    )
    DUPLICATE_SNR_VALUES: Final[str] = "SNR values must be distinct."
    EMPTY_ANNOTATIONS: Final[str] = (
        "Annotation CSV must contain at least one verified recording."
    )
    EMPTY_AUDIO_SIGNAL: Final[str] = "Cannot save an empty audio signal."
    EMPTY_DECODED_AUDIO: Final[str] = "Decoded audio is empty."
    EMPTY_EXPERIMENT_CONDITIONS: Final[str] = (
        "Mix ratios and confidence thresholds must not be empty."
    )
    EMPTY_EXTRACTED_SEGMENT: Final[str] = "Extracted audio segment is empty."
    EMPTY_MIX_SIGNALS: Final[str] = "Cannot mix empty audio signals."
    EMPTY_OUTPUT_STEM: Final[str] = "Output stem must not be empty."
    EMPTY_PREDICTION_INPUTS: Final[str] = (
        "At least one audio path is required."
    )
    EMPTY_TARGET_SEGMENT: Final[str] = "The selected target segment is empty."
    INCOMPATIBLE_CHECKPOINT: Final[str] = (
        "Checkpoint settings differ: use a separate checkpoint for a "
        "changed protocol."
    )
    INCOMPLETE_AUDIO_RESPONSE: Final[str] = (
        "Incomplete audio response for {recording_id}."
    )
    INCOMPLETE_REFERENCE: Final[str] = (
        "Reference annotations must cover all species in a recording."
    )
    INFERENCE_INTERRUPTED: Final[str] = (
        "BirdNET stopped on {identifier}; the recording remains pending. "
        "Rerun the batch to create a new session.\n\n{error}"
    )
    INSUFFICIENT_NOISE_TRIALS: Final[str] = (
        "At least two noise repetitions are required."
    )
    INVALID_ANNOTATION_COLUMNS: Final[str] = (
        "Annotation CSV requires recording_id, species_name, and complete."
    )
    INVALID_API_PAGE: Final[str] = (
        "Xeno-canto did not return a valid catalogue page."
    )
    INVALID_AUDIO_DIMENSIONS: Final[str] = (
        "Audio data must contain one or two dimensions."
    )
    INVALID_BATCH_SIZE: Final[str] = "Batch size must be positive."
    INVALID_CATALOGUE_METADATA: Final[str] = (
        "Invalid catalogue metadata for {identifier}."
    )
    INVALID_CATALOGUE_SETTINGS: Final[str] = (
        "Use a non-empty query and 50-500 results per page."
    )
    INVALID_CONFIDENCE_THRESHOLDS: Final[str] = (
        "Confidence thresholds must be within the exported score range."
    )
    INVALID_CURSOR_ADVANCEMENT: Final[str] = (
        "Cursor advancement requires the next recording's completed outcome."
    )
    INVALID_DETECTION_INTERVAL: Final[str] = (
        "Detection end time must be greater than start time."
    )
    INVALID_MODEL_BACKEND: Final[str] = (
        "Unsupported {version} acoustic backend: {backend}."
    )
    INVALID_INFERENCE_SETTINGS: Final[str] = (
        "Inference workers, audio producers and window batch size must be "
        "positive integers."
    )
    INVALID_INFERENCE_DEVICE: Final[str] = "Choose CPU or GPU:0."
    INVALID_GPU_WORKERS: Final[str] = (
        "Use one inference worker for GPU:0; increase audio producers instead."
    )
    EXCESSIVE_INFERENCE_WORKERS: Final[str] = (
        "Worker and producer counts must not exceed the {cores} CPU cores "
        "available to this runtime."
    )
    UNSUPPORTED_GPU_BACKEND: Final[str] = (
        "GPU execution requires the ONNX backend in this application."
    )
    CUDA_UNAVAILABLE: Final[str] = (
        "CUDA inference is unavailable. Select a Colab GPU runtime and "
        "install onnxruntime-gpu with compatible CUDA/cuDNN libraries, "
        "then rerun model preparation, or choose CPU."
    )
    CUDA_CHECK_TIMEOUT: Final[str] = (
        "CUDA verification timed out before audio processing started. "
        "Restart the GPU runtime and retry model preparation."
    )
    CUDA_CHECK_FAILED: Final[str] = "CUDA verification failed: {error}"
    CUDA_CHECK_CRASHED: Final[str] = (
        "CUDA verification worker exited unexpectedly (exit code {code}). "
        "Restart the GPU runtime and retry model preparation."
    )
    CUDA_SESSION_CLOSED: Final[str] = "The CUDA inference session is closed."
    INVALID_CUDA_PROBE_INPUT: Final[str] = (
        "CUDA verification requires one audio input with two dimensions."
    )
    INVALID_CUDA_PROBE_LENGTH: Final[str] = (
        "CUDA verification requires a positive audio window length. "
        "A dynamic ONNX input uses BirdNET's segment size."
    )
    INVALID_CUDA_PROBE_OUTPUT: Final[str] = (
        "CUDA verification produced invalid scores for {batch_size} windows."
        " Output {output}: shape={shape}, finite={finite}."
    )
    INVALID_PLOT_ARTIFACT: Final[str] = "{artifact} is not a plot artifact."
    INVALID_PROGRESS_INTERVAL: Final[str] = (
        "Progress interval must be positive."
    )
    INVALID_RECORDING_IDENTIFIER: Final[str] = (
        "A recording identifier must be XC followed by digits."
    )
    INVALID_REQUEST_SETTINGS: Final[str] = (
        "Request timing and attempts must be non-negative/positive."
    )
    INVALID_RESAMPLING_RATES: Final[str] = "Sampling rates must be positive."
    INVALID_SAMPLE_RATE: Final[str] = "Sample rate must be greater than zero."
    INVALID_SEGMENT_INTERVAL: Final[str] = (
        "Segment end time must be greater than start time."
    )
    INVALID_SNR_VALUES: Final[str] = "SNR values must be finite and non-empty."
    INVALID_SPECTROGRAM: Final[str] = (
        "A spectrogram needs non-empty audio and a positive sample rate."
    )
    INVALID_TABLE_PREVIEW: Final[str] = (
        "Choose a public table and a positive preview limit."
    )
    INVALID_TARGET_PEAK: Final[str] = (
        "Target peak must be in the interval (0, 1]."
    )
    MISSING_API_KEY: Final[str] = (
        "Set XENO_CANTO_API_KEY in the environment or local .env file."
    )
    MISSING_AUDIO_FILE: Final[str] = "Audio file not found: {audio_path}"
    MISSING_AUDIO_STREAM: Final[str] = "The recording has no audio stream."
    MISSING_AUDIO_URL: Final[str] = (
        "No downloadable audio URL for {identifier}."
    )
    MISSING_CATALOGUE_CLIENT: Final[str] = (
        "A Xeno-canto client is needed to fetch the next catalogue page."
    )
    MISSING_CHECKPOINT: Final[str] = (
        "Run 'bird-bioacoustics batch' to create the experiment checkpoint "
        "first."
    )
    MISSING_DOWNLOADED_AUDIO: Final[str] = (
        "The download service returned no original audio."
    )
    MISSING_PREDICTION_COLUMNS: Final[str] = (
        "Missing BirdNET columns: {missing_text}"
    )
    MISSING_PROJECT_ROOT: Final[str] = (
        "Pass an explicit project root when running outside the project."
    )
    MISSING_RECENT_UPLOADS: Final[str] = (
        "No recent uploads available to freeze the XC identifier range."
    )
    MISSING_STORED_THRESHOLDS: Final[str] = (
        "The checkpoint has no stored confidence thresholds."
    )
    NEGATIVE_MIX_WEIGHTS: Final[str] = "Mix weights must be non-negative."
    NEGATIVE_RANDOM_SEED: Final[str] = "Random seed must be non-negative."
    NON_FINITE_AUDIO: Final[str] = "Decoded audio contains non-finite samples."
    NO_DECODABLE_AUDIO: Final[str] = (
        "The recording contains no decodable audio."
    )
    PROGRESS_NOT_STARTED: Final[str] = "Batch progress has not been started."
    SAME_OVERLAP_RECORDING: Final[str] = (
        "Overlap requires two distinct recordings."
    )
    SAME_OVERLAP_SPECIES: Final[str] = (
        "Overlap requires two distinct target species."
    )
    UNEXPECTED_AUDIO_CONTENT: Final[str] = (
        "Recording {recording_id} returned {content_type} instead of audio."
    )
    UNFINISHED_BATCH: Final[str] = (
        "An unfinished batch cannot be marked complete."
    )
    UNKNOWN_ANNOTATION: Final[str] = (
        "Unknown annotated recording: {recording_id}"
    )
    ZERO_MIX_WEIGHTS: Final[str] = (
        "At least one mix weight must be greater than zero."
    )
    ZERO_NOISE_POWER: Final[str] = "Noise realization has zero power."
    ZERO_SIGNAL_POWER: Final[str] = (
        "Audio segment does not contain usable signal power."
    )


class LogMessage:
    """Centralize worker logs without placing presentation text in services."""

    FORMAT: Final[str] = "%(message)s"

    INFERENCE_RECOVERED: Final[str] = (
        "Requeued %s recordings after a cancelled session."
    )
    CUDA_CONVOLUTION_RECOVERED: Final[str] = (
        "CUDA: using compatible cuDNN convolution algorithms."
    )
    RECORDING_FAILED: Final[str] = "Recording %s failed: %s"
    CATALOGUE_CACHED: Final[str] = "Catalogue: page {page}/{pages} cached"
    FAILURE_DETAIL: Final[str] = "{kind}: {error}"


class ExperimentMessage:
    """Explain scientific exclusions without changing stored reasons."""

    UNSUPPORTED_TARGET: Final[str] = (
        "Unidentified/questioned metadata or species absent from BirdNET"
    )
    UNDETECTED_TARGET: Final[str] = (
        "Metadata target absent from exported baseline predictions"
    )
    LEGACY_ERROR_PREFIX: Final[str] = "RuntimeError:"
    LEGACY_SESSION_CANCELLED: Final[str] = (
        "Analysis was cancelled. Please check the logs:"
    )


class ProgressMessage:
    """Format the progress bar's label and persisted counts."""

    BATCH: Final[str] = "Lot {batch_id}"
    UNIT: Final[str] = "audio"
    COUNTS: Final[str] = "done={done}, failed={failed}"
    CURRENT_RECORDING: Final[str] = "{counts}, XC={identifier}"


class CliMessage:
    """Name CLI operations and explain their usage and failures."""

    PROGRAM: Final[str] = "bird-bioacoustics"
    DESCRIPTION: Final[str] = (
        "Run one resumable Xeno-canto batch or report all stored measurements."
    )
    ROOT_HELP: Final[str] = (
        "Project directory; defaults to the current project."
    )
    BATCH_HELP: Final[str] = "Process one recording batch with a progress bar."
    BATCH_SIZE_HELP: Final[str] = (
        "Maximum recordings in a new batch; unfinished batches keep their "
        "original size."
    )
    NO_PROGRESS_HELP: Final[str] = "Disable progress output."
    MODELS_HELP: Final[str] = (
        "Download and cache the configured model without processing audio."
    )
    DEVICE_HELP: Final[str] = "Inference target: CPU or the first CUDA GPU."
    WORKERS_HELP: Final[str] = (
        "Inference worker processes; use one on a single GPU."
    )
    PRODUCERS_HELP: Final[str] = "Processes preparing audio window batches."
    INFERENCE_BATCH_SIZE_HELP: Final[str] = (
        "Audio windows per inference call; independent of recording batches."
    )
    REPORT_HELP: Final[str] = (
        "Save cumulative results; refresh the notebook and HTML/PDF/SVG "
        "previews."
    )
    STATUS_HELP: Final[str] = "Read current coverage without writing results."
    INTERRUPTED: Final[str] = (
        "Batch interrupted; its next uncommitted recording remains pending."
    )
    ERROR: Final[str] = "{kind}: {error}"
