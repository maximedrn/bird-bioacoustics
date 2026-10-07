"""Configure CUDA sessions and recover unsupported cuDNN engine searches."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

from numpy import float32, isfinite, zeros

from app.audio.diagnostics import silent_native_diagnostics
from app.domain.constants import (
    ConvolutionSearch,
    CudaEngineFailure,
    ExecutionProvider,
    OnnxOption,
    RuntimeModule,
)
from app.domain.messages import ErrorMessage, LogMessage
from app.domain.types import FloatArray


class OnnxValue(Protocol):
    """Describe model inputs and outputs without loading ONNX in the parent."""

    name: str
    shape: list[int | str | None]


class OnnxSession(Protocol):
    """Expose the ONNX operations required by the acoustic backend."""

    def get_providers(self) -> list[str]:
        """Return initialized execution providers.

        :return: Active execution providers.
        :rtype: list[str]
        """
        raise NotImplementedError

    def disable_fallback(self) -> None:
        """Disable whole-session CPU retries.

        :return: None.
        :rtype: None
        """

    def get_inputs(self) -> list[OnnxValue]:
        """Return input names and dimensions.

        :return: Input metadata.
        :rtype: list[OnnxValue]
        """
        raise NotImplementedError

    def get_outputs(self) -> list[OnnxValue]:
        """Return output names and dimensions.

        :return: Output metadata.
        :rtype: list[OnnxValue]
        """
        raise NotImplementedError

    def run(
        self, output_names: list[str], input_feed: dict[str, FloatArray]
    ) -> list[FloatArray]:
        """Evaluate the requested FP32 model outputs.

        :param output_names: Requested native output names.
        :type output_names: list[str]
        :param input_feed: FP32 audio windows by input name.
        :type input_feed: dict[str, FloatArray]
        :return: Arrays for the requested outputs.
        :rtype: list[FloatArray]
        """
        raise NotImplementedError


class OnnxSessionOptions(Protocol):
    """Type the supported session logging option."""

    log_severity_level: int


class OnnxRuntime(Protocol):
    """Type the lazily imported ONNX runtime."""

    def get_available_providers(self) -> list[str]:
        """Return compiled execution provider names.

        :return: Available execution providers.
        :rtype: list[str]
        """
        raise NotImplementedError

    def set_default_logger_severity(self, severity: int) -> None:
        """Configure native logs in the current GPU process.

        :param severity: Minimum reported native logging severity.
        :type severity: int
        :return: None.
        :rtype: None
        """

    def SessionOptions(self) -> OnnxSessionOptions:
        """Create session configuration.

        :return: Mutable native session options.
        :rtype: OnnxSessionOptions
        """
        raise NotImplementedError

    def InferenceSession(
        self,
        path_or_bytes: str,
        *,
        sess_options: OnnxSessionOptions,
        providers: list[str],
        provider_options: list[dict[str, str]],
    ) -> OnnxSession:
        """Create a native session with explicit CUDA options.

        :param path_or_bytes: Cached ONNX model path.
        :type path_or_bytes: str
        :param sess_options: Native logging configuration.
        :type sess_options: OnnxSessionOptions
        :param providers: Execution providers in priority order.
        :type providers: list[str]
        :param provider_options: Options for each requested provider.
        :type provider_options: list[dict[str, str]]
        :return: Loaded native session.
        :rtype: OnnxSession
        """
        raise NotImplementedError


@silent_native_diagnostics()
def create_cuda_session(
    model_path: Path, search: ConvolutionSearch
) -> OnnxSession:
    """Load the selected cuDNN engines and reject a CPU-only initialization.

    :param model_path: Cached FP32 ONNX weights.
    :type model_path: Path
    :param search: GPU convolution engine selection strategy.
    :type search: ConvolutionSearch
    :return: Native CUDA session with CPU retry disabled.
    :rtype: OnnxSession
    """
    runtime: OnnxRuntime = cast(OnnxRuntime, import_module(RuntimeModule.ONNX))
    if ExecutionProvider.CUDA not in runtime.get_available_providers():
        raise RuntimeError(ErrorMessage.CUDA_UNAVAILABLE)
    runtime.set_default_logger_severity(OnnxOption.LOG_ERROR)
    options: OnnxSessionOptions = runtime.SessionOptions()
    options.log_severity_level = OnnxOption.LOG_ERROR
    session: OnnxSession = runtime.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=[ExecutionProvider.CUDA, ExecutionProvider.CPU],
        provider_options=[
            {
                OnnxOption.DEVICE_ID: OnnxOption.FIRST_GPU,
                OnnxOption.CONVOLUTION_SEARCH: search,
            },
            {},
        ],
    )
    if ExecutionProvider.CUDA not in session.get_providers():
        raise RuntimeError(ErrorMessage.CUDA_UNAVAILABLE)
    session.disable_fallback()
    return session


class CudaSession:
    """Retry engine selection once on CUDA without changing the input data."""

    def __init__(self, model_path: Path) -> None:
        """Create the first session using the faster cuDNN search.

        :param model_path: Cached model with unchanged precision and labels.
        :type model_path: Path
        :return: None.
        :rtype: None
        """
        self._path: Path = model_path
        self._search: ConvolutionSearch = ConvolutionSearch.HEURISTIC
        self._session: OnnxSession | None = create_cuda_session(
            model_path, self._search
        )

    @property
    def session(self) -> OnnxSession:
        """Return the currently loaded native session.

        :return: Active native session.
        :rtype: OnnxSession
        """
        if self._session is None:
            raise RuntimeError(ErrorMessage.CUDA_SESSION_CLOSED)
        return self._session

    def get_inputs(self) -> list[OnnxValue]:
        """Return input metadata for BirdNET and preparation probes.

        :return: Native input metadata.
        :rtype: list[OnnxValue]
        """
        return self.session.get_inputs()

    def get_outputs(self) -> list[OnnxValue]:
        """Return the unchanged prediction and encoding output metadata.

        :return: Native output metadata.
        :rtype: list[OnnxValue]
        """
        return self.session.get_outputs()

    @silent_native_diagnostics()
    def run(
        self, output_names: list[str], input_feed: dict[str, FloatArray]
    ) -> list[FloatArray]:
        """Recover an unsupported cuDNN engine search on the same GPU.

        Other runtime errors and a failed compatibility attempt propagate to
        BirdNET, which cancels the session and leaves the recording pending.

        :param output_names: Requested output names from the original graph.
        :type output_names: list[str]
        :param input_feed: Unmodified FP32 audio windows.
        :type input_feed: dict[str, FloatArray]
        :return: Native prediction arrays with their original row order.
        :rtype: list[FloatArray]
        """
        try:
            return self.session.run(output_names, input_feed)
        except Exception as error:
            if self._search != ConvolutionSearch.HEURISTIC:
                raise
            message: str = str(error).upper()
            if not any(marker in message for marker in CudaEngineFailure):
                raise
        # Release the old GPU allocation before loading compatibility engines.
        self._session = None
        self._search = ConvolutionSearch.COMPATIBILITY
        self._session = create_cuda_session(self._path, self._search)
        result: list[FloatArray] = self.session.run(output_names, input_feed)
        print(LogMessage.CUDA_CONVOLUTION_RECOVERED, flush=True)
        return result


def check_cuda_model(
    model_path: Path,
    batch_size: int = OnnxOption.PROBE_PARTIAL_BATCH,
    *,
    input_size_samples: int | None = None,
) -> str:
    """Run real convolutions for small, partial and full inference batches.

    :param model_path: Cached FP32 model weights.
    :type model_path: Path
    :param batch_size: Requested maximum number of audio windows.
    :type batch_size: int
    :param input_size_samples: BirdNET window length for dynamic ONNX inputs.
    :type input_size_samples: int | None
    :return: Verified CUDA execution provider.
    :rtype: str
    """
    session: CudaSession = CudaSession(model_path)
    inputs: list[OnnxValue] = session.get_inputs()
    if len(inputs) != 1 or len(inputs[0].shape) != 2:
        raise ValueError(ErrorMessage.INVALID_CUDA_PROBE_INPUT)
    sample_dimension: int | str | None = inputs[0].shape[1]
    samples: int | None = (
        sample_dimension
        if isinstance(sample_dimension, int)
        else input_size_samples
    )
    if samples is None or isinstance(samples, bool) or samples <= 0:
        raise ValueError(ErrorMessage.INVALID_CUDA_PROBE_LENGTH)
    output_name: str = session.get_outputs()[0].name
    sizes: list[int] = sorted(
        {1, min(OnnxOption.PROBE_PARTIAL_BATCH, batch_size), batch_size}
    )
    size: int
    for size in sizes:
        audio: FloatArray = zeros((size, samples), dtype=float32)
        scores: FloatArray = session.run(
            [output_name], {inputs[0].name: audio}
        )[0]
        if (
            scores.ndim != 2
            or scores.shape[0] != size
            or not bool(isfinite(scores).all())
        ):
            raise RuntimeError(
                ErrorMessage.INVALID_CUDA_PROBE_OUTPUT.format(batch_size=size)
            )
    return ExecutionProvider.CUDA
