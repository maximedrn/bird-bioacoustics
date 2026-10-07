"""Validate inference resources and verify that CUDA can load the model."""

from __future__ import annotations

from importlib import import_module
from os import cpu_count
from typing import Protocol, cast

from app.domain.constants import (
    ExecutionProvider,
    InferenceDevice,
    OnnxOption,
    RuntimeModule,
)
from app.domain.messages import ErrorMessage
from app.domain.protocols import BirdNetModelProtocol
from app.domain.settings import InferenceSettings, ModelSettings


class OnnxSession(Protocol):
    """Expose the providers actually initialized by an ONNX session."""

    def get_providers(self) -> list[str]:
        """Read initialized providers rather than compiled capabilities.

        :return: Active provider names.
        :rtype: list[str]
        """
        raise NotImplementedError


class OnnxRuntime(Protocol):
    """Type the lazy ONNX boundary without importing it during reports."""

    def get_available_providers(self) -> list[str]:
        """List providers advertised by the installed runtime.

        :return: Available provider names.
        :rtype: list[str]
        """
        raise NotImplementedError

    def InferenceSession(
        self,
        path_or_bytes: str,
        *,
        providers: list[str],
        provider_options: list[dict[str, str]],
    ) -> OnnxSession:
        """Load the model with the same provider request as BirdNET.

        :param path_or_bytes: Local ONNX model path.
        :type path_or_bytes: str
        :param providers: Requested providers in priority order.
        :type providers: list[str]
        :param provider_options: Options paired with each provider.
        :type provider_options: list[dict[str, str]]
        :return: Initialized session.
        :rtype: OnnxSession
        """
        raise NotImplementedError


def validate_inference(
    settings: InferenceSettings, model: ModelSettings
) -> None:
    """Reject unavailable process counts before changing experiment state.

    :param settings: Requested hardware and process counts.
    :type settings: InferenceSettings
    :param model: Scientific model configuration.
    :type model: ModelSettings
    :return: None.
    :rtype: None
    """
    cores: int = cpu_count() or 1
    if max(settings.n_workers, settings.n_producers) > cores:
        raise ValueError(
            ErrorMessage.EXCESSIVE_INFERENCE_WORKERS.format(cores=cores)
        )
    if settings.device == InferenceDevice.GPU and model.backend != "onnx":
        raise ValueError(ErrorMessage.UNSUPPORTED_GPU_BACKEND)


def verify_model_device(
    model: BirdNetModelProtocol, settings: InferenceSettings
) -> str:
    """Fail early if CUDA cannot load the cached model on the selected GPU.

    :param model: Acoustic model whose weights are already available.
    :type model: BirdNetModelProtocol
    :param settings: Requested inference hardware.
    :type settings: InferenceSettings
    :return: Verified provider name.
    :rtype: str
    """
    if settings.device == InferenceDevice.CPU:
        return ExecutionProvider.CPU
    runtime: OnnxRuntime = cast(OnnxRuntime, import_module(RuntimeModule.ONNX))
    if ExecutionProvider.CUDA not in runtime.get_available_providers():
        raise RuntimeError(ErrorMessage.CUDA_UNAVAILABLE)
    session: OnnxSession = runtime.InferenceSession(
        str(model.model_path),
        providers=[ExecutionProvider.CUDA, ExecutionProvider.CPU],
        provider_options=[{OnnxOption.DEVICE_ID: OnnxOption.FIRST_GPU}, {}],
    )
    if ExecutionProvider.CUDA not in session.get_providers():
        raise RuntimeError(ErrorMessage.CUDA_UNAVAILABLE)
    return ExecutionProvider.CUDA
