"""Validate inference resources and verify CUDA in an isolated process."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.connection import Connection
from multiprocessing.context import SpawnContext
from multiprocessing.process import BaseProcess
from os import cpu_count, environ
from pathlib import Path

from app.audio.onnx import check_cuda_model
from app.domain.constants import (
    ExecutionProvider,
    InferenceDevice,
    InferenceRuntime,
)
from app.domain.messages import ErrorMessage
from app.domain.protocols import BirdNetModelProtocol
from app.domain.settings import InferenceSettings, ModelSettings


@dataclass(frozen=True, slots=True)
class CudaCheckResult:
    """Transfer a completed CUDA check from its isolated process."""

    provider: str | None
    error: str | None


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


def configure_inference(settings: InferenceSettings) -> None:
    """Select safe process creation without initializing CUDA in the parent.

    :param settings: Requested inference hardware.
    :type settings: InferenceSettings
    :return: None.
    :rtype: None
    """
    if settings.device == InferenceDevice.GPU:
        environ[InferenceRuntime.START_METHOD_VARIABLE] = (
            InferenceRuntime.START_METHOD
        )


def verify_model_device(
    model: BirdNetModelProtocol, settings: InferenceSettings
) -> str:
    """Verify CUDA in a fresh process and keep the application parent clean.

    :param model: Acoustic model whose weights are already available.
    :type model: BirdNetModelProtocol
    :param settings: Requested inference hardware.
    :type settings: InferenceSettings
    :return: Verified provider name.
    :rtype: str
    """
    if settings.device == InferenceDevice.CPU:
        return ExecutionProvider.CPU
    configure_inference(settings)
    window_samples: int = model.get_segment_size_samples()
    context: SpawnContext = get_context(InferenceRuntime.START_METHOD)
    reader: Connection
    writer: Connection
    reader, writer = context.Pipe(duplex=False)
    worker: BaseProcess = context.Process(
        target=check_cuda_worker,
        args=(model.model_path, settings.batch_size, window_samples, writer),
    )
    try:
        worker.start()
        writer.close()
        return read_cuda_check(worker, reader)
    finally:
        reader.close()
        writer.close()
        stop_cuda_check(worker)


def read_cuda_check(worker: BaseProcess, reader: Connection) -> str:
    """Read a bounded verification result and reject failed process startup.

    :param worker: Isolated verification process.
    :type worker: BaseProcess
    :param reader: Parent pipe endpoint.
    :type reader: Connection
    :return: Verified CUDA provider.
    :rtype: str
    """
    if not reader.poll(InferenceRuntime.CUDA_CHECK_TIMEOUT):
        raise RuntimeError(ErrorMessage.CUDA_CHECK_TIMEOUT)
    try:
        result: object = reader.recv()
    except EOFError as error:
        worker.join(timeout=InferenceRuntime.PROCESS_EXIT_TIMEOUT)
        raise RuntimeError(
            ErrorMessage.CUDA_CHECK_CRASHED.format(code=worker.exitcode)
        ) from error
    worker.join(timeout=InferenceRuntime.PROCESS_EXIT_TIMEOUT)
    if not isinstance(result, CudaCheckResult) or worker.exitcode != 0:
        raise RuntimeError(
            ErrorMessage.CUDA_CHECK_CRASHED.format(code=worker.exitcode)
        )
    if result.error is not None:
        raise RuntimeError(
            ErrorMessage.CUDA_CHECK_FAILED.format(error=result.error)
        )
    if result.provider != ExecutionProvider.CUDA:
        raise RuntimeError(ErrorMessage.CUDA_UNAVAILABLE)
    return result.provider


def stop_cuda_check(worker: BaseProcess) -> None:
    """Reap verification resources after success, timeout or interruption.

    :param worker: Verification process, possibly not started successfully.
    :type worker: BaseProcess
    :return: None.
    :rtype: None
    """
    if worker.pid is None:
        return
    if worker.is_alive():
        worker.terminate()
        worker.join(timeout=InferenceRuntime.PROCESS_EXIT_TIMEOUT)
    if worker.is_alive():
        worker.kill()
        worker.join()
    worker.close()


def check_cuda_worker(
    model_path: Path,
    batch_size: int,
    window_samples: int,
    writer: Connection,
) -> None:
    """Load CUDA only inside the process that owns its temporary context.

    :param model_path: Already cached ONNX weights.
    :type model_path: Path
    :param batch_size: Requested number of audio windows per inference.
    :type batch_size: int
    :param window_samples: Audio window length supplied by BirdNET.
    :type window_samples: int
    :param writer: Child pipe endpoint for its completed result.
    :type writer: Connection
    :return: None.
    :rtype: None
    """
    try:
        provider: str = check_cuda_model(
            model_path, batch_size, input_size_samples=window_samples
        )
        result: CudaCheckResult = CudaCheckResult(provider, None)
    except Exception as error:
        result = CudaCheckResult(None, str(error))
    try:
        writer.send(result)
    finally:
        writer.close()
