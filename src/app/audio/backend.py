"""Resolve the external inference library only when a model is requested."""

from __future__ import annotations

from importlib import import_module
from os import environ
from typing import Literal, Protocol, cast

from app.domain.constants import RuntimeModule
from app.domain.messages import ErrorMessage
from app.domain.protocols import BirdNetModelProtocol
from app.domain.settings import ModelSettings


class BirdNetLibrary(Protocol):
    """Describe the supported public acoustic-model factory."""

    def load(
        self,
        family: Literal["acoustic"],
        version: Literal["2.4", "3.0"],
        backend: Literal["tf", "pb", "pt", "onnx"],
    ) -> BirdNetModelProtocol:
        """Construct a model using an already validated version/backend pair.

        :param family: Acoustic model family.
        :type family: Literal["acoustic"]
        :param version: Supported model version.
        :type version: Literal["2.4", "3.0"]
        :param backend: Inference backend compatible with that version.
        :type backend: Literal["tf", "pb", "pt", "onnx"]
        :return: Acoustic model exposing prediction and session methods.
        :rtype: BirdNetModelProtocol
        """
        raise NotImplementedError


def resolve_library() -> BirdNetLibrary:
    """Resolve the optional inference boundary after a batch requests it.

    :return: Installed BirdNET library's public model factory.
    :rtype: BirdNetLibrary
    """
    return cast(BirdNetLibrary, import_module(RuntimeModule.BIRDNET))


def load_model(settings: ModelSettings) -> BirdNetModelProtocol:
    """Load a compatible model while keeping reporting free of inference.

    :param settings: Acoustic version and backend configuration.
    :type settings: ModelSettings
    :return: Loaded acoustic model.
    :rtype: BirdNetModelProtocol
    """
    valid_pair: bool = (
        settings.version == "2.4" and settings.backend in ("tf", "pb")
    ) or (settings.version == "3.0" and settings.backend in ("pt", "onnx"))
    if not valid_pair:
        raise ValueError(
            ErrorMessage.INVALID_MODEL_BACKEND.format(
                version=settings.version, backend=settings.backend
            )
        )
    environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
    library: BirdNetLibrary = resolve_library()
    return library.load(settings.family, settings.version, settings.backend)
