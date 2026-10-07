"""Prepare cached model assets independently of corpus processing."""

from __future__ import annotations

from typing import TypedDict

from app.audio.backend import load_model
from app.audio.runtime import validate_inference, verify_model_device
from app.domain.protocols import BirdNetModelProtocol
from app.domain.settings import InferenceSettings, ModelSettings


class ModelPreparation(TypedDict):
    """Describe cached assets and verified execution resources."""

    model: str
    model_bytes: int
    species: int
    device: str
    provider: str
    workers: int
    producers: int
    inference_batch_size: int


def prepare_model(
    settings: ModelSettings, inference: InferenceSettings
) -> ModelPreparation:
    """Download weights and labels without opening the experiment checkpoint.

    :param settings: Existing acoustic model configuration.
    :type settings: ModelSettings
    :param inference: Hardware and parallel processing configuration.
    :type inference: InferenceSettings
    :return: Cached file details and verified hardware.
    :rtype: ModelPreparation
    """
    validate_inference(inference, settings)
    model: BirdNetModelProtocol = load_model(settings)
    provider: str = verify_model_device(model, inference)
    return ModelPreparation(
        model=str(model.model_path),
        model_bytes=model.model_path.stat().st_size,
        species=len(model.species_list),
        device=inference.device,
        provider=provider,
        workers=inference.n_workers,
        producers=inference.n_producers,
        inference_batch_size=inference.batch_size,
    )
