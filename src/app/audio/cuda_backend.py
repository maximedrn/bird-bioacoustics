"""Guard CUDA inside each spawned BirdNET inference worker."""

from __future__ import annotations

from typing import cast

from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet.acoustic.models.v3_0.onnx import AcousticOnnxBackendFP32V3_0
from ordered_set import OrderedSet

from app.audio.runtime import OnnxSession
from app.domain.constants import ExecutionProvider
from app.domain.messages import ErrorMessage
from app.domain.protocols import BirdNetModelProtocol


class RequiredCudaBackend(AcousticOnnxBackendFP32V3_0):
    """Keep FP32 inference while checking each worker's provider."""

    def load(self) -> None:
        """Reject a failed CUDA initialization before accepting any audio.

        :return: None.
        :rtype: None
        """
        super().load()
        # BirdNET exposes no public session accessor in its pinned version.
        session: OnnxSession | None = cast(
            OnnxSession | None, getattr(self, "_session", None)
        )
        if (
            session is None
            or ExecutionProvider.CUDA not in session.get_providers()
        ):
            self.unload()
            raise ValueError(ErrorMessage.CUDA_UNAVAILABLE)
        session.disable_fallback()


def require_cuda(model: BirdNetModelProtocol) -> BirdNetModelProtocol:
    """Reuse weights and labels with a provider guard in every worker.

    :param model: Existing configured FP32 acoustic model.
    :type model: BirdNetModelProtocol
    :return: Same model graph and labels using the guarded ONNX backend.
    :rtype: BirdNetModelProtocol
    """
    guarded: AcousticModelV3_0 = AcousticModelV3_0.load(
        model_path=model.model_path,
        species_list=OrderedSet(model.species_list),
        backend_type=RequiredCudaBackend,
        backend_kwargs={},
    )
    return cast(BirdNetModelProtocol, guarded)
