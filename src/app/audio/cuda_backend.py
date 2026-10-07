"""Guard CUDA inside each spawned BirdNET inference worker."""

from __future__ import annotations

from typing import cast

from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet.acoustic.models.v3_0.onnx import AcousticOnnxBackendFP32V3_0
from ordered_set import OrderedSet

from app.audio.onnx import CudaSession
from app.domain.constants import OnnxOption
from app.domain.protocols import BirdNetModelProtocol


class RequiredCudaBackend(AcousticOnnxBackendFP32V3_0):
    """Keep FP32 inference while checking each worker's provider."""

    def load(self) -> None:
        """Configure recoverable cuDNN engines before accepting any audio.

        :return: None.
        :rtype: None
        """
        session: CudaSession = CudaSession(self._model_path)
        # BirdNET exposes no public session accessor in its pinned version.
        setattr(self, OnnxOption.SESSION_ATTRIBUTE, session)
        self._input_name = session.get_inputs()[0].name
        self._output_names = [output.name for output in session.get_outputs()]


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
