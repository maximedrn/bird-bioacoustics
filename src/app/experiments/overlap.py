"""Experiments overlap services."""

from __future__ import annotations

from pathlib import Path

from pandas import DataFrame

from app.audio.prediction import BirdNetPredictor
from app.audio.processing import AudioProcessor
from app.domain.constants import PredictionColumn
from app.domain.messages import ErrorMessage
from app.domain.models import (
    NoiseExperimentOutput,
    OverlapExperimentOutput,
    OverlapResult,
)
from app.domain.settings import ExperimentSettings, ProjectPaths
from app.domain.types import FloatArray


class OverlapExperiment:
    """Mix metadata targets from two distinct source recordings."""

    def __init__(
        self,
        predictor: BirdNetPredictor,
        audio_processor: AudioProcessor,
        paths: ProjectPaths,
        settings: ExperimentSettings,
    ) -> None:
        """Initialize the experiment.

        :param predictor: BirdNET prediction service.
        :type predictor: BirdNetPredictor
        :param audio_processor: Audio processing service.
        :type audio_processor: AudioProcessor
        :param paths: Project path configuration.
        :type paths: ProjectPaths
        :param settings: Experiment settings.
        :type settings: ExperimentSettings
        :return: None.
        :rtype: None
        """
        self._predictor: BirdNetPredictor = predictor
        self._audio_processor: AudioProcessor = audio_processor
        self._paths: ProjectPaths = paths
        self._settings: ExperimentSettings = settings

    def run(
        self,
        first: NoiseExperimentOutput,
        second: NoiseExperimentOutput,
        pair_id: str,
    ) -> OverlapExperimentOutput:
        """Superpose target segments after aligning their sampling rates.

        :param first: Target segment and metadata from the first recording.
        :type first: NoiseExperimentOutput
        :param second: Target segment and metadata from the second recording.
        :type second: NoiseExperimentOutput
        :param pair_id: Stable recording-pair identifier.
        :type pair_id: str
        :return: Pair measurements and playback segments.
        :rtype: OverlapExperimentOutput
        """
        if first.recording_id == second.recording_id:
            raise ValueError(ErrorMessage.SAME_OVERLAP_RECORDING)
        if first.target.species_name == second.target.species_name:
            raise ValueError(ErrorMessage.SAME_OVERLAP_SPECIES)
        sample_rate: int = first.sample_rate
        segment_a: FloatArray = first.target_segment
        segment_b: FloatArray = self._audio_processor.resample_audio(
            second.target_segment, second.sample_rate, sample_rate
        )
        segment_a_path: Path = first.target_segment_path
        segment_b_path: Path = self._audio_processor.save_audio(
            self._paths.generated_audio / f"{pair_id}_species_b_resampled.wav",
            segment_b,
            sample_rate,
        )
        mixed_paths: list[Path] = []
        for ratio_value in self._settings.mix_ratios:
            ratio = ratio_value
            mixed_samples: FloatArray = self._audio_processor.mix_audio(
                segment_a, segment_b, ratio
            )
            mixed_paths.append(
                self._audio_processor.save_audio(
                    self._paths.generated_audio
                    / f"mix_{pair_id}_{ratio.label.replace('/', '_')}.wav",
                    mixed_samples,
                    sample_rate,
                )
            )
        predictions: DataFrame = self._predictor.predict_many(
            tuple(mixed_paths), f"overlap_predictions_{pair_id}"
        )
        result_rows: list[OverlapResult] = []
        for ratio_value, mixed_path_value in zip(
            self._settings.mix_ratios, mixed_paths, strict=True
        ):
            ratio = ratio_value
            mixed_path: Path = mixed_path_value
            selected: DataFrame = predictions.loc[
                predictions[PredictionColumn.INPUT] == mixed_path.name
            ]
            result_rows.append(
                OverlapResult(
                    pair_id=pair_id,
                    recording_a=first.recording_id,
                    recording_b=second.recording_id,
                    mix=ratio.label,
                    weight_a=ratio.first,
                    weight_b=ratio.second,
                    species_a=first.target.species_name,
                    confidence_a=self._predictor.confidence_for_species(
                        selected, first.target.species_name
                    ),
                    species_b=second.target.species_name,
                    confidence_b=self._predictor.confidence_for_species(
                        selected, second.target.species_name
                    ),
                )
            )
        records: list[dict[str, object]] = [
            result.model_dump() for result in result_rows
        ]
        results: DataFrame = DataFrame(records)
        return OverlapExperimentOutput(
            species_a=first.target,
            species_b=second.target,
            segment_a=segment_a,
            segment_b=segment_b,
            segment_a_path=segment_a_path,
            segment_b_path=segment_b_path,
            results=results,
        )
