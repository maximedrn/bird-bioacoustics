"""Experiments noise services."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from numpy import inf
from numpy.random import Generator, SeedSequence, default_rng
from pandas import DataFrame

from app.audio.prediction import BirdNetPredictor
from app.audio.processing import AudioProcessor
from app.domain.constants import (
    NoiseCondition,
    PredictionColumn,
    ResultColumn,
)
from app.domain.models import (
    DetectionCandidate,
    NoiseExperimentOutput,
    NoiseResult,
)
from app.domain.settings import ExperimentSettings, ProjectPaths
from app.domain.types import FloatArray


class NoiseRobustnessExperiment:
    """Repeat noise perturbations for the metadata target of one recording."""

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
        baseline: DataFrame,
        audio_path: Path,
        target_species: str,
        recording_id: str,
    ) -> NoiseExperimentOutput | None:
        """Perturb the best baseline segment of a known metadata species.

        :param baseline: Baseline predictions for this recording.
        :type baseline: DataFrame
        :param audio_path: Original recording path.
        :type audio_path: Path
        :param target_species: Species identified on the Xeno-canto page.
        :type target_species: str
        :param recording_id: Stable corpus identifier.
        :type recording_id: str
        :return: Trial results, or None when no target segment can be selected.
        :rtype: NoiseExperimentOutput | None
        """
        trial_value: tuple[float, int, int, Path]
        target: DetectionCandidate | None = (
            self._predictor.candidate_for_species(baseline, target_species)
        )
        if target is None:
            return None
        target_segment, sample_rate = self._audio_processor.load_segment(
            audio_path, target.start_seconds, target.end_seconds
        )
        target_segment_path: Path = self._audio_processor.save_audio(
            self._paths.generated_audio
            / f"{recording_id}_target_original.wav",
            target_segment,
            sample_rate,
        )
        audio_paths: list[Path] = [target_segment_path]
        trial_settings: list[tuple[float, int, int, Path]] = []
        # Keep earlier trials unchanged as the corpus grows.
        recording_seed: int = int.from_bytes(
            sha256(recording_id.encode("utf-8")).digest()[:4], "little"
        )
        for snr_index, snr_value in enumerate(self._settings.snr_values_db):
            snr_db: float = float(snr_value)
            for trial_index in range(self._settings.noise_repetitions):
                trial: int = trial_index + 1
                seed_sequence: SeedSequence = SeedSequence(
                    [
                        self._settings.random_seed,
                        recording_seed,
                        snr_index,
                        trial_index,
                    ]
                )
                seed: int = int(seed_sequence.generate_state(1)[0])
                generator: Generator = default_rng(seed)
                noisy_samples: FloatArray = (
                    self._audio_processor.add_gaussian_noise_at_snr(
                        target_segment, snr_db, generator
                    )
                )
                snr_label: str = f"{snr_db:g}".replace(".", "_")
                noisy_path: Path = self._audio_processor.save_audio(
                    self._paths.generated_audio
                    / f"{recording_id}_snr_{snr_label}_trial_{trial}.wav",
                    noisy_samples,
                    sample_rate,
                )
                audio_paths.append(noisy_path)
                trial_settings.append((snr_db, trial, seed, noisy_path))
        predictions: DataFrame = self._predictor.predict_many(
            tuple(audio_paths), f"noise_predictions_{recording_id}"
        )
        original_predictions: DataFrame = predictions.loc[
            predictions[PredictionColumn.INPUT] == target_segment_path.name
        ]
        result_rows: list[NoiseResult] = [
            NoiseResult(
                recording_id=recording_id,
                trial=0,
                seed=None,
                condition=NoiseCondition.ORIGINAL,
                snr_db=inf,
                target_species=target.species_name,
                confidence=self._predictor.confidence_for_species(
                    original_predictions, target.species_name
                ),
                rank=self._predictor.rank_for_species(
                    original_predictions, target.species_name
                ),
            )
        ]
        for trial_value in trial_settings:
            snr_db, trial, seed, noisy_path = trial_value
            selected: DataFrame = predictions.loc[
                predictions[PredictionColumn.INPUT] == noisy_path.name
            ]
            result_rows.append(
                NoiseResult(
                    recording_id=recording_id,
                    trial=trial,
                    seed=seed,
                    condition=f"{snr_db:g} dB",
                    snr_db=snr_db,
                    target_species=target.species_name,
                    confidence=self._predictor.confidence_for_species(
                        selected, target.species_name
                    ),
                    rank=self._predictor.rank_for_species(
                        selected, target.species_name
                    ),
                )
            )
        records: list[dict[str, object]] = [
            result.model_dump() for result in result_rows
        ]
        results: DataFrame = DataFrame(records)
        results[ResultColumn.RANK] = results[ResultColumn.RANK].astype("Int64")
        results[ResultColumn.SEED] = results[ResultColumn.SEED].astype("Int64")
        return NoiseExperimentOutput(
            recording_id=recording_id,
            target=target,
            sample_rate=sample_rate,
            target_segment=target_segment,
            target_segment_path=target_segment_path,
            results=results,
        )


class ExperimentSummary:
    """Aggregate trials with equal weight for each recording."""

    @staticmethod
    def noise(results: DataFrame) -> tuple[DataFrame, DataFrame]:
        """Separate within-recording variability from between-recording
        variability.

        :param results: Original and repeated noise measurements.
        :type results: DataFrame
        :return: Per-recording summaries and an equally weighted corpus
            summary.
        :rtype: tuple[DataFrame, DataFrame]
        """
        measured: DataFrame = results.copy()
        measured["detected"] = (
            measured[ResultColumn.RANK].notna().astype(float)
        )
        per_recording: DataFrame = (
            measured.groupby(
                [
                    ResultColumn.RECORDING_ID,
                    ResultColumn.CONDITION,
                    ResultColumn.SNR_DB,
                ],
                sort=False,
                dropna=False,
            )
            .agg(
                confidence_mean=(PredictionColumn.CONFIDENCE, "mean"),
                confidence_std=(PredictionColumn.CONFIDENCE, "std"),
                detection_rate=("detected", "mean"),
                trials=(ResultColumn.TRIAL, "count"),
            )
            .reset_index()
        )
        per_recording[ResultColumn.CONFIDENCE_STD] = per_recording[
            ResultColumn.CONFIDENCE_STD
        ].fillna(0.0)
        summary: DataFrame = (
            per_recording.groupby(
                [ResultColumn.CONDITION, ResultColumn.SNR_DB],
                sort=False,
                dropna=False,
            )
            .agg(
                confidence_mean=(ResultColumn.CONFIDENCE_MEAN, "mean"),
                confidence_std=(ResultColumn.CONFIDENCE_MEAN, "std"),
                noise_std_mean=(ResultColumn.CONFIDENCE_STD, "mean"),
                detection_rate=(ResultColumn.DETECTION_RATE, "mean"),
                recordings=(ResultColumn.RECORDING_ID, "nunique"),
                trials=(ResultColumn.TRIALS, "sum"),
            )
            .reset_index()
        )
        summary[ResultColumn.CONFIDENCE_STD] = summary[
            ResultColumn.CONFIDENCE_STD
        ].fillna(0.0)
        return per_recording, summary

    @staticmethod
    def overlap(results: DataFrame) -> DataFrame:
        """Aggregate target confidence across distinct recording pairs.

        :param results: Measurements for all mixtures and recording pairs.
        :type results: DataFrame
        :return: Mean confidence and between-pair standard deviations.
        :rtype: DataFrame
        """
        summary: DataFrame = (
            results.groupby(
                [
                    ResultColumn.MIX,
                    ResultColumn.WEIGHT_A,
                    ResultColumn.WEIGHT_B,
                ],
                sort=False,
            )
            .agg(
                confidence_a_mean=(ResultColumn.CONFIDENCE_A, "mean"),
                confidence_a_std=(ResultColumn.CONFIDENCE_A, "std"),
                confidence_b_mean=(ResultColumn.CONFIDENCE_B, "mean"),
                confidence_b_std=(ResultColumn.CONFIDENCE_B, "std"),
                pairs=("pair_id", "nunique"),
            )
            .reset_index()
        )
        summary[
            [ResultColumn.CONFIDENCE_A_STD, ResultColumn.CONFIDENCE_B_STD]
        ] = summary[
            [ResultColumn.CONFIDENCE_A_STD, ResultColumn.CONFIDENCE_B_STD]
        ].fillna(0.0)
        return summary
