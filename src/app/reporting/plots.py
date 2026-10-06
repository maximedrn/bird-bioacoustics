"""Reporting plots services."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import QuadMesh
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from matplotlib.pyplot import close
from numpy import bool_, float64, isfinite, log10, maximum
from numpy.typing import NDArray
from pandas import DataFrame, Series
from scipy.signal import spectrogram

from app.domain.constants import (
    CoverageKey,
    MetadataKey,
    NoiseCondition,
    ResultColumn,
)
from app.domain.messages import ErrorMessage
from app.domain.types import CorpusStatus, Float64Array, FloatArray
from app.reporting.messages import PlotLabel
from app.reporting.protocols import PlotAxis, PlotFigure
from app.reporting.writing import atomic_output


class ExperimentPlotter:
    """Render and save all experiment figures."""

    @staticmethod
    def _create_figure(figsize: tuple[float, float]) -> Figure:
        """Attach a typed Agg canvas for standalone PNG generation.

        :param figsize: Image dimensions in inches.
        :type figsize: tuple[float, float]
        :return: Figure with its rendering canvas.
        :rtype: Figure
        """
        figure: Figure = Figure(figsize=figsize)
        FigureCanvasAgg(figure)
        return figure

    @staticmethod
    def _pair_axes(figure: Figure) -> tuple[PlotAxis, PlotAxis]:
        """Create two axes without an object ndarray or inferred element types.

        :param figure: Owning Matplotlib figure.
        :type figure: Figure
        :return: Left and right typed plotting interfaces.
        :rtype: tuple[PlotAxis, PlotAxis]
        """
        target: PlotFigure = cast(PlotFigure, figure)
        return target.add_subplot(1, 2, 1), target.add_subplot(1, 2, 2)

    @staticmethod
    def save_figure(figure: Figure, destination: Path) -> None:
        """Save a PNG atomically and release the figure after success or
        failure.

        :param figure: Completed Matplotlib figure.
        :type figure: Figure
        :param destination: PNG destination in the results directory.
        :type destination: Path
        :return: None.
        :rtype: None
        """
        temporary: Path
        try:
            with atomic_output(destination) as temporary:
                cast(PlotFigure, figure).savefig(
                    temporary, dpi=180, format="png"
                )
        finally:
            close(figure)

    @staticmethod
    def plot_progress(status: CorpusStatus, destination: Path) -> None:
        """Plot catalogue coverage and actual processing outcomes before
        inference finishes.

        :param status: Coverage from the exported database snapshot.
        :type status: CorpusStatus
        :param destination: Output PNG path.
        :type destination: Path
        :return: None.
        :rtype: None
        """
        axis: PlotAxis
        expected: int = int(status[MetadataKey.EXPECTED_RECORDINGS.value])
        cached: int = int(status[CoverageKey.CATALOGUED_RECORDINGS.value])
        figure: Figure = ExperimentPlotter._create_figure(figsize=(11.0, 4.2))
        axes: tuple[PlotAxis, PlotAxis] = ExperimentPlotter._pair_axes(figure)
        catalogue_axis: PlotAxis = axes[0]
        processing_axis: PlotAxis = axes[1]
        catalogue_counts: tuple[int, int] = (cached, max(expected - cached, 0))
        outcome_counts: tuple[int, int, int] = (
            int(status[CoverageKey.PROCESSED_RECORDINGS.value]),
            int(status[CoverageKey.FAILED_RECORDINGS.value]),
            int(status[CoverageKey.PENDING_RECORDINGS.value]),
        )
        catalogue_bars: BarContainer = catalogue_axis.bar(
            (PlotLabel.CACHED, PlotLabel.STILL_TO_COLLECT),
            catalogue_counts,
            color=("C0", "0.7"),
        )
        catalogue_axis.bar_label(
            catalogue_bars,
            labels=[f"{value:,}" for value in catalogue_counts],
            padding=3,
        )
        catalogue_axis.set_ylim(0, max(1, *catalogue_counts) * 1.15)
        catalogue_axis.set_title(
            PlotLabel.CATALOGUE_COVERAGE.format(ratio=cached / expected)
            if expected
            else PlotLabel.CATALOGUE_COVERAGE_TOTAL_UNKNOWN
        )
        catalogue_axis.set_ylabel(
            PlotLabel.RECORDINGS_IN_THE_CATALOGUE_SNAPSHOT
        )
        outcome_bars: BarContainer = processing_axis.bar(
            (PlotLabel.ANALYSED, PlotLabel.FAILED, PlotLabel.PENDING),
            outcome_counts,
            color=("C2", "C3", "0.7"),
        )
        processing_axis.bar_label(
            outcome_bars,
            labels=[f"{value:,}" for value in outcome_counts],
            padding=3,
        )
        processing_axis.set_ylim(0, max(1, *outcome_counts) * 1.15)
        processing_axis.set_title(PlotLabel.ANALYSIS_OF_CACHED_RECORDINGS)
        processing_axis.set_ylabel(PlotLabel.CATALOGUED_RECORDINGS)
        for axis in (catalogue_axis, processing_axis):
            axis.ticklabel_format(axis="y", style="plain")
            axis.grid(axis="y", alpha=0.25)
            axis.set_axisbelow(True)
        figure.tight_layout()
        ExperimentPlotter.save_figure(figure, destination)

    @staticmethod
    def plot_noise(results: DataFrame, destination: Path) -> None:
        """Plot target confidence and detection rate against SNR.

        :param results: Noise experiment results.
        :type results: DataFrame
        :param destination: Output image path.
        :type destination: Path
        :return: None.
        :rtype: None
        """
        axis: PlotAxis
        numeric_snr: Series[float] = results[ResultColumn.SNR_DB].astype(float)
        finite_mask: NDArray[bool_] = isfinite(
            numeric_snr.to_numpy(dtype=float64)
        )
        plot_data: DataFrame = results.loc[finite_mask].sort_values(
            ResultColumn.SNR_DB
        )
        original_values: Series[float] = results.loc[
            results[ResultColumn.CONDITION] == NoiseCondition.ORIGINAL,
            ResultColumn.CONFIDENCE_MEAN,
        ]
        original_confidence: float = float(original_values.iloc[0])

        figure: Figure = ExperimentPlotter._create_figure(figsize=(12.0, 4.2))
        axes: tuple[PlotAxis, PlotAxis] = ExperimentPlotter._pair_axes(figure)
        confidence_axis: PlotAxis = axes[0]
        rate_axis: PlotAxis = axes[1]
        confidence_axis.errorbar(
            plot_data[ResultColumn.SNR_DB].astype(float),
            plot_data[ResultColumn.CONFIDENCE_MEAN].astype(float),
            yerr=plot_data[ResultColumn.CONFIDENCE_STD].astype(float),
            marker="o",
            capsize=4,
            label=PlotLabel.RECORDING_MEANS_1_SD,
        )
        confidence_axis.axhline(
            original_confidence,
            linestyle="--",
            linewidth=1.2,
            label=PlotLabel.ORIGINAL_SEGMENTS_MEAN,
        )
        confidence_axis.set_ylabel(PlotLabel.TARGET_SPECIES_CONFIDENCE)
        confidence_axis.set_title(PlotLabel.NOISE_TARGET_CONFIDENCE)
        original_rate: float = float(
            results.loc[
                results[ResultColumn.CONDITION] == NoiseCondition.ORIGINAL,
                ResultColumn.DETECTION_RATE,
            ].iloc[0]
        )
        rate_axis.plot(
            plot_data[ResultColumn.SNR_DB].astype(float),
            plot_data[ResultColumn.DETECTION_RATE].astype(float),
            marker="o",
            label=PlotLabel.MEAN_DETECTION_RATE,
        )
        rate_axis.axhline(
            original_rate,
            linestyle="--",
            linewidth=1.2,
            label=PlotLabel.ORIGINAL_SEGMENTS,
        )
        rate_axis.set_ylabel(PlotLabel.TARGET_DETECTED_IN_EXPORTED_PREDICTIONS)
        rate_axis.set_title(PlotLabel.NOISE_TARGET_DETECTION_RATE)
        for axis in (confidence_axis, rate_axis):
            axis.set_xlabel(PlotLabel.SNR_DB_LOWER_VALUES_MEAN_STRONGER_NOISE)
            axis.set_ylim(0.0, 1.05)
            axis.grid(True, alpha=0.25)
            axis.legend()
        figure.tight_layout()
        ExperimentPlotter.save_figure(figure, destination)

    @staticmethod
    def plot_overlap(results: DataFrame, destination: Path) -> None:
        """Plot confidence for two overlapping species.

        :param results: Overlap experiment results.
        :type results: DataFrame
        :param destination: Output image path.
        :type destination: Path
        :return: None.
        :rtype: None
        """
        figure: Figure = ExperimentPlotter._create_figure(figsize=(7.2, 4.2))
        axis: PlotAxis = cast(PlotFigure, figure).add_subplot(1, 1, 1)
        axis.errorbar(
            results[ResultColumn.MIX],
            results[ResultColumn.CONFIDENCE_A_MEAN].astype(float),
            yerr=results[ResultColumn.CONFIDENCE_A_STD].astype(float),
            marker="o",
            capsize=4,
            label=PlotLabel.TARGET_A_MEAN_1_SD,
        )
        axis.errorbar(
            results[ResultColumn.MIX],
            results[ResultColumn.CONFIDENCE_B_MEAN].astype(float),
            yerr=results[ResultColumn.CONFIDENCE_B_STD].astype(float),
            marker="o",
            capsize=4,
            label=PlotLabel.TARGET_B_MEAN_1_SD,
        )
        axis.set_xlabel(PlotLabel.A_B_MIX_AMPLITUDE_COEFFICIENTS)
        axis.set_ylabel(PlotLabel.BIRDNET_CONFIDENCE)
        axis.set_ylim(0.0, 1.0)
        axis.set_title(PlotLabel.EFFECT_OF_OVERLAPPING_BIRD_VOCALIZATIONS)
        axis.grid(True, alpha=0.25)
        axis.legend()
        figure.tight_layout()
        ExperimentPlotter.save_figure(figure, destination)

    @staticmethod
    def plot_threshold(results: DataFrame, destination: Path) -> None:
        """Plot retained detections and species against confidence threshold.

        :param results: Threshold experiment results.
        :type results: DataFrame
        :param destination: Output image path.
        :type destination: Path
        :return: None.
        :rtype: None
        """
        axis_value: PlotAxis
        metric_value: ResultColumn
        figure: Figure = ExperimentPlotter._create_figure(figsize=(11.0, 4.2))
        axes: tuple[PlotAxis, PlotAxis] = ExperimentPlotter._pair_axes(figure)
        count_axis: PlotAxis = axes[0]
        rate_axis: PlotAxis = axes[1]
        thresholds: Series[float] = results[ResultColumn.THRESHOLD].astype(
            float
        )
        count_axis.plot(
            thresholds,
            results[ResultColumn.DETECTIONS],
            marker="o",
            label=PlotLabel.DETECTIONS,
        )
        count_axis.plot(
            thresholds,
            results[ResultColumn.DISTINCT_SPECIES],
            marker="o",
            label=PlotLabel.DISTINCT_SPECIES,
        )
        count_axis.set_ylabel(PlotLabel.COUNT_ACROSS_RECORDINGS)
        count_axis.set_title(PlotLabel.RETAINED_PREDICTIONS)
        rate_axis.plot(
            thresholds,
            results[ResultColumn.TARGET_DETECTION_RATE],
            marker="o",
            label=PlotLabel.METADATA_TARGET_RECOVERED,
        )
        for metric_value in (
            ResultColumn.PRECISION,
            ResultColumn.RECALL,
            ResultColumn.F1_SCORE,
        ):
            metric: str = metric_value
            if results[metric].notna().any():
                rate_axis.plot(
                    thresholds, results[metric], marker="o", label=metric
                )
        rate_axis.set_ylim(0.0, 1.05)
        rate_axis.set_ylabel(PlotLabel.RECORDING_LEVEL_RATE)
        rate_axis.set_title(
            PlotLabel.TARGET_RETENTION_OPTIONAL_REFERENCE_METRICS
        )
        for axis_value in (count_axis, rate_axis):
            axis: PlotAxis = axis_value
            axis.set_xlabel(PlotLabel.CONFIDENCE_THRESHOLD)
            axis.grid(True, alpha=0.25)
            axis.legend()
        figure.tight_layout()
        ExperimentPlotter.save_figure(figure, destination)

    @staticmethod
    def plot_spectrogram(
        samples: FloatArray,
        sample_rate: int,
        destination: Path,
    ) -> None:
        """Plot a power spectrogram for an audio signal.

        :param samples: Audio samples.
        :type samples: FloatArray
        :param sample_rate: Sampling rate in hertz.
        :type sample_rate: int
        :param destination: Output image path.
        :type destination: Path
        :return: None.
        :rtype: None
        """
        if samples.size == 0 or sample_rate <= 0:
            raise ValueError(ErrorMessage.INVALID_SPECTROGRAM)
        window_size: int = min(1024, int(samples.size))
        spectrogram_result: tuple[Float64Array, Float64Array, Float64Array] = (
            spectrogram(
                samples.astype(float64),
                fs=float(sample_rate),
                nperseg=window_size,
                noverlap=min(768, window_size - 1),
                scaling="spectrum",
            )
        )
        frequencies: Float64Array = spectrogram_result[0]
        times: Float64Array = spectrogram_result[1]
        power: Float64Array = spectrogram_result[2]
        safe_power: Float64Array = maximum(power, 1e-12)
        power_db: Float64Array = 10.0 * log10(safe_power)

        figure: Figure = ExperimentPlotter._create_figure(figsize=(8.0, 4.0))
        axis: PlotAxis = cast(PlotFigure, figure).add_subplot(1, 1, 1)
        mesh: QuadMesh = axis.pcolormesh(
            times,
            frequencies,
            power_db,
            shading="auto",
        )
        axis.set_ylim(0.0, 15_000.0)
        axis.set_xlabel(PlotLabel.TIME_S)
        axis.set_ylabel(PlotLabel.FREQUENCY_HZ)
        axis.set_title(PlotLabel.TARGET_SEGMENT_SPECTROGRAM)
        cast(PlotFigure, figure).colorbar(
            mesh, ax=axis, label=PlotLabel.POWER_DB
        )
        figure.tight_layout()
        ExperimentPlotter.save_figure(figure, destination)
