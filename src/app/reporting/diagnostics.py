"""Render distributions, paired responses and cumulative corpus diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from math import ceil
from pathlib import Path
from typing import Final, cast

from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from numpy import float64, isnan, ndenumerate, zeros
from pandas import DataFrame, Series

from app.domain.constants import Artifact, NoiseCondition, ResultColumn
from app.domain.types import Float64Array
from app.reporting.constants import ChartSetting, DetailColumn
from app.reporting.diagnostic_protocols import DiagnosticAxis
from app.reporting.messages import ObservationMessage, PlotLabel
from app.reporting.plots import ExperimentPlotter
from app.reporting.protocols import PlotFigure


class DiagnosticPlotter:
    """Draw seven compact charts from detached cumulative measurements."""

    @staticmethod
    def _axis(
        figure: Figure, rows: int = 1, columns: int = 1, index: int = 1
    ) -> DiagnosticAxis:
        """Create an axis using the existing typed figure boundary.

        :param figure: Owning Agg figure.
        :type figure: Figure
        :param rows: Number of panel rows.
        :type rows: int
        :param columns: Number of panel columns.
        :type columns: int
        :param index: One-based panel index.
        :type index: int
        :return: Native Matplotlib axis for diagnostic artists.
        :rtype: DiagnosticAxis
        """
        return cast(
            DiagnosticAxis,
            cast(PlotFigure, figure).add_subplot(rows, columns, index),
        )

    @staticmethod
    def _condition(value: str) -> str:
        """Use a readable label for the unperturbed segment condition.

        :param value: Stored condition name.
        :type value: str
        :return: English condition label.
        :rtype: str
        """
        return (
            PlotLabel.ORIGINAL if value == NoiseCondition.ORIGINAL else value
        )

    @staticmethod
    def _conditions(results: DataFrame) -> list[str]:
        """Order conditions by decreasing SNR with the original first.

        :param results: Measurements with condition names and numeric SNR.
        :type results: DataFrame
        :return: Deterministically ordered condition labels.
        :rtype: list[str]
        """
        conditions: DataFrame = (
            results[[ResultColumn.CONDITION, ResultColumn.SNR_DB]]
            .drop_duplicates()
            .sort_values(ResultColumn.SNR_DB, ascending=False)
        )
        return conditions[ResultColumn.CONDITION].tolist()

    @staticmethod
    def noise_distribution(results: DataFrame, destination: Path) -> None:
        """Plot exact quartiles and full ranges of recording-level means.

        :param results: Per-condition quartiles and recording counts.
        :type results: DataFrame
        :param destination: Saved PNG destination.
        :type destination: Path
        :return: None.
        :rtype: None
        """
        figure: Figure = ExperimentPlotter.create_figure((11.0, 4.8))
        axis: DiagnosticAxis = DiagnosticPlotter._axis(figure)
        statistics: list[dict[str, float | str]] = []
        for _, row in results.sort_values(
            ResultColumn.SNR_DB, ascending=False
        ).iterrows():
            statistics.append(
                {
                    "label": PlotLabel.CONDITION_COUNT.format(
                        condition=DiagnosticPlotter._condition(
                            str(row[ResultColumn.CONDITION])
                        ),
                        count=int(row[ResultColumn.RECORDINGS]),
                    ),
                    "med": float(row[DetailColumn.MEDIAN]),
                    "q1": float(row[DetailColumn.Q1]),
                    "q3": float(row[DetailColumn.Q3]),
                    "whislo": float(row[DetailColumn.MINIMUM]),
                    "whishi": float(row[DetailColumn.MAXIMUM]),
                }
            )
        axis.bxp(statistics, showfliers=False)
        axis.set_title(PlotLabel.NOISE_DISTRIBUTION)
        axis.set_xlabel(PlotLabel.NOISE_BOX_RANGE)
        axis.set_ylabel(PlotLabel.TARGET_SPECIES_CONFIDENCE)
        axis.set_ylim(-0.03, 1.03)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        ExperimentPlotter.save_figure(figure, destination)

    @staticmethod
    def species_noise(results: DataFrame, destination: Path) -> None:
        """Plot species recovery cells with their own independent sample size.

        :param results: Equally weighted species rates by noise condition.
        :type results: DataFrame
        :param destination: Saved PNG destination.
        :type destination: Path
        :return: None.
        :rtype: None
        """
        conditions: list[str] = DiagnosticPlotter._conditions(results)
        counts: Series[int] = (
            results.groupby(ResultColumn.TARGET_SPECIES)[
                ResultColumn.RECORDINGS
            ]
            .max()
            .sort_values(ascending=False)
        )
        species: list[str] = [str(value) for value in counts.index]
        rates: DataFrame = results.pivot(
            index=ResultColumn.TARGET_SPECIES,
            columns=ResultColumn.CONDITION,
            values=ResultColumn.DETECTION_RATE,
        ).reindex(index=species, columns=conditions)
        sizes: DataFrame = results.pivot(
            index=ResultColumn.TARGET_SPECIES,
            columns=ResultColumn.CONDITION,
            values=ResultColumn.RECORDINGS,
        ).reindex(index=species, columns=conditions)
        matrix: Float64Array = rates.to_numpy(dtype=float64)
        figure: Figure = ExperimentPlotter.create_figure(
            (12.0, max(4.8, len(species) * 0.5 + 1.5))
        )
        axis: DiagnosticAxis = DiagnosticPlotter._axis(figure)
        image: AxesImage = axis.imshow(
            matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto"
        )
        axis.set_xticks(
            range(len(conditions)),
            [DiagnosticPlotter._condition(value) for value in conditions],
        )
        axis.set_yticks(
            range(len(species)), [value.split("_", 1)[0] for value in species]
        )
        DiagnosticPlotter._heatmap_counts(
            axis, matrix, sizes.to_numpy(dtype=float64)
        )
        axis.set_title(PlotLabel.SPECIES_NOISE)
        cast(PlotFigure, figure).colorbar(image, ax=axis, label=PlotLabel.RATE)
        figure.tight_layout()
        ExperimentPlotter.save_figure(figure, destination)

    @staticmethod
    def _heatmap_counts(
        axis: DiagnosticAxis, matrix: Float64Array, sizes: Float64Array
    ) -> None:
        """Annotate available cells without presenting missing values as zero.

        :param axis: Species heatmap axis.
        :type axis: DiagnosticAxis
        :param matrix: Rates with NaN for a missing species-condition pair.
        :type matrix: Float64Array
        :param sizes: Independent recording counts in the same cell order.
        :type sizes: Float64Array
        :return: None.
        :rtype: None
        """
        for coordinates, value in ndenumerate(matrix):
            row: int = coordinates[0]
            column: int = coordinates[1]
            rate: float = float(value)
            if isnan(rate):
                continue
            axis.text(
                column,
                row,
                PlotLabel.SPECIES_RATE_COUNT.format(
                    rate=rate, count=int(sizes[row, column])
                ),
                ha="center",
                va="center",
                color="black" if rate > 0.55 else "white",
                fontsize=8,
            )

    @staticmethod
    def noise_pairs(results: DataFrame, destination: Path) -> None:
        """Compare each segment's original confidence to its noisy mean.

        :param results: Paired recording observations sampled within each SNR.
        :type results: DataFrame
        :param destination: Saved PNG destination.
        :type destination: Path
        :return: None.
        :rtype: None
        """
        conditions: list[str] = DiagnosticPlotter._conditions(results)
        panel_columns: int = min(2, len(conditions))
        panel_rows: int = ceil(len(conditions) / panel_columns)
        figure: Figure = ExperimentPlotter.create_figure(
            (11.0, 4.5 * panel_rows)
        )
        for index, condition in enumerate(conditions, start=1):
            data: DataFrame = results.loc[
                results[ResultColumn.CONDITION] == condition
            ]
            axis: DiagnosticAxis = DiagnosticPlotter._axis(
                figure, panel_rows, panel_columns, index
            )
            axis.scatter(
                data[DetailColumn.ORIGINAL_CONFIDENCE].to_numpy(dtype=float64),
                data[ResultColumn.CONFIDENCE_MEAN].to_numpy(dtype=float64),
                s=9,
                alpha=0.2,
            )
            axis.plot(
                [0.0, 1.0],
                [0.0, 1.0],
                linestyle="--",
                label=PlotLabel.UNCHANGED,
            )
            axis.set_title(
                PlotLabel.SAMPLE_COUNT.format(
                    label=condition,
                    shown=len(data),
                    total=int(data[DetailColumn.TOTAL_RECORDINGS].iloc[0]),
                )
            )
            axis.set_xlabel(PlotLabel.ORIGINAL_CONFIDENCE)
            axis.set_ylabel(PlotLabel.NOISY_CONFIDENCE)
            axis.set_xlim(-0.03, 1.03)
            axis.set_ylim(-0.03, 1.03)
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.2)
        cast(PlotFigure, figure).suptitle(PlotLabel.PAIRED_NOISE)
        figure.tight_layout()
        ExperimentPlotter.save_figure(figure, destination)

    @staticmethod
    def overlap_outcomes(results: DataFrame, destination: Path) -> None:
        """Plot mutually exclusive joint recovery outcomes for every mixture.

        :param results: Pair counts in the four recovery categories.
        :type results: DataFrame
        :param destination: Saved PNG destination.
        :type destination: Path
        :return: None.
        :rtype: None
        """
        figure: Figure = ExperimentPlotter.create_figure((11.0, 4.8))
        axis: DiagnosticAxis = DiagnosticPlotter._axis(figure)
        counts: Float64Array = results[ResultColumn.PAIRS].to_numpy(
            dtype=float64
        )
        bottom: Float64Array = zeros(len(results), dtype=float64)
        labels: list[str] = [
            PlotLabel.CONDITION_COUNT.format(condition=mix, count=int(count))
            for mix, count in zip(
                results[ResultColumn.MIX], counts, strict=True
            )
        ]
        for column, label, color in (
            (DetailColumn.BOTH, PlotLabel.BOTH, "C2"),
            (DetailColumn.ONLY_A, PlotLabel.ONLY_A, "C0"),
            (DetailColumn.ONLY_B, PlotLabel.ONLY_B, "C1"),
            (DetailColumn.NEITHER, PlotLabel.NEITHER, "C3"),
        ):
            proportions: Float64Array = (
                results[column].to_numpy(dtype=float64) / counts
            )
            axis.bar(
                labels, proportions, bottom=bottom, label=label, color=color
            )
            bottom += proportions
        axis.set_title(
            PlotLabel.OVERLAP_OUTCOMES.format(
                threshold=float(results[ResultColumn.THRESHOLD].iloc[0])
            )
        )
        axis.set_xlabel(PlotLabel.A_B_MIX_AMPLITUDE_COEFFICIENTS)
        axis.set_ylabel(PlotLabel.PAIR_PROPORTION)
        axis.set_ylim(0.0, 1.05)
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=4)
        figure.tight_layout()
        ExperimentPlotter.save_figure(figure, destination)

    @staticmethod
    def species_coverage(results: DataFrame, destination: Path) -> None:
        """Show the most represented supported primary species.

        :param results: All species counts among successful recordings.
        :type results: DataFrame
        :param destination: Saved PNG destination.
        :type destination: Path
        :return: None.
        :rtype: None
        """
        data: DataFrame = results.head(ChartSetting.TOP_SPECIES).iloc[::-1]
        figure: Figure = ExperimentPlotter.create_figure(
            (12.0, max(4.5, len(data) * 0.35 + 1.5))
        )
        axis: DiagnosticAxis = DiagnosticPlotter._axis(figure)
        labels: list[str] = [
            str(species).split("_", 1)[0]
            for species in data[ResultColumn.TARGET_SPECIES]
        ]
        bars: BarContainer = axis.barh(
            labels, data[ResultColumn.RECORDINGS].to_numpy(dtype=float64)
        )
        axis.bar_label(
            bars,
            labels=[
                f"{int(value):,}" for value in data[ResultColumn.RECORDINGS]
            ],
            padding=3,
        )
        maximum_count: int = int(data[ResultColumn.RECORDINGS].max())
        axis.set_xlim(0.0, maximum_count * 1.15)
        axis.set_title(
            PlotLabel.SPECIES_COVERAGE.format(
                shown=len(data), total=len(results)
            )
        )
        axis.set_xlabel(PlotLabel.ANALYSED_RECORDINGS)
        axis.grid(axis="x", alpha=0.25)
        axis.set_axisbelow(True)
        figure.tight_layout()
        ExperimentPlotter.save_figure(figure, destination)

    @staticmethod
    def duration_confidence(results: DataFrame, destination: Path) -> None:
        """Show duration associations with a full-file target maximum.

        :param results: Supported targets with positive measured durations.
        :type results: DataFrame
        :param destination: Saved PNG destination.
        :type destination: Path
        :return: None.
        :rtype: None
        """
        figure: Figure = ExperimentPlotter.create_figure((11.0, 4.8))
        axis: DiagnosticAxis = DiagnosticPlotter._axis(figure)
        axis.scatter(
            results[DetailColumn.DURATION_SECONDS].to_numpy(dtype=float64),
            results[DetailColumn.TARGET_CONFIDENCE].to_numpy(dtype=float64),
            s=12,
            alpha=0.25,
        )
        axis.set_xscale("log")
        axis.set_ylim(-0.03, 1.03)
        axis.set_title(
            PlotLabel.SAMPLE_COUNT.format(
                label=PlotLabel.DURATION_CONFIDENCE,
                shown=len(results),
                total=int(results[DetailColumn.TOTAL_RECORDINGS].iloc[0]),
            )
        )
        axis.set_xlabel(PlotLabel.RECORDING_DURATION)
        axis.set_ylabel(PlotLabel.TARGET_MAXIMUM)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        ExperimentPlotter.save_figure(figure, destination)

    @staticmethod
    def batch_history(results: DataFrame, destination: Path) -> None:
        """Plot cumulative estimates weighted by recording count.

        :param results: Recording-weighted cumulative values in batch order.
        :type results: DataFrame
        :param destination: Saved PNG destination.
        :type destination: Path
        :return: None.
        :rtype: None
        """
        figure: Figure = ExperimentPlotter.create_figure((13.0, 4.8))
        axes: tuple[DiagnosticAxis, DiagnosticAxis] = (
            DiagnosticPlotter._axis(figure, 1, 2, 1),
            DiagnosticPlotter._axis(figure, 1, 2, 2),
        )
        condition: str
        for condition in DiagnosticPlotter._conditions(results):
            data: DataFrame = results.loc[
                results[ResultColumn.CONDITION] == condition
            ].sort_values(DetailColumn.BATCH_ID)
            label: str = PlotLabel.CUMULATIVE_COUNT.format(
                condition=DiagnosticPlotter._condition(str(condition)),
                count=int(data[ResultColumn.RECORDINGS].iloc[-1]),
            )
            DiagnosticPlotter._cumulative_series(
                axes[0], data, ResultColumn.CONFIDENCE_MEAN, label
            )
            DiagnosticPlotter._cumulative_series(
                axes[1], data, ResultColumn.DETECTION_RATE, label
            )
        axes[0].set_ylabel(PlotLabel.CUMULATIVE_CONFIDENCE)
        axes[1].set_ylabel(PlotLabel.CUMULATIVE_RATE)
        batch_ids: list[int] = sorted(
            results[DetailColumn.BATCH_ID].astype(int).unique().tolist()
        )
        stride: int = max(
            1, ceil((len(batch_ids) - 1) / (ChartSetting.BATCH_TICKS - 1))
        )
        ticks: list[int] = batch_ids[::stride]
        if ticks[-1] != batch_ids[-1]:
            ticks.append(batch_ids[-1])
        for axis in axes:
            axis.set_ylim(-0.03, 1.03)
            axis.set_xlabel(PlotLabel.BATCH_NUMBER)
            axis.set_xticks(ticks)
            axis.legend(fontsize=8)
            axis.grid(alpha=0.25)
        cast(PlotFigure, figure).suptitle(PlotLabel.BATCH_HISTORY)
        figure.tight_layout()
        ExperimentPlotter.save_figure(figure, destination)

    @staticmethod
    def _cumulative_series(
        axis: DiagnosticAxis, data: DataFrame, column: ResultColumn, label: str
    ) -> None:
        """Draw one cumulative response without changing its recording weights.

        :param axis: Confidence or recovery panel.
        :type axis: DiagnosticAxis
        :param data: One condition's values ordered by batch.
        :type data: DataFrame
        :param column: Cumulative quantity to draw.
        :type column: ResultColumn
        :param label: Condition and latest independent recording count.
        :type label: str
        :return: None.
        :rtype: None
        """
        axis.plot(
            data[DetailColumn.BATCH_ID].to_numpy(dtype=float64),
            data[column].to_numpy(dtype=float64),
            marker="o",
            label=label,
        )


DIAGNOSTIC_PLOTS: Final[dict[Artifact, Callable[[DataFrame, Path], None]]] = {
    Artifact.NOISE_DISTRIBUTION: DiagnosticPlotter.noise_distribution,
    Artifact.SPECIES_NOISE: DiagnosticPlotter.species_noise,
    Artifact.NOISE_PAIRS: DiagnosticPlotter.noise_pairs,
    Artifact.OVERLAP_OUTCOMES: DiagnosticPlotter.overlap_outcomes,
    Artifact.SPECIES_COVERAGE: DiagnosticPlotter.species_coverage,
    Artifact.DURATION_CONFIDENCE: DiagnosticPlotter.duration_confidence,
    Artifact.BATCH_HISTORY: DiagnosticPlotter.batch_history,
}

DIAGNOSTIC_OBSERVATIONS: Final[dict[Artifact, str]] = {
    Artifact.NOISE_DISTRIBUTION: ObservationMessage.NOISE_DISTRIBUTION,
    Artifact.SPECIES_NOISE: ObservationMessage.SPECIES_NOISE,
    Artifact.NOISE_PAIRS: ObservationMessage.NOISE_PAIRS,
    Artifact.OVERLAP_OUTCOMES: ObservationMessage.OVERLAP_OUTCOMES,
    Artifact.SPECIES_COVERAGE: ObservationMessage.SPECIES_COVERAGE,
    Artifact.DURATION_CONFIDENCE: ObservationMessage.DURATION_CONFIDENCE,
    Artifact.BATCH_HISTORY: ObservationMessage.BATCH_HISTORY,
}
