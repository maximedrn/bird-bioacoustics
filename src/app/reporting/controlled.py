"""Publish source validation and controlled perturbations from saved scores."""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast

from matplotlib.figure import Figure
from pandas import DataFrame, Series, concat, read_csv

from app.domain.constants import MetadataKey, ResultColumn
from app.domain.settings import ProjectPaths
from app.experiments.constants import (
    ControlColumn,
    ControlSetting,
    ExperimentColumn,
    MetricColumn,
    SoundscapeDataset,
    SoundscapeVersion,
    ValidationSetting,
)
from app.reporting.analysis import bounds_cells, interval_cells, tex_table
from app.reporting.constants import ExperimentSection
from app.reporting.messages import ExperimentLabel
from app.reporting.plots import ExperimentPlotter
from app.reporting.protocols import PlotAxis, PlotFigure
from app.reporting.writing import atomic_output, write_table

EXTENSION_OBSERVATIONS: Final[dict[str, str]] = {
    ExperimentSection.SOURCE_VALIDATION: "The threshold is chosen on calibration sources; these F1 scores use different test sources. Intervals condition on that chosen threshold.",
    ExperimentSection.CONTROLLED_NOISE: "These clips were selected from annotations, including initially missed species. Natural backgrounds come from separate sources without annotated birds.",
    ExperimentSection.CONTROLLED_NOISE_FIGURE: "Gaussian and recorded backgrounds can affect the same annotated targets differently at the same added signal-to-noise ratio.",
    ExperimentSection.CONTROLLED_OVERLAP: "The joint recovery difference compares the mixture with its two isolated signals at identical gains; each source participates in one pair.",
    ExperimentSection.CONTROLLED_OVERLAP_FIGURE: "Equal source RMS controls the relative input levels. Isolated controls show whether both targets were recoverable before superposition.",
}


def paired_validation_frame(frame: DataFrame) -> DataFrame:
    """Present paired F1 differences at independently calibrated thresholds.

    :param frame: Matched model differences on the held-out source partition.
    :type frame: DataFrame
    :return: Both dataset differences with their source-bootstrap intervals.
    :rtype: DataFrame
    """
    return DataFrame(
        {
            ExperimentLabel.DATASET: frame[ExperimentColumn.DATASET],
            ExperimentLabel.F1_DIFFERENCE_INTERVAL: bounds_cells(
                frame,
                MetricColumn.F1_DELTA_3_MINUS_2,
                ExperimentColumn.LOW,
                ExperimentColumn.HIGH,
                signed=True,
            ),
        }
    )


def extension_frame(section: str, frame: DataFrame) -> DataFrame:
    """Format complete additional measurements for notebook and LaTeX tables.

    :param section: Source validation, controlled noise or controlled overlap.
    :type section: str
    :param frame: All saved observations for this table.
    :type frame: DataFrame
    :return: Every measured row in a readable presentation frame.
    :rtype: DataFrame
    """
    if section == ExperimentSection.SOURCE_VALIDATION:
        return DataFrame(
            {
                ExperimentLabel.DATASET: frame[ExperimentColumn.DATASET],
                ExperimentLabel.VERSION: frame[MetadataKey.MODEL],
                ExperimentLabel.THRESHOLD: frame[ResultColumn.THRESHOLD].map(
                    lambda value: f"{value:.2f}"
                ),
                ExperimentLabel.SOURCE_SPLIT: [
                    f"{calibration} / {test}"
                    for calibration, test in zip(
                        frame[ValidationSetting.CALIBRATION_SOURCES],
                        frame[ExperimentColumn.SOURCE_RECORDINGS],
                        strict=True,
                    )
                ],
                ExperimentLabel.PRECISION: frame[
                    MetricColumn.PRECISION_MICRO
                ].map(lambda value: f"{value:.3f}"),
                ExperimentLabel.RECALL: frame[MetricColumn.RECALL_MICRO].map(
                    lambda value: f"{value:.3f}"
                ),
                ExperimentLabel.F1_95_INTERVAL: interval_cells(
                    frame, MetricColumn.F1_MICRO
                ),
            }
        )
    if section == ExperimentSection.CONTROLLED_NOISE:
        condition_label: str = ExperimentLabel.CONDITION
        conditions: Series[str] = frame[ControlColumn.CONDITION] + frame[
            ControlColumn.SNR
        ].map(lambda value: "" if value == float("inf") else f" {value:g} dB")
    else:
        condition_label = ExperimentLabel.MIX
        conditions = frame[ControlColumn.RATIO]
    columns: dict[str, object] = {
        ExperimentLabel.DATASET: frame[ExperimentColumn.DATASET],
        condition_label: conditions,
        ExperimentLabel.CONTROL_UNITS: frame[ControlColumn.UNITS],
        ExperimentLabel.CONTROL_GROUPS: frame[ControlColumn.GROUPS],
        ExperimentLabel.RECOVERY_95_INTERVAL: bounds_cells(
            frame,
            ControlColumn.ESTIMATE,
            ControlColumn.LOW,
            ControlColumn.HIGH,
            percent=True,
        ),
        ExperimentLabel.PAIRED_POINTS: [
            f"{100 * value:+.1f} [{100 * lower:+.1f}; {100 * upper:+.1f}]"
            for value, lower, upper in zip(
                frame[ControlColumn.DELTA],
                frame["delta_low"],
                frame["delta_high"],
                strict=True,
            )
        ],
    }
    if section == ExperimentSection.CONTROLLED_OVERLAP:
        columns[ExperimentLabel.ISOLATED_RECOVERY] = frame[
            ControlColumn.CLEAN_RECOVERY
        ].map(lambda value: f"{value:.1%}")
    return DataFrame(columns)


def plot_controls(frame: DataFrame, kind: str, root: Path) -> None:
    """Contrast perturbations with their own unperturbed or isolated controls.

    :param frame: Complete summaries for both reference datasets.
    :type frame: DataFrame
    :param kind: Noise or RMS-controlled overlap.
    :type kind: str
    :param root: Destination of the PNG and PDF comparison figure.
    :type root: Path
    :return: None.
    :rtype: None
    """
    figure: Figure = ExperimentPlotter.create_figure((10.0, 3.5))
    target: PlotFigure = cast(PlotFigure, figure)
    target.set_layout_engine("constrained")
    for index, dataset in enumerate(SoundscapeDataset, start=1):
        axis: PlotAxis = target.add_subplot(1, 2, index)
        subset: DataFrame = frame[frame[ExperimentColumn.DATASET] == dataset]
        if kind == ControlSetting.NOISE:
            axis.axhline(
                float(
                    subset.loc[
                        subset[ControlColumn.CONDITION]
                        == ControlSetting.CLEAN,
                        ControlColumn.ESTIMATE,
                    ].iloc[0]
                ),
                linestyle="--",
                linewidth=1,
                label="Sans ajout",
            )
            for condition in (ControlSetting.GAUSSIAN, ControlSetting.NATURAL):
                series: DataFrame = subset[
                    subset[ControlColumn.CONDITION] == condition
                ].sort_values(ControlColumn.SNR)
                axis.errorbar(
                    series[ControlColumn.SNR],
                    series[ControlColumn.ESTIMATE],
                    yerr=[
                        series[ControlColumn.ESTIMATE]
                        - series[ControlColumn.LOW],
                        series[ControlColumn.HIGH]
                        - series[ControlColumn.ESTIMATE],
                    ],
                    marker="o"
                    if condition == ControlSetting.GAUSSIAN
                    else "s",
                    capsize=3,
                    label=(
                        "Bruit gaussien"
                        if condition == ControlSetting.GAUSSIAN
                        else "Fond enregistré"
                    ),
                    linestyle="-"
                    if condition == ControlSetting.GAUSSIAN
                    else "--",
                )
            axis.set_xlabel(ExperimentLabel.ADDED_SNR)
            axis.set_xticks([20, 10, 5, 0])
            axis.invert_xaxis()
            axis.set_ylabel("Récupération moyenne par extrait")
        else:
            subset = subset.sort_values(ControlColumn.RATIO)
            levels: list[float] = [
                float(str(value).split("/")[0])
                for value in subset[ControlColumn.RATIO]
            ]
            axis.plot(
                levels,
                subset[ControlColumn.CLEAN_RECOVERY],
                marker="s",
                linestyle="--",
                label="Deux cibles retrouvées\n(témoins isolés)",
            )
            axis.errorbar(
                levels,
                subset[ControlColumn.ESTIMATE],
                yerr=[
                    subset[ControlColumn.ESTIMATE] - subset[ControlColumn.LOW],
                    subset[ControlColumn.HIGH]
                    - subset[ControlColumn.ESTIMATE],
                ],
                marker="o",
                capsize=3,
                label="Deux cibles retrouvées\n(mélange)",
            )
            axis.set_xticks(levels, ["50/50", "75/25", "90/10"])
            axis.set_xlabel("Coefficients d'amplitude A/B")
            axis.set_ylabel("Récupération conjointe")
        axis.set_title(str(dataset))
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
        axis.legend()
    ExperimentPlotter.save_figure_formats(
        figure, (root / "comparison.png", root / "comparison.pdf")
    )


def prepare_extensions(paths: ProjectPaths) -> None:
    """Regenerate all additional tables and figures without audio or inference.

    :param paths: Saved measured results and grouped LaTeX table destinations.
    :type paths: ProjectPaths
    :return: None.
    :rtype: None
    """
    macros: dict[str, str] = {}
    validation_path: Path = (
        paths.results / ValidationSetting.DIRECTORY / ValidationSetting.METRICS
    )
    if validation_path.exists():
        validation: DataFrame = read_csv(
            validation_path, dtype={MetadataKey.MODEL: str}
        )
        tex_table(
            extension_frame(ExperimentSection.SOURCE_VALIDATION, validation),
            paths.report_tables / "benchmark/validation_table.tex",
        )
        for dataset in SoundscapeDataset:
            for version in SoundscapeVersion:
                row: DataFrame = validation[
                    (validation[ExperimentColumn.DATASET] == dataset)
                    & (validation[MetadataKey.MODEL] == version)
                ]
                prefix: str = dataset.value + (
                    "Two" if version == SoundscapeVersion.V2_4 else "Three"
                )
                macros[prefix + "Threshold"] = (
                    f"{float(row[ResultColumn.THRESHOLD].iloc[0]):.2f}"
                )
                macros[prefix + "FOne"] = (
                    f"{float(row[MetricColumn.F1_MICRO].iloc[0]):.3f}"
                )
        paired: DataFrame = read_csv(
            paths.results
            / ValidationSetting.DIRECTORY
            / ValidationSetting.PAIRED
        )
        tex_table(
            paired_validation_frame(paired),
            paths.report_tables / "benchmark/validation_paired_table.tex",
        )
        for dataset in SoundscapeDataset:
            row = paired[paired[ExperimentColumn.DATASET] == dataset]
            for name, field in (
                ("Difference", MetricColumn.F1_DELTA_3_MINUS_2),
                ("Lower", ExperimentColumn.LOW),
                ("Upper", ExperimentColumn.HIGH),
            ):
                macros[dataset.value + "Calibrated" + name] = (
                    f"{float(row[field].iloc[0]):+.3f}"
                )
    for kind in (ControlSetting.NOISE, ControlSetting.OVERLAP):
        root: Path = paths.results / ControlSetting.DIRECTORY / kind
        frames: list[DataFrame] = [
            read_csv(root / dataset / ControlSetting.SUMMARY)
            for dataset in SoundscapeDataset
            if (root / dataset / ControlSetting.SUMMARY).exists()
        ]
        if len(frames) != len(SoundscapeDataset):
            continue
        frame: DataFrame = concat(frames, ignore_index=True)
        write_table(frame, root / ControlSetting.SUMMARY)
        section: ExperimentSection = (
            ExperimentSection.CONTROLLED_NOISE
            if kind == ControlSetting.NOISE
            else ExperimentSection.CONTROLLED_OVERLAP
        )
        tex_table(
            extension_frame(section, frame),
            paths.report_tables
            / ControlSetting.DIRECTORY
            / f"{kind}_table.tex",
        )
        plot_controls(frame, kind, root)
        for dataset in SoundscapeDataset:
            subset: DataFrame = frame[
                frame[ExperimentColumn.DATASET] == dataset
            ]
            if kind == ControlSetting.NOISE:
                macros[dataset.value + "ControlledClean"] = (
                    f"{float(subset.loc[subset[ControlColumn.CONDITION] == ControlSetting.CLEAN, ControlColumn.ESTIMATE].iloc[0]) * 100:.1f}"
                )
                for condition in (
                    ControlSetting.GAUSSIAN,
                    ControlSetting.NATURAL,
                ):
                    value: float = float(
                        subset.loc[
                            (subset[ControlColumn.CONDITION] == condition)
                            & (subset[ControlColumn.SNR] == 0),
                            ControlColumn.ESTIMATE,
                        ].iloc[0]
                    )
                    macros[dataset.value + condition.capitalize() + "Zero"] = (
                        f"{value * 100:.1f}"
                    )
            else:
                row = subset[subset[ControlColumn.RATIO] == "0.5/0.5"]
                macros[dataset.value + "ControlledPairs"] = str(
                    int(row[ControlColumn.UNITS].iloc[0])
                )
                macros[dataset.value + "IsolatedBoth"] = (
                    f"{float(row[ControlColumn.CLEAN_RECOVERY].iloc[0]) * 100:.1f}"
                )
                macros[dataset.value + "MixedBoth"] = (
                    f"{float(row[ControlColumn.ESTIMATE].iloc[0]) * 100:.1f}"
                )
    if macros:
        with atomic_output(
            paths.report_tables / ControlSetting.DIRECTORY / "numbers.tex"
        ) as temporary:
            lines: list[str] = [
                "\\newcommand{\\" + name + "}{" + value + "}"
                for name, value in macros.items()
            ]
            temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
