"""Export recording-level uncertainty from the saved Xeno-canto experiment."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from datetime import UTC, datetime
from json import dumps, loads
from pathlib import Path
from re import sub
from typing import cast

from matplotlib.figure import Figure
from matplotlib.pyplot import close
from numpy import allclose, asarray, float64
from numpy.typing import ArrayLike
from pandas import DataFrame, Series, concat, read_sql_query
from pandas.io.formats.style import Styler
from sqlalchemy import Engine, create_engine, select

from app.domain.constants import (
    CoverageKey,
    MetadataKey,
    NoiseCondition,
    PredictionColumn,
    ProcessingStatus,
    ResultColumn,
)
from app.domain.settings import ExperimentSettings, ModelSettings, ProjectPaths
from app.domain.types import Float64Array, StatisticalInterval
from app.experiments.constants import ExperimentColumn, StatisticalSetting
from app.experiments.statistics import bootstrap_mean
from app.reporting.constants import (
    FRENCH_COUNTRIES,
    LATEX_TABLE_HEADINGS,
    AnalysisSetting,
    DetailColumn,
    ExperimentFile,
    TableFormat,
)
from app.reporting.messages import AnalysisError, ExperimentLabel
from app.reporting.plots import ExperimentPlotter
from app.reporting.protocols import PlotAxis, PlotFigure
from app.reporting.writing import atomic_output, write_table
from app.storage.connections import SqliteAccess, database_url
from app.storage.schema import (
    MetadataRow,
    NoiseRow,
    NoiseSummaryRow,
    OverlapRow,
    RecordingRow,
    ThresholdRow,
)


@dataclass(frozen=True, slots=True)
class AnalysisMeasurements:
    """Keep detached input tables from one read-only database transaction."""

    completed: DataFrame
    supported: DataFrame
    noise: DataFrame
    trials: DataFrame
    thresholds: DataFrame
    overlap: DataFrame
    baseline: DataFrame
    configuration: dict[str, object]
    failed_recordings: int


def tex_table(frame: DataFrame, path: Path) -> None:
    """Write measured cells with pandas and document-defined LaTeX headings.

    :param frame: Complete table with already formatted presentation values.
    :type frame: DataFrame
    :param path: Destination LaTeX fragment consumed by the report.
    :type path: Path
    :return: None.
    :rtype: None
    """
    column_format: str = "l" + "r" * (len(frame.columns) - 1)
    table: Styler = frame.style
    table.hide(axis="index")
    for column in frame.columns:

        def french_cell(value: object, heading: str = str(column)) -> str:
            text: str = str(value)
            if heading == ExperimentLabel.CONDITION:
                for source, label in (
                    ("original", "Sans ajout"),
                    ("clean", "Sans ajout"),
                    ("gaussian", "Gaussien"),
                    ("natural", "Fond"),
                ):
                    if text == source or text.startswith(source + " "):
                        text = label + text[len(source) :]
                        break
            if heading == ExperimentLabel.COUNTRY:
                text = FRENCH_COUNTRIES.get(text, text)
            if heading == ExperimentLabel.QUALITY and text == "no score":
                text = "Non renseignée"
            if heading != ExperimentLabel.VERSION:
                text = sub(r"([+-]?\d+\.\d+)", r"\\num{\1}", text)
                text = text.replace(r"\%", r"\,\%")
                text = text.replace("; ", r"\,; ")
            return text

        table.format(french_cell, subset=[column], escape="latex")
    table.format_index(
        lambda value: LATEX_TABLE_HEADINGS[str(value)], axis="columns"
    )
    with atomic_output(path) as temporary:
        table.to_latex(
            temporary,
            column_format=column_format,
            hrules=True,
            encoding="utf-8",
        )


def bounds_cells(
    frame: DataFrame,
    point: str,
    low: str,
    high: str,
    *,
    percent: bool = False,
    signed: bool = False,
) -> list[str]:
    """Format independent measured bounds with their estimate.

    :param frame: Measurements with estimate and bound columns.
    :type frame: DataFrame
    :param point: Estimate column name.
    :type point: str
    :param low: Lower confidence-bound column name.
    :type low: str
    :param high: Upper confidence-bound column name.
    :type high: str
    :param percent: Display rates as percentages.
    :type percent: bool
    :param signed: Show explicit signs for differences.
    :type signed: bool
    :return: One formatted interval per input row, preserving its order.
    :rtype: list[str]
    """
    specification: str = (
        TableFormat.PERCENT
        if percent
        else TableFormat.SIGNED
        if signed
        else TableFormat.DECIMAL
    )
    return [
        f"{float(value):{specification}} [{float(lower):{specification}}; {float(upper):{specification}}]"
        for value, lower, upper in zip(
            frame[point], frame[low], frame[high], strict=True
        )
    ]


def interval_cells(
    frame: DataFrame, prefix: str, *, percent: bool = False
) -> list[str]:
    """Format the standard estimate and confidence-bound columns.

    :param frame: Measurements with the standard low and high suffixes.
    :type frame: DataFrame
    :param prefix: Estimate column and prefix of its confidence bounds.
    :type prefix: str
    :param percent: Display estimates and bounds as percentages.
    :type percent: bool
    :return: Formatted confidence intervals in input row order.
    :rtype: list[str]
    """
    return bounds_cells(
        frame,
        prefix,
        prefix + TableFormat.LOW_SUFFIX,
        prefix + TableFormat.HIGH_SUFFIX,
        percent=percent,
    )


def interval_row(values: ArrayLike, prefix: str) -> dict[str, float]:
    """Name an estimate and its percentile bootstrap bounds.

    :param values: Independent recording-level or species-level values.
    :type values: ArrayLike
    :param prefix: Stable measurement column prefix.
    :type prefix: str
    :return: Estimate and bound fields for one exported measurement row.
    :rtype: dict[str, float]
    """
    interval: StatisticalInterval = bootstrap_mean(values)
    return {
        prefix: interval[0],
        prefix + TableFormat.LOW_SUFFIX: interval[1],
        prefix + TableFormat.HIGH_SUFFIX: interval[2],
    }


def read_measurements(paths: ProjectPaths) -> AnalysisMeasurements:
    """Read all dependent measurements in one scoped, read-only transaction.

    :param paths: Existing project checkpoint and output directories.
    :type paths: ProjectPaths
    :return: Detached tables and the saved scientific configuration.
    :rtype: AnalysisMeasurements
    """
    engine: Engine = create_engine(
        database_url(paths.checkpoint, SqliteAccess.READ_ONLY)
    )
    try:
        with engine.connect() as connection, connection.begin():
            recordings: DataFrame = read_sql_query(
                select(
                    RecordingRow.recording_id,
                    RecordingRow.status,
                    RecordingRow.target_species,
                    RecordingRow.noise_skip_reason,
                    RecordingRow.source,
                    RecordingRow.duration_seconds,
                ).where(
                    RecordingRow.status.in_(
                        (ProcessingStatus.DONE, ProcessingStatus.FAILED)
                    )
                ),
                connection,
            )
            noise: DataFrame = read_sql_query(
                select(NoiseSummaryRow), connection
            )
            trials: DataFrame = read_sql_query(select(NoiseRow), connection)
            thresholds: DataFrame = read_sql_query(
                select(ThresholdRow), connection
            )
            overlap: DataFrame = read_sql_query(select(OverlapRow), connection)
            configuration: dict[str, object] = cast(
                dict[str, object],
                loads(
                    connection.execute(
                        select(MetadataRow.value).where(
                            MetadataRow.name == MetadataKey.CONFIGURATION
                        )
                    ).scalar_one()
                ),
            )
    finally:
        engine.dispose()
    completed: DataFrame = recordings[
        recordings[ExperimentColumn.STATUS] == ProcessingStatus.DONE
    ].copy()
    for field, source_key in (
        (ExperimentColumn.COUNTRY, "cnt"),
        (ExperimentColumn.QUALITY, "q"),
        (ExperimentColumn.RECORDIST, "rec"),
    ):
        values: list[object] = [
            cast(dict[str, object], item).get(source_key, "")
            for item in completed[ExperimentColumn.SOURCE]
        ]
        completed[field] = Series(values, index=completed.index)
    return AnalysisMeasurements(
        completed=completed,
        supported=completed[
            completed[ResultColumn.TARGET_SPECIES].notna()
        ].copy(),
        noise=noise,
        trials=trials,
        thresholds=thresholds,
        overlap=overlap,
        baseline=noise[
            noise[ResultColumn.CONDITION] == NoiseCondition.ORIGINAL
        ].set_index(ResultColumn.RECORDING_ID),
        configuration=configuration,
        failed_recordings=int(
            (
                recordings[ExperimentColumn.STATUS] == ProcessingStatus.FAILED
            ).sum()
        ),
    )


def validate_measurements(data: AnalysisMeasurements) -> None:
    """Reject inconsistent repetitions, averages or dependent pair membership.

    :param data: Detached source tables from one checkpoint transaction.
    :type data: AnalysisMeasurements
    :return: None.
    :rtype: None
    """
    if (
        data.noise.groupby(ResultColumn.CONDITION).size().nunique() != 1
        or data.noise.duplicated(
            [ResultColumn.RECORDING_ID, ResultColumn.CONDITION]
        ).any()
    ):
        raise ValueError(AnalysisError.INVALID_NOISE_COVERAGE)
    repetitions: Series[int] = data.trials.groupby(
        [ResultColumn.RECORDING_ID, ResultColumn.CONDITION]
    ).size()
    expected: Series[int] = (
        data.trials.groupby(
            [ResultColumn.RECORDING_ID, ResultColumn.CONDITION]
        )[ResultColumn.CONDITION]
        .first()
        .map(
            lambda label: (
                1
                if label == NoiseCondition.ORIGINAL
                else ExperimentSettings().noise_repetitions
            )
        )
    )
    if not repetitions.eq(expected).all():
        raise ValueError(AnalysisError.INVALID_REPETITIONS)
    if len(data.baseline) + data.supported[
        ExperimentColumn.NOISE_SKIP_REASON
    ].notna().sum() != len(data.supported):
        raise ValueError(AnalysisError.INVALID_EXCLUSIONS)
    if not data.trials[PredictionColumn.CONFIDENCE].between(0, 1).all():
        raise ValueError(AnalysisError.INVALID_CONFIDENCE)
    for condition, group in data.noise.groupby(
        ResultColumn.CONDITION, sort=False
    ):
        measured: Series[float] = (
            data.trials[data.trials[ResultColumn.CONDITION] == condition]
            .groupby(ResultColumn.RECORDING_ID)[PredictionColumn.CONFIDENCE]
            .mean()
        )
        if not allclose(
            group.set_index(ResultColumn.RECORDING_ID)[
                ResultColumn.CONFIDENCE_MEAN
            ].sort_index(),
            measured.sort_index(),
            rtol=AnalysisSetting.CONFIDENCE_RTOL,
            atol=AnalysisSetting.CONFIDENCE_ATOL,
        ):
            raise ValueError(AnalysisError.INCONSISTENT_MEANS)
    members: DataFrame = data.overlap[
        data.overlap[ResultColumn.MIX]
        == ExperimentSettings().mix_ratios[0].label
    ]
    if (
        data.overlap.groupby(ResultColumn.MIX).size().nunique() != 1
        or concat(
            [
                members[ExperimentColumn.RECORDING_A],
                members[ExperimentColumn.RECORDING_B],
            ]
        )
        .duplicated()
        .any()
    ):
        raise ValueError(AnalysisError.INVALID_PAIRS)


def noise_intervals(data: AnalysisMeasurements) -> DataFrame:
    """Bootstrap file-level, species-level and paired noise effects.

    :param data: Validated repeated trials and recording-level means.
    :type data: AnalysisMeasurements
    :return: One measured interval row per original or noisy condition.
    :rtype: DataFrame
    """
    rows: list[dict[str, object]] = []
    for condition in (
        NoiseCondition.ORIGINAL,
        *(f"{snr:g} dB" for snr in ExperimentSettings().snr_values_db),
    ):
        group: DataFrame = data.noise[
            data.noise[ResultColumn.CONDITION] == condition
        ].set_index(ResultColumn.RECORDING_ID)
        differences: Series[float] = (
            group[ResultColumn.CONFIDENCE_MEAN]
            - data.baseline[ResultColumn.CONFIDENCE_MEAN]
        )
        species: Series[str] = data.trials[
            data.trials[ResultColumn.CONDITION] == NoiseCondition.ORIGINAL
        ].set_index(ResultColumn.RECORDING_ID)[ResultColumn.TARGET_SPECIES]
        grouped: DataFrame = group.assign(species=species)
        cluster: StatisticalInterval = bootstrap_mean(
            differences.to_numpy(dtype=float64),
            groups=species.reindex(differences.index).to_numpy(),
        )
        rows.append(
            {
                ResultColumn.CONDITION: condition,
                ResultColumn.RECORDINGS: len(group),
                ExperimentColumn.SPECIES: grouped.groupby(
                    ExperimentColumn.SPECIES
                ).ngroups,
                **interval_row(
                    group[ResultColumn.CONFIDENCE_MEAN].to_numpy(
                        dtype=float64
                    ),
                    PredictionColumn.CONFIDENCE,
                ),
                **interval_row(
                    group[ResultColumn.DETECTION_RATE].to_numpy(dtype=float64),
                    ResultColumn.DETECTION_RATE,
                ),
                **interval_row(
                    differences.to_numpy(dtype=float64),
                    ExperimentColumn.PAIRED_DELTA,
                ),
                **interval_row(
                    grouped.groupby(ExperimentColumn.SPECIES)[
                        ResultColumn.CONFIDENCE_MEAN
                    ]
                    .mean()
                    .to_numpy(dtype=float64),
                    "macro_confidence",
                ),
                **interval_row(
                    grouped.groupby(ExperimentColumn.SPECIES)[
                        ResultColumn.DETECTION_RATE
                    ]
                    .mean()
                    .to_numpy(dtype=float64),
                    "macro_detection_rate",
                ),
                "species_cluster_delta": cluster[0],
                "species_cluster_delta_low": cluster[1],
                "species_cluster_delta_high": cluster[2],
            }
        )
    return DataFrame(rows)


def threshold_intervals(data: AnalysisMeasurements) -> DataFrame:
    """Estimate target recovery with equal file or equal species weights.

    :param data: Validated baseline target detections and supported species.
    :type data: AnalysisMeasurements
    :return: One measured recovery interval row per predefined threshold.
    :rtype: DataFrame
    """
    rows: list[dict[str, object]] = []
    for threshold, group in data.thresholds.groupby(ResultColumn.THRESHOLD):
        group = group.merge(
            data.supported[
                [ResultColumn.RECORDING_ID, ResultColumn.TARGET_SPECIES]
            ],
            on=ResultColumn.RECORDING_ID,
        )
        species: Series[float] = group.groupby(ResultColumn.TARGET_SPECIES)[
            ExperimentColumn.TARGET_DETECTED
        ].mean()
        rows.append(
            {
                ResultColumn.THRESHOLD: threshold,
                ResultColumn.RECORDINGS: len(group),
                **interval_row(
                    group[ExperimentColumn.TARGET_DETECTED].to_numpy(
                        dtype=float64
                    ),
                    "recovery",
                ),
                **interval_row(
                    species.to_numpy(dtype=float64), "macro_recovery"
                ),
            }
        )
    return DataFrame(rows)


def overlap_intervals(data: AnalysisMeasurements) -> DataFrame:
    """Estimate joint recovery and confidence differences for disjoint pairs.

    :param data: Validated mixture scores and original target confidences.
    :type data: AnalysisMeasurements
    :return: Recovery and paired confidence intervals for each mixture.
    :rtype: DataFrame
    """
    rows: list[dict[str, object]] = []
    original: Series[float] = data.baseline[ResultColumn.CONFIDENCE_MEAN]
    for mix, group in data.overlap.groupby(ResultColumn.MIX, sort=True):
        measurements: dict[str, Float64Array] = {
            ExperimentColumn.JOINT_RECOVERY: asarray(
                (
                    group[ResultColumn.CONFIDENCE_A]
                    >= ModelSettings().minimum_confidence
                )
                & (
                    group[ResultColumn.CONFIDENCE_B]
                    >= ModelSettings().minimum_confidence
                ),
                dtype=float64,
            ),
            ExperimentColumn.DELTA_A: (
                group[ResultColumn.CONFIDENCE_A]
                - group[ExperimentColumn.RECORDING_A].map(original)
            ).to_numpy(dtype=float64),
            ExperimentColumn.DELTA_B: (
                group[ResultColumn.CONFIDENCE_B]
                - group[ExperimentColumn.RECORDING_B].map(original)
            ).to_numpy(dtype=float64),
        }
        label: str
        values: Float64Array
        for label, values in measurements.items():
            interval: StatisticalInterval = bootstrap_mean(values)
            rows.append(
                {
                    ResultColumn.MIX: mix,
                    ExperimentColumn.QUANTITY: label,
                    ExperimentColumn.ESTIMATE: interval[0],
                    ExperimentColumn.LOW: interval[1],
                    ExperimentColumn.HIGH: interval[2],
                    ResultColumn.PAIRS: len(group),
                }
            )
    return DataFrame(rows)


def group_intervals(data: AnalysisMeasurements, field: str) -> DataFrame:
    """Describe sufficiently represented metadata groups at the export
    threshold.

    :param data: Validated target detections and source metadata.
    :type data: AnalysisMeasurements
    :param field: Country or Xeno-canto quality column.
    :type field: str
    :return: Descriptive recovery intervals ordered by recording count.
    :rtype: DataFrame
    """
    rows: list[dict[str, object]] = []
    merged: DataFrame = data.supported.merge(
        data.thresholds[
            data.thresholds[ResultColumn.THRESHOLD]
            == ModelSettings().minimum_confidence
        ],
        on=ResultColumn.RECORDING_ID,
    )
    category: Hashable
    group: DataFrame
    for category, group in merged.groupby(field):
        if len(group) < AnalysisSetting.MINIMUM_GROUP_RECORDINGS:
            continue
        rows.append(
            {
                field: str(category),
                ResultColumn.RECORDINGS: len(group),
                ExperimentColumn.SPECIES: group[
                    ResultColumn.TARGET_SPECIES
                ].nunique(),
                **interval_row(
                    group[ExperimentColumn.TARGET_DETECTED].to_numpy(
                        dtype=float64
                    ),
                    "recovery",
                ),
            }
        )
    return DataFrame(rows).sort_values(
        ResultColumn.RECORDINGS, ascending=False
    )


def write_provenance(
    data: AnalysisMeasurements, noise: DataFrame, paths: ProjectPaths
) -> None:
    """Record validated measurement counts and the report's numerical macros.

    :param data: Validated checkpoint measurements and scientific settings.
    :type data: AnalysisMeasurements
    :param noise: Exported noise intervals with species counts.
    :type noise: DataFrame
    :param paths: Destinations of measured results and LaTeX report fragments.
    :type paths: ProjectPaths
    :return: None.
    :rtype: None
    """
    completed: DataFrame = data.completed
    supported: DataFrame = data.supported
    exclusions: Series[int] = completed.groupby(
        ExperimentColumn.NOISE_SKIP_REASON, dropna=False
    ).size()
    provenance: dict[str, object] = {
        ExperimentColumn.ANALYSIS_AT: datetime.now(UTC).isoformat(),
        MetadataKey.CONFIGURATION: data.configuration,
        CoverageKey.PROCESSED_RECORDINGS: len(completed),
        CoverageKey.FAILED_RECORDINGS: data.failed_recordings,
        ExperimentColumn.SUPPORTED_RECORDINGS: len(supported),
        ExperimentColumn.SUPPORTED_SPECIES: supported[
            ResultColumn.TARGET_SPECIES
        ].nunique(),
        ExperimentColumn.NOISE_ELIGIBLE: len(data.baseline),
        ExperimentColumn.NOISE_SPECIES: int(
            noise[ExperimentColumn.SPECIES].iloc[0]
        ),
        ExperimentColumn.UNDETECTED_SUPPORTED: int(
            supported[ExperimentColumn.NOISE_SKIP_REASON].notna().sum()
        ),
        ExperimentColumn.UNSUPPORTED_OR_UNIDENTIFIED: int(
            completed[ResultColumn.TARGET_SPECIES].isna().sum()
        ),
        ExperimentColumn.COUNTRIES: completed[
            ExperimentColumn.COUNTRY
        ].nunique(),
        ExperimentColumn.RECORDISTS: completed[
            ExperimentColumn.RECORDIST
        ].nunique(),
        ExperimentColumn.AUDIO_HOURS: float(
            completed[DetailColumn.DURATION_SECONDS].sum() / 3600
        ),
        ExperimentColumn.SPECIES_SINGLETON: int(
            (supported.groupby(ResultColumn.TARGET_SPECIES).size() == 1).sum()
        ),
        ExperimentColumn.SPECIES_LT5: int(
            (supported.groupby(ResultColumn.TARGET_SPECIES).size() < 5).sum()
        ),
        ExperimentColumn.OVERLAP_PAIRS: len(
            data.overlap[
                data.overlap[ResultColumn.MIX]
                == ExperimentSettings().mix_ratios[0].label
            ]
        ),
        ExperimentColumn.BOOTSTRAP_REPETITIONS: StatisticalSetting.REPETITIONS,
        ExperimentColumn.BOOTSTRAP_SEED: StatisticalSetting.RANDOM_SEED,
        ExperimentColumn.INTEGRITY_CHECKS: AnalysisSetting.PASSED,
        ExperimentColumn.EXCLUSIONS: {
            str(k): int(v) for k, v in exclusions.items()
        },
    }
    with atomic_output(
        paths.results / ExperimentFile.CORPUS_PROVENANCE_JSON
    ) as temporary:
        temporary.write_text(
            dumps(provenance, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    macros: dict[str, object] = {
        "CorpusRecordings": len(completed),
        "CorpusSupported": len(supported),
        "CorpusSpecies": provenance[ExperimentColumn.SUPPORTED_SPECIES],
        "NoiseRecordings": len(data.baseline),
        "CorpusFailed": provenance[CoverageKey.FAILED_RECORDINGS],
        "CorpusUndetected": provenance[ExperimentColumn.UNDETECTED_SUPPORTED],
        "CorpusUnsupported": provenance[
            ExperimentColumn.UNSUPPORTED_OR_UNIDENTIFIED
        ],
        "CorpusCountries": provenance[ExperimentColumn.COUNTRIES],
        "OverlapPairs": provenance[ExperimentColumn.OVERLAP_PAIRS],
        "CorpusHours": f"{provenance[ExperimentColumn.AUDIO_HOURS]:.1f}",
        "SpeciesSingleton": provenance[ExperimentColumn.SPECIES_SINGLETON],
    }
    with atomic_output(
        paths.report_tables / ExperimentFile.NUMBERS_TEX
    ) as temporary:
        temporary.write_text(
            "\n".join(
                (
                    f"\\newcommand{{\\{key}}}{{{value}}}"
                    for key, value in macros.items()
                )
            )
            + "\n",
            encoding="utf-8",
        )


def plot_noise_intervals(results: DataFrame, output: Path) -> None:
    """Render recording-level confidence and paired-effect intervals.

    :param results: Noise estimates and bootstrap confidence bounds.
    :type results: DataFrame
    :param output: Destination directory for matching PDF and PNG figures.
    :type output: Path
    :return: None.
    :rtype: None
    """
    figure: Figure = ExperimentPlotter.create_figure((10, 3.5))
    target: PlotFigure = cast(PlotFigure, figure)
    target.set_layout_engine("constrained")
    axes: tuple[PlotAxis, PlotAxis] = (
        target.add_subplot(1, 2, 1),
        target.add_subplot(1, 2, 2),
    )
    plotted: DataFrame = results.iloc[1:]
    column: str
    label: str
    axis: PlotAxis
    try:
        for column, label, axis in zip(
            (PredictionColumn.CONFIDENCE, ExperimentColumn.PAIRED_DELTA),
            (
                ExperimentLabel.MEAN_CONFIDENCE,
                ExperimentLabel.ORIGINAL_DIFFERENCE,
            ),
            axes,
            strict=True,
        ):
            axis.errorbar(
                ExperimentSettings().snr_values_db,
                plotted[column],
                yerr=[
                    plotted[column] - plotted[column + TableFormat.LOW_SUFFIX],
                    plotted[column + TableFormat.HIGH_SUFFIX]
                    - plotted[column],
                ],
                marker="o",
                capsize=4,
            )
            axis.set_xlabel(ExperimentLabel.ADDED_SNR)
            axis.set_ylabel(label)
            axis.set_xticks(ExperimentSettings().snr_values_db)
            axis.invert_xaxis()
            axis.grid(alpha=0.2)
        ExperimentPlotter.save_figure_formats(
            figure,
            (
                output / ExperimentFile.NOISE_INTERVALS_PDF,
                output / ExperimentFile.NOISE_INTERVALS_PNG,
            ),
        )
    finally:
        close(figure)


def run_analysis(paths: ProjectPaths) -> None:
    """Validate one saved snapshot and export the experiment's uncertainty.

    :param paths: Existing checkpoint and experiment output directories.
    :type paths: ProjectPaths
    :return: None.
    :rtype: None
    """
    output: Path = paths.results
    output.mkdir(parents=True, exist_ok=True)
    data: AnalysisMeasurements = read_measurements(paths)
    validate_measurements(data)
    noise: DataFrame = noise_intervals(data)
    write_table(noise, output / ExperimentFile.NOISE_INTERVALS_CSV)
    tex_table(
        DataFrame(
            {
                ExperimentLabel.CONDITION: noise[ResultColumn.CONDITION],
                ExperimentLabel.CONFIDENCE_INTERVAL: interval_cells(
                    noise, PredictionColumn.CONFIDENCE
                ),
                ExperimentLabel.TARGET_RECOVERED: noise[
                    ResultColumn.DETECTION_RATE
                ].map(lambda x: f"{x:.1%}"),
                ExperimentLabel.PAIRED_DIFFERENCE: noise[
                    ExperimentColumn.PAIRED_DELTA
                ].map(lambda x: f"{x:+.3f}"),
            }
        ),
        paths.report_tables / ExperimentFile.NOISE_TABLE_TEX,
    )
    write_table(
        threshold_intervals(data),
        output / ExperimentFile.THRESHOLD_INTERVALS_CSV,
    )
    write_table(
        overlap_intervals(data), output / ExperimentFile.OVERLAP_INTERVALS_CSV
    )
    field: str
    destination: ExperimentFile
    for field, destination in (
        (ExperimentColumn.COUNTRY, ExperimentFile.COUNTRY_RECOVERY_CSV),
        (ExperimentColumn.QUALITY, ExperimentFile.QUALITY_RECOVERY_CSV),
    ):
        write_table(group_intervals(data, field), output / destination)
    write_provenance(data, noise, paths)
    plot_noise_intervals(noise, output)
