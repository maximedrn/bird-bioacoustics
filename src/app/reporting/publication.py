"""Generate LaTeX tables directly from the experiment measurements."""

from __future__ import annotations

from base64 import b64encode
from collections.abc import Hashable
from html import escape
from pathlib import Path
from typing import Final

from nbformat import NotebookNode
from nbformat.v4 import new_output
from pandas import DataFrame, Series, read_csv

from app.domain.constants import (
    Artifact,
    MetadataKey,
    PredictionColumn,
    ResultColumn,
)
from app.domain.settings import ProjectPaths
from app.domain.types import BooleanArray, Float64Array
from app.experiments.constants import (
    EvaluationScope,
    ExperimentColumn,
    MetricColumn,
    SoundscapeDataset,
    SoundscapeFile,
    SoundscapeSetting,
    SoundscapeVersion,
    ValidationSetting,
)
from app.experiments.statistics import bootstrap_f1_difference
from app.reporting.analysis import bounds_cells, interval_cells, tex_table
from app.reporting.constants import (
    EXPERIMENT_FIGURES,
    EXPERIMENT_TABLES,
    DetailColumn,
    ExperimentFile,
    ExperimentSection,
    NotebookContent,
    NotebookKey,
)
from app.reporting.controlled import (
    EXTENSION_OBSERVATIONS,
    extension_frame,
    paired_validation_frame,
)
from app.reporting.messages import (
    ExperimentLabel,
    ExperimentObservation,
    ReportMessage,
)
from app.reporting.writing import write_table

EXPERIMENT_SECTIONS: Final[frozenset[str]] = frozenset(
    section.value for section in ExperimentSection
)


def table_output(title: str, frame: DataFrame) -> NotebookNode:
    """Publish every row without the pandas display truncation settings.

    :param title: Visible heading of the complete table.
    :type title: str
    :param frame: Complete table of measured or formatted values.
    :type frame: DataFrame
    :return: Notebook display output with a plain-text fallback.
    :rtype: NotebookNode
    """
    html: str = frame.to_html(index=False, border=0, escape=True)
    html = html.replace(
        "<thead>", f"<caption>{escape(title)}</caption><thead>", 1
    )
    return new_output(
        NotebookContent.DISPLAY_DATA,
        data={
            NotebookContent.HTML: html,
            NotebookContent.TEXT: title + "\n" + frame.to_string(index=False),
        },
    )


def observation_output(text: str) -> NotebookNode:
    """Give every result cell a brief visible observation.

    :param text: Simple observation associated with the result cell.
    :type text: str
    :return: Notebook display output with a plain-text fallback.
    :rtype: NotebookNode
    """
    text = ExperimentObservation.PREFIX + text
    return new_output(
        NotebookContent.DISPLAY_DATA,
        data={NotebookContent.MARKDOWN: text, NotebookContent.TEXT: text},
    )


def experiment_outputs(section: str, root: Path) -> list[NotebookNode]:
    """Present the experiment as narrow tables and complete figures.

    :param section: Stable notebook result-section identifier.
    :type section: str
    :param root: Directory containing the saved experiment artifacts.
    :type root: Path
    :return: Complete notebook outputs and their simple observation.
    :rtype: list[NotebookNode]
    """
    if section in EXPERIMENT_FIGURES:
        relative: Path = EXPERIMENT_FIGURES[section]
        path: Path = root / relative
        if not path.exists():
            return [observation_output(ExperimentObservation.MISSING)]
        figure: NotebookNode = new_output(
            NotebookContent.DISPLAY_DATA,
            data={NotebookContent.PNG: b64encode(path.read_bytes()).decode()},
        )
        figure[NotebookKey.METADATA] = {
            NotebookKey.FILENAME: str(relative),
            NotebookContent.PNG: {
                NotebookKey.ALTERNATIVE_TEXT: ReportMessage.EXPERIMENT_ALTERNATIVE_TEXT.format(
                    section=section
                )
            },
        }
        text: str = EXTENSION_OBSERVATIONS.get(section) or (
            ExperimentObservation.NOISE_FIGURE
            if section == ExperimentSection.NOISE_INTERVAL_FIGURE
            else ExperimentObservation.COMPARISON
        )
        return [figure, observation_output(text)]
    if section not in EXPERIMENT_TABLES:
        raise ValueError(
            ReportMessage.UNKNOWN_EXPERIMENT.format(section=section)
        )
    path = root / EXPERIMENT_TABLES[section]
    if not path.exists():
        return [observation_output(ExperimentObservation.MISSING)]
    frame: DataFrame = read_csv(path, dtype={MetadataKey.MODEL: str})
    if section in EXTENSION_OBSERVATIONS:
        additional: list[NotebookNode] = [
            table_output(
                section.replace("_", " ").capitalize(),
                extension_frame(section, frame),
            ),
        ]
        if section == ExperimentSection.SOURCE_VALIDATION:
            additional.append(
                table_output(
                    "Paired F1 after separate calibration",
                    paired_validation_frame(
                        read_csv(
                            root
                            / ValidationSetting.DIRECTORY
                            / ValidationSetting.PAIRED
                        )
                    ),
                )
            )
        additional.append(observation_output(EXTENSION_OBSERVATIONS[section]))
        return additional
    if section == ExperimentSection.NOISE_INTERVALS:
        return noise_outputs(frame)
    if section == ExperimentSection.THRESHOLD_INTERVALS:
        return threshold_outputs(frame)
    if section in {
        ExperimentSection.SOUNDSCAPE_METRICS,
        ExperimentSection.COMMON_SPECIES_METRICS,
    }:
        return soundscape_outputs(frame, section)
    if section == ExperimentSection.PAIRED_MODELS:
        return paired_outputs(frame)
    return group_outputs(frame, section)


def noise_outputs(frame: DataFrame) -> list[NotebookNode]:
    """Present complete noise measurements and their observation.

    :param frame: Complete table of measured or formatted values.
    :type frame: DataFrame
    :return: Complete notebook outputs and their simple observation.
    :rtype: list[NotebookNode]
    """
    outputs: list[NotebookNode] = []
    outputs.append(
        table_output(
            ExperimentLabel.NOISE_FILE_TITLE,
            DataFrame(
                {
                    ExperimentLabel.CONDITION: frame[ResultColumn.CONDITION],
                    ExperimentLabel.RECORDINGS: frame[ResultColumn.RECORDINGS],
                    ExperimentLabel.SCORE_INTERVAL: interval_cells(
                        frame, PredictionColumn.CONFIDENCE
                    ),
                    ExperimentLabel.RECOVERY_INTERVAL: interval_cells(
                        frame, ResultColumn.DETECTION_RATE
                    ),
                    ExperimentLabel.PAIRED_INTERVAL: interval_cells(
                        frame, ExperimentColumn.PAIRED_DELTA
                    ),
                }
            ),
        )
    )
    outputs.append(
        table_output(
            ExperimentLabel.NOISE_SPECIES_TITLE,
            DataFrame(
                {
                    ExperimentLabel.CONDITION: frame[ResultColumn.CONDITION],
                    ExperimentLabel.SPECIES: frame[ExperimentColumn.SPECIES],
                    ExperimentLabel.SPECIES_SCORE_INTERVAL: interval_cells(
                        frame, "macro_confidence"
                    ),
                    ExperimentLabel.SPECIES_RECOVERY_INTERVAL: interval_cells(
                        frame, "macro_detection_rate"
                    ),
                    ExperimentLabel.SPECIES_CLUSTER_INTERVAL: interval_cells(
                        frame, "species_cluster_delta"
                    ),
                }
            ),
        )
    )
    text: str = ExperimentObservation.NOISE_INTERVALS
    outputs.append(observation_output(text))
    return outputs


def threshold_outputs(frame: DataFrame) -> list[NotebookNode]:
    """Present complete threshold measurements and their observation.

    :param frame: Complete table of measured or formatted values.
    :type frame: DataFrame
    :return: Complete notebook outputs and their simple observation.
    :rtype: list[NotebookNode]
    """
    outputs: list[NotebookNode] = []
    outputs.append(
        table_output(
            ExperimentLabel.THRESHOLD_TITLE,
            DataFrame(
                {
                    ExperimentLabel.THRESHOLD: frame[
                        ResultColumn.THRESHOLD
                    ].map(lambda x: f"{x:.2f}"),
                    ExperimentLabel.RECORDINGS: frame[ResultColumn.RECORDINGS],
                    ExperimentLabel.FILE_INTERVAL: interval_cells(
                        frame, "recovery"
                    ),
                    ExperimentLabel.SPECIES_INTERVAL: interval_cells(
                        frame, "macro_recovery"
                    ),
                }
            ),
        )
    )
    text: str = ExperimentObservation.THRESHOLDS
    outputs.append(observation_output(text))
    return outputs


def soundscape_outputs(frame: DataFrame, section: str) -> list[NotebookNode]:
    """Present complete soundscape measurements and their observation.

    :param frame: Complete table of measured or formatted values.
    :type frame: DataFrame
    :param section: Stable notebook result-section identifier.
    :type section: str
    :return: Complete notebook outputs and their simple observation.
    :rtype: list[NotebookNode]
    """
    outputs: list[NotebookNode] = []
    scope: str = (
        EvaluationScope.COMMON_CLASSES
        if section == ExperimentSection.COMMON_SPECIES_METRICS
        else EvaluationScope.ALL_REFERENCE_CLASSES
    )
    frame = frame[frame[ExperimentColumn.SCOPE] == scope]
    dataset: Hashable
    group: DataFrame
    for dataset, group in frame.groupby(ExperimentColumn.DATASET, sort=False):
        outputs.append(
            table_output(
                ExperimentLabel.METRICS_TITLE.format(
                    dataset=str(dataset), scope=scope
                ),
                DataFrame(
                    {
                        ExperimentLabel.VERSION: group[MetadataKey.MODEL],
                        ExperimentLabel.THRESHOLD: group[
                            ResultColumn.THRESHOLD
                        ].map(lambda x: f"{x:.2f}"),
                        ExperimentLabel.PRECISION: group[
                            MetricColumn.PRECISION_MICRO
                        ].map(lambda x: f"{x:.3f}"),
                        ExperimentLabel.RECALL: group[
                            MetricColumn.RECALL_MICRO
                        ].map(lambda x: f"{x:.3f}"),
                        ExperimentLabel.F1_INTERVAL: interval_cells(
                            group, MetricColumn.F1_MICRO
                        ),
                        ExperimentLabel.F1_MACRO: group[
                            MetricColumn.F1_MACRO
                        ].map(lambda x: f"{x:.3f}"),
                    }
                ),
            )
        )
        outputs.append(
            table_output(
                ExperimentLabel.COUNTS_TITLE.format(dataset=str(dataset)),
                DataFrame(
                    {
                        ExperimentLabel.VERSION: group[MetadataKey.MODEL],
                        ExperimentLabel.THRESHOLD: group[
                            ResultColumn.THRESHOLD
                        ].map(lambda x: f"{x:.2f}"),
                        ExperimentLabel.TP: group[MetricColumn.TP],
                        ExperimentLabel.FP: group[MetricColumn.FP],
                        ExperimentLabel.FN: group[MetricColumn.FN],
                        ExperimentLabel.NEGATIVE_FALSE_ALARMS: group[
                            MetricColumn.NEGATIVE_CLIP_FALSE_ALARM_RATE
                        ].map(lambda x: f"{x:.1%}"),
                    }
                ),
            )
        )
        context: Series[str | float | int] = group.iloc[0]
        outputs.append(
            table_output(
                ExperimentLabel.CONTEXT_TITLE.format(dataset=str(dataset)),
                DataFrame(
                    {
                        ExperimentLabel.CLIPS: [
                            int(context[ExperimentColumn.CLIPS])
                        ],
                        ExperimentLabel.SOURCES: [
                            int(context[ExperimentColumn.SOURCE_RECORDINGS])
                        ],
                        ExperimentLabel.POSITIVE_CLASSES: [
                            int(context[MetricColumn.SUPPORTED_CLASSES])
                        ],
                        ExperimentLabel.NEGATIVE_CLIPS: [
                            int(context[MetricColumn.NEGATIVE_CLIPS])
                        ],
                    }
                ),
            )
        )
    text: str = (
        ExperimentObservation.COMMON_SPECIES
        if section == ExperimentSection.COMMON_SPECIES_METRICS
        else ExperimentObservation.SOUNDSCAPES
    )
    outputs.append(observation_output(text))
    return outputs


def paired_outputs(frame: DataFrame) -> list[NotebookNode]:
    """Present complete paired measurements and their observation.

    :param frame: Complete table of measured or formatted values.
    :type frame: DataFrame
    :return: Complete notebook outputs and their simple observation.
    :rtype: list[NotebookNode]
    """
    outputs: list[NotebookNode] = []
    outputs.append(
        table_output(
            ExperimentLabel.PAIRED_TITLE,
            DataFrame(
                {
                    ExperimentLabel.DATASET: frame[ExperimentColumn.DATASET],
                    ExperimentLabel.THRESHOLD: frame[
                        ResultColumn.THRESHOLD
                    ].map(lambda x: f"{x:.2f}"),
                    ExperimentLabel.F1_DIFFERENCE: frame[
                        MetricColumn.F1_DELTA_3_MINUS_2
                    ].map(lambda x: f"{x:+.3f}"),
                    ExperimentLabel.INTERVAL: [
                        f"[{float(lo):+.3f}; {float(hi):+.3f}]"
                        for lo, hi in zip(
                            frame[ExperimentColumn.LOW],
                            frame[ExperimentColumn.HIGH],
                            strict=True,
                        )
                    ],
                }
            ),
        )
    )
    text: str = ExperimentObservation.PAIRED
    outputs.append(observation_output(text))
    return outputs


def group_outputs(frame: DataFrame, section: str) -> list[NotebookNode]:
    """Present complete group measurements and their observation.

    :param frame: Complete table of measured or formatted values.
    :type frame: DataFrame
    :param section: Stable notebook result-section identifier.
    :type section: str
    :return: Complete notebook outputs and their simple observation.
    :rtype: list[NotebookNode]
    """
    outputs: list[NotebookNode] = []
    field: str = (
        ExperimentColumn.COUNTRY
        if section == ExperimentSection.COUNTRIES
        else ExperimentColumn.QUALITY
    )
    label: str = (
        ExperimentLabel.COUNTRY
        if section == ExperimentSection.COUNTRIES
        else ExperimentLabel.QUALITY_XENO_CANTO
    )
    outputs.append(
        table_output(
            ExperimentLabel.GROUP_TITLE,
            DataFrame(
                {
                    label: frame[field],
                    ExperimentLabel.RECORDINGS: frame[ResultColumn.RECORDINGS],
                    ExperimentLabel.SPECIES: frame[ExperimentColumn.SPECIES],
                    ExperimentLabel.RECOVERY_INTERVAL: interval_cells(
                        frame, "recovery"
                    ),
                }
            ),
        )
    )
    text: str = (
        ExperimentObservation.COUNTRIES
        if section == ExperimentSection.COUNTRIES
        else ExperimentObservation.QUALITY
    )
    outputs.append(observation_output(text))
    return outputs


def prepare_recovery_tables(paths: ProjectPaths) -> None:
    """Generate recovery tables from the saved experiment.

    :param paths: Destinations of measured results and LaTeX fragments.
    :type paths: ProjectPaths
    :return: None.
    :rtype: None
    """
    root: Path = paths.results
    thresholds: DataFrame = read_csv(
        root / ExperimentFile.THRESHOLD_INTERVALS_CSV
    )
    tex_table(
        DataFrame(
            {
                ExperimentLabel.THRESHOLD: thresholds[
                    ResultColumn.THRESHOLD
                ].map(lambda x: f"{x:.2f}"),
                ExperimentLabel.FILE: interval_cells(
                    thresholds, "recovery", percent=True
                ),
                ExperimentLabel.SPECIES_WEIGHTED: interval_cells(
                    thresholds, "macro_recovery", percent=True
                ),
            }
        ),
        paths.report_tables / ExperimentFile.THRESHOLD_TABLE_TEX,
    )
    overlaps: DataFrame = read_csv(root / ExperimentFile.OVERLAP_INTERVALS_CSV)
    both: DataFrame = overlaps[
        overlaps[ExperimentColumn.QUANTITY] == ExperimentColumn.JOINT_RECOVERY
    ]
    tex_table(
        DataFrame(
            {
                ExperimentLabel.MIX: both[ResultColumn.MIX],
                ExperimentLabel.BOTH_INTERVAL: bounds_cells(
                    both,
                    ExperimentColumn.ESTIMATE,
                    ExperimentColumn.LOW,
                    ExperimentColumn.HIGH,
                    percent=True,
                ),
            }
        ),
        paths.report_tables / ExperimentFile.OVERLAP_TABLE_TEX,
    )
    noise: DataFrame = read_csv(root / ExperimentFile.NOISE_INTERVALS_CSV)
    tex_table(
        DataFrame(
            {
                ExperimentLabel.CONDITION: noise[ResultColumn.CONDITION],
                ExperimentLabel.RECOVERY_INTERVAL: interval_cells(
                    noise, ResultColumn.DETECTION_RATE, percent=True
                ),
                ExperimentLabel.PAIRED_INTERVAL: interval_cells(
                    noise, ExperimentColumn.PAIRED_DELTA
                ),
                ExperimentLabel.CLUSTERS_INTERVAL: interval_cells(
                    noise, "species_cluster_delta"
                ),
            }
        ),
        paths.report_tables / ExperimentFile.NOISE_UNCERTAINTY_TABLE_TEX,
    )
    tex_table(
        DataFrame(
            {
                ExperimentLabel.CONDITION: noise[ResultColumn.CONDITION],
                ExperimentLabel.SPECIES_SCORE_INTERVAL: interval_cells(
                    noise, "macro_confidence"
                ),
                ExperimentLabel.SPECIES_RECOVERY_INTERVAL: interval_cells(
                    noise, "macro_detection_rate", percent=True
                ),
            }
        ),
        paths.report_tables / ExperimentFile.NOISE_MACRO_TABLE_TEX,
    )


def prepare_corpus_tables(paths: ProjectPaths) -> None:
    """Generate corpus tables from the saved experiment.

    :param paths: Destinations of measured results and LaTeX fragments.
    :type paths: ProjectPaths
    :return: None.
    :rtype: None
    """
    root: Path = paths.results
    field: str
    measurements: ExperimentFile
    table: ExperimentFile
    for field, measurements, table in (
        (
            ExperimentColumn.COUNTRY,
            ExperimentFile.COUNTRY_RECOVERY_CSV,
            ExperimentFile.COUNTRY_TABLE_TEX,
        ),
        (
            ExperimentColumn.QUALITY,
            ExperimentFile.QUALITY_RECOVERY_CSV,
            ExperimentFile.QUALITY_TABLE_TEX,
        ),
    ):
        frame: DataFrame = read_csv(root / measurements)
        tex_table(
            DataFrame(
                {
                    ExperimentLabel.COUNTRY
                    if field == ExperimentColumn.COUNTRY
                    else ExperimentLabel.QUALITY: frame[field],
                    ExperimentLabel.RECORDINGS: frame[ResultColumn.RECORDINGS],
                    ExperimentLabel.SPECIES: frame[ExperimentColumn.SPECIES],
                    ExperimentLabel.RECOVERY_95_INTERVAL: interval_cells(
                        frame, "recovery", percent=True
                    ),
                }
            ),
            paths.report_tables / table,
        )
    counts: DataFrame = read_csv(Artifact.THRESHOLD.path(root, "csv"))
    tex_table(
        DataFrame(
            {
                ExperimentLabel.THRESHOLD: counts[ResultColumn.THRESHOLD].map(
                    lambda x: f"{x:.2f}"
                ),
                ExperimentLabel.PREDICTIONS: counts[ResultColumn.DETECTIONS],
                ExperimentLabel.DISTINCT_LABELS: counts[
                    ResultColumn.DISTINCT_SPECIES
                ],
                ExperimentLabel.TARGET_RECOVERED: counts[
                    ResultColumn.TARGET_DETECTION_RATE
                ].map(lambda x: f"{x:.1%}"),
            }
        ),
        paths.report_tables / ExperimentFile.THRESHOLD_COUNTS_TABLE_TEX,
    )
    distribution: DataFrame = read_csv(
        Artifact.NOISE_DISTRIBUTION.path(root, "csv")
    )
    tex_table(
        DataFrame(
            {
                ExperimentLabel.CONDITION: distribution[
                    ResultColumn.CONDITION
                ],
                ExperimentLabel.MINIMUM: distribution[
                    DetailColumn.MINIMUM
                ].map(lambda x: f"{x:.3f}"),
                ExperimentLabel.Q1: distribution[DetailColumn.Q1].map(
                    lambda x: f"{x:.3f}"
                ),
                ExperimentLabel.MEDIAN: distribution[DetailColumn.MEDIAN].map(
                    lambda x: f"{x:.3f}"
                ),
                ExperimentLabel.Q3: distribution[DetailColumn.Q3].map(
                    lambda x: f"{x:.3f}"
                ),
                ExperimentLabel.MAXIMUM: distribution[
                    DetailColumn.MAXIMUM
                ].map(lambda x: f"{x:.3f}"),
            }
        ),
        paths.report_tables / ExperimentFile.NOISE_DISTRIBUTION_TABLE_TEX,
    )
    outcomes: DataFrame = read_csv(Artifact.OVERLAP_OUTCOMES.path(root, "csv"))
    tex_table(
        DataFrame(
            {
                ExperimentLabel.MIX: outcomes[ResultColumn.MIX],
                ExperimentLabel.BOTH: outcomes[DetailColumn.BOTH],
                ExperimentLabel.ONLY_A: outcomes[DetailColumn.ONLY_A],
                ExperimentLabel.ONLY_B: outcomes[DetailColumn.ONLY_B],
                ExperimentLabel.NEITHER: outcomes[DetailColumn.NEITHER],
            }
        ),
        paths.report_tables / ExperimentFile.OVERLAP_OUTCOMES_TABLE_TEX,
    )


def prepare_benchmark_tables(paths: ProjectPaths) -> None:
    """Generate benchmark tables from the saved experiment.

    :param paths: Destinations of measured results and LaTeX fragments.
    :type paths: ProjectPaths
    :return: None.
    :rtype: None
    """
    root: Path = paths.results
    metrics: DataFrame = read_csv(
        root / ExperimentFile.METRICS_CSV, dtype={MetadataKey.MODEL: str}
    )
    main_metrics: DataFrame = metrics[
        metrics[ExperimentColumn.SCOPE]
        == EvaluationScope.ALL_REFERENCE_CLASSES
    ]
    selected: DataFrame = main_metrics
    tex_table(
        DataFrame(
            {
                ExperimentLabel.DATASET: selected[ExperimentColumn.DATASET],
                ExperimentLabel.VERSION: selected[MetadataKey.MODEL],
                ExperimentLabel.THRESHOLD: selected[
                    ResultColumn.THRESHOLD
                ].map(lambda x: f"{x:.2f}"),
                ExperimentLabel.PRECISION: selected[
                    MetricColumn.PRECISION_MICRO
                ].map(lambda x: f"{x:.3f}"),
                ExperimentLabel.RECALL: selected[
                    MetricColumn.RECALL_MICRO
                ].map(lambda x: f"{x:.3f}"),
                ExperimentLabel.F1_95_INTERVAL: interval_cells(
                    selected, MetricColumn.F1_MICRO
                ),
                ExperimentLabel.F1_MACRO: selected[MetricColumn.F1_MACRO].map(
                    lambda x: f"{x:.3f}"
                ),
            }
        ),
        paths.report_tables / ExperimentFile.BENCHMARK_TABLE_TEX,
    )
    common: DataFrame = metrics[
        metrics[ExperimentColumn.SCOPE] == EvaluationScope.COMMON_CLASSES
    ]
    tex_table(
        DataFrame(
            {
                ExperimentLabel.VERSION: common[MetadataKey.MODEL],
                ExperimentLabel.THRESHOLD: common[ResultColumn.THRESHOLD].map(
                    lambda x: f"{x:.2f}"
                ),
                ExperimentLabel.PRECISION: common[
                    MetricColumn.PRECISION_MICRO
                ].map(lambda x: f"{x:.3f}"),
                ExperimentLabel.RECALL: common[MetricColumn.RECALL_MICRO].map(
                    lambda x: f"{x:.3f}"
                ),
                ExperimentLabel.F1_95_INTERVAL: interval_cells(
                    common, MetricColumn.F1_MICRO
                ),
                ExperimentLabel.F1_MACRO: common[MetricColumn.F1_MACRO].map(
                    lambda x: f"{x:.3f}"
                ),
            }
        ),
        paths.report_tables / ExperimentFile.COMMON_SPECIES_TABLE_TEX,
    )
    tex_table(
        DataFrame(
            {
                ExperimentLabel.VERSION: common[MetadataKey.MODEL],
                ExperimentLabel.THRESHOLD: common[ResultColumn.THRESHOLD].map(
                    lambda x: f"{x:.2f}"
                ),
                ExperimentLabel.TP: common[MetricColumn.TP],
                ExperimentLabel.FP: common[MetricColumn.FP],
                ExperimentLabel.FN: common[MetricColumn.FN],
                ExperimentLabel.COMMON_FALSE_ALARMS: common[
                    MetricColumn.NEGATIVE_CLIP_FALSE_ALARM_RATE
                ].map(lambda x: f"{x:.1%}"),
            }
        ),
        paths.report_tables / ExperimentFile.COMMON_SPECIES_COUNTS_TABLE_TEX,
    )
    negative: DataFrame = main_metrics[
        main_metrics[ExperimentColumn.DATASET] == SoundscapeDataset.HSN
    ]
    tex_table(
        DataFrame(
            {
                ExperimentLabel.VERSION: negative[MetadataKey.MODEL],
                ExperimentLabel.THRESHOLD: negative[
                    ResultColumn.THRESHOLD
                ].map(lambda x: f"{x:.2f}"),
                ExperimentLabel.NEGATIVE_CLIPS: negative[
                    MetricColumn.NEGATIVE_CLIPS
                ],
                ExperimentLabel.WITH_FALSE_ALARM: negative[
                    MetricColumn.NEGATIVE_CLIP_FALSE_ALARM_RATE
                ].map(lambda x: f"{x:.1%}"),
            }
        ),
        paths.report_tables / ExperimentFile.NEGATIVE_CLIPS_TABLE_TEX,
    )
    tex_table(
        DataFrame(
            {
                ExperimentLabel.DATASET: main_metrics[
                    ExperimentColumn.DATASET
                ],
                ExperimentLabel.VERSION: main_metrics[MetadataKey.MODEL],
                ExperimentLabel.THRESHOLD: main_metrics[
                    ResultColumn.THRESHOLD
                ].map(lambda x: f"{x:.2f}"),
                ExperimentLabel.TP: main_metrics[MetricColumn.TP],
                ExperimentLabel.FP: main_metrics[MetricColumn.FP],
                ExperimentLabel.FN: main_metrics[MetricColumn.FN],
                ExperimentLabel.F1_MICRO: main_metrics[
                    MetricColumn.F1_MICRO
                ].map(lambda x: f"{x:.3f}"),
            }
        ),
        paths.report_tables / ExperimentFile.ALL_THRESHOLDS_TABLE_TEX,
    )


def prepare_paired_tables(paths: ProjectPaths) -> None:
    """Generate paired tables from the saved experiment.

    :param paths: Destinations of measured results and LaTeX fragments.
    :type paths: ProjectPaths
    :return: None.
    :rtype: None
    """
    root: Path = paths.results
    differences: list[dict[str, str | float]] = []
    dataset: str
    for dataset in [SoundscapeDataset.NBP, SoundscapeDataset.HSN]:
        truth: BooleanArray = read_csv(
            root / SoundscapeFile.REFERENCE.format(dataset=dataset)
        ).to_numpy(dtype=bool)
        manifest: DataFrame = read_csv(
            root / SoundscapeFile.MANIFEST.format(dataset=dataset)
        )
        scores: dict[str, Float64Array] = {
            v: read_csv(
                root
                / SoundscapeFile.SCORES.format(dataset=dataset, version=v),
                index_col=0,
            ).to_numpy()
            for v in [SoundscapeVersion.V2_4, SoundscapeVersion.V3_0]
        }
        threshold: float
        for threshold in SoundscapeSetting.THRESHOLDS:
            delta: float
            low: float
            high: float
            delta, low, high = bootstrap_f1_difference(
                truth,
                scores[SoundscapeVersion.V3_0] >= threshold,
                scores[SoundscapeVersion.V2_4] >= threshold,
                manifest[ExperimentColumn.SOURCE_RECORDING].to_numpy(),
                seed=SoundscapeSetting.RANDOM_SEED,
            )
            differences.append(
                {
                    ExperimentColumn.DATASET: dataset,
                    ResultColumn.THRESHOLD: threshold,
                    MetricColumn.F1_DELTA_3_MINUS_2: delta,
                    ExperimentColumn.LOW: low,
                    ExperimentColumn.HIGH: high,
                }
            )
    difference_table: DataFrame = DataFrame(differences)
    write_table(
        difference_table,
        root / ExperimentFile.PAIRED_MODEL_DIFFERENCES_CSV,
    )
    tex_table(
        DataFrame(
            {
                ExperimentLabel.DATASET: difference_table[
                    ExperimentColumn.DATASET
                ],
                ExperimentLabel.THRESHOLD: difference_table[
                    ResultColumn.THRESHOLD
                ].map(lambda x: f"{x:.2f}"),
                ExperimentLabel.F1_DIFFERENCE_INTERVAL: bounds_cells(
                    difference_table,
                    MetricColumn.F1_DELTA_3_MINUS_2,
                    ExperimentColumn.LOW,
                    ExperimentColumn.HIGH,
                    signed=True,
                ),
            }
        ),
        paths.report_tables / ExperimentFile.PAIRED_DIFFERENCE_TABLE_TEX,
    )


def prepare_tables(paths: ProjectPaths) -> None:
    """Build all numerical report fragments from saved measured tables.

    :param paths: Project checkpoint and result directories.
    :type paths: ProjectPaths
    :return: None.
    :rtype: None
    """
    root: Path = paths.results
    prepare_recovery_tables(paths)
    prepare_corpus_tables(paths)
    if (root / ExperimentFile.METRICS_CSV).exists():
        prepare_benchmark_tables(paths)
        prepare_paired_tables(paths)
