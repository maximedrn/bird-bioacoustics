"""Choose thresholds on calibration sources and evaluate held-out sources."""

from __future__ import annotations

from datetime import UTC, datetime
from json import dumps
from pathlib import Path

from numpy import float64, isfinite
from numpy.random import Generator, default_rng
from pandas import DataFrame, concat, read_csv

from app.domain.constants import MetadataKey
from app.domain.settings import ProjectPaths
from app.domain.types import BenchmarkMetricRow, BooleanArray, Float64Array
from app.experiments.constants import (
    ExperimentColumn,
    MetricColumn,
    SoundscapeDataset,
    SoundscapeFile,
    SoundscapeSetting,
    SoundscapeVersion,
    ValidationSetting,
)
from app.experiments.messages import ValidationMessage
from app.experiments.soundscapes import digest, evaluate
from app.experiments.statistics import (
    bootstrap_f1_difference,
    multilabel_metrics,
)
from app.reporting.writing import atomic_output, write_table


def split_sources(manifest: DataFrame) -> DataFrame:
    """Reserve a quarter of sources without inspecting labels or predictions.

    :param manifest: Selected segments with their original recording source.
    :type manifest: DataFrame
    :return: All segments with deterministic calibration or test membership.
    :rtype: DataFrame
    """
    sources: list[str] = sorted(
        str(value)
        for value in manifest[ExperimentColumn.SOURCE_RECORDING].unique()
    )
    if len(sources) < 4:
        raise ValueError(ValidationMessage.INSUFFICIENT_SOURCES)
    generator: Generator = default_rng(SoundscapeSetting.RANDOM_SEED)
    count: int = max(
        1, round(len(sources) * ValidationSetting.CALIBRATION_FRACTION)
    )
    calibration: set[str] = set(
        generator.choice(sources, count, replace=False)
    )
    result: DataFrame = manifest.copy().sort_index()
    result[ValidationSetting.PARTITION] = result[
        ExperimentColumn.SOURCE_RECORDING
    ].map(
        lambda source: (
            ValidationSetting.CALIBRATION
            if source in calibration
            else ValidationSetting.TEST
        )
    )
    return result


def select_threshold(truth: BooleanArray, scores: Float64Array) -> float:
    """Maximise calibration micro F1, preferring higher thresholds on ties.

    :param truth: Calibration reference labels only.
    :type truth: BooleanArray
    :param scores: Corresponding complete model scores.
    :type scores: Float64Array
    :return: Decision threshold selected without access to held-out labels.
    :rtype: float
    """
    if (
        scores.shape != truth.shape
        or not isfinite(scores).all()
        or (scores < 0).any()
        or (scores > 1).any()
    ):
        raise ValueError(ValidationMessage.INVALID_SCORES)
    return max(
        ValidationSetting.THRESHOLDS,
        key=lambda threshold: (
            multilabel_metrics(truth, scores >= threshold)[
                MetricColumn.F1_MICRO
            ],
            threshold,
        ),
    )


def run_validation(paths: ProjectPaths) -> None:
    """Export a fixed source split and independently evaluated thresholds.

    :param paths: Saved benchmark predictions and publication directories.
    :type paths: ProjectPaths
    :return: None.
    :rtype: None
    """
    rows: list[BenchmarkMetricRow] = []
    paired: list[BenchmarkMetricRow] = []
    splits: list[DataFrame] = []
    fingerprints: dict[str, str] = {}
    dataset: SoundscapeDataset
    version: SoundscapeVersion
    for dataset in SoundscapeDataset:
        manifest_path: Path = paths.results / SoundscapeFile.MANIFEST.format(
            dataset=dataset
        )
        if not manifest_path.exists():
            return
        manifest: DataFrame = read_csv(manifest_path)
        split: DataFrame = split_sources(manifest)
        split[ExperimentColumn.DATASET] = dataset.value
        splits.append(split)
        calibration: DataFrame = split[
            split[ValidationSetting.PARTITION] == ValidationSetting.CALIBRATION
        ]
        test: DataFrame = split[
            split[ValidationSetting.PARTITION] == ValidationSetting.TEST
        ]
        reference_path: Path = paths.results / SoundscapeFile.REFERENCE.format(
            dataset=dataset
        )
        reference: DataFrame = read_csv(reference_path)
        truth: BooleanArray = reference.to_numpy(dtype=bool)
        fingerprints[str(reference_path.relative_to(paths.results))] = digest(
            reference_path
        )
        fingerprints[str(manifest_path.relative_to(paths.results))] = digest(
            manifest_path
        )
        decisions: dict[str, BooleanArray] = {}
        thresholds: dict[str, float] = {}
        for version in SoundscapeVersion:
            score_path: Path = paths.results / SoundscapeFile.SCORES.format(
                dataset=dataset, version=version
            )
            frame: DataFrame = read_csv(
                score_path, index_col=ExperimentColumn.FILEPATH
            )
            if (
                frame.index.tolist()
                != manifest[ExperimentColumn.FILEPATH].tolist()
                or frame.columns.tolist() != reference.columns.tolist()
            ):
                raise ValueError(ValidationMessage.MISALIGNED_SCORES)
            scores: Float64Array = frame.to_numpy(dtype=float64)
            threshold: float = select_threshold(
                truth[calibration.index], scores[calibration.index]
            )
            decisions[version] = scores[test.index] >= threshold
            thresholds[version] = threshold
            metrics: dict[str, float | int] = evaluate(
                truth[test.index], scores[test.index], test, (threshold,)
            )[0]
            rows.append(
                {
                    ExperimentColumn.DATASET: dataset.value,
                    MetadataKey.MODEL: version.value,
                    ValidationSetting.CALIBRATION_SOURCES: calibration[
                        ExperimentColumn.SOURCE_RECORDING
                    ].nunique(),
                    ValidationSetting.CALIBRATION_F1: multilabel_metrics(
                        truth[calibration.index],
                        scores[calibration.index] >= threshold,
                    )[MetricColumn.F1_MICRO],
                    **metrics,
                }
            )
            fingerprints[str(score_path.relative_to(paths.results))] = digest(
                score_path
            )
        difference: float
        low: float
        high: float
        difference, low, high = bootstrap_f1_difference(
            truth[test.index],
            decisions[SoundscapeVersion.V3_0],
            decisions[SoundscapeVersion.V2_4],
            test[ExperimentColumn.SOURCE_RECORDING].to_numpy(),
            seed=SoundscapeSetting.RANDOM_SEED,
        )
        paired.append(
            {
                ExperimentColumn.DATASET: dataset.value,
                "threshold_2_4": thresholds[SoundscapeVersion.V2_4],
                "threshold_3_0": thresholds[SoundscapeVersion.V3_0],
                MetricColumn.F1_DELTA_3_MINUS_2: difference,
                ExperimentColumn.LOW: low,
                ExperimentColumn.HIGH: high,
            }
        )
    destination: Path = paths.results / ValidationSetting.DIRECTORY
    write_table(DataFrame(rows), destination / ValidationSetting.METRICS)
    write_table(DataFrame(paired), destination / ValidationSetting.PAIRED)
    write_table(
        concat(splits, ignore_index=True),
        destination / ValidationSetting.SPLIT,
    )
    with atomic_output(
        destination / ValidationSetting.PROVENANCE
    ) as temporary:
        temporary.write_text(
            dumps(
                {
                    "created_at": datetime.now(UTC).isoformat(),
                    "seed": SoundscapeSetting.RANDOM_SEED,
                    "calibration_fraction": ValidationSetting.CALIBRATION_FRACTION,
                    "threshold_grid": ValidationSetting.THRESHOLDS,
                    "selection": "maximum calibration micro F1; higher threshold breaks ties",
                    "interval": "test-source bootstrap conditional on selected threshold",
                    "fingerprints": fingerprints,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
