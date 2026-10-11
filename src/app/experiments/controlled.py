"""Measure annotated targets under natural noise and equal-RMS mixtures."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from datetime import UTC, datetime
from json import dumps, loads
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from numpy import (
    float64,
    isfinite,
)
from numpy.random import Generator, default_rng
from pandas import DataFrame, Series, read_csv

from app.domain.constants import MetadataKey, PredictionColumn, ResultColumn
from app.domain.protocols import CsvWritable, SoundscapeModel
from app.domain.settings import ProjectPaths
from app.domain.types import Float64Array
from app.experiments.constants import (
    ControlColumn,
    ControlSetting,
    ExperimentColumn,
    SoundscapeDataset,
    SoundscapeSetting,
    SoundscapeVersion,
    ValidationSetting,
)
from app.experiments.messages import ControlledMessage
from app.experiments.perturbations import render_audio
from app.experiments.soundscapes import digest, prepare_soundscapes
from app.experiments.statistics import bootstrap_mean
from app.reporting.writing import atomic_output, write_table


@dataclass(frozen=True, slots=True)
class ControlInputs:
    """Keep selected observations separate from all prediction scores."""

    foregrounds: dict[str, DataFrame]
    backgrounds: list[tuple[str, list[str]]]
    audio: dict[str, Path]
    mapping: dict[str, dict[str, list[str]]]
    thresholds: dict[str, float]
    binding: dict[str, object]


def select_controls(paths: ProjectPaths) -> ControlInputs:
    """Select held-out positive sources and disjoint calibration backgrounds.

    :param paths: Benchmark cache, source split and bound model provenance.
    :type paths: ProjectPaths
    :return: Reproducible input selection independent of prediction successes.
    :rtype: ControlInputs
    """
    split_path: Path = (
        paths.results / ValidationSetting.DIRECTORY / ValidationSetting.SPLIT
    )
    split: DataFrame = read_csv(split_path)
    validation: DataFrame = read_csv(
        paths.results
        / ValidationSetting.DIRECTORY
        / ValidationSetting.METRICS,
        dtype={MetadataKey.MODEL: str},
    )
    provenance_path: Path = (
        paths.results / "benchmark" / ControlSetting.PROVENANCE
    )
    provenance: dict[str, object] = loads(provenance_path.read_text())
    records: dict[str, dict[str, object]] = cast(
        dict[str, dict[str, object]], provenance[ExperimentColumn.DATASETS]
    )
    foregrounds: dict[str, DataFrame] = {}
    mappings: dict[str, dict[str, list[str]]] = {}
    thresholds: dict[str, float] = {}
    generator: Generator = default_rng(SoundscapeSetting.RANDOM_SEED)
    dataset: SoundscapeDataset
    for dataset in SoundscapeDataset:
        subset: DataFrame = split[
            (split[ExperimentColumn.DATASET] == dataset)
            & (split[ValidationSetting.PARTITION] == ValidationSetting.TEST)
        ]
        eligible: DataFrame = subset[
            (subset[ExperimentColumn.REFERENCE_EBIRD_CODES] != "[]")
            & (
                subset[PredictionColumn.END_TIME]
                - subset[PredictionColumn.START_TIME]
                >= ControlSetting.MINIMUM_SECONDS
            )
        ]
        sources: list[str] = sorted(
            str(value)
            for value in eligible[ExperimentColumn.SOURCE_RECORDING].unique()
        )
        if len(sources) < ControlSetting.POSITIVE_SOURCES:
            raise ValueError(
                ControlledMessage.INSUFFICIENT_FOREGROUNDS.format(
                    dataset=dataset
                )
            )
        chosen: list[int] = []
        source: str
        for source in generator.choice(
            sources, ControlSetting.POSITIVE_SOURCES, replace=False
        ).tolist():
            group: DataFrame = eligible[
                eligible[ExperimentColumn.SOURCE_RECORDING] == source
            ]
            chosen.append(int(generator.choice(group.index)))
        foregrounds[dataset] = eligible.loc[chosen].reset_index(drop=True)
        mapping: dict[str, dict[str, list[str]]] = cast(
            dict[str, dict[str, list[str]]],
            records[dataset][ExperimentColumn.MAPPING],
        )
        mappings[dataset] = mapping[SoundscapeVersion.V3_0]
        thresholds[dataset] = float(
            validation.loc[
                (validation[ExperimentColumn.DATASET] == dataset)
                & (validation[MetadataKey.MODEL] == SoundscapeVersion.V3_0),
                ResultColumn.THRESHOLD,
            ].iloc[0]
        )
    negative: DataFrame = split[
        (split[ExperimentColumn.DATASET] == SoundscapeDataset.HSN)
        & (split[ValidationSetting.PARTITION] == ValidationSetting.CALIBRATION)
        & (split[ExperimentColumn.REFERENCE_EBIRD_CODES] == "[]")
        & (
            split[PredictionColumn.END_TIME]
            - split[PredictionColumn.START_TIME]
            >= ControlSetting.BACKGROUND_SECONDS
        )
    ]
    background_sources: list[str] = sorted(
        str(value)
        for value in negative[ExperimentColumn.SOURCE_RECORDING].unique()
    )
    if len(background_sources) < ControlSetting.BACKGROUND_SOURCES:
        raise ValueError(ControlledMessage.INSUFFICIENT_BACKGROUNDS)
    backgrounds: list[tuple[str, list[str]]] = [
        (
            source,
            sorted(
                str(value)
                for value in negative.loc[
                    negative[ExperimentColumn.SOURCE_RECORDING] == source,
                    ExperimentColumn.FILEPATH,
                ]
            ),
        )
        for source in generator.choice(
            background_sources,
            ControlSetting.BACKGROUND_SOURCES,
            replace=False,
        ).tolist()
    ]
    audio: dict[str, Path] = {
        path.name: path for path in (paths.data / "benchmark").rglob("*.ogg")
    }
    selected_files: set[str] = {
        str(value)
        for frame in foregrounds.values()
        for value in frame[ExperimentColumn.FILEPATH]
    }
    selected_files.update(name for _, names in backgrounds for name in names)
    binding: dict[str, object] = {
        "protocol": ControlSetting.PROTOCOL_VERSION,
        "seed": SoundscapeSetting.RANDOM_SEED,
        "split_sha256": digest(split_path),
        "model": cast(dict[str, object], provenance[ExperimentColumn.MODELS])[
            SoundscapeVersion.V3_0
        ],
        "environment": provenance[ExperimentColumn.ENVIRONMENT],
        "mapping": mappings,
        "thresholds": thresholds,
        "snr_values": ControlSetting.SNR_VALUES,
        "repetitions": ControlSetting.REPETITIONS,
        "mix_ratios": ControlSetting.MIX_RATIOS,
        "rms": ControlSetting.RMS,
        "sample_rate": ControlSetting.SAMPLE_RATE,
        "minimum_seconds": ControlSetting.MINIMUM_SECONDS,
        "audio_sha256": {
            name: digest(audio[name]) for name in sorted(selected_files)
        },
        "selection": "annotation-positive test sources, no confidence filtering; annotation-negative calibration background sources",
    }
    return ControlInputs(
        foregrounds, backgrounds, audio, mappings, thresholds, binding
    )


def noise_manifest(inputs: ControlInputs, dataset: str) -> DataFrame:
    """Bind every noise repetition to one foreground and background source.

    :param inputs: Annotation-selected foregrounds and disjoint backgrounds.
    :type inputs: ControlInputs
    :param dataset: Reference vocabulary of this foreground subset.
    :type dataset: str
    :return: All clean and perturbed observations, without model outputs.
    :rtype: DataFrame
    """
    rows: list[dict[str, str | int | float]] = []
    index: Hashable
    row: Series[str | float | int]
    for index, row in inputs.foregrounds[dataset].iterrows():
        group: str
        names: list[str]
        group, names = inputs.backgrounds[
            int(str(index)) % len(inputs.backgrounds)
        ]
        base: dict[str, str | int | float] = {
            ExperimentColumn.DATASET: dataset,
            ControlColumn.UNIT: str(row[ExperimentColumn.FILEPATH]),
            ControlColumn.FIRST_FILE: str(row[ExperimentColumn.FILEPATH]),
            ControlColumn.FIRST_SOURCE: str(
                row[ExperimentColumn.SOURCE_RECORDING]
            ),
            ControlColumn.FIRST_TARGET: str(
                row[ExperimentColumn.REFERENCE_EBIRD_CODES]
            ),
            ControlColumn.GROUP: group,
        }
        rows.append(
            {
                **base,
                ExperimentColumn.FILEPATH: f"clean_{index}.wav",
                ControlColumn.CONDITION: ControlSetting.CLEAN,
                ControlColumn.SNR: float("inf"),
                ControlColumn.TRIAL: 0,
                ControlColumn.BACKGROUND: names[0],
            }
        )
        condition: str
        snr: float
        trial: int
        for condition in (ControlSetting.GAUSSIAN, ControlSetting.NATURAL):
            for snr in ControlSetting.SNR_VALUES:
                for trial in range(ControlSetting.REPETITIONS):
                    rows.append(
                        {
                            **base,
                            ExperimentColumn.FILEPATH: f"{condition}_{index}_{snr:g}_{trial}.wav",
                            ControlColumn.CONDITION: condition,
                            ControlColumn.SNR: snr,
                            ControlColumn.TRIAL: trial + 1,
                            ControlColumn.BACKGROUND: names[
                                trial % len(names)
                            ],
                        }
                    )
    return DataFrame(rows)


def overlap_manifest(inputs: ControlInputs, dataset: str) -> DataFrame:
    """Pair distinct held-out sources whose annotated species do not overlap.

    :param inputs: Fixed positive source selection before model inference.
    :type inputs: ControlInputs
    :param dataset: Reference vocabulary shared by both mixture components.
    :type dataset: str
    :return: Isolated controls and mixtures for disjoint recording pairs.
    :rtype: DataFrame
    """
    foregrounds: DataFrame = inputs.foregrounds[dataset]
    remaining: list[int] = list(range(len(foregrounds)))
    rows: list[dict[str, str | int | float]] = []
    while remaining:
        first: int = remaining.pop(0)
        labels_a: list[str] = loads(
            str(
                foregrounds.iloc[first][ExperimentColumn.REFERENCE_EBIRD_CODES]
            )
        )
        second: int | None = next(
            (
                i
                for i in remaining
                if not set(labels_a).intersection(
                    loads(
                        str(
                            foregrounds.iloc[i][
                                ExperimentColumn.REFERENCE_EBIRD_CODES
                            ]
                        )
                    )
                )
            ),
            None,
        )
        if second is None:
            continue
        remaining.remove(second)
        unit: str = f"{dataset}_{first}_{second}"
        alpha: float
        beta: float
        for alpha, beta in ControlSetting.MIX_RATIOS:
            condition: str
            for condition in (
                ControlSetting.FIRST,
                ControlSetting.SECOND,
                ControlSetting.MIXED,
            ):
                rows.append(
                    {
                        ExperimentColumn.DATASET: dataset,
                        ExperimentColumn.FILEPATH: f"{unit}_{alpha:g}_{condition}.wav",
                        ControlColumn.UNIT: unit,
                        ControlColumn.GROUP: unit,
                        ControlColumn.CONDITION: condition,
                        ControlColumn.RATIO: f"{alpha:g}/{beta:g}",
                        ControlColumn.FIRST_FILE: str(
                            foregrounds.iloc[first][ExperimentColumn.FILEPATH]
                        ),
                        ControlColumn.SECOND_FILE: str(
                            foregrounds.iloc[second][ExperimentColumn.FILEPATH]
                        ),
                        ControlColumn.FIRST_SOURCE: str(
                            foregrounds.iloc[first][
                                ExperimentColumn.SOURCE_RECORDING
                            ]
                        ),
                        ControlColumn.SECOND_SOURCE: str(
                            foregrounds.iloc[second][
                                ExperimentColumn.SOURCE_RECORDING
                            ]
                        ),
                        ControlColumn.FIRST_TARGET: labels_a[0],
                        ControlColumn.SECOND_TARGET: loads(
                            str(
                                foregrounds.iloc[second][
                                    ExperimentColumn.REFERENCE_EBIRD_CODES
                                ]
                            )
                        )[0],
                    }
                )
    if len(rows) < 18:
        raise ValueError(ControlledMessage.INSUFFICIENT_PAIRS)
    return DataFrame(rows)


def infer_controls(
    model: SoundscapeModel,
    manifest: DataFrame,
    inputs: ControlInputs,
    dataset: str,
    kind: str,
) -> DataFrame:
    """Export complete reference scores without decision-threshold truncation.

    :param model: BirdNET 3.0 with the recorded weight fingerprint.
    :type model: SoundscapeModel
    :param manifest: Predetermined audio variants in stable order.
    :type manifest: DataFrame
    :param inputs: Cached foregrounds, backgrounds and reference mappings.
    :type inputs: ControlInputs
    :param dataset: Reference vocabulary to score.
    :type dataset: str
    :param kind: Noise or overlap protocol.
    :type kind: str
    :return: Complete clip/concept scores aligned with the manifest.
    :rtype: DataFrame
    """
    with TemporaryDirectory(prefix="birdnet_controlled_") as temporary:
        directory: Path = Path(temporary)
        paths: list[Path] = render_audio(
            manifest, inputs.audio, directory, kind
        )
        result: CsvWritable = model.predict(
            paths,
            top_k=None,
            n_workers=SoundscapeSetting.INFERENCE_WORKERS,
            n_producers=SoundscapeSetting.AUDIO_PRODUCERS,
            batch_size=SoundscapeSetting.INFERENCE_BATCH_SIZE,
            default_confidence_threshold=SoundscapeSetting.EXPORT_THRESHOLD,
            custom_species_list=sorted(
                {
                    label
                    for labels in inputs.mapping[dataset].values()
                    for label in labels
                }
            ),
            show_stats="minimal",
        )
        raw_path: Path = directory / "windows.csv"
        result.to_csv(str(raw_path), silent=True)
        raw: DataFrame = read_csv(raw_path)
        raw[PredictionColumn.INPUT] = raw[PredictionColumn.INPUT].map(
            lambda value: Path(str(value)).name
        )
        if set(raw[PredictionColumn.INPUT]) != set(
            manifest[ExperimentColumn.FILEPATH]
        ):
            raise ValueError(ControlledMessage.INCOMPLETE_PREDICTIONS)
        table: DataFrame = (
            raw.groupby(
                [PredictionColumn.INPUT, PredictionColumn.SPECIES_NAME]
            )[PredictionColumn.CONFIDENCE]
            .max()
            .unstack()
        )
        scores: DataFrame = DataFrame(
            index=manifest[ExperimentColumn.FILEPATH]
        )
        code: str
        labels: list[str]
        for code, labels in inputs.mapping[dataset].items():
            scores[code] = (
                table.reindex(index=scores.index, columns=labels)
                .fillna(0.0)
                .max(axis=1)
                if labels
                else 0.0
            )
        return scores


def summarize_noise(
    manifest: DataFrame, scores: DataFrame, threshold: float
) -> tuple[DataFrame, DataFrame]:
    """Average repetitions per clip, then bootstrap shared-background groups.

    :param manifest: Noise conditions, foreground units and background groups.
    :type manifest: DataFrame
    :param scores: Full scores in manifest order and reference vocabulary.
    :type scores: DataFrame
    :param threshold: Decision fixed using distinct calibration sources.
    :type threshold: float
    :return: Repetition measurements and grouped interval summaries.
    :rtype: tuple[DataFrame, DataFrame]
    """
    measured: DataFrame = manifest.copy()
    recoveries: list[float] = []
    confidences: list[float] = []
    index: Hashable
    row: Series[str | float | int]
    for index, row in manifest.iterrows():
        codes: list[str] = loads(str(row[ControlColumn.FIRST_TARGET]))
        values: Float64Array = scores.iloc[int(str(index))][codes].to_numpy(
            dtype=float64
        )
        recoveries.append(float((values >= threshold).mean()))
        confidences.append(float(values.mean()))
    measured[ControlColumn.RECOVERY] = recoveries
    measured[ControlColumn.CONFIDENCE] = confidences
    clean: Series[float] = measured[
        measured[ControlColumn.CONDITION] == ControlSetting.CLEAN
    ].set_index(ControlColumn.UNIT)[ControlColumn.RECOVERY]
    measured[ControlColumn.CLEAN_RECOVERY] = measured[ControlColumn.UNIT].map(
        clean
    )
    measured[ControlColumn.DELTA] = (
        measured[ControlColumn.RECOVERY]
        - measured[ControlColumn.CLEAN_RECOVERY]
    )
    units: DataFrame = measured.groupby(
        [
            ControlColumn.CONDITION,
            ControlColumn.SNR,
            ControlColumn.UNIT,
            ControlColumn.GROUP,
        ],
        as_index=False,
    )[
        [ControlColumn.RECOVERY, ControlColumn.CONFIDENCE, ControlColumn.DELTA]
    ].mean()
    rows: list[dict[str, str | int | float]] = []
    key: tuple[Hashable, ...]
    group: DataFrame
    for key, group in units.groupby(
        [ControlColumn.CONDITION, ControlColumn.SNR], sort=False
    ):
        estimate: float
        low: float
        high: float
        estimate, low, high = bootstrap_mean(
            group[ControlColumn.RECOVERY].to_numpy(dtype=float64),
            groups=group[ControlColumn.GROUP].to_numpy(),
        )
        delta: float
        delta_low: float
        delta_high: float
        delta, delta_low, delta_high = bootstrap_mean(
            group[ControlColumn.DELTA].to_numpy(dtype=float64),
            groups=group[ControlColumn.GROUP].to_numpy(),
        )
        rows.append(
            {
                ControlColumn.CONDITION: str(key[0]),
                ControlColumn.SNR: float(str(key[1])),
                ControlColumn.ESTIMATE: estimate,
                ControlColumn.LOW: low,
                ControlColumn.HIGH: high,
                ControlColumn.DELTA: delta,
                "delta_low": delta_low,
                "delta_high": delta_high,
                ControlColumn.UNITS: len(group),
                ControlColumn.GROUPS: group[ControlColumn.GROUP].nunique(),
            }
        )
    return measured, DataFrame(rows)


def summarize_overlap(
    manifest: DataFrame, scores: DataFrame, threshold: float
) -> tuple[DataFrame, DataFrame]:
    """Compare joint recovery with isolated controls of the same two signals.

    :param manifest: Matched controls and mixtures for disjoint source pairs.
    :type manifest: DataFrame
    :param scores: Every reference concept score in manifest order.
    :type scores: DataFrame
    :param threshold: Decision fixed on separate calibration sources.
    :type threshold: float
    :return: Pair outcomes and paired bootstrap recovery intervals.
    :rtype: tuple[DataFrame, DataFrame]
    """
    rows: list[dict[str, str | float]] = []
    key: tuple[Hashable, ...]
    group: DataFrame
    for key, group in manifest.groupby(
        [ControlColumn.UNIT, ControlColumn.RATIO], sort=False
    ):
        observations: dict[str, int] = {
            str(row[ControlColumn.CONDITION]): int(str(index))
            for index, row in group.iterrows()
        }
        a: str = str(group[ControlColumn.FIRST_TARGET].iloc[0])
        b: str = str(group[ControlColumn.SECOND_TARGET].iloc[0])
        alone: float = float(
            scores.iloc[observations[ControlSetting.FIRST]][a] >= threshold
            and scores.iloc[observations[ControlSetting.SECOND]][b]
            >= threshold
        )
        mixed: float = float(
            scores.iloc[observations[ControlSetting.MIXED]][a] >= threshold
            and scores.iloc[observations[ControlSetting.MIXED]][b] >= threshold
        )
        rows.append(
            {
                ControlColumn.UNIT: str(key[0]),
                ControlColumn.RATIO: str(key[1]),
                ControlColumn.CLEAN_RECOVERY: alone,
                ControlColumn.RECOVERY: mixed,
                ControlColumn.DELTA: mixed - alone,
            }
        )
    measured: DataFrame = DataFrame(rows)
    summaries: list[dict[str, str | int | float]] = []
    ratio_value: Hashable
    for ratio_value, group in measured.groupby(
        ControlColumn.RATIO, sort=False
    ):
        ratio: str = str(ratio_value)
        estimate: float
        low: float
        high: float
        estimate, low, high = bootstrap_mean(
            group[ControlColumn.RECOVERY].to_numpy(dtype=float64)
        )
        delta: float
        delta_low: float
        delta_high: float
        delta, delta_low, delta_high = bootstrap_mean(
            group[ControlColumn.DELTA].to_numpy(dtype=float64)
        )
        summaries.append(
            {
                ControlColumn.RATIO: ratio,
                ControlColumn.ESTIMATE: estimate,
                ControlColumn.LOW: low,
                ControlColumn.HIGH: high,
                ControlColumn.CLEAN_RECOVERY: float(
                    group[ControlColumn.CLEAN_RECOVERY].mean()
                ),
                ControlColumn.DELTA: delta,
                "delta_low": delta_low,
                "delta_high": delta_high,
                ControlColumn.UNITS: len(group),
                ControlColumn.GROUPS: len(group),
            }
        )
    return measured, DataFrame(summaries)


def run_controls(paths: ProjectPaths) -> None:
    """Run or reuse bound perturbation scores, keeping all measured outcomes.

    :param paths: Source-separated benchmark data and result destinations.
    :type paths: ProjectPaths
    :return: None.
    :rtype: None
    """
    inputs: ControlInputs = select_controls(paths)
    model: SoundscapeModel | None = None
    dataset: SoundscapeDataset
    kind: str
    for dataset in SoundscapeDataset:
        for kind in (ControlSetting.NOISE, ControlSetting.OVERLAP):
            root: Path = (
                paths.results / ControlSetting.DIRECTORY / kind / dataset
            )
            root.mkdir(parents=True, exist_ok=True)
            manifest: DataFrame = (
                noise_manifest(inputs, dataset)
                if kind == ControlSetting.NOISE
                else overlap_manifest(inputs, dataset)
            )
            manifest_path: Path = root / ControlSetting.MANIFEST
            score_path: Path = root / ControlSetting.SCORES
            provenance_path: Path = root / ControlSetting.PROVENANCE
            binding: dict[str, object] = {
                **inputs.binding,
                ExperimentColumn.DATASET: dataset.value,
                "kind": kind,
            }
            previous: dict[str, object] = (
                loads(provenance_path.read_text())
                if provenance_path.exists()
                else {}
            )
            cached: bool = previous.get("binding") == loads(
                dumps(binding)
            ) and all(
                path.exists() and previous.get(path.name) == digest(path)
                for path in (manifest_path, score_path)
            )
            if not cached:
                if model is None:
                    model = prepare_soundscapes(paths).models[
                        SoundscapeVersion.V3_0
                    ]
                    if (
                        digest(model.model_path)
                        != cast(dict[str, str], inputs.binding["model"])[
                            ExperimentColumn.SHA256
                        ]
                    ):
                        raise ValueError(ControlledMessage.CHANGED_MODEL)
                print(
                    ControlledMessage.RUNNING_INFERENCE.format(
                        kind=kind, dataset=dataset, count=len(manifest)
                    ),
                    flush=True,
                )
                scores: DataFrame = infer_controls(
                    model, manifest, inputs, dataset, kind
                )
                write_table(manifest, manifest_path)
                write_table(scores, score_path, index=True)
                with atomic_output(provenance_path) as temporary:
                    temporary.write_text(
                        dumps(
                            {
                                "created_at": datetime.now(UTC).isoformat(),
                                "binding": binding,
                                ControlSetting.MANIFEST: digest(manifest_path),
                                ControlSetting.SCORES: digest(score_path),
                            },
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
            scores = read_csv(score_path, index_col=ExperimentColumn.FILEPATH)
            values: Float64Array = scores.to_numpy(dtype=float64)
            if (
                scores.index.tolist()
                != manifest[ExperimentColumn.FILEPATH].tolist()
                or scores.columns.tolist() != list(inputs.mapping[dataset])
                or not isfinite(values).all()
                or (values < 0).any()
                or (values > 1).any()
            ):
                raise ValueError(ControlledMessage.INVALID_SCORES)
            measured: DataFrame
            summary: DataFrame
            if kind == ControlSetting.NOISE:
                measured, summary = summarize_noise(
                    manifest, scores, inputs.thresholds[dataset]
                )
            else:
                measured, summary = summarize_overlap(
                    manifest, scores, inputs.thresholds[dataset]
                )
            summary[ExperimentColumn.DATASET] = dataset.value
            summary[ResultColumn.THRESHOLD] = inputs.thresholds[dataset]
            write_table(measured, root / ControlSetting.MEASUREMENTS)
            write_table(summary, root / ControlSetting.SUMMARY)
