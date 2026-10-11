"""Run fixed annotated soundscape tests with two BirdNET model versions.

Run from the notebook or with: uv run app soundscapes
Audio is cached locally; manifests and raw scores make metrics reproducible.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import tarfile
import urllib.request
from collections.abc import Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from importlib.metadata import version as package_version
from pathlib import Path
from re import fullmatch
from typing import IO, cast

from matplotlib.figure import Figure
from numpy import (
    array,
    float64,
    isfinite,
    quantile,
    unique,
)
from numpy.random import Generator, default_rng
from pandas import DataFrame, read_csv, read_parquet

from app.domain.constants import (
    MetadataKey,
    PredictionColumn,
    ResultColumn,
    RuntimeModule,
)
from app.domain.protocols import (
    CsvWritable,
    SoundscapeLibrary,
    SoundscapeModel,
)
from app.domain.settings import ProjectPaths
from app.domain.types import (
    BenchmarkMetricRow,
    BooleanArray,
    Float64Array,
    IntegerArray,
    MetricValues,
)
from app.experiments.constants import (
    TAXON_CONCEPTS,
    EvaluationScope,
    ExperimentColumn,
    MetricColumn,
    SoundscapeDataset,
    SoundscapeFile,
    SoundscapeSetting,
    SoundscapeVersion,
    StatisticalSetting,
    TaxonomyField,
)
from app.experiments.messages import SoundscapeMessage
from app.experiments.statistics import (
    grouped_outcome_counts,
    multilabel_metrics,
    resampled_f1,
)
from app.reporting.constants import ExperimentFile
from app.reporting.messages import ExperimentLabel
from app.reporting.plots import ExperimentPlotter
from app.reporting.protocols import PlotAxis, PlotFigure
from app.reporting.writing import atomic_output, write_table


def digest(path: Path) -> str:
    """Identify the exact dataset or model file used.

    :param path: Exact local dataset, model or generated artifact path.
    :type path: Path
    :return: SHA-256 fingerprint of the complete file.
    :rtype: str
    """
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, ExperimentColumn.SHA256).hexdigest()


def download(url: str, path: Path) -> Path:
    """Cache complete downloads; never reuse interrupted partial files.

    :param url: Public download URL for the fixed data revision.
    :type url: str
    :param path: Exact local dataset, model or generated artifact path.
    :type path: Path
    :return: Path to a complete cached download.
    :rtype: Path
    """
    if not path.exists():
        print(
            SoundscapeMessage.DOWNLOADING.format(filename=path.name),
            flush=True,
        )
        with atomic_output(path) as temporary:
            with urllib.request.urlopen(
                url, timeout=SoundscapeSetting.DOWNLOAD_TIMEOUT_SECONDS
            ) as incoming:
                with temporary.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
    return path


def archive_fingerprint(archive: Path) -> str:
    """Preserve the source checksum after removing an extracted archive.

    :param archive: Complete source archive or its original cached path.
    :type archive: Path
    :return: Verified SHA-256 fingerprint of the original source archive.
    :rtype: str
    """
    record: Path = archive.with_name(
        archive.name + SoundscapeSetting.ARCHIVE_FINGERPRINT_SUFFIX
    )
    fingerprint: str
    if archive.is_file():
        fingerprint = digest(archive)
        if record.exists() and record.read_text().strip() != fingerprint:
            raise ValueError(
                SoundscapeMessage.CHANGED_ARCHIVE.format(path=archive)
            )
        with atomic_output(record) as temporary:
            temporary.write_text(fingerprint + "\n", encoding="utf-8")
        return fingerprint
    fingerprint = record.read_text().strip() if record.is_file() else ""
    if (
        fullmatch(SoundscapeSetting.ARCHIVE_FINGERPRINT_PATTERN, fingerprint)
        is None
    ):
        raise ValueError(
            SoundscapeMessage.INVALID_ARCHIVE_FINGERPRINT.format(path=record)
        )
    return fingerprint


def select_metadata(
    dataset: str, cache: Path, output: Path
) -> tuple[DataFrame, list[str]]:
    """Keep all NBP rows; sample six windows per HSN source, without labels.

    :param dataset: Stable identifier of the fixed public test subset.
    :type dataset: str
    :param cache: Local directory retaining downloaded data.
    :type cache: Path
    :param output: Destination directory for measured results and provenance.
    :type output: Path
    :return: Fixed clip metadata and the full reference vocabulary.
    :rtype: tuple[DataFrame, list[str]]
    """
    path: Path = download(
        SoundscapeFile.METADATA_URL.format(
            base=SoundscapeSetting.BASE_URL, dataset=dataset
        ),
        cache / dataset / ExperimentFile.METADATA_PARQUET,
    )
    metadata: DataFrame = read_parquet(path).reset_index()
    metadata[ExperimentColumn.SOURCE_RECORDING] = (
        metadata[ExperimentColumn.FILEPATH].str.rsplit("_", n=2).str[0]
    )
    vocabulary: list[str] = sorted(
        {
            code
            for labels in metadata[ExperimentColumn.EBIRD_CODE_MULTILABEL]
            for code in labels
        }
    )
    if dataset == SoundscapeDataset.HSN:
        generator: Generator = default_rng(SoundscapeSetting.RANDOM_SEED)
        chosen: list[int] = []
        group: DataFrame
        for _, group in metadata.groupby(
            ExperimentColumn.SOURCE_RECORDING, sort=True
        ):
            chosen.extend(
                generator.choice(
                    group.index,
                    SoundscapeSetting.SOURCE_WINDOWS,
                    replace=False,
                )
            )
        metadata = metadata.loc[chosen].sort_values(ExperimentColumn.FILEPATH)
    else:
        metadata = metadata.sort_values(ExperimentColumn.FILEPATH)
    metadata = metadata.reset_index(drop=True)
    manifest: DataFrame = metadata[
        [
            ExperimentColumn.FILEPATH,
            ExperimentColumn.SOURCE_RECORDING,
            PredictionColumn.START_TIME,
            PredictionColumn.END_TIME,
        ]
    ].copy()
    manifest[ExperimentColumn.REFERENCE_EBIRD_CODES] = metadata[
        ExperimentColumn.EBIRD_CODE_MULTILABEL
    ].map(lambda values: json.dumps(list(cast(Collection[str], values))))
    destination: Path = output / SoundscapeFile.MANIFEST.format(
        dataset=dataset
    )
    if destination.exists():
        previous: DataFrame = read_csv(destination)
        if (
            previous[ExperimentColumn.FILEPATH].tolist()
            != manifest[ExperimentColumn.FILEPATH].tolist()
        ):
            raise ValueError(SoundscapeMessage.CHANGED_SELECTION)
    write_table(manifest, destination)
    return (metadata, vocabulary)


def extract_audio(
    dataset: str, metadata: DataFrame, cache: Path
) -> list[Path]:
    """Extract only selected regular audio files, ignoring archive paths.

    :param dataset: Stable identifier of the fixed public test subset.
    :type dataset: str
    :param metadata: Selected clip metadata with source identifiers.
    :type metadata: DataFrame
    :param cache: Local directory retaining downloaded data.
    :type cache: Path
    :return: Selected regular audio files in metadata order.
    :rtype: list[Path]
    """
    directory: Path = cache / dataset / "audio"
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = [
        directory / name for name in metadata[ExperimentColumn.FILEPATH]
    ]
    missing: set[str] = {path.name for path in paths if not path.exists()}
    archive: Path = cache / dataset / ExperimentFile.AUDIO_TAR_GZ
    if missing:
        archive = download(
            SoundscapeFile.AUDIO_URL.format(
                base=SoundscapeSetting.BASE_URL, dataset=dataset
            ),
            archive,
        )
        with tarfile.open(archive, "r:gz") as stream:
            member: tarfile.TarInfo
            for member in stream:
                name: str = Path(member.name).name
                if not member.isfile() or name not in missing:
                    continue
                source: IO[bytes] | None = stream.extractfile(member)
                if source is None:
                    raise ValueError(
                        SoundscapeMessage.MISSING_ARCHIVE_CONTENTS.format(
                            name=name
                        )
                    )
                with source, (directory / name).open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                missing.remove(name)
        if missing:
            raise ValueError(
                SoundscapeMessage.MISSING_AUDIO.format(names=sorted(missing))
            )
    archive_fingerprint(archive)
    archive.unlink(missing_ok=True)
    return paths


def labels_for_taxon(
    model: SoundscapeModel, taxon: dict[str, str]
) -> list[str]:
    """Map reference concepts, including documented later species splits.

    :param model: Typed BirdNET inference interface.
    :type model: SoundscapeModel
    :param taxon: Reference taxonomy fields for one eBird concept.
    :type taxon: dict[str, str]
    :return: Available model labels within the reference concept.
    :rtype: list[str]
    """
    labels: list[str] = list(model.species_list)
    by_latin: dict[str, str] = {
        label.split("_", 1)[0]: label for label in labels
    }
    concept: tuple[str, ...] | None = TAXON_CONCEPTS.get(
        taxon[TaxonomyField.SPECIES_CODE]
    )
    if concept:
        return [by_latin[name] for name in concept if name in by_latin]
    exact: str | None = by_latin.get(taxon[TaxonomyField.SCIENTIFIC_NAME])
    if exact:
        return [exact]
    matches: list[str] = [
        label
        for label in labels
        if label.split("_", 1)[-1] == taxon[TaxonomyField.COMMON_NAME]
    ]
    return matches if len(matches) == 1 else []


def cache_matches(
    provenance_path: Path,
    binding: dict[str, object],
    scores_path: Path,
    windows_path: Path,
) -> bool:
    """Reuse scores only with the same inputs and unchanged measured files.

    :param provenance_path: Saved input binding and measured-file fingerprints.
    :type provenance_path: Path
    :param binding: Current immutable inputs of this inference run.
    :type binding: dict[str, object]
    :param scores_path: Saved clip-level prediction scores.
    :type scores_path: Path
    :param windows_path: Saved model-window predictions.
    :type windows_path: Path
    :return: Whether inputs and measured-file fingerprints still match.
    :rtype: bool
    """
    if not all(
        path.exists() for path in (provenance_path, scores_path, windows_path)
    ):
        return False
    previous: dict[str, object] = cast(
        dict[str, object], json.loads(provenance_path.read_text())
    )
    return bool(
        previous.get(ExperimentColumn.BINDING) == binding
        and previous.get(ExperimentColumn.SCORES_SHA256) == digest(scores_path)
        and (
            previous.get(ExperimentColumn.WINDOWS_SHA256)
            == digest(windows_path)
        )
    )


def evaluate(
    truth: BooleanArray,
    scores: Float64Array,
    metadata: DataFrame,
    thresholds: tuple[float, ...] = SoundscapeSetting.THRESHOLDS,
) -> list[dict[str, float | int]]:
    """Calculate threshold metrics and source-clustered F1 uncertainty.

    :param truth: Binary labels in the evaluated reference vocabulary.
    :type truth: BooleanArray
    :param scores: Model confidences in the same clip and class order.
    :type scores: Float64Array
    :param metadata: Selected clip metadata with source identifiers.
    :type metadata: DataFrame
    :param thresholds: Decisions fixed before evaluating these observations.
    :type thresholds: tuple[float, ...]
    :return: Threshold metrics with source-clustered F1 intervals.
    :rtype: list[dict[str, float | int]]
    """
    _, groups = unique(
        metadata[ExperimentColumn.SOURCE_RECORDING], return_inverse=True
    )
    group_count: int = int(groups.max()) + 1
    generator: Generator = default_rng(SoundscapeSetting.RANDOM_SEED)
    resamples: IntegerArray = generator.integers(
        group_count, size=(StatisticalSetting.REPETITIONS, group_count)
    )
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        predictions: BooleanArray = scores >= threshold
        metrics: MetricValues = multilabel_metrics(truth, predictions)
        group_counts: IntegerArray = grouped_outcome_counts(
            truth, predictions, groups
        )
        bootstrap_f1: Float64Array = resampled_f1(group_counts, resamples)
        low, high = quantile(
            bootstrap_f1, StatisticalSetting.INTERVAL_QUANTILES
        )
        rows.append(
            {
                ResultColumn.THRESHOLD: threshold,
                **metrics,
                MetricColumn.F1_MICRO_LOW: float(low),
                MetricColumn.F1_MICRO_HIGH: float(high),
                ExperimentColumn.CLIPS: len(truth),
                ExperimentColumn.SOURCE_RECORDINGS: group_count,
            }
        )
    return rows


@dataclass(frozen=True, slots=True)
class SoundscapeContext:
    """Share immutable services and the mutable provenance of one benchmark
    run.
    """

    output: Path
    cache: Path
    taxonomy: dict[str, dict[str, str]]
    models: dict[str, SoundscapeModel]
    model_records: dict[str, dict[str, str | int]]
    environment: dict[str, object]
    provenance: dict[str, object]
    dataset_records: dict[str, dict[str, object]]


@dataclass(frozen=True, slots=True)
class SoundscapeSelection:
    """Keep selected clips and bound taxonomy mappings together."""

    metadata: DataFrame
    evaluated_codes: list[str]
    audio_paths: list[Path]
    mappings: dict[str, dict[str, list[str]]]
    common: list[str]
    excluded: list[str]
    truth: BooleanArray
    dataset_info: dict[str, object]
    inference_records: dict[str, dict[str, object]]


def prepare_soundscapes(project: ProjectPaths) -> SoundscapeContext:
    """Load models and bind the software and taxonomy used by the fixed
    experiment.

    :param project: Project directories for data caches and results.
    :type project: ProjectPaths
    :return: Bound model, taxonomy and software context.
    :rtype: SoundscapeContext
    """
    output: Path = project.results
    cache: Path = project.data / SoundscapeSetting.DIRECTORY
    output.mkdir(parents=True, exist_ok=True)
    tax_path: Path = download(
        SoundscapeSetting.TAXONOMY_URL,
        cache / ExperimentFile.EBIRD_2021_CSV,
    )
    with tax_path.open() as handle:
        taxonomy: dict[str, dict[str, str]] = {
            row[TaxonomyField.SPECIES_CODE]: row
            for row in csv.DictReader(handle)
        }
    library: SoundscapeLibrary = cast(
        SoundscapeLibrary, import_module(RuntimeModule.BIRDNET)
    )
    models: dict[str, SoundscapeModel] = {
        SoundscapeVersion.V2_4: library.load(
            "acoustic", SoundscapeVersion.V2_4.value, "tf", library="litert"
        ),
        SoundscapeVersion.V3_0: library.load(
            "acoustic", SoundscapeVersion.V3_0.value, "onnx"
        ),
    }
    environment: dict[str, object] = {
        ExperimentColumn.PYTHON: platform.python_version(),
        ExperimentColumn.PLATFORM: platform.platform(),
        ExperimentColumn.PACKAGES: {
            package: package_version(package)
            for package in SoundscapeSetting.PACKAGES
        },
    }
    model_records: dict[str, dict[str, str | int]] = {
        version: {
            ExperimentColumn.FILE: model.model_path.name,
            ExperimentColumn.SHA256: digest(model.model_path),
            ExperimentColumn.CLASSES: len(model.species_list),
            ExperimentColumn.BACKEND: "TFLite/LiteRT"
            if version == SoundscapeVersion.V2_4
            else "ONNX",
        }
        for version, model in models.items()
    }
    dataset_records: dict[str, dict[str, object]] = {}
    provenance: dict[str, object] = {
        MetadataKey.CREATED_AT: datetime.now(UTC).isoformat(),
        ExperimentColumn.DATASET_REVISION: SoundscapeSetting.REVISION,
        ResultColumn.SEED: SoundscapeSetting.RANDOM_SEED,
        ExperimentColumn.THRESHOLDS: SoundscapeSetting.THRESHOLDS,
        ExperimentColumn.BOOTSTRAP_REPETITIONS: StatisticalSetting.REPETITIONS,
        ExperimentColumn.TAXONOMY_SHA256: digest(tax_path),
        ExperimentColumn.TRAINING_INDEPENDENCE_VERIFIED: False,
        ExperimentColumn.ENVIRONMENT: environment,
        ExperimentColumn.AGGREGATION: SoundscapeSetting.AGGREGATION,
        ExperimentColumn.MODELS: model_records,
        ExperimentColumn.DATASETS: dataset_records,
    }
    return SoundscapeContext(
        output,
        cache,
        taxonomy,
        models,
        model_records,
        environment,
        provenance,
        dataset_records,
    )


def select_soundscape(
    dataset: str, context: SoundscapeContext
) -> SoundscapeSelection:
    """Select clips once and bind their reference vocabulary and source
    provenance.

    :param dataset: Stable identifier of the fixed public test subset.
    :type dataset: str
    :param context: Models, software and dataset bindings of this run.
    :type context: SoundscapeContext
    :return: Complete selected inputs and dataset provenance.
    :rtype: SoundscapeSelection
    """
    metadata: DataFrame
    vocabulary: list[str]
    metadata, vocabulary = select_metadata(
        dataset, context.cache, context.output
    )
    audio_paths: list[Path] = extract_audio(dataset, metadata, context.cache)
    mappings: dict[str, dict[str, list[str]]] = {
        version: {
            code: labels_for_taxon(model, context.taxonomy[code])
            for code in vocabulary
        }
        for version, model in context.models.items()
    }
    common: list[str] = [
        code
        for code in vocabulary
        if all(mappings[v][code] for v in tuple(SoundscapeVersion))
    ]
    excluded: list[str] = sorted(set(vocabulary) - set(common))
    if excluded:
        print(
            SoundscapeMessage.DIFFERENT_COVERAGE.format(
                dataset=dataset, excluded=excluded
            ),
            flush=True,
        )
    evaluated_codes: list[str] = vocabulary
    print(
        SoundscapeMessage.SELECTED_CLIPS.format(
            dataset=dataset, clips=len(audio_paths), classes=len(common)
        ),
        flush=True,
    )
    truth: BooleanArray = array(
        [
            [code in labels for code in evaluated_codes]
            for labels in metadata[ExperimentColumn.EBIRD_CODE_MULTILABEL]
        ],
        dtype=bool,
    )
    write_table(
        DataFrame(truth.astype(int), columns=evaluated_codes),
        context.output / SoundscapeFile.REFERENCE.format(dataset=dataset),
    )
    inference_records: dict[str, dict[str, object]] = {}
    dataset_info: dict[str, object] = {
        ExperimentColumn.METADATA_SHA256: digest(
            context.cache / dataset / ExperimentFile.METADATA_PARQUET
        ),
        ExperimentColumn.ARCHIVE_SHA256: archive_fingerprint(
            context.cache / dataset / ExperimentFile.AUDIO_TAR_GZ
        ),
        ExperimentColumn.MANIFEST_SHA256: digest(
            context.output / SoundscapeFile.MANIFEST.format(dataset=dataset)
        ),
        ExperimentColumn.CLIPS: len(audio_paths),
        MetricColumn.NEGATIVE_CLIPS: int((~truth.any(1)).sum()),
        ExperimentColumn.REFERENCE_CLASSES: evaluated_codes,
        EvaluationScope.COMMON_CLASSES: common,
        ExperimentColumn.NONCOMMON_CLASSES: excluded,
        ExperimentColumn.MAPPING: mappings,
        ExperimentColumn.AUDIO_SHA256: {
            path.name: digest(path) for path in audio_paths
        },
        ExperimentColumn.INFERENCE: inference_records,
    }
    context.dataset_records[dataset] = dataset_info
    return SoundscapeSelection(
        metadata,
        evaluated_codes,
        audio_paths,
        mappings,
        common,
        excluded,
        truth,
        dataset_info,
        inference_records,
    )


def predict_soundscape(
    dataset: str,
    version: str,
    model: SoundscapeModel,
    context: SoundscapeContext,
    selection: SoundscapeSelection,
) -> Float64Array:
    """Reuse bound predictions or run inference on the same selected clips.

    :param dataset: Stable identifier of the fixed public test subset.
    :type dataset: str
    :param version: Stable model version being evaluated.
    :type version: str
    :param model: Typed BirdNET inference interface.
    :type model: SoundscapeModel
    :param context: Models, software and dataset bindings of this run.
    :type context: SoundscapeContext
    :param selection: Selected clips, reference labels and taxonomy mappings.
    :type selection: SoundscapeSelection
    :return: Validated confidence matrix in reference clip/class order.
    :rtype: Float64Array
    """
    score_path: Path = context.output / SoundscapeFile.SCORES.format(
        dataset=dataset, version=version
    )
    raw_path: Path = context.output / SoundscapeFile.WINDOWS.format(
        dataset=dataset, version=version
    )
    inference_path: Path = context.output / SoundscapeFile.INFERENCE.format(
        dataset=dataset, version=version
    )
    binding: dict[str, object] = {
        ExperimentColumn.PROTOCOL: SoundscapeSetting.PROTOCOL_VERSION,
        ExperimentColumn.DATASET_REVISION: SoundscapeSetting.REVISION,
        ExperimentColumn.METADATA_SHA256: selection.dataset_info[
            ExperimentColumn.METADATA_SHA256
        ],
        ExperimentColumn.MANIFEST_SHA256: selection.dataset_info[
            ExperimentColumn.MANIFEST_SHA256
        ],
        ExperimentColumn.ARCHIVE_SHA256: selection.dataset_info[
            ExperimentColumn.ARCHIVE_SHA256
        ],
        ExperimentColumn.AUDIO_SHA256: selection.dataset_info[
            ExperimentColumn.AUDIO_SHA256
        ],
        ExperimentColumn.TAXONOMY_SHA256: context.provenance[
            ExperimentColumn.TAXONOMY_SHA256
        ],
        ExperimentColumn.MAPPING: selection.mappings[version],
        MetadataKey.MODEL: context.model_records[version],
        ExperimentColumn.ENVIRONMENT: context.environment,
        ExperimentColumn.AGGREGATION: context.provenance[
            ExperimentColumn.AGGREGATION
        ],
        ExperimentColumn.PREDICTION_SETTINGS: {
            ExperimentColumn.TOP_K: None,
            ExperimentColumn.N_WORKERS: SoundscapeSetting.INFERENCE_WORKERS,
            ExperimentColumn.N_PRODUCERS: SoundscapeSetting.AUDIO_PRODUCERS,
            MetadataKey.BATCH_SIZE: SoundscapeSetting.INFERENCE_BATCH_SIZE,
            ExperimentColumn.DEFAULT_CONFIDENCE_THRESHOLD: SoundscapeSetting.EXPORT_THRESHOLD,
        },
    }
    if not cache_matches(inference_path, binding, score_path, raw_path):
        print(
            SoundscapeMessage.RUNNING_INFERENCE.format(
                dataset=dataset, version=version
            ),
            flush=True,
        )
        inference_started: str = datetime.now(UTC).isoformat()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        result: CsvWritable = model.predict(
            selection.audio_paths,
            top_k=None,
            n_workers=SoundscapeSetting.INFERENCE_WORKERS,
            n_producers=SoundscapeSetting.AUDIO_PRODUCERS,
            batch_size=SoundscapeSetting.INFERENCE_BATCH_SIZE,
            default_confidence_threshold=SoundscapeSetting.EXPORT_THRESHOLD,
            custom_species_list=[
                label
                for labels in selection.mappings[version].values()
                for label in labels
            ],
            show_stats="minimal",
        )
        result.to_csv(str(raw_path), silent=True)
        raw: DataFrame = read_csv(raw_path)
        raw[PredictionColumn.INPUT] = raw[PredictionColumn.INPUT].map(
            lambda path: Path(str(path)).name
        )
        if set(raw[PredictionColumn.INPUT]) != set(
            selection.metadata[ExperimentColumn.FILEPATH]
        ):
            raise ValueError(SoundscapeMessage.MISSING_PREDICTIONS)
        table: DataFrame = (
            raw.groupby(
                [PredictionColumn.INPUT, PredictionColumn.SPECIES_NAME]
            )[PredictionColumn.CONFIDENCE]
            .max()
            .unstack()
        )
        ordered: DataFrame = DataFrame(
            index=selection.metadata[ExperimentColumn.FILEPATH]
        )
        code: str
        for code in selection.evaluated_codes:
            labels: list[str] = selection.mappings[version][code]
            ordered[code] = (
                table.reindex(
                    index=selection.metadata[ExperimentColumn.FILEPATH],
                    columns=labels,
                )
                .fillna(0.0)
                .max(axis=1)
                if labels
                else 0.0
            )
        ordered.index.name = ExperimentColumn.FILEPATH
        write_table(ordered, score_path, index=True)
        with atomic_output(inference_path) as temporary:
            temporary.write_text(
                json.dumps(
                    {
                        ExperimentColumn.STARTED_AT: inference_started,
                        ExperimentColumn.COMPLETED_AT: datetime.now(
                            UTC
                        ).isoformat(),
                        ExperimentColumn.BINDING: binding,
                        ExperimentColumn.SCORES_SHA256: digest(score_path),
                        ExperimentColumn.WINDOWS_SHA256: digest(raw_path),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
    score_table: DataFrame = read_csv(
        score_path, index_col=ExperimentColumn.FILEPATH
    )
    if (
        score_table.index.tolist()
        != selection.metadata[ExperimentColumn.FILEPATH].tolist()
        or score_table.columns.tolist() != selection.evaluated_codes
    ):
        raise ValueError(SoundscapeMessage.INVALID_PREDICTION_ORDER)
    scores: Float64Array = score_table.to_numpy(dtype=float64)
    if (
        not isfinite(scores).all()
        or len(scores) != len(selection.truth)
        or (scores < 0).any()
        or (scores > 1).any()
    ):
        raise ValueError(SoundscapeMessage.INCOMPLETE_PREDICTIONS)
    selection.dataset_info[
        SoundscapeFile.SCORE_FINGERPRINT.format(
            dataset=dataset, version=version
        )
    ] = digest(score_path)
    selection.inference_records[version] = json.loads(
        inference_path.read_text()
    )
    return scores


def append_metrics(
    dataset: str,
    version: str,
    scores: Float64Array,
    context: SoundscapeContext,
    selection: SoundscapeSelection,
    all_metrics: list[BenchmarkMetricRow],
) -> None:
    """Append full-vocabulary and common-class results and persist their
    provenance.

    :param dataset: Stable identifier of the fixed public test subset.
    :type dataset: str
    :param version: Stable model version being evaluated.
    :type version: str
    :param scores: Model confidences in the same clip and class order.
    :type scores: Float64Array
    :param context: Models, software and dataset bindings of this run.
    :type context: SoundscapeContext
    :param selection: Selected clips, reference labels and taxonomy mappings.
    :type selection: SoundscapeSelection
    :param all_metrics: Accumulated results in their original evaluation order.
    :type all_metrics: list[BenchmarkMetricRow]
    :return: None.
    :rtype: None
    """
    row: MetricValues
    for row in evaluate(selection.truth, scores, selection.metadata):
        all_metrics.append(
            {
                ExperimentColumn.DATASET: dataset,
                MetadataKey.MODEL: version,
                ExperimentColumn.SCOPE: EvaluationScope.ALL_REFERENCE_CLASSES,
                **row,
            }
        )
    if selection.excluded:
        indices: list[int] = [
            selection.evaluated_codes.index(code) for code in selection.common
        ]
        for row in evaluate(
            selection.truth[:, indices], scores[:, indices], selection.metadata
        ):
            all_metrics.append(
                {
                    ExperimentColumn.DATASET: dataset,
                    MetadataKey.MODEL: version,
                    ExperimentColumn.SCOPE: EvaluationScope.COMMON_CLASSES,
                    **row,
                }
            )
    write_table(
        DataFrame(all_metrics), context.output / ExperimentFile.METRICS_CSV
    )
    with atomic_output(
        context.output / ExperimentFile.BENCHMARK_PROVENANCE_JSON
    ) as temporary:
        temporary.write_text(
            json.dumps(context.provenance, indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
    print(
        SoundscapeMessage.COMPLETED.format(dataset=dataset, version=version),
        flush=True,
    )


def plot_soundscapes(metrics: DataFrame, output: Path) -> None:
    """Render the F1 comparison without changing the predefined thresholds.

    :param metrics: Complete measured threshold results.
    :type metrics: DataFrame
    :param output: Destination directory for measured results and provenance.
    :type output: Path
    :return: None.
    :rtype: None
    """
    fig: Figure = ExperimentPlotter.create_figure((10, 3.6))
    target: PlotFigure = cast(PlotFigure, fig)
    target.set_layout_engine("constrained")
    axes: tuple[PlotAxis, PlotAxis] = (
        target.add_subplot(1, 2, 1),
        target.add_subplot(1, 2, 2),
    )
    for ax, dataset in zip(
        axes, (SoundscapeDataset.NBP, SoundscapeDataset.HSN), strict=True
    ):
        for version in tuple(SoundscapeVersion):
            rows: DataFrame = metrics[
                (metrics[ExperimentColumn.DATASET] == dataset)
                & (metrics[MetadataKey.MODEL] == version)
                & (
                    metrics[ExperimentColumn.SCOPE]
                    == EvaluationScope.ALL_REFERENCE_CLASSES
                )
            ]
            ax.errorbar(
                rows[ResultColumn.THRESHOLD],
                rows[MetricColumn.F1_MICRO],
                yerr=[
                    rows[MetricColumn.F1_MICRO]
                    - rows[MetricColumn.F1_MICRO_LOW],
                    rows[MetricColumn.F1_MICRO_HIGH]
                    - rows[MetricColumn.F1_MICRO],
                ],
                marker="o",
                capsize=3,
                label=ExperimentLabel.MODEL_LEGEND.format(version=version),
            )
        ax.set_title(dataset)
        ax.set_xlabel(ExperimentLabel.CONFIDENCE_THRESHOLD)
        ax.set_ylabel(ExperimentLabel.F1_MICRO)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
        ax.legend()
    ExperimentPlotter.save_figure_formats(
        fig,
        (
            output / ExperimentFile.BENCHMARK_F1_PDF,
            output / ExperimentFile.BENCHMARK_F1_PNG,
        ),
    )
    print(metrics.to_string(index=False), flush=True)


def run_soundscapes(project: ProjectPaths) -> None:
    """Run the fixed annotated experiment and save all threshold results.

    :param project: Project directories for data caches and results.
    :type project: ProjectPaths
    :return: None.
    :rtype: None
    """
    context: SoundscapeContext = prepare_soundscapes(project)
    all_metrics: list[BenchmarkMetricRow] = []
    for dataset in SoundscapeDataset:
        selection: SoundscapeSelection = select_soundscape(dataset, context)
        for version, model in context.models.items():
            scores: Float64Array = predict_soundscape(
                dataset, version, model, context, selection
            )
            append_metrics(
                dataset, version, scores, context, selection, all_metrics
            )
    metrics: DataFrame = DataFrame(all_metrics)
    plot_soundscapes(metrics, context.output)


if __name__ == "__main__":
    run_soundscapes(ProjectPaths.from_working_directory())
