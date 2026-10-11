"""Compute statistics with an explicit independent sampling unit."""

from __future__ import annotations

from numpy import (
    asarray,
    bincount,
    column_stack,
    divide,
    empty,
    float64,
    isfinite,
    isin,
    ones,
    quantile,
    stack,
    unique,
    zeros,
)
from numpy.random import Generator, default_rng
from numpy.typing import ArrayLike

from app.domain.types import (
    BooleanArray,
    Float64Array,
    IntegerArray,
    MetricValues,
    StatisticalInterval,
)
from app.experiments.constants import MetricColumn, StatisticalSetting
from app.experiments.messages import StatisticalError


def bootstrap_mean(
    values: ArrayLike,
    groups: ArrayLike | None = None,
    repetitions: int = StatisticalSetting.REPETITIONS,
    seed: int = StatisticalSetting.RANDOM_SEED,
) -> StatisticalInterval:
    """Estimate a mean and its percentile bootstrap interval.

    Groups retain all dependent rows; unequal sizes use sum(x) / count(x).
    Per-species means give an explicitly species-weighted estimate.

    :param values: Finite one-dimensional independent measurements.
    :type values: ArrayLike
    :param groups: Optional source identifier for each dependent measurement.
    :type groups: ArrayLike | None
    :param repetitions: Number of independent bootstrap draws.
    :type repetitions: int
    :param seed: Reproducible generator seed.
    :type seed: int
    :return: Mean, lower bound and upper bound at 95 percent confidence.
    :rtype: StatisticalInterval
    """
    observations: Float64Array = asarray(values, dtype=float64)
    if (
        observations.ndim != 1
        or observations.size < 2
        or not isfinite(observations).all()
        or repetitions < StatisticalSetting.MINIMUM_REPETITIONS
    ):
        raise ValueError(StatisticalError.INVALID_OBSERVATIONS)
    if groups is None:
        totals: Float64Array = observations
        counts: Float64Array = ones(observations.size)
    else:
        if asarray(groups).shape != observations.shape:
            raise ValueError(StatisticalError.INVALID_GROUPS)
        inverse: IntegerArray = unique(groups, return_inverse=True)[1]
        totals = bincount(inverse, weights=observations)
        counts = asarray(bincount(inverse), dtype=float64)
    if len(totals) < 2:
        raise ValueError(StatisticalError.INSUFFICIENT_UNITS)
    generator: Generator = default_rng(seed)
    estimates: Float64Array = empty(repetitions)
    # Bound temporary memory without changing the sequence of draws.
    for start in range(0, repetitions, StatisticalSetting.CHUNK_SIZE):
        stop: int = min(start + StatisticalSetting.CHUNK_SIZE, repetitions)
        indices: IntegerArray = generator.integers(
            len(totals), size=(stop - start, len(totals))
        )
        estimates[start:stop] = totals[indices].sum(1) / counts[indices].sum(1)
    bounds: Float64Array = quantile(
        estimates, StatisticalSetting.INTERVAL_QUANTILES
    )
    return float(observations.mean()), float(bounds[0]), float(bounds[1])


def binary_matrices(
    reference: ArrayLike, predicted: ArrayLike
) -> tuple[BooleanArray, BooleanArray]:
    """Validate the binary matrices before counting clip/species outcomes.

    :param reference: Nonempty binary reference labels per clip and class.
    :type reference: ArrayLike
    :param predicted: Binary predictions in the same row and class order.
    :type predicted: ArrayLike
    :return: Boolean reference and prediction matrices with identical shapes.
    :rtype: tuple[BooleanArray, BooleanArray]
    """
    if (
        asarray(reference).ndim != 2
        or asarray(reference).shape != asarray(predicted).shape
        or asarray(reference).size == 0
        or not isin(reference, (0, 1)).all()
        or not isin(predicted, (0, 1)).all()
    ):
        raise ValueError(StatisticalError.INVALID_MATRICES)
    return asarray(reference, dtype=bool), asarray(predicted, dtype=bool)


def multilabel_metrics(
    reference: ArrayLike, predicted: ArrayLike
) -> MetricValues:
    """Count each clip/species pair once, including negative clips.

    Macro F1 uses classes with reference positives; all evaluated classes
    contribute micro false positives. Undefined ratios are reported as zero.

    :param reference: Binary labels in the evaluated reference vocabulary.
    :type reference: ArrayLike
    :param predicted: Binary predictions in the same clip and class order.
    :type predicted: ArrayLike
    :return: Micro counts, precision, recall, F1 and negative-clip outcomes.
    :rtype: MetricValues
    """
    truth, guesses = binary_matrices(reference, predicted)
    tp_class: IntegerArray = (truth & guesses).sum(0)
    fp_class: IntegerArray = (~truth & guesses).sum(0)
    fn_class: IntegerArray = (truth & ~guesses).sum(0)
    tp: int = int(tp_class.sum())
    fp: int = int(fp_class.sum())
    fn: int = int(fn_class.sum())
    represented: BooleanArray = truth.sum(0) > 0
    denominator: IntegerArray = 2 * tp_class + fp_class + fn_class
    per_class_f1: Float64Array = divide(
        2 * tp_class,
        denominator,
        out=zeros(len(tp_class)),
        where=denominator > 0,
    )
    negative: BooleanArray = ~truth.any(1)
    return {
        MetricColumn.TP: tp,
        MetricColumn.FP: fp,
        MetricColumn.FN: fn,
        MetricColumn.PRECISION_MICRO: tp / (tp + fp) if tp + fp else 0.0,
        MetricColumn.RECALL_MICRO: tp / (tp + fn) if tp + fn else 0.0,
        MetricColumn.F1_MICRO: (
            2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        ),
        MetricColumn.F1_MACRO: (
            float(per_class_f1[represented].mean())
            if represented.any()
            else 0.0
        ),
        MetricColumn.SUPPORTED_CLASSES: int(represented.sum()),
        MetricColumn.NEGATIVE_CLIPS: int(negative.sum()),
        MetricColumn.NEGATIVE_CLIP_FALSE_ALARM_RATE: (
            float(guesses[negative].any(1).mean()) if negative.any() else 0.0
        ),
    }


def grouped_outcome_counts(
    truth: BooleanArray, predictions: BooleanArray, groups: IntegerArray
) -> IntegerArray:
    """Aggregate TP, FP and FN while keeping each source's rows together.

    :param truth: Reference clip/class matrix.
    :type truth: BooleanArray
    :param predictions: Predictions in the same clip/class order.
    :type predictions: BooleanArray
    :param groups: Contiguous zero-based source index for each clip.
    :type groups: IntegerArray
    :return: One row of TP, FP and FN counts per source.
    :rtype: IntegerArray
    """
    rows: IntegerArray = column_stack(
        (
            (truth & predictions).sum(1),
            (~truth & predictions).sum(1),
            (truth & ~predictions).sum(1),
        )
    )
    return stack(
        [
            rows[groups == group].sum(0)
            for group in range(int(groups.max()) + 1)
        ]
    )


def resampled_f1(counts: IntegerArray, indices: IntegerArray) -> Float64Array:
    """Compute micro F1 for each shared draw of source-level outcome counts.

    :param counts: Source-level TP, FP and FN columns.
    :type counts: IntegerArray
    :param indices: Source indices selected by each bootstrap draw.
    :type indices: IntegerArray
    :return: F1 distribution with zero for a draw with no positive outcome.
    :rtype: Float64Array
    """
    totals: IntegerArray = counts[indices].sum(1)
    denominator: IntegerArray = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
    return divide(
        2 * totals[:, 0],
        denominator,
        out=zeros(len(indices)),
        where=denominator > 0,
    )


def bootstrap_f1_difference(
    reference: ArrayLike,
    predicted_a: ArrayLike,
    predicted_b: ArrayLike,
    groups: ArrayLike,
    repetitions: int = StatisticalSetting.REPETITIONS,
    seed: int = StatisticalSetting.RANDOM_SEED,
) -> StatisticalInterval:
    """Estimate F1(A)-F1(B) with paired resampling of source files.

    :param reference: Binary reference labels per clip and class.
    :type reference: ArrayLike
    :param predicted_a: Binary predictions of the first model.
    :type predicted_a: ArrayLike
    :param predicted_b: Binary predictions of the second model.
    :type predicted_b: ArrayLike
    :param groups: Source identifier for each clip.
    :type groups: ArrayLike
    :param repetitions: Number of shared source-level bootstrap draws.
    :type repetitions: int
    :param seed: Reproducible generator seed shared by both models.
    :type seed: int
    :return: F1 difference and its lower and upper 95 percent bounds.
    :rtype: StatisticalInterval
    """
    first: MetricValues = multilabel_metrics(reference, predicted_a)
    second: MetricValues = multilabel_metrics(reference, predicted_b)
    truth: BooleanArray = asarray(reference, dtype=bool)
    if (
        asarray(groups).shape != (len(truth),)
        or repetitions < StatisticalSetting.MINIMUM_REPETITIONS
    ):
        raise ValueError(StatisticalError.INVALID_CLIP_GROUPS)
    inverse: IntegerArray = unique(groups, return_inverse=True)[1]
    group_count: int = int(inverse.max()) + 1
    if group_count < 2:
        raise ValueError(StatisticalError.INSUFFICIENT_SOURCES)
    generator: Generator = default_rng(seed)
    indices: IntegerArray = generator.integers(
        group_count, size=(repetitions, group_count)
    )
    distributions: list[Float64Array] = []
    predicted: ArrayLike
    for predicted in (predicted_a, predicted_b):
        guesses: BooleanArray = asarray(predicted, dtype=bool)
        counts: IntegerArray = grouped_outcome_counts(truth, guesses, inverse)
        distributions.append(resampled_f1(counts, indices))
    bounds: Float64Array = quantile(
        distributions[0] - distributions[1],
        StatisticalSetting.INTERVAL_QUANTILES,
    )
    return (
        float(first[MetricColumn.F1_MICRO] - second[MetricColumn.F1_MICRO]),
        float(bounds[0]),
        float(bounds[1]),
    )
