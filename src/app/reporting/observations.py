"""Reporting observations services."""

from __future__ import annotations

from numpy import float64, isfinite
from pandas import DataFrame, Series

from app.domain.constants import NoiseCondition, ResultColumn
from app.domain.types import Float64Array
from app.reporting.messages import ObservationMessage


class NotebookReporter:
    """Generate observations from current measurements."""

    @staticmethod
    def noise(results: DataFrame, skipped: int) -> str:
        """Describe target confidence and noise variability for the current
        corpus.

        :param results: Equally weighted corpus summary.
        :type results: DataFrame
        :param skipped: Recordings without a detectable baseline target.
        :type skipped: int
        :return: English Markdown observation.
        :rtype: str
        """
        count: int = int(results[ResultColumn.RECORDINGS].max())
        original: Series[str | float | int] = results.loc[
            results[ResultColumn.CONDITION] == NoiseCondition.ORIGINAL
        ].iloc[0]
        lines: list[str] = [
            ObservationMessage.HEADING,
            (
                ObservationMessage.NOISE_BASELINE.format(
                    count=count,
                    baseline_confidence=float(
                        original[ResultColumn.CONFIDENCE_MEAN]
                    ),
                )
            ),
        ]
        noisy: DataFrame = results.loc[
            isfinite(results[ResultColumn.SNR_DB])
        ].sort_values(ResultColumn.SNR_DB, ascending=False)
        for row_value in noisy[
            [
                ResultColumn.SNR_DB,
                ResultColumn.CONFIDENCE_MEAN,
                ResultColumn.CONFIDENCE_STD,
                ResultColumn.DETECTION_RATE,
            ]
        ].to_numpy(dtype=float64):
            row: Float64Array = row_value
            lines.append(
                ObservationMessage.NOISE_LEVEL.format(
                    snr_db=float(row[0]),
                    confidence_mean=float(row[1]),
                    confidence_std=float(row[2]),
                    detection_rate=float(row[3]),
                )
            )
        lines.append(ObservationMessage.NOISE_EXPLANATION)
        if skipped:
            lines.append(
                ObservationMessage.NOISE_SKIPPED.format(skipped=skipped)
            )
        return (
            ObservationMessage.PARAGRAPH_BREAK.join(lines[:2])
            + ObservationMessage.PARAGRAPH_BREAK
            + ObservationMessage.LINE_BREAK.join(lines[2:])
        )

    @staticmethod
    def overlap(results: DataFrame) -> str:
        """Describe the two target roles across the current recording pairs.

        :param results: Mixture confidence summaries.
        :type results: DataFrame
        :return: English Markdown observation.
        :rtype: str
        """
        pair_count: int = int(results[ResultColumn.PAIRS].max())
        first: Series[str | float | int] = results.iloc[0]
        last: Series[str | float | int] = results.iloc[-1]
        return ObservationMessage.OVERLAP.format(
            pair_count=pair_count,
            first_mix=first[ResultColumn.MIX],
            last_mix=last[ResultColumn.MIX],
            first_a=float(first[ResultColumn.CONFIDENCE_A_MEAN]),
            last_a=float(last[ResultColumn.CONFIDENCE_A_MEAN]),
            first_b=float(first[ResultColumn.CONFIDENCE_B_MEAN]),
            last_b=float(last[ResultColumn.CONFIDENCE_B_MEAN]),
        )

    @staticmethod
    def threshold(results: DataFrame) -> str:
        """Describe filtering while distinguishing weak targets from reference
        metrics.

        :param results: Detection counts and optional complete-reference
            metrics.
        :type results: DataFrame
        :return: English Markdown observation.
        :rtype: str
        """
        first: Series[str | float | int] = results.iloc[0]
        last: Series[str | float | int] = results.iloc[-1]
        text: str = ObservationMessage.THRESHOLD_COUNTS.format(
            first_threshold=float(first[ResultColumn.THRESHOLD]),
            last_threshold=float(last[ResultColumn.THRESHOLD]),
            first_detections=int(first[ResultColumn.DETECTIONS]),
            last_detections=int(last[ResultColumn.DETECTIONS]),
            first_species=int(first[ResultColumn.DISTINCT_SPECIES]),
            last_species=int(last[ResultColumn.DISTINCT_SPECIES]),
        )
        if int(first[ResultColumn.TARGET_RECORDINGS]):
            text += ObservationMessage.TARGET_RECOVERY.format(
                first_rate=float(first[ResultColumn.TARGET_DETECTION_RATE]),
                last_rate=float(last[ResultColumn.TARGET_DETECTION_RATE]),
            )
        else:
            text += ObservationMessage.NO_TARGET
        if int(first[ResultColumn.ANNOTATED_RECORDINGS]):
            text += ObservationMessage.COMPLETE_ANNOTATIONS.format(
                recordings=int(first[ResultColumn.ANNOTATED_RECORDINGS])
            )
        else:
            text += ObservationMessage.NO_ANNOTATIONS
        return text
