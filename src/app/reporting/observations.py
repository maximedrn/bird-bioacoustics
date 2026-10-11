"""Simple observations attached to the measured experiment results."""

from __future__ import annotations

from pandas import DataFrame, Series

from app.domain.constants import NoiseCondition, ResultColumn
from app.reporting.messages import ObservationMessage


class NotebookReporter:
    """Describe the main result of each experiment in two sentences."""

    @staticmethod
    def noise(results: DataFrame, skipped: int) -> str:
        """Describe the strongest measured noise effect.

        :param results: Saved cumulative experiment measurements.
        :type results: DataFrame
        :param skipped: Number of files excluded from the noise experiment.
        :type skipped: int
        :return: Formatted description of the saved experiment.
        :rtype: str
        """
        original: Series[str | float | int] = results.loc[
            results[ResultColumn.CONDITION] == NoiseCondition.ORIGINAL
        ].iloc[0]
        noisy: Series[str | float | int] = results.sort_values(
            ResultColumn.SNR_DB
        ).iloc[0]
        return ObservationMessage.NOISE.format(
            original=float(original[ResultColumn.CONFIDENCE_MEAN]),
            noisy=float(noisy[ResultColumn.CONFIDENCE_MEAN]),
            skipped=skipped,
        )

    @staticmethod
    def overlap(results: DataFrame) -> str:
        """Describe the weaker target in the most unbalanced mixture.

        :param results: Saved cumulative experiment measurements.
        :type results: DataFrame
        :return: Formatted description of the saved experiment.
        :rtype: str
        """
        first: Series[str | float | int] = results.iloc[0]
        last: Series[str | float | int] = results.iloc[-1]
        return ObservationMessage.OVERLAP.format(
            first=float(first[ResultColumn.CONFIDENCE_B_MEAN]),
            last=float(last[ResultColumn.CONFIDENCE_B_MEAN]),
        )

    @staticmethod
    def threshold(results: DataFrame) -> str:
        """Describe filtering without equating output counts with truth.

        :param results: Saved cumulative experiment measurements.
        :type results: DataFrame
        :return: Formatted description of the saved experiment.
        :rtype: str
        """
        first: Series[str | float | int] = results.iloc[0]
        last: Series[str | float | int] = results.iloc[-1]
        return ObservationMessage.THRESHOLD.format(
            first_threshold=float(first[ResultColumn.THRESHOLD]),
            last_threshold=float(last[ResultColumn.THRESHOLD]),
            first_detections=int(first[ResultColumn.DETECTIONS]),
            last_detections=int(last[ResultColumn.DETECTIONS]),
        )
