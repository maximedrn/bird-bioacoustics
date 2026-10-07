"""Read full-corpus diagnostic statistics and bounded scatter samples."""

from __future__ import annotations

from math import ceil, floor

from pandas import DataFrame, Series
from sqlalchemy import Integer, Select, case, cast, func, select
from sqlalchemy.engine import Connection
from sqlalchemy.orm import aliased
from sqlalchemy.sql import ColumnElement, Executable, Subquery
from sqlalchemy.sql.functions import count as sql_count
from sqlalchemy.sql.selectable import SelectBase

from app.domain.constants import NoiseCondition, ProcessingStatus, ResultColumn
from app.domain.models import ResultDetails
from app.reporting.constants import ChartSetting, DetailColumn
from app.storage.schema import (
    BatchRow,
    NoiseSummaryRow,
    OverlapRow,
    RecordingRow,
    SpeciesScoreRow,
)


class DiagnosticResults:
    """Share the report transaction without opening a second reader."""

    def __init__(self, connection: Connection) -> None:
        """Use the connection whose snapshot already contains the summaries.

        :param connection: Active report snapshot connection.
        :type connection: Connection
        :return: None.
        :rtype: None
        """
        self._connection: Connection = connection

    def _frame(
        self, statement: Executable, columns: tuple[str, ...]
    ) -> DataFrame:
        """Materialize only aggregate rows or explicitly bounded samples.

        :param statement: SQLAlchemy query for compact report data.
        :type statement: Executable
        :param columns: Output names, including for an empty query.
        :type columns: tuple[str, ...]
        :return: Stable, possibly empty diagnostic table.
        :rtype: DataFrame
        """
        records: list[dict[str, object]] = [
            dict(row) for row in self._connection.execute(statement).mappings()
        ]
        return DataFrame(records, columns=columns)

    def _sample(
        self, statement: SelectBase, partition: str | None = None
    ) -> Subquery:
        """Select evenly spaced observations in stored processing order.

        :param statement: One observation per recording and optional group.
        :type statement: SelectBase
        :param partition: Separate condition within which to bound the sample.
        :type partition: str | None
        :return: Sample query including each group's full recording count.
        :rtype: Subquery
        """
        source: Subquery = statement.subquery()
        group: ColumnElement[object] | None = (
            source.c[partition] if partition is not None else None
        )
        ranked: Subquery = select(
            source,
            func.row_number()
            .over(
                partition_by=group,
                order_by=(
                    source.c[DetailColumn.POSITION],
                    source.c[ResultColumn.RECORDING_ID],
                ),
            )
            .label(DetailColumn.SAMPLE_RANK),
            sql_count()
            .over(partition_by=group)
            .label(DetailColumn.TOTAL_RECORDINGS),
        ).subquery()
        stride: ColumnElement[int] = cast(
            (
                ranked.c[DetailColumn.TOTAL_RECORDINGS]
                + ChartSetting.SCATTER_RECORDINGS
                - 1
            )
            / ChartSetting.SCATTER_RECORDINGS,
            Integer,
        )
        return (
            select(ranked)
            .where((ranked.c[DetailColumn.SAMPLE_RANK] - 1) % stride == 0)
            .order_by(ranked.c[DetailColumn.SAMPLE_RANK])
            .subquery()
        )

    def noise_distribution(self) -> DataFrame:
        """Compute exact quartiles of equally weighted recording means.

        :return: Quartiles and observed extrema for each noise condition.
        :rtype: DataFrame
        """
        model: type[NoiseSummaryRow] = NoiseSummaryRow
        conditions: Select[str, float, int] = (
            select(model.condition, model.snr_db, sql_count())
            .group_by(model.condition, model.snr_db)
            .order_by(model.snr_db.desc())
        )
        records: list[dict[str, object]] = []
        for row in self._connection.execute(conditions):
            condition, snr_db, count = row
            values: tuple[float, ...] = self._quantiles(condition, count)
            records.append(
                {
                    ResultColumn.CONDITION: condition,
                    ResultColumn.SNR_DB: snr_db,
                    ResultColumn.RECORDINGS: count,
                    DetailColumn.MINIMUM: values[0],
                    DetailColumn.Q1: values[1],
                    DetailColumn.MEDIAN: values[2],
                    DetailColumn.Q3: values[3],
                    DetailColumn.MAXIMUM: values[4],
                }
            )
        return DataFrame(
            records,
            columns=(
                ResultColumn.CONDITION,
                ResultColumn.SNR_DB,
                ResultColumn.RECORDINGS,
                DetailColumn.MINIMUM,
                DetailColumn.Q1,
                DetailColumn.MEDIAN,
                DetailColumn.Q3,
                DetailColumn.MAXIMUM,
            ),
        )

    def _quantiles(self, condition: str, count: int) -> tuple[float, ...]:
        """Interpolate sorted quartiles while retaining at most ten values.

        :param condition: Noise condition to stream in confidence order.
        :type condition: str
        :param count: Number of independent recording means.
        :type count: int
        :return: Minimum, quartiles and maximum with linear interpolation.
        :rtype: tuple[float, ...]
        """
        positions: tuple[float, ...] = tuple(
            (count - 1) * quantile for quantile in ChartSetting.QUARTILES
        )
        indices: set[int] = {
            index
            for position in positions
            for index in (floor(position), ceil(position))
        }
        statement: Select[float] = (
            select(NoiseSummaryRow.confidence_mean)
            .where(NoiseSummaryRow.condition == condition)
            .order_by(NoiseSummaryRow.confidence_mean)
            .execution_options(yield_per=ChartSetting.STREAM_ROWS)
        )
        selected: dict[int, float] = {}
        for index, value in enumerate(self._connection.scalars(statement)):
            if index in indices:
                selected[index] = float(value)
        return tuple(
            selected[floor(position)]
            + (position - floor(position))
            * (selected[ceil(position)] - selected[floor(position)])
            for position in positions
        )

    def species_noise(self) -> DataFrame:
        """Compare the most represented noise-eligible species at each SNR.

        :return: Species detection rates weighted equally by recording.
        :rtype: DataFrame
        """
        model: type[NoiseSummaryRow] = NoiseSummaryRow
        targets: Select[str | None] = (
            select(RecordingRow.target_species)
            .join(model, model.recording_id == RecordingRow.recording_id)
            .where(
                model.condition == NoiseCondition.ORIGINAL,
                RecordingRow.target_species.is_not(None),
            )
            .group_by(RecordingRow.target_species)
            .order_by(sql_count().desc(), RecordingRow.target_species)
            .limit(ChartSetting.TOP_SPECIES)
        )
        statement: Select[str | None, str, float, float, int] = (
            select(
                RecordingRow.target_species,
                model.condition,
                model.snr_db,
                func.avg(model.detection_rate).label(
                    ResultColumn.DETECTION_RATE
                ),
                sql_count().label(ResultColumn.RECORDINGS),
            )
            .join(model, model.recording_id == RecordingRow.recording_id)
            .where(RecordingRow.target_species.in_(targets))
            .group_by(
                RecordingRow.target_species, model.condition, model.snr_db
            )
            .order_by(RecordingRow.target_species, model.snr_db.desc())
        )
        return self._frame(
            statement,
            (
                ResultColumn.TARGET_SPECIES,
                ResultColumn.CONDITION,
                ResultColumn.SNR_DB,
                ResultColumn.DETECTION_RATE,
                ResultColumn.RECORDINGS,
            ),
        )

    def noise_pairs(self) -> DataFrame:
        """Pair each noisy recording mean with its own original segment.

        :return: Bounded paired samples and full condition counts.
        :rtype: DataFrame
        """
        original: type[NoiseSummaryRow] = aliased(NoiseSummaryRow)
        noisy: type[NoiseSummaryRow] = NoiseSummaryRow
        statement: Select[str, int | None, str, float, float, float] = (
            select(
                noisy.recording_id,
                RecordingRow.position,
                noisy.condition,
                noisy.snr_db,
                original.confidence_mean.label(
                    DetailColumn.ORIGINAL_CONFIDENCE
                ),
                noisy.confidence_mean,
            )
            .join(
                original,
                (original.recording_id == noisy.recording_id)
                & (original.condition == NoiseCondition.ORIGINAL),
            )
            .join(
                RecordingRow, RecordingRow.recording_id == noisy.recording_id
            )
            .where(noisy.condition != NoiseCondition.ORIGINAL)
        )
        sample: Subquery = self._sample(statement, ResultColumn.CONDITION)
        return self._frame(
            select(sample).order_by(
                sample.c[ResultColumn.CONDITION],
                sample.c[DetailColumn.SAMPLE_RANK],
            ),
            (
                ResultColumn.RECORDING_ID,
                ResultColumn.CONDITION,
                ResultColumn.SNR_DB,
                DetailColumn.ORIGINAL_CONFIDENCE,
                ResultColumn.CONFIDENCE_MEAN,
                DetailColumn.TOTAL_RECORDINGS,
            ),
        )

    def overlap_outcomes(self, threshold: float) -> DataFrame:
        """Count joint recovery outcomes at a configured confidence threshold.

        :param threshold: Minimum decision threshold from the saved protocol.
        :type threshold: float
        :return: Both, only A, only B and neither counts for each mixture.
        :rtype: DataFrame
        """
        model: type[OverlapRow] = OverlapRow
        a: ColumnElement[bool] = model.confidence_a >= threshold
        b: ColumnElement[bool] = model.confidence_b >= threshold
        statement: Select[str, float, int, int, int, int, int] = (
            select(
                model.mix,
                model.weight_a,
                sql_count().label(ResultColumn.PAIRS),
                func.sum(case((a & b, 1), else_=0)).label(DetailColumn.BOTH),
                func.sum(case((a & ~b, 1), else_=0)).label(
                    DetailColumn.ONLY_A
                ),
                func.sum(case((~a & b, 1), else_=0)).label(
                    DetailColumn.ONLY_B
                ),
                func.sum(case((~a & ~b, 1), else_=0)).label(
                    DetailColumn.NEITHER
                ),
            )
            .group_by(model.mix, model.weight_a)
            .order_by(model.weight_a)
        )
        table: DataFrame = self._frame(
            statement,
            (
                ResultColumn.MIX,
                ResultColumn.WEIGHT_A,
                ResultColumn.PAIRS,
                DetailColumn.BOTH,
                DetailColumn.ONLY_A,
                DetailColumn.ONLY_B,
                DetailColumn.NEITHER,
            ),
        )
        table[ResultColumn.THRESHOLD] = threshold
        return table

    def species_coverage(self) -> DataFrame:
        """Count identified supported targets across all successful files.

        :return: Complete species counts ordered by representation.
        :rtype: DataFrame
        """
        statement: Select[str | None, int] = (
            select(
                RecordingRow.target_species,
                sql_count().label(ResultColumn.RECORDINGS),
            )
            .where(
                RecordingRow.status == ProcessingStatus.DONE,
                RecordingRow.target_species.is_not(None),
            )
            .group_by(RecordingRow.target_species)
            .order_by(sql_count().desc(), RecordingRow.target_species)
        )
        return self._frame(
            statement, (ResultColumn.TARGET_SPECIES, ResultColumn.RECORDINGS)
        )

    def duration_confidence(self) -> DataFrame:
        """Relate full-file target maxima to original recording duration.

        :return: Bounded samples including zero for an unexported target.
        :rtype: DataFrame
        """
        statement: Select[str, int | None, float | None, float] = (
            select(
                RecordingRow.recording_id,
                RecordingRow.position,
                RecordingRow.duration_seconds,
                func.coalesce(SpeciesScoreRow.confidence, 0.0).label(
                    DetailColumn.TARGET_CONFIDENCE
                ),
            )
            .outerjoin(
                SpeciesScoreRow,
                (SpeciesScoreRow.recording_id == RecordingRow.recording_id)
                & (
                    SpeciesScoreRow.species_name == RecordingRow.target_species
                ),
            )
            .where(
                RecordingRow.status == ProcessingStatus.DONE,
                RecordingRow.target_species.is_not(None),
                RecordingRow.duration_seconds > 0.0,
            )
        )
        sample: Subquery = self._sample(statement)
        return self._frame(
            select(sample).order_by(sample.c[DetailColumn.SAMPLE_RANK]),
            (
                ResultColumn.RECORDING_ID,
                DetailColumn.DURATION_SECONDS,
                DetailColumn.TARGET_CONFIDENCE,
                DetailColumn.TOTAL_RECORDINGS,
            ),
        )

    def batch_history(self) -> DataFrame:
        """Accumulate noise statistics in batch order, including a partial lot.

        :return: Cumulative recording means, detection rates and sample counts.
        :rtype: DataFrame
        """
        model: type[NoiseSummaryRow] = NoiseSummaryRow
        statement: Select[int, str, str, float, float, float, int] = (
            select(
                BatchRow.batch_id,
                BatchRow.status.label(DetailColumn.BATCH_STATUS),
                model.condition,
                model.snr_db,
                func.sum(model.confidence_mean).label(
                    DetailColumn.CONFIDENCE_TOTAL
                ),
                func.sum(model.detection_rate).label(
                    DetailColumn.DETECTION_TOTAL
                ),
                sql_count().label(ResultColumn.RECORDINGS),
            )
            .join(RecordingRow, RecordingRow.batch_id == BatchRow.batch_id)
            .join(model, model.recording_id == RecordingRow.recording_id)
            .where(RecordingRow.status == ProcessingStatus.DONE)
            .group_by(BatchRow.batch_id, model.condition, model.snr_db)
            .order_by(BatchRow.batch_id, model.snr_db.desc())
        )
        columns: tuple[str, ...] = (
            DetailColumn.BATCH_ID,
            DetailColumn.BATCH_STATUS,
            ResultColumn.CONDITION,
            ResultColumn.SNR_DB,
            DetailColumn.CONFIDENCE_TOTAL,
            DetailColumn.DETECTION_TOTAL,
            ResultColumn.RECORDINGS,
        )
        history: DataFrame = self._frame(statement, columns).astype(
            {
                DetailColumn.CONFIDENCE_TOTAL: float,
                DetailColumn.DETECTION_TOTAL: float,
                ResultColumn.RECORDINGS: int,
            }
        )
        totals: DataFrame = history.groupby(ResultColumn.CONDITION)[
            [
                DetailColumn.CONFIDENCE_TOTAL,
                DetailColumn.DETECTION_TOTAL,
                ResultColumn.RECORDINGS,
            ]
        ].cumsum()
        counts: Series[int] = totals[ResultColumn.RECORDINGS]
        history[ResultColumn.RECORDINGS] = counts
        history[ResultColumn.CONFIDENCE_MEAN] = (
            totals[DetailColumn.CONFIDENCE_TOTAL] / counts
        )
        history[ResultColumn.DETECTION_RATE] = (
            totals[DetailColumn.DETECTION_TOTAL] / counts
        )
        return history.drop(
            columns=[
                DetailColumn.CONFIDENCE_TOTAL,
                DetailColumn.DETECTION_TOTAL,
            ]
        )

    def build(self, threshold: float) -> ResultDetails:
        """Read all seven diagnostic tables inside the current transaction.

        :param threshold: Stored minimum decision confidence for mixtures.
        :type threshold: float
        :return: Compact diagnostics detached from the report reader.
        :rtype: ResultDetails
        """
        return ResultDetails(
            self.noise_distribution(),
            self.species_noise(),
            self.noise_pairs(),
            self.overlap_outcomes(threshold),
            self.species_coverage(),
            self.duration_confidence(),
            self.batch_history(),
        )
