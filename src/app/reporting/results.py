"""Reporting results services."""

from __future__ import annotations

from json import dumps, loads
from pathlib import Path
from sqlite3 import SQLITE_CANTOPEN
from sqlite3 import OperationalError as SqliteOperationalError
from typing import cast

from numpy import sqrt
from pandas import DataFrame
from sqlalchemy import Engine, Select, create_engine, func, select
from sqlalchemy.engine import Connection as SqlConnection
from sqlalchemy.engine import Row, RowMapping
from sqlalchemy.exc import OperationalError
from sqlalchemy.sql.functions import count as sql_count

from app.domain.constants import (
    Artifact,
    MetadataKey,
    ProcessingStatus,
    ResultColumn,
)
from app.domain.messages import ErrorMessage
from app.domain.models import NoiseExperimentOutput, ResultSnapshot
from app.domain.types import CorpusStatus, ReferenceLabels
from app.experiments.annotations import AnnotationRepository
from app.reporting.writing import atomic_output
from app.storage.connections import (
    SqliteAccess,
    database_url,
    initialize_read_journal,
    open_read_connection,
)
from app.storage.database import ExperimentDatabase
from app.storage.schema import (
    NoiseSummaryRow,
    OverlapRow,
    PreviewSegmentRow,
    RecordingRow,
    SpeciesScoreRow,
    ThresholdRow,
)


class ExperimentResults(ExperimentDatabase):
    """Retrieve stored measurements through a separate read-only SQLite
    connection.
    """

    def __init__(self, path: Path) -> None:
        """Open existing results without an API key, model or experiment
        worker.

        :param path: Existing experiment checkpoint.
        :type path: Path
        :return: None.
        :rtype: None
        """
        if not path.is_file():
            raise FileNotFoundError(ErrorMessage.MISSING_CHECKPOINT)
        self.path: Path = path
        self.engine: Engine = create_engine(
            database_url(path, SqliteAccess.READ_ONLY),
            connect_args={"autocommit": False},
        )
        self.connection: SqlConnection
        error: OperationalError
        try:
            self.connection = open_read_connection(self.engine)
        except OperationalError as error:
            self.engine.dispose()
            if not isinstance(error.orig, SqliteOperationalError):
                raise
            if error.orig.sqlite_errorcode != SQLITE_CANTOPEN:
                raise
            initialize_read_journal(path)
            self.connection = open_read_connection(self.engine)

    @staticmethod
    def standard_deviation(
        total: float, squared_total: float, count: int
    ) -> float:
        """Compute sample deviation from first and second moments.

        :param total: Sum of observations.
        :type total: float
        :param squared_total: Sum of squared observations.
        :type squared_total: float
        :param count: Number of independent observations.
        :type count: int
        :return: Sample deviation, or zero with fewer than two observations.
        :rtype: float
        """
        variance: float = (
            (squared_total - total * total / count) / (count - 1)
            if count > 1
            else 0.0
        )
        return float(sqrt(max(variance, 0.0)))

    def noise_summary(self) -> DataFrame:
        """Aggregate equally weighted recording means in bounded memory.

        :return: Noise condition means, sample deviations and detection rates.
        :rtype: DataFrame
        """
        model: type[NoiseSummaryRow] = NoiseSummaryRow
        statement: Select[
            str, float, float, float, float, float, float, int, int
        ] = (
            select(
                model.condition,
                model.snr_db,
                func.avg(model.confidence_mean).label(
                    ResultColumn.CONFIDENCE_MEAN
                ),
                func.sum(model.confidence_mean).label("total"),
                func.sum(model.confidence_mean * model.confidence_mean).label(
                    "squared_total"
                ),
                func.avg(model.confidence_std).label(
                    ResultColumn.NOISE_STD_MEAN
                ),
                func.avg(model.detection_rate).label(
                    ResultColumn.DETECTION_RATE
                ),
                sql_count().label(ResultColumn.RECORDINGS),
                func.sum(model.trials).label(ResultColumn.TRIALS),
            )
            .group_by(model.condition, model.snr_db)
            .order_by(model.snr_db.desc())
        )
        rows: list[dict[str, object]] = []
        for row_value in self.connection.execute(statement).mappings():
            row: RowMapping = row_value
            record: dict[str, object] = dict(row)
            record.pop("total")
            record.pop("squared_total")
            record[ResultColumn.CONFIDENCE_STD] = self.standard_deviation(
                float(row["total"]),
                float(row["squared_total"]),
                int(row[ResultColumn.RECORDINGS]),
            )
            rows.append(record)
        return DataFrame(
            rows,
            columns=(
                ResultColumn.CONDITION,
                ResultColumn.SNR_DB,
                ResultColumn.CONFIDENCE_MEAN,
                ResultColumn.CONFIDENCE_STD,
                ResultColumn.NOISE_STD_MEAN,
                ResultColumn.DETECTION_RATE,
                ResultColumn.RECORDINGS,
                ResultColumn.TRIALS,
            ),
        )

    def overlap_summary(self) -> DataFrame:
        """Aggregate independent pairs without loading every mixture row.

        :return: Confidence means and sample deviations for each mixture.
        :rtype: DataFrame
        """
        model: type[OverlapRow] = OverlapRow
        statement: Select[
            str, float, float, float, float, float, float, float, float, int
        ] = (
            select(
                model.mix,
                model.weight_a,
                model.weight_b,
                func.avg(model.confidence_a).label(
                    ResultColumn.CONFIDENCE_A_MEAN
                ),
                func.avg(model.confidence_b).label(
                    ResultColumn.CONFIDENCE_B_MEAN
                ),
                func.sum(model.confidence_a).label("total_a"),
                func.sum(model.confidence_b).label("total_b"),
                func.sum(model.confidence_a * model.confidence_a).label(
                    "squared_a"
                ),
                func.sum(model.confidence_b * model.confidence_b).label(
                    "squared_b"
                ),
                sql_count().label(ResultColumn.PAIRS),
            )
            .group_by(model.mix, model.weight_a, model.weight_b)
            .order_by(model.weight_a)
        )
        rows: list[dict[str, object]] = []
        for row_value in self.connection.execute(statement).mappings():
            row: RowMapping = row_value
            record: dict[str, object] = dict(row)
            for role in ("a", "b"):
                record[f"confidence_{role}_std"] = self.standard_deviation(
                    float(row[f"total_{role}"]),
                    float(row[f"squared_{role}"]),
                    int(row[ResultColumn.PAIRS]),
                )
                record.pop(f"total_{role}")
                record.pop(f"squared_{role}")
            rows.append(record)
        return DataFrame(
            rows,
            columns=(
                ResultColumn.MIX,
                ResultColumn.WEIGHT_A,
                ResultColumn.WEIGHT_B,
                ResultColumn.CONFIDENCE_A_MEAN,
                ResultColumn.CONFIDENCE_B_MEAN,
                ResultColumn.CONFIDENCE_A_STD,
                ResultColumn.CONFIDENCE_B_STD,
                ResultColumn.PAIRS,
            ),
        )

    def threshold_summary(self, thresholds: tuple[float, ...]) -> DataFrame:
        """Count retained windows and supported metadata targets across
        successful files.

        :param thresholds: Stored confidence thresholds.
        :type thresholds: tuple[float, ...]
        :return: Threshold counts; complete-reference metrics remain undefined.
        :rtype: DataFrame
        """
        threshold: float
        successful: int = int(
            self.connection.scalar(
                select(sql_count())
                .select_from(RecordingRow)
                .where(RecordingRow.status == ProcessingStatus.DONE)
            )
            or 0
        )
        eligible: int = int(
            self.connection.scalar(
                select(sql_count())
                .select_from(RecordingRow)
                .where(
                    RecordingRow.status == ProcessingStatus.DONE,
                    RecordingRow.target_species.is_not(None),
                )
            )
            or 0
        )
        rows: list[dict[str, object]] = []
        for threshold in thresholds:
            counts: Row[int, int] = self.connection.execute(
                select(
                    func.coalesce(func.sum(ThresholdRow.detections), 0),
                    func.coalesce(func.sum(ThresholdRow.target_detected), 0),
                ).where(ThresholdRow.threshold == threshold)
            ).one()
            species: int = int(
                self.connection.scalar(
                    select(
                        sql_count(func.distinct(SpeciesScoreRow.species_name))
                    ).where(SpeciesScoreRow.confidence >= threshold)
                )
                or 0
            )
            rows.append(
                {
                    ResultColumn.THRESHOLD: threshold,
                    ResultColumn.DETECTIONS: int(counts[0]),
                    ResultColumn.DISTINCT_SPECIES: species,
                    ResultColumn.RECORDINGS: successful,
                    ResultColumn.TARGET_RECORDINGS: eligible,
                    ResultColumn.TARGET_DETECTION_RATE: int(counts[1])
                    / eligible
                    if eligible
                    else None,
                    ResultColumn.ANNOTATED_RECORDINGS: 0,
                    ResultColumn.TRUE_POSITIVES: None,
                    ResultColumn.FALSE_POSITIVES: None,
                    ResultColumn.FALSE_NEGATIVES: None,
                    ResultColumn.PRECISION: None,
                    ResultColumn.RECALL: None,
                    ResultColumn.F1_SCORE: None,
                }
            )
        return DataFrame(rows)

    def reference_metrics(
        self, results: DataFrame, annotations: ReferenceLabels
    ) -> DataFrame:
        """Evaluate complete manual labels only for successfully processed
        recordings.

        :param results: Corpus threshold measurements.
        :type results: DataFrame
        :param annotations: Independently verified complete species lists.
        :type annotations: ReferenceLabels
        :return: Micro precision, recall and F1 on the annotated subset.
        :rtype: DataFrame
        """
        completed: ReferenceLabels = {}
        for identifier, labels in annotations.items():
            status: str | None = self.connection.scalar(
                select(RecordingRow.status).where(
                    RecordingRow.recording_id == identifier
                )
            )
            if status is None:
                raise ValueError(
                    ErrorMessage.ANNOTATION_OUTSIDE_CATALOGUE.format(
                        identifier=identifier
                    )
                )
            if status == ProcessingStatus.DONE:
                completed[identifier] = labels
        evaluated: DataFrame = results.copy()
        for index, threshold in evaluated[ResultColumn.THRESHOLD].items():
            tp: int = 0
            fp: int = 0
            fn: int = 0
            for identifier, reference in completed.items():
                predicted: set[str] = set(
                    self.connection.scalars(
                        select(SpeciesScoreRow.species_name).where(
                            SpeciesScoreRow.recording_id == identifier,
                            SpeciesScoreRow.confidence >= float(threshold),
                        )
                    )
                )
                tp += len(predicted.intersection(reference))
                fp += len(predicted.difference(reference))
                fn += len(reference.difference(predicted))
            if completed:
                evaluated.loc[index, ResultColumn.ANNOTATED_RECORDINGS] = len(
                    completed
                )
                evaluated.loc[
                    index,
                    [
                        ResultColumn.TRUE_POSITIVES,
                        ResultColumn.FALSE_POSITIVES,
                        ResultColumn.FALSE_NEGATIVES,
                    ],
                ] = (tp, fp, fn)
                evaluated.loc[index, ResultColumn.PRECISION] = (
                    tp / (tp + fp) if tp + fp else 0.0
                )
                evaluated.loc[index, ResultColumn.RECALL] = (
                    tp / (tp + fn) if tp + fn else 0.0
                )
                evaluated.loc[index, ResultColumn.F1_SCORE] = (
                    2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
                )
        return evaluated

    def skipped_noise(self) -> int:
        """Count successful recordings excluded from targeted noise
        measurements.

        :return: Number of explicit target-test exclusions.
        :rtype: int
        """
        return int(
            self.connection.scalar(
                select(sql_count())
                .select_from(RecordingRow)
                .where(
                    RecordingRow.status == ProcessingStatus.DONE,
                    RecordingRow.noise_skip_reason.is_not(None),
                )
            )
            or 0
        )

    def export(
        self, directory: Path, annotation_path: Path | None = None
    ) -> ResultSnapshot:
        """Retrieve a consistent database snapshot and save compact README
        data.

        :param directory: CSV and status destination.
        :type directory: Path
        :param annotation_path: Optional complete manual annotations.
        :type annotation_path: Path | None
        :return: Consistent measurements and coverage for plotting.
        :rtype: ResultSnapshot
        """
        directory.mkdir(parents=True, exist_ok=True)
        self.connection.rollback()
        # One explicit SQLite transaction fixes the snapshot for every summary.
        with self.connection.begin():
            configuration: dict[str, object] = loads(
                self.get_metadata(MetadataKey.CONFIGURATION) or "{}"
            )
            settings: dict[str, object] = cast(
                dict[str, object], configuration.get("experiment", {})
            )
            values: list[float] = cast(
                list[float], settings.get("confidence_thresholds", [])
            )
            if not values:
                raise ValueError(ErrorMessage.MISSING_STORED_THRESHOLDS)
            noise: DataFrame = self.noise_summary()
            overlap: DataFrame = self.overlap_summary()
            thresholds: DataFrame = self.threshold_summary(tuple(values))
            if annotation_path is not None:
                annotations: ReferenceLabels | None = (
                    AnnotationRepository.load(annotation_path, None)
                )
                if annotations:
                    thresholds = self.reference_metrics(
                        thresholds, annotations
                    )
            status: CorpusStatus = self.report()
            skipped: int = self.skipped_noise()
            preview: NoiseExperimentOutput | None = self.segment(
                PreviewSegmentRow, self.path.parent
            )
        for filename, dataframe in (
            (Artifact.NOISE.filename("csv"), noise),
            (Artifact.OVERLAP.filename("csv"), overlap),
            (Artifact.THRESHOLD.filename("csv"), thresholds),
        ):
            with atomic_output(directory / filename) as temporary:
                dataframe.to_csv(temporary, index=False)
        with atomic_output(
            Artifact.STATUS.path(directory, "json")
        ) as temporary:
            temporary.write_text(
                dumps(status, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        return ResultSnapshot(
            noise, overlap, thresholds, status, skipped, preview
        )
