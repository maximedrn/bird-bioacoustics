"""Audio prediction services."""

from __future__ import annotations

from numbers import Real
from pathlib import Path
from typing import ClassVar

from pandas import DataFrame, Series, read_csv, to_numeric

from app.domain.constants import PredictionColumn
from app.domain.errors import BirdNetInferenceError
from app.domain.messages import ErrorMessage
from app.domain.models import DetectionCandidate
from app.domain.protocols import (
    BirdNetModelProtocol,
    CsvWritable,
    PredictionSession,
)
from app.domain.settings import ModelSettings, ProjectPaths


class BirdNetPredictor:
    """Provide validated and normalized access to BirdNET predictions."""

    REQUIRED_COLUMNS: ClassVar[frozenset[str]] = frozenset(
        {
            PredictionColumn.INPUT,
            PredictionColumn.START_TIME,
            PredictionColumn.END_TIME,
            PredictionColumn.SPECIES_NAME,
            PredictionColumn.CONFIDENCE,
        }
    )

    def __init__(
        self,
        model: BirdNetModelProtocol,
        paths: ProjectPaths,
        settings: ModelSettings | None = None,
        session: PredictionSession | None = None,
    ) -> None:
        """Initialize the predictor.

        :param model: Loaded BirdNET model.
        :type model: BirdNetModelProtocol
        :param paths: Project path configuration.
        :type paths: ProjectPaths
        :param settings: Minimum exported score and model configuration.
        :type settings: ModelSettings | None
        :param session: Optional reusable inference session.
        :type session: PredictionSession | None
        :return: None.
        :rtype: None
        """
        self._settings: ModelSettings = (
            settings if settings is not None else ModelSettings()
        )
        self._session: PredictionSession | None = session
        self._model: BirdNetModelProtocol = model
        self._paths: ProjectPaths = paths

    def predict(self, audio_path: Path, output_stem: str) -> DataFrame:
        """Run BirdNET and return a validated prediction table.

        :param audio_path: Input audio path.
        :type audio_path: Path
        :param output_stem: Output CSV filename without extension.
        :type output_stem: str
        :return: Cleaned BirdNET predictions.
        :rtype: DataFrame
        """
        return self.predict_many((audio_path,), output_stem)

    def predict_many(
        self, audio_paths: tuple[Path, ...], output_stem: str
    ) -> DataFrame:
        """Infer several files without truncating their species predictions.

        :param audio_paths: Distinct input audio paths.
        :type audio_paths: tuple[Path, ...]
        :param output_stem: Output CSV filename without extension.
        :type output_stem: str
        :return: Cleaned predictions with portable input filenames.
        :rtype: DataFrame
        """
        if not audio_paths:
            raise ValueError(ErrorMessage.EMPTY_PREDICTION_INPUTS)
        if len({path_value.name for path_value in audio_paths}) != len(
            audio_paths
        ):
            raise ValueError(ErrorMessage.DUPLICATE_AUDIO_FILENAMES)
        for path_value in audio_paths:
            audio_path: Path = path_value
            if not audio_path.exists():
                raise FileNotFoundError(
                    ErrorMessage.MISSING_AUDIO_FILE.format(
                        audio_path=audio_path
                    )
                )
        if not output_stem.strip():
            raise ValueError(ErrorMessage.EMPTY_OUTPUT_STEM)

        output_csv: Path = self._paths.results / f"{output_stem}.csv"
        model_inputs: tuple[str, ...] = tuple(
            str(path_value) for path_value in audio_paths
        )
        try:
            raw_predictions: CsvWritable = (
                self._session.run(model_inputs)
                if self._session is not None
                else self._model.predict(
                    model_inputs,
                    top_k=None,
                    n_workers=self._settings.n_workers,
                    default_confidence_threshold=self._settings.minimum_confidence,
                )
            )
        except (RuntimeError, ChildProcessError) as error:
            if self._session is not None:
                self._session.cancel()
            raise BirdNetInferenceError(str(error)) from error
        except KeyboardInterrupt:
            if self._session is not None:
                self._session.cancel()
            raise
        raw_predictions.to_csv(str(output_csv), silent=True)

        dataframe: DataFrame = read_csv(output_csv)
        available_columns: set[str] = {
            str(column_name) for column_name in dataframe.columns
        }
        missing_columns: frozenset[str] = frozenset(
            self.REQUIRED_COLUMNS.difference(available_columns)
        )
        if missing_columns:
            missing_text: str = ", ".join(sorted(missing_columns))
            raise ValueError(
                ErrorMessage.MISSING_PREDICTION_COLUMNS.format(
                    missing_text=missing_text
                )
            )

        confidence_values: Series[float] = to_numeric(
            dataframe[PredictionColumn.CONFIDENCE],
            errors="coerce",
        )
        dataframe[PredictionColumn.CONFIDENCE] = confidence_values
        cleaned_dataframe: DataFrame = dataframe.dropna(
            subset=[PredictionColumn.CONFIDENCE]
        ).copy()

        if PredictionColumn.INPUT in cleaned_dataframe.columns:
            input_values: Series[str] = cleaned_dataframe[
                PredictionColumn.INPUT
            ].astype(str)
            portable_inputs: Series[str] = input_values.map(self._basename)
            cleaned_dataframe[PredictionColumn.INPUT] = portable_inputs

        cleaned_dataframe.to_csv(output_csv, index=False)
        return cleaned_dataframe

    @staticmethod
    def _basename(value: object) -> str:
        """Return the basename of an arbitrary path-like value.

        :param value: Value containing a path.
        :type value: object
        :return: Basename only.
        :rtype: str
        """
        path_value: Path = Path(str(value))
        return path_value.name

    @staticmethod
    def parse_time_to_seconds(value: object) -> float:
        """Convert a BirdNET time value to seconds.

        :param value: Numeric seconds or ``HH:MM:SS`` text.
        :type value: object
        :return: Time in seconds.
        :rtype: float
        """
        if isinstance(value, Real):
            numeric_value: float = float(value)
            return numeric_value

        text_value: str = str(value)
        parts: list[str] = text_value.split(":")
        if len(parts) == 3:
            hours: float = float(parts[0])
            minutes: float = float(parts[1])
            seconds: float = float(parts[2])
            return (hours * 3600.0) + (minutes * 60.0) + seconds

        return float(text_value)

    @classmethod
    def candidate_from_row(
        cls, row: Series[str | float]
    ) -> DetectionCandidate:
        """Create a validated detection candidate from one prediction row.

        :param row: BirdNET prediction row.
        :type row: Series
        :return: Validated detection candidate.
        :rtype: DetectionCandidate
        """
        candidate: DetectionCandidate = DetectionCandidate(
            species_name=str(row[PredictionColumn.SPECIES_NAME]),
            confidence=float(row[PredictionColumn.CONFIDENCE]),
            start_seconds=cls.parse_time_to_seconds(
                row[PredictionColumn.START_TIME]
            ),
            end_seconds=cls.parse_time_to_seconds(
                row[PredictionColumn.END_TIME]
            ),
        )
        return candidate

    @classmethod
    def candidate_for_species(
        cls,
        predictions: DataFrame,
        species_name: str,
    ) -> DetectionCandidate | None:
        """Select the most confident detection of the metadata target.

        :param predictions: Baseline predictions for one recording.
        :type predictions: DataFrame
        :param species_name: Target species supplied by the recording metadata.
        :type species_name: str
        :return: Target candidate, or None when the target is absent.
        :rtype: DetectionCandidate | None
        """
        selected: DataFrame = predictions.loc[
            predictions[PredictionColumn.SPECIES_NAME] == species_name
        ].sort_values(PredictionColumn.CONFIDENCE, ascending=False)
        if selected.empty:
            return None
        return cls.candidate_from_row(selected.iloc[0])

    @staticmethod
    def confidence_for_species(
        predictions: DataFrame,
        species_name: str,
    ) -> float:
        """Return the maximum confidence for one species.

        :param predictions: BirdNET prediction table.
        :type predictions: DataFrame
        :param species_name: Target species name.
        :type species_name: str
        :return: Maximum confidence, or zero when absent.
        :rtype: float
        """
        selected: Series[float] = predictions.loc[
            predictions[PredictionColumn.SPECIES_NAME] == species_name,
            PredictionColumn.CONFIDENCE,
        ]
        if selected.empty:
            return 0.0

        maximum_confidence: float = float(selected.max())
        return maximum_confidence

    @staticmethod
    def rank_for_species(
        predictions: DataFrame,
        species_name: str,
    ) -> int | None:
        """Return the one-based rank of one species by confidence.

        :param predictions: BirdNET prediction table.
        :type predictions: DataFrame
        :param species_name: Target species name.
        :type species_name: str
        :return: One-based rank, or ``None`` when absent.
        :rtype: int | None
        """
        ordered: DataFrame = predictions.sort_values(
            PredictionColumn.CONFIDENCE,
            ascending=False,
        ).reset_index(drop=True)
        matching_mask: Series[bool] = (
            ordered[PredictionColumn.SPECIES_NAME] == species_name
        )
        matching_indices: list[int] = [
            int(index_value) for index_value in ordered.index[matching_mask]
        ]
        if not matching_indices:
            return None

        rank: int = matching_indices[0] + 1
        return rank
