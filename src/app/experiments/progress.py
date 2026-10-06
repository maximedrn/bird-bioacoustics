"""Present durable recording-batch progress in terminals and notebooks."""

from __future__ import annotations

from typing import Never, Protocol

from tqdm.auto import tqdm as create_progress
from tqdm.std import tqdm as ProgressBar

from app.domain.constants import ProcessingStatus
from app.domain.messages import ErrorMessage, ProgressMessage
from app.domain.models import RecordingBatch


class ProgressObserver(Protocol):
    """Receive progress only after recording outcomes have committed."""

    def start(
        self, batch: RecordingBatch, cursor: int, counts: dict[str, int]
    ) -> None:
        """Initialize the display at the saved cursor.

        :param batch: Current recording interval.
        :type batch: RecordingBatch
        :param cursor: Committed absolute cursor.
        :type cursor: int
        :param counts: Outcomes already stored in this batch.
        :type counts: dict[str, int]
        :return: None.
        :rtype: None
        """
        ...

    def update(
        self, cursor: int, identifier: str, status: ProcessingStatus
    ) -> None:
        """Display one committed recording outcome.

        :param cursor: New committed absolute cursor.
        :type cursor: int
        :param identifier: Just-attempted XC identifier.
        :type identifier: str
        :param status: Committed recording status.
        :type status: ProcessingStatus
        :return: None.
        :rtype: None
        """
        ...

    def close(self) -> None:
        """Close progress output after completion or interruption.

        :return: None.
        :rtype: None
        """
        ...


class TqdmBatchProgress:
    """Adapt durable batch cursor changes to a tqdm progress bar."""

    def __init__(self, enabled: bool = True) -> None:
        """Configure whether progress is visible.

        :param enabled: Display the progress bar.
        :type enabled: bool
        :return: None.
        :rtype: None
        """
        self._enabled: bool = enabled
        self._bar: ProgressBar[Never] | None = None
        self._start: int = 0
        self._counts: dict[str, int] = {}

    def start(
        self, batch: RecordingBatch, cursor: int, counts: dict[str, int]
    ) -> None:
        """Open a bar whose total is the actual saved batch size.

        :param batch: Current saved batch interval.
        :type batch: RecordingBatch
        :param cursor: Committed absolute cursor.
        :type cursor: int
        :param counts: Stored batch outcomes.
        :type counts: dict[str, int]
        :return: None.
        :rtype: None
        """

        self._start = batch.start_cursor
        self._counts = dict(counts)
        self._bar = create_progress(
            total=batch.size,
            initial=cursor - self._start,
            desc=ProgressMessage.BATCH.format(batch_id=batch.batch_id),
            unit=ProgressMessage.UNIT,
            dynamic_ncols=True,
            disable=not self._enabled,
        )
        self._bar.set_postfix_str(self._postfix())

    def update(
        self, cursor: int, identifier: str, status: ProcessingStatus
    ) -> None:
        """Advance to the committed cursor and update outcome counters.

        :param cursor: Committed cursor after this recording.
        :type cursor: int
        :param identifier: Current recording identifier.
        :type identifier: str
        :param status: Stored outcome.
        :type status: ProcessingStatus
        :return: None.
        :rtype: None
        """
        if self._bar is None:
            raise RuntimeError(ErrorMessage.PROGRESS_NOT_STARTED)
        self._counts[status] = self._counts.get(status, 0) + 1
        if self._enabled:
            self._bar.update(cursor - self._start - self._bar.n)
            self._bar.set_postfix_str(self._postfix(identifier))

    def _postfix(self, identifier: str | None = None) -> str:
        """Format explicit saved outcome counts for the progress display.

        :param identifier: Current recording identifier, if available.
        :type identifier: str | None
        :return: Progress suffix.
        :rtype: str
        """
        counts: str = ProgressMessage.COUNTS.format(
            done=self._counts.get(ProcessingStatus.DONE, 0),
            failed=self._counts.get(ProcessingStatus.FAILED, 0),
        )
        if identifier is None:
            return counts
        return ProgressMessage.CURRENT_RECORDING.format(
            counts=counts, identifier=identifier
        )

    def close(self) -> None:
        """Close an opened bar without completing uncommitted work.

        :return: None.
        :rtype: None
        """
        if self._bar is not None:
            self._bar.close()
            self._bar = None
