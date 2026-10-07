"""Bound concurrent recording analysis while yielding catalogue-order tasks."""

from __future__ import annotations

from collections import deque
from collections.abc import Generator, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from time import monotonic, sleep

from app.domain.constants import RecordingRuntime
from app.experiments.recording import RecordingMeasurements
from app.experiments.workers import RecordingWorker


@dataclass(frozen=True, slots=True)
class RecordingTask:
    """Keep a recording's resources alive until its ordered commit finishes."""

    identifier: str
    worker: RecordingWorker
    measurements: Future[RecordingMeasurements]


class RecordingPool:
    """Dispatch one recording per independent, reusable inference session.

    Threads coordinate downloads and BirdNET's own inference processes.
    Only the caller reads or writes checkpoint state and performs pairing.
    """

    def __init__(
        self,
        workers: tuple[RecordingWorker, ...],
        request_interval: float,
    ) -> None:
        """Limit temporary files and stagger recording download starts.

        :param workers: Independent recording slots owned by the caller.
        :type workers: tuple[RecordingWorker, ...]
        :param request_interval: Minimum seconds between recording launches.
        :type request_interval: float
        :return: None.
        :rtype: None
        """
        self._workers: tuple[RecordingWorker, ...] = workers
        self._interval: float = request_interval
        self._next_start: float = 0.0

    def tasks(
        self, sources: Iterator[tuple[str, dict[str, object]]]
    ) -> Generator[RecordingTask, None, None]:
        """Yield tasks in catalogue order before reusing their recording slot.

        Close this generator before releasing sessions on any interruption.
        At most one recording occupies each slot, including completed work
        that is waiting for the coordinator to commit its predecessor.

        :param sources: Pending metadata read only by the coordinator.
        :type sources: Iterator[tuple[str, dict[str, object]]]
        :return: Ordered tasks with concurrent analysis futures.
        :rtype: Generator[RecordingTask, None, None]
        """
        executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=len(self._workers),
            thread_name_prefix=RecordingRuntime.THREAD_PREFIX,
        )
        pending: deque[RecordingTask] = deque()
        complete: bool = False
        try:
            for worker in self._workers:
                self._submit(executor, worker, sources, pending)
            while pending:
                task: RecordingTask = pending.popleft()
                yield task
                self._submit(executor, task.worker, sources, pending)
            complete = True
        finally:
            if not complete:
                for worker in self._workers:
                    worker.session.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

    def _submit(
        self,
        executor: ThreadPoolExecutor,
        worker: RecordingWorker,
        sources: Iterator[tuple[str, dict[str, object]]],
        pending: deque[RecordingTask],
    ) -> None:
        """Schedule the next recording after its slot's files are released.

        :param executor: Thread pool dispatching BirdNET session operations.
        :type executor: ThreadPoolExecutor
        :param worker: Available independent recording slot.
        :type worker: RecordingWorker
        :param sources: Coordinator-owned pending metadata iterator.
        :type sources: Iterator[tuple[str, dict[str, object]]]
        :param pending: Tasks waiting for their ordered commit.
        :type pending: deque[RecordingTask]
        :return: None.
        :rtype: None
        """
        source: tuple[str, dict[str, object]] | None = next(sources, None)
        if source is None:
            return
        delay: float = max(0.0, self._next_start - monotonic())
        if delay:
            sleep(delay)
        self._next_start = monotonic() + self._interval
        identifier, metadata = source
        future: Future[RecordingMeasurements] = executor.submit(
            worker.analysis.analyze, identifier, metadata
        )
        pending.append(RecordingTask(identifier, worker, future))
