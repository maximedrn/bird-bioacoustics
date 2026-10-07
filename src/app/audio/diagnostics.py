"""Silence synchronous native diagnostics while preserving exceptions."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from os import close, devnull, dup, dup2, get_inheritable
from threading import RLock
from typing import BinaryIO, Final, Literal


class NativeStream:
    """Centralize native stderr redirection and serialize library contexts."""

    STDERR: Final[int] = 2
    SINK_MODE: Final[Literal["wb"]] = "wb"
    LOCK: Final[RLock] = RLock()


@contextmanager
def silent_native_diagnostics() -> Generator[None, None, None]:
    """Hide native printouts during a synchronous library operation.

    Process stderr is restored before batch logging resumes, including after
    an exception or interruption. Python exceptions propagate unchanged.

    :return: Context with native stderr redirected to devnull.
    :rtype: Generator[None, None, None]
    """
    with NativeStream.LOCK, ExitStack() as resources:
        sink: BinaryIO = resources.enter_context(
            open(devnull, NativeStream.SINK_MODE)
        )
        inheritable: bool = get_inheritable(NativeStream.STDERR)
        original: int = dup(NativeStream.STDERR)
        resources.callback(close, original)
        resources.callback(
            dup2, original, NativeStream.STDERR, inheritable=inheritable
        )
        dup2(sink.fileno(), NativeStream.STDERR, inheritable=inheritable)
        yield
