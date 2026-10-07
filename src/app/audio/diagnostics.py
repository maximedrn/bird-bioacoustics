"""Silence native decoder diagnostics while preserving application errors."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from os import close, devnull, dup, dup2, get_inheritable
from threading import RLock
from typing import BinaryIO, Final, Literal


class DecoderStream:
    """Centralize native stderr redirection and serialize decoder contexts."""

    STDERR: Final[int] = 2
    SINK_MODE: Final[Literal["wb"]] = "wb"
    LOCK: Final[RLock] = RLock()


@contextmanager
def silent_decoder_diagnostics() -> Generator[None, None, None]:
    """Hide native printouts during a synchronous decoding operation.

    Process stderr is restored before batch logging resumes, including after
    an exception or interruption. Decoder errors propagate unchanged.

    :return: Context with native decoder diagnostics redirected to devnull.
    :rtype: Generator[None, None, None]
    """
    with DecoderStream.LOCK, ExitStack() as resources:
        sink: BinaryIO = resources.enter_context(
            open(devnull, DecoderStream.SINK_MODE)
        )
        inheritable: bool = get_inheritable(DecoderStream.STDERR)
        original: int = dup(DecoderStream.STDERR)
        resources.callback(close, original)
        resources.callback(
            dup2, original, DecoderStream.STDERR, inheritable=inheritable
        )
        dup2(sink.fileno(), DecoderStream.STDERR, inheritable=inheritable)
        yield
