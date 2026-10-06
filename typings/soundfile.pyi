"""Typed subset of SoundFile 0.14 used by this project.

Signatures follow the installed implementation and its NumPy dtype behavior.
Upstream: https://github.com/bastibe/python-soundfile/blob/0.14.0/soundfile.py
"""

from collections.abc import Generator
from os import PathLike
from types import TracebackType
from typing import Literal, Self, overload

from numpy import float32, float64, int16, int32
from numpy.typing import NDArray

type AudioPath = str | PathLike[str]
type AudioData = NDArray[float32 | float64 | int16 | int32]

class SoundFileError(Exception): ...

class SoundFile:
    def __init__(
        self,
        file: AudioPath,
        mode: str = "r",
        samplerate: int | None = None,
        channels: int | None = None,
        subtype: str | None = None,
        endian: str | None = None,
        format: str | None = None,
        closefd: bool = True,
        compression_level: float | None = None,
        bitrate_mode: str | None = None,
    ) -> None: ...
    @property
    def samplerate(self) -> int: ...
    @property
    def channels(self) -> int: ...
    @property
    def frames(self) -> int: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
    def close(self) -> None: ...
    def seek(self, frames: int, whence: int = 0) -> int: ...
    def write(self, data: AudioData) -> None: ...
    def blocks(
        self,
        blocksize: int | None = None,
        overlap: int = 0,
        frames: int = -1,
        *,
        dtype: Literal["float32"],
        always_2d: bool = False,
        fill_value: float | None = None,
        out: NDArray[float32] | None = None,
    ) -> Generator[NDArray[float32]]: ...
    @overload
    def read(
        self,
        frames: int = -1,
        *,
        dtype: Literal["float32"],
        always_2d: bool = False,
        fill_value: float | None = None,
        out: NDArray[float32] | None = None,
    ) -> NDArray[float32]: ...
    @overload
    def read(
        self,
        frames: int = -1,
        dtype: Literal["float64"] = "float64",
        always_2d: bool = False,
        fill_value: float | None = None,
        out: NDArray[float64] | None = None,
    ) -> NDArray[float64]: ...

@overload
def read(
    file: AudioPath,
    frames: int = -1,
    start: int = 0,
    stop: int | None = None,
    *,
    dtype: Literal["float32"],
    always_2d: bool = False,
) -> tuple[NDArray[float32], int]: ...
@overload
def read(
    file: AudioPath,
    frames: int = -1,
    start: int = 0,
    stop: int | None = None,
    dtype: Literal["float64"] = "float64",
    always_2d: bool = False,
) -> tuple[NDArray[float64], int]: ...
def write(
    file: AudioPath,
    data: AudioData,
    samplerate: int,
    subtype: str | None = None,
    endian: str | None = None,
    format: str | None = None,
    closefd: bool = True,
    compression_level: float | None = None,
    bitrate_mode: str | None = None,
) -> None: ...

class _SoundFileInfo:
    duration: float
    samplerate: int
    channels: int
    frames: int

def info(file: AudioPath, verbose: bool = False) -> _SoundFileInfo: ...
