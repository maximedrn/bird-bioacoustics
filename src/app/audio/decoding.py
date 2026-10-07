"""Stream media rejected or unsupported by libsndfile."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

from av import AudioFrame, AudioResampler
from av import open as open_audio
from av.audio.stream import AudioStream
from av.container import InputContainer
from av.error import FFmpegError, InvalidDataError
from av.packet import Packet
from numpy import asarray, float32, isfinite
from soundfile import SoundFile

from app.domain.errors import AudioDecodingError
from app.domain.messages import ErrorMessage
from app.domain.types import FloatArray


class AudioFrameWriter:
    """Validate frames and write one continuous float32 audio stream."""

    def __init__(self, destination: Path, resources: ExitStack) -> None:
        """Share managed output resources for one recording.

        :param destination: Temporary decoded WAV destination.
        :type destination: Path
        :param resources: Context managing the container and output file.
        :type resources: ExitStack
        :return: None.
        :rtype: None
        """
        self._destination: Path = destination
        self._resources: ExitStack = resources
        self._outgoing: SoundFile | None = None
        self._resampler: AudioResampler | None = None
        self.damaged_tail: bool = False

    def write(self, frame: AudioFrame) -> None:
        """Reject internal damage and preserve the original channel layout and
        rate.

        :param frame: Decoded audio frame.
        :type frame: AudioFrame
        :return: None.
        :rtype: None
        """
        if self.damaged_tail:
            raise AudioDecodingError(ErrorMessage.DAMAGED_AUDIO_PACKETS)
        if self._outgoing is None:
            self._open(frame)
        assert self._outgoing is not None and self._resampler is not None
        if (
            frame.sample_rate != self._outgoing.samplerate
            or len(frame.layout.channels) != self._outgoing.channels
        ):
            raise AudioDecodingError(ErrorMessage.CHANGING_AUDIO_LAYOUT)
        for converted_value in self._resampler.resample(frame):
            converted: AudioFrame = converted_value
            samples: FloatArray = asarray(
                converted.to_ndarray().T, dtype=float32
            )
            if not bool(isfinite(samples).all()):
                raise AudioDecodingError(ErrorMessage.NON_FINITE_AUDIO)
            self._outgoing.write(samples)

    def _open(self, frame: AudioFrame) -> None:
        """Open RF64 output and a format-only resampler for the first frame.

        :param frame: First decoded audio frame.
        :type frame: AudioFrame
        :return: None.
        :rtype: None
        """
        self._outgoing = self._resources.enter_context(
            SoundFile(
                self._destination,
                mode="w",
                samplerate=frame.sample_rate,
                channels=len(frame.layout.channels),
                format="RF64",
                subtype="FLOAT",
            )
        )
        self._resampler = AudioResampler(
            format="fltp", layout=frame.layout.name, rate=frame.sample_rate
        )

    def validate_complete(self) -> None:
        """Reject recordings without valid decoded samples.

        :return: None.
        :rtype: None
        """
        if self._outgoing is None or self._outgoing.frames <= 0:
            raise AudioDecodingError(ErrorMessage.NO_DECODABLE_AUDIO)


def decode_with_ffmpeg(source: Path, destination: Path) -> Path:
    """Decode frames to float WAV and tolerate only trailing non-audio junk.

    :param source: Original compressed recording.
    :type source: Path
    :param destination: Temporary decoded WAV file.
    :type destination: Path
    :return: Decoded path with unchanged sample rate and channel layout.
    :rtype: Path
    """
    try:
        with ExitStack() as resources:
            container: InputContainer = resources.enter_context(
                open_audio(str(source), mode="r")
            )
            if not container.streams.audio:
                raise AudioDecodingError(ErrorMessage.MISSING_AUDIO_STREAM)
            stream: AudioStream = container.streams.audio[0]
            writer: AudioFrameWriter = AudioFrameWriter(destination, resources)
            for packet_value in container.demux(stream):
                packet: Packet[AudioStream] = packet_value
                try:
                    frames: list[AudioFrame] = packet.decode()
                except InvalidDataError:
                    writer.damaged_tail = True
                    continue
                for frame_value in frames:
                    frame: AudioFrame = frame_value
                    writer.write(frame)
            writer.validate_complete()
    except FFmpegError:
        destination.unlink(missing_ok=True)
        raise AudioDecodingError(
            ErrorMessage.DECODING_FAILED.format(name=source.name)
        ) from None
    return destination
