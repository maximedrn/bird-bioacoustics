"""Audio processing services."""

from __future__ import annotations

from math import gcd
from pathlib import Path

from numpy import (
    absolute,
    amax,
    asarray,
    float32,
    float64,
    isfinite,
    mean,
    sqrt,
    square,
)
from numpy.random import Generator
from scipy.signal import resample_poly
from soundfile import SoundFile, SoundFileError, read, write

from app.audio.decoding import decode_with_ffmpeg
from app.domain.errors import AudioDecodingError
from app.domain.messages import ErrorMessage
from app.domain.settings import MixRatio
from app.domain.types import Float64Array, FloatArray


class AudioProcessor:
    """Provide typed audio processing operations used by the experiments."""

    @staticmethod
    def decode_for_inference(source: Path, destination: Path) -> Path:
        """Decode continuously to avoid unreliable seeking in compressed audio.

        :param source: Original downloaded recording.
        :type source: Path
        :param destination: Temporary float32 WAV destination.
        :type destination: Path
        :return: Decoded path with the original sample rate and channels.
        :rtype: Path
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            return AudioProcessor._decode_soundfile(source, destination)
        except SoundFileError:
            destination.unlink(missing_ok=True)
            return decode_with_ffmpeg(source, destination)

    @staticmethod
    def _decode_soundfile(source: Path, destination: Path) -> Path:
        """Read every sample sequentially using the primary decoder.

        :param source: Downloaded original recording.
        :type source: Path
        :param destination: Temporary float WAV file.
        :type destination: Path
        :return: Decoded file path.
        :rtype: Path
        """
        # RF64 supports long recordings beyond the usual WAV size limit.
        # Reading blocks bounds memory and preserves every decoded sample.
        with (
            SoundFile(source) as incoming,
            SoundFile(
                destination,
                mode="w",
                samplerate=incoming.samplerate,
                channels=incoming.channels,
                format="RF64",
                subtype="FLOAT",
            ) as outgoing,
        ):
            for block in incoming.blocks(
                blocksize=65_536, dtype="float32", always_2d=True
            ):
                samples: FloatArray = asarray(block, dtype=float32)
                if not bool(isfinite(samples).all()):
                    raise AudioDecodingError(ErrorMessage.NON_FINITE_AUDIO)
                outgoing.write(samples)
            if outgoing.frames <= 0:
                raise AudioDecodingError(ErrorMessage.EMPTY_DECODED_AUDIO)
        return destination

    @staticmethod
    def load_mono_audio(path: Path) -> tuple[FloatArray, int]:
        """Load an audio file as a mono float32 signal.

        :param path: Audio file path.
        :type path: Path
        :return: Mono samples and sample rate.
        :rtype: tuple[FloatArray, int]
        """
        read_result: tuple[FloatArray, int] = read(
            path,
            dtype="float32",
            always_2d=False,
        )
        samples_raw: FloatArray = read_result[0]
        sample_rate: int = int(read_result[1])
        samples: FloatArray = asarray(samples_raw, dtype=float32)

        mono_samples: FloatArray
        if samples.ndim == 2:
            mono_samples = asarray(
                mean(samples, axis=1),
                dtype=float32,
            )
        elif samples.ndim == 1:
            mono_samples = samples
        else:
            raise ValueError(ErrorMessage.INVALID_AUDIO_DIMENSIONS)

        if sample_rate <= 0:
            raise ValueError(ErrorMessage.INVALID_SAMPLE_RATE)

        return mono_samples, sample_rate

    @staticmethod
    def save_audio(path: Path, samples: FloatArray, sample_rate: int) -> Path:
        """Save a float32 signal as PCM-16 audio.

        :param path: Destination path.
        :type path: Path
        :param samples: Audio samples.
        :type samples: FloatArray
        :param sample_rate: Sampling rate in hertz.
        :type sample_rate: int
        :return: Saved file path.
        :rtype: Path
        """
        if samples.size == 0:
            raise ValueError(ErrorMessage.EMPTY_AUDIO_SIGNAL)
        if sample_rate <= 0:
            raise ValueError(ErrorMessage.INVALID_SAMPLE_RATE)

        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, samples, sample_rate, subtype="PCM_16")
        return path

    @staticmethod
    def extract_segment(
        samples: FloatArray,
        sample_rate: int,
        start_seconds: float,
        end_seconds: float,
    ) -> FloatArray:
        """Extract a validated audio segment.

        :param samples: Source audio samples.
        :type samples: FloatArray
        :param sample_rate: Sampling rate in hertz.
        :type sample_rate: int
        :param start_seconds: Segment start time in seconds.
        :type start_seconds: float
        :param end_seconds: Segment end time in seconds.
        :type end_seconds: float
        :return: Extracted float32 segment.
        :rtype: FloatArray
        """
        if end_seconds <= start_seconds:
            raise ValueError(ErrorMessage.INVALID_SEGMENT_INTERVAL)

        start_index: int = max(0, round(start_seconds * sample_rate))
        end_index: int = min(
            len(samples),
            round(end_seconds * sample_rate),
        )
        segment: FloatArray = samples[start_index:end_index].astype(
            float32,
            copy=True,
        )
        if segment.size == 0:
            raise ValueError(ErrorMessage.EMPTY_EXTRACTED_SEGMENT)

        return segment

    @staticmethod
    def normalize_peak(samples: FloatArray, peak: float = 0.98) -> FloatArray:
        """Limit a signal to a target absolute peak.

        :param samples: Audio samples.
        :type samples: FloatArray
        :param peak: Maximum target absolute peak.
        :type peak: float
        :return: Peak-normalized float32 signal.
        :rtype: FloatArray
        """
        if not 0.0 < peak <= 1.0:
            raise ValueError(ErrorMessage.INVALID_TARGET_PEAK)

        if samples.size == 0:
            return samples.astype(float32, copy=True)

        current_peak: float = float(amax(absolute(samples)))
        if current_peak == 0.0 or current_peak <= peak:
            return samples.astype(float32, copy=True)

        scale: float = peak / current_peak
        normalized: FloatArray = (samples * scale).astype(float32)
        return normalized

    @staticmethod
    def rms(samples: FloatArray) -> float:
        """Compute the root mean square of an audio signal.

        :param samples: Audio samples.
        :type samples: FloatArray
        :return: Root mean square value.
        :rtype: float
        """
        if samples.size == 0:
            return 0.0

        samples_float64: Float64Array = samples.astype(float64)
        squared_samples: Float64Array = square(samples_float64)
        mean_power: float = float(mean(squared_samples))
        rms_value: float = float(sqrt(mean_power))
        return rms_value

    @classmethod
    def add_gaussian_noise_at_snr(
        cls,
        samples: FloatArray,
        snr_db: float,
        generator: Generator,
    ) -> FloatArray:
        """Add Gaussian noise matching the requested signal-to-noise ratio.

        :param samples: Source audio samples.
        :type samples: FloatArray
        :param snr_db: Requested signal-to-noise ratio in decibels.
        :type snr_db: float
        :param generator: NumPy random generator.
        :type generator: Generator
        :return: Noisy float32 signal.
        :rtype: FloatArray
        """
        samples_float64: Float64Array = samples.astype(float64)
        squared_samples: Float64Array = square(samples_float64)
        signal_power: float = float(mean(squared_samples))
        if signal_power <= 0.0:
            raise ValueError(ErrorMessage.ZERO_SIGNAL_POWER)

        noise_power: float = signal_power / (10.0 ** (snr_db / 10.0))
        noise_scale: float = float(sqrt(noise_power))
        noise: Float64Array = generator.normal(
            loc=0.0,
            scale=noise_scale,
            size=samples.shape,
        )
        actual_noise_power: float = float(mean(square(noise)))
        if actual_noise_power <= 0.0:
            raise ValueError(ErrorMessage.ZERO_NOISE_POWER)
        noise = noise * sqrt(noise_power / actual_noise_power)
        noisy_float64: Float64Array = samples_float64 + noise
        noisy_float32: FloatArray = noisy_float64.astype(float32)
        normalized: FloatArray = cls.normalize_peak(noisy_float32)
        return normalized

    @staticmethod
    def resample_audio(
        samples: FloatArray,
        original_rate: int,
        target_rate: int,
    ) -> FloatArray:
        """Align sample rates before superposing recordings from different
        sources.

        :param samples: Mono audio samples.
        :type samples: FloatArray
        :param original_rate: Source sampling rate in hertz.
        :type original_rate: int
        :param target_rate: Destination sampling rate in hertz.
        :type target_rate: int
        :return: Resampled float32 audio.
        :rtype: FloatArray
        """
        if original_rate <= 0 or target_rate <= 0:
            raise ValueError(ErrorMessage.INVALID_RESAMPLING_RATES)
        if original_rate == target_rate:
            return samples.astype(float32, copy=True)
        divisor: int = gcd(original_rate, target_rate)
        resampled: FloatArray = asarray(
            resample_poly(
                samples, target_rate // divisor, original_rate // divisor
            ),
            dtype=float32,
        )
        return resampled

    @classmethod
    def mix_audio(
        cls,
        first: FloatArray,
        second: FloatArray,
        ratio: MixRatio,
    ) -> FloatArray:
        """Mix two signals with explicit amplitude coefficients.

        :param first: First audio signal.
        :type first: FloatArray
        :param second: Second audio signal.
        :type second: FloatArray
        :param ratio: Amplitude coefficients for both signals.
        :type ratio: MixRatio
        :return: Mixed float32 signal.
        :rtype: FloatArray
        """
        common_length: int = min(len(first), len(second))
        if common_length == 0:
            raise ValueError(ErrorMessage.EMPTY_MIX_SIGNALS)

        first_part: FloatArray = first[:common_length]
        second_part: FloatArray = second[:common_length]
        mixed: FloatArray = (
            (ratio.first * first_part) + (ratio.second * second_part)
        ).astype(float32)
        normalized: FloatArray = cls.normalize_peak(mixed)
        return normalized

    @staticmethod
    def load_segment(
        path: Path, start_seconds: float, end_seconds: float
    ) -> tuple[FloatArray, int]:
        """Decode only the selected audio window, keeping long recordings out
        of memory.

        :param path: Original recording path.
        :type path: Path
        :param start_seconds: Window start in seconds.
        :type start_seconds: float
        :param end_seconds: Window end in seconds.
        :type end_seconds: float
        :return: Mono samples and sample rate.
        :rtype: tuple[FloatArray, int]
        """
        with SoundFile(path) as stream:
            sample_rate: int = stream.samplerate
            start_frame: int = max(
                0, min(int(start_seconds * sample_rate), stream.frames)
            )
            stop_frame: int = max(
                start_frame, min(int(end_seconds * sample_rate), stream.frames)
            )
            stream.seek(start_frame)
            samples: FloatArray = stream.read(
                stop_frame - start_frame, dtype="float32", always_2d=True
            )
        if samples.size == 0:
            raise ValueError(ErrorMessage.EMPTY_TARGET_SEGMENT)
        return asarray(mean(samples, axis=1), dtype=float32), sample_rate
