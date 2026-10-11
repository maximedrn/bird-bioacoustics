"""Build repeatable acoustic perturbations and their isolated controls."""

from __future__ import annotations

from pathlib import Path

from numpy import abs as absolute
from numpy import asarray, float32, float64, isfinite, mean, sqrt
from numpy.random import Generator, default_rng
from pandas import DataFrame, Series
from soundfile import read, write

from app.audio.processing import AudioProcessor
from app.domain.types import Float64Array, FloatArray
from app.experiments.constants import (
    ControlColumn,
    ControlSetting,
    ExperimentColumn,
    SoundscapeSetting,
)
from app.experiments.messages import ControlledMessage


def add_background(
    signal: FloatArray, background: FloatArray, snr_db: float
) -> FloatArray:
    """Scale an actual background to the requested added-noise power ratio.

    :param signal: Foreground audio with nonzero mean power.
    :type signal: FloatArray
    :param background: Equally long background with nonzero mean power.
    :type background: FloatArray
    :param snr_db: Foreground-to-added-background ratio in decibels.
    :type snr_db: float
    :return: Peak-limited mixture preserving the requested power ratio.
    :rtype: FloatArray
    """
    if signal.shape != background.shape or not signal.size:
        raise ValueError(ControlledMessage.INVALID_SHAPES)
    foreground: Float64Array = signal.astype(float64)
    noise: Float64Array = background.astype(float64)
    foreground_power: float = float(mean(foreground**2))
    noise_power: float = float(mean(noise**2))
    if (
        not all(
            isfinite(value)
            for value in (snr_db, foreground_power, noise_power)
        )
        or min(foreground_power, noise_power) <= 0
    ):
        raise ValueError(ControlledMessage.INVALID_NOISE)
    scale: float = sqrt(foreground_power / (noise_power * 10 ** (snr_db / 10)))
    return AudioProcessor.normalize_peak(
        asarray(foreground + scale * noise, dtype=float32)
    )


def controlled_mix(
    first: FloatArray, second: FloatArray, alpha: float, beta: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Equalise source RMS and retain isolated controls at the mixture gains.

    :param first: First foreground segment.
    :type first: FloatArray
    :param second: Second foreground segment.
    :type second: FloatArray
    :param alpha: First amplitude coefficient.
    :type alpha: float
    :param beta: Second amplitude coefficient.
    :type beta: float
    :return: First alone, second alone and mixture with one shared peak gain.
    :rtype: tuple[FloatArray, FloatArray, FloatArray]
    """
    length: int = min(len(first), len(second))
    if (
        length == 0
        or not all(isfinite(value) for value in (alpha, beta))
        or min(alpha, beta) <= 0
    ):
        raise ValueError(ControlledMessage.INVALID_MIXTURE)
    a: Float64Array = first[:length].astype(float64)
    b: Float64Array = second[:length].astype(float64)
    rms_a: float = float(sqrt(mean(a**2)))
    rms_b: float = float(sqrt(mean(b**2)))
    if (
        not all(isfinite(value) for value in (rms_a, rms_b))
        or min(rms_a, rms_b) <= 0
    ):
        raise ValueError(ControlledMessage.ZERO_POWER)
    a *= ControlSetting.RMS * alpha / rms_a
    b *= ControlSetting.RMS * beta / rms_b
    peak: float = max(
        float(absolute(a).max()),
        float(absolute(b).max()),
        float(absolute(a + b).max()),
    )
    gain: float = min(1.0, 0.98 / peak)
    return (
        asarray(a * gain, dtype=float32),
        asarray(b * gain, dtype=float32),
        asarray((a + b) * gain, dtype=float32),
    )


def load_audio(path: Path) -> FloatArray:
    """Decode one whole clip into mono without a model-selected time window.

    :param path: Selected cached audio with at least three seconds of context.
    :type path: Path
    :return: Mono float32 audio resampled to the fixed comparison rate.
    :rtype: FloatArray
    """
    samples: FloatArray
    rate: int
    samples, rate = read(path, dtype="float32", always_2d=True)
    mono: FloatArray = asarray(samples.mean(axis=1), dtype=float32)
    return AudioProcessor.resample_audio(
        mono, rate, ControlSetting.SAMPLE_RATE
    )


def render_audio(
    manifest: DataFrame,
    audio_paths: dict[str, Path],
    directory: Path,
    kind: str,
) -> list[Path]:
    """Generate scoped temporary audio with reproducible perturbation draws.

    :param manifest: Exact observations and repetition parameters to generate.
    :type manifest: DataFrame
    :param audio_paths: Original foreground and background paths.
    :type audio_paths: dict[str, Path]
    :param directory: Temporary directory removed after inference.
    :type directory: Path
    :param kind: Noise perturbation or equal-RMS overlap experiment.
    :type kind: str
    :return: WAV paths in manifest order.
    :rtype: list[Path]
    """
    generator: Generator = default_rng(SoundscapeSetting.RANDOM_SEED)
    audio: dict[str, FloatArray] = {}
    name: str
    for name in sorted(
        set(manifest[ControlColumn.FIRST_FILE].astype(str))
        | (
            set(manifest[ControlColumn.SECOND_FILE].astype(str))
            if kind == ControlSetting.OVERLAP
            else set(manifest[ControlColumn.BACKGROUND].astype(str))
        )
    ):
        audio[name] = load_audio(audio_paths[name])
    paths: list[Path] = []
    row: Series[str | float | int]
    for _, row in manifest.iterrows():
        first: FloatArray = audio[str(row[ControlColumn.FIRST_FILE])]
        condition: str = str(row[ControlColumn.CONDITION])
        samples: FloatArray
        if kind == ControlSetting.OVERLAP:
            alpha: float
            beta: float
            alpha, beta = (
                float(value)
                for value in str(row[ControlColumn.RATIO]).split("/")
            )
            variants: tuple[FloatArray, FloatArray, FloatArray] = (
                controlled_mix(
                    first,
                    audio[str(row[ControlColumn.SECOND_FILE])],
                    alpha,
                    beta,
                )
            )
            samples = variants[
                (
                    ControlSetting.FIRST,
                    ControlSetting.SECOND,
                    ControlSetting.MIXED,
                ).index(condition)
            ]
        elif condition == ControlSetting.CLEAN:
            samples = first
        else:
            background: FloatArray = audio[str(row[ControlColumn.BACKGROUND])]
            if condition == ControlSetting.GAUSSIAN:
                background = asarray(
                    generator.normal(size=len(first)), dtype=float32
                )
            else:
                if len(background) < len(first):
                    raise ValueError(ControlledMessage.SHORT_BACKGROUND)
                offset: int = int(
                    generator.integers(len(background) - len(first) + 1)
                )
                background = background[offset : offset + len(first)]
            samples = add_background(
                first, background, float(row[ControlColumn.SNR])
            )
        destination: Path = directory / str(row[ExperimentColumn.FILEPATH])
        write(
            destination, samples, ControlSetting.SAMPLE_RATE, subtype="FLOAT"
        )
        paths.append(destination)
    return paths
