"""Define notebook preview colors independently of scientific measurements."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final


@dataclass(frozen=True, slots=True)
class ThemePalette:
    """Share document and plot colors while retaining data color scales."""

    background: str
    foreground: str
    link: str
    grid: str

    def plot_settings(self) -> dict[str, str]:
        """Style plot surfaces and labels without changing series or colormaps.

        :return: Matplotlib settings scoped to preview figure generation.
        :rtype: dict[str, str]
        """
        return {
            "figure.facecolor": self.background,
            "figure.edgecolor": self.background,
            "axes.facecolor": self.background,
            "axes.edgecolor": self.foreground,
            "axes.labelcolor": self.foreground,
            "text.color": self.foreground,
            "xtick.color": self.foreground,
            "ytick.color": self.foreground,
            "grid.color": self.grid,
            "legend.facecolor": self.background,
            "legend.edgecolor": self.grid,
            "savefig.facecolor": self.background,
            "savefig.edgecolor": self.background,
            "boxplot.boxprops.color": self.foreground,
            "boxplot.whiskerprops.color": self.foreground,
            "boxplot.capprops.color": self.foreground,
        }


class ReportPalette:
    """Centralize light and dark colors for generated previews."""

    LIGHT: Final[ThemePalette] = ThemePalette(
        background="#ffffff",
        foreground="#24292f",
        link="#0969da",
        grid="#b0b0b0",
    )
    DARK: Final[ThemePalette] = ThemePalette(
        background="#0d1117",
        foreground="#e6edf3",
        link="#58a6ff",
        grid="#30363d",
    )


class ReportTheme(StrEnum):
    """Name preview variants and expose their presentation colors."""

    LIGHT = "light"
    DARK = "dark"

    @property
    def palette(self) -> ThemePalette:
        """Retrieve the colors associated with this preview variant.

        :return: Shared document and plotting palette.
        :rtype: ThemePalette
        """
        return (
            ReportPalette.LIGHT if self == self.LIGHT else ReportPalette.DARK
        )
