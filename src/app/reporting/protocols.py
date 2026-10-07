"""Precise contracts for the Matplotlib operations used by scientific
reporting.

Matplotlib supplies inline types, but some plotting keyword signatures remain
incomplete. These interfaces describe the project's supported calls rather than
suppressing diagnostics for whole modules.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol

from matplotlib.collections import QuadMesh
from matplotlib.colorbar import Colorbar
from matplotlib.container import BarContainer, ErrorbarContainer
from matplotlib.image import AxesImage
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.text import Text
from numpy.typing import ArrayLike


class PlotAxis(Protocol):
    """Declare the actual axis operations and named style options we use."""

    def bar(
        self,
        x: Sequence[str],
        height: ArrayLike,
        *,
        color: Sequence[str] | str,
        bottom: ArrayLike | None = None,
        label: str | None = None,
    ) -> BarContainer:
        """Draw labelled counts.

        :param x: Category labels.
        :type x: Sequence[str]
        :param height: Category counts.
        :type height: ArrayLike
        :param color: Category colors.
        :type color: Sequence[str] | str
        :param bottom: Optional stack offsets.
        :type bottom: ArrayLike | None
        :param label: Optional series name.
        :type label: str | None
        :return: Bars available for annotation.
        :rtype: BarContainer
        """
        raise NotImplementedError

    def bar_label(
        self, container: BarContainer, *, labels: Sequence[str], padding: float
    ) -> list[Text]:
        """Annotate category counts.

        :param container: Rendered bars.
        :type container: BarContainer
        :param labels: Formatted counts.
        :type labels: Sequence[str]
        :param padding: Label spacing.
        :type padding: float
        :return: Count labels.
        :rtype: list[Text]
        """
        raise NotImplementedError

    def plot(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        marker: str = "",
        label: str,
        linestyle: str = "-",
    ) -> list[Line2D]:
        """Draw a named data series.

        :param x: Horizontal values.
        :type x: ArrayLike
        :param y: Vertical values.
        :type y: ArrayLike
        :param marker: Point marker.
        :type marker: str
        :param label: Legend label.
        :type label: str
        :param linestyle: Stroke style.
        :type linestyle: str
        :return: Rendered lines.
        :rtype: list[Line2D]
        """
        raise NotImplementedError

    def errorbar(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        yerr: ArrayLike,
        marker: str,
        capsize: float,
        label: str,
    ) -> ErrorbarContainer:
        """Draw means and dispersion.

        :param x: Conditions.
        :type x: ArrayLike
        :param y: Means.
        :type y: ArrayLike
        :param yerr: Standard deviations.
        :type yerr: ArrayLike
        :param marker: Point marker.
        :type marker: str
        :param capsize: Error-bar cap length.
        :type capsize: float
        :param label: Legend label.
        :type label: str
        :return: Rendered mean and deviation artists.
        :rtype: ErrorbarContainer
        """
        raise NotImplementedError

    def axhline(
        self, y: float, *, linestyle: str, linewidth: float, label: str
    ) -> Line2D:
        """Draw a reference value.

        :param y: Reference value.
        :type y: float
        :param linestyle: Line style.
        :type linestyle: str
        :param linewidth: Stroke width.
        :type linewidth: float
        :param label: Legend label.
        :type label: str
        :return: Reference line.
        :rtype: Line2D
        """
        raise NotImplementedError

    def set_title(self, label: str) -> Text:
        """Set the panel title.

        :param label: Panel title.
        :type label: str
        :return: Title artist.
        :rtype: Text
        """
        raise NotImplementedError

    def set_xlabel(self, xlabel: str) -> Text:
        """Set the horizontal label.

        :param xlabel: Axis label.
        :type xlabel: str
        :return: Label artist.
        :rtype: Text
        """
        raise NotImplementedError

    def set_ylabel(self, ylabel: str) -> Text:
        """Set the vertical label.

        :param ylabel: Axis label.
        :type ylabel: str
        :return: Label artist.
        :rtype: Text
        """
        raise NotImplementedError

    def set_ylim(self, bottom: float, top: float) -> tuple[float, float]:
        """Set display limits.

        :param bottom: Minimum ordinate.
        :type bottom: float
        :param top: Maximum ordinate.
        :type top: float
        :return: Effective bounds.
        :rtype: tuple[float, float]
        """
        raise NotImplementedError

    def ticklabel_format(
        self,
        *,
        axis: Literal["x", "y", "both"],
        style: Literal["plain", "sci", "scientific"],
    ) -> None:
        """Choose numeric tick formatting.

        :param axis: Axes to format.
        :type axis: Literal["x", "y", "both"]
        :param style: Numeric style.
        :type style: Literal["plain", "sci", "scientific"]
        :return: None.
        :rtype: None
        """
        raise NotImplementedError

    def grid(
        self,
        visible: bool | None = None,
        *,
        axis: Literal["x", "y", "both"] = "both",
        alpha: float,
    ) -> None:
        """Display the scientific grid.

        :param visible: Grid visibility.
        :type visible: bool | None
        :param axis: Grid axes.
        :type axis: Literal["x", "y", "both"]
        :param alpha: Grid opacity.
        :type alpha: float
        :return: None.
        :rtype: None
        """
        raise NotImplementedError

    def set_axisbelow(self, b: bool) -> None:
        """Place grids behind measurements.

        :param b: Whether grids belong below artists.
        :type b: bool
        :return: None.
        :rtype: None
        """
        raise NotImplementedError

    def legend(
        self,
        *,
        fontsize: float = 10,
        loc: str = "best",
        bbox_to_anchor: tuple[float, float] | None = None,
        ncol: int = 1,
    ) -> Legend:
        """Display named measurements.

        :param fontsize: Legend text size.
        :type fontsize: float
        :param loc: Anchor location.
        :type loc: str
        :param bbox_to_anchor: Optional explicit anchor coordinates.
        :type bbox_to_anchor: tuple[float, float] | None
        :param ncol: Number of legend columns.
        :type ncol: int
        :return: Legend artist.
        :rtype: Legend
        """
        raise NotImplementedError

    def pcolormesh(
        self,
        x: ArrayLike,
        y: ArrayLike,
        c: ArrayLike,
        *,
        shading: Literal["auto"],
    ) -> QuadMesh:
        """Render a spectrogram matrix.

        :param x: Times.
        :type x: ArrayLike
        :param y: Frequencies.
        :type y: ArrayLike
        :param c: Power matrix.
        :type c: ArrayLike
        :param shading: Cell interpolation choice.
        :type shading: Literal["auto"]
        :return: Spectrogram artist.
        :rtype: QuadMesh
        """
        raise NotImplementedError


class PlotFigure(Protocol):
    """Declare the figure calls with explicit supported keyword types."""

    def add_subplot(self, nrows: int, ncols: int, index: int) -> PlotAxis:
        """Create one panel.

        :param nrows: Grid height.
        :type nrows: int
        :param ncols: Grid width.
        :type ncols: int
        :param index: One-based panel index.
        :type index: int
        :return: Plot axis.
        :rtype: PlotAxis
        """
        raise NotImplementedError

    def savefig(
        self, fname: Path, *, dpi: float, format: Literal["png"]
    ) -> None:
        """Write an explicitly configured PNG.

        :param fname: Destination.
        :type fname: Path
        :param dpi: Image resolution.
        :type dpi: float
        :param format: Image format.
        :type format: Literal["png"]
        :return: None.
        :rtype: None
        """
        raise NotImplementedError

    def colorbar(
        self, mappable: QuadMesh | AxesImage, *, ax: PlotAxis, label: str
    ) -> Colorbar:
        """Label the spectrogram's power scale.

        :param mappable: Spectrogram artist.
        :type mappable: QuadMesh | AxesImage
        :param ax: Spectrogram axis.
        :type ax: PlotAxis
        :param label: Power units.
        :type label: str
        :return: Color scale.
        :rtype: Colorbar
        """
        raise NotImplementedError

    def suptitle(self, t: str) -> Text:
        """Name a figure that contains several panels.

        :param t: Shared figure title.
        :type t: str
        :return: Title artist.
        :rtype: Text
        """
        raise NotImplementedError
