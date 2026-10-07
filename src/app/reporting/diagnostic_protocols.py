"""Type the Matplotlib keyword calls used by diagnostic charts."""

from collections.abc import Sequence
from typing import Literal, Protocol

from matplotlib.collections import PathCollection
from matplotlib.container import BarContainer
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.text import Text
from numpy.typing import ArrayLike

from app.reporting.protocols import PlotAxis


class DiagnosticAxis(PlotAxis, Protocol):
    """Extend the plot boundary with supported diagnostic keywords."""

    def bxp(
        self,
        bxpstats: Sequence[dict[str, float | str]],
        *,
        showfliers: bool,
    ) -> dict[str, list[Line2D]]:
        """Draw externally computed quartiles and observed ranges.

        :param bxpstats: Statistics using Matplotlib's boxplot field names.
        :type bxpstats: Sequence[dict[str, float | str]]
        :param showfliers: Whether to show individually supplied outliers.
        :type showfliers: bool
        :return: Boxplot line artists.
        :rtype: dict[str, list[Line2D]]
        """
        raise NotImplementedError

    def scatter(
        self, x: ArrayLike, y: ArrayLike, *, s: float, alpha: float
    ) -> PathCollection:
        """Draw individual recording observations.

        :param x: Horizontal measurements.
        :type x: ArrayLike
        :param y: Vertical measurements.
        :type y: ArrayLike
        :param s: Marker area.
        :type s: float
        :param alpha: Marker opacity.
        :type alpha: float
        :return: Scatter point collection.
        :rtype: PathCollection
        """
        raise NotImplementedError

    def imshow(
        self,
        X: ArrayLike,
        *,
        vmin: float,
        vmax: float,
        cmap: str,
        aspect: Literal["auto"],
    ) -> AxesImage:
        """Draw a species-condition rate matrix.

        :param X: Matrix of rates with missing cells represented by NaN.
        :type X: ArrayLike
        :param vmin: Lower color bound.
        :type vmin: float
        :param vmax: Upper color bound.
        :type vmax: float
        :param cmap: Named colormap.
        :type cmap: str
        :param aspect: Cell aspect handling.
        :type aspect: Literal["auto"]
        :return: Heatmap artist.
        :rtype: AxesImage
        """
        raise NotImplementedError

    def set_xticks(
        self, ticks: ArrayLike, labels: Sequence[str] | None = None
    ) -> Sequence[object]:
        """Place labelled condition or batch ticks.

        :param ticks: Tick positions.
        :type ticks: ArrayLike
        :param labels: Optional condition labels.
        :type labels: Sequence[str] | None
        :return: Tick artists.
        :rtype: Sequence[object]
        """
        raise NotImplementedError

    def set_yticks(
        self, ticks: ArrayLike, labels: Sequence[str]
    ) -> Sequence[object]:
        """Label heatmap species rows.

        :param ticks: Tick positions.
        :type ticks: ArrayLike
        :param labels: Species names.
        :type labels: Sequence[str]
        :return: Tick artists.
        :rtype: Sequence[object]
        """
        raise NotImplementedError

    def text(
        self,
        x: float,
        y: float,
        s: str,
        *,
        ha: Literal["center"],
        va: Literal["center"],
        color: str,
        fontsize: float,
    ) -> Text:
        """Annotate a heatmap cell with its rate and observation count.

        :param x: Column coordinate.
        :type x: float
        :param y: Row coordinate.
        :type y: float
        :param s: Formatted rate and sample size.
        :type s: str
        :param ha: Horizontal alignment.
        :type ha: Literal["center"]
        :param va: Vertical alignment.
        :type va: Literal["center"]
        :param color: Text color chosen for cell contrast.
        :type color: str
        :param fontsize: Text size.
        :type fontsize: float
        :return: Text artist.
        :rtype: Text
        """
        raise NotImplementedError

    def set_xscale(self, value: Literal["log"]) -> None:
        """Use a logarithmic scale for positive recording durations.

        :param value: Logarithmic scale name.
        :type value: Literal["log"]
        :return: None.
        :rtype: None
        """
        raise NotImplementedError

    def set_xlim(self, left: float, right: float) -> tuple[float, float]:
        """Set horizontal confidence or count limits.

        :param left: Lower bound.
        :type left: float
        :param right: Upper bound.
        :type right: float
        :return: Effective bounds.
        :rtype: tuple[float, float]
        """
        raise NotImplementedError

    def set_aspect(
        self, aspect: Literal["equal"], *, adjustable: Literal["box"]
    ) -> None:
        """Give paired confidence axes identical geometric scales.

        :param aspect: Relative axis scale.
        :type aspect: Literal["equal"]
        :param adjustable: Resize the axis box to retain confidence limits.
        :type adjustable: Literal["box"]
        :return: None.
        :rtype: None
        """
        raise NotImplementedError

    def barh(self, y: Sequence[str], width: ArrayLike) -> BarContainer:
        """Draw horizontal species counts.

        :param y: Species labels.
        :type y: Sequence[str]
        :param width: Recording counts.
        :type width: ArrayLike
        :return: Bar artists available for numeric annotation.
        :rtype: BarContainer
        """
        raise NotImplementedError
