"""Combine every PDF page into one self-contained SVG preview."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final
from xml.etree.ElementTree import (
    Element,
    ElementTree,
    SubElement,
    fromstring,
    register_namespace,
)

from pymupdf import Document, Page

from app.reporting.messages import ReportMessage
from app.reporting.themes import ReportTheme
from app.reporting.writing import atomic_output

SVG_NAMESPACE: Final[str] = "http://www.w3.org/2000/svg"
XLINK_NAMESPACE: Final[str] = "http://www.w3.org/1999/xlink"


class SvgTag(StrEnum):
    """Keep SVG elements in the correct XML namespace."""

    ROOT = f"{{{SVG_NAMESPACE}}}svg"
    RECTANGLE = f"{{{SVG_NAMESPACE}}}rect"
    TITLE = f"{{{SVG_NAMESPACE}}}title"
    GROUP = f"{{{SVG_NAMESPACE}}}g"
    STYLE = f"{{{SVG_NAMESPACE}}}style"


class SvgAttribute(StrEnum):
    """Name viewport, placement, background and reference attributes."""

    ID = "id"
    WIDTH = "width"
    HEIGHT = "height"
    VIEW_BOX = "viewBox"
    X = "x"
    Y = "y"
    FILL = "fill"
    ROLE = "role"


class SvgSetting:
    """Define deterministic page spacing and document serialization."""

    PAGE_GAP_POINTS: Final[float] = 12.0
    IMAGE_ROLE: Final[str] = "img"
    REFERENCE_PREFIX: Final[str] = "#"
    URL_REFERENCE_PREFIX: Final[str] = "url(#"
    PAGE_ID: Final[str] = "{theme}_page_{number}_{identifier}"
    THEME_ID: Final[str] = "theme-{theme}"
    THEME_STYLES: Final[str] = (
        "#{dark} {{ display: none; }}\n"
        "@media (prefers-color-scheme: dark) {{\n"
        "  #{light} {{ display: none; }}\n"
        "  #{dark} {{ display: inline; }}\n"
        "}}\n"
    )
    ENCODING: Final[str] = "utf-8"


@dataclass(frozen=True, slots=True)
class SvgPage:
    """Retain an SVG page and its dimensions in PDF points."""

    root: Element
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class SvgLayer:
    """Describe one complete notebook rendition and its theme."""

    theme: ReportTheme
    pages: tuple[SvgPage, ...]

    @property
    def width(self) -> float:
        """Return the width required by the widest page.

        :return: Layer width in PDF points.
        :rtype: float
        """
        return max(page.width for page in self.pages)

    @property
    def height(self) -> float:
        """Return the total height of stacked pages.

        :return: Layer height in PDF points.
        :rtype: float
        """
        return (
            sum(page.height for page in self.pages)
            + (len(self.pages) - 1) * SvgSetting.PAGE_GAP_POINTS
        )


class PdfSvgExporter:
    """Render vector text and embedded figures for all PDF pages."""

    @classmethod
    def export(
        cls, pdf: Path, destination: Path, *, dark_pdf: Path | None = None
    ) -> None:
        """Publish all PDF pages as one SVG without external dependencies.

        :param pdf: Printed notebook from the HTML conversion step.
        :type pdf: Path
        :param destination: README image destination.
        :type destination: Path
        :param dark_pdf: Optional dark rendition selected by the SVG's CSS.
        :type dark_pdf: Path | None
        :return: None.
        :rtype: None
        """
        layers: list[SvgLayer] = [
            SvgLayer(ReportTheme.LIGHT, cls._pages(pdf, ReportTheme.LIGHT))
        ]
        if dark_pdf is not None:
            layers.append(
                SvgLayer(
                    ReportTheme.DARK, cls._pages(dark_pdf, ReportTheme.DARK)
                )
            )
        root: Element = cls._combine(tuple(layers))
        register_namespace("", SVG_NAMESPACE)
        register_namespace("xlink", XLINK_NAMESPACE)
        temporary: Path
        with atomic_output(destination) as temporary:
            ElementTree(root).write(
                temporary,
                encoding=SvgSetting.ENCODING,
                xml_declaration=True,
            )

    @classmethod
    def _pages(cls, pdf: Path, theme: ReportTheme) -> tuple[SvgPage, ...]:
        """Export each PDF page and make its XML identifiers globally unique.

        :param pdf: Notebook PDF containing one or more pages.
        :type pdf: Path
        :param theme: Variant namespace for clip and glyph identifiers.
        :type theme: ReportTheme
        :return: Ordered page images with independent clip and glyph IDs.
        :rtype: tuple[SvgPage, ...]
        """
        document: Document = Document(str(pdf))
        pages: list[SvgPage] = []
        try:
            if not document.page_count:
                raise ValueError(ReportMessage.EMPTY_NOTEBOOK_PDF)
            for number in range(document.page_count):
                page: Page = document.load_page(number)
                root: Element = fromstring(
                    page.get_svg_image(text_as_path=True)
                )
                cls._unique_ids(root, number, theme)
                pages.append(SvgPage(root, page.rect.width, page.rect.height))
        finally:
            document.close()
        return tuple(pages)

    @staticmethod
    def _unique_ids(root: Element, number: int, theme: ReportTheme) -> None:
        """Update IDs and their references before combining page documents.

        :param root: One PDF page's SVG document.
        :type root: Element
        :param number: Zero-based PDF page number.
        :type number: int
        :param theme: Namespace shared by one complete notebook rendition.
        :type theme: ReportTheme
        :return: None.
        :rtype: None
        """
        references: dict[str, str] = {}
        for element in root.iter():
            identifier: str | None = element.get(SvgAttribute.ID)
            if identifier is not None:
                references[identifier] = SvgSetting.PAGE_ID.format(
                    theme=theme.value, number=number, identifier=identifier
                )
        for element in root.iter():
            for attribute, value in element.attrib.items():
                element.set(
                    attribute,
                    references[value]
                    if attribute == SvgAttribute.ID
                    else PdfSvgExporter._reference(value, references),
                )

    @staticmethod
    def _reference(value: str, references: dict[str, str]) -> str:
        """Rewrite fragment and URL references without changing other values.

        :param value: Original SVG attribute value.
        :type value: str
        :param references: Original to page-specific XML identifiers.
        :type references: dict[str, str]
        :return: Updated attribute value.
        :rtype: str
        """
        if value.startswith(SvgSetting.REFERENCE_PREFIX):
            identifier: str = value[1:]
            return SvgSetting.REFERENCE_PREFIX + references.get(
                identifier, identifier
            )
        if SvgSetting.URL_REFERENCE_PREFIX not in value:
            return value
        for original, replacement in references.items():
            value = value.replace(f"url(#{original})", f"url(#{replacement})")
        return value

    @staticmethod
    def _combine(layers: tuple[SvgLayer, ...]) -> Element:
        """Overlay theme renditions and select one using the reader's scheme.

        :param layers: Complete light and optional dark notebook renditions.
        :type layers: tuple[SvgLayer, ...]
        :return: One self-contained SVG with automatic theme selection.
        :rtype: Element
        """
        width: float = max(layer.width for layer in layers)
        height: float = max(layer.height for layer in layers)
        root: Element = Element(
            SvgTag.ROOT,
            {
                SvgAttribute.WIDTH: f"{width:g}pt",
                SvgAttribute.HEIGHT: f"{height:g}pt",
                SvgAttribute.VIEW_BOX: f"0 0 {width:g} {height:g}",
                SvgAttribute.ROLE: SvgSetting.IMAGE_ROLE,
            },
        )
        SubElement(
            root, SvgTag.TITLE
        ).text = ReportMessage.NOTEBOOK_PREVIEW_TITLE
        if len(layers) > 1:
            SubElement(
                root, SvgTag.STYLE
            ).text = SvgSetting.THEME_STYLES.format(
                light=SvgSetting.THEME_ID.format(
                    theme=ReportTheme.LIGHT.value
                ),
                dark=SvgSetting.THEME_ID.format(theme=ReportTheme.DARK.value),
            )
        layer: SvgLayer
        for layer in layers:
            PdfSvgExporter._append_layer(root, layer, width, height)
        return root

    @staticmethod
    def _append_layer(
        root: Element, layer: SvgLayer, width: float, height: float
    ) -> None:
        """Stack the pages of one theme over its own opaque background.

        :param root: SVG document shared by all variants.
        :type root: Element
        :param layer: Pages and palette of one notebook rendition.
        :type layer: SvgLayer
        :param width: Complete document width in PDF points.
        :type width: float
        :param height: Complete document height in PDF points.
        :type height: float
        :return: None.
        :rtype: None
        """
        group: Element = SubElement(
            root,
            SvgTag.GROUP,
            {
                SvgAttribute.ID: SvgSetting.THEME_ID.format(
                    theme=layer.theme.value
                )
            },
        )
        SubElement(
            group,
            SvgTag.RECTANGLE,
            {
                SvgAttribute.WIDTH: f"{width:g}",
                SvgAttribute.HEIGHT: f"{height:g}",
                SvgAttribute.FILL: layer.theme.palette.background,
            },
        )
        offset: float = 0.0
        page: SvgPage
        for page in layer.pages:
            page.root.set(SvgAttribute.X, "0")
            page.root.set(SvgAttribute.Y, f"{offset:g}")
            page.root.set(SvgAttribute.WIDTH, f"{page.width:g}")
            page.root.set(SvgAttribute.HEIGHT, f"{page.height:g}")
            group.append(page.root)
            offset += page.height + SvgSetting.PAGE_GAP_POINTS
