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
from app.reporting.writing import atomic_output

SVG_NAMESPACE: Final[str] = "http://www.w3.org/2000/svg"
XLINK_NAMESPACE: Final[str] = "http://www.w3.org/1999/xlink"


class SvgTag(StrEnum):
    """Keep SVG elements in the correct XML namespace."""

    ROOT = f"{{{SVG_NAMESPACE}}}svg"
    RECTANGLE = f"{{{SVG_NAMESPACE}}}rect"
    TITLE = f"{{{SVG_NAMESPACE}}}title"


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
    BACKGROUND: Final[str] = "#ffffff"
    IMAGE_ROLE: Final[str] = "img"
    REFERENCE_PREFIX: Final[str] = "#"
    URL_REFERENCE_PREFIX: Final[str] = "url(#"
    PAGE_ID: Final[str] = "page_{number}_{identifier}"
    ENCODING: Final[str] = "utf-8"


@dataclass(frozen=True, slots=True)
class SvgPage:
    """Retain an SVG page and its dimensions in PDF points."""

    root: Element
    width: float
    height: float


class PdfSvgExporter:
    """Render vector text and embedded figures for all PDF pages."""

    @classmethod
    def export(cls, pdf: Path, destination: Path) -> None:
        """Publish all PDF pages as one SVG without external dependencies.

        :param pdf: Printed notebook from the HTML conversion step.
        :type pdf: Path
        :param destination: README image destination.
        :type destination: Path
        :return: None.
        :rtype: None
        """
        pages: tuple[SvgPage, ...] = cls._pages(pdf)
        root: Element = cls._combine(pages)
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
    def _pages(cls, pdf: Path) -> tuple[SvgPage, ...]:
        """Export each PDF page and make its XML identifiers globally unique.

        :param pdf: Notebook PDF containing one or more pages.
        :type pdf: Path
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
                cls._unique_ids(root, number)
                pages.append(SvgPage(root, page.rect.width, page.rect.height))
        finally:
            document.close()
        return tuple(pages)

    @staticmethod
    def _unique_ids(root: Element, number: int) -> None:
        """Update IDs and their references before combining page documents.

        :param root: One PDF page's SVG document.
        :type root: Element
        :param number: Zero-based PDF page number.
        :type number: int
        :return: None.
        :rtype: None
        """
        references: dict[str, str] = {}
        for element in root.iter():
            identifier: str | None = element.get(SvgAttribute.ID)
            if identifier is not None:
                references[identifier] = SvgSetting.PAGE_ID.format(
                    number=number, identifier=identifier
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
    def _combine(pages: tuple[SvgPage, ...]) -> Element:
        """Stack page viewports on a white background, preserving their scale.

        :param pages: Ordered PDF page images.
        :type pages: tuple[SvgPage, ...]
        :return: One SVG document containing every page.
        :rtype: Element
        """
        width: float = max(page.width for page in pages)
        height: float = (
            sum(page.height for page in pages)
            + (len(pages) - 1) * SvgSetting.PAGE_GAP_POINTS
        )
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
        SubElement(
            root,
            SvgTag.RECTANGLE,
            {
                SvgAttribute.WIDTH: f"{width:g}",
                SvgAttribute.HEIGHT: f"{height:g}",
                SvgAttribute.FILL: SvgSetting.BACKGROUND,
            },
        )
        offset: float = 0.0
        for page in pages:
            page.root.set(SvgAttribute.X, "0")
            page.root.set(SvgAttribute.Y, f"{offset:g}")
            page.root.set(SvgAttribute.WIDTH, f"{page.width:g}")
            page.root.set(SvgAttribute.HEIGHT, f"{page.height:g}")
            root.append(page.root)
            offset += page.height + SvgSetting.PAGE_GAP_POINTS
        return root
