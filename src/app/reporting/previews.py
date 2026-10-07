"""Convert saved notebook results through HTML and PDF into a README SVG."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from shutil import copy2
from string import Template
from tempfile import TemporaryDirectory
from typing import Final, Literal

from matplotlib.style import context as plot_style_context
from nbconvert.exporters import HTMLExporter
from playwright.sync_api import Browser, Page, sync_playwright
from playwright.sync_api import Error as BrowserError

from app.domain.constants import Artifact, FileName
from app.domain.settings import ProjectPaths
from app.reporting.messages import ReportMessage
from app.reporting.notebooks import NotebookPublisher
from app.reporting.service import ExperimentReport
from app.reporting.svg import PdfSvgExporter
from app.reporting.themes import ReportTheme, ThemePalette
from app.reporting.writing import atomic_output


class PreviewSetting:
    """Centralize notebook export and browser rendering options."""

    PACKAGE: Final[str] = "app.reporting"
    TEMPLATE: Final[str] = "templates/notebook.html.tpl"
    HTML_TEMPLATE: Final[str] = "basic"
    PAGE_LOAD: Final[Literal["load"]] = "load"
    PRINT_MEDIA: Final[Literal["print"]] = "print"
    CSS_PIXEL_UNIT: Final[str] = "px"
    LOAD_TIMEOUT_MS: Final[int] = 30_000
    HEIGHT_PADDING_PX: Final[int] = 1
    TEMPORARY_PREFIX: Final[str] = "notebook_preview_"
    ENCODING: Final[str] = "utf-8"
    RESOURCES_READY: Final[str] = (
        "document.fonts.status === 'loaded' && "
        "Array.from(document.images).every(image => "
        "image.complete && image.naturalWidth > 0)"
    )
    PAGE_DIMENSIONS: Final[str] = (
        "() => { const bounds = document.body.getBoundingClientRect(); "
        "return [Math.ceil(bounds.width), Math.ceil(bounds.height)]; }"
    )


@dataclass(frozen=True, slots=True)
class NotebookPreviewPaths:
    """Expose each saved step of the notebook preview pipeline."""

    html: Path
    pdf: Path
    svg: Path


@dataclass(frozen=True, slots=True)
class NotebookPreviewExporter:
    """Publish a portable preview without executing notebook cells."""

    paths: ProjectPaths

    def export(
        self, notebook: Path, report: ExperimentReport
    ) -> NotebookPreviewPaths:
        """Convert saved results to HTML, a single-page PDF and adaptive SVG.

        :param notebook: Notebook whose outputs were just refreshed.
        :type notebook: Path
        :param report: Same cumulative snapshot as the saved notebook outputs.
        :type report: ExperimentReport
        :return: HTML, PDF and SVG artifacts in the results directory.
        :rtype: NotebookPreviewPaths
        """
        html: Path = self._html(notebook)
        pdf: Path = self._pdf(html)
        svg: Path = Artifact.NOTEBOOK.path(self.paths.results, "svg")
        with TemporaryDirectory(
            prefix=PreviewSetting.TEMPORARY_PREFIX
        ) as directory:
            dark_pdf: Path = self._dark_pdf(notebook, report, Path(directory))
            PdfSvgExporter.export(pdf, svg, dark_pdf=dark_pdf)
        return NotebookPreviewPaths(html, pdf, svg)

    def _html(
        self, notebook: Path, theme: ReportTheme = ReportTheme.LIGHT
    ) -> Path:
        """Render the notebook's Markdown and outputs into standalone HTML.

        :param notebook: Saved presentation notebook.
        :type notebook: Path
        :param theme: Colors used by this document variant.
        :type theme: ReportTheme
        :return: Standalone HTML with embedded figures and print styles.
        :rtype: Path
        """
        exporter: HTMLExporter = HTMLExporter(
            template_name=PreviewSetting.HTML_TEMPLATE,
            exclude_input=True,
            exclude_input_prompt=True,
            exclude_output_prompt=True,
            exclude_anchor_links=True,
        )
        body, _resources = exporter.from_filename(str(notebook))
        resource: Traversable = files(PreviewSetting.PACKAGE).joinpath(
            PreviewSetting.TEMPLATE
        )
        template: Template = Template(resource.read_text())
        palette: ThemePalette = theme.palette
        destination: Path = Artifact.NOTEBOOK.path(self.paths.results, "html")
        with atomic_output(destination) as temporary:
            temporary.write_text(
                template.substitute(
                    body=body,
                    background=palette.background,
                    foreground=palette.foreground,
                    link=palette.link,
                ),
                encoding=PreviewSetting.ENCODING,
            )
        return destination

    @staticmethod
    def _dark_pdf(
        notebook: Path, report: ExperimentReport, root: Path
    ) -> Path:
        """Render dark plots and a PDF from the existing cumulative snapshot.

        :param notebook: Refreshed notebook whose source cells are preserved.
        :type notebook: Path
        :param report: Measurements shared by both preview variants.
        :type report: ExperimentReport
        :param root: Temporary directory removed after SVG publication.
        :type root: Path
        :return: Dark single-page PDF for the adaptive SVG layer.
        :rtype: Path
        """
        paths: ProjectPaths = ProjectPaths.from_root(root)
        copy2(notebook, paths.root / FileName.NOTEBOOK)
        dark_report: ExperimentReport = ExperimentReport(
            paths, report.snapshot
        )
        with plot_style_context(ReportTheme.DARK.palette.plot_settings()):
            figures = dark_report.save_figures()
        dark_notebook: Path = NotebookPublisher(dark_report, figures).refresh()
        exporter: NotebookPreviewExporter = NotebookPreviewExporter(paths)
        html: Path = exporter._html(dark_notebook, ReportTheme.DARK)
        return exporter._pdf(html)

    def _pdf(self, html: Path) -> Path:
        """Print the saved HTML after its embedded images and fonts are ready.

        :param html: Standalone HTML created by the previous step.
        :type html: Path
        :return: Single-page PDF with a height fitted to the notebook content.
        :rtype: Path
        """
        destination: Path = Artifact.NOTEBOOK.path(self.paths.results, "pdf")
        try:
            with sync_playwright() as automation:
                browser: Browser = automation.chromium.launch()
                try:
                    page: Page = browser.new_page()
                    page.emulate_media(media=PreviewSetting.PRINT_MEDIA)
                    page.goto(
                        html.resolve().as_uri(),
                        wait_until=PreviewSetting.PAGE_LOAD,
                        timeout=PreviewSetting.LOAD_TIMEOUT_MS,
                    )
                    page.wait_for_function(
                        PreviewSetting.RESOURCES_READY,
                        timeout=PreviewSetting.LOAD_TIMEOUT_MS,
                    )
                    width, height = page.evaluate(
                        PreviewSetting.PAGE_DIMENSIONS
                    )
                    with atomic_output(destination) as temporary:
                        page.pdf(
                            path=str(temporary),
                            width=f"{width}{PreviewSetting.CSS_PIXEL_UNIT}",
                            height=(
                                f"{height + PreviewSetting.HEIGHT_PADDING_PX}"
                                f"{PreviewSetting.CSS_PIXEL_UNIT}"
                            ),
                            print_background=True,
                            prefer_css_page_size=False,
                        )
                finally:
                    browser.close()
        except BrowserError as error:
            raise RuntimeError(
                ReportMessage.NOTEBOOK_PDF_FAILED.format(error=error)
            ) from error
        return destination
