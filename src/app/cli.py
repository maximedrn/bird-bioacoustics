"""Command-line entry points for independent batch execution and reporting."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from json import dumps
from logging import INFO, basicConfig
from os import environ
from pathlib import Path
from sys import modules
from typing import Final, Protocol, TextIO, TypedDict, cast

from app.audio.models import ModelPreparation, prepare_model
from app.domain.constants import InferenceDevice, RuntimeModule
from app.domain.errors import ExperimentAlreadyRunningError
from app.domain.messages import CliMessage, LogMessage
from app.domain.settings import (
    CorpusSettings,
    InferenceSettings,
    ModelSettings,
    ProjectPaths,
)
from app.domain.types import CorpusStatus
from app.experiments.progress import TqdmBatchProgress
from app.experiments.runner import run_batch
from app.reporting.notebooks import NotebookPublisher
from app.reporting.previews import (
    NotebookPreviewExporter,
    NotebookPreviewPaths,
)
from app.reporting.results import ExperimentResults
from app.reporting.service import ExperimentReport, load_report


class RuntimeStreams(Protocol):
    """Read the system stream after notebook or test redirection."""

    @property
    def stderr(self) -> TextIO | None:
        """Retrieve the current error stream.

        :return: Active error stream or None for an embedded interpreter.
        :rtype: TextIO | None
        """


RUNTIME_STREAMS: Final[RuntimeStreams] = cast(
    RuntimeStreams, modules[RuntimeModule.SYSTEM]
)


class Command(StrEnum):
    """Name independently executable application workflows."""

    BATCH = "batch"
    MODELS = "models"
    REPORT = "report"
    STATUS = "status"


class ExitCode(IntEnum):
    """Expose stable shell results for automation."""

    SUCCESS = 0
    ERROR = 1
    BUSY = 2
    INTERRUPTED = 130


class CliNamespace(Namespace):
    """Declare argparse destinations with explicit types."""

    root: Path | None
    command: str
    batch_size: int = CorpusSettings().batch_size
    no_progress: bool = False
    device: InferenceDevice = InferenceDevice.CPU
    workers: int = InferenceSettings().n_workers
    producers: int = InferenceSettings().n_producers
    inference_batch_size: int = InferenceSettings().batch_size


@dataclass(frozen=True, slots=True)
class CliOptions:
    """Hold parsed and explicitly typed command options."""

    root: Path | None
    command: Command
    batch_size: int
    no_progress: bool
    inference: InferenceSettings


class ReportOutput(TypedDict):
    """Type the CLI's cumulative report payload."""

    status: CorpusStatus
    figures: list[str]
    notebook: str
    html: str
    pdf: str
    svg: str


class CommandParsers(Protocol):
    """Type the public subparser registration operation."""

    def add_parser(self, name: str, *, help: str) -> ArgumentParser:
        """Register one available command.

        :param name: Command name.
        :type name: str
        :param help: Command description.
        :type help: str
        :return: Command argument parser.
        :rtype: ArgumentParser
        """
        raise NotImplementedError


def parser() -> ArgumentParser:
    """Build the CLI without opening storage or reading API credentials.

    :return: Command-line argument parser.
    :rtype: ArgumentParser
    """
    result: ArgumentParser = ArgumentParser(
        prog=CliMessage.PROGRAM, description=CliMessage.DESCRIPTION
    )
    result.add_argument("--root", type=Path, help=CliMessage.ROOT_HELP)
    commands: CommandParsers = cast(
        CommandParsers, result.add_subparsers(dest="command", required=True)
    )
    batch: ArgumentParser = commands.add_parser(
        Command.BATCH, help=CliMessage.BATCH_HELP
    )
    batch.add_argument(
        "--batch-size",
        type=int,
        default=CorpusSettings().batch_size,
        help=CliMessage.BATCH_SIZE_HELP,
    )
    batch.add_argument(
        "--no-progress", action="store_true", help=CliMessage.NO_PROGRESS_HELP
    )
    add_inference_options(batch)
    models: ArgumentParser = commands.add_parser(
        Command.MODELS, help=CliMessage.MODELS_HELP
    )
    add_inference_options(models)
    commands.add_parser(Command.REPORT, help=CliMessage.REPORT_HELP)
    commands.add_parser(Command.STATUS, help=CliMessage.STATUS_HELP)
    return result


def add_inference_options(command: ArgumentParser) -> None:
    """Share hardware controls between model preparation and batch execution.

    :param command: Subcommand accepting inference resource options.
    :type command: ArgumentParser
    :return: None.
    :rtype: None
    """
    defaults: InferenceSettings = InferenceSettings()
    command.add_argument(
        "--device",
        type=InferenceDevice,
        choices=tuple(InferenceDevice),
        default=defaults.device,
        help=CliMessage.DEVICE_HELP,
    )
    command.add_argument(
        "--workers",
        type=int,
        default=defaults.n_workers,
        help=CliMessage.WORKERS_HELP,
    )
    command.add_argument(
        "--producers",
        type=int,
        default=defaults.n_producers,
        help=CliMessage.PRODUCERS_HELP,
    )
    command.add_argument(
        "--inference-batch-size",
        type=int,
        default=defaults.batch_size,
        help=CliMessage.INFERENCE_BATCH_SIZE_HELP,
    )


def parse_options(argv: list[str] | None) -> CliOptions:
    """Copy argparse destinations into an immutable typed configuration.

    :param argv: Optional command arguments.
    :type argv: list[str] | None
    :return: Valid parsed command options.
    :rtype: CliOptions
    """
    namespace: CliNamespace = CliNamespace()
    parser().parse_args(argv, namespace=namespace)
    return CliOptions(
        namespace.root,
        Command(namespace.command),
        namespace.batch_size,
        namespace.no_progress,
        InferenceSettings(
            device=namespace.device,
            n_workers=namespace.workers,
            n_producers=namespace.producers,
            batch_size=namespace.inference_batch_size,
        ),
    )


def execute_command(
    options: CliOptions, paths: ProjectPaths
) -> CorpusStatus | ReportOutput | ModelPreparation:
    """Dispatch a command without starting unrelated business workflows.

    :param options: Validated CLI configuration.
    :type options: CliOptions
    :param paths: Immutable project directories.
    :type paths: ProjectPaths
    :return: JSON-compatible command output.
    :rtype: CorpusStatus | ReportOutput | ModelPreparation
    """
    if options.command == Command.BATCH:
        return execute_batch(options, paths)
    if options.command == Command.REPORT:
        return execute_report(paths)
    if options.command == Command.MODELS:
        return prepare_model(ModelSettings(), options.inference)
    return read_status(paths)


def execute_batch(options: CliOptions, paths: ProjectPaths) -> CorpusStatus:
    """Process one batch and enable progress according to the CLI option.

    :param options: Batch size and progress preferences.
    :type options: CliOptions
    :param paths: Project directories.
    :type paths: ProjectPaths
    :return: Saved cumulative coverage.
    :rtype: CorpusStatus
    """

    return run_batch(
        paths,
        corpus=CorpusSettings(batch_size=options.batch_size),
        progress=TqdmBatchProgress(enabled=not options.no_progress),
        inference=options.inference,
    )


def execute_report(paths: ProjectPaths) -> ReportOutput:
    """Refresh notebook results and publish the HTML, PDF and SVG preview.

    :param paths: Project directories.
    :type paths: ProjectPaths
    :return: Coverage and paths to the figures, notebook and preview formats.
    :rtype: ReportOutput
    """
    environ.setdefault("MPLBACKEND", "Agg")

    report: ExperimentReport = load_report(paths)
    figures: tuple[Path, ...] = report.save_figures()
    notebook: Path = NotebookPublisher(report, figures).refresh()
    preview: NotebookPreviewPaths = NotebookPreviewExporter(paths).export(
        notebook, report
    )
    return ReportOutput(
        status=report.snapshot.status,
        figures=[str(path) for path in figures],
        notebook=str(notebook),
        html=str(preview.html),
        pdf=str(preview.pdf),
        svg=str(preview.svg),
    )


def read_status(paths: ProjectPaths) -> CorpusStatus:
    """Read current coverage without creating results or starting inference.

    :param paths: Project directories.
    :type paths: ProjectPaths
    :return: Persisted catalogue coverage.
    :rtype: CorpusStatus
    """

    reader: ExperimentResults
    with ExperimentResults(paths.checkpoint) as reader:
        return reader.report()


def print_error(message: str) -> None:
    """Use the current stderr stream, including notebook or test redirection.

    :param message: Formatted operational error.
    :type message: str
    :return: None.
    :rtype: None
    """

    print(message, file=RUNTIME_STREAMS.stderr)


def main(argv: list[str] | None = None) -> int:
    """Execute one requested workflow with resource-safe error handling.

    :param argv: Optional argument list for embedding and tests.
    :type argv: list[str] | None
    :return: Process exit code.
    :rtype: int
    """
    error: OSError | ValueError | RuntimeError
    basicConfig(level=INFO, format=LogMessage.FORMAT)
    try:
        options: CliOptions = parse_options(argv)
        paths: ProjectPaths = (
            ProjectPaths.from_root(options.root)
            if options.root
            else ProjectPaths.from_working_directory()
        )
        result: CorpusStatus | ReportOutput | ModelPreparation = (
            execute_command(options, paths)
        )
        print(dumps(result, indent=2, ensure_ascii=False))
    except ExperimentAlreadyRunningError as error:
        print_error(str(error))
        return ExitCode.BUSY
    except KeyboardInterrupt:
        print_error(CliMessage.INTERRUPTED)
        return ExitCode.INTERRUPTED
    except (OSError, ValueError, RuntimeError) as error:
        print_error(
            CliMessage.ERROR.format(kind=type(error).__name__, error=error)
        )
        return ExitCode.ERROR
    return ExitCode.SUCCESS
