"""Save result artifacts atomically without exposing partial output."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol


@contextmanager
def atomic_output(destination: Path) -> Generator[Path, None, None]:
    """Replace an artifact only after its temporary write has completed.

    :param destination: Public artifact path.
    :type destination: Path
    :return: Unique temporary path for the writer.
    :rtype: Generator[Path, None, None]
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.stem}_",
        suffix=destination.suffix,
        delete=False,
    ) as stream:
        temporary: Path = Path(stream.name)
    try:
        yield temporary
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


class TemporaryOutput(Protocol):
    """Describe the named output handle used only to retrieve a pathname."""

    @property
    def name(self) -> str:
        """Read the temporary file's pathname.

        :return: Temporary output filename.
        :rtype: str
        """
        raise NotImplementedError
