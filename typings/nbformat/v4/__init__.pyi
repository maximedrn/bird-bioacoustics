"""Typed display-output constructor used by the notebook publisher.

Upstream: nbformat/v4/nbbase.py.
"""

from collections.abc import Mapping

from nbformat import NotebookNode

def new_output(
    output_type: str,
    *,
    data: Mapping[str, str],
) -> NotebookNode: ...
