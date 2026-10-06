"""Typed display subset used by the thin notebook and reporting service.

Upstream: IPython/core/display.py and IPython/core/display_functions.py.
"""

from os import PathLike

class DisplayObject: ...

class Markdown(DisplayObject):
    def __init__(self, data: str) -> None: ...

class Image(DisplayObject):
    def __init__(self, *, filename: str | PathLike[str]) -> None: ...

def display(*objects: object) -> None: ...
