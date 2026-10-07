"""Type the PDF and SVG subset missing from upstream method annotations.

Upstream: pymupdf/__init__.py and pymupdf/utils.py.
"""

class Rect:
    width: float
    height: float

class Page:
    rect: Rect

    def get_svg_image(self, *, text_as_path: bool) -> str: ...

class Document:
    page_count: int
    def __init__(self, filename: str) -> None: ...
    def load_page(self, page_id: int) -> Page: ...
    def close(self) -> None: ...
