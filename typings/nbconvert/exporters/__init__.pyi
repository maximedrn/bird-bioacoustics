"""Correct the HTML exporter's inherited notebook return annotation.

Upstream: nbconvert/exporters/html.py and nbconvert/exporters/exporter.py.
"""

class HTMLExporter:
    def __init__(
        self,
        *,
        template_name: str,
        exclude_input: bool,
        exclude_input_prompt: bool,
        exclude_output_prompt: bool,
        exclude_anchor_links: bool,
    ) -> None: ...
    def from_filename(
        self, filename: str
    ) -> tuple[str, dict[str, object]]: ...
