"""Generates the blank PDF for the synthetic demo form.

WHY THIS EXISTS AT ALL. A form-filling pipeline whose tests never touch a PDF is
testing its own bookkeeping. But this repo cannot ship the Illinois Certificate
of Child Health Examination -- it is a state document, and a copy of it here
with guessed field coordinates would be worse than useless.

So the tests and `make demo-i01` run against a form this repo DRAWS, at exactly
the coordinates `config/forms/demo_camp_health_form.yaml` declares. That makes
the round trip checkable in a way a fixture never could be: generate the blank,
fill it, read the text back out of each box, and assert the value landed where
the template said it would. A coordinate bug shows up as text in the wrong box,
which is the actual failure mode.

PyMuPDF is imported INSIDE the function, not at module scope, for the same
reason the OCR engines are in I-06: a machine without it must still be able to
import this package, run the deterministic half, and see a clear message about
what is missing rather than an ImportError at startup.
"""

from __future__ import annotations

import os
from typing import Any

from .templates import FormTemplate

__all__ = ["PdfUnavailable", "write_blank_form", "PAGE_WIDTH", "PAGE_HEIGHT"]

#: US Letter, in points.
PAGE_WIDTH = 612.0
PAGE_HEIGHT = 792.0


class PdfUnavailable(RuntimeError):
    """Raised when a PDF operation is attempted without PyMuPDF installed."""


def _load_fitz() -> Any:
    try:
        import fitz  # noqa: PLC0415 - deliberate lazy import
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise PdfUnavailable(
            "PyMuPDF is not installed, so no PDF can be rendered. "
            "`pip install pymupdf`. The deterministic half of this module -- "
            "template loading, chart resolution, reconciliation, the discrepancy "
            "list and the review payload -- runs without it."
        ) from exc
    return fitz


def write_blank_form(template: FormTemplate, path: str | os.PathLike[str]) -> str:
    """Draw a blank form matching `template`, and return the path.

    Every field gets a labelled, ruled box at its declared coordinates, so a
    filled copy can be checked box by box. Section headers come from the page
    number alone -- this is a test fixture, not a design exercise.
    """
    fitz = _load_fitz()
    doc = fitz.open()
    grey = (0.45, 0.45, 0.45)
    black = (0.1, 0.1, 0.1)

    for page_number in range(1, template.page_count + 1):
        page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        page.insert_text(
            (72, 60), template.title, fontsize=13, color=black, fontname="helv"
        )
        page.insert_text(
            (72, 76),
            f"SYNTHETIC DEMONSTRATION FORM - page {page_number} of "
            f"{template.page_count}",
            fontsize=7, color=grey, fontname="helv",
        )
        for spec in template.fields:
            if spec.box.page != page_number:
                continue
            x0, y0, x1, y1 = spec.box.rect
            page.draw_rect(fitz.Rect(x0, y0, x1, y1), color=grey, width=0.5)
            # The label sits ABOVE the box, never beside it. Beside it, the
            # label for column 3 of the immunization grid overlapped the BOX of
            # column 1, so reading the text back out of column 1 returned the
            # generator's own label -- which would have made a coordinate test
            # pass against a form nobody could read.
            page.insert_text(
                (x0, y0 - 3),
                (spec.label or spec.name)[:34],
                fontsize=6, color=grey, fontname="helv",
            )

    path = str(path)
    doc.save(path)
    doc.close()
    return path
