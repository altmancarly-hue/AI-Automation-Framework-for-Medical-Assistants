"""Fax pages to text, with a confidence number that means something.

README I-06 classifies OCR as deterministic and the build plan pins the stack:
PaddleOCR primary, pytesseract fallback, per-page confidence, low confidence
routes to a human. Both engines are imported LAZILY -- `make lint` asserts no
module-scope import of either -- so the repo installs and the tests run on a box
with neither.

THE ONE DESIGN DECISION THAT MATTERS HERE IS HOW CONFIDENCE AGGREGATES.

A five-page discharge summary whose middle page is a black smear scores about
0.80 on a mean. The mean is the wrong statistic: the document is not 80%
readable, it is 80% readable and 20% missing, and the missing fifth is where the
discharge medications were. So a document's confidence is the MINIMUM page
confidence, and the page numbers below the threshold are named. A reviewer who
is told "page 3 of 5 is unreadable" does something different from one told "this
document scored 0.80".

The second decision is that an EMPTY page is not a confident page. An engine
that finds no text on a blank fax cover sheet returns no confidence scores at
all, and averaging an empty list produced 1.0 -- a perfectly confident reading
of nothing. A page with no extracted text has confidence 0.0 and says so.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence

__all__ = [
    "LOW_CONFIDENCE_THRESHOLD",
    "OCRUnavailable",
    "Page",
    "OCRResult",
    "OCREngine",
    "PaddleOCREngine",
    "TesseractEngine",
    "ScriptedOCR",
    "chain",
]

#: Below this a document goes to a human rather than to a classifier. Set from
#: the pilot's measured confidence distribution (README I-06: "measure OCR
#: confidence distribution during pilot"), not from this constant.
LOW_CONFIDENCE_THRESHOLD = 0.72


class OCRUnavailable(RuntimeError):
    """No OCR engine could run. Never silently returns empty text."""


@dataclass(frozen=True)
class Page:
    """One page. `confidence` is the engine's own, 0.0 when nothing was read."""

    number: int
    text: str
    confidence: float
    engine: str = ""
    #: Set when the engine reported a rotation or a very low-resolution source.
    notes: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        return not self.text.strip()


@dataclass
class OCRResult:
    """A whole transmission, with provenance and a usable confidence figure."""

    source: str
    pages: list[Page] = field(default_factory=list)
    engine: str = ""
    fallback_used: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def confidence(self) -> float:
        """The WORST page, not the average. See the module docstring."""
        if not self.pages:
            return 0.0
        return min(p.confidence for p in self.pages)

    @property
    def mean_confidence(self) -> float:
        """Reported alongside, because the distribution is what a pilot needs."""
        if not self.pages:
            return 0.0
        return sum(p.confidence for p in self.pages) / len(self.pages)

    def unreadable_pages(
        self, threshold: float = LOW_CONFIDENCE_THRESHOLD
    ) -> list[Page]:
        return [p for p in self.pages if p.confidence < threshold]

    def needs_human(self, threshold: float = LOW_CONFIDENCE_THRESHOLD) -> bool:
        return not self.pages or self.confidence < threshold

    def describe_quality(
        self, threshold: float = LOW_CONFIDENCE_THRESHOLD
    ) -> str:
        bad = self.unreadable_pages(threshold)
        if not bad:
            return (
                f"{self.page_count} page(s), worst page confidence "
                f"{self.confidence:.2f}"
            )
        numbers = ", ".join(str(p.number) for p in bad)
        return (
            f"page(s) {numbers} of {self.page_count} scored below {threshold:.2f} "
            f"(worst {self.confidence:.2f}); the document is partly unread and a "
            "person has to look at the image"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": os.path.basename(self.source),
            "pages": self.page_count,
            "engine": self.engine,
            "fallback_used": self.fallback_used,
            "confidence": round(self.confidence, 3),
            "mean_confidence": round(self.mean_confidence, 3),
            "sha256": self.sha256,
            "unreadable_pages": [p.number for p in self.unreadable_pages()],
            "errors": list(self.errors),
        }


class OCREngine(Protocol):
    name: str

    def available(self) -> bool: ...

    def read(self, path: str) -> OCRResult: ...


def _pages_from_scores(
    texts: Sequence[str], scores: Sequence[Sequence[float]], engine: str
) -> list[Page]:
    pages: list[Page] = []
    for index, text in enumerate(texts):
        page_scores = list(scores[index]) if index < len(scores) else []
        # An empty page is not a confident page. Averaging an empty list
        # returned 1.0 -- a perfectly confident reading of nothing.
        # An empty page is not a confident page, whatever the engine reported.
        # Deriving confidence from the score list alone let a third-party engine
        # return `texts=["", "real"]` with two scores for page one and produce a
        # blank page at 0.96 -- the stated invariant, broken by the one thing it
        # was written to prevent.
        confidence = (
            (sum(page_scores) / len(page_scores))
            if (page_scores and text.strip())
            else 0.0
        )
        pages.append(
            Page(number=index + 1, text=text, confidence=confidence, engine=engine)
        )
    return pages


class PaddleOCREngine:
    """PaddleOCR. Primary engine; imported only when actually used."""

    name = "paddleocr"

    def __init__(self, *, lang: str = "en", **options: Any) -> None:
        self.lang = lang
        self.options = options
        self._engine: Any = None

    def available(self) -> bool:
        try:  # pragma: no cover - depends on the deployment
            import paddleocr  # noqa: F401
        except Exception:  # noqa: BLE001
            return False
        return True

    def _load(self) -> Any:  # pragma: no cover - heavy optional dependency
        if self._engine is None:
            from paddleocr import PaddleOCR

            self._engine = PaddleOCR(
                use_angle_cls=True, lang=self.lang, show_log=False, **self.options
            )
        return self._engine

    def read(self, path: str) -> OCRResult:  # pragma: no cover - optional dep
        engine = self._load()
        raw = engine.ocr(path, cls=True)
        texts: list[str] = []
        scores: list[list[float]] = []
        for page in raw or []:
            lines = page or []
            texts.append("\n".join(str(line[1][0]) for line in lines))
            scores.append([float(line[1][1]) for line in lines])
        return OCRResult(
            source=path,
            pages=_pages_from_scores(texts, scores, self.name),
            engine=self.name,
        )


class TesseractEngine:
    """pytesseract. The fallback, and the one most likely to be installed."""

    name = "tesseract"

    def __init__(self, *, lang: str = "eng", dpi: int = 300) -> None:
        self.lang = lang
        self.dpi = dpi

    def available(self) -> bool:
        try:  # pragma: no cover - depends on the deployment
            import pytesseract  # noqa: F401
            from PIL import Image  # noqa: F401
        except Exception:  # noqa: BLE001
            return False
        return True

    def read(self, path: str) -> OCRResult:  # pragma: no cover - optional dep
        import pytesseract
        from PIL import Image, ImageSequence

        texts: list[str] = []
        scores: list[list[float]] = []
        with Image.open(path) as image:
            for frame in ImageSequence.Iterator(image):
                data = pytesseract.image_to_data(
                    frame, lang=self.lang, output_type=pytesseract.Output.DICT
                )
                words = [
                    (word, float(conf))
                    for word, conf in zip(data["text"], data["conf"])
                    # Tesseract emits -1 for whitespace boxes; averaging those
                    # in drags every page's confidence toward the floor.
                    if str(word).strip() and float(conf) >= 0
                ]
                texts.append(" ".join(w for w, _c in words))
                scores.append([c / 100.0 for _w, c in words])
        return OCRResult(
            source=path,
            pages=_pages_from_scores(texts, scores, self.name),
            engine=self.name,
        )


@dataclass
class ScriptedOCR:
    """A shipped test double, not a mock of this module's logic.

    Holds page text and per-page confidence for a named source, so the tests and
    the demo drive the real classification, matching, urgency and routing path
    without a fax machine or a 400MB model. Same role `EchoTransport` plays for
    the model layer.
    """

    name = "scripted"
    documents: dict[str, list[tuple[str, float]]] = field(default_factory=dict)

    def available(self) -> bool:
        return True

    def read(self, path: str) -> OCRResult:
        key = os.path.basename(path)
        if key not in self.documents:
            raise OCRUnavailable(f"no scripted OCR for {key!r}")
        pages = [
            Page(number=index + 1, text=text, confidence=confidence, engine=self.name)
            for index, (text, confidence) in enumerate(self.documents[key])
        ]
        return OCRResult(source=path, pages=pages, engine=self.name)


def chain(engines: Sequence[OCREngine], path: str) -> OCRResult:
    """Try each engine in order. Fall back on failure OR on low confidence.

    The second half of that is the point. An engine that is installed and
    running but reading a bad fax at 0.4 is not a success to be accepted just
    because it did not raise -- README I-06 asks for a fallback and this is when
    it earns its place. Whichever engine produced the best worst-page confidence
    wins, and the result records that a fallback ran so the pilot can measure
    how often the primary is losing.
    """
    if not engines:
        raise OCRUnavailable("no OCR engine configured")
    attempts: list[OCRResult] = []
    errors: list[str] = []
    for engine in engines:
        if not engine.available():
            errors.append(f"{engine.name}: not installed on this machine")
            continue
        try:
            result = engine.read(path)
        except Exception as exc:  # noqa: BLE001 - the next engine is the point
            errors.append(f"{engine.name}: {type(exc).__name__}: {exc}")
            continue
        # Counted against ATTEMPTS, not against position in the configured list.
        # Indexing the configured list recorded every document on a box without
        # PaddleOCR as a primary-engine failure, corrupting the one metric the
        # flag exists to produce.
        result.fallback_used = bool(attempts)
        attempts.append(result)
        if not result.needs_human():
            break
    if not attempts:
        raise OCRUnavailable(
            f"every OCR engine failed on {os.path.basename(path)}: {errors}. "
            "The document is not readable by this system and needs a person; it "
            "is NOT filed and NOT classified."
        )
    # PAGE COUNT FIRST, then confidence. Selecting on confidence alone returned
    # a one-page fallback over a five-page primary read the moment the primary
    # dipped under the threshold: four fifths of a discharge summary vanished,
    # `route()` treated it as a clean read, and the urgency scan ran over 20% of
    # the document with nothing recorded anywhere.
    pages_seen = {r.page_count for r in attempts}
    best = max(attempts, key=lambda r: (r.page_count, r.confidence, r.mean_confidence))
    best.errors.extend(errors)
    if len(pages_seen) > 1:
        best.errors.append(
            f"engines disagree on page count {sorted(pages_seen)}; kept the "
            f"{best.page_count}-page read from {best.engine}. A transmission "
            "whose length is uncertain needs a person to look at the image."
        )
        # Force the human queue: a document some engine thought was longer is a
        # document that may be missing pages.
        best.pages.append(
            Page(
                number=best.page_count + 1, text="",
                confidence=0.0, engine=best.engine,
                notes=("page-count disagreement between OCR engines",),
            )
        )
    elif len(attempts) > 1:
        best.errors.append(
            f"{len(attempts)} engine(s) tried; kept {best.engine} at "
            f"{best.confidence:.2f}"
        )
    return best
