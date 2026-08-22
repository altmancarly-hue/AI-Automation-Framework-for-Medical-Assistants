"""Writing values onto the form, with a highlight on every one of them.

README I-01, target state, step 6, emphasis in the original:

    Form is rendered as a filled PDF. **Every auto-filled field is visually
    highlighted** so the reviewing MA sees exactly what the machine wrote.

That sentence is a control, not a feature, and this module treats it as one. The
highlight is not something the caller asks for; it is emitted by the same code
path that writes the text, in the same loop, and `FilledForm.verify()` refuses to
release a document where the two counts disagree. There is no `highlight=False`
parameter, because the only reason to want one is to make machine-written text
indistinguishable from a clinician's, and that is the failure the highlight
exists to prevent.

FOUR THINGS THIS MODULE REFUSES TO DO.

  1. **Write into a signature field.** Ever, for any reason. A signature is an
     attestation and the whole pipeline exists to keep a person making it.
     `FieldSpec.machine_writable` is false for signature fields and `_write` is
     never reached for them.
  2. **Write a disputed immunization dose.** `reconcile.py` marks a dose
     disputed when the chart and the registry cannot be squared. The transform
     already renders those as None; this module checks again, because this is
     the check whose failure puts a wrong date on a legal school document, and
     one lock on that door is not enough.
  3. **Write anything a transform could not render.** A transform that returns
     None leaves the box BLANK and records why. It never falls back to the raw
     value, an empty string, or "N/A" -- on a school form a filled-looking box is
     read as an assertion, and "N/A" in the allergy box is a clinical claim.
  4. **Overflow a box.** Text wider than its box is truncated in the PDF but
     recorded at full length and flagged, because an allergy list silently cut
     at "penicillin, peanut, latex" reads as complete.

THE PDF LIBRARY IS BEHIND AN INTERFACE. `PdfBackend` has three methods.
`PyMuPDFBackend` implements them with a lazy import; `RecordingBackend` is a
shipped test double that records the same calls without a PDF. That is not
ceremony -- README I-01's risk table lists "vendor lock on the OCR layer" and the
same argument applies here, and more practically it means the entire fill,
verify and audit path is testable on a machine with no PyMuPDF.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Protocol, Sequence

from .blankforms import PdfUnavailable
from .chart import ChartRecord, MissingPath, SourceValue
from .templates import BoundingBox, FieldSpec, FormTemplate

__all__ = [
    "FieldWrite",
    "FilledForm",
    "FormFiller",
    "PdfBackend",
    "RecordingBackend",
    "PyMuPDFBackend",
    "HighlightMissing",
    "SignatureWriteAttempted",
    "TextOverflow",
    "HIGHLIGHT_RGB",
]

#: The highlight colour. Deliberately a strong yellow rather than a subtle tint:
#: a highlight the reviewer's eye can skip is a highlight that is not doing its
#: job on a page with sixty of them.
HIGHLIGHT_RGB: tuple[float, float, float] = (1.0, 0.92, 0.23)

#: Rough width of one character at a given font size, for overflow detection.
#: Helvetica averages about 0.5 em across mixed-case text; 0.55 is deliberately
#: pessimistic so the flag fires slightly early rather than slightly late.
_CHAR_WIDTH_RATIO = 0.55


class HighlightMissing(RuntimeError):
    """Raised when a filled form has a written field with no highlight.

    This cannot be caused by a caller. It fires when the fill loop and the
    highlight loop have been allowed to diverge -- which is the bug that turns
    machine-written text into text nobody can distinguish from a clinician's.
    """


class SignatureWriteAttempted(RuntimeError):
    """Raised when anything tries to machine-write a signature field."""


class TextOverflow(RuntimeError):
    """Raised when text reaches the renderer wider than the box it goes in."""


@dataclass(frozen=True)
class FieldWrite:
    """One value written onto the form, and everything the audit needs.

    README I-01's audit requirement, verbatim: "every field write records source,
    method (auto/manual), confidence, reviewing user, timestamp". `reviewed_by`
    is filled in later, by `review.py`, when a person actually signs off -- it is
    not defaulted to a system identifier here, because a system identifier in a
    reviewer column is how an unreviewed form comes to look reviewed.
    """

    field_name: str
    text: str
    box: BoundingBox
    #: "auto" | "manual" | "probe"
    method: str
    #: The template's dotted chart path. Carried so the release gate can match a
    #: stale chart value to the box it filled EXACTLY, rather than guessing from
    #: a recorded date -- which matched the wrong field whenever two values
    #: shared an observation date, i.e. constantly, since a whole visit's vitals
    #: are recorded on one day.
    source_path: str = ""
    source_system: str = ""
    source_resource: str = ""
    source_recorded: date | None = None
    #: Set when the rendered text was wider than its box.
    truncated_from: str = ""
    #: Set on a synthetic-probe write. See `probe.py`.
    synthetic: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field_name,
            "text": self.text,
            "method": self.method,
            "page": self.box.page,
            "source_path": self.source_path,
            "source_system": self.source_system,
            "source_resource": self.source_resource,
            "source_recorded": (
                self.source_recorded.isoformat() if self.source_recorded else None
            ),
            "truncated_from": self.truncated_from,
            "synthetic": self.synthetic,
        }


@dataclass(frozen=True)
class SkippedField:
    """A field the machine did not fill, and why. Every one is reported."""

    field_name: str
    reason: str
    required: bool
    human_only: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field_name,
            "reason": self.reason,
            "required": self.required,
            "human_only": self.human_only,
        }


@dataclass
class FilledForm:
    """The rendered document plus a complete account of what went onto it."""

    form_type: str
    patient_id: str
    template_version: str
    output_path: str = ""
    writes: list[FieldWrite] = field(default_factory=list)
    skipped: list[SkippedField] = field(default_factory=list)
    highlights: list[str] = field(default_factory=list)
    generated_utc: datetime | None = None
    #: Carried through from the reconciliation so the review screen and the
    #: release gate see the same list.
    discrepancies: list[dict[str, Any]] = field(default_factory=list)

    @property
    def auto_filled(self) -> list[FieldWrite]:
        return [w for w in self.writes if w.method == "auto"]

    @property
    def truncated(self) -> list[FieldWrite]:
        return [w for w in self.writes if w.truncated_from]

    @property
    def has_synthetic_write(self) -> bool:
        """True while a deliberately wrong probe value is ON THIS DOCUMENT.

        The release gate blocks on this rather than on a flag in the probe
        registry. Scoring a probe records what the reviewer did; it does not
        take the wrong value off the page, and an earlier version let a caller
        clear the registry flag and sign the document with the injected error
        still printed on it.
        """
        return any(w.synthetic for w in self.writes)

    def missing_required(self) -> list[SkippedField]:
        """Required boxes nothing filled, EXCLUDING the ones a person must fill.

        A signature block is required and empty on every machine-filled form;
        that is the correct state, not a defect, and listing it here would train
        the reviewer to ignore this list.
        """
        return [s for s in self.skipped if s.required and not s.human_only]

    def verify(self) -> None:
        """Every written field is highlighted, and no signature was written.

        Called by `FormFiller.fill` before it returns, so a caller cannot get an
        unverified `FilledForm` out of this module at all.
        """
        written = {w.field_name for w in self.writes}
        highlighted = set(self.highlights)
        if written != highlighted:
            missing = sorted(written - highlighted)
            extra = sorted(highlighted - written)
            raise HighlightMissing(
                "every machine-written field must carry a highlight annotation "
                f"(README I-01 step 6). Written but not highlighted: {missing}. "
                f"Highlighted but not written: {extra}."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "form_type": self.form_type,
            "patient_id": self.patient_id,
            "template_version": self.template_version,
            "output_path": self.output_path,
            "generated_utc": (
                self.generated_utc.isoformat() if self.generated_utc else None
            ),
            "auto_filled_count": len(self.auto_filled),
            "writes": [w.as_dict() for w in self.writes],
            "skipped": [s.as_dict() for s in self.skipped],
            "truncated": [w.field_name for w in self.truncated],
            "has_synthetic_write": self.has_synthetic_write,
            "missing_required": [s.field_name for s in self.missing_required()],
            "discrepancies": list(self.discrepancies),
        }


# -- the PDF layer -----------------------------------------------------------


class PdfBackend(Protocol):
    """Three operations. Everything else about PDFs stays out of this module."""

    name: str

    def open(self, source: str) -> None: ...
    def measure_text(self, text: str, *, font_size: float) -> float: ...
    def write_text(self, box: BoundingBox, text: str, *, font_size: float) -> None: ...
    def highlight(self, box: BoundingBox) -> None: ...
    def save(self, destination: str) -> None: ...


@dataclass
class RecordingBackend:
    """A shipped test double. Records the calls; renders nothing.

    Not a mock of this module's logic -- the fill loop, the transforms, the
    dispute checks, the overflow detection and `verify()` are all the real code.
    This only stands in for the PDF library, the way `EchoTransport` stands in
    for a model server.
    """

    name: str = "recording"
    opened: str = ""
    texts: list[tuple[str, str]] = field(default_factory=list)
    boxes: list[BoundingBox] = field(default_factory=list)
    highlighted: list[BoundingBox] = field(default_factory=list)
    saved: str = ""

    def open(self, source: str) -> None:
        self.opened = source

    def measure_text(self, text: str, *, font_size: float) -> float:
        """The 0.55-em heuristic, kept HERE where it is only a test double.

        It used to be the real rule, and it was permissive: Helvetica capitals
        run about 0.68 em, so an upper-case allergy list passed the estimate and
        then rendered past the right-hand rule.
        """
        return len(text) * font_size * _CHAR_WIDTH_RATIO

    def write_text(self, box: BoundingBox, text: str, *, font_size: float) -> None:
        self.texts.append((f"p{box.page}:{box.x},{box.y}", text))
        self.boxes.append(box)

    def highlight(self, box: BoundingBox) -> None:
        self.highlighted.append(box)

    def save(self, destination: str) -> None:
        self.saved = destination


@dataclass
class PyMuPDFBackend:
    """The real one. `fitz` is imported inside `open`, never at module scope."""

    name: str = "pymupdf"
    font_name: str = "helv"
    _doc: Any = None
    _fitz: Any = None

    def open(self, source: str) -> None:
        try:
            import fitz  # noqa: PLC0415 - deliberate lazy import
        except ImportError as exc:  # pragma: no cover - depends on the machine
            raise PdfUnavailable(
                "PyMuPDF is not installed, so the form cannot be rendered. "
                "`pip install pymupdf`. Everything up to rendering -- the chart "
                "pull, the reconciliation, the discrepancy list -- has already "
                "run and is in the FilledForm."
            ) from exc
        self._fitz = fitz
        self._doc = fitz.open(source)

    def measure_text(self, text: str, *, font_size: float) -> float:
        """The real rendered width, from the font metrics."""
        if self._fitz is None:
            try:
                import fitz  # noqa: PLC0415 - deliberate lazy import
            except ImportError as exc:  # pragma: no cover - depends on machine
                raise PdfUnavailable("PyMuPDF is not installed") from exc
            self._fitz = fitz
        return float(
            self._fitz.get_text_length(text, fontname=self.font_name, fontsize=font_size)
        )

    def _page(self, number: int) -> Any:
        if self._doc is None:
            raise RuntimeError("open() the blank form before writing to it")
        if not 1 <= number <= self._doc.page_count:
            raise IndexError(
                f"the template puts a field on page {number} of a "
                f"{self._doc.page_count}-page PDF; the blank form and the "
                "template do not match"
            )
        return self._doc[number - 1]

    def write_text(self, box: BoundingBox, text: str, *, font_size: float) -> None:
        page = self._page(box.page)
        # POST-CONDITION, checked against the font metrics rather than trusted
        # from the caller. `insert_text` draws happily past the right-hand rule,
        # so a value the width check let through rendered over the next column --
        # on an immunization grid, over the neighbouring antigen's box, outside
        # the highlight, and invisible to anything reading the box.
        #
        # `_fit` measures with the same function, so this can only fire when the
        # two have been allowed to disagree. Loud is the right failure: silent
        # ink outside its box is the one this replaced.
        available = box.width - 4.0
        width = self.measure_text(text, font_size=font_size)
        if width > available:
            raise TextOverflow(
                f"{text!r} renders {width:.1f}pt wide into a {available:.1f}pt "
                f"box. It was not trimmed before it reached the renderer."
            )
        page.insert_text(
            (box.x + 2.0, box.y + box.height - 3.5),
            text,
            fontsize=font_size,
            fontname=self.font_name,
            color=(0.0, 0.0, 0.45),
        )

    def highlight(self, box: BoundingBox) -> None:
        page = self._page(box.page)
        x0, y0, x1, y1 = box.rect
        annotation = page.add_highlight_annot(self._fitz.Rect(x0, y0, x1, y1))
        annotation.set_colors(stroke=HIGHLIGHT_RGB)
        # The annotation carries its own explanation, so the highlight survives
        # printing to paper as a visible tint AND survives being opened in any
        # PDF reader as a note saying what put it there.
        annotation.set_info(
            title="NSP forms pipeline",
            content="Auto-filled from the chart. Verify before signing.",
        )
        annotation.update()

    def save(self, destination: str) -> None:
        if self._doc is None:
            raise RuntimeError("nothing to save")
        self._doc.save(destination)
        self._doc.close()
        self._doc = None


# -- the filler --------------------------------------------------------------


@dataclass
class FormFiller:
    """Walks a template, resolves each field, writes it, highlights it."""

    backend: Any
    audit: Any = None
    font_size: float = 8.5
    INITIATIVE: str = "I-01"

    def fill(
        self,
        template: FormTemplate,
        record: ChartRecord,
        *,
        blank_pdf: str,
        destination: str,
        now: datetime,
        discrepancies: Sequence[Mapping[str, Any]] = (),
        overrides: Mapping[str, str] | None = None,
        probe_field: str = "",
        user_id: str = "system:forms",
        dry_run: bool = False,
    ) -> FilledForm:
        """Fill `template` from `record` and return a verified `FilledForm`.

        `overrides` is how a synthetic probe injects its deliberate error and
        how a reviewer's correction is re-rendered. An overridden field is
        written with `method="probe"` when it is the probe field, so nothing
        downstream can confuse the two.

        `dry_run` runs the whole resolution, render, truncation and verify path
        against a throwaway backend, producing a `FilledForm` and NOTHING ELSE:
        no file on disk and no audit event. `probe.py` needs the rendered text of
        every field to choose a target, and without this it got that by doing a
        real fill -- which left a stray `.probe-scan` PDF beside every probed
        form and wrote a second `form_filled` audit row for a document that was
        never produced. An audit log with phantom fills in it is not an audit log.
        """
        if template.is_placeholder:
            # Checked HERE as well as in `TemplateStore.for_filling`. A caller
            # holding a template from `store.get()` bypassed the store's gate
            # entirely and filled a legal document from guessed coordinates.
            from .templates import UncalibratedTemplate

            raise UncalibratedTemplate(
                f"template {template.form_type!r} has placeholder coordinates "
                f"for {len(template.placeholder_fields)} field(s) and will not "
                "be filled. Measure them against the real PDF first."
            )
        overrides = dict(overrides or {})
        backend: Any = RecordingBackend() if dry_run else self.backend
        filled = FilledForm(
            form_type=template.form_type,
            patient_id=record.patient_id,
            template_version=template.version,
            generated_utc=now,
            discrepancies=[dict(d) for d in discrepancies],
        )
        backend.open(blank_pdf)

        for spec in template.fields:
            if not spec.machine_writable:
                filled.skipped.append(
                    SkippedField(
                        field_name=spec.name,
                        reason=(
                            "signature fields are never machine-written"
                            if spec.kind == "signature"
                            else "this field is marked human-only"
                        ),
                        required=spec.required,
                        human_only=True,
                    )
                )
                continue

            if spec.name in overrides:
                text = overrides[spec.name]
                source = None
                method = "probe" if spec.name == probe_field else "manual"
            else:
                method = "auto"
                try:
                    source = record.resolve(spec.source) if spec.source else None
                except MissingPath as exc:
                    # A template that names a path the chart shape does not have
                    # is a configuration bug, and it must be loud. Filling the
                    # rest of the form and leaving this box blank would look
                    # like a patient with no data.
                    raise
                if source is None:
                    filled.skipped.append(
                        SkippedField(
                            spec.name,
                            "the chart holds no value for this field",
                            spec.required, False,
                        )
                    )
                    continue
                if _is_disputed(source.value):
                    # The second lock. `templates.dose_date` already renders a
                    # disputed dose as None; this is the check that does not
                    # depend on which transform a template happened to name.
                    filled.skipped.append(
                        SkippedField(
                            spec.name,
                            "the chart and the registry do not agree on this "
                            "dose; it is left blank and listed as a discrepancy",
                            spec.required, False,
                        )
                    )
                    continue
                text = spec.render(source.value)
                if text is None:
                    filled.skipped.append(
                        SkippedField(spec.name, _blank_reason(spec, source),
                                     spec.required, False)
                    )
                    continue

            if spec.kind == "signature":  # pragma: no cover - unreachable by design
                raise SignatureWriteAttempted(
                    f"something tried to write into signature field {spec.name!r}"
                )

            shown, truncated_from = _fit(text, spec.box, self.font_size, backend)
            backend.write_text(spec.box, shown, font_size=self.font_size)
            # SAME LOOP as the write. Not a second pass over the writes list --
            # a second pass is a thing that can be skipped, reordered, or made
            # conditional, and this one must not be.
            backend.highlight(spec.box)
            filled.highlights.append(spec.name)
            filled.writes.append(
                FieldWrite(
                    field_name=spec.name,
                    text=shown,
                    box=spec.box,
                    method=method,
                    # ALWAYS the template's path, override or not. An earlier
                    # version left it empty on an override, so probing a box
                    # erased the evidence that a stale chart value had reached
                    # the form -- missing data widening a safety threshold.
                    source_path=spec.source,
                    source_system=source.system if source else "manual",
                    source_resource=source.resource if source else "",
                    source_recorded=source.recorded if source else None,
                    truncated_from=truncated_from,
                    synthetic=method == "probe",
                )
            )

        filled.verify()
        backend.save(destination)
        if dry_run:
            return filled
        filled.output_path = destination

        if self.audit is not None:
            self.audit.record_event(
                # NOT "form_filled": `FormTracker` emits `form_<state>` and its
                # FILLED state is `form_filled`. Two different events under one
                # name make the audit log unqueryable for either of them.
                event_type="form_rendered",
                actor_id=user_id,
                initiative_id=self.INITIATIVE,
                detail={
                    "form_type": template.form_type,
                    "template_version": template.version,
                    "patient_id": record.patient_id,
                    "auto_filled": len(filled.auto_filled),
                    "skipped": len(filled.skipped),
                    "truncated": [w.field_name for w in filled.truncated],
                    "missing_required": [
                        s.field_name for s in filled.missing_required()
                    ],
                    "discrepancies": len(filled.discrepancies),
                },
                patient_id=record.patient_id,
            )
        return filled


def _blank_reason(spec: FieldSpec, source: SourceValue) -> str:
    """Why a resolved value produced no text. The distinction matters.

    An EMPTY list is not a failure to render -- it is a chart that says nothing
    here, and the box stays blank because "no known allergies" is a clinical
    assertion a person makes. Reporting that as "the transform could not render
    it" sends the MA looking for a bug instead of at the allergy box.
    """
    value = source.value
    if isinstance(value, (list, tuple)) and not value:
        return (
            "the chart holds an empty list for this field. It is left blank "
            "rather than filled with 'none': on a school form that is a "
            "clinical assertion, and it is the reviewer's to make"
        )
    return (
        f"the chart value {value!r} could not be rendered by transform "
        f"{spec.transform!r}; left blank rather than guessed"
    )


def _is_disputed(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value.get("disputed"))


def _fit(
    text: str, box: BoundingBox, font_size: float, backend: Any
) -> tuple[str, str]:
    """Trim text to its box, reporting the original when it did not fit.

    Returns `(shown, truncated_from)`. `truncated_from` is empty when the whole
    string fitted. A truncated allergy list looks complete on the page, so the
    review screen has to be told, and `review.py` treats it as blocking.

    MEASURED, not estimated. The estimate this replaced modelled every character
    as 0.55 em; Helvetica capitals are nearer 0.68, so an upper-case allergy list
    of 67 characters passed a 69-character "capacity" and then rendered 17 points
    past the right-hand rule -- unflagged, unblocked, and outside the highlight.
    """
    available = box.width - 4.0
    if available <= 0:  # pragma: no cover - templates refuse zero-size boxes
        return "", text
    try:
        width = backend.measure_text(text, font_size=font_size)
    except Exception:  # noqa: BLE001 - a backend without metrics is not a reason
        # ...to render an unbounded string. Fall back to the pessimistic
        # heuristic rather than to "it fits".
        width = len(text) * font_size * _CHAR_WIDTH_RATIO
    if width <= available:
        return text, ""

    # An ellipsis rather than a clean cut, so the page itself shows something is
    # missing even if nobody reads the review screen.
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = text[:middle] + "\u2026"
        try:
            candidate_width = backend.measure_text(candidate, font_size=font_size)
        except Exception:  # noqa: BLE001
            candidate_width = len(candidate) * font_size * _CHAR_WIDTH_RATIO
        if candidate_width <= available:
            low = middle
        else:
            high = middle - 1
    return (text[:low] + "\u2026") if low else "\u2026", text
