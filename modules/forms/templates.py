"""Where every field on a form lives, and where its value comes from.

README I-01 is unusually direct about this one:

    Locate the fillable field coordinates on a known form | Template map (built
    once per form) | Pure deterministic. DO NOT USE AN LLM FOR THIS -- it is
    slower, costlier, and less reliable than a coordinate map.

So a template is data: a form type, a list of fields, and for each field a page,
a bounding box, the path into the chart record that supplies its value, and the
name of a transform. All of it lives in `config/forms/*.yaml` where a person can
read it, diff it, and sign off on it. There is no coordinate arithmetic in this
module and no field name as a Python literal.

THE PLACEHOLDER RULE, which is the whole reason this module has a gate.

The Illinois Certificate of Child Health Examination is a state document with a
fixed printed layout. The coordinates that put a value in the right box on that
document can only come from measuring the real PDF. This repo ships the field
LIST -- which is public, and is the part that takes a day to get right -- with
bounding boxes marked `placeholder: true`.

A placeholder box is not a small inaccuracy. A tetanus date written 40 points
too low lands in the next row of the immunization grid, which means the form
says the child had a dose they did not have, on a legal school document, in a
box a physician then signs. That is a worse failure than not filling the form at
all, and it is invisible on a screen at review size.

`TemplateStore.load()` therefore REFUSES a placeholder template unless the
caller passes `allow_placeholder=True`, exactly as `ProtocolRegistry` refuses
placeholder protocol identifiers in I-04. Calibration is a deployment task with
a name and a date, not a thing that quietly never happens.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from .chart import is_known_path

__all__ = [
    "DEFAULT_FORMS_DIR",
    "TRANSFORMS",
    "BoundingBox",
    "FieldSpec",
    "FormTemplate",
    "TemplateStore",
    "UncalibratedTemplate",
    "UnknownTransform",
    "TemplateInvalid",
    "register_transform",
]

DEFAULT_FORMS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "forms",
)

#: Field kinds. `checkbox` and `signature` are called out because they are the
#: two the filler must never treat as text: a checkbox gets a mark or nothing,
#: and a signature field is never machine-written at all.
FIELD_KINDS: frozenset[str] = frozenset(
    {"text", "date", "number", "checkbox", "signature", "grid_date"}
)


class UncalibratedTemplate(RuntimeError):
    """Raised when a template with placeholder coordinates is asked to fill."""


class UnknownTransform(RuntimeError):
    """Raised when a template names a transform this build does not have."""


class TemplateInvalid(ValueError):
    """Raised when a template file is internally inconsistent."""


# -- transforms --------------------------------------------------------------
#
# A transform turns a chart value into the string that goes in the box. They are
# named in YAML and resolved here, so a template edit is a data edit. Each one
# is total: given a value it cannot render, it returns None, and a field whose
# transform returns None is left BLANK and reported -- never filled with a guess
# or an empty-looking placeholder like "N/A", which on a school form reads as a
# positive assertion that there is nothing to report.

TRANSFORMS: dict[str, Callable[[Any], str | None]] = {}


def register_transform(name: str) -> Callable[
    [Callable[[Any], str | None]], Callable[[Any], str | None]
]:
    def decorate(fn: Callable[[Any], str | None]) -> Callable[[Any], str | None]:
        TRANSFORMS[name] = fn
        return fn

    return decorate


@register_transform("verbatim")
def _verbatim(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@register_transform("date_mm_dd_yyyy")
def _date_mm_dd_yyyy(value: Any) -> str | None:
    """US convention, which is what an Illinois school form is printed for."""
    if isinstance(value, date):
        return value.strftime("%m/%d/%Y")
    return None


@register_transform("date_mm_yyyy")
def _date_mm_yyyy(value: Any) -> str | None:
    """A month-precision dose. The form gets `06/2024`, NOT `06/01/2024`.

    Anchoring a month-precision registry entry to the first of the month is a
    modelling convenience inside I-02. Printing that day on a legal document
    turns an approximation into an assertion.
    """
    if isinstance(value, date):
        return value.strftime("%m/%Y")
    return None


@register_transform("dose_date")
def _dose_date(value: Any) -> str | None:
    """One immunization dose -> the date as the form should print it.

    Takes the DOSE, not the date, because how a date should be printed depends
    on the dose's own precision and nothing else knows that. I-02 anchors a
    month-precision registry entry to the first of the month so date arithmetic
    works; printing `06/01/2024` on a school form turns that convenience into a
    claim about the day, and a school comparing it against its own record sees a
    mismatch.

    A dose the reconciliation could not settle renders as None -- the box is
    left blank and the discrepancy list says why. `fill.py` enforces that too;
    this is the second of the two locks.
    """
    if not isinstance(value, Mapping):
        return None
    if value.get("disputed"):
        return None
    given = value.get("given")
    if not isinstance(given, date):
        return None
    if str(value.get("precision", "day")) == "month":
        return given.strftime("%m/%Y")
    return given.strftime("%m/%d/%Y")


@register_transform("integer")
def _integer(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return str(int(round(float(value))))
    except (TypeError, ValueError):
        return None


@register_transform("one_decimal")
def _one_decimal(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return None


@register_transform("blood_pressure")
def _blood_pressure(value: Any) -> str | None:
    """`{"systolic": 104, "diastolic": 66}` -> `104/66`. Both or neither.

    A half-rendered blood pressure ("104/") is worse than a blank: it looks
    measured.
    """
    if not isinstance(value, Mapping):
        return None
    systolic, diastolic = value.get("systolic"), value.get("diastolic")
    if systolic is None or diastolic is None:
        return None
    left, right = _integer(systolic), _integer(diastolic)
    if left is None or right is None:
        return None
    return f"{left}/{right}"


@register_transform("pass_fail")
def _pass_fail(value: Any) -> str | None:
    """A screening result, from a closed vocabulary.

    Anything this does not recognise returns None and the field goes to a human.
    A vision screening rendered as the raw chart string could put "wnl", "20/20
    OU" or "not done" into a box the school reads as pass/fail.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    return {
        "pass": "Pass", "passed": "Pass", "normal": "Pass",
        "fail": "Fail", "failed": "Fail", "abnormal": "Fail",
        "referred": "Fail",
    }.get(text)


@register_transform("comma_list")
def _comma_list(value: Any) -> str | None:
    """A list of strings. An EMPTY list returns None, deliberately.

    "No known allergies" is a clinical assertion a person makes, not something
    an empty query result is entitled to write onto a state form. An empty list
    leaves the box blank and the reviewer fills it.
    """
    if not isinstance(value, (list, tuple)):
        return None
    items = [str(v).strip() for v in value if str(v).strip()]
    return ", ".join(items) if items else None


@register_transform("yes_no")
def _yes_no(value: Any) -> str | None:
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return None


# -- the template ------------------------------------------------------------


@dataclass(frozen=True)
class BoundingBox:
    """A box on a page, in PDF points, origin top-left."""

    page: int
    x: float
    y: float
    width: float
    height: float
    #: True when these numbers were guessed rather than measured off the real
    #: form. See the module docstring; this is why the store has a gate.
    placeholder: bool = False

    @property
    def rect(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def as_dict(self) -> dict[str, Any]:
        return {
            "page": self.page, "x": self.x, "y": self.y,
            "width": self.width, "height": self.height,
            "placeholder": self.placeholder,
        }


@dataclass(frozen=True)
class FieldSpec:
    """One box on the form and where its value comes from."""

    name: str
    kind: str
    box: BoundingBox
    #: Dotted path into the `ChartRecord` mapping, e.g. `vitals.height_cm`.
    #: Empty for fields nothing in the chart supplies.
    source: str = ""
    transform: str = "verbatim"
    #: The form itself requires this. A required field left blank blocks release
    #: rather than producing a partly-filled document somebody signs anyway.
    required: bool = False
    #: A field a machine must never write. See `machine_writable`.
    human_only: bool = False
    label: str = ""

    @property
    def machine_writable(self) -> bool:
        """False for signatures and for anything marked human-only.

        A signature block is the physician's attestation. Rendering anything
        into it -- even the physician's own name pulled from a staff table --
        manufactures an attestation, which is the one thing this whole pipeline
        exists to keep a human doing.
        """
        return not self.human_only and self.kind != "signature"

    def render(self, value: Any) -> str | None:
        fn = TRANSFORMS.get(self.transform)
        if fn is None:
            raise UnknownTransform(
                f"field {self.name!r} names transform {self.transform!r}, which "
                f"this build does not have. Known: {sorted(TRANSFORMS)}"
            )
        return fn(value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "kind": self.kind, "source": self.source,
            "transform": self.transform, "required": self.required,
            "human_only": self.human_only, "label": self.label,
            "box": self.box.as_dict(),
        }


@dataclass(frozen=True)
class FormTemplate:
    """One form type: its identity, its anchor text, and its fields."""

    form_type: str
    title: str
    version: str
    issuer: str
    page_count: int
    fields: tuple[FieldSpec, ...]
    #: Phrases that appear on the printed form and nowhere else. `detect.py`
    #: scores an incoming scan against these; they are the deterministic half of
    #: classification, and they are data for the same reason the boxes are.
    anchors: tuple[str, ...] = ()
    #: Free text naming who measured the coordinates and when. Empty on a
    #: placeholder template.
    calibration: str = ""
    source_pdf: str = ""

    @property
    def is_placeholder(self) -> bool:
        return any(f.box.placeholder for f in self.fields)

    @property
    def placeholder_fields(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields if f.box.placeholder)

    def field(self, name: str) -> FieldSpec:
        for spec in self.fields:
            if spec.name == name:
                return spec
        raise KeyError(name)

    @property
    def machine_fields(self) -> tuple[FieldSpec, ...]:
        return tuple(f for f in self.fields if f.machine_writable)

    @property
    def required_fields(self) -> tuple[FieldSpec, ...]:
        return tuple(f for f in self.fields if f.required)

    def as_dict(self) -> dict[str, Any]:
        return {
            "form_type": self.form_type, "title": self.title,
            "version": self.version, "issuer": self.issuer,
            "page_count": self.page_count, "anchors": list(self.anchors),
            "calibration": self.calibration,
            "is_placeholder": self.is_placeholder,
            "fields": [f.as_dict() for f in self.fields],
        }


def _template_from_mapping(data: Mapping[str, Any], origin: str) -> FormTemplate:
    try:
        form_type = str(data["form_type"])
        page_count = int(data["page_count"])
        raw_fields = list(data["fields"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TemplateInvalid(f"{origin}: {exc}") from exc

    seen: set[str] = set()
    fields: list[FieldSpec] = []
    for raw in raw_fields:
        name = str(raw["name"])
        if name in seen:
            # Two boxes with one name means one of them is silently unfillable,
            # and which one depends on dictionary order.
            raise TemplateInvalid(f"{origin}: duplicate field {name!r}")
        seen.add(name)
        kind = str(raw.get("kind", "text"))
        if kind not in FIELD_KINDS:
            raise TemplateInvalid(
                f"{origin}: field {name!r} has kind {kind!r}; known: "
                f"{sorted(FIELD_KINDS)}"
            )
        box_raw = raw["box"]
        page = int(box_raw["page"])
        if not 1 <= page <= page_count:
            raise TemplateInvalid(
                f"{origin}: field {name!r} is on page {page} of a "
                f"{page_count}-page form"
            )
        width = float(box_raw["width"])
        height = float(box_raw["height"])
        if width <= 0 or height <= 0:
            raise TemplateInvalid(
                f"{origin}: field {name!r} has a box of zero or negative size; "
                "a value written into it would be invisible"
            )
        transform = str(raw.get("transform", "verbatim"))
        if transform not in TRANSFORMS:
            raise UnknownTransform(
                f"{origin}: field {name!r} names transform {transform!r}, which "
                f"this build does not have. Known: {sorted(TRANSFORMS)}"
            )
        spec = FieldSpec(
            name=name,
            kind=kind,
            box=BoundingBox(
                page=page,
                x=float(box_raw["x"]), y=float(box_raw["y"]),
                width=width, height=height,
                placeholder=bool(box_raw.get("placeholder", False)),
            ),
            source=str(raw.get("source", "")),
            transform=transform,
            required=bool(raw.get("required", False)),
            human_only=bool(raw.get("human_only", False)),
            label=str(raw.get("label", "")),
        )
        if spec.required and not spec.machine_writable and not spec.human_only:
            raise TemplateInvalid(
                f"{origin}: field {name!r} is required and is a signature; mark "
                "it human_only so the release gate asks a person for it rather "
                "than blocking on a box nothing can fill"
            )
        if spec.machine_writable and spec.source and spec.kind == "signature":
            raise TemplateInvalid(
                f"{origin}: field {name!r} is a signature with a chart source"
            )
        if spec.source and not is_known_path(spec.source):
            # LOAD TIME, not fill time. A mistyped source path used to resolve to
            # None, and the box was skipped with the reason "the chart holds no
            # value for this field" -- a statement about the child, shown to the
            # MA, describing a configuration bug. Nobody would ever have found it.
            raise TemplateInvalid(
                f"{origin}: field {name!r} sources {spec.source!r}, which is not "
                "a path a chart record has. See CHART_SCHEMA in "
                "modules/forms/chart.py."
            )
        fields.append(spec)

    if not fields:
        raise TemplateInvalid(f"{origin}: template has no fields")

    return FormTemplate(
        form_type=form_type,
        title=str(data.get("title", form_type)),
        version=str(data.get("version", "unversioned")),
        issuer=str(data.get("issuer", "")),
        page_count=page_count,
        fields=tuple(fields),
        anchors=tuple(str(a) for a in data.get("anchors", ())),
        calibration=str(data.get("calibration", "")),
        source_pdf=str(data.get("source_pdf", "")),
    )


@dataclass
class TemplateStore:
    """The form library. Loaded from YAML, gated on calibration."""

    templates: dict[str, FormTemplate] = field(default_factory=dict)
    directory: str = ""
    allow_placeholder: bool = False

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike[str] = DEFAULT_FORMS_DIR,
        *,
        allow_placeholder: bool = False,
    ) -> "TemplateStore":
        """Read every `*.yaml` in `directory`.

        `allow_placeholder` is an explicit, per-call opt-in for demos and tests.
        A deployment that passes it has said in writing that it is filling a
        legal document with guessed coordinates.
        """
        store = cls(directory=str(directory), allow_placeholder=allow_placeholder)
        names = sorted(
            name for name in os.listdir(directory)
            if name.endswith((".yaml", ".yml")) and not name.startswith("_")
        )
        for name in names:
            path = os.path.join(str(directory), name)
            with open(path, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
            template = _template_from_mapping(data or {}, origin=name)
            if template.form_type in store.templates:
                raise TemplateInvalid(
                    f"{name}: form_type {template.form_type!r} is already "
                    "defined by another file"
                )
            store.templates[template.form_type] = template
        if not store.templates:
            raise TemplateInvalid(f"no form templates found in {directory}")
        return store

    def add(self, template: FormTemplate, *, confirmed_by: str) -> None:
        """Add a template to the library. A NAME IS REQUIRED.

        `detect.py` can propose a template for an unseen form. It cannot call
        this. Nothing in this repo calls this without a human identifier, which
        is the mechanical expression of README I-01's "new template proposed for
        human confirmation and permanent addition to the library".
        """
        if not confirmed_by.strip():
            raise ValueError(
                "a template enters the library only when a person confirms it; "
                "pass the confirming user's identifier"
            )
        if template.form_type in self.templates:
            raise TemplateInvalid(
                f"form_type {template.form_type!r} is already in the library"
            )
        self.templates[template.form_type] = template

    def get(self, form_type: str) -> FormTemplate:
        return self.templates[form_type]

    def for_filling(self, form_type: str) -> FormTemplate:
        """The template, or a refusal. This is the gate the filler calls."""
        template = self.templates[form_type]
        if template.is_placeholder and not self.allow_placeholder:
            raise UncalibratedTemplate(
                f"template {form_type!r} has placeholder coordinates for "
                f"{len(template.placeholder_fields)} field(s) "
                f"({', '.join(template.placeholder_fields[:4])}...). Measure them "
                f"against the real {template.issuer or 'issuer'} PDF and record "
                "who did it in `calibration:` before filling anything. A value "
                "written 40 points off lands in the next row of the immunization "
                "grid, and a physician signs it."
            )
        return template

    def known_types(self) -> list[str]:
        return sorted(self.templates)

    def calibration_report(self) -> list[dict[str, Any]]:
        """What still needs measuring. Run this before go-live."""
        return [
            {
                "form_type": t.form_type,
                "calibrated": not t.is_placeholder,
                "calibration": t.calibration,
                "placeholder_fields": list(t.placeholder_fields),
                "field_count": len(t.fields),
            }
            for t in sorted(self.templates.values(), key=lambda t: t.form_type)
        ]
