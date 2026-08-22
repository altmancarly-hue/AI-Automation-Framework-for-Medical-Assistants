"""Which form is this, and what to do when the answer is "none of them".

README I-01 splits this task in two and puts the split in the tool table:

    Read a scanned/faxed blank form and identify its type | OCR + classifier |
    Deterministic once you have a template library. LLM helps only for unseen
    form types.

    Handle a novel form the system has never seen | LLM vision model | This is
    where an LLM is genuinely superior.

So: anchor-phrase scoring against the library first, and a model only when that
comes back empty. The scoring is dull on purpose. A camp form and a school
certificate do not look alike to a keyword matcher, and the cases where they do
are exactly the cases that should go to a person anyway.

THE ONE RULE THAT MATTERS HERE. README I-01, target state, step 2:

    Unknown type -> LLM vision extraction -> new template proposed for HUMAN
    CONFIRMATION and permanent addition to the library.

A proposal is not a template. `TemplateProposer` returns a `ProposedTemplate`,
which is a dataclass with no `add` method, no reference to the store, and a
`confirm()` that demands a person's identifier and returns a `FormTemplate` the
CALLER must then hand to `TemplateStore.add`. There is no path from a model
output to the library that does not pass through a human, and the reason is
arithmetic rather than principle: a proposed template is a set of coordinates,
every subsequent form of that type is filled from them, and one bad box quietly
mis-fills every camp form for a year.

A proposal also arrives UNCALIBRATED. Every box a model proposes carries
`placeholder=True`, so even after a person confirms the field list, the store's
existing gate still refuses to fill with it until somebody measures the boxes.
Confirming "yes, that box is the tetanus date" is a different act from
confirming "yes, that box is at those coordinates", and a model that can do the
first is not evidence it did the second.

The build plan names `qwen2.5vl:7b via Ollama`. The build plan's own notes are
honest about this being the weakest local task in the repo -- *"Novel-layout
document understanding (I-01 detect.py) is the one task where frontier vision
models still lead open weights by a meaningful margin"* -- which is precisely
why the human confirmation step is not negotiable here even though it might be
elsewhere.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from nsp_core.llm import LLMClient, SchemaViolation

from .templates import (
    BoundingBox,
    FieldSpec,
    FormTemplate,
    TemplateStore,
    TRANSFORMS,
)

__all__ = [
    "DETECT_SYSTEM_PROMPT",
    "PROPOSAL_SCHEMA",
    "MIN_ANCHOR_SCORE",
    "MIN_ANCHOR_MARGIN",
    "Detection",
    "FormDetector",
    "ProposedTemplate",
    "TemplateProposer",
    "ProposalNotConfirmed",
]

#: A scan must match this share of a template's anchors to be called that form.
#: Two thirds rather than a majority: an anchor list is short, and a form that
#: matches half of a nine-phrase list is a form the reader would hesitate over.
MIN_ANCHOR_SCORE = 0.66

#: And it must beat the runner-up by this much. Two templates scoring 0.70 and
#: 0.68 is not a classification, it is a coin toss between two documents that
#: will be filled from different coordinate maps.
MIN_ANCHOR_MARGIN = 0.15


class ProposalNotConfirmed(RuntimeError):
    """Raised when a machine-proposed template is used before a person said yes."""


def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace, drop punctuation OCR invents."""
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9\s]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


@dataclass(frozen=True)
class Detection:
    """What the deterministic classifier concluded, and how sure it was."""

    form_type: str | None
    score: float
    matched_anchors: tuple[str, ...] = ()
    runner_up: str | None = None
    runner_up_score: float = 0.0
    reason: str = ""

    @property
    def known(self) -> bool:
        return self.form_type is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "form_type": self.form_type,
            "score": round(self.score, 3),
            "matched_anchors": list(self.matched_anchors),
            "runner_up": self.runner_up,
            "runner_up_score": round(self.runner_up_score, 3),
            "reason": self.reason,
        }


@dataclass
class FormDetector:
    """Anchor-phrase scoring against the template library. No model."""

    store: TemplateStore
    min_score: float = MIN_ANCHOR_SCORE
    min_margin: float = MIN_ANCHOR_MARGIN

    def score(self, text: str) -> list[tuple[str, float, tuple[str, ...]]]:
        haystack = _normalise(text)
        scored: list[tuple[str, float, tuple[str, ...]]] = []
        for form_type, template in self.store.templates.items():
            if not template.anchors:
                # A template with no anchors can never be detected, and a
                # library where that goes unnoticed silently routes every copy
                # of that form to the unknown-layout path.
                scored.append((form_type, 0.0, ()))
                continue
            hits = tuple(
                anchor for anchor in template.anchors
                if _normalise(anchor) and _normalise(anchor) in haystack
            )
            scored.append((form_type, len(hits) / len(template.anchors), hits))
        return sorted(scored, key=lambda row: (-row[1], row[0]))

    def detect(self, text: str) -> Detection:
        """The form type, or None. None is a normal outcome, not a failure."""
        scored = self.score(text)
        if not scored:
            return Detection(None, 0.0, reason="the template library is empty")
        best_type, best_score, hits = scored[0]
        runner_up, runner_score = (scored[1][0], scored[1][1]) if len(scored) > 1 else (None, 0.0)

        if best_score < self.min_score:
            return Detection(
                None, best_score, hits, runner_up, runner_score,
                reason=(
                    f"the closest template ({best_type}) matched "
                    f"{best_score:.0%} of its anchor phrases, below the "
                    f"{self.min_score:.0%} floor; this is an unseen layout"
                ),
            )
        if best_score - runner_score < self.min_margin:
            return Detection(
                None, best_score, hits, runner_up, runner_score,
                reason=(
                    f"{best_type} ({best_score:.0%}) and {runner_up} "
                    f"({runner_score:.0%}) are too close to tell apart. Filling "
                    "from the wrong coordinate map puts every value in the "
                    "wrong box, so this goes to a person."
                ),
            )
        return Detection(
            best_type, best_score, hits, runner_up, runner_score,
            reason=f"matched {len(hits)} of "
                   f"{len(self.store.get(best_type).anchors)} anchor phrases",
        )


# -- the unknown-layout path -------------------------------------------------

DETECT_SYSTEM_PROMPT = """You are reading a blank health form that a pediatric practice has never seen before.

Your job is to describe what fields the form has and roughly where they are, so
a person can review your description and build a template from it. You are NOT
filling the form and you are NOT reading any patient's data.

CONSTRAINTS:
- Report ONLY fields you can actually see on the page. Do not add fields you
  would expect a form like this to have.
- Do not invent a field label. If a box has no readable label, say so in
  `label` and set `semantic` to "unknown".
- `semantic` must come from the fixed list below and nothing else. Use
  "unknown" whenever you are not sure; "unknown" sends the field to a person,
  which is the correct outcome.
- Coordinates are approximate and will be re-measured by a person. Give your
  best estimate in PDF points from the top-left of the page.
- If the document is not a blank health form at all, return an empty field list
  and say why in `notes`.

Allowed `semantic` values:
  patient_name, patient_dob, patient_address, guardian_name, exam_date,
  height, weight, bmi, blood_pressure, vision_screening, hearing_screening,
  allergies, medications, conditions, immunization_date, provider_name,
  provider_signature, signature_date, unknown

Return ONLY valid JSON:
{
  "form_title": string,
  "issuer": string or null,
  "page_count": integer,
  "fields": [
    {
      "label": string,
      "semantic": one of the values above,
      "page": integer,
      "x": number, "y": number, "width": number, "height": number
    }
  ],
  "notes": string
}"""

_FIELD_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "semantic": {
            "type": "string",
            "enum": [
                "patient_name", "patient_dob", "patient_address", "guardian_name",
                "exam_date", "height", "weight", "bmi", "blood_pressure",
                "vision_screening", "hearing_screening", "allergies",
                "medications", "conditions", "immunization_date",
                "provider_name", "provider_signature", "signature_date",
                "unknown",
            ],
        },
        "page": {"type": "integer", "minimum": 1},
        "x": {"type": "number"},
        "y": {"type": "number"},
        "width": {"type": "number"},
        "height": {"type": "number"},
    },
    "required": ["label", "semantic", "page", "x", "y", "width", "height"],
    "additionalProperties": False,
}

PROPOSAL_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "form_title": {"type": "string"},
        "issuer": {"type": ["string", "null"]},
        "page_count": {"type": "integer", "minimum": 1},
        "fields": {"type": "array", "items": _FIELD_SCHEMA},
        "notes": {"type": "string"},
    },
    "required": ["form_title", "issuer", "page_count", "fields", "notes"],
    "additionalProperties": False,
}

#: Which chart path and transform each semantic maps to. `unknown` and the
#: attestation semantics map to nothing on purpose: a field the model could not
#: name gets no source, so even a confirmed template leaves it blank until a
#: person wires it up.
_SEMANTIC_BINDINGS: Mapping[str, tuple[str, str, str]] = {
    # semantic: (chart source path, transform, field kind)
    "patient_name": ("patient.last_name", "verbatim", "text"),
    "patient_dob": ("patient.date_of_birth", "date_mm_dd_yyyy", "date"),
    "patient_address": ("patient.address", "verbatim", "text"),
    "guardian_name": ("patient.guardian_name", "verbatim", "text"),
    "exam_date": ("exam.date", "date_mm_dd_yyyy", "date"),
    "height": ("vitals.height_in", "one_decimal", "number"),
    "weight": ("vitals.weight_lb", "one_decimal", "number"),
    "bmi": ("vitals.bmi", "one_decimal", "number"),
    "blood_pressure": ("vitals.blood_pressure", "blood_pressure", "text"),
    "vision_screening": ("screenings.vision", "pass_fail", "text"),
    "hearing_screening": ("screenings.hearing", "pass_fail", "text"),
    "allergies": ("allergies", "comma_list", "text"),
    "medications": ("medications", "comma_list", "text"),
    "conditions": ("conditions", "comma_list", "text"),
}

#: Semantics that become fields a machine may never write.
_HUMAN_ONLY_SEMANTICS: frozenset[str] = frozenset(
    {"provider_name", "provider_signature", "signature_date", "unknown"}
)


@dataclass
class ProposedTemplate:
    """A model's description of an unseen form. NOT a template.

    Deliberately has no reference to a `TemplateStore` and no `add`. The only
    way out of this class is `confirm()`, which requires a person's identifier
    and hands back a `FormTemplate` the caller must still add to the library
    themselves.
    """

    form_type: str
    title: str
    issuer: str
    page_count: int
    proposed_fields: tuple[FieldSpec, ...]
    #: Fields the model could not name. Listed separately because these are what
    #: the confirming person actually has to work on.
    unnamed: tuple[str, ...] = ()
    notes: str = ""
    model_id: str = ""
    model_version: str = ""
    provider: str = ""
    inference_id: str | None = None
    source_document: str = ""

    @property
    def needs_attention(self) -> tuple[str, ...]:
        return self.unnamed

    def confirm(self, *, confirmed_by: str, anchors: Sequence[str]) -> FormTemplate:
        """Turn the proposal into a template. A PERSON'S NAME IS REQUIRED.

        `anchors` is supplied by the confirming person rather than the model,
        because the anchors are what routes every future copy of this form, and
        a model-chosen anchor that also appears on a different form silently
        cross-routes both of them.

        The returned template's boxes stay `placeholder=True`. Confirming the
        field list is not the same act as measuring the coordinates, and
        `TemplateStore.for_filling` will still refuse it until somebody does.
        """
        if not confirmed_by.strip():
            raise ProposalNotConfirmed(
                "a proposed template becomes a real one only when a person "
                "confirms it (README I-01 step 2). Pass the confirming user."
            )
        if not anchors:
            raise ProposalNotConfirmed(
                "a template with no anchor phrases can never be detected, so "
                "every copy of this form would go back to the unknown-layout "
                "path. The confirming person supplies the anchors."
            )
        return FormTemplate(
            form_type=self.form_type,
            title=self.title,
            version=f"proposed; confirmed by {confirmed_by}",
            issuer=self.issuer,
            page_count=self.page_count,
            fields=self.proposed_fields,
            anchors=tuple(anchors),
            calibration="",  # deliberately empty: the boxes are still estimates
            source_pdf=self.source_document,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "form_type": self.form_type,
            "title": self.title,
            "issuer": self.issuer,
            "page_count": self.page_count,
            "field_count": len(self.proposed_fields),
            "unnamed": list(self.unnamed),
            "notes": self.notes,
            "model": f"{self.provider}:{self.model_id}@{self.model_version}",
            "inference_id": self.inference_id,
            "fields": [f.as_dict() for f in self.proposed_fields],
        }


@dataclass
class TemplateProposer:
    """The one model call in this module, on the one task README I-01 wants it for."""

    client: LLMClient
    audit: Any = None
    system_prompt: str = DETECT_SYSTEM_PROMPT
    INITIATIVE: str = "I-01"

    @property
    def prompt_hash(self) -> str:
        return hashlib.sha256(self.system_prompt.encode("utf-8")).hexdigest()[:16]

    def propose(
        self,
        page_text: str,
        *,
        document_id: str,
        form_type: str,
        user_id: str = "system:forms",
    ) -> ProposedTemplate | None:
        """Propose a template, or return None so the document goes to a person.

        Returns None on a schema violation rather than raising: one unreadable
        form must not stop the morning's batch, and nothing is guessed -- the
        document goes to a person with the reason attached (hard constraint 3).
        """
        try:
            result = self.client.structured(
                system=self.system_prompt,
                user=page_text,
                schema=PROPOSAL_SCHEMA,
                context={
                    "initiative_id": self.INITIATIVE,
                    "user_id": user_id,
                    "document_id": document_id,
                },
                prompt_template_id="I-01/propose-template",
                temperature=0.0,
            )
        except SchemaViolation:
            return None

        data = result.data
        page_count = int(data["page_count"])
        specs: list[FieldSpec] = []
        unnamed: list[str] = []
        seen: set[str] = set()
        for index, raw in enumerate(data["fields"]):
            semantic = str(raw["semantic"])
            page = int(raw["page"])
            if not 1 <= page <= page_count:
                # The model put a field on a page the form does not have. Drop
                # the field and say so, rather than clamping it onto page 1 --
                # a clamped coordinate is a wrong coordinate that looks right.
                unnamed.append(
                    f"(dropped) {raw['label']!r}: placed on page {page} of a "
                    f"{page_count}-page form"
                )
                continue
            width, height = float(raw["width"]), float(raw["height"])
            if width <= 0 or height <= 0:
                unnamed.append(f"(dropped) {raw['label']!r}: zero-size box")
                continue
            name = _field_name(semantic, raw["label"], index, seen)
            seen.add(name)
            source, transform, kind = _SEMANTIC_BINDINGS.get(
                semantic, ("", "verbatim", "text")
            )
            human_only = semantic in _HUMAN_ONLY_SEMANTICS
            if semantic == "provider_signature":
                kind = "signature"
            if semantic == "immunization_date":
                # A model can see that a box wants a tetanus date. It cannot
                # know WHICH dose of which antigen that box is, and guessing
                # binds the wrong dose to the box on every future form. So the
                # box is proposed with no source and flagged for the person.
                kind = "grid_date"
                human_only = True
                unnamed.append(
                    f"{name}: an immunization box the person must bind to a "
                    "specific antigen and dose number"
                )
            if semantic == "unknown":
                unnamed.append(f"{name}: the model could not name this field")
            specs.append(
                FieldSpec(
                    name=name,
                    kind=kind,
                    box=BoundingBox(
                        page=page,
                        x=float(raw["x"]), y=float(raw["y"]),
                        width=width, height=height,
                        # ALWAYS a placeholder. A model's coordinate estimate is
                        # not a measurement, and the store's gate is what stops
                        # it being treated as one.
                        placeholder=True,
                    ),
                    source=source,
                    transform=transform if transform in TRANSFORMS else "verbatim",
                    required=False,
                    human_only=human_only,
                    label=str(raw["label"])[:120],
                )
            )

        proposal = ProposedTemplate(
            form_type=form_type,
            title=str(data["form_title"]),
            issuer=str(data["issuer"] or ""),
            page_count=page_count,
            proposed_fields=tuple(specs),
            unnamed=tuple(unnamed),
            notes=str(data["notes"]),
            model_id=result.model_id,
            model_version=result.model_version,
            provider=result.provider,
            source_document=document_id,
        )
        if self.audit is not None:
            proposal.inference_id = self.audit.record_inference(
                user_id=user_id,
                initiative_id=self.INITIATIVE,
                provider=result.provider,
                model_id=result.model_id,
                model_version=result.model_version,
                prompt_template_id=result.prompt_template_id,
                prompt_template_hash=result.prompt_template_hash,
                input_token_count=result.input_token_count,
                output_token_count=result.output_token_count,
                confidence_score=None,
                constrained_decoding=result.constrained,
                repair_attempts=result.repair_attempts,
                extra={
                    "document_id": document_id,
                    "proposed_form_type": form_type,
                    "proposed_fields": len(specs),
                    "needs_attention": len(unnamed),
                },
            )
        return proposal


def _field_name(semantic: str, label: Any, index: int, seen: set[str]) -> str:
    """A stable, unique, machine-safe field name.

    Derived from the semantic rather than the label, because the label is OCR
    output and two boxes on the same form routinely OCR to the same string.
    """
    stem = semantic if semantic != "unknown" else "unnamed"
    name = f"{stem}_{index + 1}"
    while name in seen:  # pragma: no cover - index makes collisions impossible
        index += 1
        name = f"{stem}_{index + 1}"
    return name
