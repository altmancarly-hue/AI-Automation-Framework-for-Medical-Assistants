"""Reading an insurance card, and the confirmation step that follows it.

README I-09 puts this in the small column of things a model is right for:

    Read an insurance card photo | OCR + extraction -- **LLM helps here.** Card
    layouts vary enormously across 36 payers; a rules-based parser is fragile.

and immediately pairs it with a control:

    Card OCR misreads a member ID | Medium | Front desk confirms every extracted
    field against the card image side by side; **never auto-submit**

So this module extracts, and produces something a person confirms. The
confirmation is not advisory: `CardExtraction.for_submission()` raises until
every field a 270 depends on has been confirmed by a named person, and the
eligibility pipeline has no other way to obtain a member id from a photo.

WHY THE CONFIRMATION IS FIELD-BY-FIELD RATHER THAN A SINGLE "LOOKS RIGHT".
A member id is fifteen characters of alphanumeric with no checksum. Confirming a
whole card at once means confirming the payer name -- which is in large type and
obviously right -- and glancing past the digit that matters. The unit of
confirmation is the field, because the unit of error is the character.

WHAT A MISREAD COSTS. A wrong member id produces a 271 rejection that says
"subscriber not found", which reads to a front desk exactly like "this family
has no insurance". `coverage.py` separates those two outcomes for that reason,
and this module exists to make the first one rare.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence

from nsp_core.llm import LLMClient, SchemaViolation

__all__ = [
    "CARD_SYSTEM_PROMPT",
    "CARD_SCHEMA",
    "REQUIRED_FIELDS",
    "ExtractedField",
    "CardExtraction",
    "CardReader",
    "UnconfirmedCard",
]

#: The fields a 270 cannot be built without. Every one must be confirmed by a
#: person before this module will hand anything to the eligibility path.
REQUIRED_FIELDS: tuple[str, ...] = ("payer_name", "member_id")

CARD_SYSTEM_PROMPT = """You are reading a photograph of a health insurance card for a pediatric practice.

Transcribe what is printed on the card. You are not interpreting it and you are
not looking anything up.

CONSTRAINTS:
- Copy each value EXACTLY as printed, including letters, leading zeros, spaces
  and hyphens. Do not normalise, reformat, expand an abbreviation, or correct
  what looks like a typo. A member id that looks wrong is still what the card
  says, and a person will check it.
- If a field is not visible, unreadable, or you are unsure, return null for it
  and say so in `unreadable`. A null goes to a person, which is the correct
  outcome. A guess goes to a payer.
- Do not infer the payer from a logo you recognise if the name is not printed.
- `confidence` is per field, and it is about how clearly you could READ it, not
  about how plausible the value looks.
- If the image is not an insurance card at all, return nulls throughout and say
  so in `unreadable`.

Return ONLY valid JSON:
{
  "payer_name": string or null,
  "plan_name": string or null,
  "member_id": string or null,
  "group_number": string or null,
  "subscriber_name": string or null,
  "rx_bin": string or null,
  "rx_pcn": string or null,
  "payer_phone": string or null,
  "confidence": {
    "payer_name": 0.0-1.0,
    "plan_name": 0.0-1.0,
    "member_id": 0.0-1.0,
    "group_number": 0.0-1.0,
    "subscriber_name": 0.0-1.0
  },
  "unreadable": string
}"""

_STR_OR_NULL = {"type": ["string", "null"]}
_CONF = {"type": "number", "minimum": 0.0, "maximum": 1.0}

CARD_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "payer_name": _STR_OR_NULL,
        "plan_name": _STR_OR_NULL,
        "member_id": _STR_OR_NULL,
        "group_number": _STR_OR_NULL,
        "subscriber_name": _STR_OR_NULL,
        "rx_bin": _STR_OR_NULL,
        "rx_pcn": _STR_OR_NULL,
        "payer_phone": _STR_OR_NULL,
        "confidence": {
            "type": "object",
            "properties": {
                "payer_name": _CONF, "plan_name": _CONF, "member_id": _CONF,
                "group_number": _CONF, "subscriber_name": _CONF,
            },
            "required": [
                "payer_name", "plan_name", "member_id", "group_number",
                "subscriber_name",
            ],
            "additionalProperties": False,
        },
        "unreadable": {"type": "string"},
    },
    "required": [
        "payer_name", "plan_name", "member_id", "group_number",
        "subscriber_name", "rx_bin", "rx_pcn", "payer_phone", "confidence",
        "unreadable",
    ],
    "additionalProperties": False,
}


class UnconfirmedCard(RuntimeError):
    """Raised when an extraction is used before a person confirmed its fields."""


@dataclass
class ExtractedField:
    """One value the model read, and whether a person has looked at it."""

    name: str
    value: str | None
    confidence: float
    confirmed_by: str = ""
    confirmed_value: str | None = None
    confirmed_at: datetime | None = None

    @property
    def confirmed(self) -> bool:
        return bool(self.confirmed_by)

    @property
    def final(self) -> str | None:
        """What to actually use. The person's value wins, always."""
        return self.confirmed_value if self.confirmed else None

    @property
    def corrected(self) -> bool:
        return self.confirmed and self.confirmed_value != self.value

    def confirm(self, *, by: str, value: str | None, at: datetime) -> None:
        if not by.strip():
            raise ValueError("a confirmation names the person who made it")
        self.confirmed_by = by
        self.confirmed_value = value
        self.confirmed_at = at

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.name,
            "extracted": self.value,
            "confidence": round(self.confidence, 3),
            "confirmed": self.confirmed,
            "confirmed_by": self.confirmed_by,
            "final": self.final,
            "corrected": self.corrected,
        }


@dataclass
class CardExtraction:
    """What the model read off one card, pending a person's confirmation."""

    document_id: str
    fields: dict[str, ExtractedField] = field(default_factory=dict)
    unreadable: str = ""
    model_id: str = ""
    model_version: str = ""
    provider: str = ""
    inference_id: str | None = None

    @property
    def unconfirmed(self) -> list[str]:
        return sorted(
            name for name in REQUIRED_FIELDS
            if name not in self.fields or not self.fields[name].confirmed
        )

    @property
    def corrections(self) -> list[str]:
        """Fields where the person disagreed with the model.

        Worth watching over time: a member-id correction rate that climbs is
        either a prompt problem or a camera problem, and both are fixable before
        they become denials.
        """
        return sorted(n for n, f in self.fields.items() if f.corrected)

    def confirm(
        self, name: str, *, by: str, value: str | None, at: datetime
    ) -> ExtractedField:
        if name not in self.fields:
            raise KeyError(f"{name!r} is not a field on this extraction")
        self.fields[name].confirm(by=by, value=value, at=at)
        return self.fields[name]

    def for_submission(self) -> dict[str, str]:
        """The confirmed values, or a refusal. This is the only way out.

        README I-09: "Front desk confirms every extracted field against the card
        image side by side; never auto-submit." There is no bypass parameter.
        """
        outstanding = self.unconfirmed
        if outstanding:
            raise UnconfirmedCard(
                f"{outstanding} have not been confirmed against the card image. "
                "A member id is fifteen characters with no checksum, and a "
                "misread produces a 'subscriber not found' that reads to the "
                "front desk exactly like 'this family has no insurance'."
            )
        missing = [
            name for name in REQUIRED_FIELDS
            if not (self.fields[name].final or "").strip()
        ]
        if missing:
            raise UnconfirmedCard(
                f"{missing} were confirmed as empty. Re-photograph the card or "
                "key them from the physical card; an eligibility inquiry without "
                "them cannot be sent."
            )
        return {
            name: (spec.final or "")
            for name, spec in self.fields.items()
            if spec.confirmed and spec.final
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "unreadable": self.unreadable,
            "unconfirmed": self.unconfirmed,
            "corrections": self.corrections,
            "model": f"{self.provider}:{self.model_id}@{self.model_version}",
            "inference_id": self.inference_id,
            "fields": [f.as_dict() for f in self.fields.values()],
        }


@dataclass
class CardReader:
    """One model call per card. The only one in this file."""

    client: LLMClient
    audit: Any = None
    deidentifier: Any = None
    system_prompt: str = CARD_SYSTEM_PROMPT
    INITIATIVE: str = "I-09"

    @property
    def prompt_hash(self) -> str:
        return hashlib.sha256(self.system_prompt.encode("utf-8")).hexdigest()[:16]

    def read(
        self,
        card_text: str,
        *,
        document_id: str,
        user_id: str = "system:eligibility",
    ) -> CardExtraction | None:
        """Extract, or return None so the card goes to a person to key by hand.

        Returns None on a schema violation rather than raising: one unreadable
        card must not stop a batch, and nothing is guessed.

        NOTE the card is NOT de-identified before the call. The member id and the
        subscriber name are the entire payload -- a tokenised card extracts to
        tokens. That is a reason to run this locally, which is the default.
        """
        try:
            result = self.client.structured(
                system=self.system_prompt,
                user=card_text,
                schema=CARD_SCHEMA,
                context={
                    "initiative_id": self.INITIATIVE,
                    "user_id": user_id,
                    "document_id": document_id,
                },
                prompt_template_id="I-09/read-card",
                temperature=0.0,
            )
        except SchemaViolation:
            return None

        data = result.data
        confidence = data["confidence"]
        extraction = CardExtraction(
            document_id=document_id,
            unreadable=str(data["unreadable"]),
            model_id=result.model_id,
            model_version=result.model_version,
            provider=result.provider,
        )
        for name in (
            "payer_name", "plan_name", "member_id", "group_number",
            "subscriber_name", "rx_bin", "rx_pcn", "payer_phone",
        ):
            extraction.fields[name] = ExtractedField(
                name=name,
                value=data[name],
                confidence=float(confidence.get(name, 0.0)),
            )
        if self.audit is not None:
            extraction.inference_id = self.audit.record_inference(
                user_id=user_id,
                initiative_id=self.INITIATIVE,
                provider=result.provider,
                model_id=result.model_id,
                model_version=result.model_version,
                prompt_template_id=result.prompt_template_id,
                prompt_template_hash=result.prompt_template_hash,
                input_token_count=result.input_token_count,
                output_token_count=result.output_token_count,
                confidence_score=float(confidence.get("member_id", 0.0)),
                constrained_decoding=result.constrained,
                repair_attempts=result.repair_attempts,
                extra={"document_id": document_id},
            )
        return extraction
