"""The two places a model earns its keep in I-02: drafting and reply triage.

README I-02 is precise about the division of labour: "The LLM personalizes tone
and reading level; it does NOT invent clinical content. Templates are
pre-approved; the LLM fills slots and adjusts register."

Turning that sentence into something enforceable is what this module is for.

**Drafting.** The physician-approved template is rendered deterministically
first. That rendered string is the *baseline*, and it is what gets sent if
anything at all goes wrong -- a schema violation, a dropped slot, an invented
number, a missing opt-out line, an over-length SMS. The failure mode of the
personalisation layer is the approved text, not a blocked message and not a
degraded one. `_verify_draft` checks the model's output against the baseline
rather than trusting the instruction, because a prompt constraint is a request
and a post-condition is a control.

`_verify_draft` enforces six of them: no new digit runs, no recombined numeric
sequences (a set difference happily passes "555-847-0100" against a baseline
containing "847-555-0100"), no spelled-out numbers the approved text does not
use ("two shots" is a dose count no digit rule can see), no new links, no fear
or pressure language, and the opt-out sentence intact. The last two are blunt
keyword lists and are deliberately over-broad: a false positive costs nothing,
because the fallback is a message a physician already signed.

**Reply triage.** STOP is handled deterministically, before any model runs and
regardless of what a model would have said. Everything else is classified into a
fixed enum and *routed*; nothing is auto-applied. "We got it at CVS" goes to an
MA to reconcile, exactly as the README specifies -- it does not update the
immunization record, because a text message is not a source document.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from nsp_core.llm import LLMClient, SchemaViolation
from nsp_core.phi import LeakGuard

__all__ = [
    "DRAFT_SYSTEM_PROMPT",
    "DRAFT_SCHEMA",
    "REPLY_SYSTEM_PROMPT",
    "REPLY_SCHEMA",
    "ReplyIntent",
    "DraftResult",
    "MessageDrafter",
    "ReplyClassification",
    "ReplyTriage",
    "STOP_KEYWORDS",
]

# Imported from the scheduling module so there is exactly one definition of what
# a carrier considers an opt-out. Two lists would drift, and the one that drifted
# would be the one that let a message through after a STOP.
from modules.scheduling.gateway import HELP_KEYWORDS, START_KEYWORDS, STOP_KEYWORDS

SMS_MAX_CHARS = 320  # two segments; longer is billed and read as spam

DRAFT_SYSTEM_PROMPT = """You adjust the tone and reading level of a pre-approved message from a
pediatric practice to a family. You are not a clinician and you are not writing
clinical content.

RULES:
- The approved message is the source of truth. You may reword it for warmth and
  for a 6th-grade reading level.
- You MUST keep every one of the supplied slot values exactly as given: the
  child's first name, the vaccine list, and the booking link.
- You MUST NOT introduce any number, date, price, quantity, dose count, or
  phone number that is not already in the approved message.
- You MUST NOT add clinical claims, urgency language about consequences, or
  anything that could read as pressure.
- You MUST keep the opt-out sentence if the approved message has one.
- Keep an SMS under 320 characters.

Return ONLY valid JSON."""

DRAFT_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["message", "confidence"],
    "properties": {
        "message": {"type": "string", "maxLength": 2000},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "notes": {"type": "string", "maxLength": 400},
    },
}


class ReplyIntent:
    """Fixed enum. A reply that does not fit one of these goes to a person."""

    ALREADY_VACCINATED_ELSEWHERE = "already_vaccinated_elsewhere"
    WANTS_APPOINTMENT = "wants_appointment"
    HAS_QUESTIONS = "has_questions"
    DECLINES = "declines"
    OTHER = "other"
    #: Deterministic outcomes, never produced by a model.
    OPT_OUT = "opt_out"
    OPT_IN = "opt_in"
    HELP = "help"
    ALL = (
        ALREADY_VACCINATED_ELSEWHERE,
        WANTS_APPOINTMENT,
        HAS_QUESTIONS,
        DECLINES,
        OTHER,
        OPT_OUT,
        OPT_IN,
        HELP,
    )
    MODEL_CLASSIFIABLE = (
        ALREADY_VACCINATED_ELSEWHERE,
        WANTS_APPOINTMENT,
        HAS_QUESTIONS,
        DECLINES,
        OTHER,
    )


REPLY_SYSTEM_PROMPT = """You are triaging a parent's SMS reply to a pediatric practice's vaccine
reminder. Classify the reply into exactly one category so it reaches the right
person.

CATEGORIES:
- already_vaccinated_elsewhere: the parent says the child had the vaccine
  somewhere else (pharmacy, school, another clinic, another state).
- wants_appointment: the parent is asking to book, or asking about availability.
- has_questions: the parent has a clinical or safety question.
- declines: the parent says they do not want the vaccine.
- other: anything else, including replies you are not confident about.

CONSTRAINTS:
- You are classifying, not answering. Do not draft a reply.
- You MUST NOT record, confirm, or infer that a vaccine was administered. A text
  message is not a source document; a human updates the record.
- When the reply is ambiguous, short, or could be read two ways, return "other".
  Routing to a person is a correct outcome.

Return ONLY valid JSON."""

REPLY_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intent", "confidence"],
    "properties": {
        "intent": {
            "type": "string",
            "enum": list(ReplyIntent.MODEL_CLASSIFIABLE),
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rationale": {"type": "string", "maxLength": 300},
    },
}


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


_NUMBERS = re.compile(r"\d+")
#: Seven or more digits, allowing separators: phone numbers, account numbers,
#: anything a recombination attack would produce.
_LONG_NUMERIC = re.compile(r"[\d][\d\-.() ]{5,}[\d]")
_URLS = re.compile(r"https?://\S+", re.I)
_NUMBER_WORDS = (
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "first", "second", "third", "fourth", "fifth",
    "once", "twice", "single", "double", "triple", "dozen",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "tomorrow", "today", "tonight", "yesterday", "week", "weeks", "month",
    "months", "year", "years", "day", "days",
)
#: Fear, consequence and coercion language. Over-broad on purpose.
_PRESSURE_TERMS = (
    "urgent", "urgently", "immediately", "emergency", "danger", "dangerous",
    "risk", "risky", "unprotected", "vulnerable", "hospital", "hospitalised",
    "hospitalized", "die", "death", "dying", "fatal", "deadly", "serious",
    "severe", "outbreak", "epidemic", "infected", "infection", "disease",
    "illness", "sick", "measles", "whooping", "meningitis", "cancer",
    "required by law", "must", "mandatory", "expelled", "excluded", "banned",
    "final notice", "last chance", "act now", "don't wait", "do not wait",
    "penalty", "fee", "fine", "charged", "dismissed", "terminate",
)


def _word_in(haystack: str, needle: str) -> bool:
    """Whole-word (or whole-phrase) containment, so 'one' does not match 'phone'."""
    return re.search(r"(?<![a-z])" + re.escape(needle) + r"(?![a-z])", haystack) is not None


@dataclass
class DraftResult:
    message: str
    used_model: bool
    fallback_reason: str = ""
    confidence: float | None = None
    inference_id: str | None = None

    @property
    def is_approved_text(self) -> bool:
        return not self.used_model


class MessageDrafter:
    """Personalises a physician-approved template, or returns it unchanged."""

    INITIATIVE = "I-02"
    PROMPT_TEMPLATE_ID = "I-02-recall-draft"

    def __init__(
        self,
        client: LLMClient,
        *,
        audit: Any = None,
        confidence_floor: float = 0.7,
        max_sms_chars: int = SMS_MAX_CHARS,
        leak_guard: LeakGuard | None = None,
    ) -> None:
        self.client = client
        self.audit = audit
        self.confidence_floor = confidence_floor
        self.max_sms_chars = max_sms_chars
        # The slots legitimately contain a first name and a URL, so those labels
        # are permitted; everything else -- a phone number, an MRN, an address
        # leaking in from somewhere -- still raises.
        self.leak_guard = leak_guard or LeakGuard(allow_labels={"URL"})

    def draft(
        self,
        *,
        template_id: str,
        template: str,
        slots: Mapping[str, Any],
        patient_id: str = "",
        channel: str = "sms",
        user_id: str = "system:nightly",
    ) -> str:
        """Return message text. Never raises; falls back to the approved text."""
        return self.draft_detailed(
            template_id=template_id,
            template=template,
            slots=slots,
            patient_id=patient_id,
            channel=channel,
            user_id=user_id,
        ).message

    def draft_detailed(
        self,
        *,
        template_id: str,
        template: str,
        slots: Mapping[str, Any],
        patient_id: str = "",
        channel: str = "sms",
        user_id: str = "system:nightly",
    ) -> DraftResult:
        baseline = template.format(**slots)
        preserve = [
            str(slots[k]) for k in ("first_name", "vaccine_list", "booking_url")
            if slots.get(k)
        ]

        user = (
            "APPROVED MESSAGE:\n"
            f"{baseline}\n\n"
            "SLOT VALUES THAT MUST APPEAR VERBATIM:\n"
            + "\n".join(f"- {value}" for value in preserve)
            + f"\n\nCHANNEL: {channel}"
        )
        # Scan the SLOT VALUES, not the whole prompt. The approved template is a
        # fixed, physician-reviewed string that legitimately contains the
        # practice's own phone number; guarding it would mean no email template
        # could ever be personalised, and a control that blocks the normal case
        # gets switched off. The dynamic half is what needs checking, and the
        # only things that reach it are a first name, a vaccine label and a
        # booking link produced by the rules engine.
        try:
            self.leak_guard.assert_clean(list(preserve), allow_labels={"URL"})
        except Exception as exc:  # noqa: BLE001 - fail closed, send approved text
            return DraftResult(baseline, False, f"leak_guard: {type(exc).__name__}")

        try:
            result = self.client.structured(
                system=DRAFT_SYSTEM_PROMPT,
                user=user,
                schema=DRAFT_SCHEMA,
                prompt_template_id=self.PROMPT_TEMPLATE_ID,
                context={"initiative": self.INITIATIVE, "patient_id": patient_id},
            )
        except SchemaViolation as exc:
            return DraftResult(baseline, False, f"schema_violation: {exc}")

        inference_id = None
        if self.audit is not None:
            inference_id = self.audit.record_inference(
                user_id=user_id,
                initiative_id=self.INITIATIVE,
                provider=result.provider,
                model_id=result.model_id,
                model_version=result.model_version,
                prompt_template_id=self.PROMPT_TEMPLATE_ID,
                prompt_template_hash=_hash(DRAFT_SYSTEM_PROMPT + template),
                input_token_count=result.input_token_count,
                output_token_count=result.output_token_count,
                patient_id=patient_id or None,
                confidence_score=result.confidence,
                constrained_decoding=result.constrained,
                repair_attempts=result.repair_attempts,
                extra={"template_id": template_id, "channel": channel},
            )

        candidate = str(result.data["message"]).strip()
        confidence = float(result.data["confidence"])
        failure = self._verify_draft(candidate, baseline, preserve, channel)
        if failure or confidence < self.confidence_floor:
            return DraftResult(
                baseline,
                False,
                failure or f"confidence {confidence:.2f} below floor",
                confidence,
                inference_id,
            )
        return DraftResult(candidate, True, "", confidence, inference_id)

    def _verify_draft(
        self, candidate: str, baseline: str, preserve: Sequence[str], channel: str
    ) -> str:
        """Post-conditions on the model's rewrite. Any failure sends the baseline.

        These are blunt instruments and they are meant to be. Every one of them
        was written because the prompt asks for something a model can silently
        ignore, and a constraint with no post-condition is a wish.
        """
        if not candidate:
            return "empty draft"
        for value in preserve:
            if value not in candidate:
                return f"slot value dropped from the draft: {value[:24]!r}"

        # 1. No new digit runs. Blocks invented dates, dose counts and prices.
        baseline_numbers = set(_NUMBERS.findall(baseline))
        introduced = set(_NUMBERS.findall(candidate)) - baseline_numbers
        if introduced:
            return f"draft introduced numbers absent from the approved text: {sorted(introduced)}"

        # 2. No RECOMBINED digit sequences. A set difference passes
        #    "555-847-0100" against a baseline containing "847-555-0100" --
        #    same runs, different phone number. Long digit-and-separator
        #    sequences must appear verbatim.
        for token in _LONG_NUMERIC.findall(candidate):
            if token not in baseline:
                return f"draft contains a numeric sequence not in the approved text: {token!r}"

        # 3. No spelled-out numbers the approved text does not use. "two shots"
        #    is a dose count the digit rule cannot see.
        lowered = candidate.lower()
        base_lower = baseline.lower()
        for word in _NUMBER_WORDS:
            if _word_in(lowered, word) and not _word_in(base_lower, word):
                return f"draft introduced a spelled-out number: {word!r}"

        # 4. No new links. A rewrite that adds a URL has added a destination
        #    nobody approved, and a payment or credential page is exactly the
        #    thing an SMS from a medical practice must never carry.
        for url in _URLS.findall(candidate):
            if url.rstrip(".,);") not in baseline:
                return f"draft introduced a link absent from the approved text: {url!r}"

        # 5. No fear or pressure language. The prompt forbids "clinical claims,
        #    urgency language about consequences, or anything that could read as
        #    pressure"; this is the enforcement. Deliberately over-broad -- a
        #    false positive costs nothing, because the fallback is the message a
        #    physician already signed off on.
        for phrase in _PRESSURE_TERMS:
            if _word_in(lowered, phrase) and not _word_in(base_lower, phrase):
                return f"draft introduced pressure or clinical-claim language: {phrase!r}"

        if "STOP" in baseline.upper() and "STOP" not in candidate.upper():
            return "opt-out sentence removed"
        if channel == "sms" and len(candidate) > self.max_sms_chars:
            return f"draft is {len(candidate)} chars, over the {self.max_sms_chars} SMS limit"
        return ""


# --------------------------------------------------------------------------
# Reply triage
# --------------------------------------------------------------------------


@dataclass
class ReplyClassification:
    intent: str
    route: str
    confidence: float | None = None
    deterministic: bool = False
    rationale: str = ""
    inference_id: str | None = None
    #: Length only. The reply body is patient-authored text and is not copied
    #: into logs, metadata, or audit records (README 9.2).
    raw_length: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "route": self.route,
            "confidence": self.confidence,
            "deterministic": self.deterministic,
            "raw_length": self.raw_length,
        }


#: Where each intent goes. Nothing here writes to a chart.
ROUTES: Mapping[str, str] = {
    ReplyIntent.OPT_OUT: "suppression:immediate",
    ReplyIntent.OPT_IN: "suppression:release",
    ReplyIntent.HELP: "auto:help_text",
    ReplyIntent.ALREADY_VACCINATED_ELSEWHERE: "queue:ma_reconciliation",
    ReplyIntent.WANTS_APPOINTMENT: "queue:scheduling",
    ReplyIntent.HAS_QUESTIONS: "queue:nurse_line",
    ReplyIntent.DECLINES: "queue:physician_review",
    ReplyIntent.OTHER: "queue:front_desk",
}


class ReplyTriage:
    """Classifies an inbound recall reply and says where it goes."""

    INITIATIVE = "I-02"
    PROMPT_TEMPLATE_ID = "I-02-reply-triage"

    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        audit: Any = None,
        confidence_floor: float = 0.6,
    ) -> None:
        self.client = client
        self.audit = audit
        self.confidence_floor = confidence_floor

    def classify(
        self,
        body: str,
        *,
        patient_id: str = "",
        user_id: str = "system:webhook",
    ) -> ReplyClassification:
        normalised = re.sub(r"[^a-z]", "", (body or "").lower())

        # Deterministic first, unconditionally. A model is never consulted about
        # whether someone said STOP -- README I-02: "'Stop' honors opt-out
        # immediately and permanently", and immediately means before any
        # inference latency, model outage, or misclassification can intervene.
        if normalised in STOP_KEYWORDS:
            return ReplyClassification(
                ReplyIntent.OPT_OUT, ROUTES[ReplyIntent.OPT_OUT],
                deterministic=True, raw_length=len(body or ""),
            )
        if normalised in START_KEYWORDS:
            return ReplyClassification(
                ReplyIntent.OPT_IN, ROUTES[ReplyIntent.OPT_IN],
                deterministic=True, raw_length=len(body or ""),
            )
        if normalised in HELP_KEYWORDS:
            return ReplyClassification(
                ReplyIntent.HELP, ROUTES[ReplyIntent.HELP],
                deterministic=True, raw_length=len(body or ""),
            )

        if self.client is None or not (body or "").strip():
            return ReplyClassification(
                ReplyIntent.OTHER, ROUTES[ReplyIntent.OTHER],
                raw_length=len(body or ""),
            )

        try:
            result = self.client.structured(
                system=REPLY_SYSTEM_PROMPT,
                user=f"REPLY:\n{body}",
                schema=REPLY_SCHEMA,
                prompt_template_id=self.PROMPT_TEMPLATE_ID,
                context={"initiative": self.INITIATIVE, "patient_id": patient_id},
            )
        except SchemaViolation:
            return ReplyClassification(
                ReplyIntent.OTHER, ROUTES[ReplyIntent.OTHER],
                rationale="schema_violation", raw_length=len(body or ""),
            )

        inference_id = None
        if self.audit is not None:
            inference_id = self.audit.record_inference(
                user_id=user_id,
                initiative_id=self.INITIATIVE,
                provider=result.provider,
                model_id=result.model_id,
                model_version=result.model_version,
                prompt_template_id=self.PROMPT_TEMPLATE_ID,
                prompt_template_hash=_hash(REPLY_SYSTEM_PROMPT),
                input_token_count=result.input_token_count,
                output_token_count=result.output_token_count,
                patient_id=patient_id or None,
                confidence_score=result.confidence,
                constrained_decoding=result.constrained,
                repair_attempts=result.repair_attempts,
            )

        intent = str(result.data["intent"])
        confidence = float(result.data["confidence"])
        if confidence < self.confidence_floor:
            intent = ReplyIntent.OTHER
        return ReplyClassification(
            intent=intent,
            route=ROUTES[intent],
            confidence=confidence,
            rationale=str(result.data.get("rationale", "")),
            inference_id=inference_id,
            raw_length=len(body or ""),
        )
