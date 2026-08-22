"""LLM adjudication for the small set of pairs the rules cannot settle.

This is the ONLY place in the immunization module where a model is called, and
it answers exactly one question: are these two dose records the same
administration event, or two different ones? It is never asked whether a child
is due for a vaccine. README I-02: "Using an LLM to determine whether a child is
due for a vaccine would be actively negligent."

The prompt is README Appendix A.2, verbatim. It is stored as a module constant
with a hash so that a change to it is visible in a diff and in the audit log
(`prompt_template_hash`), because README 9.4 is right that prompts are code.

Four controls sit around the call, in order of how much they matter:

1. **Minimum necessary, enforced by construction.** The payload is built from
   structured fields -- CVX code, date, source, product text -- and never from a
   chart note. There is no name, no MRN, no address to strip, because none is
   ever assembled. A `LeakGuard` pass runs anyway, with DATE explicitly allowed
   because the task is date comparison and stripping dates would make it
   unanswerable.

2. **The no-invented-dates rule is verified, not requested.** A.2 tells the
   model it "MUST NOT infer, estimate, or reconstruct a date that appears in
   neither record." `_verify_grounding` checks that every date and CVX code in
   the response actually appears in the input. A response that invents one is
   discarded and routed to a human -- an instruction the model can ignore is a
   suggestion; a post-condition it cannot get past is a control.

3. **UNCERTAIN is a first-class success.** The build plan is explicit that it
   "is a CORRECT outcome that routes to a human -- do not tune it out". Nothing
   here retries an UNCERTAIN, lowers a threshold to avoid one, or treats one as
   a failure metric.

4. **Every call is audited and nothing is auto-applied.** An adjudication is a
   *proposal*. It changes the reconciliation only after
   `apply_adjudications(..., reviewed_by=...)`, which requires a named human.
   README 3.4: human in the loop is a legal requirement here, not a preference.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Mapping, Sequence

from nsp_core.llm import LLMClient, SchemaViolation
from nsp_core.phi import LeakGuard

from .matcher import AmbiguousPair, Determination, MatchedPair, Reconciliation

__all__ = [
    "ADJUDICATION_SYSTEM_PROMPT",
    "ADJUDICATION_SCHEMA",
    "PROMPT_TEMPLATE_ID",
    "AdjudicationOutcome",
    "Adjudicator",
    "HumanReviewItem",
    "apply_adjudications",
]

PROMPT_TEMPLATE_ID = "README-A.2"

#: README Appendix A.2, verbatim. Do not paraphrase, tighten, or "improve" this
#: without a change-control entry (README 9.4) -- the hash of this string is in
#: every audit record and is how a prompt change is correlated with a change in
#: output quality.
ADJUDICATION_SYSTEM_PROMPT = """You are reconciling two immunization records for the same patient.

RECORD A (practice chart) and RECORD B (state registry) are provided below.

For each dose pair presented, determine whether they represent THE SAME
administration event or TWO DISTINCT events.

CONSTRAINTS:
- You may ONLY use information present in the two records.
- You MUST NOT infer, estimate, or reconstruct a date that appears in neither
  record.
- You MUST NOT determine whether a patient is due for a vaccine. That is
  computed by a separate rules engine.
- When uncertain, return UNCERTAIN. Returning UNCERTAIN is a correct and
  expected outcome; a human will review it. Guessing is not.

Consider: CVX code equivalence, historical trade-name to generic mappings,
partial or approximate dates, and combination-vaccine components recorded
separately in one source and jointly in the other.

Return ONLY valid JSON:
{
  "determination": "MATCH" | "NO_MATCH" | "UNCERTAIN",
  "confidence": 0.0-1.0,
  "reasoning": string,
  "cvx_a": string|null,
  "cvx_b": string|null,
  "date_a": string|null,
  "date_b": string|null,
  "requires_human_review": boolean
}"""

#: The A.2 return shape, strict. Note what is NOT here: no "due", no "status",
#: no "recommendation". The schema is the architectural enforcement of README
#: R-06 (scope creep into clinical decision support) -- there is no field for
#: the model to put a clinical judgement in even if it wanted to.
ADJUDICATION_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "determination",
        "confidence",
        "reasoning",
        "cvx_a",
        "cvx_b",
        "date_a",
        "date_b",
        "requires_human_review",
    ],
    "properties": {
        "determination": {"type": "string", "enum": ["MATCH", "NO_MATCH", "UNCERTAIN"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning": {"type": "string", "maxLength": 1200},
        "cvx_a": {"type": ["string", "null"]},
        "cvx_b": {"type": ["string", "null"]},
        "date_a": {"type": ["string", "null"]},
        "date_b": {"type": ["string", "null"]},
        "requires_human_review": {"type": "boolean"},
    },
}


def _prompt_hash() -> str:
    return hashlib.sha256(ADJUDICATION_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]


@dataclass
class AdjudicationOutcome:
    pair: AmbiguousPair
    determination: str
    confidence: float | None
    reasoning: str
    requires_human_review: bool
    inference_id: str | None = None
    failure: str = ""

    @property
    def auto_resolvable(self) -> bool:
        """MATCH or NO_MATCH, confidently, with nothing flagged for a human.

        Even this only means the pair may be *proposed* for resolution.
        `apply_adjudications` still requires a named reviewer.
        """
        return (
            not self.failure
            and not self.requires_human_review
            and self.determination in (Determination.MATCH, Determination.NO_MATCH)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "chart_record_id": self.pair.chart.record_id,
            "registry_record_id": self.pair.registry.record_id,
            "determination": self.determination,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "requires_human_review": self.requires_human_review,
            "failure": self.failure,
        }


@dataclass
class HumanReviewItem:
    """One row in the MA reconciliation work queue."""

    patient_id: str
    pair: AmbiguousPair
    reason: str
    machine_suggestion: str | None = None
    machine_confidence: float | None = None
    machine_reasoning: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "reason": self.reason,
            "machine_suggestion": self.machine_suggestion,
            "machine_confidence": self.machine_confidence,
            "machine_reasoning": self.machine_reasoning,
            **self.pair.as_dict(),
        }


class Adjudicator:
    """Adjudicates ambiguous pairs. Nothing else in this module calls a model."""

    INITIATIVE = "I-02"

    def __init__(
        self,
        client: LLMClient,
        *,
        audit: Any = None,
        confidence_floor: float = 0.85,
        leak_guard: LeakGuard | None = None,
    ) -> None:
        self.client = client
        self.audit = audit
        # A pair the model is only 60% sure about is a pair a human should see.
        # This threshold gates AUTO-RESOLUTION, not the model's freedom to
        # answer: a low-confidence MATCH is still recorded and still shown to
        # the reviewer as a suggestion.
        self.confidence_floor = confidence_floor
        self.leak_guard = leak_guard or LeakGuard(allow_labels={"DATE"})

    # -- payload -----------------------------------------------------------

    def build_payload(self, pair: AmbiguousPair) -> str:
        """Assemble the minimum-necessary comparison. Structured fields only.

        Nothing here is drawn from a note, a comment field, or anything else
        that could carry an identifier. This is README 3.3 -- minimum necessary,
        enforced architecturally -- implemented as "the function that builds the
        prompt has no access to the patient record".
        """
        def render(record: Any, label: str) -> str:
            lines = [
                f"RECORD {label} (source: {record.source})",
                f"  cvx: {record.normalised_cvx or record.cvx}",
                f"  date: {record.given.isoformat()} (precision: {record.precision})",
            ]
            if record.product_text:
                lines.append(f"  product as recorded: {record.product_text}")
            if record.lot:
                lines.append(f"  lot: {record.lot}")
            return "\n".join(lines)

        body = (
            f"{render(pair.chart, 'A')}\n\n{render(pair.registry, 'B')}\n\n"
            f"Rules-engine note: {pair.reason}"
        )
        # Belt and braces. The payload is built from structured fields, so this
        # should never fire -- which is exactly why it is worth asserting.
        self.leak_guard.assert_clean(body, allow_labels={"DATE"})
        return body

    # -- one pair ----------------------------------------------------------

    def adjudicate(
        self,
        pair: AmbiguousPair,
        *,
        patient_id: str,
        user_id: str = "system:nightly",
    ) -> AdjudicationOutcome:
        payload = self.build_payload(pair)
        try:
            result = self.client.structured(
                system=ADJUDICATION_SYSTEM_PROMPT,
                user=payload,
                schema=ADJUDICATION_SCHEMA,
                prompt_template_id=PROMPT_TEMPLATE_ID,
                context={"initiative": self.INITIATIVE, "patient_id": patient_id},
            )
        except SchemaViolation as exc:
            # Fail closed (README 3.5). No retry loop with a looser schema, no
            # default determination, no silent skip.
            self._record_failure(patient_id, user_id, str(exc))
            return AdjudicationOutcome(
                pair=pair,
                determination=Determination.AMBIGUOUS,
                confidence=None,
                reasoning="",
                requires_human_review=True,
                failure=f"schema_violation: {exc}",
            )

        inference_id = None
        if self.audit is not None:
            inference_id = self.audit.record_inference(
                user_id=user_id,
                initiative_id=self.INITIATIVE,
                provider=result.provider,
                model_id=result.model_id,
                model_version=result.model_version,
                prompt_template_id=PROMPT_TEMPLATE_ID,
                prompt_template_hash=_prompt_hash(),
                input_token_count=result.input_token_count,
                output_token_count=result.output_token_count,
                patient_id=patient_id,
                confidence_score=result.confidence,
                constrained_decoding=result.constrained,
                repair_attempts=result.repair_attempts,
                extra={"pair_reason": pair.reason, "antigens": list(pair.antigens)},
            )

        data = result.data
        grounding_failure = _verify_grounding(data, pair)
        determination = str(data["determination"])
        confidence = float(data["confidence"])
        requires_review = bool(data["requires_human_review"])

        if grounding_failure:
            return AdjudicationOutcome(
                pair=pair,
                determination=Determination.AMBIGUOUS,
                confidence=confidence,
                reasoning=str(data.get("reasoning", "")),
                requires_human_review=True,
                inference_id=inference_id,
                failure=grounding_failure,
            )

        # UNCERTAIN always goes to a human. So does anything under the floor.
        # Neither is a failure; both are the system working.
        if determination == "UNCERTAIN" or confidence < self.confidence_floor:
            requires_review = True

        return AdjudicationOutcome(
            pair=pair,
            determination=determination,
            confidence=confidence,
            reasoning=str(data.get("reasoning", "")),
            requires_human_review=requires_review,
            inference_id=inference_id,
        )

    def _record_failure(self, patient_id: str, user_id: str, detail: str) -> None:
        if self.audit is None:
            return
        self.audit.record_event(
            actor_id=user_id,
            initiative_id=self.INITIATIVE,
            event_type="adjudication_schema_violation",
            patient_id=patient_id,
            # Length only. The detail string is derived from model output and
            # the raw output never enters the log (README 9.2).
            detail={"detail_length": len(detail)},
        )

    # -- a whole reconciliation -------------------------------------------

    def adjudicate_reconciliation(
        self,
        reconciliation: Reconciliation,
        *,
        patient_id: str,
        user_id: str = "system:nightly",
    ) -> list[AdjudicationOutcome]:
        return [
            self.adjudicate(pair, patient_id=patient_id, user_id=user_id)
            for pair in reconciliation.ambiguous
        ]


def _verify_grounding(data: Mapping[str, Any], pair: AmbiguousPair) -> str:
    """Check the response only contains facts that were in the input.

    A.2 forbids inventing a date. This function is what makes that forbiddance
    load-bearing: any date or CVX code in the response that is not one of the
    two supplied values means the model reconstructed something, and the whole
    response is discarded regardless of how confident it was.
    """
    permitted_dates = {
        pair.chart.given.isoformat(),
        pair.registry.given.isoformat(),
    }
    permitted_codes = {
        str(pair.chart.normalised_cvx or pair.chart.cvx),
        str(pair.registry.normalised_cvx or pair.registry.cvx),
    }
    for key, permitted in (
        ("date_a", permitted_dates),
        ("date_b", permitted_dates),
        ("cvx_a", permitted_codes),
        ("cvx_b", permitted_codes),
    ):
        value = data.get(key)
        if value is None:
            continue
        if str(value).strip() not in permitted:
            return (
                f"ungrounded_{key}: response contains a value that appears in "
                "neither record"
            )
    return ""


def apply_adjudications(
    reconciliation: Reconciliation,
    outcomes: Sequence[AdjudicationOutcome],
    *,
    reviewed_by: str,
    decisions: Mapping[tuple[str, str], str] | None = None,
    audit: Any = None,
    patient_id: str = "",
) -> tuple[Reconciliation, list[HumanReviewItem]]:
    """Fold reviewed adjudications back into the reconciliation.

    `reviewed_by` is required and must name a person. There is no
    `reviewed_by=None` path and no `auto_apply=True` flag, because README 3.4
    makes human review a legal requirement rather than a configuration option:
    an unreviewed machine conclusion must not reach a chart.

    `decisions` lets the reviewer override a machine determination per pair,
    keyed by (chart_record_id, registry_record_id). An override is recorded as
    action_taken="edited", which is how the programme learns that the prompt or
    the rules need work.

    Returns the updated reconciliation and the queue of items still needing a
    human. Anything the model was unsure about, anything it got wrong on
    grounding, and anything below the confidence floor lands in that queue.
    """
    decisions = dict(decisions or {})
    if not reviewed_by or not reviewed_by.strip():
        raise ValueError(
            "apply_adjudications requires a named reviewer; machine conclusions "
            "are not applied to a chart without one (README 3.4)"
        )

    by_pair = {
        (o.pair.chart.record_id, o.pair.registry.record_id): o for o in outcomes
    }
    updated = Reconciliation(
        matched=list(reconciliation.matched),
        ambiguous=[],
        chart_only=list(reconciliation.chart_only),
        registry_only=list(reconciliation.registry_only),
        duplicates=list(reconciliation.duplicates),
        unknown_codes=list(reconciliation.unknown_codes),
    )
    queue: list[HumanReviewItem] = []
    matched_chart_ids: set[str] = set()
    pending_chart_only: dict[str, Any] = {}
    pending_registry_only: dict[str, Any] = {}

    for pair in reconciliation.ambiguous:
        outcome = by_pair.get((pair.chart.record_id, pair.registry.record_id))
        if outcome is None or not outcome.auto_resolvable:
            updated.ambiguous.append(pair)
            queue.append(
                HumanReviewItem(
                    patient_id=patient_id,
                    pair=pair,
                    reason=(outcome.failure or "model returned UNCERTAIN or low confidence")
                    if outcome
                    else "not adjudicated",
                    machine_suggestion=outcome.determination if outcome else None,
                    machine_confidence=outcome.confidence if outcome else None,
                    machine_reasoning=outcome.reasoning if outcome else "",
                )
            )
            continue

        decision = decisions.get(
            (pair.chart.record_id, pair.registry.record_id), outcome.determination
        )
        if decision not in (Determination.MATCH, Determination.NO_MATCH):
            raise ValueError(
                f"reviewer decision must be MATCH or NO_MATCH, got {decision!r}"
            )

        if decision == Determination.MATCH:
            matched_chart_ids.add(pair.chart.record_id)
            updated.matched.append(
                MatchedPair(
                    chart=pair.chart,
                    registry=pair.registry,
                    date_delta_days=abs((pair.chart.given - pair.registry.given).days),
                    rule=f"adjudicated MATCH, confirmed by {reviewed_by}",
                )
            )
        else:
            # One combination product legitimately appears in several ambiguous
            # pairs -- Pediarix against three registry component rows. Appending
            # its chart record once per pair would report a child with one dose
            # as having had three, which is the phantom-record failure this whole
            # module exists to prevent. Each record moves to a bucket at most
            # once, and never to chart_only if another pair matched it.
            pending_chart_only[pair.chart.record_id] = pair.chart
            pending_registry_only[pair.registry.record_id] = pair.registry

        if audit is not None:
            audit.record_review(
                reviewer_id=reviewed_by,
                initiative_id=Adjudicator.INITIATIVE,
                draft=outcome.determination,
                final=decision,
                action_taken="accepted" if decision == outcome.determination else "edited",
                inference_id=outcome.inference_id,
                patient_id=patient_id or None,
                # A binary confirmation has no text to edit. Counting it toward
                # the rubber-stamp median would make a reviewer who correctly
                # confirms correct adjudications look like one who has stopped
                # reading, and a false alarm is how a real alarm gets ignored.
                edit_rate_applicable=False,
                extra={
                    "chart_record_id": pair.chart.record_id,
                    "registry_record_id": pair.registry.record_id,
                    "review_type": "binary_decision",
                },
            )

    for record_id, record in pending_chart_only.items():
        if record_id not in matched_chart_ids:
            updated.chart_only.append(record)
    matched_registry_ids = {p.registry.record_id for p in updated.matched}
    for record_id, record in pending_registry_only.items():
        if record_id not in matched_registry_ids:
            updated.registry_only.append(record)
    return updated, queue
