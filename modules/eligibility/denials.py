"""Why a claim was denied, and whether the practice caused it.

README I-09:

    Classify a denial reason | **LLM -- reasonable fit.** Denial reason codes
    are standardized but the accompanying free text is not, and payer-specific
    quirks are numerous.

Note precisely what that sentence licenses. The CARC/RARC codes are standard and
are mapped here in a table, deterministically. The model reads only the
free-text remark, and only to place the denial in one of five buckets the
practice can act on.

THE MEASUREMENT THAT MATTERS. README I-09's target state: *"Eligibility-caused
denials trend-tracked back to the verification process."* That is the number
this whole initiative is judged by. If the practice runs T-3 eligibility checks
and its eligibility-caused denial rate does not fall, the checks are not working
-- and without this classification nobody would know, because a denial rate is
one number that hides five causes moving in different directions.

WHY THE CODE BEATS THE MODEL WHEN THEY DISAGREE. A CARC code is what the payer
formally asserted; the remark is prose somebody typed. When the mapped code says
`eligibility` and the model says `coding`, this module keeps the code and records
the disagreement. A classification layer that overrides a standard with a
guess is a classification layer that drifts.

APPEALS ARE DRAFTED, NEVER SENT. Same posture as everywhere else in this repo:
`AppealDraft` is text for a human to review, sign and send, and it cites only
facts it was handed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Mapping, Sequence

from nsp_core.llm import LLMClient, SchemaViolation

__all__ = [
    "DENIAL_SYSTEM_PROMPT",
    "DENIAL_SCHEMA",
    "RootCause",
    "CARC_ROOT_CAUSE",
    "Denial",
    "Classification",
    "DenialClassifier",
    "DenialReport",
    "build_denial_report",
    "AppealDraft",
    "draft_appeal",
]


class RootCause:
    ELIGIBILITY = "eligibility"
    CODING = "coding"
    AUTHORIZATION = "authorization"
    TIMELY_FILING = "timely_filing"
    OTHER = "other"

    ALL = (ELIGIBILITY, CODING, AUTHORIZATION, TIMELY_FILING, OTHER)
    #: The ones a pre-visit eligibility check could have prevented. This is the
    #: set the trend line is drawn over.
    PREVENTABLE_BY_VERIFICATION = frozenset({ELIGIBILITY})


#: CARC (Claim Adjustment Reason Code) to root cause. The standard half.
#: Deliberately incomplete: a code that is not here is `None`, the model's
#: reading is used instead, and `Classification.from_code` records which
#: happened. Guessing a mapping for an unlisted code would put denials in the
#: eligibility bucket that belong somewhere else, and the eligibility bucket is
#: the one the initiative is measured on.
CARC_ROOT_CAUSE: Mapping[str, str] = {
    "27": RootCause.ELIGIBILITY,   # expenses incurred after coverage terminated
    "26": RootCause.ELIGIBILITY,   # expenses incurred prior to coverage
    "31": RootCause.ELIGIBILITY,   # patient cannot be identified as our insured
    "32": RootCause.ELIGIBILITY,   # our records indicate not an eligible dependent
    "33": RootCause.ELIGIBILITY,   # insured has no dependent coverage
    "177": RootCause.ELIGIBILITY,  # patient has not met the required eligibility
    "200": RootCause.ELIGIBILITY,  # expenses incurred during lapse in coverage
    "4": RootCause.CODING,         # procedure inconsistent with the modifier
    "11": RootCause.CODING,        # diagnosis inconsistent with the procedure
    "16": RootCause.CODING,        # lacks information or has submission errors
    "18": RootCause.CODING,        # exact duplicate claim
    "97": RootCause.CODING,        # payment included in another service
    "181": RootCause.CODING,       # procedure code was invalid on the date
    "182": RootCause.CODING,       # procedure modifier was invalid on the date
    "15": RootCause.AUTHORIZATION,  # missing or invalid authorization number
    "38": RootCause.AUTHORIZATION,  # services not provided or authorized
    "39": RootCause.AUTHORIZATION,  # services denied at the time auth was requested
    "197": RootCause.AUTHORIZATION,  # precertification absent
    "29": RootCause.TIMELY_FILING,  # the time limit for filing has expired
}

DENIAL_SYSTEM_PROMPT = """You are classifying the free-text remark on a denied medical claim for a pediatric practice.

You will receive the payer's remark text and nothing else. Your job is to place
it in one of five buckets so the practice can see which part of its own process
produced the denial.

The buckets:
  eligibility     the patient was not covered, not covered on that date, not
                  identifiable to the payer, or not an eligible dependent
  coding          the codes, modifiers, or claim data were wrong, missing,
                  inconsistent, or duplicated
  authorization   a prior authorization or precertification was required and
                  was absent, expired, or did not match
  timely_filing   the claim was submitted after the payer's filing deadline
  other           anything else, including remarks you cannot place

CONSTRAINTS:
- Use ONLY the remark text. Do not reason from what is typical for a payer, and
  do not infer a cause the remark does not state.
- Choose "other" whenever the remark is ambiguous or you are unsure. "other"
  routes to a person, which is the correct outcome. Guessing puts a denial in
  the wrong trend line and the practice fixes the wrong process.
- Do not recommend an appeal, state whether the denial is correct, or suggest a
  code. You are reading one sentence, not adjudicating a claim.
- `evidence` must be a span copied verbatim from the remark. If you cannot
  quote the words that led you to the bucket, the bucket is "other".

Return ONLY valid JSON:
{
  "root_cause": one of the five buckets,
  "confidence": 0.0-1.0,
  "evidence": string,
  "reasoning": string
}"""

DENIAL_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "root_cause": {"type": "string", "enum": list(RootCause.ALL)},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "evidence": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["root_cause", "confidence", "evidence", "reasoning"],
    "additionalProperties": False,
}

#: Below this the model's reading is not used and the denial goes to a person.
MIN_CONFIDENCE = 0.7

#: The shortest evidence span that can carry a bucket. A bare substring test
#: was satisfied by a SINGLE CHARACTER: `evidence="e"` appears in almost any
#: remark, so the model could place any denial in any bucket and the appeal
#: letter then read "categorised as eligibility on the basis of 'e'". That
#: bucket is the number the whole initiative is measured on.
#:
#: Calibrated against real remark phrasing rather than picked round: "coverage
#: terminated" (19 chars, 2 words) is a legitimate span and must pass, while
#: "the patient" (11 chars) must not -- it is liftable out of "the patient's
#: age" on a coding denial to justify an eligibility bucket.
MIN_EVIDENCE_CHARS = 12
MIN_EVIDENCE_WORDS = 2


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def _grounded(evidence: str, remark: str) -> tuple[bool, str]:
    """Is this span really a quotation from the remark, and is it a quotation?

    Three tests, and each one exists because the single substring check failed
    it. Whitespace is normalised (a model that reflows a line break is still
    quoting); the span must be long enough to carry meaning; and it must sit on
    word boundaries, so `"the patient"` cannot be lifted out of
    `"the patient's age"` to justify an eligibility bucket on a coding denial.
    """
    span = _normalise(evidence)
    haystack = _normalise(remark)
    if not span:
        return False, "it quoted nothing from the remark"
    if len(span) < MIN_EVIDENCE_CHARS or len(span.split()) < MIN_EVIDENCE_WORDS:
        return False, (
            f"its evidence span {evidence!r} is too short to carry a "
            f"classification (needs {MIN_EVIDENCE_CHARS} characters and "
            f"{MIN_EVIDENCE_WORDS} words)"
        )
    # TOKEN SUBSEQUENCE, not a character-boundary check. Boundaries by character
    # class treat an apostrophe as a separator, so "the patient" was accepted as
    # a quotation from "the patient's age" -- which is exactly how a coding
    # denial ended up in the eligibility bucket. Comparing tokens, "patient" and
    # "patient's" are different words and the span does not match.
    span_tokens = span.split()
    remark_tokens = haystack.split()
    matched = any(
        remark_tokens[i : i + len(span_tokens)] == span_tokens
        for i in range(len(remark_tokens) - len(span_tokens) + 1)
    )
    if not matched:
        if span in haystack:
            return False, (
                f"its evidence span {evidence!r} is a fragment of longer words "
                "in the remark, not a quotation from it"
            )
        return False, "its evidence span does not appear in the remark at all"
    return True, ""


@dataclass(frozen=True)
class Denial:
    """One denied claim line, as it arrives on the 835."""

    claim_id: str
    patient_id: str
    payer_id: str
    service_date: date
    denied_on: date
    amount_usd: float
    carc_code: str = ""
    rarc_code: str = ""
    remark_text: str = ""
    cpt_code: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id, "payer_id": self.payer_id,
            "service_date": self.service_date.isoformat(),
            "denied_on": self.denied_on.isoformat(),
            "amount_usd": self.amount_usd,
            "carc_code": self.carc_code, "rarc_code": self.rarc_code,
            "cpt_code": self.cpt_code,
        }


@dataclass
class Classification:
    """Where a denial landed, and which half of the system decided."""

    denial: Denial
    root_cause: str
    #: "carc" when the standard code settled it, "model" when the remark did,
    #: "unclassified" when neither could.
    decided_by: str
    confidence: float | None = None
    evidence: str = ""
    reasoning: str = ""
    #: Set when the mapped CARC and the model's reading disagreed. Worth
    #: watching: a rising disagreement rate means the prompt or the code table
    #: needs attention.
    model_disagreed_with: str = ""
    inference_id: str | None = None
    needs_human: bool = False

    @property
    def preventable_by_verification(self) -> bool:
        return self.root_cause in RootCause.PREVENTABLE_BY_VERIFICATION

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.denial.as_dict(),
            "root_cause": self.root_cause,
            "decided_by": self.decided_by,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "model_disagreed_with": self.model_disagreed_with,
            "needs_human": self.needs_human,
            "preventable_by_verification": self.preventable_by_verification,
        }


@dataclass
class DenialClassifier:
    """Standard codes first; the model only where the standard runs out."""

    client: LLMClient | None = None
    audit: Any = None
    system_prompt: str = DENIAL_SYSTEM_PROMPT
    min_confidence: float = MIN_CONFIDENCE
    INITIATIVE: str = "I-09"

    @property
    def prompt_hash(self) -> str:
        return hashlib.sha256(self.system_prompt.encode("utf-8")).hexdigest()[:16]

    def classify(
        self, denial: Denial, *, user_id: str = "system:eligibility"
    ) -> Classification:
        mapped = CARC_ROOT_CAUSE.get(denial.carc_code.strip())
        model_reading: dict[str, Any] | None = None
        inference_id: str | None = None

        if self.client is not None and denial.remark_text.strip():
            try:
                result = self.client.structured(
                    system=self.system_prompt,
                    user=denial.remark_text,
                    schema=DENIAL_SCHEMA,
                    context={
                        "initiative_id": self.INITIATIVE,
                        "user_id": user_id,
                        "claim_id": denial.claim_id,
                    },
                    prompt_template_id="I-09/classify-denial",
                    temperature=0.0,
                )
                model_reading = dict(result.data)
                if self.audit is not None:
                    inference_id = self.audit.record_inference(
                        user_id=user_id,
                        initiative_id=self.INITIATIVE,
                        provider=result.provider,
                        model_id=result.model_id,
                        model_version=result.model_version,
                        prompt_template_id=result.prompt_template_id,
                        prompt_template_hash=result.prompt_template_hash,
                        input_token_count=result.input_token_count,
                        output_token_count=result.output_token_count,
                        confidence_score=float(model_reading["confidence"]),
                        constrained_decoding=result.constrained,
                        repair_attempts=result.repair_attempts,
                        extra={"claim_id": denial.claim_id},
                    )
            except SchemaViolation:
                model_reading = None

        # The standard wins. Always.
        if mapped is not None:
            disagreed = ""
            if model_reading and model_reading["root_cause"] != mapped:
                disagreed = str(model_reading["root_cause"])
            return Classification(
                denial=denial,
                root_cause=mapped,
                decided_by="carc",
                confidence=None,
                evidence=f"CARC {denial.carc_code}",
                reasoning=(
                    f"CARC {denial.carc_code} is a standard code mapping to "
                    f"{mapped}. The payer formally asserted this; the free-text "
                    "remark is prose somebody typed."
                ),
                model_disagreed_with=disagreed,
                inference_id=inference_id,
            )

        if model_reading is None:
            return Classification(
                denial=denial,
                root_cause=RootCause.OTHER,
                decided_by="unclassified",
                reasoning=(
                    f"CARC {denial.carc_code!r} is not in the mapping table and "
                    "the remark could not be read. This denial is not counted "
                    "against any process until a person places it."
                ),
                needs_human=True,
                inference_id=inference_id,
            )

        confidence = float(model_reading["confidence"])
        evidence = str(model_reading["evidence"])
        grounded, ground_reason = _grounded(evidence, denial.remark_text)
        if confidence < self.min_confidence or not grounded:
            # A post-condition, not a prompt instruction. The prompt asks for a
            # verbatim span; this checks that one came back. An "evidence" field
            # nobody verifies is a field the model can fill with anything.
            return Classification(
                denial=denial,
                root_cause=RootCause.OTHER,
                decided_by="unclassified",
                confidence=confidence,
                evidence=evidence,
                reasoning=(
                    f"the model read this as {model_reading['root_cause']!r} at "
                    f"{confidence:.2f} confidence"
                    + (
                        f" but {ground_reason}"
                        if not grounded else
                        f", below the {self.min_confidence} floor"
                    )
                    + ". Left unclassified for a person."
                ),
                needs_human=True,
                inference_id=inference_id,
            )

        return Classification(
            denial=denial,
            root_cause=str(model_reading["root_cause"]),
            decided_by="model",
            confidence=confidence,
            evidence=evidence,
            reasoning=str(model_reading["reasoning"]),
            inference_id=inference_id,
        )


@dataclass
class DenialReport:
    """Denials by root cause, and the trend the initiative is judged on."""

    period: str
    classifications: list[Classification] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.classifications)

    @property
    def total_usd(self) -> float:
        return round(sum(c.denial.amount_usd for c in self.classifications), 2)

    def by_root_cause(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {
            cause: {"count": 0, "amount_usd": 0.0} for cause in RootCause.ALL
        }
        for item in self.classifications:
            row = out[item.root_cause]
            row["count"] += 1
            row["amount_usd"] = round(row["amount_usd"] + item.denial.amount_usd, 2)
        return out

    def by_payer(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.classifications:
            counts[item.denial.payer_id] = counts.get(item.denial.payer_id, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    @property
    def eligibility_caused(self) -> list[Classification]:
        return [c for c in self.classifications if c.preventable_by_verification]

    @property
    def unclassified(self) -> list[Classification]:
        return [c for c in self.classifications if c.needs_human]

    @property
    def disagreements(self) -> list[Classification]:
        return [c for c in self.classifications if c.model_disagreed_with]

    def as_dict(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "total": self.total,
            "total_usd": self.total_usd,
            "by_root_cause": self.by_root_cause(),
            "by_payer": self.by_payer(),
            "eligibility_caused": len(self.eligibility_caused),
            "eligibility_caused_usd": round(
                sum(c.denial.amount_usd for c in self.eligibility_caused), 2
            ),
            "unclassified": len(self.unclassified),
            "model_disagreements": len(self.disagreements),
        }


def build_denial_report(
    period: str, classifications: Sequence[Classification]
) -> DenialReport:
    return DenialReport(period=period, classifications=list(classifications))


@dataclass
class AppealDraft:
    """Text for a person to review, sign and send. Never sent from here."""

    claim_id: str
    body: str
    cited_facts: tuple[str, ...]
    requires_review_by: str = "billing_lead"

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "body": self.body,
            "cited_facts": list(self.cited_facts),
            "requires_review_by": self.requires_review_by,
        }


def draft_appeal(
    classification: Classification,
    *,
    practice_name: str,
    facts: Sequence[str],
) -> AppealDraft:
    """Assemble an appeal from facts the CALLER supplies. No model, no invention.

    README I-09 lists appeal drafting as a good model fit, and it is -- for
    prose. But the prose is the cheap part: what makes an appeal succeed is a
    correct claim number, a correct date, and a citation to something that
    actually happened. So this builds the letter from facts handed in, and a
    practice that wants a model to smooth the wording runs it over this output
    with a person reading the result.

    An appeal that cites a fact nobody checked is worse than no appeal: it goes
    on the record, and the payer keeps it.
    """
    if not facts:
        raise ValueError(
            "an appeal cites facts. Drafting one with nothing to cite produces a "
            "letter that says the practice disagrees, which the payer already knew."
        )
    denial = classification.denial
    lines = [
        f"Re: appeal of denied claim {denial.claim_id}",
        f"Date of service: {denial.service_date.isoformat()}",
        f"Denied: {denial.denied_on.isoformat()}"
        + (f" (CARC {denial.carc_code})" if denial.carc_code else ""),
        f"Amount: ${denial.amount_usd:,.2f}",
        "",
        f"{practice_name} requests reconsideration of the above claim.",
        "",
        "The denial was categorised as "
        f"{classification.root_cause.replace('_', ' ')}"
        + (
            f" on the basis of {classification.evidence!r}."
            if classification.evidence else "."
        ),
        "",
        "The following facts are submitted in support of this appeal:",
    ]
    for index, fact in enumerate(facts, start=1):
        lines.append(f"  {index}. {fact}")
    lines += [
        "",
        "We ask that the claim be reprocessed. Supporting documentation is",
        "attached.",
        "",
        "[REVIEW BEFORE SENDING - every fact above must be verified against the",
        " chart and the claim. This draft was assembled, not adjudicated.]",
    ]
    return AppealDraft(
        claim_id=denial.claim_id,
        body="\n".join(lines),
        cited_facts=tuple(facts),
    )
