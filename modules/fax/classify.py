"""Document type, from a fixed taxonomy, measured rather than assumed.

README I-06 makes the case for a model here and it is a good one: "Faxes are
unstructured, inconsistently formatted, frequently poor quality. Rule-based
classification on faxes fails constantly; this is exactly where LLM robustness
pays." The build plan adds the condition: "Small model is fine -- measure with
an eval set, don't guess."

So this module is half a classifier and half a measuring instrument, and the
measuring half is the part that decides whether the classifier is allowed to
route anything.

TWO THINGS THAT ARE NOT NEGOTIABLE HERE.

**The taxonomy is closed.** `DOCUMENT_TYPES` is README I-06's list verbatim and
the JSON schema constrains the field with `enum`. A classifier that can invent a
category invents one on the document it understands least, and that document is
then routed by a rule that was never written for it. `other` is the escape
hatch, and `other` goes to a human.

**Per-class recall is not one number.** An overall accuracy figure hides the
only failures that matter. Mistaking `records_request` for
`insurance_correspondence` costs somebody ten seconds of re-filing. Mistaking
`outside_lab` for `records_request` buries a critical value in an administrative
queue, which is README I-06's CRITICAL risk. `EvalResult` therefore carries a
confusion matrix and a per-class recall floor, and `ClassifierGate` refuses to
let a classifier route production documents until the floors are met -- the same
shape as I-04's `ModelPin`, for the same reason: a measurement nobody gates on
is a report, not a control.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from nsp_core.llm import LLMClient, SchemaViolation

__all__ = [
    "DOCUMENT_TYPES",
    "CLINICAL_TYPES",
    "CLASSIFY_SYSTEM_PROMPT",
    "CLASSIFY_SCHEMA",
    "MIN_CONFIDENCE",
    "RECALL_FLOORS",
    "MAX_LOW_CONFIDENCE_RATE",
    "MAX_CLINICAL_LEAK_RATE",
    "evaluate",
    "Classification",
    "DocumentClassifier",
    "EvalCase",
    "EvalResult",
    "ClassifierGate",
    "ClassifierNotValidated",
]

#: README I-06, verbatim and closed.
DOCUMENT_TYPES: tuple[str, ...] = (
    "specialist_consult",
    "hospital_discharge",
    "outside_lab",
    "outside_imaging",
    "prior_auth",
    "records_request",
    "school_form",
    "insurance_correspondence",
    "immunization_record",
    "other",
)

#: Types whose misrouting has a clinical cost rather than an administrative one.
#: These carry the strict recall floors and they never auto-file without a task.
CLINICAL_TYPES: frozenset[str] = frozenset(
    {
        "specialist_consult",
        "hospital_discharge",
        "outside_lab",
        "outside_imaging",
        "immunization_record",
    }
)

#: Below this the classifier's own answer is not used and the document goes to a
#: human. README I-06: "Include a `confidence` field and honor it."
MIN_CONFIDENCE = 0.75

#: Fraction of each class the classifier must actually find in the eval set
#: before it is allowed to route. Clinical types are held higher because their
#: failure mode is a buried result rather than a re-file.
RECALL_FLOORS: Mapping[str, float] = {
    "outside_lab": 0.95,
    "hospital_discharge": 0.95,
    "immunization_record": 0.90,
    "specialist_consult": 0.90,
    "outside_imaging": 0.90,
    "prior_auth": 0.80,
    "records_request": 0.75,
    "school_form": 0.75,
    "insurance_correspondence": 0.75,
    "other": 0.50,
}

CLASSIFY_SYSTEM_PROMPT = """You are classifying an inbound fax for a pediatric practice.

You will receive the OCR text of one document. The text may be poor quality,
truncated, or contain fragments of a cover sheet.

Assign exactly one document type from this fixed list and nothing else:
  specialist_consult      a consultation note or letter from a specialist
  hospital_discharge      a discharge summary or after-visit summary from a hospital
  outside_lab             laboratory results from an outside facility
  outside_imaging         radiology or imaging results from an outside facility
  prior_auth              prior authorization correspondence from a payer
  records_request         a request for records, or records sent in response to one
  school_form             a school, camp, daycare or sports participation form
  insurance_correspondence  eligibility, benefits, claims or coverage correspondence
  immunization_record      an immunization history or registry printout
  other                   anything that is none of the above

CONSTRAINTS:
- You MUST NOT invent a type outside the list. Use "other" when unsure.
- You MUST NOT summarise clinical content, give an opinion on it, or state
  anything the document does not say.
- If the document contains more than one kind of content, choose the type of the
  content a clinician would most need to see.
- Report a patient name and date of birth ONLY if they appear in the text.
  Copy them exactly as written. Do not correct, complete or infer them.
- Set confidence below 0.75 whenever the text is too poor or too ambiguous to
  be sure. A low confidence sends the document to a person, which is the correct
  outcome.

Return ONLY valid JSON:
{
  "document_type": one of the types above,
  "confidence": 0.0-1.0,
  "patient_name": string or null,
  "patient_dob": string or null,
  "sending_facility": string or null,
  "one_line_summary": string
}"""

_STR_OR_NULL = {"type": ["string", "null"]}

CLASSIFY_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "document_type": {"type": "string", "enum": list(DOCUMENT_TYPES)},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "patient_name": _STR_OR_NULL,
        "patient_dob": _STR_OR_NULL,
        "sending_facility": _STR_OR_NULL,
        "one_line_summary": {"type": "string"},
    },
    "required": [
        "document_type", "confidence", "patient_name", "patient_dob",
        "sending_facility", "one_line_summary",
    ],
    "additionalProperties": False,
}


class ClassifierNotValidated(RuntimeError):
    """Raised when an unmeasured classifier is asked to route documents."""


@dataclass
class Classification:
    document_type: str
    confidence: float
    patient_name: str | None
    patient_dob: str | None
    sending_facility: str | None
    one_line_summary: str
    model_id: str = ""
    model_version: str = ""
    provider: str = ""
    prompt_hash: str = ""
    inference_id: str | None = None
    warnings: list[str] = field(default_factory=list)
    #: The threshold this answer was judged against, carried on the answer
    #: itself. An earlier version read the module constant here, so
    #: `DocumentClassifier(min_confidence=0.95)` was accepted, stored, and had
    #: no effect on a single routing decision.
    min_confidence: float = MIN_CONFIDENCE

    @property
    def confident(self) -> bool:
        return self.confidence >= self.min_confidence

    @property
    def is_clinical(self) -> bool:
        return self.document_type in CLINICAL_TYPES

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_type": self.document_type,
            "confidence": round(self.confidence, 3),
            "patient_name": self.patient_name,
            "patient_dob": self.patient_dob,
            "sending_facility": self.sending_facility,
            "one_line_summary": self.one_line_summary,
            "model": f"{self.provider}:{self.model_id}@{self.model_version}",
            "inference_id": self.inference_id,
            "warnings": list(self.warnings),
        }


class DocumentClassifier:
    """One model call per document, against a closed taxonomy."""

    INITIATIVE = "I-06"

    def __init__(
        self,
        client: LLMClient,
        *,
        audit: Any = None,
        deidentifier: Any = None,
        min_confidence: float = MIN_CONFIDENCE,
        system_prompt: str = CLASSIFY_SYSTEM_PROMPT,
    ) -> None:
        self.client = client
        self.audit = audit
        self.deidentifier = deidentifier
        if min_confidence < MIN_CONFIDENCE:
            # Raising the bar is a local decision. Lowering it silently sends
            # documents the classifier is unsure about into automatic routing,
            # which is the failure README I-06 asks the confidence field to
            # prevent. A threshold may tighten; it may not widen.
            raise ValueError(
                f"min_confidence={min_confidence} is below the module floor "
                f"{MIN_CONFIDENCE}; a confidence threshold may be raised, never "
                "lowered"
            )
        self.min_confidence = min_confidence
        self.system_prompt = system_prompt
        if set(CLASSIFY_SCHEMA["properties"]["document_type"]["enum"]) != set(DOCUMENT_TYPES):
            raise ValueError("the schema enum has drifted from DOCUMENT_TYPES")

    @property
    def prompt_hash(self) -> str:
        return hashlib.sha256(self.system_prompt.encode("utf-8")).hexdigest()[:16]

    # Identity of the thing actually doing the classifying. `ClassifierGate`
    # compares these to what it approved: a validated prompt in front of a
    # swapped model is an unmeasured classifier wearing a measured prompt.
    @property
    def model_id(self) -> str:
        return getattr(self.client.transport, "model_id", "")

    @property
    def model_version(self) -> str:
        return getattr(self.client.transport, "model_version", "")

    @property
    def provider(self) -> str:
        return self.client.provider

    def classify(
        self,
        text: str,
        *,
        document_id: str,
        user_id: str = "system:fax",
        max_chars: int = 12_000,
    ) -> Classification | None:
        """Classify, or return None so the caller queues it for a human.

        Returns None rather than raising on a schema violation: one unclassifiable
        fax must not stop the morning's batch, and hard constraint 3 is satisfied
        because nothing is guessed -- the document goes to a person with the
        reason attached.

        NOTE the identity fields are NOT de-identified before the call even when
        a de-identifier is configured. Matching the document to a chart is the
        entire task; a tokenised name matches nothing. That is a reason to run
        this locally (README 3.1), which is the default.
        """
        # Truncated at a page-ish boundary rather than mid-word, and reported.
        excerpt = text[:max_chars]
        warnings: list[str] = []
        if len(text) > max_chars:
            warnings.append(
                f"document is {len(text)} characters; the classifier saw the "
                f"first {max_chars}"
            )
        try:
            result = self.client.structured(
                system=self.system_prompt,
                user=excerpt,
                schema=CLASSIFY_SCHEMA,
                context={
                    "initiative_id": self.INITIATIVE,
                    "user_id": user_id,
                    "document_id": document_id,
                },
                prompt_template_id="I-06/classify",
                temperature=0.0,
            )
        except SchemaViolation:
            return None

        data = result.data
        classification = Classification(
            document_type=str(data["document_type"]),
            confidence=float(data["confidence"]),
            patient_name=data["patient_name"],
            patient_dob=data["patient_dob"],
            sending_facility=data["sending_facility"],
            one_line_summary=str(data["one_line_summary"]),
            model_id=result.model_id,
            model_version=result.model_version,
            provider=result.provider,
            prompt_hash=self.prompt_hash,
            warnings=warnings,
            min_confidence=self.min_confidence,
        )
        # Belt and braces on the enum: the schema constrains it, and a transport
        # without native constrained decoding validates rather than constrains,
        # so the check runs again on the value that came back.
        if classification.document_type not in DOCUMENT_TYPES:
            return None
        if self.audit is not None:
            classification.inference_id = self.audit.record_inference(
                user_id=user_id,
                initiative_id=self.INITIATIVE,
                provider=result.provider,
                model_id=result.model_id,
                model_version=result.model_version,
                prompt_template_id=result.prompt_template_id,
                prompt_template_hash=result.prompt_template_hash,
                input_token_count=result.input_token_count,
                output_token_count=result.output_token_count,
                confidence_score=classification.confidence,
                constrained_decoding=result.constrained,
                repair_attempts=result.repair_attempts,
                extra={
                    "document_id": document_id,
                    "document_type": classification.document_type,
                },
            )
        return classification


# -- measurement -------------------------------------------------------------


@dataclass(frozen=True)
class EvalCase:
    """One labelled document. The label is a human's, not a model's."""

    document_id: str
    text: str
    expected_type: str
    note: str = ""


#: Ceiling on the share of the eval set the classifier was too unsure to route.
#: A classifier that is never confident scores perfectly on every document it
#: was allowed to answer for, and routes nothing.
MAX_LOW_CONFIDENCE_RATE = 0.20

#: Ceiling on clinical-to-administrative confusion, as a share of clinical
#: documents. Zero is the right target and the wrong gate: over 300 real faxes a
#: single unlucky cover sheet would block a classifier that is otherwise safe,
#: and a gate that can never pass gets switched off. Two percent is a budget,
#: and `clinical_leaks()` still lists every one of them.
MAX_CLINICAL_LEAK_RATE = 0.02


@dataclass
class EvalResult:
    total: int = 0
    correct: int = 0
    unclassifiable: int = 0
    low_confidence: int = 0
    #: expected -> predicted -> count. Two synthetic predicted-labels appear
    #: here: `__unclassifiable__` (schema violation) and `__low_confidence__`
    #: (the model answered, below threshold, so production sends it to a human).
    confusion: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    #: sha256 of each document text -> how many eval cases carried it. A tiled
    #: corpus inflates `total` without adding a single new document.
    text_digests: Counter = field(default_factory=Counter)
    model_id: str = ""
    model_version: str = ""
    provider: str = ""
    prompt_hash: str = ""
    #: The confidence threshold in force during the run being scored.
    min_confidence: float = MIN_CONFIDENCE

    @property
    def accuracy(self) -> float:
        """Share of documents ROUTED correctly, not answered correctly.

        A below-threshold answer is not counted as correct even when the label
        matches, because the production path never uses it -- it hands the
        document to a person. Scoring it as a hit measures a decision the
        system does not make.
        """
        return self.correct / self.total if self.total else 0.0

    @property
    def distinct_documents(self) -> int:
        return len(self.text_digests)

    @property
    def duplicate_documents(self) -> int:
        return self.total - self.distinct_documents

    @property
    def low_confidence_rate(self) -> float:
        return self.low_confidence / self.total if self.total else 0.0

    def clinical_support(self) -> int:
        return sum(self.support(label) for label in CLINICAL_TYPES)

    def support(self, label: str) -> int:
        return sum(self.confusion[label].values())

    def recall(self, label: str) -> float:
        support = self.support(label)
        return self.confusion[label][label] / support if support else 0.0

    def precision(self, label: str) -> float:
        predicted = sum(
            counts[label] for counts in self.confusion.values()
        )
        return self.confusion[label][label] / predicted if predicted else 0.0

    def clinical_leaks(self) -> list[dict[str, Any]]:
        """Clinical documents predicted as administrative. The dangerous cell.

        This is the confusion-matrix entry behind README I-06's CRITICAL risk:
        an outside lab result predicted as `records_request` lands in an
        administrative queue, and the critical value in it is never read.
        """
        administrative = set(DOCUMENT_TYPES) - CLINICAL_TYPES
        leaks: list[dict[str, Any]] = []
        for expected, counts in self.confusion.items():
            if expected not in CLINICAL_TYPES:
                continue
            for predicted, count in counts.items():
                if predicted in administrative and count:
                    leaks.append(
                        {"expected": expected, "predicted": predicted, "count": count}
                    )
        return sorted(leaks, key=lambda item: -item["count"])

    def blockers(
        self,
        *,
        min_cases: int = 100,
        min_per_class: int = 5,
        floors: Mapping[str, float] = RECALL_FLOORS,
        max_low_confidence_rate: float = MAX_LOW_CONFIDENCE_RATE,
        max_clinical_leak_rate: float = MAX_CLINICAL_LEAK_RATE,
    ) -> list[str]:
        """Every reason this classifier must not route production documents."""
        problems: list[str] = []
        if self.total < min_cases:
            problems.append(
                f"only {self.total} labelled document(s); README I-06 evaluates "
                f"against 300 historical faxes and this gate wants at least "
                f"{min_cases}"
            )
        # The floor is on DISTINCT documents. Ten hand-written faxes tiled
        # twelve times is a hundred and twenty rows and ten measurements; the
        # earlier gate counted the rows and approved.
        if self.distinct_documents < min_cases:
            problems.append(
                f"{self.total} case(s) over only {self.distinct_documents} "
                f"distinct document(s) ({self.duplicate_documents} repeat(s)); "
                f"a tiled corpus measures the classifier against "
                f"{self.distinct_documents} documents, not {self.total}"
            )
        thin = sorted(
            label for label in DOCUMENT_TYPES if 0 < self.support(label) < min_per_class
        )
        absent = sorted(label for label in DOCUMENT_TYPES if self.support(label) == 0)
        if absent:
            problems.append(
                f"no labelled examples at all for {absent}; a class the eval set "
                "never contained has not been measured, it has been assumed"
            )
        if thin:
            problems.append(
                f"fewer than {min_per_class} example(s) for {thin}; a recall "
                "figure over two documents is not a measurement"
            )
        # A classifier that is never sure of anything answers nothing and so
        # gets every question it answered right. Recall already reflects this
        # (a below-threshold answer counts as a miss), but say it out loud,
        # because the operator's next move differs: this is not "the model is
        # wrong", it is "the model is useless at this threshold".
        if self.low_confidence_rate > max_low_confidence_rate:
            problems.append(
                f"{self.low_confidence} of {self.total} answers "
                f"({self.low_confidence_rate:.0%}) fell below the "
                f"{self.min_confidence:.2f} confidence threshold, over the "
                f"{max_low_confidence_rate:.0%} ceiling; this classifier routes "
                "almost nothing and sends the work to a person"
            )
        weak = [
            f"{label}={self.recall(label):.2f} (floor {floors[label]})"
            for label in DOCUMENT_TYPES
            if self.support(label) and self.recall(label) < floors.get(label, 0.0)
        ]
        if weak:
            problems.append("per-class recall below floor: " + ", ".join(weak))
        leaks = self.clinical_leaks()
        leaked = sum(item["count"] for item in leaks)
        clinical = self.clinical_support()
        budget = int(clinical * max_clinical_leak_rate)
        if leaked > budget:
            problems.append(
                f"{leaked} clinical document(s) of {clinical} predicted as "
                f"administrative, over the budget of {budget} "
                f"({max_clinical_leak_rate:.0%}): "
                + ", ".join(
                    f"{l['expected']}->{l['predicted']} x{l['count']}"
                    for l in leaks[:4]
                )
            )
        return problems

    def passes(self, **kwargs: Any) -> bool:
        return not self.blockers(**kwargs)

    def summary(self) -> str:
        worst = sorted(
            (
                (self.recall(label), label)
                for label in DOCUMENT_TYPES
                if self.support(label)
            )
        )[:3]
        return (
            f"{self.model_id}@{self.model_version} prompt#{self.prompt_hash}: "
            f"{self.total} case(s) over {self.distinct_documents} distinct "
            f"document(s), accuracy {self.accuracy:.3f}, "
            f"{self.unclassifiable} unclassifiable, {self.low_confidence} below "
            f"confidence. Weakest recall: "
            + ", ".join(f"{label}={rate:.2f}" for rate, label in worst)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "distinct_documents": self.distinct_documents,
            "accuracy": round(self.accuracy, 4),
            "unclassifiable": self.unclassifiable,
            "low_confidence": self.low_confidence,
            "min_confidence": self.min_confidence,
            "per_class": {
                label: {
                    "support": self.support(label),
                    "recall": round(self.recall(label), 3),
                    "precision": round(self.precision(label), 3),
                }
                for label in DOCUMENT_TYPES
                if self.support(label)
            },
            "clinical_leaks": self.clinical_leaks(),
            "blockers": self.blockers(),
        }


def evaluate(
    classifier: DocumentClassifier, cases: Sequence[EvalCase]
) -> EvalResult:
    """Score a classifier against labelled documents. No shortcuts to the model.

    Scores what the PIPELINE would do, not what the model said. A below-threshold
    answer goes to a person in production, so it is scored as a miss here even
    when the label happens to match -- otherwise a classifier that is never
    confident scores 1.000 and gets approved to route nothing.
    """
    # `getattr` rather than attribute access: an eval harness may pass a stand-in
    # classifier, and refusing to score it teaches nobody anything. The defaults
    # are the strict ones.
    result = EvalResult(
        prompt_hash=getattr(classifier, "prompt_hash", ""),
        min_confidence=getattr(classifier, "min_confidence", MIN_CONFIDENCE),
        provider=getattr(classifier, "provider", ""),
    )
    for case in cases:
        result.total += 1
        result.text_digests[
            hashlib.sha256(case.text.encode("utf-8")).hexdigest()
        ] += 1
        classification = classifier.classify(case.text, document_id=case.document_id)
        if classification is None:
            result.unclassifiable += 1
            # An unclassifiable document is a miss for its true class, not a
            # row to drop -- dropping them inflates recall on exactly the
            # documents the classifier could not read.
            result.confusion[case.expected_type]["__unclassifiable__"] += 1
            continue
        result.model_id = classification.model_id
        result.model_version = classification.model_version
        if not classification.confident:
            result.low_confidence += 1
            result.confusion[case.expected_type]["__low_confidence__"] += 1
            continue
        result.confusion[case.expected_type][classification.document_type] += 1
        if classification.document_type == case.expected_type:
            result.correct += 1
    return result


@dataclass
class ClassifierGate:
    """A measured classifier, or nothing. The counterpart to I-04's ModelPin.

    `require()` is called on the production path, so an unmeasured or degraded
    classifier cannot route documents at all -- it does not route them badly, it
    refuses. README I-06's own plan evaluates against 300 historical faxes; this
    is that evaluation turned into a gate rather than a milestone.

    WHAT IS PINNED, and why each of them. An earlier version compared only the
    prompt hash, which meant a validated prompt in front of a swapped model
    passed the gate -- and the model is the thing that was measured.

      provider/model/version  the thing whose recall was measured
      prompt hash            prompts are code (README 9.4)
      min_confidence         the same model at a lower threshold routes
                             documents the eval never scored it on
    """

    provider: str = ""
    model_id: str = ""
    model_version: str = ""
    prompt_hash: str = ""
    min_confidence: float = MIN_CONFIDENCE
    validated_on: str = ""
    last_summary: str = ""

    def approve(self, result: EvalResult, *, on: str) -> None:
        if not result.passes():
            raise ClassifierNotValidated(
                "refusing to approve a classifier that has not passed its eval "
                f"set: {result.summary()}. Blocking: "
                + "; ".join(result.blockers())
            )
        self.provider = result.provider
        self.model_id = result.model_id
        self.model_version = result.model_version
        self.prompt_hash = result.prompt_hash
        self.min_confidence = result.min_confidence
        self.validated_on = on
        self.last_summary = result.summary()

    def require(self, classifier: DocumentClassifier) -> None:
        if not self.validated_on:
            raise ClassifierNotValidated(
                "no classifier has been approved. Run `evaluate()` against "
                "labelled historical faxes and call `approve()` before routing "
                "anything."
            )
        if classifier.prompt_hash != self.prompt_hash:
            raise ClassifierNotValidated(
                f"prompt {classifier.prompt_hash} is not the validated prompt "
                f"({self.prompt_hash}); prompts are code (README 9.4)"
            )
        approved = (self.provider, self.model_id, self.model_version)
        running = (classifier.provider, classifier.model_id, classifier.model_version)
        if running != approved:
            raise ClassifierNotValidated(
                f"model {running[0]}:{running[1]}@{running[2]} is not the "
                f"validated model ({approved[0]}:{approved[1]}@{approved[2]}); "
                "the eval measured a model, not a prompt"
            )
        if classifier.min_confidence != self.min_confidence:
            raise ClassifierNotValidated(
                f"confidence threshold {classifier.min_confidence} is not the "
                f"validated threshold ({self.min_confidence}); recall was "
                "measured at the validated one"
            )
