"""Where a document goes, and what task goes with it.

README I-06's routing rules, and one line from its risk table that changes the
shape of the whole module:

    Auto-filed document nobody reads | Medium | AUTO-FILING DOES NOT MEAN NO
    TASK. Every clinically relevant type generates a review task.

So `route()` returns a queue AND a task, and for clinical types the task is not
optional. A document that lands in a chart with no task attached is a document
in a chart nobody opens, which is one of the failure modes this initiative
exists to remove -- it is not an improvement on a stack of paper, it is the same
stack somewhere less visible.

THE URGENT PATH HAS TWO DESTINATIONS ON PURPOSE. README I-06:

    Urgent -> immediate alert to the on-duty physician AND a parallel human
    verification, because an urgency misclassification cannot be allowed to fail
    silently.

An urgent document therefore produces two tasks, not one, and the verification
task exists whether or not the alert was delivered. An alert nobody acknowledged
and a queue nobody checked are the same outcome.

THE IMMUNIZATION HANDOFF is the cross-initiative link README I-06 calls "the
highest-value cross-initiative link in the plan" -- outside immunization records
arriving by fax are the largest single source of the chart-versus-registry
discrepancies that I-02 reconciles. A detected immunization record is handed to
`modules.immunization.matcher` as structured dose records, and -- importantly --
it is handed over as a RECONCILIATION INPUT, never as a chart write. I-02's
existing controls (ambiguous pairs to a human, no unresolved dose applied
without a named reviewer) then apply unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence

from .classify import CLINICAL_TYPES, Classification
from .match import MatchOutcome, MatchResult
from .ocr import OCRResult
from .urgency import UrgencyResult

__all__ = [
    "GENERIC_CVX",
    "Queue",
    "TaskKind",
    "RoutedDocument",
    "ReviewTask",
    "route",
    "extract_immunization_doses",
    "ImmunizationHandoff",
]


class Queue:
    AUTO_FILE = "auto_file"
    PHYSICIAN_REVIEW = "physician_review"
    URGENT_ALERT = "urgent_alert"
    HUMAN_INDEXING = "human_indexing"
    IMMUNIZATION_RECONCILIATION = "immunization_reconciliation"


class TaskKind:
    ACKNOWLEDGE = "acknowledge"
    PHYSICIAN_REVIEW = "physician_review"
    URGENT_CONTACT = "urgent_contact"
    VERIFY_URGENCY = "verify_urgency"
    INDEX_BY_HAND = "index_by_hand"
    RECONCILE_IMMUNIZATIONS = "reconcile_immunizations"


@dataclass(frozen=True)
class ReviewTask:
    kind: str
    assigned_to: str
    description: str
    due_within_hours: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "assigned_to": self.assigned_to,
            "description": self.description,
            "due_within_hours": self.due_within_hours,
        }


@dataclass
class RoutedDocument:
    document_id: str
    queue: str
    tasks: list[ReviewTask] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    patient_id: str | None = None
    document_type: str = "unknown"
    urgency: str = "needs_physician_review"
    immunization_handoff: "ImmunizationHandoff | None" = None

    @property
    def auto_filed(self) -> bool:
        return self.queue == Queue.AUTO_FILE

    @property
    def needs_human(self) -> bool:
        return self.queue in {
            Queue.HUMAN_INDEXING, Queue.PHYSICIAN_REVIEW, Queue.URGENT_ALERT
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "queue": self.queue,
            "patient_id": self.patient_id,
            "document_type": self.document_type,
            "urgency": self.urgency,
            "tasks": [t.as_dict() for t in self.tasks],
            "reasons": list(self.reasons),
            "immunization_doses": (
                len(self.immunization_handoff.doses) if self.immunization_handoff else 0
            ),
        }


def route(
    *,
    document_id: str,
    ocr: OCRResult,
    classification: Classification | None,
    match: MatchResult,
    urgency: UrgencyResult,
    on_duty_physician: str = "on_duty",
    indexer: str = "front_desk",
) -> RoutedDocument:
    """Decide the queue and the tasks. Deterministic; no model runs here.

    The order of the checks is the priority order, and it starts with the two
    things that make every later question unanswerable: a document nobody could
    read, and a document nobody could attribute to a patient.
    """
    routed = RoutedDocument(document_id=document_id, queue=Queue.HUMAN_INDEXING)
    routed.urgency = urgency.urgency
    if classification is not None:
        routed.document_type = classification.document_type
    # Assigned BEFORE any early return. An urgent alert on a document whose
    # patient was successfully matched used to go out with patient_id=None,
    # because step 1 returned before this line ran.
    if match.matched is not None:
        routed.patient_id = match.matched.patient_id

    # 1. Unreadable. Nothing downstream means anything.
    if ocr.needs_human():
        routed.reasons.append(f"OCR: {ocr.describe_quality()}")
        routed.tasks.append(
            ReviewTask(
                TaskKind.INDEX_BY_HAND, indexer,
                f"Read and index by hand: {ocr.describe_quality()}",
                due_within_hours=8.0,
            )
        )
        # An unreadable document can still be urgent -- the readable half may
        # carry a critical value -- so the urgency alert is raised alongside,
        # AND the queue moves with it. Raising the tasks without setting the
        # queue meant nothing selecting on `queue == "urgent_alert"` -- a
        # worklist, a pager integration, the audit field -- ever saw it.
        if urgency.is_urgent:
            routed.queue = Queue.URGENT_ALERT
            _add_urgent_tasks(routed, urgency, on_duty_physician, indexer)
        return routed

    # 2. Unclassifiable, or classified with too little confidence to act on.
    if classification is None:
        routed.reasons.append(
            "the classifier could not produce a valid answer for this document"
        )
        routed.tasks.append(
            ReviewTask(
                TaskKind.INDEX_BY_HAND, indexer,
                "Classify and index by hand: the classifier returned nothing usable",
                due_within_hours=8.0,
            )
        )
        return routed
    if not classification.confident:
        routed.reasons.append(
            f"classifier confidence {classification.confidence:.2f} is below the "
            "threshold; README I-06 says to honour the confidence field"
        )
        routed.tasks.append(
            ReviewTask(
                TaskKind.INDEX_BY_HAND, indexer,
                f"Confirm document type (model guessed {classification.document_type} "
                f"at {classification.confidence:.2f})",
                due_within_hours=8.0,
            )
        )
        if urgency.is_urgent:
            routed.queue = Queue.URGENT_ALERT
            _add_urgent_tasks(routed, urgency, on_duty_physician, indexer)
        return routed

    # 3. Urgent. Alert AND verify, in parallel, whatever else is true.
    if urgency.is_urgent:
        routed.queue = Queue.URGENT_ALERT
        routed.reasons.append(urgency.reason)
        # An immunization record is still an immunization record when it is
        # urgent. Returning early here skipped the I-02 handoff entirely -- and
        # a middle initial on the header was enough to trigger it -- so "the
        # highest-value cross-initiative link in the plan" was silently lost.
        if (
            classification.document_type == "immunization_record"
            and match.matched is not None
        ):
            _attach_immunization_handoff(routed, ocr, classification, match, document_id)
        _add_urgent_tasks(routed, urgency, on_duty_physician, indexer)
        if match.needs_human:
            routed.reasons.append(f"patient match: {match.reason}")
            routed.tasks.append(
                ReviewTask(
                    TaskKind.INDEX_BY_HAND, indexer,
                    f"URGENT document, patient not identified: {match.reason}",
                    due_within_hours=1.0,
                )
            )
        return routed

    # 4. Patient not identified. Everything below here needs a chart.
    if match.needs_human:
        routed.reasons.append(f"patient match: {match.reason}")
        hours = 4.0 if classification.is_clinical else 8.0
        routed.tasks.append(
            ReviewTask(
                TaskKind.INDEX_BY_HAND, indexer,
                f"Identify the patient for a {classification.document_type}: "
                f"{match.reason}",
                due_within_hours=hours,
            )
        )
        return routed

    assert match.matched is not None
    routed.patient_id = match.matched.patient_id

    # 5. Immunization records go to I-02, not to a chart.
    if classification.document_type == "immunization_record":
        routed.queue = Queue.IMMUNIZATION_RECONCILIATION
        _attach_immunization_handoff(routed, ocr, classification, match, document_id)
        return routed

    # 6. `other` is the escape hatch, and the escape hatch goes to a person.
    #
    # `classify.py`'s docstring has said so from the start; the routing did not.
    # `other` is not in CLINICAL_TYPES, so a confident, matched, routine `other`
    # fell through to step 7 and was filed into a chart with no task and no
    # human review event. `other` is by definition the class the classifier
    # assigns to the document it understands LEAST -- the worst possible
    # candidate for silent filing, and hard constraint 4 forbids it outright.
    if classification.document_type == "other":
        routed.queue = Queue.HUMAN_INDEXING
        routed.reasons.append(
            "the classifier could not place this document in any type; `other` "
            "is the escape hatch and it goes to a person"
        )
        routed.tasks.append(
            ReviewTask(
                TaskKind.INDEX_BY_HAND, indexer,
                f"Classify and index by hand: {classification.one_line_summary}",
                due_within_hours=8.0,
            )
        )
        return routed

    # 7. Physician review.
    if urgency.urgency == "needs_physician_review":
        routed.queue = Queue.PHYSICIAN_REVIEW
        routed.reasons.append(urgency.reason or "clinical document requiring review")
        routed.tasks.append(
            ReviewTask(
                TaskKind.PHYSICIAN_REVIEW, on_duty_physician,
                f"{classification.document_type}: {classification.one_line_summary}",
                due_within_hours=24.0,
            )
        )
        return routed

    # 8. Routine, confidently classified, confidently matched: auto-file.
    routed.queue = Queue.AUTO_FILE
    routed.reasons.append(
        f"routine {classification.document_type}, patient matched on "
        f"{match.reason}"
    )
    if classification.is_clinical:
        # README I-06: auto-filing does not mean no task.
        routed.tasks.append(
            ReviewTask(
                TaskKind.ACKNOWLEDGE, on_duty_physician,
                f"Acknowledge filed {classification.document_type}: "
                f"{classification.one_line_summary}",
                due_within_hours=72.0,
            )
        )
    return routed


def _urgent_detail(urgency: UrgencyResult) -> str:
    """Everything that raised the level, numeric findings first.

    `escalate()` overwrites `reason` on the first rule that fires, so an
    order-priority phrase appearing before the numbers meant the one-hour page
    said "the document contains critical-value language" and NEVER MENTIONED THE
    POTASSIUM OF 7.4. The physician being paged has to be told what to look at.
    """
    numeric = [
        a.describe() for a in urgency.abnormal
        if a.source in ("reference_range", "document_flag")
    ]
    phrases = [a.describe() for a in urgency.abnormal if a.source == "critical_phrase"]
    # Last, and only when something else already made this urgent: an
    # age-dependent finding is a reason to look, not a reason to page, and it
    # must not push a settled finding off the end of the list.
    unsettled = [a.describe() for a in urgency.abnormal if a.source == "age_unknown"]
    parts = numeric + phrases + unsettled
    if not parts:
        return urgency.reason or "flagged urgent"
    return "; ".join(parts[:4]) + (
        f" (model said {urgency.model_urgency or 'nothing'})"
    )


def _add_urgent_tasks(
    routed: RoutedDocument, urgency: UrgencyResult, physician: str, indexer: str
) -> None:
    """Two tasks, always. The alert and the parallel verification.

    Separate people on purpose. A verification assigned to the same person the
    alert went to is not a verification; it is the same judgement twice.
    """
    detail = _urgent_detail(urgency)
    routed.tasks.append(
        ReviewTask(
            TaskKind.URGENT_CONTACT, physician,
            f"URGENT: {detail}", due_within_hours=1.0,
        )
    )
    routed.tasks.append(
        ReviewTask(
            TaskKind.VERIFY_URGENCY, indexer,
            "Parallel verification of an urgent classification (README I-06: an "
            "urgency misclassification cannot be allowed to fail silently): "
            + detail,
            due_within_hours=2.0,
        )
    )


def _attach_immunization_handoff(
    routed: RoutedDocument,
    ocr: OCRResult,
    classification: Classification,
    match: MatchResult,
    document_id: str,
) -> None:
    assert match.matched is not None
    doses, unresolved = extract_immunization_doses(ocr.text, with_unresolved=True)
    routed.immunization_handoff = ImmunizationHandoff(
        patient_id=match.matched.patient_id,
        document_id=document_id,
        source=classification.sending_facility or "outside fax",
        doses=doses,
        unresolved=unresolved,
    )
    routed.reasons.append(
        "immunization record detected; handed to the I-02 reconciliation "
        "pipeline as an input, not written to the chart"
    )
    detail = f"Reconcile {len(doses)} dose(s) from {routed.immunization_handoff.source}"
    if unresolved:
        detail += (
            f"; {len(unresolved)} line(s) name a vaccine this system could not "
            "resolve and must be read by hand"
        )
    routed.tasks.append(
        ReviewTask(TaskKind.RECONCILE_IMMUNIZATIONS, "ma_queue", detail, 48.0)
    )


# -- the I-02 handoff --------------------------------------------------------


#: Generic names as a registry printout spells them, mapped to CVX. I-02's
#: `cvx.TRADE_NAMES` covers brands (Pediarix, Prevnar) and not the generic
#: abbreviations a state registry prints, and a dose whose code I-02 cannot
#: resolve is skipped by its first reconciliation pass -- so an unmapped name
#: here is a dose that silently never reaches the forecaster.
#:
#: "Unspecified formulation" codes are used deliberately: a printout saying
#: "DTaP" does not say which product, and asserting one would invent a fact.
GENERIC_CVX: Mapping[str, str] = {
    "dtap": "107", "tdap": "115", "td": "113",
    "ipv": "89", "opv": "02", "polio": "89",
    "mmr": "03", "mmrv": "94",
    "varicella": "21", "var": "21", "chickenpox": "21",
    "hepb": "45", "hep b": "45", "hepatitis b": "45",
    "hepa": "85", "hep a": "85", "hepatitis a": "85",
    "hib": "17",
    "pcv": "152", "pcv13": "133", "pcv15": "152", "pcv20": "152",
    "ppsv": "33", "ppsv23": "33",
    "rotavirus": "122", "rv": "122", "rv1": "119", "rv5": "116",
    "hpv": "137", "hpv9": "165", "hpv2": "137", "hpv4": "62",
    "menacwy": "114", "menb": "162",
    "influenza": "158", "flu": "158",
    "covid19": "171", "covid-19": "171", "covid": "171",
}

#: Words that mean the line is NOT an administration. A registry printout lists
#: due, refused, contraindicated, invalid and historical-only rows in the same
#: table as the doses given, and an extractor that reads only "a vaccine name
#: and a date" turned every one of them into an administered dose -- which
#: manufactures the exact failure I-02's matcher docstring names, "a real gap
#: masked by a phantom record", and reports an unvaccinated child as covered.
_NOT_ADMINISTERED = re.compile(
    r"(?i)\b(?:not\s+administered|not\s+given|refused|declin\w+|"
    r"contraindicat\w+|deferred|due\b|overdue|invalid|"
    r"exempt\w*|scheduled|recommend\w+|pending|history\s+only|"
    r"immune|titer|serolog\w+|repeat\s+required|error)\b"
)


@dataclass
class ImmunizationHandoff:
    """Structured doses lifted off an immunization printout, for I-02.

    A RECONCILIATION INPUT, never a chart write. I-02's matcher decides what is
    the same event as what, its adjudicator handles the ambiguous pairs, and its
    existing rule -- nothing unresolved is applied without a named reviewer --
    applies unchanged. Nothing here writes anything.

    `as_dose_records()` builds real `matcher.DoseRecord` objects rather than
    loose dictionaries. An earlier version emitted `{vaccine, administered_on,
    ...}` and a test asserted only its own key names; `DoseRecord(**record)`
    raised `TypeError`, `administered_on` was a string where `reconcile()` does
    date arithmetic, and every generic vaccine name failed `cvx.is_known`. The
    "highest-value cross-initiative link in the plan" did not connect.
    """

    patient_id: str
    document_id: str
    source: str
    doses: list[dict[str, Any]] = field(default_factory=list)
    #: Lines naming a vaccine that could not be turned into a dose. Reported so
    #: the MA reads them, rather than dropped so the task says "0 dose(s)".
    unresolved: list[dict[str, Any]] = field(default_factory=list)

    def as_dose_records(self) -> list[Any]:
        """Real `DoseRecord`s, ready for `modules.immunization.matcher.reconcile`."""
        from modules.immunization.forecast import DosePrecision
        from modules.immunization.matcher import DoseRecord

        records = []
        for index, dose in enumerate(self.doses):
            records.append(
                DoseRecord(
                    record_id=f"{self.document_id}#{index}",
                    cvx=dose["cvx"],
                    given=dose["given"],
                    # I-02 knows two sources. An outside fax is a registry-shaped
                    # claim about what happened elsewhere, not the chart.
                    source="registry",
                    precision=(
                        DosePrecision.MONTH
                        if dose["precision"] == "month"
                        else DosePrecision.DAY
                    ),
                    product_text=dose["product_text"],
                )
            )
        return records

    def as_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "document_id": self.document_id,
            "source": self.source,
            "doses": [
                {
                    "cvx": d["cvx"],
                    "given": d["given"].isoformat(),
                    "precision": d["precision"],
                    "product_text": d["product_text"],
                }
                for d in self.doses
            ],
            "unresolved": list(self.unresolved),
        }


_DATE_PATTERNS = (
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"), ("m", "d", "y")),
    (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"), ("y", "m", "d")),
    # Two-digit years, which a registry prints as often as four. Without this
    # every line of such a printout was silently dropped and the MA's task read
    # "Reconcile 0 dose(s)" -- which reads as nothing to do rather than as an
    # extraction that failed.
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2})\b"), ("m", "d", "yy")),
    (re.compile(r"\b(\d{1,2})/(\d{4})\b"), ("m", "y")),
)

#: Vaccine names as they appear on a registry printout.
_VACCINE = re.compile(
    r"(?i)\b(DTaP|Tdap|Td|IPV|OPV|MMRV|MMR|Varicella|VAR|Hep\s?B|HepB|Hep\s?A|HepA|"
    r"Hib|PCV\d*|PPSV\d*|Rotavirus|RV\d?|HPV\d?|MenACWY|MenB|Influenza|Flu|"
    r"COVID-?19|Pediarix|Pentacel|ProQuad|Kinrix|Quadracel|Vaxelis|Comvax|"
    r"Twinrix|RotaTeq|Rotarix|Prevnar\s?\d*|Gardasil\s?9?|Menactra|Menveo|"
    r"Bexsero|Infanrix|Daptacel|ActHIB|PedvaxHIB|Hiberix|Engerix|Recombivax|"
    r"Havrix|IPOL)\b"
)


def _resolve_cvx(name: str) -> str | None:
    """A CVX code for a printed vaccine name, or None. Never a guess."""
    from modules.immunization import cvx as cvxlib

    cleaned = re.sub(r"\s+", " ", name.strip().lower())
    code = cvxlib.normalise_code(cleaned)
    if code is not None and cvxlib.is_known(code):
        return code
    for key in (cleaned, cleaned.replace(" ", ""), cleaned.replace("-", "")):
        if key in GENERIC_CVX:
            return GENERIC_CVX[key]
    return None


def _parse_dose_date(line: str) -> tuple[date | None, str, str]:
    """(date, precision, raw) from a printout line. Rejects impossible dates."""
    for pattern, order in _DATE_PATTERNS:
        match = pattern.search(line)
        if match is None:
            continue
        parts = dict(zip(order, match.groups()))
        year = int(parts.get("y") or 0)
        if "yy" in parts:
            year = 2000 + int(parts["yy"])
            if date(year, 1, 1) > date.today():
                year -= 100
        month = int(parts["m"])
        day = int(parts.get("d") or 1)
        try:
            # A real date object, so 13/45/2024 and 02/30/2025 cannot reach I-02
            # as "2024-13-45". They were emitted verbatim as reconciliation
            # input by a formatter that never constructed a date at all.
            parsed = date(year, month, day)
        except ValueError:
            return None, "", match.group(0)
        if parsed > date.today():
            return None, "", match.group(0)
        return parsed, ("month" if "d" not in parts else "day"), match.group(0)
    return None, "", ""


def extract_immunization_doses(
    text: str, *, with_unresolved: bool = False
) -> Any:
    """Vaccine plus date, per line. Deterministic and deliberately literal.

    Line-oriented because registry printouts are tables. No model, no inference,
    and no completion of a partial date: a record reading "03/2019" is handed to
    I-02 with `MONTH` precision, which I-02 already knows how to hold for review
    rather than resolve.

    Anything that names a vaccine and is NOT unambiguously an administration --
    a due date, a refusal, a contraindication, an invalid dose, a line whose date
    is impossible, a name with no CVX code -- goes to `unresolved` for a person
    to read. It never becomes a dose.
    """
    doses: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        matches = list(_VACCINE.finditer(line))
        if not matches:
            continue
        if _NOT_ADMINISTERED.search(line):
            unresolved.append(
                {"line": line[:160], "reason": "line does not read as an administration"}
            )
            continue
        # Every vaccine/date pair on the line, not just the first: OCR merges
        # adjacent table rows onto one line often enough that taking the first
        # pair silently halved a printout.
        segments = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
            segments.append((match.group(1), line[match.start() : end]))
        for name, segment in segments:
            given, precision, raw = _parse_dose_date(segment)
            code = _resolve_cvx(name)
            if code is None:
                unresolved.append(
                    {"line": segment.strip()[:160], "reason": f"no CVX code for {name!r}"}
                )
                continue
            if given is None:
                unresolved.append(
                    {
                        "line": segment.strip()[:160],
                        "reason": (
                            f"date {raw!r} is not a valid past date"
                            if raw
                            else "no administration date on this line"
                        ),
                    }
                )
                continue
            doses.append(
                {
                    "cvx": code,
                    "given": given,
                    "precision": precision,
                    "product_text": name,
                    "line": segment.strip()[:120],
                }
            )
    return (doses, unresolved) if with_unresolved else doses
