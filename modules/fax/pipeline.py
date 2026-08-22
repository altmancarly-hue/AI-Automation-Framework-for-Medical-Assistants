"""The inbound fax pipeline, in the order the steps have to happen.

A fixed sequence, like I-03's batch and for the same reason: there is no
open-ended task space here. OCR, classify, match, triage, route. The model is
called at most twice per document and selects no tools.

ORDER MATTERS IN TWO PLACES.

**Urgency runs on the OCR text, not on the classification.** A document the
classifier could not read is still scanned for critical values, because the
critical value may be on the page that read fine. Making urgency depend on a
successful classification would mean the least readable documents -- the ones
most likely to be a bad fax of a hospital result -- get the least scrutiny.

**Matching runs before routing and after classification.** The classifier is
what extracts the name and DOB off the page; the matcher decides, deterministically,
whether that is enough to name a patient.

Every document produces a `ProcessedDocument` whatever happens. There is no path
that drops one: a fax that fails at every stage is routed to a human with the
reasons attached, which is the correct outcome and the one the practice is
already paying for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence

from .classify import Classification, ClassifierGate, DocumentClassifier
from .match import MatchOutcome, MatchResult, PatientMatcher
from .ocr import OCRResult, OCRUnavailable, chain
from .route import Queue, RoutedDocument, route
from .urgency import UrgencyTriage

__all__ = [
    "ProcessedDocument", "FaxPipeline", "InboundMonitor",
    "parse_document_date", "date_readings", "date_order_is_ambiguous",
]

_DATE_FORMATS = (
    "%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y", "%m/%d/%y", "%d-%b-%Y", "%B %d, %Y",
    "%b %d, %Y", "%d %B %Y",
)

#: Numeric day-first equivalents of the month-first formats above, tried only to
#: DETECT ambiguity -- never to silently replace the month-first reading.
_DAY_FIRST_FORMATS = ("%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y")

#: A purely numeric slash/dash date, which is the only shape whose field order
#: cannot be read off the string. `04-Jun-2024` and `2024-06-04` are unambiguous.
_NUMERIC_DATE = re.compile(r"^\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*$")


def _pivot_two_digit_year(parsed: date, fmt: str) -> date:
    if fmt.endswith("%y") and parsed > date.today():
        return parsed.replace(year=parsed.year - 100)
    return parsed


def parse_document_date(text: str | None) -> date | None:
    """A date off a fax, or None. Never a guess.

    Month-first, because US paediatric faxes are month-first. `date_readings()`
    is the function to call when being wrong by up to eleven months matters --
    it reports whether the string could also be read day-first.

    Two-digit years are the one place this is opinionated: a fax that says
    `03/04/24` for a paediatric date of birth means 2024, not 1924, and
    `strptime` with `%y` pivots at 69. The pivot here is "not in the future",
    which is the only rule that is always true of a date of birth.
    """
    if not text:
        return None
    cleaned = str(text).strip()
    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
        return _pivot_two_digit_year(parsed, fmt)
    return None


def date_readings(text: str | None) -> tuple[date | None, date | None]:
    """`(month_first, day_first_or_None)`.

    `03/04/2024` is 4 March in Springfield and 3 April in Sherbrooke, and the
    fax does not say which. The month-first reading is returned first because
    that is the local convention; the second element is non-None ONLY when the
    string genuinely supports both readings and they are different dates.

    WHY this is not resolved here: the string does not contain the answer. The
    caller has something the string does not -- a patient panel, a chart -- and
    resolves it against that, or hands the document to a person. Picking one
    silently is how a nine-month age error reaches a reference range.
    """
    month_first = parse_document_date(text)
    cleaned = str(text).strip() if text else ""
    day_first: date | None = None
    if _NUMERIC_DATE.match(cleaned):
        for fmt in _DAY_FIRST_FORMATS:
            try:
                day_first = _pivot_two_digit_year(
                    datetime.strptime(cleaned, fmt).date(), fmt
                )
            except ValueError:
                continue
            break
    if month_first is None:
        # `13/04/2024` has no month-first reading at all, so its order is not
        # in doubt: 13 is a day. Reporting None here would send a perfectly
        # readable date from a day-first facility to manual indexing.
        return day_first, None
    if day_first is None or day_first == month_first:
        return month_first, None
    return month_first, day_first


def date_order_is_ambiguous(text: str | None) -> bool:
    return date_readings(text)[1] is not None


@dataclass
class ProcessedDocument:
    document_id: str
    ocr: OCRResult | None = None
    classification: Classification | None = None
    match: MatchResult | None = None
    urgency: Any = None
    routed: RoutedDocument | None = None
    errors: list[str] = field(default_factory=list)
    #: Things a person should know that did not stop processing. Distinct from
    #: `errors`, which are failures.
    warnings: list[str] = field(default_factory=list)

    @property
    def queue(self) -> str:
        return self.routed.queue if self.routed else Queue.HUMAN_INDEXING

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "ocr": self.ocr.as_dict() if self.ocr else None,
            "classification": self.classification.as_dict() if self.classification else None,
            "match": self.match.as_dict() if self.match else None,
            "urgency": self.urgency.as_dict() if self.urgency else None,
            "routed": self.routed.as_dict() if self.routed else None,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


class FaxPipeline:
    """OCR -> classify -> match -> triage -> route. One document at a time."""

    def _match_ambiguous_dob(
        self,
        processed: "ProcessedDocument",
        *,
        name: str,
        dob: date,
        dob_alt: date,
        raw: str,
    ) -> MatchResult:
        """Resolve a `03/04/2024` date of birth against the panel, or refuse.

        The string does not say which reading is right. The PANEL often does:
        run both, and if exactly one lands on a patient, the chart has answered
        a question the fax could not. If both land -- on the same patient or on
        two different ones -- or neither does, nothing here is entitled to pick,
        so the document goes to a person with both readings named.

        WHY NOT just take the month-first reading: a date of birth is the
        strongest identity key this module has, and matching on the wrong one
        does not fail loudly. It files a specialist's letter in a stranger's
        chart.
        """
        primary = self.matcher.match(name=name, dob=dob)
        secondary = self.matcher.match(name=name, dob=dob_alt)
        note = (
            f"date of birth {raw!r} reads as {dob.isoformat()} (month-first) or "
            f"{dob_alt.isoformat()} (day-first)"
        )

        def contested(result: MatchResult) -> bool:
            """A reading that found patients and could not choose between them.

            NOT the same as finding nothing, though `matched is None` is true of
            both. A reading that lands on flagged twins returns AMBIGUOUS with
            `matched=None`; treating that as "no patient has this date of birth"
            let the OTHER reading look unique, and the document auto-filed into a
            cousin's chart with no human review event at all -- overriding an
            outcome `match.py` says goes to a person ALWAYS.
            """
            return result.matched is None and bool(result.candidates)

        if contested(primary) or contested(secondary):
            processed.warnings.append(
                f"{note}; at least one reading lands on patients that need a "
                "person to choose between"
            )
            return MatchResult(
                outcome=MatchOutcome.AMBIGUOUS,
                document_name=name,
                document_dob=None,
                candidates=primary.candidates + secondary.candidates,
                reason=(
                    f"{note}. One of the two readings could not be resolved to a "
                    "single patient, so neither reading settles this document. "
                    "Read the date order off the sending facility's letterhead."
                ),
            )

        hits = [r for r in (primary, secondary) if r.matched is not None]
        if len(hits) == 1:
            resolved = hits[0]
            processed.warnings.append(
                f"{note}; only {resolved.document_dob.isoformat()} matches a "
                f"panel patient, so the panel settled it"
            )
            return resolved
        if not hits:
            # Neither reading is a patient here. The month-first result already
            # says NO_CANDIDATE or NO_DOB; no need to invent an outcome.
            processed.warnings.append(f"{note}; neither reading matches a patient")
            return primary
        if primary.matched is not None and secondary.matched is not None and (
            primary.matched.patient_id == secondary.matched.patient_id
        ):
            # Same child either way -- e.g. a name strong enough on its own.
            # The ambiguity is real but it changed nothing.
            processed.warnings.append(f"{note}; both readings match the same patient")
            return primary
        processed.warnings.append(f"{note}; the two readings match different patients")
        return MatchResult(
            outcome=MatchOutcome.AMBIGUOUS,
            document_name=name,
            document_dob=None,
            candidates=primary.candidates + secondary.candidates,
            reason=(
                f"{note}, and the two readings identify different patients "
                f"({primary.matched.patient_id} vs {secondary.matched.patient_id}). "
                "Read the date order off the sending facility's letterhead."
            ),
        )

    def __init__(
        self,
        *,
        engines: Sequence[Any],
        classifier: DocumentClassifier,
        matcher: PatientMatcher,
        triage: UrgencyTriage,
        gate: ClassifierGate | None = None,
        audit: Any = None,
        on_duty_physician: str = "on_duty",
        indexer: str = "front_desk",
    ) -> None:
        self.engines = list(engines)
        self.classifier = classifier
        self.matcher = matcher
        self.triage = triage
        self.gate = gate
        self.audit = audit
        self.on_duty_physician = on_duty_physician
        self.indexer = indexer

    def process(
        self,
        path: str,
        *,
        document_id: str,
        age_lookup: Mapping[str, float] | None = None,
        now: datetime | None = None,
    ) -> ProcessedDocument:
        """Never raises on one bad document. The batch is a morning's post."""
        # Both gates before anything is read: an unmeasured classifier and an
        # unreviewed range table are configuration errors, not per-document
        # ones, and they should stop the run rather than mis-route a thousand
        # faxes quietly.
        if self.gate is not None:
            self.gate.require(self.classifier)
        self.triage.ranges.require_reviewed()

        processed = ProcessedDocument(document_id=document_id)
        try:
            processed.ocr = chain(self.engines, path)
        except OCRUnavailable as exc:
            processed.errors.append(str(exc))
            processed.ocr = OCRResult(source=path, engine="none")

        text = processed.ocr.text
        try:
            processed.classification = (
                self.classifier.classify(text, document_id=document_id)
                if text.strip()
                else None
            )
        except Exception as exc:  # noqa: BLE001 - a transport blip is not a drop
            # `SchemaViolation` was caught inside the classifier; everything
            # else -- a connection reset, a model server restart -- escaped and
            # produced NO ProcessedDocument at all, dropping the document. The
            # docstring promised the opposite.
            processed.errors.append(f"classifier: {type(exc).__name__}: {exc}")
            processed.classification = None

        name = (processed.classification.patient_name if processed.classification else None) or ""
        raw_dob = processed.classification.patient_dob if processed.classification else None
        dob, dob_alt = date_readings(raw_dob)
        if dob_alt is not None:
            processed.match = self._match_ambiguous_dob(
                processed, name=name, dob=dob, dob_alt=dob_alt, raw=str(raw_dob)
            )
        else:
            processed.match = self.matcher.match(name=name, dob=dob)

        age_months: float | None = None
        if processed.match.matched is not None and age_lookup is not None:
            age_months = age_lookup.get(processed.match.matched.patient_id)
        stated_collection = self.triage.ranges.collection_date_text(text)
        collected, collected_alt = self.triage.ranges.collection_date_readings(text)
        if collected is not None and collected > date.today():
            # A mis-OCR'd year. Scoring a newborn against the adolescent band is
            # the failure the collection date exists to prevent, so an
            # impossible one is discarded rather than trusted.
            collected = None
        if collected_alt is not None and collected_alt > date.today():
            collected_alt = None

        candidate_ages: list[float] = []
        if (
            stated_collection
            and collected is None
            and collected_alt is None
            and age_months is not None
        ):
            # The document states a collection date and not one reading of it is
            # possible. Keeping the age on ARRIVAL here would be inventing a
            # collection date the document contradicts -- the exact substitution
            # hard constraint 3 forbids -- so the age is withdrawn and every
            # band applies.
            processed.warnings.append(
                f"the stated collection date {stated_collection!r} is not a "
                "usable date; no age at collection is claimed and results are "
                "scored against every paediatric band"
            )
            age_months = None
        elif (
            processed.match.matched is not None
            and age_months is not None
            and (collected is not None or collected_alt is not None)
        ):
            # Age at COLLECTION, not age on arrival. A newborn bilirubin faxed
            # six weeks late is still a newborn bilirubin, and scoring it
            # against the two-month band force-flags a normal result.
            dob_on_file = processed.match.matched.dob
            # A collection date before the child was born is a misreading, not a
            # fact, whichever reading produced it.
            ages = sorted(
                ((day - dob_on_file).days / 30.4375, day)
                for day in (collected, collected_alt)
                if day is not None and day >= dob_on_file
            )
            if len(ages) > 1:
                # `03/04/2024` is 4 March or 3 April and nothing in the document
                # says which. SCORE AT BOTH.
                #
                # An earlier version picked the younger age, on the theory that
                # paediatric bands are narrowest at the youngest ages. They are
                # not: the newborn bilirubin ceiling is 15.0 and the ceiling
                # after one month is 1.2, so the younger reading cleared a
                # bilirubin of 9.4 that the older reading pages a physician
                # about. Choosing either age is choosing an answer to a question
                # the document did not settle.
                #
                # `scan(candidate_ages=...)` therefore scores against both: a
                # result outside both bands is abnormal whichever reading is
                # right, and a result outside only one of them is reported as
                # unsettled -- which routes to a person instead of to silence or
                # to a false page.
                candidate_ages = [age for age, _day in ages]
                processed.warnings.append(
                    f"collection date reads as {ages[0][1].isoformat()} or "
                    f"{ages[1][1].isoformat()}; scored against BOTH ages, and a "
                    "finding that holds at only one of them is reported as "
                    "unsettled rather than resolved"
                )
            elif len(ages) == 1 and (collected is not None and collected_alt is not None):
                # One reading was before the child was born, so the document
                # settled itself -- but say so, because it means the date order
                # on this sender's paper is now known.
                processed.warnings.append(
                    f"collection date reads as {ages[0][1].isoformat()}; the "
                    "other reading precedes the child's date of birth"
                )
            if ages:
                age_months = ages[0][0]
            else:
                # Both readings are impossible. Scoring against the age on
                # arrival would be inventing a collection date, so the age is
                # withdrawn and `scan` falls back to every band.
                processed.warnings.append(
                    "no usable collection date: every reading is in the future "
                    "or precedes the child's date of birth; results are scored "
                    "against every paediatric band"
                )
                age_months = None

        try:
            processed.urgency = self.triage.triage(
                text,
                document_id=document_id,
                age_months=age_months,
                candidate_ages=candidate_ages or None,
                document_type=(
                    processed.classification.document_type
                    if processed.classification
                    else ""
                ),
                unreadable_pages=[p.number for p in processed.ocr.unreadable_pages()],
            )
        except Exception as exc:  # noqa: BLE001
            # Same reasoning, and the default is urgent: a model that errored is
            # not evidence that a document is routine.
            from .urgency import UrgencyResult

            processed.errors.append(f"urgency: {type(exc).__name__}: {exc}")
            processed.urgency = UrgencyResult(
                urgency="urgent", confidence=0.0,
                reason=(
                    "urgency triage could not run and the document was escalated "
                    f"rather than assumed routine ({type(exc).__name__})"
                ),
                warnings=[str(exc)],
            )

        processed.routed = route(
            document_id=document_id,
            ocr=processed.ocr,
            classification=processed.classification,
            match=processed.match,
            urgency=processed.urgency,
            on_duty_physician=self.on_duty_physician,
            indexer=self.indexer,
        )
        if self.audit is not None:
            # README I-06: "Every classification, match, and routing decision
            # logged with model version and confidence."
            self.audit.record_event(
                actor_id="system:fax",
                initiative_id="I-06",
                event_type="fax_routed",
                detail={
                    "document_id": document_id,
                    "queue": processed.routed.queue,
                    "document_type": processed.routed.document_type,
                    "urgency": processed.routed.urgency,
                    "escalated_by_rule": bool(
                        processed.urgency and processed.urgency.escalated
                    ),
                    "match_outcome": processed.match.outcome,
                    "ocr_confidence": round(processed.ocr.confidence, 3),
                    "classifier_confidence": (
                        round(processed.classification.confidence, 3)
                        if processed.classification
                        else None
                    ),
                },
            )
        return processed


@dataclass
class InboundMonitor:
    """Zero-received-documents alerting. README I-06's gateway-outage control.

    "Monitor and alert on zero-received-documents in a 4-hour window during
    business hours." A fax gateway that stops delivering produces no error
    anywhere -- it produces silence, and silence looks exactly like a quiet
    morning. This is the only failure in the initiative that a pipeline cannot
    detect from the inside, because there is nothing to process.
    """

    window_hours: float = 4.0
    business_start_hour: int = 8
    business_end_hour: int = 17
    business_days: frozenset[int] = frozenset({0, 1, 2, 3, 4})

    def is_business_time(self, moment: datetime) -> bool:
        return (
            moment.weekday() in self.business_days
            and self.business_start_hour <= moment.hour < self.business_end_hour
        )

    def check(
        self, *, last_received: datetime | None, now: datetime
    ) -> dict[str, Any]:
        if not self.is_business_time(now):
            return {"alert": False, "reason": "outside business hours"}
        if last_received is None:
            return {
                "alert": True,
                "reason": (
                    "no fax has ever been recorded as received. Either the "
                    "gateway is not delivering or nothing has been wired up."
                ),
            }
        if (now.tzinfo is None) != (last_received.tzinfo is None):
            # Refused rather than coerced. Guessing which one is UTC is how a
            # monitor reports "no fax for six hours" at 09:00 on a normal
            # morning, and an outage alert that cries wolf is switched off.
            raise ValueError(
                "last_received and now must both be timezone-aware or both "
                f"naive (got {last_received!r} and {now!r})"
            )
        idle = (now - last_received).total_seconds() / 3600.0
        if idle < self.window_hours:
            return {
                "alert": False,
                "reason": f"last document {idle:.1f}h ago",
                "idle_hours": round(idle, 2),
            }
        return {
            "alert": True,
            "idle_hours": round(idle, 2),
            "reason": (
                f"no inbound document for {idle:.1f} hours during business hours. "
                "A fax gateway failure is silent -- it looks like a quiet "
                "morning. Check the gateway and the backup analog line."
            ),
        }
