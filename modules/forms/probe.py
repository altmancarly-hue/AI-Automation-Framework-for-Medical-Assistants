"""The deliberate wrong answer, and whether anybody noticed.

The build plan calls this out by name:

    probe.py  Synthetic error injection. Insert a deliberate wrong value into
              1 in 50 forms, record whether the reviewer caught it, write to
              AuditLog with synthetic_probe=True. This is the automation-
              complacency control from README §I-01 risks and it is THE SINGLE
              MOST-OMITTED CONTROL IN REAL DEPLOYMENTS.

The reason it gets omitted is that it feels perverse: the system deliberately
does the thing the system exists to prevent. But the alternative is worse and it
is the default. A forms pipeline that is right 99% of the time trains the
reviewer, over about three weeks, to click approve. After that the review step
still exists on the screen and no longer exists in fact, and NOTHING IN THE
DATA SAYS SO -- the error rate looks the same, because the errors the reviewer
stopped catching are the same errors nobody was making. The first time it
matters is a real one.

So the only way to measure whether the review is real is to give it something to
find, at a rate low enough not to be an insult and high enough to be measurable.
One in fifty is the build plan's number: about two per week at NSP's volume.

FOUR RULES, EACH OF WHICH IS A WAY THIS CONTROL GOES WRONG IN PRACTICE.

  1. **A probe never reaches a signature.** `ReleaseGate` treats an outstanding
     probe as a blocker, and `resolve()` is what clears it. A probe that could
     be signed is not a control, it is a defect generator with paperwork.
  2. **A probe is never clinically dangerous.** `PLAUSIBLE_PERTURBATIONS` only
     touches fields whose wrongness is visible and harmless -- a transposed
     digit in a height, a date a year off. It never touches an allergy, a
     medication, a condition, or an immunization date. A "test" that puts a
     wrong tetanus date on a form and relies on the reviewer to catch it has
     placed a real risk to catch a hypothetical one, and if the reviewer misses
     it the probe IS the failure.
  3. **A probe is visibly a probe in the audit log, and only there.** It is
     written with `synthetic_probe=True` so `AuditLog.probe_catch_rate()`
     excludes it from the real edit-rate statistics -- an injected error that
     counted as a genuine correction would flatter the very number it exists to
     test.
  4. **The rate is deterministic per form, not random.** `should_probe` hashes
     the request id. Two runs over the same batch probe the same forms, which
     makes the whole thing reproducible in a test and stops a retry from
     re-rolling the dice until a form gets probed.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Iterable, Mapping, Sequence

from .fill import FilledForm
from .review import ReviewDecision, ReviewPayload
from .templates import FormTemplate

__all__ = [
    "DEFAULT_PROBE_RATE",
    "SAFE_PROBE_FIELDS",
    "UNSAFE_PROBE_FIELDS",
    "Probe",
    "ProbeProgram",
    "UnsafeProbeTarget",
]

#: One in fifty, from the build plan.
DEFAULT_PROBE_RATE = 50

#: Field-name patterns a probe may perturb. An ALLOWLIST, because the blocklist
#: version of this rule -- "don't probe anything dangerous" -- fails the moment
#: somebody adds a field nobody thought about, and the way it fails is a wrong
#: clinical value on a legal document.
SAFE_PROBE_FIELDS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^height(_\w+)?$"),
    re.compile(r"^weight(_\w+)?$"),
    re.compile(r"^bmi$"),
    re.compile(r"^bmi_percentile$"),
    re.compile(r"^blood_pressure$"),
    re.compile(r"^exam_date$"),
    re.compile(r"^dental_exam_date$"),
    re.compile(r"^child_(first|last)_name$"),
    re.compile(r"^date_of_birth$"),
    re.compile(r"^parent_guardian$"),
)

#: Named explicitly as well, so the intent survives a refactor of the patterns
#: and so a reader can see the list rather than reason about regexes. Anything
#: matching one of these is refused even if a pattern above would allow it.
UNSAFE_PROBE_FIELDS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^imm_"),
    re.compile(r"allerg", re.I),
    re.compile(r"medicat", re.I),
    re.compile(r"condition", re.I),
    re.compile(r"lead", re.I),
    re.compile(r"restriction", re.I),
    re.compile(r"signature", re.I),
    re.compile(r"vision|hearing", re.I),
)


class UnsafeProbeTarget(RuntimeError):
    """Raised when a probe is aimed at a field it must not touch."""


def _is_safe(field_name: str) -> bool:
    if any(p.search(field_name) for p in UNSAFE_PROBE_FIELDS):
        return False
    return any(p.match(field_name) for p in SAFE_PROBE_FIELDS)


# -- the perturbations -------------------------------------------------------
#
# Each returns a wrong-but-plausible string, or None if it cannot perturb this
# value. Plausible matters: a height of 999 is caught by anybody and measures
# nothing. A height of 48.5 rendered as 45.8 is exactly the transposition a
# tired person misses, which is the thing being measured.


def _transpose_digits(text: str) -> str | None:
    digits = [i for i, ch in enumerate(text) if ch.isdigit()]
    for a, b in zip(digits, digits[1:]):
        if text[a] != text[b]:
            chars = list(text)
            chars[a], chars[b] = chars[b], chars[a]
            return "".join(chars)
    return None


def _transpose_letters(text: str) -> str | None:
    """`Alvarez` -> `Alvraez`. The misread a tired person scrolls past.

    Without this, a name field was in the safe allowlist and no perturbation
    could touch it, so choosing a name as the target silently produced NO probe
    at all -- the control quietly not running is the one failure mode a
    complacency control cannot afford.
    """
    letters = [i for i, ch in enumerate(text) if ch.isalpha()]
    for a, b in zip(letters, letters[1:]):
        if text[a].lower() != text[b].lower():
            chars = list(text)
            chars[a], chars[b] = chars[b], chars[a]
            return "".join(chars)
    return None


def _shift_year(text: str) -> str | None:
    match = re.search(r"\b(19|20)(\d{2})\b", text)
    if not match:
        return None
    year = int(match.group(0))
    return text[: match.start()] + str(year - 1) + text[match.end():]


def _nudge_number(text: str) -> str | None:
    match = re.fullmatch(r"\s*(\d+)(?:\.(\d+))?\s*", text)
    if not match:
        return None
    whole = int(match.group(1))
    if match.group(2) is None:
        return str(whole + 3) if whole >= 3 else None
    return f"{whole}.{(int(match.group(2)) + 3) % 10}"


PERTURBATIONS: tuple[tuple[str, Callable[[str], str | None]], ...] = (
    ("transposed digits", _transpose_digits),
    ("year shifted by one", _shift_year),
    ("number nudged", _nudge_number),
    ("transposed letters", _transpose_letters),
)


@dataclass
class Probe:
    """One injected error, and its fate."""

    request_id: str
    field_name: str
    original: str
    injected: str
    kind: str
    injected_at: datetime
    resolved: bool = False
    caught: bool | None = None
    reviewer_id: str = ""
    resolved_at: datetime | None = None
    #: Set when the probe was cleared without ever being fairly presented -- the
    #: form was withdrawn, or re-filled before anybody reviewed it. Withdrawn
    #: probes are excluded from the catch rate: scoring a probe nobody was shown
    #: as a miss makes the reviewer look worse than they are, and scoring it as a
    #: catch makes them look better. Neither is a measurement.
    withdrawn_reason: str = ""

    @property
    def outstanding(self) -> bool:
        return not self.resolved

    @property
    def scored(self) -> bool:
        return self.resolved and not self.withdrawn_reason

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "field": self.field_name,
            "original": self.original,
            "injected": self.injected,
            "kind": self.kind,
            "injected_at": self.injected_at.isoformat(),
            "resolved": self.resolved,
            "caught": self.caught,
            "reviewer_id": self.reviewer_id,
            "withdrawn_reason": self.withdrawn_reason,
        }


@dataclass
class ProbeProgram:
    """Decides which forms get a probe, injects it, and scores the result."""

    audit: Any = None
    rate: int = DEFAULT_PROBE_RATE
    salt: str = "nsp-i01-probe"
    #: The CURRENT probe per request. A request that is re-filled gets a new
    #: one; the old one moves to `retired`.
    probes: dict[str, Probe] = field(default_factory=dict)
    #: Superseded probes. Kept because the catch rate is computed over every
    #: probe that was ever scored, and an earlier version overwrote the dict
    #: entry -- so a withdrawn probe vanished from the numbers that exist to
    #: record that it was withdrawn.
    retired: list[Probe] = field(default_factory=list)
    #: Request ids whose probe has been SCORED. `should_probe` refuses them
    #: forever after. Without this the deterministic hash re-probed the same
    #: request on every re-fill, so a probed request could never be released
    #: clean -- the only way out was to sign a form with an injected error on it.
    scored_requests: set[str] = field(default_factory=set)
    INITIATIVE: str = "I-01"

    def __post_init__(self) -> None:
        if self.rate < 2:
            raise ValueError(
                f"a probe rate of 1 in {self.rate} probes every form (or most "
                "of them). The control measures review quality; it is not a "
                "way to fill the queue with known-bad documents."
            )

    def should_probe(self, request_id: str) -> bool:
        """Deterministic per request id. See rule 4 in the module docstring.

        A request whose probe has already been scored is never probed again. The
        measurement has been taken; probing it a second time only guarantees the
        form cannot be released without an injected error on it.
        """
        if request_id in self.scored_requests:
            return False
        digest = hashlib.sha256(f"{self.salt}:{request_id}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self.rate == 0

    def candidates(self, filled: FilledForm) -> list[tuple[str, str]]:
        """Every safe, non-empty, auto-filled field, in a stable rotated order.

        Rotated by a hash of the patient and form rather than always starting at
        the first field: a probe that is always in the child's name teaches the
        reviewer to check the child's name, which is a different lesson from the
        one intended.
        """
        safe = [
            (w.field_name, w.text)
            for w in filled.writes
            if w.method == "auto" and _is_safe(w.field_name) and w.text.strip()
        ]
        if not safe:
            return []
        digest = hashlib.sha256(
            f"{self.salt}:{filled.patient_id}:{filled.form_type}".encode("utf-8")
        ).digest()
        start = int.from_bytes(digest[:4], "big") % len(safe)
        return safe[start:] + safe[:start]

    def choose_target(self, filled: FilledForm) -> tuple[str, str] | None:
        options = self.candidates(filled)
        return options[0] if options else None

    def build_override(
        self, filled: FilledForm, *, request_id: str, now: datetime
    ) -> dict[str, str] | None:
        """The `overrides` mapping `FormFiller.fill` needs, plus a recorded probe.

        Returns None when this form is not probed, or when nothing on it can be
        perturbed safely -- which is a normal outcome, not a failure. A form
        with no safe target simply is not probed; forcing one would mean
        reaching into the fields the allowlist exists to protect.
        """
        # Walk the candidates rather than taking the first. An earlier version
        # gave up when the chosen field could not be perturbed, so a form whose
        # rotation landed on a name produced no probe and nothing said so -- the
        # control silently not running.
        for field_name, original in self.candidates(filled):
            if not _is_safe(field_name):  # pragma: no cover - candidates filters
                raise UnsafeProbeTarget(field_name)
            for kind, perturb in PERTURBATIONS:
                injected = perturb(original)
                if injected is not None and injected != original:
                    break
            else:
                continue
            break
        else:
            return None

        existing = self.probes.pop(request_id, None)
        if existing is not None:
            self.retired.append(existing)
        probe = Probe(
            request_id=request_id, field_name=field_name, original=original,
            injected=injected, kind=kind, injected_at=now,
        )
        self.probes[request_id] = probe
        if self.audit is not None:
            self.audit.record_event(
                actor_id="system:probe",
                initiative_id=self.INITIATIVE,
                event_type="synthetic_probe_injected",
                patient_id=filled.patient_id,
                detail={
                    "request_id": request_id,
                    "field": field_name,
                    "kind": kind,
                    "form_type": filled.form_type,
                },
            )
        return {field_name: injected}

    def all_probes(self) -> list[Probe]:
        """Every probe ever injected, superseded ones included."""
        return list(self.retired) + list(self.probes.values())

    def outstanding_for(self, request_id: str) -> bool:
        probe = self.probes.get(request_id)
        return probe is not None and probe.outstanding

    def withdraw(self, request_id: str, *, reason: str, now: datetime) -> Probe | None:
        """Clear a probe nobody was ever shown, without scoring it.

        WHY THIS EXISTS. A probed form can be blocked for an unrelated reason --
        an unsettled dose, a stale vital -- and sent back to be re-filled. The
        first pass's probe was never presented to a reviewer as a finished form.
        Without this method it stayed outstanding, which meant two things, both
        wrong: the release gate blocked that request forever, and the outstanding
        list grew until the catch rate was computed over a shrinking, biased
        subset of probes.

        A withdrawn probe is resolved (so the gate clears) and excluded from the
        rate (so the measurement stays honest).
        """
        probe = self.probes.get(request_id)
        if probe is None or probe.resolved:
            return probe
        probe.resolved = True
        probe.caught = None
        probe.withdrawn_reason = reason
        probe.resolved_at = now
        if self.audit is not None:
            self.audit.record_event(
                actor_id="system:probe",
                initiative_id=self.INITIATIVE,
                event_type="synthetic_probe_withdrawn",
                detail={"request_id": request_id, "reason": reason[:200],
                        "field": probe.field_name},
            )
        return probe

    def resolve(
        self,
        request_id: str,
        payload: ReviewPayload,
        decision: ReviewDecision,
        *,
        now: datetime,
    ) -> Probe | None:
        """Score the review against the probe and clear the blocker.

        CAUGHT means the reviewer changed the probed field to something other
        than the injected value. Not "made any edit", and not "rejected the
        form" -- a reviewer who rejects every form catches every probe and
        reviews nothing, and a reviewer who edits an unrelated box has not found
        this one. Resubmitting the injected value verbatim is not a catch either,
        whatever action label comes with it.

        Scoring does NOT clear the injected error from the document. The caller
        must re-render; `ReleaseGate` blocks on `FilledForm.has_synthetic_write`
        until it does.
        """
        probe = self.probes.get(request_id)
        if probe is None or probe.resolved:
            return probe

        # ONE condition. An earlier version also counted "rejected the form and
        # touched this field", which never compared the value -- so a review
        # screen that resubmits fields as displayed, or a reviewer who sends the
        # form back with a note, scored as a catch while the injected error
        # stood. The number that exists to detect a rubber stamp was exactly the
        # number a rubber stamp produced.
        correction = decision.corrections.get(probe.field_name)
        caught = correction is not None and correction != probe.injected

        probe.resolved = True
        probe.caught = bool(caught)
        probe.reviewer_id = decision.reviewer_id
        probe.resolved_at = now
        self.scored_requests.add(request_id)

        if self.audit is not None:
            # `synthetic_probe=True` keeps this out of the genuine edit-rate
            # statistics and inside `probe_catch_rate()`. `edit_rate_applicable`
            # is False for the same reason: an injected error corrected by the
            # reviewer is not evidence about how good the machine's drafts are.
            self.audit.record_review(
                reviewer_id=decision.reviewer_id,
                initiative_id=self.INITIATIVE,
                draft=payload.draft_text(),
                final=_final_text(payload, decision),
                action_taken=decision.action,
                patient_id=payload.patient_id,
                synthetic_probe=True,
                probe_caught=probe.caught,
                review_seconds=decision.review_seconds,
                edit_rate_applicable=False,
                extra={
                    "request_id": request_id,
                    "field": probe.field_name,
                    "kind": probe.kind,
                    "form_type": payload.form_type,
                },
            )
        return probe

    # -- reports -----------------------------------------------------------

    def catch_rate(self) -> dict[str, Any]:
        """The number this whole module exists to produce.

        A falling catch rate is the earliest signal available that the review
        step has become a click. It is worth watching per reviewer as well as
        overall, because it is usually one person's workload that broke first.
        """
        # `scored`, not `resolved`: a withdrawn probe was never shown to
        # anybody and belongs in neither the numerator nor the denominator.
        every = self.all_probes()
        resolved = [p for p in every if p.scored]
        caught = [p for p in resolved if p.caught]
        by_reviewer: dict[str, dict[str, int]] = {}
        for probe in resolved:
            row = by_reviewer.setdefault(
                probe.reviewer_id or "unknown", {"probes": 0, "caught": 0}
            )
            row["probes"] += 1
            row["caught"] += int(bool(probe.caught))
        return {
            "injected": len(every),
            "withdrawn": sum(1 for p in every if p.withdrawn_reason),
            "scored": len(resolved),
            "caught": len(caught),
            "catch_rate": round(len(caught) / len(resolved), 3) if resolved else None,
            "outstanding": [p.request_id for p in every if p.outstanding],
            "by_reviewer": {
                reviewer: {
                    **row,
                    "catch_rate": round(row["caught"] / row["probes"], 3),
                }
                for reviewer, row in sorted(by_reviewer.items())
            },
        }


def _final_text(payload: ReviewPayload, decision: ReviewDecision) -> str:
    values = {row.field_name: row.written for row in payload.provenance}
    values.update(decision.corrections)
    return "\n".join(f"{k}={v}" for k, v in sorted(values.items()))
