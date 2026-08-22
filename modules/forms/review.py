"""The one screen the MA reviews, and the gate that decides what may leave it.

README I-01, target state, step 7:

    MA reviews in a single screen: filled form on the left, source data with
    provenance on the right. **Discrepancies surface at the top.**

`build_review()` produces that screen as a payload -- this repo has no UI, and a
payload is the honest boundary. But the interesting half of this module is not
the layout, it is `ReleaseGate`, and the reason is in README I-01's risk table:

    Staff stop reviewing carefully once accuracy is good (automation
    complacency) | HIGH | ... This is the control most implementations omit and
    most need.

A review screen is not a control. A review screen plus a gate that refuses to
release certain forms no matter how many times the reviewer clicks approve is a
control. So `ReleaseGate.blockers()` is a list of conditions under which a form
does not go to a physician's signature queue at all:

  * an unresolved immunization discrepancy
  * the registry never consulted -- README I-01's "flag the form as registry not
    reconciled", made mechanical
  * a required box nothing filled
  * a value truncated to fit its box
  * a stale vital
  * an outstanding synthetic probe

Every one of them is a condition the MA could otherwise click past, and the last
one is the point of `probe.py`.

WHAT A REVIEW COSTS. `record_review` computes the edit distance between the
machine's draft field values and what the reviewer released, so §10.3's
edit-rate report covers forms too. A form approved in four seconds with sixty
auto-filled fields and zero edits, week after week, is what automation
complacency looks like in a database, and it is visible here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence

from .chart import ChartRecord, DEFAULT_STALE_DAYS, SourceValue
from .fill import FilledForm
from .templates import FormTemplate

__all__ = [
    "ProvenanceRow",
    "ReviewPayload",
    "ReleaseGate",
    "ReviewDecision",
    "NotReleasable",
    "build_review",
    "record_review",
]


class NotReleasable(RuntimeError):
    """Raised when a form is sent for signature with blockers outstanding."""


@dataclass(frozen=True)
class ProvenanceRow:
    """One line of the right-hand pane: what was written and where it came from."""

    field_name: str
    label: str
    written: str
    method: str
    #: Where on the form this box is, so the right-hand pane reads in the same
    #: order as the page the MA is looking at.
    page: int
    y: float
    source_path: str
    system: str
    resource: str
    recorded: date | None
    age_days: int | None
    stale: bool
    #: True when this value is a historical event rather than a measurement --
    #: an immunization given in 2018 does not get fresher. Carried so the pane
    #: and the gate answer the staleness question the same way.
    historical: bool = False
    truncated_from: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field_name,
            "label": self.label,
            "written": self.written,
            "method": self.method,
            "page": self.page,
            "source": {
                "path": self.source_path,
                "historical": self.historical,
                "system": self.system,
                "resource": self.resource,
                "recorded": self.recorded.isoformat() if self.recorded else None,
                "age_days": self.age_days,
                "stale": self.stale,
            },
            "truncated_from": self.truncated_from,
        }


@dataclass
class ReviewPayload:
    """Everything the review screen renders. Discrepancies first, always."""

    form_type: str
    patient_id: str
    pdf_path: str
    #: TOP of the screen. Ordering is by severity then by kind, and the
    #: structure puts them first rather than relying on a UI to do it: a
    #: discrepancy list a front end can choose to render lower down is a
    #: discrepancy list that will be rendered lower down.
    discrepancies: list[dict[str, Any]] = field(default_factory=list)
    blockers: list[dict[str, Any]] = field(default_factory=list)
    provenance: list[ProvenanceRow] = field(default_factory=list)
    blank_fields: list[dict[str, Any]] = field(default_factory=list)
    auto_filled_count: int = 0
    generated_utc: datetime | None = None

    @property
    def releasable(self) -> bool:
        return not self.blockers

    def draft_text(self) -> str:
        """A stable serialisation of the machine's proposal.

        This is the `draft` half of the edit-distance calculation in
        `AuditLog.record_review`. Sorted by field name so a reordering in the
        UI does not read as an edit.
        """
        return "\n".join(
            f"{row.field_name}={row.written}"
            for row in sorted(self.provenance, key=lambda r: r.field_name)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "form_type": self.form_type,
            "patient_id": self.patient_id,
            "pdf": self.pdf_path,
            "generated_utc": (
                self.generated_utc.isoformat() if self.generated_utc else None
            ),
            "releasable": self.releasable,
            "discrepancies": list(self.discrepancies),
            "blockers": list(self.blockers),
            "auto_filled_count": self.auto_filled_count,
            "provenance": [row.as_dict() for row in self.provenance],
            "blank_fields": list(self.blank_fields),
        }


@dataclass
class ReleaseGate:
    """Conditions under which a form does not reach a signature queue.

    Separate from the payload so the same rules run twice: once to render the
    screen, and once at the moment of release. A gate that only exists in the
    rendering path is a gate any other caller walks around.
    """

    stale_days: int = DEFAULT_STALE_DAYS
    #: Truncation blocks by default. A clipped allergy list on a school form
    #: reads as a complete one, and the reviewer sees the clipped version.
    block_on_truncation: bool = True

    def blockers(
        self,
        filled: FilledForm,
        record: ChartRecord,
        *,
        as_of: date,
        probe_outstanding: bool = False,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []

        for row in filled.discrepancies:
            if row.get("severity") == "blocking":
                out.append(
                    {
                        "kind": row.get("kind", "discrepancy"),
                        "detail": row.get("detail", ""),
                        "resolution": "settle it in the reconciliation queue, "
                                      "then re-fill this form",
                    }
                )

        for missing in filled.missing_required():
            out.append(
                {
                    "kind": "required_field_blank",
                    "detail": f"{missing.field_name}: {missing.reason}",
                    "resolution": "enter it by hand, or correct the chart and "
                                  "re-fill",
                }
            )

        if self.block_on_truncation:
            for write in filled.truncated:
                out.append(
                    {
                        "kind": "value_truncated",
                        "detail": (
                            f"{write.field_name} did not fit its box. The form "
                            f"shows {write.text!r}; the chart says "
                            f"{write.truncated_from!r}"
                        ),
                        "resolution": "shorten it deliberately, or attach a "
                                      "continuation sheet",
                    }
                )

        for stale in record.stale_values(as_of, stale_days=self.stale_days):
            # Only for values that actually reached the form. A stale field
            # nothing printed is not this form's problem, and listing it trains
            # the reviewer to skim.
            # `writes`, not `auto_filled`. A probed field is written with
            # method="probe" and so is absent from `auto_filled` -- so probing a
            # box erased the evidence that a stale chart value had reached the
            # form. The value came from the same chart path either way; who
            # supplied the TEXT does not change where the box's data comes from.
            if not any(w.source_path == stale["path"] for w in filled.writes):
                continue
            age = stale["age_days"]
            out.append(
                {
                    "kind": "stale_value",
                    "detail": (
                        f"{stale['path']} was recorded "
                        + (f"{age} days ago" if age is not None else "on no date")
                        + f" (limit {self.stale_days}); a form filled from it "
                        "looks current and is not"
                    ),
                    "resolution": "re-measure at this visit, or confirm the "
                                  "value is still the one to report",
                }
            )

        if probe_outstanding:
            out.append(
                {
                    "kind": "synthetic_probe_outstanding",
                    "detail": (
                        "this form carries a deliberate synthetic error that "
                        "has not been scored yet. It is a test of the review "
                        "step and it must never reach a signature."
                    ),
                    "resolution": "score it with `ProbeProgram.resolve`, then "
                                  "re-fill the form",
                }
            )
        if filled.has_synthetic_write:
            # A property of the DOCUMENT, not of the probe registry. Scoring a
            # probe records what the reviewer did; it does not take the wrong
            # value off the page. An earlier version blocked only on the
            # registry flag, so clearing that flag and signing left the injected
            # error printed on the form that went to the school.
            injected = [w.field_name for w in filled.writes if w.synthetic]
            out.append(
                {
                    "kind": "synthetic_value_on_the_document",
                    "detail": (
                        f"{injected} on this PDF still hold a deliberately wrong "
                        "value. Scoring the probe does not re-render the page."
                    ),
                    "resolution": "re-fill this form; the request is not probed "
                                  "again once its probe has been scored",
                }
            )
        return out


def build_review(
    template: FormTemplate,
    filled: FilledForm,
    record: ChartRecord,
    *,
    as_of: date,
    gate: ReleaseGate | None = None,
    probe_outstanding: bool = False,
) -> ReviewPayload:
    """Assemble the split-pane payload."""
    gate = gate or ReleaseGate()
    rows: list[ProvenanceRow] = []
    for write in filled.writes:
        try:
            spec = template.field(write.field_name)
            label = spec.label or spec.name
        except KeyError:  # pragma: no cover - a write always comes from a spec
            spec, label = None, write.field_name

        # Ask the SourceValue itself, the same call `ReleaseGate` makes through
        # `record.stale_values()`. An earlier version recomputed staleness from
        # the write's date alone, so it flagged every immunization row on every
        # form -- nine meaningless "stale" marks next to the three that mattered,
        # which is the shape of alert that trains a reviewer to skip the list.
        source: SourceValue | None = None
        if write.method == "auto" and write.source_path:
            try:
                source = record.resolve(write.source_path)
            except Exception:  # noqa: BLE001 - a bad path is reported elsewhere
                source = None

        recorded = write.source_recorded
        age = (as_of - recorded).days if recorded else None
        historical = bool(source and source.historical)
        stale = (
            source.is_stale(as_of, stale_days=gate.stale_days)
            if source is not None
            else (write.method == "auto" and (age is None or age > gate.stale_days))
        )
        rows.append(
            ProvenanceRow(
                field_name=write.field_name,
                label=label,
                written=write.text,
                method=write.method,
                page=write.box.page,
                y=write.box.y,
                source_path=write.source_path,
                historical=historical,
                system=write.source_system,
                resource=write.source_resource,
                recorded=recorded,
                age_days=age,
                stale=stale,
                truncated_from=write.truncated_from,
            )
        )

    return ReviewPayload(
        form_type=filled.form_type,
        patient_id=filled.patient_id,
        pdf_path=filled.output_path,
        discrepancies=sorted(
            filled.discrepancies,
            key=lambda d: (0 if d.get("severity") == "blocking" else 1,
                           str(d.get("kind", ""))),
        ),
        blockers=gate.blockers(
            filled, record, as_of=as_of, probe_outstanding=probe_outstanding
        ),
        # Reading order on the page, not dictionary order. A provenance pane
        # ordered differently from the form makes the reviewer hunt for each row.
        provenance=sorted(rows, key=lambda r: (r.page, r.y, r.field_name)),
        blank_fields=[s.as_dict() for s in filled.skipped],
        auto_filled_count=len(filled.auto_filled),
        generated_utc=filled.generated_utc,
    )


@dataclass(frozen=True)
class ReviewDecision:
    """What a person did with the screen."""

    reviewer_id: str
    #: "accepted" | "edited" | "rejected"
    action: str
    #: Field name -> the value the reviewer put there instead.
    corrections: Mapping[str, str] = field(default_factory=dict)
    review_seconds: float | None = None
    note: str = ""

    @property
    def approved(self) -> bool:
        return self.action in ("accepted", "edited")


def record_review(
    payload: ReviewPayload,
    decision: ReviewDecision,
    *,
    audit: Any,
    synthetic_probe: bool = False,
    probe_caught: bool | None = None,
) -> str:
    """Write the review event, with the edit distance computed by the audit log.

    Refuses to record an APPROVAL of a form with outstanding blockers. That
    refusal is the gate's teeth: without it the gate is advisory, and an
    advisory gate on a legal document is a suggestion.
    """
    if decision.approved and payload.blockers:
        raise NotReleasable(
            f"{len(payload.blockers)} blocker(s) outstanding on this form: "
            + "; ".join(b["kind"] for b in payload.blockers)
            + ". A form does not reach a physician's signature queue with an "
            "unsettled immunization discrepancy, a blank required field, a "
            "truncated value or an outstanding probe on it."
        )
    if not decision.reviewer_id.strip():
        raise ValueError("a review event names the person who did the review")

    final = _apply(payload, decision.corrections)
    return audit.record_review(
        reviewer_id=decision.reviewer_id,
        initiative_id="I-01",
        draft=payload.draft_text(),
        final=final,
        action_taken=decision.action,
        patient_id=payload.patient_id,
        synthetic_probe=synthetic_probe,
        probe_caught=probe_caught,
        review_seconds=decision.review_seconds,
        extra={
            "form_type": payload.form_type,
            "auto_filled": payload.auto_filled_count,
            "corrections": sorted(decision.corrections),
            "note": decision.note[:200],
        },
    )


def _apply(payload: ReviewPayload, corrections: Mapping[str, str]) -> str:
    values = {row.field_name: row.written for row in payload.provenance}
    values.update(corrections)
    return "\n".join(f"{k}={v}" for k, v in sorted(values.items()))
