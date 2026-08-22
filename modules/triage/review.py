"""Signing, and the edit-diff that proves a human was actually there.

README I-04: "The edit-diff log is not optional. It is the mechanism by which
the practice proves that a human actually reviewed the draft rather than
rubber-stamping it, and it is the dataset that tells you whether the system is
getting better or worse over time."

README 10.3 goes further and calls the "< 5% edit rate" alarm one of the two
most important lines in the whole document: an MA whose edit distance on
AI-drafted triage notes collapses to zero has stopped reviewing, and an
unreviewed machine-generated clinical record with a human signature on it is
substantially worse than the manual process it replaced.

So this module does three things and refuses to do a fourth:

  * It computes the edit distance in `nsp_core.audit` -- one implementation, not
    a second one here that could disagree flatteringly.
  * It records the review with `AuditLog.record_review`, which is append-only at
    the database level.
  * It surfaces the per-reviewer edit-rate report so the alarm is something the
    practice looks at rather than something the code merely enables.

The fourth thing, which it will not do, is sign on anyone's behalf. There is no
auto-sign, no bulk-sign, no "accept all". `sign()` takes a named person and the
text they are actually putting their name to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence

from nsp_core.audit import ReviewOutcome, edit_distance, edit_ratio
from modules.scheduling.models import iso

from .protocols import MATaps, ProtocolRegistry
from .structure import StructuredNote

__all__ = [
    "SignedNote",
    "UnreviewedDropError",
    "NoteReviewer",
    "RUBBER_STAMP_ALARM_RATIO",
    "POOR_DRAFT_ALARM_RATIO",
]

#: README 10.3, triage notes row: healthy band 15-40%, alarm below 5%
#: (rubber-stamping) or above 60% (the drafts are bad).
RUBBER_STAMP_ALARM_RATIO = 0.05
POOR_DRAFT_ALARM_RATIO = 0.60


class UnreviewedDropError(RuntimeError):
    """Raised when a note is signed without acknowledging removed content."""


@dataclass(frozen=True)
class SignedNote:
    encounter_id: str
    signed_by: str
    signed_utc: datetime
    text: str
    baseline_text: str
    edit_distance: int
    edit_ratio: float
    review_id: str | None
    action_taken: str
    review_seconds: float | None = None

    @property
    def looks_rubber_stamped(self) -> bool:
        return self.edit_ratio < RUBBER_STAMP_ALARM_RATIO

    def as_dict(self) -> dict[str, Any]:
        return {
            "encounter_id": self.encounter_id,
            "signed_by": self.signed_by,
            "signed_utc": iso(self.signed_utc),
            "edit_distance": self.edit_distance,
            "edit_ratio": round(self.edit_ratio, 4),
            "action_taken": self.action_taken,
            "review_seconds": self.review_seconds,
        }


class NoteReviewer:
    """Turns a draft plus a human into a signed note and an audit record."""

    INITIATIVE = "I-04"

    def __init__(
        self,
        *,
        audit: Any = None,
        registry: ProtocolRegistry | None = None,
        require_drop_acknowledgement: bool = True,
    ) -> None:
        self.audit = audit
        self.registry = registry
        # A note whose machine-removed content the reviewer never saw is a note
        # reviewed against an edited version of the facts. Default on.
        self.require_drop_acknowledgement = require_drop_acknowledgement

    def sign(
        self,
        *,
        note: StructuredNote,
        taps: MATaps,
        baseline_text: str,
        final_text: str,
        signed_by: str,
        now: datetime,
        patient_id: str | None = None,
        review_seconds: float | None = None,
        acknowledged_drops: bool = False,
        synthetic_probe: bool = False,
        probe_caught: bool | None = None,
    ) -> SignedNote:
        """Sign the note. Records the diff whether or not anyone looks at it.

        `baseline_text` is the chart note the MACHINE proposed. `final_text` is
        what the MA is putting their name to. The distance between the two is
        the entire point, which is why the baseline must be the machine's
        version of the SAME document -- not the reviewer's draft view, which
        carries banners, removal lists and a model footer the chart note never
        has. Diffing those two produces a healthy-looking edit ratio for a
        signature nobody read, and the "< 5% edit rate" alarm can then never
        fire. See `render.render_note` and `render.signature_block`.
        """
        if not signed_by or not signed_by.strip():
            raise ValueError(
                "a note must be signed by a named person; there is no system "
                "signature and no auto-sign path"
            )
        if self.registry is not None:
            # Refuses if the protocol or disposition was never tapped.
            self.registry.validate(taps)
        if self.require_drop_acknowledgement and not acknowledged_drops:
            # Flagged items are gated as well as dropped ones. A flagged item is
            # machine-authored text the transcript did NOT clearly support, and
            # it is going into the chart -- marked, but going in. Gating only on
            # removals would mean the content that actually reaches the record is
            # the content nobody had to look at.
            outstanding = len(note.dropped) + len(note.flagged)
            if outstanding:
                raise UnreviewedDropError(
                    f"{len(note.dropped)} item(s) were removed and "
                    f"{len(note.flagged)} kept-but-unsupported item(s) are marked "
                    "in this note. The signer has to see both before signing -- "
                    "pass acknowledged_drops=True once they have."
                )
        if not final_text.strip():
            raise ValueError("refusing to sign an empty note")

        distance = edit_distance(baseline_text, final_text)
        ratio = edit_ratio(baseline_text, final_text)
        action = ReviewOutcome.ACCEPTED if distance == 0 else ReviewOutcome.EDITED

        review_id = None
        if self.audit is not None:
            review_id = self.audit.record_review(
                reviewer_id=signed_by,
                initiative_id=self.INITIATIVE,
                draft=baseline_text,
                final=final_text,
                action_taken=action,
                inference_id=note.inference_id,
                patient_id=patient_id,
                review_seconds=review_seconds,
                synthetic_probe=synthetic_probe,
                probe_caught=probe_caught,
                # Free-text drafts are exactly the case the edit-rate alarm was
                # designed for, so this one counts.
                edit_rate_applicable=True,
                extra={
                    "encounter_id": note.encounter_id,
                    "transcript_sha256": note.transcript_sha256,
                    "protocol_id": taps.protocol_id,
                    "disposition_id": taps.disposition_id,
                    "dropped_items": len(note.dropped),
                },
            )

        return SignedNote(
            encounter_id=note.encounter_id,
            signed_by=signed_by,
            signed_utc=now,
            text=final_text,
            baseline_text=baseline_text,
            edit_distance=distance,
            edit_ratio=ratio,
            review_id=review_id,
            action_taken=action,
            review_seconds=review_seconds,
        )

    def reject(
        self,
        *,
        note: StructuredNote,
        rejected_by: str,
        reason: str,
        baseline_text: str,
        now: datetime,
        patient_id: str | None = None,
    ) -> None:
        """Throw the draft away and document by hand.

        A first-class outcome with its own audit trail. If rejection is only
        available as "close the tab", the rejection rate is unmeasurable and the
        practice cannot tell a bad week from a bad prompt.
        """
        if not reason.strip():
            raise ValueError("a rejection must record why")
        if self.audit is not None:
            self.audit.record_review(
                reviewer_id=rejected_by,
                initiative_id=self.INITIATIVE,
                draft=baseline_text,
                final="",
                action_taken=ReviewOutcome.REJECTED,
                inference_id=note.inference_id,
                patient_id=patient_id,
                edit_rate_applicable=False,
                extra={"encounter_id": note.encounter_id, "reason": reason},
            )

    # -- the alarm ---------------------------------------------------------

    def edit_rate_report(self, *, min_reviews: int = 10) -> dict[str, Any]:
        """README 10.3's triage-note edit-rate table, per reviewer.

        Reported in both directions. Below 5% means the reviewer has stopped
        reading; above 60% means the drafts are bad and the MA is retyping them,
        which is a prompt problem, not a person problem. Conflating the two --
        or only alarming on one -- is how a documentation programme quietly
        becomes a transcription programme with extra steps.
        """
        if self.audit is None:
            return {"available": False, "reason": "no audit log configured"}
        findings = self.audit.rubber_stamp_report(
            initiative_id=self.INITIATIVE,
            min_reviews=min_reviews,
            alarm_ratio=RUBBER_STAMP_ALARM_RATIO,
        )
        reviewers = []
        for finding in findings:
            reviewers.append(
                {
                    "reviewer_id": finding.reviewer_id,
                    "reviews": finding.reviews,
                    "median_edit_ratio": round(finding.median_edit_ratio, 4),
                    "zero_edit_fraction": round(finding.zero_edit_fraction, 4),
                    "rubber_stamp_alarm": finding.alarm,
                    "poor_draft_alarm": (
                        finding.reviews >= min_reviews
                        and finding.median_edit_ratio > POOR_DRAFT_ALARM_RATIO
                    ),
                            # Guarded on sample size: one note is not a band.
                    "in_healthy_band": (
                        finding.reviews >= min_reviews
                        and 0.15 <= finding.median_edit_ratio <= 0.40
                    ),
                }
            )
        return {
            "available": True,
            "healthy_band": [0.15, 0.40],
            "rubber_stamp_alarm_below": RUBBER_STAMP_ALARM_RATIO,
            "poor_draft_alarm_above": POOR_DRAFT_ALARM_RATIO,
            "reviewers": sorted(reviewers, key=lambda r: r["median_edit_ratio"]),
        }
