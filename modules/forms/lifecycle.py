"""The tracking record, so no form is ever "in the pile somewhere".

README I-01's failure table has an entry with no clinical content at all:

    Form lost in the signature pile | Child cannot start school or a sports
    season

and its benefit model prices the consequence: twelve "where is my form" calls a
week, five minutes each, because nobody can answer the question without a
physical search. Both of those are one missing thing -- a record of where each
form is, that changes only through named transitions.

So this module is a state machine and a ledger, and it is deliberately dull.

    received -> filled -> ma_review -> physician_signature -> signed
                                    -> delivered -> closed

with `blocked` reachable from `ma_review` and `withdrawn` reachable from
anywhere before `signed`. Every transition names an actor, records a timestamp,
and is appended to an immutable list. There is no `set_state`.

THREE THINGS THE MACHINE ENFORCES, each of which is a README line:

  1. **A form cannot reach `physician_signature` while the release gate has
     blockers.** README I-01 step 8 is "MA approves -> routes to physician
     signature queue"; the approval is what the gate guards, and the transition
     asks the gate rather than trusting the caller.
  2. **Signing requires a licensed professional's identifier, and it is a
     different person from nobody.** A `signed` transition with an empty
     `actor` is refused. This is the same posture as I-04's escalation rule:
     the moment a licensed professional takes responsibility is the moment the
     record has to say who.
  3. **A form cannot be delivered before it is signed.** An unsigned school
     physical arriving at a school is a rejected form and a repeated cycle,
     which is failure mode one in the same table.

WHAT AGE MEANS HERE. `overdue()` is the report that answers "where is my form"
before the parent asks, and `stalled_at()` says which stage the practice is
actually slow at. README I-01's KPI table wants form turnaround under 24 hours;
a turnaround number with no per-stage breakdown tells you that you are slow
without telling you where.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "FormState",
    "TRANSITIONS",
    "Transition",
    "FormRequest",
    "FormTracker",
    "IllegalTransition",
    "SignatureRequiresProfessional",
    "DEFAULT_STAGE_TARGETS",
]


class FormState:
    RECEIVED = "received"
    FILLED = "filled"
    MA_REVIEW = "ma_review"
    BLOCKED = "blocked"
    PHYSICIAN_SIGNATURE = "physician_signature"
    SIGNED = "signed"
    DELIVERED = "delivered"
    CLOSED = "closed"
    WITHDRAWN = "withdrawn"

    ALL = (
        RECEIVED, FILLED, MA_REVIEW, BLOCKED, PHYSICIAN_SIGNATURE,
        SIGNED, DELIVERED, CLOSED, WITHDRAWN,
    )
    #: States from which nothing further happens.
    TERMINAL = frozenset({CLOSED, WITHDRAWN})


#: The only legal moves. Anything not in here is refused, which is what makes
#: the ledger trustworthy: a form in `delivered` provably passed through
#: `signed`, because there is no other way in.
TRANSITIONS: Mapping[str, frozenset[str]] = {
    FormState.RECEIVED: frozenset({FormState.FILLED, FormState.WITHDRAWN}),
    FormState.FILLED: frozenset({FormState.MA_REVIEW, FormState.WITHDRAWN}),
    FormState.MA_REVIEW: frozenset(
        {FormState.PHYSICIAN_SIGNATURE, FormState.BLOCKED,
         FormState.FILLED, FormState.WITHDRAWN}
    ),
    # A blocked form goes back to be re-filled once the underlying problem --
    # an unsettled dose, a stale vital -- is fixed. It does NOT go straight to
    # signature: the fix has to produce a new document.
    FormState.BLOCKED: frozenset({FormState.FILLED, FormState.WITHDRAWN}),
    FormState.PHYSICIAN_SIGNATURE: frozenset(
        {FormState.SIGNED, FormState.MA_REVIEW, FormState.WITHDRAWN}
    ),
    FormState.SIGNED: frozenset({FormState.DELIVERED}),
    FormState.DELIVERED: frozenset({FormState.CLOSED}),
    FormState.CLOSED: frozenset(),
    FormState.WITHDRAWN: frozenset(),
}

#: How long each stage should take, from README I-01's "< 24 hrs" turnaround
#: target broken into the stages a practice can actually act on. Hours.
DEFAULT_STAGE_TARGETS: Mapping[str, float] = {
    FormState.RECEIVED: 4.0,
    FormState.FILLED: 2.0,
    FormState.MA_REVIEW: 8.0,
    FormState.BLOCKED: 24.0,
    FormState.PHYSICIAN_SIGNATURE: 24.0,
    FormState.SIGNED: 8.0,
    FormState.DELIVERED: 24.0,
}


class IllegalTransition(RuntimeError):
    """Raised on a move the state machine does not have."""


class SignatureRequiresProfessional(RuntimeError):
    """Raised when a form is signed without naming who signed it."""


@dataclass(frozen=True)
class Transition:
    from_state: str
    to_state: str
    actor: str
    at: datetime
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_state, "to": self.to_state,
            "actor": self.actor, "at": self.at.isoformat(), "note": self.note,
        }


@dataclass
class FormRequest:
    """One form, from the moment it arrives to the moment it is closed."""

    request_id: str
    patient_id: str
    form_type: str
    channel: str            # "front_desk" | "fax" | "email" | "portal" | "visit"
    received_at: datetime
    due_by: datetime | None = None
    state: str = FormState.RECEIVED
    history: list[Transition] = field(default_factory=list)
    #: Set when the physician signs. There is no default and no system value.
    signed_by: str = ""
    signed_at: datetime | None = None
    delivered_to: str = ""
    #: The parent-facing status notifications sent so far. README I-01 step 10:
    #: a notification at each stage is what removes the "where is my form" call.
    notifications: list[dict[str, Any]] = field(default_factory=list)
    #: The blockers on the CURRENT fill, recorded by the pipeline when it
    #: produces the review. The tracker reads this rather than accepting a list
    #: from whoever calls `advance`: an optional keyword defaulting to `()` made
    #: "this form is clean" and "the caller did not mention it" the same thing,
    #: so any caller could sign a form with an unsettled immunization on it.
    review_blockers: list[dict[str, Any]] = field(default_factory=list)
    #: Set by the pipeline each time it fills. Cleared on any move away from
    #: `ma_review`, so a stale review cannot authorise a later document.
    reviewed_fill: str = ""

    @property
    def terminal(self) -> bool:
        return self.state in FormState.TERMINAL

    def entered_current_state_at(self) -> datetime:
        return self.history[-1].at if self.history else self.received_at

    def age_hours(self, now: datetime) -> float:
        return (now - self.received_at).total_seconds() / 3600.0

    def stage_hours(self, now: datetime) -> float:
        return (now - self.entered_current_state_at()).total_seconds() / 3600.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "patient_id": self.patient_id,
            "form_type": self.form_type,
            "channel": self.channel,
            "state": self.state,
            "received_at": self.received_at.isoformat(),
            "due_by": self.due_by.isoformat() if self.due_by else None,
            "signed_by": self.signed_by,
            "signed_at": self.signed_at.isoformat() if self.signed_at else None,
            "delivered_to": self.delivered_to,
            "reviewed_fill": self.reviewed_fill,
            "review_blockers": [b.get("kind", "?") for b in self.review_blockers],
            "history": [t.as_dict() for t in self.history],
            "notifications": list(self.notifications),
        }


@dataclass
class FormTracker:
    """The ledger. Holds requests and is the only thing that moves them."""

    requests: dict[str, FormRequest] = field(default_factory=dict)
    audit: Any = None
    stage_targets: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_STAGE_TARGETS)
    )
    INITIATIVE: str = "I-01"

    def receive(
        self,
        *,
        request_id: str,
        patient_id: str,
        form_type: str,
        channel: str,
        now: datetime,
        due_by: datetime | None = None,
    ) -> FormRequest:
        if request_id in self.requests:
            raise IllegalTransition(f"request {request_id!r} already exists")
        request = FormRequest(
            request_id=request_id, patient_id=patient_id, form_type=form_type,
            channel=channel, received_at=now, due_by=due_by,
        )
        self.requests[request_id] = request
        self._audit(request, "form_received", {"channel": channel})
        return request

    def advance(
        self,
        request: FormRequest,
        to_state: str,
        *,
        actor: str,
        now: datetime,
        note: str = "",
        blockers: Sequence[Mapping[str, Any]] = (),
        signed_by: str = "",
        delivered_to: str = "",
    ) -> FormRequest:
        """Move one step. Every guard in the module docstring lives here."""
        if to_state not in FormState.ALL:
            raise IllegalTransition(f"{to_state!r} is not a state")
        allowed = TRANSITIONS.get(request.state, frozenset())
        if to_state not in allowed:
            raise IllegalTransition(
                f"a form in {request.state!r} cannot move to {to_state!r}; "
                f"legal moves are {sorted(allowed) or 'none (terminal)'}"
            )
        if not actor.strip():
            raise IllegalTransition(
                "every transition names who made it; a ledger with anonymous "
                "moves answers 'where is my form' with 'somewhere'"
            )

        if to_state == FormState.PHYSICIAN_SIGNATURE:
            # Read from the REQUEST, not from the caller. `blockers=` is still
            # accepted and is merged in, but omitting it no longer means "clean".
            outstanding = list(request.review_blockers) + list(blockers)
            if not request.reviewed_fill:
                raise IllegalTransition(
                    "no review has been recorded for the current fill of this "
                    "form. A form reaches a signature queue because a person "
                    "reviewed the document that exists now, not because a "
                    "caller asked for the transition."
                )
            if outstanding:
                raise IllegalTransition(
                    f"this form has {len(outstanding)} outstanding blocker(s) "
                    "and does not reach a signature queue: "
                    + "; ".join(str(b.get("kind", "?")) for b in outstanding)
                )

        if to_state == FormState.SIGNED:
            if not signed_by.strip():
                raise SignatureRequiresProfessional(
                    "a signed form names the licensed professional who signed "
                    "it. There is no system signer and no default."
                )
            request.signed_by = signed_by
            request.signed_at = now

        if to_state == FormState.DELIVERED:
            if request.state != FormState.SIGNED:  # pragma: no cover - TRANSITIONS
                raise IllegalTransition("a form is delivered only after signing")
            if not delivered_to.strip():
                raise IllegalTransition(
                    "delivery records where the form went; 'delivered' with no "
                    "destination is not a tracking record"
                )
            request.delivered_to = delivered_to

        transition = Transition(request.state, to_state, actor, now, note)
        request.history.append(transition)
        request.state = to_state
        if to_state in (FormState.FILLED, FormState.BLOCKED, FormState.WITHDRAWN):
            # A new document is coming, or none is. Either way the review that
            # authorised the last one does not authorise the next.
            request.reviewed_fill = ""
            request.review_blockers = []
        self._audit(
            request,
            f"form_{to_state}",
            {
                "from": transition.from_state,
                "actor": actor,
                "note": note[:200],
                "signed_by": request.signed_by,
                "delivered_to": request.delivered_to,
            },
        )
        return request

    def record_review_outcome(
        self,
        request: FormRequest,
        *,
        fill_id: str,
        blockers: Sequence[Mapping[str, Any]],
    ) -> None:
        """Attach the current fill's blockers to the request.

        Called by the pipeline the moment a review payload exists, so the
        tracker holds the evidence rather than asking the caller for it.
        """
        request.reviewed_fill = fill_id
        request.review_blockers = [dict(b) for b in blockers]

    def notify(
        self, request: FormRequest, *, channel: str, now: datetime, message: str
    ) -> dict[str, Any]:
        """Record a parent-facing status notification.

        Recorded rather than sent: delivery is somebody else's integration. What
        matters here is that the practice can prove the parent was told, which
        is what replaces the call rather than adding to it.
        """
        entry = {
            "at": now.isoformat(), "channel": channel,
            "state": request.state, "message": message[:300],
        }
        request.notifications.append(entry)
        self._audit(request, "form_notification", {"channel": channel})
        return entry

    # -- reports -----------------------------------------------------------

    def open_requests(self) -> list[FormRequest]:
        return [r for r in self.requests.values() if not r.terminal]

    def overdue(self, now: datetime) -> list[dict[str, Any]]:
        """Forms past their stage target or their due date, worst first."""
        rows: list[dict[str, Any]] = []
        for request in self.open_requests():
            target = self.stage_targets.get(request.state)
            stage_hours = request.stage_hours(now)
            past_due = request.due_by is not None and now > request.due_by
            if past_due or (target is not None and stage_hours > target):
                rows.append(
                    {
                        "request_id": request.request_id,
                        "patient_id": request.patient_id,
                        "form_type": request.form_type,
                        "state": request.state,
                        "hours_in_state": round(stage_hours, 1),
                        "stage_target_hours": target,
                        "total_age_hours": round(request.age_hours(now), 1),
                        "past_due_date": past_due,
                    }
                )
        return sorted(rows, key=lambda r: -r["total_age_hours"])

    def stalled_at(self, now: datetime) -> dict[str, int]:
        """How many open forms are sitting in each state.

        The turnaround KPI says the practice is slow. This says where.
        """
        counts: dict[str, int] = {}
        for request in self.open_requests():
            counts[request.state] = counts.get(request.state, 0) + 1
        return dict(sorted(counts.items()))

    def turnaround_report(self) -> dict[str, Any]:
        """Median and worst hours from receipt to signature, over closed forms.

        README I-01's KPI: "Form turnaround (request -> signed) | < 24 hrs".
        """
        durations = [
            (r.signed_at - r.received_at).total_seconds() / 3600.0
            for r in self.requests.values()
            if r.signed_at is not None
        ]
        if not durations:
            return {"signed_forms": 0, "median_hours": None, "worst_hours": None}
        ordered = sorted(durations)
        middle = len(ordered) // 2
        median = (
            ordered[middle]
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2
        )
        return {
            "signed_forms": len(ordered),
            "median_hours": round(median, 2),
            "worst_hours": round(ordered[-1], 2),
            "over_24h": sum(1 for d in ordered if d > 24.0),
        }

    def _audit(
        self, request: FormRequest, event_type: str, detail: Mapping[str, Any]
    ) -> None:
        if self.audit is None:
            return
        self.audit.record_event(
            actor_id=str(detail.get("actor") or "system:forms"),
            initiative_id=self.INITIATIVE,
            event_type=event_type,
            patient_id=request.patient_id,
            detail={
                "request_id": request.request_id,
                "form_type": request.form_type,
                "state": request.state,
                **{k: v for k, v in detail.items() if k != "actor"},
            },
        )
