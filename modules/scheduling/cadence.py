"""Reminder rules engine for I-07. Deterministic. No model, by design.

WHY there is no LLM in this file:

README I-07 is explicit, and it is the single most heavily marketed claim the
practice will be pitched. Reminders, confirmations and cancellations are a cron
job, a rules table and a messaging API. A language model adds latency, cost, a
BAA, an audit obligation and a hallucination surface to a problem that is
arithmetic on datetimes. If a future maintainer finds themselves prompting a
model to decide whether it is 21:00 yet, that is the bug.

The three hard problems in this file are all about time and consent:

  1. DST. Offsets are *elapsed durations* from the appointment instant, not
     civil-calendar arithmetic. T-48h means forty-eight hours of real time.
     Doing `local_start - timedelta(days=2)` on a wall clock produces a 47- or
     49-hour reminder twice a year, which is exactly the class of bug that gets
     blamed on the carrier and never fixed.
  2. Quiet hours are wall-clock, and therefore *are* civil arithmetic — 21:00
     local means 21:00 whatever the offset happens to be that week. The two
     kinds of time reasoning live side by side and must not be confused.
  3. Consent and suppression are legal gates and are evaluated at send time,
     not at plan time. A family that opts out on Tuesday must not receive the
     message that was queued for them on Monday.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence

from .models import (
    PRACTICE_TZ,
    Channel,
    ConsentPurpose,
    Database,
    MessagePurpose,
    VisitType,
    iso,
    new_id,
    parse_iso,
    to_local,
)

__all__ = [
    "ReminderRule",
    "DEFAULT_CADENCE",
    "QuietHours",
    "FrequencyCap",
    "CapDecision",
    "SendDecision",
    "SendGate",
    "PlannedMessage",
    "format_when",
    "ReminderEngine",
    "TEMPLATES",
]


# --------------------------------------------------------------------------
# The rules table
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReminderRule:
    """One scheduled touch, expressed as elapsed time before the appointment."""

    purpose: str
    offset: timedelta
    template_id: str
    #: If True, a send that would land inside quiet hours is pushed to 08:00
    #: rather than dropped. False means the touch is simply skipped -- correct
    #: for a T-2h reminder that would arrive after the visit started.
    defer_from_quiet_hours: bool = True


DEFAULT_CADENCE: dict[str, tuple[ReminderRule, ...]] = {
    # Well visits are booked months out and are the ones parents forget.
    VisitType.WELL: (
        ReminderRule(MessagePurpose.REMINDER_T7, timedelta(days=7), "wv_t7_confirm"),
        ReminderRule(MessagePurpose.REMINDER_T48, timedelta(hours=48), "wv_t48_remind"),
        ReminderRule(MessagePurpose.REMINDER_T2, timedelta(hours=2), "wv_t2_logistics"),
    ),
    # Sick visits are usually same- or next-day. A T-7 reminder for a visit
    # booked four hours ago is noise that spends the family's weekly message
    # budget on nothing.
    VisitType.SICK: (
        ReminderRule(MessagePurpose.REMINDER_T2, timedelta(hours=2), "sv_t2_logistics"),
    ),
    VisitType.FOLLOW_UP: (
        ReminderRule(MessagePurpose.REMINDER_T48, timedelta(hours=48), "fu_t48_remind"),
        ReminderRule(MessagePurpose.REMINDER_T2, timedelta(hours=2), "fu_t2_logistics"),
    ),
    VisitType.PROCEDURE: (
        ReminderRule(MessagePurpose.REMINDER_T7, timedelta(days=7), "pr_t7_prep"),
        ReminderRule(MessagePurpose.REMINDER_T48, timedelta(hours=48), "pr_t48_remind"),
        ReminderRule(MessagePurpose.REMINDER_T2, timedelta(hours=2), "pr_t2_logistics"),
    ),
}

TEMPLATES: dict[str, str] = {
    "wv_t7_confirm": (
        "North Suburban Pediatrics: {first_name}'s well visit with {provider} is "
        "{when}. Confirm: {confirm_url}  Cancel/reschedule: {cancel_url}  "
        "Reply STOP to opt out."
    ),
    "wv_t48_remind": (
        "Reminder: {first_name} sees {provider} {when}. Confirm {confirm_url} or "
        "cancel {cancel_url}. Reply STOP to opt out."
    ),
    "wv_t2_logistics": (
        "{first_name}'s visit is at {local_time} today. 1000 Weiland Rd, Buffalo "
        "Grove - park in the north lot. Bring the insurance card and any forms. "
        "Cancel: {cancel_url}"
    ),
    "sv_t2_logistics": (
        "{first_name}'s appointment is at {local_time} today with {provider}. "
        "Park in the north lot. Cancel: {cancel_url}"
    ),
    "fu_t48_remind": (
        "Reminder: {first_name}'s follow-up with {provider} is {when}. "
        "Confirm {confirm_url} or cancel {cancel_url}. Reply STOP to opt out."
    ),
    "fu_t2_logistics": "{first_name}'s follow-up is at {local_time} today. Cancel: {cancel_url}",
    "pr_t7_prep": (
        "{first_name}'s procedure with {provider} is {when}. Preparation "
        "instructions: {confirm_url}. Reply STOP to opt out."
    ),
    "pr_t48_remind": "Reminder: {first_name}'s procedure is {when}. Cancel: {cancel_url}",
    "pr_t2_logistics": "{first_name}'s procedure is at {local_time} today.",
    "backfill_offer": (
        "An earlier appointment opened up: {when} with {provider}. First to accept "
        "gets it. Accept: {accept_url}  No thanks: {decline_url}"
    ),
    "backfill_won": "Confirmed - {first_name} is booked {when} with {provider}.",
    "backfill_lost": "That slot was just taken. You are still on the list and we will keep looking.",
    "cancel_confirmation": (
        "Cancelled: {first_name}'s {when} appointment. To rebook, call 847-555-0100 "
        "or visit {confirm_url}."
    ),
    "optout_confirmation": (
        "You are unsubscribed from North Suburban Pediatrics appointment messages. "
        "No further texts will be sent. Reply START to resume."
    ),
}


def format_when(local_dt: datetime) -> str:
    """Human date/time for a message body, without glibc-only strftime flags.

    `%-d` and `%-I` are a glibc extension: they raise on musl (Alpine images)
    and on Windows. A reminder template that crashes on the deployment platform
    is a bad way to find that out.
    """
    return (
        f"{local_dt:%a %b} {local_dt.day} at {(local_dt.hour - 1) % 12 + 1}"
        f":{local_dt:%M} {local_dt:%p}"
    )


# --------------------------------------------------------------------------
# Quiet hours
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class QuietHours:
    """No sends between 21:00 and 08:00 practice-local.

    WHY wall clock rather than a UTC window: the point of quiet hours is that
    nobody's phone buzzes at 03:00. That is a fact about the family's clock,
    which shifts with DST. A fixed UTC window would drift an hour every spring
    and start texting at 04:00.
    """

    start: time = time(21, 0)
    end: time = time(8, 0)

    def __post_init__(self) -> None:
        if self.start == self.end:
            raise ValueError(
                "QuietHours(start == end) is ambiguous: it reads as a 24-hour "
                "window and silently behaves as none. Pass a real interval, or "
                "disable quiet hours explicitly at the call site."
            )

    @property
    def wraps_midnight(self) -> bool:
        return self.start > self.end

    def contains(self, moment: datetime) -> bool:
        local = to_local(moment).time()
        if not self.wraps_midnight:  # e.g. 01:00-05:00
            return self.start <= local < self.end
        return local >= self.start or local < self.end

    def next_open(self, moment: datetime) -> datetime:
        """Earliest sendable instant at or after `moment`."""
        if not self.contains(moment):
            return moment
        local = to_local(moment)
        candidate = local.replace(
            hour=self.end.hour, minute=self.end.minute, second=0, microsecond=0
        )
        # Roll to tomorrow only when the window wraps midnight AND we are on the
        # evening side of it. Testing `local.time() >= self.start` alone is wrong
        # for a non-wrapping window such as 01:00-05:00, where every in-window
        # moment satisfies it and the deferral lands a day late.
        if self.wraps_midnight and local.time() >= self.start:
            # Late evening: the next 08:00 is tomorrow morning. Adding a day to
            # a local wall time then re-localising keeps this civil, which is
            # what "08:00 tomorrow" means to a parent.
            candidate = (local + timedelta(days=1)).replace(
                hour=self.end.hour, minute=self.end.minute, second=0, microsecond=0
            )
        # Re-attach the zone so a transition day gets the correct offset.
        candidate = candidate.replace(tzinfo=PRACTICE_TZ)
        return candidate.astimezone(timezone.utc)


# --------------------------------------------------------------------------
# Frequency cap
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CapDecision:
    allowed: bool
    count: int
    limit: int
    tier: str
    reason: str = ""


class FrequencyCap:
    """Shared message budget per family per rolling week.

    IMPORTED BY OTHER MODULES ON PURPOSE. The immunization recall engine (I-02)
    must consult the same object, because the cap the README specifies is a cap
    across *all* initiatives, not per initiative. Two modules each politely
    sending three messages a week is six messages a week to the family, and the
    family does not care which internal system sent which.

    Tiers exist so the cap degrades in the right order. Appointment reminders
    for a booked visit are tier "appointment": the family has an appointment and
    needs to be reminded of it. Waitlist backfill offers are tier "waitlist" and
    get a slightly higher allowance, because that family explicitly asked to be
    told when something opens up — but they are still capped, since losing five
    blasts a week is precisely the message fatigue README R-10 warns about.
    Population outreach — recall, gap closure, campaigns — is tier "outreach"
    and gets the lowest limit, so a busy week of clinical scheduling squeezes
    out the recall message rather than the reverse.

    Transactional replies (cancel confirmations, opt-out confirmations, backfill
    win/loss) are never counted and never blocked. They are responses to
    something the family just did.
    """

    #: tier -> messages per family per window
    DEFAULT_LIMITS: Mapping[str, int] = {"appointment": 5, "waitlist": 6, "outreach": 2}
    TIER_APPOINTMENT = "appointment"
    TIER_WAITLIST = "waitlist"
    TIER_OUTREACH = "outreach"

    def __init__(
        self,
        db: Database,
        *,
        limits: Mapping[str, int] | None = None,
        window: timedelta = timedelta(days=7),
        exempt_purposes: Iterable[str] = MessagePurpose.TRANSACTIONAL,
    ) -> None:
        self.db = db
        self.limits = dict(limits or self.DEFAULT_LIMITS)
        self.window = window
        self.exempt_purposes = frozenset(exempt_purposes)

    def limit_for(self, tier: str) -> int:
        try:
            return self.limits[tier]
        except KeyError as exc:
            raise ValueError(f"unknown frequency-cap tier {tier!r}") from exc

    def count(self, family_id: str, at: datetime) -> int:
        """Non-exempt messages actually sent to this family in the window."""
        since = iso(at - self.window)
        placeholders = ",".join("?" for _ in self.exempt_purposes) or "''"
        row = self.db.one(
            f"""SELECT COUNT(*) AS c FROM message_log
                WHERE family_id = ? AND sent_utc IS NOT NULL AND sent_utc >= ?
                  AND status = 'sent' AND purpose NOT IN ({placeholders})""",
            (family_id, since, *self.exempt_purposes),
        )
        return int(row["c"]) if row else 0

    def check(
        self,
        family_id: str,
        at: datetime,
        purpose: str,
        *,
        tier: str = TIER_APPOINTMENT,
    ) -> CapDecision:
        if purpose in self.exempt_purposes:
            return CapDecision(True, 0, 0, tier, "transactional_exempt")
        limit = self.limit_for(tier)
        count = self.count(family_id, at)
        if count >= limit:
            return CapDecision(False, count, limit, tier, "frequency_cap")
        return CapDecision(True, count, limit, tier)


# --------------------------------------------------------------------------
# The send gate
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SendDecision:
    allow: bool
    status: str  # 'sendable' | 'blocked' | 'deferred' | 'skipped'
    reason: str = ""
    send_at: datetime | None = None


class SendGate:
    """The single funnel every outbound message passes through.

    Order is deliberate and is asserted by the test suite:

        1. suppression   - a STOP is absolute and is checked before anything
                           else, so that no later rule can accidentally
                           short-circuit past it.
        2. consent       - express consent for this channel and purpose, not
                           revoked. TCPA. A missing consent row blocks the
                           send; it does not warn.
        3. quiet hours   - defer, do not drop.
        4. frequency cap - skip, do not defer, because a deferred message that
                           re-enters the queue next week is a message the
                           family did not need.

    Steps 1 and 2 are legal gates and there is no bypass parameter for them.
    Steps 3 and 4 are courtesy gates and transactional messages skip them.
    """

    def __init__(
        self,
        db: Database,
        *,
        quiet_hours: QuietHours | None = None,
        frequency_cap: FrequencyCap | None = None,
    ) -> None:
        self.db = db
        self.quiet_hours = quiet_hours or QuietHours()
        self.frequency_cap = frequency_cap or FrequencyCap(db)

    def is_suppressed(self, family_id: str, channel: str) -> bool:
        row = self.db.one(
            "SELECT 1 AS hit FROM suppression WHERE family_id = ? AND channel = ?"
            " AND released_utc IS NULL LIMIT 1",
            (family_id, channel),
        )
        return row is not None

    def has_consent(
        self,
        family_id: str,
        channel: str,
        purpose: str = ConsentPurpose.REMINDERS,
        *,
        at: datetime | None = None,
    ) -> bool:
        """True only if an unrevoked consent row exists for channel+purpose.

        WHY `at` matters: a consent granted after the message was planned still
        counts at send time, and a consent revoked before send time does not.
        Evaluating against the send instant is what makes the log defensible.
        """
        params: list[Any] = [family_id, channel, purpose]
        sql = (
            "SELECT 1 AS hit FROM consent WHERE family_id = ? AND channel = ?"
            " AND purpose = ? AND revoked_utc IS NULL"
        )
        if at is not None:
            sql += " AND granted_utc <= ?"
            params.append(iso(at))
        return self.db.one(sql + " LIMIT 1", params) is not None

    def evaluate(
        self,
        *,
        family_id: str,
        channel: str,
        purpose: str,
        at: datetime,
        tier: str = FrequencyCap.TIER_APPOINTMENT,
        consent_purpose: str = ConsentPurpose.REMINDERS,
        deadline: datetime | None = None,
        allow_defer: bool = True,
    ) -> SendDecision:
        transactional = purpose in MessagePurpose.TRANSACTIONAL

        if self.is_suppressed(family_id, channel):
            return SendDecision(False, "blocked", "suppressed")

        if not self.has_consent(family_id, channel, consent_purpose, at=at):
            return SendDecision(False, "blocked", "no_consent")

        # A deadline that has already passed is a deadline, quiet hours or not.
        # Without this, a cron outage that leaves a T-2h reminder queued until
        # 10:00 sends it while the family is already in the waiting room.
        if deadline is not None and at >= deadline:
            return SendDecision(False, "skipped", "would_arrive_after_appointment")

        send_at = at
        if not transactional and self.quiet_hours.contains(at):
            if not allow_defer:
                return SendDecision(False, "skipped", "quiet_hours")
            send_at = self.quiet_hours.next_open(at)
            if deadline is not None and send_at >= deadline:
                # Pushing a T-2h reminder to 08:00 would land it after the
                # visit. Silence beats a reminder that arrives from the waiting
                # room.
                return SendDecision(False, "skipped", "would_arrive_after_appointment")
            return SendDecision(True, "deferred", "quiet_hours", send_at)

        decision = self.frequency_cap.check(family_id, send_at, purpose, tier=tier)
        if not decision.allowed:
            return SendDecision(False, "skipped", decision.reason)

        return SendDecision(True, "sendable", "", send_at)


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannedMessage:
    purpose: str
    template_id: str
    planned_utc: datetime
    appointment_id: str
    patient_id: str
    family_id: str
    channel: str = Channel.SMS
    skipped_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def skipped(self) -> bool:
        return bool(self.skipped_reason)


def plan_reminders(
    appointment: Mapping[str, Any],
    *,
    now: datetime,
    family_id: str,
    cadence: Mapping[str, Sequence[ReminderRule]] | None = None,
    channel: str = Channel.SMS,
) -> list[PlannedMessage]:
    """Compute reminder instants for one appointment.

    THE DST-CRITICAL LINE is `start - rule.offset`, performed on aware UTC
    instants. Forty-eight hours before a 09:20 CST appointment on the Monday
    after the fall-back transition is 10:20 CDT on the Saturday, not 09:20 —
    because that weekend contains 49 civil hours between those wall times.
    Subtracting a `timedelta` from an aware datetime is elapsed-time arithmetic
    and gets this right; doing the same subtraction on a naive local datetime
    and localising afterwards does not.
    """
    cadence = cadence or DEFAULT_CADENCE
    start = parse_iso(appointment["start_utc"])
    rules = cadence.get(appointment["visit_type"], ())
    planned: list[PlannedMessage] = []
    for rule in rules:
        send_at = start - rule.offset
        reason = ""
        if send_at <= now:
            # Booked inside the window. Not an error -- a sick visit booked at
            # 08:00 for 10:00 simply has no T-48h touch.
            reason = "offset_already_elapsed"
        planned.append(
            PlannedMessage(
                purpose=rule.purpose,
                template_id=rule.template_id,
                planned_utc=send_at,
                appointment_id=appointment["appointment_id"],
                patient_id=appointment["patient_id"],
                family_id=family_id,
                channel=channel,
                skipped_reason=reason,
                metadata={"defer_from_quiet_hours": rule.defer_from_quiet_hours},
            )
        )
    return planned


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


class ReminderEngine:
    """Plan-then-dispatch. The cron job is `dispatch_due`.

    Planning and dispatch are separate passes because the gates must be applied
    at dispatch time. A design that decided "allowed" at planning time would
    send to a family that opted out in between, which is the single most common
    TCPA finding in automated messaging.
    """

    INITIATIVE = "I-07"

    def __init__(
        self,
        db: Database,
        gateway: Any,
        *,
        gate: SendGate | None = None,
        cadence: Mapping[str, Sequence[ReminderRule]] | None = None,
        templates: Mapping[str, str] | None = None,
        audit: Any | None = None,
    ) -> None:
        self.db = db
        self.gateway = gateway
        self.gate = gate or SendGate(db)
        self.cadence = cadence or DEFAULT_CADENCE
        self.templates = dict(templates or TEMPLATES)
        self.audit = audit

    # -- planning ----------------------------------------------------------

    def plan_appointment(self, appointment_id: str, *, now: datetime) -> list[str]:
        """Queue this appointment's reminders. Idempotent per (appt, purpose)."""
        appt = self.db.one(
            """SELECT a.*, p.family_id, f.primary_phone, f.primary_email
               FROM appointment a
               JOIN patient p ON p.patient_id = a.patient_id
               JOIN family f ON f.family_id = p.family_id
               WHERE a.appointment_id = ?""",
            (appointment_id,),
        )
        if appt is None:
            raise KeyError(f"unknown appointment {appointment_id!r}")
        channel = Channel.SMS
        to_address = appt["primary_phone"]
        created: list[str] = []
        for planned in plan_reminders(
            appt, now=now, family_id=appt["family_id"], cadence=self.cadence, channel=channel
        ):
            existing = self.db.one(
                "SELECT message_id FROM message_log WHERE appointment_id = ? AND purpose = ?",
                (appointment_id, planned.purpose),
            )
            if existing:
                continue
            message_id = new_id("msg")
            self.db.execute(
                """INSERT INTO message_log (message_id, family_id, patient_id,
                       appointment_id, channel, purpose, template_id, planned_utc,
                       status, block_reason, to_address)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    message_id,
                    planned.family_id,
                    planned.patient_id,
                    appointment_id,
                    channel,
                    planned.purpose,
                    planned.template_id,
                    iso(planned.planned_utc),
                    "skipped" if planned.skipped else "planned",
                    planned.skipped_reason or None,
                    to_address,
                ),
            )
            created.append(message_id)
        return created

    def plan_horizon(self, *, now: datetime, horizon: timedelta = timedelta(days=10)) -> int:
        """Plan reminders for every upcoming un-planned appointment."""
        rows = self.db.all(
            """SELECT appointment_id FROM appointment
               WHERE status IN ('scheduled','confirmed')
                 AND start_utc > ? AND start_utc <= ?""",
            (iso(now), iso(now + horizon)),
        )
        total = 0
        for row in rows:
            total += len(self.plan_appointment(row["appointment_id"], now=now))
        return total

    # -- dispatch ----------------------------------------------------------

    def dispatch_due(self, *, now: datetime, limit: int = 500) -> dict[str, int]:
        """Send everything due. Returns a status histogram for the ops dashboard."""
        due = self.db.all(
            """SELECT m.*, pt.first_name, a.start_utc, a.visit_type, a.provider_id,
                      pr.display_name AS provider_name
               FROM message_log m
               JOIN patient pt ON pt.patient_id = m.patient_id
               LEFT JOIN appointment a ON a.appointment_id = m.appointment_id
               LEFT JOIN provider pr ON pr.provider_id = a.provider_id
               WHERE m.status = 'planned'
                 AND COALESCE(m.send_after_utc, m.planned_utc) <= ?
               ORDER BY COALESCE(m.send_after_utc, m.planned_utc) LIMIT ?""",
            (iso(now), limit),
        )
        histogram: dict[str, int] = {}
        for row in due:
            outcome = self._dispatch_one(row, now=now)
            histogram[outcome] = histogram.get(outcome, 0) + 1
        return histogram

    def _dispatch_one(self, row: Mapping[str, Any], *, now: datetime) -> str:
        appointment_start = parse_iso(row["start_utc"]) if row["start_utc"] else None

        # A cancelled appointment must not keep reminding. Checked here rather
        # than at cancellation time so that a cancellation arriving by any
        # path -- SMS, web, staff, EHR sync -- is honoured without every path
        # remembering to clean up the queue.
        appt_status = self.db.one(
            "SELECT status FROM appointment WHERE appointment_id = ?",
            (row["appointment_id"],),
        )
        if appt_status and appt_status["status"] not in ("scheduled", "confirmed"):
            self._finish(row["message_id"], status="skipped", reason="appointment_not_active")
            return "skipped"

        decision = self.gate.evaluate(
            family_id=row["family_id"],
            channel=row["channel"],
            purpose=row["purpose"],
            at=now,
            deadline=appointment_start,
        )
        if decision.status == "deferred" and decision.send_at is not None:
            # planned_utc is preserved: it is what the cadence rule decided, and
            # overwriting it destroys the record of when the touch was supposed
            # to happen (and silently moves the row between reporting windows).
            self.db.execute(
                "UPDATE message_log SET send_after_utc = ? WHERE message_id = ?",
                (iso(decision.send_at), row["message_id"]),
            )
            return "deferred"
        if not decision.allow:
            self._finish(row["message_id"], status=decision.status, reason=decision.reason)
            return decision.status

        body = self.render(row)
        receipt = self.gateway.send(
            to=row["to_address"],
            body=body,
            channel=row["channel"],
            purpose=row["purpose"],
            reference=row["message_id"],
        )
        if not receipt.accepted:
            self._finish(row["message_id"], status="failed", reason=receipt.error or "gateway")
            return "failed"
        self.db.execute(
            "UPDATE message_log SET status='sent', sent_utc=?, gateway_ref=?, block_reason=NULL"
            " WHERE message_id = ?",
            (iso(now), receipt.gateway_ref, row["message_id"]),
        )
        return "sent"

    def _finish(self, message_id: str, *, status: str, reason: str) -> None:
        self.db.execute(
            "UPDATE message_log SET status = ?, block_reason = ? WHERE message_id = ?",
            (status, reason, message_id),
        )

    # -- rendering ---------------------------------------------------------

    def render(self, row: Mapping[str, Any]) -> str:
        template = self.templates.get(row["template_id"])
        if template is None:
            raise KeyError(f"no template {row['template_id']!r}")
        start = parse_iso(row["start_utc"]) if row.get("start_utc") else None
        local = to_local(start) if start else None
        return template.format(
            first_name=row.get("first_name", ""),
            provider=row.get("provider_name") or "your provider",
            when=format_when(local) if local else "",
            local_time=(f"{(local.hour - 1) % 12 + 1}:{local:%M} {local:%p}" if local else ""),
            confirm_url=f"https://nsp.example/c/{row['message_id']}",
            cancel_url=f"https://nsp.example/x/{row['message_id']}",
            accept_url=f"https://nsp.example/a/{row.get('offer_id') or ''}",
            decline_url=f"https://nsp.example/d/{row.get('offer_id') or ''}",
        )
