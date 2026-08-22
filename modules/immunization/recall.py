"""Outbound recall queue: scoring, cadence, and sending. No model in this file.

WHY the recall engine imports `FrequencyCap` from the scheduling module rather
than defining its own:

README I-02's risk table asks for a "global frequency cap per family" and I-07's
asks for the cap to be shared "when I-02 and I-07 both send". Two modules each
politely limiting themselves to three messages a week is six messages a week to
the family, and the family does not care which internal system sent which. So
this module consults the same `FrequencyCap` object, on the `outreach` tier,
which is deliberately the lowest allowance: a week busy with appointment
reminders should squeeze out the recall message, not the reverse. A parent who
is already coming in on Thursday does not need a text about a gap the clinician
will raise in the room.

WHY sending is gated on a recorded validation run:

README I-02 lists "forecast logic error creating systematic false positives" as
a High severity risk, with the control "validate against 200 known-good records
before go-live". A control that lives in a runbook is a control that gets
skipped in week nine when the launch date slips. `RecallEngine.run()` raises
`RecallNotAuthorized` until a passing `ValidationResult` is supplied or the
forecast authority is the registry itself. Queue building and dry runs always
work, so the team can see exactly what *would* go out while validation is still
in progress.

WHY the cadence stops at three messages and hands off to a person:

README I-02: Day 0 SMS, Day 7 SMS, Day 21 email, Day 45 call list -- "because at
that point the reason is probably not 'they forgot'". A family that has ignored
three messages is telling you something that a fourth message will not address.
The escalation to a human is the point, not a fallback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence

from modules.scheduling.cadence import FrequencyCap, SendGate
from modules.scheduling.gateway import Gateway
from modules.scheduling.models import (
    Channel,
    ConsentPurpose,
    Database,
    iso,
    new_id,
    parse_iso,
    to_local,
)

from .forecast import AntigenForecast, PatientForecast, Schedule, Status, ValidationResult

__all__ = [
    "RECALL_SCHEMA",
    "RecallNotAuthorized",
    "RecallStep",
    "DEFAULT_CADENCE",
    "GapCandidate",
    "RecallEngine",
    "RECALL_CONSENT_BASIS",
    "TEMPLATES",
]

#: The documented determination README I-02's TCPA risk row asks for. Recall
#: messages about a vaccine a child is already due for under a schedule the
#: practice follows are informational health-care messages, not marketing: they
#: promote no product, offer no price, and are sent only to established patients
#: about their own care. That determination is recorded here, in code, next to
#: the thing it authorises -- and the consent purpose used is REMINDERS, never
#: MARKETING, so a family who declined marketing still receives them and a
#: family who revoked reminder consent does not.
RECALL_CONSENT_BASIS = (
    "informational treatment communication to an established patient; "
    "consent purpose = reminders; not marketing"
)


RECALL_SCHEMA = """
CREATE TABLE IF NOT EXISTS recall_gap (
    gap_id         TEXT PRIMARY KEY,
    patient_id     TEXT NOT NULL,
    family_id      TEXT NOT NULL,
    antigen        TEXT NOT NULL,
    opened_utc     TEXT NOT NULL,
    closed_utc     TEXT,
    close_reason   TEXT,
    last_urgency   REAL NOT NULL DEFAULT 0
);

-- Attempts are keyed to the PATIENT, not to a single antigen gap. A child who
-- is behind on Tdap, MenACWY and HPV gets one text listing all three, not three
-- texts -- the family experiences messages, not antigens, and three separate
-- texts about the same visit is the fastest route to a STOP. Because a
-- per-patient message covers every open gap, a per-patient cap of three in
-- ninety days is strictly stronger than README I-02's "max 3 per gap".
CREATE TABLE IF NOT EXISTS recall_attempt (
    attempt_id    TEXT PRIMARY KEY,
    patient_id    TEXT NOT NULL,
    step          INTEGER NOT NULL,
    channel       TEXT NOT NULL,
    antigens      TEXT NOT NULL,      -- JSON list, for the audit trail
    attempted_utc TEXT NOT NULL,
    outcome       TEXT NOT NULL,      -- sent|blocked|skipped|failed|queued_for_human
    reason        TEXT,
    message_id    TEXT
);

-- Physician one-click exclusion (README I-02 risk table: "a vaccine-hesitant
-- family receives an automated message and escalates"). Chart-level, honoured
-- before anything else, and never expires on its own.
CREATE TABLE IF NOT EXISTS recall_exclusion (
    exclusion_id  TEXT PRIMARY KEY,
    patient_id    TEXT,
    family_id     TEXT,
    scope         TEXT NOT NULL,      -- 'patient' | 'family'
    antigen       TEXT,               -- NULL = all antigens
    reason        TEXT NOT NULL,
    excluded_by   TEXT NOT NULL,
    created_utc   TEXT NOT NULL,
    released_utc  TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_recall_gap_open
    ON recall_gap(patient_id, antigen) WHERE closed_utc IS NULL;
CREATE INDEX IF NOT EXISTS ix_recall_attempt_patient ON recall_attempt(patient_id, attempted_utc);
CREATE INDEX IF NOT EXISTS ix_recall_exclusion ON recall_exclusion(patient_id, family_id);
"""


class RecallNotAuthorized(RuntimeError):
    """Raised when outbound sending is attempted before validation."""


@dataclass(frozen=True)
class RecallStep:
    day: int
    channel: str
    template_id: str | None    # None = hand off to a human call list
    label: str


#: README I-02, "Outreach cadence", verbatim in structure.
DEFAULT_CADENCE: tuple[RecallStep, ...] = (
    RecallStep(0, Channel.SMS, "recall_sms_1", "first SMS"),
    RecallStep(7, Channel.SMS, "recall_sms_2", "second SMS, different framing"),
    RecallStep(21, Channel.EMAIL, "recall_email", "email"),
    RecallStep(45, "call_list", None, "human call list"),
)

#: Physician-approved templates. The LLM in `messaging.py` may adjust register
#: and reading level; it may not change a clinical fact, and every slot it can
#: fill is supplied by the rules engine.
TEMPLATES: Mapping[str, str] = {
    "recall_sms_1": (
        "North Suburban Pediatrics: {first_name} is due for {vaccine_list}. "
        "Book here: {booking_url}. If {first_name} already had this somewhere "
        "else, reply and let us know so we can update the record. Reply STOP to opt out."
    ),
    "recall_sms_2": (
        "A quick follow-up from North Suburban Pediatrics - {first_name} still "
        "has {gap_count} vaccine(s) outstanding: {vaccine_list}. Most visits take "
        "15 minutes. {booking_url}  Already vaccinated elsewhere? Reply and tell "
        "us. Reply STOP to opt out."
    ),
    "recall_email": (
        "Hello,\n\nOur records show {first_name} is due for: {vaccine_list}.\n\n"
        "You can book at {booking_url}, or call 847-555-0100.\n\n"
        "If {first_name} received these elsewhere, reply to this email and we "
        "will update the chart.\n\nNorth Suburban Pediatrics"
    ),
}


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


@dataclass
class GapCandidate:
    patient_id: str
    family_id: str
    first_name: str
    dob: date
    antigen: str
    label: str
    status: str
    days_overdue: int
    weight: float
    school_required: bool
    urgency: float = 0.0
    breakdown: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "antigen": self.antigen,
            "label": self.label,
            "status": self.status,
            "days_overdue": self.days_overdue,
            "urgency": round(self.urgency, 2),
            "breakdown": {k: round(v, 2) for k, v in self.breakdown.items()},
            "school_required": self.school_required,
            "notes": list(self.notes),
        }


class RecallEngine:
    INITIATIVE = "I-02"

    #: Scoring constants. Exposed as class attributes rather than buried as
    #: literals so the practice can tune them and see in a diff what changed.
    #:
    #: OVERDUE_SATURATION_DAYS is the one that matters. README I-02's formula is
    #: `days_overdue x antigen_weight + bonuses`, and taken literally the first
    #: term is unbounded: a 16-year-old who never had MMR is ~5,300 days overdue,
    #: scores ~26,000, and permanently outranks an infant whose rotavirus window
    #: shuts in nine days. That is the wrong queue. Overdue-ness also stops
    #: carrying information after a while -- a gap open for three years is not
    #: three times more urgent than one open for one year, it is a gap the
    #: previous outreach did not close. So the term saturates at a year, and the
    #: bonuses are scaled to be able to compete with it.
    OVERDUE_SATURATION_DAYS = 365
    SCHOOL_WINDOW_DAYS = 75
    #: Days after the school start date that a gap stays at peak urgency.
    SCHOOL_GRACE_DAYS = 30
    SCHOOL_BONUS_PER_DAY = 8.0
    SCHOOL_CHECKPOINT_MULTIPLIER = 2.5
    AGE_OUT_WINDOW_DAYS = 90
    AGE_OUT_BONUS_PER_DAY = 20.0
    MAX_MESSAGES_PER_PATIENT_PER_90_DAYS = 3
    #: Minimum spacing between two recall messages to the same family, whatever
    #: the cadence table says. A stale anchor must not collapse the remaining
    #: steps into consecutive nights.
    MIN_DAYS_BETWEEN_MESSAGES = 5
    #: A step that fails at the gateway or is blocked gets this many retries,
    #: this far apart, before the cadence moves on without it.
    MAX_RETRIES_PER_STEP = 3
    RETRY_BACKOFF_DAYS = 2
    #: Close reasons that mean a dose was actually given. `not_yet_due` counts:
    #: dose two of three was administered and the third is simply not due yet.
    CONVERSION_CLOSE_REASONS = ("complete", "not_yet_due")

    def __init__(
        self,
        db: Database,
        gateway: Gateway,
        *,
        schedule: Schedule | None = None,
        gate: SendGate | None = None,
        frequency_cap: FrequencyCap | None = None,
        cadence: Sequence[RecallStep] = DEFAULT_CADENCE,
        templates: Mapping[str, str] | None = None,
        validation: ValidationResult | None = None,
        registry_is_authority: bool = False,
        booking_url: str = "https://nsp.example/book",
        audit: Any = None,
        drafter: Any = None,
    ) -> None:
        self.db = db
        self.gateway = gateway
        self.schedule = schedule or Schedule.load()
        self.frequency_cap = frequency_cap or FrequencyCap(db)
        self.gate = gate or SendGate(db, frequency_cap=self.frequency_cap)
        self.cadence = tuple(cadence)
        self.templates = dict(templates or TEMPLATES)
        self.validation = validation
        self.registry_is_authority = registry_is_authority
        self.booking_url = booking_url
        self.audit = audit
        #: Optional `messaging.MessageDrafter`. Absent means templates are sent
        #: verbatim, which is a perfectly good production configuration.
        self.drafter = drafter
        self.db.conn.executescript(RECALL_SCHEMA)

    # -- authorisation -----------------------------------------------------

    @property
    def authorized(self) -> bool:
        if self.registry_is_authority:
            return True
        return self.validation is not None and self.validation.passes()

    def authorize(self, validation: ValidationResult) -> None:
        self.validation = validation

    def _require_authorization(self) -> None:
        if self.authorized:
            return
        detail = (
            self.validation.summary()
            if self.validation is not None
            else "no validation run recorded"
        )
        raise RecallNotAuthorized(
            "outbound recall is not authorised: the forecast engine has not been "
            f"validated against known-good records ({detail}). Build the queue and "
            "dry-run it as much as you like; sending needs the validation first "
            "(README I-02, forecast-error risk control)."
        )

    # -- queue -------------------------------------------------------------

    def build_queue(
        self,
        forecasts: Iterable[PatientForecast],
        *,
        as_of: date,
        patients: Mapping[str, Mapping[str, Any]],
        require_no_upcoming_appointment: bool = True,
    ) -> list[GapCandidate]:
        """Rank open gaps for patients with no upcoming visit.

        `patients` maps patient_id -> {family_id, first_name, dob}. It is passed
        in rather than queried so that this function is pure and testable, and
        so the panel extract stays the caller's concern.

        Gaps are excluded, not down-ranked, when: the antigen is not
        recall-eligible (MenB shared decision-making, COVID), the forecast is
        contested or held for review, the patient or family carries a physician
        exclusion, or the child already has an upcoming appointment -- the
        huddle sheet will surface it in the room, which is a better
        conversation than a text.
        """
        candidates: list[GapCandidate] = []
        for forecast in forecasts:
            info = patients.get(forecast.patient_id)
            if info is None:
                continue
            if require_no_upcoming_appointment and self._has_upcoming_appointment(
                forecast.patient_id, as_of
            ):
                continue
            for antigen, gap in sorted(forecast.antigens.items()):
                if not gap.is_open_gap or not gap.recall_eligible:
                    continue
                if self._is_excluded(forecast.patient_id, info["family_id"], antigen):
                    continue
                candidate = self._score(gap, forecast, info, as_of)
                candidates.append(candidate)
        candidates.sort(key=lambda c: (-c.urgency, c.patient_id, c.antigen))
        return candidates

    def _score(
        self,
        gap: AntigenForecast,
        forecast: PatientForecast,
        info: Mapping[str, Any],
        as_of: date,
    ) -> GapCandidate:
        """urgency = saturated days_overdue x weight + school bonus + age-out bonus.

        README I-02's formula, with the first term saturated -- see
        OVERDUE_SATURATION_DAYS for why the literal version produces the wrong
        queue. The breakdown is retained on the candidate because an
        unexplainable clinical priority queue does not get used: the MA working
        it needs to see why an eight-year-old's MMR outranks a teenager's HPV
        today and the reverse next month.
        """
        dob: date = info["dob"]
        base = min(gap.days_overdue, self.OVERDUE_SATURATION_DAYS) * gap.weight
        school = self._school_bonus(gap, dob, as_of)
        age_out = self._age_out_bonus(gap, dob, as_of)
        candidate = GapCandidate(
            patient_id=forecast.patient_id,
            family_id=str(info["family_id"]),
            first_name=str(info.get("first_name", "")),
            dob=dob,
            antigen=gap.antigen,
            label=gap.label,
            status=gap.status,
            days_overdue=gap.days_overdue,
            weight=gap.weight,
            school_required=gap.school_required,
            urgency=base + school + age_out,
            breakdown={
                "overdue_x_weight_saturated": base,
                "school_deadline_bonus": school,
                "age_out_bonus": age_out,
            },
            notes=list(gap.notes),
        )
        return candidate

    def _school_bonus(self, gap: AntigenForecast, dob: date, as_of: date) -> float:
        """Urgency that rises into the August deadline and stays high just past it.

        WHY the window extends past the start date: a child who is still
        non-compliant on 22 August is not in a future crisis, they are in a
        present one -- Illinois schools exclude them. Rolling straight to next
        year's deadline the moment school starts would drop those children off
        the queue at exactly the moment they matter most.
        """
        if not gap.school_required:
            return 0.0
        candidates = [
            date(year, self.schedule.school_month, self.schedule.school_day)
            for year in (as_of.year, as_of.year + 1)
        ]
        best = 0.0
        for start in candidates:
            days_until = (start - as_of).days
            if days_until > self.SCHOOL_WINDOW_DAYS or days_until < -self.SCHOOL_GRACE_DAYS:
                continue
            if days_until >= 0:
                bonus = (self.SCHOOL_WINDOW_DAYS - days_until) * self.SCHOOL_BONUS_PER_DAY
            else:
                # Past the deadline and still not compliant: hold at the peak.
                bonus = self.SCHOOL_WINDOW_DAYS * self.SCHOOL_BONUS_PER_DAY
            age_at_start = start.year - dob.year - (
                (start.month, start.day) < (dob.month, dob.day)
            )
            checkpoints = {
                int(c["typical_age_years"]) for c in self.schedule.school_checkpoints
            }
            if age_at_start in checkpoints:
                bonus *= self.SCHOOL_CHECKPOINT_MULTIPLIER
            best = max(best, bonus)
        return best

    def _age_out_bonus(self, gap: AntigenForecast, dob: date, as_of: date) -> float:
        """Urgency that rises as an irreversible deadline approaches.

        Rotavirus is the extreme case: after eight months the dose can never be
        given, so a recall that lands a week late has zero value rather than
        reduced value. HPV's threshold is softer but real -- crossing the 15th
        birthday turns a two-dose series into a three-dose one, which is an
        extra visit and an extra injection for the child.
        """
        rule = self.schedule.rule(gap.antigen) or {}
        age_out = rule.get("age_out") or {}
        deadlines: list[date] = []
        if gap.age_out_hard_date:
            deadlines.append(gap.age_out_hard_date)
        change = age_out.get("series_change_age")
        if change:
            from .forecast import add_period

            deadlines.append(add_period(dob, change))
        deadlines = [d for d in deadlines if d > as_of]
        if not deadlines:
            return 0.0
        days_remaining = (min(deadlines) - as_of).days
        if days_remaining > self.AGE_OUT_WINDOW_DAYS:
            return 0.0
        return (self.AGE_OUT_WINDOW_DAYS - days_remaining) * self.AGE_OUT_BONUS_PER_DAY

    # -- exclusions and eligibility ---------------------------------------

    def exclude(
        self,
        *,
        reason: str,
        excluded_by: str,
        patient_id: str | None = None,
        family_id: str | None = None,
        antigen: str | None = None,
        now: datetime | None = None,
    ) -> str:
        """The physician's one-click exclusion. Requires a reason and a name."""
        if not patient_id and not family_id:
            raise ValueError("exclusion needs a patient_id or a family_id")
        if not reason.strip() or not excluded_by.strip():
            raise ValueError("an exclusion must record who made it and why")
        exclusion_id = new_id("exc")
        self.db.execute(
            """INSERT INTO recall_exclusion (exclusion_id, patient_id, family_id, scope,
                   antigen, reason, excluded_by, created_utc)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                exclusion_id,
                patient_id,
                family_id,
                "patient" if patient_id else "family",
                antigen,
                reason,
                excluded_by,
                iso(now or datetime.now(timezone.utc)),
            ),
        )
        if self.audit is not None:
            self.audit.record_event(
                actor_id=excluded_by,
                initiative_id=self.INITIATIVE,
                event_type="recall_exclusion_created",
                patient_id=patient_id,
                detail={"scope": "patient" if patient_id else "family", "antigen": antigen},
            )
        return exclusion_id

    def _is_excluded(self, patient_id: str, family_id: str, antigen: str) -> bool:
        row = self.db.one(
            """SELECT 1 AS hit FROM recall_exclusion
               WHERE released_utc IS NULL
                 AND (antigen IS NULL OR antigen = ?)
                 AND ((patient_id = ?) OR (family_id = ?))
               LIMIT 1""",
            (antigen, patient_id, family_id),
        )
        return row is not None

    def _has_upcoming_appointment(self, patient_id: str, as_of: date) -> bool:
        row = self.db.one(
            """SELECT 1 AS hit FROM appointment
               WHERE patient_id = ? AND status IN ('scheduled','confirmed')
                 AND start_utc >= ? LIMIT 1""",
            (patient_id, f"{as_of.isoformat()}T00:00:00+00:00"),
        )
        return row is not None

    # -- gap lifecycle -----------------------------------------------------

    def open_gap(self, candidate: GapCandidate, *, now: datetime) -> str:
        existing = self.db.one(
            "SELECT gap_id FROM recall_gap WHERE patient_id = ? AND antigen = ?"
            " AND closed_utc IS NULL",
            (candidate.patient_id, candidate.antigen),
        )
        if existing:
            self.db.execute(
                "UPDATE recall_gap SET last_urgency = ? WHERE gap_id = ?",
                (candidate.urgency, existing["gap_id"]),
            )
            return existing["gap_id"]
        gap_id = new_id("gap")
        self.db.execute(
            """INSERT INTO recall_gap (gap_id, patient_id, family_id, antigen,
                   opened_utc, last_urgency)
               VALUES (?,?,?,?,?,?)""",
            (
                gap_id,
                candidate.patient_id,
                candidate.family_id,
                candidate.antigen,
                iso(now),
                candidate.urgency,
            ),
        )
        return gap_id

    def close_resolved_gaps(
        self, forecasts: Iterable[PatientForecast], *, now: datetime
    ) -> int:
        """Close gaps the forecaster no longer reports. Idempotent.

        Run this BEFORE sending on every cycle. A child vaccinated yesterday at
        a pharmacy, reconciled overnight, must not get this morning's recall --
        README I-02's first risk row.
        """
        closed = 0
        for forecast in forecasts:
            open_rows = self.db.all(
                "SELECT gap_id, antigen FROM recall_gap WHERE patient_id = ?"
                " AND closed_utc IS NULL",
                (forecast.patient_id,),
            )
            for row in open_rows:
                antigen = forecast.antigens.get(row["antigen"])
                if antigen is not None and antigen.is_open_gap:
                    continue
                reason = "resolved" if antigen is None else antigen.status
                self.db.execute(
                    "UPDATE recall_gap SET closed_utc = ?, close_reason = ? WHERE gap_id = ?",
                    (iso(now), reason, row["gap_id"]),
                )
                closed += 1
        return closed

    def attempts_for(
        self, patient_id: str, *, since: datetime | None = None
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM recall_attempt WHERE patient_id = ? AND outcome = 'sent'"
        params: list[Any] = [patient_id]
        if since is not None:
            sql += " AND attempted_utc >= ?"
            params.append(iso(since))
        return self.db.all(sql + " ORDER BY attempted_utc", params)

    def next_step(self, patient_id: str, *, now: datetime) -> RecallStep | None:
        """The next cadence step due for this patient, or None.

        Anchored on the patient's OLDEST still-open gap, and scoped to THAT
        EPISODE. Three properties this has to have, each of which was wrong in
        an obvious-looking earlier version:

        1. **Episode-scoped, not lifetime.** Counting every attempt this patient
           has ever received means that once one cadence completes, `next_step`
           returns None forever and the patient is silently never contacted
           again -- for any future gap, for the rest of their childhood.
           Attempts are counted only from the current anchor onwards.
        2. **Spaced against the last send, not only the anchor.** With a stale
           anchor every remaining step becomes eligible at once, so a family who
           was excluded from the queue for a month gets the second SMS, the
           email and the call-list handoff on three consecutive nights.
        3. **Bounded retries.** A failing gateway must not re-attempt step 0
           every night forever, which it will if only successful sends count.
        """
        row = self.db.one(
            "SELECT MIN(opened_utc) AS opened FROM recall_gap WHERE patient_id = ?"
            " AND closed_utc IS NULL",
            (patient_id,),
        )
        if row is None or not row["opened"]:
            return None
        opened = parse_iso(row["opened"])

        episode = self.db.all(
            "SELECT step, outcome, attempted_utc FROM recall_attempt"
            " WHERE patient_id = ? AND attempted_utc >= ?"
            " ORDER BY attempted_utc",
            (patient_id, iso(opened)),
        )
        done = {
            int(r["step"]) for r in episode
            if r["outcome"] in ("sent", "queued_for_human")
        }
        sends = [parse_iso(r["attempted_utc"]) for r in episode if r["outcome"] == "sent"]
        if sends and now - max(sends) < timedelta(days=self.MIN_DAYS_BETWEEN_MESSAGES):
            return None

        for index, step in enumerate(self.cadence):
            if index in done:
                continue
            if now < opened + timedelta(days=step.day):
                return None
            retries = [
                parse_iso(r["attempted_utc"]) for r in episode
                if int(r["step"]) == index and r["outcome"] in ("failed", "blocked")
            ]
            if len(retries) >= self.MAX_RETRIES_PER_STEP:
                continue
            if retries and now - max(retries) < timedelta(days=self.RETRY_BACKOFF_DAYS):
                return None
            return step
        return None

    def _step_index(self, step: RecallStep) -> int:
        return self.cadence.index(step)

    # -- sending -----------------------------------------------------------

    def run(
        self,
        forecasts: Sequence[PatientForecast],
        *,
        now: datetime,
        patients: Mapping[str, Mapping[str, Any]],
        limit: int = 500,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """One nightly cycle: close resolved gaps, rank, open, send what is due.

        Candidates are ranked per antigen -- that is what makes the queue
        explainable -- and then GROUPED PER PATIENT for sending. A family behind
        on three vaccines receives one message naming all three.

        `dry_run=True` does everything except send and is always permitted; that
        is how the practice inspects the queue during the validation period.
        """
        if not dry_run:
            self._require_authorization()

        as_of = to_local(now).date()
        closed = 0 if dry_run else self.close_resolved_gaps(forecasts, now=now)
        queue = self.build_queue(forecasts, as_of=as_of, patients=patients)

        grouped: dict[str, list[GapCandidate]] = {}
        for candidate in queue:
            grouped.setdefault(candidate.patient_id, []).append(candidate)
        # Patients in descending order of their most urgent gap.
        ordered = sorted(
            grouped.items(),
            key=lambda kv: (-max(c.urgency for c in kv[1]), kv[0]),
        )

        histogram: dict[str, int] = {}
        actions: list[dict[str, Any]] = []
        for patient_id, candidates in ordered[:limit]:
            top = max(candidates, key=lambda c: c.urgency)
            if dry_run:
                # A dry run must not mutate. Opening a gap row sets the cadence
                # anchor, so a "harmless" inspection would silently consume the
                # day-0 slot and the first real message would go out as the
                # second. Inspection has to be free.
                histogram["would_send"] = histogram.get("would_send", 0) + 1
                actions.append({
                    "patient_id": patient_id,
                    "antigens": [c.antigen for c in candidates],
                    "step": self.cadence[0].label,
                    "urgency": round(top.urgency, 2),
                })
                continue
            for candidate in candidates:
                self.open_gap(candidate, now=now)
            step = self.next_step(patient_id, now=now)
            if step is None:
                histogram["not_due"] = histogram.get("not_due", 0) + 1
                continue
            outcome = self._attempt(patient_id, candidates, step, now=now)
            histogram[outcome] = histogram.get(outcome, 0) + 1
            actions.append({
                "patient_id": patient_id,
                "antigens": [c.antigen for c in candidates],
                "step": step.label,
                "outcome": outcome,
            })
        return {
            "gaps_closed": closed,
            "queue_size": len(queue),
            "patients_in_queue": len(grouped),
            "outcomes": histogram,
            "actions": actions,
            "authorized": self.authorized,
            "dry_run": dry_run,
        }

    def _attempt(
        self,
        patient_id: str,
        candidates: Sequence[GapCandidate],
        step: RecallStep,
        *,
        now: datetime,
    ) -> str:
        index = self._step_index(step)
        antigens = [c.antigen for c in candidates]
        primary = max(candidates, key=lambda c: c.urgency)

        if step.template_id is None:
            self._record_attempt(patient_id, index, step.channel, antigens, now,
                                 "queued_for_human",
                                 "three messages sent without a response")
            return "queued_for_human"

        # Hard cap independent of the cadence table. WHY belt and braces: the
        # cadence is configuration and configuration gets edited.
        recent = self.attempts_for(patient_id, since=now - timedelta(days=90))
        if len(recent) >= self.MAX_MESSAGES_PER_PATIENT_PER_90_DAYS:
            self._record_attempt(patient_id, index, step.channel, antigens, now,
                                 "skipped", "patient message cap reached")
            return "skipped"

        family = self.db.one(
            "SELECT * FROM family WHERE family_id = ?", (primary.family_id,)
        )
        if family is None:
            self._record_attempt(patient_id, index, step.channel, antigens, now,
                                 "skipped", "family record not found")
            return "skipped"
        to_address = (
            family["primary_email"] if step.channel == Channel.EMAIL
            else family["primary_phone"]
        )
        if not to_address:
            self._record_attempt(patient_id, index, step.channel, antigens, now,
                                 "skipped", f"no {step.channel} address on file")
            return "skipped"

        decision = self.gate.evaluate(
            family_id=primary.family_id,
            channel=step.channel,
            purpose="recall_immunization",
            at=now,
            tier=FrequencyCap.TIER_OUTREACH,
            consent_purpose=ConsentPurpose.REMINDERS,
            allow_defer=True,
        )
        # `deferred` comes back with allow=True and a later send_at, so it must
        # be checked BEFORE the allow flag. The nightly batch runs before the
        # practice opens; a recall that is merely too early is not a failed
        # attempt, and recording one would fill the audit trail with nightly
        # noise that reads like outreach which happened. The next cron tick
        # after 08:00 picks the step up untouched.
        if decision.status == "deferred":
            return "deferred"
        if not decision.allow:
            self._record_attempt(patient_id, index, step.channel, antigens, now,
                                 decision.status, decision.reason)
            return decision.status

        body = self._render(step, candidates, primary)
        message_id = new_id("msg")
        self.db.execute(
            """INSERT INTO message_log (message_id, family_id, patient_id, channel,
                   purpose, template_id, planned_utc, status, to_address)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                message_id,
                primary.family_id,
                patient_id,
                step.channel,
                "recall_immunization",
                step.template_id,
                iso(now),
                "planned",
                to_address,
            ),
        )
        receipt = self.gateway.send(
            to=to_address,
            body=body,
            channel=step.channel,
            purpose="recall_immunization",
            reference=message_id,
        )
        if not receipt.accepted:
            self.db.execute(
                "UPDATE message_log SET status='failed', block_reason=? WHERE message_id=?",
                (receipt.error or "gateway", message_id),
            )
            self._record_attempt(patient_id, index, step.channel, antigens, now,
                                 "failed", receipt.error, message_id)
            return "failed"
        self.db.execute(
            "UPDATE message_log SET status='sent', sent_utc=?, gateway_ref=?"
            " WHERE message_id=?",
            (iso(now), receipt.gateway_ref, message_id),
        )
        self._record_attempt(patient_id, index, step.channel, antigens, now, "sent",
                             None, message_id)
        return "sent"

    def _render(
        self,
        step: RecallStep,
        candidates: Sequence[GapCandidate],
        primary: GapCandidate,
    ) -> str:
        labels = sorted({c.label for c in candidates})
        slots = {
            "first_name": primary.first_name,
            "vaccine_list": ", ".join(labels),
            "gap_count": len(labels),
            "booking_url": self.booking_url,
        }
        template = self.templates[step.template_id or ""]
        if self.drafter is None:
            return template.format(**slots)
        return self.drafter.draft(
            template_id=step.template_id or "",
            template=template,
            slots=slots,
            patient_id=primary.patient_id,
            channel=step.channel,
        )

    def _record_attempt(
        self,
        patient_id: str,
        step_index: int,
        channel: str,
        antigens: Sequence[str],
        now: datetime,
        outcome: str,
        reason: str | None = None,
        message_id: str | None = None,
    ) -> None:
        self.db.execute(
            """INSERT INTO recall_attempt (attempt_id, patient_id, step, channel,
                   antigens, attempted_utc, outcome, reason, message_id)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (new_id("att"), patient_id, step_index, channel, json.dumps(list(antigens)),
             iso(now), outcome, reason, message_id),
        )

    # -- reporting ---------------------------------------------------------

    def call_list(self, *, now: datetime) -> list[dict[str, Any]]:
        """Gaps that exhausted the message cadence and need a person.

        README I-02: at day 45 "the reason is probably not 'they forgot'."
        """
        return self.db.all(
            """SELECT g.patient_id, g.family_id,
                      GROUP_CONCAT(g.antigen) AS antigens,
                      MIN(g.opened_utc) AS opened_utc,
                      MAX(g.last_urgency) AS urgency
               FROM recall_gap g
               WHERE g.closed_utc IS NULL
                 AND g.patient_id IN (
                       SELECT patient_id FROM recall_attempt
                       WHERE outcome = 'queued_for_human'
                 )
               GROUP BY g.patient_id
               ORDER BY urgency DESC""",
        )

    def conversion_report(self, *, since: datetime | None = None) -> dict[str, Any]:
        """README 10.2: recall message -> visit conversion, target > 30%.

        Two things this has to get right, both of which a naive join gets wrong:

        * An attempt counts for a gap only if it happened DURING that gap's open
          episode. Joining on patient_id alone credits every gap the patient has
          ever had to a single message.
        * A conversion is a gap that closed because a dose was given. That
          shows up as any close reason meaning the antigen moved forward --
          `complete` for a finished series, but also `not_yet_due` for dose two
          of three. Requiring `complete` alone misses most real conversions.
        """
        clause = " AND g.opened_utc >= ?" if since is not None else ""
        params: list[Any] = [iso(since)] if since is not None else []
        rows = self.db.all(
            f"""SELECT g.gap_id, g.close_reason,
                       (SELECT COUNT(*) FROM recall_attempt a
                         WHERE a.patient_id = g.patient_id
                           AND a.outcome = 'sent'
                           AND a.attempted_utc >= g.opened_utc
                           AND (g.closed_utc IS NULL OR a.attempted_utc <= g.closed_utc)
                       ) AS sends
                FROM recall_gap g WHERE 1=1{clause}""",
            params,
        )
        contacted = [r for r in rows if r["sends"]]
        converted = [
            r for r in contacted if r["close_reason"] in self.CONVERSION_CLOSE_REASONS
        ]
        total = len(contacted)
        return {
            "gaps_contacted": total,
            "gaps_closed_by_vaccination": len(converted),
            "conversion": (len(converted) / total) if total else None,
            "target": 0.30,
        }
