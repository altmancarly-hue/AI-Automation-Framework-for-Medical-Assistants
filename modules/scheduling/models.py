"""Storage layer for I-07 — no-show reduction and waitlist backfill.

WHY SQLite and not the EHR:

The EHR's scheduling API is the system of record for appointments. This module
is the system of record for everything the EHR does not model: consent
evidence, message history, the waitlist, suppression, and the offer/accept
protocol that makes backfill atomic. Appointments are mirrored here because
backfill needs a row it can lock, and you cannot take a row lock over a REST
API. Reconciliation back to FHIR Appointment is the integration boundary; it is
deliberately not this module's job.

WHY every timestamp is UTC ISO-8601 with an explicit offset:

Local-time storage plus DST equals silent, seasonal, off-by-one-hour bugs that
appear twice a year and are attributed to "the carrier." Instants are stored in
UTC; wall-clock reasoning (quiet hours, "same day") happens at the edges, in
America/Chicago, and only there.

WHY consent is its own table with capture method and revocation:

TCPA (README R-10) is an evidentiary problem, not a technical one. When a
demand letter arrives, the practice has to produce *when* consent was captured,
*how*, for *which channel*, and for *which purpose* — reminders and marketing
are separate consents and conflating them is the classic finding. Revocation is
a row update on this table plus a suppression insert, and both are timestamped.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from calendar import monthrange
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterator, Mapping, Sequence
from zoneinfo import ZoneInfo

__all__ = [
    "PRACTICE_TZ",
    "Database",
    "AppointmentStatus",
    "VisitType",
    "Channel",
    "ConsentPurpose",
    "MessagePurpose",
    "OfferOutcome",
    "iso",
    "parse_iso",
    "to_local",
    "age_months",
    "new_id",
    "seed_practice_defaults",
    "address_for",
]

# The practice is in Buffalo Grove, Illinois. One timezone, hard-coded, because
# a configurable timezone in a single-site deployment is a bug generator with
# no user.
PRACTICE_TZ = ZoneInfo("America/Chicago")


class AppointmentStatus:
    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    RELEASED = "released"  # cancelled and handed to the backfill engine
    COMPLETED = "completed"
    NO_SHOW = "no_show"
    ALL = (SCHEDULED, CONFIRMED, CANCELLED, RELEASED, COMPLETED, NO_SHOW)


class VisitType:
    WELL = "well"
    SICK = "sick"
    FOLLOW_UP = "follow_up"
    PROCEDURE = "procedure"
    ALL = (WELL, SICK, FOLLOW_UP, PROCEDURE)
    # Only well visits may be self-scheduled (README I-07 risk table): a parent
    # should not self-select a sick slot without a triage conversation.
    SELF_SCHEDULABLE = (WELL,)


class Channel:
    SMS = "sms"
    VOICE = "voice"
    EMAIL = "email"
    ALL = (SMS, VOICE, EMAIL)


class ConsentPurpose:
    """Reminders and marketing are separate consents. Never conflate them."""

    REMINDERS = "reminders"
    MARKETING = "marketing"
    ALL = (REMINDERS, MARKETING)


class MessagePurpose:
    REMINDER_T7 = "reminder_t7"
    REMINDER_T48 = "reminder_t48"
    REMINDER_T2 = "reminder_t2"
    BACKFILL_OFFER = "backfill_offer"
    BACKFILL_LOST = "backfill_lost"
    BACKFILL_WON = "backfill_won"
    CANCEL_CONFIRMATION = "cancel_confirmation"
    OPTOUT_CONFIRMATION = "optout_confirmation"
    ALL = (
        REMINDER_T7,
        REMINDER_T48,
        REMINDER_T2,
        BACKFILL_OFFER,
        BACKFILL_LOST,
        BACKFILL_WON,
        CANCEL_CONFIRMATION,
        OPTOUT_CONFIRMATION,
    )
    # Replies to something the family just did. Exempt from the weekly
    # frequency cap and from quiet hours: a parent who taps "cancel" at 22:40
    # is owed an immediate confirmation, and withholding it to protect a
    # message budget produces a phone call at 09:00 the next morning.
    TRANSACTIONAL = (
        CANCEL_CONFIRMATION,
        OPTOUT_CONFIRMATION,
        BACKFILL_WON,
        BACKFILL_LOST,
    )


class OfferOutcome:
    PENDING = "pending"
    ACCEPTED = "accepted"
    LOST = "lost"  # someone else took the slot first
    DECLINED = "declined"
    EXPIRED = "expired"
    ALL = (PENDING, ACCEPTED, LOST, DECLINED, EXPIRED)


# --------------------------------------------------------------------------
# Time helpers
# --------------------------------------------------------------------------


def iso(dt: datetime) -> str:
    """Serialise an aware datetime to a lexicographically sortable UTC string.

    WHY normalise to UTC first: SQLite compares these as text. Mixing offsets
    would make '2026-11-02T09:20:00-06:00' sort before
    '2026-11-02T15:20:00+00:00' despite being the same instant.
    """
    if dt.tzinfo is None:
        raise ValueError("naive datetime rejected; attach a timezone")
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_iso(text: str) -> datetime:
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        raise ValueError(f"stored timestamp {text!r} has no offset")
    return dt.astimezone(timezone.utc)


def to_local(dt: datetime, tz: ZoneInfo = PRACTICE_TZ) -> datetime:
    """Convert an instant to practice-local wall time."""
    if dt.tzinfo is None:
        raise ValueError("naive datetime rejected; attach a timezone")
    return dt.astimezone(tz)


def age_months(dob: date | str, at: datetime) -> int:
    """Whole months of age at an instant, evaluated in practice-local time.

    WHY local: a child born on the 1st does not turn 24 months at 18:00 the
    previous day because the server is in UTC. Age gates schedule eligibility,
    so it is computed the way a human at the front desk would compute it.
    """
    if isinstance(dob, str):
        dob = date.fromisoformat(dob)
    local = to_local(at).date()
    months = (local.year - dob.year) * 12 + (local.month - dob.month)
    # A child born on the 31st has their monthly anniversary on the last day of
    # a shorter month. Without the clamp, someone born 2024-08-31 is "0 months"
    # on 2024-09-30 and is excluded from an age-gated panel for a day or three
    # every month -- a bug that only ever bites children born on the 29th-31st.
    days_in_month = monthrange(local.year, local.month)[1]
    anniversary_day = min(dob.day, days_in_month)
    if local.day < anniversary_day:
        months -= 1
    return max(0, months)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS family (
    family_id     TEXT PRIMARY KEY,
    display_name  TEXT NOT NULL,
    primary_phone TEXT NOT NULL,
    primary_email TEXT,
    locale        TEXT NOT NULL DEFAULT 'en'
);

CREATE TABLE IF NOT EXISTS patient (
    patient_id  TEXT PRIMARY KEY,
    family_id   TEXT NOT NULL REFERENCES family(family_id),
    first_name  TEXT NOT NULL,
    last_name   TEXT NOT NULL,
    dob         TEXT NOT NULL,               -- ISO date
    active      INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS provider (
    provider_id      TEXT PRIMARY KEY,
    display_name     TEXT NOT NULL,
    min_age_months   INTEGER NOT NULL DEFAULT 0,
    max_age_months   INTEGER NOT NULL DEFAULT 300,
    visit_types_csv  TEXT NOT NULL DEFAULT 'well,sick,follow_up,procedure',
    active           INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS appointment (
    appointment_id     TEXT PRIMARY KEY,
    patient_id         TEXT NOT NULL REFERENCES patient(patient_id),
    provider_id        TEXT NOT NULL REFERENCES provider(provider_id),
    visit_type         TEXT NOT NULL,
    start_utc          TEXT NOT NULL,
    duration_minutes   INTEGER NOT NULL,
    status             TEXT NOT NULL,
    created_utc        TEXT NOT NULL,
    confirmed_utc      TEXT,
    cancelled_utc      TEXT,
    cancel_source      TEXT,                 -- 'sms' | 'web' | 'phone' | 'staff'
    completed_utc      TEXT,
    booking_source     TEXT NOT NULL DEFAULT 'staff',
    filled_from_release TEXT,                -- slot_release.release_id
    expected_revenue   REAL NOT NULL DEFAULT 140.0
);

CREATE TABLE IF NOT EXISTS waitlist_entry (
    entry_id          TEXT PRIMARY KEY,
    patient_id        TEXT NOT NULL REFERENCES patient(patient_id),
    desired_provider  TEXT,                  -- NULL = any provider
    visit_type        TEXT NOT NULL,
    earliest_ok_utc   TEXT NOT NULL,
    latest_ok_utc     TEXT NOT NULL,
    priority          INTEGER NOT NULL DEFAULT 0,
    added_utc         TEXT NOT NULL,
    notify_channel    TEXT NOT NULL DEFAULT 'sms',
    status            TEXT NOT NULL DEFAULT 'active',   -- active|booked|withdrawn|expired
    booked_appointment TEXT
);

CREATE TABLE IF NOT EXISTS slot_release (
    release_id           TEXT PRIMARY KEY,
    appointment_id       TEXT NOT NULL REFERENCES appointment(appointment_id),
    provider_id          TEXT NOT NULL,
    visit_type           TEXT NOT NULL,
    start_utc            TEXT NOT NULL,
    duration_minutes     INTEGER NOT NULL,
    released_utc         TEXT NOT NULL,
    filled_utc           TEXT,
    filled_by_entry      TEXT,
    filled_appointment   TEXT,
    closed_utc           TEXT,
    close_reason         TEXT
);

CREATE TABLE IF NOT EXISTS backfill_offer (
    offer_id       TEXT PRIMARY KEY,
    release_id     TEXT NOT NULL REFERENCES slot_release(release_id),
    entry_id       TEXT NOT NULL REFERENCES waitlist_entry(entry_id),
    patient_id     TEXT NOT NULL,
    rank           INTEGER NOT NULL,
    offered_utc    TEXT NOT NULL,
    expires_utc    TEXT NOT NULL,
    responded_utc  TEXT,
    outcome        TEXT NOT NULL DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS message_log (
    message_id      TEXT PRIMARY KEY,
    family_id       TEXT NOT NULL,
    patient_id      TEXT,
    appointment_id  TEXT,
    offer_id        TEXT,
    channel         TEXT NOT NULL,
    purpose         TEXT NOT NULL,
    template_id     TEXT NOT NULL,
    planned_utc     TEXT NOT NULL,   -- when the cadence rule said to send
    send_after_utc  TEXT,            -- quiet-hours deferral; planned_utc is preserved
    sent_utc        TEXT,
    status          TEXT NOT NULL,   -- planned|sent|deferred|blocked|failed|skipped
    block_reason    TEXT,
    gateway_ref     TEXT,
    to_address      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS consent (
    consent_id       TEXT PRIMARY KEY,
    family_id        TEXT NOT NULL REFERENCES family(family_id),
    channel          TEXT NOT NULL,
    purpose          TEXT NOT NULL,
    granted_utc      TEXT NOT NULL,
    capture_method   TEXT NOT NULL,   -- intake_form|portal|verbal_documented|sms_double_optin
    capture_evidence TEXT,            -- form id, recording ref, staff initials
    captured_by      TEXT,
    revoked_utc      TEXT,
    revocation_method TEXT,
    UNIQUE (family_id, channel, purpose, granted_utc)
);

CREATE TABLE IF NOT EXISTS suppression (
    suppression_id  TEXT PRIMARY KEY,
    family_id       TEXT NOT NULL,
    channel         TEXT NOT NULL,
    reason          TEXT NOT NULL,   -- stop_keyword|staff|bounce|deceased|complaint
    created_utc     TEXT NOT NULL,
    source_message  TEXT,
    released_utc    TEXT             -- set only on an explicit START opt-back-in
);

CREATE INDEX IF NOT EXISTS ix_appt_start ON appointment(start_utc, status);
CREATE INDEX IF NOT EXISTS ix_appt_provider ON appointment(provider_id, start_utc);
CREATE INDEX IF NOT EXISTS ix_wait_active ON waitlist_entry(status, visit_type);
CREATE INDEX IF NOT EXISTS ix_msg_family ON message_log(family_id, sent_utc);
CREATE INDEX IF NOT EXISTS ix_msg_appt ON message_log(appointment_id, purpose);
CREATE INDEX IF NOT EXISTS ix_consent_family ON consent(family_id, channel, purpose);
CREATE INDEX IF NOT EXISTS ix_suppression_family ON suppression(family_id, channel);
CREATE INDEX IF NOT EXISTS ix_release_open ON slot_release(filled_utc, closed_utc);

-- One open release per appointment. This is the database-level backstop for the
-- check-then-act race in release_slot: a duplicate cancellation webhook cannot
-- create a second sellable copy of the same twenty minutes.
CREATE UNIQUE INDEX IF NOT EXISTS ux_release_open_per_appointment
    ON slot_release(appointment_id) WHERE filled_utc IS NULL AND closed_utc IS NULL;
"""


class Database:
    """Connection factory with per-thread connections and an IMMEDIATE helper.

    WHY per-thread connections: SQLite connection objects are not safe to share
    across threads, and the backfill race test deliberately runs five threads.
    A thread-local connection is the supported pattern and it is also what a
    web worker pool will do in production.

    WHY WAL: readers do not block the writer. The reminder cron sweeping the
    message log must not stall an accept coming in off a Twilio webhook.
    """

    def __init__(self, path: str | os.PathLike[str], *, busy_timeout: float = 30.0) -> None:
        self.path = str(path)
        if self.path == ":memory:":
            # Every thread would open its own private, schema-less database:
            # the backfill pool would crash on "no such table" and a race test
            # against it would prove nothing. Refuse loudly rather than let a
            # caller believe they are exercising the real locking behaviour.
            raise ValueError(
                "Database(':memory:') is unsupported: thread-local connections "
                "cannot share an anonymous in-memory database. Use a file path "
                "(pytest's tmp_path works) or 'file::memory:?cache=shared'."
            )
        self.busy_timeout = busy_timeout
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._local = threading.local()
        with self.connection() as conn:
            conn.executescript(SCHEMA)

    def _new_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout,
            # isolation_level=None turns OFF the driver's implicit transaction
            # management so that BEGIN IMMEDIATE means exactly what it says.
            # Without this, sqlite3 opens a deferred transaction behind our
            # back and the write lock is not taken until the first UPDATE --
            # which is precisely the window a five-way blast races in.
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(f"PRAGMA busy_timeout={int(self.busy_timeout * 1000)}")
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        existing = getattr(self._local, "conn", None)
        if existing is None:
            existing = self._new_connection()
            self._local.conn = existing
        return existing

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        yield self.conn

    @contextmanager
    def immediate(self) -> Iterator[sqlite3.Connection]:
        """Open a write transaction that takes the reserved lock immediately.

        This is SQLite's equivalent of SELECT ... FOR UPDATE. Everything inside
        the block sees a consistent snapshot that no other writer can modify,
        which is the entire basis of first-accept-wins in backfill.py.
        """
        conn = self.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except Exception:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    # -- small query helpers ----------------------------------------------

    def one(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        row = self.conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def all(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def close(self) -> None:
        existing = getattr(self._local, "conn", None)
        if existing is not None:
            existing.close()
            self._local.conn = None


# --------------------------------------------------------------------------
# Convenience writers used by tests, the demo, and the EHR sync shim
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Appointment:
    appointment_id: str
    patient_id: str
    provider_id: str
    visit_type: str
    start_utc: datetime
    duration_minutes: int
    status: str

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Appointment":
        return cls(
            appointment_id=row["appointment_id"],
            patient_id=row["patient_id"],
            provider_id=row["provider_id"],
            visit_type=row["visit_type"],
            start_utc=parse_iso(row["start_utc"]),
            duration_minutes=row["duration_minutes"],
            status=row["status"],
        )


def add_family(
    db: Database,
    *,
    family_id: str,
    display_name: str,
    primary_phone: str,
    primary_email: str | None = None,
) -> str:
    db.execute(
        "INSERT OR REPLACE INTO family (family_id, display_name, primary_phone, primary_email)"
        " VALUES (?,?,?,?)",
        (family_id, display_name, primary_phone, primary_email),
    )
    return family_id


def add_patient(
    db: Database,
    *,
    patient_id: str,
    family_id: str,
    first_name: str,
    last_name: str,
    dob: str,
) -> str:
    db.execute(
        "INSERT OR REPLACE INTO patient (patient_id, family_id, first_name, last_name, dob)"
        " VALUES (?,?,?,?,?)",
        (patient_id, family_id, first_name, last_name, dob),
    )
    return patient_id


def add_provider(
    db: Database,
    *,
    provider_id: str,
    display_name: str,
    min_age_months: int = 0,
    max_age_months: int = 300,
    visit_types: Sequence[str] = VisitType.ALL,
) -> str:
    db.execute(
        "INSERT OR REPLACE INTO provider (provider_id, display_name, min_age_months,"
        " max_age_months, visit_types_csv) VALUES (?,?,?,?,?)",
        (provider_id, display_name, min_age_months, max_age_months, ",".join(visit_types)),
    )
    return provider_id


def add_appointment(
    db: Database,
    *,
    patient_id: str,
    provider_id: str,
    visit_type: str,
    start: datetime,
    duration_minutes: int = 20,
    status: str = AppointmentStatus.SCHEDULED,
    now: datetime | None = None,
    booking_source: str = "staff",
    appointment_id: str | None = None,
    expected_revenue: float = 140.0,
) -> str:
    if visit_type not in VisitType.ALL:
        raise ValueError(f"unknown visit_type {visit_type!r}")
    if status not in AppointmentStatus.ALL:
        raise ValueError(f"unknown status {status!r}")
    appointment_id = appointment_id or new_id("appt")
    db.execute(
        """INSERT INTO appointment (appointment_id, patient_id, provider_id, visit_type,
               start_utc, duration_minutes, status, created_utc, booking_source,
               expected_revenue)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            appointment_id,
            patient_id,
            provider_id,
            visit_type,
            iso(start),
            duration_minutes,
            status,
            iso(now or datetime.now(timezone.utc)),
            booking_source,
            expected_revenue,
        ),
    )
    return appointment_id


def add_waitlist_entry(
    db: Database,
    *,
    patient_id: str,
    visit_type: str,
    earliest_ok: datetime,
    latest_ok: datetime,
    desired_provider: str | None = None,
    priority: int = 0,
    added: datetime | None = None,
    notify_channel: str = Channel.SMS,
    entry_id: str | None = None,
) -> str:
    if visit_type not in VisitType.ALL:
        raise ValueError(f"unknown visit_type {visit_type!r}")
    if notify_channel not in Channel.ALL:
        raise ValueError(f"unknown notify_channel {notify_channel!r}")
    entry_id = entry_id or new_id("wl")
    db.execute(
        """INSERT INTO waitlist_entry (entry_id, patient_id, desired_provider, visit_type,
               earliest_ok_utc, latest_ok_utc, priority, added_utc, notify_channel)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            entry_id,
            patient_id,
            desired_provider,
            visit_type,
            iso(earliest_ok),
            iso(latest_ok),
            priority,
            iso(added or datetime.now(timezone.utc)),
            notify_channel,
        ),
    )
    return entry_id


def grant_consent(
    db: Database,
    *,
    family_id: str,
    channel: str = Channel.SMS,
    purpose: str = ConsentPurpose.REMINDERS,
    granted: datetime | None = None,
    capture_method: str = "intake_form",
    capture_evidence: str | None = None,
    captured_by: str | None = None,
) -> str:
    """Record express consent. Every argument here is evidence in a TCPA file.

    `capture_method` and `capture_evidence` are not optional in spirit: a
    consent row that cannot be traced to a signed intake form, a portal event,
    or a documented verbal capture is an assertion, not evidence.
    """
    if purpose not in ConsentPurpose.ALL:
        raise ValueError(f"unknown consent purpose {purpose!r}")
    if channel not in Channel.ALL:
        raise ValueError(f"unknown channel {channel!r}")
    consent_id = new_id("con")
    db.execute(
        """INSERT INTO consent (consent_id, family_id, channel, purpose, granted_utc,
               capture_method, capture_evidence, captured_by)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            consent_id,
            family_id,
            channel,
            purpose,
            iso(granted or datetime.now(timezone.utc)),
            capture_method,
            capture_evidence,
            captured_by,
        ),
    )
    return consent_id


def revoke_consent(
    db: Database,
    *,
    family_id: str,
    channel: str = Channel.SMS,
    purpose: str = ConsentPurpose.REMINDERS,
    revoked: datetime | None = None,
    method: str = "stop_keyword",
) -> int:
    """Revoke consent and suppress the channel in the same breath.

    WHY both: revocation without suppression leaves every other code path free
    to send. The suppression list is the enforcement point; the consent row is
    the evidence. Doing one without the other is how practices end up with a
    documented opt-out and a message sent the following Tuesday.
    """
    when = iso(revoked or datetime.now(timezone.utc))
    cur = db.execute(
        "UPDATE consent SET revoked_utc = ?, revocation_method = ?"
        " WHERE family_id = ? AND channel = ? AND purpose = ? AND revoked_utc IS NULL",
        (when, method, family_id, channel, purpose),
    )
    # Plain INSERT, never OR IGNORE: a STOP that collides with an existing row
    # (same family, same channel, same second -- a webhook retry) must still
    # create a live suppression. Silently ignoring it leaves the family
    # subscribed, which is the one outcome TCPA does not forgive.
    db.execute(
        "INSERT INTO suppression (suppression_id, family_id, channel, reason,"
        " created_utc) VALUES (?,?,?,?,?)",
        (new_id("sup"), family_id, channel, method, when),
    )
    return cur.rowcount


def seed_practice_defaults(db: Database) -> None:
    """Two providers with realistic age ranges, for demos and tests."""
    add_provider(
        db,
        provider_id="dr_ruiz",
        display_name="Dr. Ruiz",
        min_age_months=0,
        max_age_months=300,
    )
    add_provider(
        db,
        provider_id="dr_okafor",
        display_name="Dr. Okafor",
        min_age_months=24,
        max_age_months=300,
        visit_types=(VisitType.WELL, VisitType.SICK, VisitType.FOLLOW_UP),
    )


def address_for(row: Mapping[str, Any], channel: str) -> str | None:
    """The destination for a family on a given channel, or None if unreachable.

    WHY this is a function rather than an inline `row["primary_phone"]`: a
    waitlist entry can name `notify_channel='email'`, and pairing that channel
    with a phone number produces a candidate that passes the consent check,
    consumes a blast slot, and is then rejected by the gateway. An unreachable
    family must be excluded at selection time, not discovered at send time.
    """
    if channel in (Channel.SMS, Channel.VOICE):
        return row.get("primary_phone") or None
    if channel == Channel.EMAIL:
        return row.get("primary_email") or None
    return None


def within(a: datetime, b: datetime, tolerance: timedelta) -> bool:
    return abs(a - b) <= tolerance
