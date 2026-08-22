"""Waitlist backfill with atomic, first-accept-wins booking.

WHY the locking discipline in this file is the whole point:

README I-07: "Blasting a slot to five people and letting two book it is worse
than not blasting it at all." A double-book is not a data-integrity nuisance —
it is two families in the waiting room for one twenty-minute slot, one of whom
was told by an automated system that they had an appointment. That is a worse
patient experience than never offering the slot, and it destroys trust in the
entire programme in one afternoon.

The guarantee is implemented as: every accept opens `BEGIN IMMEDIATE`, re-reads
`slot_release.filled_utc` *inside* that transaction, and either wins the slot or
loses it. SQLite serialises writers, so exactly one transaction observes
`filled_utc IS NULL`. There is no application-level lock, no optimistic retry
loop, and no "check then act" window — the check and the act are the same
transaction.

Two secondary rules matter almost as much:

  * No network I/O inside the write transaction. The gateway call that tells
    four families they lost happens after COMMIT. Holding a database write lock
    across an HTTP round-trip to a carrier is how a five-way blast becomes a
    thirty-second stall on every other writer.
  * A losing accept is a *normal outcome*, not an error. It gets a courteous
    message and the waitlist entry stays active.

No language model appears anywhere in this file, and none should.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence

from .cadence import FrequencyCap, SendGate, TEMPLATES, format_when
from .gateway import Gateway
from .models import (
    AppointmentStatus,
    Channel,
    address_for,
    Database,
    MessagePurpose,
    OfferOutcome,
    age_months,
    iso,
    new_id,
    parse_iso,
    to_local,
)

__all__ = [
    "Candidate",
    "AcceptResult",
    "BackfillEngine",
    "rank_candidates",
]

#: How long a blasted offer stays live before it expires. Short enough that a
#: slot is not held hostage by a parent who never looks at their phone, long
#: enough that a parent in a meeting has a fair chance.
DEFAULT_OFFER_TTL = timedelta(minutes=20)

#: How many families a single release is offered to at once. README target:
#: median fill time under ten minutes. Five is the README's number.
DEFAULT_BLAST_SIZE = 5

#: A slot starting sooner than this is not worth blasting — the family cannot
#: physically get there, and a "come now" text at T-45min reads as chaotic.
MIN_LEAD_TIME = timedelta(hours=2)


@dataclass(frozen=True)
class Candidate:
    entry_id: str
    patient_id: str
    family_id: str
    priority: int
    added_utc: datetime
    wait_seconds: float
    provider_requested: bool
    age_fit: bool
    notify_channel: str
    to_address: str
    first_name: str

    @property
    def sort_key(self) -> tuple[Any, ...]:
        """Lexicographic ranking key, in the order the build plan specifies.

        (priority, wait duration, appointment-type match, age-appropriateness)

        Visit type and age are *also* hard eligibility filters upstream, so by
        the time a candidate reaches this key both are already satisfied. They
        remain in the key as tie-breakers, which is not dead code: bulk-loaded
        waitlist entries frequently share a priority and an `added_utc` to the
        second, and a stable, explainable tie-break beats whatever order SQLite
        happens to return. `entry_id` closes the ordering so the ranking is
        fully deterministic and therefore testable.
        """
        return (
            -self.priority,
            -self.wait_seconds,
            0 if self.provider_requested else 1,
            0 if self.age_fit else 1,
            self.entry_id,
        )


@dataclass(frozen=True)
class AcceptResult:
    won: bool
    outcome: str
    offer_id: str
    release_id: str
    appointment_id: str | None = None
    reason: str = ""


def rank_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    return sorted(candidates, key=lambda c: c.sort_key)


class BackfillEngine:
    """Release → rank → blast → first accept wins."""

    INITIATIVE = "I-07"

    def __init__(
        self,
        db: Database,
        gateway: Gateway,
        *,
        gate: SendGate | None = None,
        templates: Mapping[str, str] | None = None,
        offer_ttl: timedelta = DEFAULT_OFFER_TTL,
        blast_size: int = DEFAULT_BLAST_SIZE,
        min_lead_time: timedelta = MIN_LEAD_TIME,
        audit: Any | None = None,
    ) -> None:
        self.db = db
        self.gateway = gateway
        self.gate = gate or SendGate(db)
        self.templates = dict(templates or TEMPLATES)
        self.offer_ttl = offer_ttl
        self.blast_size = blast_size
        self.min_lead_time = min_lead_time
        self.audit = audit

    # -- release -----------------------------------------------------------

    def release_slot(
        self,
        appointment_id: str,
        *,
        now: datetime,
        source: str = "sms",
        notify_family: bool = True,
    ) -> str | None:
        """Cancel an appointment and hand its slot to the backfill engine.

        Returns the release id, or None if the slot is not backfillable (too
        close to start). WHY still cancel in that case: the cancellation is
        valuable on its own — it turns a no-show into a documented cancellation,
        which is most of the after-hours capture in the README's model, even
        when the slot cannot be refilled.
        """
        # The whole state transition is one write transaction with a conditional
        # UPDATE. WHY: read-status-then-write is check-then-act, and the actors
        # here are a double-tapped cancel link, a carrier webhook retry and a
        # front-desk click -- all plausibly concurrent. Two callers passing the
        # status check would each insert a slot_release for the same twenty
        # minutes, each get blasted, and each get filled. The atomicity in
        # accept_offer is airtight but guards the wrong scope: it serialises
        # accepts *within* a release, not releases *of* an appointment.
        with self.db.immediate() as conn:
            row = conn.execute(
                """SELECT a.*, p.family_id, p.first_name, f.primary_phone, f.primary_email
                   FROM appointment a
                   JOIN patient p ON p.patient_id = a.patient_id
                   JOIN family f ON f.family_id = p.family_id
                   WHERE a.appointment_id = ?""",
                (appointment_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown appointment {appointment_id!r}")
            appt = dict(row)

            start = parse_iso(appt["start_utc"])
            backfillable = start - now >= self.min_lead_time

            cur = conn.execute(
                "UPDATE appointment SET status = ?, cancelled_utc = ?, cancel_source = ?"
                " WHERE appointment_id = ? AND status IN (?, ?)",
                (
                    AppointmentStatus.RELEASED if backfillable else AppointmentStatus.CANCELLED,
                    iso(now),
                    source,
                    appointment_id,
                    AppointmentStatus.SCHEDULED,
                    AppointmentStatus.CONFIRMED,
                ),
            )
            if cur.rowcount == 0:
                # Someone else already cancelled it. Not an error -- a parent
                # tapping the link twice is normal -- but this caller does not
                # get a second release to blast.
                return None

            # Any queued reminders for a cancelled visit are now noise.
            conn.execute(
                "UPDATE message_log SET status='skipped', block_reason='appointment_cancelled'"
                " WHERE appointment_id = ? AND status = 'planned'",
                (appointment_id,),
            )

            release_id: str | None = None
            if backfillable:
                release_id = new_id("rel")
                conn.execute(
                    """INSERT INTO slot_release (release_id, appointment_id, provider_id,
                           visit_type, start_utc, duration_minutes, released_utc)
                       VALUES (?,?,?,?,?,?,?)""",
                    (
                        release_id,
                        appointment_id,
                        appt["provider_id"],
                        appt["visit_type"],
                        appt["start_utc"],
                        appt["duration_minutes"],
                        iso(now),
                    ),
                )

        # Outside the write lock: notification is I/O.
        if notify_family:
            self._send_transactional(
                family_id=appt["family_id"],
                patient_id=appt["patient_id"],
                to_address=appt["primary_phone"],
                purpose=MessagePurpose.CANCEL_CONFIRMATION,
                template_id="cancel_confirmation",
                now=now,
                context={"first_name": appt["first_name"], "start_utc": appt["start_utc"]},
            )

        if self.audit is not None:
            self.audit.record_event(
                actor_id=f"scheduling:{source}",
                initiative_id=self.INITIATIVE,
                event_type="slot_released" if release_id else "appointment_cancelled",
                patient_id=appt["patient_id"],
                detail={"visit_type": appt["visit_type"], "provider": appt["provider_id"]},
            )
        return release_id

    # -- candidate selection ----------------------------------------------

    def eligible_candidates(self, release_id: str, *, now: datetime) -> list[Candidate]:
        """Hard-filter the waitlist, then rank what survives.

        Everything in the WHERE clause is a correctness filter, not a
        preference. An entry is excluded when offering it would produce a
        booking that should not exist:

          * wrong visit type — a sick slot is not a well slot
          * outside the family's stated date window
          * the patient's age is outside the provider's panel range
          * the family asked for a specific provider and this is not them
          * the patient already has an appointment overlapping this slot
          * the family is suppressed or has no consent, so we cannot reach them

        The last one is worth calling out: an unreachable family is not a
        lower-ranked candidate, it is not a candidate. Ranking them and then
        failing to notify them wastes a blast slot that a reachable family
        could have used, and silently lowers the fill rate.
        """
        release = self.db.one(
            "SELECT * FROM slot_release WHERE release_id = ?", (release_id,)
        )
        if release is None:
            raise KeyError(f"unknown release {release_id!r}")
        if release["filled_utc"] or release["closed_utc"]:
            return []

        start = parse_iso(release["start_utc"])
        end = start + timedelta(minutes=release["duration_minutes"])

        provider = self.db.one(
            "SELECT * FROM provider WHERE provider_id = ?", (release["provider_id"],)
        )
        if provider is None or not provider["active"]:
            return []
        if release["visit_type"] not in provider["visit_types_csv"].split(","):
            return []

        rows = self.db.all(
            """SELECT w.*, p.dob, p.first_name, p.family_id, f.primary_phone,
                      f.primary_email
               FROM waitlist_entry w
               JOIN patient p ON p.patient_id = w.patient_id
               JOIN family f ON f.family_id = p.family_id
               WHERE w.status = 'active'
                 AND w.visit_type = ?
                 AND w.earliest_ok_utc <= ?
                 AND w.latest_ok_utc >= ?
                 AND p.active = 1
                 AND (w.desired_provider IS NULL OR w.desired_provider = ?)
                 AND w.entry_id NOT IN (
                       SELECT entry_id FROM backfill_offer WHERE release_id = ?
                 )""",
            (
                release["visit_type"],
                release["start_utc"],
                release["start_utc"],
                release["provider_id"],
                release_id,
            ),
        )

        candidates: list[Candidate] = []
        for row in rows:
            months = age_months(row["dob"], start)
            if not (provider["min_age_months"] <= months <= provider["max_age_months"]):
                continue
            if self._has_conflicting_appointment(row["patient_id"], start, end):
                continue
            to_address = address_for(row, row["notify_channel"])
            if not to_address:
                # Unreachable on the channel they asked for. Excluded, not
                # ranked -- see the docstring above.
                continue
            decision = self.gate.evaluate(
                family_id=row["family_id"],
                channel=row["notify_channel"],
                purpose=MessagePurpose.BACKFILL_OFFER,
                at=now,
                tier=FrequencyCap.TIER_WAITLIST,
                deadline=start,
                allow_defer=False,  # an offer that arrives at 08:00 tomorrow is worthless
            )
            if not decision.allow:
                continue
            added = parse_iso(row["added_utc"])
            candidates.append(
                Candidate(
                    entry_id=row["entry_id"],
                    patient_id=row["patient_id"],
                    family_id=row["family_id"],
                    priority=int(row["priority"]),
                    added_utc=added,
                    wait_seconds=(now - added).total_seconds(),
                    provider_requested=row["desired_provider"] is not None,
                    age_fit=(
                        provider["min_age_months"] <= months <= provider["max_age_months"]
                    ),
                    notify_channel=row["notify_channel"],
                    to_address=to_address,
                    first_name=row["first_name"],
                )
            )
        return rank_candidates(candidates)

    def _has_conflicting_appointment(
        self, patient_id: str, start: datetime, end: datetime
    ) -> bool:
        rows = self.db.all(
            """SELECT start_utc, duration_minutes FROM appointment
               WHERE patient_id = ? AND status IN ('scheduled','confirmed')""",
            (patient_id,),
        )
        for row in rows:
            other_start = parse_iso(row["start_utc"])
            other_end = other_start + timedelta(minutes=row["duration_minutes"])
            if other_start < end and start < other_end:
                return True
        return False

    # -- blast -------------------------------------------------------------

    def blast(
        self,
        release_id: str,
        *,
        now: datetime,
        top_n: int | None = None,
    ) -> list[str]:
        """Offer the slot to the top N candidates simultaneously.

        The offer rows are all written first, then the notifications go out
        concurrently. WHY that order: if the process dies between writing and
        sending, the state is "offers exist, nobody was told", which a sweep can
        detect and retry. The reverse order produces "families were told, no
        offer exists", and an accept link that 404s.
        """
        top_n = top_n or self.blast_size
        release = self.db.one("SELECT * FROM slot_release WHERE release_id = ?", (release_id,))
        if release is None:
            raise KeyError(f"unknown release {release_id!r}")
        if release["filled_utc"] or release["closed_utc"]:
            return []

        start = parse_iso(release["start_utc"])

        # Quiet hours apply to the blast, not to the cancellation. A parent who
        # cancels at 21:40 must still be able to cancel -- that is most of the
        # after-hours capture in the README's model -- but four other families
        # do not get a 21:40 text about it. The release stays OPEN and the
        # morning sweep blasts it at 08:00. Closing it here would silently
        # discard exactly the slots this initiative exists to recover.
        if self.gate.quiet_hours.contains(now):
            resume_at = self.gate.quiet_hours.next_open(now)
            if start - resume_at < self.min_lead_time:
                self.db.execute(
                    "UPDATE slot_release SET closed_utc = ?, close_reason = ?"
                    " WHERE release_id = ?",
                    (iso(now), "quiet_hours_no_time_to_fill", release_id),
                )
            return []

        candidates = self.eligible_candidates(release_id, now=now)[:top_n]
        if not candidates:
            self.db.execute(
                "UPDATE slot_release SET closed_utc = ?, close_reason = ? WHERE release_id = ?",
                (iso(now), "no_eligible_candidates", release_id),
            )
            return []

        expires = now + self.offer_ttl
        offers: list[tuple[str, Candidate]] = []
        for rank, candidate in enumerate(candidates, start=1):
            offer_id = new_id("off")
            self.db.execute(
                """INSERT INTO backfill_offer (offer_id, release_id, entry_id, patient_id,
                       rank, offered_utc, expires_utc, outcome)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    offer_id,
                    release_id,
                    candidate.entry_id,
                    candidate.patient_id,
                    rank,
                    iso(now),
                    iso(expires),
                    OfferOutcome.PENDING,
                ),
            )
            offers.append((offer_id, candidate))

        provider = self.db.one(
            "SELECT display_name FROM provider WHERE provider_id = ?",
            (release["provider_id"],),
        )
        local = to_local(start)
        body_for = lambda offer_id: self.templates["backfill_offer"].format(  # noqa: E731
            when=format_when(local),
            provider=provider["display_name"] if provider else "your provider",
            accept_url=f"https://nsp.example/a/{offer_id}",
            decline_url=f"https://nsp.example/d/{offer_id}",
        )

        def _notify(item: tuple[str, Candidate]) -> tuple[str, bool]:
            offer_id, candidate = item
            delivered = self._log_and_send(
                family_id=candidate.family_id,
                patient_id=candidate.patient_id,
                to_address=candidate.to_address,
                channel=candidate.notify_channel,
                purpose=MessagePurpose.BACKFILL_OFFER,
                template_id="backfill_offer",
                body=body_for(offer_id),
                now=now,
                offer_id=offer_id,
                tier=FrequencyCap.TIER_WAITLIST,
            )
            return offer_id, delivered

        # Genuinely concurrent, because the README's fill-time target is a
        # median under ten minutes and serialising five carrier round-trips
        # front-loads seconds onto the family at the bottom of the list.
        if len(offers) > 1:
            with ThreadPoolExecutor(max_workers=min(8, len(offers))) as pool:
                results = list(pool.map(_notify, offers))
        else:
            results = [_notify(offers[0])]

        # An offer whose notification never left the building is a live accept
        # link nobody has. Void it rather than leaving it pending: otherwise the
        # slot looks offered, the fill-rate denominator counts it, and nothing
        # can ever accept it. This is rare -- eligibility already applied the
        # gate -- but the gate is re-evaluated at send time by design, so the
        # two can legitimately disagree.
        delivered = []
        for offer_id, ok in results:
            if ok:
                delivered.append(offer_id)
            else:
                self.db.execute(
                    "UPDATE backfill_offer SET outcome=?, responded_utc=? WHERE offer_id=?",
                    (OfferOutcome.EXPIRED, iso(now), offer_id),
                )
        if not delivered:
            self.db.execute(
                "UPDATE slot_release SET closed_utc = ?, close_reason = ? WHERE release_id = ?"
                " AND filled_utc IS NULL",
                (iso(now), "no_reachable_candidates", release_id),
            )
        return delivered

    # -- accept ------------------------------------------------------------

    def accept_offer(self, offer_id: str, *, now: datetime) -> AcceptResult:
        """Book the slot for this offer if it is still open. Atomic.

        Everything that decides the winner happens inside one BEGIN IMMEDIATE
        transaction: read the offer, read the release, write the appointment,
        stamp the release, lose the siblings. A concurrent caller either blocks
        until this commits and then sees `filled_utc` set, or blocks and then
        wins because this one rolled back. There is no interleaving in which
        both observe an open slot.
        """
        losers: list[str] = []
        result: AcceptResult

        with self.db.immediate() as conn:
            offer = conn.execute(
                "SELECT * FROM backfill_offer WHERE offer_id = ?", (offer_id,)
            ).fetchone()
            if offer is None:
                raise KeyError(f"unknown offer {offer_id!r}")
            release_id = offer["release_id"]

            if offer["outcome"] != OfferOutcome.PENDING:
                return AcceptResult(
                    False, offer["outcome"], offer_id, release_id, reason="offer_not_pending"
                )
            if parse_iso(offer["expires_utc"]) < now:
                conn.execute(
                    "UPDATE backfill_offer SET outcome=?, responded_utc=? WHERE offer_id=?",
                    (OfferOutcome.EXPIRED, iso(now), offer_id),
                )
                return AcceptResult(
                    False, OfferOutcome.EXPIRED, offer_id, release_id, reason="offer_expired"
                )

            release = conn.execute(
                "SELECT * FROM slot_release WHERE release_id = ?", (release_id,)
            ).fetchone()
            if release["filled_utc"] is not None or release["closed_utc"] is not None:
                conn.execute(
                    "UPDATE backfill_offer SET outcome=?, responded_utc=? WHERE offer_id=?",
                    (OfferOutcome.LOST, iso(now), offer_id),
                )
                return AcceptResult(
                    False, OfferOutcome.LOST, offer_id, release_id, reason="slot_already_filled"
                )

            # Re-check the entry inside the transaction. Eligibility ran at
            # blast time, before any booking existed; between then and now this
            # family may have accepted a different release for the same clock
            # time. Without this, one patient is booked into two simultaneous
            # slots and the waitlist row's booked_appointment silently
            # overwrites the first booking, orphaning it.
            entry = conn.execute(
                "SELECT status FROM waitlist_entry WHERE entry_id = ?", (offer["entry_id"],)
            ).fetchone()
            if entry is None or entry["status"] != "active":
                conn.execute(
                    "UPDATE backfill_offer SET outcome=?, responded_utc=? WHERE offer_id=?",
                    (OfferOutcome.LOST, iso(now), offer_id),
                )
                return AcceptResult(
                    False, OfferOutcome.LOST, offer_id, release_id,
                    reason="waitlist_entry_no_longer_active",
                )

            slot_start = parse_iso(release["start_utc"])
            slot_end = slot_start + timedelta(minutes=release["duration_minutes"])
            clash = conn.execute(
                """SELECT start_utc, duration_minutes FROM appointment
                   WHERE patient_id = ? AND status IN ('scheduled','confirmed')""",
                (offer["patient_id"],),
            ).fetchall()
            for other in clash:
                other_start = parse_iso(other["start_utc"])
                other_end = other_start + timedelta(minutes=other["duration_minutes"])
                if other_start < slot_end and slot_start < other_end:
                    conn.execute(
                        "UPDATE backfill_offer SET outcome=?, responded_utc=? WHERE offer_id=?",
                        (OfferOutcome.DECLINED, iso(now), offer_id),
                    )
                    return AcceptResult(
                        False, OfferOutcome.DECLINED, offer_id, release_id,
                        reason="patient_already_booked_at_that_time",
                    )

            original = conn.execute(
                "SELECT expected_revenue FROM appointment WHERE appointment_id = ?",
                (release["appointment_id"],),
            ).fetchone()

            appointment_id = new_id("appt")
            conn.execute(
                """INSERT INTO appointment (appointment_id, patient_id, provider_id,
                       visit_type, start_utc, duration_minutes, status, created_utc,
                       booking_source, filled_from_release, expected_revenue)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    appointment_id,
                    offer["patient_id"],
                    release["provider_id"],
                    release["visit_type"],
                    release["start_utc"],
                    release["duration_minutes"],
                    AppointmentStatus.CONFIRMED,
                    iso(now),
                    "backfill",
                    release_id,
                    original["expected_revenue"] if original else 140.0,
                ),
            )
            conn.execute(
                "UPDATE slot_release SET filled_utc=?, filled_by_entry=?, filled_appointment=?"
                " WHERE release_id = ?",
                (iso(now), offer["entry_id"], appointment_id, release_id),
            )
            conn.execute(
                "UPDATE backfill_offer SET outcome=?, responded_utc=? WHERE offer_id=?",
                (OfferOutcome.ACCEPTED, iso(now), offer_id),
            )
            conn.execute(
                "UPDATE waitlist_entry SET status='booked', booked_appointment=?"
                " WHERE entry_id = ?",
                (appointment_id, offer["entry_id"]),
            )
            sibling_rows = conn.execute(
                "SELECT offer_id FROM backfill_offer WHERE release_id = ? AND outcome = ?"
                " AND offer_id != ?",
                (release_id, OfferOutcome.PENDING, offer_id),
            ).fetchall()
            losers = [r["offer_id"] for r in sibling_rows]
            conn.execute(
                "UPDATE backfill_offer SET outcome=?, responded_utc=? WHERE release_id = ?"
                " AND outcome = ? AND offer_id != ?",
                (OfferOutcome.LOST, iso(now), release_id, OfferOutcome.PENDING, offer_id),
            )
            result = AcceptResult(
                True, OfferOutcome.ACCEPTED, offer_id, release_id, appointment_id
            )

        # Outside the write lock, deliberately. See the module docstring.
        if self.audit is not None:
            self.audit.record_event(
                actor_id="scheduling:backfill",
                initiative_id=self.INITIATIVE,
                event_type="slot_backfilled",
                detail={"release_id": result.release_id, "losing_offers": len(losers)},
            )
        self._notify_winner(offer_id, now=now)
        for loser in losers:
            self._notify_loser(loser, now=now)
        return result

    def decline_offer(self, offer_id: str, *, now: datetime) -> AcceptResult:
        """Record a decline. The slot stays open for the others."""
        with self.db.immediate() as conn:
            offer = conn.execute(
                "SELECT * FROM backfill_offer WHERE offer_id = ?", (offer_id,)
            ).fetchone()
            if offer is None:
                raise KeyError(f"unknown offer {offer_id!r}")
            if offer["outcome"] != OfferOutcome.PENDING:
                return AcceptResult(False, offer["outcome"], offer_id, offer["release_id"])
            conn.execute(
                "UPDATE backfill_offer SET outcome=?, responded_utc=? WHERE offer_id=?",
                (OfferOutcome.DECLINED, iso(now), offer_id),
            )
            return AcceptResult(False, OfferOutcome.DECLINED, offer_id, offer["release_id"])

    def sweep_open_releases(self, *, now: datetime, top_n: int | None = None) -> dict[str, int]:
        """Re-blast every release that is still open. Run on the same cron tick.

        This is what picks up slots released overnight, and what gives a slot a
        second chance after every offer from the first blast expired without an
        answer. Idempotent: `eligible_candidates` excludes anyone already
        offered this release, so a repeat sweep reaches the next tier of the
        waitlist rather than re-texting the same five families.
        """
        rows = self.db.all(
            """SELECT release_id, start_utc FROM slot_release
               WHERE filled_utc IS NULL AND closed_utc IS NULL""",
        )
        histogram = {"blasted": 0, "skipped": 0, "too_close": 0, "offers": 0}
        for row in rows:
            if parse_iso(row["start_utc"]) - now < self.min_lead_time:
                # Counted, not silently dropped. A rising `too_close` number is
                # the signal that quiet hours and the minimum lead time are
                # together eating the early-morning slots, which is a tunable
                # trade-off the practice should get to see and decide on.
                histogram["too_close"] += 1
                continue
            pending = self.db.one(
                "SELECT COUNT(*) c FROM backfill_offer WHERE release_id = ? AND outcome = ?",
                (row["release_id"], OfferOutcome.PENDING),
            )
            if pending and pending["c"]:
                histogram["skipped"] += 1
                continue
            offers = self.blast(row["release_id"], now=now, top_n=top_n)
            if offers:
                histogram["blasted"] += 1
                histogram["offers"] += len(offers)
            else:
                histogram["skipped"] += 1
        return histogram

    def expire_stale_offers(self, *, now: datetime) -> int:
        """Sweep expired offers. Run alongside the reminder cron."""
        cur = self.db.execute(
            "UPDATE backfill_offer SET outcome=?, responded_utc=? WHERE outcome=?"
            " AND expires_utc < ?",
            (OfferOutcome.EXPIRED, iso(now), OfferOutcome.PENDING, iso(now)),
        )
        return cur.rowcount

    def close_unfilled_releases(self, *, now: datetime) -> int:
        """Close releases whose slot time has passed without a fill."""
        cur = self.db.execute(
            "UPDATE slot_release SET closed_utc = ?, close_reason = 'slot_time_passed'"
            " WHERE filled_utc IS NULL AND closed_utc IS NULL AND start_utc <= ?",
            (iso(now), iso(now)),
        )
        return cur.rowcount

    # -- notification helpers ---------------------------------------------

    def _notify_winner(self, offer_id: str, *, now: datetime) -> None:
        row = self.db.one(
            """SELECT o.offer_id, o.patient_id, p.first_name, p.family_id, f.primary_phone,
                      f.primary_email, r.start_utc, pr.display_name AS provider_name,
                      e.notify_channel
               FROM backfill_offer o
               JOIN waitlist_entry e ON e.entry_id = o.entry_id
               JOIN patient p ON p.patient_id = o.patient_id
               JOIN family f ON f.family_id = p.family_id
               JOIN slot_release r ON r.release_id = o.release_id
               JOIN provider pr ON pr.provider_id = r.provider_id
               WHERE o.offer_id = ?""",
            (offer_id,),
        )
        if row is None:
            return
        local = to_local(parse_iso(row["start_utc"]))
        body = self.templates["backfill_won"].format(
            first_name=row["first_name"],
            when=format_when(local),
            provider=row["provider_name"],
        )
        self._log_and_send(
            family_id=row["family_id"],
            patient_id=row["patient_id"],
            to_address=address_for(row, row["notify_channel"]) or row["primary_phone"],
            channel=row["notify_channel"],
            purpose=MessagePurpose.BACKFILL_WON,
            template_id="backfill_won",
            body=body,
            now=now,
            offer_id=offer_id,
        )

    def _notify_loser(self, offer_id: str, *, now: datetime) -> None:
        row = self.db.one(
            """SELECT o.offer_id, o.patient_id, p.family_id, f.primary_phone,
                      f.primary_email, e.notify_channel
               FROM backfill_offer o
               JOIN waitlist_entry e ON e.entry_id = o.entry_id
               JOIN patient p ON p.patient_id = o.patient_id
               JOIN family f ON f.family_id = p.family_id
               WHERE o.offer_id = ?""",
            (offer_id,),
        )
        if row is None:
            return
        self._log_and_send(
            family_id=row["family_id"],
            patient_id=row["patient_id"],
            to_address=address_for(row, row["notify_channel"]) or row["primary_phone"],
            channel=row["notify_channel"],
            purpose=MessagePurpose.BACKFILL_LOST,
            template_id="backfill_lost",
            body=self.templates["backfill_lost"],
            now=now,
            offer_id=offer_id,
        )

    def _send_transactional(
        self,
        *,
        family_id: str,
        patient_id: str,
        to_address: str,
        purpose: str,
        template_id: str,
        now: datetime,
        context: Mapping[str, Any],
    ) -> None:
        local = to_local(parse_iso(context["start_utc"])) if context.get("start_utc") else None
        body = self.templates[template_id].format(
            first_name=context.get("first_name", ""),
            when=format_when(local) if local else "",
            confirm_url="https://nsp.example/book",
        )
        self._log_and_send(
            family_id=family_id,
            patient_id=patient_id,
            to_address=to_address,
            channel=Channel.SMS,
            purpose=purpose,
            template_id=template_id,
            body=body,
            now=now,
        )

    def _log_and_send(
        self,
        *,
        family_id: str,
        patient_id: str,
        to_address: str,
        channel: str,
        purpose: str,
        template_id: str,
        body: str,
        now: datetime,
        offer_id: str | None = None,
        tier: str = FrequencyCap.TIER_APPOINTMENT,
    ) -> bool:
        """Write the message row, apply the gate, send, record the outcome.

        Transactional purposes still pass through the gate. Suppression and
        consent are checked even for a cancellation confirmation, because a
        family that has issued STOP has told the carrier and us that no message
        of any kind is welcome. The gate exempts transactional messages from
        quiet hours and the frequency cap only — never from the legal gates.
        """
        message_id = new_id("msg")
        self.db.execute(
            """INSERT INTO message_log (message_id, family_id, patient_id, offer_id,
                   channel, purpose, template_id, planned_utc, status, to_address)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                message_id,
                family_id,
                patient_id,
                offer_id,
                channel,
                purpose,
                template_id,
                iso(now),
                "planned",
                to_address,
            ),
        )
        decision = self.gate.evaluate(
            family_id=family_id,
            channel=channel,
            purpose=purpose,
            at=now,
            tier=tier,
            allow_defer=False,
        )
        if not decision.allow:
            self.db.execute(
                "UPDATE message_log SET status=?, block_reason=? WHERE message_id=?",
                (decision.status, decision.reason, message_id),
            )
            return False
        receipt = self.gateway.send(
            to=to_address, body=body, channel=channel, purpose=purpose, reference=message_id
        )
        if receipt.accepted:
            self.db.execute(
                "UPDATE message_log SET status='sent', sent_utc=?, gateway_ref=?"
                " WHERE message_id=?",
                (iso(now), receipt.gateway_ref, message_id),
            )
            return True
        self.db.execute(
            "UPDATE message_log SET status='failed', block_reason=? WHERE message_id=?",
            (receipt.error or "gateway", message_id),
        )
        return False
