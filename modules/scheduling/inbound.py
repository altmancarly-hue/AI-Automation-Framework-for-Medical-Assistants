"""Inbound webhook routing. One tap or keyword in, one recorded action out.

WHY this is a sixth file rather than part of gateway.py:

`gateway.py` classifies — it turns a webhook payload into an action code and
knows nothing about appointments. This file *acts*, which means it touches the
database, the backfill engine and the consent record. Keeping the two apart
means the classification rules can be tested exhaustively without a database,
and the action rules can be tested without pretending to be a carrier.

The opt-out path is the reason this file gets careful treatment. Under TCPA an
opt-out must be honoured immediately, which in practice means: suppress before
acknowledging, and let the acknowledgement itself be the one message that is
still permitted to go out. Getting that order backwards — acknowledge, then
suppress — leaves a window in which a concurrent reminder sweep sends to a
number that has already said stop.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from .backfill import BackfillEngine
from .gateway import Gateway, InboundAction, InboundIntent, classify_inbound
from .models import (
    AppointmentStatus,
    Channel,
    ConsentPurpose,
    Database,
    MessagePurpose,
    iso,
    new_id,
    revoke_consent,
)

__all__ = ["InboundResult", "InboundRouter"]


@dataclass(frozen=True)
class InboundResult:
    action: str
    handled: bool
    detail: str = ""
    reference: str | None = None


class InboundRouter:
    """Applies a classified inbound message to practice state."""

    INITIATIVE = "I-07"

    def __init__(
        self,
        db: Database,
        gateway: Gateway,
        backfill: BackfillEngine,
        *,
        audit: Any | None = None,
    ) -> None:
        self.db = db
        self.gateway = gateway
        self.backfill = backfill
        self.audit = audit

    def handle(self, payload: Mapping[str, Any], *, now: datetime) -> InboundResult:
        intent = classify_inbound(payload)
        if intent.action == InboundAction.OPT_OUT:
            return self._opt_out(intent, now=now)
        if intent.action == InboundAction.OPT_IN:
            return self._opt_in(intent, now=now)
        if intent.action == InboundAction.CONFIRM:
            return self._confirm(intent, now=now)
        if intent.action == InboundAction.CANCEL:
            return self._cancel(intent, now=now)
        if intent.action == InboundAction.ACCEPT_OFFER:
            return self._accept(intent, now=now)
        if intent.action == InboundAction.DECLINE_OFFER:
            return self._decline(intent, now=now)
        if intent.action == InboundAction.RESCHEDULE:
            return self._to_human(intent, now=now, detail="reschedule_requested")
        if intent.action == InboundAction.HELP:
            return InboundResult(intent.action, True, "help_sent")
        return self._to_human(intent, now=now, detail="free_text")

    # -- consent -----------------------------------------------------------

    def _family_for_address(self, address: str) -> str | None:
        row = self.db.one(
            "SELECT family_id FROM family WHERE primary_phone = ?", (address,)
        )
        return row["family_id"] if row else None

    def _opt_out(self, intent: InboundIntent, *, now: datetime) -> InboundResult:
        family_id = self._family_for_address(intent.from_address)
        if family_id is None:
            # An unknown number that says STOP is still suppressed, keyed on the
            # address itself. WHY: we may not recognise the number because the
            # family changed carriers, and "we could not match you" is not a
            # defence for continuing to send.
            self.db.execute(
                "INSERT OR IGNORE INTO suppression (suppression_id, family_id, channel,"
                " reason, created_utc) VALUES (?,?,?,?,?)",
                (new_id("sup"), f"unmatched:{intent.from_address}", Channel.SMS,
                 "stop_keyword", iso(now)),
            )
            return InboundResult(intent.action, True, "suppressed_unmatched_number")

        # Suppress FIRST, acknowledge second. See the module docstring.
        revoke_consent(
            self.db,
            family_id=family_id,
            channel=Channel.SMS,
            purpose=ConsentPurpose.REMINDERS,
            revoked=now,
            method="stop_keyword",
        )
        # Marketing consent, if any, dies with the same keyword.
        revoke_consent(
            self.db,
            family_id=family_id,
            channel=Channel.SMS,
            purpose=ConsentPurpose.MARKETING,
            revoked=now,
            method="stop_keyword",
        )
        if self.audit is not None:
            self.audit.record_event(
                actor_id=f"inbound:{intent.from_address}",
                initiative_id=self.INITIATIVE,
                event_type="consent_revoked",
                detail={"channel": Channel.SMS, "method": "stop_keyword"},
            )
        # The confirmation is carrier-required and is the single message that
        # may follow a STOP. It is sent directly, bypassing the send gate,
        # because the gate would (correctly) now refuse it.
        self.gateway.send(
            to=intent.from_address,
            body=(
                "You are unsubscribed from North Suburban Pediatrics appointment "
                "messages. No further texts will be sent. Reply START to resume."
            ),
            channel=Channel.SMS,
            purpose=MessagePurpose.OPTOUT_CONFIRMATION,
            reference=f"optout-{family_id}",
        )
        return InboundResult(intent.action, True, "suppressed", family_id)

    def _opt_in(self, intent: InboundIntent, *, now: datetime) -> InboundResult:
        family_id = self._family_for_address(intent.from_address)
        if family_id is None:
            return InboundResult(intent.action, False, "unknown_number")
        self.db.execute(
            "UPDATE suppression SET released_utc = ? WHERE family_id = ? AND channel = ?"
            " AND released_utc IS NULL",
            (iso(now), family_id, Channel.SMS),
        )
        # A START keyword is express consent captured by SMS double opt-in.
        self.db.execute(
            """INSERT OR IGNORE INTO consent (consent_id, family_id, channel, purpose,
                   granted_utc, capture_method, capture_evidence, captured_by)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                new_id("con"),
                family_id,
                Channel.SMS,
                ConsentPurpose.REMINDERS,
                iso(now),
                "sms_double_optin",
                f"inbound START from {intent.from_address}",
                "system",
            ),
        )
        if self.audit is not None:
            self.audit.record_event(
                actor_id=f"inbound:{intent.from_address}",
                initiative_id=self.INITIATIVE,
                event_type="consent_granted",
                detail={"channel": Channel.SMS, "method": "sms_double_optin"},
            )
        return InboundResult(intent.action, True, "resubscribed", family_id)

    # -- appointment actions ----------------------------------------------

    def _appointment_for_message(self, message_id: str | None) -> str | None:
        if not message_id:
            return None
        row = self.db.one(
            "SELECT appointment_id FROM message_log WHERE message_id = ?", (message_id,)
        )
        return row["appointment_id"] if row else None

    def _confirm(self, intent: InboundIntent, *, now: datetime) -> InboundResult:
        appointment_id = self._appointment_for_message(intent.reference)
        if appointment_id is None:
            return InboundResult(intent.action, False, "unknown_reference", intent.reference)
        self.db.execute(
            "UPDATE appointment SET status = ?, confirmed_utc = ? WHERE appointment_id = ?"
            " AND status = ?",
            (AppointmentStatus.CONFIRMED, iso(now), appointment_id, AppointmentStatus.SCHEDULED),
        )
        return InboundResult(intent.action, True, "confirmed", appointment_id)

    def _cancel(self, intent: InboundIntent, *, now: datetime) -> InboundResult:
        """Accept a cancellation 24/7 and release the slot.

        This is the highest-value path in the whole module. README I-07: a
        parent who realises at 20:00 that they cannot make tomorrow's 09:20
        cannot currently tell anyone, and a meaningful share of those become
        no-shows. Accepting the cancellation at 20:00 converts a lost slot into
        a backfillable one.
        """
        appointment_id = self._appointment_for_message(intent.reference)
        if appointment_id is None:
            return InboundResult(intent.action, False, "unknown_reference", intent.reference)
        release_id = self.backfill.release_slot(appointment_id, now=now, source="sms")
        if release_id is None:
            return InboundResult(intent.action, True, "cancelled_not_backfillable", appointment_id)
        self.backfill.blast(release_id, now=now)
        return InboundResult(intent.action, True, "cancelled_and_blasted", release_id)

    def _accept(self, intent: InboundIntent, *, now: datetime) -> InboundResult:
        if not intent.reference:
            return InboundResult(intent.action, False, "missing_offer_id")
        try:
            result = self.backfill.accept_offer(intent.reference, now=now)
        except KeyError:
            return InboundResult(intent.action, False, "unknown_offer", intent.reference)
        return InboundResult(
            intent.action,
            result.won,
            result.reason or result.outcome,
            result.appointment_id or intent.reference,
        )

    def _decline(self, intent: InboundIntent, *, now: datetime) -> InboundResult:
        if not intent.reference:
            return InboundResult(intent.action, False, "missing_offer_id")
        try:
            result = self.backfill.decline_offer(intent.reference, now=now)
        except KeyError:
            return InboundResult(intent.action, False, "unknown_offer", intent.reference)
        return InboundResult(intent.action, True, result.outcome, intent.reference)

    def _to_human(self, intent: InboundIntent, *, now: datetime, detail: str) -> InboundResult:
        """Park it for staff. No guessing, no NLP, no auto-reply that pretends.

        Recorded as a message_log row with status 'blocked' and a reason, so the
        volume of free-text inbound is measurable. If that volume is high, the
        answer is a better link in the outbound template, not a chatbot.
        """
        family_id = self._family_for_address(intent.from_address) or f"unmatched:{intent.from_address}"
        self.db.execute(
            """INSERT INTO message_log (message_id, family_id, channel, purpose,
                   template_id, planned_utc, status, block_reason, to_address)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                new_id("msg"),
                family_id,
                Channel.SMS,
                "inbound_freetext",
                "none",
                iso(now),
                "blocked",
                detail,
                intent.from_address,
            ),
        )
        return InboundResult(intent.action, False, detail)
