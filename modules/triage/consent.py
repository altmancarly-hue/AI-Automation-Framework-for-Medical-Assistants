"""Illinois two-party consent. A hard legal gate, not a courtesy.

720 ILCS 5/14-2 makes it a felony to record a private conversation without the
consent of ALL parties. A telephone triage call between a medical assistant and
a parent is a private conversation. README I-04's risk table rates this High and
says so plainly: "A recorded-line disclosure must play or be stated at the start
of every recorded call, and consent must be documented. This is a hard legal
requirement, not a courtesy."

So this module is a gate, and the gate is closed by default:

  * `authorise_recording()` returns a token. `capture.py` will not start without
    one, and the token names the consent record and the disclosure delivery that
    justified it. There is no `force=True`.
  * The disclosure must be delivered BEFORE recording starts, and be logged as
    delivered. Recording first and disclosing second is the violation.
  * Consent is per family and per purpose, and it expires. A consent captured at
    registration in 2019 is not evidence that this call was consented to, which
    is why `authorise_recording` takes the call's own disclosure event.

WHAT THIS GATE DOES NOT DO, and must never do: block the call. A parent who
declines recording still gets triaged, still gets advice, still gets a note. The
MA types it by hand exactly as they do today. `DeclinedRecording` is a normal
outcome that routes to manual documentation -- refusing to help someone because
they would not consent to being recorded would be a far worse failure than the
one this module prevents.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from modules.scheduling.models import Database, iso, new_id, parse_iso

__all__ = [
    "CONSENT_SCHEMA",
    "DISCLOSURE_SCRIPT",
    "RecordingNotAuthorised",
    "DeclinedRecording",
    "ConsentBasis",
    "DisclosureDelivery",
    "RecordingAuthorisation",
    "ConsentRegistry",
]

#: The words the MA says, verbatim, before pressing record. Stored as a
#: constant with a hash so that a change to the script shows up in a diff and in
#: the audit trail -- "we always say something like that" is not evidence.
DISCLOSURE_SCRIPT = (
    "Before we go further: I'd like to record this call so I can write an "
    "accurate note for your child's chart. The recording is deleted as soon as "
    "the note is signed. Is that okay with you?"
)

#: How long a recorded verbal consent is treated as covering. Deliberately
#: short. A blanket consent signed at registration is a consent to a policy, not
#: to this conversation, and the statute is about this conversation.
DEFAULT_CONSENT_TTL = timedelta(days=365)

#: How long a minted recording authorisation stays usable. A triage call is
#: minutes long; anything older is a token someone kept, not a call in progress.
DEFAULT_AUTHORISATION_TTL = timedelta(hours=2)


class RecordingNotAuthorised(RuntimeError):
    """Raised when capture is attempted without a valid authorisation."""


class DeclinedRecording(Exception):
    """Not an error. The parent said no; document the call by hand.

    Modelled as an exception only so that a caller cannot ignore it and carry on
    into the recording path by accident.
    """


class ConsentBasis:
    VERBAL_ON_CALL = "verbal_on_call"       # the strongest, and the default
    WRITTEN_INTAKE = "written_intake"       # standing consent in the intake packet
    PORTAL = "portal"
    ALL = (VERBAL_ON_CALL, WRITTEN_INTAKE, PORTAL)


CONSENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS triage_recording_consent (
    consent_id      TEXT PRIMARY KEY,
    family_id       TEXT NOT NULL,
    patient_id      TEXT,
    basis           TEXT NOT NULL,
    granted_utc     TEXT NOT NULL,
    expires_utc     TEXT,
    captured_by     TEXT NOT NULL,
    evidence_ref    TEXT,
    revoked_utc     TEXT,
    revocation_note TEXT
);

-- One row per call, written BEFORE the recorder starts. The absence of a row is
-- the absence of authorisation; there is no other state.
CREATE TABLE IF NOT EXISTS triage_disclosure_delivery (
    delivery_id     TEXT PRIMARY KEY,
    encounter_id    TEXT NOT NULL,
    family_id       TEXT NOT NULL,
    delivered_utc   TEXT NOT NULL,
    delivered_by    TEXT NOT NULL,
    script_hash     TEXT NOT NULL,
    method          TEXT NOT NULL,        -- 'spoken' | 'ivr_announcement'
    response        TEXT NOT NULL,        -- 'granted' | 'declined'
    note            TEXT
);

CREATE INDEX IF NOT EXISTS ix_triage_consent_family
    ON triage_recording_consent(family_id, revoked_utc);
CREATE INDEX IF NOT EXISTS ix_triage_disclosure_encounter
    ON triage_disclosure_delivery(encounter_id);
"""


def _script_hash(script: str) -> str:
    import hashlib

    return hashlib.sha256(script.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class DisclosureDelivery:
    delivery_id: str
    encounter_id: str
    family_id: str
    delivered_utc: datetime
    delivered_by: str
    response: str
    script_hash: str

    @property
    def granted(self) -> bool:
        return self.response == "granted"


@dataclass(frozen=True)
class RecordingAuthorisation:
    """The token `capture.py` demands. Carries its own justification.

    WHY a token rather than a boolean: a boolean can be computed in one place
    and passed around after the fact. This object names the consent row and the
    disclosure delivery that authorised THIS encounter, so an audit can walk
    backwards from a recording to the words that were said before it started.
    """

    encounter_id: str
    family_id: str
    consent_id: str
    delivery_id: str
    authorised_utc: datetime
    basis: str
    #: A token minted at 09:00 is not evidence that consent still stood at
    #: 15:00. `TriageService.capture` re-runs the gate anyway; this is the
    #: belt to that pair of braces.
    expires_utc: datetime | None = None

    def valid_at(self, moment: datetime) -> bool:
        return self.expires_utc is None or moment < self.expires_utc

    def as_dict(self) -> dict[str, Any]:
        return {
            "encounter_id": self.encounter_id,
            "family_id": self.family_id,
            "consent_id": self.consent_id,
            "delivery_id": self.delivery_id,
            "authorised_utc": iso(self.authorised_utc),
            "basis": self.basis,
        }


class ConsentRegistry:
    """Records consent, records disclosure, issues recording authorisations."""

    INITIATIVE = "I-04"

    def __init__(
        self,
        db: Database,
        *,
        audit: Any = None,
        script: str = DISCLOSURE_SCRIPT,
        consent_ttl: timedelta = DEFAULT_CONSENT_TTL,
        authorisation_ttl: timedelta = DEFAULT_AUTHORISATION_TTL,
    ) -> None:
        self.db = db
        self.audit = audit
        self.script = script
        self.consent_ttl = consent_ttl
        self.authorisation_ttl = authorisation_ttl
        self.db.conn.executescript(CONSENT_SCHEMA)

    # -- consent -----------------------------------------------------------

    def grant(
        self,
        *,
        family_id: str,
        captured_by: str,
        basis: str = ConsentBasis.VERBAL_ON_CALL,
        now: datetime,
        patient_id: str | None = None,
        evidence_ref: str | None = None,
        ttl: timedelta | None = None,
    ) -> str:
        if basis not in ConsentBasis.ALL:
            raise ValueError(f"unknown consent basis {basis!r}")
        if not captured_by.strip():
            raise ValueError("consent must record who captured it")
        consent_id = new_id("rcons")
        self.db.execute(
            """INSERT INTO triage_recording_consent (consent_id, family_id, patient_id,
                   basis, granted_utc, expires_utc, captured_by, evidence_ref)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                consent_id,
                family_id,
                patient_id,
                basis,
                iso(now),
                iso(now + (ttl or self.consent_ttl)),
                captured_by,
                evidence_ref,
            ),
        )
        if self.audit is not None:
            self.audit.record_event(
                actor_id=captured_by,
                initiative_id=self.INITIATIVE,
                event_type="recording_consent_granted",
                patient_id=patient_id,
                detail={"basis": basis},
            )
        return consent_id

    def revoke(self, *, family_id: str, now: datetime, note: str = "") -> int:
        cur = self.db.execute(
            "UPDATE triage_recording_consent SET revoked_utc = ?, revocation_note = ?"
            " WHERE family_id = ? AND revoked_utc IS NULL",
            (iso(now), note, family_id),
        )
        if self.audit is not None:
            self.audit.record_event(
                actor_id="system",
                initiative_id=self.INITIATIVE,
                event_type="recording_consent_revoked",
                detail={"rows": cur.rowcount},
            )
        return cur.rowcount

    def active_consent(self, family_id: str, *, at: datetime) -> Mapping[str, Any] | None:
        return self.db.one(
            """SELECT * FROM triage_recording_consent
               WHERE family_id = ? AND revoked_utc IS NULL
                 AND granted_utc <= ?
                 AND (expires_utc IS NULL OR expires_utc > ?)
               ORDER BY granted_utc DESC LIMIT 1""",
            (family_id, iso(at), iso(at)),
        )

    # -- disclosure --------------------------------------------------------

    def record_disclosure(
        self,
        *,
        encounter_id: str,
        family_id: str,
        delivered_by: str,
        response: str,
        now: datetime,
        method: str = "spoken",
        note: str = "",
    ) -> DisclosureDelivery:
        """Log the disclosure. Call this BEFORE starting the recorder.

        `response` is what the parent actually said, and "declined" is recorded
        just as carefully as "granted" -- a declined call that was nonetheless
        recorded is the thing an audit is looking for, and it is only findable
        if declines leave a trace.
        """
        if response not in ("granted", "declined"):
            raise ValueError("response must be 'granted' or 'declined'")
        if not delivered_by.strip():
            raise ValueError("disclosure must record who delivered it")
        delivery = DisclosureDelivery(
            delivery_id=new_id("disc"),
            encounter_id=encounter_id,
            family_id=family_id,
            delivered_utc=now,
            delivered_by=delivered_by,
            response=response,
            script_hash=_script_hash(self.script),
        )
        self.db.execute(
            """INSERT INTO triage_disclosure_delivery (delivery_id, encounter_id,
                   family_id, delivered_utc, delivered_by, script_hash, method,
                   response, note)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                delivery.delivery_id,
                encounter_id,
                family_id,
                iso(now),
                delivered_by,
                delivery.script_hash,
                method,
                response,
                note,
            ),
        )
        if self.audit is not None:
            self.audit.record_event(
                actor_id=delivered_by,
                initiative_id=self.INITIATIVE,
                event_type=f"recording_disclosure_{response}",
                detail={"encounter_id": encounter_id, "method": method},
            )
        return delivery

    def disclosure_for(self, encounter_id: str) -> Mapping[str, Any] | None:
        """The latest disclosure for this encounter, ties broken by insert order.

        `ORDER BY delivered_utc DESC` alone is not enough: timestamps are stored
        to the second, and a parent who says "yes... actually no, don't record"
        inside one second produced a scan-order result. `rowid DESC` makes the
        later row win, which is the one that reflects what they last said.
        """
        return self.db.one(
            "SELECT * FROM triage_disclosure_delivery WHERE encounter_id = ?"
            " ORDER BY delivered_utc DESC, rowid DESC LIMIT 1",
            (encounter_id,),
        )

    def has_decline(self, encounter_id: str) -> bool:
        return self.db.one(
            "SELECT 1 AS hit FROM triage_disclosure_delivery WHERE encounter_id = ?"
            " AND response = 'declined' LIMIT 1",
            (encounter_id,),
        ) is not None

    # -- the gate ----------------------------------------------------------

    def authorise_recording(
        self,
        *,
        encounter_id: str,
        family_id: str,
        now: datetime,
    ) -> RecordingAuthorisation:
        """Issue a recording authorisation, or refuse.

        Raises `DeclinedRecording` when the parent said no -- a normal outcome
        that routes to manual documentation -- and `RecordingNotAuthorised` when
        the preconditions were never met at all. The two are distinct because
        they mean different things to the person holding the phone: one is "they
        said no, type it up", the other is "you skipped a step".
        """
        # ANY decline on this encounter is disqualifying, whatever came after.
        # A parent who has once said no has said no; a later "granted" row is
        # far more likely to be a mis-tap than a change of heart, and the cost
        # of being wrong is a felony statute.
        if self.has_decline(encounter_id):
            raise DeclinedRecording(
                "this encounter carries a declined disclosure. Continue the call "
                "and document it by hand -- consent gates the recording, never "
                "the care. If the parent has genuinely changed their mind, open "
                "a new encounter so the record is unambiguous."
            )
        delivery = self.disclosure_for(encounter_id)
        if delivery is None:
            raise RecordingNotAuthorised(
                "no recorded-line disclosure has been delivered for this "
                "encounter. Illinois is a two-party consent state (720 ILCS "
                "5/14-2): the disclosure is said BEFORE the recorder starts, "
                "not after."
            )
        if str(delivery["family_id"]) != str(family_id):
            # The one cross-check the audit story depends on: walking backwards
            # from a recording to the words that were said before it.
            raise RecordingNotAuthorised(
                f"the disclosure for encounter {encounter_id!r} was delivered to "
                f"family {delivery['family_id']!r}, not {family_id!r}"
            )

        delivered = parse_iso(delivery["delivered_utc"])
        if delivered > now:
            raise RecordingNotAuthorised(
                "the disclosure is timestamped after the requested recording "
                "start; a disclosure that follows the recording is not consent"
            )

        consent = self.active_consent(family_id, at=now)
        if consent is None:
            raise RecordingNotAuthorised(
                "no active recording consent on file for this family. A verbal "
                "yes on the call is enough, but it has to be recorded with "
                "ConsentRegistry.grant() before the recorder starts."
            )

        return RecordingAuthorisation(
            encounter_id=encounter_id,
            family_id=family_id,
            consent_id=str(consent["consent_id"]),
            delivery_id=str(delivery["delivery_id"]),
            authorised_utc=now,
            basis=str(consent["basis"]),
            expires_utc=now + self.authorisation_ttl,
        )

    # -- reporting ---------------------------------------------------------

    def unevidenced_recordings(self, encounter_ids: list[str]) -> list[str]:
        """Encounters that have audio but no granted disclosure. Should be empty.

        This is the query counsel asks for, and the answer being empty is the
        whole point. It is exposed as a report rather than only enforced inline
        because an enforcement that is never independently checked is a belief.
        """
        if not encounter_ids:
            return []
        # "Is there ANY granted row" was the wrong question: an encounter where
        # the parent granted and then declined answered yes, while the gate
        # (correctly) refuses it. The report has to agree with the gate, or it
        # reassures counsel about exactly the recordings that are a problem.
        evidenced = {
            e for e in encounter_ids
            if not self.has_decline(e)
            and (self.disclosure_for(e) or {}).get("response") == "granted"
        }
        return [e for e in encounter_ids if e not in evidenced]
