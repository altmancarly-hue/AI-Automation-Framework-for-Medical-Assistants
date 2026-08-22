"""The encounter state machine. What may happen, and in what order.

The ordering constraints in I-04 are all safety constraints, and every one of
them is a "before", so they are enforced by a state machine rather than by
whichever caller happens to be holding the object:

    created -> disclosed -> recording -> transcribed -> drafted -> signed
                    |
                    +-> manual  (the parent declined; the MA types the note)

  * The disclosure comes before the recorder. Illinois two-party consent.
  * The transcript comes before the draft. Nothing is structured from nothing.
  * The taps come before the render. `TapsIncomplete` is not recoverable by the
    system, only by a human tapping.
  * The signature comes before the audio deletion, and the deletion is not
    optional afterwards.

`manual` is a first-class terminal state, not an error path. A parent who
declines recording gets exactly the service they get today, and the practice can
count how often that happens -- which it cannot do if declining just means
nobody clicked anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence

from nsp_core.llm import SchemaViolation
from modules.scheduling.models import Database, iso, new_id, parse_iso, to_local

from .capture import AudioRecording, Recorder, Transcriber, Transcript
from .consent import (
    ConsentRegistry,
    DeclinedRecording,
    RecordingAuthorisation,
    RecordingNotAuthorised,
)
from .lifecycle import AudioLifecycle
from .protocols import MATaps, ProtocolRegistry, TapsIncomplete
from .render import (
    FollowUpTask,
    build_followup_task,
    render_draft,
    render_note,
    signature_block,
)
from .review import NoteReviewer, SignedNote
from .structure import NoteStructurer, StructuredNote

__all__ = [
    "ENCOUNTER_SCHEMA",
    "EncounterState",
    "IllegalTransition",
    "TriageEncounter",
    "TriageService",
]


class EncounterState:
    CREATED = "created"
    DISCLOSED = "disclosed"
    RECORDING = "recording"
    TRANSCRIBED = "transcribed"
    DRAFTED = "drafted"
    SIGNED = "signed"
    REJECTED = "rejected"
    MANUAL = "manual"
    ALL = (CREATED, DISCLOSED, RECORDING, TRANSCRIBED, DRAFTED, SIGNED, REJECTED, MANUAL)
    TERMINAL = (SIGNED, REJECTED, MANUAL)


_ALLOWED: Mapping[str, tuple[str, ...]] = {
    EncounterState.CREATED: (EncounterState.DISCLOSED, EncounterState.MANUAL),
    EncounterState.DISCLOSED: (EncounterState.RECORDING, EncounterState.MANUAL),
    EncounterState.RECORDING: (EncounterState.TRANSCRIBED, EncounterState.MANUAL),
    EncounterState.TRANSCRIBED: (EncounterState.DRAFTED, EncounterState.MANUAL),
    EncounterState.DRAFTED: (
        EncounterState.SIGNED,
        EncounterState.REJECTED,
        EncounterState.MANUAL,
    ),
    EncounterState.SIGNED: (),
    EncounterState.REJECTED: (),
    EncounterState.MANUAL: (),
}


class IllegalTransition(RuntimeError):
    """Raised on any transition the state machine does not allow."""


ENCOUNTER_SCHEMA = """
CREATE TABLE IF NOT EXISTS triage_encounter (
    encounter_id          TEXT PRIMARY KEY,
    patient_id            TEXT NOT NULL,
    family_id             TEXT NOT NULL,
    opened_by             TEXT NOT NULL,
    opened_utc            TEXT NOT NULL,
    state                 TEXT NOT NULL,
    call_started_utc      TEXT,
    call_ended_utc        TEXT,
    transcript_text       TEXT,
    transcript_sha256     TEXT,
    transcript_purged_utc TEXT,
    transcriber_model     TEXT,
    diarized              INTEGER NOT NULL DEFAULT 0,
    protocol_id           TEXT,
    disposition_id        TEXT,
    tapped_by             TEXT,
    supervising_pro_id    TEXT,
    draft_text            TEXT,
    proposed_text         TEXT,
    -- The StructuredNote itself, so `resume` can rebuild it after a crash
    -- without a second inference. No PHI class that draft_text does not
    -- already carry, and it is purged with the rest on the terminal paths.
    note_json             TEXT,
    tapped_utc            TEXT,
    signed_text           TEXT,
    signed_by             TEXT,
    signed_utc            TEXT,
    edit_distance         INTEGER,
    edit_ratio            REAL,
    manual_reason         TEXT,
    closed_utc            TEXT
);

-- README I-04 counts recovered follow-ups as real money and names the failure
-- as "patient told 'call back if worse' and nobody closes the loop". A task
-- that lives only on an in-memory object closes no loop.
CREATE TABLE IF NOT EXISTS triage_followup (
    task_id        TEXT PRIMARY KEY,
    encounter_id   TEXT NOT NULL,
    patient_id     TEXT NOT NULL,
    created_utc    TEXT NOT NULL,
    due_utc        TEXT,
    timeframe_text TEXT,
    description    TEXT NOT NULL,
    assigned_to    TEXT NOT NULL,
    completed_utc  TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_triage_followup_encounter
    ON triage_followup(encounter_id) WHERE completed_utc IS NULL;
CREATE INDEX IF NOT EXISTS ix_triage_followup_due ON triage_followup(due_utc, completed_utc);
CREATE INDEX IF NOT EXISTS ix_triage_encounter_state
    ON triage_encounter(state, opened_utc);
CREATE INDEX IF NOT EXISTS ix_triage_encounter_patient
    ON triage_encounter(patient_id, opened_utc);
"""


@dataclass
class TriageEncounter:
    encounter_id: str
    patient_id: str
    family_id: str
    opened_by: str
    opened_utc: datetime
    state: str = EncounterState.CREATED
    authorisation: RecordingAuthorisation | None = None
    recording: AudioRecording | None = None
    transcript: Transcript | None = None
    note: StructuredNote | None = None
    taps: MATaps | None = None
    draft_text: str = ""
    #: The chart note the machine proposed. This is the edit-diff BASELINE.
    proposed_text: str = ""
    signed: SignedNote | None = None
    followup: FollowUpTask | None = None

    def require(self, *states: str) -> None:
        if self.state not in states:
            raise IllegalTransition(
                f"encounter {self.encounter_id} is {self.state!r}; this step "
                f"requires one of {states}"
            )

    def move_to(self, state: str) -> None:
        if state not in _ALLOWED.get(self.state, ()):
            raise IllegalTransition(
                f"cannot move encounter {self.encounter_id} from {self.state!r} "
                f"to {state!r}"
            )
        self.state = state


class TriageService:
    """Wires consent, capture, structuring, review and retention together."""

    INITIATIVE = "I-04"

    def __init__(
        self,
        db: Database,
        *,
        consent: ConsentRegistry,
        recorder: Recorder,
        transcriber: Transcriber,
        structurer: NoteStructurer,
        registry: ProtocolRegistry,
        reviewer: NoteReviewer,
        lifecycle: AudioLifecycle,
        audit: Any = None,
    ) -> None:
        self.db = db
        self.consent = consent
        self.recorder = recorder
        self.transcriber = transcriber
        self.structurer = structurer
        self.registry = registry
        self.reviewer = reviewer
        self.lifecycle = lifecycle
        self.audit = audit
        self.db.conn.executescript(ENCOUNTER_SCHEMA)

    # -- lifecycle ---------------------------------------------------------

    def open(
        self, *, patient_id: str, family_id: str, opened_by: str, now: datetime
    ) -> TriageEncounter:
        encounter = TriageEncounter(
            encounter_id=new_id("enc"),
            patient_id=patient_id,
            family_id=family_id,
            opened_by=opened_by,
            opened_utc=now,
        )
        self.db.execute(
            """INSERT INTO triage_encounter (encounter_id, patient_id, family_id,
                   opened_by, opened_utc, state)
               VALUES (?,?,?,?,?,?)""",
            (
                encounter.encounter_id,
                patient_id,
                family_id,
                opened_by,
                iso(now),
                encounter.state,
            ),
        )
        return encounter

    #: Every column `_set` is allowed to write. `_set` interpolates its keyword
    #: names into SQL, and while every call site today passes a literal, an
    #: interpolated identifier one refactor away from a request parameter is
    #: how injection arrives. The check costs a set lookup.
    _WRITABLE_COLUMNS = frozenset(
        {
            "state", "call_started_utc", "call_ended_utc", "transcript_text",
            "transcript_sha256", "transcript_purged_utc", "transcriber_model",
            "diarized", "protocol_id", "disposition_id", "tapped_by",
            "supervising_pro_id", "draft_text", "proposed_text", "signed_text",
            "note_json", "tapped_utc",
            "signed_by", "signed_utc", "edit_distance", "edit_ratio",
            "manual_reason", "closed_utc",
        }
    )

    def _set(self, encounter: TriageEncounter, **columns: Any) -> None:
        columns["state"] = encounter.state
        unknown = sorted(set(columns) - self._WRITABLE_COLUMNS)
        if unknown:
            raise ValueError(
                f"refusing to write unknown encounter column(s) {unknown}; "
                "add them to _WRITABLE_COLUMNS deliberately"
            )
        assignments = ", ".join(f"{k} = ?" for k in columns)
        self.db.execute(
            f"UPDATE triage_encounter SET {assignments} WHERE encounter_id = ?",
            (*columns.values(), encounter.encounter_id),
        )

    # -- re-entry ----------------------------------------------------------

    def resume(self, encounter_id: str) -> TriageEncounter:
        """Rebuild an encounter from the database after a crash or a restart.

        WHY this has to exist: without it, an encounter only lived on the
        in-memory object the caller was holding. An MA whose browser reloaded
        mid-call, or a workstation that rebooted, left a row in `state =
        'recording'` and a WAV on disk that no code path could reach -- the
        audio sat there until the 24-hour sweep, and the call went undocumented,
        which README I-04 calls "indefensible".

        The state, the taps, the draft, the proposed chart text and the
        structured note all come back, so a resumed encounter can be reviewed
        and signed -- which is the entire point, since the alternative outcome
        is the MA retyping a note the machine already wrote. The note is
        DESERIALISED from the row rather than re-derived: re-running the model
        would be a second inference against a transcript that may already have
        been purged, and it would produce a different set of drops and flags
        from the ones the reviewer is looking at.

        The `authorisation` and `recording` objects are deliberately NOT
        restored. The authorisation has a TTL and has to be re-taken from
        `ConsentRegistry`, so `capture()` on a resumed encounter re-runs the
        consent gate from scratch -- which is right, because consent can have
        been revoked in the meantime.
        """
        row = self.db.one(
            "SELECT * FROM triage_encounter WHERE encounter_id = ?", (encounter_id,)
        )
        if row is None:
            raise KeyError(f"no triage encounter {encounter_id!r}")
        encounter = TriageEncounter(
            encounter_id=str(row["encounter_id"]),
            patient_id=str(row["patient_id"]),
            family_id=str(row["family_id"]),
            opened_by=str(row["opened_by"]),
            opened_utc=parse_iso(str(row["opened_utc"])),
            state=str(row["state"]),
        )
        encounter.draft_text = str(row["draft_text"] or "")
        encounter.proposed_text = str(row["proposed_text"] or "")
        if row["note_json"]:
            encounter.note = StructuredNote.from_json(str(row["note_json"]))
        if row["protocol_id"] and row["disposition_id"]:
            encounter.taps = MATaps(
                protocol_id=str(row["protocol_id"]),
                disposition_id=str(row["disposition_id"]),
                tapped_by=str(row["tapped_by"] or ""),
                # The real tap time, not the call-open time. `MATaps.as_dict`
                # publishes this as the delegation timestamp under 225 ILCS
                # 60/54.2, and a tap made seven minutes into a call must not
                # come back stamped at the moment the call was answered.
                tapped_utc=parse_iso(str(row["tapped_utc"] or row["opened_utc"])),
                supervising_professional_id=(
                    str(row["supervising_pro_id"]) if row["supervising_pro_id"] else None
                ),
            )
        return encounter

    def abandoned(self, *, now: datetime, older_than: timedelta) -> list[dict[str, Any]]:
        """Encounters stuck short of a terminal state. The re-entry work queue.

        The counterpart to `resume`: something has to notice the stuck rows
        before anyone can resume them, and "the MA remembers" is not a control.
        """
        cutoff = iso(now - older_than)
        return self.db.all(
            "SELECT encounter_id, patient_id, state, opened_utc FROM triage_encounter"
            " WHERE state NOT IN (?,?,?) AND opened_utc <= ? ORDER BY opened_utc",
            (
                EncounterState.SIGNED,
                EncounterState.REJECTED,
                EncounterState.MANUAL,
                cutoff,
            ),
        )

    # -- consent -----------------------------------------------------------

    def deliver_disclosure(
        self,
        encounter: TriageEncounter,
        *,
        response: str,
        delivered_by: str,
        now: datetime,
        method: str = "spoken",
    ) -> None:
        """Say the words, log them, and branch on the answer."""
        encounter.require(EncounterState.CREATED)
        self.consent.record_disclosure(
            encounter_id=encounter.encounter_id,
            family_id=encounter.family_id,
            delivered_by=delivered_by,
            response=response,
            now=now,
            method=method,
        )
        if response == "declined":
            encounter.move_to(EncounterState.MANUAL)
            self._set(
                encounter,
                manual_reason="parent declined recording",
                closed_utc=iso(now),
            )
            return
        encounter.move_to(EncounterState.DISCLOSED)
        self._set(encounter)

    def authorise(self, encounter: TriageEncounter, *, now: datetime) -> RecordingAuthorisation:
        encounter.require(EncounterState.DISCLOSED)
        authorisation = self.consent.authorise_recording(
            encounter_id=encounter.encounter_id,
            family_id=encounter.family_id,
            now=now,
        )
        encounter.authorisation = authorisation
        return authorisation

    def fall_back_to_manual(
        self, encounter: TriageEncounter, *, reason: str, now: datetime
    ) -> None:
        """Abandon the assisted path. Always available, from any live state.

        WHY it is always available: every failure in this pipeline -- declined
        consent, a dead GPU, a schema violation, an MA who does not trust the
        draft -- has the same correct answer, which is that the MA writes the
        note the way they did last year. A system where the fallback is hard to
        reach is a system that pressures people into signing drafts.
        """
        if encounter.state in EncounterState.TERMINAL:
            raise IllegalTransition(
                f"encounter {encounter.encounter_id} is already {encounter.state!r}"
            )
        encounter.move_to(EncounterState.MANUAL)
        self._set(encounter, manual_reason=reason, closed_utc=iso(now))
        self.lifecycle.delete_for_encounter(
            encounter.encounter_id, now=now, reason="manual_documentation"
        )
        self._forget_transcript(encounter, now=now)
        if self.audit is not None:
            self.audit.record_event(
                actor_id=encounter.opened_by,
                initiative_id=self.INITIATIVE,
                event_type="triage_manual_fallback",
                patient_id=encounter.patient_id,
                detail={"reason": reason},
            )

    # -- capture and transcription ----------------------------------------

    def capture(
        self,
        encounter: TriageEncounter,
        *,
        audio_bytes: bytes,
        started_utc: datetime,
        ended_utc: datetime,
        now: datetime,
    ) -> AudioRecording:
        encounter.require(EncounterState.DISCLOSED)
        if encounter.authorisation is None:
            raise RecordingNotAuthorised(
                "call authorise() before capture(); the disclosure and the "
                "consent record are what make the recording lawful"
            )
        # Re-evaluate at the moment audio hits disk. A token minted at 09:00 is
        # not evidence that consent still stood at 15:00, and a parent who
        # revokes between the two must not be recorded. Cheap; the gate is a
        # couple of indexed reads.
        encounter.authorisation = self.consent.authorise_recording(
            encounter_id=encounter.encounter_id,
            family_id=encounter.family_id,
            now=now,
        )
        recording = self.recorder.capture(
            authorisation=encounter.authorisation,
            encounter_id=encounter.encounter_id,
            audio_bytes=audio_bytes,
            started_utc=started_utc,
            ended_utc=ended_utc,
        )
        self.lifecycle.register(recording, now=now)
        encounter.recording = recording
        encounter.move_to(EncounterState.RECORDING)
        self._set(
            encounter,
            call_started_utc=iso(started_utc),
            call_ended_utc=iso(ended_utc),
        )
        return recording

    def transcribe(self, encounter: TriageEncounter) -> Transcript:
        encounter.require(EncounterState.RECORDING)
        if encounter.recording is None:
            raise IllegalTransition("no recording to transcribe")
        transcript = self.transcriber.transcribe(encounter.recording)
        encounter.transcript = transcript
        encounter.move_to(EncounterState.TRANSCRIBED)
        self._set(
            encounter,
            transcript_text=transcript.text,
            transcript_sha256=transcript.sha256,
            transcriber_model=f"{transcript.model_id}@{transcript.model_version}",
            diarized=int(transcript.diarized),
        )
        return transcript

    # -- drafting ----------------------------------------------------------

    def draft(
        self,
        encounter: TriageEncounter,
        *,
        taps: MATaps,
        patient_label: str,
        now: datetime,
        age_months: int | None = None,
        user_id: str | None = None,
    ) -> str:
        """Structure the transcript and render the reviewer's draft.

        The taps are validated FIRST. Running the model and then discovering
        that the MA never tapped a disposition wastes the inference and, worse,
        produces a structured note sitting in memory next to an unrenderable
        state -- which is how a "just render it without the disposition"
        shortcut gets added later.
        """
        encounter.require(EncounterState.TRANSCRIBED)
        if encounter.transcript is None:
            raise IllegalTransition("no transcript to structure")
        self.registry.validate(taps, age_months=age_months)

        note = self.structurer.structure(
            encounter.transcript,
            patient_id=encounter.patient_id,
            user_id=user_id or taps.tapped_by,
        )
        call_time = (
            encounter.recording.started_utc
            if encounter.recording
            else encounter.opened_utc
        )
        # Two documents from one note. The draft is what the reviewer reads --
        # banners, removals, model footer. The proposed text is the chart note
        # the machine would file, and it is the ONLY thing the edit diff may be
        # taken against.
        draft_text = render_draft(
            note, taps, self.registry, patient_label=patient_label, call_time=call_time
        )
        proposed_text = render_note(
            note, taps, self.registry, patient_label=patient_label, call_time=call_time
        )
        encounter.note = note
        encounter.taps = taps
        encounter.draft_text = draft_text
        encounter.proposed_text = proposed_text
        encounter.move_to(EncounterState.DRAFTED)
        self._set(
            encounter,
            protocol_id=taps.protocol_id,
            disposition_id=taps.disposition_id,
            tapped_by=taps.tapped_by,
            supervising_pro_id=taps.supervising_professional_id,
            draft_text=draft_text,
            proposed_text=proposed_text,
            note_json=note.to_json(),
            tapped_utc=iso(taps.tapped_utc),
        )
        return draft_text

    # -- signing -----------------------------------------------------------

    def sign(
        self,
        encounter: TriageEncounter,
        *,
        signed_by: str,
        now: datetime,
        patient_label: str,
        final_text: str | None = None,
        acknowledged_drops: bool = False,
        review_seconds: float | None = None,
        followup_assignee: str | None = None,
    ) -> SignedNote:
        """Sign, then delete the audio and purge the transcript. In that order.

        The diff is taken between `encounter.proposed_text` -- the chart note the
        machine wrote -- and `final_text`, the chart note the human is signing.
        The signature block is appended AFTERWARDS, so an untouched signature
        produces an edit distance of exactly zero and trips the rubber-stamp
        alarm, which is the entire reason README I-04 calls the edit-diff log
        "not optional".

        Deletion is not a separate step a caller might forget: signing is the
        event that ends the audio's purpose, so it happens here.
        """
        encounter.require(EncounterState.DRAFTED)
        # A real check, not an assert: `python -O` strips asserts, and the
        # failure mode there is an AttributeError on None halfway through a
        # signature rather than a refusal before one.
        if encounter.note is None or encounter.taps is None:
            raise IllegalTransition(
                f"encounter {encounter.encounter_id} has no structured note or "
                "no taps to sign; re-open it with TriageService.resume() or "
                "draft it first"
            )

        # Claim the transition BEFORE any audit write. `move_to` is a
        # check-then-act on an in-memory attribute; two threads calling sign()
        # both reach `reviewer.sign` and both write an append-only, permanent
        # audit record attesting that they signed it. Only one of them did.
        claimed = self.db.execute(
            "UPDATE triage_encounter SET state = ? WHERE encounter_id = ? AND state = ?",
            (EncounterState.SIGNED, encounter.encounter_id, EncounterState.DRAFTED),
        )
        if claimed.rowcount != 1:
            raise IllegalTransition(
                f"encounter {encounter.encounter_id} was already signed or moved "
                "by another session"
            )
        encounter.state = EncounterState.SIGNED

        body = final_text if final_text is not None else encounter.proposed_text
        try:
            signed = self.reviewer.sign(
                note=encounter.note,
                taps=encounter.taps,
                baseline_text=encounter.proposed_text,
                final_text=body,
                signed_by=signed_by,
                now=now,
                patient_id=encounter.patient_id,
                review_seconds=review_seconds,
                acknowledged_drops=acknowledged_drops,
            )
        except Exception:
            # Give the claim back. A refused signature must leave the encounter
            # signable, not stranded in SIGNED with no note.
            self.db.execute(
                "UPDATE triage_encounter SET state = ? WHERE encounter_id = ?",
                (EncounterState.DRAFTED, encounter.encounter_id),
            )
            encounter.state = EncounterState.DRAFTED
            raise

        chart_text = body + signature_block(signed_by, now)
        encounter.signed = signed
        self._set(
            encounter,
            signed_text=chart_text,
            signed_by=signed_by,
            signed_utc=iso(now),
            edit_distance=signed.edit_distance,
            edit_ratio=signed.edit_ratio,
            closed_utc=iso(now),
        )

        encounter.followup = build_followup_task(
            encounter.note,
            patient_id=encounter.patient_id,
            now=now,
            assigned_to=followup_assignee or signed_by,
        )
        if encounter.followup is not None:
            self._persist_followup(encounter.followup, now=now)

        # The audio has done its job. README I-04: retention creates a discovery
        # surface with no clinical benefit.
        self.lifecycle.on_signature(encounter.encounter_id, now=now)
        self._forget_transcript(encounter, now=now)
        return signed

    def _persist_followup(self, task: FollowUpTask, *, now: datetime) -> str:
        task_id = new_id("task")
        self.db.execute(
            """INSERT OR REPLACE INTO triage_followup (task_id, encounter_id,
                   patient_id, created_utc, due_utc, timeframe_text, description,
                   assigned_to)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                task_id,
                task.encounter_id,
                task.patient_id,
                iso(now),
                iso(task.due_utc) if task.due_utc else None,
                task.timeframe_text,
                task.description,
                task.assigned_to,
            ),
        )
        return task_id

    def _forget_transcript(self, encounter: TriageEncounter, *, now: datetime) -> None:
        """Purge the stored transcript AND drop the in-memory copy.

        Both halves. A transcript purged from the database but still hanging off
        a long-lived service object is a transcript, and the argument for
        deleting it -- a verbatim PHI-bearing copy of the call with weaker
        controls than the chart -- does not care where it is being kept.
        """
        self.lifecycle.purge_transcript(encounter.encounter_id, now=now)
        encounter.transcript = None

    def reject(
        self, encounter: TriageEncounter, *, rejected_by: str, reason: str, now: datetime
    ) -> None:
        encounter.require(EncounterState.DRAFTED)
        if encounter.note is None:
            raise IllegalTransition(
                f"encounter {encounter.encounter_id} has no structured note to "
                "reject"
            )
        self.reviewer.reject(
            note=encounter.note,
            rejected_by=rejected_by,
            reason=reason,
            baseline_text=encounter.proposed_text or encounter.draft_text,
            now=now,
            patient_id=encounter.patient_id,
        )
        encounter.move_to(EncounterState.REJECTED)
        self._set(encounter, manual_reason=reason, closed_utc=iso(now))
        self.lifecycle.delete_for_encounter(
            encounter.encounter_id, now=now, reason="draft_rejected"
        )
        self._forget_transcript(encounter, now=now)

    # -- reporting ---------------------------------------------------------

    def undocumented(self, *, now: datetime, older_than_hours: int = 8) -> list[dict[str, Any]]:
        """Live encounters with no signed note. README I-04's core failure mode.

        "Note not written at all during a busy session... an undocumented triage
        call that preceded a bad outcome is indefensible." This is the queue
        that stops that from being discovered months later.
        """
        from datetime import timedelta

        cutoff = iso(now - timedelta(hours=older_than_hours))
        return self.db.all(
            """SELECT encounter_id, patient_id, opened_by, opened_utc, state
               FROM triage_encounter
               WHERE state NOT IN ('signed','manual','rejected') AND opened_utc <= ?
               ORDER BY opened_utc""",
            (cutoff,),
        )

    def open_followups(self, *, now: datetime | None = None) -> list[dict[str, Any]]:
        """Follow-up tasks nobody has closed. The loop-closure work queue."""
        rows = self.db.all(
            "SELECT * FROM triage_followup WHERE completed_utc IS NULL"
            " ORDER BY COALESCE(due_utc, '9999'), created_utc"
        )
        return rows

    def complete_followup(self, encounter_id: str, *, now: datetime) -> int:
        cur = self.db.execute(
            "UPDATE triage_followup SET completed_utc = ? WHERE encounter_id = ?"
            " AND completed_utc IS NULL",
            (iso(now), encounter_id),
        )
        return cur.rowcount

    def documentation_kpis(self, *, since: datetime | None = None) -> dict[str, Any]:
        """README 10.2: triage calls documented same-shift, note completion time."""
        clause = " WHERE opened_utc >= ?" if since else ""
        params = (iso(since),) if since else ()
        rows = self.db.all(
            "SELECT state, opened_utc, signed_utc, edit_ratio, disposition_id"
            f" FROM triage_encounter{clause}",
            params,
        )
        total = len(rows)
        signed = [r for r in rows if r["state"] == "signed"]
        manual = [r for r in rows if r["state"] == "manual"]
        durations = [
            (parse_iso(r["signed_utc"]) - parse_iso(r["opened_utc"])).total_seconds()
            for r in signed
            if r["signed_utc"]
        ]
        durations.sort()
        median = _median(durations)
        # A rejected draft is a note the MA wrote by hand. It counts as
        # documented; only a call still sitting open does not.
        rejected = [r for r in rows if r["state"] == "rejected"]
        documented = len(signed) + len(manual) + len(rejected)
        return {
            "encounters": total,
            "signed": len(signed),
            "documented_by_hand": len(manual) + len(rejected),
            "documented_rate": (documented / total) if total else None,
            "documented_rate_target": 0.98,
            "median_seconds_to_signature": round(median, 1) if median is not None else None,
            "median_target_seconds": 90.0,
            "by_disposition": _counts(r["disposition_id"] for r in signed),
        }


def _median(values: Sequence[float]) -> float | None:
    """The real median. `values[len//2]` is the upper-middle for even counts."""
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _counts(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value or "unrecorded")
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))
