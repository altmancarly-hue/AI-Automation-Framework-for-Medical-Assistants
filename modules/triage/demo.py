"""Runnable end-to-end demonstration: `make demo-i04`.

Runs three synthetic calls through the whole pipeline -- disclosure, capture,
transcription, structuring, drafting, signing, audio deletion -- then shows the
consent gate refusing an undisclosed recording, the grounding check removing
invented advice, re-entry into an encounter after a crash, the retention sweep,
and the edit-rate alarm. No network, no model, no PHI, no real audio.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from modules.scheduling.models import Database
from nsp_core.audit import AuditLog
from nsp_core.llm import EchoTransport, LLMClient

from .capture import Recorder, ScriptedTranscriber, silent_wav
from .consent import ConsentRegistry, DeclinedRecording, RecordingNotAuthorised
from .encounter import TriageService
from .fixtures import CALL_TIME, by_name
from .lifecycle import AudioLifecycle
from .protocols import MATaps, ProtocolRegistry
from .render import build_followup_task
from .review import NoteReviewer
from .structure import NoteStructurer


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="nsp-i04-demo-"))
    db = Database(workdir / "practice.sqlite3")
    audit = AuditLog(workdir / "audit.sqlite3", hmac_key=b"demo-key")
    # Explicit opt-in: the shipped identifiers are placeholders and
    # ProtocolRegistry refuses them by default.
    registry = ProtocolRegistry.load(allow_placeholder=True)

    cases = [
        by_name("toddler_fever_home_care"),
        by_name("model_invents_advice"),
        by_name("newborn_fever_ed_now"),
    ]
    transcriber = ScriptedTranscriber()
    client = LLMClient(EchoTransport([c.model_response for c in cases]))
    consent = ConsentRegistry(db, audit=audit)
    lifecycle = AudioLifecycle(db, audit=audit)
    service = TriageService(
        db,
        consent=consent,
        recorder=Recorder(workdir / "audio"),
        transcriber=transcriber,
        structurer=NoteStructurer(client, audit=audit),
        registry=registry,
        reviewer=NoteReviewer(audit=audit, registry=registry),
        lifecycle=lifecycle,
        audit=audit,
    )

    print("=" * 78)
    print("I-04 demo - telephone triage documentation")
    print(f"workspace: {workdir}")
    print(f"protocol library: {registry.source} {registry.library_version}")
    if registry.is_placeholder:
        print("  !! placeholder protocol identifiers - NOT deployable")
    print("=" * 78)

    now = CALL_TIME
    signed_notes = []
    for index, case in enumerate(cases):
        print(f"\n-- call {index + 1}: {case.description}")
        encounter = service.open(
            patient_id=case.patient_id, family_id=case.family_id,
            opened_by="ma_jess", now=now,
        )
        transcriber.transcripts[encounter.encounter_id] = case.transcript_payload
        transcriber.diarized = case.diarized

        if index == 0:
            # Show the gate before consent exists.
            try:
                consent.authorise_recording(
                    encounter_id=encounter.encounter_id,
                    family_id=case.family_id, now=now,
                )
            except RecordingNotAuthorised as exc:
                print(f"   consent gate: {str(exc).splitlines()[0]}")

        consent.grant(family_id=case.family_id, captured_by="ma_jess", now=now)
        service.deliver_disclosure(
            encounter, response="granted", delivered_by="ma_jess", now=now
        )
        service.authorise(encounter, now=now)
        service.capture(
            encounter, audio_bytes=silent_wav(2.0), started_utc=now,
            ended_utc=now + timedelta(minutes=4), now=now,
        )
        audio_path = encounter.recording.path
        service.transcribe(encounter)
        taps = MATaps(
            protocol_id=case.protocol_id,
            disposition_id=case.disposition_id,
            tapped_by="ma_jess",
            tapped_utc=now,
            supervising_professional_id=case.supervising_professional_id,
        )
        service.draft(
            encounter, taps=taps, patient_label=case.patient_label, now=now,
            age_months=case.age_months,
        )
        note = encounter.note
        print(f"   protocol/disposition: {taps.protocol_id} / {taps.disposition_id}"
              f" (tapped by {taps.tapped_by})")
        for dropped in note.dropped:
            print(f"   REMOVED [{dropped['field']}] {dropped['text'][:60]}"
                  f"  <- {dropped['reason']}")
        for flag in note.flagged:
            print(f"   FLAGGED [{flag['field']}] {flag['text'][:60]}")

        # The MA edits a little, then signs.
        chart_text = (
            encounter.proposed_text
            + "\n    (MA addendum: parent verbalised understanding.)"
        )
        signed = service.sign(
            encounter, signed_by="ma_jess", now=now + timedelta(minutes=1),
            patient_label=case.patient_label, final_text=chart_text,
            acknowledged_drops=True, review_seconds=42.0,
        )
        signed_notes.append(signed)
        print(f"   signed: edit distance {signed.edit_distance}"
              f" (ratio {signed.edit_ratio:.3f})")
        print(f"   audio after signing: exists={Path(audio_path).exists()}")
        if encounter.followup:
            print(f"   follow-up task: {encounter.followup.description[:70]}")
        now += timedelta(minutes=20)

    print("\n-- a parent who declines --")
    case = by_name("very_short_call")
    encounter = service.open(
        patient_id=case.patient_id, family_id=case.family_id,
        opened_by="ma_jess", now=now,
    )
    service.deliver_disclosure(
        encounter, response="declined", delivered_by="ma_jess", now=now
    )
    print(f"   encounter state: {encounter.state} (the call still happens; the MA types the note)")
    try:
        consent.authorise_recording(
            encounter_id=encounter.encounter_id, family_id=case.family_id, now=now
        )
    except DeclinedRecording as exc:
        print(f"   recorder: {exc}")

    print("\n-- a workstation that rebooted mid-call --")
    # The MA's browser reloaded. Without a re-entry path the row sits in
    # 'recording' with a WAV beside it that no code can reach, and the call goes
    # undocumented -- which README I-04 calls indefensible.
    crashed = service.open(
        patient_id="p_crash", family_id="fam_crash", opened_by="ma_jess", now=now
    )
    stuck = service.abandoned(now=now + timedelta(hours=3), older_than=timedelta(hours=1))
    print(f"   abandoned encounters found: {len(stuck)} ({stuck[0]['state']})")
    recovered = service.resume(crashed.encounter_id)
    print(
        f"   resumed {recovered.encounter_id} in state {recovered.state!r}; "
        "consent must be re-taken before the recorder restarts"
    )

    print("\n-- retention --")
    print(f"   live audio rows: {len(lifecycle.live_audio())}")
    stale = lifecycle.stale_files_on_disk(workdir / "audio")
    print(f"   audio files still on disk: {len(stale)}")
    print(f"   orphans past the 24h ceiling: {len(lifecycle.orphans(now=now))}")

    print("\n-- edit-rate alarm (README 10.3) --")
    print(json.dumps(NoteReviewer(audit=audit).edit_rate_report(min_reviews=1), indent=2))

    print("\n-- documentation KPIs (README 10.2) --")
    print(json.dumps(service.documentation_kpis(), indent=2))

    print("\n-- one signed note as it reaches the chart --")
    print(signed_notes[0].text)


if __name__ == "__main__":
    main()
