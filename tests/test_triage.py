"""I-04 tests. Real state machine, real database, real EchoTransport.

`EchoTransport` and `ScriptedTranscriber` are shipped components, not mocks of
this module's logic: these tests drive the production structuring path --
schema validation, the clinical-judgement guard, grounding, rendering,
edit-distance -- against scripted transcripts and scripted model output.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from modules.scheduling.models import Database
from modules.triage import (
    DISCLOSURE_SCRIPT,
    HARD_RETENTION_CEILING,
    NOTE_SCHEMA,
    STRUCTURING_SYSTEM_PROMPT,
    AudioLifecycle,
    AudioRecording,
    ClinicalJudgementLeak,
    ConsentRegistry,
    DeclinedRecording,
    EncounterState,
    IllegalTransition,
    MATaps,
    ModelNotValidated,
    ModelPin,
    NoteReviewer,
    NoteStructurer,
    ProtocolRegistry,
    Recorder,
    RecordingNotAuthorised,
    RetentionBreach,
    ScriptedTranscriber,
    TapsIncomplete,
    TranscriptSegment,
    TriageService,
    UnreviewedDropError,
    assert_no_clinical_judgement_fields,
    build_followup_task,
    render_draft,
    render_note,
    signature_block,
    run_regression,
    silent_wav,
)
from modules.triage.fixtures import CALL_TIME, CASES, by_name, expand_to
from modules.triage.regression import PASS_THRESHOLD as PASS_THRESHOLD_FOR_TEST
from modules.triage.structure import BANNED_FIELD_TOKENS
from nsp_core.audit import AuditLog
from nsp_core.llm import EchoTransport, LLMClient, SchemaViolation

CHI = ZoneInfo("America/Chicago")
NOW = CALL_TIME


@pytest.fixture()
def db(tmp_path) -> Database:
    return Database(tmp_path / "triage.sqlite3")


@pytest.fixture()
def audit(tmp_path) -> AuditLog:
    return AuditLog(tmp_path / "audit.sqlite3", hmac_key=b"unit-test-key")


@pytest.fixture()
def registry() -> ProtocolRegistry:
    return ProtocolRegistry.load(allow_placeholder=True)


def make_service(db, audit, tmp_path, cases, *, responses=None):
    registry = ProtocolRegistry.load(allow_placeholder=True)
    transcriber = ScriptedTranscriber()
    client = LLMClient(
        EchoTransport(responses or [c.model_response for c in cases])
    )
    consent = ConsentRegistry(db, audit=audit)
    lifecycle = AudioLifecycle(db, audit=audit)
    service = TriageService(
        db,
        consent=consent,
        recorder=Recorder(tmp_path / "audio"),
        transcriber=transcriber,
        structurer=NoteStructurer(client, audit=audit),
        registry=registry,
        reviewer=NoteReviewer(audit=audit, registry=registry),
        lifecycle=lifecycle,
        audit=audit,
    )
    return service, transcriber, consent, lifecycle, registry


def taps_for(case, *, now=NOW, tapped_by="ma_jess") -> MATaps:
    return MATaps(
        protocol_id=case.protocol_id,
        disposition_id=case.disposition_id,
        tapped_by=tapped_by,
        tapped_utc=now,
        supervising_professional_id=case.supervising_professional_id,
    )


def run_to_draft(db, audit, tmp_path, case, *, response=None, now=NOW, tapped_utc=None):
    service, transcriber, consent, lifecycle, registry = make_service(
        db, audit, tmp_path, [case], responses=[response] if response else None
    )
    encounter = service.open(
        patient_id=case.patient_id, family_id=case.family_id,
        opened_by="ma_jess", now=now,
    )
    transcriber.transcripts[encounter.encounter_id] = case.transcript_payload
    transcriber.diarized = case.diarized
    consent.grant(family_id=case.family_id, captured_by="ma_jess", now=now)
    service.deliver_disclosure(
        encounter, response="granted", delivered_by="ma_jess", now=now
    )
    service.authorise(encounter, now=now)
    service.capture(
        encounter, audio_bytes=silent_wav(1.0), started_utc=now,
        ended_utc=now + timedelta(minutes=4), now=now,
    )
    service.transcribe(encounter)
    draft = service.draft(
        encounter, taps=taps_for(case, now=tapped_utc or now),
        patient_label=case.patient_label,
        now=now, age_months=case.age_months,
    )
    return service, encounter, draft, lifecycle, registry


def transcript_for(case):
    transcriber = ScriptedTranscriber(
        transcripts={"e1": case.transcript_payload}, diarized=case.diarized
    )
    recording = AudioRecording(
        "e1", "/dev/null", NOW, NOW + timedelta(minutes=3), "x", 180.0, None
    )
    return transcriber.transcribe(recording)


# ==========================================================================
# THE CONSTRAINT. README I-04's "single most important design constraint".
# ==========================================================================


def test_schema_contains_no_disposition_protocol_or_diagnosis_key():
    """The test the build plan asks for by name.

    "Write a test that asserts the schema contains no disposition/protocol/
    diagnosis key." An unlicensed MA in Illinois may not exercise independent
    clinical judgement; a machine-suggested disposition manufactures exactly the
    unauthorized-practice exposure 225 ILCS 60/54.2 exists to prevent.
    """
    def walk(schema, path="<root>"):
        keys = []
        for name, sub in (schema.get("properties") or {}).items():
            keys.append(f"{path}.{name}")
            if isinstance(sub, dict):
                keys.extend(walk(sub, f"{path}.{name}"))
        items = schema.get("items")
        if isinstance(items, dict):
            keys.extend(walk(items, f"{path}[]"))
        return keys

    all_keys = [k.lower() for k in walk(NOTE_SCHEMA)]
    for forbidden in ("disposition", "protocol", "diagnosis", "acuity", "triage_level"):
        assert not any(forbidden in k for k in all_keys), (forbidden, all_keys)
    # And positively: the fields that MUST be there are.
    assert "chief_complaint" in NOTE_SCHEMA["properties"]
    assert "advice_given_by_ma" in NOTE_SCHEMA["properties"]


def test_the_guard_catches_a_banned_field_at_any_depth():
    for schema in (
        {"type": "object", "additionalProperties": False,
         "properties": {"suggested_disposition": {"type": "string"}}},
        {"type": "object", "additionalProperties": False,
         "properties": {"inner": {"type": "object", "additionalProperties": False,
                                  "properties": {"diagnosis": {"type": "string"}}}}},
        {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                    "properties": {"acuity": {"type": "number"}}}},
        {"type": "object", "additionalProperties": False,
         "properties": {"x": {"anyOf": [
             {"type": "object", "additionalProperties": False, "properties": {}},
             {"type": "object", "additionalProperties": False,
              "properties": {"recommended_protocol": {"type": "string"}}},
         ]}}},
    ):
        with pytest.raises(ClinicalJudgementLeak):
            assert_no_clinical_judgement_fields(schema)


def test_the_structurer_refuses_to_be_constructed_with_a_leaky_schema():
    leaky = dict(NOTE_SCHEMA)
    leaky["properties"] = dict(NOTE_SCHEMA["properties"])
    leaky["properties"]["suggested_disposition"] = {"type": "string"}
    with pytest.raises(ClinicalJudgementLeak):
        NoteStructurer(LLMClient(EchoTransport([])), schema=leaky)


def test_a_banned_key_in_the_model_output_discards_the_extraction():
    """Belt and braces behind the validator. This constraint is load-bearing."""
    case = by_name("toddler_fever_home_care")
    payload = dict(case.extraction)
    payload["disposition"] = "see today"
    structurer = NoteStructurer(LLMClient(EchoTransport([json.dumps(payload)] * 3)))
    with pytest.raises((ClinicalJudgementLeak, SchemaViolation)):
        structurer.structure(transcript_for(case))


def test_prompt_is_appendix_a1_verbatim():
    for sentence in (
        "You MUST NOT suggest, infer, recommend, or imply any clinical disposition.",
        "You MUST NOT suggest, infer, or recommend any diagnosis.",
        "You MUST NOT suggest which triage protocol should have been used.",
        "You MUST NOT add clinical advice that was not spoken in the transcript.",
        "If information is not present in the transcript, use null. Do not infer it.",
        "The protocol used and the disposition reached are supplied separately by the\nmedical assistant",
    ):
        assert sentence in STRUCTURING_SYSTEM_PROMPT


def test_no_model_import_in_the_deterministic_half_of_the_module():
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "modules" / "triage"
    banned = {"nsp_core.llm", "openai", "anthropic"}
    offenders = []
    for name in ("protocols.py", "render.py", "lifecycle.py", "consent.py", "review.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = ["." * node.level + (node.module or "")]
            else:
                continue
            offenders += [
                f"{name}: {m}" for m in mods
                if any(m == b or m.startswith(b + ".") for b in banned)
            ]
    assert offenders == [], offenders


def test_whisper_and_torch_are_never_imported_at_module_scope():
    import ast
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "modules" / "triage" / "capture.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module_scope = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            module_scope += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            module_scope.append(node.module or "")
    assert not [m for m in module_scope if m.split(".")[0] in {"faster_whisper", "torch"}]


def test_importing_the_package_pulls_in_no_heavy_dependency():
    import pathlib
    import subprocess
    import sys

    repo = pathlib.Path(__file__).resolve().parents[1]
    code = (
        "import sys, modules.triage;"
        "bad=[n for n in sys.modules if n.split('.')[0] in "
        "{'faster_whisper','torch','transformers','boto3'}];"
        "print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=repo, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == ""


# ==========================================================================
# consent.py — Illinois two-party consent
# ==========================================================================


def test_recording_without_a_disclosure_is_refused(db, audit):
    consent = ConsentRegistry(db, audit=audit)
    consent.grant(family_id="fam1", captured_by="ma_jess", now=NOW)
    with pytest.raises(RecordingNotAuthorised, match="two-party consent"):
        consent.authorise_recording(encounter_id="e1", family_id="fam1", now=NOW)


def test_a_declined_disclosure_is_a_normal_outcome_not_an_error(db, audit):
    consent = ConsentRegistry(db, audit=audit)
    consent.grant(family_id="fam1", captured_by="ma_jess", now=NOW)
    consent.record_disclosure(
        encounter_id="e1", family_id="fam1", delivered_by="ma_jess",
        response="declined", now=NOW,
    )
    with pytest.raises(DeclinedRecording, match="never the care"):
        consent.authorise_recording(encounter_id="e1", family_id="fam1", now=NOW)


def test_a_disclosure_after_the_recording_is_not_consent(db, audit):
    consent = ConsentRegistry(db, audit=audit)
    consent.grant(family_id="fam1", captured_by="ma_jess", now=NOW)
    consent.record_disclosure(
        encounter_id="e1", family_id="fam1", delivered_by="ma_jess",
        response="granted", now=NOW + timedelta(minutes=5),
    )
    with pytest.raises(RecordingNotAuthorised, match="follows the recording"):
        consent.authorise_recording(encounter_id="e1", family_id="fam1", now=NOW)


def test_consent_must_exist_and_must_be_current(db, audit):
    consent = ConsentRegistry(db, audit=audit)
    consent.record_disclosure(
        encounter_id="e1", family_id="fam1", delivered_by="ma_jess",
        response="granted", now=NOW,
    )
    with pytest.raises(RecordingNotAuthorised, match="no active recording consent"):
        consent.authorise_recording(encounter_id="e1", family_id="fam1", now=NOW)

    consent.grant(
        family_id="fam1", captured_by="ma_jess", now=NOW - timedelta(days=400),
        ttl=timedelta(days=365),
    )
    with pytest.raises(RecordingNotAuthorised):
        consent.authorise_recording(encounter_id="e1", family_id="fam1", now=NOW)


def test_revoked_consent_blocks_recording(db, audit):
    consent = ConsentRegistry(db, audit=audit)
    consent.grant(family_id="fam1", captured_by="ma_jess", now=NOW)
    consent.record_disclosure(
        encounter_id="e1", family_id="fam1", delivered_by="ma_jess",
        response="granted", now=NOW,
    )
    assert consent.authorise_recording(encounter_id="e1", family_id="fam1", now=NOW)
    consent.revoke(family_id="fam1", now=NOW + timedelta(minutes=1))
    with pytest.raises(RecordingNotAuthorised):
        consent.authorise_recording(
            encounter_id="e1", family_id="fam1", now=NOW + timedelta(minutes=2)
        )


def test_the_authorisation_names_its_own_justification(db, audit):
    consent = ConsentRegistry(db, audit=audit)
    consent_id = consent.grant(family_id="fam1", captured_by="ma_jess", now=NOW)
    delivery = consent.record_disclosure(
        encounter_id="e1", family_id="fam1", delivered_by="ma_jess",
        response="granted", now=NOW,
    )
    auth = consent.authorise_recording(encounter_id="e1", family_id="fam1", now=NOW)
    assert auth.consent_id == consent_id
    assert auth.delivery_id == delivery.delivery_id
    assert delivery.script_hash  # the words said are pinned by hash


def test_disclosure_and_consent_require_a_named_person(db, audit):
    consent = ConsentRegistry(db, audit=audit)
    with pytest.raises(ValueError):
        consent.grant(family_id="fam1", captured_by="  ", now=NOW)
    with pytest.raises(ValueError):
        consent.record_disclosure(
            encounter_id="e1", family_id="fam1", delivered_by="",
            response="granted", now=NOW,
        )
    with pytest.raises(ValueError):
        consent.record_disclosure(
            encounter_id="e1", family_id="fam1", delivered_by="ma",
            response="maybe", now=NOW,
        )


def test_unevidenced_recordings_report_is_the_query_counsel_asks_for(db, audit):
    consent = ConsentRegistry(db, audit=audit)
    consent.record_disclosure(
        encounter_id="e_good", family_id="fam1", delivered_by="ma_jess",
        response="granted", now=NOW,
    )
    consent.record_disclosure(
        encounter_id="e_declined", family_id="fam2", delivered_by="ma_jess",
        response="declined", now=NOW,
    )
    assert consent.unevidenced_recordings(["e_good"]) == []
    assert consent.unevidenced_recordings(["e_good", "e_declined", "e_missing"]) == [
        "e_declined", "e_missing",
    ]


def test_the_disclosure_script_is_a_pinned_constant():
    assert "record this call" in DISCLOSURE_SCRIPT
    assert "deleted" in DISCLOSURE_SCRIPT
    assert DISCLOSURE_SCRIPT.strip().endswith("okay with you?")


# ==========================================================================
# capture.py
# ==========================================================================


def test_the_recorder_will_not_write_audio_without_an_authorisation(tmp_path):
    recorder = Recorder(tmp_path / "audio")
    with pytest.raises(RecordingNotAuthorised, match="two party|two-party"):
        recorder.capture(
            authorisation=None, encounter_id="e1", audio_bytes=silent_wav(0.2),
            started_utc=NOW, ended_utc=NOW + timedelta(minutes=1),
        )
    assert not os.path.exists(recorder.path_for("e1"))


def test_an_authorisation_is_not_transferable_between_encounters(db, audit, tmp_path):
    consent = ConsentRegistry(db, audit=audit)
    consent.grant(family_id="fam1", captured_by="ma_jess", now=NOW)
    consent.record_disclosure(
        encounter_id="e1", family_id="fam1", delivered_by="ma_jess",
        response="granted", now=NOW,
    )
    auth = consent.authorise_recording(encounter_id="e1", family_id="fam1", now=NOW)
    recorder = Recorder(tmp_path / "audio")
    with pytest.raises(RecordingNotAuthorised, match="not transferable"):
        recorder.capture(
            authorisation=auth, encounter_id="e2", audio_bytes=silent_wav(0.2),
            started_utc=NOW, ended_utc=NOW + timedelta(minutes=1),
        )


def test_captured_audio_is_hashed_and_written_restrictively(db, audit, tmp_path):
    consent = ConsentRegistry(db, audit=audit)
    consent.grant(family_id="fam1", captured_by="ma_jess", now=NOW)
    consent.record_disclosure(
        encounter_id="e1", family_id="fam1", delivered_by="ma_jess",
        response="granted", now=NOW,
    )
    auth = consent.authorise_recording(encounter_id="e1", family_id="fam1", now=NOW)
    payload = silent_wav(0.5)
    recording = Recorder(tmp_path / "audio").capture(
        authorisation=auth, encounter_id="e1", audio_bytes=payload,
        started_utc=NOW, ended_utc=NOW + timedelta(minutes=2),
    )
    import hashlib

    assert recording.sha256 == hashlib.sha256(payload).hexdigest()
    assert recording.duration_seconds == pytest.approx(120.0)
    assert recording.exists
    assert oct(os.stat(recording.path).st_mode)[-3:] == "600"


def test_an_undiarized_transcript_does_not_invent_speaker_labels():
    case = by_name("undiarized_transcript")
    transcript = transcript_for(case)
    assert transcript.diarized is False
    assert "MA:" not in transcript.labelled_text
    assert "CALLER:" not in transcript.labelled_text


def test_a_diarized_transcript_labels_who_spoke():
    transcript = transcript_for(by_name("toddler_fever_home_care"))
    assert transcript.diarized is True
    assert "MA:" in transcript.labelled_text
    assert "CALLER:" in transcript.labelled_text


def test_low_confidence_spans_become_gap_hints():
    transcript = transcript_for(by_name("unclear_audio_gaps"))
    hints = transcript.gap_hints()
    assert len(hints) == 2
    assert all("unclear audio" in h for h in hints)
    assert transcript.sha256 == transcript_for(by_name("unclear_audio_gaps")).sha256


def test_the_scripted_transcriber_reads_a_plain_text_sidecar(tmp_path):
    (tmp_path / "e9.txt").write_text(
        "MA: hello there\nCALLER: my child has a fever\n", encoding="utf-8"
    )
    transcriber = ScriptedTranscriber(directory=tmp_path)
    recording = AudioRecording("e9", "x", NOW, NOW, "h", 1.0, None)
    transcript = transcriber.transcribe(recording)
    assert [s.speaker for s in transcript.segments] == ["ma", "caller"]
    assert "fever" in transcript.text


# ==========================================================================
# protocols.py — the taps
# ==========================================================================


def test_no_note_can_be_assembled_without_both_taps(registry):
    with pytest.raises(TapsIncomplete, match="will not supply either"):
        registry.validate(None)


def test_unknown_protocol_or_disposition_is_refused(registry):
    base = dict(tapped_by="ma_jess", tapped_utc=NOW)
    with pytest.raises(TapsIncomplete, match="licensed registry"):
        registry.validate(MATaps(protocol_id="NOPE", disposition_id="home_care", **base))
    with pytest.raises(TapsIncomplete, match="ladder"):
        registry.validate(
            MATaps(protocol_id="PLACEHOLDER-FEVER", disposition_id="vibes", **base)
        )


def test_an_age_inappropriate_protocol_is_refused(registry):
    taps = MATaps(
        protocol_id="PLACEHOLDER-NEWBORN-FEVER", disposition_id="home_care",
        tapped_by="ma_jess", tapped_utc=NOW,
    )
    with pytest.raises(TapsIncomplete, match="does not apply"):
        registry.validate(taps, age_months=168)
    assert registry.validate(taps, age_months=1) is taps


def test_an_escalation_must_name_the_licensed_professional(registry):
    """225 ILCS 60/54.2: the record has to say who took the decision."""
    taps = MATaps(
        protocol_id="PLACEHOLDER-FEVER", disposition_id="ed_now",
        tapped_by="ma_jess", tapped_utc=NOW,
    )
    with pytest.raises(TapsIncomplete, match="54.2"):
        registry.validate(taps)
    named = MATaps(
        protocol_id="PLACEHOLDER-FEVER", disposition_id="ed_now",
        tapped_by="ma_jess", tapped_utc=NOW, supervising_professional_id="dr_ruiz",
    )
    assert registry.validate(named) is named


def test_the_protocol_list_is_alphabetical_and_age_filtered(registry):
    everything = registry.search()
    assert [p.title for p in everything] == sorted(p.title for p in everything)
    teen = registry.search(age_months=168)
    assert "PLACEHOLDER-NEWBORN-FEVER" not in {p.id for p in teen}
    assert {p.id for p in registry.search("fever", age_months=1)} == {
        "PLACEHOLDER-FEVER", "PLACEHOLDER-NEWBORN-FEVER",
    }


def test_the_registry_admits_that_it_is_a_placeholder(registry):
    """Shipping with placeholders would put PLACEHOLDER-FEVER in a chart."""
    assert registry.is_placeholder is True
    assert "licensed" in registry.source.lower()


# ==========================================================================
# structure.py — grounding
# ==========================================================================


def test_invented_advice_is_removed_from_the_note():
    case = by_name("model_invents_advice")
    note = NoteStructurer(
        LLMClient(EchoTransport([case.model_response]))
    ).structure(transcript_for(case))
    assert "give ibuprofen 10 mg per kilogram every six hours for pain" not in (
        note.data["advice_given_by_ma"]
    )
    assert any("ibuprofen" in d["text"] for d in note.dropped)
    assert "come in at four" in note.data["advice_given_by_ma"]


def test_advice_the_caller_said_is_not_credited_to_the_ma():
    case = by_name("model_misattributes_advice")
    note = NoteStructurer(
        LLMClient(EchoTransport([case.model_response]))
    ).structure(transcript_for(case))
    assert note.data["advice_given_by_ma"] == []
    dropped = [d for d in note.dropped if d["field"] == "advice_given_by_ma"]
    assert dropped and "not attributable to the medical assistant" in dropped[0]["reason"]


def test_weakly_supported_observations_are_flagged_not_deleted():
    """Deleting a documented denial makes a note look as though the question
    was never asked -- which is the gap a triage note exists to close."""
    case = by_name("denials_only")
    note = NoteStructurer(
        LLMClient(EchoTransport([case.model_response]))
    ).structure(transcript_for(case))
    assert note.flagged
    for item in note.flagged:
        assert item["field"] in (
            "symptoms_reported_present", "symptoms_explicitly_denied",
            "relevant_history_mentioned", "medications_mentioned",
        )
        assert item["text"] in note.data[item["field"]]
    assert not any(d["field"].startswith("symptoms") for d in note.dropped)
    denials = [f for f in note.flagged if f["field"] == "symptoms_explicitly_denied"]
    assert denials, "a weakly supported denial must survive as a marked item"


def test_a_grounded_note_drops_nothing():
    """A routine, fully grounded call is clean: nothing removed, nothing marked.

    Including the caller relationship. "Mom" spoken and "mother" written is the
    most common normalisation in a pediatric triage note, and a flag that fires
    on every routine call is a flag reviewers learn to click past -- which costs
    more than the flag is worth on the calls that matter.
    """
    case = by_name("toddler_fever_home_care")
    note = NoteStructurer(
        LLMClient(EchoTransport([case.model_response]))
    ).structure(transcript_for(case))
    assert note.dropped == []
    assert note.flagged == []
    assert note.data["caller"]["relationship_to_patient"] == "mother"


def test_the_relationship_synonym_table_is_not_a_free_pass():
    """It resolves "mom" to "mother". It does not resolve grandmother to mother.

    The subset test runs in one direction on purpose: every relationship the
    note claims has to have been spoken. A note that promotes the grandmother
    who placed the call into the child's mother is a consent and a
    wrong-informant problem, not a paraphrase.
    """
    case = by_name("toddler_fever_home_care")
    response = json.loads(case.model_response)
    response["caller"]["relationship_to_patient"] = "grandmother"
    note = NoteStructurer(
        LLMClient(EchoTransport([json.dumps(response)]))
    ).structure(transcript_for(case))
    assert [f["field"] for f in note.flagged] == ["caller.relationship_to_patient"]
    # Flagged, not deleted: the reviewer decides, the machine does not erase.
    assert note.data["caller"]["relationship_to_patient"] == "grandmother"


def test_a_caller_name_that_was_never_spoken_is_removed():
    case = by_name("caller_name_never_spoken")
    note = NoteStructurer(
        LLMClient(EchoTransport([case.model_response]))
    ).structure(transcript_for(case))
    assert note.data["caller"]["name"] is None
    assert any(d["field"] == "caller.name" for d in note.dropped)
    # The relationship survives -- identity is dropped, context is flagged.
    assert note.data["caller"]["relationship_to_patient"] == "mother"


def test_unclear_audio_reaches_the_note_even_if_the_model_omits_it():
    case = by_name("unclear_audio_gaps")
    assert case.extraction["transcript_gaps"] == []      # the model said nothing
    note = NoteStructurer(
        LLMClient(EchoTransport([case.model_response]))
    ).structure(transcript_for(case))
    assert len(note.data["transcript_gaps"]) == 2
    assert any("unclear audio" in w for w in note.warnings)


def test_an_undiarized_transcript_tells_the_model_not_to_guess():
    case = by_name("undiarized_transcript")
    transport = EchoTransport([case.model_response])
    NoteStructurer(LLMClient(transport)).structure(transcript_for(case))
    # The prompt grew, which is the observable effect of the warning being added.
    assert transport.calls
    plain = NoteStructurer(LLMClient(EchoTransport([case.model_response])))
    message = plain._build_user_message(transcript_for(case))
    assert "NO speaker labels" in message


def test_a_schema_violation_fails_closed_to_manual_documentation():
    case = by_name("toddler_fever_home_care")
    structurer = NoteStructurer(
        LLMClient(EchoTransport(["not json", "still not", "no"]), max_repair_attempts=2)
    )
    with pytest.raises(SchemaViolation):
        structurer.structure(transcript_for(case))


def test_the_audit_record_pins_transcript_prompt_and_model(db, audit, tmp_path):
    case = by_name("toddler_fever_home_care")
    note = NoteStructurer(
        LLMClient(EchoTransport([case.model_response])), audit=audit
    ).structure(transcript_for(case), patient_id=case.patient_id)
    row = audit.query("SELECT * FROM inference")[0]
    assert row["prompt_template_id"] == "README-A.1"
    assert row["prompt_template_hash"] == note.prompt_template_hash
    assert json.loads(row["extra_json"])["transcript_sha256"] == note.transcript_sha256
    # README 9.2: no prompt or completion payload anywhere in the record.
    assert "fever" not in json.dumps(dict(row)).lower()


# ==========================================================================
# render.py
# ==========================================================================


def test_nothing_renders_without_the_taps(registry):
    case = by_name("toddler_fever_home_care")
    note = NoteStructurer(
        LLMClient(EchoTransport([case.model_response]))
    ).structure(transcript_for(case))
    with pytest.raises(TapsIncomplete):
        render_note(note, None, registry, patient_label="x", call_time=NOW)


def test_the_chart_note_attributes_both_human_decisions(registry):
    case = by_name("newborn_fever_ed_now")
    note = NoteStructurer(
        LLMClient(EchoTransport([case.model_response]))
    ).structure(transcript_for(case))
    text = render_note(
        note, taps_for(case), registry, patient_label=case.patient_label,
        call_time=NOW,
    ) + signature_block("ma_jess", NOW)
    assert "PROTOCOL APPLIED" in text and "selected by ma_jess" in text
    assert "DISPOSITION" in text and "determined by ma_jess" in text
    assert "licensed professional: dr_ruiz" in text
    assert "225 ILCS 60/54.2" in text
    assert "Electronically signed by ma_jess" in text


def test_the_draft_shows_removals_and_the_chart_note_does_not(registry):
    case = by_name("model_invents_advice")
    note = NoteStructurer(
        LLMClient(EchoTransport([case.model_response]))
    ).structure(transcript_for(case))
    draft = render_draft(
        note, taps_for(case), registry, patient_label=case.patient_label, call_time=NOW
    )
    chart = render_note(
        note, taps_for(case), registry, patient_label=case.patient_label, call_time=NOW
    )
    assert "REMOVED BY THE SYSTEM" in draft
    assert "ibuprofen" in draft
    assert "REMOVED BY THE SYSTEM" not in chart
    assert "ibuprofen" not in chart
    assert draft.startswith("*** DRAFT")


def test_unclear_audio_appears_in_the_chart_note(registry):
    """The reader has to be able to tell "denied" from "we could not hear"."""
    case = by_name("unclear_audio_gaps")
    note = NoteStructurer(
        LLMClient(EchoTransport([case.model_response]))
    ).structure(transcript_for(case))
    chart = render_note(
        note, taps_for(case), registry, patient_label=case.patient_label, call_time=NOW
    )
    assert "PORTIONS OF THE CALL WERE UNCLEAR" in chart


def test_a_recognised_timeframe_produces_a_due_date():
    case = by_name("toddler_fever_home_care")
    note = NoteStructurer(
        LLMClient(EchoTransport([case.model_response]))
    ).structure(transcript_for(case))
    task = build_followup_task(note, patient_id="p1", now=NOW, assigned_to="ma_jess")
    assert task is not None
    assert task.due_utc == NOW + timedelta(days=1)
    assert "NO DUE DATE" not in task.description


def test_an_unrecognised_timeframe_never_invents_a_due_date():
    case = by_name("unrecognised_followup_timeframe")
    note = NoteStructurer(
        LLMClient(EchoTransport([case.model_response]))
    ).structure(transcript_for(case))
    task = build_followup_task(note, patient_id="p1", now=NOW, assigned_to="ma_jess")
    assert task is not None
    assert task.due_utc is None
    assert "NO DUE DATE SET" in task.description


def test_no_followup_discussed_produces_no_task():
    case = by_name("denials_only")
    note = NoteStructurer(
        LLMClient(EchoTransport([case.model_response]))
    ).structure(transcript_for(case))
    assert build_followup_task(note, patient_id="p1", now=NOW, assigned_to="ma") is None


# ==========================================================================
# review.py — the edit diff
# ==========================================================================


def test_signing_requires_a_named_person(db, audit, tmp_path):
    case = by_name("toddler_fever_home_care")
    service, encounter, draft, _, _ = run_to_draft(db, audit, tmp_path, case)
    with pytest.raises(ValueError, match="named person"):
        service.sign(
            encounter, signed_by="   ", now=NOW, patient_label=case.patient_label
        )


def test_signing_refuses_until_removals_are_acknowledged(db, audit, tmp_path):
    case = by_name("model_invents_advice")
    service, encounter, draft, _, _ = run_to_draft(db, audit, tmp_path, case)
    with pytest.raises(UnreviewedDropError, match="item.s. were removed"):
        service.sign(
            encounter, signed_by="ma_jess", now=NOW, patient_label=case.patient_label
        )
    signed = service.sign(
        encounter, signed_by="ma_jess", now=NOW, patient_label=case.patient_label,
        acknowledged_drops=True,
    )
    assert signed.signed_by == "ma_jess"


def test_the_edit_diff_is_recorded_and_counts_toward_the_alarm(db, audit, tmp_path):
    case = by_name("toddler_fever_home_care")
    service, encounter, draft, _, _ = run_to_draft(db, audit, tmp_path, case)
    signed = service.sign(
        encounter, signed_by="ma_jess", now=NOW, patient_label=case.patient_label,
        final_text=draft + "\n    (MA addendum: parent verbalised understanding.)",
        review_seconds=51.0,
    )
    row = audit.query("SELECT * FROM review")[0]
    assert row["edit_distance_draft_final"] == signed.edit_distance > 0
    assert row["edit_rate_applicable"] == 1
    assert row["review_seconds"] == 51.0
    assert json.loads(row["extra_json"])["protocol_id"] == case.protocol_id


def test_a_zero_edit_signature_is_visible_as_rubber_stamping(db, audit, tmp_path):
    case = by_name("toddler_fever_home_care")
    service, encounter, draft, _, _ = run_to_draft(db, audit, tmp_path, case)
    # The baseline is the PROPOSED CHART NOTE, not the reviewer-facing draft.
    # Diffing the draft (which carries the removal notices and flags) against
    # the chart body made every signature look heavily edited, so the
    # rubber-stamp alarm was structurally incapable of firing.
    signed = service.sign(
        encounter, signed_by="ma_lazy", now=NOW, patient_label=case.patient_label,
        final_text=encounter.proposed_text,
    )
    assert signed.edit_distance == 0
    assert signed.looks_rubber_stamped is True
    assert signed.action_taken == "accepted"


def test_the_edit_rate_report_alarms_in_both_directions(db, audit, tmp_path):
    reviewer = NoteReviewer(audit=audit)
    for _ in range(12):
        audit.record_review(
            reviewer_id="ma_stamp", initiative_id="I-04",
            draft="identical draft text", final="identical draft text",
            action_taken="accepted",
        )
    for i in range(12):
        audit.record_review(
            reviewer_id="ma_retypes", initiative_id="I-04",
            draft=f"a machine draft {i}",
            final=f"a completely different note written from scratch {i}",
            action_taken="edited",
        )
    report = reviewer.edit_rate_report(min_reviews=10)
    by_reviewer = {r["reviewer_id"]: r for r in report["reviewers"]}
    assert by_reviewer["ma_stamp"]["rubber_stamp_alarm"] is True
    assert by_reviewer["ma_retypes"]["poor_draft_alarm"] is True
    assert by_reviewer["ma_retypes"]["rubber_stamp_alarm"] is False


def test_rejecting_a_draft_is_a_first_class_outcome(db, audit, tmp_path):
    case = by_name("toddler_fever_home_care")
    service, encounter, draft, lifecycle, _ = run_to_draft(db, audit, tmp_path, case)
    audio_path = encounter.recording.path
    service.reject(
        encounter, rejected_by="ma_jess", reason="transcription garbled the ages",
        now=NOW,
    )
    assert encounter.state == EncounterState.REJECTED
    row = audit.query("SELECT * FROM review WHERE action_taken='rejected'")[0]
    assert json.loads(row["extra_json"])["reason"].startswith("transcription")
    assert not os.path.exists(audio_path)


# ==========================================================================
# lifecycle.py — retention
# ==========================================================================


def test_audio_is_actually_gone_after_the_note_is_signed(db, audit, tmp_path):
    """The build plan asks for this one specifically: test that the file is gone."""
    case = by_name("toddler_fever_home_care")
    service, encounter, draft, lifecycle, _ = run_to_draft(db, audit, tmp_path, case)
    audio_path = encounter.recording.path
    assert os.path.exists(audio_path)

    service.sign(
        encounter, signed_by="ma_jess", now=NOW + timedelta(minutes=1),
        patient_label=case.patient_label,
    )
    assert not os.path.exists(audio_path)
    assert lifecycle.live_audio() == []
    row = db.one("SELECT delete_reason FROM triage_audio")
    assert row["delete_reason"] == "note_signed"


def test_the_transcript_is_purged_but_its_hash_survives(db, audit, tmp_path):
    case = by_name("toddler_fever_home_care")
    service, encounter, draft, _, _ = run_to_draft(db, audit, tmp_path, case)
    before = db.one(
        "SELECT transcript_text, transcript_sha256 FROM triage_encounter"
        " WHERE encounter_id = ?", (encounter.encounter_id,)
    )
    assert before["transcript_text"]
    service.sign(
        encounter, signed_by="ma_jess", now=NOW + timedelta(minutes=1),
        patient_label=case.patient_label,
    )
    after = db.one(
        "SELECT transcript_text, transcript_sha256, transcript_purged_utc"
        " FROM triage_encounter WHERE encounter_id = ?", (encounter.encounter_id,)
    )
    assert after["transcript_text"] is None
    assert after["transcript_sha256"] == before["transcript_sha256"]
    assert after["transcript_purged_utc"]


def test_the_sweep_deletes_abandoned_audio_at_the_ceiling(db, audit, tmp_path):
    """The call nobody ever signed is exactly the one that accumulates."""
    case = by_name("toddler_fever_home_care")
    service, encounter, draft, lifecycle, _ = run_to_draft(db, audit, tmp_path, case)
    audio_path = encounter.recording.path

    # Still inside the window: untouched.
    assert lifecycle.sweep(now=NOW + timedelta(hours=23)) == []
    assert os.path.exists(audio_path)

    results = lifecycle.sweep(now=NOW + HARD_RETENTION_CEILING + timedelta(minutes=1))
    assert len(results) == 1
    assert results[0].reason == "retention_ceiling"
    assert results[0].file_removed is True
    assert not os.path.exists(audio_path)
    # The encounter is still unsigned. The sweep does not care, by design.
    assert encounter.state == EncounterState.DRAFTED


def test_retention_breach_is_detectable(db, audit, tmp_path):
    case = by_name("toddler_fever_home_care")
    service, encounter, draft, lifecycle, _ = run_to_draft(db, audit, tmp_path, case)
    late = NOW + HARD_RETENTION_CEILING + timedelta(hours=2)
    assert lifecycle.orphans(now=late)
    with pytest.raises(RetentionBreach, match="retention ceiling"):
        lifecycle.assert_clean(now=late)
    lifecycle.sweep(now=late)
    lifecycle.assert_clean(now=late, directory=tmp_path / "audio")


def test_an_unregistered_audio_file_is_reported(db, audit, tmp_path):
    lifecycle = AudioLifecycle(db, audit=audit)
    directory = tmp_path / "audio"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "stray.wav").write_bytes(silent_wav(0.2))
    assert lifecycle.stale_files_on_disk(directory) == [str(directory / "stray.wav")]
    with pytest.raises(RetentionBreach, match="no lifecycle record"):
        lifecycle.assert_clean(now=NOW, directory=directory)


# ==========================================================================
# encounter.py — the state machine
# ==========================================================================


def test_capture_before_disclosure_is_refused(db, audit, tmp_path):
    case = by_name("toddler_fever_home_care")
    service, transcriber, consent, _, _ = make_service(db, audit, tmp_path, [case])
    encounter = service.open(
        patient_id=case.patient_id, family_id=case.family_id,
        opened_by="ma_jess", now=NOW,
    )
    with pytest.raises(IllegalTransition):
        service.capture(
            encounter, audio_bytes=silent_wav(0.2), started_utc=NOW,
            ended_utc=NOW + timedelta(minutes=1), now=NOW,
        )


def test_capture_after_disclosure_but_without_authorise_is_refused(db, audit, tmp_path):
    case = by_name("toddler_fever_home_care")
    service, transcriber, consent, _, _ = make_service(db, audit, tmp_path, [case])
    encounter = service.open(
        patient_id=case.patient_id, family_id=case.family_id,
        opened_by="ma_jess", now=NOW,
    )
    consent.grant(family_id=case.family_id, captured_by="ma_jess", now=NOW)
    service.deliver_disclosure(
        encounter, response="granted", delivered_by="ma_jess", now=NOW
    )
    with pytest.raises(RecordingNotAuthorised, match="call authorise"):
        service.capture(
            encounter, audio_bytes=silent_wav(0.2), started_utc=NOW,
            ended_utc=NOW + timedelta(minutes=1), now=NOW,
        )


def test_a_declined_call_terminates_in_manual_and_stays_there(db, audit, tmp_path):
    case = by_name("toddler_fever_home_care")
    service, transcriber, consent, _, _ = make_service(db, audit, tmp_path, [case])
    encounter = service.open(
        patient_id=case.patient_id, family_id=case.family_id,
        opened_by="ma_jess", now=NOW,
    )
    service.deliver_disclosure(
        encounter, response="declined", delivered_by="ma_jess", now=NOW
    )
    assert encounter.state == EncounterState.MANUAL
    with pytest.raises(IllegalTransition):
        service.authorise(encounter, now=NOW)
    row = db.one(
        "SELECT manual_reason, closed_utc FROM triage_encounter WHERE encounter_id=?",
        (encounter.encounter_id,),
    )
    assert row["manual_reason"] == "parent declined recording"
    assert row["closed_utc"]


def test_the_manual_fallback_is_available_from_any_live_state(db, audit, tmp_path):
    case = by_name("toddler_fever_home_care")
    service, encounter, draft, _, _ = run_to_draft(db, audit, tmp_path, case)
    audio_path = encounter.recording.path
    service.fall_back_to_manual(
        encounter, reason="MA did not trust the draft", now=NOW
    )
    assert encounter.state == EncounterState.MANUAL
    assert not os.path.exists(audio_path)
    with pytest.raises(IllegalTransition):
        service.fall_back_to_manual(encounter, reason="again", now=NOW)


def test_signing_before_drafting_is_refused(db, audit, tmp_path):
    case = by_name("toddler_fever_home_care")
    service, transcriber, consent, _, _ = make_service(db, audit, tmp_path, [case])
    encounter = service.open(
        patient_id=case.patient_id, family_id=case.family_id,
        opened_by="ma_jess", now=NOW,
    )
    with pytest.raises(IllegalTransition):
        service.sign(
            encounter, signed_by="ma_jess", now=NOW, patient_label="x"
        )


def test_drafting_validates_the_taps_before_spending_an_inference(db, audit, tmp_path):
    case = by_name("newborn_fever_ed_now")
    service, transcriber, consent, _, _ = make_service(db, audit, tmp_path, [case])
    encounter = service.open(
        patient_id=case.patient_id, family_id=case.family_id,
        opened_by="ma_jess", now=NOW,
    )
    transcriber.transcripts[encounter.encounter_id] = case.transcript_payload
    consent.grant(family_id=case.family_id, captured_by="ma_jess", now=NOW)
    service.deliver_disclosure(
        encounter, response="granted", delivered_by="ma_jess", now=NOW
    )
    service.authorise(encounter, now=NOW)
    service.capture(
        encounter, audio_bytes=silent_wav(0.2), started_utc=NOW,
        ended_utc=NOW + timedelta(minutes=1), now=NOW,
    )
    service.transcribe(encounter)
    unnamed = MATaps(
        protocol_id=case.protocol_id, disposition_id="ed_now",
        tapped_by="ma_jess", tapped_utc=NOW,
    )
    with pytest.raises(TapsIncomplete):
        service.draft(
            encounter, taps=unnamed, patient_label=case.patient_label, now=NOW,
            age_months=case.age_months,
        )
    assert audit.counts()["inference"] == 0


def test_the_undocumented_queue_finds_the_call_nobody_wrote_up(db, audit, tmp_path):
    case = by_name("toddler_fever_home_care")
    service, encounter, draft, _, _ = run_to_draft(db, audit, tmp_path, case)
    assert service.undocumented(now=NOW + timedelta(hours=1)) == []
    stale = service.undocumented(now=NOW + timedelta(hours=9))
    assert [r["encounter_id"] for r in stale] == [encounter.encounter_id]


def test_documentation_kpis_track_the_readme_targets(db, audit, tmp_path):
    case = by_name("toddler_fever_home_care")
    service, encounter, draft, _, _ = run_to_draft(db, audit, tmp_path, case)
    service.sign(
        encounter, signed_by="ma_jess", now=NOW + timedelta(seconds=75),
        patient_label=case.patient_label,
    )
    kpis = service.documentation_kpis()
    assert kpis["encounters"] == 1
    assert kpis["signed"] == 1
    assert kpis["documented_rate"] == 1.0
    assert kpis["median_seconds_to_signature"] == pytest.approx(75.0)
    assert kpis["by_disposition"] == {"home_care": 1}


def test_a_full_call_runs_end_to_end(db, audit, tmp_path):
    case = by_name("vomiting_see_today")
    service, encounter, draft, lifecycle, registry = run_to_draft(
        db, audit, tmp_path, case
    )
    signed = service.sign(
        encounter, signed_by="ma_jess", now=NOW + timedelta(minutes=1),
        patient_label=case.patient_label,
        final_text=draft.replace("*** DRAFT - NOT PART OF THE CHART UNTIL SIGNED ***\n\n", ""),
        # "Throwing up" paraphrased to "vomiting approximately seven times" is a
        # marked item in this note; the signer has to have seen it.
        acknowledged_drops=True,
    )
    assert encounter.state == EncounterState.SIGNED
    assert encounter.followup is not None
    assert encounter.followup.due_utc is not None
    assert signed.edit_distance > 0
    assert not os.path.exists(encounter.recording.path)


# ==========================================================================
# regression.py — the version-bump gate
# ==========================================================================


def _regression_pieces(cases, *, mutate=None):
    transcripts = {c.name: c.transcript_payload for c in cases}
    transcriber = ScriptedTranscriber(transcripts=transcripts)
    responses = []
    for case in cases:
        payload = dict(case.extraction)
        if mutate:
            payload = mutate(payload)
        responses.append(json.dumps(payload))
    structurer = NoteStructurer(LLMClient(EchoTransport(responses)))

    def recorder_factory(case):
        return AudioRecording(
            case.name, "/dev/null", NOW, NOW + timedelta(minutes=3), "h", 180.0, None
        )

    return structurer, transcriber, recorder_factory


def test_the_regression_suite_passes_on_fifty_known_calls():
    cases = expand_to(50)
    structurer, transcriber, factory = _regression_pieces(cases)
    result = run_regression(
        structurer, cases, transcriber=transcriber, recorder_factory=factory
    )
    assert result.calls == 50
    assert result.errors == []
    assert result.passes() is True, result.summary()


def test_a_degraded_model_fails_the_suite_and_names_the_field():
    """Dropping safety-net instructions is the degradation that matters most."""
    cases = expand_to(50)

    def drop_safety_net(payload):
        payload = dict(payload)
        payload["safety_net_instructions_given"] = []
        return payload

    structurer, transcriber, factory = _regression_pieces(cases, mutate=drop_safety_net)
    result = run_regression(
        structurer, cases, transcriber=transcriber, recorder_factory=factory
    )
    assert result.passes() is False
    worst = result.worst_fields[0]
    assert worst.field_name == "safety_net_instructions_given"
    assert worst.rate < 0.9
    assert "safety_net" in result.summary()


def test_too_few_calls_cannot_pass():
    cases = expand_to(10)
    structurer, transcriber, factory = _regression_pieces(cases)
    result = run_regression(
        structurer, cases, transcriber=transcriber, recorder_factory=factory
    )
    assert result.calls == 10
    assert result.passes() is False


def test_a_model_pin_will_not_move_without_a_passing_suite():
    from datetime import date

    cases = expand_to(10)
    structurer, transcriber, factory = _regression_pieces(cases)
    weak = run_regression(
        structurer, cases, transcriber=transcriber, recorder_factory=factory
    )
    pin = ModelPin(model_id="echo-1", model_version="test")
    with pytest.raises(ModelNotValidated, match="50 known calls"):
        pin.bump(weak, on=date(2026, 8, 22))

    cases = expand_to(50)
    structurer, transcriber, factory = _regression_pieces(cases)
    good = run_regression(
        structurer, cases, transcriber=transcriber, recorder_factory=factory
    )
    pin.bump(good, on=date(2026, 8, 22))
    assert pin.validated_on == date(2026, 8, 22)
    pin.require("echo-1", "test")
    with pytest.raises(ModelNotValidated, match="not the pinned"):
        pin.require("echo-1", "test-2")


def test_the_regression_reference_excludes_content_grounding_should_remove():
    """The safety control working must not read as a regression."""
    cases = [by_name("model_invents_advice")] * 3
    structurer, transcriber, factory = _regression_pieces(cases)
    result = run_regression(
        structurer, cases, transcriber=transcriber, recorder_factory=factory
    )
    assert result.ungrounded_drops == 3
    assert result.fields["advice_given_by_ma"].rate == 1.0


def test_every_fixture_survives_the_whole_pipeline(db, audit, tmp_path):
    registry = ProtocolRegistry.load(allow_placeholder=True)
    for case in CASES:
        transcript = transcript_for(case)
        note = NoteStructurer(
            LLMClient(EchoTransport([case.model_response]))
        ).structure(transcript)
        text = render_note(
            note, taps_for(case), registry, patient_label=case.patient_label,
            call_time=NOW,
        )
        assert "PROTOCOL APPLIED" in text
        assert "DISPOSITION" in text
        for token in ("suggested disposition", "recommended protocol", "likely diagnosis"):
            assert token not in text.lower()


# ==========================================================================
# Adversarial-review regressions. One test per finding; each reproduces the
# original failure, so a revert makes exactly one of them go red.
# ==========================================================================


def test_the_edit_diff_baseline_is_the_chart_note_not_the_draft():
    """FINDING: the rubber-stamp alarm could not fire.

    `sign()` diffed the reviewer-facing draft -- removal notices, flag markers,
    the DRAFT banner -- against the chart body. Those two strings are never
    equal, so an MA who changed nothing still scored a large edit distance and
    README 10.3's "< 5% edit rate" alarm was structurally unreachable.
    """
    case = by_name("toddler_fever_home_care")
    note = NoteStructurer(
        LLMClient(EchoTransport([case.model_response]))
    ).structure(transcript_for(case))
    registry = ProtocolRegistry.load(allow_placeholder=True)
    taps = taps_for(case)
    chart = render_note(
        note, taps, registry, patient_label=case.patient_label, call_time=NOW
    )
    draft = render_draft(
        note, taps, registry, patient_label=case.patient_label, call_time=NOW
    )
    assert chart != draft, "the draft must carry reviewer-only material"
    assert "DRAFT" not in chart
    # The chart body is what the signature block gets appended to, so an
    # untouched signature is a zero-distance signature.
    reviewer = NoteReviewer()
    signed = reviewer.sign(
        note=note, taps=taps, baseline_text=chart, final_text=chart,
        signed_by="ma", now=NOW,
    )
    assert signed.edit_distance == 0 and signed.looks_rubber_stamped


def test_an_all_stopword_phrase_is_not_grounded_by_an_empty_transcript():
    """FINDING: a phrase of pure stopwords scored 1.0 against anything.

    "You should get her in" has no content words. Dividing by zero matches was
    guarded by returning 1.0, which made a fabricated disposition ground
    against an empty transcript -- the exact output the module exists to stop.
    """
    from modules.triage.structure import _grounding_score

    for phrase in ("You should get her in", "we should get him in", "get her in to us"):
        assert _grounding_score(phrase, set()) == 0.0


def test_the_prefix_rule_does_not_ground_throat_against_throwing_up():
    """FINDING: a flat four-character prefix produced false negatives.

    Grounding is a safety control, so its failure mode is the item it lets
    through. "Throat", "throwing" and "through" share four characters.
    """
    from modules.triage.structure import _matches

    corpus = {"throwing", "through", "vomiting"}
    assert not _matches("throat", corpus)
    # ...while the cases the rule exists for still work.
    assert _matches("breathing", {"breathe"})
    assert _matches("wheezing", {"wheeze"})


def test_a_kept_but_unsupported_item_is_marked_in_the_signed_chart():
    """FINDING: flagged items were shown in the draft and printed clean in the
    chart. A reader of the record could not tell them from verified content."""
    case = by_name("denials_only")
    note = NoteStructurer(
        LLMClient(EchoTransport([case.model_response]))
    ).structure(transcript_for(case))
    assert note.flagged
    text = render_note(
        note, taps_for(case), ProtocolRegistry.load(allow_placeholder=True),
        patient_label=case.patient_label, call_time=NOW,
    )
    assert "[not clearly supported by the recording]" in text


def test_model_written_gaps_never_reach_the_chart():
    """FINDING: `transcript_gaps` was a free-text channel from the model into a
    clinical record. A model writing a diagnosis there bypassed every guard."""
    case = by_name("unclear_audio_gaps")
    response = json.loads(case.model_response)
    response["transcript_gaps"] = ["findings are consistent with early sepsis"]
    note = NoteStructurer(
        LLMClient(EchoTransport([json.dumps(response)]))
    ).structure(transcript_for(case))
    assert "findings are consistent with early sepsis" in note.model_gaps
    assert note.data["transcript_gaps"] == note.system_gaps
    text = render_note(
        note, taps_for(case), ProtocolRegistry.load(allow_placeholder=True),
        patient_label=case.patient_label, call_time=NOW,
    )
    assert "sepsis" not in text.lower()
    # The reviewer still sees it and can choose to type it in.
    draft = render_draft(
        note, taps_for(case), ProtocolRegistry.load(allow_placeholder=True),
        patient_label=case.patient_label, call_time=NOW,
    )
    assert "sepsis" in draft.lower()


def test_the_schema_guard_is_an_allowlist_not_a_token_blocklist():
    """FINDING: a banned-token blocklist passed anything named innocuously.

    `next_step` carries a disposition and contains none of the banned tokens.
    The A.1 key set is the contract, so the guard checks membership.
    """
    schema = json.loads(json.dumps(NOTE_SCHEMA))
    schema["properties"]["next_step"] = {"type": "string"}
    schema["additionalProperties"] = False
    with pytest.raises(ClinicalJudgementLeak, match="A.1"):
        NoteStructurer(LLMClient(EchoTransport(["{}"])), schema=schema)


def test_the_regression_suite_fails_when_grounding_is_switched_off():
    """FINDING: the suite scored a sabotaged pipeline as a pass.

    Two bugs, one after the other. First the expected answer was derived from
    the run being scored, so it was 1.0 by construction. Then, with the
    expectation moved into the fixtures, a run with grounding disabled entirely
    still scored 0.988 -- the fabricated advice reached the note and the average
    absorbed it. Declared adversarial drops are now an absolute condition.
    """
    cases = expand_to(50)
    result = _run_suite(cases, grounding_threshold=0.0)
    assert result.missed_drops, "fabricated advice survived and nothing noticed"
    assert result.overall > PASS_THRESHOLD_FOR_TEST
    assert not result.passes()
    with pytest.raises(ModelNotValidated):
        ModelPin("echo", "old").bump(result, on=date(2026, 8, 22))


def test_the_regression_suite_passes_the_shipped_configuration():
    result = _run_suite(expand_to(50), grounding_threshold=0.6)
    assert result.missed_drops == []
    assert result.overall == 1.0
    assert result.passes()
    assert result.distinct_transcripts == len(CASES)
    # The summary is what the pin stores; it must not read as fifty real calls.
    assert f"{len(CASES)} distinct transcript(s)" in result.summary()


def test_a_failed_unlink_keeps_the_row_live(db, audit, tmp_path):
    """FINDING: an unlink failure stamped `deleted_utc` anyway.

    The file stayed on disk and the row said it was gone, so `sweep`,
    `orphans`, `live_audio` and `assert_clean` all stopped seeing it. A
    permanently invisible recording is worse than a visibly stuck one.
    """
    lifecycle = AudioLifecycle(db, audit=audit)
    recorder = Recorder(str(tmp_path / "audio"))
    recording = _recording(recorder, "enc_stuck")
    lifecycle.register(recording, now=NOW)

    original = lifecycle._purge
    lifecycle._purge = lambda path: False  # a read-only mount, an AV lock
    try:
        result = lifecycle.on_signature("enc_stuck", now=NOW)
    finally:
        lifecycle._purge = original
    assert result is not None and result.file_removed is False
    assert lifecycle.failed_deletions(), "a stuck file must stay visible"
    assert lifecycle.live_audio(), "the row must stay live so the sweep retries"
    with pytest.raises(RetentionBreach, match="could not be deleted"):
        lifecycle.assert_clean(now=NOW + timedelta(hours=48))


def test_a_missing_encounter_table_is_reported_not_silently_skipped(db, tmp_path):
    """FINDING, then the over-correction for it.

    `assert_clean` queried a table `AudioLifecycle` does not own, so a retention
    cron pointed at a database of audio rows raised OperationalError. Guarding
    the query fixed the crash and created something worse: with the lifecycle
    and the TriageService on different database handles -- an easy wiring
    mistake, both take a `Database` -- every transcript purge became a no-op and
    `assert_clean` certified the box as clean with the verbatim call still in
    it. Now it reports, and a caller asking for a specific purge gets an error
    rather than a promise this object cannot keep.
    """
    lifecycle = AudioLifecycle(db, audit=None)
    empty = tmp_path / "no-audio-here"
    empty.mkdir()
    with pytest.raises(RetentionBreach, match="NOT VERIFIABLE"):
        lifecycle.assert_clean(now=NOW, directory=str(empty))
    with pytest.raises(RetentionBreach, match="has NOT been destroyed"):
        lifecycle.purge_transcript("enc_anything", now=NOW)
    # The sweep still must not crash -- it runs unattended on a timer.
    assert lifecycle.sweep_transcripts(now=NOW) == 0
    assert lifecycle.sweep(now=NOW) == []

    # A deployment that genuinely keeps transcripts elsewhere says so, once.
    declared = AudioLifecycle(db, audit=None, transcripts_elsewhere=True)
    declared.assert_clean(now=NOW, directory=str(empty))


def test_only_one_signature_survives_a_concurrent_sign(db, audit, tmp_path):
    """FINDING: `sign()` recorded the review before claiming the transition.

    Two MAs pressing sign on the same note produced two review rows, two audit
    records and two edit-distance measurements for one chart note.
    """
    case = by_name("toddler_fever_home_care")
    service, encounter, draft, _, _ = run_to_draft(db, audit, tmp_path, case)
    service.sign(
        encounter, signed_by="ma_first", now=NOW, patient_label=case.patient_label
    )
    resumed = service.resume(encounter.encounter_id)
    resumed.note = encounter.note
    resumed.state = EncounterState.DRAFTED  # a stale tab that never saw the sign
    with pytest.raises(IllegalTransition, match="already signed"):
        service.sign(
            resumed, signed_by="ma_second", now=NOW, patient_label=case.patient_label
        )
    rows = audit.query("SELECT * FROM review")
    assert len(rows) == 1 and rows[0]["reviewer_id"] == "ma_first"


def test_an_encounter_can_be_resumed_after_a_restart(db, audit, tmp_path):
    """FINDING: an encounter only existed on the in-memory object.

    A reload mid-call left a row in `state='recording'` and a WAV on disk that
    no code path could reach: the audio sat until the sweep and the call went
    undocumented.
    """
    case = by_name("toddler_fever_home_care")
    service, encounter, draft, _, _ = run_to_draft(db, audit, tmp_path, case)

    resumed = service.resume(encounter.encounter_id)
    assert resumed.state == EncounterState.DRAFTED
    assert resumed.proposed_text == encounter.proposed_text
    assert resumed.taps is not None
    assert resumed.taps.protocol_id == case.protocol_id
    # The note comes back deserialised -- same inference id, same drops, same
    # flags -- so the encounter can actually be signed, which is the point.
    assert resumed.note is not None
    assert resumed.note.inference_id == encounter.note.inference_id
    assert resumed.note == encounter.note
    # The authorisation is NOT restored: it has a TTL and consent can have been
    # revoked, so `capture()` re-runs the gate from scratch.
    assert resumed.authorisation is None

    signed = service.sign(
        resumed, signed_by="ma_after_reboot", now=NOW + timedelta(minutes=5),
        patient_label=case.patient_label,
    )
    assert signed.signed_by == "ma_after_reboot"

    assert service.abandoned(
        now=NOW + timedelta(hours=2), older_than=timedelta(hours=1)
    ) == []  # signed is terminal


def test_a_resumed_encounter_keeps_the_real_tap_time(db, audit, tmp_path):
    """FINDING: `resume` invented `tapped_utc` from the call-open time.

    `MATaps.as_dict()` publishes it as the delegation timestamp under 225 ILCS
    60/54.2. A tap made seven minutes into a call must not come back stamped at
    the moment the call was answered.
    """
    case = by_name("toddler_fever_home_care")
    tapped_at = NOW + timedelta(minutes=7)
    service, encounter, _, _, _ = run_to_draft(
        db, audit, tmp_path, case, tapped_utc=tapped_at
    )
    assert encounter.taps.tapped_utc == tapped_at != encounter.opened_utc
    resumed = service.resume(encounter.encounter_id)
    assert resumed.taps.tapped_utc == tapped_at
    assert resumed.taps.as_dict()["tapped_utc"] == encounter.taps.as_dict()["tapped_utc"]


def test_signing_without_a_note_refuses_under_O(db, audit, tmp_path):
    """FINDING: both exits from DRAFTED were bare asserts, which `python -O`
    strips -- turning a refusal into an AttributeError mid-signature."""
    case = by_name("toddler_fever_home_care")
    service, encounter, _, _, _ = run_to_draft(db, audit, tmp_path, case)
    encounter.note = None
    with pytest.raises(IllegalTransition, match="no structured note"):
        service.sign(
            encounter, signed_by="ma", now=NOW, patient_label=case.patient_label
        )


def test_set_refuses_a_column_it_does_not_know(db, audit, tmp_path):
    """FINDING: `_set` interpolates its keyword names straight into SQL."""
    case = by_name("toddler_fever_home_care")
    service, encounter, _, _, _ = run_to_draft(db, audit, tmp_path, case)
    with pytest.raises(ValueError, match="unknown encounter column"):
        service._set(encounter, **{"signed_text = 'x' --": "y"})


def test_a_counted_followup_timeframe_is_not_silently_undated():
    """FINDING: "3 weeks" produced a task with no due date.

    One, two and three days and one and two weeks happened to be enumerated;
    three weeks did not. An MA who reads NO DUE DATE SET on an ordinary phrase
    stops reading it on the unparseable ones.
    """
    from modules.triage.render import _counted_offset

    assert _counted_offset("3 weeks") == 21
    assert _counted_offset("in 10 days") == 10
    assert _counted_offset("two weeks") == 14
    # A range resolves to the EARLIER bound: early costs a phone call.
    assert _counted_offset("in 3 to 4 days") == 3
    assert _counted_offset("3-4 days") == 3
    # Still refuses to guess, and still refuses a date nobody would action.
    assert _counted_offset("in a bit") is None
    assert _counted_offset("five months") is None
    # And the digit guard the enumerated table needed still holds.
    assert _counted_offset("12 days") == 12


def test_the_follow_up_task_is_persisted_not_just_returned(db, audit, tmp_path):
    """FINDING: the task lived on an in-memory object, so it closed no loop."""
    case = by_name("vomiting_see_today")
    service, encounter, draft, _, _ = run_to_draft(db, audit, tmp_path, case)
    service.sign(
        encounter, signed_by="ma_jess", now=NOW, patient_label=case.patient_label,
        acknowledged_drops=True,
    )
    open_tasks = service.open_followups()
    assert len(open_tasks) == 1
    assert open_tasks[0]["encounter_id"] == encounter.encounter_id
    assert open_tasks[0]["due_utc"]


def test_a_declined_recording_anywhere_in_the_history_blocks_authorisation(db, audit):
    """FINDING: `authorise_recording` looked at the latest consent row only.

    Illinois is two-party consent (720 ILCS 5/14-2). A family that said no and
    was later granted by a mis-click had the decline silently outvoted.
    """
    consent = ConsentRegistry(db, audit=audit)
    consent.grant(family_id="fam", captured_by="ma", now=NOW - timedelta(days=2))
    consent.revoke(family_id="fam", now=NOW - timedelta(days=1))
    consent.grant(family_id="fam", captured_by="ma", now=NOW)
    consent.record_disclosure(
        encounter_id="enc", family_id="fam", response="declined",
        delivered_by="ma", now=NOW,
    )
    with pytest.raises(Exception) as excinfo:
        consent.authorise_recording(encounter_id="enc", family_id="fam", now=NOW)
    assert "declin" in str(excinfo.value).lower()


def test_a_disclosure_delivered_to_another_family_does_not_authorise(db, audit):
    """FINDING: the disclosure lookup keyed on encounter alone."""
    consent = ConsentRegistry(db, audit=audit)
    consent.grant(family_id="famA", captured_by="ma", now=NOW)
    consent.record_disclosure(
        encounter_id="enc", family_id="famB", response="granted",
        delivered_by="ma", now=NOW,
    )
    with pytest.raises(RecordingNotAuthorised, match="famB"):
        consent.authorise_recording(encounter_id="enc", family_id="famA", now=NOW)


def test_placeholder_protocol_ids_cannot_reach_a_chart(db, audit, tmp_path):
    """FINDING: nothing stopped "PLACEHOLDER-FEVER" being documented as the
    protocol applied. The repo ships identifiers so the code runs; a clinic
    loads the practice's licensed list."""
    registry = ProtocolRegistry.load()  # no allow_placeholder
    assert registry.is_placeholder
    with pytest.raises(TapsIncomplete, match="placeholder"):
        registry.validate(
            MATaps(
                protocol_id="PLACEHOLDER-FEVER", disposition_id="home_care",
                tapped_by="ma", tapped_utc=NOW,
            )
        )



def _recording(recorder, encounter_id):
    """A real Recorder capture, authorisation and all. No stand-in objects."""
    from modules.triage.consent import RecordingAuthorisation

    auth = RecordingAuthorisation(
        encounter_id=encounter_id, family_id="fam", consent_id="c",
        delivery_id="d", authorised_utc=NOW, basis="verbal_on_call",
    )
    return recorder.capture(
        authorisation=auth, encounter_id=encounter_id,
        audio_bytes=silent_wav(1.0), started_utc=NOW,
        ended_utc=NOW + timedelta(minutes=1),
    )


def _run_suite(cases, *, grounding_threshold, responses=None):
    """The regression suite over the real pipeline objects."""
    from modules.triage.consent import RecordingAuthorisation

    auth = RecordingAuthorisation(
        encounter_id="x", family_id="f", consent_id="c", delivery_id="d",
        authorised_utc=NOW, basis="verbal_on_call",
    )
    transcriber = ScriptedTranscriber(
        transcripts={c.name: c.transcript_payload for c in cases}
    )
    structurer = NoteStructurer(
        LLMClient(EchoTransport(responses or [c.model_response for c in cases])),
        grounding_threshold=grounding_threshold,
    )
    return run_regression(
        structurer,
        cases,
        transcriber=transcriber,
        recorder_factory=lambda case: AudioRecording(
            case.name, "/dev/null", NOW, NOW, "0", 1.0, auth
        ),
    )



def test_a_relationship_the_caller_did_not_claim_is_flagged():
    """FINDING: the synonym table asked whether a word was said, not who said it.

    The sitter calls and mentions the mother; the note documents the mother as
    the informant. That is who the practice will say it took the history from.
    The MA's own opening -- "is mom or dad available?" -- granted the same
    licence before the caller had said anything at all.
    """
    case = by_name("third_party_relationship_spoken")
    note = NoteStructurer(
        LLMClient(EchoTransport([case.model_response]))
    ).structure(transcript_for(case))
    flagged = [f["field"] for f in note.flagged]
    assert "caller.relationship_to_patient" in flagged
    assert note.data["caller"]["relationship_to_patient"] == "mother"  # marked, not erased


def test_a_confirmed_identification_is_not_flagged():
    """The other direction: "is this Ezra's dad?" / "Yes, this is Daniel." is how
    most triage calls open, and a flag on that is a flag nobody reads."""
    case = by_name("newborn_fever_ed_now")
    note = NoteStructurer(
        LLMClient(EchoTransport([case.model_response]))
    ).structure(transcript_for(case))
    assert not any(f["field"].startswith("caller.") for f in note.flagged)
    assert note.data["caller"]["relationship_to_patient"] == "father"


def test_a_relationship_field_carrying_extra_text_is_scored_not_waved_through():
    """FINDING: one recognised word skipped the grounding score for the whole
    field, so an invented name, phone number and custody assertion rode in on
    the word "mother" -- past the drop control that had just removed that same
    name from `caller.name`."""
    case = by_name("toddler_fever_home_care")
    response = json.loads(case.model_response)
    response["caller"]["relationship_to_patient"] = (
        "mother, Katarina Petrov, cell 847-555-0148, has sole custody"
    )
    note = NoteStructurer(
        LLMClient(EchoTransport([json.dumps(response)]))
    ).structure(transcript_for(case))
    assert [f["field"] for f in note.flagged] == ["caller.relationship_to_patient"]


def test_a_conditional_followup_still_gets_a_due_date():
    """FINDING: `_NEGATED` blanked the whole timeframe, so the most ordinary
    phrasing a parent hears came back undated -- and `open_followups` sorts
    undated tasks to the bottom with COALESCE(due_utc,'9999')."""
    def due(timeframe):
        note = NoteStructurer(
            LLMClient(EchoTransport([json.dumps(_followup_response(timeframe))]))
        ).structure(transcript_for(by_name("toddler_fever_home_care")))
        task = build_followup_task(note, patient_id="p", now=NOW, assigned_to="ma")
        return None if task.due_utc is None else (task.due_utc - NOW).days

    assert due("in 2 days if she's not better") == 2
    assert due("call back in 3 days, sooner if he gets worse") == 3
    assert due("two weeks unless the rash spreads") == 14
    assert due("tomorrow if no improvement") == 1
    # A refusal is still a refusal.
    assert due("not today") is None
    # And the earliest date any rule finds wins, rather than whichever ran first.
    assert due("tomorrow, and again in 2 weeks") == 1
    assert due("tonight, then a recheck in 3 days") == 0


def _followup_response(timeframe):
    case = by_name("toddler_fever_home_care")
    response = json.loads(case.model_response)
    response["followup_discussed"] = True
    response["followup_timeframe"] = timeframe
    return response


def test_the_regression_gate_refuses_a_corpus_that_proves_nothing():
    """FINDING: "0 missed adversarial drop(s)" was printed by runs that ran no
    adversarial case, and by 50 copies of one twenty-second call."""
    clean = [c for c in CASES if not c.expect.get("expected_drops")]
    result = _run_suite(expand_to_from(clean, 50), grounding_threshold=0.0)
    assert result.overall == 1.0 and result.missed_drops == []
    assert not result.passes()
    assert any("adversarial" in b for b in result.blockers())

    one_call = expand_to_from([by_name("very_short_call")], 50)
    thin = _run_suite(one_call, grounding_threshold=0.0)
    assert not thin.passes()
    assert any("distinct transcript" in b for b in thin.blockers())


def test_the_regression_gate_refuses_a_model_that_drops_one_field():
    """FINDING: no per-field floor. Emitting no safety-net instruction on half
    the corpus scored 0.50 on that field and 0.9500 overall, and passed."""
    cases = expand_to(50)
    responses = []
    for index, case in enumerate(cases):
        payload = json.loads(case.model_response)
        if index % 2 == 0:
            payload["safety_net_instructions_given"] = []
        responses.append(json.dumps(payload))
    result = _run_suite(cases, grounding_threshold=0.6, responses=responses)
    assert not result.passes()
    assert any("safety_net_instructions_given" in b for b in result.blockers())


def expand_to_from(cases, count):
    """Tile a chosen subset, for corpus-shape tests."""
    out = []
    while len(out) < count:
        base = cases[len(out) % len(cases)]
        out.append(base if len(out) < len(cases) else _clone(base, len(out)))
    return out


def _clone(base, index):
    from dataclasses import replace

    return replace(base, name=f"{base.name}__r{index}")
