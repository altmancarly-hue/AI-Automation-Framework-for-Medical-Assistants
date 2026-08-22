"""Core-layer tests. Real objects only -- no mocks of our own logic.

EchoTransport is a shipped transport, not a test mock: these tests exercise the
production LLMClient validation/repair/refusal path against scripted model text.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from nsp_core.audit import AuditLog, AuditIntegrityError, ReviewOutcome, edit_distance, edit_ratio
from nsp_core.llm import (
    BAARequired,
    BedrockTransport,
    EchoTransport,
    LLMClient,
    SchemaViolation,
    assert_strict_schema,
    build_transport,
    extract_json,
)
from nsp_core.phi import (
    HybridDeidentifier,
    LeakGuard,
    PHILeakError,
    RegexDeidentifier,
    TokenVault,
    truncate_token_safe,
)

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "confidence"],
    "properties": {
        "verdict": {"type": "string", "enum": ["MATCH", "NO_MATCH", "UNCERTAIN"]},
        "confidence": {"type": "number"},
        "notes": {"type": "string"},
    },
}


# --------------------------------------------------------------------------
# llm.py
# --------------------------------------------------------------------------


def test_structured_returns_validated_data():
    client = LLMClient(EchoTransport([json.dumps({"verdict": "MATCH", "confidence": 0.9})]))
    result = client.structured(system="s", user="u", schema=SCHEMA, prompt_template_id="t1")
    assert result.data["verdict"] == "MATCH"
    assert result.confidence == 0.9
    assert result.provider == "echo"
    assert result.repair_attempts == 0
    assert result.prompt_template_id == "t1"
    assert len(result.prompt_template_hash) == 16


def test_structured_repairs_then_succeeds():
    client = LLMClient(
        EchoTransport(
            [
                "I think it's a match, honestly.",
                json.dumps({"verdict": "UNCERTAIN", "confidence": 0.4}),
            ]
        )
    )
    result = client.structured(system="s", user="u", schema=SCHEMA)
    assert result.repair_attempts == 1
    assert result.data["verdict"] == "UNCERTAIN"


def test_structured_raises_rather_than_defaulting():
    client = LLMClient(EchoTransport(["nope", "still nope", "nope again"]), max_repair_attempts=2)
    with pytest.raises(SchemaViolation):
        client.structured(system="s", user="u", schema=SCHEMA)


def test_additional_properties_rejected_by_validator():
    client = LLMClient(
        EchoTransport([json.dumps({"verdict": "MATCH", "confidence": 1.0, "extra": 1})] * 3)
    )
    with pytest.raises(SchemaViolation):
        client.structured(system="s", user="u", schema=SCHEMA)


def test_loose_schema_is_refused_at_the_call_site():
    loose = {"type": "object", "properties": {"a": {"type": "string"}}}
    client = LLMClient(EchoTransport([json.dumps({"a": "x"})]))
    with pytest.raises(ValueError, match="additionalProperties"):
        client.structured(system="s", user="u", schema=loose)


def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```')["a"] == 1
    assert extract_json('Sure! {"a": {"b": 2}} hope that helps')["a"]["b"] == 2
    with pytest.raises(SchemaViolation):
        extract_json("no object here")


def test_nested_strict_schema_enforced():
    nested = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"inner": {"type": "object", "properties": {}}},
    }
    with pytest.raises(ValueError):
        assert_strict_schema(nested)


def test_bedrock_refuses_without_baa():
    with pytest.raises(BAARequired):
        BedrockTransport("anthropic.claude-x")
    with pytest.raises(BAARequired):
        BedrockTransport("anthropic.claude-x", baa_on_file=True)
    t = BedrockTransport("anthropic.claude-x", baa_on_file=True, baa_reference="BAA-2026-014")
    assert t.is_cloud is True


def test_default_transport_is_local():
    assert build_transport("echo").is_cloud is False
    assert build_transport("ollama").is_cloud is False
    with pytest.raises(ValueError):
        build_transport("openai")


def test_schema_violation_never_carries_payload():
    exc = SchemaViolation("bad", raw="PATIENT NAME MAYA WHITFIELD MRN 449182")
    assert "MAYA" not in str(exc)
    assert exc.raw_length == 38


# --------------------------------------------------------------------------
# audit.py
# --------------------------------------------------------------------------


@pytest.fixture()
def log(tmp_path):
    return AuditLog(tmp_path / "audit.sqlite3", hmac_key=b"unit-test-key")


def test_edit_distance_and_ratio():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("same", "same") == 0
    assert edit_ratio("", "") == 0.0
    assert edit_ratio("abcd", "abcd") == 0.0
    assert 0.0 < edit_ratio("abcd", "abcz") < 1.0


def test_patient_ref_is_hmac_not_reversible(log):
    ref = log.patient_ref("MRN449182")
    assert ref is not None
    assert "449182" not in ref
    assert len(ref) == 64
    assert log.patient_ref("MRN449182") == ref
    assert log.patient_ref("MRN449183") != ref
    assert log.patient_ref(None) is None


def test_append_only_update_is_refused(log):
    log.record_inference(
        user_id="ma01",
        initiative_id="I-04",
        provider="ollama",
        model_id="qwen2.5:14b",
        model_version="q5_K_M",
        prompt_template_id="A.1",
        prompt_template_hash="deadbeefdeadbeef",
        input_token_count=800,
        output_token_count=200,
        patient_id="P1",
    )
    with pytest.raises(sqlite3.IntegrityError):
        log._conn.execute("UPDATE inference SET user_id='forged'")
    with pytest.raises(sqlite3.IntegrityError):
        log._conn.execute("DELETE FROM inference")
    assert log.counts()["inference"] == 1


def test_integrity_check_fails_closed_when_triggers_dropped(tmp_path):
    path = tmp_path / "a.sqlite3"
    log = AuditLog(path, hmac_key=b"k")
    log._conn.execute("DROP TRIGGER trg_review_no_delete")
    log._conn.commit()
    with pytest.raises(AuditIntegrityError):
        log.verify_integrity_controls()


def test_record_review_computes_distance_itself(log):
    rid = log.record_review(
        reviewer_id="ma01",
        initiative_id="I-04",
        draft="Mother reports fever 101.2 since last night.",
        final="Mother reports fever 101.2F since last night, no rash.",
        action_taken=ReviewOutcome.EDITED,
    )
    row = log.query("SELECT * FROM review WHERE id = ?", (rid,))[0]
    assert row["edit_distance_draft_final"] > 0
    assert 0.0 < row["edit_ratio"] < 1.0
    with pytest.raises(ValueError):
        log.record_review(
            reviewer_id="ma01", initiative_id="I-04", draft="a", final="b", action_taken="signed"
        )


def test_rubber_stamp_alarm_fires_on_zero_edit_reviewer(log):
    for _ in range(12):
        log.record_review(
            reviewer_id="ma_stamp",
            initiative_id="I-04",
            draft="identical draft text",
            final="identical draft text",
            action_taken=ReviewOutcome.ACCEPTED,
        )
    for i in range(12):
        log.record_review(
            reviewer_id="ma_careful",
            initiative_id="I-04",
            draft="a draft that needed real work " + str(i),
            final="a substantially rewritten note " + str(i) + " with added detail",
            action_taken=ReviewOutcome.EDITED,
        )
    findings = {f.reviewer_id: f for f in log.rubber_stamp_report(initiative_id="I-04")}
    assert findings["ma_stamp"].alarm is True
    assert findings["ma_stamp"].zero_edit_fraction == 1.0
    assert findings["ma_careful"].alarm is False


def test_rubber_stamp_needs_minimum_sample(log):
    for _ in range(3):
        log.record_review(
            reviewer_id="new_hire",
            initiative_id="I-04",
            draft="x",
            final="x",
            action_taken=ReviewOutcome.ACCEPTED,
        )
    assert log.rubber_stamp_report()[0].alarm is False


def test_probe_catch_rate(log):
    for caught in [True] * 7 + [False] * 3:
        log.record_review(
            reviewer_id="ma01",
            initiative_id="I-01",
            draft="d",
            final="d" if not caught else "corrected",
            action_taken=ReviewOutcome.EDITED if caught else ReviewOutcome.ACCEPTED,
            synthetic_probe=True,
            probe_caught=caught,
        )
    report = log.probe_catch_rate(initiative_id="I-01")
    assert report["probes"] == 10
    assert report["catch_rate"] == 0.7
    assert report["alarm"] is True
    assert report["meets_target"] is False


def test_probes_excluded_from_rubber_stamp_stats(log):
    log.record_review(
        reviewer_id="ma01",
        initiative_id="I-01",
        draft="d",
        final="d",
        action_taken=ReviewOutcome.ACCEPTED,
        synthetic_probe=True,
        probe_caught=False,
    )
    assert log.rubber_stamp_report() == []


def test_delegation_evidence_and_gaps(log):
    log.record_delegated_execution(
        staff_id="ma01",
        initiative_id="I-10",
        task_code="VACC_ADMIN",
        supervising_pro_id="dr_ruiz",
        supervisor_on_site=True,
        standing_order_id="SO-014",
        standing_order_version="3",
        competency_record_id="CR-77",
        competency_expires="2027-03-01",
    )
    log.record_delegated_execution(
        staff_id="ma02",
        initiative_id="I-10",
        task_code="VACC_ADMIN",
        supervising_pro_id=None,
        supervisor_on_site=False,
    )
    assert len(log.delegation_evidence()) == 2
    gaps = log.unevidenced_supervision()
    assert len(gaps) == 1
    assert gaps[0]["staff_id"] == "ma02"


def test_break_glass_requires_reason_and_always_surfaces(log):
    with pytest.raises(ValueError):
        log.record_delegated_execution(
            staff_id="ma03",
            initiative_id="I-10",
            task_code="EPI_ADMIN",
            supervising_pro_id="dr_ruiz",
            supervisor_on_site=True,
            break_glass=True,
        )
    log.record_delegated_execution(
        staff_id="ma03",
        initiative_id="I-10",
        task_code="EPI_ADMIN",
        supervising_pro_id="dr_ruiz",
        supervisor_on_site=True,
        standing_order_id="SO-002",
        competency_record_id="CR-9",
        break_glass=True,
        break_glass_reason="anaphylaxis, order system offline",
    )
    assert len(log.unevidenced_supervision()) == 1


def test_query_refuses_non_select(log):
    with pytest.raises(AuditIntegrityError):
        log.query("DELETE FROM review")


# --------------------------------------------------------------------------
# phi.py
# --------------------------------------------------------------------------


def test_regex_detects_structured_identifiers():
    text = (
        "Call mom at (847) 555-0123 or parent@example.com. MRN: 449182. "
        "DOB 03/14/2019. ZIP 60089. SSN 123-45-6789."
    )
    labels = {e.label for e in RegexDeidentifier().detect(text)}
    assert {"PHONE", "EMAIL", "MRN", "DATE", "ZIP", "SSN"} <= labels


def test_deidentify_is_reversible_and_stable():
    deid = HybridDeidentifier(use_ner=False)
    text = "Reached parent at (847) 555-0123; callback (847) 555-0123. MRN: 449182."
    result = deid.deidentify(text)
    assert "555-0123" not in result.text
    assert "449182" not in result.text
    # Same value -> same token, both occurrences.
    assert result.text.count("[[PHONE_1]]") == 2
    assert result.vault.rehydrate(result.text) == text


def test_rehydrate_obj_walks_nested_structures():
    vault = TokenVault()
    token = vault.token_for("NAME", "Maya Whitfield")
    payload = {
        "summary": f"{token} seen today",
        "items": [{"detail": f"guardian of {token}"}, ["nested", token]],
        "count": 3,
    }
    out = vault.rehydrate_obj(payload)
    assert out["summary"] == "Maya Whitfield seen today"
    assert out["items"][0]["detail"] == "guardian of Maya Whitfield"
    assert out["items"][1][1] == "Maya Whitfield"
    assert out["count"] == 3


def test_unknown_token_survives_rehydration_untouched():
    vault = TokenVault()
    assert vault.rehydrate("[[NAME_9]] here") == "[[NAME_9]] here"


def test_truncate_never_splits_a_token():
    vault = TokenVault()
    tok = vault.token_for("NAME", "Maya Whitfield")
    text = f"padding padding {tok} tail"
    cut = truncate_token_safe(text, len("padding padding [[NAME_"))
    assert "[[" not in cut
    assert truncate_token_safe(text, 10_000) == text
    assert truncate_token_safe(text, 0) == ""


def test_leak_guard_blocks_identifiers_on_egress():
    guard = LeakGuard()
    guard.assert_clean({"note": "patient seen for otitis media, afebrile"})
    with pytest.raises(PHILeakError):
        guard.assert_clean({"note": "call (847) 555-0123"})


def test_leak_guard_catches_vaulted_values_regex_would_miss():
    vault = TokenVault()
    vault.token_for("NAME", "Maya Whitfield")
    guard = LeakGuard()
    guard.assert_clean("no identifiers here", vault=vault)
    with pytest.raises(PHILeakError):
        guard.assert_clean({"a": ["Maya Whitfield attended"]}, vault=vault)


def test_deidentified_payload_passes_the_guard():
    deid = HybridDeidentifier(use_ner=False)
    result = deid.deidentify("Parent (847) 555-0123, MRN: 449182, DOB 03/14/2019")
    LeakGuard().assert_clean(result.text)


# ==========================================================================
# Regression tests — each reproduced a real defect before its fix
# ==========================================================================


def test_audit_log_is_thread_safe(tmp_path):
    """Unguarded, `check_same_thread=False` loses rows with no exception.

    Measured on the pre-fix version: 400 concurrent record_event calls, 298 rows
    persisted, 74 vanished silently. `InboundRouter` writes consent revocations
    from webhook handlers, which any real server runs concurrently -- a
    revocation that fails to persist is the exact evidentiary failure this log
    exists to prevent.
    """
    import threading

    log = AuditLog(tmp_path / "a.sqlite3", hmac_key=b"k")
    errors: list = []
    barrier = threading.Barrier(8)

    def writer(worker: int) -> None:
        try:
            barrier.wait(timeout=10)
            for i in range(50):
                log.record_event(
                    actor_id=f"w{worker}",
                    initiative_id="I-07",
                    event_type="consent_revoked",
                    detail={"i": i},
                )
        except Exception as exc:  # pragma: no cover - only on a genuine failure
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(w,)) for w in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors
    assert log.counts()["event"] == 400


def test_unevidenced_supervision_honours_the_since_window(log):
    """Without brackets, `AND timestamp_utc >= ?` binds to the last OR term only."""
    log.record_delegated_execution(
        staff_id="ma02",
        initiative_id="I-10",
        task_code="VACC_ADMIN",
        supervising_pro_id=None,
        supervisor_on_site=False,
    )
    assert len(log.unevidenced_supervision()) == 1
    assert log.unevidenced_supervision(since="2999-01-01T00:00:00+00:00") == []


def test_integrity_check_rejects_a_defanged_trigger(tmp_path):
    """A trigger that exists but no longer aborts passes a name-only check."""
    path = tmp_path / "a.sqlite3"
    log = AuditLog(path, hmac_key=b"k")
    log._conn.execute("DROP TRIGGER trg_event_no_delete")
    log._conn.execute(
        "CREATE TRIGGER trg_event_no_delete BEFORE DELETE ON event BEGIN SELECT 1; END"
    )
    log._conn.commit()
    with pytest.raises(AuditIntegrityError, match="no longer abort"):
        log.verify_integrity_controls()


def test_partially_overlapping_spans_are_trimmed_not_dropped():
    """An NER span misaligned by a few characters must still redact its tail.

    Before the fix, `_merge` discarded any span starting inside a kept span. A
    PATIENT span reported four characters early lost to the DATE span, the name
    was never tokenised, and LeakGuard could not catch it either -- it was never
    vaulted and no regex matches a name.
    """
    from nsp_core.phi import ClinicalNERDeidentifier, Entity

    text = "Well visit 12/25/2024 Maya Whitfield, DOB noted."

    class StubNER(ClinicalNERDeidentifier):
        """A real subclass, not a mock: only `detect` is replaced, and it
        reproduces the ordinary boundary-misalignment failure mode."""

        def __init__(self) -> None:
            super().__init__(eager=False)

        def detect(self, text: str) -> list[Entity]:
            start = text.index("2024 Maya Whitfield")
            end = text.index("Whitfield") + len("Whitfield")
            return [Entity(start, end, "PATIENT", text[start:end], "ner", 0.99)]

    deid = HybridDeidentifier(ner=StubNER())
    result = deid.deidentify(text)
    assert "Maya" not in result.text
    assert "Whitfield" not in result.text
    assert "12/25/2024" not in result.text
    assert result.vault.rehydrate(result.text) == text
    LeakGuard().assert_clean(result.text, vault=result.vault)


def test_fully_contained_spans_are_still_deduplicated():
    from nsp_core.phi import Entity
    from nsp_core.phi import _merge

    outer = Entity(0, 10, "A", "0123456789", "regex", 1.0)
    inner = Entity(2, 6, "B", "2345", "ner", 0.9)
    assert [e.label for e in _merge([outer, inner])] == ["A"]


def test_strict_schema_check_sees_objects_without_an_explicit_type():
    """`{"properties": {...}}` with no "type" is valid JSON Schema."""
    untyped = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"inner": {"properties": {"a": {"type": "string"}}}},
    }
    with pytest.raises(ValueError, match="additionalProperties"):
        assert_strict_schema(untyped)


def test_strict_schema_check_descends_into_composition_keywords():
    composed = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "choice": {
                "anyOf": [
                    {"type": "object", "additionalProperties": False, "properties": {}},
                    {"type": "object", "properties": {"x": {"type": "string"}}},
                ]
            }
        },
    }
    with pytest.raises(ValueError, match="anyOf"):
        assert_strict_schema(composed)


def test_strict_schema_check_refuses_unresolvable_refs():
    with pytest.raises(ValueError, match=r"\$ref"):
        assert_strict_schema(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {"a": {"$ref": "#/$defs/Thing"}},
            }
        )
