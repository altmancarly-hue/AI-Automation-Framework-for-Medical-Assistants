"""I-02 tests. Real rules engine, real database, real EchoTransport.

`EchoTransport` is a shipped transport, not a mock of our logic: these tests
drive the production `LLMClient` path -- schema validation, repair, refusal --
against scripted model text, so the grounding and fallback controls are
exercised for real rather than asserted about.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from modules.immunization import (
    Adjudicator,
    AdministeredDose,
    Antigen,
    CrossCheckForecaster,
    Determination,
    DoseRecord,
    DosePrecision,
    LocalRulesForecaster,
    MessageDrafter,
    PatientInput,
    Provenance,
    RecallEngine,
    RecallNotAuthorized,
    RegistryForecaster,
    ReplyIntent,
    ReplyTriage,
    Schedule,
    Status,
    ValidationResult,
    add_period,
    apply_adjudications,
    build_huddle,
    components_for,
    expand,
    is_known,
    normalise_code,
    reconcile,
    run_nightly,
    same_antigen_set,
    shares_any_antigen,
    unknown_codes,
    validate_against_reference,
)
from modules.immunization.adjudicate import (
    ADJUDICATION_SCHEMA,
    ADJUDICATION_SYSTEM_PROMPT,
)
from modules.immunization.fixtures import CASES, TODAY, by_name, years_before
from modules.immunization.recall import RECALL_CONSENT_BASIS
from modules.scheduling import (
    Channel,
    ConsentPurpose,
    Database,
    FrequencyCap,
    LocalGateway,
    VisitType,
    add_appointment,
    add_family,
    add_patient,
    add_provider,
    grant_consent,
    revoke_consent,
)
from nsp_core.audit import AuditLog
from nsp_core.llm import EchoTransport, LLMClient

CHI = ZoneInfo("America/Chicago")


def local(y, m, d, hh=0, mm=0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=CHI)


@pytest.fixture()
def forecaster() -> LocalRulesForecaster:
    return LocalRulesForecaster()


@pytest.fixture()
def db(tmp_path) -> Database:
    return Database(tmp_path / "i02.sqlite3")


@pytest.fixture()
def gw(tmp_path) -> LocalGateway:
    return LocalGateway(tmp_path / "outbox.jsonl")


def forecast_case(forecaster, name: str, *, as_of: date = TODAY):
    case = by_name(name)
    return case, forecaster.forecast(
        patient_id=case.patient_id,
        dob=case.dob,
        doses=case.chart_doses(),
        as_of=as_of,
    )


# ==========================================================================
# cvx.py — combination expansion is what everything else rests on
# ==========================================================================


def test_combination_product_expands_to_its_components():
    assert set(components_for("110")) == {Antigen.DTAP, Antigen.HEPB, Antigen.IPV}
    assert set(components_for("146")) == {
        Antigen.DTAP, Antigen.IPV, Antigen.HIB, Antigen.HEPB,
    }
    assert [e["antigen"] for e in expand("94")] == [Antigen.MMR, Antigen.VAR]


def test_codes_normalise_across_padding_and_trade_names():
    assert normalise_code("8") == "08"
    assert normalise_code(" 08 ") == "08"
    assert normalise_code("Pediarix") == "110"
    assert normalise_code("Gardasil 9") == "165"
    assert normalise_code("Pediarix (DTaP-HepB-IPV)") == "110"
    assert normalise_code(None) is None


def test_unknown_code_is_reported_not_silently_dropped():
    assert is_known("999") is False
    assert components_for("999") == ()
    assert unknown_codes(["08", "999", "999", "110"]) == ["999"]


def test_antigen_set_equivalence_versus_overlap():
    # Same antigen, different formulation code: equivalent.
    assert same_antigen_set("08", "45") is True
    # Combination against one of its components: overlapping, NOT equivalent.
    assert same_antigen_set("110", "20") is False
    assert shares_any_antigen("110", "20") is True
    assert shares_any_antigen("110", "03") is False


# ==========================================================================
# forecast.py — the rules engine
# ==========================================================================


def test_no_language_model_anywhere_in_the_forecasting_path():
    """The forecaster must never import a model. README I-02 calls it negligent."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "modules" / "immunization"
    banned = {"nsp_core.llm", "openai", "anthropic"}
    offenders = []
    for name in ("forecast.py", "cvx.py", "matcher.py", "recall.py", "huddle.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = ["." * node.level + (node.module or "")]
            else:
                continue
            for mod in mods:
                if any(mod == b or mod.startswith(b + ".") for b in banned):
                    offenders.append(f"{name}: {mod}")
    assert offenders == [], offenders


def test_add_period_clamps_at_month_end():
    assert add_period(date(2026, 1, 31), {"months": 1}) == date(2026, 2, 28)
    assert add_period(date(2024, 1, 31), {"months": 1}) == date(2024, 2, 29)
    assert add_period(date(2026, 8, 22), {"years": 1}) == date(2027, 8, 22)
    assert add_period(date(2026, 8, 22), {"weeks": 6}) == date(2026, 10, 3)
    assert add_period(date(2026, 8, 22), None) == date(2026, 8, 22)


def test_grace_period_counts_a_dose_given_three_days_early(forecaster):
    _, forecast = forecast_case(forecaster, "grace_period_three_days_early")
    mmr = forecast.antigens["mmr"]
    assert mmr.doses_valid == 1
    assert mmr.invalid_doses == []


def test_a_dose_given_three_weeks_early_does_not_count(forecaster):
    _, forecast = forecast_case(forecaster, "invalid_dose_too_early")
    mmr = forecast.antigens["mmr"]
    assert mmr.doses_valid == 0
    assert len(mmr.invalid_doses) == 1
    assert "minimum age" in mmr.invalid_doses[0].reason


def test_dtap_fourth_dose_after_the_fourth_birthday_completes_the_series(forecaster):
    """A naive forecaster recalls this child for a fifth DTaP they do not need."""
    _, forecast = forecast_case(forecaster, "dtap_four_doses_complete")
    dtap = forecast.antigens["dtap"]
    assert dtap.status == Status.COMPLETE
    assert dtap.doses_required == 4
    assert dtap.doses_valid == 4


def test_rotavirus_ages_out_and_stops_being_a_gap(forecaster):
    _, forecast = forecast_case(forecaster, "rotavirus_aged_out")
    rota = forecast.antigens["rota"]
    assert rota.status == Status.AGED_OUT
    assert rota.is_open_gap is False


def test_rotavirus_carries_a_hard_deadline_while_still_in_window(forecaster):
    _, forecast = forecast_case(forecaster, "rotavirus_closing_window")
    rota = forecast.antigens["rota"]
    assert rota.status in (Status.DUE, Status.OVERDUE)
    assert rota.age_out_hard_date is not None
    assert rota.age_out_hard_date > TODAY


def test_hpv_series_length_forks_at_the_fifteenth_birthday(forecaster):
    _, early = forecast_case(forecaster, "hpv_two_dose_started")
    _, late = forecast_case(forecaster, "hpv_three_dose_late_start")
    assert early.antigens["hpv"].doses_required == 2
    assert late.antigens["hpv"].doses_required == 3


def test_mixed_hib_variants_use_the_longer_conservative_schedule(forecaster):
    _, forecast = forecast_case(forecaster, "hib_mixed_variants")
    hib = forecast.antigens["hib"]
    assert hib.doses_required == 4
    assert any("mixed product variants" in n for n in hib.notes)


def test_hib_and_pcv_are_not_required_after_five(forecaster):
    _, forecast = forecast_case(forecaster, "kindergarten_school_gaps", as_of=TODAY)
    # This child is 5; Hib and PCV catch-up is not indicated for a healthy child.
    assert forecast.antigens["hib"].status == Status.NOT_REQUIRED
    assert forecast.antigens["pcv"].status == Status.NOT_REQUIRED


def test_varicella_second_dose_interval_depends_on_age(forecaster):
    _, forecast = forecast_case(forecaster, "varicella_interval_under_13")
    var = forecast.antigens["var"]
    assert var.doses_valid == 1
    assert len(var.invalid_doses) == 1
    assert "minimum interval" in var.invalid_doses[0].reason


def test_partial_date_holds_the_antigen_for_review(forecaster):
    case = by_name("partial_date_history")
    forecast = forecaster.forecast(
        patient_id=case.patient_id, dob=case.dob, doses=case.chart_doses(), as_of=TODAY
    )
    mmr = forecast.antigens["mmr"]
    assert mmr.status == Status.REQUIRES_REVIEW
    assert mmr.is_open_gap is False
    assert any("partial date" in n for n in mmr.notes)


def test_unknown_code_makes_every_open_gap_unconfirmed(forecaster):
    _, forecast = forecast_case(forecaster, "unknown_cvx_code")
    assert forecast.unknown_codes == ["999"]
    assert all(
        any("unrecognised" in n for n in gap.notes) for gap in forecast.open_gaps
    )


def test_influenza_is_seasonal_not_a_series(forecaster):
    dob = years_before(6)
    # A dose from the season that is still current (Sept 2025 - Aug 2026).
    forecast = forecaster.forecast(
        patient_id="p", dob=dob,
        doses=[AdministeredDose("150", date(TODAY.year - 1, 10, 3))],
        as_of=TODAY,
    )
    assert forecast.antigens["influenza"].status == Status.COMPLETE

    stale = forecaster.forecast(
        patient_id="p", dob=dob,
        doses=[AdministeredDose("150", date(TODAY.year - 2, 10, 3))],
        as_of=TODAY,
    )
    assert stale.antigens["influenza"].status == Status.DUE


def test_a_june_flu_shot_is_not_due_again_three_weeks_later(forecaster):
    """A July season boundary made the whole panel DUE on 1 July, months before
    the new season's vaccine exists anywhere."""
    dob = years_before(6)
    june_dose = [AdministeredDose("150", date(2026, 6, 10))]
    for as_of in (date(2026, 6, 30), date(2026, 7, 1), date(2026, 8, 22)):
        forecast = forecaster.forecast(
            patient_id="p", dob=dob, doses=june_dose, as_of=as_of
        )
        assert forecast.antigens["influenza"].status == Status.COMPLETE, as_of
    # The new season does begin, in September.
    forecast = forecaster.forecast(
        patient_id="p", dob=dob, doses=june_dose, as_of=date(2026, 9, 2)
    )
    assert forecast.antigens["influenza"].status == Status.DUE


def test_up_to_date_adolescent_has_no_school_required_gaps(forecaster):
    _, forecast = forecast_case(forecaster, "adolescent_complete")
    assert [g.antigen for g in forecast.open_gaps if g.school_required] == []


def test_adolescent_cluster_is_the_most_missed_category(forecaster):
    _, forecast = forecast_case(forecaster, "adolescent_gap_cluster")
    gaps = {g.antigen for g in forecast.open_gaps}
    assert {"tdap", "menacwy", "hpv"} <= gaps


def test_transferred_in_child_gets_a_real_catch_up_list(forecaster):
    _, forecast = forecast_case(forecaster, "transferred_in_catch_up")
    gaps = {g.antigen: g.status for g in forecast.open_gaps}
    assert gaps["mmr"] == Status.OVERDUE
    assert gaps["var"] == Status.OVERDUE
    assert forecast.antigens["dtap"].doses_valid == 2


def test_every_fixture_forecasts_without_raising(forecaster):
    for case in CASES:
        forecast = forecaster.forecast(
            patient_id=case.patient_id, dob=case.dob, doses=case.chart_doses(),
            as_of=TODAY,
        )
        assert set(forecast.antigens) == set(Schedule.load().antigens)
        for antigen in forecast.antigens.values():
            assert antigen.status in Status.ALL


# ==========================================================================
# The authority abstraction and the go-live validation gate
# ==========================================================================


def test_registry_forecaster_is_an_unimplemented_seam():
    with pytest.raises(NotImplementedError, match="integration seam"):
        RegistryForecaster().forecast(
            patient_id="p", dob=date(2020, 1, 1), doses=[], as_of=TODAY
        )


def test_cross_check_reports_disagreement_and_suppresses_recall(forecaster):
    """A contested gap is never recalled on, whichever engine is primary."""
    strict = LocalRulesForecaster()
    # A second engine with a deliberately different rule table.
    other = LocalRulesForecaster(Schedule.load())
    other.schedule.antigens["mmr"]["series"][0]["overdue_age"] = {"years": 30}

    cross = CrossCheckForecaster(strict, other)
    case = by_name("transferred_in_catch_up")
    result = cross.forecast(
        patient_id=case.patient_id, dob=case.dob, doses=case.chart_doses(), as_of=TODAY
    )
    assert cross.disagreements
    mmr = result.antigens["mmr"]
    assert mmr.recall_eligible is False
    assert any("disagree" in n for n in mmr.notes)


def test_validation_only_passes_with_enough_cases_and_agreement(forecaster):
    cases = [
        {
            "patient_id": f"v{i}",
            "dob": by_name("adolescent_complete").dob,
            "doses": by_name("adolescent_complete").chart_doses(),
            "as_of": TODAY,
            "expected": {"mmr": Status.COMPLETE},
        }
        for i in range(200)
    ]
    result = validate_against_reference(forecaster, cases)
    assert result.cases == 200
    assert result.agreement_rate == 1.0
    assert result.passes() is True

    short = validate_against_reference(forecaster, cases[:10])
    assert short.passes() is False


def test_validation_counts_mismatches_rather_than_hiding_them(forecaster):
    cases = [
        {
            "patient_id": "v1",
            "dob": by_name("transferred_in_catch_up").dob,
            "doses": by_name("transferred_in_catch_up").chart_doses(),
            "as_of": TODAY,
            "expected": {"mmr": Status.COMPLETE},   # deliberately wrong
        }
    ]
    result = validate_against_reference(forecaster, cases)
    assert result.agreement_rate == 0.0
    assert result.mismatches[0].antigen == "mmr"


# ==========================================================================
# matcher.py
# ==========================================================================


def test_equivalent_codes_on_the_same_day_are_a_clean_match():
    case = by_name("equivalent_codes_match")
    result = reconcile(case.chart, case.registry)
    assert len(result.matched) == 1
    assert result.ambiguous == []


def test_one_day_drift_is_inside_tolerance():
    case = by_name("one_day_drift")
    result = reconcile(case.chart, case.registry)
    assert len(result.matched) == 1
    assert result.matched[0].date_delta_days == 1


def test_two_week_drift_escalates_rather_than_guessing():
    case = by_name("two_week_drift_ambiguous")
    result = reconcile(case.chart, case.registry)
    assert len(result.ambiguous) == 1
    assert "transcription error or two distinct" in result.ambiguous[0].reason
    assert result.unresolved_antigens == {"var"}


def test_combination_versus_components_is_one_cluster_not_a_discrepancy():
    """Pediarix in the chart, three registry rows the same day."""
    case = by_name("combination_split_across_sources")
    result = reconcile(case.chart, case.registry)
    assert result.chart_only == []
    assert result.registry_only == []
    assert len(result.ambiguous) == 3
    assert result.unresolved_antigens == {"dtap", "hepb", "ipv"}
    # And the merge contributes ONE dose, not four.
    assert len(result.merged_doses()) == 1


def test_duplicate_dose_is_reported_and_never_auto_resolved():
    case = by_name("duplicate_dose")
    result = reconcile(case.chart, case.registry)
    assert len(result.duplicates) == 1
    assert result.duplicates[0].date_delta_days == 3
    assert result.duplicates[0].source == "chart"


def test_partial_date_pair_is_ambiguous_not_matched():
    case = by_name("partial_date_history")
    result = reconcile(case.chart, case.registry)
    assert len(result.ambiguous) == 1
    assert "partial date" in result.ambiguous[0].reason
    assert result.ambiguous[0].date_delta_days is None


def test_registry_only_dose_survives_the_merge():
    case = by_name("registry_only_dose")
    result = reconcile(case.chart, case.registry)
    assert len(result.registry_only) == 1
    assert len(result.merged_doses()) == 2


def test_partial_dates_never_auto_match_and_merge_to_one_dose():
    chart = [DoseRecord("c1", "03", date(2021, 1, 1), "chart",
                        precision=DosePrecision.YEAR)]
    registry = [DoseRecord("r1", "03", date(2021, 1, 1), "registry")]
    result = reconcile(chart, registry)
    assert result.matched == []          # a partial date can never prove a match
    assert len(result.ambiguous) == 1
    assert len(result.merged_doses()) == 1


def test_an_adjudicated_match_keeps_the_more_precise_date():
    """The precision preference is reachable through adjudication, where a
    partial-date pair can legitimately become a MATCH."""
    chart = [DoseRecord("c1", "03", date(2021, 1, 1), "chart",
                        precision=DosePrecision.YEAR)]
    registry = [DoseRecord("r1", "03", date(2021, 6, 14), "registry")]
    reconciliation = reconcile(chart, registry)
    client = LLMClient(EchoTransport([_adjudication_response()]))
    outcomes = Adjudicator(client).adjudicate_reconciliation(
        reconciliation, patient_id="p1"
    )
    updated, _ = apply_adjudications(reconciliation, outcomes, reviewed_by="ma01")
    merged = updated.merged_doses()
    assert len(merged) == 1
    assert merged[0].precision == DosePrecision.DAY
    assert merged[0].given == date(2021, 6, 14)


def test_unknown_code_pairs_are_flagged_for_a_human():
    chart = [DoseRecord("c1", "999", date(2024, 5, 1), "chart")]
    registry = [DoseRecord("r1", "03", date(2024, 5, 1), "registry")]
    result = reconcile(chart, registry)
    assert result.unknown_codes == ["999"]
    assert len(result.ambiguous) == 1
    assert "unrecognised" in result.ambiguous[0].reason


# ==========================================================================
# adjudicate.py
# ==========================================================================


def _adjudication_response(**overrides):
    payload = {
        "determination": "MATCH",
        "confidence": 0.95,
        "reasoning": "Same product recorded with equivalent codes.",
        "cvx_a": None,
        "cvx_b": None,
        "date_a": None,
        "date_b": None,
        "requires_human_review": False,
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_prompt_is_appendix_a2_verbatim():
    for sentence in (
        "You may ONLY use information present in the two records.",
        "You MUST NOT infer, estimate, or reconstruct a date that appears in neither",
        "You MUST NOT determine whether a patient is due for a vaccine.",
        "Returning UNCERTAIN is a correct and\n  expected outcome",
    ):
        assert sentence in ADJUDICATION_SYSTEM_PROMPT


def test_schema_has_no_field_for_a_clinical_judgement():
    keys = set(ADJUDICATION_SCHEMA["properties"])
    assert not keys & {"due", "status", "disposition", "recommendation", "forecast"}
    assert ADJUDICATION_SCHEMA["additionalProperties"] is False


def test_adjudicator_resolves_a_confident_match(tmp_path):
    case = by_name("two_week_drift_ambiguous")
    pair = reconcile(case.chart, case.registry).ambiguous[0]
    audit = AuditLog(tmp_path / "a.sqlite3", hmac_key=b"k")
    client = LLMClient(EchoTransport([_adjudication_response()]))
    outcome = Adjudicator(client, audit=audit).adjudicate(pair, patient_id="p1")
    assert outcome.determination == Determination.MATCH
    assert outcome.auto_resolvable is True
    assert audit.counts()["inference"] == 1


def test_uncertain_is_a_correct_outcome_that_routes_to_a_human():
    case = by_name("two_week_drift_ambiguous")
    pair = reconcile(case.chart, case.registry).ambiguous[0]
    client = LLMClient(
        EchoTransport([_adjudication_response(determination="UNCERTAIN", confidence=0.9)])
    )
    outcome = Adjudicator(client).adjudicate(pair, patient_id="p1")
    assert outcome.determination == "UNCERTAIN"
    assert outcome.requires_human_review is True
    assert outcome.auto_resolvable is False


def test_low_confidence_match_still_goes_to_a_human():
    case = by_name("two_week_drift_ambiguous")
    pair = reconcile(case.chart, case.registry).ambiguous[0]
    client = LLMClient(EchoTransport([_adjudication_response(confidence=0.4)]))
    outcome = Adjudicator(client).adjudicate(pair, patient_id="p1")
    assert outcome.auto_resolvable is False


def test_an_invented_date_discards_the_whole_response():
    """A.2 forbids reconstructing a date. This is the post-condition."""
    case = by_name("two_week_drift_ambiguous")
    pair = reconcile(case.chart, case.registry).ambiguous[0]
    client = LLMClient(
        EchoTransport([_adjudication_response(date_a="1999-01-01", confidence=0.99)])
    )
    outcome = Adjudicator(client).adjudicate(pair, patient_id="p1")
    assert outcome.auto_resolvable is False
    assert outcome.failure.startswith("ungrounded_date_a")


def test_an_invented_cvx_code_discards_the_whole_response():
    case = by_name("two_week_drift_ambiguous")
    pair = reconcile(case.chart, case.registry).ambiguous[0]
    client = LLMClient(EchoTransport([_adjudication_response(cvx_b="777")]))
    outcome = Adjudicator(client).adjudicate(pair, patient_id="p1")
    assert outcome.failure.startswith("ungrounded_cvx_b")


def test_a_grounded_date_from_the_record_is_accepted():
    case = by_name("two_week_drift_ambiguous")
    pair = reconcile(case.chart, case.registry).ambiguous[0]
    client = LLMClient(
        EchoTransport([
            _adjudication_response(
                date_a=pair.chart.given.isoformat(),
                date_b=pair.registry.given.isoformat(),
                cvx_a=pair.chart.normalised_cvx,
                cvx_b=pair.registry.normalised_cvx,
            )
        ])
    )
    outcome = Adjudicator(client).adjudicate(pair, patient_id="p1")
    assert outcome.auto_resolvable is True


def test_schema_violation_fails_closed_to_a_human(tmp_path):
    case = by_name("two_week_drift_ambiguous")
    pair = reconcile(case.chart, case.registry).ambiguous[0]
    audit = AuditLog(tmp_path / "a.sqlite3", hmac_key=b"k")
    client = LLMClient(EchoTransport(["not json", "still not", "nope"]),
                       max_repair_attempts=2)
    outcome = Adjudicator(client, audit=audit).adjudicate(pair, patient_id="p1")
    assert outcome.requires_human_review is True
    assert outcome.failure.startswith("schema_violation")
    assert audit.counts()["event"] == 1
    assert audit.counts()["inference"] == 0


def test_adjudication_payload_carries_no_identifiers():
    case = by_name("two_week_drift_ambiguous")
    pair = reconcile(case.chart, case.registry).ambiguous[0]
    client = LLMClient(EchoTransport([_adjudication_response()]))
    payload = Adjudicator(client).build_payload(pair)
    assert case.first_name not in payload
    assert case.patient_id not in payload
    assert "RECORD A" in payload and "RECORD B" in payload


def test_applying_adjudications_requires_a_named_reviewer():
    case = by_name("two_week_drift_ambiguous")
    reconciliation = reconcile(case.chart, case.registry)
    client = LLMClient(EchoTransport([_adjudication_response()]))
    outcomes = Adjudicator(client).adjudicate_reconciliation(
        reconciliation, patient_id="p1"
    )
    with pytest.raises(ValueError, match="named reviewer"):
        apply_adjudications(reconciliation, outcomes, reviewed_by="  ")


def test_reviewed_match_folds_into_the_reconciliation(tmp_path):
    case = by_name("two_week_drift_ambiguous")
    reconciliation = reconcile(case.chart, case.registry)
    audit = AuditLog(tmp_path / "a.sqlite3", hmac_key=b"k")
    client = LLMClient(EchoTransport([_adjudication_response()]))
    outcomes = Adjudicator(client, audit=audit).adjudicate_reconciliation(
        reconciliation, patient_id="p1"
    )
    updated, queue = apply_adjudications(
        reconciliation, outcomes, reviewed_by="ma01", audit=audit, patient_id="p1"
    )
    assert len(updated.matched) == 1
    assert updated.ambiguous == []
    assert queue == []
    assert audit.counts()["review"] == 1
    # A binary confirmation must not drag the rubber-stamp median to zero.
    assert audit.rubber_stamp_report() == []


def test_reviewer_override_is_recorded_as_an_edit(tmp_path):
    case = by_name("two_week_drift_ambiguous")
    reconciliation = reconcile(case.chart, case.registry)
    audit = AuditLog(tmp_path / "a.sqlite3", hmac_key=b"k")
    client = LLMClient(EchoTransport([_adjudication_response()]))
    outcomes = Adjudicator(client, audit=audit).adjudicate_reconciliation(
        reconciliation, patient_id="p1"
    )
    key = (outcomes[0].pair.chart.record_id, outcomes[0].pair.registry.record_id)
    updated, queue = apply_adjudications(
        reconciliation, outcomes, reviewed_by="dr_ruiz",
        decisions={key: Determination.NO_MATCH}, audit=audit, patient_id="p1",
    )
    assert updated.matched == []
    assert len(updated.chart_only) == 1 and len(updated.registry_only) == 1
    row = audit.query("SELECT action_taken FROM review")[0]
    assert row["action_taken"] == "edited"


def test_unresolved_pairs_stay_in_the_human_queue():
    case = by_name("two_week_drift_ambiguous")
    reconciliation = reconcile(case.chart, case.registry)
    client = LLMClient(EchoTransport([_adjudication_response(determination="UNCERTAIN")]))
    outcomes = Adjudicator(client).adjudicate_reconciliation(
        reconciliation, patient_id="p1"
    )
    updated, queue = apply_adjudications(reconciliation, outcomes, reviewed_by="ma01")
    assert len(updated.ambiguous) == 1
    assert len(queue) == 1
    assert queue[0].machine_suggestion == "UNCERTAIN"


# ==========================================================================
# messaging.py
# ==========================================================================


SLOTS = {
    "first_name": "Nia",
    "vaccine_list": "Tdap",
    "gap_count": 1,
    "booking_url": "https://nsp.example/book",
}
TEMPLATE = (
    "North Suburban Pediatrics: {first_name} is due for {vaccine_list}. "
    "Book here: {booking_url}. Reply STOP to opt out."
)


def _draft_response(message: str, confidence: float = 0.9) -> str:
    return json.dumps({"message": message, "confidence": confidence})


def test_a_clean_rewrite_is_used():
    good = (
        "Hi! Nia is due for Tdap. You can book here: https://nsp.example/book. "
        "Reply STOP to opt out."
    )
    drafter = MessageDrafter(LLMClient(EchoTransport([_draft_response(good)])))
    result = drafter.draft_detailed(
        template_id="recall_sms_1", template=TEMPLATE, slots=SLOTS
    )
    assert result.used_model is True
    assert result.message == good


def test_a_rewrite_that_invents_a_number_falls_back_to_the_approved_text():
    """The single most valuable check: a new number is a new clinical fact."""
    bad = (
        "Hi! Nia is due for Tdap - 2 doses needed by September 1. "
        "https://nsp.example/book. Reply STOP to opt out."
    )
    drafter = MessageDrafter(LLMClient(EchoTransport([_draft_response(bad)])))
    result = drafter.draft_detailed(
        template_id="recall_sms_1", template=TEMPLATE, slots=SLOTS
    )
    assert result.used_model is False
    assert result.message == TEMPLATE.format(**SLOTS)
    assert "introduced numbers" in result.fallback_reason


def test_a_rewrite_that_drops_a_slot_falls_back():
    bad = "Your child is due for a vaccine. Book: https://nsp.example/book. STOP to opt out."
    drafter = MessageDrafter(LLMClient(EchoTransport([_draft_response(bad)])))
    result = drafter.draft_detailed(
        template_id="recall_sms_1", template=TEMPLATE, slots=SLOTS
    )
    assert result.used_model is False
    assert "slot value dropped" in result.fallback_reason


def test_a_rewrite_that_removes_the_opt_out_falls_back():
    bad = "Hi! Nia is due for Tdap. Book here: https://nsp.example/book."
    drafter = MessageDrafter(LLMClient(EchoTransport([_draft_response(bad)])))
    result = drafter.draft_detailed(
        template_id="recall_sms_1", template=TEMPLATE, slots=SLOTS
    )
    assert result.used_model is False
    assert result.fallback_reason == "opt-out sentence removed"


def test_an_over_length_sms_falls_back():
    bad = (
        "Hi! Nia is due for Tdap. " + ("a very warm and friendly sentence. " * 12)
        + "https://nsp.example/book Reply STOP to opt out."
    )
    drafter = MessageDrafter(LLMClient(EchoTransport([_draft_response(bad)])))
    result = drafter.draft_detailed(
        template_id="recall_sms_1", template=TEMPLATE, slots=SLOTS
    )
    assert result.used_model is False
    assert "over the" in result.fallback_reason


def test_a_schema_violation_falls_back_rather_than_blocking_the_message():
    drafter = MessageDrafter(
        LLMClient(EchoTransport(["nope", "nope", "nope"]), max_repair_attempts=2)
    )
    result = drafter.draft_detailed(
        template_id="recall_sms_1", template=TEMPLATE, slots=SLOTS
    )
    assert result.message == TEMPLATE.format(**SLOTS)
    assert result.fallback_reason.startswith("schema_violation")


def test_stop_is_classified_without_ever_calling_a_model():
    transport = EchoTransport([])
    triage = ReplyTriage(LLMClient(transport))
    result = triage.classify("STOP")
    assert result.intent == ReplyIntent.OPT_OUT
    assert result.deterministic is True
    assert result.route == "suppression:immediate"
    assert transport.calls == []   # the model was never consulted


def test_reply_routing_sends_each_intent_to_the_right_queue():
    def triage_for(intent: str) -> ReplyTriage:
        return ReplyTriage(
            LLMClient(
                EchoTransport([json.dumps({"intent": intent, "confidence": 0.9})])
            )
        )

    assert triage_for("already_vaccinated_elsewhere").classify(
        "we got it at CVS last month"
    ).route == "queue:ma_reconciliation"
    assert triage_for("has_questions").classify("is this safe?").route == "queue:nurse_line"
    assert triage_for("wants_appointment").classify("can we come friday").route == (
        "queue:scheduling"
    )
    assert triage_for("declines").classify("no thank you").route == (
        "queue:physician_review"
    )


def test_low_confidence_classification_goes_to_a_person():
    triage = ReplyTriage(
        LLMClient(
            EchoTransport([json.dumps({"intent": "declines", "confidence": 0.2})])
        )
    )
    result = triage.classify("hmm")
    assert result.intent == ReplyIntent.OTHER
    assert result.route == "queue:front_desk"


def test_reply_metadata_never_carries_the_body():
    triage = ReplyTriage(None)
    result = triage.classify("Maya already had it at Walgreens on the 4th")
    assert "Maya" not in json.dumps(result.as_dict())
    assert result.raw_length > 0


# ==========================================================================
# recall.py
# ==========================================================================


def seed_family(db, case, *, consent=True, email=None):
    add_family(db, family_id=case.family_id, display_name=f"{case.first_name} family",
               primary_phone=f"+1847555{abs(hash(case.name)) % 10000:04d}",
               primary_email=email)
    add_patient(db, patient_id=case.patient_id, family_id=case.family_id,
                first_name=case.first_name, last_name="Demo", dob=case.dob.isoformat())
    if consent:
        grant_consent(db, family_id=case.family_id, channel=Channel.SMS,
                      purpose=ConsentPurpose.REMINDERS, granted=local(2026, 1, 1, 9),
                      capture_method="intake_form", capture_evidence="INTAKE-1")


def nightly_for(names, db=None, *, consent=True):
    cases = [by_name(n) for n in names]
    if db is not None:
        for case in cases:
            seed_family(db, case, consent=consent)
    patients = [
        PatientInput(c.patient_id, c.family_id, c.first_name, c.dob, c.chart, c.registry)
        for c in cases
    ]
    return cases, run_nightly(patients, as_of=TODAY)


def test_recall_refuses_to_send_before_validation(db, gw):
    cases, nightly = nightly_for(["adolescent_gap_cluster"], db)
    engine = RecallEngine(db, gw)
    assert engine.authorized is False
    with pytest.raises(RecallNotAuthorized, match="validated against known-good"):
        engine.run(list(nightly.forecasts.values()), now=local(2026, 8, 22, 10),
                   patients=nightly.patients)


def test_dry_run_is_always_permitted(db, gw):
    cases, nightly = nightly_for(["adolescent_gap_cluster"], db)
    engine = RecallEngine(db, gw)
    report = engine.run(list(nightly.forecasts.values()), now=local(2026, 8, 22, 10),
                        patients=nightly.patients, dry_run=True)
    assert report["dry_run"] is True
    assert report["queue_size"] > 0
    assert gw.sent == []


def _authorized_engine(db, gw, **kwargs) -> RecallEngine:
    validation = ValidationResult(
        engine="local_rules", schedule_version="test", cases=200,
        antigen_comparisons=200, agreements=200,
    )
    return RecallEngine(db, gw, validation=validation, **kwargs)


def test_authorized_engine_sends_the_first_message(db, gw):
    cases, nightly = nightly_for(["adolescent_gap_cluster"], db)
    engine = _authorized_engine(db, gw)
    report = engine.run(list(nightly.forecasts.values()), now=local(2026, 8, 22, 10),
                        patients=nightly.patients)
    assert report["outcomes"].get("sent", 0) >= 1
    assert gw.messages_for("recall_immunization")


def test_recall_uses_the_shared_frequency_cap_on_the_outreach_tier(db, gw):
    """I-07's reminders must squeeze out I-02's recall, not the reverse."""
    cases, nightly = nightly_for(["adolescent_gap_cluster"], db)
    case = cases[0]
    now = local(2026, 8, 22, 10)
    cap = FrequencyCap(db)
    limit = cap.limit_for(FrequencyCap.TIER_OUTREACH)
    for i in range(limit):
        db.execute(
            "INSERT INTO message_log (message_id, family_id, channel, purpose,"
            " template_id, planned_utc, sent_utc, status, to_address)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (f"m{i}", case.family_id, "sms", "reminder_t7", "wv_t7_confirm",
             _iso(now), _iso(now), "sent", "+1"),
        )
    engine = _authorized_engine(db, gw)
    report = engine.run(list(nightly.forecasts.values()), now=now,
                        patients=nightly.patients)
    assert gw.messages_for("recall_immunization") == []
    assert report["outcomes"].get("skipped", 0) >= 1


def _iso(dt: datetime) -> str:
    from modules.scheduling.models import iso

    return iso(dt)


def test_opted_out_family_is_never_recalled(db, gw):
    cases, nightly = nightly_for(["adolescent_gap_cluster"], db)
    revoke_consent(db, family_id=cases[0].family_id, revoked=local(2026, 8, 1))
    engine = _authorized_engine(db, gw)
    engine.run(list(nightly.forecasts.values()), now=local(2026, 8, 22, 10),
               patients=nightly.patients)
    assert gw.sent == []


def test_physician_exclusion_removes_a_family_from_the_queue(db, gw):
    cases, nightly = nightly_for(["adolescent_gap_cluster"], db)
    engine = _authorized_engine(db, gw)
    engine.exclude(reason="family declined; discussed at the 13y visit",
                   excluded_by="dr_ruiz", family_id=cases[0].family_id,
                   now=local(2026, 8, 1))
    queue = engine.build_queue(list(nightly.forecasts.values()), as_of=TODAY,
                               patients=nightly.patients)
    assert queue == []


def test_exclusion_requires_a_reason_and_an_author(db, gw):
    engine = _authorized_engine(db, gw)
    with pytest.raises(ValueError):
        engine.exclude(reason="", excluded_by="dr_ruiz", family_id="fam")
    with pytest.raises(ValueError):
        engine.exclude(reason="declined", excluded_by="", family_id="fam")
    with pytest.raises(ValueError):
        engine.exclude(reason="declined", excluded_by="dr_ruiz")


def test_a_child_with_an_upcoming_appointment_is_left_to_the_huddle_sheet(db, gw):
    cases, nightly = nightly_for(["adolescent_gap_cluster"], db)
    add_provider(db, provider_id="dr_ruiz", display_name="Dr. Ruiz")
    add_appointment(db, patient_id=cases[0].patient_id, provider_id="dr_ruiz",
                    visit_type=VisitType.WELL, start=local(2026, 9, 1, 10),
                    now=local(2026, 8, 1))
    engine = _authorized_engine(db, gw)
    queue = engine.build_queue(list(nightly.forecasts.values()), as_of=TODAY,
                               patients=nightly.patients)
    assert queue == []


def test_antigens_held_for_review_are_never_recalled(db, gw):
    cases, nightly = nightly_for(["combination_split_across_sources"], db)
    engine = _authorized_engine(db, gw)
    queue = engine.build_queue(list(nightly.forecasts.values()), as_of=TODAY,
                               patients=nightly.patients)
    assert {c.antigen for c in queue}.isdisjoint({"dtap", "hepb", "ipv"})


def test_shared_decision_and_covid_antigens_are_never_recalled(db, gw):
    cases, nightly = nightly_for(["adolescent_complete"], db)
    engine = _authorized_engine(db, gw)
    queue = engine.build_queue(list(nightly.forecasts.values()), as_of=TODAY,
                               patients=nightly.patients)
    assert "menb" not in {c.antigen for c in queue}
    assert "covid" not in {c.antigen for c in queue}


def test_urgency_is_explainable_and_school_aware(db, gw):
    cases, nightly = nightly_for(["kindergarten_school_gaps"], db)
    engine = _authorized_engine(db, gw)
    queue = engine.build_queue(list(nightly.forecasts.values()), as_of=TODAY,
                               patients=nightly.patients)
    mmr = next(c for c in queue if c.antigen == "mmr")
    assert set(mmr.breakdown) == {
        "overdue_x_weight_saturated", "school_deadline_bonus", "age_out_bonus",
    }
    assert mmr.breakdown["school_deadline_bonus"] > 0
    assert mmr.urgency == pytest.approx(sum(mmr.breakdown.values()))


def test_age_out_bonus_lifts_a_closing_rotavirus_window(db, gw):
    cases, nightly = nightly_for(["rotavirus_closing_window"], db)
    engine = _authorized_engine(db, gw)
    queue = engine.build_queue(list(nightly.forecasts.values()), as_of=TODAY,
                               patients=nightly.patients)
    rota = next(c for c in queue if c.antigen == "rota")
    assert rota.breakdown["age_out_bonus"] > 0


def test_cadence_advances_on_the_gap_open_date(db, gw):
    cases, nightly = nightly_for(["adolescent_gap_cluster"], db)
    # The weekly frequency cap is tested separately; here it would mask the
    # cadence behaviour under test.
    engine = _authorized_engine(db, gw, frequency_cap=FrequencyCap(
        db, limits={"appointment": 99, "waitlist": 99, "outreach": 99}
    ))
    forecasts = list(nightly.forecasts.values())
    day0 = local(2026, 8, 22, 10)
    engine.run(forecasts, now=day0, patients=nightly.patients)
    first = len(gw.messages_for("recall_immunization"))
    assert first >= 1

    # Same day again: nothing new is due.
    engine.run(forecasts, now=day0 + timedelta(hours=2), patients=nightly.patients)
    assert len(gw.messages_for("recall_immunization")) == first

    # Day 7: the second SMS.
    engine.run(forecasts, now=day0 + timedelta(days=7), patients=nightly.patients)
    assert len(gw.messages_for("recall_immunization")) > first


def test_cadence_stops_at_three_messages_and_queues_a_human(db, gw):
    cases, nightly = nightly_for(["adolescent_gap_cluster"], db)
    engine = _authorized_engine(db, gw, frequency_cap=FrequencyCap(
        db, limits={"appointment": 99, "waitlist": 99, "outreach": 99}
    ))
    db.execute("UPDATE family SET primary_email='parent@example.com' WHERE family_id=?",
               (cases[0].family_id,))
    grant_consent(db, family_id=cases[0].family_id, channel=Channel.EMAIL,
                  purpose=ConsentPurpose.REMINDERS, granted=local(2026, 1, 1, 9),
                  capture_method="intake_form")
    forecasts = list(nightly.forecasts.values())
    day0 = local(2026, 8, 22, 10)
    for offset in (0, 7, 21, 45):
        engine.run(forecasts, now=day0 + timedelta(days=offset),
                   patients=nightly.patients)
    call_list = engine.call_list(now=day0 + timedelta(days=45))
    assert call_list, "day 45 must hand off to a person"
    sent = db.all(
        "SELECT COUNT(*) c FROM recall_attempt WHERE outcome='sent' AND patient_id=?",
        (cases[0].patient_id,),
    )
    assert sent[0]["c"] <= RecallEngine.MAX_MESSAGES_PER_PATIENT_PER_90_DAYS


def test_a_gap_closed_by_vaccination_stops_the_cadence(db, gw):
    cases, nightly = nightly_for(["adolescent_gap_cluster"], db)
    engine = _authorized_engine(db, gw)
    forecasts = list(nightly.forecasts.values())
    day0 = local(2026, 8, 22, 10)
    engine.run(forecasts, now=day0, patients=nightly.patients)
    before = len(gw.messages_for("recall_immunization"))

    # The child gets vaccinated; the next night's forecast has no open gaps.
    case = cases[0]
    for gap in forecasts[0].open_gaps:
        gap.status = Status.COMPLETE
    closed = engine.close_resolved_gaps(forecasts, now=day0 + timedelta(days=1))
    assert closed >= 1
    engine.run(forecasts, now=day0 + timedelta(days=7), patients=nightly.patients)
    assert len(gw.messages_for("recall_immunization")) == before


def test_recall_consent_basis_is_documented_in_code():
    assert "not marketing" in RECALL_CONSENT_BASIS
    assert "reminders" in RECALL_CONSENT_BASIS


def test_recall_uses_the_drafter_when_one_is_supplied(db, gw):
    cases, nightly = nightly_for(["adolescent_gap_cluster"], db)
    good = (
        "Hi! Nia is due for Tdap. Book: https://nsp.example/book Reply STOP to opt out."
    )
    drafter = MessageDrafter(LLMClient(EchoTransport([_draft_response(good)] * 10)))
    engine = _authorized_engine(db, gw, drafter=drafter)
    engine.run(list(nightly.forecasts.values()), now=local(2026, 8, 22, 10),
               patients=nightly.patients)
    # Whatever the model returned, the message that went out still names the
    # child and still carries an opt-out.
    body = gw.messages_for("recall_immunization")[0]["body"]
    assert "STOP" in body


# ==========================================================================
# huddle.py
# ==========================================================================


def test_huddle_sheet_lists_tomorrows_patients_with_their_gaps(db):
    case = by_name("adolescent_gap_cluster")
    seed_family(db, case)
    add_provider(db, provider_id="dr_ruiz", display_name="Dr. Ruiz")
    tomorrow = TODAY + timedelta(days=1)
    add_appointment(db, patient_id=case.patient_id, provider_id="dr_ruiz",
                    visit_type=VisitType.WELL,
                    start=local(tomorrow.year, tomorrow.month, tomorrow.day, 9, 20),
                    now=local(2026, 8, 1))
    _, nightly = nightly_for(["adolescent_gap_cluster"])
    sheet = build_huddle(db, for_date=tomorrow, forecasts=nightly.forecasts,
                         reconciliations=nightly.reconciliations)
    assert len(sheet.cards) == 1
    card = sheet.cards[0]
    assert card.headline_count > 0
    text = sheet.render()
    assert "IMMUNIZATION HUDDLE" in text
    assert "Tdap" in text
    assert card.age_text.endswith("y") or " y " in card.age_text


def test_huddle_marks_ai_narrative_distinctly_from_computed_lines(db):
    case = by_name("adolescent_gap_cluster")
    seed_family(db, case)
    add_provider(db, provider_id="dr_ruiz", display_name="Dr. Ruiz")
    tomorrow = TODAY + timedelta(days=1)
    add_appointment(db, patient_id=case.patient_id, provider_id="dr_ruiz",
                    visit_type=VisitType.WELL,
                    start=local(tomorrow.year, tomorrow.month, tomorrow.day, 9, 20),
                    now=local(2026, 8, 1))
    _, nightly = nightly_for(["adolescent_gap_cluster"])
    sheet = build_huddle(
        db, for_date=tomorrow, forecasts=nightly.forecasts,
        narratives={case.patient_id: "Adolescent cluster; discuss HPV first."},
    )
    card = sheet.cards[0]
    assert card.narrative is not None
    assert card.narrative.provenance == Provenance.AI
    assert "AI" in card.render()
    # Every clinical line is computed, never AI.
    assert all(l.provenance == Provenance.COMPUTED for l in card.overdue)


def test_huddle_surfaces_reconciliation_discrepancies(db):
    case = by_name("combination_split_across_sources")
    seed_family(db, case)
    add_provider(db, provider_id="dr_ruiz", display_name="Dr. Ruiz")
    tomorrow = TODAY + timedelta(days=1)
    add_appointment(db, patient_id=case.patient_id, provider_id="dr_ruiz",
                    visit_type=VisitType.WELL,
                    start=local(tomorrow.year, tomorrow.month, tomorrow.day, 11, 0),
                    now=local(2026, 8, 1))
    _, nightly = nightly_for(["combination_split_across_sources"])
    sheet = build_huddle(db, for_date=tomorrow, forecasts=nightly.forecasts,
                         reconciliations=nightly.reconciliations)
    card = sheet.cards[0]
    assert card.discrepancies
    assert any("unresolved" in d.text for d in card.discrepancies)


def test_huddle_warns_about_a_closing_age_out_window(db):
    case = by_name("rotavirus_closing_window")
    seed_family(db, case)
    add_provider(db, provider_id="dr_ruiz", display_name="Dr. Ruiz")
    tomorrow = TODAY + timedelta(days=1)
    add_appointment(db, patient_id=case.patient_id, provider_id="dr_ruiz",
                    visit_type=VisitType.WELL,
                    start=local(tomorrow.year, tomorrow.month, tomorrow.day, 8, 40),
                    now=local(2026, 8, 1))
    _, nightly = nightly_for(["rotavirus_closing_window"])
    sheet = build_huddle(db, for_date=tomorrow, forecasts=nightly.forecasts)
    text = sheet.render()
    assert "can no longer be given" in text


def test_huddle_says_so_when_a_child_is_up_to_date(db):
    case = by_name("adolescent_complete")
    seed_family(db, case)
    add_provider(db, provider_id="dr_ruiz", display_name="Dr. Ruiz")
    tomorrow = TODAY + timedelta(days=1)
    add_appointment(db, patient_id=case.patient_id, provider_id="dr_ruiz",
                    visit_type=VisitType.WELL,
                    start=local(tomorrow.year, tomorrow.month, tomorrow.day, 14, 0),
                    now=local(2026, 8, 1))
    _, nightly = nightly_for(["adolescent_complete"])
    sheet = build_huddle(db, for_date=tomorrow, forecasts=nightly.forecasts)
    card = sheet.cards[0]
    assert card.overdue == []


# ==========================================================================
# pipeline.py
# ==========================================================================


def test_reconciliation_holds_downgrade_contested_antigens():
    _, nightly = nightly_for(["combination_split_across_sources"])
    forecast = list(nightly.forecasts.values())[0]
    for antigen in ("dtap", "hepb", "ipv"):
        entry = forecast.antigens[antigen]
        assert entry.status == Status.REQUIRES_REVIEW
        assert entry.is_open_gap is False


def test_unknown_codes_suppress_recall_eligibility():
    _, nightly = nightly_for(["unknown_cvx_code"])
    forecast = list(nightly.forecasts.values())[0]
    assert forecast.unknown_codes == ["999"]
    assert all(not g.recall_eligible for g in forecast.open_gaps)


def test_an_adjudicator_without_a_reviewer_is_a_configuration_error():
    case = by_name("two_week_drift_ambiguous")
    client = LLMClient(EchoTransport([_adjudication_response()]))
    with pytest.raises(ValueError, match="named human reviewer"):
        run_nightly(
            [PatientInput(case.patient_id, case.family_id, case.first_name, case.dob,
                          case.chart, case.registry)],
            as_of=TODAY,
            adjudicator=Adjudicator(client),
        )


def test_nightly_without_an_adjudicator_still_queues_the_ambiguity():
    _, nightly = nightly_for(["two_week_drift_ambiguous"])
    assert len(nightly.review_queue) == 1
    assert nightly.review_queue[0].reason == "no adjudicator configured"


def test_nightly_summary_reports_what_a_human_has_to_do():
    _, nightly = nightly_for([c.name for c in CASES])
    summary = nightly.summary()
    assert summary["patients"] == len(CASES)
    assert summary["antigens_held_for_review"] > 0
    assert summary["reconciliation_review_items"] > 0
    assert summary["patients_with_unknown_codes"] == 1


def test_overdue_term_saturates_so_the_bonuses_can_still_compete(db, gw):
    """A 16-year-old's ancient MMR gap must not permanently outrank an infant
    whose rotavirus window shuts in nine days.

    Taken literally, README I-02's `days_overdue x weight` is unbounded: 5,300
    days x weight 5 is ~26,000, and no bonus in the formula can reach that. The
    term saturates at a year for exactly this reason.
    """
    cases, nightly = nightly_for(
        ["hpv_three_dose_late_start", "rotavirus_closing_window"], db
    )
    engine = _authorized_engine(db, gw)
    queue = engine.build_queue(list(nightly.forecasts.values()), as_of=TODAY,
                               patients=nightly.patients)
    ancient = next(c for c in queue
                   if c.patient_id.endswith("hpv_three_dose_late_start")
                   and c.antigen == "mmr")
    closing = next(c for c in queue if c.antigen == "rota")
    assert ancient.days_overdue > 3000
    assert ancient.breakdown["overdue_x_weight_saturated"] == pytest.approx(
        RecallEngine.OVERDUE_SATURATION_DAYS * 5.0
    )
    assert closing.urgency > 0
    assert closing.breakdown["age_out_bonus"] > ancient.breakdown["age_out_bonus"]


def test_a_quiet_hours_run_defers_without_burning_a_cadence_step(db, gw):
    """The nightly batch runs at 06:30. That must not consume the day-0 message."""
    cases, nightly = nightly_for(["adolescent_gap_cluster"], db)
    engine = _authorized_engine(db, gw)
    forecasts = list(nightly.forecasts.values())
    before_open = local(2026, 8, 22, 6, 30)
    report = engine.run(forecasts, now=before_open, patients=nightly.patients)
    assert report["outcomes"] == {"deferred": 1}
    assert gw.sent == []
    assert db.all("SELECT * FROM recall_attempt") == []

    after_open = local(2026, 8, 22, 9, 0)
    report = engine.run(forecasts, now=after_open, patients=nightly.patients)
    assert report["outcomes"].get("sent") == 1


def test_one_message_covers_every_open_gap_for_a_patient(db, gw):
    """A family behind on three vaccines gets one text, not three."""
    cases, nightly = nightly_for(["adolescent_gap_cluster"], db)
    engine = _authorized_engine(db, gw)
    engine.run(list(nightly.forecasts.values()), now=local(2026, 8, 22, 9),
               patients=nightly.patients)
    messages = gw.messages_for("recall_immunization")
    assert len(messages) == 1
    body = messages[0]["body"]
    assert "Tdap" in body and "MenACWY" in body
    row = db.one("SELECT antigens FROM recall_attempt WHERE outcome='sent'")
    assert len(json.loads(row["antigens"])) >= 3


def test_covid_never_occupies_a_line_on_the_huddle_sheet(db):
    case = by_name("adolescent_gap_cluster")
    seed_family(db, case)
    add_provider(db, provider_id="dr_ruiz", display_name="Dr. Ruiz")
    tomorrow = TODAY + timedelta(days=1)
    add_appointment(db, patient_id=case.patient_id, provider_id="dr_ruiz",
                    visit_type=VisitType.WELL,
                    start=local(tomorrow.year, tomorrow.month, tomorrow.day, 9, 20),
                    now=local(2026, 8, 1))
    _, nightly = nightly_for(["adolescent_gap_cluster"])
    sheet = build_huddle(db, for_date=tomorrow, forecasts=nightly.forecasts)
    assert "COVID" not in sheet.render()
    assert "Influenza" in sheet.render()


def test_huddle_shows_one_line_per_discrepancy_not_one_per_pair(db):
    case = by_name("combination_split_across_sources")
    seed_family(db, case)
    add_provider(db, provider_id="dr_ruiz", display_name="Dr. Ruiz")
    tomorrow = TODAY + timedelta(days=1)
    add_appointment(db, patient_id=case.patient_id, provider_id="dr_ruiz",
                    visit_type=VisitType.WELL,
                    start=local(tomorrow.year, tomorrow.month, tomorrow.day, 10, 0),
                    now=local(2026, 8, 1))
    _, nightly = nightly_for(["combination_split_across_sources"])
    sheet = build_huddle(db, for_date=tomorrow, forecasts=nightly.forecasts,
                         reconciliations=nightly.reconciliations)
    unresolved = [d for d in sheet.cards[0].discrepancies if "unresolved" in d.text]
    assert len(unresolved) == 1, "three ambiguous pairs are one thing to look at"


def test_a_dry_run_mutates_nothing(db, gw):
    """Inspecting the queue must not consume the day-0 cadence slot."""
    cases, nightly = nightly_for(["adolescent_gap_cluster"], db)
    engine = _authorized_engine(db, gw)
    forecasts = list(nightly.forecasts.values())
    engine.run(forecasts, now=local(2026, 8, 22, 9), patients=nightly.patients,
               dry_run=True)
    assert db.all("SELECT * FROM recall_gap") == []
    assert db.all("SELECT * FROM recall_attempt") == []
    assert gw.sent == []

    report = engine.run(forecasts, now=local(2026, 8, 22, 9, 30),
                        patients=nightly.patients)
    assert report["outcomes"].get("sent") == 1


# ==========================================================================
# Regression tests — each of these reproduced a real defect before its fix
# ==========================================================================


def test_an_unspecified_product_selects_the_longer_series(forecaster):
    """CVX 122 is "rotavirus, unspecified". Unknown must not mean Rotarix.

    Filtering out doses with no known variant and then trusting a single
    remaining variant declared a 2-dose rotavirus series complete on the
    strength of one identified dose. Rotavirus ages out hard at eight months,
    so the third dose was then unrecoverable.
    """
    dob = TODAY.replace(year=TODAY.year - 1)
    doses = [
        AdministeredDose("122", add_period(dob, {"months": 2})),   # unspecified
        AdministeredDose("119", add_period(dob, {"months": 4})),   # Rotarix
    ]
    forecast = forecaster.forecast(patient_id="p", dob=dob, doses=doses,
                                   as_of=add_period(dob, {"months": 6}))
    rota = forecast.antigens["rota"]
    assert rota.doses_required == 3
    assert rota.status != Status.COMPLETE
    assert any("unspecified formulation" in n for n in rota.notes)


def test_an_invalid_first_dose_does_not_choose_the_hpv_series(forecaster):
    """A stray HPV dose at age 8 must not make a 15-year-old's series 2 doses."""
    dob = date(2008, 1, 1)
    doses = [
        AdministeredDose("165", date(2016, 1, 1)),    # age 8 - invalid, too early
        AdministeredDose("165", date(2023, 7, 1)),    # 15 y 6 mo - the real first
        AdministeredDose("165", date(2023, 12, 15)),
    ]
    forecast = forecaster.forecast(patient_id="p", dob=dob, doses=doses,
                                   as_of=date(2024, 6, 1))
    hpv = forecast.antigens["hpv"]
    assert hpv.doses_required == 3
    assert hpv.status != Status.COMPLETE
    assert hpv.doses_valid == 2


def test_hib_variant_changes_the_timings_not_just_the_count(forecaster):
    """PedvaxHIB is 2 primary doses plus a 12-15 month booster.

    Keeping the PRP-T dose-3 timing (14 weeks) against a 3-dose count produced
    a false OVERDUE at six months AND marked the series complete without the
    booster -- wrong in both directions, on a school-required antigen.
    """
    dob = date(2025, 1, 1)
    on_schedule = [
        AdministeredDose("49", date(2025, 3, 1)),
        AdministeredDose("49", date(2025, 5, 1)),
    ]
    at_nine_months = forecaster.forecast(
        patient_id="p", dob=dob, doses=on_schedule, as_of=date(2025, 10, 1)
    ).antigens["hib"]
    assert at_nine_months.status == Status.NOT_YET_DUE
    assert at_nine_months.days_overdue == 0

    three_early = on_schedule + [AdministeredDose("49", date(2025, 7, 1))]
    at_two_years = forecaster.forecast(
        patient_id="p", dob=dob, doses=three_early, as_of=date(2027, 1, 1)
    ).antigens["hib"]
    # The 6-month dose cannot satisfy the 12-month booster.
    assert at_two_years.status != Status.COMPLETE


def test_an_unstarted_hpv_series_is_sized_for_the_current_age(forecaster):
    dob = years_before(16)
    forecast = forecaster.forecast(patient_id="p", dob=dob, doses=[], as_of=TODAY)
    assert forecast.antigens["hpv"].doses_required == 3

    young = forecaster.forecast(patient_id="p", dob=years_before(11), doses=[],
                                as_of=TODAY)
    assert young.antigens["hpv"].doses_required == 2


def test_partial_date_reports_no_confirmed_doses(forecaster):
    """"2 of 0" was the old output for a conditional series with a partial date."""
    dob = years_before(14)
    doses = [
        AdministeredDose("165", date(2024, 1, 1), precision=DosePrecision.YEAR),
        AdministeredDose("165", date(2024, 8, 1)),
    ]
    hpv = forecaster.forecast(
        patient_id="p", dob=dob, doses=doses, as_of=TODAY
    ).antigens["hpv"]
    assert hpv.status == Status.REQUIRES_REVIEW
    assert hpv.doses_valid == 0
    assert hpv.doses_required >= 2


def test_a_completed_series_never_reports_more_doses_than_it_requires(forecaster):
    dob = date(2018, 1, 1)
    doses = [
        AdministeredDose("20", date(2018, 3, 1)),
        AdministeredDose("20", date(2018, 5, 1)),
        AdministeredDose("20", date(2018, 7, 1)),
        AdministeredDose("20", date(2022, 3, 1)),   # dose 4 at 4y -> completes
        AdministeredDose("20", date(2022, 10, 1)),  # a 5th anyway
    ]
    dtap = forecaster.forecast(
        patient_id="p", dob=dob, doses=doses, as_of=date(2024, 1, 1)
    ).antigens["dtap"]
    assert dtap.status == Status.COMPLETE
    assert dtap.doses_required >= dtap.doses_valid


def test_varicella_interval_resolves_against_the_projected_dose_date(forecaster):
    """A child days short of 13 must not be told to wait three months for a dose
    they will receive after their birthday, when four weeks suffices."""
    dob = date(2013, 9, 1)
    doses = [AdministeredDose("21", date(2014, 9, 5))]
    var = forecaster.forecast(
        patient_id="p", dob=dob, doses=doses, as_of=date(2026, 8, 25)
    ).antigens["var"]
    assert var.earliest_date is not None
    assert var.earliest_date <= date(2026, 10, 1)


def test_no_match_on_a_combination_cluster_does_not_clone_the_chart_dose():
    """One Pediarix must not become three DTaP administrations on one day."""
    case = by_name("combination_split_across_sources")
    reconciliation = reconcile(case.chart, case.registry)
    assert len(reconciliation.ambiguous) == 3
    client = LLMClient(
        EchoTransport([_adjudication_response(determination="NO_MATCH")] * 3)
    )
    outcomes = Adjudicator(client).adjudicate_reconciliation(
        reconciliation, patient_id="p1"
    )
    updated, _ = apply_adjudications(reconciliation, outcomes, reviewed_by="ma01")
    assert len(updated.chart_only) == 1
    assert len(updated.registry_only) == 3
    merged = updated.merged_doses()
    assert len(merged) == 4
    assert sum(1 for d in merged if d.cvx == "110") == 1


def test_a_mixed_verdict_cluster_never_puts_one_record_in_two_buckets():
    case = by_name("combination_split_across_sources")
    reconciliation = reconcile(case.chart, case.registry)
    client = LLMClient(EchoTransport([
        _adjudication_response(determination="MATCH"),
        _adjudication_response(determination="NO_MATCH"),
        _adjudication_response(determination="NO_MATCH"),
    ]))
    outcomes = Adjudicator(client).adjudicate_reconciliation(
        reconciliation, patient_id="p1"
    )
    updated, _ = apply_adjudications(reconciliation, outcomes, reviewed_by="ma01")
    chart_ids = [p.chart.record_id for p in updated.matched] + [
        r.record_id for r in updated.chart_only
    ]
    assert len(chart_ids) == len(set(chart_ids)), chart_ids


def test_a_registry_side_combination_clusters_too():
    """The asymmetry the pass-2 comment claims to have solved, in the other
    direction: Pediarix in I-CARE, three component rows in the chart."""
    day = date(2025, 3, 1)
    chart = [
        DoseRecord("c1", "20", day, "chart"),
        DoseRecord("c2", "08", day, "chart"),
        DoseRecord("c3", "10", day, "chart"),
    ]
    registry = [DoseRecord("r1", "110", day, "registry")]
    result = reconcile(chart, registry)
    assert result.chart_only == []
    assert result.registry_only == []
    assert len(result.ambiguous) == 3
    assert len(result.merged_doses()) == 1


def test_a_completed_cadence_does_not_silence_the_patient_forever(db, gw):
    """Lifetime-scoped attempt counting meant one finished cadence ended all
    future outreach for that child, silently."""
    cases, nightly = nightly_for(["adolescent_gap_cluster"], db)
    engine = _authorized_engine(db, gw, frequency_cap=FrequencyCap(
        db, limits={"appointment": 99, "waitlist": 99, "outreach": 99}
    ))
    db.execute("UPDATE family SET primary_email='p@example.com' WHERE family_id=?",
               (cases[0].family_id,))
    grant_consent(db, family_id=cases[0].family_id, channel=Channel.EMAIL,
                  purpose=ConsentPurpose.REMINDERS, granted=local(2026, 1, 1, 9),
                  capture_method="intake_form")
    forecasts = list(nightly.forecasts.values())
    day0 = local(2026, 8, 22, 10)
    for offset in (0, 7, 21, 45):
        engine.run(forecasts, now=day0 + timedelta(days=offset),
                   patients=nightly.patients)
    assert engine.call_list(now=day0 + timedelta(days=45))

    # The gaps resolve, then a new one opens eighteen months later.
    for gap in forecasts[0].open_gaps:
        gap.status = Status.COMPLETE
    engine.close_resolved_gaps(forecasts, now=day0 + timedelta(days=60))
    for gap in forecasts[0].antigens.values():
        gap.status = Status.NOT_YET_DUE
    forecasts[0].antigens["tdap"].status = Status.OVERDUE
    forecasts[0].antigens["tdap"].days_overdue = 30

    later = day0 + timedelta(days=550)
    report = engine.run(forecasts, now=later, patients=nightly.patients)
    assert report["outcomes"].get("sent") == 1, report["outcomes"]


def test_a_stale_anchor_does_not_collapse_the_cadence_into_consecutive_days(db, gw):
    cases, nightly = nightly_for(["adolescent_gap_cluster"], db)
    engine = _authorized_engine(db, gw, frequency_cap=FrequencyCap(
        db, limits={"appointment": 99, "waitlist": 99, "outreach": 99}
    ))
    forecasts = list(nightly.forecasts.values())
    day0 = local(2026, 8, 22, 10)
    engine.run(forecasts, now=day0, patients=nightly.patients)
    assert len(gw.messages_for("recall_immunization")) == 1

    # The family drops out of the queue for a month, then reappears at day 40.
    sent_days = []
    for offset in range(40, 47):
        engine.run(forecasts, now=day0 + timedelta(days=offset),
                   patients=nightly.patients)
        sent_days.append(len(gw.messages_for("recall_immunization")))
    # At most one further message across that whole week.
    assert sent_days[-1] <= 2, sent_days


def test_a_failing_gateway_does_not_retry_forever(db, gw):
    from modules.scheduling.gateway import Gateway, GatewayReceipt

    class DeadGateway(Gateway):
        name = "dead"

        def send(self, *, to, body, channel, purpose, reference):
            return GatewayReceipt(False, error="carrier unreachable")

    cases, nightly = nightly_for(["adolescent_gap_cluster"], db)
    engine = _authorized_engine(db, DeadGateway(), frequency_cap=FrequencyCap(
        db, limits={"appointment": 99, "waitlist": 99, "outreach": 99}
    ))
    forecasts = list(nightly.forecasts.values())
    day0 = local(2026, 8, 22, 10)
    for offset in range(0, 60, 3):
        engine.run(forecasts, now=day0 + timedelta(days=offset),
                   patients=nightly.patients)
    failures = db.all(
        "SELECT COUNT(*) c FROM recall_attempt WHERE outcome='failed' AND step=0"
    )[0]["c"]
    assert failures <= RecallEngine.MAX_RETRIES_PER_STEP, failures


def test_conversion_report_counts_a_partial_series_advance(db, gw):
    cases, nightly = nightly_for(["adolescent_gap_cluster"], db)
    engine = _authorized_engine(db, gw)
    forecasts = list(nightly.forecasts.values())
    day0 = local(2026, 8, 22, 10)
    engine.run(forecasts, now=day0, patients=nightly.patients)
    # The child comes in and gets dose 1 of 2: the gap closes as not_yet_due.
    for gap in forecasts[0].open_gaps:
        gap.status = Status.NOT_YET_DUE
    engine.close_resolved_gaps(forecasts, now=day0 + timedelta(days=3))
    report = engine.conversion_report()
    assert report["gaps_contacted"] > 0
    assert report["gaps_closed_by_vaccination"] == report["gaps_contacted"]
    assert report["conversion"] == 1.0


def test_huddle_never_asserts_up_to_date_without_a_forecast(db):
    case = by_name("adolescent_gap_cluster")
    seed_family(db, case)
    add_provider(db, provider_id="dr_ruiz", display_name="Dr. Ruiz")
    tomorrow = TODAY + timedelta(days=1)
    add_appointment(db, patient_id=case.patient_id, provider_id="dr_ruiz",
                    visit_type=VisitType.WELL,
                    start=local(tomorrow.year, tomorrow.month, tomorrow.day, 15, 0),
                    now=local(2026, 8, 1))
    sheet = build_huddle(db, for_date=tomorrow, forecasts={})   # batch missed them
    text = sheet.render()
    assert "up to date" not in text
    assert "has NOT been checked" in text


def test_draft_rejects_a_recombined_phone_number():
    template = (
        "Call us on 847-555-0100 to book {first_name}'s {vaccine_list} visit: "
        "{booking_url}. Reply STOP to opt out."
    )
    slots = {"first_name": "Nia", "vaccine_list": "Tdap",
             "booking_url": "https://nsp.example/book"}
    bad = (
        "Call us on 555-847-0100 to book Nia's Tdap visit: "
        "https://nsp.example/book. Reply STOP to opt out."
    )
    drafter = MessageDrafter(LLMClient(EchoTransport([_draft_response(bad)])))
    result = drafter.draft_detailed(template_id="t", template=template, slots=slots)
    assert result.used_model is False
    assert "numeric sequence" in result.fallback_reason


def test_draft_rejects_a_spelled_out_dose_count():
    bad = "Hi! Nia is due for Tdap - that's two shots. https://nsp.example/book Reply STOP to opt out."
    drafter = MessageDrafter(LLMClient(EchoTransport([_draft_response(bad)])))
    result = drafter.draft_detailed(template_id="t", template=TEMPLATE, slots=SLOTS)
    assert result.used_model is False
    assert "spelled-out number" in result.fallback_reason


def test_draft_rejects_an_introduced_link():
    bad = ("Hi! Nia is due for Tdap. Pay at https://nsp-billing.example/pay then "
           "book https://nsp.example/book. Reply STOP to opt out.")
    drafter = MessageDrafter(LLMClient(EchoTransport([_draft_response(bad)])))
    result = drafter.draft_detailed(template_id="t", template=TEMPLATE, slots=SLOTS)
    assert result.used_model is False
    assert "introduced a link" in result.fallback_reason


def test_draft_rejects_fear_and_pressure_language():
    for bad in (
        "URGENT: Nia is unprotected against Tdap. https://nsp.example/book Reply STOP to opt out.",
        "Nia is due for Tdap. Without it she could be hospitalized. "
        "https://nsp.example/book Reply STOP to opt out.",
        "Nia must get Tdap. https://nsp.example/book Reply STOP to opt out.",
    ):
        drafter = MessageDrafter(LLMClient(EchoTransport([_draft_response(bad)])))
        result = drafter.draft_detailed(template_id="t", template=TEMPLATE, slots=SLOTS)
        assert result.used_model is False, bad
        assert "pressure" in result.fallback_reason


def test_holds_are_idempotent():
    from modules.immunization import apply_reconciliation_holds

    case = by_name("combination_split_across_sources")
    reconciliation = reconcile(case.chart, case.registry)
    forecast = LocalRulesForecaster().forecast(
        patient_id=case.patient_id, dob=case.dob,
        doses=reconciliation.merged_doses(), as_of=TODAY,
    )
    for _ in range(3):
        apply_reconciliation_holds(forecast, reconciliation)
    notes = forecast.antigens["dtap"].notes
    assert len([n for n in notes if "disagree" in n]) == 1
