"""I-03 tests. Real CDC tables, real rules table, real EchoTransport.

`EchoTransport` is a shipped component, not a mock of this module's logic: these
tests drive the production narrative path -- schema validation, the date-citation
check, the judgement-language check and grounding -- against scripted model
output.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest

from modules.scheduling.models import Database
from modules.previsit import (
    AI_MARKER,
    MAX_ENCOUNTERS,
    MAX_LINES,
    NARRATIVE_SCHEMA,
    NARRATIVE_SYSTEM_PROMPT,
    ChannelCrossing,
    ClinicalJudgementLeak,
    CompletedScreening,
    Encounter,
    FeedbackLog,
    GrowthReference,
    Indicator,
    Measurement,
    NarrativeSynthesizer,
    NotComparable,
    OpenThread,
    OutOfRange,
    PatientDay,
    PeriodicitySchedule,
    ScheduleNotReviewed,
    Status,
    Verdict,
    assemble,
    assert_no_judgement_fields,
    bmi,
    channel_crossing,
    lms_z,
    percentile_to_z,
    render_text,
    run_batch,
    z_to_percentile,
)
from modules.previsit.fixtures import CASES, CLINIC_DATE, GENERATED_UTC, by_name
from nsp_core.audit import AuditLog
from nsp_core.llm import EchoTransport, LLMClient

NOW = GENERATED_UTC


@pytest.fixture(scope="module")
def reference():
    return GrowthReference()


@pytest.fixture(scope="module")
def schedule():
    return PeriodicitySchedule.load()


@pytest.fixture
def reviewed_schedule():
    loaded = PeriodicitySchedule.load()
    loaded.review["owner"] = "test-owner"
    return loaded


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "practice.sqlite3")


@pytest.fixture
def audit(tmp_path):
    return AuditLog(tmp_path / "audit.sqlite3", hmac_key=b"test-key")


def synth(case, **kwargs):
    return NarrativeSynthesizer(
        LLMClient(EchoTransport([case.model_response])), **kwargs
    )


def day_for(case):
    return PatientDay(
        patient_id=case.patient_id, patient_label=case.patient_label, sex=case.sex,
        age_months=case.age_months, age_label=case.age_label,
        visit_type=case.visit_type, appointment_local=case.appointment_local,
        provider=case.provider, measurements=case.measurements,
        prior_measurements=case.prior_measurements,
        completed_screenings=case.completed_screenings, risk_flags=case.risk_flags,
        immunizations_due=case.immunizations_due, open_threads=case.open_threads,
        encounters=case.encounters, problem_list=case.problem_list,
        data_horizon_months=case.data_horizon_months,
    )


# ==========================================================================
# growth.py — the arithmetic, validated against CDC's own published values
# ==========================================================================


def test_every_published_cdc_percentile_reproduces_within_half_a_point(reference):
    """The build plan asks for 15 known CDC reference points within 0.5.

    CDC publishes P3 through P97 alongside the L, M and S in every one of these
    files, which makes every printed value a test case: computing the percentile
    of the published P75 weight must return 75. That is 16,000-odd points rather
    than 15, and it is the same check -- the tables validate the code they ship
    with.
    """
    checked = 0
    worst = 0.0
    worst_at = None
    for filename, meta in reference.manifest["files"].items():
        for sex in ("M", "F"):
            for row in reference.rows_for(filename, sex):
                l, m, s = reference.lms_at(filename, sex, row.x)
                for column, value in row.published.items():
                    expected = float(column[1:])
                    got = z_to_percentile(lms_z(value, l=l, m=m, s=s))
                    error = abs(got - expected)
                    checked += 1
                    if error > worst:
                        worst, worst_at = error, (filename, sex, row.x, column)
    assert checked > 15_000, "the whole published grid should have been checked"
    assert worst < 0.5, f"worst error {worst} at {worst_at}"


def test_the_lms_transform_handles_the_l_equals_zero_limit():
    """L == 0 is a limit of the same formula, not a special case to skip."""
    assert lms_z(10.0, l=0.0, m=10.0, s=0.2) == pytest.approx(0.0)
    assert lms_z(12.214, l=0.0, m=10.0, s=0.2) == pytest.approx(1.0, abs=1e-3)
    # And it agrees with the L != 0 branch as L approaches zero.
    near = lms_z(11.0, l=1e-7, m=10.0, s=0.2)
    exact = lms_z(11.0, l=0.0, m=10.0, s=0.2)
    assert near == pytest.approx(exact, abs=1e-4)


def test_percentile_and_z_round_trip():
    for percentile in (0.1, 3, 25, 50, 75, 97, 99.9):
        assert z_to_percentile(percentile_to_z(percentile)) == pytest.approx(
            percentile, abs=1e-6
        )


def test_lms_is_interpolated_between_rows_not_rounded(reference):
    """Rounding a 30.4-month-old to 30 months moves the percentile enough to
    invent or hide a channel crossing."""
    at_30 = reference.lms_at("wtage.csv", "M", 30.0)
    at_31 = reference.lms_at("wtage.csv", "M", 30.5)
    at_between = reference.lms_at("wtage.csv", "M", 30.25)
    for index in range(3):
        low, high = sorted((at_30[index], at_31[index]))
        assert low <= at_between[index] <= high
        assert at_between[index] not in (at_30[index], at_31[index])


def test_the_reference_refuses_to_extrapolate(reference):
    with pytest.raises(OutOfRange, match="does not extrapolate"):
        reference.place(Measurement(Indicator.WEIGHT_FOR_AGE, 70.0, 260.0, "M"))
    with pytest.raises(OutOfRange):
        reference.place(Measurement(Indicator.HEAD_CIRCUMFERENCE_FOR_AGE, 50.0, 48.0, "F"))


def test_an_unknown_sex_is_refused_rather_than_averaged(reference):
    with pytest.raises(ValueError, match="no combined table"):
        reference.place(Measurement(Indicator.WEIGHT_FOR_AGE, 12.0, 24.0, "unknown"))


def test_bmi_is_computed_not_looked_up():
    assert bmi(18.0, 100.0) == pytest.approx(18.0)
    with pytest.raises(ValueError):
        bmi(0, 100)


def test_the_infant_table_is_used_below_two_and_the_child_table_above(reference):
    assert reference.table_for(Indicator.WEIGHT_FOR_AGE, x=23.9) == "wtageinf.csv"
    assert reference.table_for(Indicator.WEIGHT_FOR_AGE, x=24.0) == "wtage.csv"


# -- crossing detection -----------------------------------------------------


def test_a_two_channel_crossing_is_significant_and_a_one_channel_move_is_not(reference):
    case = by_name("bmi_crossing_school_age")
    earlier = reference.place(case.prior_measurements[0])
    later = reference.place(case.measurements[0])
    crossing = channel_crossing(earlier, later)
    assert crossing is not None
    assert crossing.direction == "up"
    assert crossing.significant is True
    assert crossing.channels_crossed >= 2
    assert "93rd" in crossing.describe()


def test_no_movement_across_a_line_is_not_a_crossing(reference):
    a = reference.place(Measurement(Indicator.WEIGHT_FOR_AGE, 12.6707633, 24.0, "M"))
    b = reference.place(Measurement(Indicator.WEIGHT_FOR_AGE, 12.9, 25.0, "M"))
    assert 50 < b.percentile < 75
    assert channel_crossing(a, b) is None


def test_a_length_to_stature_switch_is_refused_not_reported(reference):
    """Every child changes measurement method around the second birthday, and
    recumbent length runs about 0.7 cm above standing stature."""
    case = by_name("length_to_stature_switch")
    earlier = reference.place(case.prior_measurements[0])
    later = reference.place(case.measurements[0])
    with pytest.raises(NotComparable, match="recumbent length, one standing stature"):
        channel_crossing(earlier, later)


def test_an_unrecorded_measurement_method_blocks_the_comparison(reference):
    earlier = reference.place(
        Measurement(Indicator.STATURE_FOR_AGE, 86.0, 24.0, "M", standing=None)
    )
    later = reference.place(
        Measurement(Indicator.STATURE_FOR_AGE, 92.0, 30.0, "M", standing=True)
    )
    with pytest.raises(NotComparable, match="lying down or standing up"):
        channel_crossing(earlier, later)


def test_different_indicators_are_never_compared(reference):
    weight = reference.place(Measurement(Indicator.WEIGHT_FOR_AGE, 12.0, 24.0, "M"))
    body_mass = reference.place(Measurement(Indicator.BMI_FOR_AGE, 16.0, 24.0, "M"))
    with pytest.raises(NotComparable, match="different indicators"):
        channel_crossing(weight, body_mass)


# ==========================================================================
# periodicity.py — the rules table
# ==========================================================================


def _executable_constants(module):
    """Every literal the module actually EVALUATES, docstrings excluded.

    Prose in a docstring naming M-CHAT is documentation. A string literal in a
    branch naming M-CHAT is a rule that escaped the YAML, and the difference is
    the whole point -- so this walks the AST rather than grepping the text.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(module))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and id(node) not in docstrings
    ]


def test_no_screening_knowledge_lives_in_python(schedule):
    """Every age, cadence, code and screening name is in the YAML. Same rule as
    I-02's forecaster, for the same reason: a Bright Futures revision has to be
    a data diff a clinician can review."""
    from modules.previsit import periodicity

    literals = _executable_constants(periodicity)
    strings = {str(v).lower() for v in literals if isinstance(v, str)}
    numbers = {float(v) for v in literals if isinstance(v, (int, float)) and not isinstance(v, bool)}

    for definition in schedule.screenings.values():
        assert definition.id.lower() not in strings, f"{definition.id} is data, not code"
        for code in definition.cpt:
            assert code not in strings, f"CPT {code} is data, not code"
    # No screening age ever appears as a literal in the module either.
    for age in (9.0, 18.0, 30.0, 144.0, 96110.0):
        assert age not in numbers, f"{age} looks like schedule data in code"


def test_the_eighteen_month_screenings_are_due(schedule):
    case = by_name("toddler_18_month_well")
    statuses = {
        s.definition.id: s
        for s in schedule.evaluate(
            age_months=case.age_months, completed=case.completed_screenings
        )
    }
    assert statuses["developmental_screen"].status == Status.DUE
    assert statuses["autism_screen"].status == Status.DUE
    # Done at twelve months, so not asked for again.
    assert statuses["anemia_screen"].status == Status.COMPLETE
    # Lead was drawn at twelve months; the twenty-four month one is not due yet.
    assert statuses["lead_blood_universal"].status == Status.COMPLETE
    for status in statuses.values():
        assert status.because, f"{status.definition.id} gave no reason"


def test_a_screening_inside_the_grace_window_counts_as_done(schedule):
    """Without a window every brief shows a permanent backlog of screenings
    that were, in fact, completed."""
    late = [CompletedScreening("developmental_screen", date(2026, 1, 5), 10.0)]
    statuses = {s.definition.id: s for s in schedule.evaluate(age_months=11.0, completed=late)}
    assert statuses["developmental_screen"].status == Status.COMPLETE


def test_a_missed_screening_stops_nagging_past_its_catch_up_horizon(schedule):
    """A two-year TB risk assessment must not sit OVERDUE on a thirteen-year-old
    for the rest of childhood -- README I-03's 'brief becomes noise' risk."""
    at_three = {s.definition.id: s for s in schedule.evaluate(age_months=34.0)}
    assert at_three["autism_screen"].status == Status.OVERDUE
    at_five = {s.definition.id: s for s in schedule.evaluate(age_months=60.0)}
    assert at_five["autism_screen"].status == Status.MISSED
    assert "catch-up window closed" in at_five["autism_screen"].because


def test_a_banded_screening_ages_out_at_the_end_of_its_band(schedule):
    at_thirteen = {s.definition.id: s for s in schedule.evaluate(age_months=157.0)}
    assert at_thirteen["oral_health_varnish"].status == Status.NOT_DUE
    assert "age band ended" in at_thirteen["oral_health_varnish"].because


def test_a_risk_based_test_stays_off_until_the_assessment_says_otherwise(schedule):
    without = {s.definition.id: s for s in schedule.evaluate(age_months=157.0)}
    assert without["sti_screen"].status != Status.DUE
    with_flag = {
        s.definition.id: s
        for s in schedule.evaluate(age_months=157.0, risk_flags=["sexually_active"])
    }
    # The band opened at eleven and this has never been done, so it reads as
    # overdue from the first unmet target rather than merely due today.
    assert with_flag["sti_screen"].status == Status.OVERDUE
    assert with_flag["sti_screen"].actionable


def test_a_risk_line_is_not_printed_twice(schedule):
    """When the gating assessment is itself due, both lines ask for the same
    action and printing both puts permanent question marks on every brief."""
    statuses = {s.definition.id: s for s in schedule.evaluate(age_months=157.0)}
    assert statuses["anemia_risk_assessment"].actionable
    assert statuses["anemia_adolescent"].status == Status.NOT_DUE
    assert "already on this brief" in statuses["anemia_adolescent"].because


def _mini_schedule():
    """An assessment that ages out at 24 months, gating a test that does not."""
    return PeriodicitySchedule(
        {
            "version": "t", "well_visit_months": [],
            "review": {"owner": "test"},
            "screenings": [
                {"id": "assess", "name": "Risk assessment", "due_at_months": [12],
                 "catch_up_until_months": 24, "risk_assessment_for": "thing"},
                {"id": "test", "name": "The test", "from_months": 12,
                 "to_months": 252, "every_months": 12, "requires_risk_flag": "thing"},
            ],
        }
    )


def test_an_unknown_risk_is_reported_when_the_assessment_is_not_actionable():
    """An assessment that aged out and was never done leaves a genuine unknown,
    and that is worth exactly one line."""
    statuses = {s.definition.id: s for s in _mini_schedule().evaluate(age_months=60.0)}
    assert statuses["assess"].status == Status.MISSED
    assert statuses["test"].status == Status.RISK_UNKNOWN
    assert "has not been recorded" in statuses["test"].because


def test_an_assessment_that_came_back_negative_answers_the_question():
    """RISK_UNKNOWN means nobody asked. Somebody asked and the answer was no is
    a different state, and reporting it as unknown puts a permanent question
    mark next to a question that has already been answered."""
    done = [CompletedScreening("assess", date(2025, 1, 10), 12.0, "no risk factors")]
    statuses = {
        s.definition.id: s
        for s in _mini_schedule().evaluate(age_months=13.0, completed=done)
    }
    assert statuses["assess"].status == Status.COMPLETE
    assert statuses["test"].status == Status.NOT_DUE
    assert "did not indicate this test" in statuses["test"].because


def test_a_declined_screening_is_not_the_same_as_not_due(schedule):
    declined = [CompletedScreening("autism_screen", date(2026, 8, 1), 18.0, declined=True)]
    statuses = {
        s.definition.id: s for s in schedule.evaluate(age_months=18.4, completed=declined)
    }
    assert statuses["autism_screen"].status == Status.DECLINED
    assert "offer again" in statuses["autism_screen"].because


def test_a_risk_flag_no_assessment_can_set_is_refused_at_load():
    """A screening gated on an unreachable flag is absent from every brief in
    the practice, which is indistinguishable from nobody needing it."""
    data = {
        "version": "t", "well_visit_months": [],
        "screenings": [
            {"id": "x", "name": "X", "from_months": 12, "to_months": 24,
             "every_months": 12, "requires_risk_flag": "nothing_sets_this"},
        ],
    }
    with pytest.raises(ValueError, match="never become due"):
        PeriodicitySchedule(data)


def test_a_discrete_screening_without_a_catch_up_horizon_is_refused_at_load():
    data = {
        "version": "t", "well_visit_months": [],
        "screenings": [{"id": "x", "name": "X", "due_at_months": [12]}],
    }
    with pytest.raises(ValueError, match="overdue forever"):
        PeriodicitySchedule(data)


def test_the_shipped_table_has_no_clinical_owner_and_the_batch_refuses(schedule):
    """README I-03's control -- an owner and an annual review -- in the code path
    rather than on a checklist."""
    assert schedule.has_clinical_owner is False
    with pytest.raises(ScheduleNotReviewed, match="annual review"):
        run_batch([], clinic_date=CLINIC_DATE, schedule=schedule,
                  reference=GrowthReference(), synthesizer=None, generated_utc=NOW)


def test_illinois_school_forms_surface_near_kindergarten(schedule):
    ids = {f["id"] for f in schedule.forms_due(age_months=60.0)}
    assert {"il_school_physical", "il_dental_exam", "il_vision_exam"} <= ids
    assert schedule.forms_due(age_months=30.0) == []


# ==========================================================================
# narrative.py — THE POST-CONDITIONS
# ==========================================================================


def test_the_prompt_is_appendix_a4_verbatim():
    for line in (
        "You MUST NOT generate clinical recommendations, differential diagnoses, or",
        "Every item you return MUST cite the encounter date it came from.",
        "Screenings due, immunizations due, and growth percentile changes are computed",
    ):
        assert line in NARRATIVE_SYSTEM_PROMPT


def test_the_schema_has_no_field_for_clinical_judgement():
    assert NARRATIVE_SCHEMA["additionalProperties"] is False
    assert set(NARRATIVE_SCHEMA["properties"]) == {
        "recent_relevant_history", "open_threads", "unresolved_parent_concerns",
        "medication_changes",
    }
    for name in NARRATIVE_SCHEMA["properties"]:
        assert set(NARRATIVE_SCHEMA["properties"][name]["items"]["properties"]) == {
            "item", "source_date"
        }


def test_the_schema_guard_is_an_allowlist_not_a_token_blocklist():
    """`next_step` carries a plan and contains none of the obvious banned words.
    The A.4 key set is the contract."""
    schema = json.loads(json.dumps(NARRATIVE_SCHEMA))
    schema["properties"]["next_step"] = {"type": "string"}
    with pytest.raises(ClinicalJudgementLeak, match="A.4"):
        assert_no_judgement_fields(schema)
    with pytest.raises(ClinicalJudgementLeak):
        NarrativeSynthesizer(LLMClient(EchoTransport(["{}"])), schema=schema)


def test_an_item_citing_a_date_that_was_never_supplied_is_dropped():
    case = by_name("model_invents_a_date")
    result = synth(case).synthesize(case.encounters, patient_id=case.patient_id)
    assert [i.text for i in result.items] == ["Growth tracking well, no parental concerns"]
    assert len(result.dropped) == 1
    assert "not supplied" in result.dropped[0]["reason"]
    assert "Febrile seizure" in result.dropped[0]["item"]


def test_an_item_that_reads_as_a_suggested_order_is_dropped():
    case = by_name("model_suggests_an_order")
    result = synth(case).synthesize(case.encounters, patient_id=case.patient_id)
    texts = [i.text for i in result.items]
    assert not any("spirometry" in t for t in texts)
    assert any("suggested order" in d["reason"] for d in result.dropped)
    # ...and the legitimate item beside it survives.
    assert any("Albuterol used twice" in t for t in texts)


@pytest.mark.parametrize(
    "text",
    [
        "Recommend starting an inhaled steroid",
        "Consider a sweat chloride test",
        "Findings concerning for asthma",
        "Rule out reflux",
        "Increase the dose to twice daily",
        "Will need a referral to cardiology",
    ],
)
def test_judgement_language_is_caught_in_every_form(text):
    case = by_name("model_invents_a_date")
    response = json.dumps(
        {
            "recent_relevant_history": [
                {"item": text, "source_date": case.encounters[0].encounter_date.isoformat()}
            ],
            "open_threads": [], "unresolved_parent_concerns": [], "medication_changes": [],
        }
    )
    result = NarrativeSynthesizer(
        LLMClient(EchoTransport([response]))
    ).synthesize(case.encounters, patient_id="p")
    assert result.items == []
    assert len(result.dropped) == 1


def test_a_true_claim_filed_under_the_wrong_date_is_dropped():
    """Worse than no pointer: it sends the reader to the wrong place in the chart."""
    case = by_name("toddler_18_month_well")
    response = json.dumps(
        {
            "recent_relevant_history": [
                # This did happen -- on 2026-05-03, not at the twelve month visit.
                {"item": "Right tympanic membrane erythematous and bulging",
                 "source_date": "2026-02-10"},
            ],
            "open_threads": [], "unresolved_parent_concerns": [], "medication_changes": [],
        }
    )
    result = NarrativeSynthesizer(
        LLMClient(EchoTransport([response]))
    ).synthesize(case.encounters, patient_id="p")
    assert result.items == []
    assert "wrong place in the chart" in result.dropped[0]["reason"]


def test_only_the_last_three_encounters_reach_the_model():
    """README I-03: minimum necessary, and it keeps latency and cost down."""
    case = by_name("toddler_18_month_well")
    extra = list(case.encounters) + [
        Encounter(date(2024, 1, 5), "Sick", "Old note nobody needs today."),
        Encounter(date(2023, 6, 1), "Well child", "Older still."),
    ]
    synthesizer = synth(case)
    result = synthesizer.synthesize(extra, patient_id="p")
    assert len(result.source_dates) == MAX_ENCOUNTERS
    assert "2024-01-05" not in result.source_dates
    assert any("minimum necessary" in w for w in result.warnings)


def test_a_schema_violation_degrades_the_section_and_not_the_brief():
    """Hard constraint 3 with no guessing: the section is empty and says why."""
    case = by_name("toddler_18_month_well")
    result = NarrativeSynthesizer(
        LLMClient(EchoTransport(["not json at all"]), max_repair_attempts=0)
    ).synthesize(case.encounters, patient_id="p")
    assert result.items == []
    assert any("computed sections of this brief are unaffected" in w for w in result.warnings)


def test_no_encounters_means_no_narrative_and_no_error():
    result = NarrativeSynthesizer(LLMClient(EchoTransport([]))).synthesize(
        [], patient_id="p"
    )
    assert result.is_empty
    assert any("no prior encounters" in w for w in result.warnings)


# ==========================================================================
# brief.py — one screen, and the two kinds of content kept apart
# ==========================================================================


def test_ai_content_is_confined_to_its_own_marked_section():
    case = by_name("toddler_18_month_well")
    narrative = synth(case).synthesize(case.encounters, patient_id=case.patient_id)
    brief = assemble(
        patient_label=case.patient_label, age_label=case.age_label,
        visit_type=case.visit_type, appointment_local=case.appointment_local,
        provider=case.provider, generated_utc=NOW,
        immunizations_due=case.immunizations_due, narrative=narrative,
    )
    ai_sections = [s for s in brief.sections if s.ai_generated]
    assert len(ai_sections) == 1
    assert "AI-generated" in ai_sections[0].title
    for section in brief.sections:
        for marker, _text in section.lines:
            assert (marker == AI_MARKER) == section.ai_generated


def test_every_ai_line_carries_its_source_encounter_date():
    case = by_name("toddler_18_month_well")
    narrative = synth(case).synthesize(case.encounters, patient_id=case.patient_id)
    brief = assemble(
        patient_label=case.patient_label, age_label=case.age_label,
        visit_type=case.visit_type, appointment_local=case.appointment_local,
        provider=case.provider, generated_utc=NOW, narrative=narrative,
    )
    ai_lines = [t for s in brief.sections if s.ai_generated for _m, t in s.lines]
    assert ai_lines
    for line in ai_lines:
        assert line.rstrip().endswith("]") and "[20" in line


def test_medication_doses_are_removed_from_the_narrative():
    """README I-03: the brief 'deliberately omits data that must be verified
    in-chart (e.g. exact medication doses)'."""
    case = by_name("toddler_18_month_well")
    narrative = synth(case).synthesize(case.encounters, patient_id=case.patient_id)
    brief = assemble(
        patient_label=case.patient_label, age_label=case.age_label,
        visit_type=case.visit_type, appointment_local=case.appointment_local,
        provider=case.provider, generated_utc=NOW, narrative=narrative,
    )
    text = render_text(brief)
    assert "400mg" not in text and "5 mL" not in text
    assert "[dose - verify in chart]" in text
    # ...and collapsed to one marker rather than a line of rubble.
    assert "[dose - verify in chart]/[dose" not in text


def test_the_footer_says_it_is_not_a_substitute_for_chart_review():
    brief = assemble(
        patient_label="X", age_label="4y", visit_type="Well child",
        appointment_local="09:00", provider="Dr. A", generated_utc=NOW,
    )
    assert "not a substitute for chart review" in render_text(brief)


def test_the_one_screen_cap_is_enforced_and_overflow_is_reported():
    """A brief that quietly truncates hides the item nobody knew was there."""
    threads = [
        OpenThread("referral", f"Referral {i} placed", date(2026, 1, 1), 200)
        for i in range(40)
    ]
    brief = assemble(
        patient_label="X", age_label="4y", visit_type="Well child",
        appointment_local="09:00", provider="Dr. A", generated_utc=NOW,
        open_threads=threads,
    )
    assert brief.line_count <= MAX_LINES
    assert brief.overflow > 0
    assert any("did not fit the one-screen cap" in w for w in brief.warnings)


def test_an_empty_due_section_says_so_rather_than_vanishing():
    brief = assemble(
        patient_label="X", age_label="4y", visit_type="Sick", appointment_local="09:00",
        provider="Dr. A", generated_utc=NOW,
    )
    assert brief.sections[0].title == "DUE TODAY"
    assert "nothing due" in brief.sections[0].lines[0][1]


# ==========================================================================
# batch.py — the fixed pipeline
# ==========================================================================


def test_the_batch_produces_one_brief_per_patient(db, audit, reviewed_schedule, reference):
    client = LLMClient(EchoTransport([c.model_response for c in CASES if c.encounters]))
    batch = run_batch(
        [day_for(c) for c in CASES], clinic_date=CLINIC_DATE,
        schedule=reviewed_schedule, reference=reference,
        synthesizer=NarrativeSynthesizer(client, audit=audit), generated_utc=NOW,
        feedback=FeedbackLog(db, audit=audit),
    )
    assert len(batch.briefs) == len(CASES)
    assert batch.failures == []
    assert batch.as_dict()["narrative_items_dropped"] == 2


def test_one_patient_failing_does_not_stop_the_batch(reviewed_schedule, reference):
    broken = day_for(by_name("toddler_18_month_well"))
    broken.age_months = float("nan")
    good = day_for(by_name("adolescent_first_phqa"))
    batch = run_batch(
        [broken, good], clinic_date=CLINIC_DATE, schedule=reviewed_schedule,
        reference=reference, synthesizer=None, generated_utc=NOW,
    )
    assert len(batch.briefs) == 1
    assert len(batch.failures) == 1
    assert batch.failures[0]["patient_id"] == broken.patient_id


def test_an_unmeasurable_value_is_named_rather_than_silently_absent(
    reviewed_schedule, reference
):
    """A percentile that quietly does not appear is indistinguishable from a
    percentile that is fine."""
    day = day_for(by_name("adolescent_first_phqa"))
    day.measurements = (
        Measurement(Indicator.WEIGHT_FOR_AGE, 55.0, 300.0, "F", "2026-08-24"),
    )
    batch = run_batch(
        [day], clinic_date=CLINIC_DATE, schedule=reviewed_schedule,
        reference=reference, synthesizer=None, generated_utc=NOW,
    )
    assert any("outside" in w for w in batch.briefs[0].warnings)


def test_a_method_switch_is_reported_as_not_assessed_not_as_a_finding(
    reviewed_schedule, reference
):
    day = day_for(by_name("length_to_stature_switch"))
    batch = run_batch(
        [day], clinic_date=CLINIC_DATE, schedule=reviewed_schedule,
        reference=reference, synthesizer=None, generated_utc=NOW,
    )
    brief = batch.briefs[0]
    assert any("trend not assessed" in w for w in brief.warnings)
    flags = [s for s in brief.sections if s.title == "FLAGS"]
    assert not any("stature for age crossed" in t for s in flags for _m, t in s.lines)


def test_the_significant_crossing_reaches_the_flags_section(reviewed_schedule, reference):
    day = day_for(by_name("bmi_crossing_school_age"))
    batch = run_batch(
        [day], clinic_date=CLINIC_DATE, schedule=reviewed_schedule,
        reference=reference, synthesizer=None, generated_utc=NOW,
    )
    text = render_text(batch.briefs[0])
    assert "bmi for age crossed" in text
    assert "Allergy referral placed" in text


# ==========================================================================
# feedback.py — is this working at all
# ==========================================================================


def test_a_wrong_verdict_must_say_what_was_wrong(db, audit):
    log = FeedbackLog(db, audit=audit)
    log.register(brief_id="b1", patient_id="p1", clinic_date="2026-08-24",
                 generated_utc=NOW, had_narrative=True, prompt_hash="h1")
    with pytest.raises(ValueError, match="next prompt revision"):
        log.record(brief_id="b1", verdict=Verdict.WRONG, given_by="ma", now=NOW)
    log.record(brief_id="b1", verdict=Verdict.WRONG, given_by="ma", now=NOW,
               detail="wrong ear")
    assert log.wrong_items()[0]["detail"] == "wrong ear"


def test_a_second_click_corrects_rather_than_double_votes(db, audit):
    log = FeedbackLog(db, audit=audit)
    log.register(brief_id="b1", patient_id="p1", clinic_date="2026-08-24",
                 generated_utc=NOW, had_narrative=False)
    log.record(brief_id="b1", verdict=Verdict.USEFUL, given_by="ma", now=NOW)
    log.record(brief_id="b1", verdict=Verdict.NOT_USEFUL, given_by="ma", now=NOW)
    report = log.report()
    assert report["overall"]["rated"] == 1
    assert report["overall"]["not_useful"] == 1


def test_the_open_rate_denominator_is_every_brief_produced(db, audit):
    log = FeedbackLog(db, audit=audit)
    for index in range(4):
        log.register(brief_id=f"b{index}", patient_id=f"p{index}",
                     clinic_date="2026-08-24", generated_utc=NOW, had_narrative=False)
    log.mark_opened("b0", now=NOW)
    assert log.report()["open_rate"] == 0.25


def test_wrong_is_never_averaged_into_a_satisfaction_score(db, audit):
    log = FeedbackLog(db, audit=audit)
    for index in range(4):
        log.register(brief_id=f"b{index}", patient_id=f"p{index}",
                     clinic_date="2026-08-24", generated_utc=NOW, had_narrative=True)
        log.record(brief_id=f"b{index}",
                   verdict=Verdict.WRONG if index == 3 else Verdict.USEFUL,
                   given_by="ma", now=NOW,
                   detail="pointed at the wrong visit" if index == 3 else "")
    report = log.report()
    assert report["overall"]["useful_rate"] == 0.75
    assert report["overall"]["wrong"] == 1
    assert report["overall"]["wrong_rate"] == 0.25


def test_the_report_says_when_the_narrative_is_subtracting_value(db, audit):
    """If briefs with narrative rate worse than briefs without, the decision is
    whether to keep generating it -- not how to tune the prompt."""
    log = FeedbackLog(db, audit=audit)
    for index in range(4):
        log.register(brief_id=f"ai{index}", patient_id=f"p{index}",
                     clinic_date="2026-08-24", generated_utc=NOW, had_narrative=True)
        log.record(brief_id=f"ai{index}", verdict=Verdict.NOT_USEFUL,
                   given_by="ma", now=NOW)
    for index in range(4):
        log.register(brief_id=f"plain{index}", patient_id=f"q{index}",
                     clinic_date="2026-08-24", generated_utc=NOW, had_narrative=False)
        log.record(brief_id=f"plain{index}", verdict=Verdict.USEFUL,
                   given_by="ma", now=NOW)
    report = log.report()
    assert report["narrative_delta"] == -1.0
    assert "subtracting value" in report["narrative_verdict"]


def test_feedback_on_an_unregistered_brief_is_refused(db, audit):
    log = FeedbackLog(db, audit=audit)
    with pytest.raises(KeyError):
        log.record(brief_id="nope", verdict=Verdict.USEFUL, given_by="ma", now=NOW)


def test_every_fixture_runs_end_to_end(db, audit, reviewed_schedule, reference):
    client = LLMClient(EchoTransport([c.model_response for c in CASES if c.encounters]))
    batch = run_batch(
        [day_for(c) for c in CASES], clinic_date=CLINIC_DATE,
        schedule=reviewed_schedule, reference=reference,
        synthesizer=NarrativeSynthesizer(client, audit=audit), generated_utc=NOW,
    )
    for brief in batch.briefs:
        text = render_text(brief)
        assert "not a substitute for chart review" in text
        assert brief.line_count <= MAX_LINES
        for banned in ("recommend", "consider ", "rule out", "differential"):
            ai_text = " ".join(
                t for s in brief.sections if s.ai_generated for _m, t in s.lines
            ).lower()
            assert banned not in ai_text


# ==========================================================================
# Adversarial-review regressions. One test per finding; each reproduces the
# original failure, so a revert makes exactly one of them go red.
# ==========================================================================


def test_a_collapse_within_the_tail_is_flagged(reference):
    """FINDING: the chart lines stop at the 3rd and 97th, so a child already
    outside them had none left to cross.

    A boy falling from the 1.8th percentile to the 0.008th -- the single most
    alarming trajectory in the schedule -- produced no crossing, no flag and no
    line on the brief.
    """
    earlier = reference.place(Measurement(Indicator.WEIGHT_FOR_AGE, 7.35, 9.0, "M"))
    later = reference.place(Measurement(Indicator.WEIGHT_FOR_AGE, 7.45, 15.0, "M"))
    assert earlier.percentile < 3.0 and later.percentile < 3.0
    crossing = channel_crossing(earlier, later)
    assert crossing is not None
    assert crossing.lines_crossed == ()  # nothing printed lies between them
    assert crossing.significant is True
    assert crossing.direction == "down"
    assert "SD down beyond the printed chart lines" in crossing.describe()


def test_a_small_move_inside_the_tail_is_not_flagged(reference):
    """The other direction: the z rule must not make every extreme child a flag."""
    earlier = reference.place(Measurement(Indicator.WEIGHT_FOR_AGE, 7.35, 9.0, "M"))
    later = reference.place(Measurement(Indicator.WEIGHT_FOR_AGE, 7.6, 10.0, "M"))
    crossing = channel_crossing(earlier, later)
    assert crossing is None or not crossing.significant


def test_a_transposed_decimal_is_refused_not_flagged_as_growth(reference):
    """FINDING: 11.0 kg keyed as 110 kg produced a percentile of 100 and a
    four-channel crossing at the top of the brief."""
    from modules.previsit.growth import ImplausibleMeasurement

    with pytest.raises(ImplausibleMeasurement, match="data-entry error"):
        reference.place(Measurement(Indicator.WEIGHT_FOR_AGE, 110.0, 24.0, "F"))
    with pytest.raises(ImplausibleMeasurement):
        reference.place(Measurement(Indicator.WEIGHT_FOR_AGE, 1.0, 24.0, "F"))
    # ...and a real value at the same age still computes.
    assert reference.place(Measurement(Indicator.WEIGHT_FOR_AGE, 11.0, 24.0, "F"))


def test_percentiles_are_never_printed_as_zeroth_or_hundredth():
    """FINDING: `int(round(...))` produced "0th percentile" and "100th"."""
    from modules.previsit.growth import ordinal

    assert ordinal(0.06) == "<1st"
    assert ordinal(0.4) == "<1st"
    assert ordinal(99.6) == ">99th"
    assert ordinal(99.9994) == ">99th"
    assert (ordinal(1.0), ordinal(3.0), ordinal(93.0)) == ("1st", "3rd", "93rd")


def test_the_channel_index_agrees_with_the_crossing_count(reference):
    """FINDING: `bisect_left` put a point sitting exactly on a printed line in
    the band below, so moving ONTO a line counted as a crossing and moving OFF
    one did not. Two public numbers describing the same movement must agree."""
    from modules.previsit.growth import MAJOR_CHANNELS, GrowthPoint

    def point(percentile):
        return GrowthPoint("weight_for_age", 1.0, 24.0, "M", 0.0, percentile, "t")

    for a, b in ((2.9, 3.0), (3.0, 4.0), (24.9, 25.0), (25.0, 25.1), (49.0, 76.0)):
        crossing = channel_crossing(point(a), point(b))
        counted = crossing.channels_crossed if crossing else 0
        assert abs(point(b).channel - point(a).channel) == counted


def test_sex_spelled_out_is_the_same_child(reference):
    a = reference.place(Measurement(Indicator.WEIGHT_FOR_AGE, 12.0, 24.0, "female"))
    b = reference.place(Measurement(Indicator.WEIGHT_FOR_AGE, 13.0, 30.0, "F"))
    channel_crossing(a, b)  # must not raise


def test_one_completion_cannot_satisfy_two_due_ages(schedule):
    """FINDING: any two due ages closer than twice the window let one completion
    satisfy both. A single M-CHAT-R/F at 21 months read as COMPLETE for the
    18-month AND the 24-month screen, and the 24-month autism screen never
    appeared on any brief for that child."""
    one = [CompletedScreening("autism_screen", date(2026, 5, 1), 21.0)]
    for age, expected in ((21.0, Status.DUE), (24.0, Status.DUE), (28.0, Status.OVERDUE)):
        statuses = {s.definition.id: s for s in schedule.evaluate(age_months=age, completed=one)}
        assert statuses["autism_screen"].status == expected, age


def test_a_missed_earlier_screening_is_not_absorbed_by_a_later_one(schedule):
    """FINDING: only the latest reached target was examined, so OVERDUE silently
    downgraded to DUE and two missed developmental screens left no trace."""
    at_27 = {s.definition.id: s for s in schedule.evaluate(age_months=27.0)}
    status = at_27["developmental_screen"]
    assert status.status == Status.OVERDUE
    assert status.target_age_months == 9.0
    assert "outstanding" in status.because


def test_a_banded_screening_can_actually_read_overdue(schedule):
    """FINDING: whenever `every_months <= 2 * window_months`, OVERDUE was
    unreachable -- six years of no lead risk assessment rendered identically to
    one done three months ago, under a statutory obligation (410 ILCS 45)."""
    at_71 = {s.definition.id: s for s in schedule.evaluate(age_months=71.0)}
    assert at_71["lead_risk_assessment"].status == Status.OVERDUE
    assert at_71["oral_health_varnish"].status == Status.OVERDUE


def test_every_screening_can_reach_overdue_at_some_age(schedule):
    """The general form of the finding above, swept over the whole schedule."""
    reachable = {}
    for step in range(0, 521):
        for status in schedule.evaluate(age_months=step * 0.5):
            reachable.setdefault(status.definition.id, set()).add(status.status)
    gated = {
        d.id for d in schedule.screenings.values() if d.requires_risk_flag
    }
    for screening_id, statuses in reachable.items():
        if screening_id in gated:
            continue
        assert Status.OVERDUE in statuses, f"{screening_id} can never read overdue"


def test_the_next_due_date_is_reachable(schedule):
    """FINDING: `targets` was clipped to the patient's age, so the NOT_DUE
    branch that tells the MA when something is next due never fired."""
    at_two_months = {s.definition.id: s for s in schedule.evaluate(age_months=2.0)}
    status = at_two_months["developmental_screen"]
    assert status.status == Status.NOT_DUE
    assert status.because.startswith("next due at")


def test_a_once_in_band_panel_drawn_on_the_visit_it_was_requested_counts(schedule):
    """FINDING: the band DUE window opened six months before `from_months`, but
    the completion filter used `from_months` exactly -- so a lipid panel drawn
    on the very visit the brief marked DUE was rejected and the child was sent
    for a second venipuncture four years later."""
    at_102 = {s.definition.id: s for s in schedule.evaluate(age_months=102.0)}
    assert at_102["dyslipidemia_universal"].status == Status.DUE
    drawn = [CompletedScreening("dyslipidemia_universal", date(2026, 3, 1), 103.0)]
    for age in (103.0, 120.0, 126.0, 131.0):
        statuses = {
            s.definition.id: s for s in schedule.evaluate(age_months=age, completed=drawn)
        }
        assert statuses["dyslipidemia_universal"].status == Status.COMPLETE, age


def test_a_form_with_no_end_age_does_not_hang_the_batch():
    """FINDING: `while age <= float(raw.get("to_months", age))` re-evaluated the
    default from the loop variable, making the condition `age <= age`. The YAML
    is explicitly clinician-editable, so this hung the 17:00 batch."""
    loaded = PeriodicitySchedule(
        {
            "version": "t", "well_visit_months": [], "review": {"owner": "t"},
            "screenings": [{"id": "x", "name": "X", "due_at_months": [12],
                            "catch_up_until_months": 24}],
            "form_requirements": [
                {"id": "f", "name": "F", "from_months": 60, "every_months": 12},
            ],
        }
    )
    assert loaded.forms_due(age_months=60.0)  # returns, rather than spinning


def test_the_tb_risk_assessment_has_no_age_gap(schedule):
    """FINDING: the infant entry ended at 24 months (+3) and the annual entry
    started at 36, leaving ages 27-33 with no TB risk assessment at all."""
    for age in (27.0, 30.0, 33.0):
        statuses = [
            s for s in schedule.evaluate(age_months=age)
            if s.definition.risk_assessment_for == "tuberculosis" and s.actionable
        ]
        assert statuses, f"no TB risk assessment is actionable at {age} months"


# -- narrative post-conditions ---------------------------------------------


def _one_item(note_text, item, cited="2026-03-02"):
    encounters = [Encounter(date(2026, 3, 2), "Well child", note_text)]
    payload = json.dumps(
        {
            "recent_relevant_history": [{"item": item, "source_date": cited}],
            "open_threads": [], "unresolved_parent_concerns": [],
            "medication_changes": [],
        }
    )
    return NarrativeSynthesizer(
        LLMClient(EchoTransport([payload]))
    ).synthesize(encounters, patient_id="p")


@pytest.mark.parametrize(
    "note,item",
    [
        ("Screened for depression: PHQ-A negative. Denies suicidal ideation.",
         "Positive depression screen with suicidal ideation"),
        ("No wheeze on exam. Albuterol not needed.",
         "Wheeze on exam, albuterol needed"),
        ("Rapid strep negative. No fever.", "Rapid strep positive with fever"),
    ],
)
def test_an_item_that_inverts_the_notes_meaning_is_dropped(note, item):
    """FINDING: token overlap scores HIGHEST when a claim reuses the note's
    vocabulary with the sense inverted -- the most dangerous output the model
    can produce, because it reads as well-sourced and says the opposite of the
    record. "Positive depression screen with suicidal ideation" scored 0.80."""
    result = _one_item(note, item)
    assert result.items == []
    assert "contradicts the cited note" in result.dropped[0]["reason"]


def test_a_word_order_inversion_is_dropped():
    """FINDING: "the seizures have started since stopping the medication" scored
    1.00 against a note saying they stopped since starting it. Same words,
    opposite meaning, perfect score."""
    result = _one_item(
        "Mother reports the seizures have stopped since starting the medication.",
        "Mother reports the seizures have started since stopping the medication",
    )
    assert result.items == []
    assert "different order" in result.dropped[0]["reason"]


@pytest.mark.parametrize(
    "item",
    [
        "Needs inhaled steroid started for the nightly cough",
        "GI referral warranted for the reflux",
        "Asthma action plan may benefit from escalation",
        "Spirometry indicated at this visit",
        "Due for a repeat hemoglobin",
    ],
)
def test_orders_phrased_noun_first_are_still_orders(item):
    """FINDING: the blocklist anchored on verbs, and ordinary clinical word order
    puts the noun first -- so the sentences that read most like orders passed."""
    note = (
        "Mother asks about starting an inhaled steroid for the nightly cough. "
        "Discussed GI referral for the reflux. Reviewed the asthma action plan. "
        "Spirometry and hemoglobin were mentioned at this visit."
    )
    result = _one_item(note, item)
    assert result.items == [], item
    assert result.dropped


def test_a_referral_that_already_happened_is_a_fact_not_an_order():
    """The other direction, and it is README I-03's own worked example: "Allergy
    referral placed 2026-03-14, no report received" is the archetypal open
    thread. A flat blocklist on the word "referral" deleted it."""
    result = _one_item(
        "Allergy referral placed today. Parent to call for an appointment.",
        "Allergy referral placed, parent to call for an appointment",
    )
    assert [i.text for i in result.items] == [
        "Allergy referral placed, parent to call for an appointment"
    ]


def test_a_stem_shared_with_a_different_condition_does_not_ground():
    """FINDING: the 5-character prefix rule turned a vaccine into an infection --
    "pneumonia" matched "pneumococcal"."""
    result = _one_item(
        "Pneumococcal conjugate vaccine given at this visit. No fever.",
        "Pneumonia treated at this visit",
    )
    assert result.items == []


def test_a_changed_number_is_a_hard_drop():
    """FINDING: "Started amoxicillin 500 mg" scored 0.67 against a note saying
    250 mg -- two of three content words matched and the digit string was
    skipped by the prefix loop entirely."""
    wrong = _one_item(
        "Started amoxicillin 250 mg twice daily for ten days.",
        "Started amoxicillin 500 mg",
    )
    assert wrong.items == []
    assert "changed number" in wrong.dropped[0]["reason"]
    right = _one_item(
        "Started amoxicillin 250 mg twice daily for ten days.",
        "Started amoxicillin 250 mg twice daily",
    )
    assert right.items


def test_every_inference_is_recorded_in_the_audit_log(db, audit):
    """FINDING: a narrative could render on a clinician's screen with the audit
    database holding zero rows about it. `record_inference` appeared nowhere."""
    case = by_name("toddler_18_month_well")
    result = NarrativeSynthesizer(
        LLMClient(EchoTransport([case.model_response])), audit=audit
    ).synthesize(case.encounters, patient_id=case.patient_id)
    rows = audit.query("SELECT * FROM inference WHERE initiative_id = 'I-03'")
    assert len(rows) == 1
    assert rows[0]["model_id"]
    assert result.inference_id
    extra = json.loads(rows[0]["extra_json"])
    assert extra["items_kept"] == len(result.items)


def test_the_problem_list_goes_through_de_identification():
    """FINDING: only `note_text` was de-identified; the problem list -- free text
    from the same chart, routinely carrying a phone number or an MRN -- went to
    the model raw."""
    from nsp_core.phi import HybridDeidentifier, LeakGuard

    seen: list[str] = []

    class Recording(EchoTransport):
        def generate(self, *args, **kwargs):  # noqa: D102
            seen.append(kwargs.get("user", args[1] if len(args) > 1 else ""))
            return super().generate(*args, **kwargs)

    case = by_name("toddler_18_month_well")
    synthesizer = NarrativeSynthesizer(
        LLMClient(EchoTransport([case.model_response])),
        deidentifier=HybridDeidentifier(),
        leak_guard=LeakGuard(),
    )
    # The guard fails closed, so a surviving identifier raises rather than
    # reaching the transport at all.
    synthesizer.synthesize(
        case.encounters, patient_id="p",
        problem_list=["Asthma - call back on 847-555-0139"],
    )


def test_an_empty_item_is_recorded_rather_than_skipped():
    """FINDING: the one place in the module that violated "never silently skip"."""
    result = _one_item("Well child check today.", "   ")
    assert result.items == []
    assert result.dropped and result.dropped[0]["reason"] == "empty item"


# -- brief and batch --------------------------------------------------------


def test_a_declined_screening_is_visible_on_the_brief(reviewed_schedule, reference):
    """FINDING: `assemble` rendered only OVERDUE, DUE and RISK_UNKNOWN, so a
    DECLINED screening was indistinguishable from NOT_DUE and was never
    re-offered -- the opposite of what recording a decline is for."""
    day = day_for(by_name("toddler_18_month_well"))
    day.completed_screenings = tuple(day.completed_screenings) + (
        CompletedScreening("autism_screen", date(2026, 8, 1), 18.0, declined=True),
    )
    batch = run_batch(
        [day], clinic_date=CLINIC_DATE, schedule=reviewed_schedule,
        reference=reference, synthesizer=None, generated_utc=NOW,
    )
    text = render_text(batch.briefs[0])
    assert "previously declined, offer again" in text


def test_never_done_screenings_get_one_summary_line(reviewed_schedule, reference):
    """FINDING: twelve MISSED screenings on a six-year-old with an empty history
    produced a brief that mentioned none of them."""
    day = day_for(by_name("bmi_crossing_school_age"))
    day.completed_screenings = ()
    day.data_horizon_months = None  # no horizon: these really were never done
    batch = run_batch(
        [day], clinic_date=CLINIC_DATE, schedule=reviewed_schedule,
        reference=reference, synthesizer=None, generated_utc=NOW,
    )
    text = render_text(batch.briefs[0])
    assert "NEVER DONE (catch-up window closed)" in text
    assert "Autism screening" in text


def test_the_prior_measurement_is_chosen_deterministically(reference):
    """FINDING: the prior was picked by iterating a frozenset, so the same chart
    flagged a two-channel stature drop under one PYTHONHASHSEED and reported
    nothing under another."""
    from modules.previsit.batch import _growth_for

    day = PatientDay(
        patient_id="p", patient_label="P", sex="F", age_months=48.0, age_label="4y",
        visit_type="Well child", appointment_local="09:00", provider="Dr. A",
        measurements=(
            Measurement(Indicator.STATURE_FOR_AGE, 96.5, 48.0, "F", "2026-08-24", standing=True),
        ),
        prior_measurements=(
            Measurement(Indicator.LENGTH_FOR_AGE, 84.0, 20.0, "F", "2025-04-24", standing=False),
            Measurement(Indicator.STATURE_FOR_AGE, 95.0, 36.0, "F", "2025-08-24", standing=True),
        ),
    )
    _points, crossings, notes = _growth_for(reference, day)
    # The directly comparable prior wins, every time.
    assert len(crossings) == 1
    assert crossings[0].significant and crossings[0].direction == "down"
    assert notes == []


def test_priors_are_compared_newest_first(reference):
    """FINDING: the last element of the sequence was kept, not the most recent
    measurement -- so an EHR returning newest-first compared the wrong pair."""
    from modules.previsit.batch import _growth_for

    priors = (
        Measurement(Indicator.WEIGHT_FOR_AGE, 16.0, 42.0, "M", "2026-02-24"),
        Measurement(Indicator.WEIGHT_FOR_AGE, 12.5, 24.0, "M", "2024-08-24"),
    )
    day = PatientDay(
        patient_id="p", patient_label="P", sex="M", age_months=48.0, age_label="4y",
        visit_type="Well child", appointment_local="09:00", provider="Dr. A",
        measurements=(Measurement(Indicator.WEIGHT_FOR_AGE, 16.4, 48.0, "M", "2026-08-24"),),
        prior_measurements=priors,
    )
    _points, crossings, _notes = _growth_for(reference, day)
    reversed_day = PatientDay(
        patient_id="p", patient_label="P", sex="M", age_months=48.0, age_label="4y",
        visit_type="Well child", appointment_local="09:00", provider="Dr. A",
        measurements=day.measurements, prior_measurements=tuple(reversed(priors)),
    )
    _p2, crossings2, _n2 = _growth_for(reference, reversed_day)
    assert [c.as_dict() for c in crossings] == [c.as_dict() for c in crossings2]


def test_an_unknown_indicator_does_not_destroy_the_brief(reviewed_schedule, reference):
    """FINDING: `table_for` raises KeyError, which `_growth_for` did not catch,
    so one plausible EHR alias produced zero briefs for that patient -- and
    batch.py's own docstring promises the opposite."""
    day = day_for(by_name("toddler_18_month_well"))
    day.measurements = (Measurement("height_for_age", 81.5, 18.4, "F", "2026-08-24"),)
    batch = run_batch(
        [day], clinic_date=CLINIC_DATE, schedule=reviewed_schedule,
        reference=reference, synthesizer=None, generated_utc=NOW,
    )
    assert len(batch.briefs) == 1 and batch.failures == []
    assert any("height_for_age" in w for w in batch.briefs[0].warnings)
    assert "DTaP #4" in render_text(batch.briefs[0])


def test_statutory_forms_outrank_the_ai_section_when_space_runs_out():
    """FINDING: a full screen dropped two Illinois statutory forms to keep six
    unvalidated model lines."""
    case = by_name("toddler_18_month_well")
    narrative = synth(case).synthesize(case.encounters, patient_id=case.patient_id)
    brief = assemble(
        patient_label="X", age_label="5y", visit_type="Well child",
        appointment_local="09:00", provider="Dr. A", generated_utc=NOW,
        immunizations_due=[f"Vaccine {i}" for i in range(16)],
        forms_due=[
            {"name": "Illinois Certificate of Child Health Examination", "note": "K"},
            {"name": "Illinois Proof of School Dental Examination", "note": "K"},
        ],
        narrative=narrative,
    )
    titles = [s.title for s in brief.sections]
    assert "ADMIN" in titles
    assert titles.index("ADMIN") < len(titles)
    text = render_text(brief)
    assert "Illinois Certificate of Child Health Examination" in text


def test_a_truncated_section_says_how_many_it_hid():
    """FINDING: 40 overdue screenings became 25 lines and a bare overflow count,
    with no marker inside the section itself."""
    brief = assemble(
        patient_label="X", age_label="5y", visit_type="Well child",
        appointment_local="09:00", provider="Dr. A", generated_utc=NOW,
        immunizations_due=[f"Vaccine {i}" for i in range(40)],
    )
    text = render_text(brief)
    assert "more not shown - open the chart" in text
    assert brief.line_count <= MAX_LINES


@pytest.mark.parametrize(
    "text",
    [
        "Started amoxicillin 500 milligrams twice daily",
        "Albuterol 2 puffs every 4 hours",
        "Give 1 teaspoon at bedtime",
        "Ibuprofen 10 mg/kg every 6 hours",
    ],
)
def test_the_dose_scrub_covers_the_forms_people_actually_write(text):
    """FINDING: `milligrams`, `puffs`, `teaspoon` and `every N hours` were all
    absent, and `mg/kg` was unreachable behind `mg` -- leaving a half-scrubbed
    weight-based dose that reads as MORE authoritative than the original."""
    from modules.previsit.brief import _scrub_doses

    scrubbed, changed = _scrub_doses(text)
    assert changed
    assert "/kg" not in scrubbed
    assert not any(char.isdigit() for char in scrubbed)


def test_three_useful_clicks_cannot_outvote_one_wrong(db, audit):
    """FINDING: the LEFT JOIN fanned out to one row per (brief, verdict) and the
    rates were computed over clicks, so three MAs rating one brief USEFUL
    produced 75% useful / 25% wrong when the truth per brief was 50/50 -- which
    is exactly what the module docstring says must not happen."""
    log = FeedbackLog(db, audit=audit)
    for brief_id in ("b0", "b1"):
        log.register(brief_id=brief_id, patient_id=brief_id, clinic_date="2026-08-24",
                     generated_utc=NOW, had_narrative=True, prompt_hash="h1")
    for who in ("ma_a", "ma_b", "ma_c"):
        log.record(brief_id="b0", verdict=Verdict.USEFUL, given_by=who, now=NOW)
    log.record(brief_id="b1", verdict=Verdict.WRONG, given_by="dr", now=NOW,
               detail="pointed at the wrong visit")
    overall = log.report()["overall"]
    assert overall["rated"] == 2
    assert overall["useful_rate"] == 0.5
    assert overall["wrong_rate"] == 0.5
    assert overall["verdicts_recorded"] == 4


def test_a_brief_with_no_model_call_stays_in_the_control_arm(db, audit):
    """FINDING: filtering on prompt_hash dropped every brief that never had a
    model call, leaving "without narrative" composed only of briefs where
    synthesis ran and every item was dropped -- systematically the hardest
    charts, biasing the comparison in favour of the narrative."""
    log = FeedbackLog(db, audit=audit)
    log.register(brief_id="ai", patient_id="p1", clinic_date="2026-08-24",
                 generated_utc=NOW, had_narrative=True, prompt_hash="h1")
    log.register(brief_id="plain", patient_id="p2", clinic_date="2026-08-24",
                 generated_utc=NOW, had_narrative=False)
    log.record(brief_id="ai", verdict=Verdict.NOT_USEFUL, given_by="ma", now=NOW)
    log.record(brief_id="plain", verdict=Verdict.USEFUL, given_by="ma", now=NOW)
    report = log.report(prompt_hash="h1")
    assert report["with_narrative"]["briefs"] == 1
    assert report["without_narrative"]["briefs"] == 1
    assert report["narrative_delta"] == -1.0


def test_a_narrative_trimmed_off_the_page_is_not_counted_as_shown(db, audit, reviewed_schedule, reference):
    """FINDING: `had_narrative` was recorded even when the one-screen cap removed
    the AI section, so a rating was attributed to text nobody saw."""
    case = by_name("toddler_18_month_well")
    day = day_for(case)
    day.immunizations_due = tuple(f"Vaccine {i}" for i in range(40))
    log = FeedbackLog(db, audit=audit)
    batch = run_batch(
        [day], clinic_date=CLINIC_DATE, schedule=reviewed_schedule, reference=reference,
        synthesizer=synth(case), generated_utc=NOW, feedback=log,
    )
    assert batch.briefs[0].has_ai_content is False
    row = db.one("SELECT * FROM previsit_brief")
    assert row["had_narrative"] == 1 and row["narrative_shown"] == 0


def test_regex_only_de_identification_says_so_rather_than_implying_safe_harbor():
    """FINDING (partial): the regex pass removes phone numbers, MRNs and dates
    and does NOT remove person names -- and `LeakGuard` re-derives with the same
    regexes, so it does not catch them either. A caller who passed a
    de-identifier is entitled to know the pass is partial."""
    from nsp_core.phi import HybridDeidentifier

    case = by_name("toddler_18_month_well")
    result = NarrativeSynthesizer(
        LLMClient(EchoTransport([case.model_response])),
        deidentifier=HybridDeidentifier(),
    ).synthesize(
        case.encounters, patient_id="p",
        problem_list=["Asthma - call Dr. Robert Chen on 847-555-0139"],
    )
    assert any("WITHOUT the clinical NER model" in w for w in result.warnings)
    assert any("not Safe Harbor" in w for w in result.warnings)


def test_a_data_horizon_reports_unknown_rather_than_asserting_a_miss(schedule):
    """Day one of a deployment otherwise shows every established patient overdue
    for a decade of screenings that were in fact done on paper. Every brief is
    red, the red means nothing, and the periodicity section is ignored inside a
    week -- README I-03's "brief becomes noise" risk, arriving on the first
    morning."""
    without = {s.definition.id: s for s in schedule.evaluate(age_months=74.0)}
    assert without["oral_health_risk"].status == Status.MISSED
    assert without["vision_instrument"].status == Status.MISSED

    with_horizon = {
        s.definition.id: s
        for s in schedule.evaluate(age_months=74.0, data_horizon_months=60.0)
    }
    assert with_horizon["vision_instrument"].status == Status.UNKNOWN
    assert "data horizon" in with_horizon["vision_instrument"].because
    # Anything due AFTER the horizon is still held to account.
    assert with_horizon["vision_acuity"].status in (Status.DUE, Status.OVERDUE)
    assert with_horizon["blood_pressure"].status in (Status.DUE, Status.OVERDUE)


def test_the_data_horizon_never_hides_a_current_screening(schedule):
    """The horizon must not become a way to make the whole checklist go quiet."""
    for age in (18.4, 74.0, 157.0):
        statuses = schedule.evaluate(age_months=age, data_horizon_months=age - 1.0)
        assert any(s.actionable for s in statuses), age
