"""I-09 — eligibility verification and denial prevention.

The controls under test, in the order they matter:

  1. a coverage denial is never communicated to a patient by this system
  2. "not found" and "inactive" are different facts and stay different
  3. a 271 the parser only half-understood is not used at all
  4. no card is submitted before a person confirmed the fields
  5. a standard CARC code beats the model's reading of the free text
"""

from __future__ import annotations

import ast
import json
import os
from datetime import date, datetime, timedelta, timezone

import pytest

from modules.eligibility import (
    CARC_ROOT_CAUSE,
    CardReader,
    Denial,
    DenialClassifier,
    Determination,
    EligibilityRequest,
    MalformedEDI,
    Outcome,
    PatientCommunicationRefused,
    PayerRecord,
    PayerTable,
    Response271,
    RootCause,
    SubsetParser,
    UnconfirmedCard,
    build_270,
    build_denial_report,
    determine,
    draft_appeal,
    outreach_draft,
)
from modules.eligibility import coverage as coverage_module
from modules.eligibility import x12 as x12_module
from modules.eligibility.fixtures import (
    CASES,
    DENIALS,
    NOW,
    SERVICE_DATE,
    build_payer_table,
    by_name,
    card_response,
    denial_response,
)
from nsp_core.audit import AuditLog
from nsp_core.llm import EchoTransport, LLMClient

_MODULE_DIR = os.path.dirname(os.path.abspath(x12_module.__file__))


@pytest.fixture
def payers():
    return build_payer_table()


@pytest.fixture
def parser():
    return SubsetParser()


@pytest.fixture
def audit(tmp_path):
    return AuditLog(tmp_path / "audit.sqlite3", hmac_key=b"test-key")


def run(name, payers, parser, *, on=SERVICE_DATE) -> Determination:
    case = by_name(name)
    response = parser.parse_271(case.response_271) if case.response_271 else None
    return determine(case.request, response, payers, on=on)


# ==========================================================================
# structural guards
# ==========================================================================


def test_no_model_in_the_edi_or_the_coverage_determination():
    """README I-09: "the core is a solved EDI problem". A model reading a
    specified format adds latency and a failure mode."""
    for name in ("x12.py", "coverage.py"):
        tree = ast.parse(open(os.path.join(_MODULE_DIR, name), encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "nsp_core.llm" not in node.module, name
                assert node.module not in ("openai", "anthropic"), name
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in ("openai", "anthropic"), name


def test_only_two_files_may_call_a_model():
    """Card OCR and denial free-text classification. Nothing else."""
    users = set()
    for name in sorted(os.listdir(_MODULE_DIR)):
        if not name.endswith(".py") or name in ("demo.py", "__init__.py"):
            continue
        source = open(os.path.join(_MODULE_DIR, name), encoding="utf-8").read()
        if "nsp_core.llm" in source:
            users.add(name)
    assert users == {"cards.py", "denials.py"}, users


# ==========================================================================
# the 270
# ==========================================================================


def test_a_270_carries_the_dependent_because_the_patient_is_a_child(parser):
    case = by_name("rosa_active")
    edi = build_270(case.request, control_number="0001", created=NOW)
    assert "NM1*03*1*Alvarez*Rosa" in edi     # the dependent
    assert "NM1*IL*1*Alvarez*Marisol" in edi  # the subscriber
    assert "DMG*D8*20180314" in edi           # the child's date of birth
    assert edi.endswith("~")


def test_a_270_with_a_malformed_npi_is_refused(parser):
    """A rejection two seconds later reads like a coverage problem to whoever
    reads the queue."""
    case = by_name("rosa_active")
    bad = EligibilityRequest(
        **{**case.request.__dict__, "provider_npi": "19028844"}
    )
    with pytest.raises(MalformedEDI, match="ten digits"):
        build_270(bad, control_number="1", created=NOW)


def test_a_270_missing_a_required_field_is_refused():
    case = by_name("rosa_active")
    bad = EligibilityRequest(**{**case.request.__dict__, "subscriber_member_id": ""})
    with pytest.raises(MalformedEDI, match="subscriber_member_id"):
        build_270(bad, control_number="1", created=NOW)


# ==========================================================================
# the 271 parser
# ==========================================================================


def test_a_segment_the_parser_does_not_know_makes_the_response_untrusted(parser):
    """A subset parser that quietly returns its partial understanding is how a
    system tells a family they have no insurance because the payer used a
    segment nobody implemented."""
    response = parser.parse_271(by_name("ivy_unparseable").response_271)
    assert response.unparsed
    assert not response.trustworthy
    # ...and it DID read the active benefit. It just refuses to be trusted on it.
    assert response.has_active_benefit


def test_an_unknown_eb_code_is_unparsed_rather_than_assumed(parser):
    """An EB code this subset does not know could mean covered, not covered, or
    'call us'."""
    response = parser.parse_271(
        "ST*271*1~NM1*PR*2*Payer*****PI*P1~EB*Q*IND*30**PLAN~"
    )
    assert response.unparsed
    assert not response.trustworthy
    assert not response.benefits


def test_coinsurance_is_read_whichever_convention_the_payer_used(parser):
    """A payer sending `.2` and one sending `20` both mean twenty percent.
    Treating `.2` as 0.2% would tell a family their share is nothing."""
    for raw, expect in (("20", 20.0), (".2", 20.0), ("0.2", 20.0)):
        response = parser.parse_271(
            f"ST*271*1~NM1*PR*2*P*****PI*P1~EB*B*IND*30**PLAN***{raw}~"
        )
        assert response.benefits[0].percent == pytest.approx(expect)


def test_an_empty_payload_is_refused(parser):
    with pytest.raises(MalformedEDI):
        parser.parse_271("   ")


# ==========================================================================
# the determination
# ==========================================================================


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_every_fixture_reaches_its_expected_outcome(case, payers, parser):
    result = run(case.name, payers, parser)
    assert result.outcome == case.expect_outcome


def test_not_found_is_not_the_same_fact_as_inactive(payers, parser):
    """A mistyped member id and a terminated policy look identical to a front
    desk and are opposite facts."""
    not_found = run("nadia_not_found", payers, parser)
    inactive = run("theo_terminated", payers, parser)
    assert not_found.outcome == Outcome.NOT_FOUND
    assert inactive.outcome == Outcome.INACTIVE
    assert "NOT a statement that the patient is uninsured" in not_found.reason
    assert "mistyped member id" in not_found.reason


def test_no_response_is_not_a_coverage_fact(payers, parser):
    result = run("priya_no_response", payers, parser)
    assert result.outcome == Outcome.INDETERMINATE
    assert "says nothing about" in result.reason


def test_a_contradictory_271_goes_to_a_person(payers, parser):
    result = run("lucas_contradictory", payers, parser)
    assert result.outcome == Outcome.INDETERMINATE
    assert "both active and inactive" in result.reason


def test_an_untrustworthy_271_is_never_acted_on(payers, parser):
    result = run("ivy_unparseable", payers, parser)
    assert result.outcome == Outcome.INDETERMINATE
    assert result.warnings


def test_out_of_network_is_detected_before_the_visit(payers, parser):
    """README I-09: "proactive call before the visit, not a surprise bill after.
    This alone justifies the initiative on patient-relationship grounds.\""""
    result = run("omar_out_of_network", payers, parser)
    assert result.outcome == Outcome.OUT_OF_NETWORK
    assert result.in_network is False
    assert "BEFORE the visit" in result.reason


def test_a_lapsed_contract_makes_a_covered_patient_out_of_network(payers, parser):
    """Humana's contract ended in May and is still on the public list."""
    case = by_name("rosa_active")
    request = EligibilityRequest(
        **{**case.request.__dict__, "payer_id": "HUMANA", "payer_name": "Humana"}
    )
    response = parser.parse_271(case.response_271)
    result = determine(request, response, payers, on=SERVICE_DATE)
    assert result.outcome == Outcome.OUT_OF_NETWORK
    assert "not in force" in result.reason


def test_the_copay_is_surfaced_so_it_is_collected_at_the_desk(payers, parser):
    result = run("rosa_active", payers, parser)
    assert result.copay_usd == 25.0
    assert result.deductible_remaining_usd == 450.0


def test_a_payer_absent_from_the_table_is_unknown_not_out_of_network(payers, parser):
    case = by_name("rosa_active")
    request = EligibilityRequest(
        **{**case.request.__dict__, "payer_id": "NEWCO", "payer_name": "NewCo"}
    )
    result = determine(
        request, parser.parse_271(case.response_271), payers, on=SERVICE_DATE
    )
    assert result.outcome == Outcome.OUT_OF_NETWORK
    assert any("not in the payer table" in w for w in result.warnings)


# ==========================================================================
# the control this module is built around
# ==========================================================================


@pytest.mark.parametrize(
    "name",
    ["theo_terminated", "nadia_not_found", "omar_out_of_network",
     "ivy_unparseable", "lucas_contradictory", "priya_no_response"],
)
def test_no_bad_news_is_ever_drafted_for_a_patient(name, payers, parser):
    """README I-09: "Never auto-communicate a coverage denial to a patient.
    Route to a human who calls the payer to confirm before any patient
    contact.\""""
    result = run(name, payers, parser)
    assert not result.patient_safe
    with pytest.raises(PatientCommunicationRefused, match="never"):
        outreach_draft(result, family_name="the family", child_first_name="Child")


def test_the_only_drafted_message_is_good_news(payers, parser):
    result = run("rosa_active", payers, parser)
    message = outreach_draft(
        result, family_name="the Alvarez family", child_first_name="Rosa"
    )
    assert "25.00" in message
    assert result.patient_safe


def test_there_is_no_template_for_bad_news_to_reach():
    """The refusal is that the templates do not exist, not that a flag is off."""
    templates = coverage_module._OUTREACH_TEMPLATES
    assert set(templates) == {Outcome.ACTIVE}
    signature = coverage_module.outreach_draft.__code__.co_varnames
    assert "force" not in signature
    assert "override" not in signature


# ==========================================================================
# the payer table
# ==========================================================================


def test_the_public_list_is_generated_from_the_table(payers):
    """README I-09 opens by noting the published list is dated January 2016."""
    listed = payers.public_list(SERVICE_DATE)
    assert "Blue Cross Blue Shield of Illinois" in listed
    assert "Humana" not in listed        # contract lapsed in May
    assert "Oscar Health" not in listed  # never contracted


def test_lapsed_and_expiring_contracts_are_reported(payers):
    rows = {r["payer_id"]: r for r in payers.stale_records(SERVICE_DATE)}
    assert rows["HUMANA"]["state"] == "expired"
    assert rows["CIGNA"]["state"] == "ending"


def test_a_payer_record_that_ends_before_it_begins_is_refused():
    table = PayerTable()
    with pytest.raises(ValueError, match="ends before it begins"):
        table.add(
            PayerRecord("X", "X", True, date(2026, 1, 1),
                        effective_to=date(2025, 1, 1))
        )


# ==========================================================================
# card capture
# ==========================================================================


def test_a_card_is_never_submitted_unconfirmed(audit):
    """README I-09: "Front desk confirms every extracted field against the card
    image side by side; never auto-submit.\""""
    reader = CardReader(LLMClient(EchoTransport([card_response()])), audit=audit)
    extraction = reader.read("BCBS ... ID W9928311402", document_id="card_1")
    assert extraction.unconfirmed == ["member_id", "payer_name"]
    with pytest.raises(UnconfirmedCard, match="not been confirmed"):
        extraction.for_submission()


def test_the_person_s_value_wins_and_the_correction_is_recorded(audit):
    reader = CardReader(LLMClient(EchoTransport([card_response()])), audit=audit)
    extraction = reader.read("...", document_id="card_1")
    now = datetime(2026, 8, 24, tzinfo=timezone.utc)
    # The payer name was read correctly; the person confirms it verbatim.
    extraction.confirm(
        "payer_name", by="dana",
        value="Blue Cross Blue Shield of Illinois", at=now,
    )
    # The member id was not: the model dropped the trailing digit.
    extraction.confirm("member_id", by="dana", value="W99283114021", at=now)
    submitted = extraction.for_submission()
    assert submitted["member_id"] == "W99283114021"
    assert extraction.corrections == ["member_id"]


def test_a_field_confirmed_as_empty_still_blocks(audit):
    reader = CardReader(LLMClient(EchoTransport([card_response()])), audit=audit)
    extraction = reader.read("...", document_id="card_1")
    now = datetime(2026, 8, 24, tzinfo=timezone.utc)
    extraction.confirm("payer_name", by="dana", value="BCBS", at=now)
    extraction.confirm("member_id", by="dana", value="", at=now)
    with pytest.raises(UnconfirmedCard, match="confirmed as empty"):
        extraction.for_submission()


def test_a_confirmation_names_the_person(audit):
    reader = CardReader(LLMClient(EchoTransport([card_response()])), audit=audit)
    extraction = reader.read("...", document_id="card_1")
    with pytest.raises(ValueError, match="names the person"):
        extraction.confirm(
            "member_id", by="  ", value="X",
            at=datetime(2026, 8, 24, tzinfo=timezone.utc),
        )


def test_an_unreadable_card_returns_none_rather_than_stopping_the_batch():
    reader = CardReader(LLMClient(EchoTransport(["not json"])))
    assert reader.read("...", document_id="card_1") is None


def test_the_card_extraction_is_audited(audit):
    reader = CardReader(LLMClient(EchoTransport([card_response()])), audit=audit)
    extraction = reader.read("...", document_id="card_1")
    assert extraction.inference_id
    rows = audit.query("SELECT * FROM inference WHERE initiative_id = ?", ("I-09",))
    assert len(rows) == 1


# ==========================================================================
# denials
# ==========================================================================


def test_a_standard_carc_code_beats_the_model(audit):
    """The payer formally asserted the code; the remark is prose somebody typed."""
    classifier = DenialClassifier(
        LLMClient(EchoTransport([denial_response("coding", 0.99, "terminated")])),
        audit=audit,
    )
    result = classifier.classify(DENIALS[0])       # CARC 27 -> eligibility
    assert result.root_cause == RootCause.ELIGIBILITY
    assert result.decided_by == "carc"
    assert result.model_disagreed_with == "coding"


def test_an_unmapped_carc_falls_through_to_the_model(audit):
    classifier = DenialClassifier(
        LLMClient(EchoTransport([
            denial_response("eligibility", 0.92, "not enrolled in this plan")
        ])),
        audit=audit,
    )
    result = classifier.classify(DENIALS[3])       # CARC B7, unmapped
    assert result.decided_by == "model"
    assert result.root_cause == RootCause.ELIGIBILITY


def test_an_evidence_span_that_is_not_in_the_remark_is_refused(audit):
    """A post-condition, not a prompt instruction. An "evidence" field nobody
    verifies is a field the model can fill with anything."""
    classifier = DenialClassifier(
        LLMClient(EchoTransport([
            denial_response("eligibility", 0.95, "coverage lapsed in June")
        ])),
        audit=audit,
    )
    result = classifier.classify(DENIALS[4])       # remark says nothing of the kind
    assert result.root_cause == RootCause.OTHER
    assert result.needs_human
    assert "does not appear in the remark" in result.reasoning


@pytest.mark.parametrize("evidence,why", [
    ("e", "too short"),
    ("a", "too short"),
    ("", "quoted nothing"),
    ("the patient", "too short"),
])
def test_a_one_character_evidence_span_is_not_grounding(evidence, why, audit):
    """The check used to be a bare substring test, satisfied by a single
    character: `evidence="e"` appears in almost any remark, so the model could
    place any denial in any bucket -- and the span was copied verbatim into the
    appeal letter sent to the payer."""
    classifier = DenialClassifier(
        LLMClient(EchoTransport([denial_response("eligibility", 0.95, evidence)])),
        audit=audit,
    )
    result = classifier.classify(DENIALS[4])
    assert result.needs_human
    assert result.root_cause == RootCause.OTHER
    assert why in result.reasoning


def test_a_span_lifted_out_of_a_longer_word_is_not_a_quotation(audit):
    """A coding denial placed in the eligibility bucket -- the number the whole
    initiative is measured on -- by quoting a fragment."""
    coding = Denial(
        claim_id="CLM-1", patient_id="p", payer_id="UHC",
        service_date=date(2026, 7, 1), denied_on=date(2026, 8, 1),
        amount_usd=100.0, carc_code="ZZ",
        remark_text="Procedure code invalid for the patient's age.",
    )
    classifier = DenialClassifier(
        LLMClient(EchoTransport([
            denial_response("eligibility", 0.95, "for the patient")
        ])),
        audit=audit,
    )
    result = classifier.classify(coding)
    assert result.needs_human
    assert not result.preventable_by_verification


def test_a_real_quotation_still_grounds(audit):
    """The check must not push genuine classifications to the human queue."""
    classifier = DenialClassifier(
        LLMClient(EchoTransport([
            denial_response("eligibility", 0.92, "not enrolled in this plan")
        ])),
        audit=audit,
    )
    result = classifier.classify(DENIALS[3])
    assert not result.needs_human
    assert result.decided_by == "model"


def test_a_low_confidence_reading_goes_to_a_person(audit):
    classifier = DenialClassifier(
        LLMClient(EchoTransport([
            denial_response("eligibility", 0.3, "not enrolled in this plan")
        ])),
        audit=audit,
    )
    result = classifier.classify(DENIALS[3])
    assert result.needs_human
    assert result.root_cause == RootCause.OTHER


def test_a_denial_with_no_model_and_no_mapping_is_unclassified():
    classifier = DenialClassifier(client=None)
    result = classifier.classify(DENIALS[4])
    assert result.decided_by == "unclassified"
    assert result.needs_human


def test_the_report_splits_the_number_the_initiative_is_judged_on(audit):
    """A denial rate is one number hiding five causes moving in different
    directions."""
    responses = [
        denial_response("eligibility", 0.94, "coverage terminated"),
        denial_response("coding", 0.91, "diagnosis is inconsistent"),
        denial_response("timely_filing", 0.96, "timely filing limit"),
        denial_response("eligibility", 0.92, "not enrolled in this plan"),
        denial_response("eligibility", 0.88, "coverage lapsed in June"),
    ]
    classifier = DenialClassifier(LLMClient(EchoTransport(responses)), audit=audit)
    report = build_denial_report("2026-08", [classifier.classify(d) for d in DENIALS])
    counts = report.by_root_cause()
    assert counts[RootCause.ELIGIBILITY]["count"] == 2
    assert counts[RootCause.CODING]["count"] == 1
    assert counts[RootCause.TIMELY_FILING]["count"] == 1
    assert len(report.unclassified) == 1
    assert report.total_usd == pytest.approx(1170.50)


def test_an_appeal_cites_facts_and_is_never_sent():
    classification = DenialClassifier(client=None).classify(DENIALS[0])
    with pytest.raises(ValueError, match="cites facts"):
        draft_appeal(classification, practice_name="NSP", facts=[])
    draft = draft_appeal(
        classification, practice_name="NSP",
        facts=["A 271 dated 2026-07-01 reported active coverage."],
    )
    assert "REVIEW BEFORE SENDING" in draft.body
    assert draft.requires_review_by


# ==========================================================================
# adversarial-review regressions
# ==========================================================================


def test_r05_terminated_coverage_is_never_drafted_as_good_news(payers, parser):
    """#2 THE WORST ONE. A 271 with EB*1 and a plan end date before the service
    date came out `active`, `patient_safe`, and `outreach_draft` produced "we've
    confirmed your insurance". The claim then denied CARC 27 and the balance
    went to the family."""
    case = by_name("rosa_active")
    terminated = case.response_271.replace(
        "DTP*346*D8*20260101~", "DTP*346*D8*20260101~DTP*347*D8*20260531~"
    )
    result = determine(
        case.request, parser.parse_271(terminated), payers, on=SERVICE_DATE
    )
    assert result.outcome == Outcome.INACTIVE
    assert not result.patient_safe
    with pytest.raises(PatientCommunicationRefused):
        outreach_draft(result, family_name="the family", child_first_name="Rosa")


@pytest.mark.parametrize("qualifier", ["347", "349", "357"])
def test_r07_every_coverage_end_qualifier_is_read(qualifier, payers, parser):
    """#10c `349` and `307` fell through the DTP chain with no `else`, so a
    payer stating a coverage end date in one of them produced plan_ends=None,
    unparsed=[], and a trustworthy response asserting active coverage."""
    case = by_name("rosa_active")
    edi = case.response_271.replace(
        "DTP*346*D8*20260101~", f"DTP*{qualifier}*D8*20260531~"
    )
    response = parser.parse_271(edi)
    assert response.plan_ends == date(2026, 5, 31)
    result = determine(case.request, response, payers, on=SERVICE_DATE)
    assert result.outcome == Outcome.INACTIVE


def test_r07_an_unhandled_dtp_qualifier_is_collected_not_skipped(parser):
    """The design rule, applied: "every segment it does not know is COLLECTED,
    not skipped"."""
    response = parser.parse_271(
        "ST*271*1~NM1*PR*2*P*****PI*P1~EB*1*IND*30**PLAN~DTP*999*D8*20260531~"
    )
    assert response.unparsed
    assert not response.trustworthy


def test_r07_a_repeating_eb03_becomes_one_line_per_service_type(parser):
    """#10a EB03 is a REPEATING simple element (separator `^`). Splitting only
    on `:` produced a service type of `'30^98^68'` that matched nothing, while
    `trustworthy` stayed True."""
    response = parser.parse_271(
        "ST*271*1~NM1*PR*2*P*****PI*P1~EB*1*IND*30^98^68**GOLD PPO 500~"
    )
    assert response.trustworthy
    assert {b.service_type for b in response.benefits} == {"30", "98", "68"}
    assert response.benefits_for("98")
    assert all(b.eb_code == "1" for b in response.benefits)


def test_r07_a_payer_free_text_note_reaches_the_determination(parser, payers):
    """#10b MSG was on the ignored list, so "COVERAGE TERMINATED 05/31/2026 -
    VERIFY BEFORE SERVICE" was discarded while the response reported active
    coverage and `BenefitLine.message` -- a documented field -- stayed empty."""
    case = by_name("rosa_active")
    edi = case.response_271 + "MSG*COVERAGE TERMINATED 05/31/2026 - VERIFY~"
    response = parser.parse_271(edi)
    assert any("TERMINATED" in b.message for b in response.benefits) or response.messages
    result = determine(case.request, response, payers, on=SERVICE_DATE)
    assert result.outcome == Outcome.INDETERMINATE
    assert not result.patient_safe


def test_r06_an_unstated_network_is_not_in_network(payers, parser):
    """#9b `in_network=True if network is None else network` widened a threshold
    on missing data, contradicting `BenefitLine`'s own docstring: "Empty means
    the payer did not say, which is NOT the same as in-network.\""""
    case = by_name("rosa_active")
    silent = case.response_271.replace(
        "EB*A*IND*98**GOLD PPO 500**25*****Y~", "EB*A*IND*98**GOLD PPO 500**25~"
    )
    result = determine(
        case.request, parser.parse_271(silent), payers, on=SERVICE_DATE
    )
    assert result.in_network is None
    assert not result.patient_safe, "an unstated network is not confirmed good news"
    assert any("did not state a network indicator" in w for w in result.warnings)


def test_r06_an_unrecognised_network_indicator_is_unparsed(parser):
    """`IN`/`OUT` is what the docstring promised and what `_network` did not
    understand. Anything the parser cannot place routes to `unparsed`."""
    known = parser.parse_271(
        "ST*271*1~NM1*PR*2*P*****PI*P1~EB*A*IND*98**PLAN**25*****IN~"
    )
    assert known.trustworthy
    assert known.benefits[0].network_indicator == "Y"

    unknown = parser.parse_271(
        "ST*271*1~NM1*PR*2*P*****PI*P1~EB*A*IND*98**PLAN**25*****QQ~"
    )
    assert not unknown.trustworthy


def test_r06_the_coinsurance_comes_from_the_in_network_line(parser):
    """#9c `_amount` preferred the in-network line; `_percent` did not look at
    the indicator at all, so a family was quoted the out-of-network coinsurance
    beside the in-network copay."""
    response = parser.parse_271(
        "ST*271*1~NM1*PR*2*P*****PI*P1~"
        "EB*1*IND*30**PLAN~"
        "EB*B*IND*98**PLAN***40****N~"      # out of network first
        "EB*B*IND*98**PLAN***20****Y~"
        "EB*A*IND*98**PLAN**60*****N~"
        "EB*A*IND*98**PLAN**25*****Y~"
    )
    assert coverage_module._percent(response, "B") == pytest.approx(20.0)
    assert coverage_module._amount(response, "A") == pytest.approx(25.0)


def test_r06_a_full_coinsurance_of_one_is_not_one_percent(parser):
    """Mutation testing: the `<=` boundary in `_pct` could become `<`, and a
    payer sending `1` for 100% coinsurance would read as 1%."""
    from modules.eligibility.x12 import _pct

    assert _pct("1") == pytest.approx(100.0)
    assert _pct("0.2") == pytest.approx(20.0)
    assert _pct("20") == pytest.approx(20.0)


def test_r06_the_in_network_preference_on_amounts_is_asserted(parser):
    """Mutation testing: `_amount` could drop the preference its docstring
    describes and nothing failed."""
    response = parser.parse_271(
        "ST*271*1~NM1*PR*2*P*****PI*P1~"
        "EB*A*IND*98**PLAN**80*****N~"
        "EB*A*IND*98**PLAN**25*****Y~"
    )
    assert coverage_module._amount(response, "A") == pytest.approx(25.0)


def test_r06_a_mixed_network_response_does_not_default_to_in_network(parser):
    """Mutation testing: `_network` could return True unless some line said `N`."""
    from modules.eligibility.coverage import _network

    assert _network(parser.parse_271(
        "ST*271*1~NM1*PR*2*P*****PI*P1~EB*1*IND*30**PLAN~"
    )) is None
    assert _network(parser.parse_271(
        "ST*271*1~NM1*PR*2*P*****PI*P1~EB*A*IND*98**PLAN**25*****Y~"
    )) is True
    assert _network(parser.parse_271(
        "ST*271*1~NM1*PR*2*P*****PI*P1~EB*A*IND*98**PLAN**25*****N~"
    )) is False
    # The case the name promises and the assertions above did not reach: the
    # payer answered both ways in one response. Unknown, not favourable.
    mixed = parser.parse_271(
        "ST*271*1~NM1*PR*2*P*****PI*P1~"
        "EB*A*IND*98**PLAN**25*****Y~"
        "EB*A*IND*98**PLAN**80*****N~"
    )
    assert _network(mixed) is None


def test_r06_a_mixed_network_answer_reaches_the_determination_as_unknown(
    parser, payers
):
    """The consequence, end to end: a contradictory network answer must leave
    `in_network` unset, raise the warning that says so, and make the
    determination unsafe to read out at the front desk."""
    case = by_name("rosa_active")
    response = parser.parse_271(
        "ST*271*1~NM1*PR*2*P*****PI*P1~"
        "EB*1*IND*98**PLAN~"
        "EB*A*IND*98**PLAN**25*****Y~"
        "EB*A*IND*98**PLAN**80*****N~"
    )
    result = determine(case.request, response, payers, on=SERVICE_DATE)
    assert result.in_network is None
    assert any("network indicator" in w for w in result.warnings)
    assert not result.patient_safe
