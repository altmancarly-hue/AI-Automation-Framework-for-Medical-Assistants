"""I-06 tests. Real matcher, real reference ranges, real EchoTransport.

`EchoTransport` and `ScriptedOCR` are shipped components, not mocks of this
module's logic: these tests drive the production classification, matching,
urgency and routing path against scripted OCR text and scripted model output.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from modules.fax import (
    CLINICAL_TYPES,
    DOCUMENT_TYPES,
    LOW_CONFIDENCE_THRESHOLD,
    MIN_CONFIDENCE,
    URGENCY_SYSTEM_PROMPT,
    Classification,
    ClassifierGate,
    ClassifierNotValidated,
    DocumentClassifier,
    EvalCase,
    FaxPipeline,
    InboundMonitor,
    MatchOutcome,
    OCRResult,
    OCRUnavailable,
    Page,
    PanelPatient,
    PatientMatcher,
    Queue,
    ReferenceRanges,
    ScriptedOCR,
    TaskKind,
    UnreviewedRanges,
    UrgencyTriage,
    chain,
    evaluate,
    extract_immunization_doses,
    jaro,
    jaro_winkler,
    normalise_name,
    parse_document_date,
    rank,
    route,
    split_name,
)
from modules.fax.fixtures import AGE_MONTHS, CASES, PANEL, by_name, eval_cases
from nsp_core.audit import AuditLog
from nsp_core.llm import EchoTransport, LLMClient

NOW = datetime(2026, 8, 24, 9, 30, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def ranges():
    loaded = ReferenceRanges.load()
    loaded.review["owner"] = "test-owner"
    return loaded


@pytest.fixture
def audit(tmp_path):
    return AuditLog(tmp_path / "audit.sqlite3", hmac_key=b"test-key")


@pytest.fixture
def matcher():
    return PatientMatcher(PANEL)


def make_pipeline(cases, *, ranges, audit=None, gate=None, matcher=None):
    responses = []
    for case in cases:
        responses.append(case.classifier_response)
        responses.append(case.urgency_response)
    client = LLMClient(EchoTransport(responses))
    return FaxPipeline(
        engines=[ScriptedOCR(documents={c.filename: list(c.pages) for c in cases})],
        classifier=DocumentClassifier(client, audit=audit),
        matcher=matcher or PatientMatcher(PANEL),
        triage=UrgencyTriage(client, ranges=ranges, audit=audit),
        gate=gate,
        audit=audit,
        on_duty_physician="dr_alvarez",
        indexer="ma_jess",
    )


def run_one(case, *, ranges, audit=None, matcher=None):
    pipeline = make_pipeline([case], ranges=ranges, audit=audit, matcher=matcher)
    return pipeline.process(case.filename, document_id=case.name, age_lookup=AGE_MONTHS)


# ==========================================================================
# THE CONSTRAINT: the deterministic half stays deterministic
# ==========================================================================


def test_no_model_in_ocr_matching_or_routing():
    """README I-06: "Yes for classification, no for the rest." """
    from modules.fax import match, ocr, route as route_module

    banned = ("nsp_core.llm", "openai", "anthropic")
    for module in (ocr, match, route_module):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                assert not any(name.startswith(b) for b in banned), (
                    f"{module.__name__} imports {name}"
                )


def test_no_module_scope_ocr_engine_import():
    """PaddleOCR and pytesseract are lazily imported so the repo installs on a
    box with neither."""
    from modules.fax import ocr

    tree = ast.parse(inspect.getsource(ocr))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                assert not name.startswith(("paddle", "pytesseract", "PIL"))


def test_the_taxonomy_is_closed_and_matches_the_schema():
    from modules.fax.classify import CLASSIFY_SCHEMA

    assert CLASSIFY_SCHEMA["additionalProperties"] is False
    assert set(CLASSIFY_SCHEMA["properties"]["document_type"]["enum"]) == set(DOCUMENT_TYPES)
    assert len(DOCUMENT_TYPES) == 10
    assert CLINICAL_TYPES < set(DOCUMENT_TYPES)


def test_the_urgency_prompt_is_appendix_a3_verbatim():
    for line in (
        "Classify its urgency. Bias STRONGLY toward over-flagging.",
        "- Anything you are unsure about",
        'Return "routine" ONLY for clearly administrative documents',
    ):
        assert line in URGENCY_SYSTEM_PROMPT


# ==========================================================================
# ocr.py
# ==========================================================================


def test_document_confidence_is_the_worst_page_not_the_average():
    """A five-page discharge summary whose middle page is a black smear scores
    0.80 on a mean. The document is not 80% readable; it is 80% readable and
    20% missing, and the missing fifth had the discharge medications in it."""
    result = OCRResult("x", [Page(i, "text", 0.99) for i in range(1, 5)] + [Page(5, "", 0.0)])
    assert result.mean_confidence > 0.79
    assert result.confidence == 0.0
    assert result.needs_human()
    assert [p.number for p in result.unreadable_pages()] == [5]
    assert "page(s) 5 of 5" in result.describe_quality()


def test_an_empty_page_is_not_a_confident_page():
    """Averaging an empty list of word scores returned 1.0 -- a perfectly
    confident reading of nothing."""
    from modules.fax.ocr import _pages_from_scores

    pages = _pages_from_scores(["", "hello"], [[], [0.9, 0.8]], "t")
    assert pages[0].confidence == 0.0
    assert pages[1].confidence == pytest.approx(0.85)


def test_the_engine_chain_falls_back_on_low_confidence_not_only_on_failure():
    """An engine that is running but reading a bad fax at 0.4 is not a success
    to be accepted just because it did not raise."""

    class Weak:
        name = "weak"

        def available(self):
            return True

        def read(self, path):
            return OCRResult(path, [Page(1, "blurry", 0.30)], engine="weak")

    class Strong:
        name = "strong"

        def available(self):
            return True

        def read(self, path):
            return OCRResult(path, [Page(1, "clear text", 0.95)], engine="strong")

    result = chain([Weak(), Strong()], "f.tiff")
    assert result.engine == "strong"
    assert result.fallback_used is True


def test_the_chain_keeps_the_best_attempt_when_every_engine_is_poor():
    class Poor:
        def __init__(self, name, score):
            self.name = name
            self.score = score

        def available(self):
            return True

        def read(self, path):
            return OCRResult(path, [Page(1, "x", self.score)], engine=self.name)

    result = chain([Poor("a", 0.30), Poor("b", 0.55)], "f.tiff")
    assert result.engine == "b"
    assert result.needs_human()


def test_every_engine_failing_raises_rather_than_returning_empty_text():
    class Broken:
        name = "broken"

        def available(self):
            return True

        def read(self, path):
            raise RuntimeError("no")

    with pytest.raises(OCRUnavailable, match="NOT filed"):
        chain([Broken()], "f.tiff")


# ==========================================================================
# match.py — the module that refuses
# ==========================================================================


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("martha", "marhta", 0.9611),
        ("dwayne", "duane", 0.8400),
        ("dixon", "dicksonx", 0.8133),
        ("", "", 1.0),
    ],
)
def test_jaro_winkler_matches_the_canonical_reference_values(a, b, expected):
    """The number decides whether a document reaches a child's chart, so it is
    implemented here and checked against the published values rather than
    imported from a package that may change between releases."""
    assert jaro_winkler(a, b) == pytest.approx(expected, abs=1e-4)


def test_jaro_alone_matches_its_reference_values():
    assert jaro("crate", "trace") == pytest.approx(0.7333, abs=1e-4)
    assert jaro("abc", "xyz") == 0.0


def test_the_prefix_boost_does_not_lift_genuinely_different_names():
    assert jaro_winkler("anderson", "andrade") < 0.92


def test_name_normalisation_handles_the_shapes_a_fax_carries():
    assert normalise_name("NAKAMURA, Ari  Jr.") == "nakamura ari"
    assert normalise_name("Marie-Claire O'Brien") == "marie claire obrien"
    assert normalise_name("Zoë Müller") == "zoe muller"
    assert split_name("Reyes, Sofia") == ("sofia", "reyes")
    assert split_name("Sofia Reyes") == ("sofia", "reyes")


def test_a_twin_discharge_with_no_first_name_goes_to_a_human(matcher):
    """The build plan names this case: twins share a DOB, so DOB alone is never
    sufficient. Test with a twin fixture."""
    result = matcher.match(name="Nakamura", dob=date(2019, 3, 14))
    assert result.outcome == MatchOutcome.AMBIGUOUS
    assert result.needs_human
    assert result.matched is None
    assert len(result.candidates) == 2


def test_a_flagged_twin_is_never_auto_filed_even_with_a_perfect_name(matcher):
    """FINDING: `multiple_birth` appeared four times in match.py -- the dataclass
    field, an English reason string, and two lines of a report -- and influenced
    NO decision. Setting it True changed nothing.

    With twins "Mia" and "Mila" on one date of birth, a document reading "Mla"
    -- the commonest OCR confusion there is -- scored Mila at 0.97 and Mia at
    0.92, so exactly one candidate cleared the threshold and the document was
    AUTO-FILED to the wrong twin's chart with no human in the loop.
    """
    result = matcher.match(name="Nakamura, Ari", dob=date(2019, 3, 14))
    assert result.outcome == MatchOutcome.AMBIGUOUS
    assert result.matched is None
    assert "one OCR character" in result.reason
    # The best candidate is still reported, so the human has somewhere to start.
    assert result.candidates[0].patient.patient_id == "p_ari"


def test_a_near_tie_goes_to_a_human_even_without_a_multiple_birth_flag():
    """The margin rule, for the twins nobody flagged. A threshold with no margin
    is a cliff that one mis-read character pushes a sibling across."""
    panel = [
        PanelPatient("mia", "Mia", "Nakamura", date(2019, 3, 14)),
        PanelPatient("mila", "Mila", "Nakamura", date(2019, 3, 14)),
    ]
    result = PatientMatcher(panel).match(name="Nakamura, Mla", dob=date(2019, 3, 14))
    assert result.outcome == MatchOutcome.AMBIGUOUS
    assert "margin" in result.reason


def test_two_candidates_over_the_threshold_go_to_a_human_even_with_a_clear_winner():
    """Taking the highest score is exactly how one twin receives the other
    twin's discharge summary."""
    panel = [
        PanelPatient("a", "Ari", "Nakamura", date(2019, 3, 14), multiple_birth=True),
        PanelPatient("b", "Arie", "Nakamura", date(2019, 3, 14), multiple_birth=True),
    ]
    result = PatientMatcher(panel).match(name="Nakamura, Ari", dob=date(2019, 3, 14))
    assert result.outcome == MatchOutcome.AMBIGUOUS
    assert result.matched is None
    assert result.needs_human


def test_siblings_with_the_same_surname_and_different_dobs_match_correctly(matcher):
    sofia = matcher.match(name="Reyes, Sofia", dob=date(2021, 6, 30))
    mateo = matcher.match(name="Reyes, Mateo", dob=date(2019, 11, 2))
    assert sofia.matched.patient_id == "p_sofia"
    assert mateo.matched.patient_id == "p_mateo"


def test_a_surname_only_document_never_matches_on_the_surname_alone(matcher):
    """"Reyes" matches three children in this practice."""
    result = matcher.match(name="Reyes", dob=date(2021, 6, 30))
    assert result.outcome == MatchOutcome.NAME_TOO_WEAK


def test_the_date_of_birth_is_never_fuzzy_matched(matcher):
    """03/04 and 04/03 are two real children."""
    result = matcher.match(name="Reyes, Sofia", dob=date(2021, 6, 3))
    assert result.outcome == MatchOutcome.NO_CANDIDATE
    assert "never fuzzy-matched" in result.reason


def test_a_document_with_no_date_of_birth_goes_to_a_human(matcher):
    result = matcher.match(name="Reyes, Sofia", dob=None)
    assert result.outcome == MatchOutcome.NO_DOB
    assert result.needs_human


def test_a_duplicate_chart_is_reported_rather_than_guessed_between(matcher):
    result = matcher.match(name="Okafor, Wren", dob=date(2020, 9, 5))
    assert result.outcome == MatchOutcome.DUPLICATE_PANEL
    assert "the duplicate chart is itself the finding" in result.reason


def test_the_name_order_on_the_fax_does_not_matter(matcher):
    for written in ("Torres, Mila", "Mila Torres", "TORRES MILA", "torres, mila"):
        result = matcher.match(name=written, dob=date(2024, 4, 12))
        assert result.outcome == MatchOutcome.MATCHED, written
        assert result.matched.patient_id == "p_mila"


def test_a_recorded_alias_matches(matcher):
    result = matcher.match(name="Torres, Ludmila", dob=date(2024, 4, 12))
    assert result.outcome == MatchOutcome.MATCHED


def test_panel_hygiene_finds_duplicates_and_unflagged_multiples(matcher):
    duplicates = matcher.duplicate_report()
    assert len(duplicates) == 1
    assert set(duplicates[0]["patient_ids"]) == {"p_dup_a", "p_dup_b"}
    unflagged = matcher.unflagged_multiples()
    assert [u["surname"] for u in unflagged] == ["okafor"]


# ==========================================================================
# urgency.py — the deterministic override
# ==========================================================================


def test_the_shipped_ranges_have_no_clinical_owner_and_the_pipeline_refuses():
    unreviewed = ReferenceRanges.load()
    assert unreviewed.has_clinical_owner is False
    with pytest.raises(UnreviewedRanges, match="clinical owner"):
        unreviewed.require_reviewed()


def test_a_critical_value_escalates_a_model_that_said_routine(ranges):
    """README I-06's CRITICAL risk, and its stated control: every document with
    a numeric lab value outside reference range is force-flagged BY RULE
    regardless of LLM output."""
    case = by_name("critical_lab_called_routine")
    processed = run_one(case, ranges=ranges)
    assert processed.urgency.model_urgency == "routine"
    assert processed.urgency.urgency == "urgent"
    assert processed.urgency.escalated is True
    assert any("Hemoglobin" in o for o in processed.urgency.overrides)
    assert processed.routed.queue == Queue.URGENT_ALERT


def test_the_override_can_only_escalate(ranges):
    """If the model says urgent and the rules find nothing, it stays urgent.
    A.3 tells the model to flag anything it is unsure about, and second-guessing
    that with a regex is the wrong direction."""
    triage = UrgencyTriage(
        LLMClient(EchoTransport([json.dumps({
            "urgency": "urgent", "confidence": 0.6,
            "reason": "unsure", "abnormal_values_detected": [],
        })])),
        ranges=ranges,
    )
    result = triage.triage("Nothing numeric here at all.", document_id="d", age_months=60)
    assert result.urgency == "urgent"
    assert result.abnormal == []


def test_reference_ranges_are_age_banded(ranges):
    """A hemoglobin of 11.0 is normal at nine months and low at fifteen years.
    A single adult range would force-flag most healthy infants, and an alert
    that fires on everything is an alert nobody reads."""
    text = "Hemoglobin 11.0 g/dL"
    assert not [a for a in ranges.scan(text, age_months=9) if a.source == "reference_range"]
    assert [a for a in ranges.scan(text, age_months=180) if a.source == "reference_range"]


def test_a_newborn_bilirubin_is_not_flagged_by_the_child_range(ranges):
    case = by_name("newborn_bilirubin_normal_for_age")
    processed = run_one(case, ranges=ranges)
    numeric = [a for a in processed.urgency.abnormal if a.source == "reference_range"]
    assert not any(a.analyte == "Total bilirubin" for a in numeric)


def test_an_implausible_value_is_still_escalated_but_labelled(ranges):
    """A potassium of 41.0 is a lost decimal point, not a critical value. It is
    flagged anyway -- refusing to flag a number you cannot read is the wrong
    direction -- and labelled so nobody chases a phantom."""
    case = by_name("ocr_damaged_potassium")
    processed = run_one(case, ranges=ranges)
    assert processed.routed.queue == Queue.URGENT_ALERT
    shifted = [a for a in processed.urgency.abnormal if a.decimal_shift_suspected]
    assert shifted and "check whether this is 4.1" in shifted[0].describe()


def test_the_labs_own_abnormal_flag_is_read_without_a_reference_range(ranges):
    """Works on analytes this table has never heard of, which is most of them."""
    found = ranges.scan("Ferritin        8   ng/mL      L\n")
    flagged = [a for a in found if a.source == "document_flag"]
    assert flagged and flagged[0].direction == "low"


def test_a_critical_phrase_is_word_anchored(ranges):
    """FINDING from the demo: `"stat"` matched inside `"status asthmaticus"` and
    turned an ordinary discharge into a critical-value alert."""
    assert not [
        a for a in ranges.scan("Admitted with status asthmaticus.")
        if a.source == "critical_phrase"
    ]
    assert [
        a for a in ranges.scan("Critical value called to provider at 0918.")
        if a.source == "critical_phrase"
    ]


def test_stat_is_not_a_critical_value_notification(ranges):
    """FINDING: "STAT" is an ORDER PRIORITY, printed on a large share of
    hospital paper before anyone has looked at a result. Treating it as a
    critical-value notification put a portable chest radiograph on the on-duty
    physician's one-hour pager."""
    assert not [
        a for a in ranges.scan("Portable STAT chest radiograph. No acute findings.")
        if a.source == "critical_phrase"
    ]


@pytest.mark.parametrize(
    "text",
    [
        "CRITICAL  VALUE called at 0918",          # OCR double space
        "CRITICAL VALUES called at 0918",           # plural
        "Critical-value notification",              # hyphenated
        "Result phoned to the ordering provider",   # what labs actually print
        "Called and read-back verified by Dr. Ruiz",
    ],
)
def test_critical_phrases_survive_the_shapes_a_fax_carries(ranges, text):
    """FINDING: a literal single-spaced substring match missed every one of
    these, all of which are what a laboratory actually prints."""
    assert [a for a in ranges.scan(text) if a.source == "critical_phrase"], text


def test_a_failed_urgency_call_escalates_rather_than_assuming_routine(ranges):
    triage = UrgencyTriage(
        LLMClient(EchoTransport(["not json"]), max_repair_attempts=0), ranges=ranges
    )
    result = triage.triage("Some document text.", document_id="d")
    assert result.urgency == "urgent"
    assert "escalated rather than assumed routine" in result.reason


def test_an_unreadable_page_on_a_clinical_document_raises_the_level(ranges):
    triage = UrgencyTriage(None, ranges=ranges)
    result = triage.triage(
        "A consult note.", document_id="d", document_type="specialist_consult",
        unreadable_pages=[2],
    )
    assert rank(result.urgency) >= rank("needs_physician_review")
    assert any("did not OCR" in o for o in result.overrides)


def test_a_missing_age_narrows_the_ranges_and_says_so(ranges):
    """FINDING: an unknown age WIDENED every band, so a twin's document -- the
    case that most needs scrutiny -- had its potassium band widened to 3.5-5.9
    and its glucose band to the newborn 40-110, and a potassium of 5.8 and a
    glucose of 46 were both silently cleared. The same numbers on a matched
    child raised a one-hour page.

    Widening a safety threshold in response to missing data is the one thing
    this module must never do."""
    triage = UrgencyTriage(None, ranges=ranges)
    unknown = triage.triage(
        "Potassium 5.8 mmol/L\nGlucose 46 mg/dL", document_id="d", age_months=None
    )
    # Both are abnormal at some paediatric ages and normal at others, so neither
    # is silently cleared and neither invents a one-hour page.
    unsettled = [a for a in unknown.abnormal if a.source == "age_unknown"]
    assert {a.analyte for a in unsettled} == {"Potassium", "Glucose"}
    assert unknown.urgency == "needs_physician_review"
    assert any("UNSETTLED" in w for w in unknown.warnings)

    # ...and the same numbers on a matched child still raise the one-hour page.
    known = triage.triage(
        "Potassium 5.8 mmol/L\nGlucose 46 mg/dL", document_id="d", age_months=60
    )
    assert [a for a in known.abnormal if a.source == "reference_range"]
    assert known.urgency == "urgent"

    # A value outside EVERY band is abnormal whatever the age, and is urgent
    # even with no age at all.
    extreme = triage.triage("Hemoglobin 3.2 g/dL", document_id="d", age_months=None)
    assert [a for a in extreme.abnormal if a.source == "reference_range"]
    assert extreme.urgency == "urgent"

    # And an ordinary CBC on an unmatched fax does not page anybody.
    normal = triage.triage(
        "Hemoglobin 12.6 g/dL\nHematocrit 37.8 %\nWBC 8.1 K/uL",
        document_id="d", age_months=None,
    )
    assert normal.urgency != "urgent"


# ==========================================================================
# classify.py — the eval gate
# ==========================================================================


def test_an_unmeasured_classifier_cannot_route_anything(ranges):
    pipeline = make_pipeline(
        [by_name("routine_records_request")], ranges=ranges, gate=ClassifierGate()
    )
    with pytest.raises(ClassifierNotValidated, match="No classifier has been approved|no classifier has been approved"):
        pipeline.process("routine_records_request.tiff", document_id="d")


def test_the_gate_refuses_an_eval_set_with_classes_it_never_saw():
    """A class the eval set never contained has not been measured, it has been
    assumed."""
    cases = eval_cases(repeats=12)
    responses = [by_name(c.document_id.split("__r")[0]).classifier_response for c in cases]
    result = evaluate(DocumentClassifier(LLMClient(EchoTransport(responses))), cases)
    assert result.accuracy == 1.0
    assert not result.passes()
    assert any("no labelled examples at all" in b for b in result.blockers())
    with pytest.raises(ClassifierNotValidated):
        ClassifierGate().approve(result, on="2026-08-24")


def test_the_gate_refuses_a_classifier_that_buries_a_lab_result():
    """The confusion-matrix cell behind README I-06's CRITICAL risk: an outside
    lab predicted as `records_request` lands in an administrative queue and the
    critical value in it is never read."""
    text = "LAB RESULT"
    cases = [
        EvalCase(f"d{i}", text, expected_type="outside_lab") for i in range(20)
    ]
    bad = json.dumps({
        "document_type": "records_request", "confidence": 0.95,
        "patient_name": None, "patient_dob": None, "sending_facility": None,
        "one_line_summary": "x",
    })
    result = evaluate(
        DocumentClassifier(LLMClient(EchoTransport([bad] * 20))), cases
    )
    leaks = result.clinical_leaks()
    assert leaks and leaks[0]["expected"] == "outside_lab"
    assert any("predicted as" in b and "administrative" in b for b in result.blockers())
    # 20 of 20 is far over the 2% budget, so the gate refuses rather than
    # merely reporting.
    assert not result.passes()


def test_an_unclassifiable_document_counts_against_its_true_class():
    """Dropping them inflates recall on exactly the documents the classifier
    could not read."""
    cases = [EvalCase(f"d{i}", "x", expected_type="outside_lab") for i in range(10)]
    result = evaluate(
        DocumentClassifier(
            LLMClient(EchoTransport(["not json"] * 10), max_repair_attempts=0)
        ),
        cases,
    )
    assert result.unclassifiable == 10
    assert result.recall("outside_lab") == 0.0
    assert result.support("outside_lab") == 10


def test_a_prompt_change_invalidates_the_gate():
    gate = ClassifierGate(
        model_id="m", model_version="v", prompt_hash="abc", validated_on="today"
    )
    classifier = DocumentClassifier(LLMClient(EchoTransport(["{}"])))
    with pytest.raises(ClassifierNotValidated, match="prompts are code"):
        gate.require(classifier)


def test_a_low_confidence_classification_is_honoured(ranges):
    """README I-06: "Include a `confidence` field and honor it." """
    case = by_name("routine_records_request")
    unsure = json.loads(case.classifier_response)
    unsure["confidence"] = 0.40
    client = LLMClient(EchoTransport([json.dumps(unsure), case.urgency_response]))
    pipeline = FaxPipeline(
        engines=[ScriptedOCR(documents={case.filename: list(case.pages)})],
        classifier=DocumentClassifier(client),
        matcher=PatientMatcher(PANEL),
        triage=UrgencyTriage(client, ranges=ranges),
    )
    processed = pipeline.process(case.filename, document_id=case.name)
    assert processed.routed.queue == Queue.HUMAN_INDEXING
    assert any("honour the confidence field" in r for r in processed.routed.reasons)


# ==========================================================================
# route.py
# ==========================================================================


def test_a_routine_matched_administrative_document_auto_files(ranges):
    processed = run_one(by_name("routine_records_request"), ranges=ranges)
    assert processed.routed.queue == Queue.AUTO_FILE
    assert processed.routed.patient_id == "p_juno"
    assert processed.routed.tasks == []


def test_auto_filing_a_clinical_document_still_creates_a_task(ranges):
    """README I-06: "Auto-filing does not mean no task. Every clinically
    relevant type generates a review task." """
    case = by_name("unreadable_middle_page")
    routine = json.loads(case.urgency_response)
    routine["urgency"] = "routine"
    client = LLMClient(EchoTransport([case.classifier_response, json.dumps(routine)]))
    # Readable pages only, so the document reaches the auto-file branch.
    readable = [(text, 0.94) for text, _c in case.pages if text.strip()]
    pipeline = FaxPipeline(
        engines=[ScriptedOCR(documents={case.filename: readable})],
        classifier=DocumentClassifier(client),
        matcher=PatientMatcher(PANEL),
        triage=UrgencyTriage(client, ranges=ranges),
    )
    processed = pipeline.process(case.filename, document_id=case.name)
    assert processed.routed.queue == Queue.AUTO_FILE
    assert processed.classification.is_clinical
    assert [t.kind for t in processed.routed.tasks] == [TaskKind.ACKNOWLEDGE]


def test_an_urgent_document_produces_an_alert_and_a_parallel_verification(ranges):
    """README I-06: urgent goes to the on-duty physician AND a parallel human
    verification, because an urgency misclassification cannot be allowed to
    fail silently."""
    processed = run_one(by_name("critical_lab_called_routine"), ranges=ranges)
    kinds = [t.kind for t in processed.routed.tasks]
    assert TaskKind.URGENT_CONTACT in kinds
    assert TaskKind.VERIFY_URGENCY in kinds
    assignees = {t.assigned_to for t in processed.routed.tasks}
    assert len(assignees) == 2, "the verification must not go to the alerted person"


def test_an_urgent_document_with_no_patient_still_alerts(ranges):
    processed = run_one(by_name("twin_discharge_summary"), ranges=ranges)
    assert processed.routed.queue == Queue.URGENT_ALERT
    assert processed.routed.patient_id is None
    kinds = [t.kind for t in processed.routed.tasks]
    assert TaskKind.URGENT_CONTACT in kinds and TaskKind.INDEX_BY_HAND in kinds


def test_an_unreadable_document_goes_to_a_human_before_anything_else(ranges):
    processed = run_one(by_name("unreadable_middle_page"), ranges=ranges)
    assert processed.routed.queue == Queue.HUMAN_INDEXING
    assert processed.ocr.needs_human()
    assert any("page(s) 2 of 3" in r for r in processed.routed.reasons)


def test_an_unmatched_patient_never_auto_files(ranges):
    processed = run_one(by_name("unknown_patient_prior_auth"), ranges=ranges)
    assert processed.routed.queue == Queue.HUMAN_INDEXING
    assert processed.match.outcome == MatchOutcome.NO_CANDIDATE


def test_a_duplicate_chart_document_goes_to_a_human(ranges):
    processed = run_one(by_name("duplicate_chart_school_form"), ranges=ranges)
    assert processed.routed.queue == Queue.HUMAN_INDEXING
    assert processed.match.outcome in (
        MatchOutcome.DUPLICATE_PANEL, MatchOutcome.AMBIGUOUS
    )


# ==========================================================================
# the I-02 handoff
# ==========================================================================


def test_an_immunization_record_goes_to_reconciliation_not_to_the_chart(ranges):
    """README I-06 calls this "the highest-value cross-initiative link in the
    plan"."""
    from modules.immunization.matcher import DoseRecord

    processed = run_one(by_name("immunization_registry_printout"), ranges=ranges)
    assert processed.routed.queue == Queue.IMMUNIZATION_RECONCILIATION
    handoff = processed.routed.immunization_handoff
    assert handoff is not None
    assert handoff.patient_id == "p_mila"
    assert len(handoff.doses) == 6
    assert [t.kind for t in processed.routed.tasks] == [TaskKind.RECONCILE_IMMUNIZATIONS]

    records = handoff.as_dose_records()
    assert len(records) == 6
    for record in records:
        assert isinstance(record, DoseRecord)
        # An outside fax is a registry-shaped claim about care given elsewhere.
        # I-02 knows two sources; "outside_fax" is not one of them, and a dose
        # labelled with an unknown source silently loses every registry rule.
        assert record.source == "registry"
        assert record.record_id.startswith("immunization_registry_printout#")
        assert isinstance(record.given, date)
        assert record.known, record.product_text


def test_a_partial_date_is_handed_over_marked_not_completed():
    """A month-only date is a real dose at unknown precision, not a bad date.

    I-02's matcher already models this: `DosePrecision.MONTH` bars the pair from
    the rule-provable pass and sends it to adjudication instead. So the extractor
    keeps the dose, anchors it to the first of the month, and marks the
    precision -- rather than dropping it or inventing a day.
    """
    doses = extract_immunization_doses(
        "DTaP    04/12/2024\nRotavirus   06/2024\nMMR (not administered)\n"
    )
    assert len(doses) == 2

    partial = [d for d in doses if d["precision"] == "month"]
    assert len(partial) == 1
    assert partial[0]["given"] == date(2024, 6, 1)
    assert partial[0]["product_text"] == "Rotavirus"

    exact = [d for d in doses if d["precision"] == "day"]
    assert exact[0]["given"] == date(2024, 4, 12)

    # "not administered" is not an administration.
    assert not any("MMR" in d["product_text"] for d in doses)

    # ...but it is reported rather than silently dropped.
    _, unresolved = (
        extract_immunization_doses(
            "DTaP    04/12/2024\nRotavirus   06/2024\nMMR (not administered)\n",
            with_unresolved=True,
        )
    )
    assert any("MMR" in u["line"] for u in unresolved)


def test_the_handoff_records_are_shaped_for_the_i02_matcher(ranges):
    """The point of the link is that I-02's controls apply unchanged.

    The previous version of this test imported `reconcile`, never called it, and
    asserted only I-06's own dictionary keys -- so it passed while
    `DoseRecord(**record)` raised `TypeError`. It now actually runs the matcher.
    """
    from modules.immunization.forecast import DosePrecision
    from modules.immunization.matcher import DoseRecord, reconcile

    processed = run_one(by_name("immunization_registry_printout"), ranges=ranges)
    registry = processed.routed.immunization_handoff.as_dose_records()

    # A chart holding the same DTaP one day off the faxed date. Inside I-02's
    # tolerance, so the matcher should pair them rather than double-count.
    faxed_dtap = next(
        r for r in registry
        if r.product_text.upper().startswith("DTAP")
        and r.precision == DosePrecision.DAY
    )
    chart = [
        DoseRecord(
            record_id="chart#1",
            cvx=faxed_dtap.cvx,
            given=faxed_dtap.given + timedelta(days=1),
            source="chart",
        )
    ]

    result = reconcile(chart, registry)

    # The handoff carries nothing the matcher cannot code.
    assert result.unknown_codes == []
    # The overlapping dose paired; it is not in both leftovers.
    assert len(result.matched) == 1
    assert result.matched[0].registry.record_id == faxed_dtap.record_id
    assert faxed_dtap not in result.registry_only
    # Everything else on the printout is genuinely new to the chart.
    assert len(result.registry_only) == len(registry) - 1
    # And a month-precision dose, if the printout carried one, stays out of the
    # rule-provable pass -- I-02's own control, applying unchanged.
    for pair in result.matched:
        assert pair.registry.precision == DosePrecision.DAY


# ==========================================================================
# the pipeline
# ==========================================================================


def test_every_fixture_produces_a_routed_document(ranges, audit):
    pipeline = make_pipeline(CASES, ranges=ranges, audit=audit)
    for case in CASES:
        processed = pipeline.process(
            case.filename, document_id=case.name, age_lookup=AGE_MONTHS
        )
        assert processed.routed is not None, case.name
        assert processed.routed.queue in {
            Queue.AUTO_FILE, Queue.PHYSICIAN_REVIEW, Queue.URGENT_ALERT,
            Queue.HUMAN_INDEXING, Queue.IMMUNIZATION_RECONCILIATION,
        }


def test_every_routing_decision_is_audited(ranges, audit):
    """README I-06: "Every classification, match, and routing decision logged
    with model version and confidence." """
    pipeline = make_pipeline(CASES, ranges=ranges, audit=audit)
    for case in CASES:
        pipeline.process(case.filename, document_id=case.name, age_lookup=AGE_MONTHS)
    events = audit.query("SELECT * FROM event WHERE event_type = 'fax_routed'")
    assert len(events) == len(CASES)
    detail = json.loads(events[0]["detail_json"])
    assert "queue" in detail and "ocr_confidence" in detail
    inferences = audit.query("SELECT * FROM inference WHERE initiative_id = 'I-06'")
    assert len(inferences) == 2 * len(CASES)
    assert all(row["model_id"] for row in inferences)


def test_a_two_digit_year_on_a_fax_does_not_become_nineteen_twenty_four():
    assert parse_document_date("03/04/24") == date(2024, 3, 4)
    assert parse_document_date("11/08/2011") == date(2011, 11, 8)
    assert parse_document_date("2019-03-14") == date(2019, 3, 14)
    assert parse_document_date("not a date") is None
    assert parse_document_date(None) is None


def test_the_ranges_gate_runs_before_any_document_is_read():
    unreviewed = ReferenceRanges.load()
    case = by_name("routine_records_request")
    client = LLMClient(EchoTransport([case.classifier_response, case.urgency_response]))
    pipeline = FaxPipeline(
        engines=[ScriptedOCR(documents={case.filename: list(case.pages)})],
        classifier=DocumentClassifier(client),
        matcher=PatientMatcher(PANEL),
        triage=UrgencyTriage(client, ranges=unreviewed),
    )
    with pytest.raises(UnreviewedRanges):
        pipeline.process(case.filename, document_id="d")


# ==========================================================================
# the gateway-outage monitor
# ==========================================================================


def test_the_monitor_alerts_on_silence_during_business_hours():
    """A fax gateway that stops delivering produces no error anywhere -- it
    produces silence, and silence looks exactly like a quiet morning."""
    monitor = InboundMonitor()
    quiet = monitor.check(last_received=NOW - timedelta(hours=5), now=NOW)
    assert quiet["alert"] is True
    recent = monitor.check(last_received=NOW - timedelta(minutes=30), now=NOW)
    assert recent["alert"] is False


def test_the_monitor_is_quiet_outside_business_hours():
    monitor = InboundMonitor()
    night = NOW.replace(hour=3)
    assert monitor.check(last_received=NOW - timedelta(days=1), now=night)["alert"] is False
    sunday = NOW + timedelta(days=(6 - NOW.weekday()))
    assert monitor.check(last_received=NOW, now=sunday)["alert"] is False


def test_never_having_received_anything_is_an_alert():
    assert InboundMonitor().check(last_received=None, now=NOW)["alert"] is True


def test_mixed_timezone_awareness_is_refused_rather_than_coerced():
    """Guessing which one is UTC is how a monitor reports "no fax for six hours"
    at 09:00 on a normal morning."""
    with pytest.raises(ValueError, match="timezone-aware"):
        InboundMonitor().check(last_received=NOW.replace(tzinfo=None), now=NOW)


def test_reference_ranges_are_applied_at_the_age_of_collection(ranges, audit):
    """A newborn bilirubin faxed six weeks late is still a newborn bilirubin.
    Scoring it against the age on arrival force-flags a normal result, and an
    alert that fires on healthy babies is an alert nobody reads."""
    case = by_name("newborn_bilirubin_normal_for_age")
    assert AGE_MONTHS["p_ez"] > 1.0, "the child is over a month old on arrival"
    assert ranges.collection_date(case.text) == date(2026, 7, 2)
    processed = run_one(case, ranges=ranges, audit=audit)
    numeric = [a for a in processed.urgency.abnormal if a.source == "reference_range"]
    assert not numeric
    # ...and the same value with no collection date on it IS flagged, because
    # then the only age available is the age on arrival.
    stripped = [(p.replace("Collected: 07/02/2026 (36 hours of life)\n", ""), c)
                for p, c in case.pages]
    triage = UrgencyTriage(None, ranges=ranges)
    result = triage.triage(stripped[0][0], document_id="d", age_months=AGE_MONTHS["p_ez"])
    assert [a for a in result.abnormal if a.source == "reference_range"]


def test_a_unit_containing_the_letter_l_is_not_an_abnormal_flag(ranges):
    """FINDING from the tests: the boundary only excluded letters, so the "L" in
    "mmol/L" reported every potassium result in the practice as flagged low by
    the sending laboratory."""
    flagged = [
        a for a in ranges.scan("Potassium   4.1 mmol/L") if a.source == "document_flag"
    ]
    assert not flagged
    real = [
        a for a in ranges.scan("Potassium   2.9 mmol/L    L") if a.source == "document_flag"
    ]
    assert real and real[0].direction == "low"



@pytest.mark.parametrize(
    "written",
    ["PETROV JUNO A", "PETROV^JUNO^A", "Juno A. Petrov", "Petrov, Juno",
     "petrov, juno", "PETROV, JUNO A", "Juno Petrov"],
)
def test_the_fax_header_formats_a_practice_actually_receives(matcher, written):
    """FINDING: "LAST FIRST MI" -- the standard fax-header format -- scored 0.545
    and refused a correctly-addressed document, because the trailing middle
    initial was read as the surname."""
    result = matcher.match(name=written, dob=date(2011, 11, 8))
    assert result.outcome == MatchOutcome.MATCHED, written
    assert result.matched.patient_id == "p_juno"


def test_a_credential_token_never_deletes_a_real_surname():
    """FINDING: a flat stopword set containing "do" (for the D.O. credential)
    deleted the surname of every patient named Do -- one of the most common
    Vietnamese surnames -- so that child's documents were permanently
    unmatchable. "Pa", "Ms", "Miss", "II" and "IV" are real names too."""
    panel = [
        PanelPatient("p_do", "Minh", "Do", date(2018, 5, 4)),
        PanelPatient("p_pa", "Pa", "Vang", date(2017, 2, 9)),
    ]
    matcher = PatientMatcher(panel)
    for written in ("DO, MINH", "Minh Do", "Do, Minh"):
        assert matcher.match(name=written, dob=date(2018, 5, 4)).matched is not None, written
    assert matcher.match(name="Vang, Pa", dob=date(2017, 2, 9)).matched is not None
    assert normalise_name("Do") == "do"
    assert normalise_name("Dr. Juno Petrov MD") == "juno petrov"


# ==========================================================================
# adversarial-review regressions
#
# One test per finding from the I-06 review. Each reproduces the ORIGINAL
# defect, so the fix cannot be quietly reverted. They are grouped here rather
# than scattered because the review is the reason they exist, and a future
# reader deserves to see the whole list at once.
# ==========================================================================


class _FixedOCR:
    """An engine that returns exactly the pages it was handed."""

    def __init__(self, name, pages):
        self.name = name
        self.pages = pages

    def available(self):
        return True

    def read(self, path):
        return OCRResult(
            path,
            [Page(i + 1, t, c, engine=self.name) for i, (t, c) in enumerate(self.pages)],
            engine=self.name,
        )


def _band(ranges, analyte_name, age_months):
    analyte = next(a for a in ranges.analytes if a.name == analyte_name)
    return analyte.band_for(age_months)


def _ok_response(case):
    return json.dumps({
        "document_type": case.expected_type, "confidence": 0.94,
        "patient_name": None, "patient_dob": None, "sending_facility": None,
        "one_line_summary": "x",
    })


def EvalResultFactory(*, clinical, leaks):
    """An EvalResult with `leaks` of `clinical` outside_lab documents predicted
    as an administrative type, and the rest predicted correctly."""
    from modules.fax.classify import EvalResult

    result = EvalResult(total=clinical, correct=clinical - leaks)
    result.confusion["outside_lab"]["outside_lab"] = clinical - leaks
    result.confusion["outside_lab"]["records_request"] = leaks
    return result


def test_r01_a_reference_range_column_is_not_read_as_a_result(ranges):
    """#1 The old proximity parser took the nearest number to the analyte name,
    which on a real lab report is the low end of the printed reference range."""
    text = (
        "Hemoglobin        11.8    g/dL     Ref 11.0 - 14.0\n"
        "WBC                7.2    K/uL     Ref 5.0 - 15.5\n"
    )
    found = ranges.scan(text, age_months=60)
    assert not found, [f.excerpt for f in found]


def test_r02_a_bracketed_flag_is_read_as_a_flag(ranges):
    """#2 `(H)`, `[LL]` and `*CRIT*` are how labs actually print flags; the old
    pattern required a bare token and missed every one of them."""
    for rendering in ("(H)", "[LL]", "*CRIT*", "ABNORMAL"):
        found = ranges.scan(f"Potassium   6.9  mmol/L  {rendering}\n", age_months=60)
        assert found, rendering


def test_r03_stat_is_a_delivery_priority_not_a_finding(ranges):
    """#3 `STAT` on a cover sheet marked routine faxes urgent all morning. An
    alert that fires on everything fails the same way as no alert."""
    assert not ranges.scan("STAT FAX - please deliver to Dr Alvarez\n", age_months=60)
    assert ranges.scan("Status asthmaticus noted on arrival\n", age_months=60) == []


def test_r04_a_recorded_multiple_birth_is_binding(matcher):
    """#4 A perfect name score used to beat a recorded twin flag. It must not:
    the flag is a fact about the panel, the score is a guess about the fax."""
    result = matcher.match(name="Okafor, Wren", dob=date(2020, 9, 5))
    assert result.matched is None
    assert result.needs_human


def test_r05_an_unknown_age_narrows_the_bands_it_does_not_widen_them(ranges):
    """#5 `band_for(None)` returned the widest band, so an unmatched document
    was scored more leniently than a matched one -- missing data relaxed a
    safety threshold."""
    # There is no band for an unknown age. Manufacturing one produced first the
    # widest (which cleared a potassium of 5.8) and then the intersection of all
    # of them -- which, because the shipped bands are NOT nested, gave
    # Hemoglobin 13.5-13.5 and force-flagged 140 of 141 hemoglobin values.
    assert _band(ranges, "Hemoglobin", None) is None

    analyte = next(a for a in ranges.analytes if a.name == "Hemoglobin")
    assert len(analyte.candidate_bands(None)) == len(analyte.bands)
    assert analyte.candidate_bands([60]) == (analyte.band_for(60),)
    for band in analyte.candidate_bands(None):
        assert band.low < band.high, band

    # An ordinary hemoglobin is not force-flagged when the age is unknown...
    ordinary = [a for a in ranges.scan("Hemoglobin 12.6 g/dL\n")
                if a.source == "reference_range"]
    assert not ordinary
    # ...but it is not silently cleared either.
    assert [a for a in ranges.scan("Hemoglobin 12.6 g/dL\n")
            if a.source == "age_unknown"]


def test_r06_a_low_confidence_urgency_answer_is_not_used(ranges):
    """#6 The urgency call had no confidence floor, so a 0.05-confidence
    "routine" was honoured exactly like a 0.99 one."""
    from modules.fax.urgency import MIN_URGENCY_CONFIDENCE

    assert 0.0 < MIN_URGENCY_CONFIDENCE <= 1.0
    case = replace(
        by_name("routine_records_request"),
        name="unsure_case", model_urgency="routine", model_urgency_confidence=0.05,
    )
    processed = run_one(case, ranges=ranges)
    assert processed.urgency.urgency != "routine"


def test_r07_a_vaccine_line_that_is_not_an_administration_is_reported(ranges):
    """#7/#12/#19 The old extractor turned "MMR (not administered)" into a dose,
    and dropped anything it could not parse without saying so."""
    doses, unresolved = extract_immunization_doses(
        "MMR (not administered)\nTdap  refused by parent\nDTaP  99/99/2024\n",
        with_unresolved=True,
    )
    assert doses == []
    assert len(unresolved) == 3


def test_r08_the_handoff_emits_real_dose_records():
    """#8 `DoseRecord(**record)` raised TypeError on the dictionaries the old
    handoff produced, so the cross-initiative link did not connect."""
    from modules.fax.route import ImmunizationHandoff
    from modules.immunization.matcher import DoseRecord

    handoff = ImmunizationHandoff(
        patient_id="p", document_id="d", source="I-CARE",
        doses=extract_immunization_doses("DTaP  04/12/2024\n"),
    )
    record = handoff.as_dose_records()[0]
    assert isinstance(record, DoseRecord) and record.known and record.source == "registry"


def test_r09_an_urgent_immunization_document_still_hands_off(ranges):
    """#9 The urgent path returned before the handoff was built, so a registry
    printout that tripped an urgent rule silently lost its doses."""
    case = by_name("immunization_registry_printout")
    urgent = replace(
        case,
        name="urgent_immunization_printout",
        pages=case.pages + (("Critical value called to provider.\n", 0.95),),
        model_urgency="urgent",
        model_urgency_reason="critical value notification on the same fax",
    )
    processed = run_one(urgent, ranges=ranges)
    assert processed.routed.queue == Queue.URGENT_ALERT
    assert processed.routed.immunization_handoff is not None
    assert len(processed.routed.immunization_handoff.doses) == 6


def test_r10_every_routed_document_has_a_queue_and_a_patient_field(ranges):
    """#10 Some early-return paths left `queue` unset and `patient_id`
    unassigned, so the document existed in no queue at all."""
    for case in CASES:
        processed = run_one(case, ranges=ranges)
        routed = processed.routed
        assert routed.queue in {
            Queue.AUTO_FILE, Queue.PHYSICIAN_REVIEW, Queue.URGENT_ALERT,
            Queue.HUMAN_INDEXING, Queue.IMMUNIZATION_RECONCILIATION,
        }, case.name
        if processed.match.matched is not None:
            assert routed.patient_id == processed.match.matched.patient_id, case.name


def test_r11_engines_that_disagree_on_page_count_do_not_silently_win():
    """#11 The chain compared confidence only, so a one-page read of a
    three-page fax could beat the engine that saw all three."""
    short = _FixedOCR("short", [("only page", 0.99)])
    full = _FixedOCR("full", [("p1", 0.80), ("p2", 0.80), ("p3", 0.80)])
    result = chain([full, short], "f.tif")
    assert len(result.pages) >= 3
    assert result.engine == "full"


def test_r13_a_surname_is_not_deleted_as_a_credential(matcher):
    """#13 A flat noise-word set removed "Do" from "Do, Minh" -- the surname."""
    first, last = split_name("Do, Minh")
    assert first.lower() == "minh"
    assert last.lower() == "do"
    assert "do" in normalise_name("Do, Minh")


def test_r14_the_urgent_task_names_the_number_first(ranges):
    """#14 The alert text led with the model's sentence, so the MA read prose
    before the value that caused the alert."""
    processed = run_one(by_name("critical_lab_called_routine"), ranges=ranges)
    detail = processed.routed.tasks[0].description
    assert any(ch.isdigit() for ch in detail[:80]), detail


def test_r15a_a_tiled_eval_set_does_not_satisfy_the_case_floor():
    """#15 Ten fixtures repeated twelve times reported 120 cases and passed the
    100-case floor while measuring ten documents."""
    labelled = eval_cases(repeats=12)
    result = evaluate(
        DocumentClassifier(
            LLMClient(EchoTransport([_ok_response(c) for c in labelled]))
        ),
        labelled,
    )
    assert result.accuracy == 1.0
    assert result.total == 120
    assert result.distinct_documents == 10
    assert any("distinct document" in b for b in result.blockers())


def test_r15b_a_never_confident_classifier_does_not_score_perfectly():
    """#15 A classifier pinned below the threshold answered every label
    correctly, scored 1.000, and would have been approved to route nothing."""
    cases = [
        EvalCase(f"d{i}", f"LAB RESULT {i}", expected_type="outside_lab")
        for i in range(20)
    ]
    responses = [
        json.dumps({
            "document_type": "outside_lab", "confidence": 0.40,
            "patient_name": None, "patient_dob": None, "sending_facility": None,
            "one_line_summary": "x",
        })
        for _ in cases
    ]
    result = evaluate(
        DocumentClassifier(LLMClient(EchoTransport(responses))), cases
    )
    assert result.accuracy == 0.0
    assert result.recall("outside_lab") == 0.0
    assert any("confidence threshold" in b for b in result.blockers())


def test_r15c_the_confidence_threshold_cannot_be_lowered():
    """#15 `DocumentClassifier(min_confidence=...)` was stored and never read,
    because `Classification.confident` used the module constant."""
    with pytest.raises(ValueError, match="never lowered"):
        DocumentClassifier(LLMClient(EchoTransport([])), min_confidence=0.10)
    # Through the real call path, not a hand-built Classification: the defect
    # was that `classify()` never put the threshold on its answer.
    response = json.dumps({
        "document_type": "outside_lab", "confidence": 0.80, "patient_name": None,
        "patient_dob": None, "sending_facility": None, "one_line_summary": "x",
    })
    strict = DocumentClassifier(
        LLMClient(EchoTransport([response])), min_confidence=0.95
    )
    answer = strict.classify("LAB RESULT", document_id="d")
    assert answer.min_confidence == 0.95
    assert not answer.confident

    default = DocumentClassifier(LLMClient(EchoTransport([response])))
    assert default.classify("LAB RESULT", document_id="d").confident


def test_r15d_the_gate_pins_the_model_not_only_the_prompt():
    """#15 `require()` compared the prompt hash alone, so a validated prompt in
    front of a swapped model passed -- and the model is what was measured."""
    gate = ClassifierGate(
        provider="echo", model_id="echo-1", model_version="test",
        prompt_hash=DocumentClassifier(LLMClient(EchoTransport([]))).prompt_hash,
        validated_on="2026-08-24",
    )
    same = DocumentClassifier(LLMClient(EchoTransport([])))
    gate.require(same)  # no raise
    swapped = DocumentClassifier(
        LLMClient(EchoTransport([], model_id="other-7b", model_version="test"))
    )
    with pytest.raises(ClassifierNotValidated, match="not the validated model"):
        gate.require(swapped)


def test_r15e_the_clinical_leak_gate_is_a_budget_not_an_absolute():
    """#15 A single leak over 300 faxes blocked a classifier that was otherwise
    safe, and a gate that can never pass gets switched off."""
    result = EvalResultFactory(clinical=200, leaks=1)
    assert result.clinical_leaks()
    assert not any("predicted as administrative" in b for b in result.blockers())
    over = EvalResultFactory(clinical=200, leaks=40)
    assert any("predicted as" in b and "administrative" in b for b in over.blockers())
    assert not over.passes()


def test_r16_the_shipped_bands_are_internally_consistent(ranges):
    """#16 Hct was not ~3x Hgb, and the glucose ceiling was a FASTING ceiling
    applied to random draws, so ordinary results flagged."""
    for age in (1, 12, 60, 180):
        hgb = _band(ranges, "Hemoglobin", age)
        hct = _band(ranges, "Hematocrit", age)
        assert hct.low <= hgb.low * 3.3
        assert hct.high >= hgb.high * 2.7
    assert _band(ranges, "Glucose", 60).high >= 120


def test_r17_a_transport_failure_does_not_drop_the_document(ranges):
    """#17 An exception from the model server escaped `process()`, so the fax
    produced no ProcessedDocument at all and vanished."""
    class Exploding(EchoTransport):
        def generate(self, *a, **k):  # noqa: ANN002, ANN003
            raise ConnectionResetError("model server restarted")

    case = by_name("routine_records_request")
    pipeline = FaxPipeline(
        engines=[ScriptedOCR(documents={case.filename: list(case.pages)})],
        classifier=DocumentClassifier(LLMClient(Exploding([]))),
        matcher=PatientMatcher(PANEL),
        triage=UrgencyTriage(LLMClient(Exploding([])), ranges=ranges),
        on_duty_physician="dr_alvarez", indexer="ma_jess",
    )
    processed = pipeline.process(case.filename, document_id=case.name)
    assert processed.routed is not None
    assert processed.routed.queue in (Queue.HUMAN_INDEXING, Queue.URGENT_ALERT)
    assert processed.errors


def test_r20_a_collection_date_in_the_future_is_discarded(ranges):
    """#20 A mis-OCR'd year produced an age at collection that scored a child
    against the wrong band -- and a NEGATIVE one scored an adolescent against
    the newborn bands, where a bilirubin of 9.4 is normal."""
    assert ranges.collection_date("Collected: 04/12/2099") == date(2099, 4, 12)

    case = by_name("newborn_bilirubin_normal_for_age")
    future = replace(
        case,
        name="future_collection_date",
        # ONLY the collection line. The date of birth stays readable, so the
        # patient still matches and the collection-date logic is what is tested.
        pages=tuple(
            (text.replace("Collected: 07/02/2026", "Collected: 07/02/2099"), score)
            for text, score in case.pages
        ),
    )
    processed = run_one(future, ranges=ranges)
    assert processed.match.matched is not None
    # The date is unusable, so no age at collection is claimed and the result is
    # scored against every band -- reported as unsettled, never cleared.
    assert any("not a usable date" in w for w in processed.warnings)
    assert not [a for a in processed.urgency.abnormal if a.source == "reference_range"]
    assert [a for a in processed.urgency.abnormal if a.source == "age_unknown"]
    assert processed.routed.queue != Queue.AUTO_FILE


def test_r21_a_decimal_shift_is_flagged_in_both_directions(ranges):
    """#21 Only a 10x-high value was labelled a suspected decimal shift; a
    10x-LOW one was reported as a catastrophic result with no caveat."""
    high = ranges.scan("Hemoglobin  118  g/dL\n", age_months=60)
    low = ranges.scan("Hemoglobin  1.18  g/dL\n", age_months=60)
    assert high and high[0].decimal_shift_suspected
    assert low and low[0].decimal_shift_suspected


def test_r22_an_empty_page_scores_zero_not_one():
    """#22 A page with no text got the engine's default confidence, so a blank
    scan read as a perfectly confident empty document."""
    from modules.fax.ocr import _pages_from_scores

    pages = _pages_from_scores(["", "real"], [[0.96, 0.96], [0.9]], "t")
    assert pages[0].confidence == 0.0
    assert OCRResult("x", pages).confidence == 0.0


def test_r23_fallback_used_reports_whether_a_fallback_ran():
    """#23 The flag was set from the winning engine's name rather than from
    whether another engine had been tried."""
    good = _FixedOCR("primary", [("clean page", 0.98)])
    assert chain([good], "f.tif").fallback_used is False
    poor = _FixedOCR("primary", [("smudge", 0.30)])
    assert chain([poor, good], "f.tif").fallback_used is True


def test_r24_an_ambiguous_date_order_is_detected():
    """#24 `03/04/2024` was silently read month-first. It is 4 March in
    Springfield and 3 April in Sherbrooke, and the fax does not say which."""
    from modules.fax.pipeline import date_order_is_ambiguous, date_readings

    assert date_readings("03/04/2024") == (date(2024, 3, 4), date(2024, 4, 3))
    assert not date_order_is_ambiguous("04/22/2024")
    assert not date_order_is_ambiguous("2024-06-04")
    assert not date_order_is_ambiguous("04-Jun-2024")
    # No month-first reading exists, so the order is not in doubt.
    assert date_readings("13/04/2024") == (date(2024, 4, 13), None)


def test_r24_an_ambiguous_dob_is_resolved_by_the_panel_or_refused(ranges):
    """The panel knows something the string does not. When it does not, the
    document goes to a person rather than into the likelier chart."""
    panel = [
        PanelPatient(patient_id="p_a", first_name="Sam", last_name="Reyes",
                     dob=date(2024, 3, 4)),
        PanelPatient(patient_id="p_b", first_name="Sam", last_name="Reyes",
                     dob=date(2024, 4, 3)),
    ]
    case = by_name("routine_records_request")

    ambiguous = replace(
        case, name="ambiguous_dob",
        patient_name="Reyes, Sam", patient_dob="03/04/2024",
    )
    only_one = run_one(
        ambiguous, ranges=ranges, matcher=PatientMatcher(panel[:1]),
    )
    assert only_one.match.matched is not None
    assert any("panel settled it" in w for w in only_one.warnings)

    both = run_one(ambiguous, ranges=ranges, matcher=PatientMatcher(panel))
    assert both.match.matched is None
    assert both.match.outcome == MatchOutcome.AMBIGUOUS
    assert both.routed.queue == Queue.HUMAN_INDEXING


def test_r24_an_ambiguous_collection_date_is_scored_at_both_ages(ranges):
    """A day/month swap can move a draw by eleven months, which is several
    paediatric bands -- and the bands are NOT nested, so picking either age can
    widen a threshold. A bilirubin of 9.4 is normal at two days (ceiling 15.0)
    and alarming at thirty-two (ceiling 1.2). Score at both."""
    from modules.fax.urgency import UrgencyTriage

    triage = UrgencyTriage(None, ranges=ranges)
    text = "Total bilirubin   9.4  mg/dL\n"

    newborn = triage.triage(text, document_id="d", age_months=0.07)
    assert not newborn.abnormal, "normal for a two-day-old"

    older = triage.triage(text, document_id="d", age_months=1.05)
    assert [a for a in older.abnormal if a.source == "reference_range"]
    assert older.urgency == "urgent"

    # Both readings on the table: unsettled, and still urgent, because it is a
    # coin toss on a real child rather than a statement about all of childhood.
    both = triage.triage(
        text, document_id="d", age_months=0.07, candidate_ages=[1.05]
    )
    unsettled = [a for a in both.abnormal if a.source == "age_unknown"]
    assert unsettled and unsettled[0].analyte == "Total bilirubin"
    assert not [a for a in both.abnormal if a.source == "reference_range"]
    assert both.urgency == "urgent"
    assert any("UNSETTLED" in w for w in both.warnings)


def test_r24_the_pipeline_hands_both_ages_to_the_scan(ranges):
    """The same rule, end to end. The unit test above can pass while the
    pipeline still picks one age and hands `candidate_ages=None` to `scan`."""
    case = by_name("newborn_bilirubin_normal_for_age")   # born 2026-07-01

    # `07/02/2026` is 2 July (age 1 day) or 7 February (before the child was
    # born). One reading is impossible, so the document settles itself.
    settled_by_birth = run_one(case, ranges=ranges)
    assert any("precedes the child" in w for w in settled_by_birth.warnings)
    assert not settled_by_birth.urgency.abnormal

    # `08/07/2026` is 7 August (age 37 days) or 8 July (age 7 days), and both
    # are real dates in this child's life. The bilirubin ceiling is 15.0 below
    # one month and 1.2 above it, so a bilirubin of 9.4 is normal under one
    # reading and alarming under the other. Neither may be chosen silently.
    straddle = replace(
        case, name="straddling_collection_order",
        pages=tuple(
            (text.replace("Collected: 07/02/2026", "Collected: 08/07/2026"), score)
            for text, score in case.pages
        ),
    )
    processed = run_one(straddle, ranges=ranges)
    assert any("scored against BOTH ages" in w for w in processed.warnings)
    unsettled = [a for a in processed.urgency.abnormal if a.source == "age_unknown"]
    assert unsettled and unsettled[0].analyte == "Total bilirubin"
    assert not [a for a in processed.urgency.abnormal if a.source == "reference_range"]
    # A coin toss on a real child's bilirubin is an urgent page, not a 24-hour
    # queue -- and never a silent auto-file.
    assert processed.routed.queue == Queue.URGENT_ALERT


def test_r25_requires_immediate_needs_a_clinical_completion(ranges):
    """Found while re-running the review proofs: the bare stem "requires
    immediate" matched "Requires immediate return of the school clearance
    paperwork" and paged the on-duty physician about a sports form."""
    assert not ranges.scan(
        "Requires immediate return to school clearance paperwork.\n", age_months=60
    )
    assert ranges.scan("This requires immediate medical attention.\n", age_months=60)


@pytest.mark.parametrize("rendering", [
    "This requires immediate medical attention.",
    "Findings require immediate medical attention.",         # plural subject
    "Requires immediate hospitalization.",
    "Findings require immediate neurosurgical evaluation.",
    "This is a life threatening finding.",                   # un-hyphenated
    "This is a life-threatening finding.",
    "CRITICAL  VALUES called at 0918",
    "Critical-value notification",
    "Called and read-back verified by Dr Ruiz",
    "Blood cultures positive",
])
def test_r28_a_critical_phrase_survives_ordinary_inflection(ranges, rendering):
    """The compiler claimed a trailing "s" and a hyphen could not defeat it.
    Both could: the optional "s" was pinned to the LAST word and was additive
    only, and the hyphen rule applied only BETWEEN whitespace-separated words,
    so "life-threatening" in the config never matched "life threatening"."""
    assert ranges.scan(rendering + "\n", age_months=60), rendering


@pytest.mark.parametrize("rendering", [
    "Requires immediate return of the school clearance paperwork.",
    "Order: STAT CBC and CMP. Results below are within normal limits.",
    "Status asthmaticus noted on arrival.",
    "Portable STAT chest film obtained; normal.",
    "Statistically significant improvement noted.",
])
def test_r28_a_critical_phrase_does_not_fire_on_ordinary_paper(ranges, rendering):
    """The other half of the same rule. Widening the phrase matcher is only
    safe if it stays anchored -- an alert that fires on everything fails the
    same way as no alert."""
    assert not ranges.scan(rendering + "\n", age_months=60), rendering


def test_r26_an_alert_reason_is_a_whole_line_not_a_character_window(ranges):
    """The excerpt was a +/-30 character window around the phrase, so it
    straddled the line above and handed the MA a reference range and half a
    sentence as the reason for a one-hour page."""
    text = (
        "Creatinine   0.7   mg/dL    0.5-1.0\n"
        "\n"
        "Result phoned to ordering provider 08/22/2026 0918.\n"
    )
    phrase = [a for a in ranges.scan(text, age_months=168)
              if a.source == "critical_phrase"]
    assert phrase
    assert phrase[0].excerpt == "Result phoned to ordering provider 08/22/2026 0918."


@pytest.mark.parametrize("token,expected", [
    ("(L)", "low"), ("[LL]", "low"), ("LOW", "low"),
    ("(H)", "high"), ("HH", "high"), ("HIGH", "high"),
    ("*CRIT*", "abnormal"), ("CRITICAL", "abnormal"),
    ("ABN", "abnormal"), ("ABNORMAL", "abnormal"),
])
def test_r27_a_document_flag_never_invents_a_direction(ranges, token, expected):
    """`direction = "low" if token.startswith("L") else "high"` reported ABN and
    CRIT as HIGH, so the physician's one-hour page read "the sending lab flagged
    this line HIGH" about a ferritin of 8, which is low."""
    found = [a for a in ranges.scan(f"Ferritin   8 {token}  ng/mL\n", age_months=60)
             if a.source == "document_flag"]
    assert found, token
    assert found[0].direction == expected
    assert expected.upper() in found[0].describe()
    # And the placeholder zero never reaches the page.
    assert "flagged this line" in found[0].describe()


def test_r29_an_unknown_age_neither_clears_nor_pages(ranges):
    """The unknown-age band was `max(lows)..min(highs)`, which is a narrowing
    only if the bands are nested. They are not: Hemoglobin is 13.5-21.5 at 0-1
    month and 9.5-14.0 at 1-6, so the band came out 13.5-13.5 and force-flagged
    140 of 141 hemoglobin values -- i.e. the entire human-indexing queue."""
    from modules.fax.urgency import UrgencyTriage

    triage = UrgencyTriage(None, ranges=ranges)
    flagged = 0
    for tenth in range(60, 201):
        result = triage.triage(
            f"Hemoglobin {tenth / 10:.1f} g/dL", document_id="d", age_months=None
        )
        if any(a.source == "reference_range" for a in result.abnormal):
            flagged += 1
    # Only the values outside EVERY paediatric band, which is a small minority.
    assert flagged < 60, f"{flagged}/141 hemoglobin values force-flagged"
    assert flagged > 0, "a hemoglobin of 6.0 is low at every age"


def test_r30_an_ambiguous_dob_never_overrides_a_needs_human_outcome(ranges):
    """`hits = [r for r in (primary, secondary) if r.matched is not None]` could
    not tell "no patient has that date of birth" from "flagged twins, goes to a
    person ALWAYS" -- so the other reading looked unique and the document
    auto-filed into a cousin's chart with no human review event."""
    panel = [
        PanelPatient(patient_id="p_twin_a", first_name="Sam", last_name="Reyes",
                     dob=date(2024, 3, 4), multiple_birth=True),
        PanelPatient(patient_id="p_twin_b", first_name="Sasha", last_name="Reyes",
                     dob=date(2024, 3, 4), multiple_birth=True),
        PanelPatient(patient_id="p_cousin", first_name="Sam", last_name="Reyes",
                     dob=date(2024, 4, 3)),
    ]
    case = replace(
        by_name("routine_records_request"), name="ambiguous_dob_twins",
        patient_name="Reyes, Sam", patient_dob="03/04/2024",
    )
    processed = run_one(case, ranges=ranges, matcher=PatientMatcher(panel))
    assert processed.match.matched is None
    assert processed.match.outcome == MatchOutcome.AMBIGUOUS
    assert processed.routed.queue == Queue.HUMAN_INDEXING
    assert processed.routed.tasks


def test_r31_other_never_auto_files(ranges):
    """`classify.py` has always said "`other` is the escape hatch, and `other`
    goes to a human". The routing did not: `other` is not a CLINICAL_TYPE, so a
    confident, matched, routine `other` was filed into a chart with no task and
    no human review event -- hard constraint 4, on the one document class the
    classifier understands least."""
    case = replace(
        by_name("routine_records_request"), name="unclassifiable_attachment",
        document_type="other", summary="cover sheet with an unidentified attachment",
    )
    processed = run_one(case, ranges=ranges)
    assert processed.classification.confident
    assert processed.match.matched is not None
    assert processed.routed.queue == Queue.HUMAN_INDEXING
    assert [t.kind for t in processed.routed.tasks] == [TaskKind.INDEX_BY_HAND]
