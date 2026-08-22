"""I-01 — automated forms pipeline.

No mocks of this module's logic. `StaticChartSource`, `RecordingBackend` and
`EchoTransport` are shipped test doubles standing in for the EHR, the PDF
library and a model server; every rule under test is the real code.

The PDF tests run against a blank form this repo GENERATES at the template's own
coordinates, so a coordinate bug shows up as text read back out of the wrong box
rather than as a bookkeeping assertion that agrees with itself.
"""

from __future__ import annotations

import ast
import copy
import inspect
import json
import os
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from modules.forms import blankforms, chart, detect, fill, lifecycle, probe, review
from modules.forms.chart import (
    ChartRecord,
    ChartUnavailable,
    MissingPath,
    SourceValue,
    StaticChartSource,
)
from modules.forms.detect import (
    FormDetector,
    ProposalNotConfirmed,
    ProposedTemplate,
    TemplateProposer,
)
from modules.forms.fill import (
    FormFiller,
    HighlightMissing,
    PyMuPDFBackend,
    RecordingBackend,
)
from modules.forms.fixtures import (
    CASES,
    CLINIC_NOW,
    build_chart_source,
    by_name,
    chart_doses_for,
    registry_doses_for,
)
from modules.forms.lifecycle import (
    FormState,
    FormTracker,
    IllegalTransition,
    SignatureRequiresProfessional,
)
from modules.forms.pipeline import FormPipeline, PatientNotConfirmed
from modules.forms.probe import ProbeProgram, UnsafeProbeTarget, _is_safe
from modules.forms.reconcile import (
    ANTIGEN_FIELD_GROUPS,
    build_immunization_block,
    reconcile_for_form,
    _groups_for,
)
from modules.forms.review import (
    NotReleasable,
    ReleaseGate,
    ReviewDecision,
    build_review,
    record_review,
)
from modules.forms.templates import (
    TRANSFORMS,
    TemplateInvalid,
    TemplateStore,
    UncalibratedTemplate,
    UnknownTransform,
)
from modules.immunization.matcher import DoseRecord
from nsp_core.audit import AuditLog
from nsp_core.llm import EchoTransport, LLMClient

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

_MODULE_DIR = os.path.dirname(os.path.abspath(fill.__file__))


# ==========================================================================
# fixtures
# ==========================================================================


@pytest.fixture(scope="module")
def store():
    return TemplateStore.load()


@pytest.fixture(scope="module")
def camp(store):
    return store.for_filling("demo_camp_health_form")


@pytest.fixture(scope="module")
def blank(camp, tmp_path_factory):
    path = tmp_path_factory.mktemp("blank") / "camp.pdf"
    return blankforms.write_blank_form(camp, str(path))


@pytest.fixture
def audit(tmp_path):
    return AuditLog(tmp_path / "audit.sqlite3", hmac_key=b"test-key")


def make_pipeline(tmp_path, *, audit=None, probes=None, gate=None, unreachable=()):
    return FormPipeline(
        store=TemplateStore.load(),
        chart_source=build_chart_source(unreachable=unreachable),
        filler=FormFiller(PyMuPDFBackend(), audit=audit),
        tracker=FormTracker(audit=audit),
        gate=gate or ReleaseGate(),
        probes=probes,
        audit=audit,
    )


def run_case(name, tmp_path, blank, *, pipeline=None, audit=None, probes=None,
             gate=None, now=CLINIC_NOW):
    case = by_name(name)
    pipeline = pipeline or make_pipeline(tmp_path, audit=audit, probes=probes, gate=gate)
    request = pipeline.tracker.receive(
        request_id=f"rq_{name}", patient_id=case.patient_id,
        form_type=case.form_type, channel=case.channel, now=now,
    )
    prepared = pipeline.prepare(
        request,
        blank_pdf=blank,
        destination=str(tmp_path / f"{name}.pdf"),
        now=now,
        chart_doses=chart_doses_for(name),
        registry_doses=registry_doses_for(name),
        registry_note=case.registry_note,
    )
    return pipeline, prepared


# ==========================================================================
# structural guards
# ==========================================================================


def test_no_model_in_the_deterministic_half():
    """Only `detect.py` may call a model, and only on the unknown-layout path.

    README I-01: "roughly 70% deterministic plumbing, 30% LLM. Build the
    deterministic core first. A team that starts with the LLM will build
    something impressive in a demo and unreliable in production."
    """
    for name in (
        "templates.py", "chart.py", "fill.py", "reconcile.py",
        "review.py", "lifecycle.py", "probe.py", "pipeline.py", "blankforms.py",
    ):
        source = open(os.path.join(_MODULE_DIR, name), encoding="utf-8").read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "nsp_core.llm" not in node.module, (name, node.module)
                assert node.module not in ("openai", "anthropic"), name
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in ("openai", "anthropic"), name


def test_no_module_scope_pdf_import():
    """PyMuPDF is a lazy import, so a machine without it can still run the
    deterministic half and get a clear message rather than an ImportError."""
    for name in os.listdir(_MODULE_DIR):
        if not name.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(_MODULE_DIR, name), encoding="utf-8").read())
        for node in tree.body:  # module scope only
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in ("fitz", "pymupdf"), name
            if isinstance(node, ast.ImportFrom):
                assert node.module not in ("fitz", "pymupdf"), name


def test_no_field_name_or_coordinate_is_a_python_literal():
    """The template is data. A field name in code is a template that cannot be
    edited without a deploy, and a coordinate in code is one nobody reviews."""
    tree = ast.parse(open(os.path.join(_MODULE_DIR, "templates.py"), encoding="utf-8").read())
    literals = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    for forbidden in ("child_last_name", "imm_dtap_1", "height_in", "exam_date"):
        assert forbidden not in literals


# ==========================================================================
# templates.py
# ==========================================================================


def test_the_illinois_form_refuses_to_fill_with_placeholder_coordinates():
    """A tetanus date written 40 points too low lands in the next row of the
    grid, the form says the child had a dose they did not have, and a physician
    signs it. That is worse than not filling the form at all."""
    strict = TemplateStore.load()
    with pytest.raises(UncalibratedTemplate, match="placeholder coordinates"):
        strict.for_filling("il_certificate_of_child_health_examination")

    # The field list is still readable, because that is the part that is real.
    template = strict.get("il_certificate_of_child_health_examination")
    assert len(template.fields) > 50
    assert template.is_placeholder

    # And the opt-in is explicit, per call.
    loose = TemplateStore.load(allow_placeholder=True)
    assert loose.for_filling("il_certificate_of_child_health_examination")


def test_the_demo_form_is_calibrated_because_this_repo_draws_it(store):
    report = {row["form_type"]: row for row in store.calibration_report()}
    assert report["demo_camp_health_form"]["calibrated"] is True
    assert report["il_certificate_of_child_health_examination"]["calibrated"] is False


def test_a_signature_field_is_never_machine_writable(store):
    for template in store.templates.values():
        for spec in template.fields:
            if spec.kind == "signature":
                assert not spec.machine_writable, spec.name
                assert not spec.source, spec.name


def test_a_month_precision_dose_prints_as_a_month():
    """I-02 anchors a month-precision registry entry to the first of the month
    so date arithmetic works. Printing `06/01/2024` on a school form turns that
    convenience into a claim about the day."""
    render = TRANSFORMS["dose_date"]
    assert render({"given": date(2024, 6, 1), "precision": "month"}) == "06/2024"
    assert render({"given": date(2024, 6, 4), "precision": "day"}) == "06/04/2024"


def test_a_disputed_dose_renders_as_nothing():
    render = TRANSFORMS["dose_date"]
    assert render({"given": date(2024, 6, 4), "precision": "day", "disputed": True}) is None


def test_an_empty_list_does_not_become_a_clinical_assertion():
    """"No known allergies" is something a person says, not something an empty
    query result may write onto a state form."""
    assert TRANSFORMS["comma_list"]([]) is None
    assert TRANSFORMS["comma_list"](["penicillin"]) == "penicillin"


def test_a_half_measured_blood_pressure_is_not_rendered():
    render = TRANSFORMS["blood_pressure"]
    assert render({"systolic": 104, "diastolic": 66}) == "104/66"
    assert render({"systolic": 104}) is None
    assert render({"systolic": 104, "diastolic": None}) is None


def test_a_screening_result_comes_from_a_closed_vocabulary():
    """A raw chart string in a pass/fail box could put "20/20 OU" or "not done"
    where a school reads pass or fail."""
    assert TRANSFORMS["pass_fail"]("normal") == "Pass"
    assert TRANSFORMS["pass_fail"]("referred") == "Fail"
    assert TRANSFORMS["pass_fail"]("20/20 OU") is None
    assert TRANSFORMS["pass_fail"]("not done") is None


def test_a_template_with_a_duplicate_field_is_refused(tmp_path):
    _write_template(tmp_path, fields=[_field("a"), _field("a")])
    with pytest.raises(TemplateInvalid, match="duplicate field"):
        TemplateStore.load(tmp_path, allow_placeholder=True)


def test_a_template_naming_an_unknown_transform_is_refused(tmp_path):
    _write_template(tmp_path, fields=[_field("a", transform="does_not_exist")])
    with pytest.raises(UnknownTransform, match="does not have"):
        TemplateStore.load(tmp_path, allow_placeholder=True)


def test_a_template_with_a_zero_size_box_is_refused(tmp_path):
    _write_template(tmp_path, fields=[_field("a", width=0)])
    with pytest.raises(TemplateInvalid, match="zero or negative size"):
        TemplateStore.load(tmp_path, allow_placeholder=True)


def test_a_template_field_on_a_page_the_form_does_not_have_is_refused(tmp_path):
    _write_template(tmp_path, page_count=1, fields=[_field("a", page=3)])
    with pytest.raises(TemplateInvalid, match="page 3 of a 1-page"):
        TemplateStore.load(tmp_path, allow_placeholder=True)


def _field(name, *, page=1, width=100.0, transform="verbatim"):
    return {
        "name": name, "kind": "text", "transform": transform,
        "box": {"page": page, "x": 10, "y": 10, "width": width, "height": 12},
    }


def _write_template(directory, *, page_count=1, fields):
    import yaml

    path = os.path.join(str(directory), "t.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {"form_type": "t", "page_count": page_count, "fields": fields}, handle
        )
    return path


# ==========================================================================
# chart.py
# ==========================================================================


def test_a_template_typo_raises_rather_than_leaving_a_box_blank():
    """A blank box looks like a patient with no data. A raised exception looks
    like the configuration bug it is."""
    record = ChartRecord("p", {"vitals": {"height_in": SourceValue(48.0, "ehr")}})
    assert record.resolve("vitals.height_in").value == 48.0
    assert record.resolve("vitals.weight_lb") is None      # container exists, empty
    with pytest.raises(MissingPath):
        record.resolve("vitalz.height_in")                 # container does not


def test_a_dose_that_has_not_been_given_yet_is_not_an_error():
    record = ChartRecord("p", {"immunizations": {"dtap": [SourceValue(1, "ehr")]}})
    assert record.resolve("immunizations.dtap.0") is not None
    assert record.resolve("immunizations.dtap.4") is None


def test_an_undated_value_counts_as_stale():
    """"Cannot be shown to be recent" is the thing the reviewer needs told."""
    undated = SourceValue(48.0, "ehr")
    assert undated.is_stale(date(2026, 8, 24))
    fresh = SourceValue(48.0, "ehr", recorded=date(2026, 8, 1))
    assert not fresh.is_stale(date(2026, 8, 24))


def test_a_historical_event_never_goes_stale():
    """An immunization given in 2018 is not a stale height. Without this the
    staleness report fired on every dose on every form, which is the shape of
    alert that trains a reviewer to skip the list."""
    dose = SourceValue({"given": date(2018, 5, 16)}, "reconciled",
                       recorded=date(2018, 5, 16), historical=True)
    assert not dose.is_stale(date(2026, 8, 24))


def test_an_unreachable_chart_is_not_a_patient_with_no_allergies():
    source = StaticChartSource(records={}, unreachable=frozenset({"p_rosa"}))
    with pytest.raises(ChartUnavailable):
        source.fetch("p_rosa", as_of=date(2026, 8, 24))


# ==========================================================================
# reconcile.py — the I-02 reuse
# ==========================================================================


def test_the_grid_groups_name_antigens_i02_actually_has():
    from modules.immunization import cvx as cvxlib

    known = set(cvxlib.Antigen.ALL)
    for group, members in ANTIGEN_FIELD_GROUPS.items():
        assert members <= known, (group, members - known)


def test_a_combination_product_lands_in_every_row_it_belongs_to():
    """Pentacel is DTaP AND polio AND Hib, and a school form has a box for each.
    All-of membership put it in none of them and lost three doses off the grid."""
    assert _groups_for("120") == ("dtap", "hib", "polio")
    assert _groups_for("94") == ("mmr", "varicella")
    assert _groups_for("20") == ("dtap",)
    assert _groups_for("115") == ("tdap",)
    assert _groups_for("9999") == ()


def test_a_disputed_antigen_prints_nothing_at_all(tmp_path, blank):
    """README I-01: a missed dose not reconciled means an unnecessary duplicate
    vaccine or a wrong non-compliance flag. Both come from printing a number the
    sources disagree about."""
    _, prepared = run_case("theo_disputed", tmp_path, blank)
    assert "dtap" in prepared.reconciliation.disputed_groups
    written = {w.field_name for w in prepared.filled.writes}
    assert not [f for f in written if f.startswith("imm_dtap_")]
    assert any(
        b["kind"] == "unsettled_antigen" for b in prepared.review.blockers
    )


def test_an_unknown_code_holds_every_row_but_reports_once(tmp_path, blank):
    """A dose of unknown antigen could belong to any row, so no row is safe --
    but ten copies of one blocker bury the one that matters."""
    _, prepared = run_case("lucas_unknown_code", tmp_path, blank)
    assert len(prepared.reconciliation.disputed_groups) == len(ANTIGEN_FIELD_GROUPS)
    kinds = [b["kind"] for b in prepared.review.blockers]
    assert kinds.count("unknown_cvx") == 1
    assert "unsettled_antigen" not in kinds
    assert not [w for w in prepared.filled.writes if w.field_name.startswith("imm_")]


def test_the_registry_being_down_is_a_state_not_an_error(tmp_path, blank):
    """README I-01: degrade gracefully to chart-only fill, flag the form as
    'registry not reconciled', continue to function."""
    _, prepared = run_case("nadia_registry_down", tmp_path, blank)
    assert prepared.filled is not None                       # it continued
    assert prepared.filled.auto_filled                       # and filled things
    assert not prepared.reconciliation.registry_consulted
    assert not prepared.releasable                           # but does not release
    assert [b["kind"] for b in prepared.review.blockers] == ["registry_not_reconciled"]


def test_a_registry_only_dose_is_printed_and_flagged_for_the_chart():
    """The registry holds a dose the chart does not. It goes on the form -- the
    child had it -- and the practice is told to file it."""
    chart_doses = [DoseRecord("c1", "20", date(2020, 4, 1), "chart")]
    registry = [
        DoseRecord("r1", "20", date(2020, 4, 1), "registry"),
        DoseRecord("r2", "03", date(2021, 4, 5), "registry"),
    ]
    result = reconcile_for_form(chart_doses, registry)
    assert not result.disputed_groups
    advisory = [d for d in result.discrepancies() if d["kind"] == "registry_only_dose"]
    assert advisory and advisory[0]["group"] == "mmr"
    block = build_immunization_block(result)
    assert block["mmr"][0].value["given"] == date(2021, 4, 5)


def test_a_partial_date_against_a_full_one_is_unsettled_and_shown(tmp_path):
    """README A.2 names this case: "a partial or approximate date". I-02's
    rule-provable pass requires day precision on both sides, so the pair reaches
    adjudication -- and BOTH sides go on the grid, marked, because a disputed
    row the MA cannot see the doses in tells them nothing."""
    result = reconcile_for_form(
        [DoseRecord("c1", "20", date(2020, 4, 1), "chart", precision="month")],
        [DoseRecord("r1", "20", date(2020, 4, 1), "registry", precision="day")],
    )
    assert result.disputed_groups == ("dtap",)
    doses = result.antigens["dtap"].doses
    assert len(doses) == 2
    assert {d.system for d in doses} == {"chart", "registry"}
    assert all(d.disputed for d in doses)
    # ...and nothing renders from a disputed dose.
    assert all(TRANSFORMS["dose_date"](d.as_source_value().value) is None for d in doses)


def test_a_matched_pair_is_placed_once_from_the_chart():
    result = reconcile_for_form(
        [DoseRecord("c1", "20", date(2020, 4, 1), "chart")],
        [DoseRecord("r1", "20", date(2020, 4, 3), "registry")],
    )
    doses = result.antigens["dtap"].doses
    assert len(doses) == 1
    assert doses[0].system == "both"
    assert doses[0].given == date(2020, 4, 1)


# ==========================================================================
# fill.py
# ==========================================================================


def test_every_written_field_is_highlighted(tmp_path, blank, camp):
    """README I-01 step 6, and the reason it is a control rather than a feature:
    machine-written text a reviewer cannot distinguish from a clinician's is not
    reviewed, it is read."""
    _, prepared = run_case("rosa_clean", tmp_path, blank)
    assert prepared.filled.writes
    assert {w.field_name for w in prepared.filled.writes} == set(prepared.filled.highlights)


def test_a_form_with_an_unhighlighted_write_cannot_be_produced():
    """`verify()` runs inside `fill()`, so no caller can obtain one."""
    filled = fill.FilledForm(form_type="t", patient_id="p", template_version="1")
    filled.writes.append(
        fill.FieldWrite("a", "x", fill.BoundingBox(1, 0, 0, 10, 10), "auto")
    )
    with pytest.raises(HighlightMissing, match="must carry a highlight"):
        filled.verify()


def test_there_is_no_way_to_turn_the_highlight_off():
    signature = inspect.signature(FormFiller.fill)
    assert "highlight" not in signature.parameters
    source = inspect.getsource(FormFiller.fill)
    # The highlight is emitted in the same loop as the write, not in a second
    # pass -- a second pass is a thing that can be skipped or made conditional.
    assert 0 < source.index("backend.highlight(") - source.index(
        "backend.write_text("
    ) < 400


def test_a_value_lands_in_the_box_the_template_names(tmp_path, blank, camp):
    """Read back out of the PDF, box by box. A coordinate bug shows up here as
    text in the wrong box rather than as bookkeeping agreeing with itself."""
    import fitz

    _, prepared = run_case("rosa_clean", tmp_path, blank)
    document = fitz.open(prepared.filled.output_path)
    for write in prepared.filled.writes:
        page = document[write.box.page - 1]
        found = page.get_textbox(fitz.Rect(*write.box.rect)).strip()
        assert write.text in found, (write.field_name, found)
    document.close()


def test_the_highlight_is_a_real_pdf_annotation(tmp_path, blank):
    import fitz

    _, prepared = run_case("rosa_clean", tmp_path, blank)
    document = fitz.open(prepared.filled.output_path)
    kinds = [a.type[1] for page in document for a in (page.annots() or [])]
    assert len(kinds) == len(prepared.filled.writes)
    assert set(kinds) == {"Highlight"}
    document.close()


def test_nothing_is_ever_written_into_a_signature_field(tmp_path, blank, camp):
    _, prepared = run_case("rosa_clean", tmp_path, blank)
    written = {w.field_name for w in prepared.filled.writes}
    for spec in camp.fields:
        if spec.kind == "signature":
            assert spec.name not in written
            assert any(s.field_name == spec.name for s in prepared.filled.skipped)


def test_a_signature_field_is_not_counted_as_a_missing_required_field(tmp_path, blank):
    """It is required and empty on every machine-filled form; that is correct,
    and listing it would train the reviewer to ignore the list."""
    _, prepared = run_case("rosa_clean", tmp_path, blank)
    assert not prepared.filled.missing_required()
    assert any(
        s.field_name == "provider_signature" and s.human_only
        for s in prepared.filled.skipped
    )


def test_an_overflowing_value_is_truncated_on_the_page_and_reported_in_full(
    tmp_path, blank
):
    """A clipped allergy list reads as complete, and the reviewer sees the
    clipped version."""
    _, prepared = run_case("ivy_long_allergies", tmp_path, blank)
    truncated = prepared.filled.truncated
    assert truncated and truncated[0].field_name == "allergies"
    assert truncated[0].text != truncated[0].truncated_from
    assert "shellfish" in truncated[0].truncated_from
    assert "shellfish" not in truncated[0].text
    assert any(b["kind"] == "value_truncated" for b in prepared.review.blockers)


def test_a_field_write_carries_its_source_path(tmp_path, blank):
    """Matching a stale chart value to the box it filled by recorded DATE hit
    the wrong field constantly -- a whole visit's vitals share one date."""
    _, prepared = run_case("rosa_clean", tmp_path, blank)
    by_field = {w.field_name: w for w in prepared.filled.writes}
    assert by_field["height_in"].source_path == "vitals.height_in"
    assert by_field["weight_lb"].source_path == "vitals.weight_lb"


def test_the_recording_backend_exercises_the_same_fill_path(tmp_path, camp):
    """The shipped test double stands in for the PDF library only."""
    case = by_name("rosa_clean")
    record = copy.deepcopy(case.record)
    result = reconcile_for_form(list(case.chart_doses), list(case.registry_doses))
    record.data["immunizations"] = build_immunization_block(result)
    backend = RecordingBackend()
    filled = FormFiller(backend).fill(
        camp, record, blank_pdf="unused.pdf",
        destination=str(tmp_path / "out.pdf"), now=CLINIC_NOW,
    )
    assert backend.opened == "unused.pdf"
    assert backend.saved == str(tmp_path / "out.pdf")
    assert len(backend.highlighted) == len(backend.texts) == len(filled.writes)


def test_the_fill_is_audited(tmp_path, blank, audit):
    _, prepared = run_case("rosa_clean", tmp_path, blank, audit=audit)
    rows = audit.query(
        "SELECT * FROM event WHERE event_type = ? AND initiative_id = ?",
        ("form_rendered", "I-01"),
    )
    assert len(rows) == 1
    assert json.loads(rows[0]["detail_json"])["auto_filled"] == len(
        prepared.filled.auto_filled
    )


# ==========================================================================
# detect.py
# ==========================================================================


def test_a_known_form_is_classified_without_a_model(store):
    detector = FormDetector(TemplateStore.load(allow_placeholder=True))
    result = detector.detect(
        "State of Illinois\nCertificate of Child Health Examination\n"
        "To be completed by health care provider\nImmunization record\n"
        "Health history\nChild's name\nDiabetes screening\n"
        "Lead risk questionnaire\nDental examination"
    )
    assert result.form_type == "il_certificate_of_child_health_examination"
    assert result.score == 1.0


def test_an_unseen_layout_is_reported_as_unknown_not_guessed(store):
    detector = FormDetector(TemplateStore.load(allow_placeholder=True))
    result = detector.detect("Northbrook Soccer Club - Player Medical Clearance")
    assert result.form_type is None
    assert "unseen layout" in result.reason


def test_two_templates_too_close_together_go_to_a_person(tmp_path):
    """Filling from the wrong coordinate map puts every value in the wrong box."""
    import yaml

    for name, anchors in (("a", ["shared one", "shared two", "only a"]),
                          ("b", ["shared one", "shared two", "only b"])):
        with open(os.path.join(tmp_path, f"{name}.yaml"), "w", encoding="utf-8") as h:
            yaml.safe_dump(
                {"form_type": name, "page_count": 1, "anchors": anchors,
                 "fields": [_field("x")]}, h,
            )
    detector = FormDetector(TemplateStore.load(tmp_path, allow_placeholder=True))
    result = detector.detect("shared one shared two")
    assert result.form_type is None
    assert "too close to tell apart" in result.reason


def test_a_proposed_template_cannot_add_itself_to_the_library():
    """README I-01 step 2: a new template is proposed for HUMAN confirmation."""
    assert not hasattr(ProposedTemplate, "add")
    # No attribute on the class is a store, and the only public method is the
    # one that demands a person.
    proposal = _proposal()
    assert not any(
        type(value).__name__ == "TemplateStore" for value in vars(proposal).values()
    )
    public = {
        name for name in dir(proposal)
        if not name.startswith("_") and callable(getattr(proposal, name))
    }
    assert public == {"confirm", "as_dict"}
    assert "confirmed_by" in inspect.signature(ProposedTemplate.confirm).parameters


def test_a_proposal_needs_a_named_person_and_anchors():
    proposal = _proposal()
    with pytest.raises(ProposalNotConfirmed, match="a person"):
        proposal.confirm(confirmed_by="  ", anchors=["x"])
    with pytest.raises(ProposalNotConfirmed, match="anchor phrases"):
        proposal.confirm(confirmed_by="ma_jess", anchors=[])


def test_a_confirmed_proposal_is_still_uncalibrated():
    """Confirming the field list is not the same act as measuring the boxes."""
    template = _proposal().confirm(confirmed_by="ma_jess", anchors=["clearance"])
    assert template.is_placeholder
    store = TemplateStore.load()
    store.add(template, confirmed_by="ma_jess")
    with pytest.raises(UncalibratedTemplate):
        store.for_filling(template.form_type)


def test_the_library_refuses_an_anonymous_addition():
    store = TemplateStore.load()
    template = _proposal().confirm(confirmed_by="ma_jess", anchors=["clearance"])
    with pytest.raises(ValueError, match="a person confirms it"):
        store.add(template, confirmed_by="")


def test_a_proposed_immunization_box_is_not_bound_to_a_dose():
    """A model can see that a box wants a tetanus date. It cannot know WHICH
    dose of which antigen, and guessing binds the wrong dose on every form."""
    proposal = _proposal()
    grid = [f for f in proposal.proposed_fields if f.kind == "grid_date"]
    assert grid and not grid[0].source
    assert grid[0].human_only
    assert any("bind" in note for note in proposal.needs_attention)


def test_a_field_the_model_placed_off_the_form_is_dropped_not_clamped():
    """A clamped coordinate is a wrong coordinate that looks right."""
    proposal = _proposal()
    names = {f.name for f in proposal.proposed_fields}
    assert not any(n.startswith("provider_signature") for n in names)
    assert any("dropped" in note for note in proposal.needs_attention)


def test_a_schema_violation_returns_none_rather_than_stopping_the_batch():
    proposer = TemplateProposer(LLMClient(EchoTransport(["not json at all"])))
    assert proposer.propose("x", document_id="d", form_type="t") is None


def _proposal():
    payload = json.dumps({
        "form_title": "Player Medical Clearance",
        "issuer": "Northbrook Soccer Club",
        "page_count": 1,
        "fields": [
            {"label": "Player name", "semantic": "patient_name", "page": 1,
             "x": 150, "y": 100, "width": 200, "height": 14},
            {"label": "Tetanus booster date", "semantic": "immunization_date",
             "page": 1, "x": 150, "y": 148, "width": 100, "height": 14},
            {"label": "???", "semantic": "unknown", "page": 1,
             "x": 150, "y": 172, "width": 100, "height": 14},
            {"label": "Coach signature", "semantic": "provider_signature",
             "page": 2, "x": 150, "y": 196, "width": 200, "height": 20},
        ],
        "notes": "",
    })
    return TemplateProposer(LLMClient(EchoTransport([payload]))).propose(
        "x", document_id="d1", form_type="northbrook_soccer_clearance"
    )


# ==========================================================================
# review.py
# ==========================================================================


def test_a_clean_form_releases(tmp_path, blank):
    _, prepared = run_case("rosa_clean", tmp_path, blank)
    assert prepared.releasable
    assert not prepared.review.blockers


def test_the_gate_refuses_an_approval_however_many_times_it_is_clicked(
    tmp_path, blank, audit
):
    """A review screen is not a control. A screen plus a gate is."""
    _, prepared = run_case("theo_disputed", tmp_path, blank, audit=audit)
    decision = ReviewDecision(reviewer_id="ma_jess", action="accepted")
    for _ in range(3):
        with pytest.raises(NotReleasable, match="blocker"):
            record_review(prepared.review, decision, audit=audit)


def test_a_rejection_is_always_recordable(tmp_path, blank, audit):
    """The gate blocks release, not documentation. An MA must always be able to
    record that they looked and sent it back."""
    _, prepared = run_case("theo_disputed", tmp_path, blank, audit=audit)
    record_review(
        prepared.review,
        ReviewDecision(reviewer_id="ma_jess", action="rejected"),
        audit=audit,
    )


def test_a_review_names_the_reviewer(tmp_path, blank, audit):
    _, prepared = run_case("rosa_clean", tmp_path, blank, audit=audit)
    with pytest.raises(ValueError, match="names the person"):
        record_review(
            prepared.review, ReviewDecision(reviewer_id="", action="accepted"),
            audit=audit,
        )


def test_a_stale_vital_blocks_release(tmp_path, blank):
    """A form filled from a four-year-old height looks current and is not."""
    _, prepared = run_case("omar_stale_vitals", tmp_path, blank)
    stale = [b for b in prepared.review.blockers if b["kind"] == "stale_value"]
    assert stale
    assert {"vitals.height_in", "vitals.weight_lb"} <= {
        b["detail"].split(" was recorded")[0] for b in stale
    }


def test_a_stale_value_that_reached_no_box_is_not_reported(tmp_path, blank):
    """A stale field nothing printed is not this form's problem, and listing it
    trains the reviewer to skim."""
    _, prepared = run_case("rosa_clean", tmp_path, blank)
    # The fixture's lead level is 400 days old, and the camp form has no lead box.

    assert any(
        row["path"] == "labs.lead_ug_dl"
        for row in prepared.record.stale_values(CLINIC_NOW.date())
    )
    assert not [b for b in prepared.review.blockers if "lead" in b["detail"]]


def test_discrepancies_come_first_and_blocking_ones_come_first_of_all(
    tmp_path, blank
):
    _, prepared = run_case("lucas_unknown_code", tmp_path, blank)
    severities = [d["severity"] for d in prepared.review.discrepancies]
    assert severities == sorted(severities, key=lambda s: 0 if s == "blocking" else 1)


def test_the_provenance_pane_reads_in_the_order_of_the_page(tmp_path, blank):
    _, prepared = run_case("rosa_clean", tmp_path, blank)
    keys = [(r.page, r.y) for r in prepared.review.provenance]
    assert keys == sorted(keys)


def test_every_provenance_row_says_where_the_value_came_from(tmp_path, blank):
    _, prepared = run_case("rosa_clean", tmp_path, blank)
    for row in prepared.review.provenance:
        if row.method != "auto":
            continue
        assert row.system
        assert row.source_path
        assert row.recorded is not None


def test_the_edit_distance_is_computed_by_the_audit_log(tmp_path, blank, audit):
    _, prepared = run_case("rosa_clean", tmp_path, blank, audit=audit)
    record_review(
        prepared.review,
        ReviewDecision(
            reviewer_id="ma_jess", action="edited",
            corrections={"height_in": "49.0"}, review_seconds=88.0,
        ),
        audit=audit,
    )
    findings = audit.rubber_stamp_report(initiative_id="I-01", min_reviews=1)
    assert findings and findings[0].reviewer_id == "ma_jess"
    assert findings[0].reviews == 1


# ==========================================================================
# lifecycle.py
# ==========================================================================


def test_a_form_cannot_be_delivered_before_it_is_signed():
    tracker = FormTracker()
    request = tracker.receive(
        request_id="r1", patient_id="p", form_type="t", channel="fax",
        now=CLINIC_NOW,
    )
    tracker.advance(request, FormState.FILLED, actor="sys", now=CLINIC_NOW)
    tracker.advance(request, FormState.MA_REVIEW, actor="ma", now=CLINIC_NOW)
    with pytest.raises(IllegalTransition, match="cannot move to 'delivered'"):
        tracker.advance(
            request, FormState.DELIVERED, actor="fd", now=CLINIC_NOW,
            delivered_to="school",
        )


def test_signing_names_the_licensed_professional():
    tracker = FormTracker()
    request = _to_signature(tracker)
    with pytest.raises(SignatureRequiresProfessional, match="names the licensed"):
        tracker.advance(request, FormState.SIGNED, actor="sys", now=CLINIC_NOW)
    tracker.advance(
        request, FormState.SIGNED, actor="dr_alvarez", now=CLINIC_NOW,
        signed_by="dr_alvarez",
    )
    assert request.signed_by == "dr_alvarez"


def test_a_form_with_blockers_does_not_reach_the_signature_queue():
    """The tracker holds the evidence rather than accepting it.

    `blockers=` used to be an optional keyword defaulting to `()`, so omitting
    it was indistinguishable from a clean form -- any caller could sign a form
    with an unsettled immunization on it, and the ledger recorded a clean
    history. The blockers now live on the request, put there by the pipeline
    when it produced the review.
    """
    tracker = FormTracker()
    request = tracker.receive(
        request_id="r1", patient_id="p", form_type="t", channel="fax", now=CLINIC_NOW
    )
    tracker.advance(request, FormState.FILLED, actor="sys", now=CLINIC_NOW)
    tracker.advance(request, FormState.MA_REVIEW, actor="ma", now=CLINIC_NOW)

    # No review recorded at all: refused, rather than treated as clean.
    with pytest.raises(IllegalTransition, match="no review has been recorded"):
        tracker.advance(
            request, FormState.PHYSICIAN_SIGNATURE, actor="ma", now=CLINIC_NOW
        )

    # A review with blockers: refused even when the caller says nothing.
    tracker.record_review_outcome(
        request, fill_id="f1", blockers=[{"kind": "stale_value"}]
    )
    with pytest.raises(IllegalTransition, match="outstanding blocker"):
        tracker.advance(
            request, FormState.PHYSICIAN_SIGNATURE, actor="ma", now=CLINIC_NOW
        )

    # And a clean review passes.
    tracker.record_review_outcome(request, fill_id="f1", blockers=[])
    tracker.advance(
        request, FormState.PHYSICIAN_SIGNATURE, actor="ma", now=CLINIC_NOW
    )
    assert request.state == FormState.PHYSICIAN_SIGNATURE


def test_a_blocked_form_goes_back_to_be_re_filled_not_straight_to_signature():
    """The fix has to produce a new document."""
    assert FormState.PHYSICIAN_SIGNATURE not in lifecycle.TRANSITIONS[FormState.BLOCKED]
    assert FormState.FILLED in lifecycle.TRANSITIONS[FormState.BLOCKED]


def test_every_transition_names_who_made_it():
    tracker = FormTracker()
    request = tracker.receive(
        request_id="r1", patient_id="p", form_type="t", channel="fax", now=CLINIC_NOW
    )
    with pytest.raises(IllegalTransition, match="names who made it"):
        tracker.advance(request, FormState.FILLED, actor="  ", now=CLINIC_NOW)


def test_delivery_records_where_the_form_went():
    tracker = FormTracker()
    request = _to_signature(tracker)
    tracker.advance(
        request, FormState.SIGNED, actor="dr", now=CLINIC_NOW, signed_by="dr"
    )
    with pytest.raises(IllegalTransition, match="where the form went"):
        tracker.advance(request, FormState.DELIVERED, actor="fd", now=CLINIC_NOW)


def test_the_tracker_answers_where_is_my_form():
    tracker = FormTracker()
    for index in range(3):
        request = tracker.receive(
            request_id=f"r{index}", patient_id=f"p{index}", form_type="t",
            channel="fax", now=CLINIC_NOW,
        )
        tracker.advance(request, FormState.FILLED, actor="sys", now=CLINIC_NOW)
        tracker.advance(request, FormState.MA_REVIEW, actor="ma", now=CLINIC_NOW)
    assert tracker.stalled_at(CLINIC_NOW) == {"ma_review": 3}
    late = tracker.overdue(CLINIC_NOW + timedelta(hours=30))
    assert len(late) == 3
    assert all(row["state"] == "ma_review" for row in late)


def test_turnaround_measures_receipt_to_signature():
    tracker = FormTracker()
    request = _to_signature(tracker)
    tracker.advance(
        request, FormState.SIGNED, actor="dr",
        now=CLINIC_NOW + timedelta(hours=5), signed_by="dr",
    )
    report = tracker.turnaround_report()
    assert report["signed_forms"] == 1
    assert report["median_hours"] == 5.0
    assert report["over_24h"] == 0


def _to_signature(tracker, *, blockers=()):
    request = tracker.receive(
        request_id="r1", patient_id="p", form_type="t", channel="fax", now=CLINIC_NOW
    )
    tracker.advance(request, FormState.FILLED, actor="sys", now=CLINIC_NOW)
    tracker.advance(request, FormState.MA_REVIEW, actor="ma", now=CLINIC_NOW)
    tracker.record_review_outcome(request, fill_id="f1", blockers=blockers)
    tracker.advance(
        request, FormState.PHYSICIAN_SIGNATURE, actor="ma", now=CLINIC_NOW
    )
    return request


# ==========================================================================
# probe.py — the automation-complacency control
# ==========================================================================


def test_a_probe_never_touches_a_clinical_field():
    """A "test" that puts a wrong tetanus date on a form has placed a real risk
    to catch a hypothetical one."""
    for unsafe in (
        "imm_dtap_1", "imm_mmr_2", "allergies", "current_medications",
        "chronic_conditions", "lead_level", "activity_restrictions",
        "provider_signature", "vision_result", "hearing_result",
    ):
        assert not _is_safe(unsafe), unsafe
    for safe in ("height_in", "weight_lb", "bmi", "blood_pressure", "exam_date"):
        assert _is_safe(safe), safe


def test_an_unknown_field_name_is_not_probeable():
    """An allowlist, because the blocklist version fails the moment somebody
    adds a field nobody thought about."""
    assert not _is_safe("some_new_field_nobody_thought_about")


def test_the_probe_rate_is_deterministic_per_request():
    program = ProbeProgram(rate=50)
    first = [program.should_probe(f"rq_{i}") for i in range(400)]
    second = [program.should_probe(f"rq_{i}") for i in range(400)]
    assert first == second
    hits = sum(first)
    assert 2 <= hits <= 16, hits          # around 1 in 50 over 400


def test_a_probe_rate_that_probes_everything_is_refused():
    with pytest.raises(ValueError, match="probes every form"):
        ProbeProgram(rate=1)


def test_a_probed_form_cannot_be_released(tmp_path, blank, audit):
    """The one otherwise-clean form in the fixture set, probed.

    The salt is pinned rather than defaulted so this test always exercises the
    path. `should_probe` is deterministic per request id, so a default salt
    would make this test skip or run depending on a hash -- and a test that
    sometimes skips is a test that sometimes is not run.
    """
    probes = ProbeProgram(audit=audit, rate=2, salt="probe-salt-0")
    assert probes.should_probe("rq_rosa_clean")
    pipeline = make_pipeline(tmp_path, audit=audit, probes=probes)
    _, prepared = run_case(
        "rosa_clean", tmp_path, blank, pipeline=pipeline, audit=audit
    )
    assert prepared.probe_field
    assert any(
        b["kind"] == "synthetic_probe_outstanding" for b in prepared.review.blockers
    )
    with pytest.raises(NotReleasable):
        record_review(
            prepared.review,
            ReviewDecision(reviewer_id="ma_jess", action="accepted"),
            audit=audit,
        )


def test_a_caught_probe_is_the_reviewer_correcting_that_field(tmp_path, blank, audit):
    probes = ProbeProgram(audit=audit, rate=2)
    pipeline = make_pipeline(tmp_path, audit=audit, probes=probes)
    _, prepared = run_case(
        "theo_disputed", tmp_path, blank, pipeline=pipeline, audit=audit
    )
    assert prepared.probe_field, "theo is probed at rate 2"
    injected = probes.probes["rq_theo_disputed"]

    caught = probes.resolve(
        "rq_theo_disputed", prepared.review,
        ReviewDecision(
            reviewer_id="ma_jess", action="edited",
            corrections={injected.field_name: injected.original},
        ),
        now=CLINIC_NOW,
    )
    assert caught.caught is True
    assert audit.probe_catch_rate(initiative_id="I-01")["caught"] == 1


def test_editing_an_unrelated_field_is_not_catching_the_probe(tmp_path, blank, audit):
    """Not "made any edit". A reviewer who edits another box has not found this
    one, and a reviewer who rejects everything catches every probe and reviews
    nothing."""
    probes = ProbeProgram(audit=audit, rate=2)
    pipeline = make_pipeline(tmp_path, audit=audit, probes=probes)
    _, prepared = run_case(
        "theo_disputed", tmp_path, blank, pipeline=pipeline, audit=audit
    )
    injected = probes.probes["rq_theo_disputed"]
    other = next(
        r.field_name for r in prepared.review.provenance
        if r.field_name != injected.field_name
    )
    result = probes.resolve(
        "rq_theo_disputed", prepared.review,
        ReviewDecision(
            reviewer_id="ma_jess", action="edited", corrections={other: "x"}
        ),
        now=CLINIC_NOW,
    )
    assert result.caught is False

    blanket = ProbeProgram(rate=2)
    blanket.probes["r"] = probe.Probe(
        "r", "height_in", "48.5", "45.8", "transposed", CLINIC_NOW
    )
    scored = blanket.resolve(
        "r", prepared.review,
        ReviewDecision(reviewer_id="ma_jess", action="rejected"),
        now=CLINIC_NOW,
    )
    assert scored.caught is False


def test_a_probe_nobody_was_shown_is_withdrawn_not_scored(tmp_path, blank, audit):
    """A probed form blocked for an unrelated reason goes back to be re-filled.
    Carrying the probe forward blocked that request at the gate forever and
    computed the catch rate over a shrinking, biased subset."""
    probes = ProbeProgram(audit=audit, rate=2)
    pipeline = make_pipeline(tmp_path, audit=audit, probes=probes)
    _, first = run_case(
        "theo_disputed", tmp_path, blank, pipeline=pipeline, audit=audit
    )
    assert first.probe_field
    request = first.request
    pipeline.tracker.advance(
        request, FormState.BLOCKED, actor="ma_jess", now=CLINIC_NOW
    )

    second = pipeline.prepare(
        request, blank_pdf=blank, destination=str(tmp_path / "again.pdf"),
        now=CLINIC_NOW + timedelta(hours=1),
        chart_doses=chart_doses_for("theo_disputed"),
        registry_doses=registry_doses_for("theo_disputed"),
    )
    rates = probes.catch_rate()
    assert rates["withdrawn"] == 1
    assert rates["scored"] == 0
    assert rates["catch_rate"] is None
    assert not [
        b for b in second.review.blockers
        if b["kind"] == "synthetic_probe_outstanding" and not second.probe_field
    ]


def test_probe_reviews_do_not_pollute_the_real_edit_rate(tmp_path, blank, audit):
    """An injected error corrected by the reviewer is not evidence about how
    good the machine's drafts are."""
    probes = ProbeProgram(audit=audit, rate=2)
    pipeline = make_pipeline(tmp_path, audit=audit, probes=probes)
    _, prepared = run_case(
        "theo_disputed", tmp_path, blank, pipeline=pipeline, audit=audit
    )
    injected = probes.probes["rq_theo_disputed"]
    probes.resolve(
        "rq_theo_disputed", prepared.review,
        ReviewDecision(
            reviewer_id="ma_jess", action="edited",
            corrections={injected.field_name: injected.original},
        ),
        now=CLINIC_NOW,
    )
    # The probe review is excluded from the genuine edit-rate statistics...
    assert audit.rubber_stamp_report(initiative_id="I-01", min_reviews=1) == []
    # ...and counted in the one it belongs to.
    assert audit.probe_catch_rate(initiative_id="I-01")["probes"] == 1


def test_the_catch_rate_is_reported_per_reviewer():
    """It is usually one person's workload that broke first."""
    program = ProbeProgram(rate=2)
    for index, (reviewer, caught) in enumerate(
        [("a", True), ("a", True), ("b", False), ("b", False)]
    ):
        item = probe.Probe(
            f"r{index}", "height_in", "48.5", "45.8", "transposed", CLINIC_NOW
        )
        item.resolved, item.caught, item.reviewer_id = True, caught, reviewer
        program.probes[f"r{index}"] = item
    rates = program.catch_rate()
    assert rates["catch_rate"] == 0.5
    assert rates["by_reviewer"]["a"]["catch_rate"] == 1.0
    assert rates["by_reviewer"]["b"]["catch_rate"] == 0.0


def test_a_probe_goes_through_the_same_render_path_as_a_real_value(
    tmp_path, blank, audit
):
    """A probe inserted after the fill would be testing a different code path
    from the one it claims to measure."""
    probes = ProbeProgram(audit=audit, rate=2)
    pipeline = make_pipeline(tmp_path, audit=audit, probes=probes)
    _, prepared = run_case(
        "theo_disputed", tmp_path, blank, pipeline=pipeline, audit=audit
    )
    write = next(
        w for w in prepared.filled.writes if w.field_name == prepared.probe_field
    )
    assert write.method == "probe"
    assert write.synthetic
    assert write.field_name in prepared.filled.highlights


# ==========================================================================
# pipeline.py
# ==========================================================================


def test_the_pipeline_refuses_a_form_with_no_confirmed_patient(tmp_path, blank):
    """README I-01: never auto-match on a fuzzy name alone."""
    pipeline = make_pipeline(tmp_path)
    request = lifecycle.FormRequest(
        request_id="r1", patient_id="", form_type="demo_camp_health_form",
        channel="fax", received_at=CLINIC_NOW,
    )
    with pytest.raises(PatientNotConfirmed, match="fuzzy name"):
        pipeline.prepare(
            request, blank_pdf=blank, destination=str(tmp_path / "x.pdf"),
            now=CLINIC_NOW,
        )


def test_the_pipeline_refuses_an_uncalibrated_template(tmp_path, blank):
    """No document is produced, rather than a wrong one produced and discarded."""
    pipeline = make_pipeline(tmp_path)
    request = pipeline.tracker.receive(
        request_id="r1", patient_id="p_rosa",
        form_type="il_certificate_of_child_health_examination",
        channel="fax", now=CLINIC_NOW,
    )
    with pytest.raises(UncalibratedTemplate):
        pipeline.prepare(
            request, blank_pdf=blank, destination=str(tmp_path / "x.pdf"),
            now=CLINIC_NOW,
        )


def test_an_unreachable_chart_produces_no_form_at_all(tmp_path, blank):
    pipeline = make_pipeline(tmp_path, unreachable=("p_rosa",))
    request = pipeline.tracker.receive(
        request_id="r1", patient_id="p_rosa", form_type="demo_camp_health_form",
        channel="fax", now=CLINIC_NOW,
    )
    with pytest.raises(ChartUnavailable):
        pipeline.prepare(
            request, blank_pdf=blank, destination=str(tmp_path / "x.pdf"),
            now=CLINIC_NOW,
        )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_every_fixture_reaches_a_review_screen(case, tmp_path, blank):
    _, prepared = run_case(case.name, tmp_path, blank)
    assert prepared.review is not None
    assert prepared.filled is not None
    assert prepared.request.state == FormState.MA_REVIEW
    assert prepared.releasable == case.expect["releasable"]
    if "blocker" in case.expect:
        assert case.expect["blocker"] in {b["kind"] for b in prepared.review.blockers}
    if "disputed_groups" in case.expect:
        assert prepared.reconciliation.disputed_groups == case.expect["disputed_groups"]


def test_the_lifecycle_moves_with_the_pipeline(tmp_path, blank, audit):
    pipeline, prepared = run_case("rosa_clean", tmp_path, blank, audit=audit)
    states = [t.to_state for t in prepared.request.history]
    assert states == [FormState.FILLED, FormState.MA_REVIEW]
    assert prepared.request.notifications


def test_a_parent_is_notified_at_each_stage(tmp_path, blank):
    _, prepared = run_case("rosa_clean", tmp_path, blank)
    assert prepared.request.notifications[0]["state"] == FormState.MA_REVIEW
    assert "prepared" in prepared.request.notifications[0]["message"]


def test_a_re_filled_form_reflects_the_corrected_chart(tmp_path, blank):
    """A blocked form goes back to `filled`, and the second pass is a new
    document rather than the same one waved through."""
    pipeline, first = run_case("nadia_registry_down", tmp_path, blank)
    assert not first.releasable
    pipeline.tracker.advance(
        first.request, FormState.BLOCKED, actor="ma_jess", now=CLINIC_NOW
    )
    second = pipeline.prepare(
        first.request, blank_pdf=blank, destination=str(tmp_path / "again.pdf"),
        now=CLINIC_NOW + timedelta(hours=2),
        chart_doses=chart_doses_for("nadia_registry_down"),
        # This time the registry answered.
        registry_doses=[
            DoseRecord("nr1", "20", date(2016, 8, 11), "registry"),
            DoseRecord("nr2", "20", date(2016, 10, 13), "registry"),
            DoseRecord("nr3", "03", date(2017, 7, 1), "registry", precision="month"),
        ],
    )
    assert second.reconciliation.registry_consulted
    assert not [
        b for b in second.review.blockers if b["kind"] == "registry_not_reconciled"
    ]
    assert second.request.state == FormState.MA_REVIEW


def test_a_probe_is_never_silently_skipped_for_want_of_a_perturbation():
    """Found by the test above: a name field was in the safe allowlist and no
    perturbation could touch it, so a form whose rotation landed on a name
    produced NO probe and nothing said so. A complacency control that quietly
    does not run is the one failure mode it cannot afford."""
    program = ProbeProgram(rate=2)
    filled = fill.FilledForm(form_type="t", patient_id="p", template_version="1")
    for name, text in (("child_last_name", "Alvarez"), ("child_first_name", "Rosa")):
        box = fill.BoundingBox(1, 0, 0, 200, 12)
        filled.writes.append(fill.FieldWrite(name, text, box, "auto"))
        filled.highlights.append(name)
    override = program.build_override(filled, request_id="r", now=CLINIC_NOW)
    assert override is not None
    field_name, injected = next(iter(override.items()))
    assert injected != dict((w.field_name, w.text) for w in filled.writes)[field_name]
    assert sorted(injected.lower()) == sorted(
        dict((w.field_name, w.text) for w in filled.writes)[field_name].lower()
    ), "a transposition, not a different name"


def test_the_probe_target_rotates_rather_than_always_being_the_first_field():
    """A probe always in the same box teaches the reviewer to check that box."""
    program = ProbeProgram(rate=2)
    seen = set()
    for patient in [f"p{i}" for i in range(12)]:
        filled = fill.FilledForm(form_type="t", patient_id=patient, template_version="1")
        for name, text in (
            ("height_in", "48.5"), ("weight_lb", "52.0"), ("bmi", "15.4"),
            ("exam_date", "07/10/2026"),
        ):
            box = fill.BoundingBox(1, 0, 0, 200, 12)
            filled.writes.append(fill.FieldWrite(name, text, box, "auto"))
            filled.highlights.append(name)
        seen.add(program.candidates(filled)[0][0])
    assert len(seen) > 1


# ==========================================================================
# adversarial-review regressions
#
# One test per finding. Each reproduces the ORIGINAL defect, and each was
# mutation-checked: the bug reintroduced, the named test confirmed to fail.
# ==========================================================================


def test_f01_a_probe_never_reaches_a_signature(tmp_path, blank, audit):
    """#1 Scoring a probe cleared the release blocker but nothing re-rendered
    the page, so the deliberately wrong value was still on the PDF that got
    signed and sent. And `should_probe` re-probed the same request on every
    re-fill, so the request could never be released clean."""
    import fitz

    probes = ProbeProgram(audit=audit, rate=2, salt="probe-salt-0")
    pipeline = make_pipeline(tmp_path, audit=audit, probes=probes)
    _, first = run_case("rosa_clean", tmp_path, blank, pipeline=pipeline, audit=audit)
    assert first.probe_field
    injected = probes.probes["rq_rosa_clean"]

    # The blocker is a property of the DOCUMENT, not of the probe registry.
    assert first.filled.has_synthetic_write
    kinds = {b["kind"] for b in first.review.blockers}
    assert "synthetic_probe_outstanding" in kinds
    assert "synthetic_value_on_the_document" in kinds

    # The MA catches it. That scores the probe; it does not clean the page.
    probes.resolve(
        "rq_rosa_clean", first.review,
        ReviewDecision(
            reviewer_id="ma_jess", action="edited",
            corrections={injected.field_name: injected.original},
        ),
        now=CLINIC_NOW,
    )
    assert injected.caught is True
    document = fitz.open(first.filled.output_path)
    box = first.template.field(injected.field_name).box
    assert injected.injected in document[box.page - 1].get_textbox(
        fitz.Rect(*box.rect)
    )
    document.close()
    still = ReleaseGate().blockers(
        first.filled, first.record, as_of=CLINIC_NOW.date(), probe_outstanding=False
    )
    assert any(b["kind"] == "synthetic_value_on_the_document" for b in still)

    # Re-filling is what cleans it, and a scored request is never probed again.
    assert not probes.should_probe("rq_rosa_clean")
    pipeline.tracker.advance(
        first.request, FormState.BLOCKED, actor="ma_jess", now=CLINIC_NOW
    )
    second = pipeline.prepare(
        first.request, blank_pdf=blank, destination=str(tmp_path / "clean.pdf"),
        now=CLINIC_NOW + timedelta(hours=1),
        chart_doses=chart_doses_for("rosa_clean"),
        registry_doses=registry_doses_for("rosa_clean"),
    )
    assert not second.probe_field
    assert not second.filled.has_synthetic_write
    assert second.releasable


def test_f02_overflow_is_measured_not_estimated(tmp_path, blank, camp):
    """#2 The 0.55-em estimate was permissive: Helvetica capitals run ~0.68em,
    so a 67-character upper-case allergy list passed a 69-character "capacity"
    and rendered 17 points past the rule -- unflagged, unblocked, and outside
    the highlight, so the box read as a complete list."""
    import fitz

    text = "WALNUT ANAPHYLAXIS, CASHEW ANAPHYLAXIS, MACADAMIA, WASP VENOM, MILK"
    box = camp.field("allergies").box
    assert len(text) * 8.5 * 0.55 < box.width - 4.0, "the old estimate said it fits"
    assert fitz.get_text_length(text, fontname="helv", fontsize=8.5) > box.width - 4.0

    case = by_name("rosa_clean")
    record = copy.deepcopy(case.record)
    record.data["allergies"] = SourceValue(
        [text], "ehr", "AllergyIntolerance/1", recorded=CLINIC_NOW.date()
    )
    result = reconcile_for_form(list(case.chart_doses), list(case.registry_doses))
    record.data["immunizations"] = build_immunization_block(result)
    filled = FormFiller(PyMuPDFBackend()).fill(
        camp, record, blank_pdf=blank,
        destination=str(tmp_path / "wide.pdf"), now=CLINIC_NOW,
    )
    truncated = [w for w in filled.truncated if w.field_name == "allergies"]
    assert truncated, "the real width must be measured, not estimated"

    # Nothing renders past the box.
    document = fitz.open(filled.output_path)
    page = document[box.page - 1]
    rightmost = max(
        (span["bbox"][2] for block in page.get_text("dict")["blocks"]
         for line in block.get("lines", []) for span in line["spans"]
         if span["bbox"][1] >= box.y - 2 and span["bbox"][3] <= box.y + box.height + 2),
        default=0.0,
    )
    assert rightmost <= box.x + box.width + 0.5, rightmost
    document.close()


def test_f02_the_renderer_refuses_text_wider_than_its_box(camp):
    """The post-condition behind the measurement. Loud, because silent ink
    outside its box is the failure it replaced."""
    from modules.forms.fill import TextOverflow

    backend = PyMuPDFBackend()
    backend.open(blankforms.write_blank_form(camp, "/tmp/_pc.pdf"))
    with pytest.raises(TextOverflow, match="not trimmed"):
        backend.write_text(camp.field("child_last_name").box, "M" * 200, font_size=8.5)


def test_f03_doses_with_no_box_are_reported_not_dropped(tmp_path, blank, camp):
    """#3 Rows sort oldest-first, so the dose that fell off the end of a short
    form was always the MOST RECENT one -- the one a camp actually checks. A
    2024 Tdap vanished behind a 2019 Td with no skipped field, no discrepancy
    and no blocker, and the form released."""
    from modules.forms.reconcile import grid_capacity_findings

    result = reconcile_for_form(
        [
            DoseRecord("c1", "09", date(2019, 5, 2), "chart"),      # Td
            DoseRecord("c2", "115", date(2024, 6, 11), "chart"),    # Tdap
        ],
        [
            DoseRecord("r1", "09", date(2019, 5, 2), "registry"),
            DoseRecord("r2", "115", date(2024, 6, 11), "registry"),
        ],
    )
    assert len(result.antigens["tdap"].doses) == 2
    boxes = [f for f in camp.fields if f.source.startswith("immunizations.tdap.")]
    assert len(boxes) == 1, "the demo form has one Tdap box"

    findings = grid_capacity_findings(camp, result)
    overflow = [f for f in findings if f["group"] == "tdap"]
    assert overflow and overflow[0]["severity"] == "blocking"
    assert "2024-06-11" in overflow[0]["detail"]


def test_f04_a_dose_never_prints_in_a_row_labelled_for_another_product():
    """#4 `pcv` claimed PPSV and `mcv` claimed MenB, so a PPSV23 date printed in
    a box the Illinois form labels "Pneumococcal conjugate - dose 3" -- reading
    as a completed conjugate series the child has not had."""
    assert _groups_for("133") == ("pcv",)        # PCV13, a conjugate
    assert _groups_for("33") == ("ppsv",)        # PPSV23, a polysaccharide
    assert _groups_for("114") == ("mcv",)        # MenACWY
    assert _groups_for("162") == ("menb",)       # MenB
    assert ANTIGEN_FIELD_GROUPS["pcv"].isdisjoint(ANTIGEN_FIELD_GROUPS["ppsv"])
    assert ANTIGEN_FIELD_GROUPS["mcv"].isdisjoint(ANTIGEN_FIELD_GROUPS["menb"])

    result = reconcile_for_form(
        [DoseRecord("c1", "33", date(2022, 4, 5), "chart")],
        [DoseRecord("r1", "33", date(2022, 4, 5), "registry")],
    )
    assert not result.antigens["pcv"].doses
    assert len(result.antigens["ppsv"].doses) == 1


def test_f04_the_illinois_form_has_a_box_for_every_row_it_can_fill():
    """A row with no box silently loses every dose in it."""
    template = TemplateStore.load(allow_placeholder=True).get(
        "il_certificate_of_child_health_examination"
    )
    boxed = {
        f.source.split(".")[1] for f in template.fields
        if f.source.startswith("immunizations.")
    }
    assert boxed == set(ANTIGEN_FIELD_GROUPS), set(ANTIGEN_FIELD_GROUPS) - boxed


def test_f05_probing_a_field_does_not_hide_its_stale_value(tmp_path, blank, audit):
    """#5 A probed write carried `source_path=""`, and the stale gate matched on
    `source_path`. Probing a box therefore erased the evidence that a stale
    chart value had reached the form -- missing data widening a threshold."""
    # Salt pinned so this request is probed AND the probe lands on a field whose
    # chart value is stale. Left to the default, both depended on a hash and the
    # assertions below were vacuously true whenever either missed -- which is how
    # the second half of this defect survived the first fix.
    probes = ProbeProgram(audit=audit, rate=2, salt="omar-salt-7")
    assert probes.should_probe("rq_omar_stale_vitals")
    pipeline = make_pipeline(tmp_path, audit=audit, probes=probes)
    unprobed = make_pipeline(tmp_path, audit=audit)

    _, without = run_case("omar_stale_vitals", tmp_path, blank, pipeline=unprobed)
    _, with_probe = run_case("omar_stale_vitals", tmp_path, blank, pipeline=pipeline)
    assert with_probe.probe_field

    stale_without = {
        b["detail"].split(" was recorded")[0]
        for b in without.review.blockers if b["kind"] == "stale_value"
    }
    stale_with = {
        b["detail"].split(" was recorded")[0]
        for b in with_probe.review.blockers if b["kind"] == "stale_value"
    }
    assert stale_without
    assert stale_without, "the un-probed form reports the stale height"
    assert stale_without <= stale_with, "probing must not remove a stale blocker"
    # The probed field's OWN stale path is among them. The gate reads
    # `filled.writes`, not `filled.auto_filled`: a probed write has
    # method="probe" and was invisible to it.
    probed_path = next(
        w.source_path for w in with_probe.filled.writes
        if w.field_name == with_probe.probe_field
    )
    assert probed_path == "vitals.height_in"
    assert probed_path in stale_without
    assert probed_path in stale_with

    write = next(
        w for w in with_probe.filled.writes if w.field_name == with_probe.probe_field
    )
    assert write.method == "probe"
    assert write.source_path, "provenance belongs to the template, not the value"
    # And the probed field's own stale blocker is still there.
    if write.source_path in {
        b["detail"].split(" was recorded")[0] for b in without.review.blockers
    }:
        assert write.source_path in {
            b["detail"].split(" was recorded")[0]
            for b in with_probe.review.blockers if b["kind"] == "stale_value"
        }


def test_f06_the_tracker_holds_the_blockers_rather_than_accepting_them(
    tmp_path, blank
):
    """#6 `blockers=` defaulted to `()`, so omitting it was indistinguishable
    from a clean form and any caller could sign a form with an unsettled
    immunization on it. The ledger then recorded a clean history."""
    pipeline, prepared = run_case("theo_disputed", tmp_path, blank)
    assert not prepared.releasable
    assert prepared.request.review_blockers

    with pytest.raises(IllegalTransition, match="outstanding blocker"):
        pipeline.tracker.advance(
            prepared.request, FormState.PHYSICIAN_SIGNATURE,
            actor="ma_jess", now=CLINIC_NOW,
        )


def test_f06_a_review_does_not_authorise_a_later_document(tmp_path, blank):
    """Moving back to `filled` produces a new document, and the review that
    cleared the old one does not clear it."""
    pipeline, first = run_case("rosa_clean", tmp_path, blank)
    assert first.releasable and first.request.reviewed_fill

    pipeline.tracker.advance(
        first.request, FormState.BLOCKED, actor="ma_jess", now=CLINIC_NOW
    )
    assert not first.request.reviewed_fill
    assert not first.request.review_blockers


def test_f06_the_filler_consults_the_calibration_gate_itself(tmp_path, blank):
    """A caller holding a template from `store.get()` bypassed the store's gate
    and filled a legal document from guessed coordinates."""
    loose = TemplateStore.load(allow_placeholder=True)
    template = loose.get("il_certificate_of_child_health_examination")
    case = by_name("rosa_clean")
    with pytest.raises(UncalibratedTemplate, match="will not"):
        FormFiller(RecordingBackend()).fill(
            template, copy.deepcopy(case.record), blank_pdf=blank,
            destination=str(tmp_path / "x.pdf"), now=CLINIC_NOW,
        )


def test_f07_resubmitting_the_injected_value_is_not_a_catch():
    """#7 "rejected AND touched this field" never compared the value, so a
    screen that resubmits fields as displayed scored a catch while the injected
    error stood. The number that detects a rubber stamp was the number a rubber
    stamp produced."""
    program = ProbeProgram(rate=2)
    program.probes["r"] = probe.Probe(
        "r", "blood_pressure", "98/60", "89/60", "transposed", CLINIC_NOW
    )
    payload = review.ReviewPayload(form_type="t", patient_id="p", pdf_path="")
    scored = program.resolve(
        "r", payload,
        ReviewDecision(
            reviewer_id="ma_jess", action="rejected",
            corrections={"blood_pressure": "89/60"},   # the injected value
        ),
        now=CLINIC_NOW,
    )
    assert scored.caught is False
    assert program.catch_rate()["catch_rate"] == 0.0


def test_f08_the_review_pane_does_not_flag_every_dose_as_stale(tmp_path, blank):
    """#8 `build_review` recomputed staleness from the write's date instead of
    asking the SourceValue, so 9 of 21 rows on the one clean form were marked
    stale and none of them meant anything -- the shape of alert that trains a
    reviewer to skip the list the real stale vital is in."""
    _, clean = run_case("rosa_clean", tmp_path, blank)
    flagged = [r.field_name for r in clean.review.provenance if r.stale]
    assert not flagged, flagged
    assert any(r.historical for r in clean.review.provenance)

    _, stale = run_case("omar_stale_vitals", tmp_path, blank)
    marked = {r.field_name for r in stale.review.provenance if r.stale}
    assert {"height_in", "weight_lb", "blood_pressure"} <= marked
    assert not any(r.field_name.startswith("imm_") for r in stale.review.provenance if r.stale)


def test_f09_a_template_typo_is_refused_at_load_time(tmp_path):
    """#9 Only NON-FINAL path segments were checked, so `allergies_list` and
    `vitals.height_inches` both returned None and the box was skipped with the
    reason "the chart holds no value for this field" -- a statement about the
    child, shown to the MA, describing a configuration bug. A child with a
    documented penicillin allergy got a blank allergy box on a signed form."""
    from modules.forms.chart import is_known_path

    assert is_known_path("vitals.height_in")
    assert is_known_path("allergies")
    assert is_known_path("immunizations.dtap.0")
    assert not is_known_path("vitals.height_inches")
    assert not is_known_path("allergies_list")
    assert not is_known_path("patient.dateofbirth")

    import yaml

    with open(os.path.join(tmp_path, "t.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "form_type": "t", "page_count": 1,
                "fields": [{
                    "name": "allergies", "kind": "text", "source": "allergies_list",
                    "transform": "comma_list",
                    "box": {"page": 1, "x": 10, "y": 10, "width": 100, "height": 12},
                }],
            },
            handle,
        )
    with pytest.raises(TemplateInvalid, match="not\\s+a path a chart record has"):
        TemplateStore.load(tmp_path, allow_placeholder=True)


def test_f09_every_shipped_template_names_only_real_chart_paths():
    """The load-time check, applied to what actually ships."""
    from modules.forms.chart import is_known_path

    store = TemplateStore.load(allow_placeholder=True)
    for template in store.templates.values():
        for spec in template.fields:
            if spec.source:
                assert is_known_path(spec.source), (template.form_type, spec.name)


def test_f10_a_transform_returning_none_never_falls_back_to_a_raw_value(
    tmp_path, blank, camp
):
    """Found by mutation testing: fill.py's refusal #3 -- "never a raw value, an
    empty string, or 'N/A'" -- had no end-to-end test. A fallback to
    `str(source.value)` wrote `[]` into the "Current medications" box of every
    fixture form and the whole suite passed."""
    _, prepared = run_case("rosa_clean", tmp_path, blank)
    for write in prepared.filled.writes:
        assert write.text.strip()
        assert write.text not in ("[]", "{}", "None", "N/A", "n/a")
    blanked = {s.field_name for s in prepared.filled.skipped}
    assert "current_medications" in blanked


def test_f10_the_second_dispute_lock_is_load_bearing(tmp_path, blank, camp):
    """Found by mutation testing: the lock the README calls "checked twice ...
    because this is the check whose failure puts a wrong date on a legal
    document" had no test of its own -- `dose_date` covered for it in every
    fixture. Here the transform is one that would happily render a disputed
    dose, so only the fill-loop lock stands between it and the page."""
    from modules.forms.templates import TRANSFORMS

    case = by_name("theo_disputed")
    record = copy.deepcopy(case.record)
    result = reconcile_for_form(list(case.chart_doses), list(case.registry_doses))
    assert "dtap" in result.disputed_groups
    record.data["immunizations"] = build_immunization_block(result)

    disputed = record.data["immunizations"]["dtap"][0]
    assert disputed.value["disputed"]
    # A transform with no opinion about disputes at all.
    assert TRANSFORMS["verbatim"](disputed.value) is not None

    naive = replace(camp.field("imm_dtap_1"), transform="verbatim")
    template = replace(
        camp,
        fields=tuple(naive if f.name == "imm_dtap_1" else f for f in camp.fields),
    )
    filled = FormFiller(RecordingBackend()).fill(
        template, record, blank_pdf=blank,
        destination=str(tmp_path / "x.pdf"), now=CLINIC_NOW,
    )
    assert "imm_dtap_1" not in {w.field_name for w in filled.writes}
    assert any(
        s.field_name == "imm_dtap_1" and "do not agree" in s.reason
        for s in filled.skipped
    )


def test_f10_a_signature_field_is_refused_on_kind_alone(camp):
    """Found by mutation testing: the signature guarantee held, but the test for
    it was satisfied by the YAML `human_only: true` flag alone -- a template
    author who omitted that flag would have been caught by nothing."""
    from modules.forms.templates import FieldSpec

    spec = replace(camp.field("provider_signature"), human_only=False)
    assert spec.kind == "signature"
    assert not spec.machine_writable
