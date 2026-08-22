"""I-10 — standing order digitization and delegation audit.

The initiative with the smallest dollar return and the largest downside
protection. What is under test:

  1. an MA sees only what they are currently competent for
  2. the check runs again at the moment of the act, not just when the screen
     was built
  3. break glass never blocks care, and is never quiet
  4. a signed standing order is never edited, only superseded
  5. the delegation rules are configuration, so the 2027 sunset is a YAML edit
"""

from __future__ import annotations

import ast
import os
from datetime import date, datetime, time, timedelta, timezone

import pytest

from modules.delegation import (
    BreakGlassRefused,
    Competency,
    CompetencyRecord,
    DelegationRules,
    DelegationService,
    FrameworkSunset,
    NotAuthorised,
    Roster,
    RosterEntry,
    StaffMember,
    StandingOrder,
    UnreviewedRules,
    UnsignedOrder,
)
from modules.delegation import enforcement as enforcement_module
from modules.delegation.fixtures import NOW, build
from nsp_core.audit import AuditLog

_MODULE_DIR = os.path.dirname(os.path.abspath(enforcement_module.__file__))


@pytest.fixture
def audit(tmp_path):
    return AuditLog(tmp_path / "audit.sqlite3", hmac_key=b"test-key")


@pytest.fixture
def service(audit):
    rules, orders, competencies, roster = build()
    rules.review["owner"] = "dr_alvarez"
    return DelegationService(
        rules=rules, orders=orders, competencies=competencies,
        roster=roster, audit=audit,
    )


def order_ids(orders):
    return sorted(o.order_id for o in orders)


# ==========================================================================
# structural guards
# ==========================================================================


def test_no_model_in_the_enforcement_path():
    """README I-10: every enforcement sub-task is "Deterministic." A model
    belongs only in authoring aids, which are not built here."""
    for name in ("enforcement.py", "register.py"):
        tree = ast.parse(open(os.path.join(_MODULE_DIR, name), encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "nsp_core.llm" not in node.module, name
                assert node.module not in ("openai", "anthropic"), name


def test_no_statute_specific_logic_is_hard_coded():
    """The whole architectural argument: when 54.2 sunsets, the practice edits
    a YAML file rather than this application."""
    source = open(
        os.path.join(_MODULE_DIR, "enforcement.py"), encoding="utf-8"
    ).read()
    tree = ast.parse(source)
    literals = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    for forbidden in ("physician", "medical_assistant", "registered_nurse"):
        assert not any(
            literal == forbidden for literal in literals
        ), f"{forbidden!r} is a role name; roles live in the config"


def test_the_rules_come_from_config_not_from_code():
    rules = DelegationRules.load()
    assert rules.framework["citation"] == "225 ILCS 60/54.2"
    assert rules.sunsets_on == date(2027, 1, 1)
    assert {r["id"] for r in rules.requirements} == {
        "written_delegation", "individual_competency", "competency_current",
        "licensed_professional_on_site", "within_delegating_scope",
    }


def test_an_empty_rule_set_is_refused(tmp_path):
    """An empty rule set is not a permissive rule set; it is a broken one."""
    import yaml

    data = yaml.safe_load(open("config/delegation_rules.yaml", encoding="utf-8"))
    data["requirements"] = []
    path = tmp_path / "empty.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="every act would pass every check"):
        DelegationRules.load(path)


# ==========================================================================
# the screen
# ==========================================================================


def test_an_ma_sees_only_what_they_are_competent_for(service):
    """README I-10: "An MA opening a task sees ONLY orders they are currently
    competent to execute." The filter is the control -- there is no moment
    where somebody has to remember not to click."""
    experienced = order_ids(service.available_orders("ma_jess", moment=NOW))
    new_hire = order_ids(service.available_orders("ma_dana", moment=NOW))
    assert experienced == ["so_immunize", "so_screening", "so_strep", "so_vitals"]
    assert new_hire == ["so_vitals"]


def test_an_expired_competency_removes_the_order_from_the_screen(service):
    """An expired competency is not a weaker competency; it is an unverified
    one. Marta's CPR card lapsed on the 18th."""
    available = order_ids(service.available_orders("lpn_marta", moment=NOW))
    assert "so_immunize" not in available
    blocked = {
        row["order_id"]: row for row in service.blocked_orders("lpn_marta", moment=NOW)
    }
    assert blocked["so_immunize"]["reasons"] == ["competency_current"]
    assert "expired 2026-08-18" in blocked["so_immunize"]["detail"]


def test_a_missing_competency_reads_differently_from_an_expired_one(service):
    """One is a training gap and one is a renewal. The remedy differs."""
    dana = {r["order_id"]: r for r in service.blocked_orders("ma_dana", moment=NOW)}
    assert dana["so_immunize"]["reasons"] == ["individual_competency"] * 3
    assert "has no verified" in dana["so_immunize"]["detail"]


def test_the_supervisor_screen_shows_what_the_ma_screen_hides(service):
    """A practice manager needs to see that somebody is one expired card away
    from not being able to give vaccines."""
    blocked = service.blocked_orders("lpn_marta", moment=NOW)
    assert blocked
    assert all("reasons" in row and row["detail"] for row in blocked)


# ==========================================================================
# the decision
# ==========================================================================


def test_a_competency_is_held_by_a_person_not_by_a_role(service):
    """The statute requires "appropriate training and experience" for THAT
    person. Two medical assistants, same role, different worklists."""
    jess = service.authorise("ma_jess", "so_immunize", moment=NOW)
    dana = service.authorise("ma_dana", "so_immunize", moment=NOW)
    assert jess.staff.role == dana.staff.role == "medical_assistant"
    assert jess.authorised
    assert not dana.authorised


def test_the_on_site_requirement_is_evidenced_by_a_named_person(service):
    decision = service.authorise("ma_jess", "so_immunize", moment=NOW)
    assert decision.supervisor is not None
    assert decision.supervisor.staff_id == "dr_alvarez"


def test_nobody_supervises_themselves(service):
    """An RN is both a licensed role and a delegable role. Without an explicit
    check the register would record them as their own supervising professional,
    which satisfies a lookup and none of the statute."""
    lunch = NOW.replace(hour=12, minute=30)      # no physician on site
    rules, orders, competencies, roster = build()
    rules.review["owner"] = "x"
    # Make the order supervisable by an RN, so Paulo is the only candidate.
    original = orders.versions["so_immunize"][-1]
    orders.versions["so_immunize"][-1] = StandingOrder(
        **{**original.__dict__, "required_supervision_role": "registered_nurse"}
    )
    service = DelegationService(
        rules=rules, orders=orders, competencies=competencies, roster=roster
    )
    decision = service.authorise("rn_paulo", "so_immunize", moment=lunch)
    assert not decision.authorised
    assert any(
        r.rule_id == "licensed_professional_on_site" for r in decision.refusals
    )


def test_the_delegating_physician_need_not_be_the_one_on_site(service):
    """225 ILCS 60/54.2 requires "a licensed health care professional is on
    site", not that the DELEGATING physician is. Conflating the two would stop a
    practice functioning on any day its signer is away."""
    decision = service.authorise("ma_jess", "so_strep", moment=NOW)
    assert decision.order.delegating_physician_id == "dr_osei"   # not in today
    assert decision.supervisor.staff_id == "dr_alvarez"
    assert decision.authorised


def test_an_overdue_order_review_is_reported_but_does_not_block(service):
    """A protocol nobody re-read is still the protocol the physician signed.
    Refusing every vaccine because a review date slipped is a worse outcome
    than the risk it addresses."""
    decision = service.authorise("ma_jess", "so_vitals", moment=NOW)
    assert decision.authorised
    non_blocking = [r for r in decision.refusals if not r.blocking]
    assert non_blocking
    assert "may have drifted" in non_blocking[0].detail


def test_every_refusal_names_a_rule_id_from_the_config(service):
    """So an audit answer is a list of rule ids rather than a paragraph."""
    known = {r["id"] for r in service.rules.requirements}
    for staff_id in ("ma_dana", "lpn_marta", "rn_paulo"):
        for order_id in service.orders.versions:
            decision = service.authorise(staff_id, order_id, moment=NOW)
            for refusal in decision.refusals:
                assert refusal.rule_id in known


# ==========================================================================
# the act
# ==========================================================================


def test_the_check_runs_again_at_the_moment_of_the_act(service):
    """A competency can expire between the MA opening the worklist at 08:00 and
    giving the injection at 11:40, and the physician goes to lunch."""
    assert service.authorise("ma_jess", "so_immunize", moment=NOW).authorised
    lunch = NOW.replace(hour=12, minute=30)
    with pytest.raises(NotAuthorised, match="licensed_professional_on_site"):
        service.execute(
            "ma_jess", "so_immunize", patient_id="p1", moment=lunch,
            execution_id="x1",
        )


def test_an_execution_records_the_supervising_professional(service, audit):
    result = service.execute(
        "ma_jess", "so_immunize", patient_id="p_rosa",
        moment=NOW + timedelta(minutes=4), execution_id="ex_1",
    )
    rows = audit.delegation_evidence(staff_id="ma_jess")
    assert len(rows) == 1
    assert rows[0]["supervising_pro_id"] == "dr_alvarez"
    assert rows[0]["supervisor_on_site"] == 1
    assert rows[0]["standing_order_id"] == "so_immunize"
    assert rows[0]["standing_order_version"] == "2"
    assert rows[0]["competency_record_id"]
    assert result.authorisation.authorised


def test_an_execution_records_the_order_version_not_just_the_order(service, audit):
    """Every execution has to resolve to the exact text the physician signed."""
    service.execute(
        "ma_jess", "so_immunize", patient_id="p1", moment=NOW, execution_id="ex_1"
    )
    row = audit.delegation_evidence(staff_id="ma_jess")[0]
    order = service.orders.version_at("so_immunize", int(row["standing_order_version"]))
    assert order.signed_by == "dr_alvarez"
    assert "ACIP" in order.clinical_content


# ==========================================================================
# break glass
# ==========================================================================


def test_break_glass_never_blocks_patient_care(service, audit):
    """Build plan: "a break-glass path that requires a justification and never
    blocks patient care.\""""
    lunch = NOW.replace(hour=12, minute=30)
    result = service.execute(
        "ma_jess", "so_immunize", patient_id="p_theo", moment=lunch,
        execution_id="ex_bg",
        break_glass_reason=(
            "Post-exposure tetanus prophylaxis required now; physician off site "
            "at lunch, reached by phone and verbally authorised."
        ),
    )
    assert result.break_glass
    assert result.review_due_utc == lunch + timedelta(hours=24)


def test_break_glass_is_never_quiet(service, audit):
    """A break-glass nobody reviews afterwards is just an unlocked door."""
    lunch = NOW.replace(hour=12, minute=30)
    service.execute(
        "ma_jess", "so_immunize", patient_id="p_theo", moment=lunch,
        execution_id="ex_bg",
        break_glass_reason=(
            "Post-exposure tetanus prophylaxis required now; physician off site "
            "at lunch, reached by phone and verbally authorised."
        ),
    )
    gaps = audit.unevidenced_supervision()
    assert len(gaps) == 1
    assert gaps[0]["break_glass"] == 1
    assert gaps[0]["break_glass_reason"]


def test_a_break_glass_needs_a_justification_somebody_can_read(service):
    """This is the record somebody reads tomorrow to decide whether the rule or
    the situation needs changing."""
    lunch = NOW.replace(hour=12, minute=30)
    with pytest.raises(BreakGlassRefused, match="at least 40 characters"):
        service.execute(
            "ma_jess", "so_immunize", patient_id="p1", moment=lunch,
            execution_id="x", break_glass_reason="urgent",
        )


def test_break_glass_is_recorded_even_when_the_act_was_authorised_anyway(
    service, audit
):
    """Somebody who invokes it unnecessarily still gets reviewed. The path is
    not a shortcut."""
    service.execute(
        "ma_jess", "so_immunize", patient_id="p1", moment=NOW,
        execution_id="ex_bg2",
        break_glass_reason=(
            "Invoked out of caution during a busy clinic; supervision was in "
            "fact present throughout."
        ),
    )
    assert audit.unevidenced_supervision()


# ==========================================================================
# versioning and signatures
# ==========================================================================


def test_an_unsigned_standing_order_cannot_be_published(service):
    """Under 54.2 the delegation is the physician's act; an unsigned protocol
    authorises nothing."""
    with pytest.raises(UnsignedOrder, match="no physician signature"):
        service.orders.publish(
            StandingOrder(
                order_id="so_new", version=1, title="X", task_code="x",
                clinical_content="...", delegating_physician_id="dr_alvarez",
                effective_from=date(2026, 9, 1),
                required_competencies=("im_injection",),
                required_supervision_role="physician",
            )
        )


def test_the_signer_must_be_the_delegating_physician(service):
    """The delegation and the signature are the same act."""
    with pytest.raises(UnsignedOrder, match="same act"):
        service.orders.publish(
            StandingOrder(
                order_id="so_new", version=1, title="X", task_code="x",
                clinical_content="...", delegating_physician_id="dr_alvarez",
                effective_from=date(2026, 9, 1),
                required_competencies=("im_injection",),
                required_supervision_role="physician",
                signed_by="dr_osei",
                signed_utc=datetime(2026, 8, 1, tzinfo=timezone.utc),
            )
        )


def test_publishing_a_new_version_retires_the_old_one_rather_than_editing_it(
    service
):
    """A signed order is an attestation about a specific text. If that text can
    change underneath the signature, every execution logged against it becomes
    unprovable."""
    assert not hasattr(service.orders, "update")
    before = service.orders.current("so_immunize", on=NOW.date())
    service.orders.publish(
        StandingOrder(
            order_id="so_immunize", version=3, title="Revised",
            task_code="administer_vaccine", clinical_content="...2027...",
            delegating_physician_id="dr_alvarez",
            effective_from=date(2026, 9, 1),
            required_competencies=("im_injection",),
            required_supervision_role="physician",
            signed_by="dr_alvarez",
            signed_utc=datetime(2026, 8, 24, tzinfo=timezone.utc),
        )
    )
    retired = service.orders.version_at("so_immunize", before.version)
    assert retired.retired_on == date(2026, 9, 1)
    assert retired.clinical_content == before.clinical_content
    # The old version is still resolvable, which is the point.
    assert service.orders.current("so_immunize", on=date(2026, 8, 24)).version == 2
    assert service.orders.current("so_immunize", on=date(2026, 9, 2)).version == 3


def test_a_version_cannot_go_backwards(service):
    with pytest.raises(ValueError, match="does not follow"):
        service.orders.publish(
            StandingOrder(
                order_id="so_immunize", version=1, title="X",
                task_code="administer_vaccine", clinical_content="...",
                delegating_physician_id="dr_alvarez",
                effective_from=date(2026, 9, 1),
                required_competencies=("im_injection",),
                required_supervision_role="physician",
                signed_by="dr_alvarez",
                signed_utc=datetime(2026, 8, 24, tzinfo=timezone.utc),
            )
        )


# ==========================================================================
# competency records
# ==========================================================================


def test_an_unlicensed_person_cannot_verify_competency(service):
    """That is the whole shape of 54.2. A register that let an MA sign off
    another MA would produce records that prove nothing."""
    with pytest.raises(ValueError, match="not a licensed role"):
        service.competencies.verify(
            CompetencyRecord(
                "cr_bad", "ma_dana", "im_injection", "ma_jess",
                date(2026, 8, 1), "observed_demonstration",
            )
        )


def test_a_competency_cannot_be_self_verified(service):
    """The record exists to show somebody else looked."""
    with pytest.raises(ValueError, match="self-verified"):
        service.competencies.verify(
            CompetencyRecord(
                "cr_bad", "rn_paulo", "poct_strep", "rn_paulo",
                date(2026, 8, 1), "written_examination",
            )
        )


def test_the_evidence_kinds_are_an_allowlist(service):
    """"They have done it for years" is exactly what this replaces."""
    with pytest.raises(ValueError, match="not on the allowed"):
        service.competencies.verify(
            CompetencyRecord(
                "cr_bad", "ma_dana", "im_injection", "dr_alvarez",
                date(2026, 8, 1), "has done it for years",
            )
        )


def test_expiring_competencies_generate_a_task(service):
    """README I-10: "Competency expiring in 30 days generates a task." The
    expired ones are listed too, because that is the interesting case."""
    rows = {r["competency_id"]: r for r in service.competencies.expiring(NOW.date())}
    assert rows["bls_cpr"]["state"] == "expired"
    assert rows["bls_cpr"]["staff_id"] == "lpn_marta"


def test_overdue_order_reviews_generate_a_task(service):
    rows = {r["order_id"]: r for r in service.orders.review_tasks(NOW.date())}
    assert rows["so_vitals"]["state"] == "overdue"
    assert rows["so_vitals"]["physician"] == "dr_alvarez"


# ==========================================================================
# the roster
# ==========================================================================


def test_an_inverted_roster_entry_is_refused(service):
    """An inverted window covers no moment at all and would silently fail every
    on-site check."""
    with pytest.raises(ValueError, match="ends before it starts"):
        service.roster.record(
            RosterEntry(
                "dr_alvarez",
                datetime(2026, 8, 24, 17, tzinfo=timezone.utc),
                datetime(2026, 8, 24, 9, tzinfo=timezone.utc),
            )
        )


def test_only_licensed_roles_satisfy_the_on_site_requirement(service):
    on_site = {m.staff_id for m in service.roster.on_site_at(NOW)}
    assert "dr_alvarez" in on_site
    assert "rn_paulo" in on_site
    assert "ma_jess" not in on_site       # on site, but not a licensed role


# ==========================================================================
# the sunset, which is why this exists
# ==========================================================================


def test_the_register_warns_as_the_sunset_approaches(service):
    state, message = service.rules.sunset_state(date(2026, 12, 1))
    assert state == "approaching"
    assert "config/delegation_rules.yaml" in message


def test_the_register_refuses_to_certify_after_the_sunset(service):
    """A document citing a repealed statute looks like an answer and is not
    one."""
    service.certify(date(2026, 12, 31))
    with pytest.raises(FrameworkSunset, match="sunset"):
        service.certify(date(2027, 1, 2))


def test_the_rules_need_an_owner(audit):
    rules, orders, competencies, roster = build()
    service = DelegationService(
        rules=rules, orders=orders, competencies=competencies,
        roster=roster, audit=audit,
    )
    with pytest.raises(UnreviewedRules, match="no named owner"):
        service.certify(NOW.date())


def test_changing_the_framework_is_a_config_edit(tmp_path, audit):
    """The entire architectural argument for building this before the deadline."""
    import yaml

    data = yaml.safe_load(open("config/delegation_rules.yaml", encoding="utf-8"))
    data["framework"]["id"] = "il_successor_2027"
    data["framework"]["citation"] = "Public Act 104-XXXX"
    data["framework"]["sunsets_on"] = "2032-01-01"
    data["review"]["owner"] = "dr_alvarez"
    path = tmp_path / "successor.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    rules = DelegationRules.load(path)
    _, orders, competencies, roster = build(rules=rules)
    service = DelegationService(
        rules=rules, orders=orders, competencies=competencies,
        roster=roster, audit=audit,
    )
    # Same code, new framework, no longer sunset.
    service.certify(date(2027, 6, 1))
    assert service.authorise("ma_jess", "so_immunize", moment=NOW).authorised


# ==========================================================================
# the audit answer
# ==========================================================================


def test_the_audit_extract_is_a_query_not_a_project(service, audit):
    """README I-10: "produce every delegation, competency record, and execution
    for MA X between date A and date B\""""
    service.execute(
        "ma_jess", "so_immunize", patient_id="p1", moment=NOW, execution_id="e1"
    )
    extract = service.audit_extract(staff_id="ma_jess")
    assert extract["framework"] == "225 ILCS 60/54.2"
    assert extract["executions"] == 1
    assert [s["staff_id"] for s in extract["staff"]] == ["ma_jess"]
    assert all(r["staff_id"] == "ma_jess" for r in extract["competency_records"])
    assert extract["standing_orders"]


def test_readiness_names_everything_that_will_stop_somebody_working(service):
    report = service.readiness(NOW.date())
    assert report["framework_state"] == "approaching"
    assert any(t["state"] == "expired" for t in report["competency_tasks"])
    assert any(t["state"] == "overdue" for t in report["order_review_tasks"])


# ==========================================================================
# regression tests — the adversarial review of Part 8
#
# Each one names the finding it pins and fails if the fix is reverted. Every
# one was mutation-checked: the bug was reintroduced and the named test
# confirmed to fail.
# ==========================================================================


JUSTIFY = (
    "Child in the room needs this now and the usual route is blocked; "
    "proceeding and flagging for review."
)


def _signed_order(**overrides):
    """A publishable order, so each test can break exactly one thing."""
    fields = dict(
        order_id="so_new", version=1, title="A new protocol",
        task_code="administer_vaccine", clinical_content="Do the thing.",
        delegating_physician_id="dr_alvarez",
        effective_from=date(2026, 8, 1),
        required_competencies=("im_injection",),
        required_supervision_role="physician",
        signed_by="dr_alvarez",
        signed_utc=datetime(2026, 7, 30, tzinfo=timezone.utc),
    )
    fields.update(overrides)
    return StandingOrder(**fields)


def test_r03_an_order_naming_no_competency_is_refused_at_publish():
    """Finding 3: `authorise()`'s competency loop does not run for an empty
    tuple, so `individual_competency` and `competency_current` pass by vacuum.
    A blank list on an epinephrine protocol put a six-week-old hire on the
    screen for IM epinephrine. Missing data must never widen the gate."""
    _, orders, _, _ = build()
    with pytest.raises(ValueError) as exc:
        orders.publish(_signed_order(
            order_id="so_epi_anaphylaxis",
            title="Administer intramuscular epinephrine for anaphylaxis",
            task_code="administer_epinephrine",
            required_competencies=(),
        ))
    assert "no required" in str(exc.value)
    assert "so_epi_anaphylaxis" not in orders.versions


def test_r03_the_blank_order_never_reaches_the_new_hires_screen(service):
    """The consequence the refusal prevents, asserted end to end: Dana holds
    vitals only, and must not gain a task by an order requiring nothing."""
    before = {o.order_id for o in service.available_orders("ma_dana", moment=NOW)}
    with pytest.raises(ValueError):
        service.orders.publish(_signed_order(
            order_id="so_epi_anaphylaxis", required_competencies=(),
        ))
    after = {o.order_id for o in service.available_orders("ma_dana", moment=NOW)}
    assert after == before
    assert "so_epi_anaphylaxis" not in after


def test_r04_a_back_dated_version_cannot_rewrite_which_text_was_in_force(audit):
    """Finding 4: `publish()` checked the version number but not the dates, so
    a v3 back-dated to before v2 began retroactively changed which protocol was
    in force on the day of an act already logged against v2 — text signed six
    weeks AFTER the injection."""
    rules, orders, competencies, roster = build()
    rules.review["owner"] = "dr_alvarez"
    service = DelegationService(
        rules=rules, orders=orders, competencies=competencies,
        roster=roster, audit=audit,
    )
    v2 = orders.current("so_immunize", on=NOW.date())
    service.execute(
        "ma_jess", "so_immunize", patient_id="p_rosa", moment=NOW,
        execution_id="ex_1",
    )
    logged = audit.delegation_evidence(staff_id="ma_jess")[0]
    assert int(logged["standing_order_version"]) == v2.version

    with pytest.raises(ValueError) as exc:
        orders.publish(StandingOrder(
            order_id="so_immunize", version=v2.version + 1,
            title=v2.title, task_code=v2.task_code,
            clinical_content="REVISED: do not administer without a second check.",
            delegating_physician_id="dr_alvarez",
            effective_from=v2.effective_from - timedelta(days=60),
            required_competencies=v2.required_competencies,
            required_supervision_role="physician",
            signed_by="dr_alvarez",
            # Signed before it takes effect, so the signature check is not what
            # refuses this: the ordering check is.
            signed_utc=datetime.combine(
                v2.effective_from - timedelta(days=61), time(9, 0),
                tzinfo=timezone.utc,
            ),
        ))
    assert "before v" in str(exc.value)

    # The act still resolves to the text its performer worked under.
    still = orders.current("so_immunize", on=NOW.date())
    assert still.version == int(logged["standing_order_version"])
    assert still.clinical_content == v2.clinical_content
    assert orders.version_at("so_immunize", v2.version).retired_on is None


def test_r04_a_version_cannot_be_in_force_before_its_own_signature():
    """The same inverted window from the other direction: an order effective
    before the physician signed it is a delegation not yet made."""
    _, orders, _, _ = build()
    with pytest.raises(ValueError) as exc:
        orders.publish(_signed_order(
            effective_from=date(2026, 1, 1),
            signed_utc=datetime(2026, 7, 30, tzinfo=timezone.utc),
        ))
    assert "predate" in str(exc.value)


def test_r08_an_extract_for_one_ma_does_not_disclose_another_employees_incident(service):
    """Finding 8: `unevidenced_supervision()` had no `staff_id`, so an extract
    requested for one MA returned every other employee's break-glass event —
    including the free-text justification, which is an HR record."""
    lunch = NOW.replace(hour=12, minute=30)
    service.execute(
        "ma_jess", "so_immunize", patient_id="p_rosa", moment=NOW,
        execution_id="ex_jess",
    )
    service.execute(
        "rn_paulo", "so_immunize", patient_id="p_theo", moment=lunch,
        execution_id="ex_paulo",
        break_glass_reason=(
            "Patient in respiratory distress; physician off site at lunch. "
            "Proceeded on verbal order, incident under HR review."
        ),
    )
    extract = service.audit_extract(staff_id="ma_jess", as_of=NOW.date())
    assert {r["staff_id"] for r in extract["unevidenced_detail"]} <= {"ma_jess"}
    assert all("HR review" not in (r["reason"] or "")
               for r in extract["unevidenced_detail"])

    # Paulo's own extract still shows it — this is scoping, not suppression.
    paulos = service.audit_extract(staff_id="rn_paulo", as_of=NOW.date())
    assert any(r["staff_id"] == "rn_paulo" for r in paulos["unevidenced_detail"])
    assert paulos["unevidenced"] >= 1


def test_r08_the_extract_scopes_executions_to_one_staff_member(service):
    """Mutation D9: the old fixture wrote exactly one execution to the whole
    log, so `executions == 1` passed whether or not the query was scoped. A
    second staff member's execution makes the assertion mean something."""
    service.execute(
        "ma_jess", "so_immunize", patient_id="p_rosa", moment=NOW,
        execution_id="ex_jess",
    )
    service.execute(
        "lpn_marta", "so_immunize", patient_id="p_theo",
        moment=NOW.replace(hour=11), execution_id="ex_marta",
        break_glass_reason=JUSTIFY,
    )
    extract = service.audit_extract(staff_id="ma_jess", as_of=NOW.date())
    assert extract["executions"] == 1
    assert [s["staff_id"] for s in extract["staff"]] == ["ma_jess"]
    assert {r["staff_id"] for r in extract["unevidenced_detail"]} <= {"ma_jess"}

    everyone = service.audit_extract(as_of=NOW.date())
    assert everyone["executions"] == 2


def test_r08_date_b_exists_and_bounds_both_executions_and_competencies(service):
    """Finding 8, second half: the docstring promised "between date A and date
    B" and there was no date B — counsel asking for Q2 got everything since Q2.
    `competency_records` was not date-filtered at all."""
    service.execute(
        "ma_jess", "so_immunize", patient_id="p_rosa", moment=NOW,
        execution_id="ex_jess",
    )
    window = service.audit_extract(
        staff_id="ma_jess",
        since=date(2026, 1, 1), until=date(2026, 6, 30),
        as_of=NOW.date(),
    )
    assert window["window"] == ["2026-01-01", "2026-06-30"]
    assert window["executions"] == 0          # the act is in August
    verified = [date.fromisoformat(r["verified_on"])
                for r in window["competency_records"]]
    assert verified, "the window should contain at least one record"
    assert all(date(2026, 1, 1) <= d <= date(2026, 6, 30) for d in verified)

    unbounded = service.audit_extract(staff_id="ma_jess", as_of=NOW.date())
    assert len(unbounded["competency_records"]) > len(window["competency_records"])


def test_r13_break_glass_does_not_perform_an_order_that_does_not_exist(service, audit):
    """Finding 13: break glass covers the requirements it is FOR — competency,
    supervision, an overdue review. A typo in an order id is not patient care
    being blocked; performing it recorded an act as completed against no
    protocol, no version, no supervisor and no competency."""
    with pytest.raises(BreakGlassRefused) as exc:
        service.execute(
            "ma_jess", "so_does_not_exist", patient_id="p1", moment=NOW,
            execution_id="ex_ghost", break_glass_reason=JUSTIFY,
        )
    assert "no standing order" in str(exc.value)
    assert audit.delegation_evidence() == []


def test_r13_break_glass_does_not_perform_an_unsigned_order(service, audit):
    """An unsigned protocol is not a delegation the physician has made, so
    there is nothing to break glass on. `publish()` refuses one; an import or
    migration can still put one in the register."""
    service.orders.versions["so_draft"] = [StandingOrder(
        order_id="so_draft", version=1, title="Draft protocol",
        task_code="administer_vaccine", clinical_content="not yet reviewed",
        delegating_physician_id="dr_alvarez", effective_from=date(2026, 1, 1),
        required_competencies=("im_injection",),
        required_supervision_role="physician",
    )]
    with pytest.raises(BreakGlassRefused) as exc:
        service.execute(
            "ma_jess", "so_draft", patient_id="p1", moment=NOW,
            execution_id="ex_draft", break_glass_reason=JUSTIFY,
        )
    assert "unsigned" in str(exc.value)
    assert audit.delegation_evidence() == []


def test_r13_no_act_is_performed_after_the_framework_sunsets(service):
    """Finding 13, second half: `certify()` enforced the sunset and nothing on
    the execution path called it, so the register went on authorising delegated
    acts under a repealed statute."""
    after = datetime(2027, 6, 1, 10, 30, tzinfo=timezone.utc)
    # A supervisor is on site, so nothing else is standing in the way: the
    # refusal has to come from the statute having lapsed.
    service.roster.record(RosterEntry(
        "dr_alvarez",
        datetime(2027, 6, 1, 8, tzinfo=timezone.utc),
        datetime(2027, 6, 1, 18, tzinfo=timezone.utc),
    ))
    assert service.authorise("ma_jess", "so_vitals", moment=after).authorised
    with pytest.raises(FrameworkSunset):
        service.execute(
            "ma_jess", "so_vitals", patient_id="p1", moment=after,
            execution_id="ex_post_sunset",
        )
    assert service.audit.delegation_evidence() == []


def test_r13_the_compliance_extract_refuses_to_cite_a_repealed_statute(service):
    """`audit_extract()` is the one-click compliance document. Naming a statute
    that no longer exists is worse than producing nothing: it looks like an
    answer."""
    service.execute(
        "ma_jess", "so_immunize", patient_id="p_rosa", moment=NOW,
        execution_id="ex_jess",
    )
    with pytest.raises(FrameworkSunset):
        service.audit_extract(staff_id="ma_jess", as_of=date(2027, 6, 1))
    # `as_of` defaults to `until`, so a Q3-2027 extract refuses on its own.
    with pytest.raises(FrameworkSunset):
        service.audit_extract(
            staff_id="ma_jess",
            since=date(2027, 7, 1), until=date(2027, 9, 30),
        )
    # Certified inside the framework's life, it still produces the document.
    ok = service.audit_extract(staff_id="ma_jess", as_of=NOW.date())
    assert ok["framework"] == "225 ILCS 60/54.2"
    assert ok["certified_on"] == NOW.date().isoformat()


def test_r13_an_unowned_rules_file_cannot_produce_a_compliance_document(service):
    """The same gate catches the other way this document lies: an extract
    asserting compliance with a framework nobody owns."""
    service.rules.review["owner"] = ""
    with pytest.raises(UnreviewedRules):
        service.audit_extract(staff_id="ma_jess", as_of=NOW.date())
