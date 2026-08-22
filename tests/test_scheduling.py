"""I-07 tests. Real database, real engine, real gateway, real threads.

Nothing in this file mocks logic that lives in the module under test. The
gateway is `LocalGateway`, which is a shipped transport, and the database is a
real SQLite file on disk — an in-memory database would make the concurrency
test meaningless, since each thread would get its own empty database.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from modules.scheduling import (
    AppointmentStatus,
    BackfillEngine,
    Channel,
    ConsentPurpose,
    Database,
    FrequencyCap,
    InboundAction,
    InboundRouter,
    LocalGateway,
    MessagePurpose,
    OfferOutcome,
    QuietHours,
    ReminderEngine,
    SendGate,
    VisitType,
    add_appointment,
    add_family,
    add_patient,
    add_provider,
    add_waitlist_entry,
    backfill_rate,
    classify_inbound,
    fill_time_stats,
    grant_consent,
    kpi_summary,
    message_funnel,
    no_show_rate,
    plan_reminders,
    rank_candidates,
    revoke_consent,
)
from modules.scheduling.models import age_months, iso, parse_iso, to_local

CHI = ZoneInfo("America/Chicago")


def local(y, m, d, hh=0, mm=0) -> datetime:
    """Build an aware practice-local datetime."""
    return datetime(y, m, d, hh, mm, tzinfo=CHI)


@pytest.fixture()
def db(tmp_path) -> Database:
    return Database(tmp_path / "sched.sqlite3")


@pytest.fixture()
def gw(tmp_path) -> LocalGateway:
    return LocalGateway(tmp_path / "outbox.jsonl")


def make_family(db, *, fid="fam1", phone="+18475550101", consent=True, granted=None):
    add_family(db, family_id=fid, display_name="Test Family", primary_phone=phone)
    if consent:
        grant_consent(
            db,
            family_id=fid,
            channel=Channel.SMS,
            purpose=ConsentPurpose.REMINDERS,
            granted=granted or local(2026, 1, 1, 9),
            capture_method="intake_form",
            capture_evidence="INTAKE-2026-0001",
            captured_by="frontdesk_02",
        )
    return fid


# ==========================================================================
# DST — the reminder offsets are elapsed time, not civil-calendar arithmetic
# ==========================================================================


def test_t48_reminder_across_fall_back_is_exactly_48_real_hours():
    """Nov 1 2026 is a 25-hour day in Chicago. A T-48h reminder for a Monday
    09:20 CST appointment must land at 10:20 CDT on the preceding Saturday.

    Naive civil arithmetic (`start_local - 2 days`) would produce 09:20 on the
    Saturday, which is 49 elapsed hours before the visit — an hour early, once
    a year, in a way nobody notices because the message still arrives.
    """
    appointment_start = local(2026, 11, 2, 9, 20)  # Monday, CST (UTC-6)
    appt = {
        "appointment_id": "a1",
        "patient_id": "p1",
        "visit_type": VisitType.WELL,
        "start_utc": iso(appointment_start),
    }
    planned = {p.purpose: p for p in plan_reminders(appt, now=local(2026, 10, 1), family_id="f")}
    t48 = planned[MessagePurpose.REMINDER_T48].planned_utc

    # Exactly 48 hours of elapsed time.
    assert appointment_start.astimezone(timezone.utc) - t48 == timedelta(hours=48)

    # Which is 10:20 local on Oct 31, in CDT, not 09:20.
    t48_local = to_local(t48)
    assert (t48_local.month, t48_local.day) == (10, 31)
    assert (t48_local.hour, t48_local.minute) == (10, 20)
    assert t48_local.tzname() == "CDT"
    assert t48_local.utcoffset() == timedelta(hours=-5)

    # And the appointment itself is in CST. The window straddles the change.
    assert appointment_start.astimezone(CHI).tzname() == "CST"


def test_t48_reminder_across_spring_forward_is_exactly_48_real_hours():
    """March 14 2027 is a 23-hour day. The same rule, the other direction:
    a Monday 09:20 CDT appointment gets its T-48h at 08:20 CST on the Saturday.
    """
    appointment_start = local(2027, 3, 15, 9, 20)  # Monday, CDT (UTC-5)
    appt = {
        "appointment_id": "a1",
        "patient_id": "p1",
        "visit_type": VisitType.WELL,
        "start_utc": iso(appointment_start),
    }
    planned = {p.purpose: p for p in plan_reminders(appt, now=local(2027, 2, 1), family_id="f")}
    t48 = planned[MessagePurpose.REMINDER_T48].planned_utc

    assert appointment_start.astimezone(timezone.utc) - t48 == timedelta(hours=48)
    t48_local = to_local(t48)
    assert (t48_local.month, t48_local.day) == (3, 13)
    assert (t48_local.hour, t48_local.minute) == (8, 20)
    assert t48_local.tzname() == "CST"


def test_t7_reminder_across_dst_keeps_elapsed_duration():
    """The seven-day touch spans the transition too, and is treated the same."""
    appointment_start = local(2026, 11, 4, 14, 0)
    appt = {
        "appointment_id": "a1",
        "patient_id": "p1",
        "visit_type": VisitType.WELL,
        "start_utc": iso(appointment_start),
    }
    planned = {p.purpose: p for p in plan_reminders(appt, now=local(2026, 10, 1), family_id="f")}
    t7 = planned[MessagePurpose.REMINDER_T7].planned_utc
    assert appointment_start.astimezone(timezone.utc) - t7 == timedelta(days=7)
    assert to_local(t7).hour == 15  # 14:00 CST is 15:00 CDT a week earlier


def test_quiet_hours_are_wall_clock_on_both_sides_of_a_transition():
    """21:00 means 21:00 to the family, whatever the UTC offset that week."""
    qh = QuietHours()
    assert qh.contains(local(2026, 7, 1, 22, 30)) is True  # CDT summer
    assert qh.contains(local(2026, 12, 1, 22, 30)) is True  # CST winter
    assert qh.contains(local(2026, 7, 1, 20, 59)) is False
    assert qh.contains(local(2026, 7, 1, 8, 0)) is False
    assert qh.contains(local(2026, 7, 1, 7, 59)) is True


def test_quiet_hours_defer_to_eight_local_next_morning():
    qh = QuietHours()
    late = local(2026, 11, 3, 22, 40)
    opened = qh.next_open(late)
    opened_local = to_local(opened)
    assert (opened_local.month, opened_local.day) == (11, 4)
    assert (opened_local.hour, opened_local.minute) == (8, 0)

    early = local(2026, 11, 4, 6, 15)
    opened_local = to_local(qh.next_open(early))
    assert (opened_local.month, opened_local.day) == (11, 4)
    assert opened_local.hour == 8

    ok = local(2026, 11, 4, 13, 0)
    assert qh.next_open(ok) == ok


def test_quiet_hours_deferral_lands_on_eight_am_on_the_spring_forward_day():
    """08:00 exists on a spring-forward day; the deferral must not drift."""
    qh = QuietHours()
    late = local(2027, 3, 13, 23, 30)  # CST, night before the change
    opened_local = to_local(qh.next_open(late))
    assert (opened_local.month, opened_local.day) == (3, 14)
    assert (opened_local.hour, opened_local.minute) == (8, 0)
    assert opened_local.tzname() == "CDT"


def test_age_months_is_computed_in_local_time():
    # 18:30 UTC on the day before a birthday is still "the day before" locally.
    assert age_months("2024-08-22", datetime(2026, 8, 21, 23, 0, tzinfo=timezone.utc)) == 23
    assert age_months("2024-08-22", local(2026, 8, 22, 9, 0)) == 24


# ==========================================================================
# Cadence
# ==========================================================================


def test_well_visit_gets_three_touches_sick_visit_gets_one():
    start = local(2026, 9, 15, 10, 0)
    well = plan_reminders(
        {"appointment_id": "a", "patient_id": "p", "visit_type": VisitType.WELL,
         "start_utc": iso(start)},
        now=local(2026, 9, 1),
        family_id="f",
    )
    sick = plan_reminders(
        {"appointment_id": "b", "patient_id": "p", "visit_type": VisitType.SICK,
         "start_utc": iso(start)},
        now=local(2026, 9, 1),
        family_id="f",
    )
    assert [p.purpose for p in well] == [
        MessagePurpose.REMINDER_T7,
        MessagePurpose.REMINDER_T48,
        MessagePurpose.REMINDER_T2,
    ]
    assert [p.purpose for p in sick] == [MessagePurpose.REMINDER_T2]


def test_touches_already_elapsed_are_marked_skipped_not_sent_late():
    start = local(2026, 9, 15, 10, 0)
    planned = plan_reminders(
        {"appointment_id": "a", "patient_id": "p", "visit_type": VisitType.WELL,
         "start_utc": iso(start)},
        now=local(2026, 9, 14, 12, 0),  # booked 22 hours out
        family_id="f",
    )
    by_purpose = {p.purpose: p for p in planned}
    assert by_purpose[MessagePurpose.REMINDER_T7].skipped is True
    assert by_purpose[MessagePurpose.REMINDER_T48].skipped is True
    assert by_purpose[MessagePurpose.REMINDER_T2].skipped is False


def test_reminder_dispatch_end_to_end(db, gw):
    fid = make_family(db)
    add_patient(db, patient_id="p1", family_id=fid, first_name="Maya", last_name="W",
                dob="2021-04-02")
    add_provider(db, provider_id="dr_ruiz", display_name="Dr. Ruiz")
    start = local(2026, 9, 15, 10, 0)
    appt = add_appointment(db, patient_id="p1", provider_id="dr_ruiz",
                           visit_type=VisitType.WELL, start=start, now=local(2026, 9, 1))
    engine = ReminderEngine(db, gw)
    engine.plan_appointment(appt, now=local(2026, 9, 1))

    # Nothing is due yet.
    assert engine.dispatch_due(now=local(2026, 9, 2)) == {}

    # T-7 becomes due.
    assert engine.dispatch_due(now=local(2026, 9, 8, 10, 30)) == {"sent": 1}
    body = gw.messages_for(MessagePurpose.REMINDER_T7)[0]["body"]
    assert "Maya" in body and "Dr. Ruiz" in body and "STOP" in body


def test_reminder_is_blocked_when_consent_is_missing(db, gw):
    fid = make_family(db, consent=False)
    add_patient(db, patient_id="p1", family_id=fid, first_name="Ana", last_name="B",
                dob="2020-01-05")
    add_provider(db, provider_id="dr_ruiz", display_name="Dr. Ruiz")
    appt = add_appointment(db, patient_id="p1", provider_id="dr_ruiz",
                           visit_type=VisitType.WELL, start=local(2026, 9, 15, 10, 0),
                           now=local(2026, 9, 1))
    engine = ReminderEngine(db, gw)
    engine.plan_appointment(appt, now=local(2026, 9, 1))
    assert engine.dispatch_due(now=local(2026, 9, 8, 10, 30)) == {"blocked": 1}
    assert gw.sent == []
    row = db.one("SELECT block_reason FROM message_log WHERE status='blocked'")
    assert row["block_reason"] == "no_consent"


def test_optout_between_planning_and_sending_is_honoured(db, gw):
    """The gate runs at dispatch, not at plan time. This is the TCPA case."""
    fid = make_family(db)
    add_patient(db, patient_id="p1", family_id=fid, first_name="Ana", last_name="B",
                dob="2020-01-05")
    add_provider(db, provider_id="dr_ruiz", display_name="Dr. Ruiz")
    appt = add_appointment(db, patient_id="p1", provider_id="dr_ruiz",
                           visit_type=VisitType.WELL, start=local(2026, 9, 15, 10, 0),
                           now=local(2026, 9, 1))
    engine = ReminderEngine(db, gw)
    engine.plan_appointment(appt, now=local(2026, 9, 1))
    revoke_consent(db, family_id=fid, revoked=local(2026, 9, 5))
    assert engine.dispatch_due(now=local(2026, 9, 8, 10, 30)) == {"blocked": 1}
    assert db.one("SELECT block_reason FROM message_log WHERE status='blocked'")[
        "block_reason"
    ] == "suppressed"


def test_suppression_is_checked_before_consent(db):
    """Gate ordering is load-bearing: a suppressed family with valid consent
    must report 'suppressed', not fall through to a consent check that passes.
    """
    fid = make_family(db)
    db.execute(
        "INSERT INTO suppression (suppression_id, family_id, channel, reason, created_utc)"
        " VALUES ('s1', ?, 'sms', 'staff', ?)",
        (fid, iso(local(2026, 9, 1))),
    )
    gate = SendGate(db)
    decision = gate.evaluate(
        family_id=fid, channel=Channel.SMS, purpose=MessagePurpose.REMINDER_T7,
        at=local(2026, 9, 8, 12, 0),
    )
    assert decision.allow is False
    assert decision.reason == "suppressed"


def test_quiet_hours_defer_then_send(db, gw):
    fid = make_family(db)
    add_patient(db, patient_id="p1", family_id=fid, first_name="Ana", last_name="B",
                dob="2020-01-05")
    add_provider(db, provider_id="dr_ruiz", display_name="Dr. Ruiz")
    # 22:00 appointment is unrealistic; use a start whose T-7 lands at night.
    start = local(2026, 9, 15, 22, 30)
    appt = add_appointment(db, patient_id="p1", provider_id="dr_ruiz",
                           visit_type=VisitType.WELL, start=start, now=local(2026, 9, 1))
    engine = ReminderEngine(db, gw)
    engine.plan_appointment(appt, now=local(2026, 9, 1))
    assert engine.dispatch_due(now=local(2026, 9, 8, 22, 30)) == {"deferred": 1}
    assert gw.sent == []
    row = db.one(
        "SELECT planned_utc, send_after_utc FROM message_log WHERE purpose='reminder_t7'"
    )
    assert to_local(parse_iso(row["send_after_utc"])).hour == 8
    # The original plan time survives the deferral -- it is the audit record of
    # what the cadence rule decided.
    assert to_local(parse_iso(row["planned_utc"])).hour == 22
    assert engine.dispatch_due(now=local(2026, 9, 9, 8, 5)) == {"sent": 1}


def test_t2_reminder_is_dropped_rather_than_deferred_past_the_visit(db, gw):
    fid = make_family(db)
    add_patient(db, patient_id="p1", family_id=fid, first_name="Ana", last_name="B",
                dob="2020-01-05")
    add_provider(db, provider_id="dr_ruiz", display_name="Dr. Ruiz")
    # Walk-in sick hours start early; T-2h for a 07:45 slot is 05:45, deep
    # inside quiet hours, and 08:00 is already after the visit began.
    start = local(2026, 9, 15, 7, 45)
    appt = add_appointment(db, patient_id="p1", provider_id="dr_ruiz",
                           visit_type=VisitType.SICK, start=start, now=local(2026, 9, 14))
    engine = ReminderEngine(db, gw)
    engine.plan_appointment(appt, now=local(2026, 9, 14))
    assert engine.dispatch_due(now=local(2026, 9, 15, 5, 45)) == {"skipped": 1}
    assert db.one("SELECT block_reason FROM message_log WHERE status='skipped'")[
        "block_reason"
    ] == "would_arrive_after_appointment"


def test_cancelled_appointment_stops_reminding(db, gw):
    fid = make_family(db)
    add_patient(db, patient_id="p1", family_id=fid, first_name="Ana", last_name="B",
                dob="2020-01-05")
    add_provider(db, provider_id="dr_ruiz", display_name="Dr. Ruiz")
    appt = add_appointment(db, patient_id="p1", provider_id="dr_ruiz",
                           visit_type=VisitType.WELL, start=local(2026, 9, 15, 10, 0),
                           now=local(2026, 9, 1))
    engine = ReminderEngine(db, gw)
    engine.plan_appointment(appt, now=local(2026, 9, 1))
    db.execute("UPDATE appointment SET status='cancelled' WHERE appointment_id=?", (appt,))
    assert engine.dispatch_due(now=local(2026, 9, 8, 12, 0)) == {"skipped": 1}


# ==========================================================================
# Frequency cap
# ==========================================================================


def test_frequency_cap_counts_per_family_per_rolling_week(db, gw):
    fid = make_family(db)
    add_patient(db, patient_id="p1", family_id=fid, first_name="Ana", last_name="B",
                dob="2020-01-05")
    cap = FrequencyCap(db, limits={"appointment": 2, "outreach": 1})
    now = local(2026, 9, 10, 12, 0)
    for i in range(2):
        db.execute(
            "INSERT INTO message_log (message_id, family_id, channel, purpose, template_id,"
            " planned_utc, sent_utc, status, to_address) VALUES (?,?,?,?,?,?,?,?,?)",
            (f"m{i}", fid, "sms", MessagePurpose.REMINDER_T7, "wv_t7_confirm",
             iso(now - timedelta(days=1)), iso(now - timedelta(days=1)), "sent", "+1"),
        )
    assert cap.count(fid, now) == 2
    assert cap.check(fid, now, MessagePurpose.REMINDER_T48).allowed is False
    # Outside the window it no longer counts.
    assert cap.count(fid, now + timedelta(days=8)) == 0


def test_outreach_tier_yields_to_appointment_tier(db):
    """I-02 recall must lose to I-07 reminders, not the other way round."""
    fid = make_family(db)
    cap = FrequencyCap(db)
    now = local(2026, 9, 10, 12, 0)
    for i in range(3):
        db.execute(
            "INSERT INTO message_log (message_id, family_id, channel, purpose, template_id,"
            " planned_utc, sent_utc, status, to_address) VALUES (?,?,?,?,?,?,?,?,?)",
            (f"m{i}", fid, "sms", MessagePurpose.REMINDER_T7, "wv_t7_confirm",
             iso(now), iso(now), "sent", "+1"),
        )
    assert cap.check(fid, now, MessagePurpose.REMINDER_T48,
                     tier=FrequencyCap.TIER_APPOINTMENT).allowed is True
    assert cap.check(fid, now, "recall_immunization",
                     tier=FrequencyCap.TIER_OUTREACH).allowed is False


def test_transactional_messages_bypass_the_cap(db):
    fid = make_family(db)
    cap = FrequencyCap(db, limits={"appointment": 0, "outreach": 0})
    now = local(2026, 9, 10, 12, 0)
    assert cap.check(fid, now, MessagePurpose.REMINDER_T7).allowed is False
    assert cap.check(fid, now, MessagePurpose.CANCEL_CONFIRMATION).allowed is True
    assert cap.check(fid, now, MessagePurpose.BACKFILL_WON).allowed is True


def test_cancel_confirmation_is_sent_at_night(db, gw):
    """A parent cancelling at 22:40 gets an immediate acknowledgement."""
    fid = make_family(db)
    add_patient(db, patient_id="p1", family_id=fid, first_name="Ana", last_name="B",
                dob="2020-01-05")
    add_provider(db, provider_id="dr_ruiz", display_name="Dr. Ruiz")
    appt = add_appointment(db, patient_id="p1", provider_id="dr_ruiz",
                           visit_type=VisitType.WELL, start=local(2026, 9, 16, 9, 20),
                           now=local(2026, 9, 1))
    engine = BackfillEngine(db, gw)
    engine.release_slot(appt, now=local(2026, 9, 15, 22, 40))
    assert len(gw.messages_for(MessagePurpose.CANCEL_CONFIRMATION)) == 1


# ==========================================================================
# Backfill — ranking, eligibility, and the atomic booking guarantee
# ==========================================================================


def build_backfill_scenario(db, gw, *, n_candidates=5, slot_start=None):
    slot_start = slot_start or local(2026, 9, 20, 10, 0)
    add_provider(db, provider_id="dr_ruiz", display_name="Dr. Ruiz",
                 min_age_months=0, max_age_months=300)
    holder_family = make_family(db, fid="fam_holder", phone="+18475550100")
    add_patient(db, patient_id="p_holder", family_id=holder_family, first_name="Holder",
                last_name="H", dob="2019-05-05")
    appt = add_appointment(db, patient_id="p_holder", provider_id="dr_ruiz",
                           visit_type=VisitType.WELL, start=slot_start,
                           now=local(2026, 9, 1))
    entries = []
    for i in range(n_candidates):
        fid = make_family(db, fid=f"fam{i}", phone=f"+1847555{2000 + i:04d}")
        add_patient(db, patient_id=f"p{i}", family_id=fid, first_name=f"Kid{i}",
                    last_name="X", dob="2020-06-01")
        entries.append(
            add_waitlist_entry(
                db,
                patient_id=f"p{i}",
                visit_type=VisitType.WELL,
                earliest_ok=local(2026, 9, 1),
                latest_ok=local(2026, 10, 1),
                added=local(2026, 8, 1) + timedelta(days=i),
                entry_id=f"wl{i}",
            )
        )
    return appt, entries, slot_start


def test_release_creates_a_backfillable_slot_and_blast_offers_top_n(db, gw):
    appt, entries, _ = build_backfill_scenario(db, gw, n_candidates=7)
    engine = BackfillEngine(db, gw)
    now = local(2026, 9, 19, 9, 0)
    release_id = engine.release_slot(appt, now=now)
    assert release_id is not None
    offers = engine.blast(release_id, now=now)
    assert len(offers) == 5  # README blast size
    assert len(gw.messages_for(MessagePurpose.BACKFILL_OFFER)) == 5


def test_ranking_is_priority_then_longest_wait(db, gw):
    appt, entries, _ = build_backfill_scenario(db, gw, n_candidates=4)
    db.execute("UPDATE waitlist_entry SET priority = 5 WHERE entry_id = 'wl3'")
    engine = BackfillEngine(db, gw)
    now = local(2026, 9, 19, 9, 0)
    release_id = engine.release_slot(appt, now=now)
    ranked = engine.eligible_candidates(release_id, now=now)
    assert ranked[0].entry_id == "wl3"            # highest priority wins
    assert ranked[1].entry_id == "wl0"            # then longest wait
    assert [c.entry_id for c in ranked] == ["wl3", "wl0", "wl1", "wl2"]


def test_rank_candidates_is_deterministic_on_full_ties():
    from modules.scheduling.backfill import Candidate

    base = dict(patient_id="p", family_id="f", priority=0,
                added_utc=local(2026, 8, 1), wait_seconds=100.0,
                provider_requested=False, age_fit=True, notify_channel="sms",
                to_address="+1", first_name="A")
    a = Candidate(entry_id="wl_b", **base)
    b = Candidate(entry_id="wl_a", **base)
    assert [c.entry_id for c in rank_candidates([a, b])] == ["wl_a", "wl_b"]


def test_age_inappropriate_patient_is_not_offered(db, gw):
    add_provider(db, provider_id="dr_okafor", display_name="Dr. Okafor",
                 min_age_months=24, max_age_months=300)
    holder = make_family(db, fid="fam_holder", phone="+18475550100")
    add_patient(db, patient_id="p_holder", family_id=holder, first_name="H", last_name="H",
                dob="2019-01-01")
    slot_start = local(2026, 9, 20, 10, 0)
    appt = add_appointment(db, patient_id="p_holder", provider_id="dr_okafor",
                           visit_type=VisitType.WELL, start=slot_start, now=local(2026, 9, 1))
    infant_family = make_family(db, fid="fam_infant", phone="+18475552001")
    add_patient(db, patient_id="p_infant", family_id=infant_family, first_name="Baby",
                last_name="X", dob="2026-05-01")  # ~4 months at slot time
    add_waitlist_entry(db, patient_id="p_infant", visit_type=VisitType.WELL,
                       earliest_ok=local(2026, 9, 1), latest_ok=local(2026, 10, 1),
                       added=local(2026, 8, 1), entry_id="wl_infant")
    older_family = make_family(db, fid="fam_older", phone="+18475552002")
    add_patient(db, patient_id="p_older", family_id=older_family, first_name="Kid",
                last_name="Y", dob="2019-06-01")
    add_waitlist_entry(db, patient_id="p_older", visit_type=VisitType.WELL,
                       earliest_ok=local(2026, 9, 1), latest_ok=local(2026, 10, 1),
                       added=local(2026, 8, 15), entry_id="wl_older")

    engine = BackfillEngine(db, gw)
    now = local(2026, 9, 19, 9, 0)
    release_id = engine.release_slot(appt, now=now)
    ranked = engine.eligible_candidates(release_id, now=now)
    assert [c.entry_id for c in ranked] == ["wl_older"]


def test_specific_provider_request_excludes_other_providers(db, gw):
    appt, _, _ = build_backfill_scenario(db, gw, n_candidates=2)
    db.execute("UPDATE waitlist_entry SET desired_provider='dr_okafor' WHERE entry_id='wl0'")
    engine = BackfillEngine(db, gw)
    now = local(2026, 9, 19, 9, 0)
    release_id = engine.release_slot(appt, now=now)
    assert [c.entry_id for c in engine.eligible_candidates(release_id, now=now)] == ["wl1"]


def test_visit_type_mismatch_and_date_window_are_hard_filters(db, gw):
    appt, _, _ = build_backfill_scenario(db, gw, n_candidates=3)
    db.execute("UPDATE waitlist_entry SET visit_type='sick' WHERE entry_id='wl0'")
    db.execute(
        "UPDATE waitlist_entry SET latest_ok_utc=? WHERE entry_id='wl1'",
        (iso(local(2026, 9, 5)),),
    )
    engine = BackfillEngine(db, gw)
    now = local(2026, 9, 19, 9, 0)
    release_id = engine.release_slot(appt, now=now)
    assert [c.entry_id for c in engine.eligible_candidates(release_id, now=now)] == ["wl2"]


def test_unreachable_family_is_not_a_candidate(db, gw):
    appt, _, _ = build_backfill_scenario(db, gw, n_candidates=3)
    revoke_consent(db, family_id="fam0", revoked=local(2026, 9, 2))
    engine = BackfillEngine(db, gw)
    now = local(2026, 9, 19, 9, 0)
    release_id = engine.release_slot(appt, now=now)
    assert [c.entry_id for c in engine.eligible_candidates(release_id, now=now)] == ["wl1", "wl2"]


def test_patient_with_a_conflicting_appointment_is_not_offered(db, gw):
    appt, _, slot_start = build_backfill_scenario(db, gw, n_candidates=2)
    add_appointment(db, patient_id="p0", provider_id="dr_ruiz", visit_type=VisitType.SICK,
                    start=slot_start + timedelta(minutes=10), duration_minutes=20,
                    now=local(2026, 9, 10))
    engine = BackfillEngine(db, gw)
    now = local(2026, 9, 19, 9, 0)
    release_id = engine.release_slot(appt, now=now)
    assert [c.entry_id for c in engine.eligible_candidates(release_id, now=now)] == ["wl1"]


def test_slot_too_close_to_start_is_cancelled_but_not_blasted(db, gw):
    appt, _, slot_start = build_backfill_scenario(db, gw, n_candidates=2)
    engine = BackfillEngine(db, gw)
    release_id = engine.release_slot(appt, now=slot_start - timedelta(minutes=45))
    assert release_id is None
    assert db.one("SELECT status FROM appointment WHERE appointment_id=?", (appt,))[
        "status"
    ] == AppointmentStatus.CANCELLED
    assert db.all("SELECT * FROM slot_release") == []


# --------------------------------------------------------------------------
# THE ATOMICITY TEST
# --------------------------------------------------------------------------


def test_five_simultaneous_accepts_produce_exactly_one_winner(db, gw):
    """Five threads accept the same slot at the same instant. Exactly one wins.

    This is the test README I-07 asks for: "Blasting a slot to five people and
    letting two book it is worse than not blasting it at all." A barrier makes
    all five threads enter accept_offer as close to simultaneously as the
    scheduler allows, so the race is real rather than staged.
    """
    appt, _, _ = build_backfill_scenario(db, gw, n_candidates=5)
    engine = BackfillEngine(db, gw)
    now = local(2026, 9, 19, 9, 0)
    release_id = engine.release_slot(appt, now=now)
    offers = engine.blast(release_id, now=now)
    assert len(offers) == 5

    barrier = threading.Barrier(len(offers))
    results: list = [None] * len(offers)
    errors: list = []

    def accept(index: int, offer_id: str) -> None:
        try:
            barrier.wait(timeout=10)
            results[index] = engine.accept_offer(offer_id, now=now + timedelta(seconds=30))
        except Exception as exc:  # pragma: no cover - only on a genuine failure
            errors.append(exc)

    threads = [
        threading.Thread(target=accept, args=(i, offer_id))
        for i, offer_id in enumerate(offers)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, errors
    winners = [r for r in results if r is not None and r.won]
    losers = [r for r in results if r is not None and not r.won]

    assert len(winners) == 1, f"expected exactly one winner, got {len(winners)}"
    assert len(losers) == 4
    assert all(r.outcome == OfferOutcome.LOST for r in losers)

    # And the database agrees, from three independent angles.
    booked = db.all(
        "SELECT * FROM appointment WHERE filled_from_release = ?", (release_id,)
    )
    assert len(booked) == 1
    release = db.one("SELECT * FROM slot_release WHERE release_id = ?", (release_id,))
    assert release["filled_utc"] is not None
    assert release["filled_appointment"] == booked[0]["appointment_id"]
    outcomes = db.all(
        "SELECT outcome, COUNT(*) c FROM backfill_offer WHERE release_id = ? GROUP BY outcome",
        (release_id,),
    )
    assert {row["outcome"]: row["c"] for row in outcomes} == {
        OfferOutcome.ACCEPTED: 1,
        OfferOutcome.LOST: 4,
    }
    assert len(gw.messages_for(MessagePurpose.BACKFILL_WON)) == 1
    assert len(gw.messages_for(MessagePurpose.BACKFILL_LOST)) == 4


def test_concurrent_accepts_are_stable_across_many_rounds(db, gw):
    """Ten independent slots, five racing threads each. Still exactly one winner.

    A single pass of the race can get lucky with thread scheduling. Repeating it
    over fresh releases is what turns "did not double-book once" into evidence.
    """
    add_provider(db, provider_id="dr_ruiz", display_name="Dr. Ruiz")
    holder = make_family(db, fid="fam_holder", phone="+18475550100")
    add_patient(db, patient_id="p_holder", family_id=holder, first_name="H", last_name="H",
                dob="2019-05-05")

    engine = BackfillEngine(db, gw)
    now = local(2026, 9, 19, 9, 0)
    for round_no in range(10):
        slot_start = local(2026, 9, 25, 9, 0) + timedelta(hours=round_no)
        appt = add_appointment(db, patient_id="p_holder", provider_id="dr_ruiz",
                               visit_type=VisitType.WELL, start=slot_start,
                               now=local(2026, 9, 1))
        # Fresh families each round. Reusing five families across ten rounds
        # would exhaust their weekly message budget by round six -- correct
        # behaviour, tested separately, but not what this test is measuring.
        for i in range(5):
            fid = make_family(db, fid=f"fam_{round_no}_{i}",
                              phone=f"+1847{round_no:02d}55{3000 + i:04d}")
            add_patient(db, patient_id=f"p_{round_no}_{i}", family_id=fid,
                        first_name=f"Kid{i}", last_name="X", dob="2020-06-01")
            add_waitlist_entry(db, patient_id=f"p_{round_no}_{i}", visit_type=VisitType.WELL,
                               earliest_ok=local(2026, 9, 1), latest_ok=local(2026, 10, 15),
                               added=local(2026, 8, 1) + timedelta(days=i),
                               entry_id=f"wl_{round_no}_{i}")
        release_id = engine.release_slot(appt, now=now)
        offers = engine.blast(release_id, now=now)
        assert len(offers) == 5

        barrier = threading.Barrier(5)
        outcomes: list = [None] * 5

        def accept(index: int, offer_id: str) -> None:
            barrier.wait(timeout=10)
            outcomes[index] = engine.accept_offer(offer_id, now=now + timedelta(seconds=5))

        threads = [threading.Thread(target=accept, args=(i, o)) for i, o in enumerate(offers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert sum(1 for r in outcomes if r and r.won) == 1, f"round {round_no}"
        booked = db.all(
            "SELECT appointment_id FROM appointment WHERE filled_from_release = ?",
            (release_id,),
        )
        assert len(booked) == 1, f"round {round_no}: {len(booked)} bookings"
        # Reset the waitlist for the next round; the winner is now 'booked'.
        db.execute("UPDATE waitlist_entry SET status='withdrawn' WHERE status='active'")


def test_backfill_offers_are_throttled_by_the_weekly_cap(db, gw):
    """A family on the waitlist is not blasted without limit.

    Six offers in a rolling week is the tier allowance; the seventh release
    finds them ineligible. Without this, a family that keeps losing races gets
    a text every time any slot anywhere opens up, which is the message-fatigue
    failure in README R-10 and the fastest route to a STOP.
    """
    add_provider(db, provider_id="dr_ruiz", display_name="Dr. Ruiz")
    holder = make_family(db, fid="fam_holder", phone="+18475550100")
    add_patient(db, patient_id="p_holder", family_id=holder, first_name="H", last_name="H",
                dob="2019-05-05")
    fid = make_family(db, fid="fam_keen", phone="+18475552222")
    add_patient(db, patient_id="p_keen", family_id=fid, first_name="Keen", last_name="X",
                dob="2020-06-01")

    engine = BackfillEngine(db, gw)
    now = local(2026, 9, 19, 9, 0)
    limit = FrequencyCap.DEFAULT_LIMITS["waitlist"]
    for round_no in range(limit + 1):
        add_waitlist_entry(db, patient_id="p_keen", visit_type=VisitType.WELL,
                           earliest_ok=local(2026, 9, 1), latest_ok=local(2026, 10, 15),
                           added=local(2026, 8, 1), entry_id=f"wl_keen_{round_no}")
        appt = add_appointment(db, patient_id="p_holder", provider_id="dr_ruiz",
                               visit_type=VisitType.WELL,
                               start=local(2026, 9, 26, 9, 0) + timedelta(hours=round_no),
                               now=local(2026, 9, 1))
        release_id = engine.release_slot(appt, now=now)
        offers = engine.blast(release_id, now=now)
        if round_no < limit:
            assert len(offers) == 1, f"round {round_no}"
        else:
            assert offers == [], "the seventh offer in a week must be withheld"
        db.execute("UPDATE waitlist_entry SET status='withdrawn' WHERE status='active'")

    assert len(gw.messages_for(MessagePurpose.BACKFILL_OFFER)) == limit


def test_repeated_accept_of_a_won_offer_is_idempotent(db, gw):
    appt, _, _ = build_backfill_scenario(db, gw, n_candidates=3)
    engine = BackfillEngine(db, gw)
    now = local(2026, 9, 19, 9, 0)
    release_id = engine.release_slot(appt, now=now)
    offers = engine.blast(release_id, now=now)
    first = engine.accept_offer(offers[0], now=now)
    second = engine.accept_offer(offers[0], now=now)
    assert first.won is True
    assert second.won is False
    assert second.reason == "offer_not_pending"
    assert len(db.all("SELECT * FROM appointment WHERE booking_source='backfill'")) == 1


def test_expired_offer_cannot_be_accepted(db, gw):
    appt, _, _ = build_backfill_scenario(db, gw, n_candidates=2)
    engine = BackfillEngine(db, gw)
    now = local(2026, 9, 19, 9, 0)
    release_id = engine.release_slot(appt, now=now)
    offers = engine.blast(release_id, now=now)
    result = engine.accept_offer(offers[0], now=now + timedelta(hours=2))
    assert result.won is False
    assert result.outcome == OfferOutcome.EXPIRED
    assert db.one("SELECT filled_utc FROM slot_release WHERE release_id=?", (release_id,))[
        "filled_utc"
    ] is None


def test_decline_leaves_the_slot_open_for_others(db, gw):
    appt, _, _ = build_backfill_scenario(db, gw, n_candidates=3)
    engine = BackfillEngine(db, gw)
    now = local(2026, 9, 19, 9, 0)
    release_id = engine.release_slot(appt, now=now)
    offers = engine.blast(release_id, now=now)
    engine.decline_offer(offers[0], now=now)
    assert engine.accept_offer(offers[1], now=now).won is True


def test_blast_with_no_eligible_candidates_closes_the_release(db, gw):
    appt, _, _ = build_backfill_scenario(db, gw, n_candidates=0)
    engine = BackfillEngine(db, gw)
    now = local(2026, 9, 19, 9, 0)
    release_id = engine.release_slot(appt, now=now)
    assert engine.blast(release_id, now=now) == []
    row = db.one("SELECT closed_utc, close_reason FROM slot_release WHERE release_id=?",
                 (release_id,))
    assert row["close_reason"] == "no_eligible_candidates"


def test_overnight_cancellation_holds_the_slot_until_quiet_hours_end(db, gw):
    """A 21:40 cancellation must not blast at 21:40, and must not be discarded.

    This is the README's headline after-hours scenario. Closing the release at
    night because "nobody is eligible right now" would throw away precisely the
    slots this initiative exists to recover.
    """
    appt, _, slot_start = build_backfill_scenario(
        db, gw, n_candidates=3, slot_start=local(2026, 9, 22, 14, 20)
    )
    engine = BackfillEngine(db, gw)
    evening = local(2026, 9, 21, 21, 40)
    release_id = engine.release_slot(appt, now=evening)
    assert engine.blast(release_id, now=evening) == []
    assert gw.messages_for(MessagePurpose.BACKFILL_OFFER) == []

    row = db.one("SELECT closed_utc FROM slot_release WHERE release_id=?", (release_id,))
    assert row["closed_utc"] is None, "the release must stay open overnight"

    morning = local(2026, 9, 22, 8, 0)
    assert engine.sweep_open_releases(now=morning) == {
        "blasted": 1, "skipped": 0, "too_close": 0, "offers": 3
    }
    assert len(gw.messages_for(MessagePurpose.BACKFILL_OFFER)) == 3


def test_early_morning_slot_cancelled_overnight_is_closed_honestly(db, gw):
    """An 09:20 slot cannot be refilled once quiet hours end at 08:00.

    Minimum lead time is two hours, so the morning sweep would arrive too late.
    The release is closed with a reason that names the trade-off rather than
    sitting open forever pretending it might still fill.
    """
    appt, _, _ = build_backfill_scenario(
        db, gw, n_candidates=3, slot_start=local(2026, 9, 22, 9, 20)
    )
    engine = BackfillEngine(db, gw)
    evening = local(2026, 9, 21, 21, 40)
    release_id = engine.release_slot(appt, now=evening)
    assert engine.blast(release_id, now=evening) == []
    row = db.one("SELECT close_reason FROM slot_release WHERE release_id=?", (release_id,))
    assert row["close_reason"] == "quiet_hours_no_time_to_fill"
    # The cancellation itself still happened, which is most of the value.
    assert db.one("SELECT status FROM appointment WHERE appointment_id=?", (appt,))[
        "status"
    ] == AppointmentStatus.RELEASED
    assert len(gw.messages_for(MessagePurpose.CANCEL_CONFIRMATION)) == 1


def test_sweep_does_not_retext_families_already_offered(db, gw):
    """A second sweep reaches the next tier, not the same five families again."""
    appt, _, _ = build_backfill_scenario(db, gw, n_candidates=7)
    engine = BackfillEngine(db, gw, blast_size=3)
    now = local(2026, 9, 19, 9, 0)
    release_id = engine.release_slot(appt, now=now)
    first = engine.blast(release_id, now=now)
    assert len(first) == 3

    # Nothing responded; the offers expire and the sweep tries the next tier.
    later = now + timedelta(minutes=25)
    assert engine.expire_stale_offers(now=later) == 3
    engine.sweep_open_releases(now=later)
    offered = db.all(
        "SELECT DISTINCT patient_id FROM backfill_offer WHERE release_id=?", (release_id,)
    )
    assert len(offered) == 6
    assert len(gw.messages_for(MessagePurpose.BACKFILL_OFFER)) == 6


def test_sweep_counts_slots_it_could_not_reach_in_time(db, gw):
    appt, _, _ = build_backfill_scenario(
        db, gw, n_candidates=2, slot_start=local(2026, 9, 22, 9, 20)
    )
    engine = BackfillEngine(db, gw)
    release_id = engine.release_slot(appt, now=local(2026, 9, 21, 20, 0))
    engine.blast(release_id, now=local(2026, 9, 21, 20, 0))
    # Fill it back to open so the sweep sees a live release too close to start.
    db.execute("UPDATE slot_release SET closed_utc=NULL, close_reason=NULL")
    db.execute("UPDATE backfill_offer SET outcome='expired'")
    assert engine.sweep_open_releases(now=local(2026, 9, 22, 8, 30))["too_close"] == 1


def test_winner_is_removed_from_the_waitlist(db, gw):
    appt, _, _ = build_backfill_scenario(db, gw, n_candidates=2)
    engine = BackfillEngine(db, gw)
    now = local(2026, 9, 19, 9, 0)
    release_id = engine.release_slot(appt, now=now)
    offers = engine.blast(release_id, now=now)
    result = engine.accept_offer(offers[0], now=now)
    entry = db.one(
        "SELECT status, booked_appointment FROM waitlist_entry WHERE entry_id="
        "(SELECT entry_id FROM backfill_offer WHERE offer_id = ?)",
        (offers[0],),
    )
    assert entry["status"] == "booked"
    assert entry["booked_appointment"] == result.appointment_id


# ==========================================================================
# Inbound
# ==========================================================================


def test_classify_inbound_taps_and_keywords():
    assert classify_inbound({"path": "/a/off_1"}).action == InboundAction.ACCEPT_OFFER
    assert classify_inbound({"path": "/d/off_1"}).action == InboundAction.DECLINE_OFFER
    assert classify_inbound({"path": "/c/msg_1"}).action == InboundAction.CONFIRM
    assert classify_inbound({"path": "/x/msg_1"}).action == InboundAction.CANCEL
    assert classify_inbound({"body": "STOP"}).action == InboundAction.OPT_OUT
    assert classify_inbound({"body": " stop "}).action == InboundAction.OPT_OUT
    assert classify_inbound({"body": "Start"}).action == InboundAction.OPT_IN
    assert classify_inbound({"body": "HELP"}).action == InboundAction.HELP


def test_free_text_goes_to_a_human_and_keeps_no_body():
    intent = classify_inbound({"body": "can we move to next Tuesday? Maya has a fever"})
    assert intent.action == InboundAction.HUMAN
    assert "Maya" not in str(intent.metadata)
    assert intent.raw_length > 0


def test_tap_beats_body_text():
    intent = classify_inbound({"path": "/c/msg_1", "body": "STOP"})
    assert intent.action == InboundAction.CONFIRM


def test_stop_keyword_suppresses_then_acknowledges(db, gw):
    fid = make_family(db, phone="+18475550101")
    add_patient(db, patient_id="p1", family_id=fid, first_name="Ana", last_name="B",
                dob="2020-01-05")
    router = InboundRouter(db, gw, BackfillEngine(db, gw))
    now = local(2026, 9, 10, 20, 0)
    result = router.handle({"body": "STOP", "from": "+18475550101"}, now=now)
    assert result.handled is True
    assert SendGate(db).is_suppressed(fid, Channel.SMS) is True
    consent_row = db.one("SELECT revoked_utc, revocation_method FROM consent WHERE family_id=?",
                         (fid,))
    assert consent_row["revoked_utc"] is not None
    assert consent_row["revocation_method"] == "stop_keyword"
    assert len(gw.messages_for(MessagePurpose.OPTOUT_CONFIRMATION)) == 1


def test_stop_from_unknown_number_is_still_suppressed(db, gw):
    router = InboundRouter(db, gw, BackfillEngine(db, gw))
    result = router.handle({"body": "STOP", "from": "+15555550199"}, now=local(2026, 9, 10, 20, 0))
    assert result.detail == "suppressed_unmatched_number"
    assert db.one("SELECT 1 AS hit FROM suppression WHERE family_id LIKE 'unmatched:%'") is not None


def test_start_keyword_restores_consent_with_evidence(db, gw):
    fid = make_family(db, phone="+18475550101")
    router = InboundRouter(db, gw, BackfillEngine(db, gw))
    router.handle({"body": "STOP", "from": "+18475550101"}, now=local(2026, 9, 10, 20, 0))
    router.handle({"body": "START", "from": "+18475550101"}, now=local(2026, 9, 11, 9, 0))
    gate = SendGate(db)
    assert gate.is_suppressed(fid, Channel.SMS) is False
    assert gate.has_consent(fid, Channel.SMS) is True
    row = db.one(
        "SELECT capture_method, capture_evidence FROM consent WHERE family_id=?"
        " AND revoked_utc IS NULL", (fid,)
    )
    assert row["capture_method"] == "sms_double_optin"
    assert "+18475550101" in row["capture_evidence"]


def test_cancel_tap_releases_and_blasts(db, gw):
    appt, _, _ = build_backfill_scenario(db, gw, n_candidates=3)
    engine = ReminderEngine(db, gw)
    engine.plan_appointment(appt, now=local(2026, 9, 1))
    message_id = db.one(
        "SELECT message_id FROM message_log WHERE appointment_id=? AND purpose='reminder_t7'",
        (appt,),
    )["message_id"]
    router = InboundRouter(db, gw, BackfillEngine(db, gw))
    result = router.handle(
        {"path": f"/x/{message_id}", "from": "+18475550100"}, now=local(2026, 9, 19, 20, 30)
    )
    assert result.detail == "cancelled_and_blasted"
    assert len(gw.messages_for(MessagePurpose.BACKFILL_OFFER)) == 3
    assert db.one("SELECT status FROM appointment WHERE appointment_id=?", (appt,))[
        "status"
    ] == AppointmentStatus.RELEASED


def test_confirm_tap_marks_the_appointment_confirmed(db, gw):
    fid = make_family(db)
    add_patient(db, patient_id="p1", family_id=fid, first_name="Ana", last_name="B",
                dob="2020-01-05")
    add_provider(db, provider_id="dr_ruiz", display_name="Dr. Ruiz")
    appt = add_appointment(db, patient_id="p1", provider_id="dr_ruiz",
                           visit_type=VisitType.WELL, start=local(2026, 9, 15, 10, 0),
                           now=local(2026, 9, 1))
    engine = ReminderEngine(db, gw)
    engine.plan_appointment(appt, now=local(2026, 9, 1))
    mid = db.one("SELECT message_id FROM message_log WHERE appointment_id=?", (appt,))[
        "message_id"
    ]
    router = InboundRouter(db, gw, BackfillEngine(db, gw))
    router.handle({"path": f"/c/{mid}"}, now=local(2026, 9, 9, 9, 0))
    assert db.one("SELECT status FROM appointment WHERE appointment_id=?", (appt,))[
        "status"
    ] == AppointmentStatus.CONFIRMED


def test_accept_tap_books_the_slot(db, gw):
    appt, _, _ = build_backfill_scenario(db, gw, n_candidates=3)
    engine = BackfillEngine(db, gw)
    now = local(2026, 9, 19, 9, 0)
    release_id = engine.release_slot(appt, now=now)
    offers = engine.blast(release_id, now=now)
    router = InboundRouter(db, gw, engine)
    result = router.handle({"path": f"/a/{offers[1]}"}, now=now + timedelta(minutes=1))
    assert result.handled is True
    assert db.one("SELECT filled_utc FROM slot_release WHERE release_id=?", (release_id,))[
        "filled_utc"
    ] is not None


# ==========================================================================
# Metrics
# ==========================================================================


def seed_metrics(db, gw):
    add_provider(db, provider_id="dr_ruiz", display_name="Dr. Ruiz")
    add_provider(db, provider_id="dr_okafor", display_name="Dr. Okafor")
    fid = make_family(db)
    add_patient(db, patient_id="p1", family_id=fid, first_name="Ana", last_name="B",
                dob="2020-01-05")
    base = local(2026, 8, 3, 9, 0)
    # dr_ruiz: 8 completed, 2 no-show. dr_okafor: 9 completed, 1 no-show.
    for i in range(10):
        add_appointment(db, patient_id="p1", provider_id="dr_ruiz", visit_type=VisitType.WELL,
                        start=base + timedelta(days=i),
                        status=AppointmentStatus.NO_SHOW if i < 2 else AppointmentStatus.COMPLETED,
                        now=base - timedelta(days=14))
    for i in range(10):
        add_appointment(db, patient_id="p1", provider_id="dr_okafor", visit_type=VisitType.SICK,
                        start=base + timedelta(days=i, hours=2),
                        status=AppointmentStatus.NO_SHOW if i < 1 else AppointmentStatus.COMPLETED,
                        now=base - timedelta(days=1))
    return fid, base


def test_no_show_rate_overall_and_by_provider(db, gw):
    seed_metrics(db, gw)
    overall = no_show_rate(db)["overall"]
    assert overall["denominator"] == 20
    assert overall["rate"] == pytest.approx(0.15)
    grouped = no_show_rate(db, group_by=["provider_id"])["groups"]
    assert grouped["dr_ruiz"]["rate"] == pytest.approx(0.2)
    assert grouped["dr_okafor"]["rate"] == pytest.approx(0.1)


def test_no_show_rate_excludes_cancellations(db, gw):
    _, base = seed_metrics(db, gw)
    add_appointment(db, patient_id="p1", provider_id="dr_ruiz", visit_type=VisitType.WELL,
                    start=base + timedelta(days=20), status=AppointmentStatus.CANCELLED,
                    now=base)
    assert no_show_rate(db)["overall"]["denominator"] == 20


def test_no_show_rate_by_day_of_week_uses_local_days(db, gw):
    seed_metrics(db, gw)
    groups = no_show_rate(db, group_by=["dow"])["groups"]
    assert sum(g["denominator"] for g in groups.values()) == 20
    assert all(len(k) == 3 for k in groups)


def test_backfill_rate_and_fill_time(db, gw):
    appt, _, _ = build_backfill_scenario(db, gw, n_candidates=3)
    engine = BackfillEngine(db, gw)
    now = local(2026, 9, 19, 9, 0)
    release_id = engine.release_slot(appt, now=now)
    offers = engine.blast(release_id, now=now)
    engine.accept_offer(offers[0], now=now + timedelta(minutes=6))

    # A second release that nobody takes.
    fid2 = make_family(db, fid="fam_x", phone="+18475559999")
    add_patient(db, patient_id="px", family_id=fid2, first_name="Z", last_name="Z",
                dob="2018-01-01")
    appt2 = add_appointment(db, patient_id="px", provider_id="dr_ruiz",
                            visit_type=VisitType.PROCEDURE,
                            start=local(2026, 9, 21, 10, 0), now=local(2026, 9, 1))
    rel2 = engine.release_slot(appt2, now=now)
    engine.blast(rel2, now=now)  # no procedure candidates -> closed

    rate = backfill_rate(db)["overall"]
    assert rate == {"numerator": 1, "denominator": 2, "rate": 0.5}
    stats = fill_time_stats(db)["overall"]
    assert stats["n"] == 1
    assert stats["median_minutes"] == pytest.approx(6.0)


def test_message_funnel_surfaces_block_reasons(db, gw):
    fid = make_family(db, consent=False)
    add_patient(db, patient_id="p1", family_id=fid, first_name="Ana", last_name="B",
                dob="2020-01-05")
    add_provider(db, provider_id="dr_ruiz", display_name="Dr. Ruiz")
    appt = add_appointment(db, patient_id="p1", provider_id="dr_ruiz",
                           visit_type=VisitType.WELL, start=local(2026, 9, 15, 10, 0),
                           now=local(2026, 9, 1))
    engine = ReminderEngine(db, gw)
    engine.plan_appointment(appt, now=local(2026, 9, 1))
    engine.dispatch_due(now=local(2026, 9, 8, 12, 0))
    funnel = message_funnel(db)
    assert funnel["by_block_reason"]["no_consent"] == 1
    assert funnel["by_status"]["blocked"] == 1


def test_kpi_summary_refuses_to_invent_a_baseline(db, gw):
    seed_metrics(db, gw)
    summary = kpi_summary(db)
    assert summary["no_show"]["verdict"] == "baseline_not_supplied"
    assert summary["no_show"]["rate"] == pytest.approx(0.15)
    with_baseline = kpi_summary(db, baseline_no_show_rate=0.09)
    assert with_baseline["no_show"]["verdict"] == "below_target"
    improved = kpi_summary(db, baseline_no_show_rate=0.30)
    assert improved["no_show"]["verdict"] == "meets_target"


def test_kpi_summary_reports_insufficient_data_rather_than_zero(db, gw):
    summary = kpi_summary(db)
    assert summary["backfill"]["verdict"] == "insufficient_data"
    assert summary["fill_time"]["verdict"] == "insufficient_data"


# ==========================================================================
# The constraint the build plan is most emphatic about
# ==========================================================================


def _module_imports(path, *, scope="any"):
    """Module names imported by a file, via the AST rather than grep.

    WHY AST: a grep-based guard trips over its own explanatory docstrings, and a
    guard that has to be silenced with a pragma is a guard nobody trusts.

    `scope="module"` looks only at top-level statements, which is how you tell a
    hard dependency from a deliberately lazy one inside a function body.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = tree.body if scope == "module" else list(ast.walk(tree))
    names = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add("." * node.level + (node.module or ""))
    return names


def _scheduling_modules():
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "modules" / "scheduling"
    return sorted(root.glob("*.py"))


def test_no_language_model_anywhere_in_this_module():
    """I-07 is deterministic. If this fails, someone put a model in a cron job."""
    banned = ("nsp_core.llm", "openai", "anthropic", "transformers", "llama_cpp")
    offenders = [
        f"{path.name}: {name}"
        for path in _scheduling_modules()
        for name in _module_imports(path)
        if any(name == b or name.startswith(b + ".") for b in banned)
    ]
    assert offenders == [], offenders


def test_no_model_symbols_referenced_in_this_module():
    """Belt and braces: no LLMClient / structured() call sneaks in by injection."""
    import ast

    offenders = []
    for path in _scheduling_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in {"LLMClient", "SchemaViolation"}:
                offenders.append(f"{path.name}:{node.lineno}: {node.id}")
            if isinstance(node, ast.Attribute) and node.attr == "structured":
                offenders.append(f"{path.name}:{node.lineno}: .structured(")
    assert offenders == [], offenders


def test_no_cloud_sdk_imports_in_this_module():
    banned = ("boto3", "google.cloud", "azure", "botocore")
    offenders = [
        f"{path.name}: {name}"
        for path in _scheduling_modules()
        for name in _module_imports(path)
        if any(name == b or name.startswith(b + ".") for b in banned)
    ]
    assert offenders == [], offenders


def test_twilio_is_lazy_and_never_a_module_scope_dependency():
    """The vendor SDK must stay lazy so the default path has no dependency.

    Asserts both halves: nothing imports twilio at module scope, and the lazy
    import does exist inside TwilioGateway — so this test fails if someone
    "cleans up" the lazy import by hoisting it, and also if the gateway quietly
    stops importing the SDK at all.
    """
    module_scope = [
        f"{path.name}: {name}"
        for path in _scheduling_modules()
        for name in _module_imports(path, scope="module")
        if name.split(".")[0] == "twilio"
    ]
    assert module_scope == []

    lazy = {
        name
        for path in _scheduling_modules()
        for name in _module_imports(path)
        if name.split(".")[0] == "twilio"
    }
    assert lazy == {"twilio.rest"}


def test_importing_the_package_pulls_in_no_optional_dependency():
    """A fresh interpreter must import modules.scheduling with stdlib only."""
    import subprocess
    import sys
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[1]
    code = (
        "import sys; import modules.scheduling as m;"
        "bad=[n for n in sys.modules if n.split('.')[0] in "
        "{'twilio','boto3','torch','transformers','openai','anthropic'}];"
        "print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=repo, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == ""


# ==========================================================================
# Regression tests — each of these reproduced a real defect before its fix
# ==========================================================================


def test_concurrent_cancellations_cannot_create_two_releases(db, gw):
    """A double-tapped cancel link must not sell the same twenty minutes twice.

    Before the fix, `release_slot` read the appointment status and wrote the
    release in separate statements. Four threads all passed the status check and
    each inserted a slot_release for one appointment; each was then blasted and
    filled, producing two confirmed bookings in one slot -- exactly the failure
    the module exists to prevent, one scope higher than where the lock was.
    """
    appt, _, _ = build_backfill_scenario(db, gw, n_candidates=5)
    engine = BackfillEngine(db, gw)
    now = local(2026, 9, 19, 9, 0)

    barrier = threading.Barrier(4)
    releases: list = [None] * 4

    def cancel(index: int) -> None:
        barrier.wait(timeout=10)
        releases[index] = engine.release_slot(appt, now=now)

    threads = [threading.Thread(target=cancel, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    created = [r for r in releases if r is not None]
    assert len(created) == 1, f"expected one release, got {created}"
    assert len(db.all("SELECT * FROM slot_release WHERE appointment_id = ?", (appt,))) == 1
    # And the losing callers got None, not a duplicate they would go on to blast.
    assert sum(1 for r in releases if r is None) == 3


def test_open_release_is_unique_per_appointment_at_the_database_level(db, gw):
    """Belt and braces: even a hand-written INSERT cannot duplicate an open release."""
    import sqlite3

    appt, _, _ = build_backfill_scenario(db, gw, n_candidates=2)
    engine = BackfillEngine(db, gw)
    now = local(2026, 9, 19, 9, 0)
    release_id = engine.release_slot(appt, now=now)
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            """INSERT INTO slot_release (release_id, appointment_id, provider_id, visit_type,
                   start_utc, duration_minutes, released_utc)
               VALUES ('rel_dupe', ?, 'dr_ruiz', 'well', ?, 20, ?)""",
            (appt, iso(local(2026, 9, 20, 10, 0)), iso(now)),
        )
    assert release_id is not None


def test_one_patient_cannot_accept_two_simultaneous_slots(db, gw):
    """Eligibility ran before any booking existed; accept must re-check.

    Two providers, two releases at the same clock time, one keen family holding
    an offer on each. Before the fix both accepts won, the patient was booked
    twice at once, and the waitlist row's booked_appointment silently overwrote
    the first booking, orphaning it.
    """
    add_provider(db, provider_id="dr_ruiz", display_name="Dr. Ruiz")
    add_provider(db, provider_id="dr_okafor", display_name="Dr. Okafor")
    holder = make_family(db, fid="fam_holder", phone="+18475550100")
    add_patient(db, patient_id="p_h1", family_id=holder, first_name="H1", last_name="H",
                dob="2019-05-05")
    add_patient(db, patient_id="p_h2", family_id=holder, first_name="H2", last_name="H",
                dob="2019-05-06")
    keen = make_family(db, fid="fam_keen", phone="+18475552222")
    add_patient(db, patient_id="p_keen", family_id=keen, first_name="Keen", last_name="X",
                dob="2020-06-01")

    slot = local(2026, 9, 25, 10, 0)
    now = local(2026, 9, 19, 9, 0)
    engine = BackfillEngine(db, gw)
    per_release = []
    for suffix, provider, patient in [("a", "dr_ruiz", "p_h1"), ("b", "dr_okafor", "p_h2")]:
        add_waitlist_entry(db, patient_id="p_keen", visit_type=VisitType.WELL,
                           earliest_ok=local(2026, 9, 1), latest_ok=local(2026, 10, 1),
                           added=local(2026, 8, 1), entry_id=f"wl_keen_{suffix}")
        appt = add_appointment(db, patient_id=patient, provider_id=provider,
                               visit_type=VisitType.WELL, start=slot, now=local(2026, 9, 1))
        release_id = engine.release_slot(appt, now=now)
        per_release.append(engine.blast(release_id, now=now))

    assert all(per_release), per_release

    def offer_for(offer_ids, entry_id):
        for offer_id in offer_ids:
            row = db.one("SELECT entry_id FROM backfill_offer WHERE offer_id=?", (offer_id,))
            if row["entry_id"] == entry_id:
                return offer_id
        raise AssertionError(f"no offer for {entry_id}")

    # Deliberately accept via a *different, still-active* waitlist entry the
    # second time, so the guard under test is the patient-conflict check rather
    # than the already-booked-entry check.
    first = engine.accept_offer(
        offer_for(per_release[0], "wl_keen_a"), now=now + timedelta(minutes=1)
    )
    second = engine.accept_offer(
        offer_for(per_release[1], "wl_keen_b"), now=now + timedelta(minutes=2)
    )
    assert first.won is True
    assert second.won is False
    assert second.reason == "patient_already_booked_at_that_time"
    booked = db.all(
        "SELECT * FROM appointment WHERE patient_id='p_keen' AND status='confirmed'"
    )
    assert len(booked) == 1


def test_waitlist_entry_already_booked_cannot_accept_again(db, gw):
    appt, entries, _ = build_backfill_scenario(db, gw, n_candidates=2)
    engine = BackfillEngine(db, gw)
    now = local(2026, 9, 19, 9, 0)
    release_id = engine.release_slot(appt, now=now)
    offers = engine.blast(release_id, now=now)
    entry_id = db.one(
        "SELECT entry_id FROM backfill_offer WHERE offer_id = ?", (offers[0],)
    )["entry_id"]
    db.execute("UPDATE waitlist_entry SET status='withdrawn' WHERE entry_id=?", (entry_id,))
    result = engine.accept_offer(offers[0], now=now)
    assert result.won is False
    assert result.reason == "waitlist_entry_no_longer_active"


def test_repeat_stop_in_the_same_second_still_suppresses(db, gw):
    """A webhook retry must not be swallowed by a UNIQUE constraint."""
    fid = make_family(db, phone="+18475550101")
    now = local(2026, 9, 10, 20, 0)
    revoke_consent(db, family_id=fid, revoked=now)
    db.execute("UPDATE suppression SET released_utc = ? WHERE family_id = ?", (iso(now), fid))
    assert SendGate(db).is_suppressed(fid, Channel.SMS) is False
    revoke_consent(db, family_id=fid, revoked=now)  # same second, second STOP
    assert SendGate(db).is_suppressed(fid, Channel.SMS) is True


def test_in_memory_database_is_refused(tmp_path):
    with pytest.raises(ValueError, match="in-memory"):
        Database(":memory:")


def test_email_waitlist_entry_with_no_email_is_excluded(db, gw):
    """`notify_channel` and the destination address must agree at selection time."""
    appt, _, _ = build_backfill_scenario(db, gw, n_candidates=2)
    db.execute("UPDATE waitlist_entry SET notify_channel='email' WHERE entry_id='wl0'")
    grant_consent(db, family_id="fam0", channel=Channel.EMAIL,
                  granted=local(2026, 1, 1, 9), capture_method="intake_form")
    engine = BackfillEngine(db, gw)
    now = local(2026, 9, 19, 9, 0)
    release_id = engine.release_slot(appt, now=now)
    assert [c.entry_id for c in engine.eligible_candidates(release_id, now=now)] == ["wl1"]

    # With an address on file they become reachable again.
    db.execute("UPDATE family SET primary_email='parent@example.com' WHERE family_id='fam0'")
    assert {c.entry_id for c in engine.eligible_candidates(release_id, now=now)} == {"wl0", "wl1"}


def test_waitlist_entry_rejects_an_unknown_channel(db):
    make_family(db)
    add_patient(db, patient_id="p1", family_id="fam1", first_name="A", last_name="B",
                dob="2020-01-01")
    with pytest.raises(ValueError, match="notify_channel"):
        add_waitlist_entry(db, patient_id="p1", visit_type=VisitType.WELL,
                           earliest_ok=local(2026, 9, 1), latest_ok=local(2026, 10, 1),
                           notify_channel="carrier_pigeon")


def test_non_wrapping_quiet_window_does_not_defer_a_day(db):
    """QuietHours(01:00, 05:00) must open at 05:00 today, not tomorrow."""
    from datetime import time as dtime

    qh = QuietHours(dtime(1, 0), dtime(5, 0))
    opened = to_local(qh.next_open(local(2026, 6, 10, 2, 30)))
    assert (opened.month, opened.day, opened.hour) == (6, 10, 5)


def test_degenerate_quiet_window_is_refused():
    from datetime import time as dtime

    with pytest.raises(ValueError):
        QuietHours(dtime(9, 0), dtime(9, 0))


def test_overdue_reminder_is_not_sent_after_the_appointment_started(db, gw):
    """A cron outage must not deliver a T-2h reminder from the waiting room."""
    fid = make_family(db)
    add_patient(db, patient_id="p1", family_id=fid, first_name="Ana", last_name="B",
                dob="2020-01-05")
    add_provider(db, provider_id="dr_ruiz", display_name="Dr. Ruiz")
    start = local(2026, 9, 15, 10, 0)
    appt = add_appointment(db, patient_id="p1", provider_id="dr_ruiz",
                           visit_type=VisitType.SICK, start=start, now=local(2026, 9, 14))
    engine = ReminderEngine(db, gw)
    engine.plan_appointment(appt, now=local(2026, 9, 14))
    assert engine.dispatch_due(now=local(2026, 9, 15, 11, 0)) == {"skipped": 1}
    assert db.one("SELECT block_reason FROM message_log WHERE status='skipped'")[
        "block_reason"
    ] == "would_arrive_after_appointment"


def test_age_months_handles_month_end_birthdays():
    assert age_months("2024-08-31", local(2024, 9, 30, 12)) == 1
    assert age_months("2024-01-31", local(2024, 2, 29, 12)) == 1
    assert age_months("2024-01-31", local(2024, 2, 28, 12)) == 0
    assert age_months("2024-03-15", local(2026, 3, 14, 12)) == 23
    assert age_months("2024-03-15", local(2026, 3, 15, 12)) == 24


def test_metrics_reject_an_unknown_group_by(db):
    for fn in (no_show_rate, backfill_rate, fill_time_stats):
        with pytest.raises(ValueError, match="cannot group by"):
            fn(db, group_by=["provdier_id"])


def test_no_show_groups_by_day_are_ordered_by_weekday(db, gw):
    seed_metrics(db, gw)
    labels = list(no_show_rate(db, group_by=["dow"])["groups"])
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    assert labels == [d for d in order if d in labels]


def test_zero_baseline_no_show_rate_is_not_a_verdict(db, gw):
    seed_metrics(db, gw)
    assert kpi_summary(db, baseline_no_show_rate=0.0)["no_show"]["verdict"] == (
        "baseline_not_supplied"
    )


def test_message_bodies_use_no_glibc_only_strftime_flags(db, gw):
    """`%-d`/`%-I` raise on musl and Windows. Assert the rendered text is right."""
    from modules.scheduling.cadence import format_when

    assert format_when(local(2026, 9, 5, 9, 20)) == "Sat Sep 5 at 9:20 AM"
    assert format_when(local(2026, 9, 15, 13, 5)) == "Tue Sep 15 at 1:05 PM"
    assert format_when(local(2026, 9, 15, 0, 30)) == "Tue Sep 15 at 12:30 AM"
    assert format_when(local(2026, 9, 15, 12, 30)) == "Tue Sep 15 at 12:30 PM"
