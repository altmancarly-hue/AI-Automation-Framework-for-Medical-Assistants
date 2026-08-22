"""Runnable end-to-end demonstration: `make demo`.

Seeds a small synthetic practice, runs a week of the reminder cron, cancels an
appointment after hours, backfills the slot from the waitlist, and prints the
README 10.2 dashboard. No network, no model, no PHI — every name here is made up.

WHY ship a demo: the first question anyone asks about a scheduling automation is
"show me what the parent actually receives". This prints it.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .backfill import BackfillEngine
from .cadence import ReminderEngine
from .gateway import LocalGateway
from .inbound import InboundRouter
from .metrics import kpi_summary
from .models import (
    AppointmentStatus,
    Channel,
    ConsentPurpose,
    Database,
    VisitType,
    add_appointment,
    add_family,
    add_patient,
    add_provider,
    add_waitlist_entry,
    grant_consent,
)

CHI = ZoneInfo("America/Chicago")


def _local(y: int, m: int, d: int, hh: int = 0, mm: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=CHI)


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="nsp-i07-demo-"))
    db = Database(workdir / "scheduling.sqlite3")
    gw = LocalGateway(workdir / "outbox.jsonl")

    add_provider(db, provider_id="dr_ruiz", display_name="Dr. Ruiz")
    add_provider(db, provider_id="dr_okafor", display_name="Dr. Okafor", min_age_months=24)

    roster = [
        ("fam_alvarez", "+18475550111", "p_mia", "Mia", "2022-03-14"),
        ("fam_brennan", "+18475550112", "p_theo", "Theo", "2019-11-02"),
        ("fam_cho", "+18475550113", "p_iris", "Iris", "2021-06-20"),
        ("fam_dunlap", "+18475550114", "p_sam", "Sam", "2017-01-09"),
        ("fam_esparza", "+18475550115", "p_nora", "Nora", "2020-08-30"),
    ]
    for family_id, phone, patient_id, first, dob in roster:
        add_family(db, family_id=family_id, display_name=f"{first}'s family",
                   primary_phone=phone)
        grant_consent(db, family_id=family_id, channel=Channel.SMS,
                      purpose=ConsentPurpose.REMINDERS, granted=_local(2026, 1, 5, 9),
                      capture_method="intake_form", capture_evidence="INTAKE-2026-0114",
                      captured_by="frontdesk_02")
        add_patient(db, patient_id=patient_id, family_id=family_id, first_name=first,
                    last_name="Demo", dob=dob)

    booked = _local(2026, 9, 1, 9, 0)
    # Two cancellations the same evening, either side of the backfill boundary:
    # an 09:20 slot cannot be refilled after quiet hours end at 08:00, a 14:20
    # slot easily can. Showing both is more honest than showing only the win.
    early = add_appointment(db, patient_id="p_mia", provider_id="dr_ruiz",
                            visit_type=VisitType.WELL,
                            start=_local(2026, 9, 22, 9, 20), now=booked)
    later = add_appointment(db, patient_id="p_nora", provider_id="dr_ruiz",
                            visit_type=VisitType.WELL,
                            start=_local(2026, 9, 22, 14, 20), now=booked)

    for patient_id, priority, days_waiting in [("p_theo", 0, 30), ("p_iris", 2, 5),
                                               ("p_sam", 0, 45)]:
        add_waitlist_entry(db, patient_id=patient_id, visit_type=VisitType.WELL,
                           earliest_ok=_local(2026, 9, 1), latest_ok=_local(2026, 10, 31),
                           priority=priority,
                           added=_local(2026, 9, 1) - timedelta(days=days_waiting))

    reminders = ReminderEngine(db, gw)
    backfill = BackfillEngine(db, gw)
    router = InboundRouter(db, gw, backfill)

    print("=" * 72)
    print("I-07 demo — North Suburban Pediatrics scheduling automation")
    print(f"workspace: {workdir}")
    print("=" * 72)

    reminders.plan_horizon(now=booked, horizon=timedelta(days=30))
    print("\n-- reminder cron, five-minute ticks (abridged) --")
    for tick in [_local(2026, 9, 15, 9, 25), _local(2026, 9, 20, 9, 25)]:
        histogram = reminders.dispatch_due(now=tick)
        print(f"  {tick:%a %d %b %H:%M %Z}  {histogram or '{}'}")

    # 21:40 the night before: the parents finally get a moment to cancel.
    evening = _local(2026, 9, 21, 21, 40)
    print(f"\n-- {evening:%a %d %b %H:%M %Z}: parents tap 'cancel' in the reminder --")
    for label, appointment_id, phone in [("09:20 Mia", early, "+18475550111"),
                                         ("14:20 Nora", later, "+18475550115")]:
        message_id = db.one(
            "SELECT message_id FROM message_log WHERE appointment_id = ? AND status='sent'"
            " ORDER BY sent_utc DESC LIMIT 1",
            (appointment_id,),
        )["message_id"]
        result = router.handle({"path": f"/x/{message_id}", "from": phone}, now=evening)
        print(f"  {label}: {result.action} -> {result.detail}")
    for row in db.all("SELECT start_utc, close_reason FROM slot_release"):
        print(f"    release {row['start_utc']} close_reason={row['close_reason']}")

    morning = _local(2026, 9, 22, 8, 0)
    print(f"\n-- {morning:%a %d %b %H:%M %Z}: quiet hours end, the sweep blasts the slot --")
    print("  sweep:", backfill.sweep_open_releases(now=morning))

    release = db.one(
        "SELECT * FROM slot_release WHERE closed_utc IS NULL ORDER BY released_utc DESC LIMIT 1"
    )
    offers = db.all(
        "SELECT o.offer_id, o.rank, p.first_name FROM backfill_offer o"
        " JOIN patient p ON p.patient_id = o.patient_id WHERE o.release_id = ?"
        " ORDER BY o.rank",
        (release["release_id"],),
    )
    print("  blast ranking:", ", ".join(f"{o['rank']}:{o['first_name']}" for o in offers))

    accepted = backfill.accept_offer(offers[-1]["offer_id"], now=morning + timedelta(minutes=7))
    print(f"  {offers[-1]['first_name']} accepts 7 minutes later -> won={accepted.won}")
    loser = backfill.accept_offer(offers[0]["offer_id"], now=morning + timedelta(minutes=9))
    print(f"  {offers[0]['first_name']} taps accept 2 minutes after that -> "
          f"won={loser.won} ({loser.reason})")

    print("\n-- what the families actually received --")
    for message in gw.sent:
        print(f"  [{message['purpose']:>22}] {message['to']}: {message['body'][:96]}")

    # Mark history so the dashboard has something to report.
    db.execute(
        "UPDATE appointment SET status=? WHERE status=? AND start_utc < ?",
        (AppointmentStatus.COMPLETED, AppointmentStatus.CONFIRMED, "2026-09-30T00:00:00+00:00"),
    )

    print("\n-- README 10.2 weekly dashboard --")
    summary = kpi_summary(db, baseline_no_show_rate=0.09)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
