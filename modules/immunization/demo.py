"""Runnable end-to-end demonstration: `make demo-i02`.

Runs the full nightly batch over the 24 synthetic fixtures, prints the
reconciliation exceptions a human has to work, shows the recall queue refusing
to send before validation and then sending after it, and renders tomorrow's
huddle sheet. No network, no model (the drafter is optional), no PHI.
"""

from __future__ import annotations

import json
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from modules.scheduling import (
    Channel,
    ConsentPurpose,
    Database,
    LocalGateway,
    VisitType,
    add_appointment,
    add_family,
    add_patient,
    add_provider,
    grant_consent,
)

from .fixtures import CASES, TODAY, by_name
from .forecast import ValidationResult
from .huddle import build_huddle
from .pipeline import PatientInput, run_nightly
from .recall import RecallEngine, RecallNotAuthorized

CHI = ZoneInfo("America/Chicago")


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="nsp-i02-demo-"))
    db = Database(workdir / "practice.sqlite3")
    gw = LocalGateway(workdir / "outbox.jsonl")
    add_provider(db, provider_id="dr_ruiz", display_name="Dr. Ruiz")

    for index, case in enumerate(CASES):
        add_family(db, family_id=case.family_id,
                   display_name=f"{case.first_name} family",
                   primary_phone=f"+184755{5000 + index:05d}",
                   primary_email=f"{case.first_name.lower()}@example.com")
        grant_consent(db, family_id=case.family_id, channel=Channel.SMS,
                      purpose=ConsentPurpose.REMINDERS,
                      granted=datetime(2026, 1, 5, 9, tzinfo=CHI),
                      capture_method="intake_form",
                      capture_evidence="INTAKE-2026-0114",
                      captured_by="frontdesk_02")
        add_patient(db, patient_id=case.patient_id, family_id=case.family_id,
                    first_name=case.first_name, last_name="Demo",
                    dob=case.dob.isoformat())

    print("=" * 78)
    print("I-02 demo - immunization gap closure and recall")
    print(f"workspace: {workdir}")
    print("=" * 78)

    patients = [
        PatientInput(c.patient_id, c.family_id, c.first_name, c.dob, c.chart, c.registry)
        for c in CASES
    ]
    nightly = run_nightly(patients, as_of=TODAY)
    print("\n-- nightly batch --")
    print(" ", json.dumps(nightly.summary()))

    print("\n-- reconciliation exceptions for the MA queue --")
    seen = set()
    for item in nightly.review_queue:
        key = (item.patient_id, item.pair.reason)
        if key in seen:
            continue          # one cluster is one decision for the reviewer
        seen.add(key)
        print(f"  {item.patient_id:38} {item.pair.reason[:70]}")

    nightly_batch_time = datetime(TODAY.year, TODAY.month, TODAY.day, 6, 30, tzinfo=CHI)
    now = datetime(TODAY.year, TODAY.month, TODAY.day, 9, 15, tzinfo=CHI)
    engine = RecallEngine(db, gw)
    print("\n-- outbound recall, before validation --")
    try:
        engine.run(list(nightly.forecasts.values()), now=now, patients=nightly.patients)
    except RecallNotAuthorized as exc:
        print(f"  refused: {str(exc).splitlines()[0]}")

    dry = engine.run(list(nightly.forecasts.values()), now=now,
                     patients=nightly.patients, dry_run=True)
    print(f"  dry run: {dry['queue_size']} gaps across "
          f"{dry['patients_in_queue']} patients, {gw.sent and 'SENT' or 'nothing sent'}")

    print("\n-- top of the queue, with the urgency broken down --")
    queue = engine.build_queue(list(nightly.forecasts.values()), as_of=TODAY,
                               patients=nightly.patients)
    for candidate in queue[:8]:
        parts = " ".join(
            f"{k.split('_')[0]}={v:.0f}" for k, v in candidate.breakdown.items() if v
        )
        print(f"  {candidate.urgency:9.1f}  {candidate.patient_id:36} "
              f"{candidate.label:22} {parts}")

    print("\n-- outbound recall, after a recorded validation run --")
    engine.authorize(
        ValidationResult(engine="local_rules", schedule_version="demo", cases=200,
                         antigen_comparisons=1800, agreements=1800,
                         validated_on=TODAY)
    )
    early = engine.run(list(nightly.forecasts.values()), now=nightly_batch_time,
                       patients=nightly.patients)
    print(f"  06:30 batch (quiet hours): {json.dumps(early['outcomes'])} "
          f"- nothing sent, no cadence step consumed")
    report = engine.run(list(nightly.forecasts.values()), now=now,
                        patients=nightly.patients)
    print(f"  09:15 send:                {json.dumps(report['outcomes'])}")
    for message in gw.messages_for("recall_immunization")[:4]:
        print(f"    -> {message['to']}: {message['body'][:110]}")

    print("\n-- tomorrow's huddle sheet --")
    tomorrow = TODAY + timedelta(days=1)
    for hour, name in ((9, "adolescent_gap_cluster"),
                       (10, "combination_split_across_sources"),
                       (11, "adolescent_complete")):
        case = by_name(name)
        add_appointment(db, patient_id=case.patient_id, provider_id="dr_ruiz",
                        visit_type=VisitType.WELL,
                        start=datetime(tomorrow.year, tomorrow.month, tomorrow.day,
                                       hour, 20, tzinfo=CHI),
                        now=datetime(2026, 8, 1, tzinfo=CHI))
    sheet = build_huddle(
        db,
        for_date=tomorrow,
        forecasts=nightly.forecasts,
        reconciliations=nightly.reconciliations,
        narratives={
            by_name("adolescent_gap_cluster").patient_id:
                "Adolescent cluster; HPV is the one with a clock on it.",
        },
    )
    print()
    print(sheet.render())


if __name__ == "__main__":
    main()
