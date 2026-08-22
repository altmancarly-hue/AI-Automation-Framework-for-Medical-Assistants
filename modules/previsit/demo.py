"""Runnable end-to-end demonstration: `make demo-i03`.

Runs the 17:00 batch over six synthetic patients and prints the briefs, the
narrative post-conditions doing their work, and the feedback report. No network,
no real model, no PHI.
"""

from __future__ import annotations

import json
import tempfile
from datetime import timedelta
from pathlib import Path

from modules.scheduling.models import Database
from nsp_core.audit import AuditLog
from nsp_core.llm import EchoTransport, LLMClient

from .batch import PatientDay, run_batch
from .brief import render_text
from .feedback import FeedbackLog, Verdict
from .fixtures import CASES, CLINIC_DATE, GENERATED_UTC
from .growth import GrowthReference
from .narrative import NarrativeSynthesizer
from .periodicity import PeriodicitySchedule


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="nsp-i03-demo-"))
    db = Database(workdir / "practice.sqlite3")
    audit = AuditLog(workdir / "audit.sqlite3", hmac_key=b"demo-key")

    schedule = PeriodicitySchedule.load()
    # The shipped table has no clinical owner, and run_batch refuses without one.
    # A demo names a demo owner; a clinic names a physician.
    schedule.review["owner"] = "demo-only (not a real reviewer)"
    reference = GrowthReference()
    feedback = FeedbackLog(db, audit=audit)

    client = LLMClient(EchoTransport([c.model_response for c in CASES if c.encounters]))
    synthesizer = NarrativeSynthesizer(client, audit=audit)

    print("=" * 78)
    print("I-03 demo - pre-visit chart preparation")
    print(f"workspace: {workdir}")
    print(f"periodicity table: {schedule.version} ({len(schedule.screenings)} screenings)")
    print(f"growth reference: {reference.source}, {len(reference.manifest['files'])} tables")
    print("=" * 78)

    days = [
        PatientDay(
            patient_id=c.patient_id, patient_label=c.patient_label, sex=c.sex,
            age_months=c.age_months, age_label=c.age_label, visit_type=c.visit_type,
            appointment_local=c.appointment_local, provider=c.provider,
            measurements=c.measurements, prior_measurements=c.prior_measurements,
            completed_screenings=c.completed_screenings, risk_flags=c.risk_flags,
            immunizations_due=c.immunizations_due, open_threads=c.open_threads,
            encounters=c.encounters, problem_list=c.problem_list,
            data_horizon_months=c.data_horizon_months,
        )
        for c in CASES
    ]

    batch = run_batch(
        days, clinic_date=CLINIC_DATE, schedule=schedule, reference=reference,
        synthesizer=synthesizer, generated_utc=GENERATED_UTC, feedback=feedback,
    )

    print(f"\n-- batch summary --\n{json.dumps(batch.as_dict(), indent=2)}")

    for case, brief in zip(CASES, batch.briefs):
        print(f"\n{'=' * 78}\n-- {case.name}: {case.description}")
        print(render_text(brief))

    print(f"\n{'=' * 78}\n-- what the post-conditions removed --")
    for case in CASES:
        if not case.encounters:
            continue
        result = NarrativeSynthesizer(
            LLMClient(EchoTransport([case.model_response]))
        ).synthesize(case.encounters, patient_id=case.patient_id,
                     problem_list=case.problem_list)
        for dropped in result.dropped:
            print(f"  [{case.name}] {dropped['item'][:56]!r}")
            print(f"      -> {dropped['reason']}")

    print(f"\n{'=' * 78}\n-- feedback loop (README I-03) --")
    now = GENERATED_UTC + timedelta(hours=14)
    for index, brief in enumerate(batch.briefs):
        feedback.mark_opened(brief.brief_id, now=now)
        verdict = Verdict.USEFUL if index % 3 else Verdict.NOT_USEFUL
        feedback.record(
            brief_id=brief.brief_id, verdict=verdict, given_by="ma_jess", now=now
        )
    feedback.record(
        brief_id=batch.briefs[0].brief_id, verdict=Verdict.WRONG, given_by="dr_alvarez",
        now=now, detail="the ear infection was left, not right",
    )
    print(json.dumps(feedback.report(), indent=2))
    print("\nwrong-verdict corpus for the next prompt revision:")
    for row in feedback.wrong_items():
        print(f"  {row['given_by']}: {row['detail']}")


if __name__ == "__main__":
    main()
