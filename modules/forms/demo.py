"""Runnable end-to-end demonstration: `make demo-i01`.

Six synthetic forms through the whole pipeline, then the four gates that matter:
the uncalibrated-template refusal, the unknown-layout proposal that cannot add
itself to the library, the release gate refusing an approval, and the synthetic
probe. No network, no PHI, no real form.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import timedelta
from pathlib import Path

from nsp_core.audit import AuditLog
from nsp_core.llm import EchoTransport, LLMClient

from .blankforms import write_blank_form
from .detect import FormDetector, TemplateProposer
from .fill import FormFiller, PyMuPDFBackend
from .fixtures import (
    CASES,
    CLINIC_NOW,
    build_chart_source,
    by_name,
    chart_doses_for,
    registry_doses_for,
)
from .lifecycle import FormState, FormTracker
from .pipeline import FormPipeline
from .probe import ProbeProgram
from .review import NotReleasable, ReviewDecision, record_review
from .templates import TemplateStore, UncalibratedTemplate

_NOVEL_FORM = """NORTHBROOK SOCCER CLUB
2026 Player Medical Clearance

Player name ______________________
Date of birth ____________________
Tetanus booster date _____________
Cleared for full participation?  Y / N
Coach / trainer signature ________
"""

_PROPOSAL = json.dumps(
    {
        "form_title": "2026 Player Medical Clearance",
        "issuer": "Northbrook Soccer Club",
        "page_count": 1,
        "fields": [
            {"label": "Player name", "semantic": "patient_name", "page": 1,
             "x": 150, "y": 100, "width": 200, "height": 14},
            {"label": "Date of birth", "semantic": "patient_dob", "page": 1,
             "x": 150, "y": 124, "width": 100, "height": 14},
            {"label": "Tetanus booster date", "semantic": "immunization_date",
             "page": 1, "x": 150, "y": 148, "width": 100, "height": 14},
            {"label": "Cleared for full participation", "semantic": "unknown",
             "page": 1, "x": 150, "y": 172, "width": 100, "height": 14},
            {"label": "Coach / trainer signature", "semantic": "provider_signature",
             "page": 1, "x": 150, "y": 196, "width": 200, "height": 20},
        ],
        "notes": "single page, hand-ruled",
    }
)


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="nsp-i01-demo-"))
    audit = AuditLog(workdir / "audit.sqlite3", hmac_key=b"demo-key")

    print("=" * 78)
    print("I-01 demo - automated forms pipeline")
    print(f"workspace: {workdir}")
    print("=" * 78)

    # -- the template library ---------------------------------------------
    print("\n-- template library --")
    strict = TemplateStore.load()
    for row in strict.calibration_report():
        mark = "calibrated" if row["calibrated"] else "PLACEHOLDER COORDINATES"
        print(f"   {row['form_type']:<46} {row['field_count']:>3} fields  {mark}")
    try:
        strict.for_filling("il_certificate_of_child_health_examination")
    except UncalibratedTemplate as exc:
        print(f"\n   REFUSED: {str(exc).splitlines()[0][:200]}")
        print("   (the field list is real; the boxes must be measured against")
        print("    the IDPH PDF before this form fills anything)")

    template = strict.for_filling("demo_camp_health_form")
    blank = write_blank_form(template, str(workdir / "blank.pdf"))
    print(f"\n   blank demo form drawn at the template's own coordinates: {blank}")

    # -- classification ----------------------------------------------------
    print("\n-- classification (deterministic; no model) --")
    detector = FormDetector(TemplateStore.load(allow_placeholder=True))
    for label, text in (
        ("a camp form", "Camp Health Form\nCamper information\nImmunizations\n"
                        "Synthetic demonstration form"),
        ("an Illinois certificate",
         "State of Illinois\nCertificate of Child Health Examination\n"
         "To be completed by health care provider\nImmunization record\n"
         "Health history\nChild's name\nDiabetes screening\n"
         "Lead risk questionnaire\nDental examination"),
        ("a form nobody has seen", _NOVEL_FORM),
    ):
        result = detector.detect(text)
        print(f"   {label:<26} -> {result.form_type or 'UNKNOWN LAYOUT'}"
              f"  ({result.score:.0%})")
        if not result.known:
            print(f"      {result.reason}")

    # -- the unknown-layout path -------------------------------------------
    print("\n-- unknown layout: the model proposes, a person disposes --")
    proposer = TemplateProposer(LLMClient(EchoTransport([_PROPOSAL])), audit=audit)
    proposal = proposer.propose(
        _NOVEL_FORM, document_id="fax_20260824_003",
        form_type="northbrook_soccer_clearance",
    )
    print(f"   proposed {len(proposal.proposed_fields)} field(s) for "
          f"{proposal.title!r}")
    for note in proposal.needs_attention:
        print(f"      NEEDS A PERSON: {note}")
    try:
        proposal.confirm(confirmed_by="", anchors=["player medical clearance"])
    except Exception as exc:  # noqa: BLE001
        print(f"   REFUSED without a confirming person: {str(exc).splitlines()[0]}")
    confirmed = proposal.confirm(
        confirmed_by="ma_jess", anchors=["player medical clearance"]
    )
    library = TemplateStore.load(allow_placeholder=True)
    library.add(confirmed, confirmed_by="ma_jess")
    print(f"   added to the library by ma_jess; still placeholder: "
          f"{confirmed.is_placeholder} (the boxes are estimates until measured)")

    # -- the pipeline ------------------------------------------------------
    tracker = FormTracker(audit=audit)
    probes = ProbeProgram(audit=audit, rate=2)  # 1 in 2 for the demo, not 1 in 50
    pipeline = FormPipeline(
        store=strict,
        chart_source=build_chart_source(),
        filler=FormFiller(PyMuPDFBackend(), audit=audit),
        tracker=tracker,
        probes=probes,
        audit=audit,
    )

    print(f"\n{'=' * 78}\n-- {len(CASES)} form requests --")
    prepared_forms = []
    now = CLINIC_NOW
    for case in CASES:
        request = tracker.receive(
            request_id=f"rq_{case.name}", patient_id=case.patient_id,
            form_type=case.form_type, channel=case.channel, now=now,
            due_by=now + timedelta(days=3),
        )
        prepared = pipeline.prepare(
            request,
            blank_pdf=blank,
            destination=str(workdir / f"{case.name}.pdf"),
            now=now,
            chart_doses=chart_doses_for(case.name),
            registry_doses=registry_doses_for(case.name),
            registry_note=case.registry_note,
        )
        prepared_forms.append(prepared)
        print(f"\n[{case.name}] {case.description}")
        print(f"   auto-filled : {len(prepared.filled.auto_filled)} field(s), "
              f"all highlighted")
        print(f"   left blank  : {len(prepared.filled.skipped)} field(s)")
        if prepared.probe_field:
            probe = probes.probes[request.request_id]
            print(f"   PROBE       : {probe.field_name} = {probe.injected!r} "
                  f"(really {probe.original!r}) - {probe.kind}")
        for row in prepared.review.discrepancies[:2]:
            print(f"   discrepancy : [{row['severity']}] {row['detail'][:88]}")
        for blocker in prepared.review.blockers[:3]:
            print(f"   BLOCKER     : {blocker['kind']} - {blocker['detail'][:76]}")
        if prepared.releasable:
            print("   releasable  : yes")
        now += timedelta(minutes=7)

    # -- review ------------------------------------------------------------
    print(f"\n{'=' * 78}\n-- the review step --")

    # EVERY form is reviewed, including the blocked ones. A probe on a form that
    # happens to be blocked for an unrelated reason was still shown to the MA,
    # and whether they spotted it is exactly what the control measures. Scoring
    # only the clean forms would measure the reviewer on their easiest work.
    for prepared in prepared_forms:
        request = prepared.request
        probe = probes.probes.get(request.request_id)
        corrections: dict[str, str] = {}
        # The demo's stand-in for a reviewer: attentive on short forms, skimming
        # on the long one. A real deployment gets this from a real person.
        attentive = len(prepared.filled.auto_filled) < 14
        if probe is not None and attentive:
            corrections[probe.field_name] = probe.original

        action = "edited" if corrections else (
            "rejected" if not prepared.releasable else "accepted"
        )
        decision = ReviewDecision(
            reviewer_id="ma_jess", action=action, corrections=corrections,
            review_seconds=95.0 if attentive else 7.0,
        )
        if probe is not None:
            probes.resolve(request.request_id, prepared.review, decision, now=now)
            print(f"   {request.request_id:<24} probe on {probe.field_name:<16} "
                  f"{'CAUGHT' if probe.caught else 'MISSED'}")
            # Scoring the probe does NOT take the wrong value off the page.
            # The form is re-filled -- and `should_probe` refuses a request
            # whose probe has been scored, so the second pass is clean.
            prepared = pipeline.prepare(
                request, blank_pdf=blank,
                destination=str(workdir / f"{request.request_id}-rerun.pdf"),
                now=now,
                chart_doses=chart_doses_for(request.request_id[3:]),
                registry_doses=registry_doses_for(request.request_id[3:]),
                registry_note=by_name(request.request_id[3:]).registry_note,
            )
            prepared_forms[prepared_forms.index(
                next(p for p in prepared_forms if p.request is request)
            )] = prepared

        if prepared.review.blockers:
            # The MA can click approve all day; the gate does not care.
            try:
                record_review(
                    prepared.review,
                    ReviewDecision(reviewer_id="ma_jess", action="accepted",
                                   review_seconds=6.0),
                    audit=audit,
                )
            except NotReleasable as exc:
                print(f"   {request.request_id:<24} APPROVAL REFUSED: "
                      f"{str(exc).split('. ')[0][:96]}")
            record_review(
                prepared.review,
                ReviewDecision(reviewer_id="ma_jess", action="rejected",
                               corrections=corrections, review_seconds=95.0),
                audit=audit,
            )
            tracker.advance(
                request, FormState.BLOCKED, actor="ma_jess", now=now,
                note=prepared.review.blockers[0]["kind"],
            )
            continue

        record_review(prepared.review, decision, audit=audit)
        tracker.advance(
            request, FormState.PHYSICIAN_SIGNATURE, actor="ma_jess", now=now,
            blockers=prepared.review.blockers,
        )
        tracker.advance(
            request, FormState.SIGNED, actor="dr_alvarez",
            now=now + timedelta(hours=2), signed_by="dr_alvarez",
        )
        tracker.advance(
            request, FormState.DELIVERED, actor="front_desk",
            now=now + timedelta(hours=3), delivered_to="parent portal",
        )
        tracker.notify(
            request, channel="sms", now=now + timedelta(hours=3),
            message="Your form is signed and available in the portal.",
        )
        print(f"   {request.request_id:<24} released -> {request.state}, "
              f"signed by {request.signed_by}")

    print("\n   a form cannot skip the signature:")
    stuck = next(p.request for p in prepared_forms if p.request.state == FormState.BLOCKED)
    try:
        tracker.advance(
            stuck, FormState.DELIVERED, actor="front_desk", now=now,
            delivered_to="the school",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"   REFUSED: {exc}")

    # -- reports -----------------------------------------------------------
    print(f"\n{'=' * 78}\n-- where every form is --")
    print(json.dumps(tracker.stalled_at(now), indent=2))
    print("\n-- overdue (README I-01: eliminates the 'where is my form' call) --")
    late = tracker.overdue(now + timedelta(days=2))
    for row in late[:4]:
        print(f"   {row['request_id']:<24} {row['state']:<22} "
              f"{row['hours_in_state']:>6.1f}h in state "
              f"(target {row['stage_target_hours']}h)")

    print("\n-- turnaround (README I-01 KPI: < 24 hrs) --")
    print(json.dumps(tracker.turnaround_report(), indent=2))

    print("\n-- synthetic probes (README I-01: the most-omitted control) --")
    print(json.dumps(probes.catch_rate(), indent=2))
    print("\n-- the same number, from the audit log --")
    print(json.dumps(audit.probe_catch_rate(initiative_id="I-01"), indent=2))

    print("\n-- one review screen, as the MA sees it --")
    payload = prepared_forms[1].review.as_dict()
    print(json.dumps(
        {
            "form_type": payload["form_type"],
            "releasable": payload["releasable"],
            "discrepancies": payload["discrepancies"],
            "blockers": payload["blockers"],
            "provenance": payload["provenance"][:4],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
