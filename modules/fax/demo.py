"""Runnable end-to-end demonstration: `make demo-i06`.

Ten synthetic faxes through the whole pipeline, then the classifier eval gate,
the panel hygiene reports, and the gateway-outage monitor. No network, no real
model, no PHI, no fax machine.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from nsp_core.audit import AuditLog
from nsp_core.llm import EchoTransport, LLMClient

from .classify import ClassifierGate, DocumentClassifier, evaluate
from .fixtures import AGE_MONTHS, CASES, PANEL, eval_cases
from .match import PatientMatcher
from .ocr import ScriptedOCR
from .pipeline import FaxPipeline, InboundMonitor
from .urgency import ReferenceRanges, UrgencyTriage

NOW = datetime(2026, 8, 24, 9, 30, tzinfo=timezone.utc)


def main() -> None:
    import tempfile
    from pathlib import Path

    workdir = Path(tempfile.mkdtemp(prefix="nsp-i06-demo-"))
    audit = AuditLog(workdir / "audit.sqlite3", hmac_key=b"demo-key")

    ranges = ReferenceRanges.load()
    # The shipped table has no clinical owner and the pipeline refuses without
    # one. A demo names a demo owner; a clinic names a physician.
    ranges.review["owner"] = "demo-only (not a real reviewer)"

    ocr = ScriptedOCR(documents={c.filename: list(c.pages) for c in CASES})
    matcher = PatientMatcher(PANEL)

    print("=" * 78)
    print("I-06 demo - inbound fax and document ingestion")
    print(f"workspace: {workdir}")
    print(f"reference ranges: {ranges.version} ({len(ranges.analytes)} analytes)")
    print(f"panel: {len(PANEL)} patients")
    print("=" * 78)

    # -- the eval gate, before anything is routed --------------------------
    print("\n-- classifier eval gate (README I-06: evaluate against historical faxes) --")
    labelled = eval_cases(repeats=12)
    eval_client = LLMClient(
        EchoTransport([_response_for(c.document_id) for c in labelled])
    )
    result = evaluate(DocumentClassifier(eval_client), labelled)
    print(f"   {result.summary()}")
    gate = ClassifierGate()
    try:
        gate.approve(result, on="2026-08-24")
        print("   approved.")
    except Exception as exc:  # noqa: BLE001
        print(f"   REFUSED: {str(exc)[:200]}")
    print("   blockers:", result.blockers() or "none")
    # The gate must be satisfied for the demo to route anything; a real
    # deployment satisfies it with 300 real faxes.
    gate.provider = result.provider
    gate.model_id, gate.model_version = result.model_id, result.model_version
    gate.prompt_hash = result.prompt_hash
    gate.min_confidence = result.min_confidence
    gate.validated_on = gate.validated_on or "2026-08-24 (DEMO ONLY - see blockers)"

    # -- the pipeline ------------------------------------------------------
    responses: list[str] = []
    for case in CASES:
        responses.append(case.classifier_response)
        responses.append(case.urgency_response)
    client = LLMClient(EchoTransport(responses))
    pipeline = FaxPipeline(
        engines=[ocr],
        classifier=DocumentClassifier(client, audit=audit),
        matcher=matcher,
        triage=UrgencyTriage(client, ranges=ranges, audit=audit),
        gate=gate,
        audit=audit,
        on_duty_physician="dr_alvarez",
        indexer="ma_jess",
    )

    print(f"\n{'=' * 78}\n-- {len(CASES)} inbound documents --")
    queues: dict[str, int] = {}
    for case in CASES:
        processed = pipeline.process(
            case.filename, document_id=case.name, age_lookup=AGE_MONTHS
        )
        routed = processed.routed
        queues[routed.queue] = queues.get(routed.queue, 0) + 1
        print(f"\n[{case.name}] {case.description}")
        print(f"   type      : {routed.document_type}")
        print(f"   patient   : {processed.match.outcome}"
              f"{' -> ' + routed.patient_id if routed.patient_id else ''}")
        print(f"   urgency   : {routed.urgency}"
              f"{'  <- ESCALATED BY RULE' if processed.urgency.escalated else ''}")
        for override in processed.urgency.overrides:
            print(f"               rule: {override[:100]}")
        print(f"   queue     : {routed.queue}")
        for task in routed.tasks:
            print(f"   task      : [{task.kind}] {task.assigned_to} "
                  f"(within {task.due_within_hours:g}h) {task.description[:70]}")
        if routed.immunization_handoff:
            handoff = routed.immunization_handoff
            print(f"   -> I-02   : {len(handoff.doses)} dose(s) from {handoff.source}")
            for record in handoff.as_dose_records():
                print(f"               CVX {record.cvx:<4} {record.given.isoformat()}"
                      f"  {record.product_text or '':<12}"
                      f"{'  (month only)' if record.precision == 'month' else ''}")
            for line in handoff.unresolved:
                print(f"               UNRESOLVED  {line['line'][:44]}"
                      f"  <- {line['reason']}")

    print(f"\n{'=' * 78}\n-- where the morning's post went --")
    print(json.dumps(queues, indent=2))

    # -- panel hygiene -----------------------------------------------------
    print(f"\n{'=' * 78}\n-- panel hygiene (run before go-live) --")
    print("duplicate charts:", json.dumps(matcher.duplicate_report(), indent=2))
    print("same DOB, same surname, multiple-birth flag not set on all:")
    print(json.dumps(matcher.unflagged_multiples(), indent=2))

    # -- gateway outage ----------------------------------------------------
    print(f"\n{'=' * 78}\n-- gateway outage monitor (README I-06 risk table) --")
    monitor = InboundMonitor()
    for label, last in (
        ("30 minutes ago", NOW - timedelta(minutes=30)),
        ("5 hours ago", NOW - timedelta(hours=5)),
        ("never", None),
    ):
        check = monitor.check(last_received=last, now=NOW)
        flag = "ALERT" if check["alert"] else "ok   "
        print(f"   last document {label:<16} {flag}  {check['reason'][:80]}")

    print(f"\n{'=' * 78}\n-- audit --")
    for row in audit.query(
        "SELECT event_type, COUNT(*) c FROM event WHERE initiative_id='I-06'"
        " GROUP BY event_type"
    ):
        print(f"   {row['event_type']:<24} {row['c']}")
    inferences = audit.query(
        "SELECT prompt_template_id, COUNT(*) c FROM inference WHERE"
        " initiative_id='I-06' GROUP BY prompt_template_id"
    )
    for row in inferences:
        print(f"   {row['prompt_template_id']:<24} {row['c']} inference(s)")


def _response_for(document_id: str) -> str:
    from .fixtures import by_name

    return by_name(document_id.split("__r")[0]).classifier_response


if __name__ == "__main__":
    main()
