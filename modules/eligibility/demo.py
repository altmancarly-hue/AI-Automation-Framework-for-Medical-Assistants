"""Runnable end-to-end demonstration: `make demo-i09`.

The T-3 batch over seven synthetic patients, the card-capture confirmation step,
the denial classification and its trend line, and the two refusals that define
this module: no patient hears bad news from a machine, and no card is submitted
unconfirmed.
"""

from __future__ import annotations

import json
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from nsp_core.audit import AuditLog
from nsp_core.llm import EchoTransport, LLMClient

from .cards import CardReader, UnconfirmedCard
from .coverage import (
    Outcome,
    PatientCommunicationRefused,
    determine,
    outreach_draft,
)
from .denials import DenialClassifier, RootCause, build_denial_report, draft_appeal
from .fixtures import (
    CASES,
    DENIALS,
    NOW,
    SERVICE_DATE,
    build_payer_table,
    by_name,
    card_response,
    denial_response,
)
from .x12 import SubsetParser, build_270

_CARD_TEXT = """BlueCross BlueShield of Illinois
GOLD PPO 500
Member: MARISOL ALVAREZ
ID  W9928311402
Group GRP0084412
RxBIN 011552   RxPCN IL
Member services 1-800-892-2803"""


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="nsp-i09-demo-"))
    audit = AuditLog(workdir / "audit.sqlite3", hmac_key=b"demo-key")
    payers = build_payer_table()
    parser = SubsetParser()

    print("=" * 78)
    print("I-09 demo - eligibility verification and denial prevention")
    print(f"workspace: {workdir}")
    print("=" * 78)
    print("\nA model touches exactly two things here: reading a card photo, and")
    print("bucketing the free text on a denial. Coverage is decided by the 271.")

    # -- the 270 -----------------------------------------------------------
    print(f"\n{'=' * 78}\n-- the T-3 batch: one 270 per scheduled patient --")
    sample = by_name("rosa_active")
    edi = build_270(sample.request, control_number="0001", created=NOW)
    print()
    for segment in edi.split("~"):
        if segment:
            print(f"   {segment}")

    # -- the batch ---------------------------------------------------------
    print(f"\n{'=' * 78}\n-- {len(CASES)} responses --")
    determinations = []
    for case in CASES:
        response = parser.parse_271(case.response_271) if case.response_271 else None
        result = determine(case.request, response, payers, on=SERVICE_DATE)
        determinations.append((case, result))
        print(f"\n[{case.name}] {case.description}")
        print(f"   outcome : {result.outcome.upper()}")
        print(f"   why     : {result.reason[:150]}")
        if result.copay_usd is not None:
            print(f"   copay   : ${result.copay_usd:,.2f} "
                  "(surfaced to the desk so it is collected at the desk)")
        for warning in result.warnings:
            print(f"   warning : {warning[:100]}")

    print(f"\n{'=' * 78}\n-- the front-desk work queue --")
    queue = [(c, d) for c, d in determinations if d.needs_human]
    print(f"   {len(queue)} of {len(CASES)} need a person before the visit:")
    for case, result in queue:
        print(f"     {result.outcome:<16} {case.child_first_name:<8} "
              f"{case.request.payer_name}")

    # -- the refusal -------------------------------------------------------
    print(f"\n{'=' * 78}\n-- the control this module is built around --")
    print("README I-09: \"Never auto-communicate a coverage denial to a patient.")
    print(" Route to a human who calls the payer to confirm before any patient")
    print(" contact.\"\n")
    for case, result in determinations:
        try:
            message = outreach_draft(
                result,
                family_name=case.family_name,
                child_first_name=case.child_first_name,
            )
            print(f"   {case.name:<22} DRAFTED: {message[:96]}")
        except PatientCommunicationRefused as exc:
            print(f"   {case.name:<22} REFUSED: {str(exc).splitlines()[0][:96]}")
    print("\n   There is no force parameter. The templates for bad news do not")
    print("   exist, which is why there is no path to sending one.")

    # -- card capture ------------------------------------------------------
    print(f"\n{'=' * 78}\n-- card capture: extract, then confirm --")
    reader = CardReader(
        LLMClient(EchoTransport([card_response(member_id="W9928311402")])),
        audit=audit,
    )
    extraction = reader.read(_CARD_TEXT, document_id="card_rosa_front")
    print()
    for spec in extraction.fields.values():
        if spec.value is not None:
            print(f"   {spec.name:<18} {str(spec.value):<38} "
                  f"read at {spec.confidence:.2f}")
    try:
        extraction.for_submission()
    except UnconfirmedCard as exc:
        print(f"\n   REFUSED: {str(exc).splitlines()[0][:150]}")

    now = NOW
    extraction.confirm("payer_name", by="front_desk_dana",
                       value="Blue Cross Blue Shield of Illinois", at=now)
    # The person reads the card and finds the model dropped a digit.
    extraction.confirm("member_id", by="front_desk_dana",
                       value="W99283114021", at=now)
    print(f"\n   confirmed by front_desk_dana; corrections: {extraction.corrections}")
    print(f"   for submission: {json.dumps(extraction.for_submission())}")
    print("\n   The unit of confirmation is the FIELD, because the unit of error")
    print("   is the character. A member id has no checksum.")

    # -- denials -----------------------------------------------------------
    print(f"\n{'=' * 78}\n-- denial classification --")
    # One echo response per denial that reaches the model. The first three have
    # mapped CARC codes and still get read, so the disagreement is visible.
    responses = [
        denial_response("eligibility", 0.94, "coverage terminated"),
        denial_response("coding", 0.91, "diagnosis is inconsistent"),
        denial_response("timely_filing", 0.96, "timely filing limit"),
        denial_response("eligibility", 0.92, "not enrolled in this plan"),
        # Evidence the remark does not contain. The grounding check catches it.
        denial_response("eligibility", 0.88, "patient coverage lapsed in June"),
    ]
    classifier = DenialClassifier(
        LLMClient(EchoTransport(responses)), audit=audit
    )
    classifications = [classifier.classify(d) for d in DENIALS]
    print()
    for item in classifications:
        flag = "  <- NEEDS A PERSON" if item.needs_human else ""
        print(f"   {item.denial.claim_id}  CARC {item.denial.carc_code:<3} "
              f"-> {item.root_cause:<14} by {item.decided_by:<12}"
              f"${item.denial.amount_usd:>7,.2f}{flag}")
        if item.model_disagreed_with:
            print(f"      the model read this as {item.model_disagreed_with!r}; "
                  "the standard code wins")
        if item.needs_human:
            print(f"      {item.reasoning[:120]}")

    report = build_denial_report("2026-08", classifications)
    print(f"\n-- the number this initiative is judged on --")
    print(json.dumps(report.as_dict(), indent=2))
    print("\n   If the T-3 checks work, `eligibility_caused` falls quarter over")
    print("   quarter. Without this split, a denial rate is one number hiding")
    print("   five causes moving in different directions.")

    # -- appeal ------------------------------------------------------------
    print(f"\n{'=' * 78}\n-- an appeal, assembled from facts a person supplies --")
    appealable = next(
        c for c in classifications if c.root_cause == RootCause.ELIGIBILITY
    )
    draft = draft_appeal(
        appealable,
        practice_name="North Suburban Pediatrics",
        facts=[
            "A 271 eligibility response dated 2026-07-01 (trace NSP0002) reported "
            "active coverage for this member on the date of service.",
            "The member's card presented at check-in shows an effective date of "
            "2026-01-01 with no termination date printed.",
            "No termination notice was received by the practice.",
        ],
    )
    print()
    print(draft.body)

    # -- payer table -------------------------------------------------------
    print(f"\n{'=' * 78}\n-- the payer table, which is the public list --")
    print(f"   generated website list ({SERVICE_DATE.isoformat()}):")
    for name in payers.public_list(SERVICE_DATE):
        print(f"     {name}")
    print("\n   contracts needing attention:")
    for row in payers.stale_records(SERVICE_DATE):
        print(f"     [{row['state']:<8}] {row['name']:<36} {row['detail']}")
    print("\n   README I-09 opens by noting the practice's published insurance")
    print("   list is dated January 2016. A list nobody generates is a list that")
    print("   went wrong at some unknown point in the past.")

    print(f"\n{'=' * 78}\n-- audit --")
    print(json.dumps(audit.counts(), indent=2))


if __name__ == "__main__":
    main()
