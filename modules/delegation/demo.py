"""Runnable end-to-end demonstration: `make demo-i10`.

Four staff against four standing orders, the lunchtime gap, the break-glass
path, the one-click audit extract, and the sunset warning that is the reason
this initiative exists at all. No model runs anywhere in this module.
"""

from __future__ import annotations

import json
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from nsp_core.audit import AuditLog

from .enforcement import BreakGlassRefused, DelegationService, NotAuthorised
from .fixtures import NOW, build
from .register import (
    DelegationRules,
    FrameworkSunset,
    StandingOrder,
    UnreviewedRules,
    UnsignedOrder,
)


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="nsp-i10-demo-"))
    audit = AuditLog(workdir / "audit.sqlite3", hmac_key=b"demo-key")
    rules, orders, competencies, roster = build()
    service = DelegationService(
        rules=rules, orders=orders, competencies=competencies,
        roster=roster, audit=audit,
    )

    print("=" * 78)
    print("I-10 demo - standing order digitization and delegation audit")
    print(f"workspace: {workdir}")
    print(f"rules: config/delegation_rules.yaml v{rules.version} "
          f"({rules.framework.get('citation')})")
    print("=" * 78)
    print("\nTHE DELEGATION RULES ARE CONFIGURATION, NOT CODE.")
    print("225 ILCS 60/54.2 sunsets 2027-01-01. When it does, the practice edits")
    print("a YAML file. `enforcement.py` knows nothing about Illinois.")

    # -- the gate ----------------------------------------------------------
    print("\n-- the rules need an owner --")
    try:
        service.certify(NOW.date())
    except UnreviewedRules as exc:
        print(f"   REFUSED: {str(exc).splitlines()[0][:150]}")
    rules.review["owner"] = "dr_alvarez (demo)"
    service.certify(NOW.date())
    print("   owner assigned; the register may now certify compliance.")

    # -- the screen --------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("-- what each person actually SEES on their worklist, 10:30 Monday --")
    print("   README I-10: \"An MA opening a task sees ONLY orders they are")
    print("   currently competent to execute.\" The filter is the control.\n")
    for staff_id in ("ma_jess", "ma_dana", "lpn_marta", "rn_paulo"):
        member = competencies.staff[staff_id]
        available = service.available_orders(staff_id, moment=NOW)
        print(f"   {member.name} ({member.role})")
        for order in available:
            print(f"      + {order.title[:64]}")
        if not available:
            print("      (nothing)")
        for row in service.blocked_orders(staff_id, moment=NOW):
            print(f"      - {row['title'][:56]}")
            print(f"          {row['detail'][:96]}")
        print()

    # -- an act ------------------------------------------------------------
    print("=" * 78)
    print("-- Jess gives a vaccine at 10:34 --")
    moment = NOW + timedelta(minutes=4)
    result = service.execute(
        "ma_jess", "so_immunize",
        patient_id="p_rosa", moment=moment, execution_id="ex_0001",
    )
    print()
    print(json.dumps(result.as_dict(), indent=2))
    print("\n   Note `supervisor`. That field is 225 ILCS 60/54.2's on-site")
    print("   requirement, evidenced. Everybody remembers Dr Alvarez was here;")
    print("   two years from now 'everybody remembers' is not evidence.")

    # -- the lunchtime gap -------------------------------------------------
    print(f"\n{'=' * 78}\n-- 12:30. Dr Alvarez is at lunch --")
    lunch = NOW.replace(hour=12, minute=30)
    print(f"   Jess's worklist now: "
          f"{[o.order_id for o in service.available_orders('ma_jess', moment=lunch)]}")
    try:
        service.execute(
            "ma_jess", "so_immunize", patient_id="p_theo",
            moment=lunch, execution_id="ex_0002",
        )
    except NotAuthorised as exc:
        print(f"\n   REFUSED: {str(exc)[:190]}")
    print("\n   The check runs TWICE -- once to build the screen at 08:00, and")
    print("   again at the moment of the act. A competency can expire between")
    print("   them, and the physician goes to lunch.")

    # -- break glass -------------------------------------------------------
    print(f"\n{'=' * 78}\n-- break glass --")
    print("   Build plan: \"a break-glass path that requires a justification and")
    print("   NEVER BLOCKS PATIENT CARE.\"\n")
    try:
        service.execute(
            "ma_jess", "so_immunize", patient_id="p_theo", moment=lunch,
            execution_id="ex_0003", break_glass_reason="urgent",
        )
    except BreakGlassRefused as exc:
        print(f"   too short: {str(exc)[:140]}")

    emergency = service.execute(
        "ma_jess", "so_immunize", patient_id="p_theo", moment=lunch,
        execution_id="ex_0004",
        break_glass_reason=(
            "Post-exposure tetanus prophylaxis required now; Dr Alvarez off "
            "site at lunch, reached by phone and verbally authorised, returning "
            "13:00. Delay would push the dose outside the window."
        ),
    )
    print(f"\n   PERFORMED. break_glass={emergency.break_glass}, "
          f"review due {emergency.review_due_utc.isoformat()}")
    print("   It cannot be refused, and it cannot be quiet: recorded with")
    print("   break_glass=True, surfaced permanently by unevidenced_supervision(),")
    print("   and it raises a review task. A break-glass that blocks is one staff")
    print("   learn to route around; one nobody reviews is an unlocked door.")

    # -- versioning --------------------------------------------------------
    print(f"\n{'=' * 78}\n-- standing orders are versioned, never edited --")
    try:
        orders.publish(
            StandingOrder(
                order_id="so_immunize", version=3,
                title="Administer routine childhood immunizations",
                task_code="administer_vaccine",
                clinical_content="...revised...",
                delegating_physician_id="dr_alvarez",
                effective_from=date(2026, 9, 1),
                required_competencies=("im_injection",),
                required_supervision_role="physician",
            )
        )
    except UnsignedOrder as exc:
        print(f"   REFUSED unsigned: {str(exc)[:140]}")

    orders.publish(
        StandingOrder(
            order_id="so_immunize", version=3,
            title="Administer routine childhood immunizations per ACIP 2027",
            task_code="administer_vaccine",
            clinical_content="...revised for the 2027 schedule...",
            delegating_physician_id="dr_alvarez",
            effective_from=date(2026, 9, 1),
            required_competencies=("im_injection", "vaccine_storage", "bls_cpr"),
            required_supervision_role="physician",
            source_guideline="ACIP child and adolescent schedule, 2027",
            review_due=date(2027, 9, 1),
            signed_by="dr_alvarez",
            signed_utc=datetime(2026, 8, 24, 16, 0, tzinfo=timezone.utc),
        )
    )
    print(f"\n   so_immunize now has "
          f"{len(orders.versions['so_immunize'])} versions:")
    for version in orders.versions["so_immunize"]:
        print(f"      v{version.version}  from {version.effective_from} "
              f"to {version.retired_on or 'open'}  signed by {version.signed_by}")
    print("\n   v2 is retired, not deleted. Every execution logged against it")
    print("   still resolves to the exact text the physician signed.")

    # -- readiness ---------------------------------------------------------
    print(f"\n{'=' * 78}\n-- what will stop somebody working, before it does --")
    print(json.dumps(service.readiness(NOW.date()), indent=2))

    # -- the sunset --------------------------------------------------------
    print(f"\n{'=' * 78}\n-- the reason this initiative exists --")
    for when in (date(2026, 8, 24), date(2026, 12, 1), date(2027, 3, 1)):
        state, message = rules.sunset_state(when)
        print(f"\n   on {when.isoformat()}: {state}")
        if message:
            print(f"      {message}")
    print()
    try:
        service.certify(date(2027, 3, 1))
    except FrameworkSunset as exc:
        print(f"   certify() REFUSES after the sunset: {str(exc)[:160]}")

    # -- the audit answer --------------------------------------------------
    print(f"\n{'=' * 78}\n-- the one-click audit extract --")
    print("   README I-10: \"produce every delegation, competency record, and")
    print("   execution for MA X between date A and date B\". A query, not a")
    print("   project.\n")
    extract = service.audit_extract(staff_id="ma_jess")
    print(json.dumps(
        {
            "framework": extract["framework"],
            "rules_version": extract["rules_version"],
            "staff": extract["staff"],
            "competency_records": len(extract["competency_records"]),
            "standing_orders": len(extract["standing_orders"]),
            "executions": extract["executions"],
            "unevidenced": extract["unevidenced"],
            "unevidenced_detail": extract["unevidenced_detail"],
        },
        indent=2,
    ))

    print(f"\n{'=' * 78}\n-- audit --")
    print(json.dumps(audit.counts(), indent=2))


if __name__ == "__main__":
    main()
