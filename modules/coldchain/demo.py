"""Runnable end-to-end demonstration: `make demo-i08`.

Four storage units through a Monday-morning evaluation, the excursion workflow
for the compressor that died on Friday night, and the monthly compliance report.
No model, no network, no real appliance.
"""

from __future__ import annotations

import json
import tempfile
from datetime import date, timedelta
from pathlib import Path

from nsp_core.audit import AuditLog

from .excursion import LotDisposition, build_compliance_report, open_excursion
from .fixtures import INVENTORY, MANUFACTURER_CONTACTS, NOW, UNITS, build_feed, unit
from .monitor import (
    AlertKind,
    ColdChainConfig,
    ColdChainMonitor,
    UnreviewedThresholds,
)


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="nsp-i08-demo-"))
    audit = AuditLog(workdir / "audit.sqlite3", hmac_key=b"demo-key")
    config = ColdChainConfig.load()

    print("=" * 78)
    print("I-08 demo - vaccine cold chain telemetry")
    print(f"workspace: {workdir}")
    print(f"thresholds: {config.version}")
    print("=" * 78)
    print("\nTHERE IS NO MODEL IN THIS MODULE.")
    print("README I-08: \"This is a sensor, a threshold, and an alert...")
    print(" Include this initiative in an AI program specifically to demonstrate")
    print(" the discipline of not using AI where it does not belong.\"")

    # -- the gate ----------------------------------------------------------
    print("\n-- thresholds need a clinical owner --")
    unreviewed = ColdChainMonitor(config=config, units={}, feed=build_feed())
    try:
        unreviewed.require_reviewed()
    except UnreviewedThresholds as exc:
        print(f"   REFUSED: {str(exc).splitlines()[0][:150]}")

    config.review["owner"] = "dr_alvarez (demo)"
    feed = build_feed()
    monitor = ColdChainMonitor(
        config=config,
        units={u.unit_id: u for u in UNITS},
        feed=feed,
        audit=audit,
    )

    print(f"\n{'=' * 78}")
    print("-- Monday 09:00. What the twice-daily paper log would not have told you --")
    alerts = monitor.notifications(now=NOW)
    for alert in alerts:
        print(f"\n[{alert.severity.upper()}] {alert.kind} - {alert.unit_label}")
        print(f"   {alert.detail}")
        print(f"   notify: {', '.join(alert.notify) or '(nobody configured)'}")
        if alert.inventory_at_risk and alert.inventory_value_usd:
            print(f"   inventory in this unit: ${alert.inventory_value_usd:,.0f}")

    quiet = ColdChainMonitor(
        config=config, units={u.unit_id: u for u in UNITS},
        feed=build_feed("all_quiet"),
    )
    quiet_alerts = quiet.notifications(now=NOW)
    print(f"\n-- the same four units, all behaving --")
    print(f"   {len(quiet_alerts)} alert(s): "
          f"{[a.kind for a in quiet_alerts] or 'none'}")
    print("   (an alert that fires on everything fails the same way as no alert)")

    print("\n-- and the same conditions, polled every five minutes for a day --")
    replay = ColdChainMonitor(
        config=config, units={u.unit_id: u for u in UNITS}, feed=feed,
    )
    sent = 0
    for poll in range(288):
        sent += len(
            replay.notifications(now=NOW - timedelta(minutes=5 * (287 - poll)))
        )
    print(f"   conditions true at any moment : {len(monitor.evaluate(now=NOW))}")
    print(f"   messages actually sent in 24h  : {sent}")
    print("   Escalation is on the condition CHANGING, not on it persisting.")

    # -- the excursion workflow -------------------------------------------
    print(f"\n{'=' * 78}\n-- excursion workflow --")
    # Widen the window: the alert says "at least 24 hours" because that is all
    # it looked at. The record needs the real start.
    wide = monitor.evaluate(now=NOW, window=timedelta(hours=96))
    excursion_alert = next(a for a in wide if a.kind == AlertKind.SUSTAINED_EXCURSION)
    failed_unit = unit(excursion_alert.unit_id)
    record = open_excursion(
        excursion_alert,
        failed_unit,
        feed.readings(
            failed_unit.unit_id, since=NOW - timedelta(hours=96), until=NOW
        ),
        INVENTORY,
        excursion_id="EXC-2026-08-24-001",
        manufacturer_contacts=MANUFACTURER_CONTACTS,
    )
    print(f"\n{record.quarantine_label()}")

    print("\n-- the one judgement this module refuses to make --")
    print("   Whether exposed vaccine is still usable is the MANUFACTURER'S call.")
    print("   Nothing here says 'probably fine' or 'discard'. Every lot is")
    print("   quarantined until a person records what they were told.")
    try:
        record.close(closed_by="ma_jess", now=NOW)
    except ValueError as exc:
        print(f"\n   CANNOT CLOSE: {str(exc)[:180]}")

    for lot in list(record.affected_lots):
        record.record_disposition(
            lot.lot_number,
            disposition=LotDisposition.DISCARDED,
            decided_by="ma_jess",
            note=(
                f"Called {lot.manufacturer} 2026-08-24 ref#88412: 19.8C for 60h "
                "exceeds stability data for this product. Discard."
            ),
        )
    record.corrective_action = (
        "Compressor failed 2026-08-21 ~21:05. Unit taken out of service, "
        "service call placed, inventory moved to fridge_1. Cellular logger "
        "confirmed reporting throughout; nobody was configured to receive "
        "weekend escalations. On-call rotation created."
    )
    record.close(closed_by="dr_alvarez", now=NOW + timedelta(hours=3))
    print(f"\n   closed by {record.closed_by}; "
          f"{record.doses_at_risk} dose(s), ${record.value_at_risk_usd:,.0f} "
          "written off")

    print("\n-- what the paper log would have cost --")
    print(f"   discovered Monday 09:00, failed Friday 21:05: "
          f"{record.duration_minutes / 60:.0f} hours")
    print(f"   inventory in that unit: ${failed_unit.inventory_value_usd:,.0f}")

    # -- the monthly record ------------------------------------------------
    print(f"\n{'=' * 78}\n-- monthly compliance report --")
    report = build_compliance_report(
        month="2026-08",
        units=UNITS,
        feed=feed,
        config=config,
        # A complete day. 2026-08-24 is today and is only nine hours old, so
        # reporting on it would produce eight "incomplete day" findings that are
        # an artefact of the clock rather than anything to act on.
        start=date(2026, 8, 22),
        end=date(2026, 8, 23),
        now=NOW,
        excursions=[record],
        alerts=alerts,
    )
    print()
    print(report.render())

    print(f"\n{'=' * 78}\n-- audit --")
    print(json.dumps(audit.counts(), indent=2))


if __name__ == "__main__":
    main()
