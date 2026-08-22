"""`python3 -m officeops selftest` -- run every task against the bundled samples.

WHY THIS SHIPS: the person installing this is an office manager, not an
engineer, and the first question is "does it work on this machine". A self-test
that runs every task end to end answers it in one command, and it also answers
the second question -- "what does the output look like" -- without anyone having
to point it at real patient data first.

It is also the fastest way to check a NEW mapping: copy a task's mapping file,
edit the aliases for your EHR, and run the task against your export with
`--mapping`. If the self-test passes and your export fails, the difference is
column names, not the tool.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Any

SAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample")

#: task -> argv, using the bundled samples. `--today` pins the clock so the
#: expected findings do not drift as the calendar moves.
CASES: list[tuple[str, list[str]]] = [
    ("confirm-list", [f"{SAMPLE}/schedule.csv", "--today", "2026-08-24"]),
    ("day-sheets", [f"{SAMPLE}/schedule.csv", "--today", "2026-08-24"]),
    ("recall-list", [f"{SAMPLE}/roster.csv", "--today", "2026-08-24"]),
    ("interpreter-list", [f"{SAMPLE}/schedule.csv", "--today", "2026-08-24"]),
    ("fridge-log", [f"{SAMPLE}/fridge_log.csv", "--today", "2026-08-24"]),
    ("vaccine-inventory", [f"{SAMPLE}/vaccine_inventory.csv", "--today", "2026-08-24"]),
    ("lab-followup", [f"{SAMPLE}/orders.csv", "--today", "2026-08-24"]),
    ("referral-followup", [f"{SAMPLE}/orders.csv", "--today", "2026-08-24"]),
    ("qc-log", [f"{SAMPLE}/qc_log.csv", "--days", "14", "--today", "2026-08-24"]),
    ("expiry-sweep", [f"{SAMPLE}/crash_cart.csv", f"{SAMPLE}/sample_closet.csv",
                      "--today", "2026-08-24"]),
    ("credential-tracker", [f"{SAMPLE}/credentials.csv", "--today", "2026-08-24"]),
    ("standing-orders", [f"{SAMPLE}/standing_orders.csv", "--today", "2026-08-24"]),
    ("retention-sweep", [f"{SAMPLE}/records_index.csv", "--today", "2026-08-24"]),
    ("mail-merge", [f"{SAMPLE}/recall_letter.txt", f"{SAMPLE}/recipients.csv",
                    "--today", "2026-08-24"]),
    ("charge-reconcile", [f"{SAMPLE}/visits.csv", f"{SAMPLE}/charges.csv",
                          "--today", "2026-08-24"]),
    ("denial-worklist", [f"{SAMPLE}/denials.csv", "--today", "2026-08-24"]),
]


def run_selftest(verbose: bool = False) -> int:
    from .cli import TASKS, build_parser, run_task

    parser = build_parser()
    failures = 0
    print("officeops selftest -- every task against the bundled sample data\n")
    for name, argv in CASES:
        try:
            args = parser.parse_args([name] + argv)
            report = run_task(name, args)
            status = f"{len(report.findings):>3} finding(s)"
            if report.problems:
                status += f", {len(report.problems)} unreadable row(s)"
            print(f"  PASS  {name:<20} {status}")
            if verbose:
                print(report.render(limit=5))
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name:<20} {type(exc).__name__}: {exc}")
            if verbose:
                traceback.print_exc()
    missing = sorted(set(TASKS) - {name for name, _ in CASES})
    if missing:
        failures += 1
        print(f"\n  FAIL  no selftest case for: {missing}")
    print(f"\n{len(CASES) - failures}/{len(CASES)} tasks ran.")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run_selftest(verbose="-v" in sys.argv))
