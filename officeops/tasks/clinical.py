"""Back-office clinical tasks. The MA's recurring paperwork, done arithmetically.

Nothing here makes a clinical decision. These scripts find gaps in records --
a temperature outside range, a lot about to expire, an order with no result --
and hand the list to a person. That distinction is the same one I-04 draws
between documenting and deciding, and it applies just as hard to a script.
"""

from __future__ import annotations

import argparse
import datetime as dt
from collections import defaultdict
from typing import Any, Mapping

from ..core import (
    OfficeOpsError,
    load_mapping,
    Report,
    add_common_arguments,
    load_table,
    parse_float,
)
from ..core import _normal

__all__ = ["TASKS"]


def _now(args: argparse.Namespace) -> dt.date:
    return dt.date.fromisoformat(args.today) if args.today else dt.date.today()


# -- vaccine fridge / freezer temperature log --------------------------------


def _fridge_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("log", help="data-logger export (CSV) with timestamp and temperature")
    parser.add_argument("--unit", default="", help="only this storage unit")
    parser.add_argument("--min-c", type=float, default=2.0, help="lower bound, Celsius (fridge: 2)")
    parser.add_argument("--max-c", type=float, default=8.0, help="upper bound, Celsius (fridge: 8)")
    parser.add_argument(
        "--freezer", action="store_true",
        help="use freezer bounds (-50 to -15 C) instead",
    )
    parser.add_argument(
        "--expect-readings-per-day", type=int, default=2,
        help="minimum readings a compliant day has (default: 2, the CDC minimum)",
    )


def _fridge_run(args: argparse.Namespace, aliases: dict) -> Report:
    """Excursions, under-logged days, and a logger that stopped reporting.

    CDC's Vaccine Storage and Handling Toolkit asks for twice-daily readings.
    A gap in the log is not a clean day; it is an unmonitored day, and it is the
    thing that turns a power cut into a whole-fridge loss nobody can bound. So
    this reports missing days as loudly as it reports excursions, which a
    logger's own software generally does not.

    THREE THINGS THIS GETS RIGHT THAT THE OBVIOUS IMPLEMENTATION DOES NOT:

      * **Everything is partitioned by storage unit.** One excursion state and
        one day counter shared across units meant an in-range reading on Fridge
        2 closed an open excursion on Fridge 1: a continuous 48-hour breach was
        reported as two five-minute events, on the wrong fridge, and the pooled
        day counter hid every gap.
      * **The window ends TODAY, not at the last row in the file.** A logger
        that died in July produced "0 excursions, 0 under-logged days, nothing
        to report" and exit 0. Fifty-three unmonitored days read as a clean run.
      * **Readings are sorted before analysis.** A newest-first export produced
        an excursion lasting minus twenty-four hours.
    """
    table = load_table(args.log, aliases=aliases)
    table.require("timestamp", "temperature_c")
    table.require_rows()
    low, high = (-50.0, -15.0) if args.freezer else (args.min_c, args.max_c)
    if args.freezer and (args.min_c != 2.0 or args.max_c != 8.0):
        # Silently overriding an explicit bound made a -16.0 C breach invisible.
        raise OfficeOpsError(
            "--freezer sets its own bounds (-50 to -15 C) and cannot be combined "
            "with --min-c/--max-c. Pass one or the other."
        )
    now = _now(args)
    report = Report(task="vaccine storage temperature review", generated=dt.datetime.now())
    report.sources = [table.source]
    report.parameters = {
        "as_of": now.isoformat(),
        "range_c": f"{low} to {high}",
        "expect_readings_per_day": args.expect_readings_per_day,
        "unit_filter": args.unit or "(all units)",
    }

    # -- read, normalise and bucket by unit ---------------------------------
    wanted = _normal(args.unit) if args.unit else ""
    seen_units: set[str] = set()
    by_unit: dict[str, list[tuple[dt.datetime, float]]] = defaultdict(list)
    for row in table:
        try:
            unit = row.text("unit", required=False, default="unit-1")
            seen_units.add(unit)
            if wanted and _normal(unit) != wanted:
                continue
            when = row.datetime("timestamp")
            # Drop the offset rather than mixing naive and aware datetimes: one
            # ISO stamp with a zone among naive ones raised TypeError mid-run
            # and left an excursion permanently open.
            if when.tzinfo is not None:
                when = when.replace(tzinfo=None)
            by_unit[unit].append((when, row.number_("temperature_c")))
        except Exception as exc:  # noqa: BLE001
            report.problem(f"row {row.number}: {exc}")

    if wanted and not by_unit:
        raise OfficeOpsError(
            f"--unit {args.unit!r} matched no readings. A typo here is "
            f"indistinguishable from a clean fridge. Units in this file: "
            f"{sorted(seen_units) or '(none)'}"
        )

    readings = 0
    excursion_readings = 0
    under_logged = 0
    for unit in sorted(by_unit):
        points = sorted(by_unit[unit])
        readings += len(points)
        run_start: dt.datetime | None = None
        run_extreme = 0.0
        per_day: dict[dt.date, int] = defaultdict(int)
        for when, temp in points:
            per_day[when.date()] += 1
            if temp < low or temp > high:
                excursion_readings += 1
                if run_start is None:
                    run_start, run_extreme = when, temp
                else:
                    run_extreme = min(run_extreme, temp) if temp < low else max(run_extreme, temp)
            elif run_start is not None:
                report.add(
                    kind="EXCURSION", unit=unit,
                    start=run_start.strftime("%Y-%m-%d %H:%M"),
                    end=when.strftime("%Y-%m-%d %H:%M"),
                    minutes=int((when - run_start).total_seconds() // 60),
                    worst_c=f"{run_extreme:g}",
                    detail=f"outside {low} to {high} C",
                )
                run_start = None
        if run_start is not None:
            last = points[-1][0]
            report.add(
                kind="EXCURSION", unit=unit,
                start=run_start.strftime("%Y-%m-%d %H:%M"),
                end=f"STILL OUT OF RANGE AT {last:%Y-%m-%d %H:%M}",
                minutes=int((last - run_start).total_seconds() // 60),
                worst_c=f"{run_extreme:g}",
                detail=f"outside {low} to {high} C and never came back in the log",
            )

        # Gaps run to TODAY, not to the last row present.
        first_day = min(per_day) if per_day else now
        for offset in range((now - first_day).days + 1):
            day = first_day + dt.timedelta(days=offset)
            count = per_day.get(day, 0)
            if count >= args.expect_readings_per_day:
                continue
            under_logged += 1
            report.add(
                kind="NO READINGS" if count == 0 else "LOG GAP", unit=unit,
                start=day.isoformat(), end=day.isoformat(), minutes=0, worst_c="-",
                detail=f"{count} reading(s), expected {args.expect_readings_per_day}",
            )

    report.findings.sort(key=lambda f: (f["unit"], f["start"]))
    excursions = sum(1 for f in report.findings if f["kind"] == "EXCURSION")
    silent = sum(1 for f in report.findings if f["kind"] == "NO READINGS")
    report.counts = {
        "as_of": now.isoformat(),
        "units": len(by_unit),
        "readings": readings,
        "readings_out_of_range": excursion_readings,
        "excursion_events": excursions,
        "days_under_logged": under_logged,
        "days_with_no_readings_at_all": silent,
        "range_c": f"{low} to {high}",
    }
    report.headline = (
        f"{excursions} excursion(s), {under_logged} under-logged day(s) "
        f"({silent} with no reading at all) across {readings} readings in "
        f"{len(by_unit)} unit(s)."
    )
    return report


# -- vaccine inventory / expiry ----------------------------------------------


def _inventory_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("inventory", help="vaccine inventory export: lot, expiry, doses, funding")
    parser.add_argument("--warn-days", type=int, default=60, help="expiry warning window")


def _inventory_run(args: argparse.Namespace, aliases: dict) -> Report:
    """Expiring lots, expired lots still on the shelf, and VFC borrowing.

    Sorted by soonest expiry, with expired lots first: an expired dose in the
    fridge is not a planning problem, it is a dose somebody can draw up today.
    """
    table = load_table(args.inventory, aliases=aliases)
    table.require("vaccine", "lot", "expiry_date")
    table.require_rows()
    now = _now(args)
    report = Report(task="vaccine inventory and expiry", generated=dt.datetime.now())
    report.sources = [table.source]
    report.parameters = {"as_of": now.isoformat(), "warn_days": args.warn_days}
    if not table.has_column("doses_on_hand"):
        # Otherwise it silently reports "1 expired lot (0 doses)", and the dose
        # count is what makes the finding actionable.
        report.problem(
            f"{table.source} has no doses-on-hand column; dose counts are "
            "reported as zero and doses_at_risk is meaningless"
        )
    doses_expiring = 0
    by_funding: dict[str, int] = defaultdict(int)
    for row in table:
        try:
            expiry = row.date("expiry_date")
            doses = int(row.number_("doses_on_hand", required=False) or 0)
            funding = row.text("funding", required=False, default="unspecified")
            by_funding[funding] += doses
            days = (expiry - now).days
            if days > args.warn_days:
                continue
            doses_expiring += doses
            report.add(
                status="EXPIRED" if days < 0 else "expiring",
                vaccine=row.text("vaccine"),
                lot=row.text("lot"),
                expiry=expiry.isoformat(),
                days=days,
                doses=doses,
                funding=funding,
            )
        except Exception as exc:  # noqa: BLE001
            report.problem(str(exc))
    report.findings.sort(key=lambda f: f["days"])
    report.counts = {
        "as_of": now.isoformat(),
        "lots_reviewed": len(table),
        "expired_lots": sum(1 for f in report.findings if f["status"] == "EXPIRED"),
        "lots_within_window": sum(1 for f in report.findings if f["status"] == "expiring"),
        "doses_at_risk": doses_expiring,
        "warn_days": args.warn_days,
        **{f"doses_{k}": v for k, v in sorted(by_funding.items())},
    }
    report.headline = (
        f"{report.counts['expired_lots']} expired lot(s) on the shelf and "
        f"{report.counts['lots_within_window']} lot(s) expiring within "
        f"{args.warn_days} days ({doses_expiring} doses)."
    )
    return report


# -- outstanding labs and referrals ------------------------------------------


def _followup_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("orders", help="orders export: order date, type, result/report date")
    parser.add_argument("--days", type=int, default=14, help="days before an order is chased")


#: Order-type words that mean "a lab" and "a referral". A bare substring test
#: on "lab" pulled in "Collaborative care referral" and "Labor and delivery
#: records request", both of which landed on the lab chase list.
_ORDER_KINDS: Mapping[str, frozenset[str]] = {
    "lab": frozenset({"lab", "labs", "laboratory", "labwork", "labtest", "test",
                      "specimen", "bloodwork", "blooddraw"}),
    "referral": frozenset({"referral", "referrals", "refer", "consult",
                           "consultation", "specialist"}),
}


def _make_followup(kind: str, date_field: str, label: str):
    def run(args: argparse.Namespace, aliases: dict) -> Report:
        table = load_table(args.orders, aliases=aliases)
        # `order_type` and the result/report column both decide the answer. With
        # neither present, referrals counted as labs and every order looked
        # outstanding.
        table.require("patient_name", "order_date", "order_type", date_field)
        table.require_rows()
        now = _now(args)
        report = Report(task=f"outstanding {label}", generated=dt.datetime.now())
        report.sources = [table.source]
        report.parameters = {
            "as_of": now.isoformat(), "threshold_days": args.days, "order_kind": kind,
        }
        total = 0
        for row in table:
            try:
                order_type = row.text("order_type", required=False, default=kind)
                if _normal(order_type) not in _ORDER_KINDS[kind]:
                    continue
                total += 1
                returned = row.date(date_field, required=False)
                if returned is not None:
                    continue
                ordered = row.date("order_date")
                waiting = (now - ordered).days
                if waiting < args.days:
                    continue
                report.add(
                    days_open=waiting,
                    patient=row.text("patient_name"),
                    mrn=row.text("patient_id", required=False),
                    ordered=ordered.isoformat(),
                    detail=row.text("description", required=False),
                    ordered_by=row.text("provider", required=False),
                )
            except Exception as exc:  # noqa: BLE001
                report.problem(f"row {row.number}: {exc}")
        report.findings.sort(key=lambda f: -f["days_open"])
        report.counts = {
            "as_of": now.isoformat(),
            f"{label}_reviewed": total,
            "outstanding": len(report.findings),
            "threshold_days": args.days,
            "oldest_days": report.findings[0]["days_open"] if report.findings else 0,
        }
        report.headline = (
            f"{len(report.findings)} {label} placed more than {args.days} days ago "
            "with nothing back."
        )
        return report

    return run


# -- CLIA-waived QC completeness ---------------------------------------------


def _qc_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("qc_log", help="QC log export: date, test, control, result")
    parser.add_argument("--days", type=int, default=30, help="days back to check")
    parser.add_argument(
        "--tests", default="strep,flu,urine,covid",
        help="comma-separated test names that require daily QC",
    )
    parser.add_argument(
        "--skip-weekends", action="store_true",
        help="do not expect QC on days the lab is closed",
    )


#: Results that mean the control was acceptable, and results that mean it was
#: not. Anything else is UNKNOWN -- never silently a pass and never a failure.
#: An earlier version's allow-list was exactly ("pass","acceptable","in range",
#: "ok"), so "Passed", "in-range", "within range" and "2/2 pass" all reported as
#: QC FAILURES beside the one real failure, and a blank result counted the day
#: as complete -- which is precisely the row a CLIA inspector writes up.
_QC_PASS = frozenset({
    "pass", "passed", "passing", "acceptable", "accept", "inrange", "withinrange",
    "ok", "okay", "good", "valid", "22pass", "11pass", "normal", "expected",
})
_QC_FAIL = frozenset({
    "fail", "failed", "failing", "outofrange", "unacceptable", "invalid",
    "repeat", "repeated", "reject", "rejected", "abnormal", "unexpected",
})


def _qc_run(args: argparse.Namespace, aliases: dict) -> Report:
    """Days with no recorded control, per test. The gap IS the finding.

    A CLIA-waived test run on a day with no control is a result that cannot be
    defended in an inspection, and the log's own software will happily show you
    the days that ARE there.

    Test names are matched on a normal form -- case, spaces and punctuation
    removed, and either side may contain the other. `--tests strep,flu,urine`
    against a log recording "Strep A", "Influenza A/B" and "Urinalysis"
    previously matched nothing at all and reported 0% completeness on a perfect
    log, which is the most likely first run in any real practice and looks
    exactly like a catastrophic compliance finding.
    """
    table = load_table(args.qc_log, aliases=aliases)
    table.require("date", "test")
    table.require_rows()
    now = _now(args)
    # The window ENDS YESTERDAY. Including today manufactured one guaranteed
    # false "NO QC" per test on every morning run, and the alert that is always
    # wrong is the one staff learn to delete.
    end_day = now - dt.timedelta(days=1)
    start_day = end_day - dt.timedelta(days=args.days - 1)
    wanted = [t.strip() for t in args.tests.split(",") if t.strip()]
    report = Report(task="CLIA-waived QC completeness", generated=dt.datetime.now())
    report.sources = [table.source]
    report.parameters = {
        "as_of": now.isoformat(),
        "window": f"{start_day.isoformat()} to {end_day.isoformat()}",
        "tests": ", ".join(wanted),
        "skip_weekends": bool(args.skip_weekends),
    }

    # Test-name aliases are data, like the column names, and for the same
    # reason: "flu" and "Influenza A/B" are one assay, and no amount of
    # substring matching connects them.
    synonyms = load_mapping("qc_tests")
    lookup: dict[str, str] = {}
    for canonical_name, spellings in synonyms.items():
        for spelling in [canonical_name, *spellings]:
            lookup[_normal(spelling)] = canonical_name

    def canonical(name: str) -> str | None:
        normal = _normal(name)
        resolved = lookup.get(normal)
        for candidate in wanted:
            target = _normal(candidate)
            canonical_target = lookup.get(target, candidate)
            if resolved is not None and resolved == lookup.get(target, target):
                return candidate
            if normal == target or target in normal or normal in target:
                return candidate
            # The log's canonical form against the argument's canonical form.
            if resolved is not None and _normal(resolved) == _normal(canonical_target):
                return candidate
        return None

    seen: dict[tuple[str, dt.date], str] = {}
    unmatched: dict[str, int] = defaultdict(int)
    in_window = 0
    for row in table:
        try:
            when = row.date("date")
            if not start_day <= when <= end_day:
                continue
            in_window += 1
            raw_test = row.text("test")
            test = canonical(raw_test)
            if test is None:
                unmatched[raw_test] += 1
                continue
            result = row.text("result", required=False, default="").strip()
            seen[(test, when)] = result
            verdict = _normal(result)
            if not result:
                report.add(
                    kind="NO RESULT", test=test, date=when.isoformat(),
                    detail="a control was run and no result was recorded",
                )
            elif verdict in _QC_FAIL:
                report.add(
                    kind="QC FAIL", test=test, date=when.isoformat(),
                    detail=f"control result recorded as {result!r}",
                )
            elif verdict not in _QC_PASS:
                report.add(
                    kind="UNKNOWN RESULT", test=test, date=when.isoformat(),
                    detail=f"{result!r} is neither a recognised pass nor a fail",
                )
        except Exception as exc:  # noqa: BLE001
            report.problem(f"row {row.number}: {exc}")

    if unmatched and not seen:
        # Every row in the window was for a test nobody asked about. Reporting
        # 0% completeness here would be a lie about a log that may be perfect.
        raise OfficeOpsError(
            f"none of the tests in {args.qc_log} match --tests {args.tests!r}. "
            f"The log contains: {sorted(unmatched)}. Pass --tests with the names "
            "your log actually uses."
        )
    for name, count in sorted(unmatched.items()):
        report.problem(
            f"{count} row(s) for {name!r}, which matches nothing in --tests"
        )

    expected = 0
    for test in wanted:
        for offset in range(args.days):
            day = start_day + dt.timedelta(days=offset)
            if args.skip_weekends and day.weekday() >= 5:
                continue
            expected += 1
            if (test, day) not in seen:
                report.add(
                    kind="NO QC", test=test, date=day.isoformat(),
                    detail="no control recorded for this test on this day",
                )
    missing = sum(1 for f in report.findings if f["kind"] == "NO QC")
    report.findings.sort(key=lambda f: (f["test"], f["date"]))
    report.counts = {
        "window": report.parameters["window"],
        "tests_tracked": len(wanted),
        "rows_in_window": in_window,
        "test_days_expected": expected,
        "test_days_recorded": expected - missing,
        "missing_qc_days": missing,
        "qc_failures": sum(1 for f in report.findings if f["kind"] == "QC FAIL"),
        "controls_with_no_result": sum(1 for f in report.findings if f["kind"] == "NO RESULT"),
        "unrecognised_results": sum(1 for f in report.findings if f["kind"] == "UNKNOWN RESULT"),
        "completeness": f"{(expected - missing) / expected:.0%}" if expected else "n/a",
    }
    report.headline = (
        f"{missing} test-day(s) with no control recorded, "
        f"{report.counts['qc_failures']} recorded failure(s) and "
        f"{report.counts['controls_with_no_result']} control(s) with no result, "
        f"over {args.days} days."
    )
    return report


# -- generic expiry sweep ----------------------------------------------------


def _expiry_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "inventories", nargs="+",
        help="one or more inventory exports (crash cart, sample closet, supplies)",
    )
    parser.add_argument("--warn-days", type=int, default=45)


def _expiry_run(args: argparse.Namespace, aliases: dict) -> Report:
    """Everything with an expiry date, across every list the practice keeps.

    Deliberately takes several files: the emergency kit, the sample closet and
    the supply room are three spreadsheets maintained by three people, and the
    one that gets forgotten is never the one somebody is currently looking at.
    """
    now = _now(args)
    report = Report(task="expiry sweep", generated=dt.datetime.now())
    report.parameters = {"as_of": now.isoformat(), "warn_days": args.warn_days}
    reviewed = 0
    for path in args.inventories:
        table = load_table(path, aliases=aliases)
        table.require("item", "expiry_date")
        table.require_rows()
        report.sources.append(table.source)
        for row in table:
            reviewed += 1
            try:
                expiry = row.date("expiry_date")
                days = (expiry - now).days
                if days > args.warn_days:
                    continue
                report.add(
                    status="EXPIRED" if days < 0 else "expiring",
                    source=table.source,
                    item=row.text("item"),
                    location=row.text("location", required=False),
                    lot=row.text("lot", required=False),
                    expiry=expiry.isoformat(),
                    days=days,
                    quantity=row.text("quantity", required=False),
                )
            except Exception as exc:  # noqa: BLE001
                report.problem(f"{table.source}: {exc}")
    report.findings.sort(key=lambda f: f["days"])
    report.counts = {
        "as_of": now.isoformat(),
        "files": len(args.inventories),
        "items_reviewed": reviewed,
        "expired": sum(1 for f in report.findings if f["status"] == "EXPIRED"),
        "expiring": sum(1 for f in report.findings if f["status"] == "expiring"),
        "warn_days": args.warn_days,
    }
    report.headline = (
        f"{report.counts['expired']} expired item(s) still listed and "
        f"{report.counts['expiring']} expiring within {args.warn_days} days."
    )
    return report


TASKS = {
    "fridge-log": (
        "Vaccine storage excursions AND under-logged days",
        _fridge_args, _fridge_run, "log",
    ),
    "vaccine-inventory": (
        "Expired and expiring vaccine lots, by funding source",
        _inventory_args, _inventory_run, "inventory",
    ),
    "lab-followup": (
        "Labs ordered with no result back",
        _followup_args, _make_followup("lab", "result_date", "labs"), "orders",
    ),
    "referral-followup": (
        "Referrals placed with no report back",
        _followup_args, _make_followup("referral", "report_date", "referrals"), "orders",
    ),
    "qc-log": (
        "CLIA-waived QC completeness -- the missing days, not the recorded ones",
        _qc_args, _qc_run, "qc_log",
    ),
    "expiry-sweep": (
        "Expiring items across every inventory the practice keeps",
        _expiry_args, _expiry_run, "inventories",
    ),
}
