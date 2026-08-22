"""Front-desk tasks. Schedule exports in, work lists out.

None of this needs a model and none of it needs an EHR integration. Every task
here reads the schedule export the practice already prints each afternoon.
"""

from __future__ import annotations

import argparse
import datetime as dt
from collections import defaultdict
from typing import Any

from ..core import (
    OfficeOpsError,
    Report,
    Table,
    add_common_arguments,
    load_table,
    parse_date,
)

__all__ = ["TASKS"]


def _load(args: argparse.Namespace, aliases: dict, path: str) -> Table:
    return load_table(path, aliases=aliases)


def _now(args: argparse.Namespace) -> dt.date:
    return dt.date.fromisoformat(args.today) if args.today else dt.date.today()


# -- confirmation call list --------------------------------------------------


def _confirm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("schedule", help="tomorrow's schedule export (CSV or XLSX)")
    parser.add_argument(
        "--days-ahead", type=int, default=1,
        help="how many days out to build the list for (default: tomorrow)",
    )
    parser.add_argument(
        "--include-confirmed", action="store_true",
        help="list every appointment, not only the unconfirmed ones",
    )


def _confirm_run(args: argparse.Namespace, aliases: dict) -> Report:
    """Who has not confirmed tomorrow, in the order somebody would call them.

    Sorted by appointment time rather than by name, because the person working
    this list is calling to fill the front of the day first -- an unconfirmed
    8:20 is worth ten minutes more attention than an unconfirmed 4:40.
    """
    table = load_table(args.schedule, aliases=aliases)
    # `confirmed` is required, not optional. Without the column every patient
    # read NOT CONFIRMED and the rate read 0% -- a number the phasing plan then
    # hands to I-07 as a MEASURED baseline.
    table.require("patient_name", "appointment_datetime", "confirmed")
    table.require_rows()
    target = _now(args) + dt.timedelta(days=args.days_ahead)
    report = Report(task="confirmation call list", generated=dt.datetime.now())
    report.sources = [table.source]
    report.parameters = {
        "as_of": _now(args).isoformat(),
        "clinic_date": target.isoformat(),
        "days_ahead": args.days_ahead,
    }
    total = 0
    confirmed = 0
    for row in table:
        try:
            when = row.datetime("appointment_datetime")
            if when.date() != target:
                continue
            total += 1
            is_confirmed = row.flag("confirmed")
            if is_confirmed:
                confirmed += 1
                if not args.include_confirmed:
                    continue
            report.add(
                time=when.strftime("%H:%M"),
                patient=row.text("patient_name"),
                mrn=row.text("patient_id", required=False),
                phone=row.text("phone", required=False),
                provider=row.text("provider", required=False),
                visit_type=row.text("visit_type", required=False),
                status="confirmed" if is_confirmed else "NOT CONFIRMED",
            )
        except Exception as exc:  # noqa: BLE001 - a bad row is a patient nobody calls
            report.problem(f"row {row.number}: {exc}")
    report.findings.sort(key=lambda f: f["time"])
    report.counts = {
        "appointments_on": target.isoformat(),
        "total": total,
        "confirmed": confirmed,
        "unconfirmed": total - confirmed,
        "confirmation_rate": f"{confirmed / total:.0%}" if total else "n/a",
    }
    report.headline = (
        f"{total - confirmed} of {total} appointments on {target:%a %d %b} are "
        "not confirmed."
    )
    return report


# -- per-provider day sheets -------------------------------------------------


def _sheets_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("schedule", help="schedule export")
    parser.add_argument("--days-ahead", type=int, default=1)


def _sheets_run(args: argparse.Namespace, aliases: dict) -> Report:
    """One printable block per provider, in time order."""
    table = load_table(args.schedule, aliases=aliases)
    table.require("patient_name", "appointment_datetime")
    table.require_rows()
    target = _now(args) + dt.timedelta(days=args.days_ahead)
    by_provider: dict[str, list[dict[str, Any]]] = defaultdict(list)
    report = Report(task="provider day sheets", generated=dt.datetime.now())
    report.sources = [table.source]
    report.parameters = {"clinic_date": target.isoformat()}
    for row in table:
        try:
            when = row.datetime("appointment_datetime")
            if when.date() != target:
                continue
            provider = row.text("provider", required=False, default="UNASSIGNED")
            by_provider[provider].append(
                {
                    "time": when.strftime("%H:%M"),
                    "patient": row.text("patient_name"),
                    "dob": row.text("dob", required=False),
                    "visit_type": row.text("visit_type", required=False),
                    "notes": row.text("notes", required=False),
                }
            )
        except Exception as exc:  # noqa: BLE001
            report.problem(f"row {row.number}: {exc}")
    lines: list[str] = []
    for provider in sorted(by_provider):
        entries = sorted(by_provider[provider], key=lambda e: e["time"])
        lines.append("")
        lines.append("=" * 74)
        lines.append(f"{provider}  --  {target:%A %d %B %Y}  --  {len(entries)} patients")
        lines.append("=" * 74)
        for entry in entries:
            lines.append(
                f"  {entry['time']}  {entry['patient']:<28}  {entry['visit_type']:<18}"
                f"  {entry['dob']}"
            )
            if entry["notes"]:
                lines.append(f"          note: {entry['notes']}")
        report.add(provider=provider, patients=len(entries))
    report.counts = {
        "clinic_date": target.isoformat(),
        "providers": len(by_provider),
        "appointments": sum(len(v) for v in by_provider.values()),
    }
    report.headline = "\n".join(lines).strip()
    return report


# -- well-visit recall list --------------------------------------------------


def _recall_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("roster", help="patient roster export with last well-visit dates")
    parser.add_argument(
        "--overdue-months", type=int, default=15,
        help="months since the last well visit before a patient is listed (default: 15)",
    )
    parser.add_argument(
        "--max-age-years", type=int, default=21,
        help="patients older than this are no longer this practice's recall (default: 21)",
    )


def _recall_run(args: argparse.Namespace, aliases: dict) -> Report:
    """Children overdue for a well visit, worst first.

    Fifteen months rather than twelve by default: a family who comes at 12 and
    a half months is not overdue, and a recall list that includes them trains
    the front desk to ignore it.
    """
    table = load_table(args.roster, aliases=aliases)
    # Without the last-well-visit column the whole panel reads NONE ON FILE,
    # and this list becomes a mailing.
    table.require("patient_name", "dob", "last_well_visit")
    table.require_rows()
    now = _now(args)
    report = Report(task="well-visit recall list", generated=dt.datetime.now())
    report.sources = [table.source]
    report.parameters = {
        "as_of": now.isoformat(),
        "overdue_months": args.overdue_months,
        "max_age_years": args.max_age_years,
    }
    considered = 0
    for row in table:
        try:
            dob = row.date("dob")
            age_days = (now - dob).days
            if age_days / 365.25 > args.max_age_years:
                continue
            considered += 1
            last = row.date("last_well_visit", required=False)
            months = None if last is None else (now - last).days / 30.44
            if last is not None and months < args.overdue_months:
                continue
            years = (now.year - dob.year) - ((now.month, now.day) < (dob.month, dob.day))
            months_part = (now.month - dob.month - (now.day < dob.day)) % 12
            report.add(
                patient=row.text("patient_name"),
                mrn=row.text("patient_id", required=False),
                # Calendar arithmetic, not days//365 -- that produced "4y 12m".
                age=f"{years}y {months_part}m",
                last_well_visit=last.isoformat() if last else "NONE ON FILE",
                # Sorted on the exact value; the displayed figure is rounded and
                # two patients ten days apart otherwise tied.
                months_since=f"{months:.0f}" if months is not None else "-",
                _sort=-(months if months is not None else 10_000.0),
                phone=row.text("phone", required=False),
            )
        except Exception as exc:  # noqa: BLE001
            report.problem(f"row {row.number}: {exc}")
    report.findings.sort(key=lambda f: f["_sort"])
    for finding in report.findings:
        finding.pop("_sort", None)
    if "_sort" in report.columns:
        report.columns.remove("_sort")
    report.counts = {
        "as_of": now.isoformat(),
        "patients_considered": considered,
        "overdue": len(report.findings),
        "never_recorded": sum(
            1 for f in report.findings if f["last_well_visit"] == "NONE ON FILE"
        ),
        "overdue_threshold_months": args.overdue_months,
    }
    report.headline = (
        f"{len(report.findings)} of {considered} children have no well visit in "
        f"the last {args.overdue_months} months."
    )
    return report


# -- interpreter needs -------------------------------------------------------


def _interpreter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("schedule", help="schedule export including preferred language")
    parser.add_argument("--days-ahead", type=int, default=1)
    parser.add_argument(
        "--english", default="english,en,eng",
        help="comma-separated values that mean no interpreter is needed",
    )


def _interpreter_run(args: argparse.Namespace, aliases: dict) -> Report:
    """Which of tomorrow's visits need an interpreter, grouped by language.

    Grouped because booking is per language per session, not per patient: four
    Spanish visits spread across a morning is one interpreter booking, and a
    list that does not group them gets four separate phone calls or none.
    """
    table = load_table(args.schedule, aliases=aliases)
    table.require("patient_name", "appointment_datetime", "language")
    table.require_rows()
    english = {e.strip().lower() for e in args.english.split(",") if e.strip()}
    target = _now(args) + dt.timedelta(days=args.days_ahead)
    report = Report(task="interpreter needs", generated=dt.datetime.now())
    report.sources = [table.source]
    report.parameters = {"clinic_date": target.isoformat(), "english": args.english}
    by_language: dict[str, int] = defaultdict(int)
    for row in table:
        try:
            when = row.datetime("appointment_datetime")
            if when.date() != target:
                continue
            language = row.text("language", required=False, default="").strip()
            if not language or language.lower() in english:
                continue
            by_language[language] += 1
            report.add(
                time=when.strftime("%H:%M"),
                language=language,
                patient=row.text("patient_name"),
                provider=row.text("provider", required=False),
                interpreter_needed=row.text("interpreter_needed", required=False, default="?"),
            )
        except Exception as exc:  # noqa: BLE001
            report.problem(f"row {row.number}: {exc}")
    report.findings.sort(key=lambda f: (f["language"], f["time"]))
    report.counts = {"clinic_date": target.isoformat(), **dict(sorted(by_language.items()))}
    report.headline = (
        f"{len(report.findings)} visit(s) on {target:%a %d %b} in "
        f"{len(by_language)} language(s) other than English."
        if report.findings
        else f"No non-English visits scheduled for {target:%a %d %b}."
    )
    return report


TASKS = {
    "confirm-list": (
        "Tomorrow's unconfirmed appointments, in call order",
        _confirm_args, _confirm_run, "schedule",
    ),
    "day-sheets": (
        "Printable per-provider day sheets",
        _sheets_args, _sheets_run, "schedule",
    ),
    "recall-list": (
        "Children overdue for a well visit",
        _recall_args, _recall_run, "roster",
    ),
    "interpreter-list": (
        "Interpreter needs for a clinic day, grouped by language",
        _interpreter_args, _interpreter_run, "schedule",
    ),
}
