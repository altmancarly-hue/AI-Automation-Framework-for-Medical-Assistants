"""Compliance and administrative tasks.

Everything here is a deadline somebody is tracking in their head or in a
spreadsheet nobody else can find. A lapsed CPR card or an unrenewed license is
not a clinical problem until the day it is a very large one, and the failure
mode is always the same: the person who was tracking it left.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from collections import defaultdict
from typing import Any

import os

from ..core import OfficeOpsError, Report, load_table, write_text

__all__ = ["TASKS"]


def _now(args: argparse.Namespace) -> dt.date:
    return dt.date.fromisoformat(args.today) if args.today else dt.date.today()


# -- credential / training expiry --------------------------------------------


def _cred_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("credentials", help="staff credential export: person, type, expiry")
    parser.add_argument("--warn-days", type=int, default=90)


def _cred_run(args: argparse.Namespace, aliases: dict) -> Report:
    """Licences, CPR, CME, OSHA and HIPAA training, sorted by urgency.

    Expired first, then soonest. Grouped counts by credential type, because a
    practice with four expired CPR cards has a scheduling problem, and one with
    four different expired things has a tracking problem -- different fixes.
    """
    table = load_table(args.credentials, aliases=aliases)
    table.require("person", "credential", "expiry_date")
    table.require_rows()
    now = _now(args)
    report = Report(task="credential and training expiry", generated=dt.datetime.now())
    report.sources = [table.source]
    report.parameters = {"as_of": now.isoformat(), "warn_days": args.warn_days}
    by_type: dict[str, int] = defaultdict(int)
    for row in table:
        try:
            expiry = row.date("expiry_date")
            days = (expiry - now).days
            if days > args.warn_days:
                continue
            credential = row.text("credential")
            by_type[credential] += 1
            report.add(
                status="EXPIRED" if days < 0 else "expiring",
                person=row.text("person"),
                role=row.text("role", required=False),
                credential=credential,
                identifier=row.text("identifier", required=False),
                expiry=expiry.isoformat(),
                days=days,
            )
        except Exception as exc:  # noqa: BLE001
            report.problem(str(exc))
    report.findings.sort(key=lambda f: f["days"])
    report.counts = {
        "as_of": now.isoformat(),
        "records_reviewed": len(table),
        "expired": sum(1 for f in report.findings if f["status"] == "EXPIRED"),
        "expiring": sum(1 for f in report.findings if f["status"] == "expiring"),
        "warn_days": args.warn_days,
        **{f"by_{re.sub(r'[^a-z0-9]+', '_', k.lower())}": v for k, v in sorted(by_type.items())},
    }
    report.headline = (
        f"{report.counts['expired']} credential(s) already expired, "
        f"{report.counts['expiring']} expiring within {args.warn_days} days."
    )
    return report


# -- standing order delegation roster ----------------------------------------


def _standing_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("roster", help="standing-order roster: person, order, signed date")
    parser.add_argument(
        "--review-months", type=int, default=12,
        help="months before a delegation needs re-signing (default: 12)",
    )


def _standing_run(args: argparse.Namespace, aliases: dict) -> Report:
    """Who is signed off for which standing order, and which sign-offs are stale.

    Illinois context (225 ILCS 60/54.2): an unlicensed medical assistant acts
    under physician delegation. A standing order with no current signature is
    not a paperwork gap -- it is an MA performing a task with no documented
    authority to perform it, which is the thing an audit is looking for.

    This script does not decide who may do what. It reports what the roster
    says and how old the signature is.
    """
    table = load_table(args.roster, aliases=aliases)
    table.require("person", "standing_order")
    table.require_rows()
    now = _now(args)
    stale_before = now - dt.timedelta(days=int(args.review_months * 30.44))
    report = Report(task="standing order delegation roster", generated=dt.datetime.now())
    report.sources = [table.source]
    report.parameters = {"as_of": now.isoformat(), "review_months": args.review_months}
    # Built from ALL rows, then the ones with issues are subtracted. Populating
    # it only on the clean path made `orders_with_nobody_current` structurally
    # zero for every input that can exist -- and it is the one number an
    # Illinois delegation audit turns on.
    everyone: dict[str, set[str]] = defaultdict(set)
    current: dict[str, set[str]] = defaultdict(set)
    for row in table:
        try:
            person = row.text("person")
            order = row.text("standing_order")
            everyone[order].add(person)
            signed = row.date("signed_date", required=False)
            supervisor = row.text("supervising_physician", required=False)
            issues = []
            if signed is None:
                issues.append("no signature date recorded")
            elif signed < stale_before:
                issues.append(f"last signed {(now - signed).days} days ago")
            if not supervisor:
                issues.append("no supervising physician named")
            if issues:
                report.add(
                    person=person, role=row.text("role", required=False),
                    standing_order=order,
                    signed=signed.isoformat() if signed else "NEVER",
                    supervising_physician=supervisor or "MISSING",
                    issue="; ".join(issues),
                )
            else:
                current[order].add(person)
        except Exception as exc:  # noqa: BLE001
            report.problem(str(exc))
    uncovered = sorted(order for order in everyone if not current.get(order))
    for order in uncovered:
        report.add(
            person="(nobody)", role="-", standing_order=order,
            signed="-", supervising_physician="-",
            issue=f"NO ONE has a current delegation for this order "
                  f"({len(everyone[order])} person(s) listed, none current)",
        )
    report.counts = {
        "as_of": now.isoformat(),
        "delegations_reviewed": len(table),
        "distinct_orders": len(everyone),
        "people_currently_delegated": sum(len(v) for v in current.values()),
        "delegations_needing_attention": len(report.findings) - len(uncovered),
        "orders_with_nobody_current": len(uncovered),
        "review_interval_months": args.review_months,
    }
    report.headline = (
        f"{len(uncovered)} standing order(s) have NOBODY currently delegated, "
        f"and {len(report.findings) - len(uncovered)} individual delegation(s) "
        "have no current signature or no named supervising physician "
        "(225 ILCS 60/54.2)."
    )
    return report


# -- records retention -------------------------------------------------------


def _retention_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("records", help="records index: patient, dob, last activity")
    parser.add_argument(
        "--adult-years", type=int, default=10,
        help="years to retain after the last visit for an adult record",
    )
    parser.add_argument(
        "--minor-until-age", type=int, default=22,
        help="age until which a minor's record is retained (18 + statute)",
    )


def _birthday_on(dob: dt.date, years: int) -> dt.date:
    """The patient's Nth birthday, handling 29 February explicitly.

    `min(dob.day, 28)` silently pulled the horizon up to three days early for
    anyone born on the 29th to the 31st. A leap-day birthday has no 29 February
    in most years; the convention here is 1 March, which is the later of the two
    readings and therefore the safe one for a retention horizon.
    """
    try:
        return dob.replace(year=dob.year + years)
    except ValueError:
        return dt.date(dob.year + years, 3, 1)


def _retention_run(args: argparse.Namespace, aliases: dict) -> Report:
    """Records past their retention horizon. A LIST, never a deletion.

    This script deletes nothing and is not capable of deleting anything. It
    produces a list for a human to review against the practice's written
    retention policy, because retention varies by record type, by open
    litigation hold, and by whether the patient has asked for a copy -- none of
    which is in a records index.

    THE HORIZON IS THE LATER OF THE TWO RULES, NOT WHICHEVER ONE APPLIES.
    Choosing the minor rule for a minor gave teenagers a SHORTER retention than
    adults: two patients last seen the same day, one aged 17 and one aged 18,
    and the seventeen-year-old's chart was listed for destruction five years
    before the adult's -- the patient with the LONGER statute of limitations
    getting the shorter retention. Illinois runs both clocks (735 ILCS 5/13-212
    for a minor's action; 77 Ill. Adm. Code 250.1510 for the ordinary period),
    so the record has to survive both.

    Defaults reflect Illinois, but the numbers are arguments, not constants:
    this is a deadline calculator, not a legal opinion.
    """
    table = load_table(args.records, aliases=aliases)
    table.require("patient_name", "dob", "last_activity")
    table.require_rows()
    now = _now(args)
    report = Report(task="records retention horizon", generated=dt.datetime.now())
    report.sources = [table.source]
    report.parameters = {
        "as_of": now.isoformat(),
        "adult_years": args.adult_years,
        "minor_until_age": args.minor_until_age,
    }
    for row in table:
        try:
            dob = row.date("dob")
            last = row.date("last_activity")
            adult_horizon = last + dt.timedelta(days=int(args.adult_years * 365.25))
            minor_horizon = _birthday_on(dob, args.minor_until_age)
            # Compared on calendar birthdays, not on a day count divided by
            # 365.25 -- that put patients seen on their own eighteenth birthday
            # into different regimes depending on where the leap days fell.
            was_minor = last < _birthday_on(dob, 18)
            horizon = max(adult_horizon, minor_horizon) if was_minor else adult_horizon
            basis = (
                f"minor at last visit: later of age {args.minor_until_age} "
                f"({minor_horizon.isoformat()}) and {args.adult_years}y after "
                f"last activity ({adult_horizon.isoformat()})"
                if was_minor
                else f"adult: {args.adult_years} years after last activity"
            )
            if horizon > now:
                continue
            report.add(
                patient=row.text("patient_name"),
                mrn=row.text("patient_id", required=False),
                dob=dob.isoformat(),
                last_activity=last.isoformat(),
                eligible_since=horizon.isoformat(),
                years_past=f"{(now - horizon).days / 365.25:.1f}",
                basis=basis,
            )
        except Exception as exc:  # noqa: BLE001
            report.problem(f"row {row.number}: {exc}")
    report.findings.sort(key=lambda f: f["eligible_since"])
    report.counts = {
        "as_of": now.isoformat(),
        "records_reviewed": len(table),
        "past_horizon": len(report.findings),
        "adult_years": args.adult_years,
        "minor_until_age": args.minor_until_age,
    }
    report.headline = (
        f"{len(report.findings)} record(s) are past the retention horizon. "
        "THIS IS A REVIEW LIST. Nothing is deleted by this tool, a litigation "
        "hold or an outstanding records request overrides it, and the horizon "
        "is the LATER of the minor and adult rules."
    )
    return report


# -- mail merge --------------------------------------------------------------


_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


def _merge_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("template", help="text template using {{field}} placeholders")
    parser.add_argument("recipients", help="CSV of recipients")
    parser.add_argument(
        "--allow-blank", action="store_true",
        help="render a letter even when a placeholder has no value",
    )


def _merge_run(args: argparse.Namespace, aliases: dict) -> Report:
    """Letters from a template and a CSV. Refuses to render a blank placeholder.

    WHY THE REFUSAL IS THE POINT: the classic mail-merge failure is a letter
    that reads "Dear ," or "your appointment on ." posted to two hundred
    families. Every unfilled placeholder is reported with its row number and no
    letter is produced for that row unless `--allow-blank` says so explicitly.
    """
    with open(args.template, "r", encoding="utf-8") as handle:
        template = handle.read()
    fields = sorted(set(_PLACEHOLDER.findall(template)))
    table = load_table(args.recipients, aliases=aliases)
    table.require_rows()
    missing_columns = sorted(f for f in fields if not table.has_column(f))
    if missing_columns:
        # One refusal, not one skip row per recipient saying the same thing.
        raise OfficeOpsError(
            f"the template uses {{{{{missing_columns[0]}}}}} and "
            f"{table.source} has no column for {missing_columns}. Add an alias "
            "or change the template."
        )
    report = Report(task="mail merge", generated=dt.datetime.now())
    report.sources = [args.template, table.source]
    report.parameters = {
        "template": os.path.basename(args.template),
        "allow_blank": bool(args.allow_blank),
    }
    letters: list[str] = []
    rendered = 0
    for row in table:
        try:
            values = {name: str(row.get(name, "") or "").strip() for name in fields}
            blanks = sorted(name for name, value in values.items() if not value)
            if blanks and not args.allow_blank:
                report.add(
                    row=row.number, status="SKIPPED",
                    recipient=row.text("patient_name", required=False, default="?"),
                    detail=f"no value for {', '.join(blanks)}",
                )
                continue
            body = _PLACEHOLDER.sub(lambda m: values.get(m.group(1), ""), template)
            letters.append(body)
            rendered += 1
            if blanks:
                report.add(
                    row=row.number, status="rendered with blanks",
                    recipient=values.get("patient_name", "?"),
                    detail=f"empty: {', '.join(blanks)}",
                )
        except Exception as exc:  # noqa: BLE001
            report.problem(str(exc))
    report.counts = {
        "recipients": len(table),
        "letters_rendered": rendered,
        "skipped_for_blanks": sum(1 for f in report.findings if f["status"] == "SKIPPED"),
        "placeholders": ", ".join(fields),
    }
    report.headline = (
        f"{rendered} letter(s) rendered from {len(table)} recipient(s); "
        f"{report.counts['skipped_for_blanks']} skipped for missing values."
    )
    # Held OUT of `counts` and out of the rendered report. Rendering them into
    # the report body duplicated every letter -- names, addresses, visit dates --
    # into the .txt artifact AND the JSON, which put a second copy of the PHI on
    # disk in a file the retention policy did not know about.
    report.letters = letters
    return report


TASKS = {
    "credential-tracker": (
        "Licences, CPR, CME and mandatory training coming due",
        _cred_args, _cred_run, "credentials",
    ),
    "standing-orders": (
        "Delegation roster: stale signatures and unnamed supervisors",
        _standing_args, _standing_run, "standing_orders",
    ),
    "retention-sweep": (
        "Records past the retention horizon (a review list, never a deletion)",
        _retention_args, _retention_run, "records",
    ),
    "mail-merge": (
        "Letters from a template, refusing to post a blank placeholder",
        _merge_args, _merge_run, "recipients",
    ),
}
