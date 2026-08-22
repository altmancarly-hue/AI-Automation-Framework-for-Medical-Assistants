"""Revenue-cycle tasks. Arithmetic on exports the billing system already makes.

Neither of these makes a coding decision. They find visits that produced no
charge and denials nobody worked, which are both counting problems.
"""

from __future__ import annotations

import argparse
import datetime as dt
from collections import Counter, defaultdict

from ..core import Report, load_table

__all__ = ["TASKS"]


def _now(args: argparse.Namespace) -> dt.date:
    return dt.date.fromisoformat(args.today) if args.today else dt.date.today()


def _charge_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("visits", help="visit export: date, patient, provider")
    parser.add_argument("charges", help="charge export: date, patient, code, amount")
    parser.add_argument(
        "--grace-days", type=int, default=3,
        help="days a visit may go unbilled before it is listed (default: 3)",
    )


def _charge_run(args: argparse.Namespace, aliases: dict) -> Report:
    """Completed visits with no charge posted. The quietest revenue leak there is.

    Matched on (patient identifier, service date) EXACTLY. A charge's date of
    service is the visit's date of service by definition, so an earlier version
    that allowed the charge's service date to fall days after the visit was not
    modelling posting lag -- it was letting any OTHER encounter's charge within
    the window satisfy this one. A patient with an unbilled well visit on the
    10th and a billed nurse visit on the 12th came back clean.

    `--grace-days` therefore means what it says on the visit side only: a visit
    younger than the grace period is not yet expected to be billed. Posting lag
    is handled by running this against a charge export that is at least that
    fresh.

    PRESENCE of a charge row is the test, not its amount. Requiring a positive
    dollar figure reported every visit in a CPT-only export as unbilled, and
    flagged legitimate $0.00 lines (a vaccine administration billed at zero, an
    encounter and its reversal) as never billed.
    """
    visits = load_table(args.visits, aliases=aliases)
    charges = load_table(args.charges, aliases=aliases)
    visits.require("patient_name", "visit_date")
    visits.require_rows()
    charges.require("visit_date")
    now = _now(args)
    report = Report(task="charge capture reconciliation", generated=dt.datetime.now())
    report.sources = [visits.source, charges.source]
    report.parameters = {"as_of": now.isoformat(), "grace_days": args.grace_days}

    has_amount = charges.has_column("amount")
    if not has_amount:
        report.problem(
            f"{charges.source} has no amount column; matching on the presence of "
            "a charge row only, which is the correct test but means net-zero "
            "encounters cannot be identified"
        )

    posted: dict[tuple[str, dt.date], list[float]] = defaultdict(list)
    for row in charges:
        try:
            key_id = row.text("patient_id", required=False)
            key_name = row.text("patient_name", required=False).lower()
            when = row.date("visit_date")
            amount = (row.number_("amount", required=False) or 0.0) if has_amount else 0.0
            for key in {k for k in (key_id, key_name) if k}:
                posted[(key, when)].append(amount)
        except Exception as exc:  # noqa: BLE001
            report.problem(f"{charges.source} row {row.number}: {exc}")

    unbilled = 0
    net_zero = 0
    for row in visits:
        try:
            when = row.date("visit_date")
            if (now - when).days < args.grace_days:
                continue
            identifier = row.text("patient_id", required=False)
            name = row.text("patient_name")
            keys = [k for k in (identifier, name.lower()) if k]
            amounts = [a for key in keys for a in posted.get((key, when), [])]
            if amounts:
                if has_amount and abs(sum(amounts)) < 0.005:
                    # Billed and reversed, or a zero-dollar line. Interesting,
                    # but a different finding from never billed.
                    net_zero += 1
                    report.add(
                        status="net zero",
                        visit_date=when.isoformat(),
                        days_ago=(now - when).days,
                        patient=name,
                        mrn=identifier,
                        provider=row.text("provider", required=False),
                        visit_type=row.text("visit_type", required=False),
                        matched_on="mrn" if identifier else "name only (weaker)",
                    )
                continue
            unbilled += 1
            report.add(
                status="NO CHARGE",
                visit_date=when.isoformat(),
                days_ago=(now - when).days,
                patient=name,
                mrn=identifier,
                provider=row.text("provider", required=False),
                visit_type=row.text("visit_type", required=False),
                matched_on="mrn" if identifier else "name only (weaker)",
            )
        except Exception as exc:  # noqa: BLE001
            report.problem(f"{visits.source} row {row.number}: {exc}")
    report.findings.sort(key=lambda f: (f["status"] != "NO CHARGE", -f["days_ago"]))
    report.counts = {
        "as_of": now.isoformat(),
        "visits_reviewed": len(visits),
        "charges_reviewed": len(charges),
        "visits_with_no_charge": unbilled,
        "visits_netting_to_zero": net_zero,
        "grace_days": args.grace_days,
        "amount_column_present": has_amount,
    }
    report.headline = (
        f"{unbilled} completed visit(s) more than {args.grace_days} days old "
        f"have no charge posted"
        + (f", and {net_zero} netted to zero." if net_zero else ".")
    )
    return report


def _denial_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("denials", help="denial/ERA export: date, payer, code, amount")
    parser.add_argument("--days", type=int, default=90, help="window to summarise")
    parser.add_argument("--top", type=int, default=10, help="how many reason codes to rank")


def _denial_run(args: argparse.Namespace, aliases: dict) -> Report:
    """Denials grouped by reason CODE, ranked by dollars and, separately, by count.

    Grouped on the code alone, carrying the most common description alongside
    it. Grouping on code plus description split one CARC into four rows because
    four payers spell the same reason differently -- so the largest single
    reason ranked below smaller ones and the top of the work queue was wrong.

    Two rankings on purpose, and the frequency block is emitted independently
    rather than as leftovers from the dollar block. A high-count low-dollar code
    is usually one fixable front-desk habit -- an eligibility check nobody runs,
    a modifier nobody appends -- and that is the one worth a staff meeting. An
    earlier version deduped it against the dollar list and considered only the
    top three by count, so at the documented `--top 10` the frequency block came
    out empty, which is exactly the ranking the README says is invisible in a
    dollar sort.
    """
    table = load_table(args.denials, aliases=aliases)
    table.require("denial_date", "reason_code")
    table.require_rows()
    now = _now(args)
    start = now - dt.timedelta(days=args.days)
    report = Report(task="denial worklist", generated=dt.datetime.now())
    report.sources = [table.source]
    report.parameters = {
        "as_of": now.isoformat(),
        "window_days": args.days,
        "top": args.top,
    }
    amounts_by_code: dict[str, list[float]] = defaultdict(list)
    descriptions: dict[str, Counter] = defaultdict(Counter)
    by_payer: dict[str, float] = defaultdict(float)
    total = 0.0
    for row in table:
        try:
            when = row.date("denial_date")
            if not start <= when <= now:
                continue
            code = row.text("reason_code").strip().upper()
            descriptions[code][row.text("reason_description", required=False)] += 1
            payer = row.text("payer", required=False, default="unspecified")
            amount = row.number_("amount", required=False) or 0.0
            amounts_by_code[code].append(amount)
            by_payer[payer] += amount
            total += amount
        except Exception as exc:  # noqa: BLE001
            report.problem(f"row {row.number}: {exc}")

    def describe(code: str) -> str:
        common = descriptions[code].most_common(1)
        label = common[0][0] if common and common[0][0] else ""
        spellings = len([d for d in descriptions[code] if d])
        suffix = f" (+{spellings - 1} other wording)" if spellings > 1 else ""
        return f"{code} {label}{suffix}".strip()

    def emit(rank_by: str, ordering) -> None:
        for code, amounts in ordering[: args.top]:
            report.add(
                rank_by=rank_by,
                code=code,
                reason=describe(code),
                count=len(amounts),
                dollars=f"{sum(amounts):.2f}",
                average=f"{sum(amounts) / len(amounts):.2f}",
            )

    emit("dollars", sorted(amounts_by_code.items(), key=lambda kv: -sum(kv[1])))
    emit("frequency", sorted(amounts_by_code.items(), key=lambda kv: -len(kv[1])))

    report.counts = {
        "window": f"{start.isoformat()} to {now.isoformat()}",
        "denials": sum(len(v) for v in amounts_by_code.values()),
        "total_dollars": f"{total:.2f}",
        "distinct_reason_codes": len(amounts_by_code),
        "top": args.top,
        **{f"payer_{k}": f"{v:.2f}" for k, v in sorted(by_payer.items(), key=lambda kv: -kv[1])[:5]},
    }
    report.headline = (
        f"{report.counts['denials']} denial(s) worth ${total:,.2f} in the last "
        f"{args.days} days across {len(amounts_by_code)} reason code(s)."
    )
    return report


TASKS = {
    "charge-reconcile": (
        "Completed visits with no charge posted",
        _charge_args, _charge_run, "visits",
    ),
    "denial-worklist": (
        "Denials ranked by dollars and, separately, by frequency",
        _denial_args, _denial_run, "denials",
    ),
}
