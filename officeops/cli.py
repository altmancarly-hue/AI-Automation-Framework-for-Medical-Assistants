"""`python3 -m officeops <task> ...` -- one entry point for every office script.

One command rather than sixteen loose files, because a folder of scripts is a
folder nobody can inventory. `python3 -m officeops list` prints every task, what
it needs and what it produces, which is the documentation that cannot go stale.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from typing import Any, Callable, Mapping

from .core import (
    DEFAULT_MAPPING_DIR,
    OfficeOpsError,
    Report,
    add_common_arguments,
    load_mapping,
    resolve_output,
    write_csv,
    write_text,
)
from .tasks import clinical, compliance, revenue, scheduling

__all__ = ["TASKS", "build_parser", "main"]

#: name -> (help, add_arguments, run, default_mapping)
TASKS: dict[str, tuple[str, Callable, Callable, str]] = {
    **scheduling.TASKS,
    **clinical.TASKS,
    **compliance.TASKS,
    **revenue.TASKS,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="officeops",
        description=(
            "Deterministic office automation for a small independent practice. "
            "No model, no network, no vendor. Reads exports you can already "
            "produce; writes work lists."
        ),
    )
    subparsers = parser.add_subparsers(dest="task", required=True)
    subparsers.add_parser("list", help="show every task, its input and its output")
    subparsers.add_parser(
        "selftest", help="run every task against the bundled sample data"
    )
    for name, (help_text, add_arguments, _run, _mapping) in sorted(TASKS.items()):
        sub = subparsers.add_parser(name, help=help_text, description=help_text)
        add_arguments(sub)
        add_common_arguments(sub)
    return parser


def _aliases(args: argparse.Namespace, default_mapping: str) -> dict[str, list[str]]:
    """Bundled defaults first, then the caller's overrides on top.

    A `--mapping` file names the handful of columns that differ. Replacing the
    whole set with it silently dropped every alias the caller did not restate.
    """
    bundled = os.path.join(DEFAULT_MAPPING_DIR, f"{default_mapping}.yaml")
    if getattr(args, "mapping", None):
        return load_mapping(args.mapping, base=bundled)
    return load_mapping(default_mapping)


def run_task(name: str, args: argparse.Namespace) -> Report:
    _help, _add, run, default_mapping = TASKS[name]
    return run(args, _aliases(args, default_mapping))


def _emit(name: str, report: Report, args: argparse.Namespace) -> None:
    if getattr(args, "json", False):
        print(json.dumps(report.as_dict(), indent=2, default=str))
    else:
        print(report.render())
    if not getattr(args, "write", False):
        print("\n(nothing written -- pass --write to save the CSV and the report)")
        return
    # The stamp comes from --today when given, so re-running a past day writes
    # that day's file instead of overwriting today's.
    stamp = (args.today or dt.date.today().isoformat()).replace("-", "")
    slug = name.replace("-", "_")
    csv_path = write_csv(report, resolve_output(args, f"{stamp}_{slug}.csv"))
    txt_path = write_text(report.render(limit=10_000), resolve_output(args, f"{stamp}_{slug}.txt"))
    print(f"\nwrote {csv_path}\nwrote {txt_path}")
    if report.letters:
        body = ("\n\n" + "-" * 70 + "\n\n").join(report.letters)
        print("wrote " + write_text(body, resolve_output(args, f"{stamp}_{slug}_letters.txt")))


def _list() -> int:
    print("officeops -- deterministic office tasks. No AI, no network, no vendor.\n")
    groups = (
        ("front desk and scheduling", scheduling.TASKS),
        ("clinical / medical assistant", clinical.TASKS),
        ("compliance and administration", compliance.TASKS),
        ("revenue cycle", revenue.TASKS),
    )
    for title, tasks in groups:
        print(f"  {title.upper()}")
        for name, (help_text, _a, _r, mapping) in sorted(tasks.items()):
            print(f"    {name:<20} {help_text}")
            print(f"    {'':<20} input mapping: officeops/mappings/{mapping}.yaml")
        print()
    print("  every task also accepts:")
    print("    --write            save the CSV and text report (default: print only)")
    print("    --out DIR          output directory (default: ./out)")
    print("    --json             machine-readable output")
    print("    --mapping FILE     column aliases for your EHR's export")
    print("    --today YYYY-MM-DD treat this as today (re-run a past day, or test)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.task == "list":
        return _list()
    if args.task == "selftest":
        from .selftest import run_selftest

        return run_selftest()
    try:
        if getattr(args, "today", None):
            _validate_today(args.today)
        report = run_task(args.task, args)
    except OfficeOpsError as exc:
        # A refusal we anticipated.
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        # Everything else that is an INPUT problem. A missing nightly export, a
        # path that is a directory, a malformed mapping file -- all of these used
        # to escape as a traceback and exit 1, which the scheduling contract
        # reads as "findings, mail the report". The practice would have been
        # mailed a stack trace and told it was a work list.
        print(f"REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    _emit(args.task, report, args)
    if report.problems:
        # Unreadable ROWS are an input problem too, and they were invisible in
        # the exit code: an export where every row failed to parse printed twenty
        # problem lines and returned 0, which the contract reads as "stay
        # silent". Twenty families were not called and nobody was told.
        print(
            f"\n{len(report.problems)} row(s) could not be read -- exiting 2 so "
            "this reaches somebody technical.",
            file=sys.stderr,
        )
        return 2
    # Exit 1 when there are findings, so cron/Task Scheduler can mail only the
    # days that need somebody. Exit 0 on a clean run.
    return 1 if report.findings else 0


def _validate_today(text: str) -> None:
    """`--today` is ISO, and saying so beats an isoformat ValueError.

    `--today 08/25/2026` -- the format the sample data itself uses -- crashed
    with `Invalid isoformat string` and exit 1.
    """
    try:
        dt.date.fromisoformat(text)
    except ValueError as exc:
        raise OfficeOpsError(
            f"--today must be an ISO date (YYYY-MM-DD), not {text!r}"
        ) from exc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
