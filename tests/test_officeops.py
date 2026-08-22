"""officeops tests. Real CSVs, real argparse, real output.

These run the production CLI path end to end against the bundled sample data.
There are no mocks: the sample files are deliberately messy -- mixed date
formats, vendor-spelled headers, blank cells -- because sample data that is
already clean tests nothing and the first thing a practice discovers is that
its export is not clean.
"""

from __future__ import annotations

import ast
import csv
import datetime as dt
import io
import json
import os
import subprocess
import sys

import pytest

from officeops import core
from officeops.cli import TASKS, build_parser, run_task
from officeops.selftest import CASES, SAMPLE, run_selftest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = os.path.join(REPO, "officeops")
TODAY = "2026-08-24"


def run(name: str, *argv: str):
    parser = build_parser()
    args = parser.parse_args([name, *argv, "--today", TODAY])
    return run_task(name, args)


def sample(filename: str) -> str:
    return os.path.join(SAMPLE, filename)


# ==========================================================================
# THE CONSTRAINT: this layer is deterministic, offline and local
# ==========================================================================


def _python_sources():
    for root, _dirs, names in os.walk(PACKAGE):
        for name in sorted(names):
            if name.endswith(".py"):
                yield os.path.join(root, name)


def test_no_model_anywhere_in_the_office_layer():
    """The whole premise: these tasks need no AI. If one ever imports a model,
    it belongs in `modules/` where the review process for that lives."""
    banned = {"nsp_core.llm", "openai", "anthropic", "transformers", "torch", "ollama"}
    for path in _python_sources():
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                assert not any(name.startswith(b) for b in banned), f"{path} imports {name}"


def test_no_network_capability_anywhere_in_the_office_layer():
    """No BAA, no vendor review, no egress. That is the entire privacy
    architecture of this layer, and it is worth asserting rather than
    documenting."""
    banned = {
        "socket", "http", "http.client", "urllib", "urllib.request", "requests",
        "httpx", "ftplib", "smtplib", "telnetlib", "asyncio",
    }
    for path in _python_sources():
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                assert name.split(".")[0] not in {b.split(".")[0] for b in banned}, (
                    f"{path} imports {name}; this layer makes no network calls"
                )


def test_every_task_has_a_selftest_case():
    assert sorted(name for name, _ in CASES) == sorted(TASKS)


def test_the_selftest_passes():
    assert run_selftest() == 0


# ==========================================================================
# core.py — parsing and refusals
# ==========================================================================


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2026-08-24", dt.date(2026, 8, 24)),
        ("08/24/2026", dt.date(2026, 8, 24)),
        ("8/24/26", dt.date(2026, 8, 24)),
        ("24-Aug-2026", dt.date(2026, 8, 24)),
        ("August 24, 2026", dt.date(2026, 8, 24)),
        ("2026-08-24T09:20:00", dt.date(2026, 8, 24)),
    ],
)
def test_the_date_formats_real_exports_use(text, expected):
    assert core.parse_date(text) == expected


def test_an_unparseable_date_names_the_row_rather_than_being_skipped():
    """An unparseable appointment date in a confirmation-call export is not a
    row to skip. It is a patient who will not be called."""
    with pytest.raises(core.BadValue, match="row 7"):
        core.parse_date("next tuesday", field_name="appt", row=7)


def test_a_missing_column_says_what_it_looked_for(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("Some Header\nvalue\n", encoding="utf-8")
    table = core.load_table(path, aliases={"patient_id": ["mrn", "chart no"]})
    with pytest.raises(core.MissingColumn) as excinfo:
        table.require("patient_id")
    message = str(excinfo.value)
    assert "mrn" in message and "Some Header" in message


def test_an_excel_byte_order_mark_does_not_hide_the_first_column(tmp_path):
    """Excel writes a BOM, and a BOM on the first header turns `Patient ID`
    into something that matches no alias -- producing "no column for
    patient_id" on a file that plainly has one."""
    path = tmp_path / "bom.csv"
    path.write_bytes("﻿Patient ID,Name\n123,Ada\n".encode("utf-8"))
    table = core.load_table(path, aliases={"patient_id": ["patient id"]})
    table.require("patient_id")
    assert table.rows[0].text("patient_id") == "123"


def test_numbers_survive_spreadsheet_decoration():
    assert core.parse_float("$1,240.50") == 1240.5
    assert core.parse_float("4.1 C") == 4.1
    assert core.parse_float("", allow_blank=True) is None
    with pytest.raises(core.BadValue):
        core.parse_float("n/a")


def test_confirmation_flags_are_read_the_way_vendors_write_them():
    for text in ("Y", "yes", "Confirmed", "TRUE", "1"):
        assert core.parse_bool(text) is True
    for text in ("N", "no", "", "Not confirmed", "pending"):
        assert core.parse_bool(text) is False
    with pytest.raises(core.BadValue):
        core.parse_bool("maybe")


# ==========================================================================
# the tasks
# ==========================================================================


def test_confirm_list_finds_the_unconfirmed_and_orders_them_by_time():
    report = run("confirm-list", sample("schedule.csv"))
    assert report.counts["unconfirmed"] == 4
    times = [f["time"] for f in report.findings]
    assert times == sorted(times), "the caller works the front of the day first"
    assert all(f["status"] == "NOT CONFIRMED" for f in report.findings)
    # A blank confirmation cell is not confirmed.
    assert any(f["patient"] == "Nowak, Filip" for f in report.findings)


def test_confirm_list_ignores_other_days():
    report = run("confirm-list", sample("schedule.csv"))
    assert not any(f["patient"] == "Kim, Aera" for f in report.findings)


def test_day_sheets_group_by_provider():
    report = run("day-sheets", sample("schedule.csv"))
    assert report.counts["providers"] == 2
    assert "Dr. Alvarez" in report.headline and "Dr. Osei" in report.headline


def test_recall_list_puts_never_recorded_first():
    report = run("recall-list", sample("roster.csv"))
    assert report.findings[0]["last_well_visit"] == "NONE ON FILE"
    assert report.counts["never_recorded"] == 1
    # A patient seen two months ago is not on a recall list.
    assert not any(f["patient"] == "Torres, Mila" for f in report.findings)
    # Nor is a 23-year-old.
    assert not any(f["patient"] == "Bell, Marcus" for f in report.findings)


def test_interpreter_list_groups_by_language():
    report = run("interpreter-list", sample("schedule.csv"))
    languages = {f["language"] for f in report.findings}
    assert languages == {"Polish", "Spanish"}
    assert report.counts["Spanish"] == 2
    assert "English" not in report.counts


def test_fridge_log_reports_the_excursion_as_one_event_not_two_readings():
    report = run("fridge-log", sample("fridge_log.csv"))
    excursions = [f for f in report.findings if f["kind"] == "EXCURSION"]
    assert len(excursions) == 1
    assert report.counts["readings_out_of_range"] == 2
    assert float(excursions[0]["worst_c"]) > 8.0


def test_fridge_log_reports_a_day_with_too_few_readings():
    """CDC asks for twice-daily readings. A gap in the log is not a clean day;
    it is an unmonitored day, and the logger's own software will not say so."""
    report = run("fridge-log", sample("fridge_log.csv"))
    gaps = [f for f in report.findings if f["kind"] == "LOG GAP"]
    assert len(gaps) == 1
    assert "1 reading(s), expected 2" in gaps[0]["detail"]


def test_fridge_log_freezer_bounds_are_different():
    report = run("fridge-log", sample("fridge_log.csv"), "--freezer")
    assert report.counts["range_c"] == "-50.0 to -15.0"
    # Every fridge reading is an excursion against freezer bounds.
    assert report.counts["readings_out_of_range"] == report.counts["readings"]


def test_vaccine_inventory_puts_expired_lots_first():
    report = run("vaccine-inventory", sample("vaccine_inventory.csv"))
    assert report.findings[0]["status"] == "EXPIRED"
    assert report.counts["expired_lots"] == 2
    assert report.counts["doses_VFC"] == 18
    assert report.counts["doses_at_risk"] > 0


def test_lab_and_referral_followups_do_not_bleed_into_each_other():
    labs = run("lab-followup", sample("orders.csv"))
    referrals = run("referral-followup", sample("orders.csv"))
    assert {f["detail"] for f in labs.findings} == {"Lead level", "Lipid panel"}
    assert {f["detail"] for f in referrals.findings} == {"Allergy"}
    # A resulted lab and a returned referral are not outstanding.
    assert not any(f["detail"] == "Hemoglobin" for f in labs.findings)
    assert not any(f["detail"] == "Dermatology" for f in referrals.findings)


def test_followups_are_sorted_oldest_first():
    report = run("lab-followup", sample("orders.csv"))
    ages = [f["days_open"] for f in report.findings]
    assert ages == sorted(ages, reverse=True)


def test_qc_log_reports_the_missing_days_not_the_recorded_ones():
    """A CLIA-waived test run on a day with no control is a result that cannot
    be defended in an inspection."""
    report = run("qc-log", sample("qc_log.csv"), "--days", "14", "--tests", "strep,flu,urine")
    missing = [f for f in report.findings if f["kind"] == "NO QC" and f["test"] == "flu"]
    assert len(missing) == 3
    failures = [f for f in report.findings if f["kind"] == "QC FAIL"]
    assert len(failures) == 1 and failures[0]["test"] == "urine"


def test_expiry_sweep_reads_several_files_at_once():
    """Three spreadsheets maintained by three people; the one that gets
    forgotten is never the one somebody is currently looking at."""
    report = run("expiry-sweep", sample("crash_cart.csv"), sample("sample_closet.csv"))
    assert report.counts["files"] == 2
    sources = {f["source"] for f in report.findings}
    assert sources == {"crash_cart.csv", "sample_closet.csv"}
    assert report.findings[0]["status"] == "EXPIRED"


def test_credential_tracker_ranks_expired_before_expiring():
    report = run("credential-tracker", sample("credentials.csv"))
    statuses = [f["status"] for f in report.findings]
    assert statuses == sorted(statuses, key=lambda s: 0 if s == "EXPIRED" else 1)
    assert report.counts["expired"] == 2


def test_standing_orders_flags_stale_signatures_and_missing_supervisors():
    """225 ILCS 60/54.2: a standing order with no current signature is an MA
    performing a task with no documented authority to perform it."""
    report = run("standing-orders", sample("standing_orders.csv"))
    issues = {f["person"]: f["issue"] for f in report.findings}
    assert "last signed" in issues["Jess Alvarado"]
    assert "no supervising physician named" in issues["Marta Silva"]
    assert "no signature date recorded" in issues["Dana Whitfield"]


def test_retention_sweep_uses_the_minor_rule_and_deletes_nothing():
    report = run("retention-sweep", sample("records_index.csv"))
    assert "Nothing is deleted by this tool" in report.headline
    for finding in report.findings:
        assert finding["basis"].startswith(("minor", "adult"))
    # A record whose patient was a minor at last activity is held longer.
    assert not any(f["patient"] == "Lindqvist, Eva" for f in report.findings)


def test_mail_merge_refuses_to_post_a_blank_placeholder():
    """The classic failure is a letter reading "your last visit was on ."
    posted to two hundred families."""
    report = run("mail-merge", sample("recall_letter.txt"), sample("recipients.csv"))
    skipped = [f for f in report.findings if f["status"] == "SKIPPED"]
    assert len(skipped) == 1
    assert "last_well_visit" in skipped[0]["detail"]
    assert report.counts["letters_rendered"] == 2


def test_mail_merge_renders_every_placeholder_when_told_to():
    report = run(
        "mail-merge", sample("recall_letter.txt"), sample("recipients.csv"), "--allow-blank"
    )
    assert report.counts["letters_rendered"] == 3
    assert len(report.letters) == 3
    assert not any("{{" in letter for letter in report.letters)
    # Never in the report body or the JSON: that put a second copy of every
    # name, address and visit date on disk in an artifact nobody was tracking.
    assert "_letters" not in report.counts
    assert "Kingsbury" not in report.render()
    assert "Kingsbury" not in json.dumps(report.as_dict())


def test_charge_reconcile_allows_a_charge_to_lag_the_visit():
    report = run("charge-reconcile", sample("visits.csv"), sample("charges.csv"))
    missing = {f["patient"] for f in report.findings}
    # Filip's charge posted a day late; that is posted.
    assert "Nowak, Filip" not in missing
    assert missing == {"Adeyemi, Chidi", "Petrov, Juno"}


def test_denial_worklist_ranks_by_dollars_and_separately_by_frequency():
    """A high-count low-dollar code is usually one fixable front-desk habit,
    and that is the one worth a staff meeting."""
    report = run("denial-worklist", sample("denials.csv"), "--days", "60")
    by_dollars = [f for f in report.findings if f["rank_by"] == "dollars"]
    amounts = [float(f["dollars"]) for f in by_dollars]
    assert amounts == sorted(amounts, reverse=True)
    assert any("CO-16" in f["reason"] for f in report.findings)


# ==========================================================================
# the CLI
# ==========================================================================


def test_the_cli_writes_nothing_without_write(tmp_path):
    from officeops.cli import main

    out = tmp_path / "out"
    code = main(
        ["confirm-list", sample("schedule.csv"), "--today", TODAY, "--out", str(out)]
    )
    assert code == 1  # findings exist
    assert not out.exists(), "a read-only run must leave no files"


def test_the_cli_writes_a_csv_and_a_report_with_write(tmp_path):
    from officeops.cli import main

    out = tmp_path / "out"
    main(["confirm-list", sample("schedule.csv"), "--today", TODAY,
          "--out", str(out), "--write"])
    written = sorted(p.name for p in out.iterdir())
    assert any(n.endswith("_confirm_list.csv") for n in written)
    assert any(n.endswith("_confirm_list.txt") for n in written)
    rows = list(csv.DictReader(open(out / written[0], encoding="utf-8")))
    assert len(rows) == 4


def test_the_exit_code_distinguishes_clean_from_unreadable(tmp_path):
    """A scheduled job has to tell "nothing to report" from "could not read the
    input", and both from "somebody needs to act"."""
    from officeops.cli import main

    empty = tmp_path / "empty.csv"
    empty.write_text("Nope\nx\n", encoding="utf-8")
    assert main(["confirm-list", str(empty), "--today", TODAY]) == 2  # refused
    assert main(["confirm-list", sample("schedule.csv"), "--today", TODAY]) == 1  # findings
    assert main(
        ["confirm-list", sample("schedule.csv"), "--today", "2026-12-25"]
    ) == 0  # clean


def test_the_list_command_documents_every_task(capsys):
    from officeops.cli import main

    main(["list"])
    printed = capsys.readouterr().out
    for name in TASKS:
        assert name in printed


def test_json_output_is_machine_readable(capsys):
    from officeops.cli import main

    main(["confirm-list", sample("schedule.csv"), "--today", TODAY, "--json"])
    payload = json.loads(capsys.readouterr().out.split("\n(nothing written")[0])
    assert payload["task"] == "confirmation call list"
    assert len(payload["findings"]) == 4


def test_a_custom_mapping_overrides_the_bundled_one(tmp_path):
    """The fix for "no column for X" is an alias, never a renamed export --
    the export is regenerated every day by somebody who will not remember."""
    export = tmp_path / "weird.csv"
    export.write_text(
        "Kid,When It Is,Did They Say Yes\n"
        "Torres Mila,2026-08-25 08:20,no\n",
        encoding="utf-8",
    )
    mapping = tmp_path / "mine.yaml"
    mapping.write_text(
        "columns:\n"
        "  patient_name: [Kid]\n"
        "  appointment_datetime: [When It Is]\n"
        "  confirmed: [Did They Say Yes]\n",
        encoding="utf-8",
    )
    parser = build_parser()
    args = parser.parse_args(
        ["confirm-list", str(export), "--today", TODAY, "--mapping", str(mapping)]
    )
    report = run_task("confirm-list", args)
    assert report.counts["unconfirmed"] == 1


# ==========================================================================
# the PowerShell scripts
# ==========================================================================

PS_DIR = os.path.join(PACKAGE, "powershell")


@pytest.mark.parametrize(
    "script",
    ["Invoke-PhiShareAudit.ps1", "Test-BackupHealth.ps1", "Register-OfficeOpsTasks.ps1"],
)
def test_every_powershell_script_has_comment_based_help_and_examples(script):
    """`Get-Help .\\script.ps1 -Full` is how a Windows admin reads a script.
    A script without comment-based help is a script they will not run."""
    text = open(os.path.join(PS_DIR, script), encoding="utf-8").read()
    assert text.count("<#") == 1 and "#>" in text
    for section in (".SYNOPSIS", ".DESCRIPTION", ".EXAMPLE"):
        assert section in text, f"{script} has no {section}"
    assert "Set-StrictMode -Version Latest" in text
    assert text.count("{") == text.count("}")


def test_the_acl_audit_cannot_change_anything_it_audits():
    """There is no -Fix parameter and there will not be one: an ACL script that
    "corrects" permissions it does not fully understand is how a practice loses
    access to its own chart archive on a Monday morning.

    Writing the report to `-OutDir` is not a mutation of the audited share, so
    the check is on the permission and deletion cmdlets, plus a requirement that
    every write target is built from `$OutDir`.
    """
    import re as _re

    text = open(os.path.join(PS_DIR, "Invoke-PhiShareAudit.ps1"), encoding="utf-8").read()
    for mutation in ("Set-Acl", "Remove-Item", "New-Item -ItemType File",
                     "icacls", "Remove-NTFSAccess", "Set-ItemProperty"):
        assert mutation not in text, f"the audit must not call {mutation}"
    for match in _re.finditer(r"^\s*\$(\w+)\s*=\s*Join-Path\s+(\$\w+)", text, _re.M):
        assert match.group(2) == "$OutDir", "writes must target the output directory"


def test_the_backup_check_does_not_claim_to_have_tested_a_restore():
    text = open(os.path.join(PS_DIR, "Test-BackupHealth.ps1"), encoding="utf-8").read()
    assert "does NOT prove a restore works" in text


def test_the_scheduler_registration_supports_whatif():
    text = open(os.path.join(PS_DIR, "Register-OfficeOpsTasks.ps1"), encoding="utf-8").read()
    assert "SupportsShouldProcess" in text and "ShouldProcess" in text


# ==========================================================================
# Adversarial-review regressions. One per finding.
# ==========================================================================


def _csv(tmp_path, name, headers, rows):
    path = tmp_path / name
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    return str(path)


def test_fridge_state_is_partitioned_by_storage_unit(tmp_path):
    """FINDING: one excursion state and one day counter shared across units.

    An in-range reading on Fridge 2 closed an open excursion on Fridge 1, so a
    continuous 48-hour breach was reported as two five-minute events, on the
    wrong fridge, and the pooled day counter hid every gap.
    """
    rows = []
    for day in range(17, 20):
        rows.append([f"2026-08-{day} 08:00", "Fridge 1", "13.4"])
        rows.append([f"2026-08-{day} 08:05", "Fridge 2", "4.1"])
        rows.append([f"2026-08-{day} 16:00", "Fridge 1", "12.9"])
        rows.append([f"2026-08-{day} 16:05", "Fridge 2", "4.2"])
    path = _csv(tmp_path, "two.csv", ["Date/Time", "Storage Unit", "Temp (C)"], rows)
    report = run("fridge-log", path)
    excursions = [f for f in report.findings if f["kind"] == "EXCURSION"]
    assert len(excursions) == 1
    assert excursions[0]["unit"] == "Fridge 1"
    assert excursions[0]["minutes"] > 2000, "the whole run, not a fragment"
    assert not any(f["unit"] == "unit-1" for f in report.findings)
    assert report.counts["units"] == 2


def test_a_logger_that_stopped_reporting_is_not_a_clean_run(tmp_path):
    """FINDING: the gap window ended at the last row in the file, so 53 days of
    unmonitored vaccine storage read as "nothing to report", exit 0."""
    rows = [[f"2026-07-0{d} 08:00", "Fridge 1", "4.1"] for d in (1, 2)]
    rows += [[f"2026-07-0{d} 16:00", "Fridge 1", "4.2"] for d in (1, 2)]
    path = _csv(tmp_path, "dead.csv", ["Date/Time", "Storage Unit", "Temp (C)"], rows)
    report = run("fridge-log", path)
    silent = [f for f in report.findings if f["kind"] == "NO READINGS"]
    assert len(silent) > 50
    assert report.counts["days_with_no_readings_at_all"] > 50


def test_fridge_readings_are_sorted_before_analysis(tmp_path):
    """FINDING: a newest-first export produced an excursion of minus 24 hours."""
    rows = [
        ["2026-08-21 08:00", "Fridge 1", "4.0"],
        ["2026-08-20 16:00", "Fridge 1", "10.4"],
        ["2026-08-20 08:00", "Fridge 1", "9.6"],
        ["2026-08-19 16:00", "Fridge 1", "4.1"],
        ["2026-08-19 08:00", "Fridge 1", "4.2"],
    ]
    path = _csv(tmp_path, "desc.csv", ["Date/Time", "Storage Unit", "Temp (C)"], rows)
    report = run("fridge-log", path)
    excursions = [f for f in report.findings if f["kind"] == "EXCURSION"]
    assert len(excursions) == 1
    assert excursions[0]["minutes"] > 0
    assert excursions[0]["start"] < excursions[0]["end"]


def test_a_mistyped_unit_is_refused_not_reported_as_clean():
    """A typo here was indistinguishable from a clean fridge: exit 0, silence."""
    with pytest.raises(core.OfficeOpsError, match="matched no readings"):
        run("fridge-log", sample("fridge_log.csv"), "--unit", "fridge one")
    # Case and punctuation still match the real unit.
    assert run("fridge-log", sample("fridge_log.csv"), "--unit", "fridge 1").findings


def test_freezer_bounds_cannot_be_silently_overridden():
    """FINDING: `--freezer` won over an explicit --min-c/--max-c, hiding a
    -16.0 C breach that the same file reported without the flag."""
    with pytest.raises(core.OfficeOpsError, match="cannot be combined"):
        run("fridge-log", sample("fridge_log.csv"), "--freezer", "--min-c", "-25")


def test_the_retention_horizon_is_the_later_of_the_two_rules(tmp_path):
    """FINDING: choosing the minor rule for a minor gave teenagers a SHORTER
    retention than adults. Two patients last seen the same day, one 17 and one
    18: the seventeen-year-old's chart was listed for destruction five years
    before the adult's -- the one with the LONGER statute of limitations
    getting the shorter retention."""
    path = _csv(
        tmp_path, "ret.csv", ["MRN", "Patient", "Date of Birth", "Last Activity"],
        [
            ["1", "Seventeen, Sam", "2004-09-15", "2021-09-15"],
            ["2", "Eighteen, Alex", "2003-09-15", "2021-09-15"],
        ],
    )
    assert run("retention-sweep", path).findings == []
    later = run("retention-sweep", path, "--today", "2032-01-01")
    # Overriding --today via run() would duplicate the flag; check directly.
    parser = build_parser()
    args = parser.parse_args(["retention-sweep", path, "--today", "2032-01-01"])
    both = {f["patient"] for f in run_task("retention-sweep", args).findings}
    assert both == {"Seventeen, Sam", "Eighteen, Alex"}


def test_a_leap_day_birthday_does_not_shift_the_horizon(tmp_path):
    """FINDING: `min(dob.day, 28)` silently pulled the horizon up to three days
    early for anyone born on the 29th to the 31st."""
    from officeops.tasks.compliance import _birthday_on

    assert _birthday_on(dt.date(2004, 2, 29), 22) == dt.date(2026, 3, 1)
    assert _birthday_on(dt.date(2004, 1, 31), 22) == dt.date(2026, 1, 31)
    assert _birthday_on(dt.date(2004, 9, 15), 18) == dt.date(2022, 9, 15)


def test_qc_test_names_match_the_way_a_log_actually_spells_them(tmp_path):
    """FINDING: exact string equality meant `--tests strep,flu,urine` against a
    log recording "Strep A", "Influenza A/B" and "Urinalysis" matched nothing at
    all and reported 0% completeness on a perfect log."""
    rows = []
    for offset in range(1, 15):
        day = (dt.date(2026, 8, 24) - dt.timedelta(days=offset)).isoformat()
        for test in ("Strep A", "Influenza A/B", "Urinalysis"):
            rows.append([day, test, "control 1", "Passed"])
    path = _csv(tmp_path, "qc.csv", ["Date", "Test", "Control", "Result"], rows)
    report = run("qc-log", path, "--days", "14", "--tests", "strep,flu,urine")
    assert report.counts["missing_qc_days"] == 0
    assert report.counts["completeness"] == "100%"
    assert report.findings == []


def test_a_log_matching_nothing_is_refused_rather_than_reported_as_zero(tmp_path):
    path = _csv(
        tmp_path, "qc.csv", ["Date", "Test", "Control", "Result"],
        [["2026-08-20", "Cholesterol", "c1", "pass"]],
    )
    with pytest.raises(core.OfficeOpsError, match="match --tests"):
        run("qc-log", path, "--days", "14", "--tests", "strep,flu")


@pytest.mark.parametrize(
    "result,kind",
    [
        ("", "NO RESULT"),
        ("Passed", None),
        ("in-range", None),
        ("within range", None),
        ("fail", "QC FAIL"),
        ("wibble", "UNKNOWN RESULT"),
    ],
)
def test_qc_result_vocabulary(tmp_path, result, kind):
    """FINDING: four spellings of "pass" reported as QC FAILURES beside the one
    real failure, and a blank result counted the day as complete -- which is
    precisely the row a CLIA inspector writes up."""
    path = _csv(
        tmp_path, "qc.csv", ["Date", "Test", "Control", "Result"],
        [["2026-08-23", "strep", "c1", result]],
    )
    report = run("qc-log", path, "--days", "1", "--tests", "strep")
    kinds = {f["kind"] for f in report.findings}
    if kind is None:
        assert kinds == set()
    else:
        assert kinds == {kind}


def test_the_qc_window_does_not_manufacture_a_finding_every_morning(tmp_path):
    """FINDING: `range(args.days + 1)` over a window that included TODAY gave
    one guaranteed false NO-QC per test on every run. The alert that is always
    wrong is the one staff learn to delete."""
    rows = []
    for offset in range(1, 41):
        day = (dt.date(2026, 8, 24) - dt.timedelta(days=offset)).isoformat()
        for test in ("strep", "flu", "urine"):
            rows.append([day, test, "c1", "pass"])
    path = _csv(tmp_path, "qc.csv", ["Date", "Test", "Control", "Result"], rows)
    report = run("qc-log", path, "--days", "30", "--tests", "strep,flu,urine")
    assert report.counts["test_days_expected"] == 90
    assert report.findings == []


def test_a_charge_with_no_amount_column_still_counts_as_billed(tmp_path):
    """FINDING: a CPT-only charge export -- entirely normal -- reported every
    fully-billed visit as having no charge posted."""
    visits = _csv(
        tmp_path, "v.csv", ["MRN", "Patient", "Date of Service"],
        [["1", "Torres, Mila", "2026-08-10"]],
    )
    charges = _csv(
        tmp_path, "c.csv", ["MRN", "Patient", "Date of Service", "CPT"],
        [["1", "Torres, Mila", "2026-08-10", "99392"]],
    )
    report = run("charge-reconcile", visits, charges)
    assert report.counts["visits_with_no_charge"] == 0
    assert any("no amount column" in p for p in report.problems)


def test_a_reversed_charge_is_its_own_finding_not_never_billed(tmp_path):
    visits = _csv(
        tmp_path, "v.csv", ["MRN", "Patient", "Date of Service"],
        [["1", "Torres, Mila", "2026-08-10"]],
    )
    charges = _csv(
        tmp_path, "c.csv", ["MRN", "Patient", "Date of Service", "CPT", "Amount"],
        [["1", "Torres, Mila", "2026-08-10", "99392", "185.00"],
         ["1", "Torres, Mila", "2026-08-10", "99392", "-185.00"]],
    )
    report = run("charge-reconcile", visits, charges)
    assert report.counts["visits_with_no_charge"] == 0
    assert report.counts["visits_netting_to_zero"] == 1
    assert report.findings[0]["status"] == "net zero"


def test_another_encounters_charge_does_not_satisfy_this_one(tmp_path):
    """FINDING: the grace window was applied to the charge's SERVICE date, which
    let a billed nurse visit on the 12th satisfy an unbilled well visit on the
    10th. A $220 leak reported as clean."""
    visits = _csv(
        tmp_path, "v.csv", ["MRN", "Patient", "Date of Service", "Visit Type"],
        [["1", "Torres, Mila", "2026-08-10", "Well child"],
         ["1", "Torres, Mila", "2026-08-12", "Nurse"]],
    )
    charges = _csv(
        tmp_path, "c.csv", ["MRN", "Patient", "Date of Service", "CPT", "Amount"],
        [["1", "Torres, Mila", "2026-08-12", "99211", "45.00"]],
    )
    report = run("charge-reconcile", visits, charges)
    assert report.counts["visits_with_no_charge"] == 1
    assert report.findings[0]["visit_date"] == "2026-08-10"


def test_orders_with_nobody_current_is_actually_computed():
    """FINDING: `covered[order].add(person)` ran only on the no-issues path, so
    the count was structurally zero for every input that can exist -- and it is
    the one number an Illinois delegation audit turns on."""
    report = run("standing-orders", sample("standing_orders.csv"))
    assert report.counts["orders_with_nobody_current"] == 2
    assert report.counts["distinct_orders"] == 3
    nobody = [f for f in report.findings if f["person"] == "(nobody)"]
    assert len(nobody) == 2
    assert all("NO ONE" in f["issue"] for f in nobody)


def test_denials_group_on_the_code_not_the_description(tmp_path):
    """FINDING: four payer spellings of one CARC became four rows, so the
    largest single reason ranked below smaller ones and the top of the work
    queue was wrong."""
    rows = []
    for index, wording in enumerate(
        ["Missing information", "Missing info", "Information missing", "MISSING INFORMATION"]
    ):
        for _ in range(5):
            rows.append(["2026-08-20", f"Payer {index}", "CO-16", wording, "900.00"])
    for _ in range(4):
        rows.append(["2026-08-20", "Aetna", "CO-45", "Fee schedule", "4000.00"])
    path = _csv(
        tmp_path, "d.csv",
        ["Remit Date", "Payer", "Reason Code", "Reason Description", "Amount"], rows,
    )
    report = run("denial-worklist", path, "--days", "30", "--top", "5")
    dollars = [f for f in report.findings if f["rank_by"] == "dollars"]
    assert dollars[0]["code"] == "CO-16"
    assert dollars[0]["count"] == 20
    assert report.counts["distinct_reason_codes"] == 2


def test_the_frequency_ranking_is_emitted_independently(tmp_path):
    """FINDING: at the documented `--top 10` the frequency block came out empty,
    because it deduped against the dollar list and only considered the top 3."""
    report = run("denial-worklist", sample("denials.csv"), "--days", "90", "--top", "10")
    frequency = [f for f in report.findings if f["rank_by"] == "frequency"]
    assert frequency
    counts = [f["count"] for f in frequency]
    assert counts == sorted(counts, reverse=True)


def test_a_mapping_file_merges_over_the_bundled_defaults(tmp_path):
    """FINDING: `--mapping` REPLACED the bundled aliases, so the README's own
    four-key example produced a call list with no phone numbers and nothing
    saying why."""
    mapping = tmp_path / "mine.yaml"
    mapping.write_text(
        "columns:\n"
        "  patient_name: [Patient Name]\n"
        '  appointment_datetime: ["Appt Date/Time"]\n'
        "  confirmed: [Confirmation Status]\n"
        "  provider: [Provider]\n",
        encoding="utf-8",
    )
    parser = build_parser()
    args = parser.parse_args(
        ["confirm-list", sample("schedule.csv"), "--today", TODAY,
         "--mapping", str(mapping)]
    )
    report = run_task("confirm-list", args)
    assert report.counts["unconfirmed"] == 4
    # patient_id and phone came from the bundled mapping, not from this file.
    assert all(f["mrn"] and f["phone"] for f in report.findings)


def test_the_shared_common_mapping_is_actually_loaded():
    """FINDING: `_common.yaml`'s own header says "THIS IS THE FILE YOU EDIT",
    and nothing loaded it."""
    merged = core.load_mapping("schedule")
    assert "patient_id" in merged and "phone" in merged


def test_an_export_with_headers_and_no_rows_says_so(tmp_path):
    """FINDING: it was reported as a missing column, and then listed the columns
    as present -- sending an office manager into the mapping file for an
    afternoon over a holiday Monday."""
    path = _csv(
        tmp_path, "empty.csv",
        ["Chart No", "Patient Name", "Appt Date/Time", "Confirmation Status"], [],
    )
    with pytest.raises(core.EmptyExport, match="no data"):
        run("confirm-list", path)


def test_a_truncated_csv_is_refused_rather_than_half_reported(tmp_path):
    """FINDING: one unbalanced quote made the reader swallow fifteen of twenty
    rows; the report said "5 of 5 appointments" and "1 row could not be read"."""
    lines = ['Chart No,Patient Name,Appt Date/Time,Confirmation Status']
    for index in range(20):
        note = '"broken' if index == 4 else "ok"
        lines.append(f"{index},Name {index},2026-08-25 09:00,{note}")
    path = tmp_path / "broken.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(core.TruncatedFile, match="unterminated quote"):
        run("confirm-list", str(path))


def test_a_missing_confirmation_column_is_refused_not_reported_as_zero_percent(tmp_path):
    """FINDING: without the column every patient read NOT CONFIRMED and the rate
    read 0% -- a number the phasing plan hands to I-07 as a MEASURED baseline."""
    path = _csv(
        tmp_path, "s.csv", ["Patient Name", "Appt Date/Time"],
        [["Torres, Mila", "2026-08-25 08:20"]],
    )
    with pytest.raises(core.MissingColumn, match="confirmed"):
        run("confirm-list", path)


def test_a_roster_with_no_last_well_visit_column_is_refused(tmp_path):
    path = _csv(tmp_path, "r.csv", ["Patient", "DOB"], [["A", "2020-01-01"]])
    with pytest.raises(core.MissingColumn, match="last_well_visit"):
        run("recall-list", path)


def test_a_lab_list_does_not_pick_up_a_collaborative_care_referral(tmp_path):
    """FINDING: `"lab" in order_type.lower()` pulled in "Collaborative care
    referral" and "Labor and delivery records request"."""
    path = _csv(
        tmp_path, "o.csv",
        ["Patient", "Order Date", "Order Type", "Description", "Result Date", "Report Date"],
        [
            ["A", "2026-06-01", "Collaborative care referral", "Behavioral health", "", ""],
            ["B", "2026-06-01", "Labor and delivery records request", "Records", "", ""],
            ["C", "2026-06-01", "lab", "Lead level", "", ""],
        ],
    )
    report = run("lab-followup", path, "--days", "14")
    assert [f["patient"] for f in report.findings] == ["C"]


def test_recall_list_ages_are_possible_and_sorted_on_the_exact_value(tmp_path):
    """FINDING: `age_days // 365` with `% 365 // 30` produced "4y 12m", and the
    sort ran on the ROUNDED month string."""
    path = _csv(
        tmp_path, "r.csv", ["Patient", "DOB", "Last Well Visit"],
        [
            ["Older, Gap", "2020-08-25", "2024-01-01"],
            ["Newer, Gap", "2020-08-25", "2024-01-11"],
            ["Twelve, Months", "2021-08-20", "2024-06-01"],
        ],
    )
    report = run("recall-list", path)
    for finding in report.findings:
        months = int(finding["age"].split("y ")[1].rstrip("m"))
        assert 0 <= months <= 11, finding["age"]
    assert [f["patient"] for f in report.findings][:2] == ["Older, Gap", "Newer, Gap"]


def test_a_template_naming_a_column_the_csv_lacks_is_one_refusal(tmp_path):
    """FINDING: one skip row PER RECIPIENT saying the same thing, instead of one
    refusal saying the template names a column the CSV does not have."""
    template = tmp_path / "t.txt"
    template.write_text("Dear {{patient_name}}, about {{nonexistent_field}}.", encoding="utf-8")
    recipients = _csv(tmp_path, "r.csv", ["Patient Name"], [["A"], ["B"], ["C"]])
    with pytest.raises(core.OfficeOpsError, match="nonexistent_field"):
        run("mail-merge", str(template), recipients)


# -- the CLI contract -------------------------------------------------------


@pytest.mark.parametrize(
    "argv,reason",
    [
        (["confirm-list", "/nonexistent/schedule.csv"], "the nightly export did not land"),
        (["confirm-list", SAMPLE], "the path is a directory"),
        (["expiry-sweep", "/tmp", "/nonexistent/x.csv"], "one of several files missing"),
        (["mail-merge", "/nonexistent/t.txt", "/nonexistent/r.csv"], "template missing"),
    ],
)
def test_an_input_problem_exits_2_rather_than_a_traceback(argv, reason):
    """FINDING: everything except OfficeOpsError escaped as a traceback and exit
    1 -- the code the scheduling contract maps to "findings, mail the report".
    The practice would have been mailed a stack trace and told it was a work
    list."""
    from officeops.cli import main

    assert main([*argv, "--today", TODAY]) == 2, reason


def test_a_malformed_mapping_file_exits_2(tmp_path):
    from officeops.cli import main

    bad = tmp_path / "bad.yaml"
    bad.write_text("columns: [this is a list, not a mapping\n", encoding="utf-8")
    assert main(["confirm-list", sample("schedule.csv"), "--today", TODAY,
                 "--mapping", str(bad)]) == 2

    wrong_shape = tmp_path / "shape.yaml"
    wrong_shape.write_text("columns: just-a-string\n", encoding="utf-8")
    assert main(["confirm-list", sample("schedule.csv"), "--today", TODAY,
                 "--mapping", str(wrong_shape)]) == 2


def test_a_non_iso_today_is_refused_with_a_useful_message(capsys):
    from officeops.cli import main

    assert main(["confirm-list", sample("schedule.csv"), "--today", "08/25/2026"]) == 2
    assert "ISO date" in capsys.readouterr().err


def test_unreadable_rows_reach_the_exit_code(tmp_path):
    """FINDING: an export where every row failed to parse printed twenty problem
    lines and returned 0 -- "stay silent". Twenty families were not called and
    nobody was told."""
    from officeops.cli import main

    path = _csv(
        tmp_path, "s.csv", ["Patient Name", "Appt Date/Time", "Confirmation Status"],
        [[f"Name {i}", "25.08.2026 09:00", "no"] for i in range(20)],
    )
    assert main([ "confirm-list", path, "--today", TODAY]) == 2


def test_the_written_csv_carries_the_run_that_produced_it(tmp_path):
    """FINDING: the CSV a practice keeps had no task name, no as-of date, no
    threshold and no source filename. Reopened in three months it could not be
    dated or reproduced."""
    from officeops.cli import main

    out = tmp_path / "out"
    main(["credential-tracker", sample("credentials.csv"), "--today", TODAY,
          "--out", str(out), "--write"])
    written = next(p for p in out.iterdir() if p.suffix == ".csv")
    row = next(csv.DictReader(open(written, encoding="utf-8")))
    assert row["run_task"] == "credential and training expiry"
    assert row["run_as_of"] == TODAY
    assert "warn_days=90" in row["run_parameters"]


def test_today_drives_the_output_filename_so_a_rerun_is_reproducible(tmp_path):
    """FINDING: the filename used the wall clock, so re-running last Tuesday's
    job overwrote today's file."""
    from officeops.cli import main

    out = tmp_path / "out"
    main(["confirm-list", sample("schedule.csv"), "--today", "2026-08-24",
          "--out", str(out), "--write"])
    assert any(p.name.startswith("20260824_") for p in out.iterdir())


# -- PowerShell regressions (read-only review; no pwsh in CI) ---------------


def _ps(name):
    return open(os.path.join(PS_DIR, name), encoding="utf-8").read()


def test_the_scheduler_registers_every_job_its_help_advertises():
    """FINDING: the .DESCRIPTION promised "backup health at 08:00 daily" and
    `$jobs` contained no such entry. A practice that ran it believed backup
    monitoring was scheduled. It was not."""
    text = _ps("Register-OfficeOpsTasks.ps1")
    assert "Test-BackupHealth.ps1" in text
    assert "Invoke-PhiShareAudit.ps1" in text
    assert "$psJobs" in text


def test_the_scheduler_requires_elevation_and_sets_a_principal():
    """FINDING: `Register-ScheduledTask` needs an administrative token, and with
    no principal the tasks run only while that user is signed in -- so the 07:30
    fridge job would not have run on a locked workstation."""
    text = _ps("Register-OfficeOpsTasks.ps1")
    assert "#Requires -RunAsAdministrator" in text
    assert "New-ScheduledTaskPrincipal" in text and "S4U" in text
    assert "ConfirmImpact = 'High'" in text
    assert "$ErrorActionPreference = 'Stop'" in text
    # Failures must reach the exit code rather than printing six error blobs.
    assert "if ($failed -gt 0) { exit 1 } else { exit 0 }" in text


def test_the_scheduler_does_not_promise_a_task_scheduler_feature_that_does_not_exist():
    """FINDING: both the script and the README told the practice to "attach a
    send-email action on non-zero exit". Task Scheduler has no
    conditional-on-exit-code action and its e-mail action was removed after
    Windows 7 -- so the alerting the whole scheduling argument rests on did not
    exist on the platform the practice runs."""
    text = _ps("Register-OfficeOpsTasks.ps1")
    assert "cannot act on an exit code" in text
    assert "-WrapperPath" in text and "errorlevel" in text.lower()


def test_the_acl_audit_ignores_deny_aces_and_inherited_duplicates():
    """FINDING: an explicit "Deny Everyone Full Control" -- correct hardening --
    was reported as the violation; and one Everyone ACE on a share root became
    one HIGH finding per subfolder, hundreds of rows for one real problem."""
    text = _ps("Invoke-PhiShareAudit.ps1")
    assert "AccessControlType" in text
    assert "$rule.IsInherited" in text and "$IncludeInherited" in text


def test_the_acl_audit_sees_non_enum_rights_and_exits_on_review_findings():
    """FINDING: a generic-rights ACE stringifies to a raw number, matched no
    name pattern and was silently downgraded -- exactly the ACEs hardest to read
    by eye. And a run with forty REVIEW findings exited 0, "stay silent"."""
    text = _ps("Invoke-PhiShareAudit.ps1")
    assert "[int]$rule.FileSystemRights" in text
    assert "$isPowerful" in text
    assert "if ($findings.Count -gt 0) { exit 1 } else { exit 0 }" in text
    # The .PARAMETER text promised a text report and only the CSV was written.
    assert "_phi_share_audit.txt" in text


def test_the_backup_check_looks_in_more_than_one_event_log():
    """FINDING: hard-coded to Windows Server Backup. Veeam, Datto, Acronis and
    every appliance a small practice uses log elsewhere -- so the report printed
    a reassuring "0" for a check it had not performed."""
    text = _ps("Test-BackupHealth.ps1")
    assert "$EventLogs" in text
    for product in ("Veeam", "datto", "acronis"):
        assert product.lower() in text.lower()
    assert "NONE of the known backup logs exist" in text
    # And the single-full-backup case is named rather than alarming forever.
    assert "overwrites a single full backup" in text


def test_the_backup_pattern_default_matches_extensionless_files():
    """FINDING: the help said `*.*` was "every file"; it excludes files with no
    extension."""
    text = _ps("Test-BackupHealth.ps1")
    assert '[string]$Pattern = "*",' in text
    assert "excludes extensionless files" in text
