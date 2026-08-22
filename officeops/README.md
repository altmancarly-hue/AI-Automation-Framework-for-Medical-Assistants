# officeops — the no-AI layer

**Sixteen recurring office tasks, done with Python and a CSV export. No model,
no network, no vendor, no BAA, no monthly cost.**

This package is the counterpart to `modules/`, not a replacement for it.
`modules/` implements the initiatives that genuinely need a language model.
This one implements the work that does not — the recurring paperwork that fills
a medical assistant's day and every other desk in a small practice.

The honest sequencing is to do this half first. It has nothing to procure and
nothing to validate, so it can start the week it is written, and it produces the
clean exports and measured baselines that the AI work needs anyway.

```bash
python3 -m officeops list          # every task, its input, its output
python3 -m officeops selftest      # run all sixteen against bundled sample data
```

## The three rules every task follows

**1. Column names are configuration, not code.** Every EHR exports the same
facts under different headers — `Appt Date`, `APPOINTMENT_DATE`, `appt_dt`.
Aliases live in `officeops/mappings/*.yaml` and `--mapping` overrides them.
Matching normalises case, spaces, underscores and punctuation, so `Appt
Date/Time` and `appt_date_time` are one alias rather than three. When a task
says *no column for `patient_id`*, it prints what it looked for and what the
file actually has. **Fix the mapping, never the export** — the export is
regenerated every day by somebody who will not remember to rename it.

**2. Read-only unless asked.** Nothing writes to the practice's files. `--write`
saves a dated CSV and a text report to `--out` (default `./out`). A script that
quietly rewrites the export it was given cannot be run twice.

**3. Refusals are loud.** A missing column, an unparseable date, a row with no
patient — reported with the row number, never dropped. A report that silently
skipped 12% of its input is worse than no report, because somebody will act on
the 88%.

## Exit codes

Built for Task Scheduler and cron: attach a mail action to non-zero and hear
only from the days that need somebody.

| Code | Meaning |
| --- | --- |
| `0` | Ran clean — nothing to report |
| `1` | Findings — somebody needs to act |
| `2` | Refused — the input could not be read |

## The tasks

### Front desk and scheduling

```bash
# Who has not confirmed tomorrow, in the order somebody would call them
python3 -m officeops confirm-list schedule.csv
python3 -m officeops confirm-list schedule.csv --days-ahead 2 --write

# Printable per-provider day sheets
python3 -m officeops day-sheets schedule.csv --write

# Children overdue for a well visit, worst first
python3 -m officeops recall-list roster.csv --overdue-months 15
python3 -m officeops recall-list roster.csv --max-age-years 18 --write

# Interpreter needs, grouped by language (one booking per language per session)
python3 -m officeops interpreter-list schedule.csv
```

`confirm-list` sorts by appointment time, not by name: the person working the
list is filling the front of the day first, and an unconfirmed 8:20 is worth ten
minutes more attention than an unconfirmed 4:40. `recall-list` defaults to
**fifteen** months rather than twelve — a family arriving at 12½ months is not
overdue, and a recall list that includes them trains the front desk to ignore it.

### Clinical / medical assistant

```bash
# Vaccine storage: excursions AND days with too few readings
python3 -m officeops fridge-log logger_export.csv
python3 -m officeops fridge-log freezer.csv --freezer --write
python3 -m officeops fridge-log logger.csv --min-c 2 --max-c 8 --expect-readings-per-day 2

# Expired and expiring lots, by funding source (VFC vs private stock)
python3 -m officeops vaccine-inventory inventory.csv --warn-days 60 --write

# Orders with nothing back
python3 -m officeops lab-followup orders.csv --days 14
python3 -m officeops referral-followup orders.csv --days 30 --write

# CLIA-waived QC: the days with NO control, which the log's own software omits
python3 -m officeops qc-log qc.csv --days 30 --tests strep,flu,urine --skip-weekends

# Everything with an expiry date, across every list the practice keeps
python3 -m officeops expiry-sweep crash_cart.csv sample_closet.csv supplies.csv --write
```

`fridge-log` reports **under-logged days as loudly as excursions**. CDC's
Vaccine Storage and Handling Toolkit asks for twice-daily readings; a gap in the
log is not a clean day, it is an unmonitored one, and it is what turns a power
cut into a whole-fridge loss nobody can bound. A logger's own software will
happily show you the days that *are* there.

`expiry-sweep` takes several files on purpose: the emergency kit, the sample
closet and the supply room are three spreadsheets maintained by three people,
and the one that gets forgotten is never the one somebody is currently looking
at.

### Compliance and administration

```bash
# Licences, CPR, CME, OSHA and HIPAA training coming due
python3 -m officeops credential-tracker credentials.csv --warn-days 90 --write

# Delegation roster: stale signatures and unnamed supervising physicians
python3 -m officeops standing-orders standing_orders.csv --review-months 12

# Records past the retention horizon — A REVIEW LIST, never a deletion
python3 -m officeops retention-sweep records_index.csv --minor-until-age 22

# Letters, refusing to post a blank placeholder
python3 -m officeops mail-merge recall_letter.txt recipients.csv --write
```

`standing-orders` exists because of 225 ILCS 60/54.2: an unlicensed medical
assistant in Illinois acts under physician delegation, so a standing order with
no current signature is not a paperwork gap — it is an MA performing a task with
no documented authority to perform it. The script reports what the roster says
and how old the signature is. It decides nothing.

`retention-sweep` deletes nothing and is not capable of deleting anything.
Retention varies by record type, by litigation hold, and by whether a patient
has an outstanding records request — none of which is in a records index.
Defaults reflect Illinois (735 ILCS 5/13-212: a minor's action may be brought
until age 22), and they are arguments rather than constants: this is a deadline
calculator, not a legal opinion.

`mail-merge` refuses a row with an unfilled placeholder unless `--allow-blank`
says so explicitly. The classic mail-merge failure is two hundred letters
reading "your last visit was on ." and it is entirely preventable.

### Revenue cycle

```bash
# Completed visits with no charge posted
python3 -m officeops charge-reconcile visits.csv charges.csv --grace-days 3 --write

# Denials, ranked by dollars and separately by frequency
python3 -m officeops denial-worklist denials.csv --days 90 --top 10
```

`denial-worklist` produces two rankings on purpose. **Dollars** is the work
queue. **Frequency** is the staff meeting: a high-count low-dollar code is
usually one fixable habit — an eligibility check nobody runs, a modifier nobody
appends — and it is invisible in a dollar ranking.

`charge-reconcile` matches on `(MRN, service date)` with a grace window and
falls back to name matching, which it labels as the weaker match, because two
siblings seen the same day are the case that breaks it.

### Windows-native (PowerShell)

These three are in PowerShell because the answer lives in Windows ACLs, the
event log and Task Scheduler rather than in a CSV.

```powershell
# Who can read the PHI share, and what should not be there
.\officeops\powershell\Invoke-PhiShareAudit.ps1 -Path "\\NSP-FS01\Charts" -Write
.\officeops\powershell\Invoke-PhiShareAudit.ps1 -Path "D:\Scans","D:\Faxes" -Depth 3

# Did last night's backup actually happen, and is it plausibly usable
.\officeops\powershell\Test-BackupHealth.ps1 -BackupPath "E:\Backups\EHR" -MinRestorePoints 14
.\officeops\powershell\Test-BackupHealth.ps1 -BackupPath "\\NAS\backups" -Pattern "*.vbk" -Write

# Put the daily jobs on a timer. -WhatIf first, always.
.\officeops\powershell\Register-OfficeOpsTasks.ps1 -RepoRoot C:\nsp -ExportDir D:\Exports -WhatIf
```

`Invoke-PhiShareAudit` **changes nothing** and has no `-Fix` parameter. An ACL
script that "corrects" permissions on a share it does not fully understand is
how a practice loses access to its own chart archive on a Monday morning. It
produces the artifact 45 CFR 164.308(a)(4)'s periodic access review needs; a
person still performs the review.

`Test-BackupHealth` checks four things a backup product's green tick does not
always check: a file newer than the last run should have produced; a size that
has not silently collapsed (the job ran, the source path was wrong, and it
backed up an empty folder every night for four months); the retention count; and
the event log. **It does not test a restore, and no script can claim a backup is
good without one.** Restore testing is a calendar item with a human on it.

## Adapting to your EHR

1. Run the task. If it refuses, read the message — it prints the aliases it
   looked for and the headers your file has.
2. Copy the relevant file out of `officeops/mappings/`, add your header to the
   right list, and pass `--mapping path/to/yours.yaml`.
3. `python3 -m officeops selftest` still passing while your export fails means
   the difference is column names, not the tool.

## Scheduling it

```bash
# Linux / macOS
0 15 * * 1-5  cd /opt/nsp && python3 -m officeops confirm-list /exports/schedule.csv --write --out /var/officeops
30 7 * * *    cd /opt/nsp && python3 -m officeops fridge-log /exports/fridge_log.csv --write --out /var/officeops
0 8 * * 1     cd /opt/nsp && python3 -m officeops credential-tracker /exports/credentials.csv --write --out /var/officeops
```

On Windows, `Register-OfficeOpsTasks.ps1` registers the equivalent set. Every
job exits `1` on findings and `0` when clean, so a "send email on non-zero"
action delivers only the days that need somebody.

## What is deliberately not here

- **No EHR API integration.** Every task reads an export the practice can
  already produce. That is what makes this deployable in an afternoon, and it is
  what makes it survive an EHR upgrade.
- **No writes back to the EHR.** These produce work lists. A person acts on
  them.
- **No clinical decisions.** `fridge-log` reports a temperature outside a range
  a human typed in. `vaccine-inventory` reports a date arithmetic. Nothing here
  interprets, and the one that comes closest — `retention-sweep` — says so in
  its own headline.
- **No network calls of any kind.** A test walks the AST of every module in this
  package and fails if one imports `socket`, `urllib`, `requests` or anything
  like them. That is the entire privacy architecture of this layer, and it is
  why it needs no BAA and no vendor review.
