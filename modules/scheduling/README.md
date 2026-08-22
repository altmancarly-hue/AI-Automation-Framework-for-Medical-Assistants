# I-07 — No-Show Reduction & Waitlist Backfill

Deterministic scheduling automation. **No language model is used anywhere in this
package**, and four tests fail if one is added.

README §I-07 is explicit about why: reminders, confirmations, cancellations and
waitlist backfill are a cron job, a rules table and a messaging API. Adding a
model adds latency, cost, a BAA, an audit obligation and a hallucination surface
to a problem that is arithmetic on datetimes.

## Files

| File | Responsibility |
| --- | --- |
| `models.py` | SQLite schema, time helpers, `Database` (per-thread connections, `BEGIN IMMEDIATE` helper), consent/suppression writers |
| `cadence.py` | Reminder rules table, `QuietHours`, `FrequencyCap`, `SendGate`, `ReminderEngine` |
| `backfill.py` | Slot release, candidate ranking, parallel blast, **atomic first-accept-wins** |
| `gateway.py` | `Gateway` ABC, `LocalGateway` (file-backed), `TwilioGateway` (lazy SDK), deterministic inbound classification |
| `inbound.py` | Applies a classified inbound tap/keyword to practice state — including the TCPA opt-out path |
| `metrics.py` | README §10.2 KPIs: no-show rate, backfill rate, median fill time, message funnel, lead-time buckets |
| `demo.py` | `make demo` — seeds a synthetic practice and prints what each family receives |

## Wiring

```python
from modules.scheduling import (
    Database, LocalGateway, ReminderEngine, BackfillEngine, InboundRouter,
)

db = Database("var/scheduling.sqlite3")
gw = LocalGateway("var/outbox.jsonl")          # swap for TwilioGateway in prod
reminders = ReminderEngine(db, gw)
backfill  = BackfillEngine(db, gw)
router    = InboundRouter(db, gw, backfill)
```

Cron, every five minutes:

```python
reminders.plan_horizon(now=now)          # queue reminders for new appointments
reminders.dispatch_due(now=now)          # send what is due, gated at send time
backfill.expire_stale_offers(now=now)
backfill.sweep_open_releases(now=now)    # picks up overnight cancellations
backfill.close_unfilled_releases(now=now)
```

Webhook:

```python
router.handle({"path": "/a/off_123"}, now=now)              # a tap
router.handle({"body": "STOP", "from": "+1847..."}, now=now) # a keyword
```

## The five design decisions worth knowing

**1. Reminder offsets are elapsed time; quiet hours are wall-clock time.**
`T-48h` is `start_utc - timedelta(hours=48)` on aware instants. Doing the same
subtraction on a naive local datetime produces a 47- or 49-hour reminder twice a
year. Quiet hours are the opposite: 21:00 means 21:00 to the family, whatever the
UTC offset that week. Both directions are tested at both DST transitions.

**2. The send gate runs at dispatch, never at planning.** Order is
`suppression → consent → quiet hours → frequency cap`. The first two are legal
gates with no bypass parameter. The last two are courtesy gates that transactional
messages (cancel confirmations, opt-out confirmations, backfill win/loss) skip.
A family that opts out on Tuesday does not receive Monday's queued message.

**3. Booking is atomic inside `BEGIN IMMEDIATE` — at two scopes.** The check
(`filled_utc IS NULL`) and the act (insert appointment, stamp release) are the
same transaction, and no network I/O happens inside it: losers are texted after
COMMIT. `release_slot` takes the same lock, because a double-tapped cancel link
or a retried carrier webhook would otherwise create two `slot_release` rows for
one appointment, each blasted and filled independently — the lock at the accept
scope is airtight but guards accepts *within* a release, not releases *of* an
appointment. A partial unique index (`ux_release_open_per_appointment`) is the
database-level backstop. `accept_offer` also re-checks, inside the transaction,
that the waitlist entry is still active and that the patient has no conflicting
appointment; eligibility ran at blast time, before any booking existed.

Covered by `test_five_simultaneous_accepts_produce_exactly_one_winner` (five
threads through a barrier), `test_concurrent_accepts_are_stable_across_many_rounds`
(ten fresh slots), `test_concurrent_cancellations_cannot_create_two_releases`, and
`test_one_patient_cannot_accept_two_simultaneous_slots`.

**4. Inbound is a lookup table, not NLP.** Taps carry an action code. Free text
goes to a human, and only its *length* is retained in the intent metadata —
patient-authored text belongs in the queue a human reads. `CANCEL` is treated as
a carrier opt-out keyword, not an appointment cancellation, because the carrier
will unsubscribe the number regardless of what this code decides.

**5. An overnight cancellation holds the slot rather than burning it.** A 21:40
cancellation does not blast at 21:40 and does not close the release either — the
morning sweep blasts it at 08:00. Slots too early to refill after 08:00 are
closed with `quiet_hours_no_time_to_fill`, and the sweep counts them in
`too_close`, so the quiet-hours/lead-time trade-off is visible rather than silent.

## Shared with I-02

`FrequencyCap` is importable and **must** be the same object the immunization
recall engine consults. The README's cap is across all initiatives, not per
initiative. Tiers make it degrade in the right order:

| Tier | Default limit / family / week | Used by |
| --- | --- | --- |
| `appointment` | 5 | I-07 reminders |
| `waitlist` | 6 | I-07 backfill offers (the family opted in to be told) |
| `outreach` | 2 | I-02 recall, campaigns |

```python
from modules.scheduling import FrequencyCap
cap = FrequencyCap(db)
if cap.check(family_id, now, "recall_immunization", tier=FrequencyCap.TIER_OUTREACH).allowed:
    ...
```

## TCPA evidence

The `consent` table records channel, purpose, grant timestamp, capture method,
capture evidence reference, capturing staff member, revocation timestamp and
revocation method. Reminders and marketing are separate consents and are never
conflated. Revocation writes a suppression row in the same call — the consent row
is the evidence, the suppression list is the enforcement.

Pass an `AuditLog` to `InboundRouter(..., audit=log)` to mirror consent grants and
revocations into the append-only log from `nsp_core/audit.py`, so opt-out evidence
sits behind DB-level UPDATE/DELETE triggers.

## Operational notes

- `Database(":memory:")` is **refused**. Thread-local connections cannot share an
  anonymous in-memory database: the backfill pool would crash on "no such table"
  and a race test against it would prove nothing. Use a file path.
- A quiet-hours deferral writes `send_after_utc` and preserves `planned_utc`, so
  the record of what the cadence rule decided survives and reporting windows do
  not shift under the row.
- A waitlist entry's `notify_channel` must have a matching address on the family
  record (`address_for`). An unreachable family is excluded at selection time,
  not discovered at send time after it has consumed a blast slot.
- Message bodies avoid `%-d` / `%-I`, which are glibc-only and raise on musl
  (Alpine) and Windows.

## Not in scope here

- **FHIR sync.** Appointments are mirrored into this database because backfill
  needs a row it can lock. Reconciling back to `FHIR Appointment` is the
  integration boundary and belongs in the EHR adapter, not here.
- **Self-service booking widget.** `VisitType.SELF_SCHEDULABLE` encodes the rule
  (well visits only); the widget itself is a front-end.
- **No-show risk prediction.** README I-07: only after twelve months of clean
  data. `lead_time_buckets()` produces that data now.
