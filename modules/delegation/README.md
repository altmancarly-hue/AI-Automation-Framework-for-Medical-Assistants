# I-10 — Standing Order Digitization & Delegation Audit

**The initiative with the smallest dollar return and the largest downside
protection.** It answers one question, on demand, in writing: *by what authority
did that medical assistant do that?*

Today the answer lives in a binder, in the memory of MAs who have worked here
for years, and in a set of standing orders last reviewed before the schedule
changed. That is fine right up until it is a survey, a board complaint, or a
deposition.

## The 2027 problem, and why the rules are YAML

225 ILCS 60/54.2 — the Illinois clause that lets a physician delegate to an
unlicensed assistant — **sunsets on 2027-01-01.** A register that hard-codes
that statute becomes a rewrite on New Year's Day. So it does not: the framework
citation, the sunset date, the licensed and delegable roles, the requirement
list, the break-glass policy and the review cadence all live in
`config/delegation_rules.yaml`. Adopting the successor framework is a YAML edit
and a review signature.

`test_no_statute_specific_logic_is_hard_coded` and `make lint` both hold this
line.

## How much of this is a model?

**None of the enforcement path.** `make lint` and
`test_no_model_in_the_enforcement_path` assert that neither `register.py` nor
`enforcement.py` imports `nsp_core.llm`, `openai` or `anthropic`. A model is a
plausible authoring aid — drafting protocol text for a physician to read, sign
and own — and that is not built here. Whether Jess may give this injection at
11:40 today is a lookup, and a lookup that a model gets right 99% of the time is
a lookup that is wrong once a week.

## Files

| File | Responsibility |
| --- | --- |
| `register.py` | `StandingOrder`, `OrderRegister`, `Competency`, `CompetencyRecord`, `CompetencyRegister`, `Roster`, `DelegationRules`. The record layer. |
| `enforcement.py` | `DelegationService`: `authorise()`, `execute()`, `available_orders()`, `audit_extract()`, `readiness()`, `certify()`. |
| `fixtures.py` | Six synthetic staff — including a six-week hire, a lapsed CPR card, and an RN who is both a licensed and a delegable role. |
| `demo.py` | `make demo-i10`. |

## The five claims under test

**1. An MA sees only what they are currently competent for.**
`available_orders()` is not a filter over a hard-coded list; it is the same
authorisation check the execution path runs. Dana, six weeks in with a vitals
competency, sees vitals.

**2. The check runs again at the moment of the act.** `execute()` re-derives the
authorisation rather than trusting whatever the screen showed. A competency can
expire between the MA opening the worklist at 08:00 and giving the injection at
11:40, and the physician goes to lunch.

**3. A signed standing order is never edited, only superseded.** There is no
`update()`. `publish()` adds a version and retires the previous one, and it
refuses:

- an unsigned order, or one whose signer is not the named delegating physician;
- an order naming **no required competencies** — an empty list makes the
  competency loop not run, so the check passes by vacuum and every person in a
  delegable role is authorised. A blank competency list on an epinephrine
  protocol is the single most likely data-entry omission in the register, and
  missing data must never widen the gate;
- a version effective **before the previous version began** — back-dating
  retroactively rewrites which text was in force, so an act already logged
  against v2 resolves to text signed six weeks after the injection;
- a version effective **before its own signature** — a delegation cannot predate
  the act of delegating.

**4. Break glass never blocks care, and is never quiet.** It covers the
requirements it is *for*: an expired competency, an absent supervisor, an
overdue protocol review. It does not cover the existence of the act. An order id
that resolves to nothing is a typo, and performing one records a completed act
against no protocol, no version, no supervisor and no competency — an
unauditable row, which is the opposite of what this module exists to produce.
Break glass therefore refuses a nonexistent order and an unsigned one, demands a
justification of at least the configured length, and schedules the review.

**5. The audit extract is a query, not a project.** `audit_extract()` takes
`staff_id`, `since` (date A), `until` (date B) and `as_of`, and scopes
executions, competency records and unevidenced-supervision detail by all of
them. Scoping is a privacy control, not tidiness: the unevidenced-supervision
query used to have no `staff_id`, so an extract requested for one MA disclosed
another employee's break-glass event **and its free-text justification**, which
is an HR record.

`audit_extract()` and `execute()` both call `certify()`, which refuses after the
sunset and refuses when `config/delegation_rules.yaml` has no named owner. The
extract is the one-click compliance document; a document citing a repealed
statute is worse than producing nothing, because it looks like an answer.

## Commands

```bash
make test-delegation   # 50 tests
make demo-i10          # authorisations, a refusal, break glass, the extract
make lint              # includes: delegation rules are config, not code
python3 -m modules.delegation.demo
```

## Wiring

```python
from modules.delegation import DelegationService, NotAuthorised
from modules.delegation.register import DelegationRules

rules = DelegationRules.load()               # config/delegation_rules.yaml
service = DelegationService(
    rules=rules, orders=orders, competencies=competencies,
    roster=roster, audit=audit_log,
)

worklist = service.available_orders("ma_jess", moment=now)

try:
    service.execute(
        "ma_jess", "so_immunize",
        patient_id=patient_id, moment=now, execution_id=uuid4().hex,
    )
except NotAuthorised as refusal:
    show(refusal.blocking)                   # names the rule, not "denied"

report = service.readiness(today)            # what will stop somebody working
extract = service.audit_extract(             # what counsel asked for
    staff_id="ma_jess", since=date(2026, 4, 1), until=date(2026, 6, 30),
)
```
