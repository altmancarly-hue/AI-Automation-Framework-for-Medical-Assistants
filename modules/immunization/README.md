# I-02 — Immunization Gap Closure & Recall

**The forecaster contains no language model and never may.** README I-02: "Using
an LLM to determine whether a child is due for a vaccine would be actively
negligent. It is a rules problem with an authoritative published rule set, and
the correct answer is a rules engine." An AST test and a `make lint` rule both
fail if one appears.

A model is used in exactly two narrow places, each wrapped in post-conditions:

| Where | What it decides | What stops it being wrong |
| --- | --- | --- |
| `adjudicate.py` | are these two dose records the same event? | README A.2 prompt verbatim; every date and CVX code in the response must appear in the input or the whole response is discarded; UNCERTAIN routes to a human; nothing applies without a named reviewer |
| `messaging.py` | reword a physician-approved template; triage an inbound reply | six post-conditions on the rewrite, any of which sends the approved text instead; STOP is handled before any model runs |

## Files

| File | Responsibility |
| --- | --- |
| `cvx.py` | CVX table, combination-vaccine component expansion, historical trade names. Pure data. |
| `forecast.py` | Generic evaluator over the rules table. Authority abstraction + cross-check + the go-live validation harness. |
| `matcher.py` | Deterministic chart↔registry reconciliation. MATCH / NO_MATCH / AMBIGUOUS. |
| `adjudicate.py` | LLM adjudication for ambiguous pairs only. |
| `messaging.py` | Template personalisation and reply triage. |
| `recall.py` | Gap queue, urgency scoring, cadence, sending. No model. |
| `huddle.py` | Tomorrow's gap sheet, one screen per patient, every line provenance-marked. |
| `pipeline.py` | The nightly batch, in the order the steps have to happen. |
| `fixtures.py` | 24 synthetic histories. No real patient data. |
| `demo.py` | `make demo-i02` — runs the whole thing and prints it. |
| `../../config/immunization_schedule.yaml` | The ACIP rules table. Data, reviewable by a clinician. |

## Wiring

```python
from modules.immunization import PatientInput, RecallEngine, build_huddle, run_nightly

nightly = run_nightly(patients, as_of=today)          # reconcile → adjudicate → forecast → hold
recall  = RecallEngine(db, gateway, validation=validation_result)
recall.run(list(nightly.forecasts.values()), now=now, patients=nightly.patients)
sheet   = build_huddle(db, for_date=tomorrow,
                       forecasts=nightly.forecasts,
                       reconciliations=nightly.reconciliations)
```

## The six decisions worth knowing

**1. Combination products are expanded to antigens before anything else looks at
them.** A Pediarix dose *is* a DTaP dose, a HepB dose and an IPV dose. Without
expansion, a chart recording "Pediarix" and a registry recording three component
rows read as a three-dose discrepancy, and the forecaster recalls a child for
vaccines sitting in their chart.

**2. The schedule is a YAML rules table, and this engine is not the authority by
default.** README I-02 warns against implementing schedule logic yourself, and it
is right. There is no schedule knowledge in Python — every age, interval, series
length and age-out is in `config/immunization_schedule.yaml`, so an ACIP update
is a pull request a clinician can review. `CrossCheckForecaster` lets I-CARE's
own CDSi forecast be primary with this engine flagging disagreement, and a
contested gap is never recalled on.

**3. Outbound sending is gated on a recorded validation run.** README I-02's
control for "forecast logic error creating systematic false positives" is
"validate against 200 known-good records before go-live". `RecallEngine.run()`
raises `RecallNotAuthorized` until a passing `ValidationResult` exists. Queue
building and dry runs always work — and a dry run mutates nothing, so inspecting
the queue cannot silently consume the day-0 cadence slot.

**4. Unresolved means held, not guessed.** An ambiguous pair, a partial date, or
an unrecognised CVX code makes the affected antigens `REQUIRES_REVIEW`, which is
not an open gap, which means the recall engine never sees them. A gap a human
confirms in ten seconds is much cheaper than either a duplicate injection or an
accusatory text to a family who did everything right.

**5. One message per family, not one per antigen.** A child behind on Tdap,
MenACWY and HPV gets one text naming all three. Ranking stays per-antigen — that
is what makes the queue explainable — but sending groups by patient. Because a
per-patient message covers every open gap, a per-patient cap of three per ninety
days is strictly stronger than README I-02's "max 3 per gap".

**6. The urgency formula's first term saturates.** Taken literally,
`days_overdue × antigen_weight` is unbounded: a 16-year-old who never had MMR
scores ~26,000 and permanently outranks an infant whose rotavirus window shuts in
nine days. Overdue-ness also stops carrying information after a while — a gap
open three years is not three times more urgent than one open a year, it is a gap
the previous outreach did not close. The term saturates at a year and the bonuses
are scaled to compete with it. Every candidate carries its `breakdown`, because
an unexplainable clinical priority queue does not get used.

## Shared with I-07

`FrequencyCap` is imported from `modules.scheduling`, on the `outreach` tier —
the lowest allowance, so a week busy with appointment reminders squeezes out the
recall message rather than the reverse. A parent already coming in on Thursday
does not need a text about a gap the clinician will raise in the room; children
with an upcoming appointment are excluded from the queue entirely and appear on
the huddle sheet instead.

TCPA basis is recorded in code as `RECALL_CONSENT_BASIS`: recall is an
informational treatment communication to an established patient, sent under the
`reminders` consent purpose, never `marketing`.

## Scope, stated plainly

The rules table covers the routine schedule for healthy children. It does **not**
encode high-risk or immunocompromised indications, travel schedules,
outbreak-response dosing, or the full CDSi observation set. Any patient with a
condition-based indication must be forecast by a clinician or by I-CARE.

`RegistryForecaster` is a deliberate, deliberately-loud integration seam: filling
it in is an HL7 2.5.1 query and a response parser, and doing so makes I-CARE the
authority with this engine as the cross-check — which is the README's actual
recommendation.
