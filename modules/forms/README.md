# I-01 — Automated Forms Pipeline

**The highest-leverage automation available to a pediatric practice, and the one
almost nobody builds.** It is also the hardest build in the set: it touches OCR,
an EHR API, a state registry interface, PDF manipulation, e-signature and a
review screen.

An MA transcribing thirty immunization dates from a chart onto a state form is
not exercising clinical skill; they are acting as a copy-paste mechanism between
two systems that could talk to each other directly. Nine minutes per form, 6,500
forms a year, peaking in July and August when the practice is also running
back-to-school physicals.

## How much of this is a model?

Roughly 30%, and not the part you would guess.

| Sub-task | Answered by | Why |
| --- | --- | --- |
| Which form is this? | `detect.py`, anchor phrases | Deterministic once you have a library. |
| Where is each field on it? | `templates.py` + `config/forms/*.yaml` | A coordinate map. **Do not use an LLM for this** — slower, costlier, less reliable. |
| What are the child's vitals? | `chart.py`, FHIR-shaped | The data is already structured. |
| Do the chart and the registry agree? | `reconcile.py` → **I-02's matcher** | CVX code + date tolerance. Rules, not judgement. |
| Which ambiguous pairs are the same event? | I-02's A.2 adjudicator | The one genuinely hard call, already built and gated. |
| **A form nobody has ever seen** | `detect.py`, vision model | The one place a model is clearly superior — and the one place a human confirmation step is mandatory. |

`make lint` asserts no model import in `templates.py`, `chart.py`, `fill.py`,
`reconcile.py`, `review.py`, `lifecycle.py`, `probe.py`, `pipeline.py` or
`blankforms.py`. `detect.py` is the only file in the module that may call one.

## Files

| File | Responsibility |
| --- | --- |
| `templates.py` | Field list, bounding boxes, source paths, named transforms. Refuses an uncalibrated form, and refuses a source path `CHART_SCHEMA` does not have — at load time, because a mistyped path used to reach the MA as "the chart holds no value for this field", a statement about the child describing a config bug. |
| `chart.py` | The structured pull, with provenance and staleness on every value. |
| `detect.py` | Anchor-phrase classification; a vision model proposes a template for a human to confirm. |
| `reconcile.py` | Adapter over I-02. Marks what nobody has settled; never picks a winner. |
| `fill.py` | Writes values and a highlight on every one of them. PDF library behind an interface. |
| `review.py` | The split-pane payload, and the release gate with teeth. |
| `lifecycle.py` | The tracking record, so no form is "in the pile somewhere". |
| `probe.py` | The synthetic-error control. |
| `pipeline.py` | The ten target-state steps in order. No model runs here. |
| `blankforms.py` | Draws the synthetic demo form the tests fill. |
| `fixtures.py` | Six synthetic cases, one per failure mode. |
| `demo.py` | `make demo-i01`. |

## Commands

```bash
make test-forms        # the module's own suite
make demo-i01          # six forms end to end, plus every gate refusing something
make lint              # the structural guards above
python3 -m modules.forms.demo
```

## Wiring

```python
from modules.forms.fill import FormFiller, PyMuPDFBackend
from modules.forms.lifecycle import FormTracker
from modules.forms.pipeline import FormPipeline
from modules.forms.probe import ProbeProgram
from modules.forms.review import ReleaseGate
from modules.forms.templates import TemplateStore

pipeline = FormPipeline(
    store=TemplateStore.load(),          # refuses uncalibrated templates
    chart_source=your_fhir_source,       # implements ChartSource
    filler=FormFiller(PyMuPDFBackend(), audit=audit),
    tracker=FormTracker(audit=audit),
    gate=ReleaseGate(),
    probes=ProbeProgram(audit=audit),    # 1 in 50
    audit=audit,
)

prepared = pipeline.prepare(
    request, blank_pdf=blank, destination=out, now=now,
    chart_doses=chart_doses, registry_doses=icare_doses,   # None = not consulted
)
print(prepared.review.blockers)          # empty means it may go for signature
```

## The placeholder rule

The Illinois Certificate of Child Health Examination is a state document with a
fixed printed layout. This repo ships its **field list** — public, and the part
that takes a day to get right — with every bounding box marked
`placeholder: true`. `TemplateStore.load()` refuses to fill it.

That gate is not caution. A tetanus date written forty points too low lands in
the next row of the immunization grid. The form then says a child had a dose
they did not have, on a legal school document, in a box a physician signs — and
it is invisible on a screen at review size. Calibration is a deployment task
with a name and a date on it.

The `demo_camp_health_form` is different: it is synthetic, this repo draws its
blank PDF (`blankforms.py`) at the template's own coordinates, so its numbers
are measured rather than guessed. That is what makes the pipeline testable end
to end without shipping a state document.

## Where it refuses, and where it carries on flagged

| Refuses — no document at all | Carries on, marked and blocked |
| --- | --- |
| the form type is unrecognised | the registry was not consulted |
| the template is uncalibrated | a dose is disputed |
| the chart system is unreachable | a required box could not be filled |
| the patient was not confirmed | a value was truncated to fit its box |
| | a vital is stale |

The right-hand column is README I-01's *"degrade gracefully to chart-only fill,
flag the form as 'registry not reconciled', continue to function"*. The left is
its *"Never auto-match on a fuzzy name alone"*.

**The patient match is not done in this module.** I-06 already has the matcher
that treats a recorded twin flag as binding and refuses a near tie.
`pipeline.prepare` takes a patient id a person or that matcher has settled, and
refuses to run without one.

## Every auto-filled field is highlighted

README I-01 step 6, and this module treats it as a control rather than a
feature. The highlight is emitted in the **same loop** as the write — not a
second pass, which is a thing that can be skipped, reordered or made conditional
— and `FilledForm.verify()` runs inside `fill()` and refuses to return a form
where the counts disagree. There is no `highlight=False` parameter, because the
only reason to want one is to make machine-written text indistinguishable from a
clinician's.

Four things the filler will not do:

1. **Write into a signature field.** Ever. A signature is an attestation.
2. **Write a disputed immunization dose.** Checked twice — once in the
   transform, once in the fill loop — because this is the check whose failure
   puts a wrong date on a legal document.
3. **Write anything a transform could not render.** The box stays blank and the
   reason is recorded. Never a raw value, an empty string, or "N/A" — on a
   school form, "N/A" in the allergy box is a clinical claim. An empty allergy
   list is the same: "no known allergies" is something a person says.
4. **Silently overflow a box.** Width is **measured** against the font metrics,
   not estimated: a 0.55-em-per-character estimate let a 67-character upper-case
   allergy list pass and then render seventeen points past the rule, unflagged
   and outside the highlight. Text too wide is trimmed with an ellipsis,
   recorded at full length, and blocks release. The renderer re-checks and
   raises rather than drawing ink outside its box.
5. **Print a dose that has no box.** Rows sort oldest-first, so without a
   capacity check it was always the most recent dose — the one the school
   checks — that fell off the end of a short form, silently.

## What goes on the grid

`reconcile.py` calls I-02's `reconcile()` and its A.2 adjudicator. It does not
reimplement them — the CVX matcher, the four-day tolerance, the
combination-product logic and the "no machine determination without a named
reviewer" rule are all built and reviewed there, and a second copy would drift.

What this module adds is the policy: **a dose nobody has settled does not go on
the form.** An ambiguous pair, an unknown CVX code, or a duplicate inside one
source marks its antigen row disputed. Both sides of a disputed pair still
appear on the review screen, marked — a row that says "we could not settle DTaP"
with nothing to look at tells the MA nothing.

A combination product lands in every row it belongs to: Pentacel is DTaP *and*
polio *and* Hib, and a school form has a box for each.

## The release gate

A review screen is not a control. A review screen plus a gate that refuses to
release is. `ReleaseGate.blockers()` returns every reason a form does not reach
a physician's signature queue, and `record_review` refuses to record an
*approval* while any of them stand:

- an unresolved immunization discrepancy
- more doses in a row than the form has boxes for
- the registry never consulted
- a required box nothing filled
- a value truncated to fit its box
- a stale vital that actually reached a box on this form
- an outstanding synthetic probe, or an injected value still on the document

A *rejection* is always recordable. The gate blocks release, not documentation.

The state machine reads the blockers **off the request**, where the pipeline put
them when it produced the review. They used to be an optional argument to
`FormTracker.advance` defaulting to `()`, which made "this form is clean" and
"the caller did not mention it" the same thing — so any caller could sign a form
with an unsettled immunization on it and the ledger recorded a clean history.
Moving a form back to `filled` clears the review, because the review that
cleared the last document does not clear the next one.

## The synthetic probe

The build plan calls this "the single most-omitted control in real deployments",
and the reason it gets omitted is that it feels perverse: the system
deliberately does the thing the system exists to prevent.

The alternative is worse and it is the default. A pipeline that is right 99% of
the time trains the reviewer, over about three weeks, to click approve. After
that the review step still exists on the screen and no longer exists in fact,
and **nothing in the data says so** — the error rate looks the same, because the
errors the reviewer stopped catching are the same errors nobody was making. The
first time it matters is a real one.

So one form in fifty carries a deliberate error, and `probe_catch_rate()` is the
number that says whether the review is real.

Four rules, each a way this control goes wrong in practice:

1. **A probe never reaches a signature.** It is a release blocker until scored.
2. **A probe is never clinically dangerous.** An allowlist of safe fields —
   heights, weights, dates, names. Never an allergy, a medication, a condition
   or an immunization date. A "test" that puts a wrong tetanus date on a form
   has placed a real risk to catch a hypothetical one.
3. **A probe is a probe in the audit log and nowhere else.** Written with
   `synthetic_probe=True` so it stays out of the genuine edit-rate statistics.
   And a probed write keeps its template source path, so probing a box cannot
   hide that a stale chart value reached the form.
4. **The rate is deterministic per form.** Hashed from the request id, so a
   retry cannot re-roll the dice until a form gets probed.

*Caught* means the reviewer corrected **that field to something other than the
injected value**. Not "made any edit", not "rejected the form", and not
resubmitting the injected value verbatim under a rejection — a reviewer who
rejects everything would otherwise catch every probe and review nothing. A probe
on a form that was blocked for an unrelated reason and re-filled is
**withdrawn**, not scored: it was never fairly presented.

**Scoring a probe does not clean the page.** It records what the reviewer did.
The injected value is still printed, so `ReleaseGate` blocks on
`FilledForm.has_synthetic_write` — a property of the document, not a flag in the
probe registry a caller could strip — until the form is re-filled. A request
whose probe has been scored is never probed again, so the second pass is clean;
without that, the deterministic hash re-probed the same request forever and the
only way it could ever be released was with an injected error on it.

## The tracking record

README I-01's failure table has an entry with no clinical content in it at all —
*"Form lost in the signature pile → child cannot start school or a sports
season"* — and prices the consequence at twelve "where is my form" calls a week.

`lifecycle.py` is the answer: a state machine with named transitions, an
immutable history, and three guards. A form cannot reach the signature queue
with blockers outstanding. Signing names the licensed professional. Delivery
happens only after signing, and records where it went.

`overdue()` answers the parent's question before they ask it. `stalled_at()`
says which stage the practice is actually slow at — a turnaround number alone
tells you that you are slow without telling you where.

## Tests

`tests/test_forms.py` — 109 tests, no mocks of this module's logic.
`StaticChartSource`, `RecordingBackend` and `EchoTransport` are shipped test
doubles standing in for the EHR, the PDF library and a model server.

The PDF tests run against a blank form this repo **generates** at the template's
own coordinates, then read each value back out of its box. A coordinate bug
shows up as text in the wrong box rather than as bookkeeping agreeing with
itself — which is how the generator's own field labels were caught overlapping
the boxes they labelled.

The last section of the file, `adversarial-review regressions`, holds one test
per finding from the module's adversarial review. Each reproduces the ORIGINAL
defect and each was mutation-checked: the bug reintroduced, the named test
confirmed to fail. Four of the review's findings were found *by* that
mutation pass rather than by reading the code — including that the second
dispute lock, the one the section above calls load-bearing, had no test of its
own.
