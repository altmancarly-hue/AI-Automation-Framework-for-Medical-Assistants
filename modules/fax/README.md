# I-06 — Inbound Fax and Document Ingestion

**Faxes are the one place in the practice where an LLM earns its keep on
robustness alone, and the one place where a wrong answer is invisible.**

README I-06 makes the case for the model: *"Faxes are unstructured,
inconsistently formatted, frequently poor quality. Rule-based classification on
faxes fails constantly; this is exactly where LLM robustness pays."* It also
names the CRITICAL risk: a critical lab value classified as an administrative
document lands in a re-filing queue and nobody ever reads it.

So this module uses a model for exactly two questions and refuses to use one for
anything else.

| Question | Answered by | Why |
| --- | --- | --- |
| What kind of document is this? | `classify.py` (model) | Genuinely unstructured. Rules fail. |
| Does this text read as urgent? | `urgency.py` (model) | Same. |
| Is this value outside the range for this child's age? | `urgency.py` (arithmetic) | A table lookup and a comparison. |
| Which patient is this? | `match.py` (Jaro-Winkler + exact DOB) | Deterministic, auditable, testable. |
| Which queue, which task, which deadline? | `route.py` | Pure logic. No model runs here. |
| Which pages did OCR actually read? | `ocr.py` | Engine confidence, no model. |

`make lint` asserts that `ocr.py`, `match.py` and `route.py` import no model at
all, that no OCR engine is imported at module scope, and that the classifier's
JSON-schema `enum` still equals `DOCUMENT_TYPES`.

**The model never decides anything on its own.** Every model answer passes
through a deterministic rule that can only escalate it, and no document reaches
a chart without a recorded human action.

## Files

| File | Responsibility |
| --- | --- |
| `ocr.py` | Engine chain with fallback. Document confidence is the WORST page. |
| `classify.py` | The taxonomy, the one classification call, and the eval gate that decides whether the classifier may route at all. |
| `match.py` | Name/DOB matching against the panel. Twins and duplicate charts go to a human. |
| `urgency.py` | The A.3 urgency call, plus the age-banded reference ranges that override it. |
| `route.py` | Queue, tasks, deadlines, and the I-02 immunization handoff. |
| `pipeline.py` | OCR → classify → match → triage → route, one document at a time. Plus the gateway-silence monitor. |
| `fixtures.py` | Ten synthetic faxes, several adversarial, and a ten-patient panel with twins and a duplicate chart. |
| `demo.py` | `make demo-i06`. |
| `../../config/reference_ranges.yaml` | Age-banded ranges and critical phrases. Data, reviewable by a clinician. |

## Wiring

```python
from modules.fax import (
    ClassifierGate, DocumentClassifier, FaxPipeline, PatientMatcher,
    ReferenceRanges, UrgencyTriage, evaluate,
)

ranges = ReferenceRanges.load()          # refuses to run without a clinical owner
classifier = DocumentClassifier(client, audit=audit)

# The gate. Measure first, route second.
gate = ClassifierGate()
gate.approve(evaluate(classifier, labelled_historical_faxes), on="2026-08-24")

pipeline = FaxPipeline(
    engines=[PaddleEngine(), TesseractEngine()],
    classifier=classifier,
    matcher=PatientMatcher(panel),
    triage=UrgencyTriage(client, ranges=ranges, audit=audit),
    gate=gate,
    audit=audit,
    on_duty_physician="dr_alvarez",
    indexer="ma_jess",
)

processed = pipeline.process(path, document_id=doc_id, age_lookup=ages)
print(processed.routed.queue, [t.kind for t in processed.routed.tasks])
```

## Commands

```bash
make test-fax          # the module's own suite
make demo-i06          # the whole pipeline over ten synthetic faxes
make lint              # the structural guards listed above
python3 -m modules.fax.demo
```

## The five queues

| Queue | Reached when | Task raised |
| --- | --- | --- |
| `auto_file` | Routine, matched, administrative, confidently classified | `acknowledge` (72h) |
| `physician_review` | Clinical content, or a model answer below its confidence floor | `physician_review` |
| `urgent_alert` | Any rule fired: critical value, critical phrase, lab's own flag, failed triage call | `urgent_contact` (1h) + `verify_urgency` (2h) |
| `human_indexing` | Unreadable page, no patient, ambiguous patient, duplicate chart, or `other` | `index_by_hand` |
| `immunization_reconciliation` | An immunization record | `reconcile_immunizations` (48h) → I-02 |

## The eval gate

README I-06 evaluates the classifier against 300 historical faxes.
`ClassifierGate` turns that from a milestone into a control: `require()` runs on
the production path, so an unmeasured classifier does not route documents badly
— it refuses to route them at all.

`EvalResult.blockers()` is the list of reasons a classifier must not go live:

- fewer than 100 labelled documents
- fewer than 100 **distinct** documents — a corpus tiled to hit the count
  measures the classifier against the corpus, not against the count
- any document type with no examples at all, or fewer than five
- per-class recall below its floor (0.95 for `outside_lab` and
  `hospital_discharge`, 0.90 for the other clinical types)
- more than 20% of answers below the confidence threshold — a classifier that is
  never sure gets every question it answered right and routes nothing
- clinical documents predicted as administrative, above a 2% budget. This is
  README I-06's CRITICAL risk expressed as a confusion-matrix cell. The budget
  is not zero because a gate that can never pass gets switched off.

`approve()` pins the provider, the model id, the model version, the prompt hash
and the confidence threshold. Change any one of them and `require()` refuses:
the eval measured a model, not a prompt.

## Why the confidence floors point the way they do

Two model calls, two floors, and they fail in the same direction:

- **Classification below 0.75** → the answer is not used, the document goes to a
  person.
- **Urgency below 0.6** → the answer is not used, the level is raised.

And the deterministic override is one-way: `_apply_overrides` can move a
document from `routine` to `urgent` and never the other way. A model that says
"routine" about a potassium of 7.4 is overruled by arithmetic; a model that says
"urgent" about a cover sheet is believed.

## Reference ranges are applied at the age of COLLECTION

A newborn bilirubin of 9.4 is unremarkable at thirty-six hours of life and
alarming at two months. A fax that took six weeks to arrive would otherwise be
scored against the wrong band and force-flag a normal newborn result — and an
alert that fires on healthy babies is an alert nobody reads.

So `pipeline.process` reads the collection date off the document, computes the
age the specimen was taken, and scores against that band. A collection date in
the future is discarded rather than trusted. A collection date **before the
child was born** is discarded too, whichever reading produced it.

### `03/04/2024`

That is 4 March in Springfield and 3 April in Sherbrooke, and the fax does not
say which. `date_readings()` reports both; nothing in this module picks one
silently.

- **On a date of birth**, the panel resolves it. Both readings are matched; if
  exactly one lands on a patient, the chart has answered a question the string
  could not. If both land on different patients, or if *either* reading lands on
  patients that need a person to choose between them (flagged twins, a duplicate
  chart), the document goes to `human_indexing` with the reason attached.
  Matching on the wrong date of birth does not fail loudly — it files a
  specialist's letter in a stranger's chart.
- **On a collection date**, both ages are scored. Choosing either one is
  choosing an answer the document did not give, and "take the younger, the bands
  are narrower there" is false against the shipped table — the bilirubin ceiling
  is 15.0 below one month and 1.2 above it, so the younger reading *clears* a
  bilirubin of 9.4 that the older one pages a physician about. See below.

Both cases append a line to `ProcessedDocument.warnings`, since a person can
settle it off the letterhead in a second.

## What a missing or uncertain age does

**There is no band for an unknown age, so none is manufactured.**

Two versions of this were wrong before the current one, in opposite directions.
Taking the *widest* band across childhood cleared a potassium of 5.8 on exactly
the documents that most need scrutiny — a safety threshold widened by missing
data. Taking the *intersection* of every band looked like the fix and was worse:
the shipped bands are not nested (hemoglobin runs 13.5–21.5 at 0–1 month and
9.5–14.0 at 1–6), so the intersection came out 13.5–13.5 and force-flagged 140
of 141 hemoglobin values — the entire human-indexing queue, paged.

So `scan()` takes **candidate ages** and applies one rule to whatever it gets:

| Situation | Candidate bands | A value outside them all | A value outside only some |
| --- | --- | --- | --- |
| Age known | one | abnormal → **urgent** | — |
| Collection date reads two ways | two | abnormal → **urgent** | unsettled → **urgent** |
| No age at all | every band the table ships | abnormal → **urgent** | unsettled → **physician review** |

A finding is *settled* only when it holds at every age the result could belong
to. When it holds at some ages and not others, that is reported as such — the
`age_unknown` source — and never resolved by picking an age.

**Why the last two rows escalate differently.** Two candidate ages means a known
child and a coin toss between two real dates; a coin toss on a bilirubin is a
one-hour page. No age at all means an unmatched document scored against all of
childhood, where almost every ordinary CBC is abnormal at *some* age — paging on
that would page on everything, which is the failure this module names twice in
its own docstring, and there is no identified patient to page about anyway. It
still goes to a person with the values named. It is never cleared.

A stated collection date that no reading can make sense of (a mis-OCR'd year,
both readings before the child was born) **withdraws** the age rather than
falling back to the age on arrival. Falling back would be inventing a collection
date the document contradicts.

## Twins, siblings and duplicate charts

`match.py` requires an exact date of birth — a date of birth is never fuzzy
matched — and a name score of 0.92 with a 0.10 margin over the runner-up. On top
of that:

- **A recorded `multiple_birth` flag on any patient sharing the date of birth is
  binding.** It goes to a human whatever the name score. The flag is a fact
  about the panel; the score is a guess about the fax. `Ada` and `Ava` differ by
  one character and score 0.92 against each other.
- **Duplicate charts** (same name, same date of birth, two patient ids) are
  reported, never guessed between.
- `matcher.duplicate_report()` and `matcher.unflagged_multiples()` are panel
  hygiene, run before go-live.

Name parsing handles `Last, First`, `First Last`, HL7 `LAST^FIRST^MI` and
`LAST FIRST MI`. Credentials and titles are stripped from a **labelled** list,
not a flat noise set — the flat set deleted the surname of every patient named
`Do`.

## OCR

`chain()` tries each engine in order and falls back on **low confidence as well
as on failure**: an engine that is running and reading a bad fax at 0.4 is not a
success just because it did not raise.

- Document confidence is the **minimum** page confidence, not the mean. A
  five-page discharge summary whose middle page is a black smear is not 80%
  readable; it is 80% readable and 20% missing, and the missing fifth had the
  discharge medications in it.
- An empty page scores 0.0 whatever the engine reported.
- Selection is on `(page_count, confidence, mean_confidence)`. Confidence alone
  returned a one-page fallback over a five-page primary read.
- When engines disagree on page count, a zero-confidence sentinel page is
  appended, so the document needs a person.
- If every engine fails, `chain()` **raises**. The document is not filed and not
  classified.

## The I-02 handoff

README I-06 calls this "the highest-value cross-initiative link in the plan". An
immunization record never writes to a chart. It produces real
`modules.immunization.matcher.DoseRecord` objects with `source="registry"`, and
I-02's matcher, adjudicator and "nothing unresolved without a named reviewer"
rule apply unchanged.

- A month-only date (`06/2024`) is kept as a real dose anchored to the first of
  the month with `precision=MONTH`, which is exactly what I-02's matcher already
  models — it bars the pair from the rule-provable pass and sends it to
  adjudication.
- `MMR (not administered)`, `Tdap refused by parent` and an unparseable date are
  **reported** in `handoff.unresolved`, not dropped. A task that says "0 doses"
  about a page with vaccines on it is worse than no task.

## Critical phrases

`config/reference_ranges.yaml` lists the phrases a laboratory prints when it has
already decided a result cannot wait. Matching is word-anchored — `"stat"`
inside `"status asthmaticus"` turned ordinary discharge summaries into critical
alerts — and within that anchoring it is forgiving of the three things a fax
does to a phrase, none of which changes what a human reads:

- **separators**: `[\s\-]+` between every pair of words, and the config phrase
  is split on hyphens too, so `life-threatening` matches `life threatening`.
- **inflection**: a trailing `s` is optional on every word, in both directions.
  `requires immediate medical attention` in the config matches "Findings
  **require** immediate medical attention" — the ordinary plural subject of a
  radiology impression.
- **case**.

The phrases themselves are an allowlist of clinical completions, not stems. A
bare `"requires immediate"` matched "Requires immediate return of the school
clearance paperwork" and paged the on-duty physician about a sports form.

## Gateway silence

`InboundMonitor` alerts when no document has arrived for too long during
business hours, because README I-06's risk table names a silent fax gateway as a
failure mode with no natural symptom: nothing breaks, the queue is just empty.
Never having received anything is also an alert. Mixed timezone-awareness is
refused rather than coerced.

## Tests

`tests/test_fax.py` — 137 tests, no mocks of this module's logic. `ScriptedOCR`
and `EchoTransport` are shipped test doubles.

The last section of the file, `adversarial-review regressions`, holds one test
per finding from the module's two adversarial reviews. Each reproduces the
ORIGINAL defect, so a fix cannot be quietly reverted — every one of them was
mutation-checked by reintroducing the bug and confirming the named test fails.
Read them as a list of the ways a document ingestion pipeline can be wrong while
looking right.

Three tests in the first batch did **not** survive that check: they asserted
`processed.routed is not None`, which is true of every document by construction.
They now assert the scored band, the resulting finding, and the queue.
