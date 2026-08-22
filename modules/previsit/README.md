# I-03 — Pre-Visit Chart Preparation

**The rules engine produces the checklist. The model produces the narrative.**
README I-03 says they "are not interchangeable", and this package is arranged so
they cannot be mistaken for one another — in the code, in the audit trail, and
on the screen the MA reads.

| Question | Answered by | Why |
| --- | --- | --- |
| Which screenings are due at this age? | `periodicity.py` + `config/periodicity.yaml` | A lookup table. README I-03: "Do not use an LLM." |
| Are immunizations due? | I-02's CDSi engine | Already built |
| Has this child crossed growth percentiles? | `growth.py` + CDC LMS tables | Pure math |
| Open referrals, unreturned results? | Structured query | Deterministic |
| **What happened at the last three visits that matters today?** | **`narrative.py`** | Genuinely unstructured clinical narrative. No rules engine does this. |

`make lint` asserts no model import in `periodicity.py`, `growth.py`, `brief.py`
or `feedback.py`, and a test walks the AST of `periodicity.py` to prove no
screening name, CPT code or schedule age appears as a Python literal.

## Files

| File | Responsibility |
| --- | --- |
| `growth.py` | CDC LMS percentiles, z-scores, and channel-crossing detection. |
| `periodicity.py` | Bright Futures + Illinois screening schedule, evaluated per patient. |
| `narrative.py` | The one model call. Appendix A.4 prompt, four post-conditions. |
| `brief.py` | One screen. Computed and generated content structurally separated. |
| `batch.py` | The 17:00 job — a fixed pipeline, deliberately not an agent. |
| `feedback.py` | Useful / not useful / wrong, and whether the narrative is earning its space. |
| `fixtures.py` | Six synthetic patients, including three adversarial ones. |
| `demo.py` | `make demo-i03`. |
| `../../config/periodicity.yaml` | The screening schedule. Data, reviewable by a clinician. |
| `../../config/growth/` | CDC LMS tables, verbatim from cdc.gov, with a checksum manifest. |

## Wiring

```python
batch = run_batch(
    tomorrows_patients,                       # list[PatientDay]
    clinic_date=clinic_date,
    schedule=PeriodicitySchedule.load(),      # refuses without a clinical owner
    reference=GrowthReference(),
    synthesizer=NarrativeSynthesizer(client, audit=audit,
                                     deidentifier=HybridDeidentifier(),
                                     leak_guard=LeakGuard()),
    generated_utc=now,
    feedback=FeedbackLog(db, audit=audit),
)
for brief in batch.briefs:
    print(render_text(brief))
```

## The growth reference

`config/growth/` holds eight CDC LMS tables downloaded verbatim from
`cdc.gov/growthcharts/data/zscore/`, with a `MANIFEST.json` recording each
file's SHA-256. `make lint` verifies the checksums. **There are no growth
numbers in `growth.py`** — same discipline as the immunization schedule, for the
same reason: a reference update has to be a data diff a clinician can sign off.

The arithmetic is CDC's own LMS method, `z = ((X/M)^L − 1)/(L·S)`, with L, M and
S linearly interpolated to the exact age (CDC's stated method — rounding a
30.4-month-old to 30 months moves the percentile enough to invent or hide a
crossing). Every value CDC prints in those files is a test case: computing the
percentile of the published P75 weight must return 75. **16,262 published points
reproduce to within 0.007 percentile.**

**Not shipped, deliberately:** the WHO 0–24 month standards that AAP prefers for
infants. `GrowthReference` is source-agnostic and `table_for` is the single
place that decides which file applies, so dropping WHO tables in is a config
change; the current tables are CDC 0–36 months and CDC 2–20 years. The CDC 2022
*extended* BMI percentiles for severe obesity are likewise not implemented, and
the classic LMS z becomes unstable above roughly the 97th BMI percentile.

## The seven decisions worth knowing

**1. The check refuses more often than it computes.** An out-of-range age, an
unrecognised sex, a recumbent length compared to a standing stature, a weight
that gives z = +10 — all raise rather than returning a number. The
length/stature switch happens to *every* child around the second birthday, at
the same time the reference table changes underneath them, and the two methods
differ by about 0.7 cm, which is a channel at some ages. A pair measured
differently is reported as **not comparable**, not as a finding.

**2. Crossing detection works in the tails.** The printed chart lines stop at
the 3rd and the 97th, so counting lines returns zero for a child already outside
them — including a child falling from the 1.8th percentile to the 0.008th, which
is the most alarming trajectory in the schedule. A movement of ≥1.34 SD (two
channel-widths, expressed in a unit that survives the tails) is significant on
its own.

**3. A screening has a window, a catch-up horizon and an age band, and all
three are data.** Without a grace window every brief shows a permanent backlog
of screenings that were in fact completed. Without a catch-up horizon a missed
two-year TB risk assessment sits OVERDUE on a thirteen-year-old forever. Both
produce a checklist that is always red, which is a checklist nobody reads —
README I-03's own "brief becomes noise" risk. The loader **refuses** a discrete
screening with no horizon and a banded screening with no end.

**4. Each completion satisfies exactly one due age.** Any two due ages closer
together than twice the window otherwise let one completion satisfy both: a
single M-CHAT-R/F at twenty-one months read as complete for the eighteen- *and*
twenty-four-month screens, and the twenty-four-month autism screen never
appeared on any brief for that child.

**5. Risk-based screenings stay off until the risk assessment answers, and the
assessment is itself a due item.** A lipid panel due for every patient is noise;
one that never appears because nobody was asked the risk question is a miss.
Three distinct states: assessment outstanding (the test is not printed twice —
one line, the actionable one), assessment done and negative (**not indicated**),
assessment never done and now un-doable (**unknown**, worth one line).

**6. Go-live has a data horizon.** `data_horizon_months` marks the age before
which the practice holds no screening history for a patient. Older targets
report as UNKNOWN — "check the paper chart once" — instead of asserting a miss
the practice can disprove. Without it, day one shows every established patient
overdue for a decade of screenings that were done on paper, every brief is red
by lunchtime, and the periodicity section is ignored inside a week. What is due
at *today's* visit is never hidden by the horizon.

**7. The narrative is bounded by post-conditions, not by the prompt.** A.4 tells
the model to cite dates and to write no recommendations. That is a request.
Every returned item is then checked, and dropped if it fails, whether or not the
model complied:

| Check | Catches |
| --- | --- |
| Cited date was actually supplied | An invented encounter — an item that reads *better* sourced than an uncited one |
| Claim appears in **that** note | A true statement filed under the wrong date, which sends the reader to the wrong place in the chart |
| No changed numbers | "amoxicillin 500 mg" against a note saying 250 mg |
| No polarity inversion | "Positive depression screen with suicidal ideation" against "PHQ-A negative. Denies suicidal ideation." |
| Word order survives | "started since stopping the medication" against "stopped since starting" |
| No clinical judgement language | "Needs an inhaled steroid", "GI referral warranted", "may benefit from escalation" — while "Allergy referral **placed**, no report received" is a fact and survives |

Everything dropped is counted and shown. The signed count reaches the brief.

## What these checks cannot do

They are **lexical**. Polarity and word-order checks catch the common
inversions; some need a human. De-identification is regex-first — it removes
phone numbers, record numbers and identifiers, and it does **not** remove person
names unless the optional clinical NER model is installed, which is why a
warning is attached to every brief where it was not, and why `LeakGuard` cannot
catch them either. Both are reasons this runs on **local inference** (README
3.1), and reasons the whole narrative section is marked AI-generated with every
line carrying its source date. README I-03's stated control is that the brief is
a pointer, not a source of truth; these filters reduce the residue, they do not
eliminate it.

## The screen

One screen, enforced (`MAX_LINES`). Sections are appended in priority order and
the cap trims from the end, so **computed content outranks generated content**
when space runs out — an earlier version dropped two Illinois statutory school
forms to keep six unvalidated model lines. A truncated section says how many of
its own items are hidden rather than leaving the reader to notice.

Every AI line carries `~` and its source date and sits under an AI-generated
header; no computed line carries the marker and no AI text appears outside the
section. Medication doses are scrubbed from the narrative before it renders —
README I-03 asks the brief to "deliberately omit data that must be verified
in-chart", and a brief printing "amoxicillin 400mg/5mL, 7mL BID" invites someone
to dose from it.

## The feedback loop

`WRONG` is tracked separately from `NOT USEFUL` and never folded into a
satisfaction score: they are different failures, and a metric that lets three
"useful"s cancel a clinician who was pointed at something untrue cannot see the
thing it exists to see. Rates are per **brief**, not per click. A `WRONG` verdict
must say what was wrong — that free text is the entire input to the next prompt
revision.

`report()` splits by whether the AI section actually *reached the page*. If
briefs with narrative rate worse than briefs without, the decision is whether to
keep generating it — not how to tune the prompt.

## Before production

- `config/periodicity.yaml` has **no clinical owner** and `run_batch` refuses
  until one is assigned. That is README I-03's control, in the code path rather
  than on a checklist.
- `config/growth/biv.yaml` likewise needs an owner.
- The screening table covers routine well-child care. It does not encode
  condition-specific surveillance, and a patient with a chronic condition needs
  a clinician's schedule, not this one.
