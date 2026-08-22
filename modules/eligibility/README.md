# I-09 — Eligibility Verification & Denial Prevention

**A solved EDI problem with two small, well-fenced model calls bolted to the
edges.** The core — build a 270, parse the 271, decide whether this child is
covered for this visit — is deterministic and stays deterministic. A model reads
an insurance card photograph and buckets the free text on a denial. That is all
it does.

The practice's published insurance list is dated **January 2016**. A payer table
with no effective dates is a list that became wrong at some unknown point in the
past; `PayerRecord` therefore carries `effective_from` / `effective_to` and
`PayerTable.add()` refuses an inverted window outright.

## How much of this is a model?

Two calls, both schema-bound, both gated:

| Sub-task | Answered by | Human gate |
| --- | --- | --- |
| Build the 270 / parse the 271 | `x12.py` | — deterministic |
| Is this child covered for this visit? | `coverage.py` | anything but a clean yes goes to a person |
| Read the card photo | `cards.py`, `CARD_SCHEMA` | **`UnconfirmedCard` until a human confirms every required field** |
| What kind of denial is this? | `denials.py`, `DENIAL_SCHEMA` | classification is triage, never an answer to the payer |
| Draft the appeal | `draft_appeal()` | drafted for a person to send, never sent |

`make lint` asserts no model import in `x12.py` or `coverage.py`.

## Files

| File | Responsibility |
| --- | --- |
| `x12.py` | `build_270()`, `X12Parser`, `SubsetParser`, `BenefitLine`, `Response271`. 005010X279A1. |
| `coverage.py` | `determine()` → `Determination`; `PayerTable`; `outreach_draft()`. |
| `cards.py` | `CardReader` over a card photograph. Every field carries its own confidence. |
| `denials.py` | CARC/RARC root-cause mapping, `DenialClassifier`, `draft_appeal()`, `build_denial_report()`. |
| `fixtures.py` | Synthetic payers, 271 responses (including the ugly ones) and denials. |
| `demo.py` | `make demo-i09`. |

## The five things that are easy to get wrong

**1. The EDI subset parser must fail loudly.** `SubsetParser` exists because
real payers send segments outside the implementation guide. Anything it does not
understand goes to `unparsed` and surfaces; it never silently drops a segment.
EB03 repeats use the `^` repetition separator, and the DTP chain ends in an
`else` that appends rather than falls through. An MSG segment attaches to the
benefit line it follows, and `determine()` reads both `response.messages` and
per-benefit messages — a payer note saying "well-child limited to one per twelve
months" is the whole answer, and it does not arrive in an EB segment.

**2. A missing network indicator does not mean in-network.** `_network_indicator()`
normalises `Y`/`N`/`IN`/`OUT` and returns `None` when the payer said nothing.
`Determination.in_network` is then `None`, not `True`. An unstated field is not
a favourable field. `_amount()` and `_percent()` both prefer the in-network line
when one exists and neither invents one when it does not.

**3. `plan_ends < on` is blocking.** Terminated coverage is `INACTIVE`, not a
warning on an otherwise-active determination. `patient_safe` requires
`not self.warnings` — a determination with anything unresolved on it is not
something to read out to a parent at the front desk.

**4. There is no template for communicating a denial.** `_OUTREACH_TEMPLATES`
contains exactly one key, `Outcome.ACTIVE`, and `make lint` asserts it.
`outreach_draft()` raises `PatientCommunicationRefused` for anything else. A
system that can generate "your insurance has denied this" will eventually
generate it wrongly, at scale, to a parent, about a child. Every not-a-clean-yes
outcome goes to the front-desk queue and a person makes the call.

**5. Evidence for a denial classification must be grounded in the denial text.**
`_grounded()` is a **token-subsequence** test, not a substring test: a model can
assemble a plausible quotation out of scattered words, and a substring check
catches only the laziest version of that. `MIN_EVIDENCE_CHARS = 12` and
`MIN_EVIDENCE_WORDS = 2` stop `"the"` from qualifying as a citation.

## Commands

```bash
make test-eligibility  # 67 tests
make demo-i09          # a batch verification, an ugly 271, a denial worklist
make lint              # includes: no template for communicating a denial
python3 -m modules.eligibility.demo
```

## Wiring

```python
from modules.eligibility import PayerTable, X12Parser, build_270, determine

edi = build_270(request)                       # send to the clearinghouse
response = X12Parser().parse(raw_271)          # or SubsetParser for a partner
result = determine(
    response, payer_table=payers, on=visit_date, service_type="98",
)

if result.outcome == "active" and result.patient_safe:
    front_desk.show(result.copay_amount, result.deductible_remaining)
else:
    exception_queue.add(result)                # a person, always
```

The clearinghouse is the integration seam: this module produces and consumes
X12, and knows nothing about how it travels.
