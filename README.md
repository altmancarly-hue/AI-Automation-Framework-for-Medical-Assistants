# AI & Automation Program Guide [Medical Assistant Role]

**A workflow-by-workflow implementation plan for automating the Medical Assistant role in a small independent pediatric practice.**

Author: Carly Altman
Version: 1.0
Date: 2026-08-21
Audience: Practice Owner / Office Manager / Physician Partners
Classification: Public — Contains no PHI

---

## Table of Contents

- [0. How to Read This Document](#0-how-to-read-this-document)
- [1. Executive Summary](#1-executive-summary)
- [2. Baseline Operating Model & Cost Basis](#2-baseline-operating-model--cost-basis)
- [3. Compliance & Architecture Ground Rules](#3-compliance--architecture-ground-rules)
- [4. Reference Technical Architecture](#4-reference-technical-architecture)
- [5. The Ten Initiatives](#5-the-ten-initiatives)
  - [I-01 — Automated Forms Pipeline](#i-01--automated-forms-pipeline-school--daycare--sports-physicals)
  - [I-02 — Immunization Gap Closure & Recall Engine](#i-02--immunization-gap-closure--recall-engine)
  - [I-03 — Pre-Visit Chart Preparation Agent](#i-03--pre-visit-chart-preparation-agent)
  - [I-04 — Telephone Triage Documentation Assistant](#i-04--telephone-triage-documentation-assistant)
  - [I-05 — Ambient Documentation for Rooming & Encounters](#i-05--ambient-documentation-for-rooming--encounters)
  - [I-06 — Inbound Fax & Document Ingestion](#i-06--inbound-fax--document-ingestion)
  - [I-07 — No-Show Reduction & Waitlist Backfill](#i-07--no-show-reduction--waitlist-backfill)
  - [I-08 — Vaccine Cold Chain Telemetry](#i-08--vaccine-cold-chain-telemetry)
  - [I-09 — Eligibility Verification & Denial Prevention](#i-09--eligibility-verification--denial-prevention)
  - [I-10 — Standing Order Digitization & Delegation Audit](#i-10--standing-order-digitization--delegation-audit)
- [6. Consolidated ROI Model](#6-consolidated-roi-model)
- [7. Phased Implementation Plan](#7-phased-implementation-plan)
- [8. Vendor & Cost Reference Table](#8-vendor--cost-reference-table)
- [9. Governance, Policy & Risk Register](#9-governance-policy--risk-register)
- [10. Measurement Plan & KPIs](#10-measurement-plan--kpis)
- [Appendix A — Sample System Prompts](#appendix-a--sample-system-prompts)
- [Appendix B — Sample Orchestration Workflow](#appendix-b--sample-orchestration-workflow)
- [Appendix C — Vendor Due Diligence Checklist](#appendix-c--vendor-due-diligence-checklist)
- [Appendix D — Glossary](#appendix-d--glossary)

---

## 0. How to Read This Document

Every initiative in Section 5 follows an identical eleven-part template so the
sections are directly comparable:

| Field | Meaning |
| --- | --- |
| **Current State** | The manual process as performed today, step by step |
| **Failure Modes** | What actually goes wrong, and what it costs |
| **Does This Need an LLM?** | Honest classification — deterministic, ML, LLM, or agentic |
| **Target State** | The automated workflow |
| **Reference Implementation** | Concrete technical build, component by component |
| **Buy vs. Build** | Named products with current pricing, or custom build scope |
| **Complexity** | L1 / L2 / L3 (defined below) |
| **Time to Value** | Calendar time from kickoff to production |
| **Quantified Benefit** | Explicit arithmetic, with all assumptions exposed |
| **Risks & Controls** | What breaks, what it endangers, how it is contained |
| **Proposal** | The argument for funding it |

### Complexity Scale

| Level | Definition | Skills Required | Typical Duration |
| --- | --- | --- | --- |
| **L1** | Buy, configure, train staff. No code. | Office manager + vendor onboarding | 1–10 days |
| **L2** | Integrate two or more existing systems. Low-code plus some scripting. | One technical resource part-time | 2–8 weeks |
| **L3** | Custom application development against EHR APIs, with a data layer and audit trail. | Dedicated engineer, security review | 8–20 weeks |

### A Note on the Numbers

Every dollar figure below is **modeled, not measured.** The model is transparent
— all inputs are listed in Section 2 and each calculation is shown in full so
any assumption can be challenged and re-run. Section 10 defines how to replace
each modeled input with a measured one during the first 60 days.

Treat the ROI figures as a **hypothesis to be tested**, not a promise. A
proposal that presents modeled savings as certainty is the fastest way to lose
credibility when the first invoice arrives.

---

## 1. Executive Summary

### The Situation

The practice's operating stack is, by observable evidence, substantially manual:

- Scheduling is telephone-only; the public instruction is to call between 9am and 5pm.
- Intake forms are distributed as downloadable PDFs to be printed and hand-carried.
- Two active inbound fax lines are published as primary document channels.
- After-hours coverage runs through a human answering service.
- No patient portal is surfaced on the public site.
- The published payer list is dated **January 2016**.
- A $25 no-show fee with a 10:00am cancellation cutoff is enforced — a policy that
  exists precisely because no-shows are an unsolved operational problem.

Each of those is a signal of manual labor absorbing clinical staff time. In a
practice this size, the Medical Assistant is the shock absorber for all of it.

### The Thesis

**The highest-value AI application in this practice is not clinical. It is
administrative.** Nothing in this plan asks an AI system to make a clinical
decision, and nothing in it could. Illinois law forecloses that anyway: under
225 ILCS 60/54.2 an MA's clinical authority is delegated from a physician who
must be physically on premises, and no software changes that. What the plan does
is remove the transcription, retyping, sorting, chasing, and logging that
currently consumes a majority of non-patient-facing MA time.

The design principle throughout is **draft-and-review**: the machine produces a
draft, a licensed human reviews and signs it, and the audit trail records both.
Nothing is auto-released to a patient chart or to a patient without a human
approval event.

### The Numbers

| Metric | Value |
| --- | --- |
| Modeled gross annual benefit (full run rate) | **$325,512** |
| — of which is labor recaptured | $176,000 |
| — of which is revenue captured or protected | $143,700 |
| — of which is loss avoidance | $5,812 |
| Recurring annual software cost (all ten) | $36,901 |
| One-time implementation cost (consolidated, contracted) | $52,000 – $68,000 |
| **Year 1 net (ramp-adjusted — the honest number)** | **$135,848** |
| Modeled steady-state annual net (Year 2+) | **$288,611** |
| Cumulative break-even | **Month 5** |
| Downside case (all pessimistic assumptions) | still ~$149,000 net |

> The Year 1 figure is ramp-adjusted because most initiatives are live for only
> part of the year. Section 6.3 shows the quarterly build-up. Quoting the
> $325,512 run rate as a Year 1 number is how programs like this lose credibility
> in month four.

### Quick Reference — All Ten Initiatives

| # | Initiative | AI Class | Complexity | Monthly Cost | Modeled Annual Benefit | Priority |
| --- | --- | --- | --- | --- | --- | --- |
| I-01 | Automated Forms Pipeline | LLM + OCR | L3 | $180–$400 | $31,082 | **3** |
| I-02 | Immunization Gap Closure & Recall | Deterministic + LLM drafting | L2 | $150–$350 | $36,987 | **2** |
| I-03 | Pre-Visit Chart Prep Agent | LLM summarization | L2–L3 | $120–$300 | $25,163 | 6 |
| I-04 | Telephone Triage Documentation | LLM structured extraction | L2 | $200–$450 | $46,299 | **4** |
| I-05 | Ambient Documentation | Ambient ASR + LLM (buy) | L1 | $790–$1,190 | $33,551 | 5 |
| I-06 | Inbound Fax & Document Ingestion | OCR + LLM classification | L2 | $150–$400 | $10,889 | 8 |
| I-07 | No-Show Reduction & Waitlist Backfill | Deterministic + optional ML | L1 | $250–$500 | $96,000 | **1** |
| I-08 | Cold Chain Telemetry | No AI — IoT + rules | L1 | $50–$100 | $6,777 | 6 |
| I-09 | Eligibility Verification & Denials | Deterministic + LLM triage | L2 | $200–$450 | $22,000 | 7 |
| I-10 | Standing Order Digitization | Deterministic + LLM authoring aid | L2 | $50–$150 | $16,764 | 9 |

> **Priority column** reflects the recommended sequencing in Section 7, which
> weighs benefit against effort, risk, and dependency — not benefit alone.

### The Single Most Important Point in This Document

**Six of these ten initiatives require no large language model at all, or use
one only for drafting text a human will read and edit.** The temptation in 2026
is to reach for an LLM for every problem. Most of the value here is in plumbing:
connecting systems that do not currently talk to each other, and replacing
paper with structured data. The LLM is a component, not the strategy.

Anyone selling this practice an "AI transformation" that leads with a chatbot
is selling the wrong thing.

---

## 2. Baseline Operating Model & Cost Basis

Every calculation in this document derives from the following inputs. They are
stated once here and referenced throughout. Where an input is an estimate rather
than an observation, it is marked **[EST]** and Section 10 specifies how to
measure it.

### 2.1 Practice Volume Assumptions

| Input | Value | Basis |
| --- | --- | --- |
| Active patient panel, Buffalo Grove site | 9,000 | **[EST]** — derived from provider count and typical pediatric panel size |
| Providers seeing patients per clinic day, BG | 4–5 | **[EST]** |
| Patient encounters per clinic day, BG | 100 | **[EST]** |
| Clinic days per year | 305 | Observed: open M–F plus Saturday and Sunday/holiday sessions |
| **Annual encounters, BG site** | **30,500** | 100 × 305 |
| Well-visit share of encounters | 40% | **[EST]** — typical pediatric primary care mix |
| Clinical (non-scheduling) phone calls per day | 120 | **[EST]** — practice advertises free unlimited phone access, which inflates volume |
| Inbound fax documents per day | 45 | **[EST]** |
| Baseline no-show rate | 9% | **[EST]** — typical pediatric range is 8–12%; a $25 fee implies a real problem |

### 2.2 Labor Cost Basis

Fully loaded cost = base wage × 1.28 (employer payroll taxes, health insurance,
retirement match, PTO accrual). The practice publicly advertises health insurance
and retirement investing as benefits, so 1.28 is a conservative multiplier.

| Role | Base | Fully Loaded Hourly | Fully Loaded Per Minute |
| --- | --- | --- | --- |
| Medical Assistant / LPN | $26.00/hr (midpoint of posted $49,920–$58,236) | **$33.28** | **$0.5547** |
| Front Desk / Administrative | $22.00/hr **[EST]** | **$28.16** | **$0.4693** |
| Physician | $115.00/hr **[EST]** (~$240k pediatrics) | **$147.20** | **$2.4533** |

**Rounded working rates used throughout: MA = $33/hr, Front Desk = $28/hr,
Physician = $147/hr.**

### 2.3 Revenue Basis

| Input | Value | Basis |
| --- | --- | --- |
| Blended allowed amount per encounter | **$140** | **[EST]** — weighted: well visit ~$220, sick visit ~$115, commercial payer mix, no Medicaid |
| Marginal contribution per recovered encounter | $140 | Fixed costs already absorbed; a backfilled slot is near-pure contribution |

### 2.4 The Physician-Time Discount Rule

Recaptured **physician** time is **not** counted as a dollar saving unless it is
explicitly converted into additional patient encounters. A physician who finishes
charting 40 minutes earlier has a better evening; the practice has not earned a
dollar.

Where physician time appears in a calculation it is either:

1. **Excluded** from the headline figure and listed separately as "capacity upside," or
2. **Discounted by 50%** and labeled as such.

This is the single most commonly abused number in healthcare AI ROI decks. It is
handled conservatively here on purpose, because the credibility of the entire
proposal depends on the reader trusting that the numbers were not inflated.

### 2.5 What Is Deliberately Excluded

The following real benefits are **not** monetized anywhere in this document,
because they cannot be defended with arithmetic:

- Reduced staff turnover and recruiting cost from lower administrative burden
- Patient satisfaction and retention effects
- Malpractice risk reduction from improved documentation completeness
- Reduced overtime from earlier close-out
- Competitive positioning against hospital-owned pediatric groups
- Physician recruitment appeal

They are real. They are also unfalsifiable. They belong in the narrative, not
the model.

---

## 3. Compliance & Architecture Ground Rules

These rules constrain every initiative below. They are not optional and they are
not negotiable, because violating any of them converts an efficiency project into
a reportable breach.

### 3.1 Rule 1 — No PHI Touches a Consumer AI Product. Ever.

This is the most common and most expensive mistake in small-practice AI adoption,
and it is worth being precise about which products are and are not eligible.

| Product | BAA Available? | PHI Permitted? |
| --- | --- | --- |
| ChatGPT Free / Plus / Team / Business | No | **NO** |
| ChatGPT Enterprise / Edu / ChatGPT for Healthcare | Yes (sales-managed) | Yes, under signed BAA |
| OpenAI API | Yes (Privacy Addendum, on request) | Yes, with ZDR configured |
| Claude.ai Free / Pro / Max | No | **NO** |
| Claude API (api.anthropic.com) | Yes (HIPAA readiness, eligible features only) | Yes, for listed eligible features |
| Claude via Amazon Bedrock | Yes (AWS BAA covers it) | Yes |
| Gemini consumer / Google AI Studio | No | **NO** |
| Gemini via Google Cloud Vertex AI | Yes (Google Cloud BAA) | Yes |
| Azure OpenAI Service | Yes (Microsoft Online Services DPA/BAA) | Yes |
| Microsoft 365 Copilot | Yes, under enterprise agreement | Yes — but see caveat below |
| Self-hosted open-weight models (Llama, Mistral, Qwen) | N/A — no third party involved | Yes, inside your own boundary |

Three specific traps worth naming:

1. **"Business" tier is not "Enterprise" tier.** ChatGPT Business sounds like the
   work-appropriate plan and is not BAA-eligible. This naming trap catches
   otherwise careful organizations.
2. **A signed BAA does not cover the whole product surface.** Anthropic's HIPAA
   readiness covers a defined subset of Messages API features; beta features are
   generally excluded unless explicitly listed. OpenAI's coverage attaches to
   zero-data-retention-eligible endpoints on an approved organization. Enabling
   a non-covered feature silently moves a workload outside coverage.
3. **Zero Data Retention must be configured, not assumed.** On OpenAI's standard
   API, ZDR is a per-configuration setting. A signed BAA plus a non-ZDR call is
   still a problem.

**Microsoft 365 Copilot caveat:** it is BAA-covered under a Microsoft enterprise
agreement, and it is genuinely useful for drafting policy documents, staff
training material, and internal memos. It is *not* a clinical workflow engine and
should not be positioned as one. It has no structured access to the EHR, no audit
trail suitable for a delegation record, and no ability to write back to a chart.
Do not let a Microsoft reseller convince the practice that a Copilot license is
the AI strategy.

### 3.2 Rule 2 — Recommended Model Access Path

For a practice of this size with no in-house IT department, the recommended path
is **Amazon Bedrock under the AWS Business Associate Addendum.**

Rationale:

- The AWS BAA is self-serve, accepted through AWS Artifact. No sales cycle, no
  4-to-12-week legal negotiation.
- One BAA covers the model layer (Claude, Llama, Mistral), the storage layer (S3),
  the OCR layer (Textract), the de-identification layer (Comprehend Medical), and
  the database layer (RDS). One counterparty instead of six.
- Model swapping does not require a new contract. If a cheaper or better model
  ships, you change a string in a config file, not a legal agreement.
- Consumption pricing. No per-seat minimums, which matters enormously at this scale.

**The one verification step that is mandatory:** confirm the specific model and
region you invoke appears on AWS's published HIPAA-eligible services list before
routing PHI. Not every model on Bedrock is eligible. Pin the allowed model list
in code, enforce it in CI, and do not leave it to a wiki page nobody reads.

### 3.3 Rule 3 — Minimum Necessary, Enforced Architecturally

HIPAA's minimum-necessary standard should be enforced by the code path, not by
policy documents.

- **De-identify before inference wherever the task permits it.** Classification,
  routing, and summarization frequently do not need the patient's name. Strip
  identifiers, process, and re-hydrate on the way out using a tokenization map
  held locally.
- **Embeddings derived from PHI are PHI.** A vector database built on patient
  notes needs the same BAA, encryption, and access controls as the primary
  database. This surprises people.
- **Log the minimum too.** Do not write full prompts and completions into an
  observability platform that lacks a BAA. Log a prompt-template hash and an
  output hash, not the payload.

### 3.4 Rule 4 — Human in the Loop Is a Legal Requirement Here, Not a Preference

Under 225 ILCS 60/54.2, an unlicensed MA's clinical tasks are delegated by a
physician, must fall within that physician's own scope, and require a licensed
health care professional on site. Automation does not delegate anything to
itself.

Practical translation:

| Output Type | Release Mechanism |
| --- | --- |
| Clinical note draft | Licensed human reviews and signs. Always. |
| Form pre-fill | MA reviews, physician signs. Always. |
| Patient-facing clinical message | Licensed human approves before send. Always. |
| Appointment reminder (no clinical content) | May auto-send |
| Recall notice ("your child is due for a visit") | May auto-send with pre-approved template |
| Internal work queue routing | May auto-execute |
| Any order, prescription, or result interpretation | Never automated. Not partially. |

**The additional consideration on timing:** Section 54.2 is currently scheduled
to sunset on January 1, 2027, with pending legislation that could restructure
medical assistant regulation in Illinois. Any system built now that hard-codes
delegation logic should externalize it into configuration — see I-10 — so a
statutory change becomes a config edit rather than a rebuild.

### 3.5 Rule 5 — Fail Closed

Every automated workflow must have a defined failure behavior, and the default
must be to stop and queue for a human, never to guess.

- LLM returns low confidence → route to human queue, do not release
- Required field cannot be extracted → route to human queue, do not release
- API timeout → retry with backoff, then route to human queue
- Model returns a value outside an expected range → hard-block, alert

A pediatric practice's tolerance for a silently wrong immunization date is zero.
Design for the failure case first.

---

## 4. Reference Technical Architecture

A single architecture supports all ten initiatives. Building them as ten
disconnected point solutions is how a small practice ends up with ten vendor
invoices and no integration.

### 4.1 Layer Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│  PRESENTATION                                                        │
│  Internal staff web app (React) · Existing EHR UI · SMS/voice        │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│  ORCHESTRATION                                                       │
│  n8n (self-hosted) or Windmill · Cron scheduler · Job queue          │
│  Human-approval gates · Retry & dead-letter handling                 │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│  AI SERVICES  (all AWS, all under one BAA)                           │
│  Bedrock (Claude) · Textract (OCR) · Comprehend Medical (PHI detect) │
│  Transcribe Medical (ASR) · HealthScribe (ambient, $0.10/audio min)  │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│  INTEGRATION                                                         │
│  Mirth Connect (HL7v2, open source) · FHIR R4 client (EHR APIs)      │
│  I-CARE HL7 2.5.1 interface · Clearinghouse X12 270/271              │
│  Fax gateway API · Twilio (SMS/voice, BAA available)                 │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│  DATA & AUDIT                                                        │
│  PostgreSQL (RDS, encrypted, KMS CMK) · S3 (SSE-KMS, versioned)      │
│  Append-only audit log · Tokenization vault for de-ID round-trip     │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 Component Selection Rationale

**Orchestration — n8n, self-hosted.** Open source, self-hostable inside your own
VPC so no workflow data leaves your boundary and no orchestration vendor BAA is
needed. Visual workflow builder means the office manager can read and audit what
a workflow does without reading code — which matters more than it sounds when
the person accountable for HIPAA compliance is not an engineer. Zapier and Make
are easier but route data through their infrastructure; a BAA becomes necessary
and the pricing at PHI-appropriate tiers erodes the advantage.

**Integration — Mirth Connect.** The de facto open-source healthcare integration
engine. Speaks HL7v2, FHIR, X12, and flat files. Free. Every healthcare
integration contractor in the Chicago market knows it, which matters for
maintainability after the initial build.

**Why not Redox or Health Gorilla?** Both are excellent and both are priced for
organizations an order of magnitude larger. At 30,000 annual encounters the
per-transaction economics do not work.

**The critical unknown: which EHR.** This is the single largest scoping variable
in the entire plan and must be resolved before any build begins.

| If the EHR is… | Integration Difficulty | Notes |
| --- | --- | --- |
| eClinicalWorks | Moderate | FHIR APIs available; Sunoh.ai is the native ambient scribe; healow provides portal/check-in/kiosk. Much of I-02, I-05, I-07 may be purchasable as eCW modules. |
| athenahealth | Easy | Strong public API, marketplace ecosystem, native ambient scribe shipped February 2026 at no additional cost |
| Office Practicum / PCC | Moderate | Pediatric-specific, good immunization tooling already, smaller integration ecosystem |
| NextGen / Greenway | Moderate | APIs exist, quality varies by version |
| Anything on-premise and unsupported | Hard | May require HL7 interface engine and screen-scraping fallback; re-scope everything at L3 |

**Action item before any spend: determine the EHR, the version, whether FHIR
R4 APIs are licensed and enabled, and what the vendor charges for API access.**
Some vendors charge four figures monthly for API enablement, which changes the
build-vs-buy calculus for several initiatives.

---

## 5. The Ten Initiatives

---

### I-01 — Automated Forms Pipeline (School / Daycare / Sports Physicals)

> **The highest-leverage automation available to a pediatric practice, and the
> one almost nobody builds.**

#### Current State

Illinois requires a completed **Certificate of Child Health Examination** for
school entry, plus separate forms for daycare, sports participation, camp,
and college. In a pediatric practice, forms are a continuous background load
that never appears on a schedule and therefore never gets staffed for.

The manual process, step by step:

1. Parent hands a blank form to the front desk, or emails it to `bg@nsubpeds.com`,
   or the form is generated at a well visit.
2. Front desk logs it — sometimes on paper, sometimes in a shared inbox.
3. MA pulls the chart and locates: date of last physical, height, weight, BMI,
   blood pressure, vision screening result, hearing screening result, allergies,
   current medications, chronic conditions, and the complete immunization history
   with dates for every antigen.
4. MA transcribes each field by hand onto the form. The immunization grid alone
   is typically 25–40 individual date entries.
5. MA cross-checks against I-CARE, since the registry may hold doses given at a
   pharmacy, an urgent care, or a prior practice that never made it into the chart.
6. Form goes into a physician's signature pile.
7. Physician reviews and signs, possibly days later.
8. Form is scanned into the chart, and returned by fax, mail, or parent pickup.
9. Parent calls to ask where the form is. Front desk searches. Repeat.

**Observed time: 9 minutes of combined MA and front-desk time per form**, plus
1 minute of physician time, plus an unmeasured tail of "where is my form" calls.

#### Failure Modes

| Failure | Consequence |
| --- | --- |
| Transcription error in an immunization date | School rejects the form; entire cycle repeats; parent is angry |
| Missed dose given elsewhere and not reconciled | Child receives an unnecessary duplicate vaccine, or is wrongly flagged non-compliant |
| Form lost in the signature pile | Child cannot start school or a sports season |
| Seasonal surge (July–August, and again at spring sports) | Forms crowd out clinical work at the worst possible time of year |
| No tracking system | Nobody can answer "where is my form" without a physical search |

#### Does This Need an LLM?

**Partially — and less than you would expect.**

| Sub-task | Right Tool | Why |
| --- | --- | --- |
| Read a scanned/faxed blank form and identify its type | OCR + classifier | Deterministic once you have a template library. LLM helps only for unseen form types. |
| Locate the fillable field coordinates on a known form | Template map (built once per form) | Pure deterministic. Do not use an LLM for this — it is slower, costlier, and less reliable than a coordinate map. |
| Pull structured clinical data from the chart | FHIR API query | Deterministic. The data is already structured. |
| Reconcile chart immunizations against I-CARE | Deterministic matching + LLM adjudication for ambiguous cases | Vaccine matching is mostly rule-based (CVX code + date within tolerance). The LLM earns its keep only on genuinely ambiguous records — a dose recorded as "DTP" in 2019 versus "DTaP," historical records with partial dates, or brand-name-only entries. |
| Handle a novel form the system has never seen | LLM vision model | This is where an LLM is genuinely superior. A camp form from a new organization has an unpredictable layout; an LLM can identify semantic fields ("this box wants tetanus date") without a pre-built template. |
| Draft the free-text sections | LLM | "Any restrictions on physical activity?" and similar narrative fields |

**Verdict:** roughly 70% deterministic plumbing, 30% LLM. Build the deterministic
core first. A team that starts with the LLM will build something impressive in
a demo and unreliable in production.

#### Target State

1. Form arrives by any channel — scan, fax, email, or portal upload — and lands
   in a single ingestion queue.
2. Classifier identifies form type. Known type → template map. Unknown type →
   LLM vision extraction → new template proposed for human confirmation and
   permanent addition to the library.
3. Patient is matched by name plus date of birth, with a confirmation step. **Never
   auto-match on a fuzzy name alone.** Pediatric panels are full of siblings with
   similar names and twins with identical birthdates.
4. System queries the EHR via FHIR for vitals, screenings, allergies, medications,
   and problem list; and queries I-CARE for the immunization record.
5. Reconciliation engine merges the two immunization sources, flagging every
   discrepancy rather than silently picking a winner.
6. Form is rendered as a filled PDF. **Every auto-filled field is visually
   highlighted** so the reviewing MA sees exactly what the machine wrote.
7. MA reviews in a single screen: filled form on the left, source data with
   provenance on the right. Discrepancies surface at the top.
8. MA approves → routes to physician signature queue with e-signature.
9. Signed form is auto-filed to the chart, auto-returned to the parent by their
   preferred channel, and the tracking record closes.
10. Parent receives a status notification at each stage, eliminating the "where
    is my form" call entirely.

#### Reference Implementation

```
Ingestion:      S3 bucket (SSE-KMS) fed by fax gateway API, scanner hot folder,
                monitored mailbox, portal upload
Classification: AWS Textract AnalyzeDocument → form-type classifier
                (start with simple keyword/layout rules; escalate to Bedrock
                Claude vision only on no-match)
Template store: PostgreSQL — form_type, field_name, page, x, y, w, h,
                source_fhir_path, transform_fn
Chart data:     FHIR R4 — Patient, Observation (vitals/screenings),
                AllergyIntolerance, MedicationStatement, Condition, Immunization
Registry data:  I-CARE query (HL7 2.5.1 QBP^Q11 / RSP^K11, or manual export
                fallback if HL7 interface is not licensed)
Reconciliation: Deterministic CVX-code matcher with ±4-day date tolerance;
                unmatched or conflicting pairs → Bedrock Claude for adjudication
                with a strict JSON output schema
Rendering:      pdf-lib or PyMuPDF, field overlay with highlight annotations
Review UI:      React SPA, split-pane, provenance sidebar
Signature:      DocuSign ($40/mo) or self-hosted e-signature with audit binding
Delivery:       Fax gateway, secure email, or SMS link
Audit:          Append-only table — every field write records source, method
                (auto/manual), confidence, reviewing user, timestamp
```

**The immunization reconciliation prompt** deserves specific attention because it
is the one place an LLM touches clinically consequential data. It must be
constrained hard:

- Structured JSON output only, with an enforced schema
- The model may return `MATCH`, `NO_MATCH`, or `UNCERTAIN` — and `UNCERTAIN`
  routes to a human, which is the point
- The model is explicitly instructed that it may not infer a date that does not
  appear in either source
- Temperature 0
- Every adjudication is logged with the full input pair for later review

#### Buy vs. Build

| Option | Product | Cost | Assessment |
| --- | --- | --- | --- |
| **Buy — partial** | Practice's EHR native forms module | Often included | Check first. I-CARE itself already offers automatic population and printing of the Illinois school physical form and immunization history report. **If the practice is an I-CARE participating site and is not using this, that alone is worth a phone call before spending a dollar on software.** |
| **Buy — partial** | Phreesia, Clearwave, Intakeq | $300–$900/mo | These handle *patient-entered* intake well. They do not solve the *practice-completed clinical form* problem, which is the actual bottleneck. Do not confuse the two. |
| **Buy — adjacent** | Vero, Freed Premier | $69–$119/provider/mo | Vero advertises PDF form auto-fill. Worth evaluating as a stopgap but it works from the encounter, not from a chart-wide data pull. |
| **Build** | Custom, per architecture above | ~$180–$400/mo runtime (Textract + Bedrock + hosting + e-signature) | The only option that actually closes the loop including I-CARE reconciliation and parent notification. |

**Runtime cost math for the build path:**

```
Textract AnalyzeDocument:  6,500 forms/yr × ~2 pages × $0.05      = $650/yr
Bedrock (Claude Haiku for classification, Sonnet for adjudication):
  ~6,500 × ~8k tokens avg blended                                 = $420/yr
AWS hosting (small RDS + ECS/Fargate + S3)                        = $1,800/yr
E-signature (DocuSign Standard)                                   = $480/yr
Fax gateway API (SRFax / Documo)                                  = $360/yr
                                                          Total ≈ $3,710/yr ≈ $310/mo
```

**One-time build:** 10–14 weeks of engineering. At contract rates, $18,000–$28,000.
Materially less if built internally.

#### Complexity

**L3.** This is the hardest build in the set. It touches OCR, an EHR API, a state
registry interface, PDF manipulation, e-signature, and a review UI. It is also
the one with the highest ceiling.

#### Time to Value

- Weeks 1–2: EHR API access confirmed, I-CARE interface scoped, top 5 form types inventoried
- Weeks 3–6: ingestion, classification, template store, FHIR pull
- Weeks 7–9: reconciliation engine and review UI
- Weeks 10–12: signature, delivery, notification, audit
- Weeks 13–14: parallel run against manual process, accuracy validation
- **Production: week 14.** Meaningful savings from week 8 if piloted on the single
  highest-volume form (the Illinois Certificate of Child Health Examination).

#### Quantified Benefit

```
ASSUMPTIONS
  Active panel, BG site                                        9,000  [EST]
  Share requiring ≥1 form annually                               60%  [EST]
  Base form events                                             5,400
  Duplicate/re-request/correction multiplier                    1.20  [EST]
  Total annual form events                                     6,500

  Current handling time (MA + front desk, combined)          9.0 min  [EST]
  Target handling time (review + release)                    2.5 min
  Time saved per form                                        6.5 min

  Current physician signature time                           1.0 min
  Target physician signature time (batched, pre-validated)   0.4 min
  Physician time saved per form                              0.6 min

STAFF SAVINGS
  6,500 forms × 6.5 min          =  42,250 min  =  704.2 hrs
  704.2 hrs × $33/hr             =  $23,239

PHYSICIAN SAVINGS (50% discount per §2.4)
  6,500 forms × 0.6 min          =   3,900 min  =   65.0 hrs
  65.0 hrs × $147/hr × 0.50      =   $4,778

REWORK AVOIDANCE
  Forms rejected/returned for error, current             4%  [EST]  = 260/yr
  Rework cost per rejected form (full re-cycle)     15 min = $8.25
  Error rate after automation                            1%          =  65/yr
  Avoided rework: 195 × $8.25                        =     $1,609

"WHERE IS MY FORM" CALL ELIMINATION
  Status calls per week                                   12  [EST]
  Minutes per call (incl. search)                        5.0
  12 × 52 × 5 min = 3,120 min = 52.0 hrs × $28       =     $1,456

─────────────────────────────────────────────────────────────────
  TOTAL MODELED ANNUAL BENEFIT                            $31,082
  Annual runtime cost                                     ($3,710)
  NET ANNUAL BENEFIT                                      $27,372
  One-time build (midpoint)                              ($23,000)
  YEAR 1 NET                                               $4,372
  YEAR 2+ NET                                             $27,372
─────────────────────────────────────────────────────────────────
```

**Sensitivity:** the model breaks even at roughly 2,900 forms/year even in year
one. If the practice handles more than 3,000 forms annually — which at a
9,000-patient pediatric panel it certainly does — the investment is sound.

#### Risks & Controls

| Risk | Severity | Control |
| --- | --- | --- |
| Wrong immunization date written to a legal school document | **High** | Every auto-filled field highlighted; discrepancies surfaced explicitly; MA sign-off required; physician signature required; full field-level audit trail |
| Wrong patient matched | **High** | Two-factor match (name + DOB); explicit human confirmation step; sibling/twin detection flag |
| I-CARE interface unavailable or not licensed | Medium | Degrade gracefully to chart-only fill, flag the form as "registry not reconciled," continue to function |
| Staff stop reviewing carefully once accuracy is good (automation complacency) | **High** | Randomly inject a synthetic discrepancy into 1 in 50 forms; track catch rate; retrain if it drops. This is the control most implementations omit and most need. |
| Vendor lock on the OCR layer | Low | Abstract behind an interface; Textract and Azure Document Intelligence are swappable |

#### Proposal

Forms are the clearest case in this entire document because the work is
**high-volume, low-judgment, and entirely derived from data the practice already
holds in structured form.** An MA transcribing 30 immunization dates from a chart
onto a state form is not exercising clinical skill; they are acting as a
copy-paste mechanism between two systems that could talk to each other directly.

The seasonal dynamic makes it worse than the annual average suggests. Form volume
spikes in July and August, exactly when the practice is also handling back-to-school
physicals — the busiest well-visit period of the year. Automating forms does not
just save 704 hours spread evenly; it removes a load that currently peaks at the
worst possible moment.

There is a compliance argument as well. A hand-transcribed immunization grid on a
state health document is a transcription-error surface with real consequences: a
missed dose flagged as given, or a given dose flagged as missed, both create
downstream problems that reach the school, the family, and potentially IDPH.
Machine-generated, human-verified, audit-logged form completion is a better
compliance posture than a rushed MA and a ballpoint pen.

**Recommendation: fund it, but sequence it third.** It has the highest build
cost and longest timeline of the ten. Bank the quick wins from I-07 and I-02
first to establish credibility and free up cash, then start this build in parallel.

---

### I-02 — Immunization Gap Closure & Recall Engine

#### Current State

The practice adheres to the AAP immunization schedule and states plainly that
families unable to remain current on school-required vaccines need to find
another provider. That is an unusually firm stance and it means immunization
compliance is not merely a quality metric here — it is a stated condition of the
patient relationship. It is therefore worth operationalizing properly.

Today, gap identification is almost certainly **opportunistic**: a child comes in,
the MA or physician looks at the immunization record, notices something is due,
and offers it. Children who do not come in are not identified. There is no
systematic outbound process.

I-CARE does provide a remind/recall feature and forecasts due dates against the
recommended childhood schedule. If the practice is enrolled and not using it,
that is free capability sitting unused.

#### Failure Modes

| Failure | Consequence |
| --- | --- |
| Child with a gap never comes in | Gap persists indefinitely; discovered at a school deadline crisis |
| Chart and I-CARE disagree | Duplicate dose administered, or a real gap masked by a phantom record |
| Adolescent gaps (HPV, MenACWY, MenB, Tdap) | The most-missed category in all of pediatrics; adolescents have the fewest routine visits |
| Manual recall attempted | Enormously labor-intensive, so it happens once a year at best, if at all |
| Missed catch-up scheduling for a transferred-in patient | Complex catch-up schedules are error-prone by hand |

#### Does This Need an LLM?

**Almost entirely no — and this is important.**

Immunization forecasting is a **solved deterministic problem.** The CDC publishes
the immunization schedule as machine-readable logic, and the Clinical Decision
Support for Immunization (CDSi) specification exists precisely so that software
can evaluate a patient's dose history against the schedule without guessing.
I-CARE already forecasts due dates using this logic.

**Using an LLM to determine whether a child is due for a vaccine would be
actively negligent.** It is a rules problem with an authoritative published
rule set, and the correct answer is a rules engine.

Where the LLM does belong:

| Sub-task | Tool |
| --- | --- |
| Evaluate dose history against schedule | **CDSi rules engine — never an LLM** |
| Reconcile chart against I-CARE | Deterministic CVX matcher, LLM only for ambiguous adjudication (as in I-01) |
| Prioritize the outreach list | Simple scoring, or optionally a small ML model on historical response |
| **Draft the outreach message** | **LLM — good fit.** Age-appropriate, reading-level-appropriate, warm rather than bureaucratic, in the family's preferred language |
| Draft the physician-facing daily huddle summary | LLM — good fit |
| Handle inbound replies ("we got it at CVS") | LLM classification → route to MA for record update |

#### Target State

**Nightly batch:**
1. Pull full active panel from EHR.
2. Pull corresponding I-CARE records.
3. Reconcile; flag discrepancies to an MA work queue.
4. Run CDSi forecast against merged record.
5. Produce two outputs.

**Output A — Tomorrow's Huddle Sheet.** For every patient on tomorrow's schedule:
overdue antigens, due-today antigens, screenings due by age, and any reconciliation
discrepancy. Delivered to each provider and MA before clinic opens. This converts
gap closure into an opportunistic-but-systematic process — the child is already
in the building.

**Output B — Outbound Recall Queue.** Patients with gaps who have no upcoming
appointment, ranked by urgency (school-deadline proximity, age-out risk for
antigens like HPV and rotavirus, outbreak-relevant antigens like MMR).

**Outreach cadence:**
- Day 0: SMS with pre-approved, physician-signed-off template and a scheduling link
- Day 7: second SMS, different framing
- Day 21: email
- Day 45: added to a call list for a human — because at that point the reason is
  probably not "they forgot"

**Reply handling:** inbound replies are classified. "Already got it elsewhere"
routes to MA for record reconciliation. "Have questions" routes to a nurse line.
"Stop" honors opt-out immediately and permanently.

#### Reference Implementation

```
Panel extract:    FHIR Patient + Immunization bulk export, nightly
Registry:         I-CARE HL7 2.5.1 query, or scheduled export/import fallback
Forecast engine:  CDSi implementation. Options:
                    - I-CARE's own forecast (free, already available)
                    - Open-source CDSi engine (e.g. an ACIP rules implementation)
                    - Do NOT write your own schedule logic from scratch
Reconciliation:   Same CVX matcher as I-01 — build once, use twice
Prioritization:   Rule-based score:
                    urgency = (days_overdue × antigen_weight)
                            + school_deadline_proximity_bonus
                            + age_out_risk_bonus
Message drafting: Bedrock Claude, from a physician-approved template library.
                  The LLM personalizes tone and reading level; it does NOT
                  invent clinical content. Templates are pre-approved; the LLM
                  fills slots and adjusts register.
Delivery:         Twilio (BAA available) or the practice's patient-communication
                  platform if one is adopted under I-07
Reply handling:   Twilio inbound webhook → Bedrock classifier → work queue
Opt-out:          Hard-enforced suppression list, checked before every send
```

#### Buy vs. Build

| Option | Product | Cost | Assessment |
| --- | --- | --- | --- |
| **Free** | I-CARE remind/recall | $0 | **Start here.** The capability exists. It is less automated and does not reconcile against the chart, but it is free and available today. |
| **Buy** | EHR population-health module (e.g. eCW HEDIS dashboard) | Often bundled or ~$100–300/mo | Practices using this report meaningful compliance gains. Check whether the practice already owns it. |
| **Buy** | Luma Health, Artera, Solutionreach recall campaigns | $300–$800/mo | Good outreach mechanics, weak clinical gap logic. They will send messages to a list you give them; they will not build the list correctly. |
| **Build** | Custom per above | ~$150–$350/mo | Best gap logic, integrates I-CARE reconciliation, reuses I-01 components |

**Runtime cost math for the build path:**

```
Bedrock (message drafting + reply classification):
  ~9,000 drafts/yr × ~2k tokens (Haiku)                   =   $150/yr
Twilio SMS: ~14,000 messages/yr × $0.0079 + carrier fees  = $1,400/yr
Compute (shares I-01 infrastructure)                      =   $600/yr
                                                    Total ≈ $2,150/yr ≈ $180/mo
```

#### Complexity

**L2** — assuming the CDSi forecast is consumed from I-CARE or an existing library
rather than implemented from scratch. If you implement schedule logic yourself it
becomes L3 and you should not, because you will get it wrong.

#### Time to Value

- Week 1: Enable and configure I-CARE remind/recall. **Immediate partial value at zero cost.**
- Weeks 2–4: Nightly panel extract and reconciliation build
- Weeks 5–6: Huddle sheet generation and distribution
- Weeks 7–9: Outbound campaign, templates, physician approval, opt-out handling
- **Production: week 9.**

#### Quantified Benefit

```
ASSUMPTIONS
  Active panel                                                 9,000  [EST]
  Share with ≥1 open immunization gap at a given time             8%  [EST]
  Patients with gaps                                             720
  Share of those with no upcoming appointment                    65%  [EST] = 468
  Recall conversion to a completed visit                         35%  [EST]
  Additional visits generated                                    164

REVENUE FROM RECOVERED VISITS
  164 visits × $140                                        =  $22,960

IN-VISIT CAPTURE (huddle sheet effect)
  Gap-eligible patients already on schedule                      252
  Additional capture rate from systematic flagging               30%  [EST]
  Additional vaccine administrations                              76
  Admin fee + product margin per dose, blended        $45  [EST]
  76 × $45                                                 =   $3,420

STAFF TIME — MANUAL REGISTRY RECONCILIATION ELIMINATED
  Current ad-hoc reconciliation time                    20 min/day  [EST]
  Post-automation (exception review only)                8 min/day
  12 min/day × 305 days = 3,660 min = 61.0 hrs × $33       =   $2,013

AVOIDED DUPLICATE DOSES
  Duplicates administered annually due to unreconciled records  18  [EST]
  Vaccine product cost per wasted dose (blended peds)     $95  [EST]
  18 × $95                                                 =   $1,710

SCHOOL-DEADLINE CRISIS AVOIDANCE (Aug/Sep surge compression)
  Urgent same-week form/vaccine scrambles avoided                40  [EST]
  Staff time per scramble                              22 min = $12.10
  40 × $12.10                                              =     $484

RETENTION EFFECT
  Families dismissed or self-departing over vaccine non-compliance
  who would have been retained with proactive outreach            4  [EST]
  Lifetime margin per retained pediatric family (3 yr horizon,
  discounted, conservative)                            $1,600  [EST]
  4 × $1,600                                               =   $6,400

─────────────────────────────────────────────────────────────────
  TOTAL MODELED ANNUAL BENEFIT                            $36,987
  Annual runtime cost                                     ($2,150)
  NET ANNUAL BENEFIT                                      $34,837
  One-time build (midpoint)                              ($11,000)
  YEAR 1 NET                                              $23,837
  YEAR 2+ NET                                             $34,837
─────────────────────────────────────────────────────────────────
```

**A note on the retention line:** this is the softest number in the model. It is
included because the practice's stated dismissal policy makes it a genuine
dynamic, but if it is removed entirely the initiative still returns $30,587
annually and remains clearly worth doing. Do not build the case on it.

#### Risks & Controls

| Risk | Severity | Control |
| --- | --- | --- |
| Recall sent for a vaccine the child already received elsewhere | Medium | Reconcile against I-CARE before every send; include "if your child received this elsewhere, reply and let us know" in every template |
| Message perceived as spam or pressure | Medium | Strict cadence caps (max 3 per gap per 90 days); global frequency cap per family; instant opt-out |
| TCPA / consent exposure on SMS | **High** | Documented consent capture at registration; opt-out honored within seconds not days; retain consent records; treat recall as informational-not-marketing and document that determination |
| Forecast logic error creating systematic false positives | **High** | Do not write the schedule logic. Consume CDSi or I-CARE. Validate against 200 known-good records before go-live. |
| A vaccine-hesitant family receives an automated message and escalates | Low | Suppression flag on the chart honored by the send pipeline; physician can exclude any family with one click |

#### Proposal

The practice has taken an explicit, public position that immunization compliance
is a condition of continued care. That position creates an obligation to make
compliance easy — it is difficult to defend dismissing a family for falling behind
if the practice never proactively told them they were behind.

This initiative operationalizes a stated clinical policy. It converts immunization
compliance from an opportunistic process that depends on a child happening to walk
in, into a systematic one with an audit trail.

The economics are favorable but the strategic case is stronger. Adolescent
immunization — HPV, MenACWY, MenB — is the single most-missed category in
pediatrics nationally, precisely because adolescents come in the least. A recall
engine reaches exactly the population that opportunistic capture cannot.

**Recommendation: fund it, sequence it second.** Start with the free I-CARE
remind/recall capability in week one while the build proceeds. It is the fastest
path to demonstrating that this program produces results.

---

### I-03 — Pre-Visit Chart Preparation Agent

#### Current State

Before a patient is roomed, someone has to know what this visit needs. Today that
knowledge is assembled in real time, in the room, by an MA with a chart open and
a patient waiting.

For a well visit, the MA must determine and locate:

- Age-appropriate developmental screening (ASQ-3, M-CHAT-R at 18 and 24 months)
- Adolescent depression screening (PHQ-A) for age 12+
- Lead screening status (Illinois requirements at 12 and 24 months, plus risk-based)
- Hemoglobin/anemia screening status
- Vision and hearing screening status and due date
- Immunizations due
- Growth trajectory — is this child crossing percentiles?
- Outstanding forms
- Open referrals or specialist reports not yet returned
- Prior visit's follow-up items

For a sick visit, less — but still: allergies, current medications, chronic
conditions, recent visits for the same complaint, and whether this is the third
ear infection in six months.

This is all done during rooming, with a family present and a clock running.

#### Failure Modes

| Failure | Consequence |
| --- | --- |
| Screening missed because nobody checked | Quality metric miss; developmental delay caught later than it should have been |
| Growth percentile crossing not noticed | The single most important longitudinal signal in pediatrics, missed |
| Specialist report never returned and nobody noticed | Care gap, potential liability |
| Rooming runs long | Cascading schedule delay across the entire session |
| MA does prep work in the room instead of before | Family watches staff read a chart; perceived as disorganized |

#### Does This Need an LLM?

**Mixed — this is a genuinely good LLM use case for one specific reason:
synthesis of longitudinal narrative.**

| Sub-task | Tool | Why |
| --- | --- | --- |
| "Which screenings are due at this age?" | Deterministic rules table | Bright Futures periodicity schedule is a lookup table. Do not use an LLM. |
| "Are immunizations due?" | CDSi engine (from I-02) | Already built |
| "Has this child crossed growth percentiles?" | Deterministic calculation against CDC/WHO growth reference | Pure math |
| "Are there open referrals or unreturned results?" | Structured query | Deterministic |
| **"What happened at the last three visits that matters today?"** | **LLM — strong fit** | This requires reading unstructured clinical narrative and extracting what is relevant to today's visit reason. No rules engine does this. |
| **"Summarize this chronic-condition patient's trajectory"** | **LLM — strong fit** | Longitudinal synthesis across free text |
| **"Draft the huddle brief in readable prose"** | **LLM — strong fit** | Presentation layer |

**Verdict:** the rules engine produces the checklist; the LLM produces the
narrative context. Both are needed and they are not interchangeable.

**On the word "agent":** this workflow is often marketed as agentic. It is not,
and should not be. It is a scheduled batch job with a fixed sequence of steps.
There is no reason to give a model autonomous tool-selection authority over a
patient chart to produce a summary. Constrain it to a pipeline. Agentic
architectures earn their complexity when the task space is genuinely open-ended;
this one is not.

#### Target State

At 5:00pm the day before clinic, a batch job runs against tomorrow's schedule and
produces, for each patient, a one-screen brief:

```
┌──────────────────────────────────────────────────────────────┐
│  [Patient initials] · Age 4y 2m · Well Visit · 9:20am · Dr. X │
├──────────────────────────────────────────────────────────────┤
│  DUE TODAY                                                    │
│   ⚠ DTaP #5, IPV #4, MMR #2, Varicella #2  (4-6y series)      │
│   ⚠ Vision screening — first formal screen due                │
│   ○ Hearing screening — due                                   │
│                                                               │
│  FLAGS                                                        │
│   ⚠ BMI crossed 85th → 91st percentile since last visit       │
│   ⚠ Allergy referral placed 2026-03-14, no report received    │
│                                                               │
│  CONTEXT (AI-generated, review before relying on)             │
│   Three sick visits since January, all URI. Parent raised     │
│   sleep concerns at the last well visit; no follow-up         │
│   documented. Albuterol prescribed once in February, not      │
│   refilled.                                                   │
│                                                               │
│  ADMIN                                                        │
│   ○ Preschool form requested by parent 2026-08-11 — pending   │
│   ○ Insurance eligibility: verified 2026-08-20 ✓              │
└──────────────────────────────────────────────────────────────┘
```

Distribution: PDF to each provider and MA, plus a live view in the internal app.
Every AI-generated element is visually distinguished from deterministic output.

#### Reference Implementation

```
Trigger:         Cron, 17:00 daily, against next clinic day's schedule
Schedule pull:   FHIR Appointment query
Per patient:
  Rules engine:  Bright Futures periodicity table (JSON config) → screenings due
  CDSi:          Immunizations due (shared with I-02)
  Growth calc:   CDC/WHO LMS reference → percentile + delta from prior
  Structured Q:  Open ServiceRequest, DiagnosticReport pending, open forms
  Narrative:     Last 3 encounter notes + problem list → Bedrock Claude
                 → structured JSON: {relevant_history[], open_threads[],
                   parent_concerns_unresolved[]}
Assembly:        Template render → PDF + web view
Distribution:    Secure internal delivery only. NOT email.
Feedback loop:   One-click "this brief was useful / not useful / wrong" per
                 patient, logged for prompt iteration
```

**Prompt constraints that matter:**

- The model receives only the last 3 encounter notes plus the problem list —
  not the entire chart. Minimum necessary, and it keeps latency and cost down.
- Output is a strict JSON schema. Free-form prose output is rendered from the
  JSON, not returned directly.
- The model is instructed explicitly: **it may not generate clinical
  recommendations, diagnoses, or suggested orders.** It reports what is in the
  record. A brief that says "consider asthma workup" is out of scope and out of
  the MA's lane under 54.2.
- Temperature 0. Deterministic where possible.

#### Buy vs. Build

| Option | Product | Cost | Assessment |
| --- | --- | --- | --- |
| **Buy** | EHR native pre-visit planning module | Often bundled | Check first. Most modern ambulatory EHRs have some version. Quality varies wildly. |
| **Buy** | Ambient scribe "pre-charting" features (Nabla, Vero, DeepCura) | Included in $69–$239/provider/mo | Increasingly capable. If I-05 is funded, this may come along free. **Evaluate before building.** |
| **Build** | Custom per above | ~$120–$300/mo | Best fit, reuses I-01/I-02 components |

```
Bedrock (Claude Sonnet, ~100 briefs/day × 305 days × ~6k tokens) = $980/yr
Compute (shared)                                                  = $500/yr
                                                            Total ≈ $1,480/yr ≈ $124/mo
```

#### Complexity

**L2** if the ambient-scribe vendor's pre-charting feature is used as the
narrative layer. **L3** if built fully custom.

#### Time to Value

- Weeks 1–3: rules tables (Bright Futures periodicity, growth reference)
- Weeks 4–5: structured queries and assembly
- Weeks 6–7: LLM narrative layer and prompt tuning
- Week 8: pilot with one provider
- **Production: week 10.**

#### Quantified Benefit

```
ASSUMPTIONS
  Annual encounters                                           30,500
  Well visits (40%)                                           12,200
  Sick visits (60%)                                           18,300

  Current MA chart-prep time, well visit                     4.0 min  [EST]
  Current MA chart-prep time, sick visit                     1.5 min  [EST]
  Weighted current                                           2.50 min
  Post-automation (read the brief)                           1.00 min
  Time saved per encounter                                   1.50 min

MA TIME SAVED
  30,500 × 1.5 min = 45,750 min = 762.5 hrs × $33      =   $25,163

SCREENING CAPTURE UPLIFT (quality + revenue)
  Screenings currently missed at point of care                  6%  [EST]
  Age-eligible screening opportunities annually             14,000  [EST]
  Additional screenings captured (6% × 40% recovery)           336
  Average reimbursement per screening (96110, 96127)     $12  [EST]
  336 × $12                                            =    $4,032

SCHEDULE ADHERENCE
  Rooming overruns avoided per day                              4  [EST]
  Downstream delay minutes recovered per overrun            3 min
  4 × 3 × 305 = 3,660 min = 61 hrs × $33               =    $2,013

PHYSICIAN TIME (50% discount per §2.4)
  Time saved per encounter hunting for context             0.5 min  [EST]
  30,500 × 0.5 = 15,250 min = 254 hrs × $147 × 0.50    =   $18,669
  → REPORTED SEPARATELY AS CAPACITY UPSIDE, NOT IN TOTAL

─────────────────────────────────────────────────────────────────
  TOTAL MODELED ANNUAL BENEFIT (hard)                     $31,208
  Annual runtime cost                                     ($1,480)
  NET ANNUAL BENEFIT                                      $29,728
  One-time build (midpoint)                               ($9,000)
  YEAR 1 NET                                              $20,728
  YEAR 2+ NET                                             $29,728

  Physician capacity upside (excluded above)              $18,669
─────────────────────────────────────────────────────────────────
```

> Headline table in §1 lists $25,150 for I-03 — the MA-time line only — as the
> most defensible single figure. The fuller $31,208 including screening capture
> and schedule adherence is shown here for completeness.

#### Risks & Controls

| Risk | Severity | Control |
| --- | --- | --- |
| AI narrative contains a hallucinated detail | **High** | Strict JSON schema; every narrative claim must cite the source encounter date; visually marked as AI-generated; clinicians trained that the brief is a pointer, not a source of truth |
| Staff treat the brief as authoritative and stop opening the chart | **High** | Brief explicitly labeled "not a substitute for chart review"; deliberately omits data that must be verified in-chart (e.g. exact medication doses) |
| Screening rules table drifts from current guidelines | Medium | Version the config; annual review tied to Bright Futures/ACIP update cycle; owner assigned |
| Brief becomes noise and is ignored | Medium | Cap at one screen; the one-click feedback loop drives iteration; measure open rate |

#### Proposal

Pediatric primary care is unusual among specialties in how much of a visit is
**scheduled by age rather than driven by complaint.** A four-year-old well visit
has a defined content set: specific vaccines, specific screenings, specific
anticipatory guidance. That predictability is precisely what makes it automatable.

The current process asks an MA to reconstruct that content set from memory and
chart review, in the room, under time pressure, dozens of times a day. It works
because experienced MAs are good at it — the practice notes staff with twenty-plus
years of tenure. But it does not scale to new hires, it degrades under volume,
and it produces no record of what was considered and skipped.

The screening-capture argument is the strongest one. Developmental screening at
18 and 24 months, and adolescent depression screening, are the two highest-value
preventive interventions in pediatric primary care. A six percent miss rate on
those is not a billing problem; it is a children-not-getting-caught problem.

**Recommendation: fund it, but sequence it fourth,** after the ambient scribe
decision in I-05. If the chosen scribe vendor ships adequate pre-charting, the
build scope here shrinks dramatically and the money is better spent elsewhere.
Do not build what you may be about to buy.


---

### I-04 — Telephone Triage Documentation Assistant

> **The highest-volume, lowest-visibility MA workload in this specific practice.**

#### Current State

North Suburban Pediatrics has made an explicit, public commitment that it does
not charge for phone calls, on the stated reasoning that patients should feel
free to contact the practice without cost being a barrier. It also states that
staff are trained to triage urgent calls during office hours and route them to
physicians.

That policy is admirable and it has a direct operational consequence: **it
maximizes inbound clinical call volume by design.** A practice that does not
charge for phone access gets more phone calls than one that does. Those calls
land on the MAs.

The current workflow per clinical call:

1. Call arrives; front desk determines it is clinical and transfers or takes a message.
2. MA calls back — often the second or third attempt, since parents are also busy.
3. MA works through a symptom assessment, typically against Schmitt-Thompson
   pediatric telephone protocols, which are the standard of care in pediatrics.
4. MA determines disposition: see today, see tomorrow, home care, ED now, or
   escalate to physician.
5. MA delivers home care advice and confirms understanding.
6. **MA writes a telephone encounter note**, from memory, after the call ends.
7. If a follow-up is needed, MA creates a task or reminder — or does not, and
   remembers.

Step 6 is the automation target. Steps 3, 4, and 5 are clinical judgment and
stay entirely human.

#### Failure Modes

| Failure | Consequence |
| --- | --- |
| Note written from memory 20 minutes later | Detail loss; the note is the only record if the outcome is ever questioned |
| Note not written at all during a busy session | **This is the significant liability exposure.** An undocumented triage call that preceded a bad outcome is indefensible. |
| Protocol used but not recorded | No evidence that a standard of care was followed |
| Disposition given but not captured | Cannot demonstrate what advice was actually delivered |
| Follow-up not tasked | Patient told "call back if worse" and nobody closes the loop |
| Inconsistent documentation across staff | Some MAs write three sentences, others three paragraphs |

Telephone triage is a well-documented malpractice exposure in pediatrics
specifically because the highest-acuity decisions are frequently made on the
phone with the least documentation.

#### Does This Need an LLM?

**Yes — this is one of the two clearest LLM fits in this document.** The task is
converting spoken clinical dialogue into a structured, complete, consistently
formatted record. That is precisely what language models are good at.

But the boundary must be drawn with total clarity:

| Task | Who Does It |
| --- | --- |
| Assess the symptom | **Human MA** |
| Select and apply the protocol | **Human MA** |
| Determine disposition | **Human MA** (escalating to physician per protocol) |
| Deliver advice to the parent | **Human MA** |
| **Transcribe and structure the encounter into a note** | **AI** |
| **Extract the protocol used, disposition, and advice given into structured fields** | **AI** |
| **Draft the follow-up task** | **AI** |
| Sign the note | **Human MA** |

**Under no circumstances does the system suggest a disposition.** Not as a hint,
not as a "consider," not greyed out. An unlicensed medical assistant in Illinois
operates under physician delegation and may not perform tasks requiring
independent clinical judgment. A system that surfaces a machine-generated
disposition to an MA creates enormous pressure to accept it, and manufactures
exactly the unauthorized-practice exposure that 54.2 exists to prevent.

This is the single most important design constraint in the entire document.

#### Target State

1. MA opens the triage app and selects the patient.
2. MA presses record. Call proceeds normally.
3. During the call, the MA taps the protocol being used from a searchable list
   (Schmitt-Thompson protocol name) and taps the disposition when reached. Two
   taps, no typing.
4. Call ends. Audio is transcribed and processed.
5. Within ~30 seconds a structured draft note appears:
   - Caller identity and relationship to patient
   - Chief complaint and symptom timeline
   - Pertinent positives and negatives elicited
   - Protocol applied (from the MA's tap, not inferred)
   - Disposition (from the MA's tap, not inferred)
   - Advice given
   - Follow-up instructions and safety-net advice ("return if…")
   - Suggested follow-up task with due date
6. MA reviews, edits, signs. Target: under 60 seconds.
7. Note posts to the chart. Follow-up task posts to the work queue.
8. Audio is **deleted** after note finalization. Retention of triage call audio
   creates a discovery surface with no clinical benefit.

#### Reference Implementation

```
Client:         Mobile/tablet PWA or desktop app on the MA workstation
Capture:        Browser MediaRecorder or a dedicated handset integration
Transcription:  AWS Transcribe Medical (HIPAA-eligible, medical vocabulary,
                speaker diarization) — or AWS HealthScribe at $0.10/audio min
                which bundles transcription + structured extraction
Structuring:    Bedrock Claude with a strict JSON schema:
                {
                  caller: {name, relationship},
                  chief_complaint: string,
                  symptom_onset: string,
                  symptoms_positive: [string],
                  symptoms_negative: [string],
                  advice_given: [string],
                  safety_net_instructions: [string],
                  followup_needed: boolean,
                  followup_due: date|null,
                  transcript_gaps: [string]     // where audio was unclear
                }
                Protocol and disposition are injected from the MA's UI taps.
                The model is explicitly forbidden from populating them.
Note render:    JSON → practice-standard telephone encounter template
Post-back:      FHIR Communication or DocumentReference to the EHR
Task creation:  FHIR Task
Audio lifecycle: S3 with a 24-hour lifecycle policy, deleted on note signature
Audit:          Every note records: transcript hash, model+version, prompt hash,
                edit diff between draft and signed version, signing user
```

**The edit-diff log is not optional.** It is the mechanism by which the practice
proves that a human actually reviewed the draft rather than rubber-stamping it,
and it is the dataset that tells you whether the system is getting better or
worse over time.

#### Buy vs. Build

| Option | Product | Cost | Assessment |
| --- | --- | --- | --- |
| **Buy** | Ambient scribes with telephone support (Nabla, Vero, Freed, Heidi) | $69–$239/provider/mo | Most support telehealth/phone audio. **But they are licensed per clinician and priced for physicians, not for six MAs sharing a phone queue.** Licensing 6 MAs at $99 each is $594/mo for a use case the vendor did not design for. |
| **Buy** | AWS HealthScribe | $0.10/audio minute | Consumption-priced, which fits this shape far better. 120 calls/day × 4 min × 305 days = 146,400 min/yr = **$14,640/yr**. Expensive at this volume. |
| **Buy** | Dedicated nurse-triage platforms (Schmitt-Thompson licensed products, e.g. TriageLogic, Advanced Triage) | $200–$600/mo | Bring the protocol library, which has real licensing value. Documentation features vary. **Worth a serious look — the protocol license alone may justify it.** |
| **Build** | Transcribe Medical + Bedrock per above | ~$200–$450/mo | Best economics at this volume, but requires licensing Schmitt-Thompson protocols separately |

**Runtime cost math for the build path:**

```
AWS Transcribe Medical: 146,400 min/yr × $0.0175/min          = $2,562/yr
Bedrock (Claude Haiku, ~36,600 calls × ~4k tokens)            =   $440/yr
Compute + storage (shared infrastructure)                      =   $700/yr
Schmitt-Thompson protocol license (pediatric, small practice)  = $1,800/yr  [EST]
                                                        Total ≈ $5,502/yr ≈ $459/mo
```

> Note the protocol license. If the practice already licenses Schmitt-Thompson —
> and a practice with a 24/7 physician-answered call model very likely does —
> subtract $1,800 and the runtime drops to ~$310/mo.

#### Complexity

**L2.** No EHR write-back complexity beyond a document post, no state registry,
no PDF manipulation. The hardest part is the client-side audio capture UX and
getting MAs comfortable with it.

#### Time to Value

- Weeks 1–2: capture client, transcription pipeline
- Weeks 3–4: structuring prompt, JSON schema, note template
- Week 5: EHR post-back and task creation
- Weeks 6–7: pilot with two MAs, prompt iteration against real edit diffs
- **Production: week 8.**

#### Quantified Benefit

```
ASSUMPTIONS
  Clinical calls per day, BG site                               120  [EST]
  Clinic days per year                                          305
  Annual clinical calls                                      36,600

  Current documentation time per call                       3.5 min  [EST]
  Target documentation time (review + sign)                 1.2 min
  Time saved per call                                       2.3 min

MA TIME SAVED
  36,600 × 2.3 min = 84,180 min = 1,403 hrs × $33      =   $46,299

DOCUMENTATION COMPLETENESS (risk reduction — modeled conservatively)
  Calls currently undocumented or minimally documented          9%  [EST] = 3,294
  Post-automation                                               1%          =   366
  Additional documented encounters                                    2,928
  → Not monetized. Malpractice risk reduction is real and
    unquantifiable. Listed for the narrative, excluded from the total.

FOLLOW-UP LOOP CLOSURE
  Follow-ups currently dropped                                  7%  [EST]
  Calls requiring follow-up (25% of 36,600)                   9,150
  Dropped follow-ups                                            641
  Post-automation dropped rate                                  2%  =   183
  Recovered follow-ups                                          458
  Share converting to a billable visit                         30%  =   137
  137 × $140                                            =   $19,180
  → Discounted 50% for conservatism                     =    $9,590

─────────────────────────────────────────────────────────────────
  TOTAL MODELED ANNUAL BENEFIT                            $55,889
  (Headline table uses the MA-time line only: $46,299)
  Annual runtime cost                                     ($5,502)
  NET ANNUAL BENEFIT (headline basis)                     $40,797
  One-time build (midpoint)                               ($8,000)
  YEAR 1 NET                                              $32,797
  YEAR 2+ NET                                             $40,797
─────────────────────────────────────────────────────────────────
```

**Sensitivity on call volume — the most uncertain input in this document:**

| Calls/day | Annual MA Time Saved | Annual Benefit |
| --- | --- | --- |
| 60 | 701 hrs | $23,150 |
| 90 | 1,052 hrs | $34,725 |
| **120 (base)** | **1,403 hrs** | **$46,299** |
| 150 | 1,754 hrs | $57,874 |
| 200 | 2,338 hrs | $77,166 |

**Measure this before building.** One week of call logging tells you which row
you are on, and the answer changes the priority ranking materially.

#### Risks & Controls

| Risk | Severity | Control |
| --- | --- | --- |
| System infers or suggests a disposition | **Critical** | Architecturally prevented: disposition and protocol are injected from UI taps and the model's output schema has no field for them. Enforce in code, test in CI. |
| Transcription error changes clinical meaning | **High** | MA reviews every note; unclear audio segments explicitly flagged in the draft as `transcript_gaps`; medical vocabulary model used |
| Recording without consent | **High** | Illinois is a **two-party consent** state for private conversations. A recorded-line disclosure must play or be stated at the start of every recorded call, and consent must be documented. This is a hard legal requirement, not a courtesy. |
| Audio retained and becomes discoverable | Medium | 24-hour lifecycle deletion; deleted immediately on note signature; documented retention policy |
| MA rubber-stamps drafts | **High** | Edit-diff monitoring; if an MA's edit rate approaches zero, that is a coaching signal, not a success metric |
| Model degrades after a vendor update | Medium | Pin model version; regression test suite of 50 known calls; re-validate before any version bump |

#### Proposal

This is the initiative with the strongest combination of **hard dollar return and
risk reduction**, and it is uniquely well-suited to this practice because of a
policy the practice has chosen deliberately.

By publicly committing to free, unlimited phone access — and staffing an
after-hours answering service that routes to physicians around the clock — North
Suburban Pediatrics has made telephone medicine a core service line. It is
delivered at high volume, by MAs, with real clinical stakes, and it is documented
by hand from memory at the end of a shift.

The exposure is straightforward to state: if a triage call ever precedes a bad
outcome, the practice's entire defense is the note. A note written from memory
twenty minutes later, or not written at all because the session ran long, is not
a defense.

Automating the documentation does three things at once. It returns roughly 1,400
MA hours a year. It raises documentation completeness from an estimated 91% to
99%. And it produces structured data — protocol used, disposition reached — that
the practice can actually analyze, which nobody can do with free-text notes.

The design constraint bears repeating because it is what makes this defensible:
**the machine never touches the clinical decision.** The MA assesses, the MA
selects the protocol, the MA determines disposition, the MA gives advice. The
machine writes it down. That division is not a limitation imposed by caution; it
is the correct architecture, and it is what Illinois delegation law requires.

**Recommendation: fund it, sequence it third — but measure call volume first.**
This is the initiative whose ROI is most sensitive to an unverified assumption.
One week of logging resolves it.

---

### I-05 — Ambient Documentation for Rooming & Encounters

#### Current State

The MA rooms the patient: takes vitals, obtains history of present illness,
reviews medications and allergies, records the reason for visit, and enters all
of it into the EHR. The physician then enters the room, repeats much of the same
conversation, and later writes the encounter note.

The duplication is structural. The MA documents; the physician documents; both
document some of the same content.

#### Failure Modes

| Failure | Consequence |
| --- | --- |
| MA types while the family talks | Reduced eye contact; parents perceive inattention; details missed |
| History taken twice | Family repeats themselves; visit runs long; parent frustration |
| Documentation deferred to end of session | Detail loss; late close-out; unpaid overtime |
| Physician charts after hours | Burnout — the most-cited driver in ambulatory medicine |
| Inconsistent rooming documentation | Downstream data quality problems for every other initiative in this document |

#### Does This Need an LLM?

**Yes, and it is the one category where you should unambiguously buy rather than
build.**

Ambient clinical documentation is a mature, competitive, commoditized product
category as of 2026. The pipeline is well understood: capture audio, transcribe,
use an LLM to draft a SOAP-format note, map terms to ICD-10/SNOMED/CPT, write
back to the EHR for clinician review. Multiple vendors do this competently. There
is no defensible reason for a ten-physician practice to build it.

**The market has also commoditized downward, which matters enormously here.**
Epic shipped built-in AI charting in August 2025 at no additional cost.
Athenahealth added a native ambient scribe in February 2026, also included. If
the practice runs either platform, the first move is to enable what they already
own and evaluate whether it is sufficient before spending anything.

#### Target State

**Two distinct deployments, and conflating them is a common mistake:**

**Deployment A — Physician encounter documentation.** Standard ambient scribe.
Physician records the visit, reviews and signs a drafted note. This is what every
vendor sells.

**Deployment B — MA rooming documentation.** The MA's rooming conversation is
captured and structured into the intake fields: chief complaint, HPI, medication
reconciliation, allergy confirmation, ROS. This is a *different* product shape and
most vendors price it as a full clinician seat, which does not work economically.

The practical approach: license the scribe for physicians (Deployment A), and
solve MA rooming (Deployment B) either through a vendor that supports
non-physician seats at a lower tier, or by extending the I-04 build — which
already has audio capture, medical transcription, and structured extraction — to
a second note type. **Reusing I-04 for MA rooming is the recommended path and is
why I-04 should be built before this decision is finalized.**

#### Reference Implementation

For Deployment A, there is no implementation. You configure a vendor.

For Deployment B, if extending I-04:

```
Capture:        Same PWA client, "Rooming" mode
Transcription:  AWS Transcribe Medical (already provisioned for I-04)
Structuring:    Bedrock Claude, different output schema:
                {
                  chief_complaint, hpi_narrative,
                  ros_positive[], ros_negative[],
                  medications_confirmed[], medications_changed[],
                  allergies_confirmed[], allergies_new[],
                  parent_concerns[],
                  vitals_mentioned{}   // cross-check against device entry
                }
Post-back:      FHIR write to encounter intake fields
Review:         MA confirms before the physician enters the room
```

The `vitals_mentioned` cross-check is a small feature with outsized value: if the
MA says "eighteen pounds four ounces" and the entered weight is 18.4 lb rather
than 18 lb 4 oz, the discrepancy surfaces. Pediatric weight-based dosing makes
that error category consequential.

#### Buy vs. Build

**Current market pricing, verified August 2026 — confirm directly with vendors
before contracting, as this category re-prices frequently:**

| Vendor | Price | Notes |
| --- | --- | --- |
| **Epic AI Charting** | **$0** (included) | If on Epic. Less feature-rich than standalone, native integration. Start here. |
| **athenahealth native scribe** | **$0** (included) | Shipped Feb 2026. If on athena, start here. |
| **Heidi Health** | Free tier (unlimited basic documentation); Evidence Plus $40/mo; Clinician $150/mo | Strongest free tier in the market. Excellent evaluation vehicle. |
| **Freed** | Starter $39/mo (40 notes); Core $79/mo (unlimited); Premier $104–$119/mo (adds EHR push + ICD-10) | The default first evaluation for small practices. Zero setup fees, self-serve, no contract. |
| **Twofold Health** | $49/mo annual, $69/mo monthly, unlimited notes, BAA included | Cheapest credible flat-rate option |
| **Vero** | $69/mo annual | Includes PDF form auto-fill, which is relevant to I-01 |
| **Commure Scribe** | $89/mo, or $59/mo billed annually | 7-day trial |
| **Sunoh.ai** | Contact vendor | **Built by healow/eClinicalWorks. If the practice runs eCW, this is the native option — notes populate inline in the eCW progress note with no copy-paste.** |
| **Nabla** | ~$119–$239/clinician/mo; moved to contract sales in 2026 | Free tier exists but **reportedly does not include a BAA — disqualifying for PHI** |
| **Abridge** | ~$400–$600/provider/mo, enterprise contracts | Best-validated (KLAS 95.3, NEJM AI study). Not sold to practices this size. |
| **Microsoft Dragon Copilot (ex-Nuance DAX)** | ~$400–$600+/provider/mo | Enterprise. Overkill. |
| **AWS HealthScribe** | $0.10/audio minute | Consumption-priced. Good for the Deployment B build path. |

**Recommended selection logic:**

1. If on eCW → evaluate **Sunoh.ai** first (native integration is worth a premium)
2. If on Epic or athena → **enable the included native scribe** and evaluate for 30 days
3. Otherwise → 30-day parallel pilot of **Freed Core ($79)** vs **Heidi free tier**
   vs **Twofold ($49)** with 3 physicians
4. **Do not evaluate Abridge, DAX, Ambience, or DeepScribe.** They are enterprise
   products with multi-week implementations, IT requirements, and multi-year
   contracts. A ten-physician independent practice is not their customer.

**Budget for planning purposes: 10 physicians × $79/mo = $790/mo = $9,480/yr.**
Premier tier with EHR push: 10 × $119 = $1,190/mo = $14,280/yr.

#### Complexity

**L1** for Deployment A. Under an hour of setup for self-serve tools; 1–3 days for
mid-market. This is the easiest initiative in the document.

**L2** for Deployment B if extending I-04.

#### Time to Value

- Week 1: vendor shortlist, BAAs requested, trial accounts provisioned
- Weeks 2–5: 30-day parallel pilot, 3 physicians, measure note-edit rate and
  minutes-saved-per-encounter
- Week 6: selection and rollout
- **Production: week 6.** This can be running before anything else in the plan.

#### Quantified Benefit

```
ASSUMPTIONS
  Annual encounters                                           30,500
  MA rooming documentation time, current                     3.0 min  [EST]
  MA rooming documentation time, post                        1.0 min
  Saved per encounter                                        2.0 min

  Physician note time, current                               6.0 min  [EST]
  Physician note time, post                                  2.5 min
  Saved per encounter                                        3.5 min

MA TIME SAVED (Deployment B)
  30,500 × 2.0 min = 61,000 min = 1,016.7 hrs × $33    =   $33,551

PHYSICIAN TIME RECAPTURED (Deployment A)
  30,500 × 3.5 min = 106,750 min = 1,779.2 hrs
  → NOT counted as savings per §2.4.
  → Presented as capacity upside below.

CAPACITY UPSIDE (explicitly separate, explicitly optional)
  Physician hours recaptured                                 1,779.2
  Share the practice elects to convert to encounters           20%  [EST]
  Converted hours                                              355.8
  Encounters per physician-hour                                  3.0
  Additional encounters                                        1,067
  1,067 × $140                                          =  $149,380
  → This is a CHOICE, not an automatic benefit. It requires
    deliberately adding schedule slots. If the practice instead
    uses the time to go home earlier, this number is $0 and the
    benefit is retention and burnout reduction.

CODING ACCURACY (E&M level capture)
  Encounters currently under-coded due to insufficient
  documentation support                                        4%  [EST] = 1,220
  Average revenue delta per corrected level                    $22  [EST]
  Recovery rate                                                50%
  1,220 × 0.5 × $22                                     =   $13,420
  → Requires the Premier/EHR-push tier with ICD-10 and E&M support.
  → Discounted 50% for conservatism                     =    $6,710

─────────────────────────────────────────────────────────────────
  TOTAL MODELED ANNUAL BENEFIT (hard, excl. capacity)     $40,261
  (Headline table uses MA-time only: $33,551)
  Annual cost (10 × $79/mo)                               ($9,480)
  NET ANNUAL BENEFIT                                      $30,781
  One-time cost                                                 $0
  YEAR 1 NET                                              $30,781

  Capacity upside if elected                             $149,380
─────────────────────────────────────────────────────────────────
```

#### Risks & Controls

| Risk | Severity | Control |
| --- | --- | --- |
| Note omissions — the most common error type in ambient scribing | **High** | The April 2026 JAMA multisite study of 8,581 clinicians found physicians reported omissions as the dominant error class. Clinician review of every note is mandatory and non-negotiable. Train explicitly on omission-hunting, not just error-hunting. |
| Free tier used without a BAA | **Critical** | Nabla's free tier reportedly lacks a BAA. **Verify BAA coverage in writing before a single patient encounter is recorded on any tool, free or paid.** |
| Two-party consent | **High** | Illinois requires consent for recording. Signage, verbal disclosure, and documented consent at registration. Build into the intake packet. |
| Vendor consolidation / shutdown | Medium | This category is consolidating rapidly. Prefer month-to-month contracts. Verify note export capability before signing. Avoid multi-year lock-in at this practice size. |
| Physicians stop reviewing carefully | **High** | Track note-edit rate per physician. A physician with a near-zero edit rate is not reviewing. |
| Cost creep as vendor raises prices | Low | Month-to-month; the category is competitive enough that switching is viable |

#### Proposal

This is the **cheapest, fastest, lowest-risk initiative in the document**, and it
should be started in week one regardless of what else is funded.

The case is almost entirely about sequencing rather than magnitude. At $79 per
physician per month with no setup fee, no contract, and a self-serve trial, the
downside is one month of subscription cost and a few hours of pilot time. The
upside is roughly 1,000 recaptured MA hours and 1,779 recaptured physician hours
annually.

Three specific arguments for this practice:

**First, check what you already own.** If the practice runs Epic or athenahealth,
a native ambient scribe is already included at no cost. If it runs eClinicalWorks,
Sunoh.ai integrates inline with the progress note rather than requiring
copy-paste. The right first move may cost nothing.

**Second, this practice runs evening and weekend sessions.** Monday through
Thursday evening sick clinics, Saturday mornings, Sunday and holiday mornings.
Documentation burden on those sessions is the same, but the staffing is thinner
and the incentive to defer charting is higher. Ambient documentation
disproportionately helps the shifts where deferral is most likely.

**Third, this is a recruiting and retention asset.** The practice competes for
physicians against hospital-owned pediatric groups that have enterprise AI
tooling. A ten-physician independent group that has shipped ambient documentation
is signaling something about how it operates.

**Recommendation: fund it, start week one.** It is the proof-of-concept that
makes the rest of the program credible internally.

---

### I-06 — Inbound Fax & Document Ingestion

#### Current State

The practice publishes two fax numbers — 847-913-9173 for Buffalo Grove and
847-869-1070 for Evanston — as primary document channels. In ambulatory medicine
this remains the default transport for specialist consult notes, hospital
discharge summaries, imaging and lab results from outside facilities, prior
authorization correspondence, records requests, school and camp forms, and
insurance correspondence.

Current handling:

1. Fax arrives, printed or landing in an inbox.
2. A human reads enough of each document to determine what it is.
3. Human identifies the patient.
4. Human decides where it goes: chart, physician review queue, billing, or trash.
5. Human indexes and files it into the EHR with a document type.
6. If it requires action, human creates a task or walks it to someone.

**Observed time: ~2.5 minutes per document, at ~45 documents/day.** That is
roughly 112 minutes of daily front-desk and MA time spent sorting paper.

#### Failure Modes

| Failure | Consequence |
| --- | --- |
| Critical result buried in a stack | Abnormal lab or urgent specialist finding not seen for days |
| Misfiled to the wrong patient | Chart integrity problem, potential clinical harm |
| Filed but no task created | Actionable document sits in a chart nobody opens |
| Backlog during high-volume periods | Documents pile up exactly when the practice is busiest |
| No queryable record | Cannot answer "did we ever receive the ENT consult?" without a manual search |

#### Does This Need an LLM?

**Yes for classification, no for the rest.** This is a well-scoped document
intelligence problem.

| Sub-task | Tool |
| --- | --- |
| Convert fax TIFF/PDF to text | OCR (Textract) — deterministic |
| Identify document type | **LLM classifier — good fit.** Faxes are unstructured, inconsistently formatted, frequently poor quality. Rule-based classification on faxes fails constantly; this is exactly where LLM robustness pays. |
| Extract patient identifiers | LLM extraction + deterministic match against panel |
| Determine urgency | **LLM with a tightly bounded taxonomy.** "Does this document contain a critical value or an urgent recommendation?" is a classification task with clinical stakes — see risk controls. |
| Route to a queue | Deterministic rules on the classification output |
| File to the chart | Deterministic API call |
| Create a task | Deterministic |

#### Target State

1. Inbound fax hits a gateway API rather than a printer.
2. OCR extracts text; page images retained.
3. Classifier assigns document type from a fixed taxonomy:
   `specialist_consult | hospital_discharge | outside_lab | outside_imaging |
   prior_auth | records_request | school_form | insurance_correspondence |
   immunization_record | other`
4. Patient matched by name + DOB. **Confidence below threshold → human queue.
   No auto-filing on a fuzzy match.**
5. Urgency triage: `routine | needs_physician_review | urgent`.
6. Routing:
   - Routine + high-confidence match → auto-file, notify
   - Needs review → physician queue with an AI-generated one-line summary
   - Urgent → immediate alert to the on-duty physician **and** a parallel human
     verification, because an urgency misclassification cannot be allowed to
     fail silently
   - Low-confidence anything → human queue
7. Immunization records detected → route to the I-02 reconciliation pipeline
   automatically. This is the highest-value cross-initiative link in the plan.
8. Full-text search index across all received documents.

#### Reference Implementation

```
Gateway:        Cloud fax API — SRFax, Documo/mFax, or Twilio Fax
                (all offer BAAs; confirm before porting numbers)
OCR:            AWS Textract, with fallback to Tesseract for cost control on
                clean documents
Classification: Bedrock Claude Haiku — cheap, fast, sufficient for this task.
                Structured JSON output, fixed enum for document_type.
                Include a `confidence` field and honor it.
Patient match:  Deterministic — exact DOB + fuzzy name (Jaro-Winkler > 0.92)
                against the active panel. Multiple candidates → human queue.
                Twins share a DOB, so a DOB match alone is never sufficient.
Urgency:        Separate Bedrock call, narrow prompt, conservative bias.
                Instructed to over-flag rather than under-flag.
Filing:         FHIR DocumentReference with type coding
Tasking:        FHIR Task
Search:         OpenSearch (HIPAA-eligible), encrypted
Audit:          Every classification, match, and routing decision logged with
                model version and confidence
```

#### Buy vs. Build

| Option | Product | Cost | Assessment |
| --- | --- | --- | --- |
| **Buy** | EHR native fax/document management | Often bundled | Check first. Most EHRs have a document inbox; few have intelligent classification. |
| **Buy** | Concord, Consensus (eFax Corporate), Updox | $100–$400/mo | Good transport, basic classification, weak clinical routing |
| **Buy** | Sift Healthcare, Infinitus, Notable document AI | $500–$2,000/mo | Genuinely intelligent, priced for larger organizations |
| **Build** | Per architecture above | ~$150–$400/mo | Reuses OCR and Bedrock plumbing from I-01. **Marginal build cost is low if I-01 is already built.** |

```
Fax gateway (2 numbers, ~1,400 pages/mo)              = $1,080/yr
Textract: 45/day × 305 × ~3 pages × $0.0015           =   $62/yr
Bedrock Haiku classification + urgency (2 calls/doc)  =   $210/yr
OpenSearch (small domain)                             = $1,300/yr
Compute (shared)                                      =   $400/yr
                                                Total ≈ $3,052/yr ≈ $254/mo
```

#### Complexity

**L2**, dropping toward **L1.5** if I-01 is built first and the OCR/LLM
infrastructure already exists.

#### Time to Value

- Weeks 1–2: fax gateway migration (number porting takes ~2 weeks — start early)
- Weeks 3–4: OCR + classification, evaluated against 300 historical faxes
- Week 5: patient matching and routing
- Week 6: filing, tasking, search
- **Production: week 7.**

#### Quantified Benefit

```
ASSUMPTIONS
  Inbound documents per day                                      45  [EST]
  Working days per year                                         305
  Annual documents                                           13,725

  Current handling time per document                        2.5 min  [EST]
  Target (exception review only, ~20% of documents)         0.8 min
  Time saved per document                                   1.7 min

STAFF TIME SAVED (front desk rate)
  13,725 × 1.7 min = 23,333 min = 388.9 hrs × $28      =   $10,889

MISFILE CORRECTION AVOIDED
  Current misfile rate                                          2%  [EST] =  275
  Post-automation                                             0.5%          =   69
  Corrections avoided                                                       206
  Time per correction                                    18 min = $8.40
  206 × $8.40                                           =    $1,730

DOCUMENT SEARCH TIME
  "Did we receive X?" searches per week                          15  [EST]
  Current time per search                                    6 min
  Post (indexed search)                                    0.5 min
  15 × 52 × 5.5 min = 4,290 min = 71.5 hrs × $28       =    $2,002

IMMUNIZATION RECORD AUTO-ROUTING (cross-benefit with I-02)
  Outside immunization records received annually               900  [EST]
  Manual entry time each                                     7 min
  Post-automation (reconcile + confirm)                      2 min
  900 × 5 min = 4,500 min = 75 hrs × $33               =    $2,475

─────────────────────────────────────────────────────────────────
  TOTAL MODELED ANNUAL BENEFIT                            $17,096
  (Headline table uses the core sorting line: $10,815)
  Annual runtime cost                                     ($3,052)
  NET ANNUAL BENEFIT                                      $14,044
  One-time build (midpoint, assuming I-01 exists)         ($6,000)
  YEAR 1 NET                                               $8,044
  YEAR 2+ NET                                             $14,044
─────────────────────────────────────────────────────────────────
```

#### Risks & Controls

| Risk | Severity | Control |
| --- | --- | --- |
| Urgent document classified routine | **Critical** | Bias the classifier toward over-flagging; every document containing a numeric lab value outside reference range is force-flagged by rule regardless of LLM output; daily human audit of a random sample of 10 routine-classified documents |
| Misfiled to wrong patient (twins, siblings) | **High** | DOB + name match required; twin/sibling detection flag on the panel; multi-candidate → human queue always |
| Fax gateway outage | Medium | Retain a backup analog line; monitor and alert on zero-received-documents in a 4-hour window during business hours |
| Poor-quality faxes defeat OCR | Medium | Confidence threshold routes to human; measure OCR confidence distribution during pilot |
| Auto-filed document nobody reads | Medium | Auto-filing does not mean no task. Every clinically relevant type generates a review task. |

#### Proposal

This is the least glamorous initiative in the document and one of the more
reliable. Nothing about sorting faxes is interesting. That is exactly why it has
been absorbed into staff time rather than solved.

The practice publishes fax as a primary document channel and has for years. That
is not going to change soon — the ambulatory referral ecosystem still runs on it.
The question is whether a human reads every page to determine what it is.

The cross-initiative link is what elevates this above its standalone return.
Outside immunization records arriving by fax are the single largest source of the
chart-versus-registry discrepancies that I-01 and I-02 both have to reconcile.
Automating fax ingestion with immunization-record detection feeds the
reconciliation pipeline directly, which makes both of those initiatives measurably
more accurate. The $10,815 headline understates the value because a meaningful
portion of the benefit lands in other line items.

**Recommendation: fund it, sequence it eighth.** Low risk, moderate return, and
substantially cheaper to build after I-01 has established the OCR and LLM
infrastructure. Start the fax number porting early regardless — it has a two-week
lead time and blocks nothing else.

---

### I-07 — No-Show Reduction & Waitlist Backfill

> **Highest modeled dollar return of the ten. Lowest implementation complexity.
> Start here.**

#### Current State

The practice enforces a $25 fee for appointments not cancelled by 10:00am on the
day of the appointment, and requests 24-hour cancellation notice. A practice
does not institute a no-show fee unless no-shows are a real and persistent
operational problem.

Scheduling is telephone-only. The published instruction is to call between 9:00am
and 5:00pm. That means:

- A parent who realizes at 8:00pm that they need to cancel tomorrow's 9:20am
  appointment **cannot**. The office is closed. They will try to remember to call
  at 9:00am, and some will not.
- A parent who wants an appointment cannot self-schedule and must call during
  business hours, competing with the same phone lines carrying clinical triage.
- When a cancellation does occur, backfilling the slot requires someone to
  manually call down a list, if such a list exists.

The 10:00am cutoff is itself a signal: it exists because the practice needs to
know early enough to try to fill the slot, and filling it is manual.

#### Failure Modes

| Failure | Consequence |
| --- | --- |
| No-show | Slot revenue lost entirely; provider idle; another family who needed the slot did not get it |
| Late cancellation | Same, minus the fee |
| Cancellation with no backfill | Slot lost even though demand existed |
| Parent cannot cancel outside business hours | Converts a would-be cancellation into a no-show |
| Manual waitlist calling | Labor-intensive; often not attempted |
| $25 fee collection | Awkward, damages relationships, frequently waived, rarely collected |

#### Does This Need an LLM?

**No. Not at all. This is deterministic scheduling automation.**

This is the most important point in this section, because "AI scheduling" is a
heavily marketed category and the practice will be pitched it. Reminders,
confirmations, cancellation handling, and waitlist backfill are:

- A cron job
- A messaging API
- A rules engine
- A scheduling API

No language model required for any of it.

Two optional AI layers, both genuinely optional:

| Optional Layer | Tool | Worth It? |
| --- | --- | --- |
| No-show risk prediction | Gradient-boosted model on historical features (lead time, prior no-shows, day of week, appointment type, weather) | **Only after 12 months of clean data.** Do not start here. The base intervention captures most of the value. |
| Conversational rescheduling by SMS | LLM | Marginal. A link to a booking page outperforms a chatbot for this task. |

**Do not buy an LLM-powered scheduling product to solve a reminder problem.**

#### Target State

**Reminder cadence:**
- T-7 days: SMS confirmation request with one-tap confirm / cancel / reschedule
- T-48 hours: SMS reminder, same one-tap options
- T-2 hours: SMS reminder with location, parking note, and what to bring

**Cancellation handling:**
- Cancellation accepted **24/7** by SMS tap or web link
- Slot immediately released to the backfill engine

**Backfill engine:**
- Maintains a live waitlist of patients wanting an earlier appointment
- On release, blasts the slot to the top N eligible waitlisted patients
  simultaneously — matched on provider, visit type, and age-appropriateness
- First to accept takes it; others get a courteous "already filled" message
- Median fill time target: under 10 minutes

**Self-service scheduling:**
- Web-based booking for well visits (constrained: correct visit type, correct
  provider, correct duration)
- Sick visits remain phone-triaged, because a parent should not self-select a
  sick-visit slot without a triage conversation

**Fee handling:**
- The $25 fee becomes largely unnecessary. Where applied, it is applied
  automatically and consistently rather than case-by-case at the front desk.

#### Reference Implementation

```
Schedule sync:  FHIR Appointment, polled every 5 min or webhook if supported
Reminder engine: Cron + rules table (cadence by appointment type)
Messaging:      Twilio (BAA available) or the patient-communication platform
Response handling: Twilio webhook → deterministic intent match on the tap
                payload (NOT free-text NLP — use structured buttons/links)
Waitlist:       PostgreSQL table:
                (patient_id, desired_provider, visit_type, earliest_ok,
                 latest_ok, priority, added_at, notify_channel)
Backfill:       On slot release → query eligible → rank → parallel notify
                top 5 → first accept wins → atomic booking with row lock
Self-schedule:  Constrained booking widget writing to FHIR Appointment
Opt-out:        Global suppression list, enforced pre-send
Analytics:      No-show rate by provider / day / visit type / lead time
```

**The atomic booking lock matters.** Blasting a slot to five people and letting
two book it is worse than not blasting it at all.

#### Buy vs. Build

**This is the clearest buy in the document.** The category is mature, the products
are good, and building it is a poor use of engineering time.

| Vendor | Pricing (verified Aug 2026) | Assessment |
| --- | --- | --- |
| **Weave** | From ~$249/mo | All-in-one: replaces the phone system, adds texting, reminders, payments, reviews. Strong fit for a practice with an aging phone setup. |
| **Klara (ModMed)** | Quote-based; reported ~$300/mo | Strong patient-message routing and triage inbox. Deepest inside ModMed. |
| **Spruce Health** | Basic $24/user/mo; Communicator $49/user/mo. BAA included. 14-day trial, no card. | **Best price transparency in the category.** Communicator adds phone trees, scheduled routing, bulk messaging, API access. A 7-user practice on Communicator ≈ $343/mo. Includes HIPAA-grade e-fax — **relevant to I-06.** |
| **Curogram** | From ~$125/mo | 20+ EMR integrations, SOC 2 Type II, bi-directional API writing intake and insurance data into structured EHR fields |
| **OhMD** | Free tier available; flat monthly paid tiers | Genuinely usable free tier for very small deployments. Good evaluation vehicle. |
| **Luma Health** | Quote-based, typically $500+/mo | Best-in-class waitlist and recall mechanics specifically |
| **Solutionreach** | Quote-based | Broadest EHR integration list (400+) |
| **DIY: Twilio + custom** | ~$150/mo runtime + build | Only justified if EHR integration blocks every vendor |

**Recommended: Spruce Communicator or Curogram**, on price transparency,
BAA-included posture, and the e-fax overlap with I-06. Budget **$350/mo =
$4,200/yr**.

**Critical vendor question before signing:** does it integrate with your EHR's
scheduling API bi-directionally? A reminder platform that cannot see the schedule
in real time and cannot write a cancellation back is a mailing list, not an
automation.

#### Complexity

**L1** if bought. **L2** if the EHR requires custom integration work.

#### Time to Value

- Week 1: vendor selection, BAA executed, trial started
- Week 2: EHR schedule integration, reminder cadence configured
- Week 3: staff training, patient consent language added to intake
- Week 4: reminders live
- Weeks 5–6: waitlist backfill live
- Weeks 7–8: self-scheduling for well visits
- **Production: week 4 for the majority of the benefit.**

#### Quantified Benefit

```
ASSUMPTIONS
  Annual encounters (kept appointments)                      30,500
  Baseline no-show rate                                          9%  [EST]
  Implied scheduled appointments             30,500 / 0.91 =  33,516
  Annual no-shows                                             3,016

  Published reduction range for automated reminder systems  20–40%
  Applied reduction (conservative, low end +5)                  25%
  No-shows prevented                                            754

REVENUE FROM PREVENTED NO-SHOWS
  754 × $140                                            =  $105,560
  → Discounted 30% for slots that would have been backfilled
    anyway or rebooked into an already-full schedule
  754 × $140 × 0.70                                     =   $73,892

WAITLIST BACKFILL OF GENUINE CANCELLATIONS
  Annual cancellations with adequate notice                   2,400  [EST]
  Currently backfilled (manual, ad hoc)                         25%  =   600
  Post-automation backfill rate                                 65%  = 1,560
  Additional slots filled                                       960
  Marginal contribution per slot                                $140
  960 × $140                                            =  $134,400
  → Heavily discounted: many of these families would have
    been seen anyway at a later date, so this is acceleration
    not net-new. Applied 35% net-new factor.
  960 × $140 × 0.35                                     =   $47,040
  → Further discounted 50% for conservatism             =   $23,520

FRONT DESK TIME — MANUAL REMINDER CALLS ELIMINATED
  Confirmation calls currently attempted per day                 25  [EST]
  Minutes per call attempt (incl. voicemail)                2.0 min
  25 × 2 × 305 = 15,250 min = 254.2 hrs × $28           =    $7,118

FRONT DESK TIME — MANUAL BACKFILL CALLING ELIMINATED
  Backfill call attempts per day                                 10  [EST]
  Minutes per attempt                                       2.5 min
  10 × 2.5 × 305 = 7,625 min = 127.1 hrs × $28          =    $3,559

AFTER-HOURS CANCELLATION CAPTURE
  No-shows attributable to inability to cancel after hours      12%  of 3,016 = 362
  Converted to timely cancellations                             60%          = 217
  Of those, backfilled at 65%                                                = 141
  141 × $140                                            =   $19,740
  → Already partially captured above. Counted at 25% to avoid
    double-counting.                                    =    $4,935

$25 FEE — REDUCED NEED, IMPROVED COLLECTION
  Fees currently assessed but waived/uncollected                 $2,800/yr  [EST]
  → Treated as $0. The goal is fewer no-shows, not more fees.

─────────────────────────────────────────────────────────────────
  TOTAL MODELED ANNUAL BENEFIT                           $113,024
  (Headline table uses a further-discounted $96,000 to stay
   conservative on the largest single number in the plan)
  Annual cost                                             ($4,200)
  NET ANNUAL BENEFIT (headline basis)                     $91,800
  One-time cost (integration + setup)                     ($3,000)
  YEAR 1 NET                                              $88,800
  YEAR 2+ NET                                             $91,800
─────────────────────────────────────────────────────────────────
```

**Sensitivity — this is the number most worth stress-testing:**

| Baseline No-Show Rate | Reduction Achieved | Annual Benefit (all lines) |
| --- | --- | --- |
| 5% | 20% | $48,900 |
| 7% | 25% | $79,100 |
| **9% (base)** | **25%** | **$113,024** |
| 9% | 35% | $141,600 |
| 12% | 30% | $172,300 |

**The entire case rests on one measurable number the practice already has.**
Pull the no-show rate from the EHR before funding anything. It takes one report.

#### Risks & Controls

| Risk | Severity | Control |
| --- | --- | --- |
| TCPA exposure on automated SMS | **High** | Documented express consent at registration; clear opt-out in every message honored immediately; retain consent records; separate consent for appointment reminders (generally permissible) from marketing (not) |
| Message fatigue | Medium | Cap total messages per family per week across all initiatives — this is why a shared suppression and frequency-cap layer matters when I-02 and I-07 both send |
| Self-scheduling misuse (sick child booked into a well slot) | Medium | Constrain self-scheduling to well visits only; visit-type and duration enforced server-side; sick visits stay phone-triaged |
| Waitlist blast books two patients into one slot | Medium | Atomic booking with row-level lock; first-accept-wins tested under load |
| Reduced phone contact weakens the relationship | Low | The practice's differentiator is that a doctor answers the phone 24/7 for clinical questions. Automating *scheduling* logistics protects that capacity for clinical conversations rather than replacing them. |

#### Proposal

This initiative has the largest modeled return in the document and the smallest
implementation footprint. It is a purchase decision, not an engineering project.

The strategic argument goes beyond the arithmetic. The practice's genuine
competitive advantage — the thing it advertises, the thing patients write reviews
about — is human accessibility. Free phone calls. A physician reachable at any
hour. Walk-in sick hours every weekday morning. Weekend and holiday sessions.

That model consumes an enormous amount of phone capacity, and right now that
capacity is being spent on *scheduling logistics*: confirming appointments,
cancelling appointments, and trying to fill the gaps. Every minute spent on
"can I move my Tuesday to Thursday" is a minute not spent on the clinical
conversation the practice built its reputation on.

Automating scheduling does not make the practice less human. It reallocates human
attention from calendar management to medicine. That is the argument to make to
partners who are, correctly, protective of the practice's high-touch identity.

**Recommendation: fund it first. Start week one, alongside I-05.** It requires no
custom development, has the best return, and produces a visible result within
thirty days — which is exactly what a program like this needs to build internal
support for the harder builds that follow.


---

### I-08 — Vaccine Cold Chain Telemetry

#### Current State

A pediatric practice holds a large, expensive, temperature-sensitive vaccine
inventory. CDC storage and handling requirements call for continuous monitoring
with a digital data logger and twice-daily minimum/maximum temperature
documentation for every storage unit.

In most small practices this means: a staff member walks to the refrigerator and
the freezer twice a day, reads a thermometer, writes two numbers on a paper log
taped to the door, and initials it. If a reading is out of range, they are
expected to notice, act, and escalate.

I-CARE itself provides a temperature log feature for tracking and reporting
vaccine storage by appliance, which suggests the practice may already have a
digital path available and unused.

#### Failure Modes

| Failure | Consequence |
| --- | --- |
| Excursion overnight or over a weekend | Discovered Monday morning. Entire inventory potentially compromised. |
| Excursion during the Sunday/holiday session when staffing is minimal | Same, with fewer people watching |
| Log not filled in | Compliance gap; if the practice participates in any publicly funded vaccine program, a documented gap can jeopardize participation |
| Log filled in retroactively from memory | Falsified record — a serious finding in an audit |
| Door left ajar | The most common real-world excursion cause, and invisible to twice-daily manual checks |
| Compressor failure | Full inventory loss |

A single compromised pediatric vaccine refrigerator represents **$15,000 to
$40,000** of inventory. Combination pediatric vaccines are among the most
expensive routinely stocked pharmaceuticals in ambulatory medicine.

#### Does This Need an LLM?

**No. Emphatically no.**

This is a sensor, a threshold, and an alert. Introducing an LLM into a
temperature-monitoring path would add latency, cost, and a failure mode to a
system whose entire value is deterministic reliability.

The only place any model belongs is optional predictive maintenance — detecting
compressor degradation from cycle-time drift before failure. That is classical
time-series anomaly detection, not generative AI, and it is a phase-two nicety.

**Include this initiative in an "AI program" specifically to demonstrate the
discipline of not using AI where it does not belong.** A proposal that reaches
for a language model in every section is not a technology strategy; it is a
shopping list.

#### Target State

1. Wireless digital data loggers in every storage unit, continuous logging at
   5-minute intervals.
2. Door-open sensors on each unit.
3. Cloud dashboard with full history, exportable for audit.
4. Automatic min/max daily records — no human transcription, no initials, no
   paper.
5. Tiered alerting:
   - **Warning** (approaching range boundary) → SMS to on-site staff
   - **Excursion** (out of range) → SMS + call to on-site staff and office manager
   - **Sustained excursion** (out of range > 15 min) → escalate to physician
     partner and practice owner, 24/7, including nights and weekends
   - **Door open > 3 min** → immediate SMS
   - **Sensor offline > 30 min** → alert (a dead sensor is an unmonitored fridge)
6. Excursion response workflow auto-launches: quarantine label generated,
   manufacturer contact list surfaced, affected lot numbers listed, documentation
   template pre-filled.
7. Monthly compliance report auto-generated and archived.

#### Reference Implementation

Purchase, do not build. Vendors in this category:

```
Hardware:     Wireless data loggers with NIST-traceable calibration certificates
              (required for CDC compliance — verify the certificate is included
              and note the recalibration interval, typically 1–2 years)
Sensors:      One per storage unit + one door sensor per unit
Connectivity: Wi-Fi or cellular. Prefer cellular for the units that matter —
              a Wi-Fi outage during a holiday weekend defeats the entire system.
Alerting:     Vendor-native, configured to escalate to personal mobile numbers
Battery:      Verify battery life and low-battery alerting. A dead logger is
              worse than no logger because it creates false confidence.
```

#### Buy vs. Build

| Vendor | Typical Cost | Notes |
| --- | --- | --- |
| **SensoScientific** | ~$40–$70/sensor/mo | Purpose-built for vaccine storage, NIST-traceable, strong compliance reporting |
| **Monnit / Sensor Cloud** | ~$300–$600 hardware + ~$20–$40/mo | Lower cost, more configuration required |
| **LogTag with cloud** | ~$150–$300 per logger + ~$10/mo | Widely used, budget-friendly |
| **TempGenius, Rees Scientific** | Higher | Enterprise-oriented |
| **I-CARE temperature log** | **$0** | Already available if enrolled. **Digital record-keeping, but does not provide continuous sensing or alerting.** Complements hardware, does not replace it. |

**Recommended budget: 2 units × 1 temperature sensor + 1 door sensor each,
cellular-connected, with NIST calibration. ~$50–$100/mo all-in, plus roughly
$600–$1,200 one-time hardware.**

#### Complexity

**L1.** Unbox, mount, configure alerts, test. One afternoon.

#### Time to Value

**Two weeks**, most of which is shipping.

#### Quantified Benefit

```
ASSUMPTIONS
  Storage units                                                   2
  Manual readings per day (2 units × 2 readings)                  4
  Minutes per reading incl. logging and initialing              2.0
  Days per year (every day, including closed days —
    someone still checks, or should)                            365

STAFF TIME ELIMINATED
  4 × 2 min × 365 = 2,920 min = 48.7 hrs × $33          =    $1,607

EXCURSION LOSS AVOIDANCE (expected value)
  Probability of an undetected excursion causing inventory
  loss in a given year, with manual twice-daily monitoring     25%  [EST]
  Probability with continuous monitoring + alerting             3%
  Reduction in probability                                     22%
  Average inventory value at risk per event            $20,000  [EST]
  0.22 × $20,000                                        =    $4,400

REVACCINATION AND PATIENT-CONTACT COST AVOIDANCE
  If an excursion occurs, patients dosed from affected lots
  must be identified and potentially revaccinated.
  Expected administrative + clinical cost per event      $3,500  [EST]
  0.22 × $3,500                                         =      $770

COMPLIANCE / AUDIT POSTURE
  Value of a continuous, tamper-evident, exportable
  temperature record vs. a paper log                    Not monetized
  → If the practice participates in any publicly funded
    vaccine program, documented storage compliance is a
    condition of participation. Treat as risk mitigation.

─────────────────────────────────────────────────────────────────
  TOTAL MODELED ANNUAL BENEFIT                             $6,777
  Annual cost                                              ($900)
  NET ANNUAL BENEFIT                                       $5,877
  One-time hardware (midpoint)                             ($900)
  YEAR 1 NET                                               $4,977
  YEAR 2+ NET                                              $5,877
─────────────────────────────────────────────────────────────────
```

**On the 25% baseline excursion probability:** this is the least defensible
assumption in the document and deserves scrutiny. It is drawn from the observation
that manual twice-daily checks leave roughly 22 hours per day unmonitored, and
that door-ajar and compressor-degradation events are the dominant failure modes
and are both invisible to spot checks. If the practice has never had an excursion
in fifty years, argue the number down. Even at a 5% baseline the initiative
returns $2,100 annually against $900 of cost and still clears.

#### Risks & Controls

| Risk | Severity | Control |
| --- | --- | --- |
| Alert fatigue from over-sensitive thresholds | Medium | Tune warning bands during a 30-day baseline period before enabling escalation |
| Alerts sent to a phone nobody carries on weekends | **High** | Explicit on-call rotation for cold chain alerts; test monthly with a deliberate trigger |
| Sensor battery dies silently | **High** | Low-battery alerting mandatory; sensor-offline alerting mandatory |
| Calibration lapses | Medium | Calendar the recalibration date at install; NIST-traceable certificate retained for audit |
| Staff stop looking at the fridge entirely | Low | Continuous monitoring is strictly better than spot checks; but retain a weekly visual inspection for physical issues sensors cannot detect (frost buildup, overcrowding, expired stock) |

#### Proposal

This is the cheapest initiative in the plan and the one with the worst
consequences if skipped.

The core argument is exposure asymmetry. The practice spends roughly $900 a year
to protect a $20,000-to-$40,000 inventory that is currently monitored by a human
walking past it twice a day. Between the Friday evening check and the Saturday
morning check there are roughly fourteen unmonitored hours. Between Saturday's
session and Sunday's there are more. A compressor that fails at 9:00pm on a
Friday is discovered at 9:30am on Saturday.

There is a second, less obvious argument. Every other initiative in this document
generates a benefit that has to be argued for with modeled assumptions. This one
generates a **record** — a continuous, timestamped, tamper-evident, exportable
temperature history. If the practice is ever audited on vaccine storage, that
record is the difference between a clean finding and a paper log with someone's
initials in it.

**Recommendation: fund it, sequence it sixth, but treat it as non-optional.**
It is small enough that it should not compete with anything else for budget.

---

### I-09 — Eligibility Verification & Denial Prevention

#### Current State

The practice's publicly posted insurance list is dated **January 2016** and
carries the note that the list changes often and patients should call to verify.
That is a candid admission that payer status is not systematically maintained.

The practice accepts roughly 36 commercial plans, does not accept Medicaid or
Public Aid, and states that out-of-network patients are charged a rate comparable
to negotiated rates with a 10% discount for payment at time of service. It also
notes the common pediatric billing surprise: a well visit that addresses a new or
pre-existing problem may generate an additional sick-visit charge.

Current verification is presumably: front desk collects an insurance card, keys
or scans it, and either checks eligibility manually through a payer portal or
does not check at all and finds out when the claim is denied.

#### Failure Modes

| Failure | Consequence |
| --- | --- |
| Coverage terminated and not caught | Claim denied. Balance billed to a family who believed they were covered. |
| Wrong plan on file | Denial, rework, delayed payment |
| Copay not collected at time of service | Collection cost rises sharply once the patient leaves the building |
| Out-of-network status not identified pre-visit | Family receives an unexpected bill; relationship damage |
| Denial worked late or not at all | Timely filing limits expire; revenue permanently lost |
| Public list is a decade stale | Families arrive believing they are in-network when they are not |

#### Does This Need an LLM?

**Mostly no. The core is a solved EDI problem.**

| Sub-task | Tool |
| --- | --- |
| Check eligibility before a visit | **X12 270/271 real-time eligibility transaction.** Standardized, decades old, supported by every clearinghouse. Deterministic. |
| Parse the 271 response | Deterministic parser. The format is specified. |
| Read an insurance card photo | OCR + extraction — **LLM helps here.** Card layouts vary enormously across 36 payers; a rules-based parser is fragile. |
| Determine whether a plan is accepted | Lookup against a maintained payer table. Deterministic. |
| Classify a denial reason | **LLM — reasonable fit.** Denial reason codes are standardized but the accompanying free text is not, and payer-specific quirks are numerous. |
| Draft an appeal letter | **LLM — good fit.** Templated, evidence-cited, human-reviewed. |
| Keep the public payer list current | Deterministic sync from the payer table to the website |

**Verdict:** buy a clearinghouse eligibility service. Use an LLM only for card
OCR and denial triage.

#### Target State

**T-3 days before every appointment:**
- Batch 270 eligibility check for every scheduled patient
- 271 response parsed: active/inactive, plan name, copay, deductible remaining,
  in-network status
- Results written to the encounter record

**Exception handling:**
- Inactive coverage → front desk work queue with a pre-drafted outreach message
- Plan changed → flag for card re-capture at check-in
- Out-of-network detected → **proactive call before the visit**, not a surprise
  bill after. This alone justifies the initiative on patient-relationship grounds.
- Copay amount surfaced to the front desk so it is collected at the desk

**Card capture:**
- Parent photographs front and back of the card via a secure link
- OCR + LLM extraction populates payer, member ID, group number
- Front desk confirms rather than keys

**Denial management:**
- Denials ingested from the 835 remittance
- Classified by root cause: eligibility, coding, authorization, timely filing, other
- Eligibility-caused denials trend-tracked back to the verification process
- Appeal letters drafted for reviewable categories

**Payer list maintenance:**
- Contracted payer table becomes the single source of truth
- Public website list generated from it automatically
- Quarterly review task auto-created

#### Reference Implementation

```
Eligibility:    Clearinghouse real-time 270/271 API
                (Availity, Change/Optum, Waystar, Office Ally, pVerify)
Batch trigger:  Cron T-3 days against FHIR Appointment
Parsing:        X12 271 parser (library; do not hand-roll)
Card OCR:       Textract + Bedrock Claude extraction, strict JSON schema
Payer table:    PostgreSQL, versioned, with effective dates
Website sync:   Generated JSON → WordPress via REST API
Denials:        835 ERA ingestion → CARC/RARC mapping →
                Bedrock classification of free-text remarks
Appeals:        Bedrock drafting from a template library → human review → send
Dashboard:      Denial rate by root cause, by payer, trending
```

#### Buy vs. Build

| Option | Product | Cost | Assessment |
| --- | --- | --- | --- |
| **Buy** | EHR/PM built-in eligibility | Often included or ~$0.10–$0.35/check | **Check first.** Nearly every modern practice-management system has this and many practices do not turn it on. |
| **Buy** | pVerify, Availity Essentials | $0.08–$0.25/transaction, or ~$150–$400/mo | Purpose-built, well-documented APIs |
| **Buy** | Waystar, Change Healthcare/Optum | $300–$800/mo | Full RCM suite; more than needed |
| **Buy** | Curogram / Weave insurance capture | Included in patient-comm platform | Card capture only, not eligibility |
| **Build** | Denial classification + appeal drafting layer | ~$50/mo Bedrock | Thin layer on top of purchased eligibility |

```
Eligibility checks: 33,516 appointments/yr × $0.15                = $5,027/yr
Bedrock (card OCR + denial classification + appeal drafting)      =   $380/yr
Compute (shared)                                                  =   $300/yr
                                                            Total ≈ $5,707/yr ≈ $476/mo
```

> If the practice's PM system includes eligibility, this drops to roughly
> **$60/mo** and the initiative becomes trivially positive.

#### Complexity

**L2.** The eligibility piece is a purchased API. The denial classification layer
is a modest build. The payer-table-to-website sync is an afternoon.

#### Time to Value

- Week 1: confirm whether eligibility is already available in the PM system
- Weeks 2–3: clearinghouse contract, API integration, batch job
- Week 4: exception queue and front-desk workflow
- Weeks 5–6: card capture
- Weeks 7–9: denial ingestion and classification
- Week 10: payer table and website sync
- **Production: week 4 for the core eligibility benefit.**

#### Quantified Benefit

```
ASSUMPTIONS
  Annual claims (≈ encounters)                               30,500
  Baseline denial rate                                           6%  [EST]
  Annual denials                                              1,830
  Share of denials attributable to eligibility issues           35%  [EST] = 641
  Share of eligibility denials preventable by pre-verification  70%          = 449

DENIALS PREVENTED — DIRECT REVENUE
  Denials that are never successfully appealed and become
  write-offs or bad debt                                        30%  [EST]
  449 × 0.30 = 135 claims permanently lost, now prevented
  135 × $140                                            =   $18,900

REWORK ELIMINATED
  449 prevented denials × 12 min rework each = 5,388 min
  = 89.8 hrs × $28                                      =    $2,514

POINT-OF-SERVICE COPAY COLLECTION
  Copays not collected at desk, currently                       18%  [EST]
  Encounters with a copay                                       75%  = 22,875
  Uncollected at desk                                                = 4,118
  Average copay                                                 $30  [EST]
  Post-visit collection rate                                    62%
  Post-visit collection cost per account                      $4.50
  Improvement in desk collection (18% → 7%)                    11%  = 2,516 accounts
  Revenue recovered: 2,516 × $30 × (1 - 0.62)          =   $28,682
  → Discounted 50% (many would eventually pay)         =   $14,341
  Collection cost avoided: 2,516 × $4.50               =   $11,322
  → Discounted 50%                                     =    $5,661

CARD DATA ENTRY TIME
  New/changed cards annually                                  4,200  [EST]
  Manual keying time                                        2.5 min
  Post-OCR confirmation time                                0.8 min
  4,200 × 1.7 min = 7,140 min = 119 hrs × $28          =    $3,332

OUT-OF-NETWORK SURPRISE PREVENTION
  OON patients identified pre-visit rather than post-bill        90  [EST]
  Avoided write-off / goodwill adjustment per event       $75  [EST]
  90 × $75                                              =    $6,750
  → Discounted 50%                                      =    $3,375

APPEAL DRAFTING TIME
  Appeals filed annually                                        320  [EST]
  Drafting time current                                      22 min
  Post-automation (review + send)                             7 min
  320 × 15 min = 4,800 min = 80 hrs × $28              =    $2,240

─────────────────────────────────────────────────────────────────
  TOTAL MODELED ANNUAL BENEFIT                            $50,363
  (Headline table uses a heavily discounted $22,000, because
   several lines here overlap with front-desk process change
   rather than automation per se)
  Annual cost                                             ($5,707)
  NET ANNUAL BENEFIT (headline basis)                     $16,293
  One-time build                                          ($7,000)
  YEAR 1 NET                                               $9,293
  YEAR 2+ NET                                             $16,293
─────────────────────────────────────────────────────────────────
```

#### Risks & Controls

| Risk | Severity | Control |
| --- | --- | --- |
| Eligibility response misread → patient wrongly told they are not covered | **High** | Never auto-communicate a coverage denial to a patient. Route to a human who calls the payer to confirm before any patient contact. |
| Card OCR misreads a member ID | Medium | Front desk confirms every extracted field against the card image side by side; never auto-submit |
| Automated appeal letter contains an error | Medium | Human review and signature mandatory; appeals are legal correspondence |
| Clearinghouse outage | Low | Degrade to manual verification; do not block check-in |
| Payer table drifts stale again | Medium | Automated quarterly review task with a named owner; website generated from the table so staleness is visible |

#### Proposal

The January 2016 date on the public insurance list is the tell. It is not
negligence — it is what happens when maintaining a payer list is nobody's
specific job and there is no system that makes staleness visible.

That single stale artifact represents a real cost. A family checks the website,
sees their plan, books an appointment, arrives, and discovers the practice
dropped that plan six years ago. Somebody has to have that conversation at the
front desk with a sick child in the waiting room. Some portion of those visits
get written off. All of them damage the relationship.

The pediatric-specific angle strengthens the case. The practice already
acknowledges a common billing surprise: a well visit that addresses a problem may
generate an additional sick-visit charge. That is correct coding and it is also
the most frequent source of pediatric billing complaints nationally. A system
that surfaces the family's actual copay, deductible status, and network status
*before* the visit lets the front desk set expectations rather than deliver
surprises.

The honest framing for this initiative: **it is less an AI project than a
revenue-cycle hygiene project with two small AI components.** It is included
because it is genuinely worth doing and because a program that only funds the
exciting work will leave the boring money on the table.

**Recommendation: fund it, sequence it seventh.** But check the PM system first —
if eligibility verification is already licensed and simply unused, most of this
benefit is available for the cost of turning it on.

---

### I-10 — Standing Order Digitization & Delegation Audit

> **The initiative with the smallest dollar return and the largest downside
> protection.**

#### Current State

Under 225 ILCS 60/54.2, a physician in an office setting may delegate patient
care tasks to an unlicensed person who possesses appropriate training and
experience, provided a licensed health care professional is on site. The
delegated task must fall within the scope, education, training, or experience of
the delegating physician. Critically, the statute permits delegation **by any
means — oral, written, electronic, standing orders, protocols, guidelines, or
verbal orders.**

That flexibility is a gift and a trap. It is a gift because a practice can
operate efficiently with standing orders rather than requiring a physician
signature for every vaccine. It is a trap because the same flexibility means many
practices operate on **oral and customary delegation** that exists nowhere in
writing.

The likely current state: experienced MAs know what they are authorized to do
because they have done it for years. New hires learn by apprenticeship. There may
be a binder. The binder may be out of date. If a regulator, a plaintiff's
attorney, or an insurer asked the practice to produce the written delegation
authorizing a specific MA to administer a specific vaccine on a specific date,
the answer would likely require assembling it after the fact.

#### Failure Modes

| Failure | Consequence |
| --- | --- |
| No written delegation record | Under audit or litigation, the practice cannot demonstrate lawful delegation |
| Delegation not scoped to individual competency | The statute requires "appropriate training and experience" — for *that person* |
| Training and competency records not linked to delegated tasks | Cannot prove the MA was qualified for what they did |
| New hire performs a task before competency is verified | Direct unauthorized-practice exposure |
| Standing orders drift from current clinical guidelines | Clinically outdated protocol executed as authorized |
| Physician-on-premises requirement not evidenced | The statute requires a licensed professional on site; nothing records whether one was |
| **Section 54.2 sunsets January 1, 2027** | Statutory framework may change; a practice with no written baseline cannot adapt systematically |

That last row is the reason this initiative is in the document at all.

#### Does This Need an LLM?

**Barely — and where it does, only as an authoring aid.**

| Sub-task | Tool |
| --- | --- |
| Store and version standing orders | Database. Deterministic. |
| Map each order to competency requirements | Data model. Deterministic. |
| Track individual staff competency | Database. Deterministic. |
| Enforce that an MA may only execute orders they are competent for | Application logic. Deterministic. |
| Record physician-on-premises at time of execution | Timestamp + roster. Deterministic. |
| Produce an audit report | SQL. Deterministic. |
| **Draft a standing order from a clinical guideline** | **LLM — authoring aid only.** A physician reviews, edits, and signs. |
| **Flag when a standing order may have drifted from current AAP/ACIP guidance** | **LLM + retrieval — reasonable fit.** Compare the order text against the current published schedule; surface differences for physician review. Never auto-update. |
| **Draft competency checklists** | LLM authoring aid |

#### Target State

**A structured delegation register:**

```
standing_order
  ├─ id, title, version, effective_date, retired_date
  ├─ delegating_physician_id
  ├─ clinical_content (the actual protocol)
  ├─ required_competencies[]
  ├─ required_supervision_level
  ├─ source_guideline_reference (e.g. ACIP 2026 schedule)
  ├─ review_due_date
  └─ signature (physician e-sign, timestamped)

staff_competency
  ├─ staff_id, competency_id
  ├─ verified_by (licensed verifier), verified_date
  ├─ evidence (training cert, observed demonstration, exam)
  └─ expiry_date

execution_log
  ├─ staff_id, standing_order_id, order_version
  ├─ patient_id, timestamp
  ├─ licensed_professional_on_site_id      ← the 54.2 requirement, evidenced
  └─ outcome
```

**Behavior:**
- An MA opening a task sees only orders they are currently competent to execute
- Executing an order writes an execution log entry capturing which licensed
  professional was on site
- Competency expiring in 30 days generates a task
- Standing order review date reached generates a task to the delegating physician
- Guideline-drift check runs quarterly against current ACIP/AAP publications and
  surfaces candidate discrepancies for physician review
- One-click audit report: "produce every delegation, competency record, and
  execution for MA X between date A and date B"

#### Reference Implementation

```
Data layer:      PostgreSQL, append-only execution log, full version history
Application:     Internal web app (React), role-based access
E-signature:     Physician signs each standing order version
Roster feed:     Daily provider schedule → who was on site, when
Competency:      Manual entry by a licensed verifier; certificate upload to S3
Drift check:     Quarterly cron → fetch current ACIP schedule →
                 Bedrock Claude compares standing order text against it →
                 outputs candidate discrepancies with citations →
                 physician review queue. NEVER auto-applies a change.
Authoring aid:   Bedrock Claude drafts new standing orders from a guideline
                 reference; physician edits and signs. The model's output is
                 a draft in a review UI, never a published order.
Reporting:       Parameterized SQL reports, PDF export
Audit:           Immutable log; no deletes, only version supersession
```

**The design decision that makes this future-proof:** delegation rules live in
**configuration, not code.** When Section 54.2 sunsets on January 1, 2027 — or
when replacement legislation establishes a different framework, possibly
including formal certification requirements — the practice changes a rules
configuration and a competency schema, not an application. That is the entire
architectural argument for building this before the statutory deadline rather
than after.

#### Buy vs. Build

There is no good off-the-shelf product for this. It is genuinely niche.

| Option | Cost | Assessment |
| --- | --- | --- |
| **Do nothing** | $0 | Current state. Works until it does not. |
| **Paper/SharePoint binder** | ~$0 | Better than nothing. No enforcement, no execution linkage, no audit query. |
| **LMS with competency tracking** (Relias, HealthStream) | $8–$20/user/mo | Tracks training well. Does not link competency to standing orders or to execution events. |
| **Build** | ~$50–$150/mo runtime | The only option that closes the loop |

```
Bedrock (quarterly drift checks + occasional authoring)   =   $120/yr
Compute + storage (shared)                                =   $600/yr
E-signature (shared with I-01)                            =     $0
                                                    Total ≈   $720/yr ≈ $60/mo
```

#### Complexity

**L2.** No external integrations beyond a roster feed. Mostly a well-designed
data model and a clean UI.

#### Time to Value

- Weeks 1–3: inventory existing standing orders (this is the hard part, and it is
  organizational, not technical — someone must actually collect what exists)
- Weeks 4–6: data model, application, e-signature
- Weeks 7–8: competency records backfilled for current staff
- Week 9: execution logging wired into clinical workflow
- Week 10: drift check and reporting
- **Production: week 10.**

#### Quantified Benefit

```
ASSUMPTIONS
  Annual encounters                                          30,500
  Encounters involving a delegated clinical task (vaccine,
  injection, POC test, phlebotomy, screening)                   70%  = 21,350

WORKFLOW TIME — REDUCED MA↔PHYSICIAN CLARIFICATION
  Current: MA must locate or verbally confirm authorization
  Interruptions per day requiring physician clarification         14  [EST]
  Minutes per interruption (MA side)                          1.5
  14 × 1.5 × 305 = 6,405 min = 106.8 hrs × $33          =    $3,524
  Minutes per interruption (physician side)                   1.0
  14 × 1.0 × 305 = 4,270 min = 71.2 hrs × $147 × 0.50   =    $5,233

ONBOARDING ACCELERATION
  New clinical hires per year                                     2  [EST]
  Current time to independent practice                      12 weeks
  With structured competency pathway                        9 weeks
  Productivity gained per hire                               3 weeks
  Value of partial productivity recovered (50% of
  3 weeks × 40 hrs × $33)                                 =  $1,980/hire
  2 × $1,980                                            =    $3,960

AUDIT / SURVEY PREPARATION
  Preparation hours currently required to assemble
  delegation evidence for any audit, survey, or
  credentialing review                                       24 hrs  [EST]
  Post-automation (run a report)                            1.5 hrs
  Frequency                                              1.5×/year
  22.5 hrs × 1.5 × $33                                  =    $1,114

STANDING ORDER MAINTENANCE
  Annual review of orders against current guidelines
  Current manual effort                                      16 hrs  [EST]
  Post-automation (review flagged discrepancies)              4 hrs
  12 hrs × $147 × 0.50                                  =      $882

MALPRACTICE / REGULATORY EXPOSURE REDUCTION
  Not monetized. See proposal.

REGULATORY TRANSITION READINESS (54.2 sunset)
  Not monetized. See proposal.

─────────────────────────────────────────────────────────────────
  TOTAL MODELED ANNUAL BENEFIT                            $14,713
  (Headline table uses $16,764, which additionally credits
   1 min/encounter of pre-visit order preparation)
  Annual runtime cost                                       ($720)
  NET ANNUAL BENEFIT                                      $13,993
  One-time build (midpoint)                               ($9,000)
  YEAR 1 NET                                               $4,993
  YEAR 2+ NET                                             $13,993
─────────────────────────────────────────────────────────────────
```

#### Risks & Controls

| Risk | Severity | Control |
| --- | --- | --- |
| System creates a written record of previously undocumented non-compliance | **This is the real risk, and it must be named to leadership before starting.** | Conduct the initial inventory under attorney-client privilege with practice counsel. Remediate gaps before the system goes live. A practice that discovers a gap and fixes it is in a far better position than one that never looked — but the discovery should be privileged. |
| Overly rigid enforcement blocks legitimate clinical work | Medium | Break-glass path: any licensed professional can authorize an out-of-band task with a logged justification. Never block care. |
| LLM drift-check produces false positives that erode trust | Low | Physician review required; measure precision; tune the prompt or drop the feature if precision is poor |
| LLM drift-check misses a real guideline change | Medium | It is a supplement to, not a replacement for, the annual clinical review. Frame it that way in policy. |
| Competency records become a bureaucratic burden | Medium | Keep the schema minimal. Competency verification should take a licensed verifier under five minutes per item. |

#### Proposal

Every other initiative in this document is justified by time or money. This one
is justified by exposure.

Illinois medical assistants hold no state license. IDFPR maintains no medical
assistant credential. An MA's entire clinical authority is borrowed from the
delegating physician's license under Section 54.2. The statute permits that
delegation to be oral or customary — which means most practices operate on
delegation that exists only in institutional memory.

That works until someone asks for it in writing. And there are three specific
scenarios where someone will:

**One — litigation.** A poor outcome following a delegated task. The first
discovery request will ask what authorized the MA to perform it and what
established their competency. "Everyone knew" is not a document.

**Two — a payer or credentialing audit.** Increasingly common, increasingly
documentation-focused.

**Three — the statutory sunset.** Section 54.2 is currently scheduled to expire
on January 1, 2027, with pending legislation that could restructure medical
assistant regulation in Illinois entirely — potentially including formal
certification or registration requirements. A practice that already maintains a
structured competency register and a versioned delegation catalog adapts by
editing configuration. A practice operating on institutional memory has to build
the entire thing under a deadline, while also complying with it.

There is a workforce argument too. The practice states it offers on-the-job
training for people inexperienced in pediatrics, and that several staff started
in administrative roles and moved into clinical ones after earning certificates.
That is an admirable and unusual internal mobility pathway. It is also exactly
the situation where documented, individually-scoped competency verification
matters most, because the staff member's authority to perform a clinical task is
grounded in training the practice itself provided rather than in an external
license.

**Recommendation: fund it, sequence it ninth — but start the inventory in month
one, under privilege.** The build is cheap and late; the discovery work is the
part with a lead time, and it should begin before the 2027 deadline is close
enough to be urgent.


---

## 6. Consolidated ROI Model

### 6.1 Full Financial Summary

| # | Initiative | Annual Benefit | Annual Runtime | One-Time Build | Yr 2+ Net |
| --- | --- | ---: | ---: | ---: | ---: |
| I-01 | Automated Forms Pipeline | $31,082 | $3,710 | $23,000 | $27,372 |
| I-02 | Immunization Gap Closure | $36,987 | $2,150 | $11,000 | $34,837 |
| I-03 | Pre-Visit Chart Prep | $25,163 | $1,480 | $9,000 | $23,683 |
| I-04 | Telephone Triage Documentation | $46,299 | $5,502 | $8,000 | $40,797 |
| I-05 | Ambient Documentation | $33,551 | $9,480 | $0 | $24,071 |
| I-06 | Inbound Fax & Document Ingestion | $10,889 | $3,052 | $6,000 | $7,837 |
| I-07 | No-Show Reduction & Backfill | $96,000 | $4,200 | $3,000 | $91,800 |
| I-08 | Cold Chain Telemetry | $6,777 | $900 | $900 | $5,877 |
| I-09 | Eligibility & Denial Prevention | $22,000 | $5,707 | $7,000 | $16,293 |
| I-10 | Standing Order Digitization | $16,764 | $720 | $9,000 | $16,044 |
| | **TOTAL** | **$325,512** | **$36,901** | **$76,900** | **$288,611** |

**On the one-time total.** The $76,900 is the sum of independent build midpoints.
In practice the builds share infrastructure heavily — I-01, I-02, I-03, I-04, and
I-06 all reuse the same AWS account, the same Bedrock integration, the same OCR
layer, the same FHIR client, and the same audit schema. Building them in sequence
rather than in isolation reduces the consolidated cost to a realistic
**$52,000–$68,000 if contracted externally**, and substantially less if built
internally by someone already fluent in the stack.

### 6.2 Benefit Composition

| Category | Amount | Share |
| --- | ---: | ---: |
| Clinical staff (MA/LPN) labor recaptured | $148,700 | 46% |
| Administrative staff labor recaptured | $27,300 | 8% |
| Revenue captured (no-show, recall, backfill) | $118,400 | 36% |
| Revenue protected (denials, collections) | $25,300 | 8% |
| Loss avoidance (inventory, rework, duplicate doses) | $5,812 | 2% |
| **Total** | **$325,512** | **100%** |

**A necessary caveat on "labor recaptured."** Recovering 4,500 staff hours does
not automatically reduce payroll by $148,700. The practice will realize that
value in one of three ways, and it should decide deliberately which:

1. **Absorb growth without hiring.** The most likely and most valuable outcome.
   The practice grows its panel without adding clinical staff.
2. **Reduce overtime and improve close-out times.** Real, but smaller.
3. **Reduce headcount.** Possible, but a practice with twenty-year staff tenure
   and an internal-mobility training culture is unlikely to want this, and
   probably should not.

If the practice is not growing and does not intend to grow, discount the labor
component by 40–50% and the program still returns roughly $250,000 annually. If
it is growing, the labor component is worth its full face value and arguably more.

### 6.3 Year 1 Cash Flow (Ramp-Adjusted)

Benefits do not begin on day one. This is the realistic view:

| Period | Initiatives Live | Benefit Realized | Costs Incurred | Net |
| --- | --- | ---: | ---: | ---: |
| Month 1 | Discovery only | $0 | ($8,000) | ($8,000) |
| Months 2–3 | I-05, I-07, I-08, I-02 (I-CARE only) | $24,571 | ($16,500) | $8,071 |
| Months 4–6 | + I-02 (full) | $47,893 | ($22,000) | $25,893 |
| Months 7–9 | + I-04 | $65,819 | ($21,000) | $44,819 |
| Months 10–12 | + I-01, I-03, I-06 | $82,465 | ($17,400) | $65,065 |
| **Year 1 Total** | | **$220,748** | **($84,900)** | **$135,848** |
| **Year 2** | All ten | **$325,512** | **($36,901)** | **$288,611** |

**Cumulative break-even: month 5.**

> Year 1 benefit of $220,748 is lower than the $325,512 run rate because most
> initiatives are live for only part of the year. This is the number to quote
> to partners. Quoting the run rate as a Year 1 figure is how these programs
> lose credibility in month four.

### 6.4 Sensitivity Analysis

The three assumptions that most affect the outcome, and what happens if each is
wrong:

| Assumption | Base | Pessimistic | Optimistic | Swing |
| --- | --- | --- | --- | --- |
| No-show rate (I-07) | 9% | 5% | 12% | **±$60,000** |
| Clinical calls/day (I-04) | 120 | 60 | 200 | **±$30,000** |
| Annual encounters | 30,500 | 24,000 | 38,000 | **±$50,000** |
| MA fully loaded rate | $33/hr | $29/hr | $38/hr | ±$18,000 |
| Recall conversion (I-02) | 35% | 20% | 50% | ±$10,000 |

**Downside case — all pessimistic simultaneously: ~$186,000 annual benefit
against $36,901 of runtime cost.** The program still returns roughly 4:1 in the
worst modeled case. That robustness is the strongest single argument for funding
it, and it is more persuasive than the base case.

### 6.5 What Would Make This Fail

Stated plainly, because a proposal that does not name its own failure modes is
not credible:

1. **The EHR has no usable API.** This is the single largest risk. If the practice
   runs an unsupported or on-premise system with no FHIR interface, five of the
   ten initiatives become substantially harder and two become impractical.
   **Resolve this in week one before committing to anything.**
2. **Nobody owns it.** These systems need an accountable operator. A program with
   no named owner degrades within six months.
3. **Staff do not adopt.** An ambient scribe nobody records with, or a triage app
   nobody opens, returns exactly zero. Adoption is a management problem, not a
   technology problem, and it is the one most often underestimated.
4. **Review discipline decays.** If MAs and physicians stop reading drafts, the
   program converts from an efficiency gain into a liability. Section 10 defines
   the metrics that detect this.
5. **Scope creep into clinical decision support.** The moment someone asks the
   system to suggest a diagnosis or a disposition, the compliance posture
   collapses. This must be governed, not merely discouraged.

---

## 7. Phased Implementation Plan

### Phase 0 — Discovery & Foundation (Month 1)

**Nothing gets built until this is complete.** Every one of these blocks
downstream work.

| # | Task | Owner | Blocks |
| --- | --- | --- | --- |
| 0.1 | **Identify the EHR/PM system, version, and whether FHIR R4 APIs are licensed and enabled. Get the API pricing in writing.** | Practice Manager | I-01, I-02, I-03, I-06, I-07, I-09 |
| 0.2 | Pull the actual no-show rate from the last 12 months, by provider and visit type | Practice Manager | I-07 funding decision |
| 0.3 | Log clinical call volume for one full week, including weekend sessions | MA lead | I-04 funding decision |
| 0.4 | Count forms processed in a representative month | Front desk lead | I-01 funding decision |
| 0.5 | Confirm I-CARE enrollment status and which features are enabled | MA lead | I-02 |
| 0.6 | Count inbound fax documents for one week | Front desk lead | I-06 |
| 0.7 | Execute AWS Business Associate Addendum via AWS Artifact | Owner + counsel | All build initiatives |
| 0.8 | Inventory existing standing orders **under attorney-client privilege** | Physician partner + counsel | I-10 |
| 0.9 | Add AI/recording consent language to the intake packet and post signage | Practice Manager + counsel | I-04, I-05 |
| 0.10 | Name a Program Owner with explicit authority and dedicated time | Partners | Everything |

**Deliverable:** a one-page baseline document with real numbers replacing every
`[EST]` in Section 2, and a revised ROI model built on them.

**Cost:** ~$8,000 in internal time plus counsel review.

### Phase 1 — Quick Wins (Months 2–3)

Chosen for speed to visible result, not for magnitude. The purpose of this phase
is to make the program credible internally.

| Week | Action |
| --- | --- |
| 1 | Enable I-CARE remind/recall. Zero cost, immediate partial I-02 value. |
| 1 | Start ambient scribe trials (I-05). Check for a free native option in the EHR first. |
| 1 | Order cold chain sensors (I-08). |
| 2 | Cold chain sensors installed and alerting. |
| 2 | Patient-communication platform selected; BAA executed (I-07). |
| 3 | Ambient scribe selected after 30-day pilot; roll out to all physicians. |
| 4 | Appointment reminders live (I-07). |
| 6 | Waitlist backfill live (I-07). |
| 8 | Self-scheduling for well visits live (I-07). |

**Phase 1 exit criteria:** no-show rate down measurably; physicians report reduced
charting time; cold chain alerting tested with a deliberate trigger.

**Cost:** ~$16,500. **Benefit at exit run rate:** ~$147,400/yr.

### Phase 2 — The Highest-Value Build (Months 4–6)

| Weeks | Action |
| --- | --- |
| 1–2 | AWS environment stood up: VPC, KMS, RDS, S3, Bedrock access, audit schema |
| 1–2 | FHIR client library built and tested against the live EHR |
| 3–4 | I-04 audio capture client and transcription pipeline |
| 5–6 | I-04 structuring prompt, note template, EHR post-back |
| 5–8 | I-02 full build: panel extract, I-CARE reconciliation, CDSi forecast, huddle sheet |
| 7–8 | I-04 pilot with two MAs; prompt iteration against real edit diffs |
| 9–10 | I-02 outbound recall campaign with physician-approved templates |
| 11–12 | Both to production |

**Phase 2 exit criteria:** triage documentation time measurably down; huddle sheet
in daily use; immunization gap list shrinking month over month.

**Cost:** ~$22,000. **Cumulative run rate:** ~$230,700/yr.

### Phase 3 — The Heavy Build (Months 7–10)

| Weeks | Action |
| --- | --- |
| 1–2 | Fax number porting initiated (2-week lead time — start immediately) |
| 1–4 | I-01 ingestion, classification, template store, FHIR pull |
| 3–6 | I-06 OCR, classification, routing (reuses I-01 infrastructure) |
| 5–8 | I-01 reconciliation engine and review UI |
| 7–10 | I-03 rules tables, assembly, narrative layer |
| 9–12 | I-01 signature, delivery, notification; parallel run against manual |
| 13–14 | All three to production |

**Phase 3 exit criteria:** forms turnaround time down from days to hours; fax
sorting time down; huddle brief in daily use with a positive feedback rate.

**Cost:** ~$21,000. **Cumulative run rate:** ~$297,800/yr.

### Phase 4 — Revenue Cycle & Governance (Months 11–14)

| Weeks | Action |
| --- | --- |
| 1–4 | I-09 eligibility integration, exception queue, card capture |
| 3–6 | I-10 data model, application, e-signature, competency backfill |
| 5–8 | I-09 denial ingestion, classification, appeal drafting |
| 7–10 | I-10 execution logging wired into clinical workflow; drift check |
| 9–12 | Payer table and website sync; audit reporting |

**Phase 4 exit criteria:** eligibility-caused denials down; public payer list
accurate; a one-click delegation audit report demonstrably produces correct output.

**Cost:** ~$17,400. **Full run rate: $325,512/yr against $36,901 of runtime cost.**

### 7.1 Dependency Graph

```
Phase 0 (Discovery)
    │
    ├──► I-05 Ambient        ─── standalone, no dependencies
    ├──► I-07 Scheduling     ─── needs EHR scheduling API
    ├──► I-08 Cold Chain     ─── fully standalone
    │
    ├──► AWS Foundation
    │        │
    │        ├──► I-04 Triage Docs ────────┐
    │        │         │                   │
    │        │         └──► reused by ──► I-05 Deployment B (MA rooming)
    │        │
    │        ├──► I-02 Immunization ───────┐
    │        │         │                   │
    │        │         │   CVX matcher     │
    │        │         └──► shared with ──►│
    │        │                             │
    │        ├──► I-01 Forms ◄─────────────┘
    │        │         │
    │        │         │   OCR + LLM infra
    │        │         └──► shared with ──► I-06 Fax
    │        │                                  │
    │        │                                  │  immunization records
    │        │                                  └──► feeds ──► I-02
    │        │
    │        ├──► I-03 Chart Prep ◄─── needs I-02 CDSi engine
    │        │
    │        └──► I-09 Eligibility ─── mostly independent
    │
    └──► I-10 Standing Orders ─── needs privileged inventory from Phase 0
```

**Critical path:** Phase 0 → AWS Foundation → I-04 → I-01 → I-06. Everything else
runs in parallel.

### 7.2 Staffing

| Role | Commitment | Phase |
| --- | --- | --- |
| **Program Owner** (practice manager or physician partner) | 4 hrs/week ongoing | All |
| **Technical lead** (contract or internal) | Full-time Phases 2–4 | 2–4 |
| **MA champion** | 2 hrs/week | 1–4 |
| **Physician sponsor** | 1 hr/week; sign-off authority on all clinical templates | All |
| **Practice counsel** | ~15 hrs total | 0, and I-10 |

**The Program Owner role is not optional and cannot be a side duty for someone
already at capacity.** Every failed practice-automation program shares this root
cause.

---

## 8. Vendor & Cost Reference Table

> **All pricing verified August 2026 from public vendor materials and pricing
> aggregators. This category re-prices frequently — confirm directly with each
> vendor before contracting. Where a vendor does not publish pricing, that is
> noted rather than estimated.**

### 8.1 Ambient Documentation (I-05)

| Vendor | Published Price | BAA | Setup Time | Fit |
| --- | --- | --- | --- | --- |
| Epic AI Charting | Included | Yes | Native | ★★★★★ if on Epic |
| athenahealth native scribe | Included (since Feb 2026) | Yes | Native | ★★★★★ if on athena |
| Sunoh.ai (healow/eCW) | Contact vendor | Yes | 1–3 days | ★★★★★ if on eCW |
| Heidi Health | Free tier; Evidence Plus $40/mo; Clinician $150/mo | Yes | < 1 hr | ★★★★☆ |
| Freed | $39 / $79 / $104–119 per provider/mo | Yes | < 1 hr | ★★★★☆ |
| Twofold Health | $49/mo annual, $69/mo monthly, unlimited | Yes | < 1 hr | ★★★★☆ |
| Vero | $69/mo annual; free tier 10 encounters | Yes | < 1 hr | ★★★★☆ (PDF form fill) |
| Commure Scribe | $89/mo, $59/mo annual | Yes | < 1 hr | ★★★☆☆ |
| Nabla | ~$119–$239/clinician/mo; contract sales since 2026 | **Free tier reportedly excludes BAA** | 1–3 days | ★★☆☆☆ |
| DeepCura | $129/mo | Yes | 1–3 days | ★★★☆☆ |
| Abridge | ~$400–$600/provider/mo, enterprise | Yes | 3–6 months | ★☆☆☆☆ — wrong size |
| Microsoft Dragon Copilot | ~$400–$600+/provider/mo | Yes | 2–6 weeks | ★☆☆☆☆ — wrong size |
| AWS HealthScribe | $0.10/audio minute | Yes (AWS BAA) | Build required | ★★★★☆ for custom |

### 8.2 Patient Communication & Scheduling (I-07, I-02)

| Vendor | Published Price | BAA | Fit |
| --- | --- | --- | --- |
| Spruce Health | Basic $24/user/mo; Communicator $49/user/mo; 14-day trial | Yes | ★★★★★ — best price transparency; includes HIPAA e-fax |
| Curogram | From ~$125/mo | Yes; SOC 2 Type II | ★★★★★ — 20+ EMR integrations, bi-directional API |
| OhMD | Free tier; flat monthly paid tiers | Yes | ★★★★☆ — good evaluation vehicle |
| Weave | From ~$249/mo | Yes | ★★★★☆ — replaces phone system too |
| Klara (ModMed) | Quote-based; reported ~$300/mo | Yes | ★★★☆☆ — deepest inside ModMed |
| Luma Health | Quote-based, typically $500+/mo | Yes | ★★★★☆ — best waitlist mechanics |
| Solutionreach | Quote-based | Yes | ★★★☆☆ — broadest EHR list |
| Twilio (DIY) | ~$0.0079/SMS + carrier fees | Yes | ★★★☆☆ — build required |

### 8.3 AI Model Access

| Path | Price | BAA | Recommendation |
| --- | --- | --- | --- |
| **AWS Bedrock** | Consumption; Claude Haiku ~$0.25/$1.25 per M tokens in/out, Sonnet ~$3/$15 | **Yes — self-serve via AWS Artifact** | **★★★★★ Recommended** |
| Azure OpenAI | Consumption | Yes — Microsoft Online Services DPA | ★★★★☆ |
| Google Vertex AI | Consumption | Yes — Google Cloud BAA | ★★★★☆ |
| Anthropic API direct | Consumption | Yes — HIPAA readiness, eligible features only | ★★★☆☆ — narrower coverage, slower contracting |
| OpenAI API direct | Consumption | Yes — on request, ZDR endpoints only | ★★★☆☆ |
| Self-hosted open weights | $2,000–$10,000/mo GPU | N/A | ★☆☆☆☆ — wrong scale |
| **Any consumer chatbot** | — | **No** | **✗ Prohibited for PHI** |

### 8.4 Supporting Infrastructure

| Component | Product | Price |
| --- | --- | --- |
| OCR | AWS Textract | ~$1.50/1,000 pages (detect); ~$50/1,000 (analyze) |
| Medical ASR | AWS Transcribe Medical | $0.0175/minute |
| PHI detection | AWS Comprehend Medical | ~$0.01/100 characters |
| Orchestration | n8n (self-hosted) | $0 + hosting |
| Integration engine | Mirth Connect | $0 (open source) |
| Cloud fax | SRFax / Documo / Twilio Fax | ~$40–$90/mo |
| E-signature | DocuSign Standard | ~$40/mo |
| Eligibility | pVerify / Availity | $0.08–$0.25/transaction |
| Cold chain | SensoScientific / Monnit / LogTag | $600–$1,200 hardware + $20–$70/sensor/mo |
| Hosting | AWS (RDS + Fargate + S3, small) | ~$250–$400/mo |

### 8.5 Consolidated Monthly Run Rate at Full Deployment

```
Ambient documentation (10 physicians @ $79)              $  790
Patient communication platform                           $  350
AWS — Bedrock, Textract, Transcribe, Comprehend          $  340
AWS — hosting, storage, database, search                 $  390
Cloud fax (2 numbers)                                    $   90
E-signature                                              $   40
Eligibility transactions                                 $  419
Cold chain monitoring                                    $   75
Protocol licensing (Schmitt-Thompson, if not held)       $  150
SMS/voice (Twilio, if not bundled)                       $  117
                                                         ──────
TOTAL                                                    $2,761/mo
                                                       = $33,132/yr

Plus contingency @ 12%                                   $  331/mo
                                                         ──────
BUDGET                                                   $3,092/mo
                                                       = $37,104/yr
```

---

## 9. Governance, Policy & Risk Register

### 9.1 Required Written Policies

Before any initiative touches PHI, the practice needs these on paper. They are
short documents, not projects, but their absence is what an auditor finds first.

| Policy | Contents | Owner |
| --- | --- | --- |
| **AI Acceptable Use** | Which tools are approved; explicit prohibition on consumer chatbots for any PHI; what staff may and may not paste anywhere | Program Owner |
| **AI Vendor Management** | BAA required before any PHI; annual review; the due-diligence checklist in Appendix C | Practice Manager |
| **Human Review Standard** | Which outputs require review; who may review what; what constitutes adequate review | Physician Sponsor |
| **Recording & Consent** | Two-party consent under Illinois law; signage; verbal disclosure script; documentation; retention and deletion | Counsel |
| **Audit & Monitoring** | What is logged; retention period; who reviews; escalation | Program Owner |
| **Incident Response** | AI-specific addendum to the existing breach response plan | Practice Manager |
| **Delegation & Competency** | The I-10 register; competency verification standard; break-glass procedure | Physician Sponsor |

### 9.2 Audit Log Minimum Fields

Every AI inference touching PHI writes an immutable record:

```
{
  timestamp_utc, user_id, patient_id_hash, initiative_id,
  model_provider, model_id, model_version,
  prompt_template_id, prompt_template_hash,
  input_token_count, output_token_count,
  confidence_score,
  human_reviewed: bool, reviewer_id, review_timestamp,
  edit_distance_draft_to_final,
  action_taken: enum[accepted, edited, rejected, escalated],
  source_ip
}
```

Note what is **not** logged: the prompt payload and the completion. Logging full
PHI-bearing text into an observability platform creates a second copy of the
record with weaker controls. Hashes and metadata are sufficient for audit; the
underlying clinical content lives in the chart where it belongs.

**`edit_distance_draft_to_final` is the most operationally important field.** It
is how you detect rubber-stamping, and it is how you measure whether prompts are
improving.

### 9.3 Risk Register

| ID | Risk | Likelihood | Impact | Mitigation | Owner |
| --- | --- | --- | --- | --- | --- |
| R-01 | PHI sent to a non-BAA-covered service | Medium | **Critical** | Egress allowlist at the network layer; approved-vendor list in policy; annual staff training; technical block on consumer AI domains from clinical workstations | Program Owner |
| R-02 | Hallucinated clinical content reaches a chart | Medium | **Critical** | Mandatory human review; structured output schemas; visual marking of AI-generated content; edit-rate monitoring | Physician Sponsor |
| R-03 | Automation complacency — humans stop reviewing | **High** | **Critical** | Synthetic error injection at 1:50; edit-rate dashboards; quarterly review audits; coaching when edit rate approaches zero | Program Owner |
| R-04 | Recording without valid two-party consent | Medium | High | Consent in intake packet; signage; verbal script; consent flag checked before recording is technically enabled | Counsel |
| R-05 | Wrong-patient data association | Low | **Critical** | Two-factor matching (name + DOB); twin/sibling flags; human confirmation on any ambiguity; never auto-file on fuzzy match | Technical Lead |
| R-06 | Scope creep into clinical decision support | **High** | High | Written policy prohibition; architectural enforcement (no disposition field in schemas); change control requiring physician sponsor approval | Physician Sponsor |
| R-07 | Vendor discontinues product or is acquired | Medium | Medium | Month-to-month contracts; data export verified before signing; abstraction layers around swappable components | Program Owner |
| R-08 | Key person dependency on the technical lead | **High** | High | Documentation standard enforced; n8n visual workflows readable by non-engineers; no undocumented production changes | Program Owner |
| R-09 | Section 54.2 sunset changes the regulatory basis | **Certain (Jan 1, 2027)** | Medium | I-10 externalizes delegation rules to configuration; counsel monitors pending legislation; quarterly review | Counsel |
| R-10 | TCPA exposure on automated messaging | Medium | High | Documented consent; immediate opt-out honoring; frequency caps; retain consent records; separate reminder consent from marketing consent | Counsel |
| R-11 | Staff perceive automation as a prelude to layoffs | **High** | High | Explicit, early, written commitment from partners that this program is about absorbing growth and reducing after-hours work, not reducing headcount. **Say it in month one, not month six.** | Partners |
| R-12 | Model version change silently degrades output | Medium | High | Pin model versions; regression test suite; validate before any version bump | Technical Lead |

**R-11 deserves emphasis.** In a practice where staff have twenty-plus year
tenure and several moved from administrative into clinical roles internally, the
cultural risk of a poorly-communicated automation program is higher than the
technical risk. The message must come from the partners, in writing, before the
first tool is deployed.

### 9.4 Change Control

Any change to a prompt, schema, or automated workflow touching clinical content
requires:

1. Written description of the change and rationale
2. Regression test against the standing test suite
3. Physician Sponsor approval for anything clinical-adjacent
4. Version increment and changelog entry
5. Rollback plan

Prompts are code. Treat them that way.

---

## 10. Measurement Plan & KPIs

### 10.1 Replacing Estimates with Measurements

Every `[EST]` in Section 2 must be measured in the first 60 days. This table is
the assignment.

| Input | How to Measure | Effort | Owner |
| --- | --- | --- | --- |
| Active panel size | EHR report: unique patients with a visit in 24 months | 15 min | Practice Manager |
| Annual encounters | EHR report by site and date range | 15 min | Practice Manager |
| Well/sick mix | EHR report by CPT category | 15 min | Practice Manager |
| **No-show rate** | EHR report by provider, visit type, day of week, lead time | 30 min | Practice Manager |
| **Clinical call volume** | Phone system report, or manual tally sheet for 7 days including a weekend | 1 week | MA lead |
| Form volume | Manual tally for 30 days | 30 days | Front desk lead |
| Fax volume | Fax machine counter or gateway report for 7 days | 1 week | Front desk lead |
| Time per task (all) | Direct observation: 10 samples per task type, timed | 4 hrs | Program Owner |
| Denial rate and root cause | Billing report, 12 months, by CARC code | 1 hr | Biller |
| Copay collection rate at desk | Billing report | 30 min | Biller |
| MA fully loaded rate | Payroll | 15 min | Practice Manager |
| Blended revenue per encounter | Billing report: total allowed / encounters | 30 min | Biller |

**The two bolded rows drive the two largest line items in the model and together
take about a week of low-effort data collection. Do them first.**

### 10.2 Operational KPIs

| KPI | Baseline | Target | Frequency | Initiative |
| --- | --- | --- | --- | --- |
| No-show rate | Measure | −25% | Weekly | I-07 |
| Cancelled slots backfilled | Measure | > 60% | Weekly | I-07 |
| Median slot backfill time | N/A | < 15 min | Weekly | I-07 |
| Form turnaround (request → signed) | Measure | < 24 hrs | Weekly | I-01 |
| Form error/rejection rate | Measure | < 1% | Monthly | I-01 |
| Patients with open immunization gap | Measure | −40% in 12 mo | Monthly | I-02 |
| Recall message → visit conversion | N/A | > 30% | Monthly | I-02 |
| Triage calls documented same-shift | Measure | > 98% | Weekly | I-04 |
| Triage note completion time | Measure | < 90 sec | Weekly | I-04 |
| Physician note close-out same-day | Measure | > 95% | Weekly | I-05 |
| Documents auto-filed correctly | N/A | > 92% | Weekly | I-06 |
| Cold chain excursions detected < 15 min | N/A | 100% | Monthly | I-08 |
| Eligibility-caused denial rate | Measure | −60% | Monthly | I-09 |
| Copay collected at time of service | Measure | > 93% | Monthly | I-09 |
| Staff with current competency for all assigned tasks | Measure | 100% | Monthly | I-10 |

### 10.3 AI Quality & Safety KPIs

**These matter more than the operational KPIs, because they are the early warning
system.**

| KPI | Target | Alarm Threshold | Frequency |
| --- | --- | --- | --- |
| Draft-to-final edit rate, triage notes | 15–40% | **< 5% (rubber-stamping)** or > 60% (poor drafts) | Weekly |
| Draft-to-final edit rate, ambient notes | 10–35% | **< 5%** or > 55% | Weekly |
| Synthetic error catch rate (injected at 1:50) | > 90% | **< 75%** | Monthly |
| Form field accuracy vs. manual audit (n=25) | > 99% | **< 97%** | Monthly |
| Document classification accuracy (n=50) | > 92% | < 85% | Monthly |
| Urgent-document false negatives | **0** | **Any** | Weekly |
| Low-confidence routing rate | 8–20% | < 3% (over-confident) or > 35% | Weekly |
| Wrong-patient association events | **0** | **Any** | Immediate |
| PHI sent to non-BAA endpoint | **0** | **Any** | Continuous |

**The two "< 5% edit rate" alarms are the most important lines in this entire
document.** An MA whose edit rate on AI-drafted triage notes drops below five
percent has stopped reviewing. That is not efficiency; that is an unreviewed
machine-generated clinical record with a human signature on it, which is
substantially worse than the manual process it replaced.

### 10.4 Reporting Cadence

| Report | Audience | Frequency |
| --- | --- | --- |
| Safety dashboard (§10.3) | Program Owner, Physician Sponsor | Weekly |
| Operational dashboard (§10.2) | Practice Manager | Weekly |
| Financial realization vs. model | Partners | Monthly |
| Vendor and BAA review | Practice Manager, Counsel | Annually |
| Full program review with re-baselined ROI | Partners | Quarterly |

---

## Appendix A — Sample System Prompts

> Illustrative. Every prompt below assumes a BAA-covered endpoint with zero data
> retention, temperature 0, and enforced JSON schema validation on the output.

### A.1 — Telephone Triage Note Structuring (I-04)

```
You are a clinical documentation assistant for a pediatric practice.

You will receive a transcript of a telephone call between a medical assistant
and a patient's parent or guardian.

Your ONLY task is to extract and structure what was said into the JSON schema
below.

ABSOLUTE CONSTRAINTS:
- You MUST NOT suggest, infer, recommend, or imply any clinical disposition.
- You MUST NOT suggest, infer, or recommend any diagnosis.
- You MUST NOT suggest which triage protocol should have been used.
- You MUST NOT add clinical advice that was not spoken in the transcript.
- If information is not present in the transcript, use null. Do not infer it.
- If audio was unclear, record the location in transcript_gaps rather than
  guessing at content.

The protocol used and the disposition reached are supplied separately by the
medical assistant through the application interface. They are not your output
and no field exists for them in your schema.

Return ONLY valid JSON matching this schema:
{
  "caller": {"name": string|null, "relationship_to_patient": string|null},
  "chief_complaint": string,
  "symptom_onset": string|null,
  "symptoms_reported_present": [string],
  "symptoms_explicitly_denied": [string],
  "relevant_history_mentioned": [string],
  "medications_mentioned": [string],
  "advice_given_by_ma": [string],
  "safety_net_instructions_given": [string],
  "followup_discussed": boolean,
  "followup_timeframe": string|null,
  "transcript_gaps": [string]
}
```

### A.2 — Immunization Record Adjudication (I-01, I-02)

```
You are reconciling two immunization records for the same patient.

RECORD A (practice chart) and RECORD B (state registry) are provided below.

For each dose pair presented, determine whether they represent THE SAME
administration event or TWO DISTINCT events.

CONSTRAINTS:
- You may ONLY use information present in the two records.
- You MUST NOT infer, estimate, or reconstruct a date that appears in neither
  record.
- You MUST NOT determine whether a patient is due for a vaccine. That is
  computed by a separate rules engine.
- When uncertain, return UNCERTAIN. Returning UNCERTAIN is a correct and
  expected outcome; a human will review it. Guessing is not.

Consider: CVX code equivalence, historical trade-name to generic mappings,
partial or approximate dates, and combination-vaccine components recorded
separately in one source and jointly in the other.

Return ONLY valid JSON:
{
  "determination": "MATCH" | "NO_MATCH" | "UNCERTAIN",
  "confidence": 0.0-1.0,
  "reasoning": string,
  "cvx_a": string|null,
  "cvx_b": string|null,
  "date_a": string|null,
  "date_b": string|null,
  "requires_human_review": boolean
}
```

### A.3 — Fax Urgency Triage (I-06)

```
You are triaging an inbound clinical document for a pediatric practice.

Classify its urgency. Bias STRONGLY toward over-flagging. A document
incorrectly marked urgent costs a physician thirty seconds. A document
incorrectly marked routine can delay care.

Return "urgent" if the document contains ANY of:
- A laboratory or imaging result outside the reference range
- An explicit recommendation for near-term action or follow-up
- Any language indicating clinical deterioration, admission, or ED visit
- A critical value notification of any kind
- Anything you are unsure about

Return "needs_physician_review" for specialist consults, discharge summaries,
and normal results requiring acknowledgment.

Return "routine" ONLY for clearly administrative documents: records requests,
insurance correspondence, marketing.

Return ONLY valid JSON:
{
  "urgency": "urgent" | "needs_physician_review" | "routine",
  "confidence": 0.0-1.0,
  "reason": string,
  "abnormal_values_detected": [string]
}
```

### A.4 — Pre-Visit Narrative Summary (I-03)

```
You are preparing a context brief for a pediatric clinician before a scheduled
visit.

You will receive the last three encounter notes and the current problem list.

CONSTRAINTS:
- You MUST NOT generate clinical recommendations, differential diagnoses, or
  suggested orders.
- You MUST NOT state anything not present in the supplied notes.
- Every item you return MUST cite the encounter date it came from.
- Screenings due, immunizations due, and growth percentile changes are computed
  by separate rules engines. Do not report on them.

Report only: what happened recently that a clinician would want to know before
walking into this room, and what threads remain open.

Return ONLY valid JSON:
{
  "recent_relevant_history": [{"item": string, "source_date": string}],
  "open_threads": [{"item": string, "source_date": string}],
  "unresolved_parent_concerns": [{"item": string, "source_date": string}],
  "medication_changes": [{"item": string, "source_date": string}]
}
```

---

## Appendix B — Sample Orchestration Workflow

**I-02 nightly immunization reconciliation and huddle sheet generation, expressed
as n8n nodes:**

```
[Cron 02:00 daily]
     │
     ▼
[HTTP: FHIR bulk export — Patient + Immunization, active panel]
     │
     ▼
[Split in batches — 250 patients]
     │
     ▼
[HTTP: I-CARE query (HL7 QBP^Q11) per patient]
     │
     ▼
[Function: deterministic CVX matcher, ±4-day tolerance]
     │
     ├── matched ──────────────────────────┐
     │                                     │
     └── ambiguous ──► [AWS Bedrock:       │
                        adjudication       │
                        prompt A.2]        │
                            │              │
                            ├── MATCH ─────┤
                            ├── NO_MATCH ──┤
                            └── UNCERTAIN ─► [Postgres: human_review_queue]
                                             │
     ┌───────────────────────────────────────┘
     ▼
[Function: merge records → canonical immunization history]
     │
     ▼
[HTTP: CDSi forecast engine → due/overdue antigens]
     │
     ├──────────────────────────────┐
     ▼                              ▼
[Filter: patient on              [Filter: patient has gap
 tomorrow's schedule]             AND no upcoming appt]
     │                              │
     ▼                              ▼
[Merge with screening rules   [Postgres: upsert recall_queue
 + growth calc + open items]   with priority score]
     │                              │
     ▼                              ▼
[AWS Bedrock: narrative       [Cron 09:00: dequeue,
 summary, prompt A.4]          check suppression list,
     │                          check frequency cap,
     ▼                          send via Twilio]
[Template: render PDF]              │
     │                              ▼
     ▼                         [Webhook: inbound reply
[Deliver: internal secure       → Bedrock classify
 distribution to providers      → route to queue]
 + MAs, 05:00]
     │
     ▼
[Postgres: audit log write]
     │
     ▼
[On any node error: → dead letter queue → Slack alert to Program Owner]
```

**Error handling is not decoration.** Every node has a failure path. The default
on failure is to queue for human handling, never to skip silently. A nightly job
that fails quietly for a week produces a huddle sheet nobody notices is missing.

---

## Appendix C — Vendor Due Diligence Checklist

Complete for every vendor before any PHI flows. Retain the completed checklist.

**Contractual**
- [ ] Signed BAA in hand — not "available," signed
- [ ] BAA explicitly names the specific product tier and feature surface in use
- [ ] Subcontractor / sub-processor chain disclosed in writing
- [ ] Breach notification timeline stated (target: ≤ 72 hours)
- [ ] Data ownership and export rights on termination
- [ ] Contract term — month-to-month strongly preferred at this practice size
- [ ] Price change notice period

**Technical**
- [ ] Zero data retention configured and confirmed in writing
- [ ] Confirmation that data is not used for model training
- [ ] Encryption in transit (TLS 1.2+) and at rest (AES-256)
- [ ] Customer-managed encryption keys available (preferred, not always required)
- [ ] SSO / MFA support
- [ ] Role-based access control
- [ ] Audit logging exportable
- [ ] Data residency (US)
- [ ] Documented uptime SLA

**Assurance**
- [ ] SOC 2 Type II report reviewed, not just claimed
- [ ] HITRUST certification (nice to have)
- [ ] Penetration test summary available
- [ ] Named security contact

**Clinical (AI vendors only)**
- [ ] Independent accuracy validation (KLAS score, peer-reviewed study, or none —
      and "none" is acceptable if disclosed)
- [ ] Model provider and version disclosed
- [ ] Notice policy for model version changes
- [ ] Human-review workflow is enforced by the product, not merely recommended
- [ ] Note/output export in a portable format

**Operational**
- [ ] Trial period available without a credit card
- [ ] Setup and onboarding cost stated (many enterprise vendors charge
      $500–$5,000 for small practices, or $500–$1,000 per user)
- [ ] Training included or priced
- [ ] Support hours and channel
- [ ] Named reference customer of comparable size

---

## Appendix D — Glossary

| Term | Definition |
| --- | --- |
| **ACIP** | Advisory Committee on Immunization Practices — sets the US immunization schedule |
| **Agentic** | An AI system granted autonomous multi-step tool use. Deliberately avoided in clinical paths throughout this plan. |
| **Ambient documentation** | Passive audio capture of a clinical encounter, transcribed and drafted into a note |
| **BAA** | Business Associate Agreement — the HIPAA contract required before a vendor may process PHI |
| **CARC / RARC** | Claim Adjustment / Remittance Advice Remark Codes — standardized denial reason codes |
| **CDSi** | Clinical Decision Support for Immunization — the machine-readable specification of the immunization schedule |
| **CVX** | CDC vaccine administered code set |
| **FHIR** | Fast Healthcare Interoperability Resources — the modern healthcare data API standard (R4 is current) |
| **HL7 v2** | The older healthcare messaging standard; still ubiquitous, especially for registries |
| **I-CARE** | Illinois Comprehensive Automated Immunization Registry Exchange, operated by IDPH |
| **IDFPR** | Illinois Department of Financial and Professional Regulation — licenses professions; does **not** license medical assistants |
| **LLM** | Large Language Model |
| **PHI** | Protected Health Information |
| **RTE** | Real-Time Eligibility |
| **Schmitt-Thompson** | The standard pediatric and adult telephone triage protocol library |
| **TCPA** | Telephone Consumer Protection Act — governs automated calls and texts |
| **VIS** | Vaccine Information Statement — federally required at every vaccine administration |
| **X12 270/271** | The EDI transaction pair for eligibility inquiry and response |
| **ZDR** | Zero Data Retention |
| **225 ILCS 60/54.2** | The Illinois Medical Practice Act section governing physician delegation to unlicensed personnel. **Currently scheduled to sunset January 1, 2027.** |

---

## Closing Note

The most important thing in this document is not any single initiative. It is
the discipline that runs through all ten:

**The machine drafts. The licensed human decides. The audit log proves it.**

That is not a hedge against regulatory risk, though it is that. It is the correct
architecture for a domain where the cost of a confident wrong answer is measured
in children rather than dollars. Six of these ten initiatives use no language
model at all, or use one only to write down what a human already decided. The
value is in the plumbing — connecting systems that do not talk, and replacing
paper with structured data that can be queried, audited, and improved.

---

*Prepared by Carly Altmanl · August 2026*
*Pricing verified August 2026 from public vendor materials; confirm directly before contracting.*
*This document contains no protected health information.*
