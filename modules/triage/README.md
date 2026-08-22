# I-04 — Telephone Triage Documentation

**The machine transcribes and writes down. The medical assistant assesses,
selects the protocol, determines the disposition and gives the advice.** README
I-04 calls that division "the single most important design constraint in the
entire document", and it is not caution — under 225 ILCS 60/54.2 an unlicensed
MA in Illinois practises under physician delegation, so a machine-supplied
disposition is a machine practising medicine.

The constraint is enforced in four independent places, in ascending order of how
much each one protects anyone:

| # | Control | Where |
| --- | --- | --- |
| 1 | The output schema has no field for a protocol, a disposition or a diagnosis | `structure.NOTE_SCHEMA` (README Appendix A.1, verbatim) |
| 2 | An **allowlist** refuses any schema whose key set is not exactly A.1's | `structure.assert_matches_a1_shape`, at import and per call |
| 3 | Every extracted string is checked against the transcript; advice credited to the MA must match something the MA actually said | `structure.NoteStructurer._ground` |
| 4 | The protocol list is alphabetical, age-filtered, and offers nothing else | `protocols.ProtocolRegistry.search` |

There is no `suggest_protocol`, no `likely_disposition`, no ranking by
plausibility, no reordering based on the transcript. A searchable list is the
entire user interface, because anything cleverer is a machine-generated clinical
judgement wearing a convenience costume.

## Files

| File | Responsibility |
| --- | --- |
| `consent.py` | Illinois two-party consent (720 ILCS 5/14-2). Consent rows, the disclosure script, and the authorisation token the recorder demands. |
| `capture.py` | Recording and transcription. faster-whisper behind a lazy import; `ScriptedTranscriber` for tests and demos. |
| `structure.py` | The one model call. A.1 prompt verbatim, A.1 schema verbatim, plus the schema guards and the grounding check. |
| `protocols.py` | The MA's two taps. **Contains no intelligence at all**, on purpose. |
| `render.py` | The chart note, the reviewer draft, and the follow-up task. |
| `review.py` | Signature, edit distance, and the rubber-stamp / poor-draft alarms. |
| `encounter.py` | The state machine, the database, KPIs, and crash re-entry. |
| `lifecycle.py` | Audio and transcript retention: delete on signature, 24-hour ceiling regardless. |
| `regression.py` | The fifty-call suite and the version-bump gate it feeds. |
| `fixtures.py` | 15 synthetic calls, hand-written. No real patient data. |
| `demo.py` | `make demo-i04` — runs the whole thing and prints it. |
| `../../config/triage_protocols.yaml` | Protocol **identifiers** only. See the licensing note below. |

## Wiring

```python
service = TriageService(
    db,
    consent=ConsentRegistry(db, audit=audit),
    recorder=Recorder(audio_dir),
    transcriber=FasterWhisperTranscriber(),
    structurer=NoteStructurer(client, audit=audit),
    registry=ProtocolRegistry.load(),        # refuses placeholder data
    reviewer=NoteReviewer(audit=audit, registry=registry),
    lifecycle=AudioLifecycle(db, audit=audit),
)

encounter = service.open(patient_id=..., family_id=..., opened_by=..., now=now)
service.deliver_disclosure(encounter, response="granted", delivered_by=..., now=now)
service.authorise(encounter, now=now)
service.capture(encounter, audio_bytes=..., started_utc=..., ended_utc=..., now=now)
service.transcribe(encounter)
service.draft(encounter, taps=taps, patient_label=..., now=now, age_months=...)
service.sign(encounter, signed_by=..., now=now, patient_label=...,
             final_text=edited, acknowledged_drops=True)
```

## Protocol content is licensed and is not in this repo

The Schmitt-Thompson pediatric telephone protocols are copyrighted. Their
**content** is not reproduced here and must not be. `triage_protocols.yaml`
carries identifiers and a library version; the MA reads the protocol from the
practice's licensed copy and this system records which one was used. An
identifier *plus a library version* is what demonstrates a standard of care was
followed — "used the fever protocol" is not evidence, because protocols change.

The shipped identifiers are `PLACEHOLDER-*` and `ProtocolRegistry.load()`
**refuses them by default**; the tests and the demo opt in explicitly. Shipping
without loading the practice's list would put `PLACEHOLDER-FEVER` in a chart.

## The eight decisions worth knowing

**1. Grounding is a post-condition, not a prompt instruction.** A.1 tells the
model not to invent advice. That is a request. Every string it returns is then
checked against the transcript, and the check runs whether or not the model
complied. Fabricated advice and safety-net instructions are **removed**;
weakly-supported symptoms, denials, history and medications are **kept and
marked**, because deleting a documented denial makes a note look as though the
question was never asked — which is the gap a triage note exists to close.

**2. A marked item is marked in the chart, not just in the draft.** Kept-but-
unsupported text prints as `[not clearly supported by the recording]` in the
signed note. A reader who cannot tell verified content from unverified content
has a note that reads better than the evidence behind it.

**3. Deterministic things are deterministic — and narrowly so.** Follow-up
timeframes are parsed by a numeric rule with a small idiom table beside it, and
the **earliest** date any rule finds wins ("tomorrow, and again in 2 weeks" is a
task for tomorrow). A conditional tail is split off rather than blanking the
string, because "in 2 days if she's not better" is the most ordinary follow-up
phrasing there is and it was coming back undated — which sinks it to the bottom
of the loop-closure queue.

Caller relationship is a closed vocabulary too, but the licence to skip the
grounding check is granted only by the **caller identifying themselves** ("it's
her mom", or the MA's "is this Ezra's dad?" answered yes), never by the word
appearing somewhere on the call. Reading it off the whole transcript let the
MA's own opening grant it, and let a sitter's "her mom is at work until six"
document the mother as the informant. A relationship field carrying anything
beyond a relationship term is scored like ordinary free text.

**4. The edit diff's baseline is the chart note, never the draft.** README 10.3
asks for an alarm when the edit rate drops below 5%, which detects rubber-
stamping. Diffing the reviewer-facing draft (removal notices, flag markers, the
DRAFT banner) against the chart body made every untouched signature look heavily
edited, so the alarm was *structurally incapable of firing*. `render_note`
returns the chart body; `signature_block` is separate; the diff has a clean
baseline.

**5. `transcript_gaps` is split in two.** Unclear-audio spans belong in the
chart — a reader has to be able to tell "the parent denied it" from "we could
not hear that part". But that made the field a free-text channel from the model
straight into a clinical record. The chart gets `system_gaps`, derived
arithmetically from transcription confidence; anything the model wrote goes to
`model_gaps`, which the reviewer sees and can choose to type in.

**6. Consent gates the recording, never the care.** A declined disclosure moves
the encounter to `manual`: the call happens, the MA documents by hand, and the
recorder refuses. Any decline anywhere in a family's history disqualifies —
a later grant does not outvote it — and the disclosure must have been delivered
to *that* family for *that* encounter.

**7. Retention has a floor and a ceiling.** Audio is deleted on signature and,
regardless, at 24 hours — signed or not, finished or not, with no exemption
parameter, because the recordings that would be exempted are precisely the
abandoned ones the ceiling exists for. Transcripts go the same way and keep
their SHA-256, so the audit record can still prove which transcript produced
which note. A failed unlink keeps the row **live** rather than stamping it
deleted: a file that is on disk and invisible to every report is worse than one
that is visibly stuck.

**8. The regression suite is a gate, not a report.** `ModelPin.bump()` refuses a
version that has not passed, and the pin covers the prompt hash as well as the
model, because prompts are code (README 9.4). `RegressionResult.blockers()`
lists every reason a run must not be pinned, and four of them are conditions no
average can absorb: a declared fabrication that survived, any single field below
`FIELD_PASS_THRESHOLD`, a corpus of fewer than ten distinct transcripts, and a
run that exercised no adversarial item at all. Each was a way a degraded or
sabotaged model passed — grounding switched off scored 0.988; a model emitting
no safety-net instruction on half the corpus scored 0.9500; fifty copies of one
twenty-second call scored 1.0 and printed "0 missed adversarial drop(s)".

## Crash re-entry

`TriageService.resume(encounter_id)` rebuilds an encounter from the database and
`abandoned()` lists the ones stuck short of a terminal state. Without them an
encounter existed only on the in-memory object the caller held: a browser reload
mid-call left a row in `recording` and a WAV nobody could reach, the audio sat
until the sweep, and the call went undocumented.

State, taps, draft, proposed chart text and the structured note all come back,
so a resumed encounter can be reviewed and **signed** — which is the point, since
the alternative outcome is the MA retyping a note the machine already wrote. The
note is deserialised from its row rather than re-derived: re-running the model
would be a second inference against a transcript that may already have been
purged, producing a different set of drops and flags from the ones the reviewer
is looking at. The authorisation is **not** restored — it has a TTL and consent
can have been revoked, so `capture()` re-runs the gate from scratch.

## Scope, stated plainly

`FasterWhisperTranscriber` is the real transcription path and is lazily
imported; nothing in the deterministic half of the module touches it, which
`make lint` asserts. Diarization degrades gracefully: with no speaker labels the
transcript is scored against the whole text and the note says so, and advice
cannot be attributed to the MA at all — which is the safe direction.

The fifty-call regression corpus is **fifteen hand-written transcripts tiled**,
and `RegressionResult.summary()` says so in the line `ModelPin` stores, because
"50 calls" reads like fifty real ones. Before go-live it should be fifty real
de-identified calls with clinician-written reference extractions.
