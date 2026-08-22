"""Structured note to the practice's telephone-encounter text. No model.

Rendering is string formatting. A model here would add nothing and would give a
second opportunity for machine-generated language to enter a chart, so there
isn't one.

Three things the layout is doing deliberately:

  * **Protocol and disposition are printed with their source.** "Protocol
    applied (selected by MA jrivera)" is not decoration -- it is the sentence
    that distinguishes a documented human decision from a machine's guess, and
    it is what a reader in a deposition needs to see.
  * **The draft and the chart note are different documents.** The draft shows
    the reviewer what was dropped for lack of transcript support and what audio
    was unclear. The signed note does not: dropped content is not part of the
    record, and a chart cluttered with the machine's rejected guesses is worse
    to read and worse to defend.
  * **Nothing renders without the taps.** `TapsIncomplete` propagates. There is
    no "(disposition pending)" placeholder, because a note that can be signed
    with a blank disposition will eventually be signed with a blank
    disposition -- and README I-04 calls an undocumented triage call
    indefensible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence

from modules.scheduling.models import to_local

from .protocols import MATaps, ProtocolRegistry
from .structure import StructuredNote

__all__ = [
    "FollowUpTask",
    "render_draft",
    "render_note",
    "signature_block",
    "build_followup_task",
]


def _bullets(
    items: Sequence[str],
    empty: str = "none documented",
    *,
    flagged: Mapping[str, str] | None = None,
) -> list[str]:
    """Render a list, marking anything the transcript did not clearly support.

    The mark goes in the CHART, not only in the draft. A weakly-supported line
    that reads like every other line is a machine's paraphrase presented as a
    documented fact; a reader six months later has no way to tell. Marking it is
    the same honesty as printing the unclear-audio spans.
    """
    if not items:
        return [f"    {empty}"]
    flagged = flagged or {}
    out = []
    for item in items:
        suffix = "  [not clearly supported by the recording]" if item in flagged else ""
        out.append(f"    - {item}{suffix}")
    return out


def _field(value: Any, empty: str = "not documented") -> str:
    text = "" if value is None else str(value).strip()
    return text or empty


@dataclass(frozen=True)
class FollowUpTask:
    """The loop-closure task. Derived arithmetically, never invented.

    README I-04's benefit model counts recovered follow-ups as real money and
    the failure mode as "patient told 'call back if worse' and nobody closes the
    loop". The task is generated from two structured fields the MA can see and
    correct -- `followup_discussed` and `followup_timeframe` -- rather than from
    a model's reading of intent, because a follow-up task nobody can trace to a
    spoken sentence is one more thing to ignore.
    """

    encounter_id: str
    patient_id: str
    due_utc: datetime | None
    timeframe_text: str
    description: str
    assigned_to: str

    def as_dict(self) -> dict[str, Any]:
        from modules.scheduling.models import iso

        return {
            "encounter_id": self.encounter_id,
            "patient_id": self.patient_id,
            "due_utc": iso(self.due_utc) if self.due_utc else None,
            "timeframe_text": self.timeframe_text,
            "description": self.description,
            "assigned_to": self.assigned_to,
        }


#: Spoken timeframes to a concrete offset. Deliberately small and literal: a
#: phrase not on this list produces a task with no due date and a note telling
#: the MA to set one, which is honest. Guessing that "in a bit" means 48 hours
#: would put a wrong date on a clinical follow-up.
_TIMEFRAME_OFFSETS: Mapping[str, int] = {
    "today": 0,
    "tonight": 0,
    "this evening": 0,
    "tomorrow": 1,
    "24 hours": 1,
    "in a day": 1,
    "2 days": 2,
    "48 hours": 2,
    "two days": 2,
    "3 days": 3,
    "three days": 3,
    "a week": 7,
    "1 week": 7,
    "one week": 7,
    "2 weeks": 14,
    "two weeks": 14,
}


#: A conditional tail: "…, sooner if he gets worse", "…unless the rash spreads".
#: Split off and discarded rather than blanking the whole string. Blanking it
#: was how "in 2 days if she's not better" -- the single most ordinary way a
#: parent is told to follow up -- came back with no due date at all, and
#: `open_followups` sorts undated tasks to the bottom of the queue with
#: COALESCE(due_utc,'9999'), which is precisely the "nobody closes the loop"
#: failure README I-04 counts as money.
_CONDITIONAL_TAIL = re.compile(r"\b(?:if|unless|in case|should she|should he)\b")

#: A refusal of follow-up rather than a condition on one. Applied to the head
#: of the string only, once the conditional tail is gone.
_NEGATED = re.compile(r"\b(?:not|no|don'?t|do not|never)\b")
_PHRASE_CACHE: dict[str, re.Pattern[str]] = {}

#: Number words a parent actually says. Stops at twelve on purpose -- past that
#: people use digits, and a longer table is more surface for no benefit.
_NUMBER_WORDS: Mapping[str, int] = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12,
}
_UNIT_DAYS: Mapping[str, int] = {"day": 1, "week": 7, "month": 30}

_NUM = r"(?:\d{1,3}|" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")"

#: "<n> <unit>" in either digits or words. This is the general rule; the table
#: below it only exists for idioms that have no number in them at all.
_COUNTED = re.compile(
    r"(?<!\w)(" + _NUM + r")\s+(day|week|month)s?(?!\w)"
)

#: "three to four days", "3-4 days", "2 or 3 weeks". Matched BEFORE the single
#: form, and resolved to the EARLIER bound: a follow-up done a day early costs
#: a phone call, a follow-up done a day late is the one that gets deposed.
_COUNTED_RANGE = re.compile(
    r"(?<!\w)(" + _NUM + r")\s*(?:to|or|-|\u2013)\s*(" + _NUM
    + r")\s+(day|week|month)s?(?!\w)"
)


def _as_count(token: str) -> int:
    return int(token) if token.isdigit() else _NUMBER_WORDS[token]

#: Beyond this a "follow-up" is not a triage follow-up, it is a well visit, and
#: a task dated eleven months out sitting in a triage queue is noise. Over the
#: cap the task ships with no date and says so, same as an unrecognised phrase.
MAX_FOLLOWUP_DAYS = 120


def _counted_offset(text: str) -> int | None:
    """Days from an explicit "<n> <unit>" phrase, or None.

    WHY this and not a bigger lookup table: the table approach silently gave
    "3 weeks" no due date at all, because 1, 2 and 3 days and 1 and 2 weeks
    happened to be enumerated and 3 weeks did not. An MA reading "NO DUE DATE
    SET" on a perfectly ordinary phrase learns to ignore the warning, which is
    what makes the genuinely unparseable cases dangerous.
    """
    ranged = _COUNTED_RANGE.search(text)
    if ranged is not None:
        low, high, unit = ranged.groups()
        count = min(_as_count(low), _as_count(high))
    else:
        match = _COUNTED.search(text)
        if match is None:
            return None
        token, unit = match.groups()
        count = _as_count(token)
    days = count * _UNIT_DAYS[unit]
    return days if 0 < days <= MAX_FOLLOWUP_DAYS else None


def _phrase_re(phrase: str) -> re.Pattern[str]:
    """Whole-phrase match with a digit guard on both ends.

    Naked substring containment made "12 days" match "2 days" (+2 instead of
    +12) and "in a weekend" match "a week" (+7). Anchoring on word boundaries
    and refusing an adjacent digit is the difference between a follow-up task
    on the right day and one on a day nobody chose.
    """
    cached = _PHRASE_CACHE.get(phrase)
    if cached is None:
        cached = re.compile(r"(?<!\d)(?<!\w)" + re.escape(phrase) + r"(?!\w)")
        _PHRASE_CACHE[phrase] = cached
    return cached


def build_followup_task(
    note: StructuredNote,
    *,
    patient_id: str,
    now: datetime,
    assigned_to: str,
) -> FollowUpTask | None:
    """A task, or None when no follow-up was discussed. Never a guess."""
    data = note.data
    if not data.get("followup_discussed"):
        return None
    timeframe = str(data.get("followup_timeframe") or "").strip()
    from datetime import timedelta

    due: datetime | None = None
    lowered = " ".join(timeframe.lower().split())
    # The conditional tail goes first: "in 2 days if she's not better" is a
    # two-day follow-up with a caveat, not a refusal.
    conditional = _CONDITIONAL_TAIL.search(lowered)
    head = lowered[: conditional.start()] if conditional else lowered
    # Then the refusals. "Not today" containing "today" is not a due date of
    # today, and a wrong date on a clinical follow-up is worse than no date.
    if _NEGATED.search(head):
        head = ""
    offsets = [] if not head else [
        days for phrase, days in _TIMEFRAME_OFFSETS.items()
        if _phrase_re(phrase).search(head)
    ]
    counted = _counted_offset(head) if head else None
    if counted is not None:
        offsets.append(counted)
    if offsets:
        # The EARLIEST date any rule found, not whichever rule ran first.
        # Ordering decided it before, so "tomorrow, and again in 2 weeks" was
        # scheduled a fortnight out -- the counted rule simply ran first. Same
        # reasoning as the range rule: early costs a phone call, late is the
        # one that gets deposed.
        due = now + timedelta(days=min(offsets))
    over_cap = head and counted is None and _COUNTED.search(head) is not None
    description = (
        f"Triage follow-up: {_field(data.get('chief_complaint'), 'see note')}"
    )
    if timeframe:
        description += f" (parent told: {timeframe})"
    if due is None:
        description += (
            " -- NO DUE DATE SET: the spoken timeframe is beyond "
            f"{MAX_FOLLOWUP_DAYS} days and is not a triage follow-up"
            if over_cap
            else " -- NO DUE DATE SET: the spoken timeframe was not recognised"
        )
    return FollowUpTask(
        encounter_id=note.encounter_id,
        patient_id=patient_id,
        due_utc=due,
        timeframe_text=timeframe,
        description=description,
        assigned_to=assigned_to,
    )


def _core_body(
    note: StructuredNote,
    taps: MATaps,
    registry: ProtocolRegistry,
    *,
    patient_label: str,
    call_time: datetime,
) -> list[str]:
    data = note.data
    flagged_by_field: dict[str, dict[str, str]] = {}
    for item in note.flagged:
        flagged_by_field.setdefault(item["field"], {})[item["text"]] = item["reason"]

    def bullets(field_name: str, empty: str = "none documented") -> list[str]:
        return _bullets(
            data.get(field_name) or [], empty, flagged=flagged_by_field.get(field_name)
        )
    described = registry.describe(taps)
    caller = data.get("caller") or {}
    local = to_local(call_time)

    lines = [
        "TELEPHONE ENCOUNTER NOTE",
        f"Patient: {patient_label}",
        f"Call time: {local:%Y-%m-%d %H:%M %Z}",
        f"Documented by: {taps.tapped_by}",
        "",
        "CALLER",
        f"    {_field(caller.get('name'), 'name not documented')}"
        f" ({_field(caller.get('relationship_to_patient'), 'relationship not documented')})",
        "",
        "CHIEF COMPLAINT",
        f"    {_field(data.get('chief_complaint'))}",
        "",
        "ONSET",
        f"    {_field(data.get('symptom_onset'))}",
        "",
        "SYMPTOMS REPORTED",
        *bullets("symptoms_reported_present"),
        "",
        "EXPLICITLY DENIED",
        *bullets("symptoms_explicitly_denied"),
        "",
        "RELEVANT HISTORY MENTIONED",
        *bullets("relevant_history_mentioned"),
        "",
        "MEDICATIONS MENTIONED",
        *bullets("medications_mentioned"),
        "",
        # The two human decisions, each labelled with who made it. This block is
        # the difference between a defensible note and a transcript.
        "PROTOCOL APPLIED",
        f"    {described['protocol_title']} [{described['protocol_id']}]",
        f"    selected by {taps.tapped_by} from {described['protocol_library']}",
        "",
        "DISPOSITION",
        f"    {described['disposition_label']} [{described['disposition_id']}]",
        f"    determined by {taps.tapped_by}",
    ]
    if taps.supervising_professional_id:
        lines.append(
            f"    licensed professional: {taps.supervising_professional_id} "
            "(225 ILCS 60/54.2)"
        )
    lines += [
        "",
        "ADVICE GIVEN",
        *bullets("advice_given_by_ma"),
        "",
        "SAFETY-NET INSTRUCTIONS GIVEN",
        *bullets("safety_net_instructions_given"),
        "",
        "FOLLOW-UP",
        f"    discussed: {'yes' if data.get('followup_discussed') else 'no'}"
        + (
            f"; timeframe: {data['followup_timeframe']}"
            if data.get("followup_timeframe")
            else ""
        ),
    ]
    if taps.ma_note.strip():
        lines += ["", "ADDITIONAL NOTE (typed by the MA)", f"    {taps.ma_note.strip()}"]
    return lines


def render_draft(
    note: StructuredNote,
    taps: MATaps,
    registry: ProtocolRegistry,
    *,
    patient_label: str,
    call_time: datetime,
) -> str:
    """The reviewer's view: the note, plus what the machine did and did not keep."""
    registry.validate(taps)
    lines = ["*** DRAFT - NOT PART OF THE CHART UNTIL SIGNED ***", ""]
    lines += _core_body(
        note, taps, registry, patient_label=patient_label, call_time=call_time
    )

    gaps = note.system_gaps or note.data.get("transcript_gaps") or []
    if gaps:
        lines += ["", "UNCLEAR AUDIO (measured from transcription confidence)",
                  *_bullets(gaps)]
    if note.model_gaps:
        lines += [
            "",
            "GAP NOTES WRITTEN BY THE MODEL - not filed to the chart",
            *_bullets(note.model_gaps),
        ]
    if note.flagged:
        lines += [
            "",
            "KEPT BUT NOT CLEARLY SUPPORTED - check each of these",
            *[f"    - [{f['field']}] {f['text']}" for f in note.flagged],
        ]
    if note.dropped:
        lines += [
            "",
            "REMOVED BY THE SYSTEM - the transcript did not support these",
            *[
                f"    - [{d['field']}] {d['text']}  ({d['reason']})"
                for d in note.dropped
            ],
        ]
    if note.warnings:
        lines += ["", "CHECK BEFORE SIGNING", *_bullets(note.warnings)]

    lines += [
        "",
        "-" * 70,
        f"Draft produced by {note.provider}:{note.model_id}@{note.model_version}",
        f"Prompt {note.prompt_template_id}#{note.prompt_template_hash}",
        f"Transcript sha256 {note.transcript_sha256[:16]}...",
        "The protocol and disposition above were selected by the medical "
        "assistant. The system does not and cannot suggest either.",
    ]
    return "\n".join(lines)


def render_note(
    note: StructuredNote,
    taps: MATaps,
    registry: ProtocolRegistry,
    *,
    patient_label: str,
    call_time: datetime,
) -> str:
    """The chart body. No signature block -- see `signature_block`.

    WHY the signature is separate: this string is the BASELINE the edit-diff is
    taken against, and README I-04 calls that diff "the mechanism by which the
    practice proves that a human actually reviewed the draft rather than
    rubber-stamping it". If the baseline carried a signature block and the final
    text carried a different one, every untouched signature would show a
    non-zero edit distance and the "< 5% edit rate" alarm could never fire.
    """
    registry.validate(taps)
    lines = _core_body(
        note, taps, registry, patient_label=patient_label, call_time=call_time
    )
    gaps = note.system_gaps or note.data.get("transcript_gaps") or []
    if gaps:
        # Unclear audio DOES belong in the chart. It is the difference between
        # "the parent denied vomiting" and "we could not hear that part", and a
        # reader six months later cannot otherwise tell them apart. Only
        # MEASURED spans reach here; anything the model wrote stays in the draft.
        lines += ["", "PORTIONS OF THE CALL WERE UNCLEAR", *_bullets(gaps)]
    return "\n".join(lines)


def signature_block(signed_by: str, signed_at: datetime | None = None) -> str:
    """Appended AFTER the edit diff is computed. Never part of the baseline."""
    stamp = to_local(signed_at) if signed_at else None
    return "\n".join(
        [
            "",
            "-" * 70,
            f"Electronically signed by {signed_by}"
            + (f" on {stamp:%Y-%m-%d %H:%M %Z}" if stamp else ""),
            "Documentation assisted by an automated transcription and "
            "structuring system; reviewed and signed by the above.",
        ]
    )
