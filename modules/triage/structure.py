"""Transcript to structured note. README Appendix A.1, verbatim.

THE CONSTRAINT THIS MODULE EXISTS TO ENFORCE

README I-04: "Under no circumstances does the system suggest a disposition. Not
as a hint, not as a 'consider,' not greyed out... This is the single most
important design constraint in the entire document."

An unlicensed medical assistant in Illinois works under physician delegation
(225 ILCS 60/54.2) and may not exercise independent clinical judgement. A
machine-suggested disposition on the screen creates enormous pressure to accept
it and manufactures exactly the unauthorized-practice exposure the statute
exists to prevent. Prompt instructions are not enough for this, because a prompt
is a request. So the constraint is enforced four ways, in ascending order of how
much they actually protect anyone:

1. The A.1 schema has no field for a disposition, protocol or diagnosis. The
   model has nowhere to put one.
2. `assert_no_clinical_judgement_fields` walks the schema at import time and
   before every call, and refuses any key whose name reads as a clinical
   judgement. This is what stops a future maintainer "helpfully" adding
   `suggested_protocol` in a hurry.
3. Every extracted string is checked against the transcript. A.1 forbids adding
   "clinical advice that was not spoken in the transcript"; `_ground` verifies
   it. An instruction the model can ignore is a request; a post-condition it
   cannot get past is a control.
4. When the transcript is diarized, items claimed as `advice_given_by_ma` must
   match something the MA actually said. Advice attributed to the wrong speaker
   reads in the chart as advice the practice gave.

What happens to ungrounded content depends on which way the field fails.
Fabricated ADVICE is dropped -- it is the worst thing this system could produce
and the hardest for a busy reviewer to spot, because it reads exactly like
advice an MA would give. Weakly-supported OBSERVATIONS are kept and flagged,
because deleting a documented symptom on a lexical heuristic is itself a
documentation harm: a missing denial makes a note look as though the question
was never asked. See `_FIELD_POLICY`. Either way the reviewer sees it, and the
rate is the signal that tells the practice whether the prompt is drifting.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from nsp_core.llm import LLMClient, SchemaViolation

from .capture import Transcript

__all__ = [
    "STRUCTURING_SYSTEM_PROMPT",
    "NOTE_SCHEMA",
    "PROMPT_TEMPLATE_ID",
    "BANNED_FIELD_TOKENS",
    "ClinicalJudgementLeak",
    "StructuredNote",
    "NoteStructurer",
    "assert_no_clinical_judgement_fields",
    "assert_matches_a1_shape",
    "assert_output_keys_are_a1",
    "A1_KEYS",
    "GROUNDING_THRESHOLD",
]

PROMPT_TEMPLATE_ID = "README-A.1"

#: README Appendix A.1, verbatim. Its hash goes into every audit record, so a
#: change to this string is visible both in a diff and in the quality data
#: (README 9.4: prompts are code).
STRUCTURING_SYSTEM_PROMPT = """You are a clinical documentation assistant for a pediatric practice.

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
}"""

_STR_OR_NULL = {"type": ["string", "null"]}
_STR_LIST = {"type": "array", "items": {"type": "string", "maxLength": 400}}

#: The A.1 shape, strict. Note what is absent and must stay absent: no
#: disposition, no protocol, no diagnosis, no acuity, no recommendation, no
#: "suggested" anything.
NOTE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "caller",
        "chief_complaint",
        "symptom_onset",
        "symptoms_reported_present",
        "symptoms_explicitly_denied",
        "relevant_history_mentioned",
        "medications_mentioned",
        "advice_given_by_ma",
        "safety_net_instructions_given",
        "followup_discussed",
        "followup_timeframe",
        "transcript_gaps",
    ],
    "properties": {
        "caller": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "relationship_to_patient"],
            "properties": {
                "name": _STR_OR_NULL,
                "relationship_to_patient": _STR_OR_NULL,
            },
        },
        "chief_complaint": {"type": "string", "maxLength": 400},
        "symptom_onset": _STR_OR_NULL,
        "symptoms_reported_present": _STR_LIST,
        "symptoms_explicitly_denied": _STR_LIST,
        "relevant_history_mentioned": _STR_LIST,
        "medications_mentioned": _STR_LIST,
        "advice_given_by_ma": _STR_LIST,
        "safety_net_instructions_given": _STR_LIST,
        "followup_discussed": {"type": "boolean"},
        "followup_timeframe": _STR_OR_NULL,
        "transcript_gaps": _STR_LIST,
    },
}

#: Key-name fragments that mean a clinical judgement. Matched on word-ish
#: boundaries against schema property names.
#:
#: `advice_given_by_ma` is deliberately NOT caught: it is a record of what a
#: human said, verified against the transcript. The distinction this list draws
#: is between "what the MA decided" (allowed, because a human decided it and it
#: was spoken aloud) and "what the system thinks" (never).
BANNED_FIELD_TOKENS = (
    "disposition",
    "protocol",
    "diagnosis",
    "diagnostic",
    "differential",
    "acuity",
    "severity",
    "urgency",
    "triage_level",
    "risk_level",
    "recommendation",
    "recommended",
    "recommends",
    "suggested",
    "suggestion",
    "impression",
    "assessment",
    "plan_of_care",
    "care_level",
    "escalate",
    "escalation",
    "should_",
    "advise",
    "advised",
)


class ClinicalJudgementLeak(RuntimeError):
    """Raised when a schema or an output carries a clinical judgement field."""


#: The exact key set A.1 defines, at each level. An ALLOWLIST, because a
#: blocklist of judgement-sounding words is a losing game: "dispo", "next_step",
#: "red_flags", "level_of_care", "esi", "dx" and "send_to" all walk past any
#: list of banned substrings anyone will actually write, and each of them would
#: put a machine-generated disposition on an MA's screen.
A1_KEYS: Mapping[str, frozenset[str]] = {
    "<root>": frozenset(
        {
            "caller",
            "chief_complaint",
            "symptom_onset",
            "symptoms_reported_present",
            "symptoms_explicitly_denied",
            "relevant_history_mentioned",
            "medications_mentioned",
            "advice_given_by_ma",
            "safety_net_instructions_given",
            "followup_discussed",
            "followup_timeframe",
            "transcript_gaps",
        }
    ),
    "<root>.caller": frozenset({"name", "relationship_to_patient"}),
}

#: Schema keywords that can introduce properties. `assert_no_clinical_judgement_fields`
#: walks every one of them, because a banned key three levels down inside a
#: `$defs` referenced from a `dependentSchemas` is still a banned key.
_SUBSCHEMA_KEYWORDS = (
    "items", "contains", "not", "if", "then", "else", "propertyNames",
    "additionalProperties", "unevaluatedProperties",
)
_SUBSCHEMA_LISTS = ("anyOf", "oneOf", "allOf", "prefixItems")
_SUBSCHEMA_MAPS = ("$defs", "definitions", "dependentSchemas", "patternProperties")


def assert_matches_a1_shape(schema: Mapping[str, Any], *, _path: str = "<root>") -> None:
    """The note schema must be EXACTLY the A.1 key set. No more, no fewer.

    This is the primary control and the blocklist below is the secondary one.
    A future maintainer who adds a field -- with any name at all, helpful or
    otherwise -- fails here, which is the point: README I-04 says the protocol
    and the disposition "are not your output and no field exists for them in
    your schema", and the only way to keep that true is to fix the schema's
    shape rather than to guess at the names of the fields nobody should add.
    """
    expected = A1_KEYS.get(_path)
    if expected is None:
        return
    actual = frozenset((schema.get("properties") or {}).keys())
    if actual != expected:
        extra = sorted(actual - expected)
        missing = sorted(expected - actual)
        raise ClinicalJudgementLeak(
            f"note schema at {_path} does not match README Appendix A.1: "
            f"unexpected {extra}, missing {missing}. The A.1 key set is the "
            "contract; a field that is not in it must not be in the schema."
        )
    for name, sub in (schema.get("properties") or {}).items():
        if isinstance(sub, dict):
            assert_matches_a1_shape(sub, _path=f"{_path}.{name}")


def assert_no_clinical_judgement_fields(
    schema: Mapping[str, Any], *, _path: str = "<root>"
) -> None:
    """Refuse any schema with a field that would hold a clinical judgement.

    Secondary to `assert_matches_a1_shape` and kept because it applies to
    schemas this module does not own. Walks EVERY keyword that can introduce a
    property -- `$defs`, `patternProperties`, `prefixItems`, `if`/`then`,
    `dependentSchemas`, `contains` -- because a guard that inspects
    `properties`, `items` and `anyOf` is a guard with nine doors left open.
    """
    if "patternProperties" in schema:
        # A key matched by patternProperties is admitted even under
        # additionalProperties:false, so a pattern is a hole this guard cannot
        # reason about. Refuse rather than pretend to check it.
        raise ClinicalJudgementLeak(
            f"schema at {_path} uses patternProperties, which admits keys this "
            "guard cannot enumerate; spell the properties out"
        )
    for name, sub in (schema.get("properties") or {}).items():
        lowered = str(name).lower()
        for token in BANNED_FIELD_TOKENS:
            if token in lowered:
                raise ClinicalJudgementLeak(
                    f"schema field {_path}.{name!r} reads as a clinical "
                    f"judgement ({token!r}). The protocol and the disposition "
                    "come from the medical assistant's taps; the model has no "
                    "field for them and must not acquire one (README I-04)."
                )
        if isinstance(sub, dict):
            assert_no_clinical_judgement_fields(sub, _path=f"{_path}.{name}")
    for keyword in _SUBSCHEMA_KEYWORDS:
        sub = schema.get(keyword)
        if isinstance(sub, dict):
            assert_no_clinical_judgement_fields(sub, _path=f"{_path}.{keyword}")
    for keyword in _SUBSCHEMA_LISTS:
        for index, sub in enumerate(schema.get(keyword) or []):
            if isinstance(sub, dict):
                assert_no_clinical_judgement_fields(
                    sub, _path=f"{_path}.{keyword}[{index}]"
                )
    for keyword in _SUBSCHEMA_MAPS:
        for name, sub in (schema.get(keyword) or {}).items():
            if isinstance(sub, dict):
                assert_no_clinical_judgement_fields(
                    sub, _path=f"{_path}.{keyword}.{name}"
                )


def assert_output_keys_are_a1(data: Mapping[str, Any], *, _path: str = "<root>") -> None:
    """The same allowlist, applied to what actually came back, recursively."""
    expected = A1_KEYS.get(_path)
    if expected is not None:
        extra = sorted(set(data) - expected)
        if extra:
            raise ClinicalJudgementLeak(
                f"model output at {_path} contains fields outside README A.1: "
                f"{extra}; discarding the extraction"
            )
    for key, value in data.items():
        lowered = str(key).lower()
        for token in BANNED_FIELD_TOKENS:
            if token in lowered:
                raise ClinicalJudgementLeak(
                    f"model output contains a clinical-judgement field "
                    f"{_path}.{key!r}; discarding the extraction"
                )
        if isinstance(value, dict):
            assert_output_keys_are_a1(value, _path=f"{_path}.{key}")


# Fail at import. A broken constraint should not wait for a call.
assert_matches_a1_shape(NOTE_SCHEMA)
assert_no_clinical_judgement_fields(NOTE_SCHEMA)


# --------------------------------------------------------------------------
# Grounding
# --------------------------------------------------------------------------

#: Fraction of a phrase's content words that must appear in the transcript.
#: Tuned for paraphrase, not for quotation: an MA saying "she's been throwing
#: up since about six" becoming "vomiting since approximately 6pm" should pass,
#: while an invented "advised parent to give ibuprofen" should not.
GROUNDING_THRESHOLD = 0.6

_WORD = re.compile(r"[a-z0-9']+")
_STOPWORDS = frozenset(
    """a an and any are as at be been being but by can could did do does for from
    had has have he her him his how i if in into is it its just like me more most
    my no not of on or our out over said say says she should so some such than
    that the their them then there these they this to too us was we were what
    when where which who will with would you your about with been also get got
    going gonna okay ok yeah yes right well um uh""".split()
)


def _content_words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2]


#: Longest first: the loop returns on the first hit. Deliberately conservative
#: -- "ation"/"ine" are here because "medication" and "medicine" are the same
#: instruction and the prefix rule below is (correctly) too strict to see it.
_SUFFIXES = (
    "iness", "ingly", "ation", "ness", "edly", "ing", "ies", "ful", "ish",
    "ine", "ed", "es", "ly", "er", "s", "y",
)


def _stem(word: str) -> str:
    """Chop a common suffix. Crude on purpose; a real stemmer is a dependency
    and one more thing that can disagree with itself between versions."""
    for suffix in _SUFFIXES:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _near(a: str, b: str) -> bool:
    """True when two words differ by at most one character.

    Exists for exactly one recurring case: a model normalising "lose
    consciousness" to "loss of consciousness". Prefix matching cannot see that
    -- "loss" and "lose" share three characters -- and treating it as ungrounded
    would strike a documented negative out of a triage note.
    """
    if abs(len(a) - len(b)) > 1:
        return False
    if a == b:
        return True
    if len(a) == len(b):
        return sum(1 for x, y in zip(a, b) if x != y) <= 1
    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    for index in range(len(longer)):
        if longer[:index] + longer[index + 1 :] == shorter:
            return True
    return False


def _common_prefix(a: str, b: str) -> int:
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += 1
    return count


#: Below this length a word is matched by exact text, shared stem or a
#: one-character difference, and not by prefix at all. Four- and five-letter
#: words are where accidental containment lives -- bath/bathroom, rest/restless,
#: tube/tuberculosis, drop/dropper, mist/mistake -- and a fabricated "give her a
#: bath" grounding against "trips to the bathroom" is this control failing in
#: the direction that puts invented advice in a chart.
_MIN_PREFIX_WORD = 5


def _prefix_matches(a: str, b: str) -> bool:
    """True when two words share most of the shorter one, and it is long enough.

    A flat four-character prefix was the loosest rule in this module and made
    the grounding check produce false negatives -- fabricated items that were
    supposed to be caught and were not. In fifteen short fixture transcripts it
    already grounded "throat" against "throwing" and "through". Both halves of
    the replacement matter: a proportional threshold alone still degenerates to
    "share four characters" on any short word, which is exactly the band the
    accidents are in.
    """
    shorter = min(len(a), len(b))
    if shorter < _MIN_PREFIX_WORD:
        return False
    needed = max(_MIN_PREFIX_WORD, -(-shorter * 3 // 5))
    return _common_prefix(a, b) >= needed


def _matches(word: str, corpus: set[str]) -> bool:
    """Exact, shared stem, near-identical, or sharing most of the shorter word."""
    if word in corpus:
        return True
    if len(word) < 4:
        return False
    stem = _stem(word)
    for other in corpus:
        if len(other) >= 4 and (_stem(other) == stem or _near(word, other)):
            return True
        if _prefix_matches(word, other):
            return True
    return False


#: Caller relationship is a closed vocabulary, so it gets a deterministic table
#: rather than a lexical matcher. Hard constraint 5: deterministic logic is
#: implemented in Python and the model is used only where the input is genuinely
#: unstructured. "Mom" spoken and "mother" written is the single most common
#: normalisation in a pediatric triage note; flagging it on every routine call
#: is how a flag becomes something reviewers click past.
#:
#: Every entry has to be a word that means a relationship AND NOTHING ELSE.
#: "Pop", "patient", "foster" and "self" were in an earlier version of this
#: table and each of them appears in ordinary triage speech -- "her ears pop",
#: "is the patient allergic to anything" -- where they would have admitted a
#: relationship nobody claimed.
_RELATIONSHIP_SYNONYMS: Mapping[str, str] = {
    "mom": "mother", "mommy": "mother", "mum": "mother", "mama": "mother",
    "momma": "mother", "mother": "mother", "stepmom": "mother",
    "stepmother": "mother",
    "dad": "father", "daddy": "father", "father": "father",
    "stepdad": "father", "stepfather": "father",
    "grandma": "grandmother", "granny": "grandmother", "grandmother": "grandmother",
    "grandmom": "grandmother",
    "grandpa": "grandfather", "grandad": "grandfather", "granddad": "grandfather",
    "grandfather": "grandfather",
    "aunt": "aunt", "auntie": "aunt",
    "uncle": "uncle",
    "guardian": "guardian", "caregiver": "guardian",
    "babysitter": "babysitter", "nanny": "babysitter",
    "sister": "sibling", "brother": "sibling", "sibling": "sibling",
    "parent": "parent",
}

#: What a spoken term also establishes. Applied to the SPOKEN side only, never
#: the claimed side: "mom" on the call supports a note that says "parent", but
#: "parent" on the call does not support a note that says "mother".
_RELATIONSHIP_IMPLIES: Mapping[str, frozenset[str]] = {
    "mother": frozenset({"parent"}),
    "father": frozenset({"parent"}),
    "guardian": frozenset({"parent"}),
}


def _canonical_relationships(text: str) -> set[str]:
    """Every canonical relationship term a piece of text mentions."""
    return {
        _RELATIONSHIP_SYNONYMS[w]
        for w in _WORD.findall(text.lower())
        if w in _RELATIONSHIP_SYNONYMS
    }


_TERM_ALTERNATION = "|".join(sorted(_RELATIONSHIP_SYNONYMS, key=len, reverse=True))

#: A caller identifying THEMSELVES: "it's her mom", "I'm the grandmother",
#: "this is dad", "mom speaking". Bounded to one sentence -- the `[^.?!]` class
#: is what stops "I'm watching her this afternoon. Her mom is at work" from
#: reading as a self-identification.
_SELF_ID = re.compile(
    r"\b(?:i am|i'?m|it is|it'?s|this is|you'?re speaking to|speaking to)\b"
    r"[^.?!]{0,40}?\b(" + _TERM_ALTERNATION + r")\b"
    r"|\b(" + _TERM_ALTERNATION + r")\b\s+(?:here|speaking)\b",
    re.IGNORECASE,
)


def _relationships_established_by(transcript: "Transcript") -> set[str]:
    """Relationships the caller established ABOUT THEMSELVES, plus implications.

    Two narrowings, and both are load-bearing, because what this returns is a
    licence to skip the grounding check on the field that says who the practice
    took the history from.

    First, the caller's turns rather than the whole transcript: the MA's own
    script otherwise granted the vocabulary, and "Hi, is mom or dad available?"
    admitted "mother" and "father" for the rest of the call.

    Second, a self-identification pattern rather than any mention: a caller
    saying "I'm watching her this afternoon, her mom is at work until six" is
    the babysitter, and mentioning the mother is not claiming to be her. A bare
    mention admitted a note that documented the sitter as the mother, with no
    flag, in a signed chart.

    An undiarized transcript has no caller channel, so the pattern runs over the
    whole text. That is looser -- the MA could say "it's mom on the line" -- and
    it is the reason the undiarized path still warns.

    The third case is the MA asking and the caller agreeing, which is how most
    triage calls actually open: "is this Ezra's dad?" / "Yes, this is Daniel."
    Nothing in the caller's own words names the relationship, so the pattern
    above cannot see it, and treating a confirmed identification as ungrounded
    would put a flag on the most ordinary opening there is.
    """
    source = transcript.text
    if transcript.diarized:
        caller = " ".join(
            s.text for s in transcript.segments if s.speaker == "caller"
        )
        if caller.strip():
            source = caller
    spoken = {
        _RELATIONSHIP_SYNONYMS[(a or b).lower()]
        for a, b in _SELF_ID.findall(source)
    }
    spoken |= _confirmed_relationships(transcript)
    for term in list(spoken):
        spoken |= set(_RELATIONSHIP_IMPLIES.get(term, ()))
    return spoken


#: The caller agreeing to what was just asked. Anchored at the start of the
#: turn: "right now?" three turns later is not a confirmation of anything.
_AFFIRMATION = re.compile(
    r"^\W*(?:yes|yeah|yep|yup|correct|that'?s right|right|speaking|"
    r"uh[- ]?huh|mm[- ]?hmm|sure|i am|it is)\b",
    re.IGNORECASE,
)


def _confirmed_relationships(transcript: "Transcript") -> set[str]:
    """Terms the MA named in a question the caller then affirmed.

    Requires both halves and requires them adjacent, so a relationship word
    anywhere in the MA's speech grants nothing on its own.
    """
    if not transcript.diarized:
        return set()
    confirmed: set[str] = set()
    segments = transcript.segments
    for index, segment in enumerate(segments[:-1]):
        if segment.speaker != "ma":
            continue
        terms = _canonical_relationships(segment.text)
        if not terms:
            continue
        nxt = segments[index + 1]
        if nxt.speaker == "caller" and _AFFIRMATION.match(nxt.text):
            # One term only. "Is this mom or dad?" answered "yes" identifies
            # nobody, and picking one of them would be the machine guessing.
            if len(terms) == 1:
                confirmed |= terms
    return confirmed


def _relationship_is_established(value: str, spoken: set[str]) -> bool:
    """True only for a bare relationship term the caller actually established.

    The residue check is the important half. Skipping the grounding score for
    the whole field, on the strength of one recognised word in it, let
    "mother, Katarina Petrov, cell 847-555-0148, has sole custody" into a signed
    chart untouched -- including the caller name the drop policy had just
    removed as a wrong-patient risk, an invented phone number and an invented
    custody assertion, in no removal list and behind no acknowledgement gate.
    A relationship field that is anything more than a relationship gets scored
    like any other free text.
    """
    claimed = _canonical_relationships(value)
    if len(claimed) != 1:
        return False
    residue = [
        word for word in _content_words(value)
        if word not in _RELATIONSHIP_SYNONYMS
    ]
    if residue:
        return False
    return claimed <= spoken


def _grounding_score(phrase: str, corpus: set[str]) -> float:
    """Fraction of the phrase's content words present in the corpus.

    A phrase with NO content words scores ZERO, not one. "You should get her
    in" is entirely stopwords, and returning 1.0 for it made a fabricated
    disposition -- the exact output this module exists to prevent -- ground
    against any transcript, including an empty one.
    """
    words = _content_words(phrase)
    if not words:
        return 0.0
    return sum(1 for w in words if _matches(w, corpus)) / len(words)


@dataclass(frozen=True)
class _FieldPolicy:
    drop: bool
    ma_attributed: bool


#: What happens to an item the transcript does not support, per field.
#:
#: DROP for advice and safety-net instructions. A.1's prohibition is specific:
#: "You MUST NOT add clinical advice that was not spoken in the transcript."
#: Fabricated instructions in a chart are the worst thing this system could
#: produce, and they are also the thing a busy reviewer is least likely to
#: notice, because they read exactly like advice an MA would give.
#:
#: FLAG for observations, history and medications. Deleting a documented symptom
#: because a crude lexical heuristic could not match the model's paraphrase is
#: itself a documentation harm -- a dropped denial makes a note look as though
#: the question was never asked, which is precisely the gap a triage note exists
#: to close. These are kept, marked, and shown to the reviewer.
#:
#: The asymmetry is the point: the two categories fail in opposite directions,
#: so they get opposite defaults.
_FIELD_POLICY: Mapping[str, _FieldPolicy] = {
    "symptoms_reported_present": _FieldPolicy(drop=False, ma_attributed=False),
    "symptoms_explicitly_denied": _FieldPolicy(drop=False, ma_attributed=False),
    "relevant_history_mentioned": _FieldPolicy(drop=False, ma_attributed=False),
    "medications_mentioned": _FieldPolicy(drop=False, ma_attributed=False),
    "advice_given_by_ma": _FieldPolicy(drop=True, ma_attributed=True),
    "safety_net_instructions_given": _FieldPolicy(drop=True, ma_attributed=True),
}


@dataclass
class StructuredNote:
    """The extraction, plus everything a reviewer needs to trust it."""

    encounter_id: str
    data: dict[str, Any]
    transcript_sha256: str
    model_id: str
    model_version: str
    provider: str
    prompt_template_id: str
    prompt_template_hash: str
    diarized: bool
    #: Items the model produced that the transcript does not support, REMOVED
    #: from `data`. Kept here so the MA can see what went and the practice can
    #: watch the rate.
    dropped: list[dict[str, Any]] = field(default_factory=list)
    #: Items kept but weakly supported. See `_FIELD_POLICY` for why these are
    #: not simply deleted too. `render.py` marks them in the chart and
    #: `review.py` will not sign until they are acknowledged.
    flagged: list[dict[str, Any]] = field(default_factory=list)
    #: Unclear-audio spans MEASURED from transcription confidence. These reach
    #: the chart.
    system_gaps: list[str] = field(default_factory=list)
    #: Gap notes the model WROTE. Shown to the reviewer, never auto-filed.
    model_gaps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    inference_id: str | None = None

    @property
    def clean(self) -> bool:
        return not self.dropped and not self.flagged and not self.warnings

    def to_json(self) -> str:
        """A lossless round-trip for `TriageService.resume`.

        Distinct from `as_dict`, which is an audit VIEW -- it flattens the model
        and prompt identifiers into display strings and would come back as a
        different object. This keeps every field, so a note reconstructed after
        a crash carries the same inference id, the same drops and the same flags
        into `review.sign` as the original would have. Re-running the model
        instead would be a second inference with no second review event.
        """
        return json.dumps(
            {
                f.name: getattr(self, f.name)
                for f in dataclasses.fields(self)
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, payload: str) -> "StructuredNote":
        return cls(**json.loads(payload))

    def as_dict(self) -> dict[str, Any]:
        return {
            "encounter_id": self.encounter_id,
            "data": self.data,
            "transcript_sha256": self.transcript_sha256,
            "model": f"{self.provider}:{self.model_id}@{self.model_version}",
            "prompt": f"{self.prompt_template_id}#{self.prompt_template_hash}",
            "diarized": self.diarized,
            "dropped": list(self.dropped),
            "flagged": list(self.flagged),
            "system_gaps": list(self.system_gaps),
            "model_gaps": list(self.model_gaps),
            "warnings": list(self.warnings),
        }


class NoteStructurer:
    """Runs A.1 and enforces the constraints the prompt only asks for."""

    INITIATIVE = "I-04"

    def __init__(
        self,
        client: LLMClient,
        *,
        audit: Any = None,
        grounding_threshold: float = GROUNDING_THRESHOLD,
        schema: Mapping[str, Any] = NOTE_SCHEMA,
        system_prompt: str = STRUCTURING_SYSTEM_PROMPT,
    ) -> None:
        assert_matches_a1_shape(schema)
        assert_no_clinical_judgement_fields(schema)
        self.client = client
        self.audit = audit
        self.grounding_threshold = grounding_threshold
        self.schema = schema
        self.system_prompt = system_prompt

    @property
    def prompt_hash(self) -> str:
        return hashlib.sha256(self.system_prompt.encode("utf-8")).hexdigest()[:16]

    def structure(
        self,
        transcript: Transcript,
        *,
        patient_id: str | None = None,
        user_id: str = "system:triage",
    ) -> StructuredNote:
        """Extract a note from a transcript. Raises SchemaViolation to a human.

        There is no fallback note, no partial extraction, no "best effort"
        result. README 3.5 is fail-closed: if the model cannot produce a valid
        structure, the MA types the note the way they do today. That is a worse
        minute for the MA and a much better outcome than a plausible note nobody
        can vouch for.
        """
        # Re-check on the way in. The import-time check protects the module
        # constant; this protects whatever the caller actually passed.
        assert_matches_a1_shape(self.schema)
        assert_no_clinical_judgement_fields(self.schema)

        user = self._build_user_message(transcript)
        result = self.client.structured(
            system=self.system_prompt,
            user=user,
            schema=self.schema,
            prompt_template_id=PROMPT_TEMPLATE_ID,
            context={"initiative": self.INITIATIVE, "patient_id": patient_id},
        )

        inference_id = None
        if self.audit is not None:
            inference_id = self.audit.record_inference(
                user_id=user_id,
                initiative_id=self.INITIATIVE,
                provider=result.provider,
                model_id=result.model_id,
                model_version=result.model_version,
                prompt_template_id=PROMPT_TEMPLATE_ID,
                prompt_template_hash=self.prompt_hash,
                input_token_count=result.input_token_count,
                output_token_count=result.output_token_count,
                patient_id=patient_id,
                confidence_score=None,
                constrained_decoding=result.constrained,
                repair_attempts=result.repair_attempts,
                extra={
                    "transcript_sha256": transcript.sha256,
                    "transcriber_model": transcript.model_id,
                    "diarized": transcript.diarized,
                },
            )

        note = StructuredNote(
            encounter_id=transcript.encounter_id,
            data=dict(result.data),
            transcript_sha256=transcript.sha256,
            model_id=result.model_id,
            model_version=result.model_version,
            provider=result.provider,
            prompt_template_id=PROMPT_TEMPLATE_ID,
            prompt_template_hash=self.prompt_hash,
            diarized=transcript.diarized,
            inference_id=inference_id,
        )
        self._verify_output_keys(note)
        self._ground(note, transcript)
        self._carry_gap_hints(note, transcript)
        return note

    # -- prompt ------------------------------------------------------------

    def _build_user_message(self, transcript: Transcript) -> str:
        parts = [
            "TRANSCRIPT:",
            transcript.labelled_text or "(no speech detected)",
        ]
        if not transcript.diarized:
            # Say so rather than letting the model assume. An undiarized
            # transcript is a dialogue with the name tags removed, and a model
            # that does not know that will confidently attribute the parent's
            # words to the MA.
            parts.append(
                "\nNOTE: this transcript has NO speaker labels. Do not assume who "
                "said any given line. If you cannot tell whether the medical "
                "assistant or the caller said something, leave the corresponding "
                "field empty rather than guessing."
            )
        hints = transcript.gap_hints()
        if hints:
            parts.append(
                "\nAUDIO QUALITY: the following spans were unclear. Include them "
                "in transcript_gaps:\n" + "\n".join(f"- {h}" for h in hints)
            )
        return "\n".join(parts)

    # -- post-conditions ---------------------------------------------------

    def _verify_output_keys(self, note: StructuredNote) -> None:
        """The allowlist, applied recursively to what actually came back.

        The schema forbids additional properties and the validator enforces it,
        so this should never fire. It exists because "should never fire" is a
        claim about a library's behaviour, and this particular constraint is one
        the practice's legal position rests on.
        """
        assert_output_keys_are_a1(note.data)

    def _ground(self, note: StructuredNote, transcript: Transcript) -> None:
        """Drop anything the transcript does not support.

        A.1 forbids adding clinical advice that was not spoken. This is where
        that stops being an instruction. Two corpora are used: everything said,
        and -- when the transcript is diarized -- what the MA specifically said,
        because advice attributed to the wrong speaker reads in the chart as
        advice the practice gave.
        """
        everything = set(_content_words(transcript.text))
        ma_only = everything
        ma_attributable = False
        if transcript.diarized:
            candidate = set(
                _content_words(
                    " ".join(s.text for s in transcript.segments if s.speaker == "ma")
                )
            )
            if candidate:
                ma_only = candidate
                ma_attributable = True
            else:
                # Diarized, but nothing is labelled "ma" -- a diarizer emitting
                # SPEAKER_00/SPEAKER_01, or a mislabelled call. Attributing
                # advice against an empty corpus deletes every genuinely spoken
                # safety-net instruction and records a true statement ("not
                # attributable to the MA") about a false premise.
                note.warnings.append(
                    "speaker labels are present but none identify the medical "
                    "assistant; advice attribution could not be checked"
                )

        for field_name, policy in _FIELD_POLICY.items():
            corpus = ma_only if policy.ma_attributed else everything
            kept: list[str] = []
            for item in note.data.get(field_name) or []:
                text = str(item).strip()
                if not text:
                    continue
                score = _grounding_score(text, corpus)
                if score >= self.grounding_threshold:
                    kept.append(text)
                    continue
                record = {
                    "field": field_name,
                    "text": text,
                    "grounding": round(score, 2),
                    "reason": (
                        "not attributable to the medical assistant in the transcript"
                        if policy.ma_attributed and ma_attributable
                        else "not supported by the transcript"
                    ),
                }
                if policy.drop:
                    note.dropped.append(record)
                else:
                    kept.append(text)
                    note.flagged.append(record)
            note.data[field_name] = kept
        if note.flagged:
            note.warnings.append(
                f"{len(note.flagged)} item(s) are weakly supported by the "
                "transcript and are marked for checking, not removed"
            )

        for field_name in ("chief_complaint", "symptom_onset", "followup_timeframe"):
            value = note.data.get(field_name)
            if not value:
                continue
            score = _grounding_score(str(value), everything)
            if score < self.grounding_threshold:
                note.warnings.append(
                    f"{field_name} is weakly supported by the transcript "
                    f"(grounding {score:.2f}); check it before signing"
                )

        # Identity is dropped; relationship is flagged. A caller NAME the
        # transcript does not contain is a wrong-patient risk and names are
        # spoken verbatim, so an unmatched one is almost certainly invented. A
        # RELATIONSHIP is the opposite: "mom" in the transcript becoming
        # "mother" in the note is the normal clinical normalisation, no lexical
        # matcher this crude will ever see it, and deleting it throws away
        # context that costs nothing to keep and mark.
        caller = note.data.get("caller") or {}
        spoken_relationships = _relationships_established_by(transcript)
        for key, drop in (("name", True), ("relationship_to_patient", False)):
            value = caller.get(key)
            if not value:
                continue
            if key == "relationship_to_patient" and _relationship_is_established(
                str(value), spoken_relationships
            ):
                continue
            score = _grounding_score(str(value), everything)
            if score >= self.grounding_threshold:
                continue
            record = {
                "field": f"caller.{key}",
                "text": str(value),
                "grounding": round(score, 2),
                "reason": "does not appear in the transcript",
            }
            if drop:
                note.dropped.append(record)
                caller[key] = None
            else:
                note.flagged.append(record)

    def _carry_gap_hints(self, note: StructuredNote, transcript: Transcript) -> None:
        """Separate the machine-MEASURED gaps from the machine-WRITTEN ones.

        `render_note` prints unclear-audio spans into the signed chart, which is
        right -- a reader has to be able to tell "the parent denied it" from "we
        could not hear that part". But that makes `transcript_gaps` a free-text
        channel from the model straight into a clinical record, and a model that
        writes "findings are consistent with early sepsis" there has put a
        diagnosis in the chart through the one field nobody was checking.

        So the chart gets `system_gaps` only: spans derived arithmetically from
        transcription confidence. Anything the model wrote goes to `model_gaps`,
        which the reviewer sees in the draft and can choose to type in.
        """
        system_gaps = transcript.gap_hints()
        model_gaps = [
            str(g).strip()
            for g in (note.data.get("transcript_gaps") or [])
            if str(g).strip() and str(g).strip() not in system_gaps
        ]
        note.system_gaps = list(system_gaps)
        note.model_gaps = model_gaps
        note.data["transcript_gaps"] = list(system_gaps)
        if system_gaps:
            note.warnings.append(
                f"{len(system_gaps)} span(s) of unclear audio; the note may be "
                "incomplete"
            )
        if model_gaps:
            note.warnings.append(
                f"the model reported {len(model_gaps)} additional gap note(s); "
                "they are shown in the draft and are NOT auto-filed to the chart"
            )
