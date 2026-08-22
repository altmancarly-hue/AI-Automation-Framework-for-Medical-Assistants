"""The one model call in I-03, and the post-conditions that contain it.

README I-03 is unusually clear about what the model is for and what it is not:

    "the rules engine produces the checklist; the LLM produces the narrative
    context. Both are needed and they are not interchangeable."

And about what it must not do:

    "The model is instructed explicitly: it may not generate clinical
    recommendations, diagnoses, or suggested orders. It reports what is in the
    record. A brief that says 'consider asthma workup' is out of scope and out
    of the MA's lane under 54.2."

THE INSTRUCTION IS A REQUEST. THE POST-CONDITIONS ARE THE CONTROL. Appendix A.4
tells the model every item must cite its source encounter date; this module
checks, and drops what does not comply, whether or not the model complied. Four
checks run on every response, in this order:

  1. **Strict schema.** `additionalProperties: false`, no field for a
     recommendation, an assessment, a plan or a diagnosis -- the same allowlist
     discipline as I-04's note schema, for the same reason.
  2. **The cited date must be real.** Not merely present: it has to match one of
     the encounter dates actually supplied. A model that invents "2026-03-14"
     produces an item that looks better sourced than an uncited one.
  3. **The claim must appear in THAT encounter's note.** A true statement filed
     under the wrong date sends a clinician to the wrong place in the chart,
     which is worse than no pointer at all.
  4. **No clinical judgement language.** "Consider", "recommend", "rule out",
     "start", "increase the dose" -- an item carrying any of it is dropped and
     recorded, because the failure this prevents is a machine-authored order
     appearing in front of a person who is not licensed to write one.

WHAT THESE CHECKS CANNOT DO, stated plainly rather than left to be discovered.
They are lexical. `_polarity_conflict` catches an inverted negation and
`_order_grounding` catches a reversed clause, but neither is a language model
and some inversions need a human. De-identification is regex-first: it removes
phone numbers, record numbers and identifiers, and it does NOT remove person
names unless the optional clinical NER model is installed -- which is why a
warning is attached to every brief where it was not. Both of those are reasons
the README puts this on LOCAL inference (Section 3.1) and marks the whole
narrative section as AI-generated: the residue is handled by a clinician who
has been told the brief is a pointer, not by a claim that the filters are
complete.

Everything dropped is kept and shown. README I-03's highest-severity risk is "AI
narrative contains a hallucinated detail", and its control is that the clinician
is trained the brief is a pointer rather than a source of truth. A drop count
the reviewer can see is what turns that from training into evidence.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Mapping, Sequence

from nsp_core.llm import LLMClient, SchemaViolation

__all__ = [
    "NARRATIVE_SYSTEM_PROMPT",
    "NARRATIVE_SCHEMA",
    "NARRATIVE_SECTIONS",
    "MAX_ENCOUNTERS",
    "GROUNDING_THRESHOLD",
    "ORDER_THRESHOLD",
    "Encounter",
    "NarrativeItem",
    "NarrativeContext",
    "NarrativeSynthesizer",
    "ClinicalJudgementLeak",
    "assert_no_judgement_fields",
]

#: README I-03: "The model receives only the last 3 encounter notes plus the
#: problem list -- not the entire chart. Minimum necessary, and it keeps latency
#: and cost down." Enforced here rather than trusted to the caller.
MAX_ENCOUNTERS = 3

#: README Appendix A.4, verbatim. Editing this string is a change-controlled
#: event (README 9.4: prompts are code) -- `prompt_hash` is recorded on every
#: inference so an edit is visible in the audit log.
NARRATIVE_SYSTEM_PROMPT = """You are preparing a context brief for a pediatric clinician before a scheduled
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
}"""

NARRATIVE_SECTIONS: tuple[str, ...] = (
    "recent_relevant_history",
    "open_threads",
    "unresolved_parent_concerns",
    "medication_changes",
)

_ITEM = {
    "type": "object",
    "properties": {
        "item": {"type": "string"},
        "source_date": {"type": "string"},
    },
    "required": ["item", "source_date"],
    "additionalProperties": False,
}

NARRATIVE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {name: {"type": "array", "items": _ITEM} for name in NARRATIVE_SECTIONS},
    "required": list(NARRATIVE_SECTIONS),
    "additionalProperties": False,
}


class ClinicalJudgementLeak(RuntimeError):
    """Raised when a schema grows a field that would carry clinical judgement."""


#: The A.4 key set IS the contract. An allowlist rather than a banned-token
#: list, because the field that carries a disposition is as likely to be called
#: `next_step` as `recommendation` -- a lesson from I-04, where a blocklist
#: passed exactly that.
_A4_KEYS = frozenset(NARRATIVE_SECTIONS)
_A4_ITEM_KEYS = frozenset({"item", "source_date"})


def assert_no_judgement_fields(schema: Mapping[str, Any]) -> None:
    """Refuse any narrative schema that is not exactly Appendix A.4's shape."""
    if schema.get("additionalProperties") is not False:
        raise ClinicalJudgementLeak("the narrative schema must be closed")
    keys = frozenset(schema.get("properties", {}))
    if keys != _A4_KEYS:
        raise ClinicalJudgementLeak(
            f"narrative schema does not match README Appendix A.4: unexpected "
            f"{sorted(keys - _A4_KEYS)}, missing {sorted(_A4_KEYS - keys)}. The "
            "A.4 key set is the contract; a field outside it is a channel for "
            "clinical judgement this module has no authority to produce."
        )
    for name, sub in schema["properties"].items():
        item = (sub or {}).get("items") or {}
        if item.get("additionalProperties") is not False:
            raise ClinicalJudgementLeak(f"{name}.items must be closed")
        if frozenset(item.get("properties", {})) != _A4_ITEM_KEYS:
            raise ClinicalJudgementLeak(
                f"{name}.items must carry exactly {sorted(_A4_ITEM_KEYS)}"
            )


assert_no_judgement_fields(NARRATIVE_SCHEMA)


#: Language that turns a report into an order.
#:
#: THIS IS A BLOCKLIST AND THAT IS A KNOWN WEAKNESS. The schema above is an
#: allowlist -- the A.4 key set is closed and nothing outside it can be
#: returned. But the ITEM TEXT is free prose, and there is no allowlist for
#: free prose. So this list is deliberately wide, it is deliberately biased
#: toward over-dropping, and it is not the only control: every surviving item
#: is marked AI-generated, carries its source date, and sits under a header
#: telling the reader to review before relying on it.
#:
#: The first version of this list matched `recommend`, `consider` and
#: `refer(ral) to|for`, and let all of these through:
#:     "Needs inhaled steroid started for the nightly cough"
#:     "GI referral warranted for the reflux"
#:     "Asthma action plan may benefit from escalation"
#: Ordinary clinical word order puts the noun first, so anchoring on verbs
#: missed the sentences that read most like orders.
_JUDGEMENT_PATTERNS: tuple[tuple[str, "re.Pattern[str]"], ...] = tuple(
    (label, re.compile(pattern, re.IGNORECASE))
    for label, pattern in (
        # DEONTIC: something ought to happen. Never exempt -- see below.
        (
            "recommendation",
            r"\b(?:recommend\w*|advis\w*|suggest\w*|should\b|ought to|"
            r"warrant\w*|indicated\b|appropriate to|reasonable to|"
            r"(?:may|would|might|could) benefit|worth\b|needs?\b|requir\w*|"
            r"due for|ensure\b|escalat\w*)",
        ),
        # ACTION: an order or a task. Exempt when the item is plainly reporting
        # that it already happened -- see `_PAST_REPORT`.
        (
            "suggested order",
            r"\b(?:consider\w*|obtain\w*|order(?:ing|ed)?\b|referr?al\b|refer\b|"
            r"work[- ]?up|initiate\w*|trial of|prescrib\w*|check\b|recheck\w*|"
            r"repeat\b|screen for|test for|schedul\w*|arrange\w*|monitor\w*|"
            r"follow[- ]?up (?:in|with|at))",
        ),
        (
            "differential",
            r"\b(?:rule out|r/o\b|differential|likely (?:represents|due to|"
            r"secondary to)|concerning for|suspicious for|consistent with|"
            r"suggestive of|presumed\b|probable\b|cannot exclude|query\b|"
            r"possible \w+ (?:disease|disorder|syndrome))",
        ),
        (
            "medication change",
            r"\b(?:increas\w*|decreas\w*|titrat\w*|taper\w*|discontinu\w*|"
            r"restart\w*|switch\w* to|step (?:up|down))\b",
        ),
        (
            "prognosis",
            r"\b(?:will (?:need|require|likely)|expect(?:ed)? to|prognosis|"
            r"anticipate\w*|at risk (?:for|of))",
        ),
    )
)

#: Past-tense reporting. "Allergy referral placed 2026-03-14, no report
#: received" is the README's own example of an open thread and it is a FACT, not
#: an order -- but it contains the word "referral", and a flat blocklist deleted
#: it. Only the ACTION class is exempted by this: a deontic item stays blocked
#: however it is phrased, so "needs the referral placed" is still an order.
_PAST_REPORT = re.compile(
    r"\b(?:was|were|had|has been|have been|placed|made|sent|given|done|"
    r"completed|performed|obtained|drawn|prescribed|received|declined|"
    r"discussed|reviewed|referred|took|attended|returned)\b",
    re.IGNORECASE,
)
_EXEMPTIBLE = frozenset({"suggested order"})

_WORD = re.compile(r"[a-z0-9']+")
_STOPWORDS = frozenset(
    """a an and any are as at be been being but by can could did do does for from
    had has have he her him his how i if in into is it its just like me more most
    my of on or our out over said say says she so some such than
    that the their them then there these they this to too us was we were what
    when where which who will with would you your about also get got""".split()
)

#: Words that flip the meaning of what follows. Deliberately NOT in _STOPWORDS
#: -- an earlier version stripped "no" and "not" as noise, which is how
#: "Positive depression screen with suicidal ideation" scored 0.80 against a
#: note reading "PHQ-A negative. Denies suicidal ideation."
_NEGATION_TRIGGERS = frozenset(
    """no not never none negative denies denied denying without absent lacks
    unremarkable nil neither nor free ruled""".split()
)

#: How far a negation reaches. Clinical negation scope is short and ends at a
#: clause boundary; six tokens is the conventional NegEx window.
_NEGATION_WINDOW = 6
_CLAUSE_BREAK = re.compile(r"[.;:,]|\b(?:but|however|although|except)\b")


def _content_words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2]


def _negated_terms(text: str) -> set[str]:
    """Content words that appear ONLY inside a negation's scope.

    A word negated in one sentence and asserted in another is not treated as
    negated -- "No wheeze today. Wheeze last week." asserts wheeze somewhere,
    and dropping an item about it would delete a true finding.
    """
    negated: set[str] = set()
    asserted: set[str] = set()
    for clause in _CLAUSE_BREAK.split(text.lower()):
        tokens = _WORD.findall(clause)
        scope_left = 0
        for token in tokens:
            if token in _NEGATION_TRIGGERS:
                scope_left = _NEGATION_WINDOW
                continue
            if token in _STOPWORDS or len(token) <= 2:
                continue
            (negated if scope_left > 0 else asserted).add(token)
            if scope_left > 0:
                scope_left -= 1
    return negated - asserted


def _bigrams(words: Sequence[str]) -> set[tuple[str, str]]:
    return {(a, b) for a, b in zip(words, words[1:])}


def _polarity_conflict(item: str, note: str) -> str | None:
    """A term the note negates and the item asserts, or the reverse.

    This is the only check in the module that can see MEANING rather than
    vocabulary, and it exists because the token-overlap score is at its highest
    exactly when a claim reuses the note's words with the sense inverted. Those
    are the most dangerous items the model can produce: they read as
    well-sourced and they are the opposite of the record.

    It is a lexical approximation of negation, not a language model, and it does
    not catch every inversion -- word-order reversals ("started since stopping"
    for "stopped since starting") need the bigram check below, and some
    inversions need a human. That residue is why every item on the brief carries
    its source date and sits under an AI-generated header: README I-03's stated
    control is that the brief is a pointer, not a source of truth.
    """
    note_negated = _negated_terms(note)
    item_negated = _negated_terms(item)
    item_terms = set(_content_words(item))
    flipped_to_positive = sorted((item_terms - item_negated) & note_negated)
    if flipped_to_positive:
        return (
            f"the {', '.join(flipped_to_positive[:3])} in the cited note is "
            "negated and this item asserts it"
        )
    flipped_to_negative = sorted(
        w for w in item_negated if w not in note_negated and w in set(_content_words(note))
    )
    if flipped_to_negative:
        return (
            f"this item negates {', '.join(flipped_to_negative[:3])}, which the "
            "cited note asserts"
        )
    return None


#: Fraction of an item's content words that must appear in the cited note. Not
#: a semantic check and never claimed as one -- see `_polarity_conflict`.
GROUNDING_THRESHOLD = 0.6

#: Fraction of the item's adjacent content-word pairs that must survive in the
#: note. Lower than the unigram bar because a legitimate summary reorders and
#: compresses, but not zero, because a total reordering is how the same words
#: state the opposite thing.
ORDER_THRESHOLD = 0.34

_HAS_DIGIT = re.compile(r"\d")

#: Below this length a word matches only exactly or by a shared stem. The 5-char
#: prefix rule accepted "pneumonia" for "pneumococcal" -- a vaccine turned into
#: an infection -- and any two long clinical terms sharing a stem matched each
#: other.
_MIN_PREFIX_WORD = 6


def _stemmed_hit(word: str, corpus: set[str]) -> bool:
    if _HAS_DIGIT.search(word):
        # Numbers are never approximated. "500 mg" must not ground against
        # "250 mg", and a token containing a digit that is skipped by the prefix
        # loop is a changed number that was never compared to anything.
        return False
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and word[: -len(suffix)] in corpus:
            return True
    if len(word) < _MIN_PREFIX_WORD:
        return False
    for other in corpus:
        if len(other) < _MIN_PREFIX_WORD or _HAS_DIGIT.search(other):
            continue
        shared = 0
        for a, b in zip(word, other):
            if a != b:
                break
            shared += 1
        if shared >= max(_MIN_PREFIX_WORD, -(-min(len(word), len(other)) * 4 // 5)):
            return True
    return False


def _unmatched_numbers(item: str, corpus: set[str]) -> list[str]:
    """Numeric tokens in the item that do not appear in the cited note.

    A changed number is a hard drop rather than a fraction off the score.
    "Started amoxicillin 500 mg" against a note saying 250 mg scored 0.67 and
    passed, because two of its three content words matched and the third was a
    digit string the prefix rule skipped entirely. A dose, a count or a lab
    value that differs from the record is precisely what a pointer must never
    invent.
    """
    return [
        word for word in _content_words(item)
        if _HAS_DIGIT.search(word) and word not in corpus
    ]


def _grounding(item: str, corpus: set[str]) -> float:
    words = _content_words(item)
    if not words:
        return 0.0
    return sum(1 for w in words if w in corpus or _stemmed_hit(w, corpus)) / len(words)


def _order_grounding(item: str, note_words: Sequence[str]) -> float:
    """How much of the item's word ORDER survives in the note.

    Bag-of-words scored "the seizures started since stopping the medication" at
    1.00 against a note saying they stopped since starting it. Same words,
    opposite meaning, perfect score.
    """
    item_words = _content_words(item)
    if len(item_words) < 2:
        return 1.0
    note_pairs = _bigrams(list(note_words))
    item_pairs = _bigrams(item_words)
    return sum(1 for pair in item_pairs if pair in note_pairs) / len(item_pairs)


@dataclass(frozen=True)
class Encounter:
    """One prior encounter. `note_text` is what the model is allowed to see."""

    encounter_date: date
    visit_type: str
    note_text: str

    @property
    def iso(self) -> str:
        return self.encounter_date.isoformat()


@dataclass(frozen=True)
class NarrativeItem:
    section: str
    text: str
    source_date: str

    def as_dict(self) -> dict[str, Any]:
        return {"section": self.section, "item": self.text, "source_date": self.source_date}


@dataclass
class NarrativeContext:
    """The narrative half of the brief, plus everything that was rejected."""

    patient_id: str
    items: list[NarrativeItem] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    model_id: str = ""
    model_version: str = ""
    provider: str = ""
    prompt_template_id: str = "I-03/A.4"
    prompt_hash: str = ""
    inference_id: str | None = None
    source_dates: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.items

    def section(self, name: str) -> list[NarrativeItem]:
        return [i for i in self.items if i.section == name]

    def as_dict(self) -> dict[str, Any]:
        return {
            "items": [i.as_dict() for i in self.items],
            "dropped": list(self.dropped),
            "warnings": list(self.warnings),
            "model": f"{self.provider}:{self.model_id}@{self.model_version}",
            "prompt": f"{self.prompt_template_id}#{self.prompt_hash}",
            "inference_id": self.inference_id,
        }


class NarrativeSynthesizer:
    """Turns the last three notes into cited pointers, or into nothing at all."""

    INITIATIVE = "I-03"

    def __init__(
        self,
        client: LLMClient,
        *,
        audit: Any = None,
        deidentifier: Any = None,
        leak_guard: Any = None,
        grounding_threshold: float = GROUNDING_THRESHOLD,
        order_threshold: float = ORDER_THRESHOLD,
        schema: Mapping[str, Any] | None = None,
        system_prompt: str = NARRATIVE_SYSTEM_PROMPT,
    ) -> None:
        self.client = client
        self.audit = audit
        self.deidentifier = deidentifier
        self.leak_guard = leak_guard
        self.grounding_threshold = grounding_threshold
        self.order_threshold = order_threshold
        self.schema = dict(schema or NARRATIVE_SCHEMA)
        assert_no_judgement_fields(self.schema)
        self.system_prompt = system_prompt

    def _ner_available(self) -> bool:
        ner = getattr(self.deidentifier, "ner", None)
        if ner is None:
            return False
        available = getattr(ner, "available", None)
        return bool(available() if callable(available) else available)

    @property
    def prompt_hash(self) -> str:
        return hashlib.sha256(self.system_prompt.encode("utf-8")).hexdigest()[:16]

    # -- prompt assembly ---------------------------------------------------

    def _user_message(
        self, encounters: Sequence[Encounter], problem_list: Sequence[str]
    ) -> str:
        parts = ["PROBLEM LIST:"]
        parts.extend(f"- {p}" for p in problem_list) if problem_list else parts.append("- none recorded")
        for enc in encounters:
            parts.append("")
            parts.append(f"ENCOUNTER {enc.iso} ({enc.visit_type}):")
            parts.append(enc.note_text.strip())
        return "\n".join(parts)

    # -- the call ----------------------------------------------------------

    def synthesize(
        self,
        encounters: Sequence[Encounter],
        *,
        patient_id: str,
        problem_list: Sequence[str] = (),
        user_id: str = "system:previsit",
    ) -> NarrativeContext:
        """One inference, then four post-conditions. Never raises on model junk.

        A brief whose narrative section failed is still a useful brief -- the
        screenings, immunizations and growth flags are all computed. So a
        SchemaViolation degrades this section to empty with a visible warning
        rather than taking down the batch. Hard constraint 3 is satisfied
        because nothing is guessed or defaulted: the section is simply absent
        and says why.
        """
        # Minimum necessary, enforced. Most recent three, newest first.
        ordered = sorted(encounters, key=lambda e: e.encounter_date, reverse=True)
        selected = ordered[:MAX_ENCOUNTERS]
        context = NarrativeContext(
            patient_id=patient_id,
            prompt_hash=self.prompt_hash,
            source_dates=tuple(e.iso for e in selected),
        )
        if len(ordered) > MAX_ENCOUNTERS:
            context.warnings.append(
                f"{len(ordered)} encounters supplied; the model saw the most "
                f"recent {MAX_ENCOUNTERS} (minimum necessary, README I-03)"
            )
        if not selected:
            context.warnings.append("no prior encounters; no narrative context")
            return context

        vault = None
        visible = selected
        visible_problems = list(problem_list)
        if self.deidentifier is not None:
            from nsp_core.phi import TokenVault

            vault = TokenVault()
            visible = [
                Encounter(
                    e.encounter_date,
                    e.visit_type,
                    self.deidentifier.deidentify(e.note_text, vault=vault).text,
                )
                for e in selected
            ]
            # The problem list went to the model raw. It is free text from the
            # same chart and routinely carries a phone number, an MRN or a
            # sibling's name -- everything in the user message has to go through
            # the same pass, not just the part that was obviously prose.
            visible_problems = [
                self.deidentifier.deidentify(str(p), vault=vault).text
                for p in problem_list
            ]
            if not self._ner_available():
                # Loud, every time, rather than a footnote in a runbook.
                # Regex de-identification removes phone numbers, MRNs and dates.
                # It does NOT remove person names -- "Followed by Dr. Robert Chen
                # at Lurie" survives it intact, and `LeakGuard` re-derives with
                # the same regexes so it does not catch them either. A caller who
                # passed a de-identifier is entitled to know the pass is partial.
                context.warnings.append(
                    "de-identification ran WITHOUT the clinical NER model, so "
                    "person names were not removed from the text sent to the "
                    "model. This is only acceptable on a local-inference "
                    "deployment (README 3.1); it is not Safe Harbor."
                )

        message = self._user_message(visible, visible_problems)
        if self.leak_guard is not None:
            # Fails closed. Encounter DATES are allowed through deliberately --
            # Appendix A.4 requires every item to cite one, so the citation
            # control and Safe Harbor pull in opposite directions here. That is
            # a reason to run this locally (README 3.1), not a reason to drop
            # the citation.
            self.leak_guard.assert_clean(message, vault=vault, allow_labels=("DATE",))

        try:
            result = self.client.structured(
                system=self.system_prompt,
                user=message,
                schema=self.schema,
                context={
                    "initiative_id": self.INITIATIVE,
                    "user_id": user_id,
                    "patient_id": patient_id,
                },
                prompt_template_id=context.prompt_template_id,
                temperature=0.0,
            )
        except SchemaViolation as exc:
            # Hard constraint 3: no guess, no default. The section is empty and
            # the brief says so, which is a smaller failure than a wrong pointer.
            context.warnings.append(
                f"narrative synthesis failed schema validation and was discarded "
                f"({exc}); the computed sections of this brief are unaffected"
            )
            return context

        data = result.data
        if vault is not None:
            data = vault.rehydrate_obj(data)
        context.model_id = result.model_id
        context.model_version = result.model_version
        context.provider = result.provider
        self._accept(context, data, selected)

        # Hard constraint: every inference is recorded. This was missing
        # entirely -- a narrative could render on a clinician's screen with the
        # audit database holding zero rows about it, so there was no way to
        # answer "which model wrote this line" after the fact, which is exactly
        # what README 9.2 and risk R-12 require.
        if self.audit is not None:
            context.inference_id = self.audit.record_inference(
                user_id=user_id,
                initiative_id=self.INITIATIVE,
                provider=result.provider,
                model_id=result.model_id,
                model_version=result.model_version,
                prompt_template_id=result.prompt_template_id,
                prompt_template_hash=result.prompt_template_hash,
                input_token_count=result.input_token_count,
                output_token_count=result.output_token_count,
                patient_id=patient_id,
                confidence_score=result.confidence,
                constrained_decoding=result.constrained,
                repair_attempts=result.repair_attempts,
                extra={
                    "encounters_supplied": len(ordered),
                    "encounters_used": len(selected),
                    "items_kept": len(context.items),
                    "items_dropped": len(context.dropped),
                },
            )
        return context

    # -- post-conditions ---------------------------------------------------

    def _accept(
        self,
        context: NarrativeContext,
        data: Mapping[str, Any],
        encounters: Sequence[Encounter],
    ) -> None:
        by_date = {e.iso: e.note_text for e in encounters}
        words_by_date = {iso: _content_words(note) for iso, note in by_date.items()}
        corpus_by_date = {iso: set(words) for iso, words in words_by_date.items()}
        for section in NARRATIVE_SECTIONS:
            for raw in data.get(section) or []:
                text = str(raw.get("item", "")).strip()
                cited = str(raw.get("source_date", "")).strip()
                if not text:
                    # Recorded, not skipped. Every other rejection in this
                    # module is counted and shown, and an item that vanished
                    # silently is the one nobody knows to ask about.
                    self._drop(context, section, str(raw.get("item", "")), cited,
                               "empty item")
                    continue
                if cited not in by_date:
                    self._drop(
                        context, section, text, cited,
                        "cites an encounter date that was not supplied"
                        if cited
                        else "cites no encounter date",
                    )
                    continue
                judgement = self._judgement_label(text)
                if judgement is not None:
                    self._drop(
                        context, section, text, cited,
                        f"reads as a {judgement}; A.4 forbids recommendations, "
                        "differentials and suggested orders",
                    )
                    continue
                stray = _unmatched_numbers(text, corpus_by_date[cited])
                if stray:
                    self._drop(
                        context, section, text, cited,
                        f"contains {', '.join(stray[:3])}, which does not appear "
                        f"in the {cited} note; a changed number is the one thing "
                        "a pointer must never invent",
                    )
                    continue
                score = _grounding(text, corpus_by_date[cited])
                if score < self.grounding_threshold:
                    self._drop(
                        context, section, text, cited,
                        f"not supported by the {cited} note (grounding "
                        f"{score:.2f}); a claim filed under the wrong date sends "
                        "the reader to the wrong place in the chart",
                    )
                    continue
                conflict = _polarity_conflict(text, by_date[cited])
                if conflict is not None:
                    self._drop(
                        context, section, text, cited,
                        f"contradicts the cited note: {conflict}",
                    )
                    continue
                order = _order_grounding(text, words_by_date[cited])
                if order < self.order_threshold:
                    self._drop(
                        context, section, text, cited,
                        f"reuses the note's vocabulary in a different order "
                        f"(order agreement {order:.2f}); the same words can state "
                        "the opposite thing",
                    )
                    continue
                context.items.append(NarrativeItem(section, text, cited))
        if context.dropped:
            context.warnings.append(
                f"{len(context.dropped)} narrative item(s) failed a post-condition "
                "and were removed before the brief was assembled"
            )

    @staticmethod
    def _judgement_label(text: str) -> str | None:
        hits = [label for label, pattern in _JUDGEMENT_PATTERNS if pattern.search(text)]
        if not hits:
            return None
        if set(hits) <= _EXEMPTIBLE and _PAST_REPORT.search(text):
            # Reports what already happened. The deontic classes are never
            # exempted, so an item that both reports and prescribes stays out.
            return None
        return next(label for label in hits if label not in _EXEMPTIBLE or True)

    @staticmethod
    def _drop(
        context: NarrativeContext, section: str, text: str, cited: str, reason: str
    ) -> None:
        context.dropped.append(
            {"section": section, "item": text, "source_date": cited, "reason": reason}
        )
