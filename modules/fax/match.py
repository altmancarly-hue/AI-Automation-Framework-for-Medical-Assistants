"""Patient matching. Deterministic, and it refuses far more often than it matches.

The build plan is exact about the rule and about the failure it is guarding:

    Patient matching: exact DOB + Jaro-Winkler name > 0.92. Multiple candidates
    -> human queue, ALWAYS. Twins share a DOB, so DOB alone is never sufficient.

README I-06 rates "misfiled to wrong patient (twins, siblings)" as HIGH and
requires "DOB + name match required; twin/sibling detection flag on the panel;
multi-candidate -> human queue always".

SO THE WHOLE MODULE IS BUILT AROUND ONE ASYMMETRY. A document sent to the human
queue costs someone forty seconds. A document filed to the wrong child is a
chart-integrity event that may never be found, and in a practice full of
siblings with the same surname and twins with the same date of birth, it is not
a rare shape. Every ambiguity therefore resolves toward the queue:

  * Two candidates over the threshold -> queue, even when one scores higher.
    "Best match wins" is exactly how a twin gets the other twin's discharge
    summary.
  * A DOB that does not match a panel patient exactly -> queue. No fuzzy dates,
    ever: 03/04 and 04/03 are two real children.
  * No DOB found on the document -> queue. A name-only match on "M. Reyes" in a
    practice with three of them is a coin toss with a chart.
  * A panel that lists the same person twice -> queue, and say so, because the
    duplicate chart is itself the finding.

There is no model here and there never will be; `make lint` asserts it.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "NAME_THRESHOLD",
    "NAME_MARGIN",
    "Candidate",
    "MatchOutcome",
    "MatchResult",
    "PanelPatient",
    "PatientMatcher",
    "jaro",
    "jaro_winkler",
    "normalise_name",
    "split_name",
]

#: The build plan's threshold. Applied to the better of (first, last) compared
#: in both name orders -- see `PatientMatcher._score`.
NAME_THRESHOLD = 0.92

#: A winner must beat the runner-up by at least this much. Without a margin the
#: threshold is a cliff a single OCR character can push a sibling across: with
#: twins "Mia" and "Mila" on one date of birth, a document reading "Mla" -- the
#: commonest OCR confusion there is -- scored Mila at 0.97 and Mia at 0.92, so
#: exactly one candidate cleared the bar and the document was AUTO-FILED to the
#: wrong twin's chart with no human in the loop.
NAME_MARGIN = 0.10


# -- string similarity -------------------------------------------------------


def jaro(a: str, b: str) -> float:
    """Jaro similarity. Written out rather than pulled in as a dependency.

    A one-file, fully-tested implementation is preferable here to a package
    because the number it produces decides whether a document reaches a child's
    chart, and "we upgraded a fuzzy-matching library" is not an acceptable
    explanation for a filing change.
    """
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    window = max(len(a), len(b)) // 2 - 1
    if window < 0:
        window = 0
    a_flags = [False] * len(a)
    b_flags = [False] * len(b)
    matches = 0
    for i, ch in enumerate(a):
        start = max(0, i - window)
        end = min(i + window + 1, len(b))
        for j in range(start, end):
            if not b_flags[j] and b[j] == ch:
                a_flags[i] = b_flags[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0
    transpositions = 0
    k = 0
    for i, matched in enumerate(a_flags):
        if not matched:
            continue
        while not b_flags[k]:
            k += 1
        if a[i] != b[k]:
            transpositions += 1
        k += 1
    transpositions //= 2
    return (
        matches / len(a) + matches / len(b) + (matches - transpositions) / matches
    ) / 3.0


def jaro_winkler(a: str, b: str, *, prefix_weight: float = 0.1) -> float:
    """Jaro with the standard shared-prefix boost, capped at four characters."""
    base = jaro(a, b)
    if base < 0.7:
        # The standard guard. Without it the prefix boost lifts genuinely
        # different names -- "Anderson"/"Andrade" -- over the threshold.
        return base
    prefix = 0
    for x, y in zip(a[:4], b[:4]):
        if x != y:
            break
        prefix += 1
    return base + prefix * prefix_weight * (1.0 - base)


_PUNCT = re.compile(r"[^a-z0-9 ]+")
_SPACE = re.compile(r"\s+")
#: Titles, which only ever appear FIRST.
_TITLES = frozenset("mr mrs ms miss dr prof".split())
#: Suffixes and credentials, which only ever appear LAST. Deliberately short:
#: "do", "pa", "ii", "iv" and "v" are all real surnames and given names, and a
#: generous list here silently deletes them. A missed "D.O." costs a fractional
#: similarity point; a deleted surname costs the patient every document they
#: ever receive.
_SUFFIXES = frozenset("jr sr iii md dds dmd phd esq rn np".split())
#: Labels an OCR'd fax header carries.
_LABELS = frozenset("patient name dob mrn".split())


def normalise_name(text: str) -> str:
    """Fold accents, drop punctuation and titles, collapse whitespace.

    Hyphens become spaces rather than disappearing: "Marie-Claire" and "Marie
    Claire" are the same child, and "MarieClaire" is a token that matches
    neither well.

    TITLES AND SUFFIXES ARE STRIPPED ONLY IN TITLE AND SUFFIX POSITION, and a
    name is never reduced to nothing. A flat stopword set containing "do" (for
    the D.O. credential) deleted the surname of every patient named Do -- one of
    the most common Vietnamese surnames -- so `normalise_name("Do")` returned
    the empty string and that child's documents were permanently unmatchable.
    The same held for "Pa", "Ms", "Miss", "II" and "IV", all of which are real
    names.
    """
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = folded.lower().replace("-", " ").replace("'", "")
    folded = _PUNCT.sub(" ", folded)
    words = [w for w in _SPACE.split(folded) if w]
    words = [w for w in words if w not in _LABELS] or words
    while len(words) > 1 and words[0] in _TITLES:
        words = words[1:]
    # Only when a name remains after it. "Minh Do" is two words and stripping
    # the second leaves one, which is how the surname disappeared.
    while len(words) > 2 and words[-1] in _SUFFIXES:
        words = words[:-1]
    return " ".join(words)


def split_name(text: str) -> tuple[str, str]:
    """(first, last) from the shapes a fax header actually carries.

    Handled: "Last, First", "First Last", HL7's "LAST^FIRST^MI", and
    "LAST FIRST MI" -- the standard fax-header format, in which a bare trailing
    single letter is a MIDDLE INITIAL and not a surname. Reading it as a surname
    made `PETROV JUNO A` score 0.545 and refused a correctly-addressed document.

    Both orders are scored anyway in `_score`, so this only has to be right
    often, not always.
    """
    if "^" in text:
        parts = [normalise_name(p) for p in text.split("^")]
        parts = [p for p in parts if p]
        if len(parts) >= 2:
            return parts[1], parts[0]
    if "," in text:
        last, _, rest = text.partition(",")
        return normalise_name(rest), normalise_name(last)
    words = normalise_name(text).split()
    if not words:
        return "", ""
    if len(words) == 1:
        return "", words[0]
    # A trailing single letter is a middle initial, not a surname.
    if len(words) > 2 and len(words[-1]) == 1:
        words = words[:-1]
    return " ".join(words[:-1]), words[-1]


# -- the panel ---------------------------------------------------------------


@dataclass(frozen=True)
class PanelPatient:
    patient_id: str
    first_name: str
    last_name: str
    dob: date
    #: Set by the practice on the panel. README I-06 asks for it explicitly:
    #: "twin/sibling detection flag on the panel".
    multiple_birth: bool = False
    aliases: tuple[str, ...] = ()

    @property
    def display(self) -> str:
        return f"{self.last_name}, {self.first_name} ({self.dob.isoformat()})"


class MatchOutcome:
    MATCHED = "matched"
    NO_DOB = "no_dob_on_document"
    NO_CANDIDATE = "no_panel_patient_with_that_dob"
    NAME_TOO_WEAK = "dob_matched_but_name_did_not"
    AMBIGUOUS = "multiple_candidates"
    DUPLICATE_PANEL = "panel_contains_duplicates"

    #: Everything except MATCHED goes to a person.
    HUMAN = frozenset(
        {NO_DOB, NO_CANDIDATE, NAME_TOO_WEAK, AMBIGUOUS, DUPLICATE_PANEL}
    )


@dataclass(frozen=True)
class Candidate:
    patient: PanelPatient
    score: float
    order_used: str  # "as-written" | "swapped"

    def as_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient.patient_id,
            "display": self.patient.display,
            "score": round(self.score, 4),
            "order_used": self.order_used,
        }


@dataclass
class MatchResult:
    outcome: str
    document_name: str
    document_dob: date | None
    matched: PanelPatient | None = None
    candidates: list[Candidate] = field(default_factory=list)
    reason: str = ""

    @property
    def needs_human(self) -> bool:
        return self.outcome in MatchOutcome.HUMAN

    @property
    def auto_fileable(self) -> bool:
        return self.outcome == MatchOutcome.MATCHED

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "document_name": self.document_name,
            "document_dob": self.document_dob.isoformat() if self.document_dob else None,
            "matched_patient_id": self.matched.patient_id if self.matched else None,
            "candidates": [c.as_dict() for c in self.candidates],
            "reason": self.reason,
            "needs_human": self.needs_human,
        }


class PatientMatcher:
    """Exact DOB, then name. Anything less than one clear answer goes to a human."""

    def __init__(
        self,
        panel: Sequence[PanelPatient],
        *,
        name_threshold: float = NAME_THRESHOLD,
    ) -> None:
        self.panel = list(panel)
        self.name_threshold = name_threshold
        self._by_dob: dict[date, list[PanelPatient]] = {}
        for patient in self.panel:
            self._by_dob.setdefault(patient.dob, []).append(patient)

    # -- scoring -----------------------------------------------------------

    def _score(self, document_name: str, patient: PanelPatient) -> tuple[float, str]:
        """Best score across name order and any recorded alias.

        Both orders are tried because a fax header may print "REYES SOFIA" and
        the panel holds "Sofia Reyes", and neither the fax nor the panel is
        reliably one convention. Taking the better of the two is safe here
        precisely BECAUSE ambiguity between two patients still goes to a human:
        a generous per-patient score cannot cause a wrong auto-file on its own,
        it can only widen the candidate set, which is the safe direction.
        """
        first, last = split_name(document_name)
        panel_first = normalise_name(patient.first_name)
        panel_last = normalise_name(patient.last_name)
        options = [
            ((first, last), "as-written"),
            ((last, first), "swapped"),
        ]
        best = (0.0, "as-written")
        for alias in ("",) + tuple(patient.aliases):
            candidate_first = normalise_name(alias) if alias else panel_first
            for (doc_first, doc_last), label in options:
                if not doc_last:
                    continue
                last_score = jaro_winkler(doc_last, panel_last)
                first_score = (
                    jaro_winkler(doc_first, candidate_first) if doc_first else 0.0
                )
                # The surname carries more weight, and a document with no first
                # name at all cannot reach the threshold on a surname alone --
                # "Reyes" matches three children in this practice.
                combined = (
                    0.6 * last_score + 0.4 * first_score if doc_first else last_score * 0.6
                )
                if combined > best[0]:
                    best = (combined, label)
        return best

    # -- the decision ------------------------------------------------------

    def match(self, *, name: str, dob: date | None) -> MatchResult:
        result = MatchResult(
            outcome=MatchOutcome.MATCHED, document_name=name, document_dob=dob
        )
        if dob is None:
            result.outcome = MatchOutcome.NO_DOB
            result.reason = (
                "the document carries no date of birth. A name-only match is a "
                "coin toss with a chart in a practice with siblings; this needs "
                "a person."
            )
            return result

        same_dob = self._by_dob.get(dob, [])
        if not same_dob:
            result.outcome = MatchOutcome.NO_CANDIDATE
            result.reason = (
                f"no active patient has the date of birth {dob.isoformat()}. The "
                "DOB is never fuzzy-matched -- 03/04 and 04/03 are two real "
                "children -- so this is either a new patient, a different "
                "practice's document, or an OCR error on the date."
            )
            return result

        scored = []
        for patient in same_dob:
            score, order = self._score(name, patient)
            scored.append(Candidate(patient, score, order))
        scored.sort(key=lambda c: -c.score)
        result.candidates = scored

        # A recorded multiple birth on this date of birth is binding. The flag
        # existed on the panel, was documented in README I-06's own controls,
        # and influenced NO decision -- setting it True changed nothing. Now it
        # forces the human queue at any score, because the whole point of
        # recording it is that these two children cannot be told apart from a
        # fax header.
        if len(same_dob) > 1 and any(p.multiple_birth for p in same_dob):
            result.outcome = MatchOutcome.AMBIGUOUS
            result.reason = (
                f"{len(same_dob)} patients share {dob.isoformat()} and at least "
                "one is flagged as a multiple birth. A twin's document is never "
                "auto-filed on a name score -- one OCR character is the whole "
                "difference between the two charts."
            )
            return result

        over = [c for c in scored if c.score >= self.name_threshold]
        if not over:
            result.outcome = MatchOutcome.NAME_TOO_WEAK
            best = scored[0]
            result.reason = (
                f"{len(same_dob)} patient(s) share this date of birth but the "
                f"best name similarity was {best.score:.3f}, below "
                f"{self.name_threshold}. DOB alone is never sufficient."
            )
            return result

        # Duplicates are checked BEFORE ambiguity. Both outcomes send the
        # document to a person, but "the panel holds this child twice" tells
        # them what to fix, and "two candidates" does not.
        by_identity: dict[str, list[Candidate]] = {}
        for candidate in over:
            key = normalise_name(
                f"{candidate.patient.first_name} {candidate.patient.last_name}"
            )
            by_identity.setdefault(key, []).append(candidate)
        duplicated = [group for group in by_identity.values() if len(group) > 1]
        if duplicated:
            result.outcome = MatchOutcome.DUPLICATE_PANEL
            result.reason = (
                f"the panel holds {len(duplicated[0])} records with this name and "
                "date of birth. Filing to either one is a guess, and the "
                "duplicate chart is itself the finding."
            )
            return result

        if len(over) > 1:
            twins = any(c.patient.multiple_birth for c in over)
            result.outcome = MatchOutcome.AMBIGUOUS
            result.reason = (
                f"{len(over)} patients match on date of birth and name"
                + (" and are flagged as a multiple birth" if twins else "")
                + ". This goes to a person ALWAYS -- taking the highest score is "
                "exactly how one twin receives the other twin's discharge "
                "summary."
            )
            return result

        winner = over[0]
        runner_up = scored[1] if len(scored) > 1 else None
        if runner_up is not None and winner.score - runner_up.score < NAME_MARGIN:
            result.outcome = MatchOutcome.AMBIGUOUS
            result.reason = (
                f"the best match scored {winner.score:.3f} and the next scored "
                f"{runner_up.score:.3f}, inside the {NAME_MARGIN} margin. A "
                "threshold with no margin is a cliff that one mis-read character "
                "pushes a sibling across."
            )
            return result
        result.matched = winner.patient
        result.reason = (
            f"exact DOB {dob.isoformat()} and name similarity {winner.score:.3f} "
            f"({winner.order_used}) against a single candidate"
        )
        return result

    # -- panel hygiene -----------------------------------------------------

    def duplicate_report(self) -> list[dict[str, Any]]:
        """Panel entries sharing a name and DOB. Run this before go-live.

        A duplicate chart makes every document for that child ambiguous forever.
        Finding them is a one-line query and nobody runs it.
        """
        seen: dict[tuple[str, date], list[PanelPatient]] = {}
        for patient in self.panel:
            key = (normalise_name(f"{patient.first_name} {patient.last_name}"), patient.dob)
            seen.setdefault(key, []).append(patient)
        return [
            {
                "name": key[0],
                "dob": key[1].isoformat(),
                "patient_ids": [p.patient_id for p in group],
            }
            for key, group in sorted(seen.items(), key=lambda kv: kv[0])
            if len(group) > 1
        ]

    def unflagged_multiples(self) -> list[dict[str, Any]]:
        """Same DOB, different names, not flagged as a multiple birth.

        Almost always twins whose panel flag was never set. README I-06 asks for
        the flag; this finds where it is missing, which is the only way the flag
        ever gets set on an established panel.
        """
        found: list[dict[str, Any]] = []
        for dob, patients in sorted(self._by_dob.items()):
            if len(patients) < 2:
                continue
            surnames = {normalise_name(p.last_name) for p in patients}
            if len(surnames) != 1:
                continue
            if all(p.multiple_birth for p in patients):
                continue
            found.append(
                {
                    "dob": dob.isoformat(),
                    "surname": next(iter(surnames)),
                    "patient_ids": [p.patient_id for p in patients],
                    "flagged": [p.patient_id for p in patients if p.multiple_birth],
                }
            )
        return found
