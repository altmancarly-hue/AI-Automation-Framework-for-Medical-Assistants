"""Urgency triage, and the rule that overrules it.

README I-06's top risk is CRITICAL and its control is explicit:

    Urgent document classified routine | CRITICAL | Bias the classifier toward
    over-flagging; EVERY DOCUMENT CONTAINING A NUMERIC LAB VALUE OUTSIDE
    REFERENCE RANGE IS FORCE-FLAGGED BY RULE REGARDLESS OF LLM OUTPUT; daily
    human audit of a random sample of 10 routine-classified documents.

The build plan says the same thing in one line: *rules beat models on safety
paths.* So this module runs the Appendix A.3 prompt and then runs a parser over
the same text, and the parser can only ever move the answer in one direction.

THE ESCALATION IS ONE-WAY. `_apply_overrides` can raise routine to urgent. It
can never lower anything. If the model says urgent and the parser finds nothing,
the document is urgent -- A.3 tells the model to flag "anything you are unsure
about", and second-guessing that with a regex is precisely the wrong direction.
The only thing the parser adds is a floor.

WHAT THE PARSER LOOKS FOR, in descending order of how much it is trusted:

  1. **A critical phrase.** "Critical value", "called and read back", "positive
     blood culture". These are the words a laboratory uses when it has already
     decided the result cannot wait, and they appear on documents whose numbers
     may be on a page OCR could not read.
  2. **A numeric result outside an age-banded reference range**, from
     `config/reference_ranges.yaml`. Age-banded because a hemoglobin of 11.0 is
     normal at nine months and low at fifteen years, and a single adult range
     would force-flag most healthy infants -- an alert that fires on everything
     is an alert nobody reads, which fails the same way as not alerting.
  3. **The document's own abnormal marker.** Labs print H, L, HIGH, LOW,
     CRITICAL, ABNORMAL and (*) next to the value. Reading the lab's own flag
     needs no reference table at all and survives an analyte this system has
     never heard of.
  4. **Unreadable pages on a clinical document.** A discharge summary whose
     middle page did not OCR is not a routine document; it is a document nobody
     has read. That is a human's problem, not a classifier's.

AND WHAT IT DOES WITH A NUMBER IT CANNOT BELIEVE. A potassium of 41.0 is not a
critical value, it is a decimal point lost in a bad scan. It is STILL escalated
-- refusing to flag a number you cannot read is the wrong direction -- but it is
labelled as probable OCR damage so a human is not sent chasing a phantom.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Mapping, Sequence

import yaml

from nsp_core.llm import LLMClient, SchemaViolation

__all__ = [
    "DEFAULT_RANGES_PATH",
    "URGENCY_LEVELS",
    "MIN_URGENCY_CONFIDENCE",
    "URGENCY_SYSTEM_PROMPT",
    "URGENCY_SCHEMA",
    "AbnormalValue",
    "ReferenceRanges",
    "UnreviewedRanges",
    "UrgencyResult",
    "UrgencyTriage",
    "rank",
]

DEFAULT_RANGES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "reference_ranges.yaml",
)

#: Ordered least to most urgent. Comparison is by index, never by string.
URGENCY_LEVELS: tuple[str, ...] = ("routine", "needs_physician_review", "urgent")

#: Below this the urgency model's own answer is not trusted to CLEAR a document.
#: `classify.py` honoured its confidence field and the safety-path call did not:
#: a radiology report whose impression was unreadable came back
#: `{"urgency": "routine", "confidence": 0.03}` and was auto-filed with a
#: 72-hour acknowledge task. Hard constraint 3 says low confidence routes to a
#: human, and this is the one call where that matters most.
MIN_URGENCY_CONFIDENCE = 0.6


def rank(level: str) -> int:
    return URGENCY_LEVELS.index(level)


class UnreviewedRanges(RuntimeError):
    """Raised when the reference-range table has no assigned clinical owner."""


#: README Appendix A.3, verbatim. Editing this is change-controlled (README 9.4).
URGENCY_SYSTEM_PROMPT = """You are triaging an inbound clinical document for a pediatric practice.

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
}"""

URGENCY_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "urgency": {"type": "string", "enum": list(URGENCY_LEVELS)},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string"},
        "abnormal_values_detected": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["urgency", "confidence", "reason", "abnormal_values_detected"],
    "additionalProperties": False,
}


# -- the reference table -----------------------------------------------------


@dataclass(frozen=True)
class _Band:
    age_months_min: float
    age_months_max: float
    low: float
    high: float


@dataclass(frozen=True)
class _Analyte:
    id: str
    name: str
    units: tuple[str, ...]
    aliases: tuple[str, ...]
    bands: tuple[_Band, ...]

    def band_for(self, age_months: float | None) -> _Band | None:
        """The band for a KNOWN age, or None when the age is unknown.

        An earlier version manufactured a band for `None` -- first the widest
        across childhood, then, after that was shown to clear a potassium of
        5.8, the intersection of every band. Both were wrong, and the second was
        wrong in a way that is easy to miss: the shipped bands are not nested.
        Hemoglobin runs 13.5-21.5 at 0-1 month and 9.5-14.0 at 1-6, so the
        intersection is 13.5-13.5 -- a zero-width band that force-flagged 140 of
        141 hemoglobin values, i.e. every unmatched CBC in the practice.

        There is no band for an unknown age, so this returns None and
        `candidate_bands` handles the question the caller actually has.
        """
        if not self.bands:
            return None
        if age_months is None:
            return None
        for band in self.bands:
            if band.age_months_min <= age_months < band.age_months_max:
                return band
        return self.bands[-1]

    def candidate_bands(
        self, ages: Sequence[float | None] | None
    ) -> tuple[_Band, ...]:
        """Every band this result could be scored against.

        One known age gives one band. Two possible collection dates give two.
        No age at all gives every band the table ships, because the child is
        somewhere in childhood and that is all anybody knows.

        `_scan_numbers` then applies one rule to whatever comes back: a value
        outside EVERY candidate band is abnormal whatever the age; a value
        outside SOME of them cannot be settled without knowing the age, and
        that is a question for a person rather than an answer to invent.
        """
        if not self.bands:
            return ()
        known = [a for a in (ages or ()) if a is not None]
        if not known:
            return tuple(self.bands)
        bands = []
        for age in known:
            band = self.band_for(age)
            if band is not None and band not in bands:
                bands.append(band)
        return tuple(bands)


def _inflect(word: str) -> str:
    """One config word as a regex that survives a singular/plural swap.

    "values" and "value", "requires" and "require" are the same word to the
    person reading the fax, and a critical-value rule that turns on which one
    the lab's template used is not a rule. The stem must stay at least four
    characters, so short words ("is", "as") are not shortened into something
    that matches a fragment.
    """
    stem = word[:-1] if word.endswith("s") and len(word) > 4 else word
    return re.escape(stem) + r"s?"


def _containing_line(text: str, start: int, end: int) -> str:
    """The whole line a match sits on, trimmed.

    A +/-30 character window straddled the line above and handed the MA
    "mg/dL 0.5-1.0\n\nResult phoned to ordering provider" as the reason for a
    one-hour page to the on-duty physician.
    """
    left = text.rfind("\n", 0, start) + 1
    right = text.find("\n", end)
    if right == -1:
        right = len(text)
    return text[left:right].strip()[:160]


@dataclass(frozen=True)
class AbnormalValue:
    """One numeric result the parser believes is outside range."""

    analyte: str
    value: float
    units: str
    low: float
    high: float
    direction: str  # "low" | "high"
    source: str  # "reference_range" | "document_flag" | "critical_phrase"
    excerpt: str = ""
    #: True when dividing the value by ten lands it inside the range -- i.e. it
    #: is exactly what a decimal point lost in a bad scan looks like. It is a
    #: precise, checkable statement ("check whether this is 4.1") rather than a
    #: guess at plausibility, and it changes NOTHING about the escalation.
    decimal_shift_suspected: bool = False

    def describe(self) -> str:
        if self.source == "critical_phrase":
            return f"document says: {self.excerpt!r}"
        if self.source == "age_unknown":
            return (
                f"{self.analyte} {self.value:g} {self.units} is {self.direction} "
                f"against {self.low:g}-{self.high:g} at SOME paediatric ages and "
                "normal at others; this document's age at collection is not "
                "established, so a person has to settle it"
            )
        if self.source == "document_flag":
            # The line itself, because this rule reads a MARKER and not a
            # number: it fires on analytes the table has never heard of, so it
            # has no value to report. An earlier version rendered a placeholder
            # zero here and produced "(flagged by sending lab) 0  flagged HIGH",
            # which reads as a result of zero.
            return (
                f"the sending lab flagged this line {self.direction.upper()}: "
                f"{self.excerpt!r}"
            )

        note = (
            f" -- check whether this is {self.value / 10:g}; a lost decimal point "
            "looks exactly like this, and it is flagged either way"
            if self.decimal_shift_suspected
            else ""
        )
        return (
            f"{self.analyte} {self.value:g} {self.units} is {self.direction} "
            f"against {self.low:g}-{self.high:g}{note}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "analyte": self.analyte,
            "value": self.value,
            "units": self.units,
            "reference": [self.low, self.high],
            "direction": self.direction,
            "source": self.source,
            "decimal_shift_suspected": self.decimal_shift_suspected,
            "excerpt": self.excerpt,
        }


#: A bare number. NO leading minus: `Cr-51 clearance study 45 mL/min` parsed as
#: creatinine = -51.0 and force-flagged every nuclear-medicine report.
_NUMBER = r"(\d{1,4}(?:[.,]\d{1,3})?)"
_NUM_TOKEN = re.compile(r"(?<![\d.\-])" + _NUMBER + r"(?![\d.])")

#: A printed reference range: "3.5-5.1", "3.5 - 5.1", "3.5 to 5.1". The numbers
#: inside one are NOT the result, and taking the first number after the analyte
#: name read the range's lower bound as the value -- so a potassium of 7.2 on a
#: line reading "Potassium  3.5-5.1 mmol/L  7.2  HH" was declared normal.
_RANGE_SPAN = re.compile(
    r"(?<![\d.])\d{1,4}(?:\.\d{1,3})?\s*(?:-|\u2013|to)\s*\d{1,4}(?:\.\d{1,3})?(?![\d.])"
)

#: Units a laboratory prints. Used for two things: deciding whether a line is a
#: RESULT LINE at all, and disambiguating two-letter analyte aliases.
_UNIT_TOKEN = re.compile(
    r"(?i)(?<![A-Za-z/])(?:mg/dl|g/dl|gm/dl|mcg/dl|ug/dl|\u00b5g/dl|ng/ml|pg/ml|"
    r"mmol/l|meq/l|mmol/mol|k/ul|10\*3/ul|x10e3/ul|thou/ul|10\^3/ul|m/ul|"
    r"u/l|iu/l|miu/l|uiu/ml|miu/ml|mm/hr|sec|ratio|fl|pg|%)(?![A-Za-z])"
)

#: The lab's own abnormal marker. Accepts the bracketed and punctuated forms
#: labs actually print -- "(H)", "[H]", "H*", "*H", "H!" -- all of which a
#: whitespace-only boundary missed, and parenthesised flags are the standard
#: layout.
_FLAG = re.compile(
    r"(?:(?<=\s)|^)[\(\[\{\*]?"
    r"(HH|LL|HIGH|LOW|CRIT(?:ICAL)?|ABN(?:ORMAL)?|H|L)"
    r"[\)\]\}\*!]*(?=\s|$)"
)

#: Words that mean the number beside them is not a result.
_NOT_A_RESULT = re.compile(
    r"(?i)\b(?:ref(?:erence)?|range|normal|expected|page|acct|account|mrn|"
    r"id|phone|fax|dob|room|bed|order|accession|specimen\s*id)\b"
)


def _range_spans(line: str) -> list[tuple[int, int]]:
    return [m.span() for m in _RANGE_SPAN.finditer(line)]


def _inside(position: int, spans: Sequence[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in spans)


class ReferenceRanges:
    """Age-banded ranges from config, plus the lab's own abnormal markers."""

    def __init__(self, data: Mapping[str, Any]) -> None:
        self.version = str(data.get("version", "unversioned"))
        self.review: dict[str, Any] = dict(data.get("review") or {})
        self.critical_phrases: tuple[str, ...] = tuple(
            str(p).lower() for p in data.get("critical_phrases", ())
        )
        self.analytes: list[_Analyte] = []
        for raw in data.get("analytes", []):
            self.analytes.append(
                _Analyte(
                    id=str(raw["id"]),
                    name=str(raw["name"]),
                    units=tuple(str(u).lower() for u in raw.get("units", ())),
                    aliases=tuple(str(a).lower() for a in raw.get("aliases", ())),
                    bands=tuple(
                        _Band(
                            float(b["age_months_min"]), float(b["age_months_max"]),
                            float(b["low"]), float(b["high"]),
                        )
                        for b in raw.get("bands", ())
                    ),
                )
            )
        if not self.analytes:
            raise ValueError("the reference-range table is empty")
        # Word-anchored, not substring. `"stat"` matched inside `"status
        # asthmaticus"` and turned an ordinary discharge summary into a critical
        # value alert -- and an urgent queue full of false alarms is how the real
        # one stops being read, which is the same failure as not alerting.
        #
        # Within that anchoring the pattern is deliberately forgiving of the
        # three things a fax does to a phrase, because none of them changes what
        # a human reads:
        #
        #   separators   `[\s\-]+` between every pair of words, and the phrase
        #                is split on hyphens too. An earlier version split on
        #                whitespace only, so `life-threatening` in the config
        #                matched a hyphen but not the space OCR often puts there.
        #   inflection   `_inflect` makes a trailing `s` optional on EVERY word,
        #                in both directions. Pinned to the last word and additive
        #                only, the config's "requires immediate medical
        #                attention" missed "Findings REQUIRE immediate medical
        #                attention" -- the ordinary plural subject of a radiology
        #                impression.
        #   case         `(?i)`.
        self._phrase_patterns = [
            (
                phrase,
                re.compile(
                    r"(?i)(?<!\w)"
                    + r"[\s\-]+".join(
                        _inflect(word) for word in re.split(r"[\s\-]+", phrase)
                    )
                    + r"(?!\w)"
                ),
            )
            for phrase in self.critical_phrases
        ]
        self._alias_patterns = [
            (
                analyte,
                alias,
                re.compile(r"(?i)(?<![A-Za-z])" + re.escape(alias) + r"(?![A-Za-z])"),
            )
            for analyte in self.analytes
            for alias in analyte.aliases
        ]

    @classmethod
    def load(cls, path: str | os.PathLike[str] = DEFAULT_RANGES_PATH) -> "ReferenceRanges":
        with open(path, "r", encoding="utf-8") as handle:
            return cls(yaml.safe_load(handle))

    @property
    def has_clinical_owner(self) -> bool:
        owner = str(self.review.get("owner", "")).strip()
        return bool(owner) and "UNASSIGNED" not in owner.upper()

    def require_reviewed(self) -> None:
        if not self.has_clinical_owner:
            raise UnreviewedRanges(
                f"config/reference_ranges.yaml ({self.version}) has no assigned "
                "clinical owner. These ranges decide whether a result reaches a "
                "physician today; they must be replaced with the reporting "
                "laboratories' own ranges and signed off before use."
            )

    # -- parsing -----------------------------------------------------------

    @staticmethod
    def collection_date(text: str) -> "date | None":
        """The date the specimen was taken, if the document says.

        WHY THIS MATTERS MORE THAN IT LOOKS: reference ranges must be applied at
        the age the specimen was COLLECTED, not the age the fax arrived. A
        newborn bilirubin of 9.4 is unremarkable at thirty-six hours of life and
        alarming at two months, and a document that took six weeks to reach the
        practice would otherwise be scored against the wrong band and force-flag
        a normal newborn result -- an alert that fires on healthy babies is an
        alert nobody reads.
        """
        # One label pattern, one date pattern, and `parse_document_date` does
        # the rest -- it already handles "22-Aug-2026" and "August 22, 2026",
        # which numeric-only patterns silently missed.
        _DATE_TOKEN = (
            r"([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4}"
            r"|[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}"
            r"|[0-9]{1,2}-[A-Za-z]{3}-[0-9]{2,4}"
            r"|[A-Za-z]{3,9}\s+[0-9]{1,2},\s*[0-9]{4})"
        )
        for pattern in (
            r"(?i)coll(?:ected|ection|\.)?\s*(?:date)?\s*[:\-]?\s*" + _DATE_TOKEN,
            r"(?i)drawn\s*(?:on)?\s*[:\-]?\s*" + _DATE_TOKEN,
            r"(?i)specimen\s*(?:date|collected)\s*[:\-]?\s*" + _DATE_TOKEN,
            r"(?i)date\s+of\s+service\s*[:\-]?\s*" + _DATE_TOKEN,
        ):
            match = re.search(pattern, text)
            if match is None:
                continue
            from .pipeline import parse_document_date

            parsed = parse_document_date(match.group(1))
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def collection_date_readings(text: str) -> "tuple[date | None, date | None]":
        """`(month_first, day_first_or_None)` for the collection date.

        A `03/04/2024` draw is either 4 March or 3 April, and the gap between
        those two readings can be eleven months. Eleven months is several
        paediatric reference bands. The caller resolves it; this only reports
        that there is something to resolve.
        """
        raw = ReferenceRanges.collection_date_text(text)
        from .pipeline import date_readings

        return date_readings(raw)

    @staticmethod
    def collection_date_text(text: str) -> "str | None":
        """The matched date SUBSTRING, before any parsing. See `collection_date`."""
        _DATE_TOKEN = (
            r"([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4}"
            r"|[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}"
            r"|[0-9]{1,2}-[A-Za-z]{3}-[0-9]{2,4}"
            r"|[A-Za-z]{3,9}\s+[0-9]{1,2},\s*[0-9]{4})"
        )
        for pattern in (
            r"(?i)coll(?:ected|ection|\.)?\s*(?:date)?\s*[:\-]?\s*" + _DATE_TOKEN,
            r"(?i)drawn\s*(?:on)?\s*[:\-]?\s*" + _DATE_TOKEN,
            r"(?i)specimen\s*(?:date|collected)\s*[:\-]?\s*" + _DATE_TOKEN,
            r"(?i)date\s+of\s+service\s*[:\-]?\s*" + _DATE_TOKEN,
        ):
            match = re.search(pattern, text)
            if match is not None:
                return match.group(1)
        return None

    def scan(
        self,
        text: str,
        *,
        age_months: float | None = None,
        candidate_ages: Sequence[float] | None = None,
    ) -> list[AbnormalValue]:
        """Everything in this text that says a human should look now.

        `candidate_ages` is for a collection date whose field order is
        ambiguous: `03/04/2024` is two dates and therefore two ages, and the
        difference between them can be eleven months, which is several
        paediatric bands. Pass both. A result outside every candidate band is
        abnormal whatever the truth is; a result outside only some of them is
        reported as unsettled rather than resolved by a coin toss.

        With no age at all the candidates are every band the table ships, which
        keeps the same rule honest: a hemoglobin of 3.2 is low at every age and
        is flagged; a hemoglobin of 12.6 is normal at some ages and abnormal at
        others, so it goes to a person instead of to a one-hour page.
        """
        found: list[AbnormalValue] = []
        for phrase, pattern in self._phrase_patterns:
            match = pattern.search(text)
            if match is not None:
                found.append(
                    AbnormalValue(
                        analyte="-", value=0.0, units="", low=0.0, high=0.0,
                        direction="critical", source="critical_phrase",
                        # The LINE the phrase sits on, not a character window.
                        # A window straddles the line above and hands the MA
                        # "mg/dL 0.5-1.0\n\nResult phoned to ordering provider"
                        # as the reason for a one-hour page.
                        excerpt=_containing_line(text, match.start(), match.end()),
                    )
                )
        ages: list[float] = []
        if age_months is not None:
            ages.append(float(age_months))
        for extra in candidate_ages or ():
            if extra is not None and float(extra) not in ages:
                ages.append(float(extra))
        found.extend(self._scan_numbers(text, ages or None))
        found.extend(self._scan_document_flags(text))
        return found

    def _scan_numbers(
        self, text: str, ages: Sequence[float | None] | None
    ) -> list[AbnormalValue]:
        """One line at a time, parsed as a result row rather than by proximity.

        A laboratory report is a table. The earlier version took the first
        number within forty characters of an analyte name, which failed three
        ways at once on ordinary layouts: it read the reference range as the
        result, it missed the result entirely on any table wider than about
        fifty columns, and two-letter aliases matched inside unit strings so
        `Platelets 310 K/uL` reported a potassium of 310.
        """
        found: list[AbnormalValue] = []
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            if not line.strip():
                continue
            spans = _range_spans(line)
            numbers = [
                (m.start(), float(m.group(1).replace(",", ".")))
                for m in _NUM_TOKEN.finditer(line)
                if not _inside(m.start(), spans)
            ]
            if not numbers:
                continue
            has_unit = _UNIT_TOKEN.search(line) is not None
            for analyte, alias, pattern in self._alias_patterns:
                match = pattern.search(line)
                if match is None:
                    continue
                # A two-letter alias -- k, na, cr, hb, pb, bg -- needs the
                # analyte's OWN unit on the line. Without that rule `K/uL`
                # became potassium and `Hb A1c` became a hemoglobin of 1.
                if len(alias) <= 2:
                    if not analyte.units:
                        continue
                    if not any(u in line.lower() for u in analyte.units):
                        continue
                after = [
                    (position, value) for position, value in numbers
                    if position > match.end()
                ]
                if not after:
                    continue
                position, value = after[0]
                # A number introduced by "Ref", "Range", "MRN", "Page" is not a
                # result.
                prefix = line[max(0, position - 24) : position]
                if _NOT_A_RESULT.search(prefix):
                    continue
                bands = analyte.candidate_bands(ages)
                if not bands:
                    break
                outside = [b for b in bands if not (b.low <= value <= b.high)]
                if not outside:
                    # Normal under every age this result could belong to.
                    break
                # The band that makes the smallest claim: the one this value is
                # closest to being inside. Reporting the most extreme band would
                # overstate how far out the result is.
                worst = min(
                    outside, key=lambda b: min(abs(value - b.low), abs(value - b.high))
                )
                direction = "low" if value < worst.low else "high"
                settled = len(outside) == len(bands)
                # A decimal point lost, or gained. Both are ordinary scan damage
                # and both send a reviewer somewhere useful.
                shifted = any(
                    worst.low <= candidate <= worst.high
                    for candidate in (value / 10.0, value * 10.0)
                )
                found.append(
                    AbnormalValue(
                        analyte=analyte.name, value=value,
                        units=analyte.units[0] if analyte.units else "",
                        low=worst.low, high=worst.high, direction=direction,
                        source="reference_range" if settled else "age_unknown",
                        decimal_shift_suspected=shifted,
                        excerpt=line.strip()[:120],
                    )
                )
                break
        return found

    def _scan_document_flags(self, text: str) -> list[AbnormalValue]:
        """The lab's own H / L / CRITICAL markers on a RESULT LINE.

        Works on analytes this table has never heard of, which is most of them:
        a reference table only ever covers what somebody remembered to add, and
        the sending laboratory has already decided this value was abnormal.

        Restricted to lines that look like results -- a known analyte or a
        printed unit. Without that restriction a bare `*` between a phone and a
        fax number on a cover sheet, and a middle initial `L` on a school form,
        both raised a one-hour page to the on-duty physician.
        """
        found: list[AbnormalValue] = []
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            if not _NUM_TOKEN.search(line):
                continue
            looks_like_a_result = _UNIT_TOKEN.search(line) is not None or any(
                pattern.search(line) and len(alias) > 2
                for _analyte, alias, pattern in self._alias_patterns
            )
            if not looks_like_a_result:
                continue
            flag = _FLAG.search(line)
            if flag is None:
                continue
            token = flag.group(1).upper()
            # `ABN`, `ABNORMAL`, `CRIT` and `CRITICAL` say that the value is out
            # of range and say NOTHING about which way. Rendering them as "high"
            # put "the sending lab flagged this line HIGH" in the physician's
            # one-hour page for a ferritin of 8, which is low.
            if token.startswith("L"):
                direction = "low"
            elif token.startswith("H"):
                direction = "high"
            else:
                direction = "abnormal"
            found.append(
                AbnormalValue(
                    analyte="(flagged by sending lab)", value=0.0, units="",
                    low=0.0, high=0.0, direction=direction,
                    source="document_flag", excerpt=line.strip()[:120],
                )
            )
        return found


# -- the triage --------------------------------------------------------------


@dataclass
class UrgencyResult:
    urgency: str
    confidence: float
    reason: str
    model_urgency: str = ""
    model_reason: str = ""
    abnormal: list[AbnormalValue] = field(default_factory=list)
    overrides: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    model_id: str = ""
    model_version: str = ""
    provider: str = ""
    inference_id: str | None = None

    @property
    def escalated(self) -> bool:
        return bool(self.model_urgency) and self.urgency != self.model_urgency

    @property
    def is_urgent(self) -> bool:
        return self.urgency == "urgent"

    def as_dict(self) -> dict[str, Any]:
        return {
            "urgency": self.urgency,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "model_urgency": self.model_urgency,
            "escalated_by_rule": self.escalated,
            "overrides": list(self.overrides),
            "abnormal_values": [a.as_dict() for a in self.abnormal],
            "warnings": list(self.warnings),
            "model": f"{self.provider}:{self.model_id}@{self.model_version}",
            "inference_id": self.inference_id,
        }


class UrgencyTriage:
    """A.3 prompt, then a rule that can only escalate."""

    INITIATIVE = "I-06"

    def __init__(
        self,
        client: LLMClient | None,
        *,
        ranges: ReferenceRanges,
        audit: Any = None,
        system_prompt: str = URGENCY_SYSTEM_PROMPT,
    ) -> None:
        self.client = client
        self.ranges = ranges
        self.audit = audit
        self.system_prompt = system_prompt

    @property
    def prompt_hash(self) -> str:
        return hashlib.sha256(self.system_prompt.encode("utf-8")).hexdigest()[:16]

    def triage(
        self,
        text: str,
        *,
        document_id: str,
        age_months: float | None = None,
        candidate_ages: Sequence[float] | None = None,
        document_type: str = "",
        unreadable_pages: Sequence[int] = (),
        user_id: str = "system:fax",
    ) -> UrgencyResult:
        """Never raises. A document that cannot be triaged is urgent by default.

        That default is the whole posture of this module. The failure mode being
        guarded is a critical value sitting in a routine queue, and a model that
        errored is not evidence that a document is routine.
        """
        result = UrgencyResult(
            urgency="needs_physician_review", confidence=0.0, reason=""
        )
        if self.client is not None:
            try:
                inference = self.client.structured(
                    system=self.system_prompt,
                    user=text[:12_000],
                    schema=URGENCY_SCHEMA,
                    context={
                        "initiative_id": self.INITIATIVE,
                        "user_id": user_id,
                        "document_id": document_id,
                    },
                    prompt_template_id="I-06/A.3",
                    temperature=0.0,
                )
            except SchemaViolation as exc:
                result.urgency = "urgent"
                result.reason = (
                    "urgency triage failed and the document was escalated rather "
                    f"than assumed routine ({exc})"
                )
                result.warnings.append("model output failed schema validation")
                inference = None
            else:
                data = inference.data
                result.model_urgency = str(data["urgency"])
                result.model_reason = str(data["reason"])
                result.urgency = result.model_urgency
                result.confidence = float(data["confidence"])
                result.reason = result.model_reason
                result.model_id = inference.model_id
                result.model_version = inference.model_version
                result.provider = inference.provider
                if self.audit is not None:
                    result.inference_id = self.audit.record_inference(
                        user_id=user_id,
                        initiative_id=self.INITIATIVE,
                        provider=inference.provider,
                        model_id=inference.model_id,
                        model_version=inference.model_version,
                        prompt_template_id=inference.prompt_template_id,
                        prompt_template_hash=inference.prompt_template_hash,
                        input_token_count=inference.input_token_count,
                        output_token_count=inference.output_token_count,
                        confidence_score=result.confidence,
                        constrained_decoding=inference.constrained,
                        repair_attempts=inference.repair_attempts,
                        extra={
                            "document_id": document_id,
                            "model_urgency": result.model_urgency,
                        },
                    )
        else:
            result.reason = "no urgency model configured; rules only"

        if age_months is None:
            result.warnings.append(
                "no patient age available, so every numeric result was scored "
                "against EVERY paediatric band. A value outside all of them is "
                "reported as abnormal; a value that is abnormal at some ages and "
                "normal at others is reported as UNSETTLED and needs an age "
                "before it means anything"
            )
        if candidate_ages:
            result.warnings.append(
                "the collection date could be read two ways, so results were "
                f"scored at {sorted(set(float(a) for a in candidate_ages))} months "
                "as well; a finding that holds at only one of them is UNSETTLED"
            )
        result.abnormal = self.ranges.scan(
            text, age_months=age_months, candidate_ages=candidate_ages
        )
        self._apply_overrides(
            result, document_type, unreadable_pages,
            enumerated_ages=bool(candidate_ages),
        )
        if (
            result.model_urgency
            and result.confidence < MIN_URGENCY_CONFIDENCE
            and rank(result.urgency) < rank("needs_physician_review")
        ):
            # Low confidence never CLEARS a document. It can still be raised by
            # a rule above, and it is never lowered.
            result.urgency = "needs_physician_review"
            result.overrides.append(
                f"the urgency model reported confidence {result.confidence:.2f}, "
                f"below {MIN_URGENCY_CONFIDENCE}; a low-confidence answer does "
                "not clear a document"
            )
            result.reason = result.overrides[-1]
        return result

    def _apply_overrides(
        self,
        result: UrgencyResult,
        document_type: str,
        unreadable_pages: Sequence[int],
        enumerated_ages: bool = False,
    ) -> None:
        """Raise the level. Never lower it. See the module docstring."""

        def escalate(to: str, why: str) -> None:
            if rank(to) > rank(result.urgency):
                result.urgency = to
                result.overrides.append(why)
                result.reason = f"{why} (model said {result.model_urgency or 'nothing'})"
            elif why not in result.overrides:
                # Recorded even when it changes nothing, so the daily audit can
                # see that the rule agreed rather than that it never fired.
                result.overrides.append(f"{why} [model already at this level]")

        phrases = [a for a in result.abnormal if a.source == "critical_phrase"]
        numeric = [a for a in result.abnormal if a.source == "reference_range"]
        unsettled = [a for a in result.abnormal if a.source == "age_unknown"]
        flagged = [a for a in result.abnormal if a.source == "document_flag"]

        if phrases:
            escalate(
                "urgent",
                f"the document contains critical-value language: {phrases[0].excerpt!r}",
            )
        if numeric:
            worst = numeric[0]
            escalate(
                "urgent",
                "a numeric result is outside the reference range by rule: "
                + worst.describe(),
            )
        if flagged:
            escalate(
                "urgent",
                "the sending laboratory flagged a value abnormal: "
                + flagged[0].excerpt[:80],
            )
        if unsettled:
            # HOW FAR this escalates depends on how big the ambiguity is, and
            # the difference is not a technicality.
            #
            # Two candidate ages means a known child and a collection date that
            # reads two ways -- `03/04/2024`. The finding is true under one of
            # exactly two readings, so it is a coin toss on a real patient, and
            # a coin toss on a bilirubin is an urgent page.
            #
            # No age at all means an unmatched document scored against every
            # band in childhood. Almost every ordinary CBC is abnormal at SOME
            # paediatric age, so paging on that would page on everything -- the
            # failure this module names twice in its own docstring -- and there
            # is no identified patient to page about. It still goes to a person,
            # with the values named. It is not cleared.
            if enumerated_ages:
                escalate(
                    "urgent",
                    f"{len(unsettled)} numeric result(s) are abnormal under one "
                    "of the two possible collection dates on this document: "
                    + unsettled[0].describe(),
                )
            else:
                escalate(
                    "needs_physician_review",
                    f"{len(unsettled)} numeric result(s) cannot be scored without "
                    "the patient's age at collection: " + unsettled[0].describe(),
                )
        if unreadable_pages and document_type in {
            "outside_lab", "outside_imaging", "hospital_discharge", "specialist_consult",
        }:
            escalate(
                "needs_physician_review",
                f"page(s) {list(unreadable_pages)} of a clinical document did not "
                "OCR; nobody has read them",
            )
