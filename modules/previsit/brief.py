"""One screen per patient, with the machine's two kinds of output kept apart.

README I-03's target state is a single screen and its risk table says why:
"Brief becomes noise and is ignored | Medium | Cap at one screen." So the cap
here is enforced arithmetic, not a layout aspiration -- `MAX_LINES` is real and
overflow is reported as a count rather than silently scrolling off.

THE SEPARATION THAT MATTERS. Two things on this screen have completely
different epistemic status:

  * COMPUTED -- screenings due, immunizations due, growth percentiles, open
    referrals. Arithmetic and table lookups. If one of these is wrong, the rules
    table or the chart data is wrong.
  * AI-GENERATED -- the narrative context. A language model's reading of three
    notes. If one of these is wrong, it may be wrong in a way that reads
    perfectly.

README I-03: "Every AI-generated element is visually distinguished from
deterministic output." That is implemented as a hard structural boundary -- the
AI section has its own heading, its own marker on every line, its own caveat,
and every line carries the encounter date it came from so the reader can go
check. Computed lines never carry the AI marker and AI lines never appear
outside the AI section.

AND THE OTHER WARNING IN THE RISK TABLE: "Staff treat the brief as authoritative
and stop opening the chart | High | Brief explicitly labeled 'not a substitute
for chart review'; deliberately omits data that must be verified in-chart (e.g.
exact medication doses)." Both halves are implemented: the footer says it, and
`_scrub_doses` removes anything that looks like a dose from the narrative before
it renders. A brief that prints "amoxicillin 400mg/5mL, 7mL BID" invites someone
to dose from it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence

from .growth import ChannelCrossing, GrowthPoint, ordinal
from .narrative import NARRATIVE_SECTIONS, NarrativeContext
from .periodicity import ScreeningStatus, Status

__all__ = [
    "MAX_LINES",
    "AI_MARKER",
    "COMPUTED_MARKER",
    "BriefSection",
    "PreVisitBrief",
    "OpenThread",
    "assemble",
    "render_text",
]

#: One screen. Not a suggestion -- see the module docstring.
MAX_LINES = 26

#: Every AI-generated line starts with this. Every computed line does not.
AI_MARKER = "~"
COMPUTED_MARKER = " "

_SECTION_TITLES: Mapping[str, str] = {
    "recent_relevant_history": "Recent",
    "open_threads": "Still open",
    "unresolved_parent_concerns": "Parent raised",
    "medication_changes": "Medications",
}

#: A number next to a unit, or a frequency abbreviation. Deliberately crude and
#: deliberately over-broad: the cost of scrubbing a harmless number is a reader
#: opening the chart, which is the behaviour the brief is supposed to preserve.
_DOSE = re.compile(
    # A number with a unit. The compound units come FIRST: with `mg` earlier in
    # the alternation it matched and the word boundary fired before the slash,
    # leaving "Ibuprofen [dose - verify in chart]/kg every 6 hours" -- a
    # half-scrubbed weight-based dose that reads as more authoritative than the
    # original.
    r"\b\d+(?:\.\d+)?\s*(?:mg/kg|mcg/kg|mg/ml|mg/5\s?ml|mg|mcg|µg|g|kg|ml|cc|"
    r"units?|iu|milligrams?|micrograms?|grams?|puffs?|sprays?|drops?|tablets?|"
    r"capsules?|teaspoons?|tsp|tablespoons?|tbsp|mls?)\b"
    # A frequency, spelled out or abbreviated.
    r"|\b(?:bid|tid|qid|qhs|qam|prn|po|pr|q\d+\s?h(?:rs?|ours?)?)\b"
    r"|\bevery\s+(?:\d+|one|two|three|four|six|eight|twelve)\s*"
    r"(?:-\s*(?:to\s*)?(?:\d+|six)\s*)?(?:hours?|hrs?|days?)\b"
    r"|\b(?:once|twice|three times|four times)\s+(?:a|per)\s+day\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class OpenThread:
    """A structured-query result: a referral or a result that never came back.

    Deterministic by construction -- README I-03 classifies "are there open
    referrals or unreturned results?" as a structured query, not a model task.
    """

    kind: str  # "referral" | "result" | "form" | "eligibility"
    description: str
    opened_on: date
    days_open: int
    detail: str = ""

    def line(self) -> str:
        return f"{self.description} {self.opened_on.isoformat()}, {self.detail or f'{self.days_open}d with no response'}"


@dataclass
class BriefSection:
    title: str
    lines: list[tuple[str, str]] = field(default_factory=list)  # (marker, text)
    ai_generated: bool = False


@dataclass
class PreVisitBrief:
    patient_label: str
    age_label: str
    visit_type: str
    appointment_local: str
    provider: str
    generated_utc: datetime
    sections: list[BriefSection] = field(default_factory=list)
    #: Lines that did not fit the one-screen cap, reported rather than dropped
    #: silently. A brief that quietly truncates is a brief that hides the item
    #: nobody knew was there.
    overflow: int = 0
    narrative_dropped: int = 0
    warnings: list[str] = field(default_factory=list)
    brief_id: str = ""

    @property
    def line_count(self) -> int:
        return sum(len(s.lines) + 1 for s in self.sections)

    @property
    def has_ai_content(self) -> bool:
        return any(s.ai_generated and s.lines for s in self.sections)

    def as_dict(self) -> dict[str, Any]:
        return {
            "brief_id": self.brief_id,
            "patient_label": self.patient_label,
            "age_label": self.age_label,
            "visit_type": self.visit_type,
            "appointment_local": self.appointment_local,
            "provider": self.provider,
            "generated_utc": self.generated_utc.isoformat(),
            "sections": [
                {
                    "title": s.title,
                    "ai_generated": s.ai_generated,
                    "lines": [{"marker": m, "text": t} for m, t in s.lines],
                }
                for s in self.sections
            ],
            "overflow": self.overflow,
            "narrative_dropped": self.narrative_dropped,
            "warnings": list(self.warnings),
        }


_SCRUB = "[dose - verify in chart]"
_RUN_OF_SCRUBS = re.compile(
    r"(?:" + re.escape(_SCRUB) + r")(?:[\s,/-]*" + re.escape(_SCRUB) + r")+"
)


def _scrub_doses(text: str) -> tuple[str, bool]:
    """Remove anything dose-shaped, then collapse the wreckage into one marker.

    "amoxicillin 400mg/5mL, 5 mL twice daily" contains three dose-shaped tokens
    and substituting each one separately produced a line of placeholder rubble
    that was harder to read than the dose it replaced -- and unreadable lines
    are how a brief becomes something people skip.
    """
    scrubbed = _DOSE.sub(_SCRUB, text)
    scrubbed = _RUN_OF_SCRUBS.sub(_SCRUB, scrubbed)
    return scrubbed, scrubbed != text


def assemble(
    *,
    patient_label: str,
    age_label: str,
    visit_type: str,
    appointment_local: str,
    provider: str,
    generated_utc: datetime,
    screenings: Sequence[ScreeningStatus] = (),
    immunizations_due: Sequence[str] = (),
    growth: Sequence[GrowthPoint] = (),
    crossings: Sequence[ChannelCrossing] = (),
    open_threads: Sequence[OpenThread] = (),
    forms_due: Sequence[Mapping[str, Any]] = (),
    narrative: NarrativeContext | None = None,
    brief_id: str = "",
    max_lines: int = MAX_LINES,
) -> PreVisitBrief:
    """Build the screen. Computed content first, AI content last and marked."""
    brief = PreVisitBrief(
        patient_label=patient_label,
        age_label=age_label,
        visit_type=visit_type,
        appointment_local=appointment_local,
        provider=provider,
        generated_utc=generated_utc,
        brief_id=brief_id,
    )

    due = BriefSection("DUE TODAY")
    for name in immunizations_due:
        due.lines.append(("!", name))
    for status in screenings:
        if status.status == Status.OVERDUE:
            due.lines.append(("!", f"{status.definition.name} - {status.because}"))
    for status in screenings:
        if status.status == Status.DUE:
            due.lines.append(("o", status.definition.name))
    for status in screenings:
        if status.status == Status.RISK_UNKNOWN:
            due.lines.append(("?", f"{status.definition.name} - risk not assessed"))
    # A screening the family DECLINED belongs on the page. `periodicity.py`
    # records it as a distinct state precisely so it can be offered again, and
    # rendering only DUE/OVERDUE made it indistinguishable from NOT_DUE -- so it
    # was never re-offered, which is the opposite of what recording a decline is
    # for.
    for status in screenings:
        if status.status == Status.DECLINED:
            due.lines.append(
                ("x", f"{status.definition.name} - previously declined, offer again")
            )
    if not due.lines:
        due.lines.append((COMPUTED_MARKER, "nothing due by the periodicity schedule"))
    brief.sections.append(due)

    # MISSED screenings are not actionable today and must not compete with what
    # is. They are still a fact about this chart that nobody would otherwise
    # see, so they get one summary line rather than one line each.
    missed = [s for s in screenings if s.status == Status.MISSED]
    if missed:
        gaps = BriefSection("NEVER DONE (catch-up window closed)")
        names = ", ".join(sorted(s.definition.name for s in missed))
        gaps.lines.append((COMPUTED_MARKER, names))
        brief.sections.append(gaps)

    unknown = [s for s in screenings if s.status == Status.UNKNOWN]
    if unknown:
        pre = BriefSection("NO RECORD IN THIS SYSTEM (check the paper chart once)")
        pre.lines.append(
            (COMPUTED_MARKER, ", ".join(sorted(s.definition.name for s in unknown)))
        )
        brief.sections.append(pre)

    flags = BriefSection("FLAGS")
    for crossing in crossings:
        if crossing.significant:
            flags.lines.append(("!", crossing.describe()))
    for thread in open_threads:
        flags.lines.append(("!", thread.line()))
    if flags.lines:
        brief.sections.append(flags)

    if growth:
        measurements = BriefSection("GROWTH")
        for point in growth:
            measurements.lines.append(
                (
                    COMPUTED_MARKER,
                    f"{point.indicator.replace('_', ' ')}: {point.value:g} "
                    f"({ordinal(point.percentile)} percentile)",
                )
            )
        brief.sections.append(measurements)

    admin = BriefSection("ADMIN")
    for form in forms_due:
        admin.lines.append(("o", f"{form['name']} - {form.get('note', '')}".strip(" -")))
    if admin.lines:
        # BEFORE the AI section, not after. Section order is priority order for
        # `_apply_cap`, and putting the narrative first meant a full screen
        # dropped two Illinois statutory forms to keep six unvalidated model
        # lines. Computed content outranks generated content when space runs
        # out; that is the same principle as marking them differently.
        brief.sections.append(admin)

    if narrative is not None and narrative.items:
        context = BriefSection("CONTEXT (AI-generated - review before relying on)",
                               ai_generated=True)
        for section_name in NARRATIVE_SECTIONS:
            for item in narrative.section(section_name):
                text, scrubbed = _scrub_doses(item.text)
                label = _SECTION_TITLES.get(section_name, section_name)
                context.lines.append((AI_MARKER, f"{label}: {text} [{item.source_date}]"))
                if scrubbed:
                    brief.warnings.append(
                        "a dose was removed from the narrative; doses are verified "
                        "in the chart, never read off this brief"
                    )
        brief.sections.append(context)

    if narrative is not None:
        brief.narrative_dropped = len(narrative.dropped)
        brief.warnings.extend(narrative.warnings)

    _apply_cap(brief, max_lines)
    return brief


def _apply_cap(brief: PreVisitBrief, max_lines: int) -> None:
    """Trim from the BOTTOM of the least urgent section, and say how much.

    The order the sections were appended in is the priority order, so trimming
    from the end takes admin detail before it takes an overdue immunization.
    """
    budget = max_lines
    kept: list[BriefSection] = []
    for section in brief.sections:
        budget -= 1  # the heading
        if budget <= 0:
            brief.overflow += len(section.lines)
            continue
        if len(section.lines) <= budget:
            budget -= len(section.lines)
            kept.append(section)
            continue
        dropped = len(section.lines) - budget
        brief.overflow += dropped
        # Keep one line for the count, so a truncated section says how many of
        # ITS items are missing rather than leaving the reader to notice.
        section.lines = section.lines[: max(budget - 1, 0)] + [
            ("+", f"{dropped + 1} more not shown - open the chart")
        ]
        budget = 0
        kept.append(section)
    brief.sections = kept
    if brief.overflow:
        brief.warnings.append(
            f"{brief.overflow} line(s) did not fit the one-screen cap and are in "
            "the chart, not on this brief"
        )


_FOOTER = (
    "This brief is a pointer, not a source of truth. It is not a substitute for "
    "chart review, and it deliberately omits medication doses."
)


def render_text(brief: PreVisitBrief, *, width: int = 78) -> str:
    """The screen, as text. The PDF and web views render from the same object."""
    rule = "-" * width
    out = [
        rule,
        f"{brief.patient_label} | {brief.age_label} | {brief.visit_type} | "
        f"{brief.appointment_local} | {brief.provider}",
        rule,
    ]
    for section in brief.sections:
        out.append("")
        out.append(section.title)
        for marker, text in section.lines:
            out.append(f"  {marker} {text}")
    out.append("")
    out.append(rule)
    if brief.narrative_dropped:
        out.append(
            f"{brief.narrative_dropped} AI item(s) were removed for citing no "
            "source encounter, citing one that was not supplied, or reading as "
            "a clinical recommendation."
        )
    for warning in dict.fromkeys(brief.warnings):
        out.append(f"! {warning}")
    out.append(_FOOTER)
    return "\n".join(out)
