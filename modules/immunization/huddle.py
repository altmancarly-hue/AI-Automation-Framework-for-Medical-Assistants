"""Tomorrow's gap sheet. One screen per patient, before clinic opens.

WHY this is the highest-value output in I-02 despite being the least clever:

The recall engine reaches children who are not coming in. The huddle sheet
catches the ones who already are. README I-02: "This converts gap closure into
an opportunistic-but-systematic process -- the child is already in the
building." No message is sent, no consent is needed, no TCPA question arises,
and the conversation happens face to face with the clinician the family already
trusts. It is also the cheapest: the forecast has already been computed for the
recall queue, and this is a different view of the same data.

WHY every line carries a provenance marker:

README R-02 asks for "visual marking of AI-generated content" and README I-03
makes it explicit. Everything on this sheet is COMPUTED -- rules-engine output
with a deterministic derivation -- except an optional narrative summary, which
is marked AI and is never the only place a fact appears. A clinician glancing at
this before a twelve-patient morning needs to know instantly which lines they
can act on without checking and which are a machine's summary.

The sheet deliberately does NOT say what to administer. It says what the
schedule shows as outstanding and what the record cannot confirm. The order is
a clinical decision made in the room by a licensed clinician.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Iterable, Mapping, Sequence

from modules.scheduling.models import Database, PRACTICE_TZ, iso, parse_iso, to_local

from .forecast import PatientForecast, Status
from .matcher import Reconciliation

__all__ = ["Provenance", "HuddleLine", "PatientCard", "HuddleSheet", "build_huddle"]


class Provenance:
    """Where a line on the sheet came from. Rendered, not implied."""

    COMPUTED = "computed"     # rules engine; deterministic; safe to act on
    RECORD = "record"         # copied from the chart or the schedule
    AI = "ai"                 # model-generated; a summary, never a source
    ALL = (COMPUTED, RECORD, AI)

    MARKS = {COMPUTED: "  ", RECORD: "  ", AI: "AI"}


@dataclass(frozen=True)
class HuddleLine:
    text: str
    provenance: str
    severity: str = "info"    # info | attention | urgent

    def render(self) -> str:
        mark = Provenance.MARKS.get(self.provenance, "  ")
        flag = {"urgent": "!!", "attention": " *", "info": "  "}[self.severity]
        return f"{mark}{flag} {self.text}"


@dataclass
class PatientCard:
    patient_id: str
    display_name: str
    dob: date
    age_text: str
    appointment_local: datetime
    provider: str
    visit_type: str
    overdue: list[HuddleLine] = field(default_factory=list)
    due: list[HuddleLine] = field(default_factory=list)
    review: list[HuddleLine] = field(default_factory=list)
    discrepancies: list[HuddleLine] = field(default_factory=list)
    narrative: HuddleLine | None = None

    @property
    def has_anything(self) -> bool:
        return bool(self.overdue or self.due or self.review or self.discrepancies)

    @property
    def headline_count(self) -> int:
        return len(self.overdue) + len(self.due)

    def render(self, width: int = 78) -> str:
        head = (
            f"{self.appointment_local:%H:%M}  {self.display_name}  "
            f"({self.age_text}, DOB {self.dob.isoformat()})"
        )
        lines = [head, f"       {self.provider} | {self.visit_type}", "-" * width]
        def section(title: str, items: Sequence[HuddleLine]) -> None:
            if not items:
                return
            lines.append(f"  {title}")
            lines.extend(f"    {item.render()}" for item in items)

        section("OVERDUE", self.overdue)
        section("DUE TODAY", self.due)
        section("CANNOT CONFIRM - needs a human look", self.review)
        section("RECORD DISCREPANCIES", self.discrepancies)
        if self.narrative:
            lines.append("  SUMMARY")
            lines.append(f"    {self.narrative.render()}")
        if not self.has_anything:
            lines.append("    up to date on the routine schedule")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        def dump(items: Sequence[HuddleLine]) -> list[dict[str, str]]:
            return [
                {"text": i.text, "provenance": i.provenance, "severity": i.severity}
                for i in items
            ]

        return {
            "patient_id": self.patient_id,
            "display_name": self.display_name,
            "dob": self.dob.isoformat(),
            "age": self.age_text,
            "appointment_local": self.appointment_local.isoformat(),
            "provider": self.provider,
            "visit_type": self.visit_type,
            "overdue": dump(self.overdue),
            "due": dump(self.due),
            "review": dump(self.review),
            "discrepancies": dump(self.discrepancies),
            "narrative": (
                {"text": self.narrative.text, "provenance": self.narrative.provenance}
                if self.narrative
                else None
            ),
        }


@dataclass
class HuddleSheet:
    for_date: date
    generated_at: datetime
    schedule_version: str
    cards: list[PatientCard] = field(default_factory=list)

    @property
    def patients_with_gaps(self) -> int:
        return sum(1 for c in self.cards if c.headline_count)

    def by_provider(self) -> dict[str, list[PatientCard]]:
        out: dict[str, list[PatientCard]] = {}
        for card in self.cards:
            out.setdefault(card.provider, []).append(card)
        for cards in out.values():
            cards.sort(key=lambda c: c.appointment_local)
        return out

    def render(self, provider: str | None = None, width: int = 78) -> str:
        groups = self.by_provider()
        if provider is not None:
            groups = {provider: groups.get(provider, [])}
        blocks: list[str] = []
        for name, cards in sorted(groups.items()):
            header = (
                f"IMMUNIZATION HUDDLE - {self.for_date:%a %d %b %Y} - {name}\n"
                f"{len(cards)} patients, {sum(1 for c in cards if c.headline_count)} "
                f"with an outstanding vaccine\n"
                f"schedule {self.schedule_version} | generated "
                f"{to_local(self.generated_at):%Y-%m-%d %H:%M %Z}\n"
                "Lines marked AI are machine-written summaries. Everything else is\n"
                "computed from the rules engine or copied from the record.\n"
                + "=" * width
            )
            blocks.append(header + "\n" + "\n\n".join(c.render(width) for c in cards))
        return "\n\n".join(blocks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "for_date": self.for_date.isoformat(),
            "generated_at": iso(self.generated_at),
            "schedule_version": self.schedule_version,
            "patients": len(self.cards),
            "patients_with_gaps": self.patients_with_gaps,
            "cards": [c.as_dict() for c in self.cards],
        }


def _age_text(dob: date, as_of: date) -> str:
    months = (as_of.year - dob.year) * 12 + (as_of.month - dob.month)
    if as_of.day < dob.day:
        months -= 1
    months = max(0, months)
    if months < 24:
        return f"{months} mo"
    return f"{months // 12} y {months % 12} mo" if months % 12 else f"{months // 12} y"


def build_huddle(
    db: Database,
    *,
    for_date: date,
    forecasts: Mapping[str, PatientForecast],
    reconciliations: Mapping[str, Reconciliation] | None = None,
    generated_at: datetime | None = None,
    narratives: Mapping[str, str] | None = None,
    schedule_version: str = "",
) -> HuddleSheet:
    """Assemble the sheet for one clinic day from data already computed.

    `forecasts` and `reconciliations` are passed in rather than computed here.
    WHY: the nightly batch has already produced them for the recall queue, and a
    huddle sheet that recomputed them could disagree with the queue that was
    built four minutes earlier. One computation, two views.

    `narratives` is optional model-written text keyed by patient id. Every entry
    is rendered with an AI marker and is additive -- no fact appears only in the
    narrative, so a clinician who ignores those lines entirely loses nothing.
    """
    reconciliations = reconciliations or {}
    narratives = narratives or {}
    generated_at = generated_at or datetime.now(PRACTICE_TZ)

    day_start = datetime.combine(for_date, datetime.min.time(), tzinfo=PRACTICE_TZ)
    day_end = day_start + timedelta(days=1)
    rows = db.all(
        """SELECT a.appointment_id, a.patient_id, a.start_utc, a.visit_type,
                  p.first_name, p.last_name, p.dob,
                  COALESCE(pr.display_name, a.provider_id) AS provider
           FROM appointment a
           JOIN patient p ON p.patient_id = a.patient_id
           LEFT JOIN provider pr ON pr.provider_id = a.provider_id
           WHERE a.status IN ('scheduled','confirmed')
             AND a.start_utc >= ? AND a.start_utc < ?
           ORDER BY a.start_utc""",
        (iso(day_start), iso(day_end)),
    )

    sheet = HuddleSheet(
        for_date=for_date,
        generated_at=generated_at,
        schedule_version=schedule_version,
        cards=[],
    )

    for row in rows:
        dob = date.fromisoformat(row["dob"])
        card = PatientCard(
            patient_id=row["patient_id"],
            display_name=f"{row['first_name']} {row['last_name']}",
            dob=dob,
            age_text=_age_text(dob, for_date),
            appointment_local=to_local(parse_iso(row["start_utc"])),
            provider=row["provider"],
            visit_type=row["visit_type"],
        )

        forecast = forecasts.get(row["patient_id"])
        if forecast is None:
            # "up to date on the routine schedule" is an affirmative clinical
            # assertion. Printing it because the nightly batch happened to skip
            # this patient would be the worst kind of quiet failure -- a
            # clinician reads it as a checked result.
            card.review.append(
                HuddleLine(
                    "no immunization forecast was produced for this patient last "
                    "night; their record has NOT been checked",
                    Provenance.RECORD,
                    "urgent",
                )
            )
        if forecast is not None:
            if schedule_version == "" and forecast.schedule_version:
                sheet.schedule_version = forecast.schedule_version
            for antigen in sorted(
                forecast.antigens.values(), key=lambda a: (-a.weight, a.antigen)
            ):
                if antigen.status == Status.OVERDUE:
                    detail = f"{antigen.label}: dose {antigen.next_dose} of {antigen.doses_required}"
                    if antigen.days_overdue:
                        detail += f", {antigen.days_overdue} days overdue"
                    if antigen.school_required:
                        detail += " [school]"
                    card.overdue.append(
                        HuddleLine(
                            detail,
                            Provenance.COMPUTED,
                            "urgent" if antigen.school_required or antigen.weight >= 4
                            else "attention",
                        )
                    )
                elif antigen.status == Status.DUE and antigen.huddle_eligible:
                    card.due.append(
                        HuddleLine(
                            f"{antigen.label}: dose {antigen.next_dose} of "
                            f"{antigen.doses_required}",
                            Provenance.COMPUTED,
                            "attention",
                        )
                    )
                elif antigen.status == Status.REQUIRES_REVIEW:
                    note = antigen.notes[0] if antigen.notes else "record cannot be verified"
                    card.review.append(
                        HuddleLine(f"{antigen.label}: {note}", Provenance.COMPUTED, "attention")
                    )
                if antigen.age_out_hard_date and antigen.is_open_gap:
                    remaining = (antigen.age_out_hard_date - for_date).days
                    if 0 < remaining <= 90:
                        card.overdue.append(
                            HuddleLine(
                                f"{antigen.label}: {remaining} days left before this "
                                "vaccine can no longer be given",
                                Provenance.COMPUTED,
                                "urgent",
                            )
                        )
            if forecast.unknown_codes:
                card.review.append(
                    HuddleLine(
                        "unrecognised vaccine code(s) in the record: "
                        + ", ".join(forecast.unknown_codes),
                        Provenance.RECORD,
                        "attention",
                    )
                )

        reconciliation = reconciliations.get(row["patient_id"])
        if reconciliation is not None:
            # A combination product against three component rows is three
            # ambiguous pairs but ONE thing for a human to look at. Printing it
            # three times buries the other discrepancies underneath it.
            seen: set[tuple[str, str]] = set()
            for pair in reconciliation.ambiguous:
                key = (", ".join(pair.antigens), pair.reason)
                if key in seen:
                    continue
                seen.add(key)
                card.discrepancies.append(
                    HuddleLine(
                        f"chart/registry unresolved ({key[0]}): {pair.reason}",
                        Provenance.RECORD,
                        "attention",
                    )
                )
            for duplicate in reconciliation.duplicates:
                card.discrepancies.append(
                    HuddleLine(
                        f"possible duplicate dose in the {duplicate.source} on "
                        f"{duplicate.first.given.isoformat()} and "
                        f"{duplicate.second.given.isoformat()}",
                        Provenance.RECORD,
                        "urgent",
                    )
                )
            for record in reconciliation.registry_only:
                card.discrepancies.append(
                    HuddleLine(
                        f"in I-CARE but not the chart: {record.normalised_cvx} on "
                        f"{record.given.isoformat()}",
                        Provenance.RECORD,
                    )
                )

        narrative = narratives.get(row["patient_id"])
        if narrative:
            card.narrative = HuddleLine(narrative.strip(), Provenance.AI)

        sheet.cards.append(card)

    if not sheet.schedule_version:
        sheet.schedule_version = schedule_version
    return sheet
