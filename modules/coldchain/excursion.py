"""What happens after the alarm: quarantine, lots, and the monthly record.

README I-08, target state step 6:

    Excursion response workflow auto-launches: quarantine label generated,
    manufacturer contact list surfaced, affected lot numbers listed,
    documentation template pre-filled.

and step 7:

    Monthly compliance report auto-generated and archived.

Those two are one module because they are the same idea at two timescales. The
value of this initiative is not the alert — a $30 thermometer with a shrieking
buzzer produces an alert. The value is that afterwards the practice can say
exactly which lots were affected, for how long, at what temperature, and what it
did about it. README I-08 puts it plainly: every other initiative produces a
benefit you argue for with modelled assumptions; this one produces **a record**.

THE ONE JUDGEMENT THIS MODULE REFUSES TO MAKE. Whether vaccine exposed to an
excursion is still usable is a call for the manufacturer, and only for the
manufacturer. Each product has its own stability data and the answer differs by
product, by temperature, by duration, and sometimes by how many prior excursions
that vial has seen. Nothing here says "probably fine" or "discard".
`ExcursionRecord` states the exposure, lists the affected lots, surfaces the
contact numbers, and marks every lot QUARANTINED until a person records what the
manufacturer said.

A system that guessed here would be worse than useless in both directions: a
wrong "fine" gives children ineffective vaccine, and a wrong "discard" throws
away $20,000 and starts a recall of patients who did not need one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Iterable, Mapping, Sequence

from .monitor import Alert, AlertKind, ColdChainConfig, Reading, StorageUnit

__all__ = [
    "LotDisposition",
    "VaccineLot",
    "ExcursionRecord",
    "open_excursion",
    "DailyRecord",
    "ComplianceReport",
    "build_compliance_report",
]


#: The only prose this module puts on a quarantine label. It states what must
#: happen and who is entitled to answer the potency question; it offers no view
#: of its own about whether the doses are still good. A wrong "fine" gives
#: children ineffective vaccine; a wrong "discard" throws away $20,000 and
#: recalls patients who did not need it. Neither call is this module's to make.
DISPOSITION_ADVISORY: tuple[str, ...] = (
    "These doses MUST NOT be administered until the manufacturer has",
    "been contacted and a disposition recorded. Stability after an",
    "excursion is product-specific and only the manufacturer can say.",
)


class LotDisposition:
    """What has been decided about a lot exposed to an excursion."""

    QUARANTINED = "quarantined"
    #: The manufacturer said it is usable. A PERSON records this, with who said
    #: it and when. There is no path that sets it any other way.
    RELEASED = "released"
    DISCARDED = "discarded"
    ALL = (QUARANTINED, RELEASED, DISCARDED)


@dataclass
class VaccineLot:
    """One lot in one unit. The thing an excursion actually threatens."""

    lot_number: str
    product: str
    manufacturer: str
    unit_id: str
    doses_on_hand: int
    expires: date
    unit_cost_usd: float = 0.0
    #: Filled in by a person, after the manufacturer answers.
    disposition: str = LotDisposition.QUARANTINED
    disposition_by: str = ""
    disposition_note: str = ""

    @property
    def value_usd(self) -> float:
        return round(self.doses_on_hand * self.unit_cost_usd, 2)

    def as_dict(self) -> dict[str, Any]:
        return {
            "lot_number": self.lot_number, "product": self.product,
            "manufacturer": self.manufacturer, "unit_id": self.unit_id,
            "doses_on_hand": self.doses_on_hand,
            "expires": self.expires.isoformat(),
            "value_usd": self.value_usd,
            "disposition": self.disposition,
            "disposition_by": self.disposition_by,
            "disposition_note": self.disposition_note,
        }


@dataclass
class ExcursionRecord:
    """One excursion, from detection to a documented disposition per lot."""

    excursion_id: str
    unit: StorageUnit
    started_utc: datetime
    detected_utc: datetime
    min_temperature_c: float
    max_temperature_c: float
    limit_low_c: float
    limit_high_c: float
    duration_minutes: float
    affected_lots: list[VaccineLot] = field(default_factory=list)
    manufacturer_contacts: dict[str, str] = field(default_factory=dict)
    #: Free text a person writes. Not generated.
    corrective_action: str = ""
    closed_utc: datetime | None = None
    closed_by: str = ""

    @property
    def value_at_risk_usd(self) -> float:
        return round(sum(lot.value_usd for lot in self.affected_lots), 2)

    @property
    def doses_at_risk(self) -> int:
        return sum(lot.doses_on_hand for lot in self.affected_lots)

    @property
    def open_lots(self) -> list[VaccineLot]:
        return [
            lot for lot in self.affected_lots
            if lot.disposition == LotDisposition.QUARANTINED
        ]

    @property
    def closeable(self) -> bool:
        """Every lot has a recorded decision, and a person wrote what was done.

        Both halves are required. An excursion closed with lots still
        quarantined is an excursion nobody finished; one closed with no
        corrective action is a temperature reading, not a record.
        """
        return not self.open_lots and bool(self.corrective_action.strip())

    def record_disposition(
        self,
        lot_number: str,
        *,
        disposition: str,
        decided_by: str,
        note: str,
    ) -> VaccineLot:
        """A PERSON records what the manufacturer said. See the module docstring.

        `note` is required and is meant to hold who at the manufacturer said
        what, on what date. "Called Merck 8/24, ref #114982, stable at 12C for
        up to 72h, release" is a record. "OK" is not, and this refuses it only
        insofar as it refuses empty -- the rest is a training problem, not a
        software one.
        """
        if disposition not in LotDisposition.ALL:
            raise ValueError(
                f"{disposition!r} is not a disposition ({LotDisposition.ALL})"
            )
        if not decided_by.strip():
            raise ValueError(
                "a disposition names the person who obtained it. Whether an "
                "exposed vaccine is usable is the manufacturer's call, and the "
                "record has to say who asked and what they were told."
            )
        if not note.strip():
            raise ValueError(
                "a disposition records what the manufacturer said. Without it "
                "the practice has a decision it cannot defend."
            )
        for lot in self.affected_lots:
            if lot.lot_number == lot_number:
                lot.disposition = disposition
                lot.disposition_by = decided_by
                lot.disposition_note = note
                return lot
        raise KeyError(f"lot {lot_number!r} is not on this excursion")

    def close(self, *, closed_by: str, now: datetime) -> None:
        if not self.closeable:
            outstanding = [lot.lot_number for lot in self.open_lots]
            raise ValueError(
                "this excursion is not finished: "
                + (f"lots {outstanding} have no recorded disposition. " if outstanding else "")
                + ("No corrective action has been written. " if not self.corrective_action.strip() else "")
                + "Closing it now would leave a temperature reading where the "
                "record should be."
            )
        self.closed_utc = now
        self.closed_by = closed_by

    def quarantine_label(self) -> str:
        """The label that goes on the door. Deliberately blunt.

        Every sentence of prose on this label comes from
        `DISPOSITION_ADVISORY`. Nothing is composed here, because a label is
        where a reassuring sentence would be most tempting and most damaging:
        "most refrigerated vaccines remain potent after a brief excursion" is
        true in general, useless about this lot, and enough to get the doses
        used. `test_the_quarantine_advisory_is_fixed_policy_text` pins the
        constant verbatim, so changing what this label claims is a code review
        rather than a wording tweak.
        """
        lines = [
            "*** DO NOT USE - QUARANTINED VACCINE ***",
            f"Unit: {self.unit.label} ({self.unit.unit_id})",
            f"Excursion: {self.excursion_id}",
            f"Exposure: {self.min_temperature_c}C to {self.max_temperature_c}C "
            f"against a {self.limit_low_c}-{self.limit_high_c}C range",
            f"Duration: {self.duration_minutes:.0f} minutes from "
            f"{self.started_utc.isoformat()}",
            f"Doses affected: {self.doses_at_risk} "
            f"(approx ${self.value_at_risk_usd:,.0f})",
            "",
            *DISPOSITION_ADVISORY,
            "",
            "Manufacturer contacts:",
        ]
        for name, phone in sorted(self.manufacturer_contacts.items()):
            lines.append(f"  {name}: {phone}")
        lines.append("")
        lines.append("Affected lots:")
        for lot in sorted(self.affected_lots, key=lambda l: l.lot_number):
            lines.append(
                f"  {lot.lot_number:<14} {lot.product:<28} "
                f"{lot.doses_on_hand:>4} dose(s)  [{lot.disposition}]"
            )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "excursion_id": self.excursion_id,
            "unit": self.unit.as_dict(),
            "started_utc": self.started_utc.isoformat(),
            "detected_utc": self.detected_utc.isoformat(),
            "min_temperature_c": self.min_temperature_c,
            "max_temperature_c": self.max_temperature_c,
            "limits": [self.limit_low_c, self.limit_high_c],
            "duration_minutes": round(self.duration_minutes, 1),
            "doses_at_risk": self.doses_at_risk,
            "value_at_risk_usd": self.value_at_risk_usd,
            "affected_lots": [lot.as_dict() for lot in self.affected_lots],
            "open_lots": [lot.lot_number for lot in self.open_lots],
            "corrective_action": self.corrective_action,
            "closed_utc": self.closed_utc.isoformat() if self.closed_utc else None,
            "closed_by": self.closed_by,
        }


def open_excursion(
    alert: Alert,
    unit: StorageUnit,
    readings: Sequence[Reading],
    inventory: Sequence[VaccineLot],
    *,
    excursion_id: str,
    manufacturer_contacts: Mapping[str, str] | None = None,
) -> ExcursionRecord:
    """Build the record the moment an excursion is detected.

    Every lot in the unit is affected, not only the ones somebody thinks are
    sensitive. The exposure happened to the appliance; which products tolerate
    it is the manufacturer's question, and narrowing the list here would be
    answering it.
    """
    if alert.kind not in AlertKind.INVENTORY_AT_RISK:
        raise ValueError(
            f"{alert.kind!r} does not put inventory at risk; an excursion record "
            "is for an excursion"
        )
    window = [
        r for r in readings
        if r.temperature_c is not None
        and r.taken_utc >= alert.detected_utc - timedelta(
            minutes=alert.duration_minutes or 0.0
        )
    ]
    temps = [float(r.temperature_c) for r in window] or [
        float(alert.temperature_c or 0.0)
    ]
    lots = [lot for lot in inventory if lot.unit_id == unit.unit_id]
    contacts = dict(manufacturer_contacts or {})
    for lot in lots:
        contacts.setdefault(lot.manufacturer, "(no contact number on file)")
    return ExcursionRecord(
        excursion_id=excursion_id,
        unit=unit,
        started_utc=alert.detected_utc - timedelta(
            minutes=alert.duration_minutes or 0.0
        ),
        detected_utc=alert.detected_utc,
        min_temperature_c=min(temps),
        max_temperature_c=max(temps),
        limit_low_c=float(alert.limit_low_c or 0.0),
        limit_high_c=float(alert.limit_high_c or 0.0),
        duration_minutes=float(alert.duration_minutes or 0.0),
        affected_lots=lots,
        manufacturer_contacts=contacts,
    )


# -- the monthly record ------------------------------------------------------


@dataclass(frozen=True)
class DailyRecord:
    """One day, one unit: the min/max CDC asks for, computed not transcribed."""

    day: date
    unit_id: str
    min_c: float | None
    max_c: float | None
    reading_count: int
    expected_count: int
    #: Total minutes in the day with no reading, computed from the actual
    #: timestamps rather than from `(expected - count) * interval`.
    gap_minutes: float
    #: The LONGEST single stretch with no reading. This, not the count, is what
    #: decides whether the day was monitored.
    longest_gap_minutes: float = 0.0
    #: The stretch a gap may not exceed. Carried on the record so the report can
    #: say what it measured against.
    max_gap_minutes: float = 30.0

    @property
    def complete(self) -> bool:
        """Was this day actually monitored?

        A DISTRIBUTION question, not a count. The first version of this asked
        whether 90% of expected readings arrived, which at a five-minute
        interval tolerates 28 missing samples -- and those 28 may be one
        contiguous block. A logger unplugged from 20:40 to 23:00 during a
        service call produced 260 of 288 readings, reported `complete`, and the
        report declared 100% coverage with no findings, for a day with two hours
        and twenty minutes of no data. The CDC min/max record for that day was
        presented as authoritative and was silently false.

        So: no single gap longer than the sensor-offline threshold. A logger
        that drops the odd poll over a cellular link still passes; one that
        stopped for two hours does not, whatever its total count.
        """
        if self.expected_count <= 0 or self.reading_count == 0:
            return False
        return self.longest_gap_minutes <= self.max_gap_minutes

    def as_dict(self) -> dict[str, Any]:
        return {
            "day": self.day.isoformat(), "unit_id": self.unit_id,
            "min_c": self.min_c, "max_c": self.max_c,
            "readings": self.reading_count, "expected": self.expected_count,
            "gap_minutes": round(self.gap_minutes, 1),
            "longest_gap_minutes": round(self.longest_gap_minutes, 1),
            "complete": self.complete,
        }


@dataclass
class ComplianceReport:
    """The artefact this whole initiative exists to produce."""

    month: str
    generated_utc: datetime
    config_version: str
    units: list[StorageUnit]
    days: list[DailyRecord]
    excursions: list[ExcursionRecord]
    alerts_by_kind: dict[str, int] = field(default_factory=dict)

    @property
    def days_monitored(self) -> int:
        return sum(1 for d in self.days if d.complete)

    @property
    def coverage(self) -> float:
        return self.days_monitored / len(self.days) if self.days else 0.0

    @property
    def incomplete_days(self) -> list[DailyRecord]:
        return [d for d in self.days if not d.complete]

    @property
    def unclosed_excursions(self) -> list[ExcursionRecord]:
        return [e for e in self.excursions if e.closed_utc is None]

    def findings(self) -> list[str]:
        """What an auditor would ask about. Stated by the practice first.

        A report that says "100% compliant" every month is a report nobody
        checks. This one names its own gaps.
        """
        out: list[str] = []
        for day in self.incomplete_days:
            out.append(
                f"{day.day.isoformat()} {day.unit_id}: {day.reading_count} of "
                f"{day.expected_count} expected readings, and a single "
                f"unmonitored stretch of {day.longest_gap_minutes:.0f} minutes "
                f"(limit {day.max_gap_minutes:.0f}); "
                f"{day.gap_minutes:.0f} minutes unmonitored in total"
            )
        for excursion in self.unclosed_excursions:
            out.append(
                f"{excursion.excursion_id}: excursion on "
                f"{excursion.detected_utc.date().isoformat()} is still open"
                + (
                    f"; lots {[l.lot_number for l in excursion.open_lots]} have no "
                    "recorded disposition"
                    if excursion.open_lots else ""
                )
            )
        for unit in self.units:
            if unit.calibration_due is not None and unit.calibration_due < (
                self.generated_utc.date()
            ):
                out.append(
                    f"{unit.unit_id}: NIST calibration lapsed "
                    f"{unit.calibration_due.isoformat()}; readings since then are "
                    "not audit evidence"
                )
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "month": self.month,
            "generated_utc": self.generated_utc.isoformat(),
            "config_version": self.config_version,
            "units": [u.as_dict() for u in self.units],
            "days_in_report": len(self.days),
            "days_monitored": self.days_monitored,
            "coverage": round(self.coverage, 4),
            "alerts_by_kind": dict(sorted(self.alerts_by_kind.items())),
            "excursions": [e.as_dict() for e in self.excursions],
            "findings": self.findings(),
        }

    def render(self) -> str:
        lines = [
            f"VACCINE STORAGE COMPLIANCE REPORT - {self.month}",
            f"generated {self.generated_utc.isoformat()} "
            f"from thresholds {self.config_version}",
            "",
            f"Units: {', '.join(u.label for u in self.units)}",
            f"Day-records: {self.days_monitored} of {len(self.days)} complete "
            f"({self.coverage:.1%})",
            "",
            "DAILY MIN/MAX (computed from continuous logging; no transcription)",
        ]
        for day in sorted(self.days, key=lambda d: (d.day, d.unit_id)):
            mark = " " if day.complete else "!"
            lines.append(
                f" {mark} {day.day.isoformat()}  {day.unit_id:<10} "
                f"min {_fmt(day.min_c):>7}  max {_fmt(day.max_c):>7}  "
                f"{day.reading_count:>4}/{day.expected_count} readings, "
                f"longest gap {day.longest_gap_minutes:>5.0f}min"
            )
        lines += ["", "ALERTS"]
        for kind, count in sorted(self.alerts_by_kind.items()):
            lines.append(f"   {kind:<22} {count}")
        if not self.alerts_by_kind:
            lines.append("   none")
        lines += ["", "EXCURSIONS"]
        if not self.excursions:
            lines.append("   none")
        for excursion in self.excursions:
            state = "CLOSED" if excursion.closed_utc else "OPEN"
            lines.append(
                f"   {excursion.excursion_id} [{state}] "
                f"{excursion.min_temperature_c}C-{excursion.max_temperature_c}C for "
                f"{excursion.duration_minutes:.0f}min, "
                f"{excursion.doses_at_risk} dose(s) affected"
            )
        findings = self.findings()
        lines += ["", "FINDINGS THIS REPORT DECLARES ABOUT ITSELF"]
        if not findings:
            lines.append("   none")
        for finding in findings:
            lines.append(f"   - {finding}")
        return "\n".join(lines)


def _gaps(
    readings: Sequence[Reading],
    day_start: datetime,
    day_end: datetime,
    interval: float,
) -> tuple[float, float]:
    """`(longest_gap_minutes, total_gap_minutes)` from the real timestamps.

    A gap is any interval between consecutive readings longer than the expected
    polling interval, plus the stretches at either end of the day. Counting
    missing samples instead -- `(expected - count) * interval` -- gives the same
    total for 28 scattered drops and one 140-minute outage, and only one of
    those is a day nobody was watching the fridge.
    """
    if not readings:
        whole = (day_end - day_start).total_seconds() / 60.0
        return whole, whole
    edges = [day_start] + [r.taken_utc for r in readings] + [day_end]
    gaps = [
        max(0.0, (later - earlier).total_seconds() / 60.0 - interval)
        for earlier, later in zip(edges, edges[1:])
    ]
    return (max(gaps) if gaps else 0.0), sum(gaps)


def _fmt(value: float | None) -> str:
    return "--" if value is None else f"{value:.1f}C"


def build_compliance_report(
    *,
    month: str,
    units: Sequence[StorageUnit],
    feed: Any,
    config: ColdChainConfig,
    start: date,
    end: date,
    now: datetime,
    excursions: Sequence[ExcursionRecord] = (),
    alerts: Sequence[Alert] = (),
) -> ComplianceReport:
    """The monthly artefact, computed from the log rather than transcribed.

    README I-08's target state step 4: "Automatic min/max daily records -- no
    human transcription, no initials, no paper." The transcription is the part
    that produces the audit finding: a log filled in retroactively from memory
    is a falsified record, and it happens because somebody was off sick on the
    Tuesday.
    """
    interval = config.minutes("expected_reading_interval_minutes", 5.0)
    max_gap = config.minutes("sensor_offline_minutes", 30.0)
    expected_per_day = int((24 * 60) / interval) if interval > 0 else 0
    days: list[DailyRecord] = []
    cursor = start
    while cursor <= end:
        day_start = datetime.combine(cursor, datetime.min.time()).replace(
            tzinfo=now.tzinfo
        )
        day_end = day_start + timedelta(days=1) - timedelta(microseconds=1)
        for unit in units:
            readings = sorted(
                (
                    r for r in feed.readings(
                        unit.unit_id, since=day_start, until=day_end
                    )
                    if r.temperature_c is not None
                ),
                key=lambda r: r.taken_utc,
            )
            temps = [float(r.temperature_c) for r in readings]
            longest, total = _gaps(readings, day_start, day_end, interval)
            days.append(
                DailyRecord(
                    day=cursor,
                    unit_id=unit.unit_id,
                    min_c=min(temps) if temps else None,
                    max_c=max(temps) if temps else None,
                    reading_count=len(readings),
                    expected_count=expected_per_day,
                    gap_minutes=total,
                    longest_gap_minutes=longest,
                    max_gap_minutes=max_gap,
                )
            )
        cursor += timedelta(days=1)

    counts: dict[str, int] = {}
    for alert in alerts:
        counts[alert.kind] = counts.get(alert.kind, 0) + 1

    return ComplianceReport(
        month=month,
        generated_utc=now,
        config_version=config.version,
        units=list(units),
        days=days,
        excursions=list(excursions),
        alerts_by_kind=counts,
    )
