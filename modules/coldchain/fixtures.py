"""Synthetic units, readings and inventory for I-08. No real appliance, no PHI.

Six scenarios, chosen because between them they are every way a cold chain
fails in a real practice:

    steady               nothing wrong. The case that must stay quiet.
    drifting             4.0C climbing to 7.6C. Still in range, still fixable.
    friday_night         a compressor dies at 21:05 on a Friday and nobody is
                         in the building until Monday.
    door_ajar            the most common real cause, and invisible to a
                         twice-daily check.
    dead_logger          the sensor stopped reporting at 02:00 and the fridge
                         has read "no problems" ever since.
    freezer_deep         a freezer scored against the freezer band, to prove the
                         bands are per unit type and not global.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Iterable

from .excursion import VaccineLot
from .monitor import Reading, ScriptedFeed, StorageUnit

__all__ = [
    "NOW",
    "UNITS",
    "INVENTORY",
    "MANUFACTURER_CONTACTS",
    "build_feed",
    "unit",
]

#: A Monday morning. The Friday-night failure is 60 hours behind it.
NOW = datetime(2026, 8, 24, 9, 0, tzinfo=timezone.utc)

UNITS: tuple[StorageUnit, ...] = (
    StorageUnit(
        unit_id="fridge_1", label="Vaccine refrigerator (clinical hallway)",
        unit_type="refrigerator", location="Clinical hallway",
        calibration_due=date(2027, 3, 1), inventory_value_usd=21400.0,
    ),
    StorageUnit(
        unit_id="fridge_2", label="Vaccine refrigerator (back office)",
        unit_type="refrigerator", location="Back office",
        calibration_due=date(2026, 9, 10), inventory_value_usd=9800.0,
    ),
    StorageUnit(
        unit_id="freezer_1", label="Vaccine freezer", unit_type="freezer",
        location="Back office", calibration_due=date(2027, 1, 15),
        inventory_value_usd=4200.0,
    ),
    StorageUnit(
        unit_id="fridge_3", label="Vaccine refrigerator (satellite room)",
        unit_type="refrigerator", location="Satellite room",
        # Lapsed. Every reading since is not audit evidence.
        calibration_due=date(2026, 8, 1), inventory_value_usd=6100.0,
        has_door_sensor=False,
    ),
)

MANUFACTURER_CONTACTS = {
    "Sanofi": "1-800-822-2463",
    "Merck": "1-800-672-6372",
    "Pfizer": "1-800-438-1985",
    "GSK": "1-888-825-5249",
}

INVENTORY: tuple[VaccineLot, ...] = (
    VaccineLot("U7284AA", "DTaP (Daptacel)", "Sanofi", "fridge_1", 42,
               date(2027, 4, 30), 31.50),
    VaccineLot("K0193BC", "MMR-II", "Merck", "fridge_1", 28,
               date(2027, 1, 31), 92.40),
    VaccineLot("PF9921X", "Prevnar 20", "Pfizer", "fridge_1", 35,
               date(2026, 12, 15), 268.00),
    VaccineLot("G4410RT", "Havrix (HepA peds)", "GSK", "fridge_1", 24,
               date(2027, 6, 30), 44.80),
    VaccineLot("S2210KK", "IPOL (IPV)", "Sanofi", "fridge_2", 30,
               date(2027, 3, 31), 39.90),
    VaccineLot("M8813QQ", "Varivax", "Merck", "freezer_1", 18,
               date(2027, 2, 28), 168.00),
    VaccineLot("PF7742Z", "Comirnaty peds", "Pfizer", "fridge_3", 20,
               date(2026, 11, 30), 148.00),
)


def unit(unit_id: str) -> StorageUnit:
    for candidate in UNITS:
        if candidate.unit_id == unit_id:
            return candidate
    raise KeyError(unit_id)


def _series(
    unit_id: str,
    *,
    start: datetime,
    minutes: int,
    interval: int = 5,
    temps: Iterable[float] | None = None,
    constant: float | None = None,
    door_open_from: datetime | None = None,
    battery: float | None = None,
    door_channel: bool = True,
) -> list[Reading]:
    out: list[Reading] = []
    values = list(temps) if temps is not None else None
    count = minutes // interval
    for index in range(count):
        taken = start + timedelta(minutes=index * interval)
        if values is not None:
            value = values[min(index, len(values) - 1)]
        else:
            value = constant
        out.append(
            Reading(
                unit_id=unit_id,
                taken_utc=taken,
                temperature_c=value,
                # A working door sensor reports False when the door is shut. Only
                # a unit with NO door sensor reports None -- which is what the
                # door-channel-offline check exists to catch.
                door_open=(
                    None if not door_channel
                    else (door_open_from is not None and taken >= door_open_from)
                ),
                battery_percent=battery,
                sensor_id=f"logger_{unit_id}",
            )
        )
    return out


#: How far back the fixtures generate. 72 hours rather than 24 so that
#: 2026-08-23 is a COMPLETE day for every unit -- otherwise the compliance
#: report's own gap findings are dominated by the edge of the fixture window
#: rather than by anything a practice would act on.
FIXTURE_HOURS = 72


def build_feed(scenario: str = "monday_morning") -> ScriptedFeed:
    """Readings for the 72 hours ending at `NOW`.

    `scenario` selects which units misbehave. The default is the one the demo
    runs: a normal fridge, a drifting one, a dead compressor discovered on
    Monday, an open door, and a silent logger.
    """
    day_start = NOW - timedelta(hours=FIXTURE_HOURS)
    minutes = FIXTURE_HOURS * 60
    data: dict[str, list[Reading]] = {}

    # fridge_1 -- steady, with a door left open for the last eleven minutes.
    data["fridge_1"] = _series(
        "fridge_1", start=day_start, minutes=minutes, constant=4.4,
        door_open_from=NOW - timedelta(minutes=11), battery=88.0,
    )

    # fridge_2 -- the compressor died on Friday evening. By Monday the unit has
    # been out of range for sixty hours. The readings exist; nobody looked.
    failure = NOW - timedelta(hours=60)
    warm = []
    for index in range(FIXTURE_HOURS * 12):
        taken = day_start + timedelta(minutes=index * 5)
        warm.append(
            Reading(
                unit_id="fridge_2", taken_utc=taken,
                # Sitting at room temperature since Friday evening.
                temperature_c=19.8 if taken >= failure else 4.6,
                door_open=False, battery_percent=71.0, sensor_id="logger_fridge_2",
            )
        )
    data["fridge_2"] = warm

    # freezer_1 -- drifting up but still inside the -50 to -15 band.
    climb = [-20.0] * ((FIXTURE_HOURS - 1) * 12) + [
        -20.0 + step * 0.4 for step in range(12)
    ]
    data["freezer_1"] = _series(
        "freezer_1", start=day_start, minutes=minutes, temps=climb, battery=17.0,
    )

    # fridge_3 -- the logger stopped at 02:00. Seven hours of silence, and the
    # last thing it said was that everything was fine.
    silent_from = NOW - timedelta(hours=7)
    data["fridge_3"] = [
        r for r in _series(
            "fridge_3", start=day_start, minutes=minutes, constant=4.9,
            battery=41.0, door_channel=False,      # this unit has no door sensor
        )
        if r.taken_utc <= silent_from
    ]

    if scenario == "all_quiet":
        for unit_id in list(data):
            data[unit_id] = _series(
                unit_id, start=day_start, minutes=minutes,
                constant=-20.0 if unit_id == "freezer_1" else 4.5,
                battery=90.0, door_channel=unit_id != "fridge_3",
            )
    return ScriptedFeed(data=data)
