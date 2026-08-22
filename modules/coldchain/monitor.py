"""Vaccine cold chain telemetry. A sensor, a threshold, and an alert.

README I-08 asks the "does this need an LLM?" question and answers it in two
words: **"No. Emphatically no."**

    This is a sensor, a threshold, and an alert. Introducing an LLM into a
    temperature-monitoring path would add latency, cost, and a failure mode to a
    system whose entire value is deterministic reliability.

    Include this initiative in an "AI program" specifically to demonstrate the
    discipline of not using AI where it does not belong. A proposal that reaches
    for a language model in every section is not a technology strategy; it is a
    shopping list.

So there is no model in this package, `make lint` asserts it, and an AST test
asserts it again. This module is here to be the one that doesn't.

WHAT IT IS FOR. A pediatric practice holds $15,000-$40,000 of vaccine in two
appliances that are currently watched by a person walking past them twice a day.
Between Friday evening's check and Saturday morning's there are about fourteen
unmonitored hours. A compressor that fails at 9pm Friday is found at 9:30am
Saturday, and the whole inventory is gone.

THE FIVE CONDITIONS, and why each one is separate rather than a severity level
of the others:

  warning              approaching a limit. One unit drifting from 4C to 7.4C is
                       not yet an excursion and is exactly when a person can
                       still fix it.
  excursion            out of range, now.
  sustained_excursion  out of range for longer than the configured window. This
                       is the one that wakes the physician partner at 3am, and
                       it is separate because a door opened for ninety seconds
                       and a compressor that died are the same reading and very
                       different events.
  door_open            the most common real cause of an excursion, and invisible
                       to twice-daily checks.
  sensor_offline       THE IMPORTANT ONE. A logger that has stopped reporting
                       reads as a fridge with no problems. README I-08's risk
                       table rates a silently dead sensor HIGH for exactly that
                       reason: "a dead logger is worse than no logger because it
                       creates false confidence."

The last one is why this module is built around a CLOCK rather than around a
stream of readings. Everything else here reacts to data arriving. Silence is the
condition that has no data to react to, so `evaluate()` takes a `now` and asks
what it has NOT heard, and nothing in the design lets a caller evaluate only the
readings it happens to have.
"""

from __future__ import annotations

import os
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Iterable, Mapping, Protocol, Sequence

import yaml

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "AlertKind",
    "Severity",
    "Reading",
    "StorageUnit",
    "Alert",
    "ColdChainConfig",
    "ColdChainMonitor",
    "SensorFeed",
    "ScriptedFeed",
    "UnreviewedThresholds",
    "UnknownUnitType",
]

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "coldchain.yaml",
)


class AlertKind:
    WARNING = "warning"
    EXCURSION = "excursion"
    SUSTAINED_EXCURSION = "sustained_excursion"
    DOOR_OPEN = "door_open"
    SENSOR_OFFLINE = "sensor_offline"
    LOW_BATTERY = "low_battery"
    CALIBRATION_DUE = "calibration_due"

    ALL = (
        WARNING, EXCURSION, SUSTAINED_EXCURSION, DOOR_OPEN,
        SENSOR_OFFLINE, LOW_BATTERY, CALIBRATION_DUE,
    )
    #: The ones that mean vaccine is at risk RIGHT NOW.
    INVENTORY_AT_RISK = frozenset({EXCURSION, SUSTAINED_EXCURSION})


class Severity:
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    ORDER = {INFO: 0, WARNING: 1, CRITICAL: 2}


class UnreviewedThresholds(RuntimeError):
    """Raised when the shipped thresholds have no named clinical owner."""


class UnknownUnitType(KeyError):
    """Raised when a unit names a type the config does not define.

    Refused rather than defaulted. A freezer silently scored against the
    refrigerator band reads as a permanent, screaming excursion; a refrigerator
    scored against the freezer band reads as permanently fine. Both are worse
    than an error at startup.
    """


@dataclass(frozen=True)
class Reading:
    """One sample from one logger."""

    unit_id: str
    taken_utc: datetime
    temperature_c: float | None = None
    #: True while the door is open. None when the unit has no door sensor.
    door_open: bool | None = None
    battery_percent: float | None = None
    sensor_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "taken_utc": self.taken_utc.isoformat(),
            "temperature_c": self.temperature_c,
            "door_open": self.door_open,
            "battery_percent": self.battery_percent,
            "sensor_id": self.sensor_id,
        }


@dataclass(frozen=True)
class StorageUnit:
    """One appliance, and the paperwork that makes its readings admissible."""

    unit_id: str
    label: str
    unit_type: str
    location: str = ""
    #: NIST-traceable calibration is a CDC requirement, and a lapsed certificate
    #: makes every reading since the lapse unusable as evidence. README I-08:
    #: "Calendar the recalibration date at install."
    calibration_due: date | None = None
    #: Inventory value, used to say what an excursion is actually risking rather
    #: than making the reader look it up during an emergency.
    inventory_value_usd: float | None = None
    has_door_sensor: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id, "label": self.label,
            "unit_type": self.unit_type, "location": self.location,
            "calibration_due": (
                self.calibration_due.isoformat() if self.calibration_due else None
            ),
            "inventory_value_usd": self.inventory_value_usd,
            "has_door_sensor": self.has_door_sensor,
        }


@dataclass(frozen=True)
class Alert:
    """One condition, on one unit, at one time."""

    kind: str
    severity: str
    unit_id: str
    unit_label: str
    detected_utc: datetime
    detail: str
    notify: tuple[str, ...] = ()
    #: Set for temperature conditions.
    temperature_c: float | None = None
    limit_low_c: float | None = None
    limit_high_c: float | None = None
    #: How long the condition has been true, when that is knowable.
    duration_minutes: float | None = None
    inventory_value_usd: float | None = None

    @property
    def inventory_at_risk(self) -> bool:
        return self.kind in AlertKind.INVENTORY_AT_RISK

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "unit_id": self.unit_id,
            "unit_label": self.unit_label,
            "detected_utc": self.detected_utc.isoformat(),
            "detail": self.detail,
            "notify": list(self.notify),
            "temperature_c": self.temperature_c,
            "limits": [self.limit_low_c, self.limit_high_c],
            "duration_minutes": (
                round(self.duration_minutes, 1)
                if self.duration_minutes is not None else None
            ),
            "inventory_value_usd": self.inventory_value_usd,
        }


@dataclass
class ColdChainConfig:
    """Thresholds and routing, loaded from `config/coldchain.yaml`."""

    version: str
    review: dict[str, Any]
    escalation: dict[str, Any]
    unit_types: dict[str, dict[str, Any]]
    routing: dict[str, list[str]]

    @classmethod
    def load(cls, path: str | os.PathLike[str] = DEFAULT_CONFIG_PATH) -> "ColdChainConfig":
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        config = cls(
            version=str(data.get("version", "unversioned")),
            review=dict(data.get("review") or {}),
            escalation=dict(data.get("escalation") or {}),
            unit_types={k: dict(v) for k, v in (data.get("unit_types") or {}).items()},
            routing={k: list(v) for k, v in (data.get("routing") or {}).items()},
        )
        if not config.unit_types:
            raise ValueError("the cold chain config defines no unit types")
        for name, spec in config.unit_types.items():
            low, high = float(spec["min_c"]), float(spec["max_c"])
            if low >= high:
                raise ValueError(
                    f"unit type {name!r} has min_c {low} >= max_c {high}; a unit "
                    "with an inverted range is either always in excursion or "
                    "never in one, and both are silent failures"
                )
            margin = float(spec.get("warning_margin_c", 0.0))
            if margin < 0:
                raise ValueError(f"unit type {name!r} has a negative warning margin")
            if margin * 2 >= (high - low):
                # The warning band would swallow the whole in-range window, so
                # every normal reading warns. An alert that fires on everything
                # fails the same way as no alert.
                raise ValueError(
                    f"unit type {name!r} has a warning margin of {margin} inside a "
                    f"{high - low} degree range; every in-range reading would warn"
                )
        for kind in config.routing:
            if kind not in AlertKind.ALL:
                raise ValueError(f"routing names unknown alert kind {kind!r}")
        return config

    @property
    def has_clinical_owner(self) -> bool:
        owner = str(self.review.get("owner") or "")
        return bool(owner) and "UNASSIGNED" not in owner.upper()

    def band(self, unit_type: str) -> dict[str, Any]:
        try:
            return self.unit_types[unit_type]
        except KeyError as exc:
            raise UnknownUnitType(
                f"{unit_type!r} is not a configured unit type "
                f"({sorted(self.unit_types)}). A unit scored against the wrong "
                "band is either permanently alarming or permanently silent."
            ) from exc

    def minutes(self, key: str, default: float) -> float:
        return float(self.escalation.get(key, default))

    def notify_for(self, kind: str) -> tuple[str, ...]:
        return tuple(self.routing.get(kind, ()))


# -- the sensor layer --------------------------------------------------------


class SensorFeed(Protocol):
    """Where readings come from. A real one polls HTTP or subscribes to MQTT."""

    name: str

    def readings(
        self, unit_id: str, *, since: datetime, until: datetime
    ) -> Sequence[Reading]: ...


@dataclass
class ScriptedFeed:
    """A shipped test double holding pre-built readings.

    Not a mock of this module's logic: the threshold arithmetic, the duration
    tracking, the offline detection and the report are all the real code. This
    stands in for the logger hardware, the way `ScriptedOCR` stands in for an OCR
    engine in I-06.

    A real feed is a thin adapter -- an HTTP poll against the vendor's API, or an
    MQTT subscription -- and the vendor's SDK is imported inside it, never at
    module scope, so a machine without it can still run everything else.
    """

    name: str = "scripted"
    data: dict[str, list[Reading]] = field(default_factory=dict)

    def readings(
        self, unit_id: str, *, since: datetime, until: datetime
    ) -> Sequence[Reading]:
        return [
            r for r in self.data.get(unit_id, [])
            if since <= r.taken_utc <= until
        ]


# -- the monitor -------------------------------------------------------------


@dataclass
class ColdChainMonitor:
    """Evaluates units against the config. Deterministic; no model, no network.

    `evaluate()` returns every condition that is TRUE. `notifications()` returns
    the subset somebody should be told about, which is a different question:
    with a five-minute poll, "true" fires 288 times a day per condition, and a
    calibration date thirty days out would page the office manager 8,640 times
    before it came due. That is the failure this module's own config comments
    name -- an alert that fires on everything fails the same way as no alert.
    """

    config: ColdChainConfig
    units: dict[str, StorageUnit] = field(default_factory=dict)
    feed: Any = None
    audit: Any = None
    #: `(unit_id, kind)` -> when it was last notified. Held here rather than in
    #: the caller so that forgetting to dedupe is not the default.
    notified: dict[tuple[str, str], datetime] = field(default_factory=dict)
    INITIATIVE: str = "I-08"

    def __post_init__(self) -> None:
        for unit in self.units.values():
            # Fail at construction, not at 3am on the reading that matters.
            self.config.band(unit.unit_type)

    def require_reviewed(self) -> None:
        """Refuse to run against thresholds nobody has signed off.

        Same gate as I-06's reference ranges, for the same reason: the shipped
        numbers are the CDC published ones, but which appliance at THIS practice
        holds what, and who is accountable for the band, is a decision with a
        name on it.
        """
        if not self.config.has_clinical_owner:
            raise UnreviewedThresholds(
                "config/coldchain.yaml has no named owner "
                f"(review.owner = {self.config.review.get('owner')!r}). Assign "
                "one before monitoring anything: these thresholds decide whether "
                "$20,000 of vaccine is thrown away or given to a child."
            )

    # -- the five conditions -----------------------------------------------

    def evaluate(
        self,
        *,
        now: datetime,
        window: timedelta = timedelta(hours=24),
    ) -> list[Alert]:
        """Every condition true right now, across every unit. STATE, not events.

        Takes `now` rather than deriving it, because the most important
        condition in this module -- a sensor that has stopped reporting -- is
        detected by comparing the clock against the last reading. A monitor that
        only reacts to data arriving cannot see silence, and silence is the
        failure that looks exactly like everything being fine.

        Nothing is routed or audited here. A condition that is true on 288
        consecutive five-minute polls is one situation, not 288 of them, and
        `notifications()` is what decides who hears about it and how often.
        """
        self.require_reviewed()
        alerts: list[Alert] = []
        for unit_id in sorted(self.units):
            alerts.extend(self.evaluate_unit(unit_id, now=now, window=window))
        alerts.sort(key=lambda a: (-Severity.ORDER[a.severity], a.unit_id, a.kind))
        return alerts

    def notifications(
        self,
        *,
        now: datetime,
        window: timedelta = timedelta(hours=24),
        alerts: Sequence[Alert] | None = None,
    ) -> list[Alert]:
        """The alerts to actually SEND, after re-notification suppression.

        A condition is announced when it first appears and then no more often
        than its re-notify interval. Escalation is on the CONDITION CHANGING,
        not on it persisting: a compressor that died at 21:00 should wake the
        physician partner once and then at the configured cadence, not 288
        times before breakfast.

        Clearing is implicit -- a condition that stops being true stops
        appearing, and its entry is dropped, so it announces again if it
        returns. That is deliberate: an excursion that recurs after being fixed
        is news.
        """
        current = list(alerts if alerts is not None else self.evaluate(now=now, window=window))
        live = {(a.unit_id, a.kind) for a in current}
        for key in list(self.notified):
            if key not in live:
                del self.notified[key]

        out: list[Alert] = []
        for alert in current:
            key = (alert.unit_id, alert.kind)
            interval = self._renotify_minutes(alert)
            last = self.notified.get(key)
            if last is None or (now - last).total_seconds() / 60.0 >= interval:
                self.notified[key] = now
                out.append(alert)
                if self.audit is not None:
                    # Audited HERE rather than in `evaluate()`. An audit row per
                    # condition per poll is 1,234 rows a day on four units, and
                    # a log nobody can read is the same failure as an alert
                    # nobody reads.
                    self.audit.record_event(
                        actor_id="system:coldchain",
                        initiative_id=self.INITIATIVE,
                        event_type=f"coldchain_{alert.kind}",
                        detail={
                            "unit_id": alert.unit_id,
                            "severity": alert.severity,
                            "temperature_c": alert.temperature_c,
                            "duration_minutes": alert.duration_minutes,
                            "notify": list(alert.notify),
                        },
                    )
        return out

    def _renotify_minutes(self, alert: Alert) -> float:
        """How often a standing condition is repeated, by severity.

        Critical conditions repeat often enough that a missed page is caught;
        a calibration reminder repeats daily rather than every five minutes.
        """
        defaults = {
            Severity.CRITICAL: 60.0,
            Severity.WARNING: 240.0,
            Severity.INFO: 1440.0,
        }
        configured = self.config.escalation.get("renotify_minutes") or {}
        by_kind = configured.get(alert.kind)
        if by_kind is not None:
            return float(by_kind)
        return float(configured.get(alert.severity, defaults[alert.severity]))

    def evaluate_unit(
        self, unit_id: str, *, now: datetime, window: timedelta
    ) -> list[Alert]:
        unit = self.units[unit_id]
        band = self.config.band(unit.unit_type)
        low, high = float(band["min_c"]), float(band["max_c"])
        margin = float(band.get("warning_margin_c", 0.0))
        readings = sorted(
            self.feed.readings(unit_id, since=now - window, until=now),
            key=lambda r: r.taken_utc,
        )
        alerts: list[Alert] = []

        def make(
            kind: str, severity: str, detail: str, **extra: Any
        ) -> Alert:
            return Alert(
                kind=kind, severity=severity, unit_id=unit_id,
                unit_label=unit.label, detected_utc=now, detail=detail,
                notify=self.config.notify_for(kind),
                inventory_value_usd=unit.inventory_value_usd,
                **extra,
            )

        # 1. SILENCE. First, and unconditionally, because every check below it
        #    reasons about readings and this is the one that fires when there
        #    are none.
        offline_after = self.config.minutes("sensor_offline_minutes", 30.0)
        # The last reading that actually CARRIED A TEMPERATURE, not merely the
        # last packet. A logger whose thermistor fails but whose radio lives on
        # transmits every five minutes with `temperature_c=None`: measuring
        # silence from the packet made that unit look perfectly monitored while
        # the threshold checks below -- which filter on `temperature_c is not
        # None` -- ran over nothing. Zero alerts, indefinitely, on a fridge
        # holding $21,000 of vaccine.
        last = next(
            (r for r in reversed(readings) if r.temperature_c is not None), None
        )
        heard_from = readings[-1] if readings else None
        silent_for = (
            (now - last.taken_utc).total_seconds() / 60.0
            if last is not None
            else window.total_seconds() / 60.0
        )
        if silent_for > offline_after:
            alerts.append(
                make(
                    AlertKind.SENSOR_OFFLINE,
                    Severity.CRITICAL,
                    (
                        f"no reading for {silent_for:.0f} minutes "
                        f"(limit {offline_after:.0f}). "
                        + (
                            f"Last was {last.temperature_c}C at "
                            f"{last.taken_utc.isoformat()}."
                            if last is not None
                            else "No reading in the window carried a temperature "
                            "at all."
                        )
                        + (
                            f" The logger IS still transmitting (last packet "
                            f"{heard_from.taken_utc.isoformat()}) but without a "
                            "temperature: the probe has failed, not the radio."
                            if heard_from is not None
                            and (last is None or heard_from.taken_utc > last.taken_utc)
                            else ""
                        )
                        + " This unit is NOT being monitored; a dead logger reads "
                        "exactly like a healthy fridge."
                    ),
                    duration_minutes=silent_for,
                )
            )
            # Deliberately does NOT return. A unit can be both silent now and
            # have been in excursion when it last spoke, and the excursion is
            # the half that puts vaccine at risk.

        temps = [r for r in readings if r.temperature_c is not None]
        if temps:
            current = temps[-1]
            value = float(current.temperature_c)
            out_of_range = value < low or value > high
            sustained_after = self.config.minutes("sustained_excursion_minutes", 15.0)
            # CUMULATIVE exposure across the window, computed whether or not the
            # unit happens to be in range at this instant. A failing thermostat
            # cycling 11.5C / 6.0C spends half its life out of range and is
            # in-range on every other poll: measuring only the current
            # consecutive run reported "5 minutes" after eight cumulative hours
            # above 8C, and computing it only when the latest reading was out
            # meant half the polls saw nothing at all. CDC excursion assessment
            # is about total time out of range.
            cumulative = _cumulative_minutes(temps, low, high, now)

            if out_of_range:
                # How long has it been out? Walk back through consecutive
                # out-of-range readings. Measured from the first one, not from
                # the last in-range one: a gap in the data is not evidence the
                # unit was fine during it.
                since_utc = current.taken_utc
                bounded = True
                for reading in reversed(temps):
                    if reading.temperature_c is None:
                        break
                    if low <= float(reading.temperature_c) <= high:
                        bounded = False
                        break
                    since_utc = reading.taken_utc
                duration = (now - since_utc).total_seconds() / 60.0
                # `bounded` means the walk reached the edge of the evaluation
                # window without ever finding an in-range reading, so the unit
                # was ALREADY out when the window opened and this duration is a
                # floor, not a measurement. Saying "1440 minutes" about a
                # compressor that died sixty hours ago understates it by a
                # factor of two and a half, and the number goes on a document.
                at_least = "at least " if bounded else ""

                if duration > sustained_after or cumulative > sustained_after:
                    alerts.append(
                        make(
                            AlertKind.SUSTAINED_EXCURSION,
                            Severity.CRITICAL,
                            (
                                f"{value}C has been outside {low}-{high}C for "
                                f"{at_least}{duration:.0f} minutes"
                                + (
                                    " -- the unit was already out of range when "
                                    "this evaluation window opened, so widen the "
                                    "window to find when it started"
                                    if bounded else ""
                                )
                                + (
                                    f", and {cumulative:.0f} cumulative minutes "
                                    "out of range in this window"
                                    if cumulative > duration + 1 else ""
                                )
                                + ". Quarantine the inventory and start the "
                                "excursion workflow."
                            ),
                            temperature_c=value, limit_low_c=low, limit_high_c=high,
                            duration_minutes=max(duration, cumulative),
                        )
                    )
                else:
                    alerts.append(
                        make(
                            AlertKind.EXCURSION,
                            Severity.CRITICAL,
                            f"{value}C is outside {low}-{high}C "
                            f"(for {at_least}{duration:.0f} minutes so far).",
                            temperature_c=value, limit_low_c=low, limit_high_c=high,
                            duration_minutes=duration,
                        )
                    )
            elif cumulative > sustained_after:
                # In range AT THIS INSTANT and badly out of range recently. This
                # is what a failing thermostat looks like from a poll that
                # happened to land on a good minute, and it is the case that
                # escaped every check.
                alerts.append(
                    make(
                        AlertKind.SUSTAINED_EXCURSION,
                        Severity.CRITICAL,
                        (
                            f"{value}C is in range right now, but this unit has "
                            f"spent {cumulative:.0f} cumulative minutes outside "
                            f"{low}-{high}C in the last "
                            f"{window.total_seconds() / 3600:.0f} hours. A unit "
                            "cycling in and out of range is a failing thermostat, "
                            "and the exposure is cumulative. Quarantine the "
                            "inventory and start the excursion workflow."
                        ),
                        temperature_c=value, limit_low_c=low, limit_high_c=high,
                        duration_minutes=cumulative,
                    )
                )
            elif margin > 0 and (value < low + margin or value > high - margin):
                alerts.append(
                    make(
                        AlertKind.WARNING,
                        Severity.WARNING,
                        f"{value}C is within {margin}C of the {low}-{high}C "
                        "limit. Still in range; this is the point at which a "
                        "person can still fix it.",
                        temperature_c=value, limit_low_c=low, limit_high_c=high,
                    )
                )

        # 2. DOOR. The most common real cause of an excursion.
        door_limit = self.config.minutes("door_open_minutes", 3.0)
        door_open_since = _door_open_since(readings)
        if door_open_since is not None:
            open_for = (now - door_open_since).total_seconds() / 60.0
            if open_for > door_limit:
                alerts.append(
                    make(
                        AlertKind.DOOR_OPEN,
                        Severity.WARNING,
                        f"the door has been open for {open_for:.0f} minutes "
                        f"(limit {door_limit:.0f}).",
                        duration_minutes=open_for,
                    )
                )

        # 2b. A DOOR CHANNEL THAT SAID NOTHING. `has_door_sensor` was declared
        #     on the unit and read by nothing, so a door channel that died
        #     produced no alert of any kind -- the same false confidence as a
        #     dead thermometer, on the failure mode README I-08 calls the most
        #     common real cause of an excursion.
        if unit.has_door_sensor and readings and not any(
            r.door_open is not None for r in readings
        ):
            alerts.append(
                make(
                    AlertKind.SENSOR_OFFLINE,
                    Severity.WARNING,
                    "this unit has a door sensor and not one reading in the "
                    "window carried a door state. The door channel is not "
                    "reporting, and a door left ajar is the most common cause "
                    "of an excursion.",
                )
            )

        # 3. BATTERY. A logger about to become the failure.
        battery = next(
            (r.battery_percent for r in reversed(readings)
             if r.battery_percent is not None),
            None,
        )
        floor = self.config.minutes("low_battery_percent", 20.0)
        if battery is not None and battery <= floor:
            alerts.append(
                make(
                    AlertKind.LOW_BATTERY,
                    Severity.WARNING,
                    f"logger battery at {battery:.0f}% (floor {floor:.0f}%). "
                    "A dead logger is worse than no logger: it creates false "
                    "confidence.",
                )
            )

        # 4. CALIBRATION. A lapsed NIST certificate makes every reading since
        #    the lapse unusable as evidence, which is a compliance failure that
        #    produces no symptom at all until an auditor asks.
        if unit.calibration_due is not None:
            days = (unit.calibration_due - now.date()).days
            if days <= 30:
                alerts.append(
                    make(
                        AlertKind.CALIBRATION_DUE,
                        Severity.CRITICAL if days < 0 else Severity.WARNING,
                        (
                            f"NIST calibration LAPSED {abs(days)} day(s) ago; "
                            "readings since then are not audit evidence."
                            if days < 0
                            else f"NIST calibration due in {days} day(s)."
                        ),
                    )
                )
        return alerts


def _cumulative_minutes(
    temps: Sequence[Reading], low: float, high: float, now: datetime
) -> float:
    """Total minutes out of range across the window, however interrupted.

    Each out-of-range reading is credited with the interval up to the NEXT
    reading, so a poll that catches a 90-second spike does not count as five
    minutes of exposure just because the logger reports every five.
    """
    total = 0.0
    for index, reading in enumerate(temps):
        value = float(reading.temperature_c)  # type: ignore[arg-type]
        if low <= value <= high:
            continue
        following = (
            temps[index + 1].taken_utc if index + 1 < len(temps) else now
        )
        total += max(0.0, (following - reading.taken_utc).total_seconds() / 60.0)
    return total


def _door_open_since(readings: Sequence[Reading]) -> datetime | None:
    """When the door last opened, if it is still open. None if it is shut.

    Walks backwards to the start of the current open stretch. A unit whose last
    reading says the door is open has been open since the FIRST consecutive
    reading that said so, not since the last one -- measuring from the last
    reading would restart the clock on every poll and the three-minute rule
    would never fire.

    A `door_open=None` sample means the channel said NOTHING, which is not the
    same as saying "closed". Treating it as closed let a single null cancel a
    ninety-minute open-door alarm, and a null in the middle of the stretch cut
    the reported duration to a quarter of the truth. Nulls are stepped over; the
    walk stops only at an explicit `False`.
    """
    known = [r for r in readings if r.door_open is not None]
    if not known or known[-1].door_open is not True:
        return None
    since = known[-1].taken_utc
    for reading in reversed(known):
        if reading.door_open is not True:
            break
        since = reading.taken_utc
    return since
