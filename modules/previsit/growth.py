"""Growth percentiles, z-scores, and the channel-crossing check.

README I-03 names percentile crossing "the single most important longitudinal
signal in pediatrics" and lists "growth percentile crossing not noticed" as a
failure mode. It also classifies the calculation, correctly, as deterministic:

    "Has this child crossed growth percentiles?" -> Deterministic calculation
    against CDC/WHO growth reference. Pure math.

So there is no model anywhere in this file, and `make lint` asserts it. The
arithmetic is the LMS method the CDC publishes with its own tables:

    z = ((X/M)**L - 1) / (L*S)     for L != 0
    z = ln(X/M) / S                for L == 0
    percentile = 100 * Phi(z)

Everything that could drift -- the reference values, which table applies at
which age, where the channel lines sit -- lives in `config/growth/` as CSV
shipped verbatim from cdc.gov with a checksum manifest. There are no growth
numbers in this module, for the same reason there is no immunization schedule
in `modules/immunization/forecast.py`: a reference update has to be a
data diff a clinician can read, not a code change.

WHAT THIS DELIBERATELY DOES NOT DO, because a growth flag that is wrong is
worse than no growth flag:

  * It does not interpret. "Crossed two channels downward" is a fact; "failure
    to thrive" is a diagnosis and belongs to the physician.
  * It does not compare a recumbent LENGTH to a standing HEIGHT. The two differ
    by roughly 0.7 cm systematically, which is easily a channel at some ages,
    and a switch of measurement method around the second birthday is the single
    most common source of a false crossing. A pair measured differently is
    reported as not comparable rather than as a finding.
  * It does not extrapolate past the ends of a table.
"""

from __future__ import annotations

import bisect
import csv
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "DEFAULT_GROWTH_DIR",
    "Indicator",
    "Measurement",
    "GrowthPoint",
    "GrowthReference",
    "ChannelCrossing",
    "MAJOR_CHANNELS",
    "SIGNIFICANT_Z_CHANGE",
    "ImplausibleMeasurement",
    "OutOfRange",
    "NotComparable",
    "lms_z",
    "z_to_percentile",
    "percentile_to_z",
    "bmi",
    "channel_crossing",
    "ordinal",
]

DEFAULT_GROWTH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "growth",
)

#: The printed lines on a CDC growth chart. "Crossing two major channels" is
#: the clinical rule of thumb README I-03 is pointing at, and it only means
#: anything relative to a fixed set of lines -- so the set is named here and
#: carried on every result rather than assumed.
MAJOR_CHANNELS: tuple[float, ...] = (3.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 97.0)

#: A change of this many standard deviations counts as significant on its own,
#: whether or not any printed line lies between the two points.
#:
#: WHY THIS EXISTS: `MAJOR_CHANNELS` runs from the 3rd to the 97th, so a child
#: who is ALREADY outside those lines has none left to cross. A boy falling from
#: z = -2.10 to z = -3.77 -- the 1.8th percentile to the 0.008th, the single most
#: alarming trajectory in the schedule -- produced no crossing, no flag and no
#: line on the brief. The threshold is two channel-widths: the printed lines sit
#: roughly 0.67 SD apart through the middle of the chart, so 2 x 0.67 is the same
#: "two major channels" rule expressed in a unit that still works in the tails.
SIGNIFICANT_Z_CHANGE = 1.34

#: Below this age the infant tables and recumbent length apply; at and above it,
#: the 2-20 year tables and standing stature. CDC publishes overlapping ranges
#: and the choice has to be made explicitly somewhere.
INFANT_TABLE_MAX_MONTHS = 24.0


class OutOfRange(ValueError):
    """The measurement falls outside the published reference table."""


class ImplausibleMeasurement(ValueError):
    """The value is not a growth finding; it is almost certainly a typo.

    Raised rather than returned, because the alternative was worse than useless:
    11.0 kg entered as 110 kg for a two-year-old produced a percentile of 100
    and a four-channel crossing at the top of the brief. A number that extreme
    needs a human to look at the chart, which is hard constraint 3.
    """


class NotComparable(ValueError):
    """Two measurements cannot be compared. See the module docstring."""


class Indicator:
    WEIGHT_FOR_AGE = "weight_for_age"
    LENGTH_FOR_AGE = "length_for_age"
    STATURE_FOR_AGE = "stature_for_age"
    HEAD_CIRCUMFERENCE_FOR_AGE = "head_circumference_for_age"
    BMI_FOR_AGE = "bmi_for_age"
    WEIGHT_FOR_LENGTH = "weight_for_length"
    WEIGHT_FOR_STATURE = "weight_for_stature"


# -- the normal distribution, without scipy ---------------------------------


def z_to_percentile(z: float) -> float:
    """Phi(z) as a percentage. `math.erf` is exact enough and has no dependency."""
    return 50.0 * (1.0 + math.erf(z / math.sqrt(2.0)))


def percentile_to_z(percentile: float) -> float:
    """Inverse normal CDF by bisection.

    Bisection rather than a rational approximation on purpose: it is obviously
    correct, it converges to machine precision in about sixty iterations on a
    bounded interval, and this runs a few hundred times a night rather than a
    few million.
    """
    if not 0.0 < percentile < 100.0:
        raise ValueError("percentile must be strictly between 0 and 100")
    low, high = -12.0, 12.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if z_to_percentile(mid) < percentile:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def lms_z(value: float, *, l: float, m: float, s: float) -> float:
    """The CDC LMS transform. The L == 0 branch is a limit, not an edge case."""
    if value <= 0 or m <= 0 or s <= 0:
        raise ValueError("LMS requires positive value, M and S")
    if abs(l) < 1e-9:
        return math.log(value / m) / s
    return ((value / m) ** l - 1.0) / (l * s)


def ordinal(percentile: float) -> str:
    """"93rd", not "93th" -- and never "0th" or "100th".

    Rounding produced both. A growth line reading "weight for age: 7.45 (0th
    percentile)" states something that cannot be true and, worse, reads as a
    rounding artefact rather than as the extreme value it is. "<1st" and ">99th"
    are the conventional forms and they carry the alarm the rounded integer
    threw away.
    """
    if percentile < 1.0:
        return "<1st"
    if percentile > 99.0:
        return ">99th"
    n = int(round(percentile))
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def bmi(weight_kg: float, height_cm: float) -> float:
    if weight_kg <= 0 or height_cm <= 0:
        raise ValueError("BMI requires positive weight and height")
    return weight_kg / (height_cm / 100.0) ** 2


# -- reference tables --------------------------------------------------------


@dataclass(frozen=True)
class _Row:
    x: float
    l: float
    m: float
    s: float
    published: Mapping[str, float]


@dataclass(frozen=True)
class Measurement:
    """One recorded measurement. `standing` matters -- see the module docstring."""

    indicator: str
    value: float
    #: Age in months, or the length/stature in cm for the weight-for-X charts.
    x: float
    sex: str  # "M" | "F"
    taken_on: str = ""
    #: True when stature was measured standing, False when recumbent length was
    #: measured, None when the record does not say. None is not a synonym for
    #: either: an unknown method blocks a comparison rather than assuming one.
    standing: bool | None = None


@dataclass(frozen=True)
class GrowthPoint:
    """A measurement placed on the reference. Carries its own provenance."""

    indicator: str
    value: float
    x: float
    sex: str
    z: float
    percentile: float
    table: str
    taken_on: str = ""
    standing: bool | None = None

    @property
    def channel(self) -> int:
        """How many major lines this point is at or above. 0 = below the 3rd.

        Counted with `<=` so that it agrees exactly with `_lines_between`, which
        is what `channels_crossed` uses. `bisect_left` disagreed with it at every
        printed line -- a point sitting exactly on the 25th landed in the band
        below, so moving ONTO a line counted as a crossing and moving OFF one did
        not. Two public numbers describing the same movement must not contradict
        each other.
        """
        return sum(1 for c in MAJOR_CHANNELS if c <= self.percentile)

    def as_dict(self) -> dict[str, Any]:
        return {
            "indicator": self.indicator,
            "value": round(self.value, 3),
            "x": self.x,
            "z": round(self.z, 3),
            "percentile": round(self.percentile, 1),
            "table": self.table,
            "taken_on": self.taken_on,
        }


@dataclass(frozen=True)
class ChannelCrossing:
    """A movement across major chart lines between two measurements."""

    indicator: str
    earlier: GrowthPoint
    later: GrowthPoint
    channels_crossed: int
    direction: str  # "up" | "down"
    lines_crossed: tuple[float, ...]

    @property
    def z_change(self) -> float:
        return self.later.z - self.earlier.z

    @property
    def significant(self) -> bool:
        """Two major channels, or the same distance measured in SD.

        The second clause is not a loosening. It is what makes the rule apply to
        the children who are already off the printed part of the chart, where
        counting lines returns zero no matter how far they move.
        """
        return (
            self.channels_crossed >= 2
            or abs(self.z_change) >= SIGNIFICANT_Z_CHANGE
        )

    def describe(self) -> str:
        if not self.lines_crossed:
            return (
                f"{self.indicator.replace('_', ' ')} moved "
                f"{abs(self.z_change):.1f} SD {self.direction} beyond the printed "
                f"chart lines ({ordinal(self.earlier.percentile)} -> "
                f"{ordinal(self.later.percentile)}) between "
                f"{self.earlier.taken_on or 'the prior visit'} and "
                f"{self.later.taken_on or 'today'}"
            )
        return (
            f"{self.indicator.replace('_', ' ')} crossed "
            f"{self.channels_crossed} major channel(s) {self.direction} "
            f"({ordinal(self.earlier.percentile)} -> "
            f"{ordinal(self.later.percentile)}) "
            f"between {self.earlier.taken_on or 'the prior visit'} and "
            f"{self.later.taken_on or 'today'}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "indicator": self.indicator,
            "channels_crossed": self.channels_crossed,
            "z_change": round(self.z_change, 3),
            "direction": self.direction,
            "lines_crossed": list(self.lines_crossed),
            "from": self.earlier.as_dict(),
            "to": self.later.as_dict(),
            "significant": self.significant,
        }


class GrowthReference:
    """The CDC LMS tables, loaded from `config/growth/`.

    One object holds every table and picks the right one, because the picking
    is itself a rule that belongs next to the data: which table applies at
    which age is exactly the sort of thing that gets re-decided inconsistently
    when it lives at three call sites.
    """

    def __init__(self, directory: str | os.PathLike[str] = DEFAULT_GROWTH_DIR) -> None:
        self.directory = str(directory)
        manifest_path = os.path.join(self.directory, "MANIFEST.json")
        with open(manifest_path, "r", encoding="utf-8") as fh:
            self.manifest: dict[str, Any] = json.load(fh)
        self.source = str(self.manifest.get("source", "unknown"))
        self.biv: dict[str, tuple[float, float]] = self._load_biv()
        self._tables: dict[str, list[_Row]] = {}
        self._published: dict[str, list[str]] = {}
        for filename, meta in self.manifest["files"].items():
            for sex in ("1", "2"):
                key = f"{filename}:{sex}"
                self._tables[key] = self._load(filename, meta, sex)
            self._published[filename] = list(meta["published_percentiles"])

    def _load_biv(self) -> dict[str, tuple[float, float]]:
        """Implausible-value bounds. Data, like the tables themselves."""
        path = os.path.join(self.directory, "biv.yaml")
        if not os.path.exists(path):
            return {}
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return {
            str(name): (float(limits["z_min"]), float(limits["z_max"]))
            for name, limits in (raw.get("bounds") or {}).items()
        }

    # -- loading -----------------------------------------------------------

    def _load(self, filename: str, meta: Mapping[str, Any], sex: str) -> list[_Row]:
        path = os.path.join(self.directory, filename)
        x_column = str(meta["x_column"])
        percentile_columns = list(meta["published_percentiles"])
        rows: list[_Row] = []
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            for raw in csv.DictReader(fh):
                if raw["Sex"] != sex:
                    continue
                rows.append(
                    _Row(
                        x=float(raw[x_column]),
                        l=float(raw["L"]),
                        m=float(raw["M"]),
                        s=float(raw["S"]),
                        published={c: float(raw[c]) for c in percentile_columns},
                    )
                )
        rows.sort(key=lambda r: r.x)
        if not rows:
            raise ValueError(f"{filename} has no rows for sex {sex}")
        return rows

    # -- table selection ---------------------------------------------------

    def table_for(self, indicator: str, *, x: float) -> str:
        """Which CSV answers this indicator at this age or length."""
        if indicator == Indicator.WEIGHT_FOR_AGE:
            return "wtageinf.csv" if x < INFANT_TABLE_MAX_MONTHS else "wtage.csv"
        if indicator == Indicator.LENGTH_FOR_AGE:
            return "lenageinf.csv"
        if indicator == Indicator.STATURE_FOR_AGE:
            return "statage.csv"
        if indicator == Indicator.HEAD_CIRCUMFERENCE_FOR_AGE:
            return "hcageinf.csv"
        if indicator == Indicator.BMI_FOR_AGE:
            return "bmiagerev.csv"
        if indicator == Indicator.WEIGHT_FOR_LENGTH:
            return "wtleninf.csv"
        if indicator == Indicator.WEIGHT_FOR_STATURE:
            return "wtstat.csv"
        raise KeyError(f"no reference table for indicator {indicator!r}")

    def lms_at(self, filename: str, sex: str, x: float) -> tuple[float, float, float]:
        """L, M and S at an exact x, linearly interpolated between table rows.

        CDC's own guidance is to interpolate L, M and S rather than to round the
        age to the nearest tabulated row. Rounding a 30.4-month-old to 30 months
        moves the percentile by enough to invent or hide a channel crossing,
        which is the one thing this module exists to get right.
        """
        rows = self._tables[f"{filename}:{self._sex_code(sex)}"]
        lo, hi = rows[0].x, rows[-1].x
        if not lo <= x <= hi:
            raise OutOfRange(
                f"{x} is outside {filename} (published range {lo}-{hi}); the "
                "reference does not cover this measurement and this module does "
                "not extrapolate"
            )
        index = bisect.bisect_left([r.x for r in rows], x)
        if index < len(rows) and rows[index].x == x:
            row = rows[index]
            return row.l, row.m, row.s
        before, after = rows[index - 1], rows[index]
        span = after.x - before.x
        weight = (x - before.x) / span
        return (
            before.l + weight * (after.l - before.l),
            before.m + weight * (after.m - before.m),
            before.s + weight * (after.s - before.s),
        )

    @staticmethod
    def _sex_code(sex: str) -> str:
        # Normalise before comparing. "F" and "female" are the same child, and
        # comparing the raw strings made `channel_crossing` refuse a perfectly
        # good pair because one visit spelled it out.
        normalised = sex.strip().upper()[:1]
        if normalised in ("M", "1"):
            return "1"
        if normalised in ("F", "2"):
            return "2"
        raise ValueError(
            f"sex {sex!r} is not M or F. The growth reference is sex-specific "
            "and there is no combined table to fall back to; route to a human."
        )

    # -- the calculation ---------------------------------------------------

    def place(self, measurement: Measurement) -> GrowthPoint:
        """Put one measurement on the reference. Raises rather than guessing."""
        filename = self.table_for(measurement.indicator, x=measurement.x)
        l, m, s = self.lms_at(filename, measurement.sex, measurement.x)
        z = lms_z(measurement.value, l=l, m=m, s=s)
        bounds = self.biv.get(measurement.indicator)
        if bounds is not None and not bounds[0] <= z <= bounds[1]:
            raise ImplausibleMeasurement(
                f"{measurement.indicator} of {measurement.value:g} at "
                f"x={measurement.x:g} gives z={z:.1f}, outside the plausible "
                f"range {bounds[0]} to {bounds[1]}. This is a data-entry error, "
                "not a growth finding; check the chart before anything is "
                "computed from it."
            )
        return GrowthPoint(
            indicator=measurement.indicator,
            value=measurement.value,
            x=measurement.x,
            sex=measurement.sex,
            z=z,
            percentile=z_to_percentile(z),
            table=filename,
            taken_on=measurement.taken_on,
            standing=measurement.standing,
        )

    def published_value(
        self, filename: str, sex: str, x: float, percentile_column: str
    ) -> float:
        """A value CDC printed, for validating our arithmetic against theirs."""
        rows = self._tables[f"{filename}:{self._sex_code(sex)}"]
        for row in rows:
            if row.x == x:
                return row.published[percentile_column]
        raise KeyError(f"{filename} has no row at x={x}")

    def rows_for(self, filename: str, sex: str) -> list[_Row]:
        return list(self._tables[f"{filename}:{self._sex_code(sex)}"])

    def published_columns(self, filename: str) -> list[str]:
        return list(self._published[filename])


# -- crossing detection ------------------------------------------------------


def _lines_between(a: float, b: float) -> tuple[float, ...]:
    low, high = (a, b) if a <= b else (b, a)
    return tuple(c for c in MAJOR_CHANNELS if low < c <= high)


def channel_crossing(
    earlier: GrowthPoint, later: GrowthPoint
) -> ChannelCrossing | None:
    """Count the major chart lines between two points, or refuse to compare.

    The refusals are the point of this function. Comparing weight-for-age to
    BMI-for-age, or a recumbent length to a standing height, produces a number
    that looks exactly like a finding and is not one -- and the length/stature
    switch happens for every single patient at around the second birthday,
    which is also when the reference table changes underneath them.
    """
    family = {Indicator.LENGTH_FOR_AGE, Indicator.STATURE_FOR_AGE}
    if earlier.indicator != later.indicator:
        if {earlier.indicator, later.indicator} == family:
            # The commonest version of this in a real chart: the same child,
            # the same quantity, measured lying down before the second birthday
            # and standing after it, against two different reference tables.
            raise NotComparable(
                "these measurements used different methods (one recumbent "
                "length, one standing stature) and different reference tables. "
                "Recumbent length runs about 0.7 cm above standing stature, "
                "which is a channel at some ages, so the apparent change is at "
                "least partly the method, not the child."
            )
        raise NotComparable(
            f"{earlier.indicator} and {later.indicator} are different indicators"
        )
    if GrowthReference._sex_code(earlier.sex) != GrowthReference._sex_code(later.sex):
        raise NotComparable("the two measurements record different sexes")
    if later.x < earlier.x:
        raise NotComparable("the later measurement is at a younger age")
    length_indicators = (
        Indicator.LENGTH_FOR_AGE,
        Indicator.STATURE_FOR_AGE,
        Indicator.WEIGHT_FOR_LENGTH,
        Indicator.WEIGHT_FOR_STATURE,
    )
    if earlier.indicator in length_indicators:
        if earlier.standing is None or later.standing is None:
            raise NotComparable(
                "one of these measurements does not record whether the child was "
                "measured lying down or standing up. Recumbent length runs about "
                "0.7 cm above standing stature, which is a channel at some ages, "
                "so the pair is not comparable until the record says."
            )
        if earlier.standing != later.standing:
            raise NotComparable(
                "these measurements used different methods (one recumbent, one "
                "standing). The apparent change is at least partly the method, "
                "not the child."
            )
    crossed = _lines_between(earlier.percentile, later.percentile)
    if not crossed and abs(later.z - earlier.z) < SIGNIFICANT_Z_CHANGE:
        return None
    return ChannelCrossing(
        indicator=earlier.indicator,
        earlier=earlier,
        later=later,
        channels_crossed=len(crossed),
        direction="up" if later.z > earlier.z else "down",
        lines_crossed=crossed,
    )
