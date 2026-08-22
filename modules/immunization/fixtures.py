"""Synthetic immunization histories. No real patient data, ever.

Twenty-four cases chosen to exercise the parts of the schedule that are actually
hard, not the parts that are easy to write. Every name is invented, every date
is synthetic, and every DOB is anchored to `TODAY` so the fixtures do not rot:
a fixture set that silently changes meaning as the calendar advances is worse
than no fixture set, because it fails months after the change that broke it.

The four cases the build plan names explicitly are here by design:

  * `transferred_in_catch_up` -- a 3-year-old arriving from out of state with a
    partial series and no registry record.
  * `combination_split_across_sources` -- Pediarix in the chart, DTaP + HepB +
    IPV as three registry rows on the same day.
  * `duplicate_dose` -- the same MMR recorded twice, three days apart.
  * `partial_date_history` -- a paper record with "MMR 2021", month unknown.

The rest cover the things that break naive forecasters: the DTaP fourth dose
that completes the series when given at four years, a dose given three days
early that the grace period must still count, a dose given three weeks early
that it must not, the rotavirus age-out, the HPV two-dose/three-dose fork at the
fifteenth birthday, mixed Hib product variants, and an unrecognised CVX code
that must not be silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from .forecast import AdministeredDose, DosePrecision
from .matcher import DoseRecord

__all__ = ["Case", "CASES", "by_name", "TODAY", "months_before", "years_before"]

#: Fixed reference date. Cases state ages relative to this, so "a 4-month-old"
#: stays a 4-month-old regardless of when the suite runs.
TODAY = date(2026, 8, 22)


def months_before(months: int, *, day_offset: int = 0) -> date:
    """A DOB that makes the child exactly `months` old on TODAY."""
    year = TODAY.year
    month = TODAY.month - months
    year += (month - 1) // 12
    month = (month - 1) % 12 + 1
    from calendar import monthrange

    day = min(TODAY.day, monthrange(year, month)[1])
    return date(year, month, day) + timedelta(days=day_offset)


def years_before(years: int, *, day_offset: int = 0) -> date:
    return months_before(years * 12, day_offset=day_offset)


def _at(dob: date, *, weeks: int = 0, months: int = 0, days: int = 0) -> date:
    """A dose date expressed as an age."""
    year = dob.year
    month = dob.month + months
    year += (month - 1) // 12
    month = (month - 1) % 12 + 1
    from calendar import monthrange

    day = min(dob.day, monthrange(year, month)[1])
    return date(year, month, day) + timedelta(weeks=weeks, days=days)


@dataclass
class Case:
    name: str
    description: str
    first_name: str
    dob: date
    chart: list[DoseRecord] = field(default_factory=list)
    registry: list[DoseRecord] = field(default_factory=list)
    #: What a reviewer should see. Used by tests as documentation-with-teeth.
    expect: dict[str, Any] = field(default_factory=dict)

    @property
    def patient_id(self) -> str:
        return f"p_{self.name}"

    @property
    def family_id(self) -> str:
        return f"fam_{self.name}"

    def chart_doses(self) -> list[AdministeredDose]:
        return [r.to_administered() for r in self.chart]

    def patient_info(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "first_name": self.first_name,
            "dob": self.dob,
        }


def _c(case_name: str, index: int, cvx: str, given: date, **kw: Any) -> DoseRecord:
    return DoseRecord(
        record_id=f"{case_name}-c{index}", cvx=cvx, given=given, source="chart", **kw
    )


def _r(case_name: str, index: int, cvx: str, given: date, **kw: Any) -> DoseRecord:
    return DoseRecord(
        record_id=f"{case_name}-r{index}", cvx=cvx, given=given, source="registry", **kw
    )


def _build() -> list[Case]:
    cases: list[Case] = []

    # ---- 1. newborn, nothing but the birth dose -------------------------
    dob = months_before(1)
    cases.append(
        Case(
            "newborn_on_track",
            "6-week-old with only the birth HepB. Nothing is overdue yet.",
            "Amara",
            dob,
            chart=[_c("newborn_on_track", 1, "08", dob + timedelta(days=1))],
            expect={"open_gaps_empty": False, "hepb_status": "not_yet_due"},
        )
    )

    # ---- 2. on-schedule 6-month-old using a combination product ---------
    dob = months_before(6)
    name = "infant_pediarix_on_track"
    chart = []
    for i, age_months in enumerate((2, 4, 6), start=1):
        chart.append(_c(name, i, "110", _at(dob, months=age_months)))       # Pediarix
        chart.append(_c(name, i + 10, "133", _at(dob, months=age_months)))  # PCV13
        chart.append(_c(name, i + 20, "48", _at(dob, months=age_months)))   # ActHIB
    chart.append(_c(name, 30, "08", dob + timedelta(days=1)))
    chart.append(_c(name, 31, "119", _at(dob, months=2)))
    chart.append(_c(name, 32, "119", _at(dob, months=4)))
    cases.append(
        Case(
            name,
            "6-month-old fully on schedule on Pediarix, PCV13, ActHIB, Rotarix.",
            "Beau",
            dob,
            chart=chart,
            expect={"dtap_status": "not_yet_due", "rota_status": "complete"},
        )
    )

    # ---- 3. THE COMBINATION SPLIT CASE ----------------------------------
    dob = months_before(8)
    name = "combination_split_across_sources"
    day = _at(dob, months=2)
    cases.append(
        Case(
            name,
            "Pediarix in the chart; DTaP + HepB + IPV as three registry rows, "
            "same day. Must NOT read as a three-dose discrepancy.",
            "Cleo",
            dob,
            chart=[_c(name, 1, "110", day)],
            registry=[
                _r(name, 1, "20", day),
                _r(name, 2, "08", day),
                _r(name, 3, "10", day),
            ],
            expect={"ambiguous_min": 1, "matched": 0},
        )
    )

    # ---- 4. THE DUPLICATE DOSE CASE -------------------------------------
    dob = months_before(30)
    name = "duplicate_dose"
    first = _at(dob, months=12)
    cases.append(
        Case(
            name,
            "MMR recorded twice, three days apart, in the chart. One of the two "
            "is a data-entry error or a genuine duplicate administration.",
            "Dev",
            dob,
            chart=[
                _c(name, 1, "03", first),
                _c(name, 2, "03", first + timedelta(days=3)),
            ],
            registry=[_r(name, 1, "03", first)],
            expect={"duplicates": 1},
        )
    )

    # ---- 5. THE PARTIAL DATE CASE ---------------------------------------
    dob = years_before(6)
    name = "partial_date_history"
    cases.append(
        Case(
            name,
            "Paper record from another state: 'MMR 2021', month and day unknown.",
            "Elena",
            dob,
            chart=[
                _c(name, 1, "03", date(2021, 1, 1), precision=DosePrecision.YEAR,
                   product_text="MMR (year only on transfer record)"),
            ],
            registry=[_r(name, 1, "03", date(2021, 6, 14))],
            expect={"mmr_status": "requires_review", "ambiguous_min": 1},
        )
    )

    # ---- 6. THE TRANSFERRED-IN CATCH-UP CASE ----------------------------
    dob = years_before(3)
    name = "transferred_in_catch_up"
    cases.append(
        Case(
            name,
            "3-year-old arriving from out of state with two DTaP, one IPV, no "
            "MMR and no varicella. Needs a real catch-up plan.",
            "Fiona",
            dob,
            chart=[
                _c(name, 1, "08", dob + timedelta(days=1)),
                _c(name, 2, "20", _at(dob, months=2)),
                _c(name, 3, "20", _at(dob, months=4)),
                _c(name, 4, "10", _at(dob, months=2)),
            ],
            registry=[],
            expect={"mmr_status": "overdue", "var_status": "overdue",
                    "dtap_status": "overdue"},
        )
    )

    # ---- 7. DTaP 4th dose at 4 years completes the series ---------------
    dob = years_before(6)
    name = "dtap_four_doses_complete"
    cases.append(
        Case(
            name,
            "Four DTaP doses with the 4th given after the 4th birthday. The 5th "
            "is not needed and a naive forecaster will recall for it.",
            "Gus",
            dob,
            chart=[
                _c(name, 1, "20", _at(dob, months=2)),
                _c(name, 2, "20", _at(dob, months=4)),
                _c(name, 3, "20", _at(dob, months=6)),
                _c(name, 4, "20", _at(dob, months=49)),  # 4 y 1 mo
            ],
            expect={"dtap_status": "complete", "dtap_required": 4},
        )
    )

    # ---- 8. grace period: three days early still counts -----------------
    dob = months_before(14)
    name = "grace_period_three_days_early"
    cases.append(
        Case(
            name,
            "MMR given three days before the first birthday. ACIP's four-day "
            "grace rule makes it valid; every registry counts it.",
            "Hana",
            dob,
            chart=[_c(name, 1, "03", _at(dob, months=12) - timedelta(days=3))],
            expect={"mmr_valid_doses": 1},
        )
    )

    # ---- 9. genuinely too early: does not count -------------------------
    dob = months_before(20)
    name = "invalid_dose_too_early"
    cases.append(
        Case(
            name,
            "MMR given three weeks before the first birthday. Invalid; the child "
            "still needs two valid doses.",
            "Idris",
            dob,
            chart=[_c(name, 1, "03", _at(dob, months=12) - timedelta(days=21))],
            expect={"mmr_valid_doses": 0, "mmr_invalid_doses": 1},
        )
    )

    # ---- 10. rotavirus aged out -----------------------------------------
    dob = months_before(11)
    name = "rotavirus_aged_out"
    cases.append(
        Case(
            name,
            "11-month-old who never started rotavirus. Past 8 months it can "
            "never be given; recalling for it wastes a message.",
            "Juno",
            dob,
            chart=[_c(name, 1, "08", dob + timedelta(days=1))],
            expect={"rota_status": "aged_out"},
        )
    )

    # ---- 11. rotavirus urgently in window --------------------------------
    dob = months_before(2, day_offset=-10)
    name = "rotavirus_closing_window"
    cases.append(
        Case(
            name,
            "Infant approaching the 14-week 6-day first-dose deadline. The "
            "age-out bonus should put this near the top of the queue.",
            "Kai",
            dob,
            chart=[_c(name, 1, "08", dob + timedelta(days=1))],
            expect={"rota_status_in": ("due", "overdue")},
        )
    )

    # ---- 12. HPV two-dose series, started before 15 ----------------------
    dob = years_before(13)
    name = "hpv_two_dose_started"
    cases.append(
        Case(
            name,
            "13-year-old with one HPV dose. Two-dose series; second due 5 months "
            "after the first.",
            "Lena",
            dob,
            chart=[_c(name, 1, "165", _at(dob, months=143))],  # ~11 y 11 mo
            expect={"hpv_required": 2},
        )
    )

    # ---- 13. HPV three-dose series, started at 15 ------------------------
    dob = years_before(16)
    name = "hpv_three_dose_late_start"
    cases.append(
        Case(
            name,
            "Started HPV at 15 y 2 mo, so the series is three doses, not two.",
            "Mateo",
            dob,
            chart=[_c(name, 1, "165", _at(dob, months=182))],  # 15 y 2 mo
            expect={"hpv_required": 3},
        )
    )

    # ---- 14. adolescent with the classic missed cluster ------------------
    dob = years_before(14)
    name = "adolescent_gap_cluster"
    chart = [
        _c(name, 1, "08", dob + timedelta(days=1)),
        _c(name, 2, "110", _at(dob, months=2)),
        _c(name, 3, "110", _at(dob, months=4)),
        _c(name, 4, "110", _at(dob, months=6)),
        _c(name, 5, "20", _at(dob, months=18)),
        _c(name, 6, "20", _at(dob, months=52)),
        _c(name, 7, "03", _at(dob, months=12)),
        _c(name, 8, "03", _at(dob, months=52)),
        _c(name, 9, "21", _at(dob, months=12)),
        _c(name, 10, "21", _at(dob, months=52)),
        _c(name, 11, "10", _at(dob, months=52)),
    ]
    cases.append(
        Case(
            name,
            "14-year-old current on childhood vaccines but with no Tdap, no "
            "MenACWY and no HPV -- the most-missed category in pediatrics.",
            "Nia",
            dob,
            chart=chart,
            expect={"tdap_status": "overdue", "menacwy_status": "overdue",
                    "hpv_status": "overdue"},
        )
    )

    # ---- 15. kindergarten entry with school-required gaps ---------------
    dob = years_before(5, day_offset=-30)
    name = "kindergarten_school_gaps"
    cases.append(
        Case(
            name,
            "Turning 5 just before the school year, missing the 4-6 year "
            "boosters. Highest school-deadline urgency in the queue.",
            "Omar",
            dob,
            chart=[
                _c(name, 1, "08", dob + timedelta(days=1)),
                _c(name, 2, "110", _at(dob, months=2)),
                _c(name, 3, "110", _at(dob, months=4)),
                _c(name, 4, "110", _at(dob, months=6)),
                _c(name, 5, "20", _at(dob, months=18)),
                _c(name, 6, "03", _at(dob, months=12)),
                _c(name, 7, "21", _at(dob, months=12)),
            ],
            expect={"mmr_status": "overdue", "school_bonus_positive": True},
        )
    )

    # ---- 16. mixed Hib product variants ---------------------------------
    dob = months_before(18)
    name = "hib_mixed_variants"
    cases.append(
        Case(
            name,
            "Hib series started on PedvaxHIB (3-dose) and continued on ActHIB "
            "(4-dose). The conservative choice is the longer schedule.",
            "Priya",
            dob,
            chart=[
                _c(name, 1, "49", _at(dob, months=2)),
                _c(name, 2, "48", _at(dob, months=4)),
                _c(name, 3, "48", _at(dob, months=6)),
            ],
            expect={"hib_required": 4},
        )
    )

    # ---- 17. unknown CVX code -------------------------------------------
    dob = years_before(4)
    name = "unknown_cvx_code"
    cases.append(
        Case(
            name,
            "A code this table does not know. It must surface for a human, not "
            "be dropped -- a dropped dose looks exactly like a missing one.",
            "Quinn",
            dob,
            chart=[
                _c(name, 1, "03", _at(dob, months=12)),
                _c(name, 2, "999", _at(dob, months=15), product_text="unknown import"),
            ],
            expect={"unknown_codes_min": 1},
        )
    )

    # ---- 18. trade name only, no code -----------------------------------
    dob = years_before(2)
    name = "trade_name_only"
    cases.append(
        Case(
            name,
            "Transcribed paper record listing 'Pediarix' rather than a CVX code.",
            "Rafa",
            dob,
            chart=[_c(name, 1, "Pediarix", _at(dob, months=2))],
            expect={"resolves_to": "110"},
        )
    )

    # ---- 19. same dose, different equivalent codes ----------------------
    dob = years_before(2)
    name = "equivalent_codes_match"
    day = _at(dob, months=6)
    cases.append(
        Case(
            name,
            "Chart says CVX 08 (pediatric HepB), registry says CVX 45 "
            "(unspecified). Same antigen set, same day: a clean MATCH.",
            "Suri",
            dob,
            chart=[_c(name, 1, "08", day)],
            registry=[_r(name, 1, "45", day)],
            expect={"matched": 1, "ambiguous": 0},
        )
    )

    # ---- 20. one-day date drift between sources -------------------------
    dob = years_before(2)
    name = "one_day_drift"
    day = _at(dob, months=4)
    cases.append(
        Case(
            name,
            "Chart records the visit date, registry the transmission date, one "
            "day apart. Inside tolerance: MATCH, not a discrepancy.",
            "Tomas",
            dob,
            chart=[_c(name, 1, "133", day)],
            registry=[_r(name, 1, "133", day + timedelta(days=1))],
            expect={"matched": 1},
        )
    )

    # ---- 21. two-week drift: genuinely ambiguous ------------------------
    dob = years_before(3)
    name = "two_week_drift_ambiguous"
    day = _at(dob, months=12)
    cases.append(
        Case(
            name,
            "Same vaccine two weeks apart: a transcription slip, or two real "
            "doses. The rules cannot tell, so it escalates.",
            "Uma",
            dob,
            chart=[_c(name, 1, "21", day)],
            registry=[_r(name, 1, "21", day + timedelta(days=14))],
            expect={"ambiguous": 1},
        )
    )

    # ---- 22. registry has a dose the chart does not ---------------------
    dob = years_before(7)
    name = "registry_only_dose"
    cases.append(
        Case(
            name,
            "Flu shot given at a pharmacy, in I-CARE only. The huddle sheet must "
            "show it so the chart gets updated.",
            "Vik",
            dob,
            chart=[_c(name, 1, "03", _at(dob, months=12))],
            registry=[_r(name, 1, "150", date(TODAY.year, 10, 3)
                         if TODAY.month >= 10 else date(TODAY.year - 1, 10, 3))],
            expect={"registry_only": 1},
        )
    )

    # ---- 23. varicella dose 2 too soon under 13 -------------------------
    dob = years_before(5)
    name = "varicella_interval_under_13"
    first = _at(dob, months=14)
    cases.append(
        Case(
            name,
            "Second varicella dose six weeks after the first, given at 15.5 "
            "months so the minimum AGE is satisfied. Under 13 the minimum "
            "INTERVAL is three months, so the dose is still invalid.",
            "Wren",
            dob,
            chart=[
                _c(name, 1, "21", first),
                _c(name, 2, "21", first + timedelta(weeks=6)),
            ],
            expect={"var_valid_doses": 1, "var_invalid_doses": 1},
        )
    )

    # ---- 24. fully up to date adolescent --------------------------------
    dob = years_before(17)
    name = "adolescent_complete"
    chart = [
        _c(name, 1, "08", dob + timedelta(days=1)),
        _c(name, 2, "110", _at(dob, months=2)),
        _c(name, 3, "110", _at(dob, months=4)),
        _c(name, 4, "110", _at(dob, months=6)),
        _c(name, 5, "20", _at(dob, months=18)),
        _c(name, 6, "20", _at(dob, months=52)),
        _c(name, 7, "10", _at(dob, months=52)),
        _c(name, 8, "03", _at(dob, months=12)),
        _c(name, 9, "03", _at(dob, months=52)),
        _c(name, 10, "21", _at(dob, months=12)),
        _c(name, 11, "21", _at(dob, months=52)),
        _c(name, 12, "115", _at(dob, months=132)),
        _c(name, 13, "114", _at(dob, months=132)),
        _c(name, 14, "114", _at(dob, months=192)),
        _c(name, 15, "165", _at(dob, months=132)),
        _c(name, 16, "165", _at(dob, months=138)),
        _c(name, 17, "83", _at(dob, months=12)),
        _c(name, 18, "83", _at(dob, months=18)),
    ]
    cases.append(
        Case(
            name,
            "17-year-old fully current. Should generate no recall at all.",
            "Xime",
            dob,
            chart=chart,
            expect={"no_school_required_gaps": True},
        )
    )

    return cases


CASES: list[Case] = _build()


def by_name(name: str) -> Case:
    for case in CASES:
        if case.name == name:
            return case
    raise KeyError(f"no fixture named {name!r}")
