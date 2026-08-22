"""Synthetic patients for I-01. No PHI, no real children, no real forms.

Six cases, chosen so that between them they exercise every branch that matters:

    rosa_clean            everything agrees; the only case that releases
    theo_disputed         chart and registry hold DTaP doses 9 days apart --
                          inside I-02's ambiguity window, outside its tolerance
    nadia_registry_down   the registry was never asked; chart-only fill
    omar_stale_vitals     a height and weight from four years ago
    ivy_long_allergies    an allergy list wider than its box
    lucas_unknown_code    a dose with a CVX code the system does not know

The two twins in the panel (`theo` and `tessa`) share a date of birth and are
both flagged, which is what I-06's matcher is for -- this module deliberately
does not resolve patients, and the fixtures are here so a test can prove it
refuses rather than guesses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from modules.immunization.matcher import DoseRecord

from .chart import ChartRecord, SourceValue, StaticChartSource

__all__ = [
    "CLINIC_NOW",
    "FormCase",
    "CASES",
    "by_name",
    "build_chart_source",
    "chart_doses_for",
    "registry_doses_for",
]

#: A fixed "now" so every fixture's ages and staleness are deterministic.
CLINIC_NOW = datetime(2026, 8, 24, 9, 0, tzinfo=timezone.utc)
_TODAY = CLINIC_NOW.date()


def _v(value: Any, *, days_ago: int = 45, system: str = "ehr", resource: str = "") -> SourceValue:
    return SourceValue(
        value=value,
        system=system,
        resource=resource or "Observation/synthetic",
        recorded=_TODAY - timedelta(days=days_ago),
    )


@dataclass(frozen=True)
class FormCase:
    """One synthetic form request and the data behind it."""

    name: str
    description: str
    patient_id: str
    form_type: str
    channel: str
    record: ChartRecord
    chart_doses: tuple[DoseRecord, ...] = ()
    registry_doses: tuple[DoseRecord, ...] | None = ()
    registry_note: str = ""
    #: What a correct run should produce, asserted in the tests.
    expect: Mapping[str, Any] = field(default_factory=dict)


def _base_record(
    patient_id: str,
    *,
    last: str,
    first: str,
    dob: date,
    guardian: str,
    exam_days_ago: int = 45,
    vitals_days_ago: int | None = None,
    height_in: float = 48.5,
    weight_lb: float = 52.0,
    allergies: Sequence[str] = ("penicillin",),
    medications: Sequence[str] = (),
    conditions: Sequence[str] = (),
) -> ChartRecord:
    vitals_days = exam_days_ago if vitals_days_ago is None else vitals_days_ago
    return ChartRecord(
        patient_id=patient_id,
        data={
            "patient": {
                "last_name": _v(last, days_ago=0, resource="Patient/" + patient_id),
                "first_name": _v(first, days_ago=0, resource="Patient/" + patient_id),
                "middle_initial": _v("", days_ago=0),
                "date_of_birth": _v(dob, days_ago=0, resource="Patient/" + patient_id),
                "sex_on_record": _v("F", days_ago=0),
                "address": _v("1 Demo Lane, Buffalo Grove IL", days_ago=0),
                "guardian_name": _v(guardian, days_ago=0),
            },
            "exam": {"date": _v(_TODAY - timedelta(days=exam_days_ago), days_ago=exam_days_ago)},
            "vitals": {
                "height_in": _v(height_in, days_ago=vitals_days),
                "weight_lb": _v(weight_lb, days_ago=vitals_days),
                "bmi": _v(round(703 * weight_lb / (height_in ** 2), 1), days_ago=vitals_days),
                "bmi_percentile": _v(62, days_ago=vitals_days),
                "blood_pressure": _v(
                    {"systolic": 98, "diastolic": 60}, days_ago=vitals_days
                ),
            },
            "screenings": {
                "vision": _v("pass", days_ago=exam_days_ago),
                "hearing": _v("pass", days_ago=exam_days_ago),
                "dental_exam_date": _v(_TODAY - timedelta(days=120), days_ago=120),
                "lead_risk_assessed": _v(True, days_ago=exam_days_ago),
                "tb_risk_assessed": _v(True, days_ago=exam_days_ago),
                "diabetes_risk_assessed": _v(True, days_ago=exam_days_ago),
            },
            "labs": {
                "hemoglobin": _v(12.8, days_ago=exam_days_ago),
                "lead_ug_dl": _v(1.4, days_ago=400),
            },
            "allergies": _v(list(allergies), days_ago=exam_days_ago),
            "medications": _v(list(medications), days_ago=exam_days_ago),
            "conditions": _v(list(conditions), days_ago=exam_days_ago),
            "immunizations": {},
        },
    )


def _series(
    prefix: str, cvx: str, source: str, dates: Sequence[date], precision: str = "day"
) -> tuple[DoseRecord, ...]:
    return tuple(
        DoseRecord(
            record_id=f"{prefix}{index + 1}",
            cvx=cvx,
            given=given,
            source=source,
            precision=precision,
        )
        for index, given in enumerate(dates)
    )


# -- the cases ---------------------------------------------------------------

_ROSA_DOB = date(2018, 3, 14)
_ROSA_CHART = (
    _series("rc_d", "20", "chart",
            [date(2018, 5, 16), date(2018, 7, 18), date(2018, 9, 19), date(2019, 6, 3)])
    + _series("rc_p", "10", "chart",
              [date(2018, 5, 16), date(2018, 7, 18), date(2019, 6, 3)])
    + _series("rc_m", "03", "chart", [date(2019, 3, 20)])
    + _series("rc_v", "21", "chart", [date(2019, 3, 20)])
)
_ROSA_REGISTRY = (
    _series("rr_d", "20", "registry",
            [date(2018, 5, 16), date(2018, 7, 18), date(2018, 9, 19), date(2019, 6, 3)])
    + _series("rr_p", "10", "registry",
              [date(2018, 5, 16), date(2018, 7, 18), date(2019, 6, 3)])
    + _series("rr_m", "03", "registry", [date(2019, 3, 20)])
    + _series("rr_v", "21", "registry", [date(2019, 3, 20)])
)

_THEO_DOB = date(2017, 11, 2)
_THEO_CHART = _series("tc_d", "20", "chart", [date(2018, 1, 4), date(2018, 3, 6)])
# Nine days apart: past I-02's four-day tolerance, inside its 45-day ambiguity
# window. One administration recorded twice, or two real doses given close
# together -- the rules cannot tell, which is exactly the point.
_THEO_REGISTRY = _series("tr_d", "20", "registry", [date(2018, 1, 4), date(2018, 3, 15)])

_NADIA_DOB = date(2016, 6, 9)
_NADIA_CHART = (
    _series("nc_d", "20", "chart", [date(2016, 8, 11), date(2016, 10, 13)])
    + _series("nc_m", "03", "chart", [date(2017, 7, 1)], precision="month")
)

_OMAR_DOB = date(2011, 2, 18)
_OMAR_CHART = _series("oc_t", "115", "chart", [date(2022, 3, 4)])
_OMAR_REGISTRY = _series("or_t", "115", "registry", [date(2022, 3, 4)])

_IVY_DOB = date(2015, 9, 30)
_IVY_CHART = _series("ic_d", "20", "chart", [date(2015, 12, 1)])
_IVY_REGISTRY = _series("ir_d", "20", "registry", [date(2015, 12, 1)])

_LUCAS_DOB = date(2014, 4, 22)
_LUCAS_CHART = _series("lc_d", "20", "chart", [date(2014, 6, 24)]) + (
    DoseRecord("lc_x1", "9999", date(2016, 5, 5), "chart"),
)
_LUCAS_REGISTRY = _series("lr_d", "20", "registry", [date(2014, 6, 24)])


CASES: tuple[FormCase, ...] = (
    FormCase(
        name="rosa_clean",
        description="Everything agrees. The only case that reaches a signature.",
        patient_id="p_rosa",
        form_type="demo_camp_health_form",
        channel="portal",
        record=_base_record(
            "p_rosa", last="Alvarez", first="Rosa", dob=_ROSA_DOB,
            guardian="Marisol Alvarez", conditions=("asthma, mild intermittent",),
        ),
        chart_doses=_ROSA_CHART,
        registry_doses=_ROSA_REGISTRY,
        expect={"releasable": True, "disputed_groups": ()},
    ),
    FormCase(
        name="theo_disputed",
        description=(
            "Chart and registry hold DTaP doses nine days apart. One event or "
            "two? The grid row stays blank."
        ),
        patient_id="p_theo",
        form_type="demo_camp_health_form",
        channel="fax",
        record=_base_record(
            "p_theo", last="Nakamura", first="Theo", dob=_THEO_DOB,
            guardian="Kenji Nakamura", height_in=52.0, weight_lb=61.0,
        ),
        chart_doses=_THEO_CHART,
        registry_doses=_THEO_REGISTRY,
        expect={"releasable": False, "blocker": "unsettled_antigen"},
    ),
    FormCase(
        name="nadia_registry_down",
        description=(
            "I-CARE was not consulted. The form is still produced, from the "
            "chart alone, and says so."
        ),
        patient_id="p_nadia",
        form_type="demo_camp_health_form",
        channel="email",
        record=_base_record(
            "p_nadia", last="Okonkwo", first="Nadia", dob=_NADIA_DOB,
            guardian="Chidi Okonkwo", height_in=55.0, weight_lb=70.0,
        ),
        chart_doses=_NADIA_CHART,
        registry_doses=None,
        registry_note="I-CARE HL7 interface is not licensed at this site",
        expect={"releasable": False, "blocker": "registry_not_reconciled"},
    ),
    FormCase(
        name="omar_stale_vitals",
        description=(
            "Height and weight from four years ago. They fill the boxes and "
            "the form looks current."
        ),
        patient_id="p_omar",
        form_type="demo_camp_health_form",
        channel="front_desk",
        record=_base_record(
            "p_omar", last="Haddad", first="Omar", dob=_OMAR_DOB,
            guardian="Layla Haddad", exam_days_ago=30, vitals_days_ago=1490,
            height_in=58.0, weight_lb=88.0,
        ),
        chart_doses=_OMAR_CHART,
        registry_doses=_OMAR_REGISTRY,
        expect={"releasable": False, "blocker": "stale_value"},
    ),
    FormCase(
        name="ivy_long_allergies",
        description=(
            "An allergy list wider than its box. Clipped on the page, it reads "
            "as complete."
        ),
        patient_id="p_ivy",
        form_type="demo_camp_health_form",
        channel="portal",
        record=_base_record(
            "p_ivy", last="Petrov", first="Ivy", dob=_IVY_DOB,
            guardian="Anya Petrov", height_in=50.0, weight_lb=57.0,
            allergies=(
                "penicillin (hives)", "amoxicillin (hives)", "peanut (anaphylaxis)",
                "tree nuts (anaphylaxis)", "shellfish", "latex", "bee sting",
                "sulfa drugs", "eggs (mild)",
            ),
        ),
        chart_doses=_IVY_CHART,
        registry_doses=_IVY_REGISTRY,
        expect={"releasable": False, "blocker": "value_truncated"},
    ),
    FormCase(
        name="lucas_unknown_code",
        description=(
            "A dose with a CVX code the system does not know. Until it is "
            "identified, no row of the grid is complete."
        ),
        patient_id="p_lucas",
        form_type="demo_camp_health_form",
        channel="fax",
        record=_base_record(
            "p_lucas", last="Byrne", first="Lucas", dob=_LUCAS_DOB,
            guardian="Sean Byrne", height_in=59.0, weight_lb=92.0,
        ),
        chart_doses=_LUCAS_CHART,
        registry_doses=_LUCAS_REGISTRY,
        expect={"releasable": False, "blocker": "unknown_cvx"},
    ),
)


def by_name(name: str) -> FormCase:
    for case in CASES:
        if case.name == name:
            return case
    raise KeyError(name)


def build_chart_source(*, unreachable: Sequence[str] = ()) -> StaticChartSource:
    """A fresh source holding a fresh copy of every record.

    A copy, because `FormPipeline.prepare` writes the reconciled immunization
    block into the record it is handed, and a shared record would carry one
    case's grid into the next test.
    """
    import copy

    return StaticChartSource(
        records={
            case.patient_id: copy.deepcopy(case.record) for case in CASES
        },
        unreachable=frozenset(unreachable),
    )


def chart_doses_for(name: str) -> list[DoseRecord]:
    return list(by_name(name).chart_doses)


def registry_doses_for(name: str) -> list[DoseRecord] | None:
    doses = by_name(name).registry_doses
    return None if doses is None else list(doses)
