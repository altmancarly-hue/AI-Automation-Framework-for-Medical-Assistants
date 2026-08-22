"""Synthetic patients for the I-03 tests and demo. No real patient data.

Each case exercises one thing the brief has to get right, including the ones it
has to get right by REFUSING: a length/stature method switch that looks like a
percentile crossing, a narrative item citing a date nobody supplied, and a
narrative item that reads as a suggested order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Mapping, Sequence

from .brief import OpenThread
from .growth import Indicator, Measurement
from .narrative import NARRATIVE_SECTIONS, Encounter
from .periodicity import CompletedScreening

__all__ = ["CLINIC_DATE", "GENERATED_UTC", "PreVisitCase", "CASES", "by_name"]

CLINIC_DATE = date(2026, 8, 24)
GENERATED_UTC = datetime(2026, 8, 23, 22, 0, tzinfo=timezone.utc)


@dataclass
class PreVisitCase:
    name: str
    description: str
    patient_id: str
    patient_label: str
    sex: str
    age_months: float
    age_label: str
    visit_type: str
    appointment_local: str
    provider: str
    measurements: tuple[Measurement, ...] = ()
    prior_measurements: tuple[Measurement, ...] = ()
    completed_screenings: tuple[CompletedScreening, ...] = ()
    risk_flags: tuple[str, ...] = ()
    immunizations_due: tuple[str, ...] = ()
    open_threads: tuple[OpenThread, ...] = ()
    encounters: tuple[Encounter, ...] = ()
    problem_list: tuple[str, ...] = ()
    data_horizon_months: float | None = None
    narrative_response: Mapping[str, Any] = field(default_factory=dict)
    expect: Mapping[str, Any] = field(default_factory=dict)

    @property
    def model_response(self) -> str:
        payload = {name: [] for name in NARRATIVE_SECTIONS}
        payload.update(self.narrative_response)
        return json.dumps(payload)


def _blank() -> dict[str, list]:
    return {name: [] for name in NARRATIVE_SECTIONS}


CASES: list[PreVisitCase] = []


CASES.append(PreVisitCase(
    name="toddler_18_month_well",
    description="18-month well visit: the two screenings that matter most are due.",
    patient_id="p_t18", patient_label="M.T. (18 mo)", sex="F",
    age_months=18.4, age_label="1y 6m", visit_type="Well child",
    appointment_local="2026-08-24 09:20 CDT", provider="Dr. Alvarez",
    measurements=(
        Measurement(Indicator.WEIGHT_FOR_AGE, 10.9, 18.4, "F", "2026-08-24"),
        Measurement(Indicator.LENGTH_FOR_AGE, 81.5, 18.4, "F", "2026-08-24", standing=False),
        Measurement(Indicator.HEAD_CIRCUMFERENCE_FOR_AGE, 46.8, 18.4, "F", "2026-08-24"),
    ),
    prior_measurements=(
        Measurement(Indicator.WEIGHT_FOR_AGE, 9.4, 12.1, "F", "2026-02-10"),
        Measurement(Indicator.LENGTH_FOR_AGE, 75.0, 12.1, "F", "2026-02-10", standing=False),
    ),
    completed_screenings=(
        CompletedScreening("newborn_metabolic", date(2025, 2, 18), 0.1, "normal"),
        CompletedScreening("newborn_hearing", date(2025, 2, 18), 0.1, "pass"),
        CompletedScreening("newborn_cchd", date(2025, 2, 18), 0.1, "pass"),
        CompletedScreening("newborn_bilirubin", date(2025, 2, 18), 0.1, "low risk"),
        CompletedScreening("maternal_depression", date(2025, 3, 20), 1.0, "negative"),
        CompletedScreening("maternal_depression", date(2025, 4, 18), 2.0, "negative"),
        CompletedScreening("maternal_depression", date(2025, 6, 17), 4.0, "negative"),
        CompletedScreening("maternal_depression", date(2025, 8, 15), 6.0, "negative"),
        CompletedScreening("tb_risk_assessment", date(2025, 3, 20), 1.0, "no risk"),
        CompletedScreening("tb_risk_assessment", date(2025, 8, 15), 6.0, "no risk"),
        CompletedScreening("tb_risk_assessment", date(2026, 2, 10), 12.1, "no risk"),
        CompletedScreening("lead_risk_assessment", date(2025, 8, 15), 6.0, "no risk"),
        CompletedScreening("lead_risk_assessment", date(2025, 11, 12), 9.2, "no risk"),
        CompletedScreening("lead_risk_assessment", date(2026, 2, 10), 12.1, "no risk"),
        CompletedScreening("oral_health_risk", date(2025, 8, 15), 6.0, "low risk"),
        CompletedScreening("oral_health_risk", date(2025, 11, 12), 9.2, "low risk"),
        CompletedScreening("oral_health_risk", date(2026, 2, 10), 12.1, "low risk"),
        CompletedScreening("oral_health_varnish", date(2025, 8, 15), 6.0, "applied"),
        CompletedScreening("oral_health_varnish", date(2026, 2, 10), 12.1, "applied"),
        CompletedScreening("vision_instrument", date(2026, 2, 10), 12.1, "pass"),
        CompletedScreening("developmental_screen", date(2025, 11, 12), 9.2, "pass"),
        CompletedScreening("anemia_screen", date(2026, 2, 10), 12.1, "11.9 g/dL"),
        CompletedScreening("lead_blood_universal", date(2026, 2, 10), 12.1, "2 mcg/dL"),
    ),
    immunizations_due=("DTaP #4 (was due at 15 mo)", "Hib #4"),
    encounters=(
        Encounter(date(2026, 2, 10), "Well child", "Twelve month well visit. Weight 9.4 kg. Parent asked about picky eating and limited variety at meals. Hemoglobin 11.9. Lead 2. Anticipatory guidance on toddler nutrition given."),
        Encounter(date(2026, 5, 3), "Sick", "Fever and pulling at right ear for two days. Right tympanic membrane erythematous and bulging. Acute otitis media. Amoxicillin 400mg/5mL, 5 mL twice daily for ten days."),
        Encounter(date(2026, 6, 21), "Sick", "Cough and congestion five days, no fever. Lungs clear. Viral upper respiratory infection. Parent again raised eating; says she still refuses most vegetables."),
    ),
    problem_list=("Acute otitis media, resolved",),
    narrative_response={
        "recent_relevant_history": [
            {"item": "Acute otitis media treated with amoxicillin 400mg/5mL, 5 mL twice daily", "source_date": "2026-05-03"},
            {"item": "Viral upper respiratory infection, lungs clear", "source_date": "2026-06-21"},
        ],
        "unresolved_parent_concerns": [
            {"item": "Parent asked about picky eating and limited variety at meals", "source_date": "2026-02-10"},
            {"item": "Parent again raised eating; she still refuses most vegetables", "source_date": "2026-06-21"},
        ],
        "medication_changes": [
            {"item": "Amoxicillin course for otitis media, ten days", "source_date": "2026-05-03"},
        ],
    },
    expect={"screenings_due": ["developmental_screen", "autism_screen"], "dose_scrubbed": True},
))


CASES.append(PreVisitCase(
    name="bmi_crossing_school_age",
    description="Six-year-old whose BMI crossed two channels upward since last year.",
    patient_id="p_bmi", patient_label="J.R. (6 y)", sex="M",
    age_months=74.0, age_label="6y 2m", visit_type="Well child",
    appointment_local="2026-08-24 10:40 CDT", provider="Dr. Alvarez",
    measurements=(
        Measurement(Indicator.BMI_FOR_AGE, 18.1, 74.0, "M", "2026-08-24"),
        Measurement(Indicator.STATURE_FOR_AGE, 118.0, 74.0, "M", "2026-08-24", standing=True),
    ),
    prior_measurements=(
        Measurement(Indicator.BMI_FOR_AGE, 15.9, 62.0, "M", "2025-08-20"),
        Measurement(Indicator.STATURE_FOR_AGE, 112.0, 62.0, "M", "2025-08-20", standing=True),
    ),
    completed_screenings=(
        CompletedScreening("vision_acuity", date(2025, 8, 20), 62.0, "20/25 both eyes"),
        CompletedScreening("hearing_audiometry", date(2025, 8, 20), 62.0, "pass"),
        CompletedScreening("lead_risk_assessment", date(2025, 8, 20), 62.0, "no risk"),
        CompletedScreening("oral_health_varnish", date(2025, 8, 20), 62.0, "applied"),
        CompletedScreening("blood_pressure", date(2025, 8, 20), 62.0, "98/60"),
        CompletedScreening("bmi_percentile", date(2025, 8, 20), 62.0, "65th"),
        CompletedScreening("tb_risk_assessment_annual", date(2025, 8, 20), 62.0, "no risk"),
    ),
    # This practice went live shortly before the five-year visit. Everything
    # earlier is on paper, so it is reported as unknown rather than asserted as
    # a miss -- otherwise day one of the deployment shows every established
    # patient overdue for years of screenings that were in fact done.
    data_horizon_months=60.0,
    open_threads=(
        OpenThread("referral", "Allergy referral placed", date(2026, 3, 14), 163,
                   detail="no report received in 163 days"),
    ),
    encounters=(
        Encounter(date(2025, 8, 20), "Well child", "Five year well visit. Vision 20/25 both eyes. Hearing pass. Parent reports intermittent nighttime cough."),
        Encounter(date(2026, 3, 14), "Sick", "Recurrent nighttime cough and wheeze with exercise. Allergy referral placed today. Parent to call for appointment."),
    ),
    problem_list=("Recurrent wheeze",),
    narrative_response={
        "open_threads": [
            {"item": "Allergy referral placed, parent to call for an appointment", "source_date": "2026-03-14"},
        ],
        "recent_relevant_history": [
            {"item": "Intermittent nighttime cough reported at the five year visit", "source_date": "2025-08-20"},
        ],
    },
    expect={"significant_crossing": True},
))


CASES.append(PreVisitCase(
    name="adolescent_first_phqa",
    description="Thirteen-year-old: the adolescent psychosocial screening block.",
    patient_id="p_teen", patient_label="A.K. (13 y)", sex="F",
    age_months=157.0, age_label="13y 1m", visit_type="Well child",
    appointment_local="2026-08-24 14:00 CDT", provider="Dr. Osei",
    measurements=(
        Measurement(Indicator.BMI_FOR_AGE, 21.4, 157.0, "F", "2026-08-24"),
        Measurement(Indicator.STATURE_FOR_AGE, 158.0, 157.0, "F", "2026-08-24", standing=True),
    ),
    completed_screenings=(
        CompletedScreening("vision_acuity", date(2025, 7, 2), 144.0, "20/20"),
    ),
    immunizations_due=("HPV #1", "MenACWY #1", "Tdap"),
    encounters=(
        Encounter(date(2025, 7, 2), "Well child", "Twelve year well visit. Vision 20/20. Doing well in school. No concerns raised."),
        Encounter(date(2026, 1, 18), "Sick", "Sore throat three days. Rapid strep negative. Symptomatic care advised."),
    ),
    problem_list=(),
    narrative_response={
        "recent_relevant_history": [
            {"item": "Sore throat in January, rapid strep negative", "source_date": "2026-01-18"},
        ],
    },
    expect={"screenings_due": ["adolescent_depression", "suicide_risk", "substance_use"]},
))


CASES.append(PreVisitCase(
    name="model_invents_a_date",
    description="ADVERSARIAL: the narrative cites an encounter that was never supplied.",
    patient_id="p_date", patient_label="R.P. (4 y)", sex="M",
    age_months=50.0, age_label="4y 2m", visit_type="Well child",
    appointment_local="2026-08-24 11:20 CDT", provider="Dr. Osei",
    measurements=(
        Measurement(Indicator.BMI_FOR_AGE, 15.6, 50.0, "M", "2026-08-24"),
    ),
    encounters=(
        Encounter(date(2026, 4, 2), "Well child", "Four year well visit. Growth tracking well. No parental concerns."),
    ),
    narrative_response={
        "recent_relevant_history": [
            {"item": "Growth tracking well, no parental concerns", "source_date": "2026-04-02"},
            {"item": "Febrile seizure managed in the emergency department", "source_date": "2026-03-14"},
        ],
    },
    expect={"dropped_reasons": ["cites an encounter date that was not supplied"]},
))


CASES.append(PreVisitCase(
    name="model_suggests_an_order",
    description="ADVERSARIAL: the narrative recommends a workup. A.4 forbids it.",
    patient_id="p_order", patient_label="S.N. (9 y)", sex="F",
    age_months=112.0, age_label="9y 4m", visit_type="Well child",
    appointment_local="2026-08-24 15:20 CDT", provider="Dr. Alvarez",
    measurements=(
        Measurement(Indicator.BMI_FOR_AGE, 17.0, 112.0, "F", "2026-08-24"),
    ),
    encounters=(
        Encounter(date(2026, 5, 9), "Sick", "Third episode of wheeze with exertion this year. Albuterol given in office with good response."),
        Encounter(date(2026, 6, 30), "Sick", "Cough at night for a week. Albuterol used twice. Lungs clear today."),
    ),
    problem_list=("Recurrent wheeze",),
    narrative_response={
        "recent_relevant_history": [
            {"item": "Third episode of exertional wheeze this year, albuterol given in office", "source_date": "2026-05-09"},
        ],
        "open_threads": [
            {"item": "Consider spirometry and an asthma workup at this visit", "source_date": "2026-06-30"},
            {"item": "Albuterol used twice for night cough", "source_date": "2026-06-30"},
        ],
    },
    expect={"dropped_reasons": ["suggested order"]},
))


CASES.append(PreVisitCase(
    name="length_to_stature_switch",
    description="The measurement method changed at two. That is not a growth finding.",
    patient_id="p_switch", patient_label="D.L. (2 y)", sex="M",
    age_months=25.0, age_label="2y 1m", visit_type="Well child",
    appointment_local="2026-08-24 08:40 CDT", provider="Dr. Osei",
    measurements=(
        Measurement(Indicator.STATURE_FOR_AGE, 86.0, 25.0, "M", "2026-08-24", standing=True),
        Measurement(Indicator.WEIGHT_FOR_AGE, 12.4, 25.0, "M", "2026-08-24"),
    ),
    prior_measurements=(
        # Recumbent, before the second birthday, against the infant table.
        Measurement(Indicator.LENGTH_FOR_AGE, 82.0, 21.0, "M", "2026-04-24", standing=False),
        Measurement(Indicator.WEIGHT_FOR_AGE, 11.6, 21.0, "M", "2026-04-24"),
    ),
    encounters=(),
    expect={"not_comparable": True},
))


def by_name(name: str) -> PreVisitCase:
    for case in CASES:
        if case.name == name:
            return case
    raise KeyError(name)
