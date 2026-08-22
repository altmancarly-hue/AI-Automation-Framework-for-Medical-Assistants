"""Synthetic faxes for the I-06 tests, demo and eval set. No real patient data.

Includes the twin fixture the build plan asks for by name, a sibling pair, a
document with an unreadable middle page, an outside lab whose critical value the
model calls routine, and a registry printout for the I-02 handoff.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping, Sequence

from .match import PanelPatient

__all__ = ["PANEL", "AGE_MONTHS", "FaxCase", "CASES", "by_name", "eval_cases"]

#: Two sets of twins, a sibling pair with the same surname, and a duplicate
#: chart -- all four of the shapes that make patient matching dangerous.
PANEL: tuple[PanelPatient, ...] = (
    PanelPatient("p_ari", "Ari", "Nakamura", date(2019, 3, 14), multiple_birth=True),
    PanelPatient("p_ren", "Ren", "Nakamura", date(2019, 3, 14), multiple_birth=True),
    PanelPatient("p_sofia", "Sofia", "Reyes", date(2021, 6, 30)),
    PanelPatient("p_mateo", "Mateo", "Reyes", date(2019, 11, 2)),
    PanelPatient("p_mila", "Mila", "Torres", date(2024, 4, 12), aliases=("Ludmila",)),
    PanelPatient("p_juno", "Juno", "Petrov", date(2011, 11, 8)),
    PanelPatient("p_chidi", "Chidi", "Adeyemi", date(2015, 1, 22)),
    PanelPatient("p_dup_a", "Wren", "Okafor", date(2020, 9, 5)),
    PanelPatient("p_dup_b", "Wren", "Okafor", date(2020, 9, 5)),
    PanelPatient("p_ez", "Ezra", "Lindqvist", date(2026, 7, 1)),
)

#: patient_id -> age in months, as of the fixture clock (2026-08-24).
AGE_MONTHS: Mapping[str, float] = {
    "p_ari": 89.0, "p_ren": 89.0, "p_sofia": 62.0, "p_mateo": 81.0,
    "p_mila": 28.0, "p_juno": 177.0, "p_chidi": 139.0,
    "p_dup_a": 71.0, "p_dup_b": 71.0, "p_ez": 1.8,
}


@dataclass
class FaxCase:
    name: str
    description: str
    #: (page text, page confidence)
    pages: tuple[tuple[str, float], ...]
    document_type: str
    patient_name: str | None
    patient_dob: str | None
    sending_facility: str | None
    summary: str
    classifier_confidence: float = 0.94
    model_urgency: str = "routine"
    model_urgency_confidence: float = 0.9
    model_urgency_reason: str = "administrative document"
    model_abnormal: tuple[str, ...] = ()
    expect: Mapping[str, Any] = field(default_factory=dict)

    @property
    def filename(self) -> str:
        return f"{self.name}.tiff"

    @property
    def text(self) -> str:
        return "\n\n".join(text for text, _c in self.pages)

    @property
    def classifier_response(self) -> str:
        return json.dumps(
            {
                "document_type": self.document_type,
                "confidence": self.classifier_confidence,
                "patient_name": self.patient_name,
                "patient_dob": self.patient_dob,
                "sending_facility": self.sending_facility,
                "one_line_summary": self.summary,
            }
        )

    @property
    def urgency_response(self) -> str:
        return json.dumps(
            {
                "urgency": self.model_urgency,
                "confidence": self.model_urgency_confidence,
                "reason": self.model_urgency_reason,
                "abnormal_values_detected": list(self.model_abnormal),
            }
        )


CASES: list[FaxCase] = []


CASES.append(FaxCase(
    name="routine_records_request",
    description="An administrative records request. The only shape that auto-files.",
    pages=(
        ("NORTHSHORE FAMILY MEDICINE\nRECORDS REQUEST\n\n"
         "Patient: Petrov, Juno\nDOB: 11/08/2011\n\n"
         "Please forward immunization and visit history for the above patient.\n"
         "Signed release attached.\n", 0.95),
    ),
    document_type="records_request",
    patient_name="Petrov, Juno", patient_dob="11/08/2011",
    sending_facility="NorthShore Family Medicine",
    summary="Records request with signed release",
    model_urgency="routine", model_urgency_reason="administrative records request",
    expect={"queue": "auto_file", "tasks": 0},
))


CASES.append(FaxCase(
    name="critical_lab_called_routine",
    description="ADVERSARIAL: a critical hemoglobin the model calls routine.",
    pages=(
        ("LAKESIDE REFERENCE LABORATORY\nCOMPLETE BLOOD COUNT\n\n"
         "Patient: Adeyemi, Chidi        DOB: 01/22/2015\n"
         "Collected: 08/22/2026\n\n"
         "Hemoglobin      6.8   g/dL     L\n"
         "Hematocrit      21.0  %        L\n"
         "WBC             7.4   K/uL\n"
         "Platelets       310   K/uL\n", 0.93),
    ),
    document_type="outside_lab",
    patient_name="Adeyemi, Chidi", patient_dob="01/22/2015",
    sending_facility="Lakeside Reference Laboratory",
    summary="CBC with low hemoglobin",
    # The model gets this wrong on purpose. The rule has to catch it.
    model_urgency="routine", model_urgency_confidence=0.88,
    model_urgency_reason="routine complete blood count",
    expect={"queue": "urgent_alert", "escalated": True},
))


CASES.append(FaxCase(
    name="twin_discharge_summary",
    description="A discharge summary for one of two twins. Must go to a human.",
    pages=(
        ("EVANSTON CHILDREN'S HOSPITAL\nDISCHARGE SUMMARY\n\n"
         "Patient: Nakamura        DOB: 03/14/2019\n"
         "Admitted 08/18/2026, discharged 08/21/2026\n\n"
         "Admitted with status asthmaticus. Treated with continuous albuterol\n"
         "and systemic steroids. Improved and discharged in stable condition.\n"
         "Follow up with primary care within 48 hours.\n", 0.91),
    ),
    document_type="hospital_discharge",
    patient_name="Nakamura", patient_dob="03/14/2019",
    sending_facility="Evanston Children's Hospital",
    summary="Discharge after admission for status asthmaticus",
    model_urgency="urgent", model_urgency_confidence=0.93,
    model_urgency_reason="explicit recommendation for follow-up within 48 hours",
    model_abnormal=("follow up within 48 hours",),
    expect={"queue": "urgent_alert", "match_outcome": "multiple_candidates"},
))


CASES.append(FaxCase(
    name="twin_named_discharge",
    description="The same discharge, with a first name. Still two twins on that DOB.",
    pages=(
        ("EVANSTON CHILDREN'S HOSPITAL\nDISCHARGE SUMMARY\n\n"
         "Patient: Nakamura, Ari       DOB: 03/14/2019\n"
         "Discharged 08/21/2026 in stable condition.\n"
         "Follow up with primary care within one week.\n", 0.92),
    ),
    document_type="hospital_discharge",
    patient_name="Nakamura, Ari", patient_dob="03/14/2019",
    sending_facility="Evanston Children's Hospital",
    summary="Discharge after admission",
    model_urgency="needs_physician_review",
    model_urgency_reason="discharge summary requiring acknowledgment",
    expect={"match_outcome": "multiple_candidates"},  # flagged twins are never auto-filed
))


CASES.append(FaxCase(
    name="unreadable_middle_page",
    description="A three-page consult whose middle page did not OCR.",
    pages=(
        ("LURIE CHILDREN'S ALLERGY\nCONSULTATION NOTE\n\n"
         "Patient: Reyes, Sofia      DOB: 06/30/2021\n", 0.94),
        ("", 0.0),
        ("Plan discussed with the family. Return in three months.\n", 0.90),
    ),
    document_type="specialist_consult",
    patient_name="Reyes, Sofia", patient_dob="06/30/2021",
    sending_facility="Lurie Children's Allergy",
    summary="Allergy consultation note",
    model_urgency="needs_physician_review",
    model_urgency_reason="specialist consult requiring acknowledgment",
    expect={"queue": "human_indexing", "ocr_needs_human": True},
))


CASES.append(FaxCase(
    name="immunization_registry_printout",
    description="An I-CARE printout. Goes to I-02, never straight to the chart.",
    pages=(
        ("ILLINOIS COMPREHENSIVE AUTOMATED IMMUNIZATION REGISTRY EXCHANGE\n"
         "IMMUNIZATION HISTORY\n\n"
         "Patient: Torres, Mila     DOB: 04/12/2024\n\n"
         "DTaP            06/14/2024\n"
         "DTaP            08/16/2024\n"
         "IPV             06/14/2024\n"
         "Hep B           04/12/2024\n"
         "PCV13           06/14/2024\n"
         "Rotavirus       06/2024\n"
         "MMR             (not administered)\n", 0.93),
    ),
    document_type="immunization_record",
    patient_name="Torres, Mila", patient_dob="04/12/2024",
    sending_facility="I-CARE",
    summary="Immunization history from the state registry",
    model_urgency="routine", model_urgency_reason="immunization history printout",
    expect={"queue": "immunization_reconciliation", "doses": 6},
))


CASES.append(FaxCase(
    name="unknown_patient_prior_auth",
    description="A prior auth for a child who is not on the panel.",
    pages=(
        ("MERIDIAN HEALTH PLAN\nPRIOR AUTHORIZATION DETERMINATION\n\n"
         "Member: Hollis, Wren      DOB: 02/03/2017\n"
         "Request for montelukast has been approved through 08/2027.\n", 0.96),
    ),
    document_type="prior_auth",
    patient_name="Hollis, Wren", patient_dob="02/03/2017",
    sending_facility="Meridian Health Plan",
    summary="Prior authorization approved",
    model_urgency="routine", model_urgency_reason="payer correspondence",
    expect={"queue": "human_indexing", "match_outcome": "no_panel_patient_with_that_dob"},
))


CASES.append(FaxCase(
    name="duplicate_chart_school_form",
    description="A school form for a child who is on the panel twice.",
    pages=(
        ("BUFFALO GROVE SCHOOL DISTRICT 21\n"
         "CERTIFICATE OF CHILD HEALTH EXAMINATION\n\n"
         "Student: Okafor, Wren     DOB: 09/05/2020\n"
         "Please complete and return for kindergarten enrollment.\n", 0.95),
    ),
    document_type="school_form",
    patient_name="Okafor, Wren", patient_dob="09/05/2020",
    sending_facility="Buffalo Grove School District 21",
    summary="Kindergarten health examination form",
    model_urgency="routine", model_urgency_reason="school form",
    expect={"queue": "human_indexing", "match_outcome": "multiple_candidates"},
))


CASES.append(FaxCase(
    name="newborn_bilirubin_normal_for_age",
    description="A bilirubin that is alarming for a child and normal for a newborn.",
    pages=(
        ("EVANSTON HOSPITAL NEWBORN NURSERY\nLABORATORY REPORT\n\n"
         "Patient: Lindqvist, Ezra     DOB: 07/01/2026\n"
         "Collected: 07/02/2026 (36 hours of life)\n"
         "Total bilirubin   9.4  mg/dL\n", 0.94),
    ),
    document_type="outside_lab",
    patient_name="Lindqvist, Ezra", patient_dob="07/01/2026",
    sending_facility="Evanston Hospital Newborn Nursery",
    summary="Newborn bilirubin",
    model_urgency="needs_physician_review",
    model_urgency_reason="newborn laboratory result requiring acknowledgment",
    expect={"not_escalated_by_numbers": True},
))


CASES.append(FaxCase(
    name="ocr_damaged_potassium",
    description="A potassium of 41.0 -- a lost decimal point, flagged anyway.",
    pages=(
        ("LAKESIDE REFERENCE LABORATORY\nBASIC METABOLIC PANEL\n\n"
         "Patient: Reyes, Mateo     DOB: 11/02/2019\n"
         "Sodium      139  mmol/L\n"
         "Potassium   41.0 mmol/L\n"
         "Creatinine  0.4  mg/dL\n", 0.78),
    ),
    document_type="outside_lab",
    patient_name="Reyes, Mateo", patient_dob="11/02/2019",
    sending_facility="Lakeside Reference Laboratory",
    summary="Basic metabolic panel",
    model_urgency="needs_physician_review",
    model_urgency_reason="laboratory result requiring acknowledgment",
    expect={"queue": "urgent_alert", "decimal_shift": True},
))


def by_name(name: str) -> FaxCase:
    for case in CASES:
        if case.name == name:
            return case
    raise KeyError(name)


def eval_cases(repeats: int = 12) -> list[Any]:
    """A labelled eval set built by tiling the fixtures.

    Tiled, and `EvalResult.blockers` will say so: a corpus of ten hand-written
    faxes repeated twelve times measures the classifier against ten documents,
    not a hundred and twenty. README I-06 evaluates against 300 historical
    faxes, and that is what a real deployment must do.
    """
    from .classify import EvalCase

    cases: list[EvalCase] = []
    for index in range(repeats):
        for fixture in CASES:
            cases.append(
                EvalCase(
                    document_id=f"{fixture.name}__r{index}",
                    text=fixture.text,
                    expected_type=fixture.document_type,
                    note=fixture.description,
                )
            )
    return cases
