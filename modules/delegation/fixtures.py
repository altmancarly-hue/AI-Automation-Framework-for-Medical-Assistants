"""Synthetic staff, orders and competencies for I-10. No real people.

The practice as README I-10 describes it before this initiative: experienced
MAs who know what they may do because they have done it for years, one new hire
who does not, a CPR card that lapsed last week, and a standing order that has
not been reviewed since the schedule changed.

    ma_jess     three years in post. Competent at everything, current.
    ma_dana     new hire, six weeks. Competent at vitals only.
    lpn_marta   LPN. Competent at vaccines; her CPR card expired on the 18th.
    rn_paulo    RN. Both a licensed role and a delegable role -- the case that
                lets somebody supervise themselves if nothing stops it.
    dr_alvarez  physician, delegating, on site most days.
    dr_osei     physician, delegating, on site Tuesdays.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from .register import (
    Competency,
    CompetencyRecord,
    CompetencyRegister,
    DelegationRules,
    OrderRegister,
    Roster,
    RosterEntry,
    StaffMember,
    StandingOrder,
)

__all__ = [
    "NOW",
    "STAFF",
    "COMPETENCIES",
    "build",
]

#: A Monday, mid-morning. Dr Alvarez is in; Dr Osei is not.
NOW = datetime(2026, 8, 24, 10, 30, tzinfo=timezone.utc)

STAFF: tuple[StaffMember, ...] = (
    StaffMember("ma_jess", "Jess Alvarado", "medical_assistant",
                started_on=date(2023, 6, 5)),
    StaffMember("ma_dana", "Dana Whitfield", "medical_assistant",
                started_on=date(2026, 7, 13)),
    StaffMember("lpn_marta", "Marta Silva", "licensed_practical_nurse",
                licence_number="043-556120", started_on=date(2021, 2, 1)),
    StaffMember("rn_paulo", "Paulo Reyes", "registered_nurse",
                licence_number="041-889201", started_on=date(2022, 9, 12)),
    StaffMember("dr_alvarez", "Dr Ines Alvarez", "physician",
                licence_number="036-119284", started_on=date(2011, 1, 4)),
    StaffMember("dr_osei", "Dr Kwame Osei", "physician",
                licence_number="036-224417", started_on=date(2016, 8, 15)),
)

COMPETENCIES: tuple[Competency, ...] = (
    Competency("bls_cpr", "BLS / CPR",
               "Current American Heart Association BLS certification"),
    Competency("im_injection", "Intramuscular injection technique",
               "Observed demonstration of IM injection in a paediatric patient"),
    Competency("vaccine_storage", "Vaccine storage and handling",
               "CDC storage and handling training, current toolkit"),
    Competency("vitals_peds", "Paediatric vital signs",
               "Age-appropriate measurement of HR, RR, BP and temperature"),
    Competency("poct_strep", "CLIA-waived rapid strep testing",
               "Competency in specimen collection and CLIA-waived testing"),
    Competency("vision_hearing", "Vision and hearing screening",
               "Age-appropriate screening technique and referral criteria"),
)


def build(*, rules: DelegationRules | None = None):
    """The whole register, populated. Returns `(rules, orders, comps, roster)`."""
    rules = rules or DelegationRules.load()

    competencies = CompetencyRegister(rules=rules)
    for member in STAFF:
        competencies.add_staff(member)
    for competency in COMPETENCIES:
        competencies.define(competency)

    signed = datetime(2026, 1, 6, 15, 0, tzinfo=timezone.utc)

    # -- competency records ------------------------------------------------
    records = (
        # Jess: everything, all current.
        CompetencyRecord("cr_001", "ma_jess", "bls_cpr", "dr_alvarez",
                         date(2025, 9, 2), "external_credential",
                         "AHA-88121", date(2027, 9, 2)),
        CompetencyRecord("cr_002", "ma_jess", "im_injection", "dr_alvarez",
                         date(2024, 3, 11), "observed_demonstration",
                         "obs-2024-03-11", date(2027, 3, 11)),
        CompetencyRecord("cr_003", "ma_jess", "vaccine_storage", "dr_alvarez",
                         date(2026, 1, 20), "training_certificate",
                         "cdc-toolkit-2026", date(2027, 1, 20)),
        CompetencyRecord("cr_004", "ma_jess", "vitals_peds", "dr_alvarez",
                         date(2023, 6, 20), "return_demonstration",
                         "ret-2023-06-20", None),
        CompetencyRecord("cr_005", "ma_jess", "poct_strep", "rn_paulo",
                         date(2025, 11, 4), "written_examination",
                         "exam-2025-11", date(2027, 11, 4)),
        CompetencyRecord("cr_006", "ma_jess", "vision_hearing", "dr_alvarez",
                         date(2024, 8, 1), "observed_demonstration",
                         "obs-2024-08-01", date(2027, 8, 1)),
        # Dana: six weeks in. Vitals only, which is the correct state and the
        # one an apprenticeship model gets wrong.
        CompetencyRecord("cr_007", "ma_dana", "vitals_peds", "rn_paulo",
                         date(2026, 7, 24), "return_demonstration",
                         "ret-2026-07-24", None),
        # Marta: vaccines yes, CPR card lapsed on the 18th.
        CompetencyRecord("cr_008", "lpn_marta", "im_injection", "dr_osei",
                         date(2023, 5, 9), "observed_demonstration",
                         "obs-2023-05-09", date(2027, 5, 9)),
        CompetencyRecord("cr_009", "lpn_marta", "vaccine_storage", "dr_osei",
                         date(2025, 6, 2), "training_certificate",
                         "cdc-toolkit-2025", date(2027, 6, 2)),
        CompetencyRecord("cr_010", "lpn_marta", "bls_cpr", "dr_osei",
                         date(2024, 8, 18), "external_credential",
                         "AHA-70233", date(2026, 8, 18)),
        CompetencyRecord("cr_011", "lpn_marta", "vitals_peds", "dr_osei",
                         date(2021, 3, 1), "return_demonstration",
                         "ret-2021-03-01", None),
        # Paulo: an RN who is also delegable.
        CompetencyRecord("cr_012", "rn_paulo", "bls_cpr", "dr_alvarez",
                         date(2025, 4, 14), "external_credential",
                         "AHA-55019", date(2027, 4, 14)),
        CompetencyRecord("cr_013", "rn_paulo", "im_injection", "dr_alvarez",
                         date(2022, 10, 3), "observed_demonstration",
                         "obs-2022-10-03", None),
        CompetencyRecord("cr_014", "rn_paulo", "vaccine_storage", "dr_alvarez",
                         date(2026, 2, 11), "training_certificate",
                         "cdc-toolkit-2026", date(2027, 2, 11)),
    )
    for record in records:
        competencies.verify(record)

    # -- standing orders ---------------------------------------------------
    orders = OrderRegister(rules=rules)
    orders.publish(
        StandingOrder(
            order_id="so_immunize", version=2,
            title="Administer routine childhood immunizations per ACIP schedule",
            task_code="administer_vaccine",
            clinical_content=(
                "Administer vaccines due per the current ACIP schedule as "
                "identified by the immunization forecaster, after confirming "
                "contraindications and providing the current VIS."
            ),
            delegating_physician_id="dr_alvarez",
            effective_from=date(2026, 1, 6),
            required_competencies=("im_injection", "vaccine_storage", "bls_cpr"),
            required_supervision_role="physician",
            source_guideline="ACIP child and adolescent schedule, 2026",
            review_due=date(2027, 1, 6),
            signed_by="dr_alvarez", signed_utc=signed,
        )
    )
    orders.publish(
        StandingOrder(
            order_id="so_vitals", version=1,
            title="Obtain vital signs and growth measurements at every visit",
            task_code="obtain_vitals",
            clinical_content=(
                "Measure and record height, weight, head circumference where "
                "age-appropriate, blood pressure from age 3, heart rate, "
                "respiratory rate and temperature."
            ),
            delegating_physician_id="dr_alvarez",
            effective_from=date(2025, 4, 1),
            required_competencies=("vitals_peds",),
            required_supervision_role="physician",
            review_due=date(2026, 4, 1),   # OVERDUE. Non-blocking, reported.
            signed_by="dr_alvarez", signed_utc=datetime(
                2025, 3, 28, 12, 0, tzinfo=timezone.utc
            ),
        )
    )
    orders.publish(
        StandingOrder(
            order_id="so_strep", version=1,
            title="Rapid strep testing for sore throat by protocol",
            task_code="poct_strep",
            clinical_content=(
                "Collect a throat swab and perform CLIA-waived rapid antigen "
                "testing for patients meeting the protocol criteria."
            ),
            delegating_physician_id="dr_osei",
            effective_from=date(2026, 2, 2),
            required_competencies=("poct_strep",),
            required_supervision_role="physician",
            source_guideline="IDSA group A streptococcal pharyngitis, 2012",
            review_due=date(2027, 2, 2),
            signed_by="dr_osei",
            signed_utc=datetime(2026, 1, 30, 9, 0, tzinfo=timezone.utc),
        )
    )
    orders.publish(
        StandingOrder(
            order_id="so_screening", version=1,
            title="Vision and hearing screening at Bright Futures intervals",
            task_code="vision_hearing_screen",
            clinical_content=(
                "Perform age-appropriate vision and hearing screening at the "
                "intervals in the periodicity schedule; refer per criteria."
            ),
            delegating_physician_id="dr_alvarez",
            effective_from=date(2025, 9, 1),
            required_competencies=("vision_hearing",),
            required_supervision_role="physician",
            review_due=date(2027, 9, 1),
            signed_by="dr_alvarez",
            signed_utc=datetime(2025, 8, 25, 14, 0, tzinfo=timezone.utc),
        )
    )

    # -- the roster --------------------------------------------------------
    roster = Roster(rules=rules, staff={m.staff_id: m for m in STAFF})
    day = NOW.date()

    def shift(staff_id: str, start_h: int, end_h: int, on: date = day) -> None:
        roster.record(
            RosterEntry(
                staff_id,
                datetime.combine(on, datetime.min.time()).replace(
                    hour=start_h, tzinfo=timezone.utc
                ),
                datetime.combine(on, datetime.min.time()).replace(
                    hour=end_h, tzinfo=timezone.utc
                ),
            )
        )

    # Dr Alvarez is in, except between 12:00 and 13:00.
    shift("dr_alvarez", 8, 12)
    shift("dr_alvarez", 13, 18)
    # Dr Osei is not in today at all. Note what that does NOT do: `so_strep`,
    # which he signed, is still executable, because 225 ILCS 60/54.2 requires
    # that "a licensed health care professional is on site" -- not that the
    # DELEGATING physician is. The delegation is Dr Osei's act, made when he
    # signed; the on-site requirement is satisfied by Dr Alvarez. Conflating the
    # two would stop a practice functioning on any day its signer is away, which
    # is not what the statute says.
    shift("rn_paulo", 8, 17)
    shift("ma_jess", 8, 17)
    shift("ma_dana", 9, 17)
    shift("lpn_marta", 8, 16)

    return rules, orders, competencies, roster
