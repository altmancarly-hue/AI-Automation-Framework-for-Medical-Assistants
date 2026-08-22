"""Synthetic patients, 271 responses and denials for I-09. No PHI, no real payer.

Seven eligibility cases, one per outcome plus the two that are commonly
confused:

    rosa_active           covered, in network, $25 copay. The quiet case.
    theo_terminated       the plan ended in June. The family does not know.
    nadia_not_found       a mistyped member id. Reads exactly like "uninsured"
                          to a front desk, and is a completely different fact.
    omar_out_of_network   covered, and the practice has no contract. README
                          I-09: call BEFORE the visit.
    ivy_unparseable       the payer used a segment this subset parser does not
                          know. The response is not trusted at all.
    lucas_contradictory   active and inactive benefits in one response.
    priya_no_response     the clearinghouse timed out.

and five denials spanning the CARC table, an unmapped code, and a remark whose
"evidence" span is not in the remark at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Mapping, Sequence

from .coverage import PayerRecord, PayerTable
from .denials import Denial
from .x12 import EligibilityRequest

__all__ = [
    "NOW",
    "SERVICE_DATE",
    "CASES",
    "DENIALS",
    "build_payer_table",
    "by_name",
    "card_response",
    "denial_response",
]

NOW = datetime(2026, 8, 24, 7, 0, tzinfo=timezone.utc)
#: T-3 days, which is when README I-09 says the batch runs.
SERVICE_DATE = date(2026, 8, 27)


def build_payer_table() -> PayerTable:
    """The contracted-payer table, with the effective dates that make it usable.

    Includes one contract that lapsed in May and is still on the public list,
    which is the failure README I-09 opens with: a published insurance list
    dated January 2016.
    """
    table = PayerTable()
    for record in (
        PayerRecord("BCBSIL", "Blue Cross Blue Shield of Illinois", True,
                    date(2015, 1, 1), aliases=("BCBS IL", "Blue Cross IL")),
        PayerRecord("AETNA", "Aetna", True, date(2018, 7, 1)),
        PayerRecord("UHC", "UnitedHealthcare", True, date(2016, 1, 1)),
        PayerRecord("CIGNA", "Cigna", True, date(2019, 3, 1),
                    effective_to=date(2026, 10, 1),
                    notes="renewal under negotiation"),
        PayerRecord("HUMANA", "Humana", True, date(2017, 1, 1),
                    effective_to=date(2026, 5, 31),
                    notes="CONTRACT LAPSED - still on the public website list"),
        PayerRecord("OSCAR", "Oscar Health", False, date(2020, 1, 1),
                    notes="never contracted; out-of-network"),
    ):
        table.add(record)
    return table


@dataclass(frozen=True)
class EligibilityCase:
    name: str
    description: str
    request: EligibilityRequest
    #: None means the clearinghouse did not answer at all.
    response_271: str | None
    family_name: str = ""
    child_first_name: str = ""
    expect_outcome: str = ""


def _request(
    *, member_id: str, last: str, sub_first: str, sub_dob: date,
    payer_id: str, payer_name: str, child_first: str, child_dob: date,
    trace: str,
) -> EligibilityRequest:
    return EligibilityRequest(
        subscriber_member_id=member_id,
        subscriber_last_name=last,
        subscriber_first_name=sub_first,
        subscriber_dob=sub_dob,
        payer_id=payer_id,
        payer_name=payer_name,
        provider_npi="1902884416",
        provider_name="North Suburban Pediatrics",
        service_date=SERVICE_DATE,
        dependent_last_name=last,
        dependent_first_name=child_first,
        dependent_dob=child_dob,
        trace_number=trace,
    )


def _seg(*segments: str) -> str:
    return "~".join(segments) + "~"


CASES: tuple[EligibilityCase, ...] = (
    EligibilityCase(
        name="rosa_active",
        description="Covered, in network, $25 copay. The case that stays quiet.",
        request=_request(
            member_id="W9928311402", last="Alvarez", sub_first="Marisol",
            sub_dob=date(1989, 4, 2), payer_id="BCBSIL",
            payer_name="Blue Cross Blue Shield of Illinois",
            child_first="Rosa", child_dob=date(2018, 3, 14), trace="NSP0001",
        ),
        response_271=_seg(
            "ST*271*0001*005010X279A1",
            "NM1*PR*2*Blue Cross Blue Shield of Illinois*****PI*BCBSIL",
            "NM1*IL*1*Alvarez*Marisol****MI*W9928311402",
            "NM1*03*1*Alvarez*Rosa",
            "DMG*D8*20180314",
            "DTP*346*D8*20260101",
            "EB*1*IND*30**GOLD PPO 500",
            "EB*A*IND*98**GOLD PPO 500**25*****Y",
            "EB*C*IND*30**GOLD PPO 500**450",
        ),
        family_name="the Alvarez family", child_first_name="Rosa",
        expect_outcome="active",
    ),
    EligibilityCase(
        name="theo_terminated",
        description="The plan ended in June. The family does not know yet.",
        request=_request(
            member_id="A4471220098", last="Nakamura", sub_first="Kenji",
            sub_dob=date(1986, 11, 30), payer_id="AETNA", payer_name="Aetna",
            child_first="Theo", child_dob=date(2017, 11, 2), trace="NSP0002",
        ),
        response_271=_seg(
            "ST*271*0002*005010X279A1",
            "NM1*PR*2*Aetna*****PI*AETNA",
            "NM1*IL*1*Nakamura*Kenji****MI*A4471220098",
            "NM1*03*1*Nakamura*Theo",
            "DTP*347*D8*20260630",
            "EB*6*IND*30**AETNA CHOICE POS II",
        ),
        family_name="the Nakamura family", child_first_name="Theo",
        expect_outcome="inactive",
    ),
    EligibilityCase(
        name="nadia_not_found",
        description=(
            "A mistyped member id. Reads exactly like 'uninsured' to a front "
            "desk, and is an entirely different fact."
        ),
        request=_request(
            member_id="U110029384X", last="Okonkwo", sub_first="Chidi",
            sub_dob=date(1990, 2, 14), payer_id="UHC",
            payer_name="UnitedHealthcare",
            child_first="Nadia", child_dob=date(2016, 6, 9), trace="NSP0003",
        ),
        response_271=_seg(
            "ST*271*0003*005010X279A1",
            "NM1*PR*2*UnitedHealthcare*****PI*UHC",
            "NM1*IL*1*Okonkwo*Chidi****MI*U110029384X",
            "AAA*N**75*C",
        ),
        family_name="the Okonkwo family", child_first_name="Nadia",
        expect_outcome="not_found",
    ),
    EligibilityCase(
        name="omar_out_of_network",
        description="Covered, and the practice has no contract with this payer.",
        request=_request(
            member_id="OSC88120043", last="Haddad", sub_first="Layla",
            sub_dob=date(1988, 8, 19), payer_id="OSCAR",
            payer_name="Oscar Health",
            child_first="Omar", child_dob=date(2011, 2, 18), trace="NSP0004",
        ),
        response_271=_seg(
            "ST*271*0004*005010X279A1",
            "NM1*PR*2*Oscar Health*****PI*OSCAR",
            "NM1*IL*1*Haddad*Layla****MI*OSC88120043",
            "NM1*03*1*Haddad*Omar",
            "EB*1*IND*30**OSCAR SIMPLE BRONZE",
            "EB*A*IND*98**OSCAR SIMPLE BRONZE**60*****N",
        ),
        family_name="the Haddad family", child_first_name="Omar",
        expect_outcome="out_of_network",
    ),
    EligibilityCase(
        name="ivy_unparseable",
        description=(
            "The payer used a segment this subset parser has never seen. The "
            "whole response is untrusted rather than half-read."
        ),
        request=_request(
            member_id="C7719920011", last="Petrov", sub_first="Anya",
            sub_dob=date(1991, 5, 5), payer_id="CIGNA", payer_name="Cigna",
            child_first="Ivy", child_dob=date(2015, 9, 30), trace="NSP0005",
        ),
        response_271=_seg(
            "ST*271*0005*005010X279A1",
            "NM1*PR*2*Cigna*****PI*CIGNA",
            "NM1*IL*1*Petrov*Anya****MI*C7719920011",
            "EB*1*IND*30**CIGNA CONNECT",
            # A segment nobody here implements, carrying who-knows-what.
            "ZZ*CUSTOM*PLAN-TIER-CHANGE*20260801*SEE-PORTAL",
        ),
        family_name="the Petrov family", child_first_name="Ivy",
        expect_outcome="indeterminate",
    ),
    EligibilityCase(
        name="lucas_contradictory",
        description="Active and inactive benefits in one response.",
        request=_request(
            member_id="B2210077731", last="Byrne", sub_first="Sean",
            sub_dob=date(1985, 1, 22), payer_id="BCBSIL",
            payer_name="Blue Cross Blue Shield of Illinois",
            child_first="Lucas", child_dob=date(2014, 4, 22), trace="NSP0006",
        ),
        response_271=_seg(
            "ST*271*0006*005010X279A1",
            "NM1*PR*2*Blue Cross Blue Shield of Illinois*****PI*BCBSIL",
            "NM1*IL*1*Byrne*Sean****MI*B2210077731",
            "EB*1*IND*30**PPO BRONZE",
            "EB*6*IND*98**PPO BRONZE",
        ),
        family_name="the Byrne family", child_first_name="Lucas",
        expect_outcome="indeterminate",
    ),
    EligibilityCase(
        name="priya_no_response",
        description="The clearinghouse timed out. This is not a coverage fact.",
        request=_request(
            member_id="H5540098812", last="Raman", sub_first="Meera",
            sub_dob=date(1987, 7, 7), payer_id="HUMANA", payer_name="Humana",
            child_first="Priya", child_dob=date(2019, 12, 1), trace="NSP0007",
        ),
        response_271=None,
        family_name="the Raman family", child_first_name="Priya",
        expect_outcome="indeterminate",
    ),
)


def by_name(name: str) -> EligibilityCase:
    for case in CASES:
        if case.name == name:
            return case
    raise KeyError(name)


DENIALS: tuple[Denial, ...] = (
    Denial(
        claim_id="CLM-88201", patient_id="p_theo", payer_id="AETNA",
        service_date=date(2026, 7, 8), denied_on=date(2026, 7, 29),
        amount_usd=214.00, carc_code="27", cpt_code="99392",
        remark_text="Expenses incurred after coverage terminated on 06/30/2026.",
    ),
    Denial(
        claim_id="CLM-88214", patient_id="p_rosa", payer_id="BCBSIL",
        service_date=date(2026, 7, 11), denied_on=date(2026, 8, 1),
        amount_usd=88.00, carc_code="11", cpt_code="90686",
        remark_text="The diagnosis is inconsistent with the procedure billed.",
    ),
    Denial(
        claim_id="CLM-88230", patient_id="p_omar", payer_id="UHC",
        service_date=date(2026, 6, 2), denied_on=date(2026, 8, 12),
        amount_usd=340.00, carc_code="29", cpt_code="99213",
        remark_text="Claim received after the 90 day timely filing limit.",
    ),
    Denial(
        # No CARC mapping. The model's reading is what decides, if it can.
        claim_id="CLM-88245", patient_id="p_nadia", payer_id="UHC",
        service_date=date(2026, 7, 20), denied_on=date(2026, 8, 14),
        amount_usd=126.50, carc_code="B7", cpt_code="99391",
        remark_text=(
            "Member was not enrolled in this plan on the date of service; "
            "coverage moved to a different group effective 07/01/2026."
        ),
    ),
    Denial(
        # A remark the model will read as eligibility, with an evidence span it
        # cannot actually quote. The grounding post-condition catches it.
        claim_id="CLM-88251", patient_id="p_ivy", payer_id="CIGNA",
        service_date=date(2026, 7, 25), denied_on=date(2026, 8, 18),
        amount_usd=402.00, carc_code="ZQ", cpt_code="99394",
        remark_text="See attached explanation of benefits.",
    ),
)


def denial_response(root_cause: str, confidence: float, evidence: str) -> str:
    return json.dumps(
        {
            "root_cause": root_cause,
            "confidence": confidence,
            "evidence": evidence,
            "reasoning": "as stated in the remark",
        }
    )


def card_response(
    *,
    payer_name: str | None = "Blue Cross Blue Shield of Illinois",
    member_id: str | None = "W9928311402",
    group_number: str | None = "GRP0084412",
    member_confidence: float = 0.93,
) -> str:
    return json.dumps(
        {
            "payer_name": payer_name,
            "plan_name": "Gold PPO 500",
            "member_id": member_id,
            "group_number": group_number,
            "subscriber_name": "MARISOL ALVAREZ",
            "rx_bin": "011552",
            "rx_pcn": "IL",
            "payer_phone": "1-800-892-2803",
            "confidence": {
                "payer_name": 0.99,
                "plan_name": 0.95,
                "member_id": member_confidence,
                "group_number": 0.88,
                "subscriber_name": 0.97,
            },
            "unreadable": "",
        }
    )
