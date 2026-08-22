"""What the 271 means, and the one thing this module will never do with it.

README I-09's risk table, the row that shapes this entire file:

    Eligibility response misread -> patient wrongly told they are not covered |
    HIGH | **Never auto-communicate a coverage denial to a patient.** Route to a
    human who calls the payer to confirm before any patient contact.

So `determine()` produces a `Determination` and `Determination.patient_safe` is
False for every outcome that is bad news. Nothing in this package sends anything;
`outreach_draft()` exists and returns a draft for a HUMAN to send, and it raises
outright if asked to draft a message about a denial. The refusal is in the code
rather than in a policy document because the failure mode is a front desk under
pressure copying a status field into a phone call.

WHY THE DETERMINATION IS DETERMINISTIC. The 271 is structured. A model reading
it would add latency and a failure mode to a decision that is a lookup and three
comparisons, and README I-09 says so. The only model calls in I-09 are card
extraction and denial free-text classification, and they live in other files.

FIVE OUTCOMES, and the reason none of them is "probably":

  active            the payer says the coverage is in force on the service date
  inactive          the payer says it is not
  not_found         the payer could not identify the member. NOT the same as
                    inactive, and the difference is the whole point: a
                    mistyped member id and a terminated policy look identical to
                    a front desk and are opposite facts.
  out_of_network    in force, but this practice is not contracted with the plan
  indeterminate     the response could not be trusted, the payer rejected the
                    inquiry, or the answer was self-contradictory

Everything except `active` routes to a person. `indeterminate` is not a failure
of the module; it is the module working.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Mapping, Sequence

from .x12 import BenefitLine, EligibilityRequest, Response271

__all__ = [
    "Outcome",
    "PayerRecord",
    "PayerTable",
    "Determination",
    "determine",
    "outreach_draft",
    "PatientCommunicationRefused",
]


class Outcome:
    ACTIVE = "active"
    INACTIVE = "inactive"
    NOT_FOUND = "not_found"
    OUT_OF_NETWORK = "out_of_network"
    INDETERMINATE = "indeterminate"

    ALL = (ACTIVE, INACTIVE, NOT_FOUND, OUT_OF_NETWORK, INDETERMINATE)
    #: Everything that is not a clean yes goes to a person.
    NEEDS_HUMAN = frozenset({INACTIVE, NOT_FOUND, OUT_OF_NETWORK, INDETERMINATE})


class PatientCommunicationRefused(RuntimeError):
    """Raised when something tries to draft patient-facing text about a denial."""


@dataclass(frozen=True)
class PayerRecord:
    """One contracted plan, with the effective dates that make it answerable.

    README I-09's opening observation is that the practice's published insurance
    list is dated **January 2016**. A payer table without effective dates is a
    list that is wrong at some unknown point in the past, which is what that is.
    """

    payer_id: str
    name: str
    contracted: bool
    effective_from: date
    effective_to: date | None = None
    #: The name the payer uses on the card, which is routinely different from
    #: the name in the contract.
    aliases: tuple[str, ...] = ()
    notes: str = ""

    def in_force(self, on: date) -> bool:
        if on < self.effective_from:
            return False
        return self.effective_to is None or on <= self.effective_to

    def as_dict(self) -> dict[str, Any]:
        return {
            "payer_id": self.payer_id, "name": self.name,
            "contracted": self.contracted,
            "effective_from": self.effective_from.isoformat(),
            "effective_to": (
                self.effective_to.isoformat() if self.effective_to else None
            ),
            "aliases": list(self.aliases), "notes": self.notes,
        }


@dataclass
class PayerTable:
    """The single source of truth for who the practice is contracted with.

    README I-09 target state: "Contracted payer table becomes the single source
    of truth. Public website list generated from it automatically." The
    generation is the point -- a hand-maintained public list and a
    hand-maintained internal list disagree within a quarter, and the public one
    is the one families read before they choose a paediatrician.
    """

    records: dict[str, PayerRecord] = field(default_factory=dict)

    def add(self, record: PayerRecord) -> None:
        if record.payer_id in self.records:
            raise ValueError(f"payer {record.payer_id!r} is already in the table")
        if record.effective_to is not None and record.effective_to < record.effective_from:
            raise ValueError(
                f"payer {record.payer_id!r} ends before it begins; a record with "
                "an inverted window is in force on no date at all and silently "
                "makes every patient on that plan out-of-network"
            )
        self.records[record.payer_id] = record

    def get(self, payer_id: str) -> PayerRecord | None:
        return self.records.get(payer_id)

    def contracted_on(self, payer_id: str, service_date: date) -> bool:
        record = self.records.get(payer_id)
        return bool(record and record.contracted and record.in_force(service_date))

    def public_list(self, on: date) -> list[str]:
        """The website list, generated. Sorted, deduplicated, current."""
        return sorted(
            {
                record.name for record in self.records.values()
                if record.contracted and record.in_force(on)
            }
        )

    def stale_records(self, on: date, *, warn_days: int = 60) -> list[dict[str, Any]]:
        """Contracts ending soon, and any that already ended.

        A contract that lapsed three months ago and is still on the public list
        is how a family arrives believing they are in-network.
        """
        rows: list[dict[str, Any]] = []
        for record in sorted(self.records.values(), key=lambda r: r.payer_id):
            if not record.contracted or record.effective_to is None:
                continue
            days = (record.effective_to - on).days
            if days < 0:
                rows.append({
                    "payer_id": record.payer_id, "name": record.name,
                    "state": "expired", "days": days,
                    "detail": f"contract ended {record.effective_to.isoformat()}",
                })
            elif days <= warn_days:
                rows.append({
                    "payer_id": record.payer_id, "name": record.name,
                    "state": "ending", "days": days,
                    "detail": f"contract ends {record.effective_to.isoformat()}",
                })
        return rows


@dataclass
class Determination:
    """What the practice concluded, and what it is allowed to do with it."""

    outcome: str
    request: EligibilityRequest
    response: Response271 | None
    reason: str
    copay_usd: float | None = None
    coinsurance_percent: float | None = None
    deductible_remaining_usd: float | None = None
    plan_description: str = ""
    in_network: bool | None = None
    checked_on: date | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def needs_human(self) -> bool:
        return self.outcome in Outcome.NEEDS_HUMAN

    @property
    def patient_safe(self) -> bool:
        """True only when this may be repeated to a family without a call first.

        `active` with NO WARNINGS is safe: telling somebody they ARE covered,
        when the payer just said so and nothing about the response was odd,
        carries no risk that a phone call would remove.

        Everything else is a claim about somebody's insurance that could be
        wrong for a reason the front desk cannot see -- a mistyped digit, a
        payer outage, a plan that transferred to a new id last week -- and the
        cost of being wrong is a family told they are uninsured, or told they
        are covered when they are not.

        The warnings clause is load-bearing. An earlier version ignored it, so a
        determination carrying "the payer reports a plan end date before the
        service date" was still `patient_safe` and `outreach_draft` produced
        "we've confirmed your insurance".
        """
        return self.outcome == Outcome.ACTIVE and not self.warnings

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "reason": self.reason,
            "needs_human": self.needs_human,
            "patient_safe": self.patient_safe,
            "copay_usd": self.copay_usd,
            "coinsurance_percent": self.coinsurance_percent,
            "deductible_remaining_usd": self.deductible_remaining_usd,
            "plan_description": self.plan_description,
            "in_network": self.in_network,
            "checked_on": self.checked_on.isoformat() if self.checked_on else None,
            "warnings": list(self.warnings),
            "member_id": self.request.subscriber_member_id,
            "payer": self.request.payer_name,
        }


#: AAA reject codes that mean "we could not find this member", as opposed to
#: "we could not answer you". The distinction decides whether the front desk
#: re-checks the card or calls the payer.
_NOT_FOUND_CODES = frozenset({"67", "71", "72", "73", "75", "64", "65", "58"})


def determine(
    request: EligibilityRequest,
    response: Response271 | None,
    payers: PayerTable,
    *,
    on: date,
) -> Determination:
    """Read the 271. Deterministic; no model runs here.

    The order of the checks is the order of certainty. Anything that makes the
    response untrustworthy is settled before any benefit line is read, because
    a benefit line from a response the parser only half-understood is worse than
    no answer.
    """
    def build(outcome: str, reason: str, **extra: Any) -> Determination:
        return Determination(
            outcome=outcome, request=request, response=response,
            reason=reason, checked_on=on, **extra,
        )

    # 1. No answer at all. A clearinghouse timeout is not a coverage fact.
    if response is None:
        return build(
            Outcome.INDETERMINATE,
            "the payer did not respond. This says nothing about the patient's "
            "coverage and must not be treated as though it did.",
        )

    # 2. The payer told us it could not answer.
    if response.rejections:
        codes = {r["code"] for r in response.rejections}
        detail = "; ".join(
            f"{r['code']} {r['reason']}" for r in response.rejections[:3]
        )
        if codes & _NOT_FOUND_CODES:
            return build(
                Outcome.NOT_FOUND,
                f"the payer could not identify this member ({detail}). This is "
                "NOT a statement that the patient is uninsured -- the commonest "
                "cause is a mistyped member id or a date of birth that does not "
                "match the payer's record. Re-check the card, then call.",
            )
        return build(
            Outcome.INDETERMINATE,
            f"the payer rejected the inquiry ({detail}).",
        )

    # 3. We could not fully read what it sent.
    if not response.trustworthy:
        return build(
            Outcome.INDETERMINATE,
            f"{len(response.unparsed)} segment(s) of this 271 were not "
            "understood by the parser, so its benefit lines cannot be relied on. "
            "A partial read of an eligibility response is how a covered child "
            "gets turned away.",
            warnings=[f"unparsed: {seg[:60]}" for seg in response.unparsed[:5]],
        )

    # 4. Contradiction. A payer that says both is a payer to phone.
    if response.has_active_benefit and response.has_inactive_benefit:
        return build(
            Outcome.INDETERMINATE,
            "this 271 reports both active and inactive benefits. That is usually "
            "a plan change mid-period or two policies on one member id, and it "
            "needs a person to disentangle before anybody is billed.",
        )

    # 5. Plainly inactive.
    if response.has_inactive_benefit and not response.has_active_benefit:
        ended = (
            f" Coverage ended {response.plan_ends.isoformat()}."
            if response.plan_ends else ""
        )
        return build(
            Outcome.INACTIVE,
            f"the payer reports this coverage inactive on {on.isoformat()}."
            + ended
            + " Confirm with the payer BEFORE contacting the family: a wrong "
            "'you are not covered' is the most damaging thing this system can say.",
        )

    if not response.has_active_benefit:
        return build(
            Outcome.INDETERMINATE,
            "the 271 carried no active or inactive benefit line, so the payer "
            "has not actually answered the question.",
        )

    # 6. The payer said "active" AND stated a plan window that excludes the
    #    service date. An earlier version appended this to `warnings` and
    #    carried on, so a 271 with EB*1 and a plan end date of 2026-05-31 came
    #    out `active`, `patient_safe`, and the family was sent a message saying
    #    their insurance was confirmed. The claim then denied CARC 27 and the
    #    balance went to them.
    if response.plan_ends is not None and response.plan_ends < on:
        return build(
            Outcome.INACTIVE,
            f"the payer reports an active benefit AND a plan end date of "
            f"{response.plan_ends.isoformat()}, which is before the service "
            f"date {on.isoformat()}. A stale benefit line against a stated end "
            "date is a terminated policy. Confirm with the payer BEFORE "
            "contacting the family.",
        )
    if response.plan_begins is not None and response.plan_begins > on:
        return build(
            Outcome.INACTIVE,
            f"the payer reports an active benefit AND a plan start date of "
            f"{response.plan_begins.isoformat()}, which is after the service "
            f"date {on.isoformat()}. The coverage has not begun.",
        )
    notes = list(response.messages) + [
        b.message for b in response.benefits if b.message.strip()
    ]
    for message in notes:
        # A payer's free-text limitation on an otherwise-active response is not
        # decoration. "COVERAGE TERMINATED 05/31/2026 - VERIFY BEFORE SERVICE"
        # arrives as MSG, and it means what it says.
        return build(
            Outcome.INDETERMINATE,
            f"the payer attached a free-text note to this response: "
            f"{message!r}. Nothing here reads payer prose, so a person must.",
        )

    # 7. Active on the payer's side. Now: are WE contracted with them?
    copay = _amount(response, "A")
    coinsurance = _percent(response, "B")
    deductible = _amount(response, "C")
    plan = next(
        (b.plan_description for b in response.benefits if b.plan_description), ""
    )
    network = _network(response)
    contracted = payers.contracted_on(request.payer_id, on)
    warnings: list[str] = []
    if network is None:
        warnings.append(
            "the payer did not state a network indicator, so in-network status "
            "is unknown rather than confirmed"
        )
    if payers.get(request.payer_id) is None:
        warnings.append(
            f"payer {request.payer_id!r} is not in the payer table at all, so "
            "network status is unknown rather than out-of-network"
        )

    if not contracted:
        record = payers.get(request.payer_id)
        why = (
            "the practice has no contract with this payer"
            if record is None or not record.contracted
            else f"the contract with this payer was not in force on {on.isoformat()}"
        )
        return build(
            Outcome.OUT_OF_NETWORK,
            f"coverage is active, but {why}. README I-09: call the family BEFORE "
            "the visit. An out-of-network surprise after the fact is the bill "
            "that ends the relationship.",
            copay_usd=copay, coinsurance_percent=coinsurance,
            deductible_remaining_usd=deductible, plan_description=plan,
            in_network=False, warnings=warnings,
        )

    return build(
        Outcome.ACTIVE,
        f"active on {on.isoformat()}"
        + (f", copay ${copay:,.2f}" if copay is not None else "")
        + (f", deductible remaining ${deductible:,.2f}" if deductible else ""),
        copay_usd=copay, coinsurance_percent=coinsurance,
        deductible_remaining_usd=deductible, plan_description=plan,
        # None, not True. "The payer did not say" is not "in network", and
        # defaulting it widened a threshold on missing data.
        in_network=network, warnings=warnings,
    )


def _amount(response: Response271, eb_code: str) -> float | None:
    """The amount for one benefit kind, preferring an in-network line.

    Preferring rather than requiring: a payer that does not state a network
    indicator has not said the amount is out-of-network, and refusing to report
    a copay because the segment was terse would send every patient to the desk
    with no number.
    """
    lines = [b for b in response.benefits if b.eb_code == eb_code and b.amount is not None]
    if not lines:
        return None
    in_network = [b for b in lines if b.network_indicator == "Y"]
    return (in_network or lines)[0].amount


def _percent(response: Response271, eb_code: str) -> float | None:
    """The same in-network preference `_amount` applies.

    Without it, a payer sending an in-network copay line and an out-of-network
    coinsurance line produced the right copay and the WRONG coinsurance, on a
    determination the front desk would read out to a family.
    """
    lines = [
        b for b in response.benefits if b.eb_code == eb_code and b.percent is not None
    ]
    if not lines:
        return None
    in_network = [b for b in lines if b.network_indicator == "Y"]
    return (in_network or lines)[0].percent


def _network(response: Response271) -> bool | None:
    indicators = {
        b.network_indicator for b in response.benefits if b.network_indicator
    }
    if not indicators:
        return None
    if indicators == {"Y"}:
        return True
    if indicators == {"N"}:
        return False
    return None


#: What a person may be handed to send. Deliberately short, and deliberately
#: about logistics rather than about coverage.
_OUTREACH_TEMPLATES: Mapping[str, str] = {
    Outcome.ACTIVE: (
        "Hi {family}, we've confirmed your insurance for {child}'s visit on "
        "{service_date}. Your copay will be ${copay}. See you then."
    ),
}


def outreach_draft(
    determination: Determination,
    *,
    family_name: str,
    child_first_name: str,
) -> str:
    """A draft for a person to send. Refuses to draft anything about a denial.

    This function is the mechanical form of README I-09's control. There is no
    `force=True`, no template for the inactive case, and no way to reach one:
    the refusal is that the templates for bad news do not exist.

    A family being told they have no insurance needs to hear it from somebody
    who has phoned the payer and can say what to do next. Automating it saves a
    minute and costs the relationship, and roughly one time in ten it is wrong
    because a digit was mistyped.
    """
    if not determination.patient_safe:
        raise PatientCommunicationRefused(
            f"this determination is {determination.outcome!r} and must not be "
            "communicated to a family by this system. README I-09: never "
            "auto-communicate a coverage denial to a patient; route it to a "
            "human who calls the payer to confirm first. "
            f"Reason on file: {determination.reason}"
        )
    template = _OUTREACH_TEMPLATES[determination.outcome]
    return template.format(
        family=family_name,
        child=child_first_name,
        service_date=determination.request.service_date.strftime("%B %-d"),
        copay=(
            f"{determination.copay_usd:,.2f}"
            if determination.copay_usd is not None else "confirmed at the desk"
        ),
    )
