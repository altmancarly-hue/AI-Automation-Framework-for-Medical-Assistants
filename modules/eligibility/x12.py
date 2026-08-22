"""X12 270 request and 271 response. The EDI, kept boring on purpose.

README I-09 is unambiguous about where this sits:

    Check eligibility before a visit | X12 270/271 real-time eligibility
    transaction. Standardized, decades old, supported by every clearinghouse.
    Deterministic.

    Parse the 271 response | Deterministic parser. The format is specified.

and the build plan adds: *"use a library, do not hand-roll the EDI"*.

WHY THERE IS A PARSER HERE ANYWAY, and what it is and is not for.

The right production answer is a maintained X12 library behind
`X12Parser` -- the protocol at the bottom of this file -- because the standard
is large, the implementation guide is 200 pages, and every clearinghouse has
quirks. What this module ships is a **strict subset parser** that handles the
segments a pediatric practice's eligibility check actually turns on, and
**refuses everything else** rather than guessing.

That refusal is the whole design. A hand-rolled EDI parser fails in one of two
ways: it silently misreads a segment it half-understands, or it throws. The
first produces a confident wrong answer about whether a child is covered, which
this module treats as the worst outcome available to it. So:

  * every segment it does not know is COLLECTED, not skipped, and
    `Response271.unparsed` is non-empty whenever that happened
  * `Response271.trustworthy` is False if anything was unparsed, and the
    coverage layer refuses to act on an untrustworthy response
  * a benefit it cannot classify becomes `unknown`, never `active`

The result is a parser that answers correctly or says it could not, which is
what you want from a subset. Swap in a real library by implementing `X12Parser`;
everything downstream is unchanged.

NOTHING HERE CALLS A MODEL. An LLM in an EDI parser would be slower, costlier
and less reliable than a specification, and README I-09 says exactly that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Protocol, Sequence

__all__ = [
    "SEGMENT_TERMINATOR",
    "ELEMENT_SEPARATOR",
    "EligibilityRequest",
    "BenefitLine",
    "Response271",
    "X12Parser",
    "SubsetParser",
    "build_270",
    "MalformedEDI",
    "SERVICE_TYPE_CODES",
    "EB_CODES",
]

SEGMENT_TERMINATOR = "~"
ELEMENT_SEPARATOR = "*"
COMPONENT_SEPARATOR = ":"
#: EB03 is a REPEATING simple element in 005010X279A1. A payer answering one
#: EB segment for three service types sends `30^98^68`, and splitting only on
#: the component separator produced a "service type" of `'30^98^68'` that
#: matched nothing -- `benefits_for("98")` returned nothing while
#: `trustworthy` stayed True.
REPETITION_SEPARATOR = "^"

#: The service type codes a pediatric practice asks about. `30` is the general
#: "health benefit plan coverage" ask that every payer answers.
SERVICE_TYPE_CODES: Mapping[str, str] = {
    "30": "Health benefit plan coverage",
    "98": "Professional (physician) visit - office",
    "96": "Professional (physician) visit - inpatient",
    "68": "Well baby care",
    "62": "MRI/CAT scan",
    "AL": "Vision (optometry)",
    "35": "Dental care",
    "88": "Pharmacy",
}

#: EB01, the eligibility-or-benefit code. Only the values this subset claims to
#: understand. Anything else lands in `unparsed` and makes the response
#: untrustworthy, which is the point.
EB_CODES: Mapping[str, str] = {
    "1": "active",
    "2": "active_pending_investigation",
    "3": "active_full_risk_capitation",
    "4": "active_services_capitated",
    "5": "active_services_capitated_primary_care",
    "6": "inactive",
    "7": "inactive_pending_eligibility_update",
    "8": "inactive_pending_investigation",
    "A": "copay",
    "B": "coinsurance",
    "C": "deductible",
    "G": "out_of_pocket_stop_loss",
    "I": "non_covered",
    "L": "primary_care_provider",
    "R": "other_or_additional_payor",
    "U": "contact_following_entity",
    "V": "cannot_process",
    "Y": "spend_down",
}

#: EB01 values that mean the member has coverage right now.
_ACTIVE = frozenset({"1", "2", "3", "4", "5"})
#: EB01 values that mean they do not.
_INACTIVE = frozenset({"6", "7", "8"})


class MalformedEDI(ValueError):
    """Raised when a transaction cannot be read as X12 at all."""


@dataclass(frozen=True)
class EligibilityRequest:
    """Everything a 270 needs. Deliberately not a patient record."""

    subscriber_member_id: str
    subscriber_last_name: str
    subscriber_first_name: str
    subscriber_dob: date
    payer_id: str
    payer_name: str
    provider_npi: str
    provider_name: str
    service_date: date
    #: The dependent, when the patient is not the subscriber. In paediatrics
    #: this is the normal case: the child is a dependent on a parent's plan.
    dependent_last_name: str = ""
    dependent_first_name: str = ""
    dependent_dob: date | None = None
    service_type_codes: tuple[str, ...] = ("30", "98")
    trace_number: str = ""

    @property
    def is_dependent(self) -> bool:
        return bool(self.dependent_last_name and self.dependent_dob)

    def as_dict(self) -> dict[str, Any]:
        return {
            "member_id": self.subscriber_member_id,
            "payer_id": self.payer_id,
            "payer_name": self.payer_name,
            "provider_npi": self.provider_npi,
            "service_date": self.service_date.isoformat(),
            "is_dependent": self.is_dependent,
            "service_type_codes": list(self.service_type_codes),
            "trace_number": self.trace_number,
        }


def build_270(
    request: EligibilityRequest, *, control_number: str, created: datetime
) -> str:
    """Assemble a 270 inquiry.

    Kept to the segments a real-time eligibility inquiry requires. The interchange
    envelope (ISA/GS) is the clearinghouse's business and differs per contract,
    so this produces the transaction set and the adapter wraps it.

    WHY IT VALIDATES BEFORE IT BUILDS. A 270 with a missing NPI comes back as a
    rejection two seconds later and somebody re-keys it. A 270 with a subtly
    wrong date of birth comes back as "not found", which reads exactly like "not
    covered" -- and the front desk tells a family they are uninsured.
    """
    for name, value in (
        ("subscriber_member_id", request.subscriber_member_id),
        ("payer_id", request.payer_id),
        ("provider_npi", request.provider_npi),
        ("subscriber_last_name", request.subscriber_last_name),
    ):
        if not str(value).strip():
            raise MalformedEDI(f"a 270 needs {name}; refusing to send a partial inquiry")
    if not re.fullmatch(r"\d{10}", request.provider_npi):
        raise MalformedEDI(
            f"provider NPI {request.provider_npi!r} is not ten digits. A 270 with "
            "a malformed NPI is rejected by the payer, and the rejection reads "
            "like a coverage problem to whoever reads the queue."
        )
    unknown = [c for c in request.service_type_codes if c not in SERVICE_TYPE_CODES]
    if unknown:
        raise MalformedEDI(
            f"unknown service type code(s) {unknown}; known: "
            f"{sorted(SERVICE_TYPE_CODES)}"
        )

    stamp = created.strftime("%Y%m%d")
    clock = created.strftime("%H%M")
    segments: list[list[str]] = [
        ["ST", "270", control_number, "005010X279A1"],
        ["BHT", "0022", "13", request.trace_number or control_number, stamp, clock],
        # 2100A - the payer
        ["HL", "1", "", "20", "1"],
        ["NM1", "PR", "2", request.payer_name, "", "", "", "", "PI", request.payer_id],
        # 2100B - the provider
        ["HL", "2", "1", "21", "1"],
        ["NM1", "1P", "2", request.provider_name, "", "", "", "", "XX",
         request.provider_npi],
        # 2100C - the subscriber
        ["HL", "3", "2", "22", "1" if request.is_dependent else "0"],
        ["NM1", "IL", "1", request.subscriber_last_name, request.subscriber_first_name,
         "", "", "", "MI", request.subscriber_member_id],
        ["DMG", "D8", request.subscriber_dob.strftime("%Y%m%d")],
    ]
    if request.is_dependent:
        segments += [
            # 2100D - the dependent, which in paediatrics is the patient
            ["HL", "4", "3", "23", "0"],
            ["NM1", "03", "1", request.dependent_last_name,
             request.dependent_first_name],
            ["DMG", "D8", request.dependent_dob.strftime("%Y%m%d")],  # type: ignore[union-attr]
        ]
    segments.append(["DTP", "291", "D8", request.service_date.strftime("%Y%m%d")])
    for code in request.service_type_codes:
        segments.append(["EQ", code])
    segments.append(["SE", str(len(segments) + 1), control_number])

    return SEGMENT_TERMINATOR.join(
        ELEMENT_SEPARATOR.join(_trim(seg)) for seg in segments
    ) + SEGMENT_TERMINATOR


def _trim(segment: Sequence[str]) -> list[str]:
    """Drop trailing empty elements, as X12 requires."""
    out = list(segment)
    while len(out) > 1 and out[-1] == "":
        out.pop()
    return out


@dataclass(frozen=True)
class BenefitLine:
    """One EB segment: what the payer said about one kind of benefit."""

    eb_code: str
    meaning: str
    service_type: str = ""
    service_type_label: str = ""
    coverage_level: str = ""
    plan_description: str = ""
    amount: float | None = None
    percent: float | None = None
    #: "IN" / "OUT" as the payer stated it. Empty means the payer did not say,
    #: which is NOT the same as in-network.
    network_indicator: str = ""
    time_period: str = ""
    message: str = ""

    @property
    def is_active(self) -> bool:
        return self.eb_code in _ACTIVE

    @property
    def is_inactive(self) -> bool:
        return self.eb_code in _INACTIVE

    def as_dict(self) -> dict[str, Any]:
        return {
            "eb_code": self.eb_code, "meaning": self.meaning,
            "service_type": self.service_type,
            "service_type_label": self.service_type_label,
            "coverage_level": self.coverage_level,
            "plan_description": self.plan_description,
            "amount": self.amount, "percent": self.percent,
            "network": self.network_indicator,
            "time_period": self.time_period, "message": self.message,
        }


@dataclass
class Response271:
    """A parsed 271, plus an honest account of what was NOT parsed."""

    control_number: str = ""
    payer_name: str = ""
    payer_id: str = ""
    member_id: str = ""
    patient_last_name: str = ""
    patient_first_name: str = ""
    patient_dob: date | None = None
    plan_begins: date | None = None
    plan_ends: date | None = None
    benefits: list[BenefitLine] = field(default_factory=list)
    #: AAA segments: the payer could not answer, and why.
    rejections: list[dict[str, str]] = field(default_factory=list)
    #: MSG free text not attached to a particular benefit line.
    messages: list[str] = field(default_factory=list)
    #: Segments this subset does not understand. See the module docstring: they
    #: are collected rather than skipped, and their presence makes the whole
    #: response untrustworthy.
    unparsed: list[str] = field(default_factory=list)
    raw: str = ""

    @property
    def trustworthy(self) -> bool:
        """False when anything was unparsed or the payer rejected the inquiry.

        The coverage layer refuses to determine anything from an untrustworthy
        response. A subset parser that quietly returns its partial understanding
        is how a system tells a family they have no insurance because the payer
        used a segment nobody implemented.
        """
        return not self.unparsed and not self.rejections

    @property
    def has_active_benefit(self) -> bool:
        return any(b.is_active for b in self.benefits)

    @property
    def has_inactive_benefit(self) -> bool:
        return any(b.is_inactive for b in self.benefits)

    def benefits_for(self, service_type: str) -> list[BenefitLine]:
        return [b for b in self.benefits if b.service_type == service_type]

    def as_dict(self) -> dict[str, Any]:
        return {
            "payer_name": self.payer_name, "payer_id": self.payer_id,
            "member_id": self.member_id,
            "patient": f"{self.patient_last_name}, {self.patient_first_name}",
            "plan_begins": self.plan_begins.isoformat() if self.plan_begins else None,
            "plan_ends": self.plan_ends.isoformat() if self.plan_ends else None,
            "trustworthy": self.trustworthy,
            "benefits": [b.as_dict() for b in self.benefits],
            "rejections": list(self.rejections),
            "messages": list(self.messages),
            "unparsed": list(self.unparsed),
        }


class X12Parser(Protocol):
    """Swap `SubsetParser` for a real library by implementing this."""

    name: str

    def parse_271(self, payload: str) -> Response271: ...


#: AAA03 reject reason codes, the ones a practice actually sees.
_AAA_REASONS: Mapping[str, str] = {
    "15": "required application data missing",
    "33": "input errors",
    "35": "out of network",
    "42": "unable to respond at current time",
    "43": "invalid or missing provider identification",
    "45": "invalid or missing provider specialty",
    "47": "invalid or missing provider state",
    "48": "invalid or missing referring provider identification number",
    "50": "provider ineligible for inquiries",
    "51": "provider not on file",
    "56": "inappropriate date",
    "57": "invalid or missing date of service",
    "58": "invalid or missing date of birth",
    "60": "date of birth follows date of service",
    "61": "date of death precedes date of service",
    "62": "date of service not within allowable inquiry period",
    "63": "date of service in future",
    "64": "invalid or missing patient id",
    "65": "invalid or missing patient name",
    "66": "invalid or missing patient gender",
    "67": "patient not found",
    "68": "duplicate patient id number",
    "71": "patient birth date does not match that for the patient on the database",
    "72": "invalid or missing subscriber/insured id",
    "73": "invalid or missing subscriber/insured name",
    "75": "subscriber/insured not found",
    "76": "duplicate subscriber/insured id number",
}

#: Segment ids this subset understands. Everything else is `unparsed`.
_KNOWN_SEGMENTS = frozenset(
    {"ST", "BHT", "HL", "NM1", "DMG", "DTP", "EB", "EQ", "SE", "TRN", "REF", "AAA",
     "MSG", "INS", "PER", "N3", "N4", "III", "LS", "LE", "HI"}
)
#: ...and the ones that carry meaning we actually extract. A segment that is
#: known-but-ignored (an address, say) does not make the response untrustworthy;
#: a segment nobody has heard of does.
#: MSG is NOT here. A payer's free-text limitation -- "COVERAGE TERMINATED
#: 05/31/2026 - VERIFY BEFORE SERVICE" -- was being discarded while
#: `BenefitLine.message` stayed empty and the response reported active coverage.
_IGNORED_SEGMENTS = frozenset(
    {"PER", "N3", "N4", "III", "LS", "LE", "HI", "INS", "REF", "TRN"}
)


@dataclass
class SubsetParser:
    """A strict subset. Correct on what it claims, loud about the rest."""

    name: str = "subset"

    def parse_271(self, payload: str) -> Response271:
        text = payload.strip()
        if not text:
            raise MalformedEDI("empty 271 payload")
        if "ST" not in text:
            raise MalformedEDI("no ST segment; this is not an X12 transaction set")

        response = Response271(raw=text)
        segments = [s for s in text.split(SEGMENT_TERMINATOR) if s.strip()]
        current_entity = ""
        pending_benefit: dict[str, Any] | None = None

        def flush() -> None:
            nonlocal pending_benefit
            if pending_benefit is not None:
                response.benefits.append(BenefitLine(**pending_benefit))
                pending_benefit = None

        for raw_segment in segments:
            parts = raw_segment.split(ELEMENT_SEPARATOR)
            tag = parts[0].strip().upper()
            if tag not in _KNOWN_SEGMENTS:
                flush()
                response.unparsed.append(raw_segment)
                continue
            if tag in _IGNORED_SEGMENTS:
                continue

            if tag == "MSG":
                text = _at(parts, 1)
                if pending_benefit is not None:
                    pending_benefit["message"] = (
                        f"{pending_benefit['message']} {text}".strip()
                    )
                elif text:
                    response.messages.append(text)
            elif tag == "ST":
                response.control_number = _at(parts, 2)
            elif tag == "NM1":
                entity = _at(parts, 1)
                current_entity = entity
                if entity == "PR":
                    response.payer_name = _at(parts, 3)
                    response.payer_id = _at(parts, 9)
                elif entity in ("IL", "03"):
                    response.patient_last_name = _at(parts, 3)
                    response.patient_first_name = _at(parts, 4)
                    if _at(parts, 8) == "MI" and _at(parts, 9):
                        response.member_id = _at(parts, 9)
            elif tag == "DMG":
                if _at(parts, 1) == "D8":
                    response.patient_dob = _d8(_at(parts, 2))
            elif tag == "DTP":
                qualifier, fmt, value = _at(parts, 1), _at(parts, 2), _at(parts, 3)
                if qualifier in ("346", "356"):             # plan / eligibility begin
                    response.plan_begins = _d8(value) if fmt == "D8" else None
                elif qualifier in ("347", "349", "357"):    # plan / benefit end
                    response.plan_ends = _d8(value) if fmt == "D8" else None
                elif qualifier == "307":                    # eligibility
                    if fmt == "D8":
                        response.plan_begins = _d8(value)
                    elif fmt == "RD8" and "-" in value:
                        begin, _, end = value.partition("-")
                        response.plan_begins = _d8(begin)
                        response.plan_ends = _d8(end)
                    else:
                        response.unparsed.append(raw_segment)
                elif qualifier in ("291", "348") and fmt == "RD8" and "-" in value:
                    begin, _, end = value.partition("-")
                    response.plan_begins = _d8(begin)
                    response.plan_ends = _d8(end)
                elif qualifier in ("291", "348", "472"):
                    # A service-date qualifier we do not need. Known and ignored.
                    pass
                else:
                    # THE DESIGN RULE, applied. `349` and `307` used to fall
                    # through this chain with no `else`, so a payer stating a
                    # coverage end date in a qualifier this parser did not
                    # handle produced plan_ends=None, unparsed=[], and a
                    # trustworthy response asserting active coverage.
                    response.unparsed.append(raw_segment)
            elif tag == "AAA":
                code = _at(parts, 3)
                response.rejections.append(
                    {
                        "valid": _at(parts, 1),
                        "code": code,
                        "reason": _AAA_REASONS.get(code, f"unmapped reject code {code}"),
                        "follow_up": _at(parts, 4),
                        "entity": current_entity,
                    }
                )
            elif tag == "EB":
                flush()
                eb_code = _at(parts, 1)
                if eb_code not in EB_CODES:
                    # An EB code this subset does not know is exactly the case
                    # that must not be guessed: it could mean covered, not
                    # covered, or "call us".
                    response.unparsed.append(raw_segment)
                    continue
                raw_service = _at(parts, 3)
                service_types = [
                    part.split(COMPONENT_SEPARATOR)[0]
                    for part in raw_service.split(REPETITION_SEPARATOR)
                    if part
                ] or [""]
                network, network_ok = _network_indicator(_at(parts, 12))
                if not network_ok:
                    # An indicator this parser does not recognise is not
                    # evidence of in-network status, and defaulting it would
                    # widen a threshold on missing data.
                    response.unparsed.append(raw_segment)
                    continue
                shared = {
                    "eb_code": eb_code,
                    "meaning": EB_CODES[eb_code],
                    "coverage_level": _at(parts, 2),
                    "plan_description": _at(parts, 5),
                    "time_period": _at(parts, 6),
                    "amount": _num(_at(parts, 7)),
                    "percent": _pct(_at(parts, 8)),
                    "network_indicator": network,
                    "message": "",
                }
                # One line per repetition. A payer answering three service types
                # in one EB segment is answering three questions.
                for extra in service_types[1:]:
                    response.benefits.append(
                        BenefitLine(
                            **{**shared, "service_type": extra,
                               "service_type_label": SERVICE_TYPE_CODES.get(extra, "")}
                        )
                    )
                pending_benefit = {
                    **shared,
                    "service_type": service_types[0],
                    "service_type_label": SERVICE_TYPE_CODES.get(
                        service_types[0], ""
                    ),
                }
        flush()
        return response


def _at(parts: Sequence[str], index: int) -> str:
    return parts[index].strip() if index < len(parts) else ""


def _d8(value: str) -> date | None:
    try:
        return datetime.strptime(value.strip(), "%Y%m%d").date()
    except (ValueError, AttributeError):
        return None


def _num(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


#: How payers actually spell the in-network indicator (EB12). `Y`/`N` is the
#: 005010 standard; `IN`/`OUT` shows up in the wild and in this module's own
#: `BenefitLine` docstring. Anything else routes to `unparsed`.
_NETWORK_INDICATORS: Mapping[str, str] = {
    "": "", "Y": "Y", "N": "N", "IN": "Y", "OUT": "N", "W": "",
    "U": "",   # unknown, per the standard
}


def _network_indicator(raw: str) -> tuple[str, bool]:
    """`(normalised, recognised)`. Empty normalised means the payer did not say."""
    key = raw.strip().upper()
    if key in _NETWORK_INDICATORS:
        return _NETWORK_INDICATORS[key], True
    return "", False


def _pct(value: str) -> float | None:
    number = _num(value)
    if number is None:
        return None
    # X12 states coinsurance as a decimal fraction. A payer that sends `20`
    # meaning twenty percent and one that sends `.2` are both common; treating
    # `.2` as 0.2% would tell a family their share is nothing.
    return number * 100.0 if number <= 1.0 else number
