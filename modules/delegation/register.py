"""Standing orders, competencies, and the roster. The register itself.

README I-10 calls this *"the initiative with the smallest dollar return and the
largest downside protection"*, and describes the state it replaces:

    The likely current state: experienced MAs know what they are authorized to do
    because they have done it for years. New hires learn by apprenticeship. There
    may be a binder. The binder may be out of date. If a regulator, a plaintiff's
    attorney, or an insurer asked the practice to produce the written delegation
    authorizing a specific MA to administer a specific vaccine on a specific
    date, the answer would likely require assembling it after the fact.

"Assembling it after the fact" is the whole problem. Everything here is designed
so that the answer already exists.

THREE DESIGN DECISIONS WORTH THE WORDS.

**Standing orders are VERSIONED and SUPERSEDED, never edited.** A signed order is
an attestation by a named physician about a specific text on a specific date. If
that text can change underneath the signature, the signature means nothing, and
every execution logged against it becomes unprovable. `publish()` creates a new
version and retires the old one; there is no `update()`.

**A competency is held by a PERSON, not by a role.** The statute requires
"appropriate training and experience" for that individual. A practice-level
competency register is exactly the thing that does not satisfy it, and it is the
most natural thing in the world to build by accident.

**An expired competency is not a weaker competency; it is an unverified one.**
There is no grace period in the shipped config and the code does not assume one.
A CPR card that lapsed yesterday is evidence about last year.

THE ROSTER IS HERE because 54.2's on-site requirement is the one that is almost
always satisfied in fact and almost never evidenced. Nobody writes down that Dr
Alvarez was in the building at 14:32; they all remember it. `Roster.on_site_at`
is that memory, made into a record before it is needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Iterable, Mapping, Sequence

import yaml

__all__ = [
    "DEFAULT_RULES_PATH",
    "DelegationRules",
    "StandingOrder",
    "OrderRegister",
    "Competency",
    "CompetencyRecord",
    "CompetencyRegister",
    "StaffMember",
    "RosterEntry",
    "Roster",
    "UnsignedOrder",
    "OrderSuperseded",
    "UnreviewedRules",
    "FrameworkSunset",
]

DEFAULT_RULES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "delegation_rules.yaml",
)


class UnsignedOrder(RuntimeError):
    """Raised when an unsigned standing order is published or executed."""


class OrderSuperseded(RuntimeError):
    """Raised when a retired order version is used."""


class UnreviewedRules(RuntimeError):
    """Raised when the delegation rules have no named owner."""


class FrameworkSunset(RuntimeError):
    """Raised when the register is asked to certify compliance past the sunset.

    225 ILCS 60/54.2 sunsets 2027-01-01. A system that keeps citing a repealed
    statute after that date is producing evidence of compliance with a framework
    that no longer exists, which is worse than producing nothing: it looks like
    an answer.
    """


@dataclass
class DelegationRules:
    """`config/delegation_rules.yaml`, loaded. The statute as data."""

    version: str
    framework: dict[str, Any]
    review: dict[str, Any]
    requirements: list[dict[str, Any]]
    licensed_roles: frozenset[str]
    delegable_to_roles: frozenset[str]
    competency: dict[str, Any]
    standing_orders: dict[str, Any]
    break_glass: dict[str, Any]

    @classmethod
    def load(cls, path: str | os.PathLike[str] = DEFAULT_RULES_PATH) -> "DelegationRules":
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        rules = cls(
            version=str(data.get("version", "unversioned")),
            framework=dict(data.get("framework") or {}),
            review=dict(data.get("review") or {}),
            requirements=[dict(r) for r in data.get("requirements") or []],
            licensed_roles=frozenset(data.get("licensed_roles") or ()),
            delegable_to_roles=frozenset(data.get("delegable_to_roles") or ()),
            competency=dict(data.get("competency") or {}),
            standing_orders=dict(data.get("standing_orders") or {}),
            break_glass=dict(data.get("break_glass") or {}),
        )
        if not rules.requirements:
            raise ValueError(
                "the delegation rules define no requirements, so every act would "
                "pass every check. An empty rule set is not a permissive rule "
                "set; it is a broken one."
            )
        if not rules.licensed_roles:
            raise ValueError(
                "no licensed roles are configured, so nobody could ever satisfy "
                "the on-site requirement"
            )
        overlap = rules.licensed_roles & rules.delegable_to_roles
        # An RN is both: they hold a licence AND may perform delegated tasks.
        # That is legitimate, and it is also the thing that lets one person
        # supervise themselves. `enforcement.py` refuses that specifically; this
        # only records that the overlap is expected.
        rules.framework.setdefault("dual_roles", sorted(overlap))
        return rules

    @property
    def has_owner(self) -> bool:
        owner = str(self.review.get("owner") or "")
        return bool(owner) and "UNASSIGNED" not in owner.upper()

    @property
    def sunsets_on(self) -> date | None:
        raw = self.framework.get("sunsets_on")
        if not raw:
            return None
        return date.fromisoformat(str(raw))

    def sunset_state(self, on: date) -> tuple[str, str]:
        """`(state, message)` where state is ok | approaching | expired."""
        sunset = self.sunsets_on
        if sunset is None:
            return "ok", ""
        days = (sunset - on).days
        citation = self.framework.get("citation", "the delegation framework")
        if days < 0:
            return "expired", (
                f"{citation} sunset {sunset.isoformat()}, {abs(days)} day(s) ago. "
                "This register can still produce a historical record, but it "
                "cannot certify compliance with a framework that no longer "
                "exists. Update config/delegation_rules.yaml to the replacement."
            )
        warn = int(self.framework.get("sunset_warning_days", 180))
        if days <= warn:
            return "approaching", (
                f"{citation} sunsets in {days} day(s) ({sunset.isoformat()}). "
                "The replacement framework is a change to "
                "config/delegation_rules.yaml, not to this application -- which "
                "is the reason this initiative was built before the deadline."
            )
        return "ok", ""

    def requirement(self, rule_id: str) -> dict[str, Any]:
        for rule in self.requirements:
            if rule["id"] == rule_id:
                return rule
        raise KeyError(rule_id)

    @property
    def evidence_kinds(self) -> frozenset[str]:
        return frozenset(self.competency.get("evidence_kinds") or ())


# -- standing orders ---------------------------------------------------------


@dataclass(frozen=True)
class StandingOrder:
    """One version of one protocol. Immutable once signed."""

    order_id: str
    version: int
    title: str
    task_code: str
    clinical_content: str
    delegating_physician_id: str
    effective_from: date
    required_competencies: tuple[str, ...] = ()
    required_supervision_role: str = "physician"
    source_guideline: str = ""
    review_due: date | None = None
    retired_on: date | None = None
    #: The physician's e-signature over this exact text. Unsigned orders cannot
    #: be published and cannot authorise anything.
    signed_by: str = ""
    signed_utc: datetime | None = None

    @property
    def signed(self) -> bool:
        return bool(self.signed_by and self.signed_utc)

    def in_force(self, on: date) -> bool:
        if on < self.effective_from:
            return False
        return self.retired_on is None or on < self.retired_on

    def review_overdue(self, on: date) -> bool:
        return self.review_due is not None and self.review_due < on

    def as_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id, "version": self.version,
            "title": self.title, "task_code": self.task_code,
            "delegating_physician_id": self.delegating_physician_id,
            "effective_from": self.effective_from.isoformat(),
            "retired_on": self.retired_on.isoformat() if self.retired_on else None,
            "required_competencies": list(self.required_competencies),
            "required_supervision_role": self.required_supervision_role,
            "source_guideline": self.source_guideline,
            "review_due": self.review_due.isoformat() if self.review_due else None,
            "signed_by": self.signed_by,
            "signed_utc": self.signed_utc.isoformat() if self.signed_utc else None,
        }


@dataclass
class OrderRegister:
    """Every version of every standing order. Append-only by construction."""

    rules: DelegationRules
    versions: dict[str, list[StandingOrder]] = field(default_factory=dict)

    def publish(self, order: StandingOrder) -> StandingOrder:
        """Add a new version, retiring the previous one. There is no `update`.

        A signed order is an attestation by a named physician about a specific
        text on a specific date. Editing it in place would leave the signature
        attached to words the signer never read, and every execution logged
        against that order becomes unprovable.
        """
        if not order.signed:
            raise UnsignedOrder(
                f"standing order {order.order_id!r} v{order.version} has no "
                "physician signature. Under 225 ILCS 60/54.2 the delegation is "
                "the physician's act; an unsigned protocol authorises nothing."
            )
        if order.delegating_physician_id != order.signed_by:
            raise UnsignedOrder(
                f"order {order.order_id!r} names {order.delegating_physician_id!r} "
                f"as the delegating physician but was signed by {order.signed_by!r}. "
                "The delegation and the signature are the same act."
            )
        if not order.required_competencies:
            # `authorise()`'s competency loop simply does not run for an empty
            # list, so `individual_competency` and `competency_current` pass by
            # vacuum and every person in a delegable role is authorised. A blank
            # competency list on an epinephrine protocol put a six-week-old hire
            # on the screen for IM epinephrine with no verified training --
            # missing data widening the gate, on the one statute clause that is
            # explicitly about the individual.
            raise ValueError(
                f"standing order {order.order_id!r} names no required "
                "competencies. Under 225 ILCS 60/54.2 the delegate must have "
                "appropriate training and experience FOR THIS TASK; an order "
                "requiring nothing authorises everyone in a delegable role. If "
                "the task genuinely needs no specific competency, define one "
                "that says so and verify it."
            )
        if order.required_supervision_role not in self.rules.licensed_roles:
            raise ValueError(
                f"order {order.order_id!r} requires supervision by "
                f"{order.required_supervision_role!r}, which is not a licensed "
                f"role ({sorted(self.rules.licensed_roles)})"
            )
        if order.signed_utc is not None and order.effective_from < order.signed_utc.date():
            # An order in force before its own signature is a delegation the
            # physician had not yet made.
            raise ValueError(
                f"order {order.order_id!r} v{order.version} is effective "
                f"{order.effective_from.isoformat()} but was signed "
                f"{order.signed_utc.date().isoformat()}. A delegation cannot "
                "predate the act of delegating."
            )
        history = self.versions.setdefault(order.order_id, [])
        if history:
            previous = history[-1]
            if order.version <= previous.version:
                raise ValueError(
                    f"order {order.order_id!r} v{order.version} does not follow "
                    f"v{previous.version}"
                )
            if order.effective_from < previous.effective_from:
                # A back-dated publish retroactively rewrote which version was
                # in force. A logged execution then resolved to text signed six
                # weeks AFTER the act, and the previous version was left
                # "effective 2025-04-01, retired 2020-01-01" -- in force on no
                # date at all, the same inverted window `PayerTable.add` and
                # `Roster.record` refuse outright.
                raise ValueError(
                    f"order {order.order_id!r} v{order.version} is effective "
                    f"{order.effective_from.isoformat()}, before v"
                    f"{previous.version} began ({previous.effective_from.isoformat()}). "
                    "Publishing backwards rewrites which text was in force on a "
                    "day that has already happened, and executions logged against "
                    "the old version stop resolving to what the physician signed."
                )
            if previous.retired_on is None:
                history[-1] = StandingOrder(
                    **{**previous.__dict__, "retired_on": order.effective_from}
                )
        history.append(order)
        return order

    def current(self, order_id: str, *, on: date) -> StandingOrder | None:
        """The version in force on `on`. None if none is."""
        for candidate in reversed(self.versions.get(order_id, [])):
            if candidate.in_force(on):
                return candidate
        return None

    def version_at(self, order_id: str, version: int) -> StandingOrder:
        for candidate in self.versions.get(order_id, []):
            if candidate.version == version:
                return candidate
        raise KeyError(f"{order_id!r} v{version}")

    def for_task(self, task_code: str, *, on: date) -> list[StandingOrder]:
        return [
            order
            for order_id in sorted(self.versions)
            if (order := self.current(order_id, on=on)) is not None
            and order.task_code == task_code
        ]

    def review_tasks(self, on: date) -> list[dict[str, Any]]:
        """Orders due for review, and those already overdue.

        README I-10's failure table: "Standing orders drift from current clinical
        guidelines -> clinically outdated protocol executed as authorized."
        """
        warn = int(self.rules.standing_orders.get("review_warning_days", 60))
        rows: list[dict[str, Any]] = []
        for order_id in sorted(self.versions):
            order = self.current(order_id, on=on)
            if order is None or order.review_due is None:
                continue
            days = (order.review_due - on).days
            if days < 0:
                rows.append({
                    "order_id": order_id, "version": order.version,
                    "title": order.title, "state": "overdue", "days": days,
                    "physician": order.delegating_physician_id,
                })
            elif days <= warn:
                rows.append({
                    "order_id": order_id, "version": order.version,
                    "title": order.title, "state": "due", "days": days,
                    "physician": order.delegating_physician_id,
                })
        return rows


# -- competencies ------------------------------------------------------------


@dataclass(frozen=True)
class Competency:
    """A named capability an order can require."""

    competency_id: str
    title: str
    description: str = ""


@dataclass(frozen=True)
class CompetencyRecord:
    """One person, one competency, verified by one licensed person, once."""

    record_id: str
    staff_id: str
    competency_id: str
    verified_by: str
    verified_on: date
    evidence_kind: str
    evidence_reference: str = ""
    expires_on: date | None = None

    def current_on(self, day: date, *, grace_days: int = 0) -> bool:
        if day < self.verified_on:
            return False
        if self.expires_on is None:
            return True
        return day <= self.expires_on + timedelta(days=grace_days)

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id, "staff_id": self.staff_id,
            "competency_id": self.competency_id,
            "verified_by": self.verified_by,
            "verified_on": self.verified_on.isoformat(),
            "evidence_kind": self.evidence_kind,
            "evidence_reference": self.evidence_reference,
            "expires_on": self.expires_on.isoformat() if self.expires_on else None,
        }


@dataclass(frozen=True)
class StaffMember:
    staff_id: str
    name: str
    role: str
    licence_number: str = ""
    started_on: date | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "staff_id": self.staff_id, "name": self.name, "role": self.role,
            "licence_number": self.licence_number,
        }


@dataclass
class CompetencyRegister:
    """Who is competent at what, and until when."""

    rules: DelegationRules
    staff: dict[str, StaffMember] = field(default_factory=dict)
    competencies: dict[str, Competency] = field(default_factory=dict)
    records: list[CompetencyRecord] = field(default_factory=list)

    def add_staff(self, member: StaffMember) -> None:
        self.staff[member.staff_id] = member

    def define(self, competency: Competency) -> None:
        self.competencies[competency.competency_id] = competency

    def verify(self, record: CompetencyRecord) -> CompetencyRecord:
        """Record a competency. The verifier must hold a licence.

        An unlicensed person cannot attest that another unlicensed person is
        competent -- that is the whole shape of 54.2, and a register that let an
        MA sign off another MA would produce records that prove nothing.
        """
        verifier = self.staff.get(record.verified_by)
        if verifier is None:
            raise ValueError(
                f"verifier {record.verified_by!r} is not on the staff list. A "
                "competency record names who verified it, and that name has to "
                "be somebody."
            )
        if verifier.role not in self.rules.licensed_roles:
            raise ValueError(
                f"{verifier.name} ({verifier.role}) is not a licensed role and "
                f"cannot verify competency. Licensed roles: "
                f"{sorted(self.rules.licensed_roles)}."
            )
        if record.staff_id == record.verified_by:
            raise ValueError(
                "a competency cannot be self-verified, whatever the verifier's "
                "licence. The record exists to show somebody else looked."
            )
        if record.competency_id not in self.competencies:
            raise KeyError(
                f"competency {record.competency_id!r} is not defined. An order "
                "requiring an undefined competency can never be satisfied."
            )
        kinds = self.rules.evidence_kinds
        if kinds and record.evidence_kind not in kinds:
            raise ValueError(
                f"evidence kind {record.evidence_kind!r} is not on the allowed "
                f"list {sorted(kinds)}. 'They have done it for years' is exactly "
                "what this register exists to replace."
            )
        self.records.append(record)
        return record

    def current(
        self, staff_id: str, competency_id: str, *, on: date
    ) -> CompetencyRecord | None:
        """The most recently verified current record, or None."""
        grace = int(self.rules.competency.get("grace_days", 0))
        candidates = [
            r for r in self.records
            if r.staff_id == staff_id
            and r.competency_id == competency_id
            and r.current_on(on, grace_days=grace)
        ]
        return max(candidates, key=lambda r: r.verified_on) if candidates else None

    def held_by(self, staff_id: str, *, on: date) -> set[str]:
        return {
            r.competency_id for r in self.records
            if r.staff_id == staff_id
            and r.current_on(on, grace_days=int(
                self.rules.competency.get("grace_days", 0)
            ))
        }

    def expiring(self, on: date) -> list[dict[str, Any]]:
        """Competencies expiring soon or already expired, per person.

        README I-10 target state: "Competency expiring in 30 days generates a
        task." The expired ones are listed too, because the interesting case is
        the one nobody actioned.
        """
        warn = int(self.rules.competency.get("renewal_warning_days", 30))
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for record in sorted(self.records, key=lambda r: -r.verified_on.toordinal()):
            key = (record.staff_id, record.competency_id)
            if key in seen or record.expires_on is None:
                continue
            seen.add(key)
            days = (record.expires_on - on).days
            if days > warn:
                continue
            member = self.staff.get(record.staff_id)
            rows.append({
                "staff_id": record.staff_id,
                "name": member.name if member else record.staff_id,
                "competency_id": record.competency_id,
                "expires_on": record.expires_on.isoformat(),
                "days": days,
                "state": "expired" if days < 0 else "expiring",
            })
        return sorted(rows, key=lambda r: r["days"])


# -- the roster --------------------------------------------------------------


@dataclass(frozen=True)
class RosterEntry:
    """One licensed professional, on site, between two times."""

    staff_id: str
    start_utc: datetime
    end_utc: datetime

    def covers(self, moment: datetime) -> bool:
        return self.start_utc <= moment <= self.end_utc


@dataclass
class Roster:
    """Who was in the building, when. The 54.2 requirement, evidenced.

    README I-10's failure table: "Physician-on-premises requirement not
    evidenced -> the statute requires a licensed professional on site; nothing
    records whether one was."

    Everybody remembers that Dr Alvarez was here on Tuesday. Nobody wrote it
    down, and two years later "everybody remembers" is not evidence.
    """

    rules: DelegationRules
    staff: dict[str, StaffMember] = field(default_factory=dict)
    entries: list[RosterEntry] = field(default_factory=list)

    def record(self, entry: RosterEntry) -> RosterEntry:
        member = self.staff.get(entry.staff_id)
        if member is None:
            raise ValueError(f"{entry.staff_id!r} is not on the staff list")
        if entry.end_utc < entry.start_utc:
            raise ValueError(
                f"roster entry for {entry.staff_id!r} ends before it starts; an "
                "inverted window covers no moment at all and would silently fail "
                "every on-site check"
            )
        self.entries.append(entry)
        return entry

    def on_site_at(self, moment: datetime) -> list[StaffMember]:
        """Every licensed professional on site at `moment`."""
        return [
            member
            for entry in self.entries
            if entry.covers(moment)
            and (member := self.staff.get(entry.staff_id)) is not None
            and member.role in self.rules.licensed_roles
        ]

    def supervisor_at(
        self, moment: datetime, *, role: str, excluding: str = ""
    ) -> StaffMember | None:
        """A licensed professional of `role` on site, who is not `excluding`.

        `excluding` is how a person is stopped from supervising themselves. An
        RN is both a licensed role and a delegable role, so without it the
        register would happily record an RN as their own supervising
        professional -- which satisfies the letter of a lookup and none of the
        statute.
        """
        for member in self.on_site_at(moment):
            if member.staff_id == excluding:
                continue
            if member.role == role:
                return member
        return None
