"""What this person may do right now, and the record of them doing it.

README I-10, target state:

    An MA opening a task sees ONLY orders they are currently competent to
    execute.

That word "only" is the enforcement layer, and it is the difference between this
initiative and a binder. A binder tells you what the rules are; this decides
what appears on the screen. The MA never sees the order they are not competent
for, so there is no moment where somebody has to remember not to click it.

FOUR THINGS TO NOTICE.

**Every refusal names a rule id from `config/delegation_rules.yaml`.** So an
audit answer is "this act failed `competency_current`" rather than a paragraph,
and when the statute changes the reasons change with the config.

**The check runs twice.** `available_orders()` filters the screen and
`execute()` checks again at the moment of the act. Not belt and braces: a
competency can expire between the MA opening the worklist at 08:00 and giving
the injection at 11:40, and the roster changes when the physician goes to lunch.
The second check is the one that is true.

**Break glass cannot be refused.** The build plan: *"Include a break-glass path
that requires a justification and never blocks patient care."* So `execute()`
with `break_glass=` set never raises for a delegation reason. It is loud instead
-- recorded with `break_glass=True`, surfaced permanently by
`AuditLog.unevidenced_supervision()`, and it raises a review task. A break-glass
that blocks is a break-glass staff learn to route around; one nobody reviews is
an unlocked door.

**Nobody supervises themselves.** An RN is both a licensed role and a delegable
role. Without an explicit check the register would record an RN as their own
supervising professional, which satisfies a lookup and none of the statute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence

from .register import (
    CompetencyRegister,
    DelegationRules,
    OrderRegister,
    Roster,
    StandingOrder,
    StaffMember,
)

__all__ = [
    "Refusal",
    "Authorisation",
    "ExecutionResult",
    "DelegationService",
    "NotAuthorised",
    "BreakGlassRefused",
]


class NotAuthorised(RuntimeError):
    """Raised when an act is attempted without satisfying the framework."""

    def __init__(self, message: str, refusals: Sequence["Refusal"]) -> None:
        super().__init__(message)
        self.refusals = list(refusals)


class BreakGlassRefused(ValueError):
    """Raised when break glass is invoked without a usable justification.

    NOT a delegation refusal. Break glass never blocks care; this fires when the
    justification is missing or too short to mean anything, which is a data
    problem in the request rather than a reason to stop.
    """


@dataclass(frozen=True)
class Refusal:
    """One rule, not satisfied, in the words of the config."""

    rule_id: str
    detail: str
    blocking: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id, "detail": self.detail,
            "blocking": self.blocking,
        }


@dataclass
class Authorisation:
    """Whether this person may perform this order at this moment, and why."""

    staff: StaffMember
    order: StandingOrder | None
    task_code: str
    moment: datetime
    refusals: list[Refusal] = field(default_factory=list)
    supervisor: StaffMember | None = None
    competency_record_ids: tuple[str, ...] = ()
    competency_expires: date | None = None

    @property
    def authorised(self) -> bool:
        return not [r for r in self.refusals if r.blocking]

    def as_dict(self) -> dict[str, Any]:
        return {
            "staff_id": self.staff.staff_id,
            "task_code": self.task_code,
            "order_id": self.order.order_id if self.order else None,
            "order_version": self.order.version if self.order else None,
            "authorised": self.authorised,
            "supervisor": self.supervisor.staff_id if self.supervisor else None,
            "competency_records": list(self.competency_record_ids),
            "competency_expires": (
                self.competency_expires.isoformat()
                if self.competency_expires else None
            ),
            "refusals": [r.as_dict() for r in self.refusals],
        }


@dataclass
class ExecutionResult:
    """A delegated act, performed and recorded."""

    execution_id: str
    authorisation: Authorisation
    patient_id: str
    performed_utc: datetime
    break_glass: bool = False
    break_glass_reason: str = ""
    review_due_utc: datetime | None = None
    outcome: str = "completed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "staff_id": self.authorisation.staff.staff_id,
            "task_code": self.authorisation.task_code,
            "order_id": (
                self.authorisation.order.order_id
                if self.authorisation.order else None
            ),
            "order_version": (
                self.authorisation.order.version
                if self.authorisation.order else None
            ),
            "supervisor": (
                self.authorisation.supervisor.staff_id
                if self.authorisation.supervisor else None
            ),
            "performed_utc": self.performed_utc.isoformat(),
            "break_glass": self.break_glass,
            "break_glass_reason": self.break_glass_reason,
            "review_due_utc": (
                self.review_due_utc.isoformat() if self.review_due_utc else None
            ),
            "outcome": self.outcome,
            "refusals": [r.rule_id for r in self.authorisation.refusals],
        }


@dataclass
class DelegationService:
    """The enforcement layer. Reads the rules; knows nothing about Illinois."""

    rules: DelegationRules
    orders: OrderRegister
    competencies: CompetencyRegister
    roster: Roster
    audit: Any = None
    INITIATIVE: str = "I-10"

    # -- the screen ---------------------------------------------------------

    def available_orders(
        self, staff_id: str, *, moment: datetime
    ) -> list[StandingOrder]:
        """Only the orders this person may execute right now.

        The filter is the control. An MA who never sees the order they are not
        competent for never has to remember not to click it, and a new hire on
        their first morning sees a short list rather than a long one with
        warnings on it.
        """
        member = self.competencies.staff.get(staff_id)
        if member is None:
            return []
        out: list[StandingOrder] = []
        for order_id in sorted(self.orders.versions):
            order = self.orders.current(order_id, on=moment.date())
            if order is None:
                continue
            if self.authorise(
                staff_id, order.order_id, moment=moment
            ).authorised:
                out.append(order)
        return out

    def blocked_orders(
        self, staff_id: str, *, moment: datetime
    ) -> list[dict[str, Any]]:
        """What this person cannot do, and why. For the supervisor's screen.

        Not shown to the MA -- that is the point of `available_orders` -- but a
        practice manager needs to see that three people are one expired CPR card
        away from not being able to give vaccines.
        """
        rows: list[dict[str, Any]] = []
        for order_id in sorted(self.orders.versions):
            order = self.orders.current(order_id, on=moment.date())
            if order is None:
                continue
            decision = self.authorise(staff_id, order_id, moment=moment)
            if not decision.authorised:
                rows.append({
                    "order_id": order_id,
                    "title": order.title,
                    "reasons": [r.rule_id for r in decision.refusals if r.blocking],
                    "detail": "; ".join(
                        r.detail for r in decision.refusals if r.blocking
                    ),
                })
        return rows

    # -- the decision -------------------------------------------------------

    def authorise(
        self, staff_id: str, order_id: str, *, moment: datetime
    ) -> Authorisation:
        """Check every requirement. Deterministic; no model runs here."""
        day = moment.date()
        member = self.competencies.staff.get(staff_id)
        if member is None:
            raise KeyError(f"{staff_id!r} is not on the staff list")

        order = self.orders.current(order_id, on=day)
        decision = Authorisation(
            staff=member,
            order=order,
            task_code=order.task_code if order else "",
            moment=moment,
        )

        def refuse(rule_id: str, detail: str) -> None:
            rule = self.rules.requirement(rule_id)
            decision.refusals.append(
                Refusal(rule_id, detail, blocking=bool(rule.get("blocking", True)))
            )

        # 0. Is this person even someone tasks may be delegated to?
        if member.role not in self.rules.delegable_to_roles:
            refuse(
                "written_delegation",
                f"{member.name} is a {member.role}, which is not a role this "
                f"framework delegates to ({sorted(self.rules.delegable_to_roles)})",
            )

        # 1. Is there a signed, in-force order at all?
        if order is None:
            refuse(
                "written_delegation",
                f"no version of standing order {order_id!r} is in force on "
                f"{day.isoformat()}",
            )
            return decision
        if not order.signed:
            refuse(
                "written_delegation",
                f"standing order {order_id!r} v{order.version} is unsigned",
            )
        if order.review_overdue(day):
            # Overdue review does NOT block: a protocol that nobody re-read is
            # still the protocol the physician signed, and refusing every
            # vaccine because a review date slipped would be a worse outcome
            # than the risk it addresses. It is recorded and reported.
            decision.refusals.append(
                Refusal(
                    "written_delegation",
                    f"the review of {order_id!r} was due "
                    f"{order.review_due.isoformat() if order.review_due else '?'} "
                    "and has not happened; the protocol may have drifted from "
                    "current guidance",
                    blocking=False,
                )
            )

        # 2 & 3. Individual competency, and current.
        held_ids: list[str] = []
        expiries: list[date] = []
        for competency_id in order.required_competencies:
            record = self.competencies.current(staff_id, competency_id, on=day)
            if record is None:
                lapsed = [
                    r for r in self.competencies.records
                    if r.staff_id == staff_id and r.competency_id == competency_id
                ]
                if lapsed:
                    newest = max(lapsed, key=lambda r: r.verified_on)
                    refuse(
                        "competency_current",
                        f"{member.name}'s {competency_id} competency expired "
                        f"{newest.expires_on.isoformat() if newest.expires_on else '?'}. "
                        "An expired competency is not a weaker competency; it is "
                        "an unverified one.",
                    )
                else:
                    refuse(
                        "individual_competency",
                        f"{member.name} has no verified {competency_id} "
                        "competency. The statute requires appropriate training "
                        "and experience for THAT PERSON.",
                    )
            else:
                held_ids.append(record.record_id)
                if record.expires_on is not None:
                    expiries.append(record.expires_on)
        decision.competency_record_ids = tuple(held_ids)
        decision.competency_expires = min(expiries) if expiries else None

        # 4. A licensed professional on site, who is not this person.
        supervisor = self.roster.supervisor_at(
            moment, role=order.required_supervision_role, excluding=staff_id
        )
        if supervisor is None:
            others = [
                m.role for m in self.roster.on_site_at(moment)
                if m.staff_id != staff_id
            ]
            refuse(
                "licensed_professional_on_site",
                f"no {order.required_supervision_role} is on site at "
                f"{moment.isoformat()}"
                + (f" (on site: {sorted(set(others))})" if others else " (nobody is)")
                + ". 225 ILCS 60/54.2 requires a licensed health care "
                "professional on the premises.",
            )
        decision.supervisor = supervisor

        # 5. Within the delegating physician's scope. Expressed as: the
        #    physician who signed must be someone the practice knows, because
        #    a scope attaches to a person and an unknown signer has none.
        if order.delegating_physician_id not in self.competencies.staff:
            refuse(
                "within_delegating_scope",
                f"the delegating physician {order.delegating_physician_id!r} is "
                "not on the staff list, so their scope cannot be established",
            )
        return decision

    # -- the act ------------------------------------------------------------

    def execute(
        self,
        staff_id: str,
        order_id: str,
        *,
        patient_id: str,
        moment: datetime,
        execution_id: str,
        break_glass_reason: str = "",
        outcome: str = "completed",
    ) -> ExecutionResult:
        """Perform a delegated act, or refuse — unless break glass is invoked.

        The authorisation is re-checked HERE, not trusted from whatever the
        screen showed. A competency can expire between the MA opening the
        worklist at 08:00 and giving the injection at 11:40, and the physician
        goes to lunch.
        """
        decision = self.authorise(staff_id, order_id, moment=moment)
        break_glass = bool(break_glass_reason)

        if break_glass:
            # Break glass covers the requirements it is FOR -- competency,
            # supervision, an overdue review -- and not the existence of the
            # act. An order id that resolves to nothing is a typo, not patient
            # care being blocked, and performing one recorded an act as
            # "completed" against no protocol, no version, no supervisor and no
            # competency: an unauditable row, which is the opposite of what this
            # module exists to produce.
            if decision.order is None:
                raise BreakGlassRefused(
                    f"there is no standing order {order_id!r} in force on "
                    f"{moment.date().isoformat()}. Break glass performs an act "
                    "the framework would not currently permit; it does not "
                    "perform an act that does not exist. Check the order id."
                )
            if not decision.order.signed:
                raise BreakGlassRefused(
                    f"standing order {order_id!r} v{decision.order.version} is "
                    "unsigned. An unsigned protocol is not a delegation the "
                    "physician has made, so there is nothing to break glass on."
                )
            minimum = int(self.rules.break_glass.get("min_justification_chars", 40))
            if not self.rules.break_glass.get("enabled", False):
                raise BreakGlassRefused(
                    "break glass is disabled in config/delegation_rules.yaml"
                )
            if staff_id and (
                (member := self.competencies.staff.get(staff_id)) is not None
                and member.role not in set(self.rules.break_glass.get("roles", ()))
            ):
                raise BreakGlassRefused(
                    f"{member.role!r} may not invoke break glass "
                    f"({sorted(self.rules.break_glass.get('roles', ()))})"
                )
            if len(break_glass_reason.strip()) < minimum:
                raise BreakGlassRefused(
                    f"a break-glass justification must be at least {minimum} "
                    f"characters; got {len(break_glass_reason.strip())}. This is "
                    "the record somebody reads tomorrow to decide whether the "
                    "rule or the situation needs changing."
                )
        elif not decision.authorised:
            blocking = [r for r in decision.refusals if r.blocking]
            raise NotAuthorised(
                f"{decision.staff.name} may not perform {order_id!r} at "
                f"{moment.isoformat()}: "
                + "; ".join(f"[{r.rule_id}] {r.detail}" for r in blocking),
                blocking,
            )

        # The framework has to still exist. `certify()` refused after the sunset
        # and nothing called it, so `authorise` and `execute` went on
        # authorising acts under a repealed statute -- and the audit extract
        # named it as the governing framework.
        self.certify(moment.date())

        review_due = None
        if break_glass:
            hours = float(self.rules.break_glass.get("review_within_hours", 24))
            review_due = moment + _hours(hours)

        result = ExecutionResult(
            execution_id=execution_id,
            authorisation=decision,
            patient_id=patient_id,
            performed_utc=moment,
            break_glass=break_glass,
            break_glass_reason=break_glass_reason,
            review_due_utc=review_due,
            outcome=outcome,
        )

        if self.audit is not None:
            self.audit.record_delegated_execution(
                staff_id=staff_id,
                initiative_id=self.INITIATIVE,
                task_code=decision.task_code or order_id,
                supervising_pro_id=(
                    decision.supervisor.staff_id if decision.supervisor else None
                ),
                supervisor_on_site=decision.supervisor is not None,
                patient_id=patient_id,
                standing_order_id=(
                    decision.order.order_id if decision.order else None
                ),
                standing_order_version=(
                    str(decision.order.version) if decision.order else None
                ),
                competency_record_id=(
                    decision.competency_record_ids[0]
                    if decision.competency_record_ids else None
                ),
                competency_expires=(
                    decision.competency_expires.isoformat()
                    if decision.competency_expires else None
                ),
                break_glass=break_glass,
                break_glass_reason=break_glass_reason or None,
                extra={
                    "execution_id": execution_id,
                    "outcome": outcome,
                    "refusals": [r.rule_id for r in decision.refusals],
                    "rules_version": self.rules.version,
                },
            )
        return result

    # -- the audit answer ---------------------------------------------------

    def audit_extract(
        self,
        *,
        staff_id: str | None = None,
        since: date | None = None,
        until: date | None = None,
        as_of: date | None = None,
    ) -> dict[str, Any]:
        """README I-10's one-click report: "every delegation, competency record,
        and execution for MA X between date A and date B".

        BOTH ENDS OF THE RANGE. An earlier version had no `until` at all, so
        "date B" could not be expressed, and its unevidenced-supervision query
        was unscoped -- an extract for one MA disclosed another employee's
        break-glass incident record to whoever asked.

        `as_of` is the date the extract is certified against; it defaults to
        `until` and then to today. The certification is what stops this document
        naming a repealed statute as the governing framework.
        """
        if self.audit is None:
            raise RuntimeError("no audit log is configured; there is nothing to extract")
        certify_on = as_of or until or date.today()
        # Refuses after the sunset, and refuses without a named rules owner.
        # `certify()` existed and nothing called it, so the one-click compliance
        # document happily cited 225 ILCS 60/54.2 six months after it lapsed.
        self.certify(certify_on)

        executions = self.audit.delegation_evidence(
            staff_id=staff_id,
            since=since.isoformat() if since else None,
            until=until.isoformat() if until else None,
        )
        gaps = self.audit.unevidenced_supervision(
            since=since.isoformat() if since else None,
            until=until.isoformat() if until else None,
            staff_id=staff_id,
        )
        members = (
            [self.competencies.staff[staff_id]]
            if staff_id and staff_id in self.competencies.staff
            else sorted(self.competencies.staff.values(), key=lambda m: m.staff_id)
        )
        return {
            "rules_version": self.rules.version,
            "framework": self.rules.framework.get("citation", ""),
            "staff": [m.as_dict() for m in members],
            "certified_on": certify_on.isoformat(),
            "window": [
                since.isoformat() if since else None,
                until.isoformat() if until else None,
            ],
            "competency_records": [
                r.as_dict() for r in self.competencies.records
                if (staff_id is None or r.staff_id == staff_id)
                # Date-scoped too. `since`/`until` used to filter executions and
                # silently return every competency record ever held.
                and (since is None or r.verified_on >= since)
                and (until is None or r.verified_on <= until)
            ],
            "standing_orders": [
                order.as_dict()
                for versions in self.orders.versions.values()
                for order in versions
            ],
            "executions": len(executions),
            "unevidenced": len(gaps),
            "unevidenced_detail": [
                {
                    "timestamp": row["timestamp_utc"],
                    "staff_id": row["staff_id"],
                    "task_code": row["task_code"],
                    "break_glass": bool(row["break_glass"]),
                    "reason": row["break_glass_reason"],
                }
                for row in gaps[:20]
            ],
        }

    def readiness(self, on: date) -> dict[str, Any]:
        """Everything that will stop somebody working, before it does."""
        state, message = self.rules.sunset_state(on)
        return {
            "rules_version": self.rules.version,
            "framework_state": state,
            "framework_message": message,
            "competency_tasks": self.competencies.expiring(on),
            "order_review_tasks": self.orders.review_tasks(on),
        }

    def certify(self, on: date) -> None:
        """Raise if this register cannot claim compliance on `on`.

        Called before producing anything that asserts the practice is compliant.
        After the sunset it refuses: a document citing a repealed statute looks
        like an answer and is not one.
        """
        if not self.rules.has_owner:
            from .register import UnreviewedRules

            raise UnreviewedRules(
                "config/delegation_rules.yaml has no named owner. The delegating "
                "physician owns this file; an unowned delegation framework is "
                "the binder problem with better formatting."
            )
        state, message = self.rules.sunset_state(on)
        if state == "expired":
            from .register import FrameworkSunset

            raise FrameworkSunset(message)


def _hours(count: float):
    from datetime import timedelta

    return timedelta(hours=count)
