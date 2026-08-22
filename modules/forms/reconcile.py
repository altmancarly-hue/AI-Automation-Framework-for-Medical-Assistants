"""The immunization grid, reconciled — by calling I-02, not by rewriting it.

The build plan is one line on this file: *"Reuse modules/immunization/matcher.py.
Do not reimplement."* That instruction is worth more than it looks. The CVX
matcher, the four-day tolerance, the combination-product logic, the A.2
adjudication prompt with its `UNCERTAIN` outcome, and the rule that no machine
determination reaches a chart without a named reviewer are all built, tested and
reviewed in I-02. A second copy here would drift, and the copy that drifted would
be the one writing dates onto a legal school document.

So this module is an adapter and a policy, and the policy is the interesting
half:

**A DOSE NOBODY HAS SETTLED DOES NOT GO ON THE FORM.** I-02's job is to tell you
what is certain and what is not. This module's job is to make sure only the first
kind is printed. An ambiguous pair, an unknown CVX code, a duplicate within the
same source -- every one of them marks its antigen DISPUTED, and a disputed
antigen's boxes are left blank with the reason attached, rather than filled with
whichever side happened to sort first.

That is not caution for its own sake. README I-01's failure table:

    Missed dose given elsewhere and not reconciled | Child receives an
    unnecessary duplicate vaccine, or is wrongly flagged non-compliant

Both halves of that are caused by printing a number the sources disagree about.
A blank box with "chart and registry disagree, see reconciliation queue" beside
it costs the MA thirty seconds. A confidently wrong date costs a school year.

**THE REGISTRY BEING DOWN IS A STATE, NOT AN ERROR.** README I-01 again: *"I-CARE
interface unavailable or not licensed | degrade gracefully to chart-only fill,
flag the form as 'registry not reconciled', continue to function."* So
`reconcile_for_form` takes registry doses that may be None, and the difference
between "the registry says nothing" and "the registry was not asked" is carried
all the way to the review screen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Mapping, Sequence

from modules.immunization import cvx as cvxlib
from modules.immunization.forecast import DosePrecision
from modules.immunization.matcher import DoseRecord, Reconciliation, reconcile

from .chart import SourceValue

__all__ = [
    "ANTIGEN_FIELD_GROUPS",
    "GridDose",
    "GridAntigen",
    "FormReconciliation",
    "reconcile_for_form",
    "build_immunization_block",
]

#: Which template field prefix each antigen group fills, and which I-02 antigen
#: names satisfy it. A school form's "DTaP / DTP / DT" row is one column of boxes
#: covering several products; I-02 works in antigens. This mapping is the only
#: place the two vocabularies meet.
#:
#: Membership is ANY-of, not all-of. A dose belongs in a row if it delivers any
#: of that row's antigens, which is what makes a combination product land in
#: every row it belongs to: Pentacel is dtap + ipv + hib, and a school form has a
#: box for each of those three. All-of membership would have put it in none of
#: them and quietly lost three doses off the grid.
#:
#: It is a module constant rather than config because it is a statement about
#: I-02's antigen vocabulary. `_assert_groups_are_known()` runs at import so a
#: rename in `cvx.Antigen` fails here immediately rather than silently emptying
#: a row.
ANTIGEN_FIELD_GROUPS: Mapping[str, frozenset[str]] = {
    "dtap": frozenset({cvxlib.Antigen.DTAP}),
    # The adolescent booster row. Td and Tdap both fill it; a childhood DTaP
    # does NOT, which is why dtap has its own row above.
    "tdap": frozenset({cvxlib.Antigen.TDAP, cvxlib.Antigen.TD}),
    "polio": frozenset({cvxlib.Antigen.IPV}),
    "hib": frozenset({cvxlib.Antigen.HIB}),
    "mmr": frozenset({cvxlib.Antigen.MMR}),
    "hepb": frozenset({cvxlib.Antigen.HEPB}),
    "varicella": frozenset({cvxlib.Antigen.VAR}),
    # One row per PRODUCT FAMILY, not per rough clinical area. PPSV23 is a
    # polysaccharide vaccine and PCV13 is a conjugate; MenB is not MenACWY, and
    # an Illinois school requirement written for MenACWY is not satisfied by
    # one. Folding them together printed a PPSV23 date in a box the form labels
    # "Pneumococcal conjugate - dose 3", which reads as a completed conjugate
    # series that the child has not had.
    "pcv": frozenset({cvxlib.Antigen.PCV}),
    "ppsv": frozenset({cvxlib.Antigen.PPSV}),
    "hepa": frozenset({cvxlib.Antigen.HEPA}),
    "mcv": frozenset({cvxlib.Antigen.MENACWY}),
    "menb": frozenset({cvxlib.Antigen.MENB}),
}


def _assert_groups_are_known() -> None:
    known = set(cvxlib.Antigen.ALL)
    unknown = {
        antigen
        for members in ANTIGEN_FIELD_GROUPS.values()
        for antigen in members
        if antigen not in known
    }
    if unknown:
        raise RuntimeError(
            f"ANTIGEN_FIELD_GROUPS names antigen(s) I-02 does not have: "
            f"{sorted(unknown)}. A grid row mapped to a name that no longer "
            "exists is a row that silently stays empty on every form."
        )


_assert_groups_are_known()


#: Prefix on a hold reason that came from an unknown CVX code rather than from a
#: conflict in that row. See `FormReconciliation.discrepancies`.
_UNKNOWN_CODE_MARK = "[unknown-code]"


@dataclass(frozen=True)
class GridDose:
    """One dose as it would appear in one box of the immunization grid."""

    given: date
    precision: str
    #: "chart" | "registry" | "both"
    system: str
    record_ids: tuple[str, ...]
    cvx: str
    product_text: str | None = None
    disputed: bool = False
    dispute_reason: str = ""

    def as_source_value(self) -> SourceValue:
        """The shape `templates.dose_date` and the review screen both consume."""
        return SourceValue(
            value={
                "given": self.given,
                "precision": self.precision,
                "disputed": self.disputed,
                "cvx": self.cvx,
            },
            system="reconciled" if self.system == "both" else self.system,
            resource=", ".join(self.record_ids),
            recorded=self.given,
            derived_from=tuple(self.record_ids),
            # A dose is a historical event, not a measurement that goes stale.
            historical=True,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "given": self.given.isoformat(),
            "precision": self.precision,
            "system": self.system,
            "cvx": self.cvx,
            "product_text": self.product_text,
            "record_ids": list(self.record_ids),
            "disputed": self.disputed,
            "dispute_reason": self.dispute_reason,
        }


@dataclass
class GridAntigen:
    """One row of the grid: an antigen group and its doses in date order."""

    group: str
    doses: list[GridDose] = field(default_factory=list)
    disputed: bool = False
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "disputed": self.disputed,
            "reasons": list(self.reasons),
            "doses": [d.as_dict() for d in self.doses],
        }


@dataclass
class FormReconciliation:
    """What I-02 concluded, arranged the way a form needs it."""

    reconciliation: Reconciliation
    antigens: dict[str, GridAntigen] = field(default_factory=dict)
    #: True when the registry was never asked or did not answer. NOT the same as
    #: a registry that answered with nothing.
    registry_consulted: bool = True
    registry_note: str = ""

    @property
    def disputed_groups(self) -> tuple[str, ...]:
        return tuple(sorted(g for g, a in self.antigens.items() if a.disputed))

    @property
    def fully_reconciled(self) -> bool:
        return self.registry_consulted and not self.disputed_groups

    def discrepancies(self) -> list[dict[str, Any]]:
        """Everything a person has to settle, worst first.

        This is the list README I-01 puts at the TOP of the review screen, above
        the form. A discrepancy buried under sixty filled boxes is a discrepancy
        nobody reads.
        """
        rows: list[dict[str, Any]] = []
        if not self.registry_consulted:
            rows.append(
                {
                    "severity": "blocking",
                    "kind": "registry_not_reconciled",
                    "group": "",
                    "detail": self.registry_note
                    or "the state registry was not consulted; this form is "
                    "filled from the chart alone",
                }
            )
        if self.reconciliation.unknown_codes:
            rows.append(
                {
                    "severity": "blocking",
                    "kind": "unknown_cvx",
                    "group": "",
                    "detail": (
                        "this record carries vaccine code(s) this system does "
                        f"not know ({', '.join(self.reconciliation.unknown_codes)}). "
                        "A dose of unknown antigen makes every row of the grid "
                        "unsafe to conclude anything about."
                    ),
                }
            )
        for group in self.disputed_groups:
            antigen = self.antigens[group]
            reasons = [r for r in antigen.reasons if _UNKNOWN_CODE_MARK not in r]
            if not reasons:
                # This row is held ONLY because of the unknown code already
                # reported above. Repeating it ten times -- once per row --
                # buries the one blocker that matters under nine copies of
                # itself, which is how a discrepancy list stops being read.
                continue
            rows.append(
                {
                    "severity": "blocking",
                    "kind": "unsettled_antigen",
                    "group": group,
                    "detail": "; ".join(reasons),
                }
            )
        for dose in _registry_only_doses(self):
            rows.append(
                {
                    "severity": "advisory",
                    "kind": "registry_only_dose",
                    "group": dose.group,
                    "detail": (
                        f"the registry holds a dose the chart does not "
                        f"({dose.given.isoformat()}); it is printed on the form "
                        "and should also be filed to the chart"
                    ),
                }
            )
        return rows

    def as_dict(self) -> dict[str, Any]:
        return {
            "registry_consulted": self.registry_consulted,
            "registry_note": self.registry_note,
            "fully_reconciled": self.fully_reconciled,
            "disputed_groups": list(self.disputed_groups),
            "antigens": {g: a.as_dict() for g, a in sorted(self.antigens.items())},
            "discrepancies": self.discrepancies(),
        }


@dataclass(frozen=True)
class _RegistryOnly:
    group: str
    given: date


def _registry_only_doses(result: FormReconciliation) -> list[_RegistryOnly]:
    out: list[_RegistryOnly] = []
    for group, antigen in sorted(result.antigens.items()):
        for dose in antigen.doses:
            if dose.system == "registry" and not dose.disputed:
                out.append(_RegistryOnly(group=group, given=dose.given))
    return out


def _groups_for(code: str) -> tuple[str, ...]:
    """Which grid rows a CVX code belongs in.

    A combination product lands in more than one: Pentacel is DTaP AND polio AND
    Hib, and a school form has a box for each. Getting this wrong in the
    permissive direction prints a dose the child did not have; getting it wrong
    in the strict direction hides one they did. So it is derived from I-02's own
    antigen expansion rather than from a hand-written code list.

    An unknown code yields NO groups, which is correct and is why
    `reconcile_for_form` holds every row when I-02 reports unknown codes: a dose
    that lands nowhere is not a dose that does not count.
    """
    antigens = set(cvxlib.components_for(code))
    if not antigens:
        return ()
    return tuple(
        sorted(
            group
            for group, members in ANTIGEN_FIELD_GROUPS.items()
            if antigens & members
        )
    )


def reconcile_for_form(
    chart_doses: Sequence[DoseRecord],
    registry_doses: Sequence[DoseRecord] | None,
    *,
    registry_note: str = "",
) -> FormReconciliation:
    """Run I-02's reconciliation and arrange the result as grid rows.

    `registry_doses=None` means the registry was not consulted -- not that it
    had nothing. The form is still produced, from the chart alone, flagged.
    """
    chart_list = list(chart_doses)
    consulted = registry_doses is not None
    registry_list = list(registry_doses or ())

    result = reconcile(chart_list, registry_list)

    antigens: dict[str, GridAntigen] = {
        group: GridAntigen(group=group) for group in ANTIGEN_FIELD_GROUPS
    }

    def place(dose: GridDose) -> None:
        for group in _groups_for(dose.cvx):
            antigens[group].doses.append(dose)

    for pair in result.matched:
        # The chart copy wins, except where the registry has the more precise
        # date -- the same rule I-02's own `merged_doses` applies, for the same
        # reason. Reimplementing the preference here would be exactly the drift
        # the build plan warned about, so the winner is picked by the same test.
        #
        # NOTE this branch is currently unreachable: I-02's rule-provable pass
        # requires DAY precision on BOTH sides, so a matched pair never differs.
        # It is kept, and kept identical to I-02's, because the alternative is a
        # silent wrong answer the day that rule is relaxed. A month-precision
        # pair reaches `result.ambiguous` instead and is placed below, marked.
        rank = {DosePrecision.DAY: 0, DosePrecision.MONTH: 1, DosePrecision.YEAR: 2}
        winner = (
            pair.registry
            if rank[pair.registry.precision] < rank[pair.chart.precision]
            else pair.chart
        )
        place(
            GridDose(
                given=winner.given,
                precision=str(winner.precision),
                system="both",
                record_ids=(pair.chart.record_id, pair.registry.record_id),
                cvx=winner.normalised_cvx or winner.cvx,
                product_text=winner.product_text,
            )
        )

    # An AMBIGUOUS pair is two records that might be one event. Both sides go
    # on the grid, marked, because the review screen's whole job here is to show
    # the MA what the machine could not settle. An earlier version placed only
    # matched and single-source doses, so a disputed row arrived at the review
    # screen empty -- "we could not settle DTaP" with nothing to look at.
    for pair in result.ambiguous:
        for record, system in ((pair.chart, "chart"), (pair.registry, "registry")):
            place(
                GridDose(
                    given=record.given,
                    precision=str(record.precision),
                    system=system,
                    record_ids=(record.record_id,),
                    cvx=record.normalised_cvx or record.cvx,
                    product_text=record.product_text,
                    disputed=True,
                    dispute_reason="this dose is one side of an unsettled pair",
                )
            )

    for record, system in (
        [(r, "chart") for r in result.chart_only]
        + [(r, "registry") for r in result.registry_only]
    ):
        place(
            GridDose(
                given=record.given,
                precision=str(record.precision),
                system=system,
                record_ids=(record.record_id,),
                cvx=record.normalised_cvx or record.cvx,
                product_text=record.product_text,
            )
        )

    # -- now the holds ------------------------------------------------------

    def hold(group: str, reason: str) -> None:
        antigen = antigens.get(group)
        if antigen is None:
            return
        antigen.disputed = True
        if reason not in antigen.reasons:
            antigen.reasons.append(reason)
        for index, dose in enumerate(antigen.doses):
            antigen.doses[index] = GridDose(
                **{**dose.__dict__, "disputed": True, "dispute_reason": reason}
            )

    for antigen_name in sorted(result.unresolved_antigens):
        for group, members in ANTIGEN_FIELD_GROUPS.items():
            if antigen_name in members:
                hold(
                    group,
                    f"the chart and the registry hold doses of {antigen_name} "
                    "that could be one administration or two; I-02 could not "
                    "settle it and nobody has yet",
                )

    for duplicate in result.duplicates:
        for group in _groups_for(duplicate.first.normalised_cvx or duplicate.first.cvx):
            hold(
                group,
                f"{duplicate.source} holds two doses "
                f"{duplicate.date_delta_days} day(s) apart; one of them is "
                "probably a double entry, and which one changes the dose count",
            )

    if result.unknown_codes:
        # Marked so `discrepancies()` can tell a row held by the unknown code
        # apart from a row with a genuine chart/registry conflict of its own.
        # A dose of unknown antigen could belong to any row, so no row is safe.
        # This is I-02's own reasoning (`Reconciliation.unresolved_antigens`
        # documents why unknown codes are a whole-patient condition) applied to
        # a document that will be read as a complete record.
        for group in antigens:
            hold(
                group,
                f"{_UNKNOWN_CODE_MARK} this patient has a dose with a vaccine "
                f"code the system does not know ({', '.join(result.unknown_codes)}); "
                "until it is identified no row of the grid can be called complete",
            )

    for antigen in antigens.values():
        antigen.doses.sort(key=lambda d: (d.given, d.cvx))

    return FormReconciliation(
        reconciliation=result,
        antigens=antigens,
        registry_consulted=consulted,
        registry_note=registry_note
        or ("" if consulted else "the state registry was not consulted"),
    )


def grid_capacity_findings(
    template: Any, result: FormReconciliation
) -> list[dict[str, Any]]:
    """Doses that have no box on this form.

    A form is a fixed number of boxes and a chart is not. Nothing used to
    compare the two, and because rows are sorted oldest-first it was always the
    MOST RECENT dose -- the one a school or a camp actually checks -- that fell
    off the end: no skipped field, no discrepancy, no blocker. A teenager with a
    2024 Tdap and a 2019 Td printed the 2019 date in the form's single Tdap box
    and the 2024 booster appeared nowhere on the document.

    Every dose without a box is reported as blocking. Which doses a short form
    should carry is a decision for a person, and this is the point at which they
    find out there is one to make.
    """
    boxes: dict[str, int] = {}
    for spec in getattr(template, "fields", ()):  # duck-typed for testability
        if not spec.source.startswith("immunizations."):
            continue
        parts = spec.source.split(".")
        if len(parts) < 2:
            continue
        boxes[parts[1]] = boxes.get(parts[1], 0) + 1

    findings: list[dict[str, Any]] = []
    for group, antigen in sorted(result.antigens.items()):
        capacity = boxes.get(group, 0)
        if not antigen.doses or capacity >= len(antigen.doses):
            continue
        if capacity == 0:
            # The form has no box for this antigen at all, which is normal --
            # a camp form does not ask about Hib. Only worth saying when there
            # are doses to lose AND the form asks about the antigen elsewhere.
            continue
        overflow = antigen.doses[capacity:]
        findings.append(
            {
                "severity": "blocking",
                "kind": "more_doses_than_boxes",
                "group": group,
                "detail": (
                    f"the {group} row holds {len(antigen.doses)} dose(s) and this "
                    f"form has {capacity} box(es); "
                    + ", ".join(d.given.isoformat() for d in overflow)
                    + " are not on the document"
                ),
            }
        )
    return findings


def build_immunization_block(result: FormReconciliation) -> dict[str, Any]:
    """The `immunizations` sub-tree of a `ChartRecord`, ready for templates.

    Template source paths read `immunizations.dtap.0`; this is what they walk
    into. Disputed doses are present and MARKED rather than removed, so the
    review screen can show the MA what the machine found and why it refused to
    print it. `templates.dose_date` renders a disputed dose as None, and
    `fill.py` refuses it a second time.
    """
    block: dict[str, Any] = {}
    for group, antigen in result.antigens.items():
        block[group] = [dose.as_source_value() for dose in antigen.doses]
    return block
