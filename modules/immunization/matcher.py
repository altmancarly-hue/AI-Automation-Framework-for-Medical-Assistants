"""Deterministic dose reconciliation between the chart and the state registry.

WHY the deterministic layer is large and the model layer is tiny:

Almost every real discrepancy between a practice chart and I-CARE is one of four
mechanical things: the same dose recorded with two different CVX codes for the
same product, the same dose recorded a day or two apart, a combination product
recorded jointly in one source and as components in the other, or a dose the
other source simply does not have. Three of those four are rules. Sending them
to a model would be slower, more expensive, non-deterministic, and would require
a BAA -- to answer a question that `08 == 45` settles.

So this module resolves everything it can prove and escalates only what it
cannot. `AMBIGUOUS` is the output that goes to `adjudicate.py`. In the fixture
set that ships with this module, roughly one pair in eight reaches it.

WHY the merge is conservative:

An unresolved pair means the module does not know whether the child had one dose
or two. Counting both risks a masked gap -- the failure mode README I-02 names
as "a real gap masked by a phantom record". Counting one risks a duplicate dose.
Neither is acceptable to guess at, so `unresolved_antigens` is returned and the
forecaster marks those antigens REQUIRES_REVIEW rather than producing a
confident status. A gap that a human confirms in ten seconds is much cheaper
than either a duplicate injection or a school-deadline surprise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Iterable, Sequence

from . import cvx as cvxlib
from .forecast import AdministeredDose, DosePrecision

__all__ = [
    "Determination",
    "DoseRecord",
    "MatchedPair",
    "AmbiguousPair",
    "Duplicate",
    "Reconciliation",
    "reconcile",
    "DEFAULT_DATE_TOLERANCE_DAYS",
    "AMBIGUITY_WINDOW_DAYS",
]

#: ACIP-adjacent convention and the build plan's number: two records of the same
#: vaccine within four days are the same administration. Registries and charts
#: routinely disagree by a day or two because one records the visit date and the
#: other the transmission date.
DEFAULT_DATE_TOLERANCE_DAYS = 4

#: Beyond the tolerance but inside this window, two doses of the same vaccine
#: are *probably* one dose with a transcription error -- but they could be two
#: real doses given close together. The rules cannot tell. This is the band that
#: goes to adjudication.
AMBIGUITY_WINDOW_DAYS = 45


class Determination:
    MATCH = "MATCH"
    NO_MATCH = "NO_MATCH"
    AMBIGUOUS = "AMBIGUOUS"
    ALL = (MATCH, NO_MATCH, AMBIGUOUS)


@dataclass(frozen=True)
class DoseRecord:
    """One dose as one source recorded it."""

    record_id: str
    cvx: str
    given: date
    source: str                       # "chart" | "registry"
    precision: str = DosePrecision.DAY
    lot: str | None = None
    product_text: str | None = None   # free text as transcribed, if any

    @property
    def normalised_cvx(self) -> str | None:
        return cvxlib.normalise_code(self.cvx)

    @property
    def known(self) -> bool:
        return cvxlib.is_known(self.cvx)

    @property
    def antigens(self) -> tuple[str, ...]:
        return cvxlib.components_for(self.cvx)

    def to_administered(self) -> AdministeredDose:
        return AdministeredDose(
            cvx=self.normalised_cvx or self.cvx,
            given=self.given,
            source=self.source,
            precision=self.precision,
            lot=self.lot,
            record_id=self.record_id,
        )


@dataclass(frozen=True)
class MatchedPair:
    chart: DoseRecord
    registry: DoseRecord
    date_delta_days: int
    rule: str


@dataclass(frozen=True)
class AmbiguousPair:
    chart: DoseRecord
    registry: DoseRecord
    reason: str
    date_delta_days: int | None
    antigens: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "chart_record_id": self.chart.record_id,
            "registry_record_id": self.registry.record_id,
            "reason": self.reason,
            "date_delta_days": self.date_delta_days,
            "antigens": list(self.antigens),
        }


@dataclass(frozen=True)
class Duplicate:
    """Two doses of the same vaccine, in the same source, within tolerance."""

    first: DoseRecord
    second: DoseRecord
    source: str
    date_delta_days: int


@dataclass
class Reconciliation:
    matched: list[MatchedPair] = field(default_factory=list)
    ambiguous: list[AmbiguousPair] = field(default_factory=list)
    chart_only: list[DoseRecord] = field(default_factory=list)
    registry_only: list[DoseRecord] = field(default_factory=list)
    duplicates: list[Duplicate] = field(default_factory=list)
    unknown_codes: list[str] = field(default_factory=list)

    @property
    def unresolved_antigens(self) -> set[str]:
        """Antigens named by an ambiguous pair, whose dose count is unsettled.

        Note what this does NOT cover: a dose carrying an unrecognised code
        could belong to any antigen, so no antigen is safe to conclude anything
        about. That is a whole-patient condition rather than a per-antigen one
        and `has_unknown_codes` reports it; `pipeline.apply_reconciliation_holds`
        consumes both.
        """
        out: set[str] = set()
        for pair in self.ambiguous:
            out.update(pair.antigens)
        return out

    @property
    def has_unknown_codes(self) -> bool:
        return bool(self.unknown_codes)

    def merged_doses(self) -> list[AdministeredDose]:
        """The union, with each matched pair contributing exactly once.

        The chart copy wins a matched pair, except when the registry copy has a
        more precise date. WHY the chart by default: it is the record the
        practice is legally responsible for and the one a clinician will open.
        WHY precision overrides that: a full date beats a year, whichever side
        it came from, because the forecaster can actually use it.
        """
        rank = {DosePrecision.DAY: 0, DosePrecision.MONTH: 1, DosePrecision.YEAR: 2}
        doses: list[AdministeredDose] = []
        for pair in self.matched:
            winner = (
                pair.registry
                if rank[pair.registry.precision] < rank[pair.chart.precision]
                else pair.chart
            )
            doses.append(winner.to_administered())
        seen_ambiguous: set[str] = set()
        for pair in self.ambiguous:
            # Take one side only, and each record only once -- a combination
            # product paired against three component rows must contribute one
            # dose, not three. Taking both sides would manufacture doses the
            # child may never have had; taking neither would erase one they
            # certainly did. The antigen is held for review either way, so this
            # choice affects a display count, not a clinical conclusion.
            key = (
                pair.chart.record_id
                if _combination(pair.chart)
                else pair.registry.record_id
                if _combination(pair.registry)
                else pair.chart.record_id
            )
            if key in seen_ambiguous:
                continue
            seen_ambiguous.add(key)
            winner = pair.registry if _combination(pair.registry) else pair.chart
            doses.append(winner.to_administered())
        for record in self.chart_only + self.registry_only:
            doses.append(record.to_administered())
        return sorted(doses, key=lambda d: (d.given, d.cvx))

    def summary(self) -> dict[str, int]:
        return {
            "matched": len(self.matched),
            "ambiguous": len(self.ambiguous),
            "chart_only": len(self.chart_only),
            "registry_only": len(self.registry_only),
            "duplicates": len(self.duplicates),
            "unknown_codes": len(self.unknown_codes),
        }


def _combination(record: DoseRecord) -> bool:
    product = cvxlib.product_for(record.cvx)
    return bool(product and product.is_combination)


def _coarse_equal(a: DoseRecord, b: DoseRecord) -> bool:
    """Do two dates agree at the coarser of the two precisions?"""
    if DosePrecision.YEAR in (a.precision, b.precision):
        return a.given.year == b.given.year
    if DosePrecision.MONTH in (a.precision, b.precision):
        return (a.given.year, a.given.month) == (b.given.year, b.given.month)
    return a.given == b.given


def _find_duplicates(
    records: Sequence[DoseRecord], tolerance: int
) -> list[Duplicate]:
    """Two administrations of the same vaccine days apart, in one source.

    This is the "avoided duplicate doses" line in the README's benefit model. It
    is reported, never auto-resolved: one of the two rows might be a data-entry
    error and one might be a genuine second dose given too early, and those have
    completely different remedies.
    """
    found: list[Duplicate] = []
    ordered = sorted(records, key=lambda r: (r.given, r.record_id))
    for i, first in enumerate(ordered):
        for second in ordered[i + 1 :]:
            delta = (second.given - first.given).days
            if delta > tolerance:
                break
            if cvxlib.same_antigen_set(first.cvx, second.cvx):
                found.append(Duplicate(first, second, first.source, delta))
    return found


def reconcile(
    chart: Iterable[DoseRecord],
    registry: Iterable[DoseRecord],
    *,
    tolerance_days: int = DEFAULT_DATE_TOLERANCE_DAYS,
    ambiguity_window_days: int = AMBIGUITY_WINDOW_DAYS,
) -> Reconciliation:
    """Reconcile two dose lists. Only genuinely ambiguous pairs escalate.

    The pairing is greedy over candidate pairs sorted by (date delta, record
    ids). WHY greedy rather than an optimal assignment: the candidate sets are a
    handful of doses per antigen, deltas are zero to four days, and ties are
    broken deterministically by record id so the result is reproducible. An
    optimal matching would give the same answer at more cost and less clarity.
    """
    chart_list = list(chart)
    registry_list = list(registry)

    result = Reconciliation()
    result.unknown_codes = cvxlib.unknown_codes(
        [r.cvx for r in chart_list + registry_list]
    )
    result.duplicates = _find_duplicates(chart_list, tolerance_days) + _find_duplicates(
        registry_list, tolerance_days
    )

    tolerance = timedelta(days=tolerance_days)
    window = timedelta(days=ambiguity_window_days)

    # -- pass 1: exact rule-provable matches -------------------------------
    candidates: list[tuple[int, str, str, DoseRecord, DoseRecord]] = []
    for c in chart_list:
        for r in registry_list:
            if not (c.known and r.known):
                continue
            if not cvxlib.same_antigen_set(c.cvx, r.cvx):
                continue
            if DosePrecision.DAY not in (c.precision, r.precision):
                continue
            if c.precision != DosePrecision.DAY or r.precision != DosePrecision.DAY:
                continue
            delta = abs((c.given - r.given).days)
            if delta <= tolerance.days:
                candidates.append((delta, c.record_id, r.record_id, c, r))

    used_chart: set[str] = set()
    used_registry: set[str] = set()
    for delta, _, _, c, r in sorted(candidates, key=lambda t: (t[0], t[1], t[2])):
        if c.record_id in used_chart or r.record_id in used_registry:
            continue
        used_chart.add(c.record_id)
        used_registry.add(r.record_id)
        rule = (
            "identical CVX and date"
            if c.normalised_cvx == r.normalised_cvx and delta == 0
            else "equivalent antigen set within date tolerance"
        )
        result.matched.append(MatchedPair(c, r, delta, rule))

    remaining_chart = [c for c in chart_list if c.record_id not in used_chart]
    remaining_registry = [r for r in registry_list if r.record_id not in used_registry]

    # -- pass 2: everything the rules cannot settle ------------------------
    #
    # A combination product may legitimately pair with SEVERAL opposite-side
    # rows -- Pediarix against three registry component rows on the same day is
    # one clinical event recorded two ways, and pairing it with only the first
    # component would leave the other two looking like doses the chart is
    # missing. So a combination record is allowed to appear in more than one
    # ambiguous pair; every other kind of ambiguity pairs one-to-one.
    ambiguous_chart: set[str] = set()
    ambiguous_registry: set[str] = set()
    def _is_combo(record: DoseRecord) -> bool:
        product = cvxlib.product_for(record.cvx)
        return bool(product and product.is_combination)

    for c in remaining_chart:
        for r in remaining_registry:
            reason, delta = _ambiguity(c, r, tolerance, window)
            if reason is None:
                continue
            # A record may take part in several pairs only if IT is the
            # combination product being split. The asymmetry matters: a
            # registry-recorded Pediarix against three chart component rows is
            # the same clinical situation as the reverse, and capping the
            # registry side at one pair would dump two chart rows into
            # `chart_only` that the registry demonstrably has.
            if c.record_id in ambiguous_chart and not _is_combo(c):
                break
            if r.record_id in ambiguous_registry and not _is_combo(r):
                continue
            antigens = tuple(sorted(set(c.antigens) | set(r.antigens)))
            result.ambiguous.append(AmbiguousPair(c, r, reason, delta, antigens))
            ambiguous_chart.add(c.record_id)
            ambiguous_registry.add(r.record_id)

    result.chart_only = [
        c for c in remaining_chart if c.record_id not in ambiguous_chart
    ]
    result.registry_only = [
        r for r in remaining_registry if r.record_id not in ambiguous_registry
    ]
    return result


def _ambiguity(
    c: DoseRecord, r: DoseRecord, tolerance: timedelta, window: timedelta
) -> tuple[str | None, int | None]:
    """Classify a leftover pair. Returns (reason, delta) or (None, None).

    Each branch is a real recording pattern, not a hypothetical:

      * combination vs components -- Pediarix in the chart, DTaP + HepB + IPV in
        the registry, same day. The rules can see the overlap but cannot prove
        the registry rows came from the same syringe.
      * a partial date that is consistent at its own precision -- "MMR 2019" in
        a transferred-in paper record against a registry row dated 2019-06-04.
      * same vaccine, date apart by more than the tolerance but less than the
        window -- either a transcription slip or two genuine doses.
      * an unrecognised code on one side against a plausible dose on the other.
    """
    delta = abs((c.given - r.given).days)

    if not c.known or not r.known:
        if delta <= window.days:
            return ("unrecognised vaccine code on one side", delta)
        return (None, None)

    if cvxlib.shares_any_antigen(c.cvx, r.cvx) and delta <= tolerance.days:
        return (
            "combination product recorded jointly in one source and as "
            "components in the other",
            delta,
        )

    if DosePrecision.DAY not in (c.precision, r.precision) or (
        c.precision != DosePrecision.DAY or r.precision != DosePrecision.DAY
    ):
        if (
            cvxlib.same_antigen_set(c.cvx, r.cvx) or cvxlib.shares_any_antigen(c.cvx, r.cvx)
        ) and _coarse_equal(c, r):
            return ("partial date consistent at the recorded precision", None)
        return (None, None)

    if cvxlib.same_antigen_set(c.cvx, r.cvx) and tolerance.days < delta <= window.days:
        return (
            "same vaccine recorded more than the date tolerance apart; either a "
            "transcription error or two distinct administrations",
            delta,
        )

    return (None, None)
