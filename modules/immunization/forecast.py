"""CDSi-lite forecaster. A generic evaluator over a declarative rules table.

READ THIS BEFORE CHANGING ANYTHING IN THIS FILE.

**No language model is used here and none ever may be.** README I-02: "Using an
LLM to determine whether a child is due for a vaccine would be actively
negligent. It is a rules problem with an authoritative published rule set, and
the correct answer is a rules engine." A test asserts no model import reaches
this package's forecasting path.

**There is no schedule knowledge in this file.** Every age, interval, series
length and age-out lives in `config/immunization_schedule.yaml`. If you find
yourself adding `if antigen == "hpv"` here, the rules table is missing an
expressive feature and that is what needs extending. The separation is the
whole point: a clinician can review a YAML diff and sign it off; nobody is
going to code-review a nest of elifs for ACIP compliance.

**This engine is not the authority by default.** README I-02 warns against
implementing schedule logic yourself, and it is right. `CrossCheckForecaster`
lets I-CARE's own CDSi forecast be primary with this engine running alongside to
flag disagreement. Using this engine as the sole authority requires a recorded
validation run (`validate_against_reference`), and `recall.py` refuses to send
outbound messages until one exists. That is the README's "validate against 200
known-good records before go-live" turned into something the code enforces
rather than something a runbook asks for.

The evaluator implements: minimum age, minimum interval from the previous valid
dose, minimum interval from the first dose, the ACIP four-day grace period,
conditional series selection (HPV 2-dose vs 3-dose by age at first dose),
product-dependent series length (Hib PRP-OMP vs PRP-T, Rotarix vs RotaTeq),
early-completion rules (a DTaP 4th dose on or after the 4th birthday completes
the series), age-outs, and annual antigens.
"""

from __future__ import annotations

import os
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Iterable, Mapping, Protocol, Sequence

import yaml

from .cvx import Antigen, components_for, is_known, normalise_code, product_for

__all__ = [
    "Status",
    "DosePrecision",
    "AdministeredDose",
    "DoseEvaluation",
    "AntigenForecast",
    "PatientForecast",
    "Schedule",
    "Forecaster",
    "LocalRulesForecaster",
    "RegistryForecaster",
    "CrossCheckForecaster",
    "Disagreement",
    "ValidationResult",
    "validate_against_reference",
    "add_period",
    "age_in_days",
    "DEFAULT_SCHEDULE_PATH",
]

DEFAULT_SCHEDULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "immunization_schedule.yaml",
)


class Status:
    COMPLETE = "complete"
    NOT_YET_DUE = "not_yet_due"
    DUE = "due"
    OVERDUE = "overdue"
    AGED_OUT = "aged_out"
    NOT_REQUIRED = "not_required"
    #: A partial date, an unknown code, or a contradiction the rules cannot
    #: resolve. NEVER auto-recalled. A human looks at it.
    REQUIRES_REVIEW = "requires_review"
    ALL = (COMPLETE, NOT_YET_DUE, DUE, OVERDUE, AGED_OUT, NOT_REQUIRED, REQUIRES_REVIEW)
    #: The statuses that represent an open gap for the recall engine.
    OPEN_GAP = (DUE, OVERDUE)


class DosePrecision:
    """How much of an administration date the source actually recorded.

    WHY this is modelled rather than coerced: transferred-in paper records
    routinely carry "MMR 2019" with no month. Coercing that to 2019-01-01 in
    order to make the interval arithmetic run invents a fact, and the invented
    fact can validate a dose that was actually given too early. A partial date
    makes the antigen REQUIRES_REVIEW, which is the honest answer.
    """

    DAY = "day"
    MONTH = "month"
    YEAR = "year"
    ALL = (DAY, MONTH, YEAR)


# --------------------------------------------------------------------------
# Date arithmetic
# --------------------------------------------------------------------------


def add_period(base: date, period: Mapping[str, int] | None) -> date:
    """Add {years, months, weeks, days} to a date, clamping at month end.

    Years and months are calendar arithmetic because ACIP states them that way
    ("12 months of age", not "365 days"). Weeks and days are exact, for the same
    reason ("6 weeks", "4 days grace"). Mixing the two is not sloppiness; it is
    what the source specification does, and normalising everything to days would
    make a child born on 29 February late for their 12-month vaccines.
    """
    if not period:
        return base
    year = base.year + int(period.get("years", 0))
    month = base.month + int(period.get("months", 0))
    year += (month - 1) // 12
    month = (month - 1) % 12 + 1
    day = min(base.day, monthrange(year, month)[1])
    shifted = date(year, month, day)
    return shifted + timedelta(
        weeks=int(period.get("weeks", 0)), days=int(period.get("days", 0))
    )


def age_in_days(dob: date, as_of: date) -> int:
    return (as_of - dob).days


def _age_years(dob: date, as_of: date) -> int:
    years = as_of.year - dob.year
    if (as_of.month, as_of.day) < (dob.month, dob.day):
        years -= 1
    return years


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AdministeredDose:
    """One dose as recorded by one source, before antigen expansion."""

    cvx: str
    given: date
    source: str = "chart"            # "chart" | "registry" | "reported"
    precision: str = DosePrecision.DAY
    lot: str | None = None
    record_id: str | None = None

    @property
    def antigens(self) -> tuple[str, ...]:
        return components_for(self.cvx)

    @property
    def known(self) -> bool:
        return is_known(self.cvx)


@dataclass(frozen=True)
class _AntigenDose:
    """One antigen component of one administered dose."""

    antigen: str
    cvx: str
    given: date
    source: str
    precision: str
    series_variant: str | None
    record_id: str | None


@dataclass
class DoseEvaluation:
    """What the rules concluded about a single dose."""

    antigen: str
    cvx: str
    given: date
    source: str
    valid: bool
    sequence: int | None          # position in the series, if it counted
    reason: str = ""              # why it did not count, when it did not

    def as_dict(self) -> dict[str, Any]:
        return {
            "antigen": self.antigen,
            "cvx": self.cvx,
            "given": self.given.isoformat(),
            "source": self.source,
            "valid": self.valid,
            "sequence": self.sequence,
            "reason": self.reason,
        }


@dataclass
class AntigenForecast:
    antigen: str
    label: str
    status: str
    doses_valid: int
    doses_required: int
    next_dose: int | None = None
    earliest_date: date | None = None
    recommended_date: date | None = None
    overdue_date: date | None = None
    days_overdue: int = 0
    evaluations: list[DoseEvaluation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    school_required: bool = False
    weight: float = 1.0
    recall_eligible: bool = True
    #: False for antigens that are reported but should not occupy a line on a
    #: clinician's morning sheet. COVID is the case: guidance has changed
    #: repeatedly, it is not an Illinois school requirement, and a "DUE TODAY"
    #: line against every patient every day trains people to skim the section
    #: that also contains the MMR gap.
    huddle_eligible: bool = True
    age_out_hard_date: date | None = None

    @property
    def is_open_gap(self) -> bool:
        return self.status in Status.OPEN_GAP

    @property
    def invalid_doses(self) -> list[DoseEvaluation]:
        return [e for e in self.evaluations if not e.valid]

    def as_dict(self) -> dict[str, Any]:
        return {
            "antigen": self.antigen,
            "label": self.label,
            "status": self.status,
            "doses_valid": self.doses_valid,
            "doses_required": self.doses_required,
            "next_dose": self.next_dose,
            "earliest_date": self.earliest_date.isoformat() if self.earliest_date else None,
            "recommended_date": (
                self.recommended_date.isoformat() if self.recommended_date else None
            ),
            "overdue_date": self.overdue_date.isoformat() if self.overdue_date else None,
            "days_overdue": self.days_overdue,
            "notes": list(self.notes),
            "school_required": self.school_required,
            "invalid_doses": [e.as_dict() for e in self.invalid_doses],
        }


@dataclass
class PatientForecast:
    patient_id: str
    dob: date
    as_of: date
    antigens: dict[str, AntigenForecast]
    unknown_codes: list[str] = field(default_factory=list)
    engine: str = "local_rules"
    schedule_version: str = ""

    @property
    def open_gaps(self) -> list[AntigenForecast]:
        return [a for a in self.antigens.values() if a.is_open_gap]

    @property
    def needs_review(self) -> list[AntigenForecast]:
        return [a for a in self.antigens.values() if a.status == Status.REQUIRES_REVIEW]

    def as_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "dob": self.dob.isoformat(),
            "as_of": self.as_of.isoformat(),
            "engine": self.engine,
            "schedule_version": self.schedule_version,
            "unknown_codes": list(self.unknown_codes),
            "antigens": {k: v.as_dict() for k, v in sorted(self.antigens.items())},
        }


# --------------------------------------------------------------------------
# Schedule
# --------------------------------------------------------------------------


class Schedule:
    """The rules table, loaded once and treated as immutable."""

    def __init__(self, data: Mapping[str, Any]) -> None:
        self.data = data
        self.version = str(data.get("version", "unversioned"))
        self.grace = timedelta(days=int(data.get("grace_period_days", 0)))
        self.antigens: Mapping[str, Any] = data.get("antigens", {})
        self.school_checkpoints = data.get("school_checkpoints", [])
        self.school_month = int(data.get("school_year_start_month", 8))
        self.school_day = int(data.get("school_year_start_day", 15))
        self.influenza_season_start_month = int(
            data.get("influenza_season_start_month", 7)
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str] = DEFAULT_SCHEDULE_PATH) -> "Schedule":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh))

    def rule(self, antigen: str) -> Mapping[str, Any] | None:
        return self.antigens.get(antigen)


# --------------------------------------------------------------------------
# Forecaster interface
# --------------------------------------------------------------------------


class Forecaster(Protocol):
    """Anything that can produce a forecast. I-CARE is one; this engine is one."""

    name: str

    def forecast(
        self,
        *,
        patient_id: str,
        dob: date,
        doses: Sequence[AdministeredDose],
        as_of: date,
    ) -> PatientForecast:
        ...


class RegistryForecaster:
    """Adapter placeholder for I-CARE's own CDSi forecast.

    Deliberately unimplemented and deliberately loud. WHY it exists at all: the
    README's recommendation is to consume I-CARE's forecast rather than compute
    one, and an architecture that has no seam for that recommendation quietly
    becomes an architecture that ignores it. This is the seam. Filling it in is
    an HL7 2.5.1 query and a response parser, and it belongs in the integration
    layer, not here.
    """

    name = "icare_registry"

    def __init__(self, client: Any = None) -> None:
        self.client = client

    def forecast(
        self,
        *,
        patient_id: str,
        dob: date,
        doses: Sequence[AdministeredDose],
        as_of: date,
    ) -> PatientForecast:
        raise NotImplementedError(
            "RegistryForecaster is an integration seam, not an implementation. "
            "Wire I-CARE's HL7 2.5.1 forecast response here and it becomes the "
            "authority, with LocalRulesForecaster running as the cross-check."
        )


class LocalRulesForecaster:
    """Evaluates a dose history against the declarative schedule."""

    name = "local_rules"

    def __init__(self, schedule: Schedule | None = None) -> None:
        self.schedule = schedule or Schedule.load()

    # -- public ------------------------------------------------------------

    def forecast(
        self,
        *,
        patient_id: str,
        dob: date,
        doses: Sequence[AdministeredDose],
        as_of: date,
    ) -> PatientForecast:
        expanded = self._expand(doses)
        unknown = [d.cvx for d in doses if not d.known]
        # De-duplicate while preserving order, so the review queue shows each
        # unrecognised code once rather than once per occurrence.
        unknown = list(dict.fromkeys(normalise_code(c) or c for c in unknown))

        results: dict[str, AntigenForecast] = {}
        for antigen, rule in self.schedule.antigens.items():
            antigen_doses = sorted(
                (d for d in expanded if d.antigen == antigen),
                key=lambda d: d.given,
            )
            results[antigen] = self._evaluate_antigen(
                antigen=antigen,
                rule=rule,
                dob=dob,
                as_of=as_of,
                doses=antigen_doses,
            )

        if unknown:
            # An unrecognised code might be any antigen. Rather than guess which
            # forecast it would have changed, say so at the patient level and
            # let the MA resolve the code. Silently ignoring it would produce a
            # confident recall for a vaccine sitting in the chart.
            for forecast in results.values():
                if forecast.status in Status.OPEN_GAP:
                    forecast.notes.append(
                        "patient record contains an unrecognised vaccine code; "
                        "gap unconfirmed until it is resolved"
                    )

        return PatientForecast(
            patient_id=patient_id,
            dob=dob,
            as_of=as_of,
            antigens=results,
            unknown_codes=unknown,
            engine=self.name,
            schedule_version=self.schedule.version,
        )

    # -- internals ---------------------------------------------------------

    def _expand(self, doses: Iterable[AdministeredDose]) -> list[_AntigenDose]:
        out: list[_AntigenDose] = []
        for dose in doses:
            product = product_for(dose.cvx)
            if product is None:
                continue  # surfaced separately as an unknown code
            for antigen in product.components:
                out.append(
                    _AntigenDose(
                        antigen=antigen,
                        cvx=product.code,
                        given=dose.given,
                        source=dose.source,
                        precision=dose.precision,
                        series_variant=product.series_variant,
                        record_id=dose.record_id,
                    )
                )
        return out

    def _series_for(
        self,
        rule: Mapping[str, Any],
        dob: date,
        doses: Sequence[_AntigenDose],
        *,
        as_of: date,
        first_valid: _AntigenDose | None = None,
    ) -> tuple[list[Mapping[str, Any]], int, list[str]]:
        """Pick the series and required-dose count for this patient.

        Handles conditional series (HPV, by age at the first VALID dose) and
        product variants (Hib, rotavirus, by what was actually administered).

        `first_valid` comes from a first evaluation pass. WHY it cannot be
        `doses[0]`: an invalid dose -- an HPV dose given at age 8, before the
        9-year minimum -- would otherwise choose the series. A child whose real
        first dose is at 15y6m needs three doses; letting a stray age-8 entry
        select the two-dose branch reports them COMPLETE after two and they
        never receive the third.
        """
        notes: list[str] = []
        conditional = rule.get("conditional_series")
        if conditional:
            branches = [b for b in conditional if not b.get("default")]
            default = next((b for b in conditional if b.get("default")), None)
            chosen = None
            if first_valid is not None:
                for branch in branches:
                    threshold = branch.get("when_age_at_first_dose_under")
                    if threshold and first_valid.given < add_period(dob, threshold):
                        chosen = branch
                        break
                if chosen is None:
                    chosen = default
            else:
                # Not started. Forecast the series the child would get if they
                # started TODAY -- an unstarted 16-year-old needs three doses,
                # and telling them two understates the course in the recall
                # message and on the huddle sheet.
                for branch in branches:
                    threshold = branch.get("when_age_at_first_dose_under")
                    if threshold and as_of < add_period(dob, threshold):
                        chosen = branch
                        break
                if chosen is None:
                    chosen = default
                notes.append("series length assumes a start at the current age")
            chosen = chosen or (branches[0] if branches else {})
            series = list(chosen.get("series", []))
            return series, int(chosen.get("doses_required", len(series))), notes

        series = list(rule.get("series", []))
        required = int(rule.get("doses_required", len(series)))
        variants = rule.get("variants") or {}
        if variants:
            longest = max(int(v.get("doses_required", required)) for v in variants.values())
            # A dose recorded as an unspecified formulation (CVX 122 rotavirus,
            # CVX 17 Hib) has NO known variant. Filtering those out and then
            # treating a single remaining variant as authoritative selects the
            # SHORTER series on the strength of one identified dose -- which for
            # rotavirus means declaring a 2-dose series complete and letting the
            # third dose age out permanently at eight months. Unknown means
            # unknown: take the longer schedule.
            unspecified = any(d.series_variant is None for d in doses)
            administered = {d.series_variant for d in doses if d.series_variant}
            if unspecified and doses:
                required = longest
                notes.append(
                    "a dose is recorded as an unspecified formulation; using the "
                    "longer schedule, which is the conservative choice"
                )
            elif len(administered) == 1:
                variant = administered.pop()
                if variant in variants:
                    spec = variants[variant]
                    required = int(spec.get("doses_required", required))
                    if spec.get("series"):
                        series = list(spec["series"])
                    notes.append(f"series length set by product variant {variant}")
            elif len(administered) > 1:
                notes.append(
                    "mixed product variants in the series; using the longer "
                    "schedule, which is the conservative choice"
                )
                required = longest
        return series, required, notes

    def _walk(
        self,
        antigen: str,
        doses: Sequence[_AntigenDose],
        series: Sequence[Mapping[str, Any]],
        dob: date,
    ) -> tuple[list[DoseEvaluation], list[_AntigenDose]]:
        """Evaluate each dose in date order against the series it lands on."""
        evaluations: list[DoseEvaluation] = []
        valid: list[_AntigenDose] = []
        for dose in doses:
            number = len(valid) + 1
            dose_rule = self._dose_rule(series, number)
            reason = self._invalid_reason(dose, dose_rule, dob, valid)
            if reason:
                evaluations.append(
                    DoseEvaluation(antigen, dose.cvx, dose.given, dose.source, False,
                                   None, reason)
                )
                continue
            valid.append(dose)
            evaluations.append(
                DoseEvaluation(antigen, dose.cvx, dose.given, dose.source, True, number)
            )
        return evaluations, valid

    def _dose_rule(self, series: Sequence[Mapping[str, Any]], number: int) -> Mapping[str, Any]:
        for entry in series:
            if int(entry.get("dose", 0)) == number:
                return entry
        return series[-1] if series else {}

    def _min_interval(
        self, dose_rule: Mapping[str, Any], dob: date, reference: date
    ) -> Mapping[str, int] | None:
        """Resolve a conditional interval (varicella) against age at the dose."""
        conditional = dose_rule.get("conditional_interval")
        if not conditional:
            return dose_rule.get("min_interval")
        for branch in conditional:
            if branch.get("default"):
                continue
            threshold = branch.get("when_age_under")
            if threshold and reference < add_period(dob, threshold):
                return branch.get("min_interval")
        for branch in conditional:
            if branch.get("default"):
                return branch.get("min_interval")
        return None

    def _projected_interval(
        self,
        dose_rule: Mapping[str, Any],
        dob: date,
        previous: date,
    ) -> Mapping[str, int] | None:
        """The interval that will actually apply on the day the dose is given.

        Resolving a conditional interval against `as_of` is wrong for a child
        who is days away from the threshold: varicella's three-month interval
        applies under thirteen and four weeks at thirteen, and a child eleven
        months short of their birthday would be told to wait three months for a
        dose that will be given after it. Each branch is tested against the date
        it would itself produce, and the earliest self-consistent answer wins.
        """
        conditional = dose_rule.get("conditional_interval")
        if not conditional:
            return dose_rule.get("min_interval")
        candidates: list[tuple[date, Mapping[str, int] | None]] = []
        for branch in conditional:
            interval = branch.get("min_interval")
            projected = add_period(previous, interval)
            threshold = branch.get("when_age_under")
            if branch.get("default"):
                if not any(
                    b.get("when_age_under")
                    and projected < add_period(dob, b["when_age_under"])
                    for b in conditional
                    if not b.get("default")
                ):
                    candidates.append((projected, interval))
            elif threshold and projected < add_period(dob, threshold):
                candidates.append((projected, interval))
        if not candidates:
            return self._min_interval(dose_rule, dob, previous)
        return min(candidates, key=lambda c: c[0])[1]

    def _evaluate_antigen(
        self,
        *,
        antigen: str,
        rule: Mapping[str, Any],
        dob: date,
        as_of: date,
        doses: Sequence[_AntigenDose],
    ) -> AntigenForecast:
        label = str(rule.get("label", antigen))
        weight = float(rule.get("weight", 1.0))
        school_required = bool(rule.get("school_required", False))
        recall_eligible = bool(rule.get("recall_eligible", True))
        huddle_eligible = bool(rule.get("huddle_eligible", True))
        forecast = AntigenForecast(
            antigen=antigen,
            label=label,
            status=Status.NOT_YET_DUE,
            doses_valid=0,
            doses_required=int(rule.get("doses_required", 0)),
            school_required=school_required,
            weight=weight,
            recall_eligible=recall_eligible,
            huddle_eligible=huddle_eligible,
        )
        if rule.get("shared_decision"):
            forecast.notes.append(
                "shared clinical decision-making; not a routine recommendation"
            )

        # A partial date anywhere in this antigen's history makes every interval
        # calculation a guess. Say so and stop.
        if any(d.precision != DosePrecision.DAY for d in doses):
            forecast.status = Status.REQUIRES_REVIEW
            # Zero, not len(doses): the whole point is that these doses cannot
            # be validated. Reporting them as valid produces "2 of 0" on a
            # conditional series and, worse, reads as a confirmed count.
            forecast.doses_valid = 0
            _, forecast.doses_required, _ = self._series_for(
                rule, dob, doses, as_of=as_of
            )
            forecast.notes.append(
                "history contains a partial date; intervals cannot be verified "
                "and this antigen will not be auto-recalled"
            )
            forecast.evaluations = [
                DoseEvaluation(antigen, d.cvx, d.given, d.source, False, None,
                               "partial date")
                if d.precision != DosePrecision.DAY
                else DoseEvaluation(antigen, d.cvx, d.given, d.source, True, None)
                for d in doses
            ]
            return forecast

        if rule.get("annual"):
            return self._evaluate_annual(forecast, rule, dob, as_of, doses)

        # Two passes, because series selection depends on the first VALID dose
        # and validity depends on the series. Pass one uses the branch chosen
        # without validity information; if that pass reveals a different first
        # valid dose, pass two re-runs with the corrected branch. Converges in
        # two: the second pass's branch is chosen from real valid doses.
        series, required, notes = self._series_for(rule, dob, doses, as_of=as_of)
        evaluations, valid = self._walk(antigen, doses, series, dob)
        if rule.get("conditional_series"):
            corrected = self._series_for(
                rule, dob, doses, as_of=as_of, first_valid=valid[0] if valid else None
            )
            changed = corrected[0] != series or corrected[1] != required
            # Always take the corrected notes: pass one had to guess, and its
            # "assumes a start at the current age" caveat is simply false once a
            # real first valid dose is known.
            series, required, notes = corrected
            if changed:
                evaluations, valid = self._walk(antigen, doses, series, dob)

        forecast.doses_required = required
        forecast.notes.extend(notes)
        forecast.evaluations = evaluations
        forecast.doses_valid = len(valid)

        # Early completion (a DTaP 4th dose at 4 years ends the series).
        for condition in rule.get("series_complete_when", []) or []:
            index = int(condition.get("dose", 0)) - 1
            if 0 <= index < len(valid):
                dose = valid[index]
                if dose.given < add_period(dob, condition.get("min_age_at_dose")) - self.schedule.grace:
                    continue
                gap_rule = condition.get("min_interval_from_previous")
                if gap_rule and index > 0:
                    if dose.given < add_period(valid[index - 1].given, gap_rule) - self.schedule.grace:
                        continue
                forecast.doses_required = index + 1
                forecast.notes.append(
                    f"series complete at {index + 1} doses: dose {index + 1} given on or "
                    "after the qualifying age"
                )
                break

        if len(valid) >= forecast.doses_required:
            forecast.status = Status.COMPLETE
            # Never display "5 of 4": an early-completion rule shortens the
            # requirement, it does not un-give a dose.
            forecast.doses_required = max(forecast.doses_required, len(valid))
            return forecast

        # Age-outs and no-longer-required.
        not_required_after = rule.get("not_required_after")
        if not_required_after and as_of >= add_period(dob, not_required_after):
            forecast.status = Status.NOT_REQUIRED
            forecast.notes.append(
                "routine catch-up not indicated at this age for a healthy child"
            )
            return forecast

        age_out = rule.get("age_out") or {}
        hard = age_out.get("hard_age")
        if hard:
            forecast.age_out_hard_date = add_period(dob, hard)
            if as_of >= forecast.age_out_hard_date:
                forecast.status = Status.AGED_OUT
                forecast.notes.append("past the maximum age for this vaccine")
                return forecast
        max_first = age_out.get("max_age_first_dose")
        if max_first and not valid:
            deadline = add_period(dob, max_first)
            forecast.age_out_hard_date = min(
                forecast.age_out_hard_date or deadline, deadline
            )
            if as_of > deadline:
                forecast.status = Status.AGED_OUT
                forecast.notes.append(
                    f"first dose could not be given after {deadline.isoformat()}"
                )
                return forecast

        next_number = len(valid) + 1
        dose_rule = self._dose_rule(series, next_number)
        forecast.next_dose = next_number

        earliest = add_period(dob, dose_rule.get("min_age"))
        if valid:
            interval = self._projected_interval(dose_rule, dob, valid[-1].given)
            if interval:
                earliest = max(earliest, add_period(valid[-1].given, interval))
            from_first = dose_rule.get("min_interval_from_first")
            if from_first:
                earliest = max(earliest, add_period(valid[0].given, from_first))
        forecast.earliest_date = earliest
        forecast.recommended_date = max(
            earliest, add_period(dob, dose_rule.get("recommended_age"))
        )
        overdue_rule = dose_rule.get("overdue_age")
        forecast.overdue_date = (
            max(earliest, add_period(dob, overdue_rule)) if overdue_rule else None
        )

        if forecast.overdue_date and as_of >= forecast.overdue_date:
            forecast.status = Status.OVERDUE
            forecast.days_overdue = (as_of - forecast.overdue_date).days
        elif as_of >= forecast.recommended_date:
            forecast.status = Status.DUE
        else:
            forecast.status = Status.NOT_YET_DUE
        return forecast

    def _invalid_reason(
        self,
        dose: _AntigenDose,
        dose_rule: Mapping[str, Any],
        dob: date,
        valid: Sequence[_AntigenDose],
    ) -> str:
        """Why this dose does not count, or "" if it does.

        The grace period is subtracted from every threshold: ACIP's four-day
        rule says a dose given up to four days early is still valid, and every
        registry in the country counts it. Omitting it would invalidate real
        doses and generate recalls for children who are up to date.
        """
        grace = self.schedule.grace
        min_age = dose_rule.get("min_age")
        if min_age and dose.given < add_period(dob, min_age) - grace:
            return f"given before the minimum age for dose {dose_rule.get('dose')}"
        if valid:
            interval = self._min_interval(dose_rule, dob, dose.given)
            if interval and dose.given < add_period(valid[-1].given, interval) - grace:
                return (
                    f"given less than the minimum interval after the previous valid "
                    f"dose on {valid[-1].given.isoformat()}"
                )
            from_first = dose_rule.get("min_interval_from_first")
            if from_first and dose.given < add_period(valid[0].given, from_first) - grace:
                return "given less than the minimum interval after the first dose"
        return ""

    def _evaluate_annual(
        self,
        forecast: AntigenForecast,
        rule: Mapping[str, Any],
        dob: date,
        as_of: date,
        doses: Sequence[_AntigenDose],
    ) -> AntigenForecast:
        """Seasonal antigens: "due" means no dose since the season started."""
        min_age = rule.get("min_age")
        if min_age and as_of < add_period(dob, min_age):
            forecast.status = Status.NOT_YET_DUE
            forecast.recommended_date = add_period(dob, min_age)
            return forecast
        month = self.schedule.influenza_season_start_month
        season_start = date(as_of.year if as_of.month >= month else as_of.year - 1, month, 1)
        forecast.evaluations = [
            DoseEvaluation(forecast.antigen, d.cvx, d.given, d.source, True, None)
            for d in doses
        ]
        this_season = [d for d in doses if d.given >= season_start]
        forecast.doses_required = 1
        forecast.doses_valid = len(this_season)
        forecast.recommended_date = season_start
        forecast.earliest_date = season_start
        if this_season:
            forecast.status = Status.COMPLETE
        else:
            forecast.status = Status.DUE
            forecast.next_dose = 1
            forecast.notes.append(f"no dose recorded since {season_start.isoformat()}")
        return forecast


# --------------------------------------------------------------------------
# Cross-check and validation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Disagreement:
    patient_id: str
    antigen: str
    primary_status: str
    secondary_status: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"{self.patient_id}/{self.antigen}: primary={self.primary_status} "
            f"secondary={self.secondary_status}"
        )


class CrossCheckForecaster:
    """Runs two forecasters and reports where they disagree.

    The primary's answer is returned unchanged -- this wrapper never invents a
    third opinion or splits the difference. Disagreements are recorded in the
    forecast's notes and collected in `disagreements` for the nightly exception
    report. WHY: two independent implementations of a published rule set
    agreeing is meaningful evidence; a blend of them is meaningless.
    """

    name = "cross_check"

    def __init__(self, primary: Forecaster, secondary: Forecaster) -> None:
        self.primary = primary
        self.secondary = secondary
        self.disagreements: list[Disagreement] = []

    def forecast(
        self,
        *,
        patient_id: str,
        dob: date,
        doses: Sequence[AdministeredDose],
        as_of: date,
    ) -> PatientForecast:
        result = self.primary.forecast(
            patient_id=patient_id, dob=dob, doses=doses, as_of=as_of
        )
        other = self.secondary.forecast(
            patient_id=patient_id, dob=dob, doses=doses, as_of=as_of
        )
        for antigen, forecast in result.antigens.items():
            comparison = other.antigens.get(antigen)
            if comparison is None or comparison.status == forecast.status:
                continue
            self.disagreements.append(
                Disagreement(patient_id, antigen, forecast.status, comparison.status)
            )
            forecast.notes.append(
                f"forecast engines disagree: {self.primary.name}={forecast.status}, "
                f"{self.secondary.name}={comparison.status}; treat as unconfirmed"
            )
            if forecast.status in Status.OPEN_GAP:
                # Never recall on a contested gap.
                forecast.recall_eligible = False
        return result


@dataclass
class ValidationResult:
    """The record that lets `recall.py` unlock outbound sending."""

    engine: str
    schedule_version: str
    cases: int
    antigen_comparisons: int
    agreements: int
    mismatches: list[Disagreement] = field(default_factory=list)
    validated_on: date | None = None

    @property
    def agreement_rate(self) -> float:
        if not self.antigen_comparisons:
            return 0.0
        return self.agreements / self.antigen_comparisons

    def passes(self, *, min_cases: int = 200, min_agreement: float = 0.99) -> bool:
        """README I-02: validate against 200 known-good records before go-live."""
        return self.cases >= min_cases and self.agreement_rate >= min_agreement

    def summary(self) -> str:
        return (
            f"{self.engine} @ {self.schedule_version}: {self.cases} cases, "
            f"{self.agreements}/{self.antigen_comparisons} antigen statuses agree "
            f"({self.agreement_rate:.4f})"
        )


def validate_against_reference(
    engine: Forecaster,
    cases: Iterable[Mapping[str, Any]],
    *,
    validated_on: date | None = None,
) -> ValidationResult:
    """Compare this engine against known-good forecasts, case by case.

    `cases` are dicts of {patient_id, dob, doses, as_of, expected: {antigen:
    status}}. The expected statuses come from I-CARE or from a clinician review
    -- this function does not care which, only that they are external. Antigens
    absent from `expected` are skipped rather than assumed to agree, because a
    reference set that only covers six antigens must not be able to report
    100% agreement across eighteen.
    """
    result = ValidationResult(
        engine=getattr(engine, "name", "unknown"),
        schedule_version=getattr(getattr(engine, "schedule", None), "version", ""),
        cases=0,
        antigen_comparisons=0,
        agreements=0,
        validated_on=validated_on,
    )
    for case in cases:
        forecast = engine.forecast(
            patient_id=str(case["patient_id"]),
            dob=case["dob"],
            doses=case["doses"],
            as_of=case["as_of"],
        )
        result.cases += 1
        for antigen, expected in (case.get("expected") or {}).items():
            got = forecast.antigens.get(antigen)
            actual = got.status if got else "missing"
            result.antigen_comparisons += 1
            if actual == expected:
                result.agreements += 1
            else:
                result.mismatches.append(
                    Disagreement(str(case["patient_id"]), antigen, expected, actual)
                )
    return result
