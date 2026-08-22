"""Which screenings are due at this age. A lookup table, and nothing else.

README I-03 is explicit about the classification:

    "Which screenings are due at this age?" -> Deterministic rules table.
    Bright Futures periodicity schedule is a lookup table. Do not use an LLM.

So this module has no screening knowledge in it. Every age, cadence, window and
citation is in `config/periodicity.yaml`, which makes a Bright Futures revision
a data diff a clinician can read and sign off, rather than a code change nobody
outside engineering can check. `make lint` asserts no model import here.

THREE THINGS THIS GETS RIGHT THAT A NAIVE "is age in list" DOES NOT:

  * **A window, not a point.** A nine-month developmental screen done at ten
    months is done. Without a grace window every brief in the practice shows a
    permanent backlog of screenings that were, in fact, completed -- and a
    checklist that is always red is a checklist nobody reads.
  * **A catch-up horizon.** A missed 18-month M-CHAT is still worth doing at
    twenty months and is not worth putting on a five-year-old's brief. The
    horizon is per-screening data, because it differs per screening.
  * **Risk-based screenings stay OFF until the risk assessment says otherwise,
    and the assessment itself is a due item.** A lipid panel that shows as due
    for every patient is noise; one that never shows because nobody was ever
    asked the risk question is a miss. Both failure modes are avoided by making
    the assessment a first-class entry that gates the test.

Nothing here decides that a child does not need something. A screening the
clinician declines is recorded as DECLINED by a human, which is a different
state from NOT_DUE and is visible on the next brief.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Mapping, Sequence

import yaml

__all__ = [
    "DEFAULT_PERIODICITY_PATH",
    "Status",
    "ScreeningDefinition",
    "ScreeningStatus",
    "CompletedScreening",
    "PeriodicitySchedule",
    "ScheduleNotReviewed",
]

DEFAULT_PERIODICITY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "periodicity.yaml",
)


class ScheduleNotReviewed(RuntimeError):
    """Raised when the periodicity table has no assigned clinical owner."""


class Status:
    DUE = "due"
    OVERDUE = "overdue"
    NOT_DUE = "not_due"
    COMPLETE = "complete"
    DECLINED = "declined"
    #: The gating risk assessment has not been done, so the system cannot say
    #: whether the test is indicated. Deliberately NOT "not_due".
    RISK_UNKNOWN = "risk_unknown"
    #: Past the catch-up horizon and never done. Reported once, not forever.
    MISSED = "missed"
    #: Due before this system held any data for the patient. Not a miss and not
    #: a pass -- a prompt to look at the paper chart once. See `evaluate`.
    UNKNOWN = "unknown_pre_horizon"


@dataclass(frozen=True)
class CompletedScreening:
    """Something that actually happened, from the chart."""

    screening_id: str
    performed_on: date
    age_months_at: float
    result: str = ""
    declined: bool = False


@dataclass(frozen=True)
class ScreeningDefinition:
    id: str
    name: str
    category: str
    cpt: tuple[str, ...] = ()
    due_at_months: tuple[float, ...] = ()
    from_months: float | None = None
    to_months: float | None = None
    every_months: float | None = None
    once_in_band: bool = False
    window_months: float = 3.0
    catch_up_until_months: float | None = None
    requires_risk_flag: str | None = None
    risk_assessment_for: str | None = None
    authority: str = ""

    def target_ages(self) -> list[float]:
        """EVERY age at which this screening is expected, over the whole schedule.

        Not clipped to the patient's age. Clipping was how a missed screening
        disappeared: only the latest reached target was ever examined, so a
        child who never had the nine- OR eighteen-month developmental screen read
        as merely DUE for the thirty-month one, and the two misses left no trace
        anywhere. The caller decides which of these have been reached; this
        method's only job is to say what the schedule asks for.
        """
        if self.due_at_months:
            return sorted(self.due_at_months)
        if self.from_months is None or self.every_months is None:
            return []
        if self.once_in_band:
            # One screen anywhere in the band. The band OPENING is the single
            # target; `to_months` is when it ages out. Generating a target every
            # `every_months` across the band produced two "once" targets for the
            # 17-21 lipid panel and re-asked for it four years later.
            return [self.from_months]
        ages: list[float] = []
        age = self.from_months
        ceiling = self.to_months if self.to_months is not None else self.from_months
        while age <= ceiling:
            ages.append(age)
            age += self.every_months
        return ages


@dataclass
class ScreeningStatus:
    """One line of the brief's DUE TODAY section, with its reasoning attached."""

    definition: ScreeningDefinition
    status: str
    target_age_months: float | None = None
    last_done_on: date | None = None
    #: Plain-language reasoning. README I-03 warns the brief "becomes noise and
    #: is ignored"; a line that cannot explain itself is the first to be ignored.
    because: str = ""

    @property
    def actionable(self) -> bool:
        return self.status in (Status.DUE, Status.OVERDUE, Status.RISK_UNKNOWN)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.definition.id,
            "name": self.definition.name,
            "category": self.definition.category,
            "cpt": list(self.definition.cpt),
            "status": self.status,
            "target_age_months": self.target_age_months,
            "last_done_on": self.last_done_on.isoformat() if self.last_done_on else None,
            "because": self.because,
            "authority": self.definition.authority,
        }


class PeriodicitySchedule:
    """The Bright Futures table, evaluated against one patient."""

    def __init__(self, data: Mapping[str, Any]) -> None:
        self.version = str(data.get("version", "unversioned"))
        self.source = str(data.get("source", "unspecified"))
        self.review: dict[str, Any] = dict(data.get("review") or {})
        self.well_visit_months: tuple[float, ...] = tuple(
            float(m) for m in data.get("well_visit_months", ())
        )
        self.screenings: dict[str, ScreeningDefinition] = {}
        for raw in data.get("screenings", []):
            definition = ScreeningDefinition(
                id=str(raw["id"]),
                name=str(raw["name"]),
                category=str(raw.get("category", "other")),
                cpt=tuple(str(c) for c in raw.get("cpt", ())),
                due_at_months=tuple(float(a) for a in raw.get("due_at_months", ())),
                from_months=_opt_float(raw.get("from_months")),
                to_months=_opt_float(raw.get("to_months")),
                every_months=_opt_float(raw.get("every_months")),
                once_in_band=bool(raw.get("once_in_band", False)),
                window_months=float(raw.get("window_months", 3)),
                catch_up_until_months=_opt_float(raw.get("catch_up_until_months")),
                requires_risk_flag=_opt_str(raw.get("requires_risk_flag")),
                risk_assessment_for=_opt_str(raw.get("risk_assessment_for")),
                authority=str(raw.get("authority", "")),
            )
            self.screenings[definition.id] = definition
        self.form_requirements: list[Mapping[str, Any]] = list(
            data.get("form_requirements", [])
        )
        if not self.screenings:
            raise ValueError("the periodicity table is empty")
        self._validate_risk_wiring()
        self._validate_horizons()

    @classmethod
    def load(
        cls, path: str | os.PathLike[str] = DEFAULT_PERIODICITY_PATH
    ) -> "PeriodicitySchedule":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh))

    def _validate_risk_wiring(self) -> None:
        """A risk flag no assessment can ever set is a screening that never fires.

        Caught at load rather than at run time: the symptom otherwise is a
        screening quietly absent from every brief in the practice, which is
        indistinguishable from "nobody needed it".
        """
        settable = {
            d.risk_assessment_for for d in self.screenings.values() if d.risk_assessment_for
        }
        orphans = sorted(
            f"{d.id} -> {d.requires_risk_flag}"
            for d in self.screenings.values()
            if d.requires_risk_flag and d.requires_risk_flag not in settable
        )
        if orphans:
            raise ValueError(
                "these screenings are gated on a risk flag that no risk "
                f"assessment in this table can set, so they can never become "
                f"due: {orphans}"
            )

    def _validate_horizons(self) -> None:
        """A discrete-age screening with no catch-up horizon never stops nagging.

        A banded entry ages out on its own -- `to_months` says when. A discrete
        one has no such end, so without a horizon a missed two-year TB risk
        assessment shows as OVERDUE on a thirteen-year-old's brief, 132 months
        past its window, every single visit for the rest of childhood. That is
        the "brief becomes noise and is ignored" risk in README I-03's own risk
        table, and the fix belongs in the data, not in a guess in code.
        """
        unbounded = sorted(
            d.id for d in self.screenings.values()
            if not d.due_at_months and d.from_months is not None and d.to_months is None
        )
        if unbounded:
            raise ValueError(
                "these banded screenings have no to_months, so they never age "
                f"out and the target list is unbounded: {unbounded}"
            )
        missing = sorted(
            d.id for d in self.screenings.values()
            if d.due_at_months and d.catch_up_until_months is None
        )
        if missing:
            raise ValueError(
                "these screenings list discrete due ages but no "
                f"catch_up_until_months, so they would read as overdue forever: "
                f"{missing}"
            )

    @property
    def has_clinical_owner(self) -> bool:
        owner = str(self.review.get("owner", "")).strip()
        return bool(owner) and "UNASSIGNED" not in owner.upper()

    def require_reviewed(self) -> None:
        """README I-03's control, in the code path rather than on a checklist."""
        if not self.has_clinical_owner:
            raise ScheduleNotReviewed(
                f"config/periodicity.yaml ({self.version}) has no assigned "
                "clinical owner. README I-03 requires an owner and an annual "
                "review tied to the Bright Futures update cycle before this "
                "drives a clinical brief."
            )

    # -- evaluation --------------------------------------------------------

    def evaluate(
        self,
        *,
        age_months: float,
        completed: Sequence[CompletedScreening] = (),
        risk_flags: Iterable[str] = (),
        data_horizon_months: float | None = None,
    ) -> list[ScreeningStatus]:
        """Status of every screening for one patient at one moment.

        `data_horizon_months` is the age before which this system has no
        screening history for this patient -- typically because the practice
        went live after the child was born and the paper chart was not
        back-loaded. Targets older than the horizon are reported as UNKNOWN
        rather than OVERDUE.

        WHY THIS IS NOT OPTIONAL IN PRACTICE: without it, day one of a
        deployment shows every established patient overdue for a decade of
        screenings that were in fact done on paper. Every brief is red, the
        red means nothing, and the periodicity section is ignored inside a
        week -- which is README I-03's "brief becomes noise" risk arriving on
        the first morning. Reporting them as UNKNOWN is both true and useful:
        it tells the MA to check the paper chart once, rather than asserting a
        miss the practice can disprove.
        """
        flags = set(risk_flags)
        by_id: dict[str, list[CompletedScreening]] = {}
        for item in completed:
            by_id.setdefault(item.screening_id, []).append(item)
        for records in by_id.values():
            records.sort(key=lambda c: c.performed_on)

        results = [
            self._evaluate_one(
                definition, age_months, by_id.get(definition.id, []), flags,
                data_horizon_months,
            )
            for definition in self.screenings.values()
        ]
        self._collapse_redundant_risk_lines(results)
        self._resolve_answered_risks(results, flags)
        order = {
            Status.OVERDUE: 0, Status.DUE: 1, Status.RISK_UNKNOWN: 2,
            Status.MISSED: 3, Status.UNKNOWN: 4, Status.COMPLETE: 5,
            Status.DECLINED: 6, Status.NOT_DUE: 7,
        }
        results.sort(key=lambda r: (order[r.status], r.definition.category, r.definition.name))
        return results

    def _evaluate_one(
        self,
        definition: ScreeningDefinition,
        age_months: float,
        history: Sequence[CompletedScreening],
        flags: set[str],
        data_horizon: float | None = None,
    ) -> ScreeningStatus:
        declined = [h for h in history if h.declined]
        done = [h for h in history if not h.declined]

        targets = definition.target_ages()
        if not targets:
            return ScreeningStatus(
                definition, Status.NOT_DUE,
                because=f"not scheduled before {_age(age_months)}",
            )

        if definition.once_in_band:
            return self._evaluate_once_in_band(
                definition, age_months, done, declined, targets[0]
            )

        window = definition.window_months
        satisfied = _attribute(done, targets, window)
        reached = [t for t in targets if age_months >= t - window]
        if not reached:
            nxt = min(targets)
            return ScreeningStatus(
                definition, Status.NOT_DUE, target_age_months=nxt,
                because=f"next due at {_age(nxt)}",
            )

        unmet = [t for t in reached if t not in satisfied]
        if not unmet:
            latest = max(t for t in reached if t in satisfied)
            record = satisfied[latest]
            return ScreeningStatus(
                definition, Status.COMPLETE, target_age_months=latest,
                last_done_on=record.performed_on,
                because=(
                    f"due at {_age(latest)}; done {record.performed_on.isoformat()} "
                    f"at {_age(record.age_months_at)}"
                ),
            )

        # The deployment's DATA horizon, applied before anything is called a
        # miss: a target older than the practice's records is unknown, not
        # unmet. (Distinct from the per-screening CATCH-UP horizon below, which
        # is a clinical judgement about when catching up stops being useful.)
        if data_horizon is not None:
            # What is due at TODAY'S visit is never hidden by the horizon. The
            # patient is in front of you; a screening whose target falls inside
            # its own window of today is outstanding whether or not the practice
            # holds the older records. Anything older than that is genuinely
            # unknown -- claiming a three-year-old vision screen was "never
            # done" when the practice has no records from that year is the
            # false assertion this whole parameter exists to prevent.
            due_today = max(
                (t for t in reached if abs(age_months - t) <= window), default=None
            )
            keep = {due_today} if due_today is not None else set()
            before_horizon = [t for t in unmet if t < data_horizon and t not in keep]
            unmet = [t for t in unmet if t >= data_horizon or t in keep]
            if not unmet:
                if before_horizon:
                    return ScreeningStatus(
                        definition, Status.UNKNOWN,
                        target_age_months=max(before_horizon),
                        because=(
                            f"{len(before_horizon)} target(s) fall before this "
                            f"system's data horizon at {_age(data_horizon)}; check "
                            "the paper chart once rather than recording a miss"
                        ),
                    )
                return ScreeningStatus(
                    definition, Status.NOT_DUE, target_age_months=max(reached),
                    because="nothing outstanding since the data horizon",
                )

        # A banded screening that the guideline has stopped asking for.
        if (
            definition.to_months is not None
            and age_months > definition.to_months + window
        ):
            return ScreeningStatus(
                definition, Status.NOT_DUE, target_age_months=max(unmet),
                because=f"the recommended age band ended at {_age(definition.to_months)}",
            )

        horizon = definition.catch_up_until_months
        if horizon is not None and age_months > horizon:
            return ScreeningStatus(
                definition, Status.MISSED, target_age_months=min(unmet),
                because=(
                    f"{len(unmet)} screening(s) from {_age(min(unmet))} were never "
                    f"done and the catch-up window closed at {_age(horizon)}; "
                    "record as missed rather than carrying it forward"
                ),
            )

        # Risk-based screenings stay OFF until the assessment says otherwise.
        # This has to run before DUE/OVERDUE: a lipid panel that shows as due
        # for every patient is noise, and one that never shows because nobody
        # was ever asked the risk question is a miss.
        if definition.requires_risk_flag and definition.requires_risk_flag not in flags:
            assessment = self._assessment_for(definition.requires_risk_flag)
            return ScreeningStatus(
                definition,
                Status.NOT_DUE if assessment is None else Status.RISK_UNKNOWN,
                target_age_months=min(unmet),
                because=(
                    f"risk-based; the {definition.requires_risk_flag} risk "
                    f"assessment ({assessment}) has not been recorded, so whether "
                    "this is indicated is unknown"
                    if assessment
                    else f"risk-based; no {definition.requires_risk_flag} risk recorded"
                ),
            )

        # Report the EARLIEST unmet target, so lateness is measured from the
        # first miss rather than from whichever one happens to be nearest.
        first = min(unmet)
        if declined and any(abs(d.age_months_at - first) <= window for d in declined):
            last = declined[-1]
            return ScreeningStatus(
                definition, Status.DECLINED, target_age_months=first,
                last_done_on=last.performed_on,
                because=(
                    f"declined {last.performed_on.isoformat()}; offer again rather "
                    "than assuming the answer has not changed"
                ),
            )
        overdue = age_months > first + window
        outstanding = (
            f"; {len(unmet)} outstanding from {_age(first)} to {_age(max(unmet))}"
            if len(unmet) > 1
            else ""
        )
        return ScreeningStatus(
            definition, Status.OVERDUE if overdue else Status.DUE,
            target_age_months=first,
            because=(
                f"due at {_age(first)}, patient is {_age(age_months)}"
                + (f", {_months(age_months - first)} past the window" if overdue else "")
                + outstanding
            ),
        )

    def _evaluate_once_in_band(
        self,
        definition: ScreeningDefinition,
        age_months: float,
        done: Sequence[CompletedScreening],
        declined: Sequence[CompletedScreening],
        opens_at: float,
    ) -> ScreeningStatus:
        """One screen anywhere in the band satisfies it, forever."""
        window = definition.window_months
        closes_at = definition.to_months if definition.to_months is not None else opens_at
        # The band's own window on BOTH ends. A panel drawn on the very visit
        # this brief marked DUE has to count, and the brief marks it DUE from
        # `opens_at - window` -- so a completion at `opens_at - 1` was rejected
        # and the child was sent for a second venipuncture four years later.
        in_band = [
            h for h in done
            if opens_at - window <= h.age_months_at <= closes_at + window
        ]
        if in_band:
            last = in_band[-1]
            return ScreeningStatus(
                definition, Status.COMPLETE, target_age_months=opens_at,
                last_done_on=last.performed_on,
                because=(
                    f"one screen in this age band is enough; done "
                    f"{last.performed_on.isoformat()} at {_age(last.age_months_at)}"
                ),
            )
        if age_months < opens_at - window:
            return ScreeningStatus(
                definition, Status.NOT_DUE, target_age_months=opens_at,
                because=f"next due at {_age(opens_at)}",
            )
        if definition.requires_risk_flag and definition.requires_risk_flag not in flags:
            assessment = self._assessment_for(definition.requires_risk_flag)
            return ScreeningStatus(
                definition,
                Status.NOT_DUE if assessment is None else Status.RISK_UNKNOWN,
                target_age_months=opens_at,
                because=(
                    f"risk-based; the {definition.requires_risk_flag} risk "
                    f"assessment ({assessment}) has not been recorded"
                    if assessment
                    else f"risk-based; no {definition.requires_risk_flag} risk recorded"
                ),
            )
        if age_months > closes_at + window:
            return ScreeningStatus(
                definition, Status.MISSED, target_age_months=opens_at,
                because=(
                    f"the {_age(opens_at)}-{_age(closes_at)} band closed and this "
                    "was never done"
                ),
            )
        if declined and any(d.age_months_at >= opens_at - window for d in declined):
            last = declined[-1]
            return ScreeningStatus(
                definition, Status.DECLINED, target_age_months=opens_at,
                last_done_on=last.performed_on,
                because=f"declined {last.performed_on.isoformat()}; offer again",
            )
        overdue = age_months > opens_at + window
        return ScreeningStatus(
            definition, Status.OVERDUE if overdue else Status.DUE,
            target_age_months=opens_at,
            because=(
                f"one screen due between {_age(opens_at)} and {_age(closes_at)}, "
                f"patient is {_age(age_months)}"
            ),
        )

    def _collapse_redundant_risk_lines(self, results: list[ScreeningStatus]) -> None:
        """Drop a RISK_UNKNOWN whose gating assessment is already on this brief.

        Both lines ask for the same action -- do the risk assessment -- and
        printing both puts a permanent pair of unanswerable question marks on
        every brief in the practice. The one the MA can act on is the
        assessment; the test underneath it becomes actionable only once the
        assessment says so, at which point it appears on its own.

        The RISK_UNKNOWN line still fires when the assessment is NOT actionable:
        an adolescent whose lead risk assessment aged out at six and was never
        done genuinely has an unknown, and that is worth one line.
        """
        actionable_assessments = {
            r.definition.risk_assessment_for
            for r in results
            if r.definition.risk_assessment_for and r.status in (Status.DUE, Status.OVERDUE)
        }
        for result in results:
            flag = result.definition.requires_risk_flag
            if result.status == Status.RISK_UNKNOWN and flag in actionable_assessments:
                result.status = Status.NOT_DUE
                result.because = (
                    f"risk-based; the {flag} risk assessment is already on this "
                    "brief and this test becomes actionable only once it is done"
                )

    def _resolve_answered_risks(
        self, results: list[ScreeningStatus], flags: set[str]
    ) -> None:
        """A risk that was assessed and came back negative is ANSWERED, not unknown.

        RISK_UNKNOWN means nobody asked. An assessment recorded COMPLETE with no
        flag set means somebody asked and the answer was no, and reporting that
        as unknown puts a permanent question mark next to a question that has
        already been answered -- which is how the brief teaches people that the
        question marks mean nothing.
        """
        answered = {
            r.definition.risk_assessment_for
            for r in results
            if r.definition.risk_assessment_for and r.status == Status.COMPLETE
        }
        for result in results:
            flag = result.definition.requires_risk_flag
            if (
                result.status == Status.RISK_UNKNOWN
                and flag in answered
                and flag not in flags
            ):
                result.status = Status.NOT_DUE
                result.because = (
                    f"the {flag} risk assessment was completed and did not "
                    "indicate this test"
                )

    def _assessment_for(self, flag: str) -> str | None:
        for definition in self.screenings.values():
            if definition.risk_assessment_for == flag:
                return definition.name
        return None

    # -- forms -------------------------------------------------------------

    def forms_due(self, *, age_months: float, window_months: float = 6.0) -> list[dict[str, Any]]:
        """Illinois school/sport forms coming up. Not clinical, still the reason
        a family comes back a second time if nobody mentions it."""
        due: list[dict[str, Any]] = []
        for raw in self.form_requirements:
            ages = [float(a) for a in raw.get("due_at_months", ())]
            if not ages and raw.get("from_months") is not None:
                # `to_months` is read ONCE. Reading it inside the condition with
                # `raw.get("to_months", age)` re-evaluated the default from the
                # loop variable, so a form with no `to_months` -- which nothing
                # validated against, in a file clinicians are told to edit --
                # made the condition `age <= age` and hung the 17:00 batch for
                # the entire practice.
                start = float(raw["from_months"])
                end = float(raw.get("to_months", start))
                step = float(raw.get("every_months", 12)) or 12.0
                age = start
                while age <= end:
                    ages.append(age)
                    age += step
            for target in ages:
                if target - window_months <= age_months <= target + window_months:
                    due.append(
                        {
                            "id": str(raw["id"]),
                            "name": str(raw["name"]),
                            "target_age_months": target,
                            "note": str(raw.get("note", "")),
                            "authority": str(raw.get("authority", "")),
                        }
                    )
                    break
        return due


def _attribute(
    done: Sequence[CompletedScreening], targets: Sequence[float], window: float
) -> dict[float, CompletedScreening]:
    """Assign each completion to the single target it satisfies.

    A completion satisfies ONE target -- the nearest one it is within window of,
    ties going to the earlier. Without this, any two due ages closer together
    than twice the window let one completion satisfy both: a single M-CHAT-R/F
    at twenty-one months read as COMPLETE for the eighteen-month AND the
    twenty-four-month screen, and the twenty-four-month autism screen never
    appeared on any brief for that child.
    """
    assigned: dict[float, CompletedScreening] = {}
    for record in sorted(done, key=lambda c: c.performed_on):
        candidates = [t for t in targets if abs(record.age_months_at - t) <= window]
        # A completion later than every target still satisfies the last one --
        # a screening done at four years is not undone by being late.
        if not candidates:
            later = [t for t in targets if record.age_months_at >= t]
            candidates = [max(later)] if later else []
        if not candidates:
            continue
        free = [t for t in candidates if t not in assigned] or candidates
        best = min(free, key=lambda t: (abs(record.age_months_at - t), t))
        assigned[best] = record
    return assigned


def _opt_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _opt_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _age(months: float) -> str:
    if months < 1:
        return "the newborn visit"
    if months < 24:
        return f"{months:g} mo"
    years, remainder = divmod(int(round(months)), 12)
    return f"{years}y" if remainder == 0 else f"{years}y {remainder}m"


def _months(delta: float) -> str:
    return f"{delta:.0f} month(s)" if delta >= 1 else "under a month"
