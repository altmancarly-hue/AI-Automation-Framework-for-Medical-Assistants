"""The 17:00 job. A fixed sequence of steps, deliberately not an agent.

README I-03, on the word "agent":

    "this workflow is often marketed as agentic. It is not, and should not be.
    It is a scheduled batch job with a fixed sequence of steps. There is no
    reason to give a model autonomous tool-selection authority over a patient
    chart to produce a summary. Constrain it to a pipeline."

So the sequence below is literal, ordered, and the same for every patient. The
model is called exactly once, near the end, with three notes and a problem list.
It selects no tools, reads no chart, and cannot cause another step to run.

ORDER MATTERS IN ONE PLACE. The narrative call is last, after every computed
section, because A.4 tells the model that screenings, immunizations and growth
are computed elsewhere and it must not report on them. If the narrative ran
first there would be nothing to compare against when it does anyway.

A patient whose narrative fails still gets a brief. A patient whose GROWTH
calculation fails still gets a brief, with the growth section replaced by a line
naming the problem -- a missing height is a data-entry gap, and hiding it means
nobody ever fixes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence

from modules.scheduling.models import new_id

from .brief import OpenThread, PreVisitBrief, assemble
from .growth import (
    ChannelCrossing,
    GrowthReference,
    ImplausibleMeasurement,
    Indicator,
    Measurement,
    NotComparable,
    OutOfRange,
    channel_crossing,
)

#: Recumbent length and standing stature are the same clinical quantity against
#: two reference tables. Pairing them by exact indicator meant the switch every
#: child makes at two produced NO comparison and NO note -- the trend simply
#: vanished from the brief for a year, which looks exactly like a trend that is
#: fine.
#:
#: TUPLES, NOT SETS. An earlier version iterated a `frozenset` to pick the prior
#: measurement, so with both a recumbent length and a standing stature on file
#: the chosen prior depended on string hashing: the same chart flagged a
#: two-channel stature drop under one PYTHONHASHSEED and reported nothing under
#: another. In a module whose first principle is determinism, the flagship
#: signal was a coin flip.
_FAMILIES: tuple[tuple[str, ...], ...] = (
    (Indicator.LENGTH_FOR_AGE, Indicator.STATURE_FOR_AGE),
    (Indicator.WEIGHT_FOR_LENGTH, Indicator.WEIGHT_FOR_STATURE),
)


def _family_of(indicator: str) -> tuple[str, ...]:
    for family in _FAMILIES:
        if indicator in family:
            return family
    return (indicator,)
from .narrative import Encounter, NarrativeSynthesizer
from .periodicity import CompletedScreening, PeriodicitySchedule

__all__ = ["PatientDay", "BriefBatch", "run_batch"]


@dataclass
class PatientDay:
    """Everything the batch is given about one scheduled patient.

    Assembled by the caller from the EHR. Kept as a plain dataclass so the FHIR
    query layer is swappable without touching any of the logic below -- the same
    seam `RegistryForecaster` is in I-02.
    """

    patient_id: str
    patient_label: str
    sex: str
    age_months: float
    age_label: str
    visit_type: str
    appointment_local: str
    provider: str
    measurements: Sequence[Measurement] = ()
    prior_measurements: Sequence[Measurement] = ()
    completed_screenings: Sequence[CompletedScreening] = ()
    risk_flags: Sequence[str] = ()
    immunizations_due: Sequence[str] = ()
    open_threads: Sequence[OpenThread] = ()
    encounters: Sequence[Encounter] = ()
    problem_list: Sequence[str] = ()
    #: Age before which this system holds no screening history for this patient.
    #: See `PeriodicitySchedule.evaluate`.
    data_horizon_months: float | None = None


@dataclass
class BriefBatch:
    clinic_date: date
    briefs: list[PreVisitBrief] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "clinic_date": self.clinic_date.isoformat(),
            "briefs": len(self.briefs),
            "failures": list(self.failures),
            "with_narrative": sum(1 for b in self.briefs if b.has_ai_content),
            "narrative_items_dropped": sum(b.narrative_dropped for b in self.briefs),
        }


def _growth_for(
    reference: GrowthReference, day: PatientDay
) -> tuple[list[Any], list[ChannelCrossing], list[str]]:
    points, crossings, notes = [], [], []
    prior_by_indicator: dict[str, list[Any]] = {}
    # Sorted by age, so "the prior" is the most recent one rather than whichever
    # element the EHR query happened to put last. With priors supplied
    # newest-first -- an entirely ordinary query order -- the comparison ran
    # against a two-year-old measurement instead of a six-month-old one and
    # reported a different crossing from the same chart.
    for measurement in sorted(day.prior_measurements, key=lambda m: m.x):
        try:
            prior_by_indicator.setdefault(measurement.indicator, []).append(
                reference.place(measurement)
            )
        except (OutOfRange, ValueError) as exc:
            notes.append(f"prior {measurement.indicator}: {exc}")
    for measurement in day.measurements:
        try:
            point = reference.place(measurement)
        except (OutOfRange, KeyError, ValueError) as exc:
            # Named, not hidden. A percentile that silently does not appear is
            # indistinguishable from a percentile that is fine.
            notes.append(f"{measurement.indicator}: {exc}")
            continue
        points.append(point)
        # Prefer a prior of the SAME indicator -- it is directly comparable.
        # Only fall back to the other member of the family, which is where the
        # method-switch refusal lives and is worth reporting rather than
        # silently skipping.
        family = _family_of(measurement.indicator)
        ordered = [measurement.indicator] + [
            name for name in family if name != measurement.indicator
        ]
        candidates = [
            prior_by_indicator[name][-1] for name in ordered if prior_by_indicator.get(name)
        ]
        if not candidates:
            continue
        first_error: str | None = None
        for earlier in candidates:
            try:
                crossing = channel_crossing(earlier, point)
            except NotComparable as exc:
                first_error = first_error or str(exc)
                continue
            if crossing is not None:
                crossings.append(crossing)
            first_error = None
            break
        if first_error is not None:
            notes.append(f"{measurement.indicator} trend not assessed: {first_error}")
    return points, crossings, notes


def run_batch(
    days: Sequence[PatientDay],
    *,
    clinic_date: date,
    schedule: PeriodicitySchedule,
    reference: GrowthReference,
    synthesizer: NarrativeSynthesizer | None,
    generated_utc: datetime,
    feedback: Any = None,
) -> BriefBatch:
    """Run tomorrow's schedule. One brief per patient, or one recorded failure."""
    schedule.require_reviewed()
    batch = BriefBatch(clinic_date=clinic_date)
    for day in days:
        try:
            batch.briefs.append(
                _one(
                    day,
                    schedule=schedule,
                    reference=reference,
                    synthesizer=synthesizer,
                    generated_utc=generated_utc,
                    feedback=feedback,
                    clinic_date=clinic_date,
                )
            )
        except Exception as exc:  # noqa: BLE001 - one patient must not stop the batch
            batch.failures.append(
                {"patient_id": day.patient_id, "error": f"{type(exc).__name__}: {exc}"}
            )
    return batch


def _one(
    day: PatientDay,
    *,
    schedule: PeriodicitySchedule,
    reference: GrowthReference,
    synthesizer: NarrativeSynthesizer | None,
    generated_utc: datetime,
    feedback: Any,
    clinic_date: date,
) -> PreVisitBrief:
    screenings = schedule.evaluate(
        age_months=day.age_months,
        completed=day.completed_screenings,
        risk_flags=day.risk_flags,
        data_horizon_months=day.data_horizon_months,
    )
    forms = schedule.forms_due(age_months=day.age_months)
    points, crossings, growth_notes = _growth_for(reference, day)

    narrative = None
    if synthesizer is not None and day.encounters:
        narrative = synthesizer.synthesize(
            day.encounters, patient_id=day.patient_id, problem_list=day.problem_list
        )

    brief_id = new_id("brf")
    brief = assemble(
        patient_label=day.patient_label,
        age_label=day.age_label,
        visit_type=day.visit_type,
        appointment_local=day.appointment_local,
        provider=day.provider,
        generated_utc=generated_utc,
        screenings=screenings,
        immunizations_due=day.immunizations_due,
        growth=points,
        crossings=crossings,
        open_threads=day.open_threads,
        forms_due=forms,
        narrative=narrative,
        brief_id=brief_id,
    )
    brief.warnings.extend(growth_notes)

    if feedback is not None:
        feedback.register(
            brief_id=brief_id,
            patient_id=day.patient_id,
            clinic_date=clinic_date.isoformat(),
            generated_utc=generated_utc,
            had_narrative=bool(narrative and narrative.items),
            # Whether it survived the one-screen cap, which is what the
            # clinician actually rated.
            narrative_shown=brief.has_ai_content,
            narrative_items=len(narrative.items) if narrative else 0,
            narrative_dropped=len(narrative.dropped) if narrative else 0,
            prompt_hash=narrative.prompt_hash if narrative else "",
            model_id=narrative.model_id if narrative else "",
        )
    return brief
