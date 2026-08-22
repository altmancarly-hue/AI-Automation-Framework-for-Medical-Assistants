"""The nightly batch, in the order the steps have to happen.

README I-02's target state is five numbered steps, and the order is not
cosmetic. Two of the transitions are the whole safety argument:

  * **Reconcile before forecasting.** Forecasting an unreconciled chart is how
    you generate a recall for a vaccine the child had at a pharmacy last month.
    README I-02's first risk row.
  * **Hold contested antigens before recalling.** An antigen whose dose count is
    unresolved gets `REQUIRES_REVIEW`, which is not an open gap, which means the
    recall engine never sees it. A gap a human confirms in ten seconds costs far
    less than either a duplicate injection or an accusatory text to a family who
    did everything right.

Adjudication sits between reconciliation and forecasting and is optional: with
no `Adjudicator` the ambiguous pairs simply stay ambiguous and their antigens
stay held. That is a slower but entirely correct configuration, and it is what
the practice runs on day one before any model is deployed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence

from .adjudicate import Adjudicator, HumanReviewItem, apply_adjudications
from .forecast import (
    AdministeredDose,
    Forecaster,
    LocalRulesForecaster,
    PatientForecast,
    Status,
)
from .matcher import DoseRecord, Reconciliation, reconcile

__all__ = ["PatientInput", "NightlyResult", "apply_reconciliation_holds", "run_nightly"]


@dataclass
class PatientInput:
    patient_id: str
    family_id: str
    first_name: str
    dob: date
    chart: Sequence[DoseRecord] = field(default_factory=list)
    registry: Sequence[DoseRecord] = field(default_factory=list)

    def info(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "first_name": self.first_name,
            "dob": self.dob,
        }


@dataclass
class NightlyResult:
    forecasts: dict[str, PatientForecast] = field(default_factory=dict)
    reconciliations: dict[str, Reconciliation] = field(default_factory=dict)
    review_queue: list[HumanReviewItem] = field(default_factory=list)
    patients: dict[str, dict[str, Any]] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        open_gaps = sum(len(f.open_gaps) for f in self.forecasts.values())
        held = sum(len(f.needs_review) for f in self.forecasts.values())
        return {
            "patients": len(self.forecasts),
            "open_gaps": open_gaps,
            "antigens_held_for_review": held,
            "reconciliation_review_items": len(self.review_queue),
            "patients_with_unknown_codes": sum(
                1 for f in self.forecasts.values() if f.unknown_codes
            ),
        }


def apply_reconciliation_holds(
    forecast: PatientForecast, reconciliation: Reconciliation
) -> PatientForecast:
    """Downgrade antigens whose dose count the reconciliation could not settle.

    The forecast is mutated in place and returned. WHY downgrade rather than
    annotate: `recall.build_queue` filters on `is_open_gap`, and a note attached
    to an otherwise-confident OVERDUE would be read by the code as a confident
    OVERDUE. The status is the interface; making the status honest is the only
    way the hold actually holds.
    """
    unresolved = reconciliation.unresolved_antigens
    for antigen in unresolved:
        entry = forecast.antigens.get(antigen)
        if entry is None:
            continue
        if entry.status in (Status.COMPLETE, Status.NOT_REQUIRED, Status.AGED_OUT):
            # Nothing to protect: no message would be sent for these anyway, and
            # flagging them would bury the real review items in noise.
            continue
        entry.status = Status.REQUIRES_REVIEW
        note = (
            "chart and registry disagree about this antigen; held until a human "
            "reconciles the records"
        )
        if note not in entry.notes:      # idempotent: safe to call twice
            entry.notes.append(note)
    if reconciliation.has_unknown_codes:
        for entry in forecast.antigens.values():
            if entry.is_open_gap:
                entry.recall_eligible = False
                note = (
                    "unrecognised vaccine code in the record; recall withheld until "
                    "the code is resolved"
                )
                if note not in entry.notes:
                    entry.notes.append(note)
    return forecast


def run_nightly(
    patients: Iterable[PatientInput],
    *,
    as_of: date,
    forecaster: Forecaster | None = None,
    adjudicator: Adjudicator | None = None,
    reviewed_by: str | None = None,
    audit: Any = None,
) -> NightlyResult:
    """Reconcile, adjudicate, forecast, hold. Produces everything downstream needs.

    `reviewed_by` is required whenever an `adjudicator` is supplied: an
    adjudication is a proposal and `apply_adjudications` will not fold a machine
    conclusion into the record without a named human (README 3.4). Passing an
    adjudicator with no reviewer is a configuration error, not a permission to
    auto-apply, so it raises.
    """
    if adjudicator is not None and not reviewed_by:
        raise ValueError(
            "an adjudicator requires reviewed_by: machine reconciliation "
            "conclusions are not applied without a named human reviewer"
        )
    forecaster = forecaster or LocalRulesForecaster()
    result = NightlyResult()

    for patient in patients:
        reconciliation = reconcile(patient.chart, patient.registry)

        if adjudicator is not None and reconciliation.ambiguous:
            outcomes = adjudicator.adjudicate_reconciliation(
                reconciliation, patient_id=patient.patient_id
            )
            reconciliation, queue = apply_adjudications(
                reconciliation,
                outcomes,
                reviewed_by=str(reviewed_by),
                audit=audit,
                patient_id=patient.patient_id,
            )
            result.review_queue.extend(queue)
        elif reconciliation.ambiguous:
            result.review_queue.extend(
                HumanReviewItem(
                    patient_id=patient.patient_id,
                    pair=pair,
                    reason="no adjudicator configured",
                )
                for pair in reconciliation.ambiguous
            )

        doses: list[AdministeredDose] = reconciliation.merged_doses()
        forecast = forecaster.forecast(
            patient_id=patient.patient_id,
            dob=patient.dob,
            doses=doses,
            as_of=as_of,
        )
        apply_reconciliation_holds(forecast, reconciliation)

        result.forecasts[patient.patient_id] = forecast
        result.reconciliations[patient.patient_id] = reconciliation
        result.patients[patient.patient_id] = patient.info()

    return result
