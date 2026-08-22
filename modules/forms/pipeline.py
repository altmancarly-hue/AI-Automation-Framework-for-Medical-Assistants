"""One form, end to end: arrive, detect, pull, reconcile, fill, review, release.

README I-01's target state is ten numbered steps. This is those steps in order,
with the gates in the places the README puts them, and nothing clever in
between. It is a function and a small dataclass rather than an agent, for the
same reason I-03's nightly batch is: the order never varies, every branch is
known in advance, and a model choosing the order could only ever choose wrong.

WHERE IT REFUSES TO CONTINUE, and where it carries on with a flag -- the
distinction is the whole design:

    REFUSES                              CARRIES ON, FLAGGED
    the form type is unrecognised        the registry was not consulted
    the template is uncalibrated         a dose is disputed
    the chart system is unreachable      a required box could not be filled
    the patient was not confirmed        a value was truncated
                                         a vital is stale

The left column means no document is produced at all. The right column means a
document is produced, marked, and blocked from the signature queue until a
person deals with it. README I-01 asks for exactly that shape: *"degrade
gracefully to chart-only fill, flag the form as 'registry not reconciled',
continue to function"* -- but also *"Never auto-match on a fuzzy name alone."*

THE PATIENT MATCH IS NOT DONE HERE, ON PURPOSE. README I-01 step 3 is "Patient
is matched by name plus date of birth, WITH A CONFIRMATION STEP", and I-06
already has that matcher -- the one that treats a recorded twin flag as binding
and refuses a near tie. So `process_form` takes a `patient_id` that a person or
that matcher has already settled, and refuses to run without one. A pipeline
that resolves its own patient is a pipeline that can resolve it wrongly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence

from modules.immunization.matcher import DoseRecord

from .chart import ChartRecord, ChartSource, ChartUnavailable
from .fill import FilledForm, FormFiller
from .lifecycle import FormRequest, FormState, FormTracker
from .probe import ProbeProgram
from .reconcile import (
    FormReconciliation,
    build_immunization_block,
    grid_capacity_findings,
    reconcile_for_form,
)
from .review import ReleaseGate, ReviewPayload, build_review
from .templates import FormTemplate, TemplateStore, UncalibratedTemplate

__all__ = ["PreparedForm", "FormPipeline", "PatientNotConfirmed"]


class PatientNotConfirmed(RuntimeError):
    """Raised when a form is prepared without a settled patient identity."""


@dataclass
class PreparedForm:
    """Everything one pass produced. The review screen reads this."""

    request: FormRequest
    template: FormTemplate
    record: ChartRecord | None = None
    reconciliation: FormReconciliation | None = None
    filled: FilledForm | None = None
    review: ReviewPayload | None = None
    probe_field: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def releasable(self) -> bool:
        return self.review is not None and self.review.releasable

    def as_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.as_dict(),
            "form_type": self.template.form_type,
            "releasable": self.releasable,
            "probe_field": self.probe_field,
            "errors": list(self.errors),
            "reconciliation": (
                self.reconciliation.as_dict() if self.reconciliation else None
            ),
            "filled": self.filled.as_dict() if self.filled else None,
            "review": self.review.as_dict() if self.review else None,
        }


@dataclass
class FormPipeline:
    """Wires the deterministic steps together. No model runs in this file."""

    store: TemplateStore
    chart_source: ChartSource
    filler: FormFiller
    tracker: FormTracker
    gate: ReleaseGate = field(default_factory=ReleaseGate)
    probes: ProbeProgram | None = None
    audit: Any = None
    INITIATIVE: str = "I-01"

    def prepare(
        self,
        request: FormRequest,
        *,
        blank_pdf: str,
        destination: str,
        now: datetime,
        registry_doses: Sequence[DoseRecord] | None = None,
        chart_doses: Sequence[DoseRecord] | None = None,
        registry_note: str = "",
        actor: str = "system:forms",
    ) -> PreparedForm:
        """Steps 4 through 7. Raises only for the left-column conditions above."""
        if not request.patient_id.strip():
            raise PatientNotConfirmed(
                "this form has no confirmed patient. README I-01: never "
                "auto-match on a fuzzy name alone -- the match happens before "
                "the pipeline runs, and it happens with a person or with I-06's "
                "matcher, which treats a recorded twin flag as binding."
            )
        # Raises `UncalibratedTemplate` if the boxes were never measured. That
        # refusal belongs here rather than at render time: the point is that no
        # document is produced, not that a wrong one is produced and discarded.
        template = self.store.for_filling(request.form_type)

        prepared = PreparedForm(request=request, template=template)

        as_of = now.date()
        record = self.chart_source.fetch(request.patient_id, as_of=as_of)

        reconciliation = reconcile_for_form(
            chart_doses if chart_doses is not None else [],
            registry_doses,
            registry_note=registry_note,
        )
        record.data["immunizations"] = build_immunization_block(reconciliation)
        # A form is a fixed number of boxes and a chart is not. Rows sort
        # oldest-first, so without this it is the most recent dose -- the one the
        # school checks -- that silently falls off the end.
        discrepancies = reconciliation.discrepancies() + grid_capacity_findings(
            template, reconciliation
        )
        if registry_doses is None and "registry" not in record.unavailable:
            record.unavailable.append("registry")

        prepared.record = record
        prepared.reconciliation = reconciliation

        # -- the probe, before the fill, so the injected value goes through the
        # -- same render, highlight, audit and review path as a real one. A probe
        # -- inserted afterwards would be testing a different code path from the
        # -- one it claims to measure.
        overrides: dict[str, str] = {}
        if self.probes is not None and self.probes.outstanding_for(request.request_id):
            # This request was probed on an earlier pass and is being re-filled,
            # so that probe was never presented as a finished form. Withdraw it
            # rather than carrying it forward: an unresolved probe blocks this
            # request at the release gate permanently.
            self.probes.withdraw(
                request.request_id,
                reason="the form was re-filled before this probe was reviewed",
                now=now,
            )
        if self.probes is not None and self.probes.should_probe(request.request_id):
            dry_run = self.filler.fill(
                template, record, blank_pdf=blank_pdf, destination=destination,
                now=now, discrepancies=discrepancies,
                user_id=actor, dry_run=True,
            )
            built = self.probes.build_override(
                dry_run, request_id=request.request_id, now=now
            )
            if built:
                overrides.update(built)
                prepared.probe_field = next(iter(built))

        filled = self.filler.fill(
            template,
            record,
            blank_pdf=blank_pdf,
            destination=destination,
            now=now,
            discrepancies=discrepancies,
            overrides=overrides,
            probe_field=prepared.probe_field,
            user_id=actor,
        )
        prepared.filled = filled

        fill_id = f"{request.request_id}@{now.isoformat()}"
        prepared.review = build_review(
            template, filled, record,
            as_of=as_of,
            gate=self.gate,
            probe_outstanding=(
                self.probes is not None
                and self.probes.outstanding_for(request.request_id)
            ),
        )

        if request.state == FormState.RECEIVED:
            self.tracker.advance(request, FormState.FILLED, actor=actor, now=now)
        elif request.state in (FormState.BLOCKED, FormState.MA_REVIEW):
            self.tracker.advance(
                request, FormState.FILLED, actor=actor, now=now,
                note="re-filled after the underlying issue was addressed",
            )
        self.tracker.advance(
            request, FormState.MA_REVIEW, actor=actor, now=now,
            note=f"{len(filled.auto_filled)} field(s) auto-filled",
        )
        # AFTER the transitions: moving to `filled` clears any review attached
        # to the previous document, which is the point of clearing it.
        self.tracker.record_review_outcome(
            request, fill_id=fill_id, blockers=prepared.review.blockers
        )
        self.tracker.notify(
            request, channel="sms", now=now,
            message="Your form has been prepared and is being checked by our staff.",
        )
        return prepared
