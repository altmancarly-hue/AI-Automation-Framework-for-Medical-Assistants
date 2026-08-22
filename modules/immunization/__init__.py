"""I-02 — Immunization gap closure and recall.

Deterministic everywhere it matters. A model appears in exactly two places, both
of them narrow and both of them wrapped in post-conditions:

  * `adjudicate.py` — is this chart dose and this registry dose the same event?
    Only for pairs the rules could not settle. README Appendix A.2 verbatim.
  * `messaging.py` — rewording a physician-approved template, and triaging an
    inbound reply into a fixed enum.

The forecaster NEVER calls a model. README I-02: "Using an LLM to determine
whether a child is due for a vaccine would be actively negligent."

Typical wiring:

    from modules.immunization import (
        LocalRulesForecaster, PatientInput, RecallEngine, build_huddle, run_nightly,
    )

    nightly = run_nightly(patients, as_of=today)
    recall = RecallEngine(db, gateway, validation=validation_result)
    recall.run(list(nightly.forecasts.values()), now=now, patients=nightly.patients)
    sheet = build_huddle(db, for_date=tomorrow, forecasts=nightly.forecasts,
                         reconciliations=nightly.reconciliations)
"""

from .adjudicate import (
    ADJUDICATION_SCHEMA,
    ADJUDICATION_SYSTEM_PROMPT,
    AdjudicationOutcome,
    Adjudicator,
    HumanReviewItem,
    apply_adjudications,
)
from .cvx import (
    CVX,
    TRADE_NAMES,
    Antigen,
    VaccineProduct,
    antigens_for,
    components_for,
    expand,
    is_known,
    normalise_code,
    product_for,
    same_antigen_set,
    shares_any_antigen,
    unknown_codes,
)
from .forecast import (
    AdministeredDose,
    AntigenForecast,
    CrossCheckForecaster,
    Disagreement,
    DoseEvaluation,
    DosePrecision,
    Forecaster,
    LocalRulesForecaster,
    PatientForecast,
    RegistryForecaster,
    Schedule,
    Status,
    ValidationResult,
    add_period,
    validate_against_reference,
)
from .huddle import HuddleSheet, PatientCard, Provenance, build_huddle
from .matcher import (
    AmbiguousPair,
    Determination,
    DoseRecord,
    Duplicate,
    MatchedPair,
    Reconciliation,
    reconcile,
)
from .messaging import (
    DraftResult,
    MessageDrafter,
    ReplyClassification,
    ReplyIntent,
    ReplyTriage,
)
from .pipeline import (
    NightlyResult,
    PatientInput,
    apply_reconciliation_holds,
    run_nightly,
)
from .recall import (
    DEFAULT_CADENCE,
    RECALL_CONSENT_BASIS,
    GapCandidate,
    RecallEngine,
    RecallNotAuthorized,
    RecallStep,
)

__all__ = [
    "ADJUDICATION_SCHEMA",
    "ADJUDICATION_SYSTEM_PROMPT",
    "AdjudicationOutcome",
    "Adjudicator",
    "AdministeredDose",
    "AmbiguousPair",
    "Antigen",
    "AntigenForecast",
    "CVX",
    "CrossCheckForecaster",
    "DEFAULT_CADENCE",
    "Determination",
    "Disagreement",
    "DoseEvaluation",
    "DosePrecision",
    "DoseRecord",
    "DraftResult",
    "Duplicate",
    "Forecaster",
    "GapCandidate",
    "HuddleSheet",
    "HumanReviewItem",
    "LocalRulesForecaster",
    "MatchedPair",
    "MessageDrafter",
    "NightlyResult",
    "PatientCard",
    "PatientForecast",
    "PatientInput",
    "Provenance",
    "RECALL_CONSENT_BASIS",
    "Reconciliation",
    "RecallEngine",
    "RecallNotAuthorized",
    "RecallStep",
    "RegistryForecaster",
    "ReplyClassification",
    "ReplyIntent",
    "ReplyTriage",
    "Schedule",
    "Status",
    "TRADE_NAMES",
    "VaccineProduct",
    "ValidationResult",
    "add_period",
    "antigens_for",
    "apply_adjudications",
    "apply_reconciliation_holds",
    "build_huddle",
    "components_for",
    "expand",
    "is_known",
    "normalise_code",
    "product_for",
    "reconcile",
    "run_nightly",
    "same_antigen_set",
    "shares_any_antigen",
    "unknown_codes",
    "validate_against_reference",
]
