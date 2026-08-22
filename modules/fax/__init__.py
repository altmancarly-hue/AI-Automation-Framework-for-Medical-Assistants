"""I-06 — Inbound fax and document ingestion.

README I-06: "Yes for classification, no for the rest." The split is enforced in
the package layout and by `make lint`:

  * `ocr.py`, `match.py`, `route.py` contain no model call and never will.
    Patient matching in particular is entirely deterministic -- exact date of
    birth plus Jaro-Winkler on the name -- and every ambiguity resolves toward a
    human queue, because a document filed to the wrong child is a chart-integrity
    event that may never be found.
  * `classify.py` is one model call against a CLOSED taxonomy, gated on a
    measured eval set with per-class recall floors. A classifier that has not
    been measured cannot route anything.
  * `urgency.py` runs README Appendix A.3 and then overrules it. The override is
    one-way: rules can escalate to urgent and can never lower anything, because
    the failure being guarded is a critical value sitting in a routine queue.

Typical wiring:

    pipeline = FaxPipeline(
        engines=[PaddleOCREngine(), TesseractEngine()],
        classifier=DocumentClassifier(client, audit=audit),
        matcher=PatientMatcher(panel),
        triage=UrgencyTriage(client, ranges=ReferenceRanges.load(), audit=audit),
        gate=approved_gate,
        audit=audit,
    )
    processed = pipeline.process("fax_00412.tiff", document_id="fax_00412")
"""

from .classify import (
    CLASSIFY_SCHEMA,
    CLASSIFY_SYSTEM_PROMPT,
    CLINICAL_TYPES,
    DOCUMENT_TYPES,
    MIN_CONFIDENCE,
    RECALL_FLOORS,
    Classification,
    ClassifierGate,
    ClassifierNotValidated,
    DocumentClassifier,
    EvalCase,
    EvalResult,
    evaluate,
)
from .match import (
    NAME_THRESHOLD,
    Candidate,
    MatchOutcome,
    MatchResult,
    PanelPatient,
    PatientMatcher,
    jaro,
    jaro_winkler,
    normalise_name,
    split_name,
)
from .ocr import (
    LOW_CONFIDENCE_THRESHOLD,
    OCREngine,
    OCRResult,
    OCRUnavailable,
    Page,
    PaddleOCREngine,
    ScriptedOCR,
    TesseractEngine,
    chain,
)
from .pipeline import (
    FaxPipeline,
    InboundMonitor,
    ProcessedDocument,
    parse_document_date,
)
from .route import (
    ImmunizationHandoff,
    Queue,
    ReviewTask,
    RoutedDocument,
    TaskKind,
    extract_immunization_doses,
    route,
)
from .urgency import (
    DEFAULT_RANGES_PATH,
    URGENCY_LEVELS,
    URGENCY_SCHEMA,
    URGENCY_SYSTEM_PROMPT,
    AbnormalValue,
    ReferenceRanges,
    UnreviewedRanges,
    UrgencyResult,
    UrgencyTriage,
    rank,
)

__all__ = [
    "AbnormalValue", "CLASSIFY_SCHEMA", "CLASSIFY_SYSTEM_PROMPT", "CLINICAL_TYPES",
    "Candidate", "Classification", "ClassifierGate", "ClassifierNotValidated",
    "DEFAULT_RANGES_PATH", "DOCUMENT_TYPES", "DocumentClassifier", "EvalCase",
    "EvalResult", "FaxPipeline", "ImmunizationHandoff", "InboundMonitor",
    "LOW_CONFIDENCE_THRESHOLD", "MIN_CONFIDENCE", "MatchOutcome", "MatchResult",
    "NAME_THRESHOLD", "OCREngine", "OCRResult", "OCRUnavailable", "Page",
    "PaddleOCREngine", "PanelPatient", "PatientMatcher", "ProcessedDocument",
    "Queue", "RECALL_FLOORS", "ReferenceRanges", "ReviewTask", "RoutedDocument",
    "ScriptedOCR", "TaskKind", "TesseractEngine", "URGENCY_LEVELS",
    "URGENCY_SCHEMA", "URGENCY_SYSTEM_PROMPT", "UnreviewedRanges", "UrgencyResult",
    "UrgencyTriage", "chain", "evaluate", "extract_immunization_doses", "jaro",
    "jaro_winkler", "normalise_name", "parse_document_date", "rank", "route",
    "split_name",
]
