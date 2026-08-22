"""I-09 — eligibility verification and denial prevention.

The core is a solved EDI problem. A model touches exactly two things: reading an
insurance card photograph, and bucketing the free text on a denial. Coverage
determinations are deterministic, and a coverage denial is never communicated to
a patient by this system.
"""

from .cards import (
    CARD_SCHEMA,
    CARD_SYSTEM_PROMPT,
    REQUIRED_FIELDS,
    CardExtraction,
    CardReader,
    ExtractedField,
    UnconfirmedCard,
)
from .coverage import (
    Determination,
    Outcome,
    PatientCommunicationRefused,
    PayerRecord,
    PayerTable,
    determine,
    outreach_draft,
)
from .denials import (
    CARC_ROOT_CAUSE,
    DENIAL_SCHEMA,
    DENIAL_SYSTEM_PROMPT,
    AppealDraft,
    Classification,
    Denial,
    DenialClassifier,
    DenialReport,
    RootCause,
    build_denial_report,
    draft_appeal,
)
from .x12 import (
    EB_CODES,
    SERVICE_TYPE_CODES,
    BenefitLine,
    EligibilityRequest,
    MalformedEDI,
    Response271,
    SubsetParser,
    X12Parser,
    build_270,
)

__all__ = [
    "AppealDraft", "BenefitLine", "CARC_ROOT_CAUSE", "CARD_SCHEMA",
    "CARD_SYSTEM_PROMPT", "CardExtraction", "CardReader", "Classification",
    "DENIAL_SCHEMA", "DENIAL_SYSTEM_PROMPT", "Denial", "DenialClassifier",
    "DenialReport", "Determination", "EB_CODES", "EligibilityRequest",
    "ExtractedField", "MalformedEDI", "Outcome", "PatientCommunicationRefused",
    "PayerRecord", "PayerTable", "REQUIRED_FIELDS", "Response271", "RootCause",
    "SERVICE_TYPE_CODES", "SubsetParser", "UnconfirmedCard", "X12Parser",
    "build_270", "build_denial_report", "determine", "draft_appeal",
    "outreach_draft",
]
