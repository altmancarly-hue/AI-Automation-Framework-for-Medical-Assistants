"""I-08 — vaccine cold chain telemetry. Contains no model, deliberately.

README I-08: *"Include this initiative in an 'AI program' specifically to
demonstrate the discipline of not using AI where it does not belong. A proposal
that reaches for a language model in every section is not a technology strategy;
it is a shopping list."*
"""

from .excursion import (
    DISPOSITION_ADVISORY,
    ComplianceReport,
    DailyRecord,
    ExcursionRecord,
    LotDisposition,
    VaccineLot,
    build_compliance_report,
    open_excursion,
)
from .monitor import (
    Alert,
    AlertKind,
    ColdChainConfig,
    ColdChainMonitor,
    Reading,
    ScriptedFeed,
    SensorFeed,
    Severity,
    StorageUnit,
    UnknownUnitType,
    UnreviewedThresholds,
)

__all__ = [
    "DISPOSITION_ADVISORY",
    "Alert", "AlertKind", "ColdChainConfig", "ColdChainMonitor",
    "ComplianceReport", "DailyRecord", "ExcursionRecord", "LotDisposition",
    "Reading", "ScriptedFeed", "SensorFeed", "Severity", "StorageUnit",
    "UnknownUnitType", "UnreviewedThresholds", "VaccineLot",
    "build_compliance_report", "open_excursion",
]
