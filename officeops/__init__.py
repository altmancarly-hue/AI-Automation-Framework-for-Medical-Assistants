"""Deterministic office automation. No model, no network, no vendor.

This package is the counterpart to `modules/` and not a replacement for it.
`modules/` implements the initiatives that need a language model or genuinely
benefit from one. This package implements the ones that do not need anything at
all beyond Python and a CSV export -- the recurring paperwork that fills a
medical assistant's day and every other desk in a small practice.

Both halves matter, and the honest sequencing is: do this one first. It has no
BAA, no vendor review, no model to validate and no monthly cost, so it can start
the week it is written, and it makes the AI work easier by producing the clean
exports and the measured baselines that Section 10 asks for.

    python3 -m officeops list                     # every task
    python3 -m officeops confirm-list schedule.csv
    python3 -m officeops fridge-log logger.csv --write
"""

from .core import (
    BadValue,
    MissingColumn,
    OfficeOpsError,
    Report,
    Row,
    Table,
    load_mapping,
    load_table,
    parse_bool,
    parse_date,
    parse_datetime,
    parse_float,
)

__all__ = [
    "BadValue", "MissingColumn", "OfficeOpsError", "Report", "Row", "Table",
    "load_mapping", "load_table", "parse_bool", "parse_date", "parse_datetime",
    "parse_float",
]
