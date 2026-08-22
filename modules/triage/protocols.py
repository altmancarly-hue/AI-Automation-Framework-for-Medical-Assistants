"""The MA's two taps: which protocol, and which disposition.

THIS IS THE MOST IMPORTANT MODULE IN I-04 AND IT CONTAINS NO INTELLIGENCE AT ALL.

README I-04: "Under no circumstances does the system suggest a disposition. Not
as a hint, not as a 'consider,' not greyed out... This is the single most
important design constraint in the entire document."

The protocol and the disposition enter the system exactly one way: a human
tapped them. So this module offers a registry to tap FROM and a record of what
was tapped, and it deliberately offers nothing else. There is no
`suggest_protocol`, no `likely_disposition`, no ranking of options by
plausibility, no reordering of the list based on the transcript. A searchable
alphabetical list is the entire user interface, because anything cleverer is a
machine-generated clinical judgement wearing a convenience costume.

Two smaller points that matter:

  * **Protocol CONTENT is licensed and is not in this repo.** Schmitt-Thompson
    protocols are copyrighted. `config/triage_protocols.yaml` holds identifiers
    only; the MA reads the protocol from the practice's licensed copy and this
    system records which one. An identifier plus a library version is what
    demonstrates a standard of care was followed -- "used the fever protocol" is
    not evidence.
  * **Escalations name the licensed professional.** Under 225 ILCS 60/54.2 an
    unlicensed MA acts under physician delegation. An escalation is the moment a
    licensed professional takes the decision, so the record has to say who. The
    registry enforces this per-disposition rather than in application code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

import yaml

from modules.scheduling.models import iso

__all__ = [
    "DEFAULT_PROTOCOL_PATH",
    "Protocol",
    "Disposition",
    "ProtocolRegistry",
    "MATaps",
    "TapsIncomplete",
]

DEFAULT_PROTOCOL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "triage_protocols.yaml",
)


class TapsIncomplete(RuntimeError):
    """Raised when a note is assembled before the MA has tapped both fields.

    Fails closed and stays closed. There is no default disposition, no "most
    likely" fallback, and no rendering path that omits the field -- a note that
    silently lacks a disposition is exactly the undocumented triage call README
    I-04 calls "indefensible".
    """


@dataclass(frozen=True)
class Protocol:
    id: str
    title: str
    age_min_months: int = 0
    age_max_months: int = 216

    def applies_at(self, age_months: int) -> bool:
        return self.age_min_months <= age_months <= self.age_max_months


@dataclass(frozen=True)
class Disposition:
    id: str
    label: str
    rank: int
    requires_supervising_professional: bool = False


@dataclass(frozen=True)
class MATaps:
    """What the human selected. The only source of protocol and disposition."""

    protocol_id: str
    disposition_id: str
    tapped_by: str
    tapped_utc: datetime
    supervising_professional_id: str | None = None
    #: Free text the MA typed, if any. Never machine-generated.
    ma_note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "disposition_id": self.disposition_id,
            "tapped_by": self.tapped_by,
            "tapped_utc": iso(self.tapped_utc),
            "supervising_professional_id": self.supervising_professional_id,
        }


class ProtocolRegistry:
    """The searchable list the MA taps from. No ranking, no suggestion."""

    def __init__(
        self, data: Mapping[str, Any], *, allow_placeholder: bool = False
    ) -> None:
        self.allow_placeholder = allow_placeholder
        self.source = str(data.get("source", "unspecified"))
        self.library_version = str(data.get("library_version", "unversioned"))
        self.protocols: dict[str, Protocol] = {
            str(p["id"]): Protocol(
                id=str(p["id"]),
                title=str(p["title"]),
                age_min_months=int(p.get("age_min_months", 0)),
                age_max_months=int(p.get("age_max_months", 216)),
            )
            for p in data.get("protocols", [])
        }
        self.dispositions: dict[str, Disposition] = {
            str(d["id"]): Disposition(
                id=str(d["id"]),
                label=str(d["label"]),
                rank=int(d.get("rank", 0)),
                requires_supervising_professional=bool(
                    d.get("requires_supervising_professional", False)
                ),
            )
            for d in data.get("dispositions", [])
        }
        if not self.protocols or not self.dispositions:
            raise ValueError(
                "the protocol registry is empty; the MA has nothing to tap and "
                "no note can be assembled"
            )

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str] = DEFAULT_PROTOCOL_PATH,
        *,
        allow_placeholder: bool = False,
    ) -> "ProtocolRegistry":
        """Load the registry. Refuses placeholders unless asked explicitly.

        The repo ships placeholder identifiers so the code runs and the tests
        have something to select. Shipping them to a clinic would put
        "PLACEHOLDER-FEVER" in a chart, so validation refuses by default and
        the demo and the tests opt in on purpose.
        """
        with open(path, "r", encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh), allow_placeholder=allow_placeholder)

    @property
    def is_placeholder(self) -> bool:
        """True while the licensed protocol list has not been loaded.

        Surfaced so the deployment checklist can assert on it. Shipping with
        placeholder identifiers would put "PLACEHOLDER-FEVER" in a chart.
        """
        return self.library_version.startswith("placeholder")

    def search(
        self, term: str = "", *, age_months: int | None = None
    ) -> list[Protocol]:
        """Alphabetical, filtered by substring and by age applicability.

        Age filtering is a safety filter, not a suggestion: a newborn-fever
        protocol offered for a fourteen-year-old is a mis-tap waiting to happen.
        The order is alphabetical and never changes based on anything the model
        saw -- see the module docstring.
        """
        term = term.strip().lower()
        found = [
            p for p in self.protocols.values()
            if (not term or term in p.title.lower() or term in p.id.lower())
            and (age_months is None or p.applies_at(age_months))
        ]
        return sorted(found, key=lambda p: p.title.lower())

    def disposition_ladder(self) -> list[Disposition]:
        return sorted(self.dispositions.values(), key=lambda d: d.rank)

    # -- validation --------------------------------------------------------

    def validate(self, taps: MATaps | None, *, age_months: int | None = None) -> MATaps:
        """Check a tap set. Raises TapsIncomplete; never repairs or defaults."""
        if self.is_placeholder and not self.allow_placeholder:
            raise TapsIncomplete(
                f"the protocol registry is still placeholder data ({self.source} "
                f"{self.library_version}). Load the practice's licensed "
                "Schmitt-Thompson identifier list before documenting a real call."
            )
        if taps is None:
            raise TapsIncomplete(
                "no protocol or disposition has been tapped. Both come from the "
                "medical assistant and the system will not supply either."
            )
        protocol = self.protocols.get(taps.protocol_id)
        if protocol is None:
            raise TapsIncomplete(
                f"protocol {taps.protocol_id!r} is not in the licensed registry "
                f"({self.source} {self.library_version})"
            )
        if age_months is not None and not protocol.applies_at(age_months):
            raise TapsIncomplete(
                f"protocol {protocol.id!r} does not apply at {age_months} months "
                "of age; re-select rather than recording a mismatch"
            )
        disposition = self.dispositions.get(taps.disposition_id)
        if disposition is None:
            raise TapsIncomplete(
                f"disposition {taps.disposition_id!r} is not on the ladder"
            )
        if not taps.tapped_by.strip():
            raise TapsIncomplete("a tap must record who made it")
        if (
            disposition.requires_supervising_professional
            and not (taps.supervising_professional_id or "").strip()
        ):
            raise TapsIncomplete(
                f"disposition {disposition.id!r} is a licensed-professional "
                "decision (225 ILCS 60/54.2); the record must name who took it"
            )
        return taps

    def describe(self, taps: MATaps) -> dict[str, str]:
        """Human-readable labels for the render layer."""
        return {
            "protocol_id": taps.protocol_id,
            "protocol_title": self.protocols[taps.protocol_id].title,
            "protocol_library": f"{self.source} {self.library_version}",
            "disposition_id": taps.disposition_id,
            "disposition_label": self.dispositions[taps.disposition_id].label,
        }
