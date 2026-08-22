"""The structured data the form is filled from, and where each value came from.

README I-01: *"Pull structured clinical data from the chart | FHIR API query |
Deterministic. The data is already structured."* That sentence is the whole
design of this module. There is no model here and there is no parsing: the
practice's EHR already holds a height as a number, and the job is to carry that
number to a box on a form without losing track of where it came from.

PROVENANCE IS NOT DECORATION. README I-01's review screen puts "source data with
provenance" next to the filled form, and its audit requirement is "every field
write records source, method, confidence, reviewing user, timestamp". An MA
reviewing a form has one question about each filled box -- *where did that come
from and when was it measured* -- and a pipeline that cannot answer it turns the
review into a rubber stamp, which is the automation-complacency risk the README
rates High.

So every value in a `ChartRecord` is a `SourceValue`: the value, the system it
came from, the resource that holds it, and the date it was recorded. `resolve()`
walks a template's dotted source path and returns the whole thing, never the
bare value.

STALENESS IS A FIRST-CLASS PROPERTY. A height from a visit four years ago is
still a number, and it will fill a box, and the form will look complete. Illinois
requires the examination to be recent; a form filled from a stale vital is a
compliance failure that looks like a success. `SourceValue.age_days` and
`ChartRecord.stale_values()` exist so the review screen and the release gate can
both see it.

WHAT THIS MODULE DOES NOT DO. It does not fetch. `ChartSource` is a protocol with
one method, and `StaticChartSource` -- a shipped test double, not a mock of this
module's logic -- is what the tests and the demo run against. A real deployment
implements the protocol over FHIR R4 (Patient, Observation, AllergyIntolerance,
MedicationStatement, Condition, Immunization) and everything downstream is
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Mapping, Protocol, Sequence

__all__ = [
    "CHART_SCHEMA",
    "is_known_path",
    "SourceValue",
    "ChartRecord",
    "ChartSource",
    "StaticChartSource",
    "ChartUnavailable",
    "MissingPath",
    "DEFAULT_STALE_DAYS",
]

#: A vital older than this is reported as stale. Illinois school physicals are
#: valid for a year for most purposes; the number lives here as one constant a
#: practice can change, rather than in three call sites.
DEFAULT_STALE_DAYS = 365


#: The shape a `ChartRecord` is allowed to have, and therefore the set of source
#: paths a template may name. `"*"` means "any key or index here".
#:
#: This exists because `resolve()` alone could not tell a template typo from an
#: empty chart: `_container_is_known` only ever checked NON-FINAL segments, so
#: `allergies_list` and `vitals.height_inches` both returned None and the box was
#: skipped with the reason "the chart holds no value for this field" -- a
#: statement about the child, shown to the MA, about a configuration bug. A
#: child with a documented penicillin allergy got a blank allergy box on a form
#: a physician signed.
CHART_SCHEMA: Mapping[str, Any] = {
    "patient": {
        "last_name": None, "first_name": None, "middle_initial": None,
        "date_of_birth": None, "sex_on_record": None, "address": None,
        "guardian_name": None, "phone": None, "mrn": None,
    },
    "exam": {"date": None, "performed_by": None},
    "vitals": {
        "height_in": None, "height_cm": None, "weight_lb": None, "weight_kg": None,
        "bmi": None, "bmi_percentile": None, "blood_pressure": None,
        "head_circumference_cm": None, "pulse": None, "temperature_f": None,
    },
    "screenings": {
        "vision": None, "hearing": None, "dental_exam_date": None,
        "lead_risk_assessed": None, "tb_risk_assessed": None,
        "diabetes_risk_assessed": None, "scoliosis": None, "developmental": None,
    },
    "labs": {
        "hemoglobin": None, "hematocrit": None, "lead_ug_dl": None,
        "glucose": None, "urinalysis": None,
    },
    "allergies": None,
    "medications": None,
    "conditions": None,
    "immunizations": {"*": {"*": None}},
}


def is_known_path(path: str, schema: Mapping[str, Any] | None = None) -> bool:
    """True when `path` names something `CHART_SCHEMA` allows.

    Called by `TemplateStore.load()`, so a renamed or mistyped source path is a
    LOAD-TIME refusal rather than a run-time blank box.
    """
    node: Any = CHART_SCHEMA if schema is None else schema
    if not path:
        return False
    for part in path.split("."):
        if node is None:
            return False
        if not isinstance(node, Mapping):
            return False
        if part in node:
            node = node[part]
        elif "*" in node:
            node = node["*"]
        else:
            return False
    return True


class ChartUnavailable(RuntimeError):
    """Raised when the chart system cannot be reached at all.

    Distinct from a chart that HAS no value for a field. An unreachable EHR must
    not look like a patient with no allergies.
    """


class MissingPath(KeyError):
    """Raised when a template names a source path the chart shape does not have.

    A template edit that renames a path must fail loudly at fill time. Returning
    None would leave the box blank and the form would look merely incomplete
    rather than misconfigured, and nobody would ever find it.
    """


@dataclass(frozen=True)
class SourceValue:
    """One value, and everything the reviewer needs to judge it."""

    value: Any
    #: "ehr" | "registry" | "reconciled" | "practice"
    system: str
    #: The record it came from -- a FHIR resource reference, a registry export
    #: id, whatever identifies it in the source system.
    resource: str = ""
    #: When the value was RECORDED (the observation date), not when it was
    #: fetched. A height fetched this morning can be four years old.
    recorded: date | None = None
    #: Set on values this pipeline derived rather than read, so the review
    #: screen can say so. Nothing here is derived by a model.
    derived_from: tuple[str, ...] = ()
    #: True when the value IS a historical event rather than a measurement of a
    #: current state. An immunization given in 2018 is not a stale height -- its
    #: date is the datum, and it does not get fresher. Without this flag the
    #: staleness report fired on every dose on every form, which is the shape of
    #: alert that trains a reviewer to skip the list the real stale vital is in.
    historical: bool = False

    def age_days(self, as_of: date) -> int | None:
        if self.recorded is None:
            return None
        return (as_of - self.recorded).days

    def is_stale(self, as_of: date, *, stale_days: int = DEFAULT_STALE_DAYS) -> bool:
        """True when the value is older than the window, or UNDATED.

        An undated value counts as stale. A chart value with no recorded date
        cannot be shown to be recent, and "cannot be shown to be recent" is the
        thing the reviewer needs told -- the alternative is a form filled from a
        value of unknown vintage that nothing on the screen questions.
        """
        if self.historical:
            return False
        age = self.age_days(as_of)
        return age is None or age > stale_days

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": _plain(self.value),
            "system": self.system,
            "resource": self.resource,
            "recorded": self.recorded.isoformat() if self.recorded else None,
            "derived_from": list(self.derived_from),
            "historical": self.historical,
        }


def _plain(value: Any) -> Any:
    """JSON-safe rendering. Dates become ISO strings, mappings recurse."""
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


@dataclass
class ChartRecord:
    """Everything pulled for one patient, keyed the way templates address it.

    The shape is deliberately flat and boring: `patient`, `vitals`, `exam`,
    `screenings`, `labs`, `allergies`, `medications`, `conditions`,
    `immunizations`. Template source paths are dotted walks into this, so
    `vitals.height_in` and `immunizations.dtap.0` both work, and adding a field
    to a form is a YAML edit.
    """

    patient_id: str
    data: dict[str, Any] = field(default_factory=dict)
    #: Systems that were asked and did not answer. A form filled while the
    #: registry was down is filled from half the sources, and the form has to
    #: say so -- README I-01: "degrade gracefully to chart-only fill, flag the
    #: form as 'registry not reconciled', continue to function".
    unavailable: list[str] = field(default_factory=list)
    fetched: date | None = None

    def resolve(self, path: str) -> SourceValue | None:
        """Walk a dotted source path. Returns None when the chart has no value.

        Raises `MissingPath` when the path names a CONTAINER that does not
        exist -- a typo in a template -- and returns None when the container
        exists and is simply empty for this patient. The difference matters: one
        is a bug and one is a fact about the child.
        """
        if not path:
            return None
        parts = path.split(".")
        node: Any = self.data
        walked: list[str] = []
        for index, part in enumerate(parts):
            walked.append(part)
            last = index == len(parts) - 1
            if isinstance(node, Mapping):
                if part not in node:
                    # A path the SCHEMA allows but this chart does not carry is a
                    # fact about the child (no allergies recorded); a path the
                    # schema does not allow is a template bug. `TemplateStore`
                    # refuses the second at load time, and this is the backstop
                    # for a record built by hand.
                    if is_known_path(path):
                        return None
                    raise MissingPath(
                        f"{path!r}: no {'.'.join(walked)!r} in this chart record, "
                        "and it is not a path CHART_SCHEMA allows -- this is a "
                        "template naming a field that does not exist, not a "
                        "patient with no data"
                    )
                node = node[part]
            elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
                if not part.isdigit():
                    raise MissingPath(
                        f"{path!r}: {'.'.join(walked[:-1])!r} is a list and "
                        f"{part!r} is not an index"
                    )
                position = int(part)
                if position >= len(node):
                    # A dose that has not been given yet. Not an error: the box
                    # for DTaP #5 is empty for a four-year-old.
                    return None
                node = node[position]
            else:
                raise MissingPath(
                    f"{path!r}: {'.'.join(walked[:-1])!r} holds a plain value "
                    f"and cannot be walked into"
                )
        if node is None:
            return None
        if isinstance(node, SourceValue):
            return node
        # A bare value in the record means somebody built a ChartRecord by hand
        # without provenance. Say so rather than inventing a source.
        return SourceValue(value=node, system="unknown", resource="")

    def stale_values(
        self, as_of: date, *, stale_days: int = DEFAULT_STALE_DAYS
    ) -> list[dict[str, Any]]:
        """Every dated value older than the window, deepest path first."""
        found: list[dict[str, Any]] = []
        _walk(self.data, [], found, as_of, stale_days)
        return sorted(found, key=lambda item: item["path"])

    def as_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "fetched": self.fetched.isoformat() if self.fetched else None,
            "unavailable": list(self.unavailable),
            "data": _plain_record(self.data),
        }


def _container_is_known(root: Any, parts: Sequence[str]) -> bool:
    node: Any = root
    for part in parts:
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        elif (
            isinstance(node, Sequence)
            and not isinstance(node, (str, bytes))
            and part.isdigit()
            and int(part) < len(node)
        ):
            node = node[int(part)]
        else:
            return False
    return True


def _walk(
    node: Any,
    trail: list[str],
    found: list[dict[str, Any]],
    as_of: date,
    stale_days: int,
) -> None:
    if isinstance(node, SourceValue):
        if node.is_stale(as_of, stale_days=stale_days):
            found.append(
                {
                    "path": ".".join(trail),
                    "system": node.system,
                    "recorded": node.recorded.isoformat() if node.recorded else None,
                    "age_days": node.age_days(as_of),
                }
            )
        return
    if isinstance(node, Mapping):
        for key, value in node.items():
            _walk(value, trail + [str(key)], found, as_of, stale_days)
    elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        for index, value in enumerate(node):
            _walk(value, trail + [str(index)], found, as_of, stale_days)


def _plain_record(node: Any) -> Any:
    if isinstance(node, SourceValue):
        return node.as_dict()
    if isinstance(node, Mapping):
        return {k: _plain_record(v) for k, v in node.items()}
    if isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        return [_plain_record(v) for v in node]
    return _plain(node)


class ChartSource(Protocol):
    """One method. A real implementation queries FHIR R4; this one does not."""

    name: str

    def fetch(self, patient_id: str, *, as_of: date) -> ChartRecord: ...


@dataclass
class StaticChartSource:
    """A shipped test double holding pre-built records.

    Not a mock of this module's logic: it stands in for the EHR, the way
    `EchoTransport` stands in for a model server. Everything downstream --
    resolution, transforms, reconciliation, filling, review -- is the real code.
    """

    name: str = "static-ehr"
    records: dict[str, ChartRecord] = field(default_factory=dict)
    #: Patient ids for which this source raises, so the degraded path is
    #: exercisable without unplugging anything.
    unreachable: frozenset[str] = frozenset()

    def fetch(self, patient_id: str, *, as_of: date) -> ChartRecord:
        if patient_id in self.unreachable:
            raise ChartUnavailable(
                f"the chart system did not answer for {patient_id!r}. No form is "
                "filled from a partial pull: an unreachable EHR must not look "
                "like a patient with no allergies."
            )
        record = self.records.get(patient_id)
        if record is None:
            raise ChartUnavailable(f"no chart record for {patient_id!r}")
        record.fetched = as_of
        return record
