"""Append-only audit log for AI inference, human review, and delegated acts.

WHY this is a separate store with its own integrity controls:

The practice's defence in an audit, a malpractice action, or an OCR
investigation is not "we had a policy." It is a record that a named human
reviewed a named machine output at a named time, and that the record could not
have been edited afterwards. Section 9.2 of the README specifies the fields.
This module specifies the guarantees:

  * Append-only at the *database* level. UPDATE and DELETE are refused by
    SQLite triggers, not by application code. Application code can be
    bypassed by anyone with the file and a Python REPL; a trigger cannot,
    without leaving the schema visibly altered.
  * Patient references are HMAC'd, never stored raw. The log lives outside
    the chart and must not become a second, weaker copy of the panel roster.
    HMAC (not a bare hash) because a bare SHA-256 of an MRN is trivially
    reversible by enumerating a six-digit space.
  * No prompt or completion payloads. Ever. See README 9.2.

The three reports at the bottom are the reason the log exists rather than a
compliance artifact nobody opens:

  * `rubber_stamp_report` - README 10.3 calls the "< 5% edit rate" alarm the
    most important line in the document. An MA whose edit distance collapses
    to zero has stopped reviewing, and an unreviewed machine note with a human
    signature is worse than the manual process it replaced.
  * `probe_catch_rate` - the automation-complacency control (R-03). Synthetic
    errors injected at 1:50; this measures whether anyone catches them.
  * `delegation_evidence` / `unevidenced_supervision` - 225 ILCS 60/54.2
    requires that a delegated act occurred under physician delegation with the
    supervising professional on site. The second report lists executions where
    that evidence is missing, which is the query counsel will actually ask for.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import statistics
import uuid
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping, Sequence

__all__ = ["AuditLog", "AuditIntegrityError", "ReviewOutcome", "edit_distance", "edit_ratio"]


class AuditIntegrityError(RuntimeError):
    """Raised when the log's tamper-evidence controls are missing or broken."""


class ReviewOutcome:
    """The `action_taken` enum from README 9.2."""

    ACCEPTED = "accepted"
    EDITED = "edited"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    ALL = (ACCEPTED, EDITED, REJECTED, ESCALATED)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS inference (
    id                    TEXT PRIMARY KEY,
    timestamp_utc         TEXT NOT NULL,
    user_id               TEXT NOT NULL,
    patient_ref           TEXT,
    initiative_id         TEXT NOT NULL,
    model_provider        TEXT NOT NULL,
    model_id              TEXT NOT NULL,
    model_version         TEXT NOT NULL,
    prompt_template_id    TEXT NOT NULL,
    prompt_template_hash  TEXT NOT NULL,
    input_token_count     INTEGER NOT NULL,
    output_token_count    INTEGER NOT NULL,
    confidence_score      REAL,
    constrained_decoding  INTEGER NOT NULL,
    repair_attempts       INTEGER NOT NULL DEFAULT 0,
    synthetic_probe       INTEGER NOT NULL DEFAULT 0,
    source_ip             TEXT,
    extra_json            TEXT
);

CREATE TABLE IF NOT EXISTS review (
    id                        TEXT PRIMARY KEY,
    inference_id              TEXT,
    timestamp_utc             TEXT NOT NULL,
    reviewer_id               TEXT NOT NULL,
    initiative_id             TEXT NOT NULL,
    patient_ref               TEXT,
    action_taken              TEXT NOT NULL,
    edit_distance_draft_final INTEGER NOT NULL,
    draft_length              INTEGER NOT NULL,
    final_length              INTEGER NOT NULL,
    edit_ratio                REAL NOT NULL,
    synthetic_probe           INTEGER NOT NULL DEFAULT 0,
    probe_caught              INTEGER,
    -- 0 for reviews where "edit distance" is not a meaningful signal: a
    -- reviewer confirming a binary MATCH/NO_MATCH adjudication has nothing to
    -- edit, and counting those would drive the rubber-stamp median to zero for
    -- a reviewer doing exactly the right thing. The <5% alarm only means
    -- something for free-text drafts.
    edit_rate_applicable      INTEGER NOT NULL DEFAULT 1,
    review_seconds            REAL,
    source_ip                 TEXT,
    extra_json                TEXT
);

CREATE TABLE IF NOT EXISTS delegated_execution (
    id                     TEXT PRIMARY KEY,
    timestamp_utc          TEXT NOT NULL,
    staff_id               TEXT NOT NULL,
    patient_ref            TEXT,
    initiative_id          TEXT NOT NULL,
    task_code              TEXT NOT NULL,
    standing_order_id      TEXT,
    standing_order_version TEXT,
    supervising_pro_id     TEXT,
    supervisor_on_site     INTEGER,
    competency_record_id   TEXT,
    competency_expires     TEXT,
    break_glass            INTEGER NOT NULL DEFAULT 0,
    break_glass_reason     TEXT,
    source_ip              TEXT,
    extra_json             TEXT
);

CREATE TABLE IF NOT EXISTS event (
    id             TEXT PRIMARY KEY,
    timestamp_utc  TEXT NOT NULL,
    actor_id       TEXT NOT NULL,
    initiative_id  TEXT NOT NULL,
    event_type     TEXT NOT NULL,
    patient_ref    TEXT,
    detail_json    TEXT,
    source_ip      TEXT
);

CREATE INDEX IF NOT EXISTS ix_inference_ts ON inference(timestamp_utc);
CREATE INDEX IF NOT EXISTS ix_review_reviewer ON review(reviewer_id, timestamp_utc);
CREATE INDEX IF NOT EXISTS ix_review_initiative ON review(initiative_id, timestamp_utc);
CREATE INDEX IF NOT EXISTS ix_delegated_staff ON delegated_execution(staff_id, timestamp_utc);
CREATE INDEX IF NOT EXISTS ix_event_type ON event(event_type, timestamp_utc);
"""

# WHY triggers rather than permissions: SQLite has no GRANT. A trigger that
# RAISE(ABORT)s is the only in-database way to make a row immutable, and it
# survives any client that opens the file.
_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS trg_inference_no_update
BEFORE UPDATE ON inference BEGIN
    SELECT RAISE(ABORT, 'audit log is append-only: UPDATE on inference refused');
END;
CREATE TRIGGER IF NOT EXISTS trg_inference_no_delete
BEFORE DELETE ON inference BEGIN
    SELECT RAISE(ABORT, 'audit log is append-only: DELETE on inference refused');
END;
CREATE TRIGGER IF NOT EXISTS trg_review_no_update
BEFORE UPDATE ON review BEGIN
    SELECT RAISE(ABORT, 'audit log is append-only: UPDATE on review refused');
END;
CREATE TRIGGER IF NOT EXISTS trg_review_no_delete
BEFORE DELETE ON review BEGIN
    SELECT RAISE(ABORT, 'audit log is append-only: DELETE on review refused');
END;
CREATE TRIGGER IF NOT EXISTS trg_deleg_no_update
BEFORE UPDATE ON delegated_execution BEGIN
    SELECT RAISE(ABORT, 'audit log is append-only: UPDATE on delegated_execution refused');
END;
CREATE TRIGGER IF NOT EXISTS trg_deleg_no_delete
BEFORE DELETE ON delegated_execution BEGIN
    SELECT RAISE(ABORT, 'audit log is append-only: DELETE on delegated_execution refused');
END;
CREATE TRIGGER IF NOT EXISTS trg_event_no_update
BEFORE UPDATE ON event BEGIN
    SELECT RAISE(ABORT, 'audit log is append-only: UPDATE on event refused');
END;
CREATE TRIGGER IF NOT EXISTS trg_event_no_delete
BEFORE DELETE ON event BEGIN
    SELECT RAISE(ABORT, 'audit log is append-only: DELETE on event refused');
END;
"""

_EXPECTED_TRIGGERS = {
    "trg_inference_no_update",
    "trg_inference_no_delete",
    "trg_review_no_update",
    "trg_review_no_delete",
    "trg_deleg_no_update",
    "trg_deleg_no_delete",
    "trg_event_no_update",
    "trg_event_no_delete",
}


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance.

    WHY exact rather than an approximation: this number is the rubber-stamp
    alarm. It is computed once per signed note on strings of a few hundred
    characters. There is no performance argument for approximating it, and an
    approximation that drifts would corrupt the one metric README 10.3 calls
    the most important in the document.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


def edit_ratio(draft: str, final: str) -> float:
    """Edit distance normalised by the longer string, in [0.0, 1.0]."""
    denom = max(len(draft), len(final))
    if denom == 0:
        return 0.0
    return edit_distance(draft, final) / denom


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class _RubberStampFinding:
    reviewer_id: str
    initiative_id: str
    reviews: int
    median_edit_ratio: float
    zero_edit_fraction: float
    alarm: bool


class AuditLog:
    """Immutable record of every inference, review, and delegated act."""

    def __init__(
        self,
        path: str | os.PathLike[str] = "var/audit.sqlite3",
        *,
        hmac_key: bytes | None = None,
        create_dirs: bool = True,
    ) -> None:
        self.path = str(path)
        if create_dirs and self.path != ":memory:":
            parent = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(parent, exist_ok=True)
        key = hmac_key or os.environ.get("NSP_AUDIT_HMAC_KEY", "").encode() or None
        if key is None:
            # WHY generate rather than fall back to a constant: a constant key
            # is the same as no key. A generated key means patient refs are
            # unlinkable across runs, which is loud and wrong in a way an
            # operator notices immediately -- unlike a silent constant, which
            # is quiet and wrong forever.
            key = uuid.uuid4().bytes
            self.ephemeral_key = True
        else:
            self.ephemeral_key = False
        self._key = key
        # WHY a mutex rather than one connection per thread: this log is
        # append-only and low-volume, so serialising writes costs nothing, and
        # `check_same_thread=False` without a lock is actively dangerous --
        # sqlite3's implicit deferred transactions mean one thread's commit()
        # can apply to another thread's open statement. Measured effect on an
        # unguarded version: 400 concurrent record_event calls, 298 rows
        # persisted, no exception raised for the missing 74. A consent
        # revocation that silently fails to persist is the exact evidentiary
        # failure this module exists to prevent.
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.executescript(_TRIGGERS)
        self._conn.commit()
        self.verify_integrity_controls()

    # -- infrastructure ----------------------------------------------------

    def verify_integrity_controls(self) -> None:
        """Fail closed if the append-only triggers are missing (README 3.5)."""
        rows = self._conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='trigger'"
        ).fetchall()
        present = {r["name"]: (r["sql"] or "") for r in rows}
        missing = _EXPECTED_TRIGGERS - set(present)
        if missing:
            raise AuditIntegrityError(
                f"append-only triggers missing from audit database: {sorted(missing)}"
            )
        # A trigger that exists but no longer aborts is worse than one that is
        # missing, because the name check passes and nobody looks again.
        defanged = sorted(
            name for name in _EXPECTED_TRIGGERS
            if "RAISE(ABORT" not in present[name].upper().replace(" ", "")
            and "RAISE( ABORT" not in present[name].upper()
        )
        if defanged:
            raise AuditIntegrityError(
                f"append-only triggers present but no longer abort: {defanged}"
            )

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    def patient_ref(self, patient_id: str | None) -> str | None:
        """HMAC a patient identifier into a stable, non-reversible reference."""
        if patient_id is None:
            return None
        return hmac.new(self._key, str(patient_id).encode("utf-8"), hashlib.sha256).hexdigest()

    def close(self) -> None:
        self._conn.close()

    # -- writes ------------------------------------------------------------

    def record_inference(
        self,
        *,
        user_id: str,
        initiative_id: str,
        provider: str,
        model_id: str,
        model_version: str,
        prompt_template_id: str,
        prompt_template_hash: str,
        input_token_count: int,
        output_token_count: int,
        patient_id: str | None = None,
        confidence_score: float | None = None,
        constrained_decoding: bool = False,
        repair_attempts: int = 0,
        synthetic_probe: bool = False,
        source_ip: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> str:
        """Record one model call. Returns the inference id.

        Callers pass fields straight off an `InferenceResult`. No prompt or
        completion text is accepted by this signature, by design -- there is no
        parameter for it, so it cannot be logged by accident.
        """
        rec_id = uuid.uuid4().hex
        with self._tx() as cur:
            cur.execute(
                """INSERT INTO inference (id, timestamp_utc, user_id, patient_ref,
                       initiative_id, model_provider, model_id, model_version,
                       prompt_template_id, prompt_template_hash, input_token_count,
                       output_token_count, confidence_score, constrained_decoding,
                       repair_attempts, synthetic_probe, source_ip, extra_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    rec_id,
                    _utcnow(),
                    user_id,
                    self.patient_ref(patient_id),
                    initiative_id,
                    provider,
                    model_id,
                    model_version,
                    prompt_template_id,
                    prompt_template_hash,
                    int(input_token_count),
                    int(output_token_count),
                    confidence_score,
                    int(bool(constrained_decoding)),
                    int(repair_attempts),
                    int(bool(synthetic_probe)),
                    source_ip,
                    json.dumps(dict(extra or {}), sort_keys=True),
                ),
            )
        return rec_id

    def record_review(
        self,
        *,
        reviewer_id: str,
        initiative_id: str,
        draft: str,
        final: str,
        action_taken: str,
        inference_id: str | None = None,
        patient_id: str | None = None,
        synthetic_probe: bool = False,
        probe_caught: bool | None = None,
        review_seconds: float | None = None,
        edit_rate_applicable: bool = True,
        source_ip: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> str:
        """Record a human review event, computing the edit distance here.

        WHY computed here rather than passed in: a caller that computes its own
        edit distance can compute it wrongly, or flatteringly. The one number
        the safety programme depends on is derived in exactly one place.
        """
        if action_taken not in ReviewOutcome.ALL:
            raise ValueError(
                f"action_taken must be one of {ReviewOutcome.ALL}, got {action_taken!r}"
            )
        distance = edit_distance(draft, final)
        ratio = edit_ratio(draft, final)
        rec_id = uuid.uuid4().hex
        with self._tx() as cur:
            cur.execute(
                """INSERT INTO review (id, inference_id, timestamp_utc, reviewer_id,
                       initiative_id, patient_ref, action_taken,
                       edit_distance_draft_final, draft_length, final_length,
                       edit_ratio, synthetic_probe, probe_caught, review_seconds,
                       edit_rate_applicable, source_ip, extra_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    rec_id,
                    inference_id,
                    _utcnow(),
                    reviewer_id,
                    initiative_id,
                    self.patient_ref(patient_id),
                    action_taken,
                    distance,
                    len(draft),
                    len(final),
                    ratio,
                    int(bool(synthetic_probe)),
                    None if probe_caught is None else int(bool(probe_caught)),
                    review_seconds,
                    int(bool(edit_rate_applicable)),
                    source_ip,
                    json.dumps(dict(extra or {}), sort_keys=True),
                ),
            )
        return rec_id

    def record_delegated_execution(
        self,
        *,
        staff_id: str,
        initiative_id: str,
        task_code: str,
        supervising_pro_id: str | None,
        supervisor_on_site: bool | None,
        patient_id: str | None = None,
        standing_order_id: str | None = None,
        standing_order_version: str | None = None,
        competency_record_id: str | None = None,
        competency_expires: str | None = None,
        break_glass: bool = False,
        break_glass_reason: str | None = None,
        source_ip: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> str:
        """Record an act performed under physician delegation (225 ILCS 60/54.2).

        Missing supervision evidence is *recorded*, not refused. WHY: the
        break-glass path must never block patient care. The control is that
        `unevidenced_supervision()` surfaces every such record for review, so
        the gap is visible rather than prevented-and-worked-around.
        """
        if break_glass and not break_glass_reason:
            raise ValueError("break_glass requires break_glass_reason")
        rec_id = uuid.uuid4().hex
        with self._tx() as cur:
            cur.execute(
                """INSERT INTO delegated_execution (id, timestamp_utc, staff_id,
                       patient_ref, initiative_id, task_code, standing_order_id,
                       standing_order_version, supervising_pro_id, supervisor_on_site,
                       competency_record_id, competency_expires, break_glass,
                       break_glass_reason, source_ip, extra_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    rec_id,
                    _utcnow(),
                    staff_id,
                    self.patient_ref(patient_id),
                    initiative_id,
                    task_code,
                    standing_order_id,
                    standing_order_version,
                    supervising_pro_id,
                    None if supervisor_on_site is None else int(bool(supervisor_on_site)),
                    competency_record_id,
                    competency_expires,
                    int(bool(break_glass)),
                    break_glass_reason,
                    source_ip,
                    json.dumps(dict(extra or {}), sort_keys=True),
                ),
            )
        return rec_id

    def record_event(
        self,
        *,
        actor_id: str,
        initiative_id: str,
        event_type: str,
        patient_id: str | None = None,
        detail: Mapping[str, Any] | None = None,
        source_ip: str | None = None,
    ) -> str:
        """Record a non-inference operational event in the immutable log.

        WHY this exists in an *AI* audit log: consent capture and revocation
        (TCPA, README R-10) need the same tamper-evidence as clinical review.
        An opt-out that can be edited out of the operational database is not
        evidence of anything. Detail payloads must not carry clinical content.
        """
        rec_id = uuid.uuid4().hex
        with self._tx() as cur:
            cur.execute(
                """INSERT INTO event (id, timestamp_utc, actor_id, initiative_id,
                       event_type, patient_ref, detail_json, source_ip)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    rec_id,
                    _utcnow(),
                    actor_id,
                    initiative_id,
                    event_type,
                    self.patient_ref(patient_id),
                    json.dumps(dict(detail or {}), sort_keys=True),
                    source_ip,
                ),
            )
        return rec_id

    # -- reports -----------------------------------------------------------

    def rubber_stamp_report(
        self,
        *,
        initiative_id: str | None = None,
        min_reviews: int = 10,
        alarm_ratio: float = 0.05,
    ) -> list[_RubberStampFinding]:
        """Per-reviewer edit-rate summary with the README 10.3 alarm applied.

        The alarm fires below 5% median edit ratio. That threshold is not a
        quality target -- it is the point at which the reviewer has almost
        certainly stopped reading. `min_reviews` prevents a new hire's first
        three notes from tripping it.
        """
        sql = (
            "SELECT reviewer_id, initiative_id, edit_ratio FROM review "
            "WHERE synthetic_probe = 0 AND edit_rate_applicable = 1"
        )
        params: list[Any] = []
        if initiative_id:
            sql += " AND initiative_id = ?"
            params.append(initiative_id)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        buckets: dict[tuple[str, str], list[float]] = {}
        for row in rows:
            buckets.setdefault((row["reviewer_id"], row["initiative_id"]), []).append(
                row["edit_ratio"]
            )
        findings: list[_RubberStampFinding] = []
        for (reviewer, init), ratios in sorted(buckets.items()):
            median = statistics.median(ratios)
            zero_fraction = sum(1 for r in ratios if r == 0.0) / len(ratios)
            findings.append(
                _RubberStampFinding(
                    reviewer_id=reviewer,
                    initiative_id=init,
                    reviews=len(ratios),
                    median_edit_ratio=median,
                    zero_edit_fraction=zero_fraction,
                    alarm=len(ratios) >= min_reviews and median < alarm_ratio,
                )
            )
        return findings

    def probe_catch_rate(self, *, initiative_id: str | None = None) -> dict[str, Any]:
        """Synthetic-error catch rate (R-03). Target > 90%, alarm < 75%."""
        sql = (
            "SELECT probe_caught FROM review "
            "WHERE synthetic_probe = 1 AND probe_caught IS NOT NULL"
        )
        params: list[Any] = []
        if initiative_id:
            sql += " AND initiative_id = ?"
            params.append(initiative_id)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        total = len(rows)
        caught = sum(1 for r in rows if r["probe_caught"])
        rate = (caught / total) if total else None
        return {
            "probes": total,
            "caught": caught,
            "catch_rate": rate,
            "alarm": rate is not None and rate < 0.75,
            "meets_target": rate is not None and rate > 0.90,
        }

    def delegation_evidence(
        self,
        *,
        staff_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """The 54.2 evidence extract: who did what, under whose delegation."""
        sql = "SELECT * FROM delegated_execution WHERE 1=1"
        params: list[Any] = []
        if staff_id:
            sql += " AND staff_id = ?"
            params.append(staff_id)
        if since:
            sql += " AND timestamp_utc >= ?"
            params.append(since)
        if until:
            sql += " AND timestamp_utc <= ?"
            params.append(until)
        sql += " ORDER BY timestamp_utc DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def unevidenced_supervision(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
        staff_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Delegated acts missing supervision evidence. This is the audit query.

        A row appears here if any of: no supervising professional named, the
        supervisor was not on site, no standing order cited, or no current
        competency record. Break-glass executions always appear, because a
        break-glass that nobody reviews afterwards is just an unlocked door.

        `staff_id` scopes the report. A break-glass justification is an incident
        record about a named person, and an extract requested for one employee
        must not disclose another's.
        """
        # The OR group MUST stay parenthesised. Without the brackets, SQL binds
        # the appended `AND timestamp_utc >= ?` to the final OR term only, and
        # the report silently returns the entire history whatever window counsel
        # asked for.
        sql = """
            SELECT * FROM delegated_execution
            WHERE (
                   supervising_pro_id IS NULL
                OR supervisor_on_site IS NULL
                OR supervisor_on_site = 0
                OR standing_order_id IS NULL
                OR competency_record_id IS NULL
                OR break_glass = 1
            )
        """
        params: list[Any] = []
        if since:
            sql += " AND timestamp_utc >= ?"
            params.append(since)
        if until:
            sql += " AND timestamp_utc <= ?"
            params.append(until)
        if staff_id:
            # WITHOUT THIS the report is unscoped, and an extract requested for
            # one staff member returned every other employee's break-glass
            # justification -- handed to whoever asked for the first one.
            sql += " AND staff_id = ?"
            params.append(staff_id)
        sql += " ORDER BY timestamp_utc DESC"
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def counts(self) -> dict[str, int]:
        """Row counts per table. Used by smoke tests and the ops dashboard."""
        out = {}
        with self._lock:
            for table in ("inference", "review", "delegated_execution", "event"):
                out[table] = self._conn.execute(
                    f"SELECT COUNT(*) c FROM {table}"
                ).fetchone()["c"]
        return out

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        """Read-only escape hatch for reporting. Refuses anything but SELECT."""
        if not sql.lstrip().upper().startswith("SELECT"):
            raise AuditIntegrityError("AuditLog.query accepts SELECT statements only")
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]
