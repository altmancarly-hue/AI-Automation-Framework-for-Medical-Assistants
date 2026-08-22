"""Audio retention. Deleted on signature, and never past twenty-four hours.

README I-04: "Audio is deleted after note finalization. Retention of triage call
audio creates a discovery surface with no clinical benefit." The risk table adds
the ceiling: "24-hour lifecycle deletion; deleted immediately on note signature;
documented retention policy."

Both halves matter and they fail differently:

  * **Deletion on signature** is the normal path. The note is the record; once
    it is signed the audio has served its entire purpose.
  * **The 24-hour ceiling** is the backstop for every call that never reaches a
    signature -- the MA went home, the shift ran long, the browser crashed. Those
    are the recordings that quietly accumulate, and they are exactly the ones
    nobody remembers exist until they are listed in discovery.

The sweep is therefore unconditional. It does not ask whether the note was
signed, whether anyone is still working on it, or whether the encounter looks
important. An unsigned call at hour 25 loses its audio and keeps its transcript
gap, which is a worse note and a much better legal position.

The transcript goes too. It is a verbatim PHI-bearing copy of the conversation
with weaker controls than the chart, which is the same argument the README makes
about audio. Its SHA-256 survives, so the audit record can still prove which
transcript produced which note without keeping the transcript itself.

ON "SECURE" DELETION, honestly: `purge` overwrites before unlinking, which is
meaningful on a spinning disk and largely theatre on a copy-on-write filesystem
or an SSD with wear levelling. It is done because it costs nothing and
occasionally helps. The control that actually matters is full-disk encryption on
the workstation, which is a deployment requirement, not something this module
can enforce.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping, Sequence

from modules.scheduling.models import Database, iso, new_id, parse_iso

from .capture import AudioRecording

__all__ = [
    "LIFECYCLE_SCHEMA",
    "HARD_RETENTION_CEILING",
    "RetentionBreach",
    "AudioLifecycle",
]

#: The absolute maximum age of any call audio, signed or not. README I-04.
HARD_RETENTION_CEILING = timedelta(hours=24)


class RetentionBreach(RuntimeError):
    """Raised by `assert_clean` when audio exists past the ceiling."""


LIFECYCLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS triage_audio (
    audio_id       TEXT PRIMARY KEY,
    encounter_id   TEXT NOT NULL,
    path           TEXT NOT NULL,
    sha256         TEXT NOT NULL,
    bytes          INTEGER NOT NULL,
    created_utc    TEXT NOT NULL,
    expires_utc    TEXT NOT NULL,
    deleted_utc    TEXT,
    delete_reason  TEXT,
    -- Set when the unlink itself failed. The row stays LIVE so the sweep keeps
    -- trying and `orphans()` keeps reporting it; stamping deleted_utc on a file
    -- that is still on disk hides it from every report this module has.
    delete_failed_utc TEXT,
    delete_failures   INTEGER NOT NULL DEFAULT 0
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_triage_audio_encounter
    ON triage_audio(encounter_id) WHERE deleted_utc IS NULL;
CREATE INDEX IF NOT EXISTS ix_triage_audio_expiry ON triage_audio(expires_utc, deleted_utc);
"""


@dataclass(frozen=True)
class DeletionResult:
    encounter_id: str
    path: str
    reason: str
    file_removed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "encounter_id": self.encounter_id,
            "reason": self.reason,
            "file_removed": self.file_removed,
        }


class AudioLifecycle:
    """Registers audio, deletes it on signature, and sweeps the stragglers."""

    INITIATIVE = "I-04"

    def __init__(
        self,
        db: Database,
        *,
        audit: Any = None,
        ceiling: timedelta = HARD_RETENTION_CEILING,
        transcripts_elsewhere: bool = False,
    ) -> None:
        #: Set only by a deployment that keeps transcripts in a different
        #: database and enforces their retention there. It suppresses the
        #: "transcript retention not verifiable" finding and nothing else --
        #: it does not make this object stop looking.
        self.transcripts_elsewhere = transcripts_elsewhere
        self.db = db
        self.audit = audit
        self.ceiling = ceiling
        self.db.conn.executescript(LIFECYCLE_SCHEMA)

    def _encounter_table_exists(self) -> bool:
        """Whether the encounter table lives in THIS database handle.

        The lifecycle owns `triage_audio`; `triage_encounter` belongs to
        `TriageService`. Point the two at different databases -- an easy wiring
        mistake, since both take a `Database` -- and the transcript half of the
        retention rule silently does nothing.

        This predicate exists so that condition can be REPORTED, not so it can
        be skipped quietly. An earlier version made every transcript call a
        no-op and `assert_clean` still certified the box as clean, which is
        strictly worse than the OperationalError it replaced: before, the first
        run failed loudly and somebody fixed the wiring. See `assert_clean` and
        `purge_transcript`.
        """
        row = self.db.one(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            ("triage_encounter",),
        )
        return row is not None

    # -- registration ------------------------------------------------------

    def register(self, recording: AudioRecording, *, now: datetime) -> str:
        """Record that audio exists. Called the moment it hits disk.

        WHY a database row for a file: a file nobody has a record of is a file
        nobody sweeps. The row is what makes the ceiling enforceable and what
        makes `orphans()` able to answer "is there any call audio on this box
        older than a day", which is the question an auditor asks.
        """
        audio_id = new_id("aud")
        size = os.path.getsize(recording.path) if os.path.exists(recording.path) else 0
        self.db.execute(
            """INSERT INTO triage_audio (audio_id, encounter_id, path, sha256, bytes,
                   created_utc, expires_utc)
               VALUES (?,?,?,?,?,?,?)""",
            (
                audio_id,
                recording.encounter_id,
                recording.path,
                recording.sha256,
                size,
                iso(now),
                iso(now + self.ceiling),
            ),
        )
        if self.audit is not None:
            self.audit.record_event(
                actor_id="system:triage",
                initiative_id=self.INITIATIVE,
                event_type="triage_audio_registered",
                detail={
                    "encounter_id": recording.encounter_id,
                    "bytes": size,
                    "expires_utc": iso(now + self.ceiling),
                    "consent_id": recording.authorisation.consent_id,
                },
            )
        return audio_id

    # -- deletion ----------------------------------------------------------

    def _purge(self, path: str) -> bool:
        """Overwrite then unlink. Returns True if the file is gone afterwards."""
        if not os.path.exists(path):
            return True
        try:
            size = os.path.getsize(path)
            with open(path, "r+b") as handle:
                handle.write(b"\x00" * size)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:  # pragma: no cover - best effort, see module docstring
            pass
        try:
            os.remove(path)
        except FileNotFoundError:  # pragma: no cover - raced with another sweep
            pass
        except OSError:
            return False
        return not os.path.exists(path)

    def delete_for_encounter(
        self, encounter_id: str, *, now: datetime, reason: str
    ) -> DeletionResult | None:
        row = self.db.one(
            "SELECT * FROM triage_audio WHERE encounter_id = ? AND deleted_utc IS NULL",
            (encounter_id,),
        )
        if row is None:
            return None
        removed = self._purge(str(row["path"]))
        if removed:
            self.db.execute(
                "UPDATE triage_audio SET deleted_utc = ?, delete_reason = ?"
                " WHERE audio_id = ?",
                (iso(now), reason, row["audio_id"]),
            )
        else:
            # A read-only mount, an antivirus lock, an SMB share. Recording it
            # as deleted would remove it from sweep(), orphans(), live_audio()
            # and assert_clean() -- the file would stay on disk, permanently
            # invisible to every control in this module.
            self.db.execute(
                "UPDATE triage_audio SET delete_failed_utc = ?,"
                " delete_failures = delete_failures + 1 WHERE audio_id = ?",
                (iso(now), row["audio_id"]),
            )
        if self.audit is not None:
            self.audit.record_event(
                actor_id="system:triage",
                initiative_id=self.INITIATIVE,
                event_type=(
                    "triage_audio_deleted" if removed else "triage_audio_delete_failed"
                ),
                detail={
                    "encounter_id": encounter_id,
                    "reason": reason,
                    "file_removed": removed,
                },
            )
        return DeletionResult(encounter_id, str(row["path"]), reason, removed)

    def on_signature(self, encounter_id: str, *, now: datetime) -> DeletionResult | None:
        """The normal path: the note is signed, the audio is finished."""
        return self.delete_for_encounter(
            encounter_id, now=now, reason="note_signed"
        )

    def sweep_transcripts(self, *, now: datetime) -> int:
        """Purge stored transcripts past the ceiling, signed or not.

        The audio sweep alone left the verbatim conversation in the database
        forever for any call that was abandoned, rejected, or documented by
        hand. The transcript is a PHI-bearing copy of the call with weaker
        controls than the chart -- the same argument that deletes the audio.
        """
        if not self._encounter_table_exists():
            return 0
        cutoff = iso(now - self.ceiling)
        cur = self.db.execute(
            "UPDATE triage_encounter SET transcript_text = NULL,"
            " transcript_purged_utc = ? WHERE transcript_text IS NOT NULL"
            " AND opened_utc <= ?",
            (iso(now), cutoff),
        )
        return cur.rowcount

    def sweep(self, *, now: datetime) -> list[DeletionResult]:
        """Delete every recording past the ceiling. Signed or not, done or not.

        Deliberately takes no arguments beyond the clock. A sweep with an
        exemption parameter is a sweep with an exemption, and the recordings
        that would be exempted are precisely the abandoned ones the ceiling
        exists for.
        """
        rows = self.db.all(
            "SELECT * FROM triage_audio WHERE deleted_utc IS NULL AND expires_utc <= ?",
            (iso(now),),
        )
        results: list[DeletionResult] = []
        for row in rows:
            result = self.delete_for_encounter(
                str(row["encounter_id"]), now=now, reason="retention_ceiling"
            )
            if result is not None:
                results.append(result)
        self.sweep_transcripts(now=now)
        return results

    # -- transcripts -------------------------------------------------------

    def purge_transcript(self, encounter_id: str, *, now: datetime) -> None:
        """Drop the transcript text, keep its hash.

        The transcript is a verbatim PHI-bearing copy of the call with weaker
        controls than the chart -- the same argument README I-04 makes about the
        audio. The hash stays so the audit record can still prove which
        transcript produced which note.
        """
        if not self._encounter_table_exists():
            # The caller is asking for a specific transcript to be destroyed. A
            # silent return here is a promise this object cannot keep, and the
            # caller has no way to find out.
            raise RetentionBreach(
                f"asked to purge the transcript for {encounter_id!r}, but this "
                "AudioLifecycle's database has no triage_encounter table. The "
                "lifecycle and the TriageService are pointed at different "
                "databases and the transcript has NOT been destroyed."
            )
        self.db.execute(
            "UPDATE triage_encounter SET transcript_text = NULL,"
            " transcript_purged_utc = ? WHERE encounter_id = ?",
            (iso(now), encounter_id),
        )

    # -- reporting ---------------------------------------------------------

    def live_audio(self) -> list[dict[str, Any]]:
        return self.db.all(
            "SELECT encounter_id, path, created_utc, expires_utc, bytes"
            " FROM triage_audio WHERE deleted_utc IS NULL ORDER BY created_utc"
        )

    def orphans(self, *, now: datetime) -> list[dict[str, Any]]:
        """Audio past its expiry that the sweep has not yet removed.

        Should always be empty in a system whose cron is running. A non-empty
        answer means the sweep is not running, which is the failure that has no
        symptom until the day it has a very large symptom.
        """
        return self.db.all(
            "SELECT encounter_id, path, created_utc, expires_utc FROM triage_audio"
            " WHERE deleted_utc IS NULL AND expires_utc <= ?",
            (iso(now),),
        )

    def stale_files_on_disk(self, directory: str | os.PathLike[str]) -> list[str]:
        """Audio files with no live row. Belt and braces against a lost write.

        If `register` ever fails after the file lands, the sweep would never see
        it. This walks the directory instead of the table, so the two disagree
        loudly rather than silently.
        """
        known = {
            os.path.realpath(str(r["path"]))
            for r in self.db.all("SELECT path FROM triage_audio WHERE deleted_utc IS NULL")
        }
        found: list[str] = []
        # realpath on both sides, and os.walk rather than listdir: a relative
        # recorder directory compared against an absolute check path flagged
        # every file as stale, and a subdirectory hid files from the check
        # entirely.
        for root, _dirs, names in os.walk(str(directory)):
            for name in sorted(names):
                path = os.path.realpath(os.path.join(root, name))
                if os.path.isfile(path) and path not in known:
                    found.append(path)
        return sorted(found)

    def failed_deletions(self) -> list[dict[str, Any]]:
        """Rows whose unlink failed. Non-empty means a file is stuck on disk."""
        return self.db.all(
            "SELECT encounter_id, path, delete_failures, delete_failed_utc"
            " FROM triage_audio WHERE deleted_utc IS NULL AND delete_failed_utc"
            " IS NOT NULL ORDER BY delete_failed_utc"
        )

    def assert_clean(self, *, now: datetime, directory: str | os.PathLike[str] | None = None) -> None:
        """Raise if anything is retained past the ceiling. For the daily check."""
        problems: list[str] = []
        failures = self.failed_deletions()
        if failures:
            problems.append(
                f"{len(failures)} recording(s) could not be deleted and are still "
                "on disk"
            )
        if not self._encounter_table_exists():
            stale_transcripts = None
            if not self.transcripts_elsewhere:
                # Reported, never assumed clean. The module docstring promises
                # "the transcript goes too", and a check that cannot see the
                # transcripts has not verified that promise.
                problems.append(
                    "transcript retention is NOT VERIFIABLE from this database: "
                    "no triage_encounter table. Either the lifecycle and the "
                    "TriageService are pointed at different databases, or this "
                    "check is running before any call exists. Pass "
                    "transcripts_elsewhere=True only if another system enforces "
                    "transcript retention."
                )
        else:
            stale_transcripts = self.db.one(
                "SELECT COUNT(*) c FROM triage_encounter WHERE transcript_text"
                " IS NOT NULL AND opened_utc <= ?",
                (iso(now - self.ceiling),),
            )
        if stale_transcripts and stale_transcripts["c"]:
            problems.append(
                f"{stale_transcripts['c']} transcript(s) retained past the "
                f"{self.ceiling} ceiling"
            )
        overdue = self.orphans(now=now)
        if overdue:
            problems.append(
                f"{len(overdue)} recording(s) past the {self.ceiling} retention "
                "ceiling have not been deleted"
            )
        if directory is not None:
            stale = self.stale_files_on_disk(directory)
            if stale:
                problems.append(
                    f"{len(stale)} audio file(s) on disk with no lifecycle record"
                )
        if problems:
            raise RetentionBreach("; ".join(problems))
