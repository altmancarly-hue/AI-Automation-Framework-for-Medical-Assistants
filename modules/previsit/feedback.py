"""One click per brief: useful, not useful, or wrong. And what happens to it.

README I-03 asks for exactly this -- "One-click 'this brief was useful / not
useful / wrong' per patient, logged for prompt iteration" -- and lists "brief
becomes noise and is ignored" as a risk whose control is "the one-click feedback
loop drives iteration; measure open rate."

A feedback button that writes to a table nobody reads is theatre, so this module
is mostly the reading half:

  * `report()` splits the rate by whether the brief HAD an AI section. If briefs
    with narrative score worse than briefs without, the narrative is subtracting
    value, and that is a decision to stop generating it -- not a prompt to tune.
  * WRONG is tracked separately from NOT_USEFUL and never averaged into a
    satisfaction score. They are different failures: not-useful is noise, wrong
    is a clinician who was pointed at something untrue. README I-03's top risk
    is a hallucinated detail, and a metric that lets three "useful"s cancel a
    "wrong" cannot see it.
  * `wrong_items()` returns the free-text detail attached to WRONG verdicts,
    which is the corpus a prompt revision is actually written against.

Feedback is keyed to the brief AND to the prompt hash that produced it, because
feedback gathered before a prompt change does not describe the prompt after it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

from modules.scheduling.models import Database, iso, new_id

__all__ = ["FEEDBACK_SCHEMA", "Verdict", "BriefFeedback", "FeedbackLog"]


class Verdict:
    USEFUL = "useful"
    NOT_USEFUL = "not_useful"
    WRONG = "wrong"

    ALL = (USEFUL, NOT_USEFUL, WRONG)


FEEDBACK_SCHEMA = """
CREATE TABLE IF NOT EXISTS previsit_brief (
    brief_id        TEXT PRIMARY KEY,
    patient_id      TEXT NOT NULL,
    clinic_date     TEXT NOT NULL,
    generated_utc   TEXT NOT NULL,
    had_narrative   INTEGER NOT NULL DEFAULT 0,
    narrative_items INTEGER NOT NULL DEFAULT 0,
    narrative_dropped INTEGER NOT NULL DEFAULT 0,
    -- Whether the AI section actually SURVIVED onto the page. A brief whose
    -- narrative was trimmed by the one-screen cap was still recorded as
    -- had_narrative, so a rating was attributed to text the clinician never saw.
    narrative_shown INTEGER NOT NULL DEFAULT 0,
    prompt_hash     TEXT NOT NULL DEFAULT '',
    model_id        TEXT NOT NULL DEFAULT '',
    opened_utc      TEXT
);

CREATE TABLE IF NOT EXISTS previsit_feedback (
    feedback_id  TEXT PRIMARY KEY,
    brief_id     TEXT NOT NULL,
    verdict      TEXT NOT NULL,
    given_by     TEXT NOT NULL,
    given_utc    TEXT NOT NULL,
    detail       TEXT NOT NULL DEFAULT '',
    prompt_hash  TEXT NOT NULL DEFAULT ''
);

-- One verdict per person per brief. A second click is a correction, applied by
-- replacing the row, not a second vote.
CREATE UNIQUE INDEX IF NOT EXISTS ux_previsit_feedback_once
    ON previsit_feedback(brief_id, given_by);
CREATE INDEX IF NOT EXISTS ix_previsit_brief_date
    ON previsit_brief(clinic_date, patient_id);
"""


@dataclass(frozen=True)
class BriefFeedback:
    brief_id: str
    verdict: str
    given_by: str
    given_utc: datetime
    detail: str = ""


class FeedbackLog:
    """Registers briefs, records verdicts, and answers whether this is working."""

    INITIATIVE = "I-03"

    def __init__(self, db: Database, *, audit: Any = None) -> None:
        self.db = db
        self.audit = audit
        self.db.conn.executescript(FEEDBACK_SCHEMA)

    def register(
        self,
        *,
        brief_id: str,
        patient_id: str,
        clinic_date: str,
        generated_utc: datetime,
        had_narrative: bool,
        narrative_shown: bool | None = None,
        narrative_items: int = 0,
        narrative_dropped: int = 0,
        prompt_hash: str = "",
        model_id: str = "",
    ) -> None:
        """Record that a brief was produced. Registered whether or not it is read.

        The denominator matters: an open rate computed only over briefs somebody
        clicked on is 100% by construction.
        """
        self.db.execute(
            """INSERT OR REPLACE INTO previsit_brief (brief_id, patient_id,
                   clinic_date, generated_utc, had_narrative, narrative_shown,
                   narrative_items, narrative_dropped, prompt_hash, model_id)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                brief_id, patient_id, clinic_date, iso(generated_utc),
                1 if had_narrative else 0,
                1 if (had_narrative if narrative_shown is None else narrative_shown) else 0,
                narrative_items, narrative_dropped, prompt_hash, model_id,
            ),
        )

    def mark_opened(self, brief_id: str, *, now: datetime) -> None:
        self.db.execute(
            "UPDATE previsit_brief SET opened_utc = ? WHERE brief_id = ?"
            " AND opened_utc IS NULL",
            (iso(now), brief_id),
        )

    def record(
        self, *, brief_id: str, verdict: str, given_by: str, now: datetime, detail: str = ""
    ) -> BriefFeedback:
        if verdict not in Verdict.ALL:
            raise ValueError(f"verdict must be one of {Verdict.ALL}, not {verdict!r}")
        if not given_by.strip():
            raise ValueError("feedback must record who gave it")
        row = self.db.one(
            "SELECT prompt_hash FROM previsit_brief WHERE brief_id = ?", (brief_id,)
        )
        if row is None:
            raise KeyError(f"no registered brief {brief_id!r}")
        if verdict == Verdict.WRONG and not detail.strip():
            # The whole point of WRONG is the corpus it builds. A verdict with
            # no detail cannot be acted on and would dilute the count of ones
            # that can.
            raise ValueError(
                "a 'wrong' verdict has to say what was wrong -- that free text is "
                "the entire input to the next prompt revision"
            )
        self.db.execute(
            """INSERT OR REPLACE INTO previsit_feedback (feedback_id, brief_id,
                   verdict, given_by, given_utc, detail, prompt_hash)
               VALUES (?,?,?,?,?,?,?)""",
            (
                new_id("fbk"), brief_id, verdict, given_by, iso(now), detail.strip(),
                str(row["prompt_hash"]),
            ),
        )
        if self.audit is not None:
            self.audit.record_event(
                actor_id=given_by,
                initiative_id=self.INITIATIVE,
                event_type="previsit_brief_feedback",
                detail={"brief_id": brief_id, "verdict": verdict},
            )
        return BriefFeedback(brief_id, verdict, given_by, now, detail.strip())

    # -- reporting ---------------------------------------------------------

    def report(self, *, prompt_hash: str | None = None) -> dict[str, Any]:
        """Is the brief earning its screen, and is the narrative helping or not?"""
        # Filter on the prompt that produced the AI half. Briefs with NO model
        # call have an empty hash and must stay in the comparison -- they are
        # the control arm. Excluding them left "without_narrative" composed only
        # of briefs where synthesis ran and every item was dropped, which are
        # systematically the hardest charts, and biased the comparison in favour
        # of the narrative.
        where, params = ("", ())
        if prompt_hash:
            where, params = (
                " WHERE (b.prompt_hash = ? OR b.prompt_hash = '')",
                (prompt_hash,),
            )
        rows = self.db.all(
            "SELECT b.brief_id, b.narrative_shown AS had_narrative, b.opened_utc,"
            " f.verdict"
            " FROM previsit_brief b LEFT JOIN previsit_feedback f"
            " ON f.brief_id = b.brief_id" + where,
            params,
        )
        if not rows:
            return {"available": False, "reason": "no briefs registered yet"}

        def bucket(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            """Rates are PER BRIEF, not per click.

            The LEFT JOIN fans out to one row per (brief, verdict), so counting
            verdicts let three MAs rating one brief USEFUL outvote one clinician
            rating another brief WRONG: 75% useful, 25% wrong, when the truth
            per brief was 50/50. This module's own docstring says a metric that
            lets three "useful"s cancel a "wrong" cannot see the thing it exists
            to see -- so a brief with ANY wrong verdict counts as wrong, and
            counts once.
            """
            by_brief: dict[str, set[str]] = {}
            for row in subset:
                by_brief.setdefault(str(row["brief_id"]), set())
                if row["verdict"]:
                    by_brief[str(row["brief_id"])].add(str(row["verdict"]))
            rated = {b: v for b, v in by_brief.items() if v}
            wrong = sum(1 for v in rated.values() if Verdict.WRONG in v)
            # A brief is "useful" only if somebody said so and nobody said wrong.
            useful = sum(
                1 for v in rated.values()
                if Verdict.USEFUL in v and Verdict.WRONG not in v
            )
            not_useful = sum(
                1 for v in rated.values()
                if Verdict.WRONG not in v and Verdict.USEFUL not in v
            )
            return {
                "briefs": len(by_brief),
                "rated": len(rated),
                "useful": useful,
                "not_useful": not_useful,
                # Never folded into a satisfaction score. See the module docstring.
                "wrong": wrong,
                "verdicts_recorded": sum(1 for r in subset if r["verdict"]),
                "useful_rate": round(useful / len(rated), 3) if rated else None,
                "wrong_rate": round(wrong / len(rated), 3) if rated else None,
            }

        opened = len({r["brief_id"] for r in rows if r["opened_utc"]})
        total = len({r["brief_id"] for r in rows})
        with_ai = [r for r in rows if r["had_narrative"]]
        without_ai = [r for r in rows if not r["had_narrative"]]
        result = {
            "available": True,
            "briefs": total,
            "open_rate": round(opened / total, 3),
            "overall": bucket(rows),
            "with_narrative": bucket(with_ai) if with_ai else None,
            "without_narrative": bucket(without_ai) if without_ai else None,
        }
        both = result["with_narrative"], result["without_narrative"]
        if all(b and b["useful_rate"] is not None for b in both):
            delta = both[0]["useful_rate"] - both[1]["useful_rate"]
            result["narrative_delta"] = round(delta, 3)
            if delta < 0:
                result["narrative_verdict"] = (
                    "briefs WITH the AI narrative are rated less useful than briefs "
                    "without it; the narrative is subtracting value and the "
                    "decision is whether to keep generating it, not how to tune it"
                )
        return result

    def wrong_items(self, *, prompt_hash: str | None = None) -> list[dict[str, Any]]:
        """The free text behind every WRONG verdict. The prompt-revision corpus."""
        where, params = ("WHERE f.verdict = ?", (Verdict.WRONG,))
        if prompt_hash:
            where += " AND f.prompt_hash = ?"
            params = params + (prompt_hash,)
        return self.db.all(
            "SELECT f.brief_id, f.given_by, f.given_utc, f.detail, f.prompt_hash,"
            " b.patient_id, b.model_id FROM previsit_feedback f"
            " JOIN previsit_brief b ON b.brief_id = f.brief_id "
            + where + " ORDER BY f.given_utc DESC",
            params,
        )
