"""Operational KPIs for I-07 (README 10.2).

WHY these four numbers and not a dashboard of forty:

README 10.1 makes the point that the entire I-07 business case rests on one
number the practice already has — the no-show rate — and that it must be
*measured*, by provider, visit type, day of week and lead time, before anything
is funded. This module computes that number from the same tables the automation
writes to, so the "after" figure is produced by the same code path as the
"before" figure. A benefit measured with a different instrument than the
baseline is not a measurement, it is a comparison of two instruments.

The four KPIs mirror the README's operational table exactly:

    no-show rate                 target -25%
    cancelled slots backfilled   target > 60%
    median slot backfill time    target < 15 min
    (plus the message funnel, which is how you find out that "reminders are on"
     actually means "reminders are being blocked by a missing consent row")

The last one is not in the README's table and is here anyway. A reminder system
whose sends are silently blocked reports a flat no-show rate and no error, and
the practice concludes the intervention does not work.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping, Sequence

from .models import Database, PRACTICE_TZ, iso, parse_iso, to_local

__all__ = [
    "KPI_TARGETS",
    "no_show_rate",
    "backfill_rate",
    "fill_time_stats",
    "message_funnel",
    "lead_time_buckets",
    "kpi_summary",
]

#: README 10.2, verbatim. Baselines marked "Measure" are supplied by the caller
#: once the practice has pulled the EHR report.
KPI_TARGETS: Mapping[str, Mapping[str, Any]] = {
    "no_show_rate": {"target": "-25% vs baseline", "frequency": "weekly"},
    "backfill_rate": {"target": 0.60, "comparator": ">", "frequency": "weekly"},
    "median_fill_minutes": {"target": 15.0, "comparator": "<", "frequency": "weekly"},
}


_DOW_ORDER = {d: i for i, d in enumerate(("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"))}


def _check_group_by(group_by: Sequence[str], valid: set[str]) -> None:
    unknown = set(group_by) - valid
    if unknown:
        raise ValueError(f"cannot group by {sorted(unknown)}; valid keys: {sorted(valid)}")


def _label(key: Sequence[Any]) -> str:
    """Join group-key parts with a separator that cannot occur in an id.

    WHY not "|": a provider or visit type containing a pipe would collide two
    distinct groups into one row, and the report would silently under-count.
    """
    return "\u241f".join(str(k) for k in key)


def _group_sort_key(key: Sequence[Any], group_by: Sequence[str]) -> tuple[Any, ...]:
    """Sort day-of-week groups by weekday, everything else alphabetically."""
    parts: list[Any] = []
    for field, value in zip(group_by, key):
        if field == "dow":
            parts.append((0, _DOW_ORDER.get(str(value), 99)))
        else:
            parts.append((1, str(value)))
    return tuple(parts)


def _window(since: datetime | None, until: datetime | None) -> tuple[str, str]:
    lo = iso(since) if since else "0000"
    hi = iso(until) if until else "9999"
    return lo, hi


@dataclass(frozen=True)
class Rate:
    numerator: int
    denominator: int

    @property
    def value(self) -> float | None:
        return (self.numerator / self.denominator) if self.denominator else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "rate": self.value,
        }


def no_show_rate(
    db: Database,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    group_by: Sequence[str] = (),
) -> dict[str, Any]:
    """No-shows over resolved appointments.

    Denominator is completed + no_show — appointments that reached their time
    and resolved one way or the other. Cancellations are deliberately excluded:
    a cancelled slot is a different failure with a different remedy (backfill),
    and folding it into the no-show rate makes a successful cancellation
    campaign look like a worsening no-show problem.

    `group_by` accepts any of: provider_id, visit_type, dow (practice-local day
    of week). The README asks for provider, visit type, day of week and lead
    time; lead time has its own function because it needs bucketing.
    """
    lo, hi = _window(since, until)
    _check_group_by(group_by, {"provider_id", "visit_type", "dow"})

    rows = db.all(
        """SELECT provider_id, visit_type, status, start_utc FROM appointment
           WHERE status IN ('completed','no_show') AND start_utc >= ? AND start_utc <= ?""",
        (lo, hi),
    )
    overall = Rate(
        sum(1 for r in rows if r["status"] == "no_show"),
        len(rows),
    )
    result: dict[str, Any] = {"overall": overall.as_dict()}
    if not group_by:
        return result

    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(
            to_local(parse_iso(row["start_utc"])).strftime("%a")
            if field == "dow"
            else row[field]
            for field in group_by
        )
        buckets.setdefault(key, []).append(row)
    grouped = {}
    for key, group in sorted(
        buckets.items(), key=lambda kv: _group_sort_key(kv[0], group_by)
    ):
        grouped[_label(key)] = Rate(
            sum(1 for r in group if r["status"] == "no_show"), len(group)
        ).as_dict()
    result["groups"] = grouped
    result["group_by"] = list(group_by)
    return result


def backfill_rate(
    db: Database,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    group_by: Sequence[str] = (),
) -> dict[str, Any]:
    """Filled releases over resolved releases (filled or closed).

    A release that is still open is excluded from both sides. Counting an
    in-flight blast as a failure understates the rate every time the report is
    run mid-morning, which is when it is run.
    """
    _check_group_by(group_by, {"provider_id", "visit_type"})
    lo, hi = _window(since, until)
    rows = db.all(
        """SELECT provider_id, visit_type, filled_utc, closed_utc FROM slot_release
           WHERE released_utc >= ? AND released_utc <= ?
             AND (filled_utc IS NOT NULL OR closed_utc IS NOT NULL)""",
        (lo, hi),
    )
    overall = Rate(sum(1 for r in rows if r["filled_utc"]), len(rows))
    result: dict[str, Any] = {"overall": overall.as_dict()}
    if not group_by:
        return result
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(tuple(row[f] for f in group_by), []).append(row)
    result["groups"] = {
        _label(key): Rate(sum(1 for r in group if r["filled_utc"]), len(group)).as_dict()
        for key, group in sorted(
            buckets.items(), key=lambda kv: _group_sort_key(kv[0], group_by)
        )
    }
    result["group_by"] = list(group_by)
    return result


def fill_time_stats(
    db: Database,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    group_by: Sequence[str] = (),
) -> dict[str, Any]:
    """Release-to-fill latency in minutes. Median is the KPI; p90 is the tell.

    WHY p90 as well: a median under fifteen minutes with a p90 of six hours
    means most slots fill from the first blast and a meaningful tail is being
    rescued by a human the next morning. That tail is the part worth working on,
    and the median hides it entirely.
    """
    _check_group_by(group_by, {"provider_id", "visit_type"})
    lo, hi = _window(since, until)
    rows = db.all(
        """SELECT provider_id, visit_type, released_utc, filled_utc FROM slot_release
           WHERE filled_utc IS NOT NULL AND released_utc >= ? AND released_utc <= ?""",
        (lo, hi),
    )

    def minutes(group: Iterable[Mapping[str, Any]]) -> list[float]:
        return [
            (parse_iso(r["filled_utc"]) - parse_iso(r["released_utc"])).total_seconds() / 60.0
            for r in group
        ]

    def summarise(values: Sequence[float]) -> dict[str, Any]:
        if not values:
            return {"n": 0, "median_minutes": None, "p90_minutes": None, "mean_minutes": None}
        ordered = sorted(values)
        p90_index = min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))
        return {
            "n": len(ordered),
            "median_minutes": round(statistics.median(ordered), 2),
            "p90_minutes": round(ordered[p90_index], 2),
            "mean_minutes": round(statistics.fmean(ordered), 2),
        }

    result: dict[str, Any] = {"overall": summarise(minutes(rows))}
    if not group_by:
        return result
    buckets: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(tuple(row[f] for f in group_by), []).append(row)
    result["groups"] = {
        _label(key): summarise(minutes(group))
        for key, group in sorted(
            buckets.items(), key=lambda kv: _group_sort_key(kv[0], group_by)
        )
    }
    result["group_by"] = list(group_by)
    return result


def message_funnel(
    db: Database,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
) -> dict[str, Any]:
    """Where outbound messages went, by status and block reason.

    This is the diagnostic that separates "the intervention did not work" from
    "the intervention did not run". A high `blocked/no_consent` count means the
    intake process is not capturing consent, which is a front-desk fix, not an
    engineering one.
    """
    lo, hi = _window(since, until)
    rows = db.all(
        "SELECT status, block_reason, purpose FROM message_log"
        " WHERE planned_utc >= ? AND planned_utc <= ?",
        (lo, hi),
    )
    by_status: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    by_purpose: dict[str, dict[str, int]] = {}
    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
        if row["block_reason"]:
            by_reason[row["block_reason"]] = by_reason.get(row["block_reason"], 0) + 1
        purpose_bucket = by_purpose.setdefault(row["purpose"], {})
        purpose_bucket[row["status"]] = purpose_bucket.get(row["status"], 0) + 1
    total = len(rows)
    sent = by_status.get("sent", 0)
    return {
        "total": total,
        "by_status": dict(sorted(by_status.items())),
        "by_block_reason": dict(sorted(by_reason.items())),
        "by_purpose": {k: dict(sorted(v.items())) for k, v in sorted(by_purpose.items())},
        "delivery_rate": (sent / total) if total else None,
    }


def lead_time_buckets(
    db: Database,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    edges_days: Sequence[int] = (1, 3, 7, 14, 30),
) -> dict[str, Any]:
    """No-show rate by booking lead time — the strongest single predictor.

    README I-07 lists lead time as a feature for the optional no-show risk model
    "only after 12 months of clean data". This function is how that data gets
    clean: it is the same bucketing the model would use, computed now, so the
    practice can see the relationship long before anyone trains anything.
    """
    lo, hi = _window(since, until)
    rows = db.all(
        """SELECT status, created_utc, start_utc FROM appointment
           WHERE status IN ('completed','no_show') AND start_utc >= ? AND start_utc <= ?""",
        (lo, hi),
    )
    labels: list[str] = []
    previous = 0
    for edge in edges_days:
        labels.append(f"{previous}-{edge}d")
        previous = edge
    labels.append(f"{previous}d+")

    buckets: dict[str, list[dict[str, Any]]] = {label: [] for label in labels}
    for row in rows:
        lead_days = (
            parse_iso(row["start_utc"]) - parse_iso(row["created_utc"])
        ).total_seconds() / 86400.0
        placed = False
        previous = 0
        for edge, label in zip(edges_days, labels):
            if lead_days < edge:
                buckets[label].append(row)
                placed = True
                break
            previous = edge
        if not placed:
            buckets[labels[-1]].append(row)
    return {
        label: Rate(sum(1 for r in group if r["status"] == "no_show"), len(group)).as_dict()
        for label, group in buckets.items()
    }


def kpi_summary(
    db: Database,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    baseline_no_show_rate: float | None = None,
) -> dict[str, Any]:
    """One call producing the weekly operational dashboard (README 10.4).

    `baseline_no_show_rate` is the pre-automation figure pulled from the EHR.
    Without it the report shows the current rate and explicitly says the target
    cannot be evaluated, rather than inventing a baseline — which is the failure
    mode README 10.1 is written to prevent.
    """
    ns = no_show_rate(db, since=since, until=until)
    bf = backfill_rate(db, since=since, until=until)
    ft = fill_time_stats(db, since=since, until=until)

    current = ns["overall"]["rate"]
    # `not baseline` rather than `is None`: a baseline of 0.0 makes the relative
    # change undefined, and reporting "below_target" for an undefined comparison
    # is worse than saying the comparison cannot be made.
    if not baseline_no_show_rate or current is None:
        no_show_verdict = "baseline_not_supplied"
        relative_change = None
    else:
        relative_change = (
            (current - baseline_no_show_rate) / baseline_no_show_rate
            if baseline_no_show_rate
            else None
        )
        no_show_verdict = (
            "meets_target"
            if relative_change is not None and relative_change <= -0.25
            else "below_target"
        )

    backfill_value = bf["overall"]["rate"]
    median_fill = ft["overall"]["median_minutes"]
    return {
        "window": {"since": iso(since) if since else None, "until": iso(until) if until else None},
        "no_show": {
            **ns["overall"],
            "baseline": baseline_no_show_rate,
            "relative_change": relative_change,
            "verdict": no_show_verdict,
        },
        "backfill": {
            **bf["overall"],
            "target": KPI_TARGETS["backfill_rate"]["target"],
            "verdict": (
                "insufficient_data"
                if backfill_value is None
                else ("meets_target" if backfill_value > 0.60 else "below_target")
            ),
        },
        "fill_time": {
            **ft["overall"],
            "target_minutes": KPI_TARGETS["median_fill_minutes"]["target"],
            "verdict": (
                "insufficient_data"
                if median_fill is None
                else ("meets_target" if median_fill < 15.0 else "below_target")
            ),
        },
        "messages": message_funnel(db, since=since, until=until),
    }
