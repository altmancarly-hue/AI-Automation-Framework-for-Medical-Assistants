"""I-07 — No-show reduction and waitlist backfill.

Deterministic scheduling automation. There is no language model in this package
and there must not be one: README I-07 is explicit that reminders, cancellations
and backfill are a cron job, a rules table and a messaging API. `make lint`
enforces this by grepping for any import of `nsp_core.llm` under this directory.

Typical wiring:

    from modules.scheduling import (
        Database, LocalGateway, ReminderEngine, BackfillEngine, InboundRouter,
    )

    db = Database("var/scheduling.sqlite3")
    gw = LocalGateway("var/outbox.jsonl")
    reminders = ReminderEngine(db, gw)
    backfill = BackfillEngine(db, gw)
    router = InboundRouter(db, gw, backfill)

    # cron, every five minutes
    reminders.plan_horizon(now=now)
    reminders.dispatch_due(now=now)
    backfill.expire_stale_offers(now=now)
    backfill.close_unfilled_releases(now=now)
"""

from .backfill import AcceptResult, BackfillEngine, Candidate, rank_candidates
from .cadence import (
    DEFAULT_CADENCE,
    CapDecision,
    FrequencyCap,
    QuietHours,
    ReminderEngine,
    ReminderRule,
    SendDecision,
    SendGate,
    plan_reminders,
)
from .gateway import (
    Gateway,
    GatewayReceipt,
    InboundAction,
    InboundIntent,
    LocalGateway,
    TwilioGateway,
    classify_inbound,
)
from .inbound import InboundResult, InboundRouter
from .metrics import (
    backfill_rate,
    fill_time_stats,
    kpi_summary,
    lead_time_buckets,
    message_funnel,
    no_show_rate,
)
from .models import (
    PRACTICE_TZ,
    AppointmentStatus,
    Channel,
    ConsentPurpose,
    Database,
    MessagePurpose,
    OfferOutcome,
    VisitType,
    add_appointment,
    add_family,
    add_patient,
    add_provider,
    add_waitlist_entry,
    grant_consent,
    revoke_consent,
    seed_practice_defaults,
)

__all__ = [
    "PRACTICE_TZ",
    "AcceptResult",
    "AppointmentStatus",
    "BackfillEngine",
    "Candidate",
    "CapDecision",
    "Channel",
    "ConsentPurpose",
    "DEFAULT_CADENCE",
    "Database",
    "FrequencyCap",
    "Gateway",
    "GatewayReceipt",
    "InboundAction",
    "InboundIntent",
    "InboundResult",
    "InboundRouter",
    "LocalGateway",
    "MessagePurpose",
    "OfferOutcome",
    "QuietHours",
    "ReminderEngine",
    "ReminderRule",
    "SendDecision",
    "SendGate",
    "TwilioGateway",
    "VisitType",
    "add_appointment",
    "add_family",
    "add_patient",
    "add_provider",
    "add_waitlist_entry",
    "backfill_rate",
    "classify_inbound",
    "fill_time_stats",
    "grant_consent",
    "kpi_summary",
    "lead_time_buckets",
    "message_funnel",
    "no_show_rate",
    "plan_reminders",
    "rank_candidates",
    "revoke_consent",
    "seed_practice_defaults",
]
