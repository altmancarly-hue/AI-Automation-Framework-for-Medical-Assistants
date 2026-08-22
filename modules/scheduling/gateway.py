"""Messaging transport and deterministic inbound intent handling.

WHY an abstraction rather than calling Twilio directly:

README R-07 — vendor discontinuation or acquisition — is a live risk in the
patient-communication category, and the README's own recommendation (Spruce or
Curogram, with Twilio as the DIY fallback) may change before this ships. A
practice that has wired a vendor SDK into its reminder logic cannot switch
without a rewrite. The interface here is four fields wide; swapping vendors is
a new subclass.

WHY inbound handling is a lookup table and not NLP:

README I-07: "deterministic intent match on the tap payload (NOT free-text
NLP - use structured buttons/links)". Every outbound message carries signed
action links. A tap is an unambiguous action code. The only free text this
system interprets is the carrier-standard keyword set (STOP / START / HELP),
which is a fixed vocabulary that carriers already enforce and that has legal
consequences if mishandled. Anything else a parent types goes to a human,
because "no I meant next week" is a conversation, not an intent class.
"""

from __future__ import annotations

import abc
import json
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from .models import Channel, iso

__all__ = [
    "GatewayReceipt",
    "Gateway",
    "LocalGateway",
    "TwilioGateway",
    "InboundAction",
    "InboundIntent",
    "classify_inbound",
    "STOP_KEYWORDS",
    "START_KEYWORDS",
    "HELP_KEYWORDS",
]


@dataclass(frozen=True)
class GatewayReceipt:
    accepted: bool
    gateway_ref: str | None = None
    error: str | None = None


class Gateway(abc.ABC):
    """Outbound messaging transport."""

    name: str = "abstract"

    @abc.abstractmethod
    def send(
        self,
        *,
        to: str,
        body: str,
        channel: str,
        purpose: str,
        reference: str,
    ) -> GatewayReceipt:
        """Attempt delivery. Never raises for a delivery failure -- returns a
        receipt with accepted=False, so the caller records the failure in the
        message log rather than losing it in a traceback."""


class LocalGateway(Gateway):
    """Writes messages to a JSONL file. The default in dev and test.

    WHY it is a real, shipped gateway and not a test mock: the build plan
    forbids mocking our own logic. Tests need to assert on what was actually
    sent, and the reminder engine must run its genuine dispatch path while
    doing so. A file-backed gateway gives both, and doubles as the dry-run mode
    an operator wants on day one of a rollout.
    """

    name = "local"

    def __init__(self, path: str | os.PathLike[str] = "var/outbox.jsonl") -> None:
        self.path = str(path)
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._lock = threading.Lock()
        self.sent: list[dict[str, Any]] = []

    def send(
        self,
        *,
        to: str,
        body: str,
        channel: str,
        purpose: str,
        reference: str,
    ) -> GatewayReceipt:
        record = {
            "ts": iso(datetime.now(timezone.utc)),
            "to": to,
            "channel": channel,
            "purpose": purpose,
            "reference": reference,
            "body": body,
        }
        ref = f"local-{reference}"
        with self._lock:
            self.sent.append(record)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        return GatewayReceipt(True, ref)

    def messages_for(self, purpose: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if purpose is None:
                return list(self.sent)
            return [m for m in self.sent if m["purpose"] == purpose]

    def clear(self) -> None:
        with self._lock:
            self.sent.clear()


class TwilioGateway(Gateway):
    """Twilio SMS/voice. The SDK is imported lazily inside send().

    WHY lazy: the default deployment path must not require the dependency, and
    `pip install twilio` must not be a prerequisite for running the test suite
    or the reminder cron in dry-run mode.
    """

    name = "twilio"

    def __init__(
        self,
        *,
        account_sid: str | None = None,
        auth_token: str | None = None,
        from_number: str | None = None,
        messaging_service_sid: str | None = None,
        status_callback: str | None = None,
    ) -> None:
        self.account_sid = account_sid or os.environ.get("TWILIO_ACCOUNT_SID", "")
        self.auth_token = auth_token or os.environ.get("TWILIO_AUTH_TOKEN", "")
        self.from_number = from_number or os.environ.get("TWILIO_FROM_NUMBER", "")
        self.messaging_service_sid = messaging_service_sid or os.environ.get(
            "TWILIO_MESSAGING_SERVICE_SID", ""
        )
        self.status_callback = status_callback
        if not self.account_sid or not self.auth_token:
            raise ValueError(
                "TwilioGateway requires account_sid and auth_token. A BAA must be "
                "executed with Twilio before any patient identifier is sent."
            )
        if not self.from_number and not self.messaging_service_sid:
            raise ValueError("provide either from_number or messaging_service_sid")
        self._client: Any = None

    def _get_client(self) -> Any:  # pragma: no cover - requires the SDK
        if self._client is None:
            from twilio.rest import Client  # lazy: see class docstring

            self._client = Client(self.account_sid, self.auth_token)
        return self._client

    def send(
        self,
        *,
        to: str,
        body: str,
        channel: str,
        purpose: str,
        reference: str,
    ) -> GatewayReceipt:  # pragma: no cover - requires network + credentials
        if channel != Channel.SMS:
            return GatewayReceipt(False, error=f"TwilioGateway does not handle {channel}")
        try:
            kwargs: dict[str, Any] = {"to": to, "body": body}
            if self.messaging_service_sid:
                kwargs["messaging_service_sid"] = self.messaging_service_sid
            else:
                kwargs["from_"] = self.from_number
            if self.status_callback:
                kwargs["status_callback"] = self.status_callback
            message = self._get_client().messages.create(**kwargs)
            return GatewayReceipt(True, message.sid)
        except Exception as exc:
            return GatewayReceipt(False, error=f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# Inbound
# --------------------------------------------------------------------------


class InboundAction:
    CONFIRM = "confirm"
    CANCEL = "cancel"
    RESCHEDULE = "reschedule"
    ACCEPT_OFFER = "accept_offer"
    DECLINE_OFFER = "decline_offer"
    OPT_OUT = "opt_out"
    OPT_IN = "opt_in"
    HELP = "help"
    HUMAN = "human"  # anything we will not guess at
    ALL = (
        CONFIRM,
        CANCEL,
        RESCHEDULE,
        ACCEPT_OFFER,
        DECLINE_OFFER,
        OPT_OUT,
        OPT_IN,
        HELP,
        HUMAN,
    )


@dataclass(frozen=True)
class InboundIntent:
    action: str
    reference: str | None = None
    from_address: str = ""
    raw_length: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


# Carrier-standard keyword sets. These are not a natural-language model; they
# are a fixed vocabulary with regulatory meaning. STOP in particular must be
# honoured immediately and unconditionally.
STOP_KEYWORDS = frozenset({"stop", "stopall", "unsubscribe", "cancel", "end", "quit"})
START_KEYWORDS = frozenset({"start", "yes", "unstop"})
HELP_KEYWORDS = frozenset({"help", "info"})

_TAP_PATH = re.compile(r"^/(?P<code>[cxrad])/(?P<ref>[A-Za-z0-9_-]+)$")
_TAP_CODES = {
    "c": InboundAction.CONFIRM,
    "x": InboundAction.CANCEL,
    "r": InboundAction.RESCHEDULE,
    "a": InboundAction.ACCEPT_OFFER,
    "d": InboundAction.DECLINE_OFFER,
}


def classify_inbound(payload: Mapping[str, Any]) -> InboundIntent:
    """Map a webhook payload to exactly one action, or to a human.

    Accepts either:
      * a tap: {"path": "/a/offer_123"} — an action link from a sent message
      * a keyword: {"body": "STOP", "from": "+18475550123"}

    Precedence is tap first, then keyword, then human. WHY that order: a tap is
    an unambiguous signal from a link we generated; body text is a guess about
    what a person meant. When both are present the tap wins.

    "CANCEL" is deliberately in STOP_KEYWORDS even though the practice would
    prefer it to mean "cancel my appointment". Carriers treat CANCEL as an
    opt-out keyword and will unsubscribe the number regardless of what this
    code decides. Interpreting it as an appointment cancellation would leave
    our state and the carrier's state disagreeing about whether the family is
    subscribed — and the carrier's state is the one that governs. The message
    templates therefore route cancellation through a link, never a keyword.
    """
    path = str(payload.get("path") or "").strip()
    if path:
        match = _TAP_PATH.match(path)
        if match:
            return InboundIntent(
                action=_TAP_CODES[match.group("code")],
                reference=match.group("ref"),
                from_address=str(payload.get("from", "")),
                metadata={"source": "tap"},
            )

    body = str(payload.get("body") or "").strip()
    normalised = re.sub(r"[^a-z]", "", body.lower())
    if normalised in STOP_KEYWORDS:
        return InboundIntent(
            InboundAction.OPT_OUT,
            from_address=str(payload.get("from", "")),
            raw_length=len(body),
            metadata={"source": "keyword", "keyword": normalised},
        )
    if normalised in START_KEYWORDS:
        return InboundIntent(
            InboundAction.OPT_IN,
            from_address=str(payload.get("from", "")),
            raw_length=len(body),
            metadata={"source": "keyword", "keyword": normalised},
        )
    if normalised in HELP_KEYWORDS:
        return InboundIntent(
            InboundAction.HELP,
            from_address=str(payload.get("from", "")),
            raw_length=len(body),
            metadata={"source": "keyword", "keyword": normalised},
        )

    # Everything else is a person trying to have a conversation. Route it to
    # staff. Note that only the length of the message is retained here -- the
    # body is patient-authored text and belongs in the message queue a human
    # reads, not in an intent-classification metadata blob.
    return InboundIntent(
        InboundAction.HUMAN,
        from_address=str(payload.get("from", "")),
        raw_length=len(body),
        metadata={"source": "freetext"},
    )
