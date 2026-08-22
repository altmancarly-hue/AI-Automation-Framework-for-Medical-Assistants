"""De-identification with reversible tokenization, and an outbound leak guard.

WHY reversible rather than redaction:

Most of this repo's tasks are structured extraction from clinical text. The
model does not need to know that the patient is Maya Whitfield -- but the
*output* must come back naming Maya Whitfield, or a human has to re-identify it
by hand and the automation has bought nothing. So identifiers are replaced with
stable, typed tokens (`[[NAME_1]]`), the model sees only tokens, and
`TokenVault.rehydrate_obj` walks the returned structure and puts the real values
back. The vault never leaves the process.

WHY hybrid regex + NER:

Regex catches the structured identifiers exhaustively and cheaply -- MRNs,
phones, SSNs, dates, emails, ZIPs. It cannot catch a name it has never seen.
A small clinical NER encoder (obi/deid_roberta_i2b2, 125M params, CPU-viable)
catches names and places, and beats a general LLM at this specific job while
costing a fraction of one. Neither alone is sufficient; the union is what
Safe Harbor requires you to attempt.

WHY the leak guard:

De-identification that is never checked is a belief, not a control. `LeakGuard`
runs immediately before any egress and raises if a known identifier survived
into the outbound payload. It is deliberately independent of the de-identifier:
a bug in the substitution pass should not also be a bug in the check that
catches it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Pattern

__all__ = [
    "PHILeakError",
    "Entity",
    "TokenVault",
    "RegexDeidentifier",
    "ClinicalNERDeidentifier",
    "HybridDeidentifier",
    "LeakGuard",
    "truncate_token_safe",
]


class PHILeakError(RuntimeError):
    """Raised when identifiable content is detected on an outbound path."""


@dataclass(frozen=True)
class Entity:
    """One detected identifier."""

    start: int
    end: int
    label: str
    text: str
    source: str  # "regex" | "ner"
    score: float = 1.0


# --------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------

# Ordered most-specific first. Overlaps are resolved by span priority in
# _merge, so SSN wins over a bare number sequence.
_PATTERNS: list[tuple[str, Pattern[str]]] = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b")),
    (
        "PHONE",
        re.compile(r"(?<!\d)(?:\+1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}(?!\d)"),
    ),
    ("MRN", re.compile(r"\b(?:MRN|MR#|Chart|Record)[:# ]*\s*([A-Z]{0,3}\d{5,10})\b", re.I)),
    ("DATE", re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")),
    (
        "DATE",
        re.compile(
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b"
        ),
    ),
    ("ZIP", re.compile(r"\b\d{5}(?:-\d{4})?\b")),
    ("URL", re.compile(r"\bhttps?://\S+\b")),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("POLICY", re.compile(r"\b(?:policy|member|subscriber)\s*(?:id|#|no\.?)[:# ]*\s*([A-Z0-9-]{6,20})\b", re.I)),
    ("DEVICE", re.compile(r"\b(?:VFC|Lot)\s*#?\s*([A-Z0-9-]{4,20})\b")),
]

_TOKEN_RE = re.compile(r"\[\[([A-Z_]+)_(\d+)\]\]")


# --------------------------------------------------------------------------
# Vault
# --------------------------------------------------------------------------


class TokenVault:
    """Maps surrogate tokens back to original values. Process-local only.

    Values are keyed by (label, original_text) so the same name appearing five
    times in a transcript yields the same token five times. That consistency is
    what lets a model reason about "the patient" across a document without ever
    seeing who the patient is.
    """

    def __init__(self) -> None:
        self._forward: dict[tuple[str, str], str] = {}
        self._reverse: dict[str, str] = {}
        self._counters: dict[str, int] = {}

    def token_for(self, label: str, value: str) -> str:
        key = (label, value)
        existing = self._forward.get(key)
        if existing is not None:
            return existing
        self._counters[label] = self._counters.get(label, 0) + 1
        token = f"[[{label}_{self._counters[label]}]]"
        self._forward[key] = token
        self._reverse[token] = value
        return token

    def original(self, token: str) -> str | None:
        return self._reverse.get(token)

    def __len__(self) -> int:
        return len(self._reverse)

    @property
    def tokens(self) -> tuple[str, ...]:
        return tuple(self._reverse)

    @property
    def originals(self) -> tuple[str, ...]:
        return tuple(self._reverse.values())

    def rehydrate(self, text: str) -> str:
        """Replace every known token in a string with its original value."""
        def _sub(match: re.Match[str]) -> str:
            token = match.group(0)
            return self._reverse.get(token, token)

        return _TOKEN_RE.sub(_sub, text)

    def rehydrate_obj(self, obj: Any) -> Any:
        """Walk an arbitrary structure and rehydrate every string within it.

        WHY recursive: model output is a nested dict of lists of dicts. A
        rehydration pass that only handled top-level strings would silently
        leave `[[NAME_1]]` inside `findings[2].detail`, and that string would
        reach a chart. Depth is not optional.
        """
        if isinstance(obj, str):
            return self.rehydrate(obj)
        if isinstance(obj, list):
            return [self.rehydrate_obj(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self.rehydrate_obj(v) for v in obj)
        if isinstance(obj, dict):
            return {self.rehydrate_obj(k): self.rehydrate_obj(v) for k, v in obj.items()}
        return obj


# --------------------------------------------------------------------------
# Detectors
# --------------------------------------------------------------------------


def _merge(entities: Iterable[Entity]) -> list[Entity]:
    """Resolve overlapping spans, keeping the longest (then highest-scoring).

    A span that starts inside a kept span is TRIMMED to its non-overlapping
    tail, never discarded. WHY that distinction matters more than it looks:
    this function resolves the regex-union-NER result, and boundary
    misalignment is the normal NER failure mode. Dropping the whole span means
    that in

        "Well visit 12/25/2024 Maya Whitfield, DOB noted."

    a PATIENT span the model reported as starting four characters early loses
    to the DATE span and the name is never redacted at all. It then reaches the
    model, and LeakGuard cannot catch it either -- it was never vaulted, and no
    regex matches a name. Trimming keeps the tail, so the name is still
    tokenised.
    """
    ordered = sorted(entities, key=lambda e: (e.start, -(e.end - e.start), -e.score))
    kept: list[Entity] = []
    last_end = -1
    for ent in ordered:
        if ent.start >= last_end:
            kept.append(ent)
            last_end = ent.end
            continue
        if ent.end <= last_end:
            continue  # fully contained: genuinely redundant
        trimmed = Entity(
            start=last_end,
            end=ent.end,
            label=ent.label,
            text=ent.text[last_end - ent.start :],
            source=ent.source,
            score=ent.score,
        )
        if trimmed.text.strip():
            kept.append(trimmed)
            last_end = trimmed.end
    return kept


class RegexDeidentifier:
    """Structured-identifier detection. Deterministic, exhaustive, cheap."""

    def __init__(self, extra_patterns: Iterable[tuple[str, Pattern[str]]] = ()) -> None:
        self.patterns = list(_PATTERNS) + list(extra_patterns)

    def detect(self, text: str) -> list[Entity]:
        found: list[Entity] = []
        for label, pattern in self.patterns:
            for match in pattern.finditer(text):
                # Use group(1) when the pattern isolates the identifier from a
                # label like "MRN:", so the word "MRN" stays in the text and
                # the model retains the context it needs.
                if pattern.groups:
                    start, end = match.span(1)
                else:
                    start, end = match.span(0)
                found.append(
                    Entity(start, end, label, text[start:end], "regex", 1.0)
                )
        return _merge(found)


class ClinicalNERDeidentifier:
    """Wraps a clinical de-ID NER encoder. Optional at import time.

    Default model: obi/deid_roberta_i2b2. WHY that one: it is 125M parameters,
    runs on a laptop CPU, and outperforms general instruct LLMs on i2b2
    de-identification. This is the single place in the repo where a
    domain-specific model genuinely beats a general one.

    If transformers is not installed, `available` is False and the hybrid
    de-identifier degrades to regex-only *and says so* -- it does not pretend
    to be doing NER it is not doing.
    """

    def __init__(
        self,
        model_id: str = "obi/deid_roberta_i2b2",
        *,
        min_score: float = 0.5,
        eager: bool = False,
    ) -> None:
        self.model_id = model_id
        self.min_score = min_score
        self._pipe: Any = None
        self._unavailable_reason: str | None = None
        if eager:
            self._load()

    def _load(self) -> Any:  # pragma: no cover - heavy optional dependency
        if self._pipe is None and self._unavailable_reason is None:
            try:
                from transformers import pipeline

                self._pipe = pipeline(
                    "token-classification",
                    model=self.model_id,
                    aggregation_strategy="simple",
                )
            except Exception as exc:
                self._unavailable_reason = f"{type(exc).__name__}: {exc}"
        return self._pipe

    @property
    def available(self) -> bool:
        return self._load() is not None  # pragma: no cover

    def detect(self, text: str) -> list[Entity]:  # pragma: no cover - optional dep
        pipe = self._load()
        if pipe is None:
            return []
        out = []
        for span in pipe(text):
            score = float(span.get("score", 0.0))
            if score < self.min_score:
                continue
            label = str(span.get("entity_group", "NAME")).upper().replace("-", "_")
            out.append(
                Entity(
                    int(span["start"]),
                    int(span["end"]),
                    label,
                    text[int(span["start"]) : int(span["end"])],
                    "ner",
                    score,
                )
            )
        return out


@dataclass
class DeidResult:
    text: str
    vault: TokenVault
    entities: list[Entity] = field(default_factory=list)
    ner_used: bool = False


class HybridDeidentifier:
    """Regex + NER union, substituted right-to-left into surrogate tokens."""

    def __init__(
        self,
        *,
        regex: RegexDeidentifier | None = None,
        ner: ClinicalNERDeidentifier | None = None,
        use_ner: bool = True,
    ) -> None:
        self.regex = regex or RegexDeidentifier()
        self.ner = ner if ner is not None else (ClinicalNERDeidentifier() if use_ner else None)

    def detect(self, text: str) -> tuple[list[Entity], bool]:
        entities = list(self.regex.detect(text))
        ner_used = False
        if self.ner is not None:
            ner_entities = self.ner.detect(text)
            if ner_entities:
                ner_used = True
                entities.extend(ner_entities)
        # Regex wins ties: score 1.0 beats any NER confidence, and structured
        # identifiers are the ones with legal weight.
        return _merge(entities), ner_used

    def deidentify(self, text: str, *, vault: TokenVault | None = None) -> DeidResult:
        """Return text with identifiers replaced by stable surrogate tokens."""
        vault = vault or TokenVault()
        entities, ner_used = self.detect(text)
        out = text
        # Right-to-left so earlier spans keep their offsets.
        for ent in sorted(entities, key=lambda e: e.start, reverse=True):
            token = vault.token_for(ent.label, ent.text)
            out = out[: ent.start] + token + out[ent.end :]
        return DeidResult(text=out, vault=vault, entities=entities, ner_used=ner_used)


# --------------------------------------------------------------------------
# Egress control
# --------------------------------------------------------------------------


class LeakGuard:
    """Final check before a payload leaves the process.

    WHY it re-derives rather than trusting the de-identifier: the two passes
    fail independently. If the substitution loop has an off-by-one, the guard
    still catches the surviving identifier. Fail closed (README 3.5).
    """

    def __init__(
        self,
        *,
        regex: RegexDeidentifier | None = None,
        allow_labels: Iterable[str] = (),
    ) -> None:
        self.regex = regex or RegexDeidentifier()
        # Some tasks genuinely require an identifier class to survive to the
        # model. Immunization adjudication (README A.2) has to compare
        # administration DATES -- stripping them would make the task
        # unanswerable. Allowing a label is a deliberate, named exception made
        # at one call site, not a global loosening: everything else still
        # raises, and the vault check below is unaffected.
        self.allow_labels = frozenset(l.upper() for l in allow_labels)

    def scan(
        self,
        payload: Any,
        *,
        vault: TokenVault | None = None,
        allow_labels: Iterable[str] = (),
    ) -> list[Entity]:
        allowed = self.allow_labels | frozenset(l.upper() for l in allow_labels)
        text = payload if isinstance(payload, str) else _flatten(payload)
        findings = [e for e in self.regex.detect(text) if e.label not in allowed]
        if vault is not None:
            for original in vault.originals:
                # A known original value appearing verbatim is a leak even if
                # no pattern matches it -- this is how NER-detected names get
                # caught by a regex-based guard.
                for match in re.finditer(re.escape(original), text):
                    findings.append(
                        Entity(match.start(), match.end(), "VAULTED", original, "vault", 1.0)
                    )
        return _merge(findings)

    def assert_clean(
        self,
        payload: Any,
        *,
        vault: TokenVault | None = None,
        allow_labels: Iterable[str] = (),
    ) -> None:
        findings = self.scan(payload, vault=vault, allow_labels=allow_labels)
        if findings:
            labels = sorted({f.label for f in findings})
            raise PHILeakError(
                f"outbound payload contains {len(findings)} identifier(s) "
                f"of type(s) {labels}; refusing to transmit"
            )


def _flatten(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, Mapping):
        return " ".join(_flatten(k) + " " + _flatten(v) for k, v in obj.items())
    if isinstance(obj, (list, tuple, set)):
        return " ".join(_flatten(v) for v in obj)
    return str(obj)


def truncate_token_safe(text: str, max_chars: int) -> str:
    """Truncate without ever splitting a surrogate token in half.

    WHY this matters: a truncation that leaves `[[NAME_` at the end of the
    prompt produces a token the vault cannot rehydrate, and a model that sees
    a malformed token may echo it into output as literal text. Truncation
    backs up to the last complete token boundary.
    """
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    open_idx = cut.rfind("[[")
    close_idx = cut.rfind("]]")
    if open_idx > close_idx:
        cut = cut[:open_idx]
    return cut.rstrip()
