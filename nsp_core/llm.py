"""Provider-agnostic, schema-constrained LLM access for the NSP clinical stack.

WHY this module exists at all:

Every other module in this repo is forbidden from calling a model directly.
That is not stylistic. A clinical automation system has exactly three
non-negotiable properties at the inference boundary, and all three are easier
to guarantee in one place than in nine:

  1. PHI must not leave the building unless a BAA covers the destination
     (README 3.1). A cloud transport that cannot be *constructed* without an
     explicit BAA assertion is stronger than a policy document.
  2. Output must be structurally valid or the call must fail loudly. A model
     that returns prose where a schema was demanded has produced a defect, not
     a degraded result. `SchemaViolation` is raised; callers route to a human.
     Nothing in this repo is permitted to substitute a default value.
  3. Model identity must be recorded for every inference (README 9.2, R-12).
     `InferenceResult` carries provider/model/version so the caller can hand
     them straight to `AuditLog.record_inference`.

Constrained decoding is preferred over prompting wherever the transport
supports it, because a grammar makes invalid JSON *impossible* rather than
merely unlikely. A 7B model with a grammar beats a 70B model without one for
extraction work, which is all this repo asks of a model.
"""

from __future__ import annotations

import abc
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

try:  # pragma: no cover - exercised implicitly by every structured() call
    import jsonschema
except ImportError:  # pragma: no cover
    jsonschema = None  # type: ignore[assignment]

__all__ = [
    "SchemaViolation",
    "BAARequired",
    "InferenceResult",
    "Transport",
    "EchoTransport",
    "OllamaTransport",
    "LlamaCppTransport",
    "VLLMTransport",
    "TransformersTransport",
    "BedrockTransport",
    "LLMClient",
    "build_transport",
]


class SchemaViolation(RuntimeError):
    """Raised when a model's output cannot be coerced to the demanded schema.

    WHY it is an exception and not a return value: a caller that receives a
    value has to remember to check it. A caller that receives an exception
    cannot forget. Every call site in this repo catches this and routes to a
    human queue.
    """

    def __init__(self, message: str, *, raw: str = "", attempts: int = 0) -> None:
        super().__init__(message)
        self.raw_length = len(raw)  # length only - never the payload itself
        self.attempts = attempts


class BAARequired(RuntimeError):
    """Raised when a cloud transport is constructed without a BAA assertion."""


# --------------------------------------------------------------------------
# Result envelope
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InferenceResult:
    """Everything the audit log needs, and nothing it must not have.

    WHY no prompt/completion text: README 9.2. Logging PHI-bearing text into an
    observability store creates a second copy of the record under weaker
    controls than the chart it came from. Hashes and counts are sufficient to
    prove what ran; the clinical content stays in the chart.
    """

    data: dict[str, Any]
    provider: str
    model_id: str
    model_version: str
    prompt_template_id: str
    prompt_template_hash: str
    input_token_count: int
    output_token_count: int
    confidence: float | None
    constrained: bool
    repair_attempts: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _approx_tokens(text: str) -> int:
    """Cheap token estimate.

    WHY approximate: the count exists for cost/anomaly monitoring and for
    detecting a prompt that suddenly doubled in size (R-12). Exactness would
    require a tokenizer per provider for no operational gain.
    """
    return max(1, len(text) // 4)


def _hash_template(system: str, schema: Mapping[str, Any]) -> str:
    payload = system + "\x00" + json.dumps(schema, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# JSON extraction / validation
# --------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(raw: str) -> dict[str, Any]:
    """Pull the first plausible JSON object out of a model response.

    WHY tolerant parsing is acceptable here: this path only runs for transports
    without grammar support. It tolerates fences and leading prose, and nothing
    else. It never repairs semantics -- a missing required field still fails
    validation downstream, which is the point.
    """
    candidate = raw.strip()
    fenced = _FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    if not candidate.startswith("{"):
        start = candidate.find("{")
        if start == -1:
            raise SchemaViolation("no JSON object in response", raw=raw)
        candidate = candidate[start:]
    depth = 0
    end = -1
    in_string = False
    escape = False
    for i, ch in enumerate(candidate):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        raise SchemaViolation("unterminated JSON object", raw=raw)
    try:
        parsed = json.loads(candidate[:end])
    except json.JSONDecodeError as exc:
        raise SchemaViolation(f"malformed JSON: {exc.msg}", raw=raw) from exc
    if not isinstance(parsed, dict):
        raise SchemaViolation("top-level JSON value is not an object", raw=raw)
    return parsed


def validate(data: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    """Validate against the schema, raising SchemaViolation on any failure."""
    if jsonschema is None:  # pragma: no cover - dependency is in requirements
        _validate_minimal(data, schema)
        return
    validator_cls = jsonschema.validators.validator_for(schema)
    validator = validator_cls(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
    if errors:
        first = errors[0]
        path = "/".join(str(p) for p in first.path) or "<root>"
        raise SchemaViolation(f"schema violation at {path}: {first.message}")


def _validate_minimal(data: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    """Last-resort validator for environments without jsonschema installed.

    WHY it exists: fail-closed (README 3.5) means the absence of a validation
    library must not silently become the absence of validation.
    """
    required = schema.get("required", [])
    for key in required:
        if key not in data:
            raise SchemaViolation(f"missing required property {key!r}")
    props = schema.get("properties", {})
    if schema.get("additionalProperties") is False:
        extra = set(data) - set(props)
        if extra:
            raise SchemaViolation(f"additional properties not allowed: {sorted(extra)}")


def assert_strict_schema(schema: Mapping[str, Any], *, _path: str = "<root>") -> None:
    """Reject any schema that leaves a hole for unexpected keys.

    Hard constraint 2 in the build plan: every LLM call uses a strict schema
    with additionalProperties false. This is enforced, not documented.

    Object-ness is inferred from the presence of `properties`, not only from an
    explicit `"type": "object"`. WHY: `{"properties": {...}}` with no type is
    valid JSON Schema and is exactly the shape a strictness check keyed on
    `type` walks straight past -- which would make the guarantee depend on how
    the schema happened to be written. Composition keywords are descended into
    for the same reason.
    """
    looks_like_object = schema.get("type") == "object" or "properties" in schema
    if looks_like_object:
        if schema.get("additionalProperties") is not False:
            raise ValueError(
                f"schema at {_path} must set additionalProperties: false"
            )
        for name, sub in schema.get("properties", {}).items():
            if isinstance(sub, dict):
                assert_strict_schema(sub, _path=f"{_path}.{name}")
    items = schema.get("items")
    if isinstance(items, dict):
        assert_strict_schema(items, _path=f"{_path}[]")
    elif isinstance(items, list):  # tuple-form arrays
        for index, sub in enumerate(items):
            if isinstance(sub, dict):
                assert_strict_schema(sub, _path=f"{_path}[{index}]")
    for keyword in ("anyOf", "oneOf", "allOf"):
        for index, sub in enumerate(schema.get(keyword, []) or []):
            if isinstance(sub, dict):
                assert_strict_schema(sub, _path=f"{_path}.{keyword}[{index}]")
    if "$ref" in schema:
        # A $ref points somewhere this function cannot see. Refusing is the
        # fail-closed choice: an unresolvable reference means the strictness
        # guarantee cannot be established, and "cannot establish" must not read
        # as "established".
        raise ValueError(
            f"schema at {_path} uses $ref; inline the definition so strictness "
            "can be verified before the call is made"
        )


# --------------------------------------------------------------------------
# Transports
# --------------------------------------------------------------------------


class Transport(abc.ABC):
    """A single way of reaching a model.

    `supports_constrained_decoding` is the property that matters. Transports
    that can enforce a JSON grammar server-side get the schema handed to them;
    the rest fall back to prompt-and-repair in LLMClient.
    """

    name: str = "abstract"
    is_cloud: bool = False
    supports_constrained_decoding: bool = False

    def __init__(self, model_id: str, *, model_version: str = "unpinned") -> None:
        self.model_id = model_id
        # WHY a version string at all for local models: R-12. A silent model
        # swap that degrades extraction is invisible without it.
        self.model_version = model_version

    @abc.abstractmethod
    def generate(
        self,
        *,
        system: str,
        user: str,
        schema: Mapping[str, Any] | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Return raw model text."""

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return f"<{type(self).__name__} {self.model_id}>"


class EchoTransport(Transport):
    """Deterministic test double. Never reaches a network.

    WHY it lives in the shipped module rather than the test tree: the build
    plan forbids mocking our own logic in tests. A first-class, dependency-free
    transport lets tests exercise the *real* LLMClient path -- validation,
    repair, refusal -- against scripted responses.
    """

    name = "echo"
    supports_constrained_decoding = False

    def __init__(
        self,
        responses: Sequence[str] | None = None,
        *,
        model_id: str = "echo-1",
        model_version: str = "test",
    ) -> None:
        super().__init__(model_id, model_version=model_version)
        self._responses = list(responses or [])
        self._index = 0
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        *,
        system: str,
        user: str,
        schema: Mapping[str, Any] | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        self.calls.append({"system_len": len(system), "user_len": len(user)})
        if self._index < len(self._responses):
            out = self._responses[self._index]
            self._index += 1
            return out
        return json.dumps({})


class OllamaTransport(Transport):
    """Local Ollama. Uses the `format` parameter for grammar-constrained JSON."""

    name = "ollama"
    supports_constrained_decoding = True

    def __init__(
        self,
        model_id: str = "qwen2.5:14b-instruct-q5_K_M",
        *,
        host: str = "http://127.0.0.1:11434",
        model_version: str = "unpinned",
        timeout: float = 120.0,
    ) -> None:
        super().__init__(model_id, model_version=model_version)
        self.host = host.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        *,
        system: str,
        user: str,
        schema: Mapping[str, Any] | None,
        temperature: float,
        max_tokens: int,
    ) -> str:  # pragma: no cover - requires a running daemon
        import urllib.request

        body: dict[str, Any] = {
            "model": self.model_id,
            "stream": False,
            "system": system,
            "prompt": user,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if schema is not None:
            # Ollama accepts a JSON schema here and constrains decoding to it.
            body["format"] = schema
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload.get("response", "")


class LlamaCppTransport(Transport):
    """llama.cpp server. Uses `json_schema` (GBNF grammar under the hood)."""

    name = "llama.cpp"
    supports_constrained_decoding = True

    def __init__(
        self,
        model_id: str = "local-gguf",
        *,
        host: str = "http://127.0.0.1:8080",
        model_version: str = "unpinned",
        timeout: float = 120.0,
    ) -> None:
        super().__init__(model_id, model_version=model_version)
        self.host = host.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        *,
        system: str,
        user: str,
        schema: Mapping[str, Any] | None,
        temperature: float,
        max_tokens: int,
    ) -> str:  # pragma: no cover - requires a running server
        import urllib.request

        body: dict[str, Any] = {
            "prompt": f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n",
            "temperature": temperature,
            "n_predict": max_tokens,
        }
        if schema is not None:
            body["json_schema"] = schema
        req = urllib.request.Request(
            f"{self.host}/completion",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload.get("content", "")


class VLLMTransport(Transport):
    """vLLM OpenAI-compatible server. Uses `guided_json`."""

    name = "vllm"
    supports_constrained_decoding = True

    def __init__(
        self,
        model_id: str,
        *,
        host: str = "http://127.0.0.1:8000",
        model_version: str = "unpinned",
        timeout: float = 120.0,
    ) -> None:
        super().__init__(model_id, model_version=model_version)
        self.host = host.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        *,
        system: str,
        user: str,
        schema: Mapping[str, Any] | None,
        temperature: float,
        max_tokens: int,
    ) -> str:  # pragma: no cover - requires a running server
        import urllib.request

        body: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if schema is not None:
            body["guided_json"] = schema
        req = urllib.request.Request(
            f"{self.host}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"]


class TransformersTransport(Transport):
    """In-process HuggingFace transformers. No constrained decoding.

    WHY it is still offered: the de-identification NER path and small
    classifiers run fine here, and it removes the need for a daemon on a
    single-workstation deployment.
    """

    name = "transformers"
    supports_constrained_decoding = False

    def __init__(
        self,
        model_id: str,
        *,
        model_version: str = "unpinned",
        device: str = "auto",
        dtype: str = "auto",
    ) -> None:
        super().__init__(model_id, model_version=model_version)
        self.device = device
        self.dtype = dtype
        self._pipe = None

    def _pipeline(self):  # pragma: no cover - heavy optional dependency
        if self._pipe is None:
            from transformers import pipeline  # lazy: keeps import cost off the CLI

            self._pipe = pipeline(
                "text-generation",
                model=self.model_id,
                device_map=self.device,
                torch_dtype=self.dtype,
            )
        return self._pipe

    def generate(
        self,
        *,
        system: str,
        user: str,
        schema: Mapping[str, Any] | None,
        temperature: float,
        max_tokens: int,
    ) -> str:  # pragma: no cover - heavy optional dependency
        pipe = self._pipeline()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        out = pipe(
            messages,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
        )
        generated = out[0]["generated_text"]
        if isinstance(generated, list):
            return generated[-1]["content"]
        return str(generated)


class BedrockTransport(Transport):
    """AWS Bedrock. The ONLY cloud path, and it is gated.

    WHY the gate is a constructor argument rather than a config flag: a config
    flag can be flipped by anyone editing YAML. A constructor argument shows up
    in code review as `baa_on_file=True` on a specific line, authored by a
    specific person, in a specific commit. README 3.1 is a legal boundary; it
    deserves a legal-grade paper trail.

    boto3 is imported lazily inside the request path so that the default,
    local-only deployment never imports an AWS SDK at all.
    """

    name = "bedrock"
    is_cloud = True
    supports_constrained_decoding = False

    def __init__(
        self,
        model_id: str,
        *,
        baa_on_file: bool = False,
        baa_reference: str = "",
        region: str = "us-east-2",
        model_version: str = "unpinned",
    ) -> None:
        if not baa_on_file:
            raise BAARequired(
                "BedrockTransport requires baa_on_file=True. No PHI may reach a "
                "cloud endpoint without an executed BAA (README 3.1)."
            )
        if not baa_reference:
            raise BAARequired(
                "baa_on_file=True requires a baa_reference identifying the "
                "executed agreement, so the audit trail names the document."
            )
        super().__init__(model_id, model_version=model_version)
        self.region = region
        self.baa_reference = baa_reference

    def generate(
        self,
        *,
        system: str,
        user: str,
        schema: Mapping[str, Any] | None,
        temperature: float,
        max_tokens: int,
    ) -> str:  # pragma: no cover - requires AWS credentials
        import boto3  # lazy, cloud-gated: see class docstring

        client = boto3.client("bedrock-runtime", region_name=self.region)
        response = client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"temperature": temperature, "maxTokens": max_tokens},
        )
        return response["output"]["message"]["content"][0]["text"]


_TRANSPORTS: dict[str, type[Transport]] = {
    "echo": EchoTransport,
    "ollama": OllamaTransport,
    "llamacpp": LlamaCppTransport,
    "vllm": VLLMTransport,
    "transformers": TransformersTransport,
    "bedrock": BedrockTransport,
}


def build_transport(spec: str | None = None, **kwargs: Any) -> Transport:
    """Construct a transport from a name, defaulting to local Ollama.

    WHY the default is local: hard constraint 1. An operator who configures
    nothing gets a local model, not a cloud bill and a compliance incident.
    """
    spec = spec or os.environ.get("NSP_LLM_PROVIDER", "ollama")
    try:
        cls = _TRANSPORTS[spec]
    except KeyError as exc:
        raise ValueError(
            f"unknown provider {spec!r}; known: {sorted(_TRANSPORTS)}"
        ) from exc
    return cls(**kwargs)


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------

_REPAIR_INSTRUCTION = (
    "Your previous response did not satisfy the required JSON schema.\n"
    "Error: {error}\n"
    "Return ONLY a single JSON object satisfying the schema. No prose, no "
    "code fences, no explanation."
)


class LLMClient:
    """The only sanctioned way to call a model in this repo.

    `structured()` is the entire public surface. There is deliberately no
    `complete()` or `chat()`: free-text generation with no schema has no
    legitimate use in this system, and offering it would invite one.
    """

    def __init__(
        self,
        transport: Transport | None = None,
        *,
        max_repair_attempts: int = 2,
        default_temperature: float = 0.0,
        default_max_tokens: int = 1500,
    ) -> None:
        self.transport = transport or build_transport()
        self.max_repair_attempts = max_repair_attempts
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

    @property
    def provider(self) -> str:
        return self.transport.name

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
        prompt_template_id: str = "adhoc",
        temperature: float | None = None,
        max_tokens: int | None = None,
        confidence_key: str = "confidence",
    ) -> InferenceResult:
        """Run one inference and return validated structured output.

        Raises SchemaViolation if the model cannot produce conforming output
        within `max_repair_attempts`. It never returns partial or defaulted
        data -- callers route the exception to a human queue (hard constraint 3).

        `context` is metadata for the audit record (initiative_id, user_id,
        patient reference). It is NOT interpolated into the prompt; the caller
        is responsible for what the model sees, and for de-identifying it.
        """
        assert_strict_schema(schema)
        temperature = self.default_temperature if temperature is None else temperature
        max_tokens = self.default_max_tokens if max_tokens is None else max_tokens
        constrained = self.transport.supports_constrained_decoding
        schema_arg: Mapping[str, Any] | None = schema if constrained else None

        prompt_user = user
        if not constrained:
            # Prompted fallback. The schema goes in the prompt because the
            # transport cannot enforce it; validation still enforces it.
            prompt_user = (
                f"{user}\n\nRespond with ONLY a JSON object matching this schema:\n"
                f"{json.dumps(schema, indent=2)}"
            )

        last_error = ""
        raw = ""
        for attempt in range(self.max_repair_attempts + 1):
            raw = self.transport.generate(
                system=system,
                user=prompt_user,
                schema=schema_arg,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            try:
                data = extract_json(raw)
                validate(data, schema)
            except SchemaViolation as exc:
                last_error = str(exc)
                if attempt == self.max_repair_attempts:
                    raise SchemaViolation(
                        f"model failed schema after {attempt + 1} attempts: {last_error}",
                        raw=raw,
                        attempts=attempt + 1,
                    ) from exc
                prompt_user = (
                    f"{user}\n\n{_REPAIR_INSTRUCTION.format(error=last_error)}\n"
                    f"Schema:\n{json.dumps(schema, indent=2)}"
                )
                continue

            confidence = data.get(confidence_key)
            return InferenceResult(
                data=data,
                provider=self.transport.name,
                model_id=self.transport.model_id,
                model_version=self.transport.model_version,
                prompt_template_id=prompt_template_id,
                prompt_template_hash=_hash_template(system, schema),
                input_token_count=_approx_tokens(system) + _approx_tokens(prompt_user),
                output_token_count=_approx_tokens(raw),
                confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
                constrained=constrained,
                repair_attempts=attempt,
                metadata=dict(context or {}),
            )
        raise AssertionError("unreachable")  # pragma: no cover
