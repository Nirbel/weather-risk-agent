"""The only door to the LLM: structured output, local validation, one retry, provider fallback.

Two separate failure paths (docs/DECISIONS.md D3, D5):
  - provider failure (5xx/timeout/auth, long rate limits) → try the next model in `models`;
    a short rate limit (≤ 12 s, e.g. Groq's 8K tokens/min) or a temporary overload (HTTP 503,
    "high demand") is waited out once on the same model first;
  - invalid output (schema, a provider rejecting its own JSON, or the extra `check`)
    → one retry with the errors, then LLMValidationError.
"""

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from weather_risk.agent.schemas import strict_json_schema

T = TypeVar("T", bound=BaseModel)

GENERATION_REJECTED = ("json_validate_failed", "does not match the expected schema", "failed_generation")
RETRY_AFTER = re.compile(r"try again in ([\d.]+)\s*s", re.IGNORECASE)
MAX_RATE_LIMIT_WAIT_S = 12.0
TRANSIENT = ("503", "ServiceUnavailable", "overloaded", "high demand")
TRANSIENT_WAIT_S = 3.0


class _GenerationRejected(ValueError):
    """The provider refused its own output as not matching the schema (counts as invalid output)."""


class LLMError(Exception):
    meta: "CallMeta | None" = None


class LLMUnavailableError(LLMError):
    """Every configured provider failed."""


class LLMValidationError(LLMError):
    """Output was still invalid after one corrective retry."""


@dataclass
class CallMeta:
    model: str | None = None
    fallback_used: bool = False
    attempts: int = 0
    latency_ms: int = 0
    provider_errors: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)


def _content(response: Any) -> str:
    message = response.choices[0].message
    content = message.content
    if not content and getattr(message, "tool_calls", None):
        content = message.tool_calls[0].function.arguments  # some providers return JSON mode as a tool call
    if not content:
        raise ValueError("empty response")
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())


def _describe(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        lines = [f"- {'.'.join(str(p) for p in e['loc']) or 'output'}: {e['msg']}" for e in exc.errors()[:8]]
        return "\n".join(lines)
    return f"- {exc}"


def _kwargs(model: str, temperature: float) -> dict:
    # Gemini 3+ deprecates sampling parameters; everything else gets the requested temperature.
    return {} if model.startswith("gemini/") else {"temperature": temperature}


async def _complete(completion: Callable[..., Awaitable[Any]], messages: list[dict], models: list[str],
                    response_format: dict, meta: CallMeta, temperature: float, timeout: float) -> str:
    for i, model in enumerate(models):
        for try_ in (1, 2):
            try:
                response = await completion(model=model, messages=messages, response_format=response_format,
                                            timeout=timeout, **_kwargs(model, temperature))
                content = _content(response)
            except Exception as exc:  # provider failure → wait out a short rate limit, else next model
                text = str(exc)
                if any(marker in text for marker in GENERATION_REJECTED):
                    raise _GenerationRejected(f"{model} rejected its own output: it did not match the JSON schema")
                wait = RETRY_AFTER.search(text)
                if try_ == 1 and wait and float(wait.group(1)) <= MAX_RATE_LIMIT_WAIT_S:
                    meta.provider_errors.append(f"{model}: rate limited, waited {float(wait.group(1)):.1f}s")
                    await asyncio.sleep(float(wait.group(1)))
                    continue
                if try_ == 1 and any(marker in text for marker in TRANSIENT):
                    meta.provider_errors.append(f"{model}: temporarily unavailable, waited {TRANSIENT_WAIT_S:g}s")
                    await asyncio.sleep(TRANSIENT_WAIT_S)
                    continue
                meta.provider_errors.append(f"{model}: {exc.__class__.__name__}: {text[:200]}")
                break
            meta.model, meta.fallback_used = model, i > 0
            return content
    raise LLMUnavailableError("All LLM providers failed: " + " | ".join(meta.provider_errors))


async def structured_call(
    messages: list[dict],
    model_cls: type[T],
    *,
    models: list[str],
    check: Callable[[T], None] | None = None,
    completion: Callable[..., Awaitable[Any]] | None = None,
    temperature: float = 0.0,
    timeout: float = 30.0,
) -> tuple[T, CallMeta]:
    if completion is None:
        import litellm

        litellm.suppress_debug_info = True
        completion = litellm.acompletion
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": model_cls.__name__, "strict": True, "schema": strict_json_schema(model_cls)},
    }
    meta = CallMeta()
    started = time.perf_counter()
    msgs = list(messages)
    try:
        for attempt in (1, 2):
            meta.attempts = attempt
            content = None
            try:
                content = await _complete(completion, msgs, models, response_format, meta, temperature, timeout)
                obj = model_cls.model_validate_json(content)
                if check:
                    check(obj)
                return obj, meta
            except ValueError as exc:  # pydantic ValidationError and _GenerationRejected are ValueErrors too
                errors = _describe(exc)
                meta.validation_errors.append(errors)
                if attempt == 2:
                    raise LLMValidationError(f"LLM output invalid after retry:\n{errors}") from exc
                previous = [{"role": "assistant", "content": content}] if content else []
                msgs = msgs + previous + [
                    {"role": "user", "content": f"That output was rejected:\n{errors}\nReturn corrected JSON only."},
                ]
        raise AssertionError("unreachable")
    except LLMError as exc:
        exc.meta = meta
        raise
    finally:
        meta.latency_ms = int((time.perf_counter() - started) * 1000)


async def transcribe(audio: bytes, filename: str, *, model: str, transcription: Callable[..., Awaitable[Any]] | None = None,
                     timeout: float = 30.0) -> str:
    """Speech → text for voice input (bonus). The text then goes through the normal chat turn."""
    if transcription is None:
        import litellm

        transcription = litellm.atranscription
    response = await transcription(model=model, file=(filename, audio), timeout=timeout)
    return (response.text or "").strip()
