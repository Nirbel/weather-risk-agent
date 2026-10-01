"""The only door to the LLM: structured output, local validation, one retry, provider fallback.

Two separate failure paths (docs/DECISIONS.md D3, D5):
  - provider failure (429/5xx/timeout/auth) → try the next model in `models`;
  - invalid output (schema or extra `check`) → one retry with the errors, then LLMValidationError.
"""

import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from weather_risk.agent.schemas import strict_json_schema

T = TypeVar("T", bound=BaseModel)


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


async def _complete(completion: Callable[..., Awaitable[Any]], messages: list[dict], models: list[str],
                    response_format: dict, meta: CallMeta, temperature: float, timeout: float) -> str:
    for i, model in enumerate(models):
        try:
            response = await completion(model=model, messages=messages, response_format=response_format,
                                        temperature=temperature, timeout=timeout)
            content = _content(response)
        except Exception as exc:  # any provider failure → next model
            meta.provider_errors.append(f"{model}: {exc.__class__.__name__}: {str(exc)[:200]}")
            continue
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
            content = await _complete(completion, msgs, models, response_format, meta, temperature, timeout)
            try:
                obj = model_cls.model_validate_json(content)
                if check:
                    check(obj)
                return obj, meta
            except ValueError as exc:  # pydantic ValidationError is a ValueError too
                errors = _describe(exc)
                meta.validation_errors.append(errors)
                if attempt == 2:
                    raise LLMValidationError(f"LLM output invalid after retry:\n{errors}") from exc
                msgs = msgs + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"That output was rejected:\n{errors}\nReturn corrected JSON only."},
                ]
        raise AssertionError("unreachable")
    except LLMError as exc:
        exc.meta = meta
        raise
    finally:
        meta.latency_ms = int((time.perf_counter() - started) * 1000)
