"""LLM wrapper: validation retry, provider fallback, clear errors. LiteLLM is mocked."""

import pytest
from pydantic import BaseModel, ConfigDict

from conftest import FakeCompletion, llm_response as response
from weather_risk.agent.llm import LLMUnavailableError, LLMValidationError, structured_call


class Thing(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    size: int


MODELS = ["groq/openai/gpt-oss-20b", "gemini/gemini-3.8-flash"]
MSGS = [{"role": "user", "content": "go"}]


async def test_valid_output_first_try():
    fake = FakeCompletion('{"name": "a", "size": 1}')
    obj, meta = await structured_call(MSGS, Thing, models=MODELS, completion=fake)
    assert obj == Thing(name="a", size=1)
    assert (meta.model, meta.attempts, meta.fallback_used) == (MODELS[0], 1, False)
    sent = fake.calls[0]["response_format"]
    assert sent["type"] == "json_schema" and sent["json_schema"]["strict"] is True


async def test_invalid_output_is_retried_once_with_the_error():
    fake = FakeCompletion('{"name": "a"}', '{"name": "a", "size": 2}')
    obj, meta = await structured_call(MSGS, Thing, models=MODELS, completion=fake)
    assert obj.size == 2 and meta.attempts == 2
    retry_messages = fake.calls[1]["messages"]
    assert retry_messages[-2]["role"] == "assistant"
    assert "size" in retry_messages[-1]["content"] and "rejected" in retry_messages[-1]["content"]


async def test_invalid_twice_raises_clear_error():
    fake = FakeCompletion('{"name": "a"}', "not json")
    with pytest.raises(LLMValidationError) as exc:
        await structured_call(MSGS, Thing, models=MODELS, completion=fake)
    assert "size" in str(exc.value) or "JSON" in str(exc.value)


async def test_extra_check_failure_triggers_retry():
    def check(obj: Thing) -> None:
        if obj.size > 10:
            raise ValueError("size must come from the data")

    fake = FakeCompletion('{"name": "a", "size": 99}', '{"name": "a", "size": 3}')
    obj, meta = await structured_call(MSGS, Thing, models=MODELS, completion=fake, check=check)
    assert obj.size == 3
    assert "size must come from the data" in fake.calls[1]["messages"][-1]["content"]


async def test_provider_failure_falls_back_to_second_model():
    fake = FakeCompletion(RuntimeError("429 rate limited"), '{"name": "b", "size": 1}')
    obj, meta = await structured_call(MSGS, Thing, models=MODELS, completion=fake)
    assert meta.model == MODELS[1] and meta.fallback_used
    assert fake.calls[1]["model"] == MODELS[1]


async def test_all_providers_down_raises_unavailable():
    fake = FakeCompletion(RuntimeError("down"), RuntimeError("also down"))
    with pytest.raises(LLMUnavailableError):
        await structured_call(MSGS, Thing, models=MODELS, completion=fake)


async def test_code_fences_and_tool_call_payloads_are_accepted():
    fenced = FakeCompletion('```json\n{"name": "a", "size": 1}\n```')
    assert (await structured_call(MSGS, Thing, models=MODELS, completion=fenced))[0].size == 1
    tool = FakeCompletion(response(content=None, tool_args='{"name": "t", "size": 5}'))
    assert (await structured_call(MSGS, Thing, models=MODELS, completion=tool))[0].size == 5


async def test_provider_schema_rejection_uses_the_validation_retry():
    # Groq strict mode can return HTTP 400 "Generated JSON does not match the expected schema"
    rejected = RuntimeError('GroqException - {"error":{"message":"Generated JSON does not match the expected schema",'
                            '"code":"json_validate_failed"}}')
    fake = FakeCompletion(rejected, '{"name": "a", "size": 1}')
    obj, meta = await structured_call(MSGS, Thing, models=MODELS, completion=fake)
    assert obj.size == 1 and meta.attempts == 2
    assert fake.calls[1]["model"] == MODELS[0]  # retried on the primary, not treated as an outage
    assert "schema" in meta.validation_errors[0]


async def test_short_rate_limit_is_waited_out_on_the_same_model():
    limited = RuntimeError("Rate limit reached for model on tokens per minute (TPM). Please try again in 0.01s.")
    fake = FakeCompletion(limited, '{"name": "a", "size": 1}')
    obj, meta = await structured_call(MSGS, Thing, models=MODELS, completion=fake)
    assert meta.model == MODELS[0] and not meta.fallback_used
    assert "rate limited" in meta.provider_errors[0]


async def test_long_rate_limit_falls_back():
    limited = RuntimeError("Rate limit reached. Please try again in 45.5s.")
    fake = FakeCompletion(limited, '{"name": "a", "size": 1}')
    _, meta = await structured_call(MSGS, Thing, models=MODELS, completion=fake)
    assert meta.model == MODELS[1] and meta.fallback_used


async def test_gemini_calls_omit_temperature():
    fake = FakeCompletion(RuntimeError("down"), '{"name": "a", "size": 1}')
    await structured_call(MSGS, Thing, models=MODELS, completion=fake, temperature=0.2)
    assert fake.calls[0]["temperature"] == 0.2
    assert "temperature" not in fake.calls[1]
