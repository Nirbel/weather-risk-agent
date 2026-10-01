"""One conversational turn: plan → execute → explain → assemble → persist."""

import time
from dataclasses import asdict

from weather_risk.agent.answer import assemble, error_answer
from weather_risk.agent.executor import ExecContext, execute
from weather_risk.agent.explainer import explain, template_explanation
from weather_risk.agent.llm import CallMeta, LLMError, LLMUnavailableError
from weather_risk.agent.planner import plan_question
from weather_risk.agent.schemas import AgentAnswer, QueryPlan
from weather_risk.db import Database
from weather_risk.settings import Settings

HISTORY_TURNS = 3


class ConversationAccessError(Exception):
    """The conversation belongs to another user."""


def _meta(call: CallMeta | None) -> dict | None:
    if call is None:
        return None
    meta = {k: v for k, v in asdict(call).items() if k in ("model", "fallback_used", "attempts", "latency_ms")}
    meta["validation_errors"] = [e[:300] for e in call.validation_errors]
    meta["provider_errors"] = [e[:200] for e in call.provider_errors]
    return meta


class Agent:
    def __init__(self, db: Database, analyzer, settings: Settings, *, completion=None):
        self.db, self.analyzer, self.settings, self.completion = db, analyzer, settings, completion

    async def _conversation(self, conversation_id: str | None, user_email: str | None, title: str) -> str:
        if conversation_id:
            owner = await self.db.conversation_owner(conversation_id)
            if owner is not None and owner != user_email:
                raise ConversationAccessError(conversation_id)
            if owner is not None:
                return conversation_id
        return await self.db.create_conversation(user_email, title)

    async def ask(self, message: str, conversation_id: str | None = None, user_email: str | None = None) -> AgentAnswer:
        started = time.perf_counter()
        conversation_id = await self._conversation(conversation_id, user_email, message)
        history = await self.db.recent_turns(conversation_id, HISTORY_TURNS)
        try:
            plan, planner_meta = await plan_question(
                message, history, settings=self.settings, today=self.analyzer.today,
                window=self.analyzer.window(), data_end=self.analyzer.data_end(), completion=self.completion,
            )
        except LLMError as exc:
            kind = ("The language model is unavailable" if isinstance(exc, LLMUnavailableError)
                    else "The model's query plan failed validation twice")
            reason = f"{kind}: {str(exc).splitlines()[-1][:200]}"
            answer = error_answer(reason, conversation_id, {"planner": _meta(exc.meta),
                                                            "latency_ms": int((time.perf_counter() - started) * 1000)})
            await self.db.add_turn(conversation_id, message, None, answer.model_dump(mode="json"))
            return answer
        previous = history[-1]["user_text"] if history else None
        answer = await self.answer_plan(plan, message, use_llm=True, conversation_id=conversation_id,
                                        planner_meta=planner_meta, started=started, previous_question=previous)
        await self.db.add_turn(conversation_id, message, plan.model_dump(mode="json"), answer.model_dump(mode="json"))
        return answer

    async def answer_plan(self, plan: QueryPlan, question: str, *, use_llm: bool, conversation_id: str | None = None,
                          planner_meta: CallMeta | None = None, started: float | None = None,
                          previous_question: str | None = None) -> AgentAnswer:
        """Execute a plan and explain it. `use_llm=False` is the offline path (eval, Analytics)."""
        started = started or time.perf_counter()
        bundle = await execute(plan, ExecContext(self.analyzer))
        explainer_meta, source = None, "template"
        if use_llm and bundle.rows:
            explanation, explainer_meta, source = await explain(question, bundle, self.settings, self.completion,
                                                                previous_question)
        else:
            explanation = template_explanation(bundle)
        meta = {"planner": _meta(planner_meta), "explainer": _meta(explainer_meta), "explanation_source": source,
                "latency_ms": int((time.perf_counter() - started) * 1000)}
        return assemble(plan, bundle, explanation, await self.db.source_statuses(), meta, conversation_id)
