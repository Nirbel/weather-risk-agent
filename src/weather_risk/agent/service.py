"""One conversational turn: plan → execute → explain → assemble → persist."""

import time
from dataclasses import asdict
from datetime import date

from weather_risk.agent.answer import assemble, error_answer
from weather_risk.agent.executor import ExecContext, execute
from weather_risk.agent.explainer import explain, template_explanation
from weather_risk.agent.llm import CallMeta, LLMError, LLMUnavailableError
from weather_risk.agent.planner import plan_question
from weather_risk.agent.schemas import AgentAnswer, QueryPlan
from weather_risk.db import Store
from weather_risk.hubs import load_hubs
from weather_risk.settings import Settings
from weather_risk.sources.open_meteo import archive_end

HISTORY_TURNS = 3


def _meta(call: CallMeta | None) -> dict | None:
    if call is None:
        return None
    return {k: v for k, v in asdict(call).items() if k in ("model", "fallback_used", "attempts", "latency_ms")}


class Agent:
    def __init__(self, store: Store, settings: Settings, *, completion=None, today: date | None = None):
        self.store = store
        self.settings = settings
        self.completion = completion
        self.fixed_today = today

    @property
    def today(self) -> date:
        return self.fixed_today or date.today()

    def _data_range(self) -> tuple[date, date]:
        ranges = [r for h in load_hubs() if (r := self.store.daily_range(h.id))]
        if not ranges:
            return date(2016, 1, 1), archive_end(self.today)
        return max(r[0] for r in ranges), min(min(r[1] for r in ranges), archive_end(self.today))

    async def ask(self, message: str, conversation_id: str | None = None, user_email: str | None = None) -> AgentAnswer:
        started = time.perf_counter()
        if not conversation_id or not self.store.conversation_exists(conversation_id):
            conversation_id = self.store.create_conversation(user_email)
        history = self.store.recent_turns(conversation_id, HISTORY_TURNS)
        data_start, data_end = self._data_range()
        try:
            plan, planner_meta = await plan_question(
                message, history, settings=self.settings, today=self.today,
                data_start=data_start, data_end=data_end, completion=self.completion,
            )
        except LLMError as exc:
            kind = "The language model is unavailable" if isinstance(exc, LLMUnavailableError) else \
                "The model's query plan failed validation twice"
            reason = f"{kind}: {str(exc).splitlines()[-1][:200]}"
            answer = error_answer(reason, conversation_id, {"planner": _meta(exc.meta),
                                                            "latency_ms": int((time.perf_counter() - started) * 1000)})
            self.store.add_turn(conversation_id, message, None, answer.model_dump(mode="json"))
            return answer
        answer = await self.answer_plan(plan, message, use_llm=True, conversation_id=conversation_id,
                                        planner_meta=planner_meta, started=started)
        self.store.add_turn(conversation_id, message, plan.model_dump(mode="json"), answer.model_dump(mode="json"))
        return answer

    async def answer_plan(self, plan: QueryPlan, question: str, *, use_llm: bool, conversation_id: str | None = None,
                          planner_meta: CallMeta | None = None, started: float | None = None) -> AgentAnswer:
        """Execute a plan and explain it. `use_llm=False` is the offline path (eval, LLM outage)."""
        started = started or time.perf_counter()
        bundle = execute(plan, ExecContext(store=self.store, today=self.today))
        explainer_meta, source = None, "template"
        if use_llm and bundle.rows:
            explanation, explainer_meta, source = await explain(question, bundle, self.settings, self.completion)
        else:
            explanation = template_explanation(bundle)
        meta = {
            "planner": _meta(planner_meta),
            "explainer": _meta(explainer_meta),
            "explanation_source": source,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
        return assemble(plan, bundle, explanation, self.store.source_statuses(), meta, conversation_id)
