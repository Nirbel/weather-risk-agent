"""Online end-to-end evaluation: the real application flow, nothing recorded or mocked.

    uv run python -m eval.run --online                          # fresh cache: downloads real data (~4 min)
    uv run python -m eval.run --online --cache data/eval.db     # keep the downloaded data between runs

Each turn goes through Agent.ask — real LLM planner → real Open-Meteo / FEMA NRI APIs →
deterministic computation → real LLM explanation → validation — exactly as /chat does.

Live public data can change, so no score or percentage is hard-coded. Instead, every answer is
compared with what the deterministic code produces *in the same run*:

  plan       the QueryPlan is schema-valid; intent, hubs, hazards and time period match the case
  numbers    the answer's table equals the code's result for the expected plan (ranking + values)
  own plan   the answer's table equals re-executing the plan the LLM chose (no number invented)
  grounding  every number in the explanation text appears in the computed result or the question
  window     the answer's time window equals the code's window for the expected plan
  contract   assumptions, sources, uncertainty and time window are present and match the code's
  follow-up  later turns resolve context from earlier ones (checked through their expected plans)

Fault cases then inject failures (LLM model down, all LLMs down, Open-Meteo unreachable,
FEMA NRI unreachable) and check that the answer degrades gracefully instead of crashing.
"""

import asyncio
import json
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import text

from eval.run import ROOT, build_plan, load_cases, plan_matches, resolve_expected, run_checks
from weather_risk.agent.executor import ExecContext, execute
from weather_risk.agent.grounding import allowed_numbers, ungrounded_numbers
from weather_risk.agent.schemas import QueryPlan
from weather_risk.agent.service import Agent
from weather_risk.analysis import build_analyzer
from weather_risk.config import validate_config
from weather_risk.db import Database
from weather_risk.settings import Settings, get_settings, llm_configured
from weather_risk.sources.http import make_client

# Checks in cases.yaml that assume the recorded data or the template wording (offline mode only).
DATA_DEPENDENT = {"top_hub", "order", "stat_matches_raw", "uncertainty_mentions", "headline_contains"}
FIELD_GROUPS = {"intent": ("intent",), "hubs": ("hubs", "region"), "hazards": ("hazards",),
                "time period": ("metric", "time_preset", "year")}
OPEN_METEO, ARCGIS = "archive-api.open-meteo.com", "services.arcgis.com"


class FailingHosts(httpx.AsyncBaseTransport):
    """Real network, except the listed hosts, which fail as if unreachable (fault injection)."""

    def __init__(self, hosts: set[str]):
        self.hosts, self.inner = hosts, httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.host in self.hosts:
            raise httpx.ConnectError(f"{request.url.host} unreachable (fault injected by eval)", request=request)
        return await self.inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self.inner.aclose()


# -- per-turn validation ------------------------------------------------------------------

def _names(answer: dict) -> list[str]:
    return [s["name"] for s in answer["sources"]]


async def check_turn(agent: Agent, question: str, previous: str | None, answer: dict, expected: dict) -> dict:
    """All online checks for one turn; each value is True, False, or None (not applicable)."""
    checks: dict[str, bool | None] = {}
    notes: list[str] = []
    plan_dict, plan = answer["query_plan"], None
    try:
        plan = QueryPlan.model_validate(plan_dict) if plan_dict else None
    except ValueError as exc:
        notes.append(f"plan invalid: {exc}")
    checks["plan valid"] = plan is not None
    if plan_dict is None:
        notes.append(f"no plan — {answer['answer'][:120]}")

    _, _, mismatches = plan_matches(expected, plan_dict)
    notes += [f"plan — {m}" for m in mismatches]
    for group, fields in FIELD_GROUPS.items():
        asked = [f for f in fields if f in expected]
        checks[group] = None if not asked else plan_dict is not None and not any(
            m.split(":")[0] in asked for m in mismatches)
    checks["plan fields"] = not mismatches

    reference = (await agent.answer_plan(build_plan(expected), question, use_llm=False)).model_dump(mode="json")
    computes = bool(reference["sources"])  # rank / compare / stat / explain (not clarify / out of scope)
    checks["numbers"] = answer["table"] == reference["table"]
    if not checks["numbers"]:
        notes.append(f"numbers: answer table {answer['table'][:2]} ≠ code {reference['table'][:2]}")
    checks["window"] = answer["time_windows"] == reference["time_windows"] if computes else None
    if checks["window"] is False:
        notes.append(f"window: {answer['time_windows']} ≠ code {reference['time_windows']}")

    if plan is not None:
        own = (await agent.answer_plan(plan, question, use_llm=False)).model_dump(mode="json")
        checks["own plan"] = answer["table"] == own["table"]
        bundle = await execute(plan, ExecContext(agent.analyzer))
        allowed = allowed_numbers(bundle.for_llm(), question) | allowed_numbers(None, previous or "")
        bad = ungrounded_numbers(" ".join([answer["answer"], *answer["reasoning"]]), allowed)
        checks["grounding"] = not bad
        if bad:
            notes.append(f"grounding: numbers not in the result: {bad}")
    else:
        checks["own plan"] = checks["grounding"] = False

    if computes:
        missing = [k for k in ("assumptions", "sources", "time_windows") if not answer[k]]
        missing += [] if answer["uncertainty"]["reasons"] else ["uncertainty"]
        missing += [f"assumption from code: {a[:60]}…" for a in reference["assumptions"]
                    if a not in answer["assumptions"]]
        missing += [] if _names(answer) == _names(reference) else [f"sources {_names(answer)} ≠ {_names(reference)}"]
        missing += [] if answer["uncertainty"]["level"] == reference["uncertainty"]["level"] else [
            f"uncertainty {answer['uncertainty']['level']} ≠ code {reference['uncertainty']['level']}"]
        checks["contract"] = not missing
        notes += [f"contract: {m}" for m in missing]
    else:
        checks["contract"] = None
    return {"checks": checks, "notes": notes}


async def run_case(case: dict, agent: Agent, pause: float) -> dict:
    started = time.perf_counter()
    conversation_id, previous, turns, failures = None, None, [], []
    answer: dict = {}
    for i, turn in enumerate(case["turns"]):
        if i and pause:
            await asyncio.sleep(pause)  # pace every LLM turn (Groq free tier: 8K tokens/min)
        expected = resolve_expected(turn["expect_plan"], agent.analyzer.today)
        answer = (await agent.ask(turn["question"], conversation_id, user_email="eval-online")).model_dump(mode="json")
        conversation_id = answer["conversation_id"]
        result = await check_turn(agent, turn["question"], previous, answer, expected)
        result |= {"question": turn["question"], "follow_up": i > 0, "answer": answer["answer"],
                   "explanation_source": answer["meta"].get("explanation_source"),
                   "fallback_used": any((answer["meta"].get(k) or {}).get("fallback_used")
                                        for k in ("planner", "explainer"))}
        turns.append(result)
        failed = [name for name, ok in result["checks"].items() if ok is False]
        if failed:
            failures.append(f"turn {i + 1} failed: {', '.join(failed)}")
            failures += [f"turn {i + 1}: {note}" for note in result["notes"]]
        previous = turn["question"]
    portable = {k: v for k, v in case.get("checks", {}).items() if k not in DATA_DEPENDENT}
    failures += run_checks(portable, answer)
    return {"id": case["id"], "group": case["group"], "passed": not failures, "failures": failures, "turns": turns,
            "latency_ms": int((time.perf_counter() - started) * 1000)}


# -- fault cases ---------------------------------------------------------------------------

async def fault_cases(db: Database, client: httpx.AsyncClient, settings: Settings,
                      only: set[str] | None = None) -> list[dict]:
    """Inject failures and check graceful degradation (no crash, honest answer, high uncertainty)."""
    results = []

    def want(cid: str) -> bool:
        return not only or cid in only

    async def case(cid: str, name: str, question: str, agent: Agent, verify) -> None:
        started = time.perf_counter()
        skipped = None
        try:
            answer = (await agent.ask(question, None, user_email="eval-online")).model_dump(mode="json")
            failures = verify(answer)
            if cid == "X1" and failures:
                # The fault forces the fallback; if the fallback provider is itself out (quota, overload),
                # the run can't tell whether fallback works — say so instead of reporting a code failure.
                errors = (answer["meta"].get("planner") or {}).get("provider_errors", [])
                fallback_errors = [e for e in errors if e.startswith(settings.llm_fallback_model)]
                if fallback_errors:
                    skipped, failures = f"inconclusive — fallback provider unavailable: {fallback_errors[-1][:140]}", []
        except Exception as exc:  # the point of these cases: nothing may raise
            answer, failures = {}, [f"raised {exc.__class__.__name__}: {exc}"]
        results.append({"id": cid, "group": "Failures", "fault": name, "passed": not failures, "failures": failures,
                        "skipped": skipped, "answer": answer.get("answer"),
                        "latency_ms": int((time.perf_counter() - started) * 1000)})

    def expect(*conditions: tuple[bool, str]) -> list[str]:
        return [message for ok, message in conditions if not ok]

    analyzer = build_analyzer(db, client)
    compare = "Compare Miami and Houston in terms of hurricane and flood exposure."

    if want("X1"):  # primary LLM down: the fallback provider answers; numbers unchanged
        if settings.gemini_api_key or not settings.llm_fallback_model.startswith("gemini/"):
            broken = settings.model_copy(update={"llm_model": "groq/eval-fault-no-such-model"})
            reference = (await Agent(db, analyzer, settings).answer_plan(
                build_plan({"intent": "compare", "hubs": ["miami", "houston"], "hazards": ["hurricane", "flood"]}),
                compare, use_llm=False)).model_dump(mode="json")
            await case("X1", "primary LLM model unavailable", compare, Agent(db, analyzer, broken), lambda a: expect(
                (a["intent"] == "compare", f"intent {a['intent']}"),
                ((a["meta"].get("planner") or {}).get("fallback_used") is True, "planner did not fall back"),
                (a["table"] == reference["table"], "numbers differ from the code's result")))
        else:
            results.append({"id": "X1", "group": "Failures", "fault": "primary LLM model unavailable", "passed": True,
                            "skipped": "no GEMINI_API_KEY for the fallback", "failures": [], "latency_ms": 0})
    if want("X2"):  # every LLM down: a clear error answer, no crash, no invented numbers
        dead = settings.model_copy(update={"llm_model": "groq/eval-fault-no-such-model",
                                           "llm_fallback_model": "gemini/eval-fault-no-such-model"})
        await case("X2", "all LLM models unavailable", compare, Agent(db, analyzer, dead), lambda a: expect(
            (a["intent"] == "error", f"intent {a['intent']}"), (a["table"] == [], "a table was returned"),
            (a["uncertainty"]["level"] == "high", "uncertainty not high"), (bool(a["reasoning"]), "no reason given")))
    if want("X3"):
        await _open_meteo_down(case, expect, settings)
    if want("X4"):
        await _nri_down(case, expect, db, settings, compare)
    return results


async def _open_meteo_down(case, expect, settings: Settings) -> None:
    """X3 — Open-Meteo unreachable (fresh cache): nothing ranked as if complete; the gap is reported."""
    tmp = Path(tempfile.mkdtemp(prefix="eval-fault-"))
    fresh = Database(f"sqlite+aiosqlite:///{tmp / 'fault.db'}")
    await fresh.create_all()
    async with make_client(settings.contact_email, transport=FailingHosts({OPEN_METEO})) as down:
        await case("X3", "Open-Meteo unreachable", "Which hubs in the Midwest are most exposed to winter disruption?",
                   Agent(fresh, build_analyzer(fresh, down), settings), lambda a: expect(
                       (a["intent"] == "rank", f"intent {a['intent']}"),
                       (bool(a["table"]) and all(r.get("rank") is None and r.get("score") is None for r in a["table"]),
                        "a hub was ranked without weather data"),
                       (a["uncertainty"]["level"] == "high", "uncertainty not high"),
                       (any("Open-Meteo" in r for r in a["uncertainty"]["reasons"]), "the outage is not reported")))
    await fresh.dispose()
    shutil.rmtree(tmp, ignore_errors=True)


async def _nri_down(case, expect, db: Database, settings: Settings, compare: str) -> None:
    """X4 — FEMA NRI unreachable (weather cached): hurricane can't be scored, so no combined comparison."""
    async with db.engine.begin() as conn:
        await conn.execute(text("DELETE FROM http_cache WHERE key = 'nri'"))
    async with make_client(settings.contact_email, transport=FailingHosts({ARCGIS})) as down:
        await case("X4", "FEMA NRI unreachable", compare, Agent(db, build_analyzer(db, down), settings),
                   lambda a: expect(
                       (a["intent"] == "compare", f"intent {a['intent']}"),
                       (len(a["table"]) == 2 and all(r.get("Hurricane / tropical storm") is None
                                                     and r.get("combined") is None for r in a["table"]),
                        "hurricane or a combined score was computed without NRI"),
                       (a["uncertainty"]["level"] == "high", "uncertainty not high"),
                       (any("National Risk Index" in r for r in a["uncertainty"]["reasons"]),
                        "the outage is not reported")))


# -- runner --------------------------------------------------------------------------------

def _rate(results: list[dict], check: str, follow_up_only: bool = False) -> float | None:
    values = [t["checks"][check] for r in results for t in r.get("turns", [])
              if t["checks"].get(check) is not None and (t["follow_up"] or not follow_up_only)]
    return round(sum(values) / len(values), 3) if values else None


async def main(only: set[str] | None, pause: float, cache: str | None) -> int:
    validate_config()
    settings = get_settings()
    if not llm_configured(settings):
        raise SystemExit("--online needs GROQ_API_KEY and/or GEMINI_API_KEY in .env")
    cases = [c for c in load_cases() if not only or c["id"] in only]
    tmp = None
    if cache:
        path = Path(cache).resolve()
    else:
        tmp = Path(tempfile.mkdtemp(prefix="eval-online-"))
        path = tmp / "eval.db"
    db = Database(f"sqlite+aiosqlite:///{path}")
    await db.create_all()
    results: list[dict] = []
    fetch_s = None
    try:
        async with make_client(settings.contact_email) as client:
            analyzer = build_analyzer(db, client)
            agent = Agent(db, analyzer, settings)
            today = analyzer.today
            start, end = analyzer.window()
            print(f"Online eval · today {today} · exposure window {start}–{end} · cache {path}", flush=True)
            print("Fetching live data (Open-Meteo ERA5 for 10 hubs × 5 years, FEMA NRI); a cold cache takes "
                  "about 3–4 minutes on the free tier…", flush=True)
            t0 = time.perf_counter()
            exposure = await analyzer.exposure()
            fetch_s = round(time.perf_counter() - t0, 1)
            incomplete = [h for h, bd in exposure.breakdowns.items() if not bd["meta"]["observed_window"]["complete"]]
            print(f"Data ready in {fetch_s} s · NRI {exposure.nri_version} · incomplete weather: "
                  f"{', '.join(incomplete) or 'none'}", flush=True)
            for h in incomplete:  # e.g. a download that failed; the next request retries the missing year
                for gap in exposure.breakdowns[h]["data_gaps"]:
                    print(f"  {h}: {gap}", flush=True)
            print(flush=True)

            for i, case in enumerate(c for c in cases if not c["id"].startswith("X")):
                if i and pause:
                    await asyncio.sleep(pause)  # Groq free tier: 8K tokens/min
                result = await run_case(case, agent, pause)
                results.append(result)
                explained_by = ",".join(sorted({t["explanation_source"] or "-" for t in result["turns"]}))
                print(f"{'PASS' if result['passed'] else 'FAIL'}  {case['id']:3} {case['group']:17} "
                      f"{result['latency_ms']:>6} ms  {explained_by:8}  turns {len(result['turns'])}", flush=True)
                for failure in result["failures"]:
                    print(f"        - {failure}", flush=True)
            if not only or any(cid.startswith("X") for cid in only):
                print("\nFault injection:", flush=True)
                for result in await fault_cases(db, client, settings, only):
                    results.append(result)
                    status = "SKIP" if result.get("skipped") else "PASS" if result["passed"] else "FAIL"
                    print(f"{status}  {result['id']:3} {result['fault']:32} {result['latency_ms']:>6} ms"
                          f"  {result.get('skipped') or (result['answer'] or '')[:90]}", flush=True)
                    for failure in result["failures"]:
                        print(f"        - {failure}", flush=True)
    finally:
        await db.dispose()
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)

    flow = [r for r in results if "turns" in r]
    turns = [t for r in flow for t in r["turns"]]
    passed = sum(r["passed"] for r in results)
    summary: dict[str, Any] = {
        "mode": "online", "today": today.isoformat(), "data_fetch_s": fetch_s,
        "cases": len(results), "passed": passed, "pass_rate": round(passed / len(results), 3) if results else None,
        "plan_valid": _rate(flow, "plan valid"), "intent": _rate(flow, "intent"), "hubs": _rate(flow, "hubs"),
        "hazards": _rate(flow, "hazards"), "time_period": _rate(flow, "time period"),
        "follow_ups": _rate(flow, "plan fields", follow_up_only=True), "numbers_match_code": _rate(flow, "numbers"),
        "numbers_match_own_plan": _rate(flow, "own plan"), "grounding": _rate(flow, "grounding"),
        "window": _rate(flow, "window"), "contract": _rate(flow, "contract"),
        "llm_explanations": sum(t["explanation_source"] == "llm" for t in turns),
        "template_explanations": sum(t["explanation_source"] == "template" for t in turns),
        "provider_fallbacks": sum(bool(t["fallback_used"]) for t in turns),
        "faults_passed": f"{sum(r['passed'] and not r.get('skipped') for r in results if 'fault' in r)}/"
                         f"{sum('fault' in r for r in results)}",
        "faults_inconclusive": sum(bool(r.get("skipped")) for r in results if "fault" in r),
        "latency_p50_ms": sorted(r["latency_ms"] for r in flow)[len(flow) // 2] if flow else None,
    }
    print("\n" + "  ".join(f"{k}={v}" for k, v in summary.items()))
    report = ROOT / "eval" / "reports" / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_online.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False) + "\n")
    print(f"report: {report.relative_to(ROOT)}")
    return 0 if passed == len(results) else 1
