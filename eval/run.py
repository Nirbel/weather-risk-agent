"""Run the evaluation set.

    uv run python -m eval.run              # offline: replayed real data, expected plans, template explainer
    uv run python -m eval.run --live       # real planner + explainer (Groq → Gemini) on the same replayed data
    uv run python -m eval.run --only A1,C1 # a subset

Data is always the recorded real API responses (tests/fixtures/recorded), served "as of" the
recording date, so expected numbers are stable. Exit code is non-zero if any case fails.
"""

import argparse
import asyncio
import gzip
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from weather_risk import replay
from weather_risk.agent.schemas import QueryPlan
from weather_risk.agent.service import Agent
from weather_risk.analysis import build_analyzer
from weather_risk.config import validate_config
from weather_risk.data import RateLimiter
from weather_risk.db import Database
from weather_risk.settings import get_settings, llm_configured

ROOT = Path(__file__).resolve().parents[1]
PLAN_FIELDS = ("intent", "hubs", "region", "hazards", "metric", "time_preset", "year", "top_k", "weight_overrides")


# -- plans -------------------------------------------------------------------------

def build_plan(expected: dict) -> QueryPlan:
    """Offline: turn a case's expected plan fields into a full QueryPlan."""
    intent = expected["intent"][0] if isinstance(expected["intent"], list) else expected["intent"]
    data = {f: None for f in QueryPlan.model_fields} | {"interpretation_notes": []}
    data |= {k: v for k, v in expected.items() if k != "intent"} | {"intent": intent}
    if data["weight_overrides"]:
        data["weight_overrides"] = {f: None for f in ("winter", "hurricane", "flood", "severe_storm", "heat")} \
            | data["weight_overrides"]
    if intent == "out_of_scope":
        data["out_of_scope_reason"] = "Not covered by this agent."
    if intent == "clarify":
        data["clarification_question"] = "Which hub do you mean?"
    return QueryPlan.model_validate(data)


def plan_matches(expected: dict, actual: dict | None) -> tuple[int, int, list[str]]:
    """(fields matched, fields expected, mismatch notes) — order-insensitive for lists."""
    if actual is None:
        return 0, len(expected), ["no plan (planner failed)"]
    ok, notes = 0, []
    for field, want in expected.items():
        got = actual.get(field)
        if field == "intent":
            good = got in (want if isinstance(want, list) else [want])
        elif field == "weight_overrides":
            good = bool(got) and all(got.get(k) == v for k, v in want.items())
        elif isinstance(want, list):
            good = sorted(got or []) == sorted(want)
        else:
            good = got == want
        ok += good
        if not good:
            notes.append(f"{field}: expected {want}, got {got}")
    return ok, len(expected), notes


# -- checks ------------------------------------------------------------------------

def raw_pct(hub: str, years: list[int], threshold: float) -> float:
    """Independent recomputation from the raw recorded Open-Meteo JSON."""
    values = []
    for year in years:
        with gzip.open(ROOT / "tests" / "fixtures" / "recorded" / "open_meteo" / f"{hub}_{year}.json.gz", "rt") as f:
            values += [v for v in json.load(f)["daily"]["snowfall_sum"] if v is not None]
    return round(100 * sum(v >= threshold for v in values) / len(values), 1)


def run_checks(checks: dict, answer: dict) -> list[str]:
    """Returns failure messages (empty = all checks passed)."""
    failures = []
    table_hubs = [row.get("hub") for row in answer["table"]]
    text = " ".join([answer["answer"], *answer["reasoning"]])
    for name, want in checks.items():
        match name:
            case "contract":
                missing = [k for k in ("assumptions", "sources", "time_windows") if not answer[k]]
                if not answer["uncertainty"]["reasons"]:
                    missing.append("uncertainty")
                if missing:
                    failures.append(f"contract: missing {', '.join(missing)}")
            case "top_hub" if not table_hubs or table_hubs[0] != want:
                failures.append(f"top_hub: expected {want}, got {table_hubs[:1]}")
            case "hubs_subset" if not set(table_hubs) <= set(want):
                failures.append(f"hubs_subset: unexpected {sorted(set(table_hubs) - set(want))}")
            case "hubs_include" if not set(want) <= set(table_hubs):
                failures.append(f"hubs_include: missing {sorted(set(want) - set(table_hubs))}")
            case "rows" if len(answer["table"]) != want:
                failures.append(f"rows: expected {want}, got {len(answer['table'])}")
            case "order" if not (want[0] in table_hubs and want[1] in table_hubs
                                 and table_hubs.index(want[0]) < table_hubs.index(want[1])):
                failures.append(f"order: expected {want[0]} above {want[1]}")
            case "headline_contains":
                failures += [f"headline_contains: {s!r} not in answer" for s in want if s not in text]
            case "uncertainty_mentions" if not any(want in r for r in answer["uncertainty"]["reasons"]):
                failures.append(f"uncertainty_mentions: {want!r} not found")
            case "assumption_mentions" if not any(want in a for a in answer["assumptions"]):
                failures.append(f"assumption_mentions: {want!r} not found")
            case "stat_matches_raw":
                expected = raw_pct(want["hub"], want["years"], want["threshold"])
                got = answer["table"][0]["% of days"] if answer["table"] else None
                if got != expected:
                    failures.append(f"stat_matches_raw: raw data gives {expected}%, answer gives {got}%")
            case "not_contains":
                failures += [f"not_contains: {s!r} appears" for s in want if s.lower() in text.lower()]
    return failures


# -- runner ------------------------------------------------------------------------

async def run_case(case: dict, agent: Agent, live: bool) -> dict:
    started = time.perf_counter()
    plan_ok = plan_total = 0
    plan_notes: list[str] = []
    conversation_id, answer = None, None
    for turn in case["turns"]:
        if live:
            result = await agent.ask(turn["question"], conversation_id, user_email="eval")
            answer = result.model_dump(mode="json")
            conversation_id = answer["conversation_id"]
            ok, total, notes = plan_matches(turn["expect_plan"], answer["query_plan"])
            plan_ok, plan_total, plan_notes = plan_ok + ok, plan_total + total, plan_notes + notes
        else:
            result = await agent.answer_plan(build_plan(turn["expect_plan"]), turn["question"], use_llm=False)
            answer = result.model_dump(mode="json")
    failures = run_checks(case.get("checks", {}), answer)
    expected_intent = case["turns"][-1]["expect_plan"]["intent"]
    if answer["intent"] not in (expected_intent if isinstance(expected_intent, list) else [expected_intent]):
        failures.append(f"intent: expected {expected_intent}, got {answer['intent']}")
    if live:
        failures += [f"plan — {n}" for n in plan_notes]
    return {
        "id": case["id"], "group": case["group"], "passed": not failures, "failures": failures,
        "plan_fields": [plan_ok, plan_total] if live else None,
        "explanation_source": answer["meta"].get("explanation_source"),
        "fallback_used": any((answer["meta"].get(k) or {}).get("fallback_used") for k in ("planner", "explainer")),
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "answer": answer["answer"],
    }


async def main(live: bool, only: set[str] | None, pause: float) -> int:
    validate_config()
    settings = get_settings()
    if live and not llm_configured(settings):
        raise SystemExit("--live needs GROQ_API_KEY and/or GEMINI_API_KEY in .env")
    cases = [c for c in yaml.safe_load((ROOT / "eval" / "cases.yaml").read_text()) if not only or c["id"] in only]
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.create_all()
    today = replay.recorded_today()
    results = []
    async with httpx.AsyncClient(transport=replay.transport()) as data_client:
        analyzer = build_analyzer(db, data_client, today_fn=lambda: today, limiter=RateLimiter(float("inf")))
        agent = Agent(db, analyzer, settings)
        for i, case in enumerate(cases):
            if live and i and pause:
                await asyncio.sleep(pause)  # Groq free tier: 8K tokens/min
            result = await run_case(case, agent, live)
            results.append(result)
            mark = "PASS" if result["passed"] else "FAIL"
            extra = f"  plan {result['plan_fields'][0]}/{result['plan_fields'][1]}" if live else ""
            print(f"{mark}  {case['id']:3} {case['group']:17} {result['latency_ms']:>6} ms"
                  f"  {result['explanation_source'] or '-':8}{extra}", flush=True)
            for failure in result["failures"]:
                print(f"        - {failure}", flush=True)
    await db.dispose()

    passed = sum(r["passed"] for r in results)
    summary: dict[str, Any] = {"mode": "live" if live else "offline", "cases": len(results), "passed": passed,
                               "pass_rate": round(passed / len(results), 3) if results else None}
    if live:
        ok = sum(r["plan_fields"][0] for r in results)
        total = sum(r["plan_fields"][1] for r in results)
        computed = [r for r in results if r["explanation_source"]]
        summary |= {
            "plan_field_accuracy": round(ok / total, 3) if total else None,
            "llm_explanations_grounded": sum(r["explanation_source"] == "llm" for r in computed),
            "template_fallbacks": sum(r["explanation_source"] == "template" for r in computed),
            "provider_fallbacks": sum(r["fallback_used"] for r in results),
            "latency_p50_ms": sorted(r["latency_ms"] for r in results)[len(results) // 2],
        }
    print("\n" + "  ".join(f"{k}={v}" for k, v in summary.items()))
    report = ROOT / "eval" / "reports" / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{summary['mode']}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False) + "\n")
    print(f"report: {report.relative_to(ROOT)}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="use the real LLM planner and explainer")
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--pause", type=float, default=8.0, help="seconds between live cases (rate limits)")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.live, set(args.only.split(",")) if args.only else None, args.pause)))
