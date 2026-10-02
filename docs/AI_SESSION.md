# Working with the AI agent

The project was built with **Claude Code** (desktop app, Claude Opus) in one long session that was resumed
several times. The full transcript is the deliverable the task asks for; this page explains how the work
went and where to find it.

## Where the transcript is

- **[docs/session/](session/)** holds the exported session (Claude desktop → session menu → *Export*):
  the conversation, every tool call and its result, and the plan.
- The plan, the decision log and the commit history are the condensed record:
  [PLAN.md](PLAN.md) · [DECISIONS.md](DECISIONS.md) · `git log`.

## How the work was done

1. **Plan before code.** The agent read the task, then checked every candidate data source with live,
   read-only API calls before choosing one (PLAN.md §0). That is how it found, for example, that the FEMA
   NRI headline score cannot rank metro hubs (D7) and that the "snow day" threshold swings an answer from
   2.7 % to 12.1 % (D13).
2. **Agreed rules in [CLAUDE.md](../CLAUDE.md).** Numbers come from code, all LLM output is
   schema-validated, real data only, every answer states assumptions, uncertainty, sources and window, and
   pragmatic TDD for core logic.
3. **Milestones, each ending with a green suite and a commit:** data and scoring (M1–M2), agent core (M3),
   API and UI (M4), eval (M5), auth (M6), docs (M7), bonuses: alerts and voice (M8).
4. **Test-first for core logic** with small hand-computed fixtures. Expected values are worked out in
   comments, never copied from the implementation (scoring, stats, date windows, plan validation, grounding,
   executor, answer contract, alert diff, coverage rule, email and password rules).

## Where the human steered

| When | What the user decided | Effect |
|---|---|---|
| Planning | Equal-weight overall score, Open-Meteo only, Groq + Gemini, one run script | D5, D12, D15 |
| Mid-build | Async SQLAlchemy instead of stdlib `sqlite3` | The agent checked it wasn't materially worse, then switched (D14) |
| Mid-build | Narrow the scope to one day: 10 hubs, 2 sources, 5 years on demand | D21 |
| After M8 | Run with the script, not Docker, on this machine | D15 |
| Review round | Email + password auth with `@moveo.co.il` sign-up, per-user webhooks, a coverage rule, a real online eval, clean-clone startup, a two-service Compose file | D18, D22, D24, D25, D15 |

## Things the process caught

- **Calibration on real data** (D20): the first real run ranked Kansas City above Minneapolis on winter.
  The cause was measured in the data, not tuned toward an expected ranking.
- **Clean-clone failure**: the default SQLite path's folder did not exist on a fresh checkout. Fixed with
  a regression test (D15).
- **Security review of a commit** flagged the demo admin password (kept as the task requires, but now
  configurable, rotatable and warned about) and SSRF through user-chosen webhook URLs (destination check
  added, D22).
- **The first online eval run** (D25) found a planner that dropped a named hazard, a follow-up planned with
  the wrong intent, and a comparison that silently dropped a hazard whose only source was down. All three
  were fixed and re-verified live. It also caught a real download failure: a Dallas year was missing, so
  the coverage rule kept Dallas out of the ranking until the year was re-fetched.
