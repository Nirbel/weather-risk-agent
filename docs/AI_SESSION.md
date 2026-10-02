# Working with the AI agent

The project was built with **Claude Code** (desktop app, Claude Opus) across several development sessions.
The full transcript is the deliverable the task asks for; this page gives a short overview of the process
and points to the complete conversation.

## Full development transcript

The complete Claude Code development transcript is available here:

**[Full AI development session](session/full-session.md)**

The transcript combines the Claude Code sessions used to plan, implement, test and refine the project.
It was generated directly from the local Claude Code session history and is not a summary.

For a condensed view of the engineering process, see:
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
| After M8 | Keep the script as the simplest local run path; later add two-service Docker Compose for deployment | D15 |
| Review round | Email + password auth with `@moveo.co.il` sign-up, per-user webhooks, a coverage rule, a real online eval, clean-clone startup, a two-service Compose file | D18, D22, D24, D25, D15 |

## Things the process caught

- **Calibration on real data** (D20): the first real run ranked Kansas City above Minneapolis on winter.
  The cause was measured in the data, not tuned toward an expected ranking.
- **Clean-clone failure**: the default SQLite path's folder did not exist on a fresh checkout. Fixed with
  a regression test (D15).
- **Security review of a commit** flagged the documented demo admin credential and SSRF through
  user-chosen webhook URLs. The demo credential was kept intentionally for reviewer access but made
  configurable and rotatable; webhook destination checks were added (D22).
- **The first online eval run** (D25) found a planner that dropped a named hazard, a follow-up planned with
  the wrong intent, and a comparison that silently dropped a hazard whose only source was down. All three
  were fixed and re-verified live. It also caught a real download failure: a Dallas year was missing, so
  the coverage rule kept Dallas out of the ranking until the year was re-fetched.
