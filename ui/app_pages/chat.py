"""Chat with the agent. The sidebar holds conversation history only."""

import html
import re

import pandas as pd
import streamlit as st

import api_client

LEVEL = {"low": ("green", ":material/check_circle:"), "medium": ("orange", ":material/info:"),
         "high": ("red", ":material/error:")}
SUGGESTIONS = {
    ":material/ac_unit: Midwest winter exposure": "Which hubs in the Midwest are most exposed to winter disruption?",
    ":material/flood: Miami vs Houston": "Compare Miami and Houston in terms of hurricane and flood exposure.",
    ":material/weather_snowy: Denver snow days": "What percentage of days in Denver last year had snowfall?",
    ":material/thunderstorm: Why Dallas?": "Why is the Dallas hub's weather disruption risk high?",
}


RTL = re.compile(r"[\u0590-\u05FF\u0600-\u06FF]")  # Hebrew and Arabic letters
# The chat's own labels; the answer's text comes from the API in its `language` (en or he).
UI = {
    "en": {"uncertainty": "Uncertainty", "low": "low", "medium": "medium", "high": "high",
           "context": "Assumptions, sources and time window", "assumptions": "Assumptions",
           "no_assumptions": "None beyond the computed result.", "window": "Time window", "not_applicable": "Not applicable",
           "sources": "Sources", "no_sources": "None (nothing computed)", "retrieved": "retrieved",
           "how": "How this was computed", "llm": "LLM, grounding-checked", "template": "deterministic template",
           "numbers": "Numbers: deterministic Python", "explanation": "explanation",
           "plan": "**Query plan** (LLM output, schema-validated)", "details": "**Result details**"},
    "he": {"uncertainty": "אי-ודאות", "low": "נמוכה", "medium": "בינונית", "high": "גבוהה",
           "context": "הנחות, מקורות וחלון זמן", "assumptions": "הנחות",
           "no_assumptions": "אין, מעבר לתוצאה המחושבת.", "window": "חלון זמן", "not_applicable": "לא רלוונטי",
           "sources": "מקורות", "no_sources": "אין (לא חושב דבר)", "retrieved": "נשלף",
           "how": "איך זה חושב", "llm": "LLM, עם בדיקת עיגון", "template": "תבנית דטרמיניסטית",
           "numbers": "מספרים: Python דטרמיניסטי", "explanation": "הסבר",
           "plan": "**תוכנית השאילתה** (פלט LLM, נבדק מול סכמה)", "details": "**פרטי התוצאה**"},
}


def write_text(text: str) -> None:
    """Markdown, or a right-to-left block when the answer is in Hebrew or Arabic."""
    if RTL.search(text):
        text = "• " + text[2:] if text.startswith("- ") else text
        st.html(f'<div dir="rtl" style="text-align: right">{html.escape(text)}</div>')
    else:
        st.markdown(text.replace("$", r"\$"))  # "$807 per $1M" must not render as LaTeX


def bullets(items: list[str], rtl: bool, title: str | None = None) -> None:
    """A list; right-to-left as one HTML block (items are escaped, except prepared links)."""
    if not rtl:
        if title:
            st.markdown(f"**{title}**")
        for item in items:
            st.markdown(f"- {item}".replace("$", r"\$"))
        return
    head = f"<b>{html.escape(title)}</b>" if title else ""
    st.html(f'<div dir="rtl" style="text-align: right">{head}<ul>'
            + "".join(f"<li>{item}</li>" for item in items) + "</ul></div>")


def start_new_chat() -> None:
    st.session_state.messages, st.session_state.conversation_id = [], None


def open_conversation(conversation_id: str) -> None:
    response = api_client.request("GET", f"/conversations/{conversation_id}")
    if response is None or response.status_code != 200:
        st.toast("Could not open that conversation.", icon=":material/error:")
        return
    messages = []
    for turn in response.json()["turns"]:
        messages += [{"role": "user", "content": turn["user_text"]}, {"role": "assistant", "answer": turn["answer"]}]
    st.session_state.messages, st.session_state.conversation_id = messages, conversation_id


def render_answer(answer: dict, key: str, latest: bool) -> None:
    lang = answer.get("language", "en")  # answers stored before i18n have no language
    t, rtl = UI.get(lang, UI["en"]), lang == "he"
    esc = html.escape if rtl else (lambda x: x)
    write_text(answer["answer"])
    for point in answer["reasoning"]:
        write_text(f"- {point}")
    if answer["table"]:
        st.dataframe(pd.DataFrame(answer["table"]).rename(columns=answer.get("column_labels") or {}), hide_index=True)

    level = answer["uncertainty"]["level"]
    color, icon = LEVEL[level]
    with st.expander(f"{t['uncertainty']}: {t[level]}", icon=icon, expanded=level == "high"):
        st.badge(t[level], color=color, icon=icon)
        bullets([esc(r) for r in answer["uncertainty"]["reasons"]], rtl)
    with st.expander(t["context"], icon=":material/fact_check:"):
        bullets([esc(a) for a in answer["assumptions"] or [t["no_assumptions"]]], rtl, t["assumptions"])
        windows = []
        for w in answer["time_windows"] or [{"label": t["not_applicable"], "start": None, "end": None}]:
            dates = f": {w['start'] or '…'} → {w['end'] or '…'}" if w["start"] or w["end"] else ""
            windows.append(esc(f"{w['label']}{dates}"))
        bullets(windows, rtl, t["window"])
        sources = []
        for s in answer["sources"] or [{"name": t["no_sources"], "url": "", "retrieved_at": None}]:
            name = (f'<a href="{html.escape(s["url"])}">{html.escape(s["name"])}</a>' if rtl else
                    f"[{s['name']}]({s['url']})") if s["url"] else esc(s["name"])
            detail = f" · {esc(s['detail'])}" if s.get("detail") else ""
            sources.append(f"{name} — {t['retrieved']} {esc(s['retrieved_at'] or 'n/a')}{detail}")
        bullets(sources, rtl, t["sources"])
    with st.expander(t["how"], icon=":material/function:"):
        meta = answer["meta"] or {}
        planner, explainer = meta.get("planner") or {}, meta.get("explainer") or {}
        explained_by = t["llm"] if meta.get("explanation_source") == "llm" else t["template"]
        st.caption(f"{t['numbers']} · {t['explanation']}: {explained_by} · planner: {planner.get('model')} · "
                   f"explainer: {explainer.get('model')} · {meta.get('latency_ms')} ms")
        st.markdown(t["plan"])
        st.json({k: v for k, v in (answer["query_plan"] or {}).items() if v not in (None, [])}, expanded=False)
        if answer["breakdown"]:
            st.markdown(t["details"])
            st.json(answer["breakdown"], expanded=False)
    if latest and answer["suggested_follow_ups"]:
        choice = st.pills("Follow up", answer["suggested_follow_ups"], key=f"follow-{key}",
                          label_visibility="collapsed")
        if choice:
            st.session_state.pending_prompt = choice
            st.rerun()


def ask(prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.status(":shimmer[Planning the query, computing, explaining]", type="compact") as status:
            response = api_client.request("POST", "/chat", json={
                "message": prompt, "conversation_id": st.session_state.conversation_id})
            body = response.json() if response is not None and "json" in response.headers.get("content-type", "") \
                else {}
            if response is None or "intent" not in body:  # transport error, not an AgentAnswer
                status.update(label="Request failed", state="error")
                detail = response.text[:300] if response is not None else "API unreachable"
                st.session_state.messages.append({"role": "error", "content": detail})
                return
            answer = body  # 200, or 422/503 with an error answer that is stored in the conversation too
            failed = response.status_code != 200
            status.update(label="Could not answer" if failed else
                          f"Answered in {answer['meta'].get('latency_ms', 0) / 1000:.1f} s",
                          state="error" if failed else "complete")
    st.session_state.conversation_id = answer["conversation_id"]
    st.session_state.messages.append({"role": "assistant", "answer": answer})
    st.rerun()


# -- sidebar: conversation history only ----------------------------------------------
with st.sidebar:
    st.button("New chat", icon=":material/add:", on_click=start_new_chat, width="stretch")
    st.caption("Previous conversations")
    response = api_client.request("GET", "/conversations")
    conversations = response.json() if response is not None and response.status_code == 200 else []
    for conv in conversations:
        current = conv["id"] == st.session_state.conversation_id
        st.button(conv["title"], key=f"conv-{conv['id']}", on_click=open_conversation, args=(conv["id"],),
                  icon=":material/chat_bubble:" if current else ":material/chat_bubble_outline:",
                  type="secondary" if current else "tertiary", width="stretch",
                  help=f"{conv['turns']} turn(s) · {conv['updated_at'][:16].replace('T', ' ')}")
    if not conversations:
        st.caption("No conversations yet.")

# -- conversation ---------------------------------------------------------------------
last_assistant = max((i for i, m in enumerate(st.session_state.messages) if m["role"] == "assistant"), default=-1)
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        st.chat_message("user").markdown(message["content"])
    elif message["role"] == "error":
        st.chat_message("assistant").error(message["content"], icon=":material/error:")
    else:
        with st.chat_message("assistant"):
            render_answer(message["answer"], key=str(i), latest=i == last_assistant)

if not st.session_state.messages:
    st.markdown("##### Ask about the long-term weather exposure of your hubs")
    picked = st.pills("Try asking", list(SUGGESTIONS), label_visibility="collapsed")
    if picked:
        st.session_state.pending_prompt = SUGGESTIONS[picked]
        st.rerun()

voice = bool((api_client.get("/health") or {}).get("voice"))
submitted = st.chat_input("Ask about hub weather exposure…", accept_audio=voice, submit_mode="disable",
                          key="chat_input")
prompt = st.session_state.pop("pending_prompt", None)
if submitted is not None:
    if voice and getattr(submitted, "audio", None) is not None:
        response = api_client.request("POST", "/transcribe",
                                      files={"audio": ("question.wav", submitted.audio.getvalue(), "audio/wav")})
        prompt = response.json().get("text") if response is not None and response.status_code == 200 else None
        if not prompt:
            st.toast("Could not transcribe the recording.", icon=":material/mic_off:")
    else:
        prompt = submitted.text if hasattr(submitted, "text") else submitted
if prompt:
    ask(prompt)
