"""Chat with the agent. The sidebar holds conversation history only."""

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
    st.markdown(answer["answer"])
    for point in answer["reasoning"]:
        st.markdown(f"- {point}")
    if answer["table"]:
        st.dataframe(pd.DataFrame(answer["table"]), hide_index=True)

    level = answer["uncertainty"]["level"]
    color, icon = LEVEL[level]
    with st.expander(f"Uncertainty: {level}", icon=icon, expanded=level == "high"):
        st.badge(level, color=color, icon=icon)
        for reason in answer["uncertainty"]["reasons"]:
            st.markdown(f"- {reason}")
    with st.expander("Assumptions, sources and time window", icon=":material/fact_check:"):
        st.markdown("**Assumptions**")
        for a in answer["assumptions"] or ["None beyond the computed result."]:
            st.markdown(f"- {a}")
        st.markdown("**Time window**")
        for w in answer["time_windows"] or [{"label": "Not applicable", "start": None, "end": None}]:
            dates = f": {w['start'] or '…'} → {w['end'] or '…'}" if w["start"] or w["end"] else ""
            st.markdown(f"- {w['label']}{dates}")
        st.markdown("**Sources**")
        for s in answer["sources"] or [{"name": "None (nothing computed)", "url": "", "retrieved_at": None}]:
            link = f"[{s['name']}]({s['url']})" if s["url"] else s["name"]
            st.markdown(f"- {link} — retrieved {s['retrieved_at'] or 'n/a'}" + (f" · {s['detail']}" if s.get("detail") else ""))
    with st.expander("How this was computed", icon=":material/function:"):
        meta = answer["meta"] or {}
        planner, explainer = meta.get("planner") or {}, meta.get("explainer") or {}
        explained_by = "LLM, grounding-checked" if meta.get("explanation_source") == "llm" else "deterministic template"
        st.caption(f"Numbers: deterministic Python · explanation: {explained_by} · planner: {planner.get('model')} · "
                   f"explainer: {explainer.get('model')} · {meta.get('latency_ms')} ms")
        st.markdown("**Query plan** (LLM output, schema-validated)")
        st.json({k: v for k, v in (answer["query_plan"] or {}).items() if v not in (None, [])}, expanded=False)
        if answer["breakdown"]:
            st.markdown("**Result details**")
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
            if response is None or response.status_code != 200:
                status.update(label="Request failed", state="error")
                detail = response.text[:300] if response is not None else "API unreachable"
                st.session_state.messages.append({"role": "error", "content": detail})
                return
            answer = response.json()
            status.update(label=f"Answered in {answer['meta'].get('latency_ms', 0) / 1000:.1f} s", state="complete")
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
