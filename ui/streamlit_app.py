"""Weather Risk Intelligence Agent — Streamlit UI (talks to the FastAPI backend over HTTP).

    uv run python scripts/run.py        (starts API + UI together)
"""

import streamlit as st

import api_client

st.set_page_config(page_title="Weather risk agent", page_icon=":material/thunderstorm:", layout="wide")
st.session_state.setdefault("messages", [])
st.session_state.setdefault("conversation_id", None)


def login_page() -> None:
    st.title("Weather risk intelligence agent", anchor=False)
    st.caption("Which distribution hubs are most exposed to weather disruption — and why.")
    with st.form("login", width=420):
        email = st.text_input("Work email", placeholder="you@company.com",
                              help="Access is limited to an allow-list of emails.")
        if st.form_submit_button("Log in", icon=":material/login:", type="primary") and email:
            response = api_client.request("POST", "/auth/login", json={"email": email})
            if response is not None and response.status_code == 200:
                body = response.json()
                st.session_state.update(token=body["token"], email=body["email"], role=body["role"])
                st.rerun()
            elif response is not None:
                st.error(response.json().get("detail", response.text), icon=":material/block:")


if not api_client.token():
    st.navigation([st.Page(login_page, title="Log in", icon=":material/login:")], position="hidden").run()
    st.stop()

home = st.Page("app_pages/home.py", title="Home", icon=":material/home:", default=True)
chat = st.Page("app_pages/chat.py", title="Chat", icon=":material/chat:")
analytics = st.Page("app_pages/analytics.py", title="Analytics", icon=":material/bar_chart:")
st.session_state.pages = {"home": home, "chat": chat, "analytics": analytics}
page = st.navigation([home, chat, analytics], position="top")

with st.container(horizontal=True, horizontal_alignment="right"):
    with st.popover(st.session_state.email, icon=":material/account_circle:", type="tertiary"):
        st.caption(f"Role: {st.session_state.role}")
        if st.button("Log out", icon=":material/logout:"):
            st.session_state.clear()
            st.rerun()

page.run()
