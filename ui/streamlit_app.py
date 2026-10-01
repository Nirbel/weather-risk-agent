"""Weather Risk Intelligence Agent — Streamlit UI (talks to the FastAPI backend over HTTP).

    uv run python scripts/run.py        (starts API + UI together)
"""

import streamlit as st

import api_client

st.set_page_config(page_title="Weather risk agent", page_icon=":material/thunderstorm:", layout="wide")
st.session_state.setdefault("messages", [])
st.session_state.setdefault("conversation_id", None)


def _start_session(response) -> None:
    if response is None:
        return
    if response.status_code in (200, 201):
        body = response.json()
        st.session_state.update(token=body["token"], email=body["email"], role=body["role"])
        st.rerun()
    detail = response.json().get("detail", response.text)
    st.error(detail if isinstance(detail, str) else "Check the email and password.", icon=":material/block:")


def login_page() -> None:
    st.title("Weather risk intelligence agent", anchor=False)
    st.caption("Which distribution hubs are most exposed to weather disruption — and why.")
    with st.container(width=440):
        log_in, sign_up = st.tabs(["Log in", "Sign up"])
        with log_in, st.form("login"):
            email = st.text_input("Email", placeholder="you@moveo.co.il")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Log in", icon=":material/login:", type="primary") and email and password:
                _start_session(api_client.request("POST", "/auth/login", json={"email": email, "password": password}))
        with sign_up, st.form("signup"):
            st.caption("Open to @moveo.co.il addresses. New accounts get the analyst role.")
            email = st.text_input("Work email", placeholder="you@moveo.co.il", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password",
                                     help="At least 8 characters.")
            if st.form_submit_button("Create account", icon=":material/person_add:", type="primary") \
                    and email and password:
                _start_session(api_client.request("POST", "/auth/signup",
                                                  json={"email": email, "password": password}))


if not api_client.token():
    st.navigation([st.Page(login_page, title="Log in", icon=":material/login:")], position="hidden").run()
    st.stop()

home = st.Page("app_pages/home.py", title="Home", icon=":material/home:", default=True)
chat = st.Page("app_pages/chat.py", title="Chat", icon=":material/chat:")
analytics = st.Page("app_pages/analytics.py", title="Analytics", icon=":material/bar_chart:")
alerts = st.Page("app_pages/alerts.py", title="Alerts", icon=":material/notifications:")
st.session_state.pages = {"home": home, "chat": chat, "analytics": analytics, "alerts": alerts}
page = st.navigation([home, chat, analytics, alerts], position="top")

with st.container(horizontal=True, horizontal_alignment="right"):
    with st.popover(st.session_state.email, icon=":material/account_circle:", type="tertiary"):
        st.caption(f"Role: {st.session_state.role}")
        if st.button("Log out", icon=":material/logout:"):
            st.session_state.clear()
            st.rerun()

page.run()
