"""Thin HTTP client for the FastAPI backend. The UI holds no business logic."""

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 300  # a cold cache can take ~3 minutes (Open-Meteo free-tier rate limit)


def token() -> str | None:
    return st.session_state.get("token")


def request(method: str, path: str, **kwargs) -> httpx.Response | None:
    headers = {"Authorization": f"Bearer {token()}"} if token() else {}
    try:
        response = httpx.request(method, f"{API_URL}{path}", headers=headers, timeout=TIMEOUT, **kwargs)
    except httpx.HTTPError as exc:
        st.error(f"Cannot reach the API at {API_URL} ({exc.__class__.__name__}). Is the backend running?",
                 icon=":material/cloud_off:")
        return None
    if response.status_code == 401 and token():
        st.session_state.clear()
        st.rerun()
    return response


@st.cache_data(ttl=600, show_spinner=False)
def cached_get(path: str, auth_token: str | None) -> dict | list | None:
    """GETs of deterministic results, cached per user token for 10 minutes."""
    headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
    response = httpx.get(f"{API_URL}{path}", headers=headers, timeout=TIMEOUT)
    return response.json() if response.status_code == 200 else None


def get(path: str):
    try:
        return cached_get(path, token())
    except httpx.HTTPError as exc:
        st.error(f"Cannot reach the API at {API_URL} ({exc.__class__.__name__}).", icon=":material/cloud_off:")
        return None
