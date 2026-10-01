"""Alerts: each user's own score-change webhook — URL, on/off, threshold, Save, Test webhook."""

import pandas as pd
import streamlit as st

import api_client

EXAMPLE_PAYLOAD = {
    "event": "hub_exposure_score_changed",
    "text": "Weather risk: 1 hub exposure score change(s) of at least 5 points (2021–2025 window) — "
            "Miami 41.0 → 47.3 (+6.3)",
    "checked_at": "2026-10-02T06:00:00+00:00",
    "threshold": 5.0,
    "window": {"start": "2021-01-01", "end": "2025-12-31"},
    "changes": [{"hub_id": "miami", "hub": "Miami", "old_score": 41.0, "new_score": 47.3, "delta": 6.3,
                 "direction": "up", "top_hazard": "hurricane"}],
}


def when(ts: str | None) -> str:
    return f"{ts[:16].replace('T', ' ')} UTC" if ts else "never"


def call(method: str, path: str, **kwargs) -> dict | None:
    """Call the API; on failure leave an error note for the next run and return None."""
    response = api_client.request(method, path, **kwargs)
    if response is not None and response.status_code == 200:
        return response.json()
    if response is not None:
        detail = response.json().get("detail", response.text) if response.headers.get("content-type", "").startswith(
            "application/json") else response.text
        st.session_state.alert_note = ("error", detail if isinstance(detail, str) else "Check the values and try again.")
    return None


st.title("Score-change alerts", anchor=False)
st.markdown(
    "Get a message when a hub's **overall exposure score** moves by at least your threshold. "
    "Point the webhook at **any** endpoint that accepts a JSON `POST` — a Make or Zapier scenario, an n8n "
    "workflow, a Slack or Teams workflow, or your own service. The payload has a plain `text` summary plus "
    "structured `changes`."
)
if note := st.session_state.pop("alert_note", None):
    getattr(st, note[0])(note[1], icon=":material/notifications:")

feed = call("GET", "/alerts")
cfg = call("GET", "/alerts/settings")
if feed is None or cfg is None:
    st.error("Alerts are unavailable right now.", icon=":material/error:")
    st.stop()

every = feed["check_every_hours"]
with st.container(horizontal=True):
    st.metric("Last check", when(feed["last_check"]), border=True,
              help=f"All hubs are re-scored every {every:g} h." if every else "The scheduler is off.")
    st.metric("Your threshold", f"{cfg['threshold']:g} points", border=True)
    st.metric("Your webhook", "On" if feed["delivering"] else "Off", border=True)

left, right = st.columns([3, 2])
with left:
    with st.container(border=True):
        st.subheader(":material/webhook: Your webhook", anchor=False)
        with st.form("webhook_settings", border=False):
            url = st.text_input("Webhook URL", value=cfg["webhook_url"] or "", placeholder="https://…",
                                help="Any http(s) URL that accepts a JSON POST. Requests time out after 10 s "
                                     "and redirects are not followed.")
            enabled = st.toggle("Send alerts to this webhook", value=cfg["enabled"])
            threshold = st.number_input("Score-change threshold (points on the 0–100 overall score)",
                                        min_value=0.5, max_value=100.0, value=float(cfg["threshold"]), step=0.5)
            if st.form_submit_button("Save", icon=":material/save:", type="primary"):
                if call("PUT", "/alerts/settings",
                        json={"webhook_url": url.strip() or None, "enabled": enabled, "threshold": threshold}):
                    st.session_state.alert_note = ("success", "Webhook settings saved.")
                st.rerun()
        with st.container(horizontal=True, vertical_alignment="center"):
            if st.button("Test webhook", icon=":material/send:", disabled=not cfg["webhook_url"],
                         help="Posts a sample payload (event \"test\") to your saved URL."):
                if body := call("POST", "/alerts/test"):
                    st.session_state.alert_note = ("success" if body["ok"] else "error",
                                                   f"Test webhook: {body['delivery']}.")
                st.rerun()
            if cfg["updated_at"]:
                st.caption(f"Saved {when(cfg['updated_at'])}")
            elif not cfg["webhook_url"]:
                st.caption("Save a URL first, then send a test.")
with right:
    with st.container(border=True):
        st.subheader(":material/info: How alerts work", anchor=False)
        st.markdown(
            "- Each check scores all hubs, then compares them with the scores **you** were last told about.\n"
            "- A hub moving by at least your threshold is listed below and posted to your webhook "
            "(when it is on).\n"
            "- Scores cover 5 completed years, so they change only when the window rolls over in January, "
            "FEMA publishes a new NRI release, or the scoring config changes — alerts are rare by design.\n"
            "- Hubs with data gaps are skipped, so a failed download never looks like a score change."
        )
        with st.expander("Example payload", icon=":material/data_object:"):
            st.json(EXAMPLE_PAYLOAD)

st.subheader("Your alerts", anchor=False)
if feed["alerts"]:
    st.dataframe(pd.DataFrame(feed["alerts"])[["created_at", "hub", "old_score", "new_score", "delta", "delivery"]],
                 hide_index=True, column_config={
                     "created_at": st.column_config.DatetimeColumn("When", format="YYYY-MM-DD HH:mm"),
                     "hub": "Hub", "old_score": st.column_config.NumberColumn("Was", format="%.1f"),
                     "new_score": st.column_config.NumberColumn("Now", format="%.1f"),
                     "delta": st.column_config.NumberColumn("Change", format="%+.1f"), "delivery": "Webhook"})
else:
    st.info("No score changes yet. Your first check records the baseline.", icon=":material/notifications:")

if st.session_state.get("role") == "admin":
    with st.container(border=True):
        st.markdown("**Admin** · run the check for every subscriber now instead of waiting for the schedule.")
        if st.button("Check now", icon=":material/refresh:"):
            with st.spinner("Scoring all hubs and comparing with each subscriber's baseline…"):
                body = call("POST", "/alerts/run")
            if body:
                changed = [u for u in body["users"] if u["changes"]]
                text = (f"Checked {len(body['users'])} subscriber(s); {len(changed)} notified"
                        + (": " + ", ".join(f"{u['user']} ({u['changes']}, {u['delivery']})" for u in changed)
                           if changed else "."))
                if body["skipped"]:
                    text += f" Skipped (data gaps): {', '.join(body['skipped'])}."
                st.session_state.alert_note = ("warning" if body["skipped"] else "success", text)
            st.rerun()
