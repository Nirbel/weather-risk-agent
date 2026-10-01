"""Home: what the agent does, how it scores, where the data comes from, and its limits."""

import streamlit as st

import api_client

EXAMPLES = [
    "Which hubs in the Midwest are most exposed to winter disruption?",
    "Compare Miami and Houston in terms of hurricane and flood exposure.",
    "What percentage of days in Denver last year had snowfall?",
    "Why is the Dallas hub's weather disruption risk high?",
]

st.title("Weather risk intelligence agent", anchor=False)
st.markdown(
    "Ask which distribution hubs are most exposed to weather disruption — and why. "
    "**Every score, rank and percentage is computed by code from public data;** the language model only "
    "turns your question into a query plan and explains the computed result."
)

health = api_client.get("/health") or {}
window = health.get("exposure_window", {})
with st.container(horizontal=True):
    st.metric("Hubs", health.get("hubs", "—"), border=True)
    st.metric("Exposure window", f"{window.get('start', '—')[:4]}–{window.get('end', '—')[:4]}", border=True,
              help="The previous 5 completed calendar years, fetched on demand and cached.")
    st.metric("Hazard families", 5, border=True, help="Winter, hurricane, flood, severe storm, heat.")

st.subheader("Try a question", anchor=False)
if health.get("voice"):
    st.caption(":material/mic: You can also ask by voice: use the microphone in the chat box.")
with st.container(horizontal=True):
    for i, question in enumerate(EXAMPLES):
        if st.button(question, key=f"example-{i}", icon=":material/arrow_forward:"):
            st.session_state.pending_prompt = question
            st.switch_page(st.session_state.pages["chat"])

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.subheader(":material/function: Methodology", anchor=False)
        st.markdown(
            "- **Exposure score (0–100)** per hub and hazard family, for investment decisions.\n"
            "- Two kinds of evidence per hazard:\n"
            "  - **Observed weather**: days per year above an operational threshold (e.g. snowfall ≥ 2.5 cm), "
            "scored on fixed anchors.\n"
            "  - **Modeled loss**: the county's FEMA NRI building-loss rate, as a percentile of all US counties.\n"
            "- **Overall** = equal-weight average of the 5 hazard families. You can ask to re-weight.\n"
            "- Rankings report near-ties and whether the order survives ±25 % weight changes."
        )
    with st.container(border=True):
        st.subheader(":material/database: Data sources", anchor=False)
        st.markdown(
            "- [Open-Meteo historical weather](https://open-meteo.com/en/docs/historical-weather-api) — "
            "ERA5 reanalysis, daily, CC BY 4.0.\n"
            "- [FEMA National Risk Index](https://hazards.fema.gov/nri/) — county hazard loss rates.\n\n"
            "Data is fetched when a question needs it and cached. A cold first ranking takes about "
            "3 minutes on Open-Meteo's free tier; later answers are instant."
        )
with right:
    with st.container(border=True):
        st.subheader(":material/warning: Limitations", anchor=False)
        st.markdown(
            "- **Hub locations are assumed** (each metro's main airport logistics zone); county data stands in "
            "for the site.\n"
            "- **Reanalysis is modeled, not measured**: it smooths daily extremes and can differ from airport "
            "stations by several days a year.\n"
            "- **Hurricane and severe-storm scores rest on FEMA NRI alone** — reanalysis can't resolve "
            "hurricanes, tornadoes or hail.\n"
            "- **Thresholds and weights are judgment calls**, kept in config and shown in every answer.\n"
            "- **Incomplete weather is never ranked**: a hub needs at least 99 % of days with data and no gap "
            "over 3 days, otherwise its weather-based scores are withheld and the answer says why.\n"
            "- **Long-term exposure only** — no forecasts. Score-change alerts (set your webhook on the *Alerts* "
            "page) fire when the 5-year window rolls over, FEMA publishes a new NRI release, or the scoring "
            "config changes, so they are rare by design."
        )
    with st.container(border=True):
        st.subheader(":material/fact_check: How to read an answer", anchor=False)
        st.markdown(
            "Every answer lists its **assumptions**, **uncertainty**, **data sources** and **time window**, and a "
            "*How this was computed* section with the query plan and the score breakdown. "
            "The *Analytics* page charts the same numbers."
        )
