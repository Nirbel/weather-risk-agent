"""Analytics: the same deterministic results chat uses — rankings, hub comparisons,
hazard-family scores and historical exposure metrics."""

import altair as alt
import pandas as pd
import streamlit as st

import api_client

FAMILIES = {"winter": "Winter", "hurricane": "Hurricane", "flood": "Flood", "severe_storm": "Severe storm",
            "heat": "Heat"}
REGIONS = {"midwest": "Midwest", "south": "South", "west": "West", "northeast": "Northeast"}
# Validated categorical slots 1–3 (all-pairs) and the sequential blue ramp — see the dataviz palette.
THEME = {
    "light": {"series": ["#2a78d6", "#eb6834", "#1baf7a"], "ink": "#0b0b0b", "ink2": "#52514e", "surface": "#ffffff"},
    "dark": {"series": ["#3987e5", "#d95926", "#199e70"], "ink": "#ffffff", "ink2": "#c3c2b7", "surface": "#0e1117"},
}
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
MAX_COMPARE = 3

mode = "dark" if getattr(st.context.theme, "type", "light") == "dark" else "light"
colors = THEME[mode]

st.markdown("Every number here comes from the same deterministic scoring the chat agent cites.")
with st.spinner("Loading exposure scores — a cold cache fetches 5 years of weather per hub (up to ~3 minutes)…"):
    data = api_client.get("/analytics/exposure")
if not data:
    st.error("Exposure scores are unavailable right now.", icon=":material/error:")
    st.stop()

hubs = pd.DataFrame([{"hub_id": h["hub_id"], "hub": h["hub"], "region": h["region"], "overall": h["overall"],
                      "top hazard": FAMILIES.get(h["top_family"], "—"), "data gaps": len(h["data_gaps"]),
                      "weather complete": h["weather_complete"], "weather coverage %": h["coverage_pct"],
                      **{f: h["families"][f]["score"] for f in FAMILIES}} for h in data["hubs"]])
roster_order = list(hubs["hub"])  # stable order → stable colors
hub_ids = {h["hub"]: h["hub_id"] for h in data["hubs"]}

# -- controls: one row above the charts --------------------------------------------
with st.container(horizontal=True, vertical_alignment="bottom"):
    focus = st.segmented_control("Score", ["overall", *FAMILIES], default="overall", required=True,
                                 format_func=lambda f: "Overall" if f == "overall" else FAMILIES[f])
    regions = st.pills("Region", list(REGIONS), selection_mode="multi", default=list(REGIONS),
                       format_func=REGIONS.get)
view = hubs[hubs["region"].isin(regions or list(REGIONS))].copy()
view["score"] = view[focus]
view = view.sort_values("score", ascending=False)
# Coverage rule: a hub whose weather record is incomplete has no score for weather-based hazards → not ranked.
unscored = view[view["score"].isna()]
ranked = view.dropna(subset=["score"])

with st.container(horizontal=True):
    st.metric("Exposure window", f"{data['window']['start'][:4]}–{data['window']['end'][:4]}", border=True)
    st.metric("FEMA NRI release", data["nri_version"] or "unavailable", border=True)
    if not ranked.empty:
        st.metric(f"Most exposed ({'overall' if focus == 'overall' else FAMILIES[focus].lower()})",
                  ranked.iloc[0]["hub"], f"{ranked.iloc[0]['score']:.1f} / 100", delta_color="off", border=True)
    st.metric("Hubs with data gaps", int((hubs["data gaps"] > 0).sum()), border=True)

# -- ranking + hazard grid -----------------------------------------------------------
left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.markdown(f"**Ranking — {'overall exposure' if focus == 'overall' else FAMILIES[focus]}**")
        base = alt.Chart(ranked).encode(
            y=alt.Y("hub:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
            x=alt.X("score:Q", scale=alt.Scale(domain=[0, 100]), title="Exposure score (0–100)"),
            tooltip=[alt.Tooltip("hub:N", title="Hub"), alt.Tooltip("score:Q", title="Score", format=".1f"),
                     alt.Tooltip("top hazard:N", title="Top hazard")],
        )
        bars = base.mark_bar(cornerRadiusEnd=4, color=colors["series"][0], size=18)
        labels = base.mark_text(align="left", dx=4, color=colors["ink2"]).encode(text=alt.Text("score:Q", format=".1f"))
        st.altair_chart(bars + labels, height=36 * max(len(ranked), 1) + 40)
        if not unscored.empty:
            st.caption(":material/warning: Not ranked — incomplete weather data (needs ≥ 99 % of days and no gap "
                       "over 3 days): " + ", ".join(unscored["hub"]))
with right:
    with st.container(border=True):
        st.markdown("**Hazard-family scores by hub**")
        grid = view.melt(id_vars=["hub"], value_vars=list(FAMILIES), var_name="family", value_name="value")
        grid["family"] = grid["family"].map(FAMILIES)
        cells = alt.Chart(grid).encode(
            x=alt.X("family:N", sort=list(FAMILIES.values()), title=None, axis=alt.Axis(labelAngle=0, orient="top")),
            y=alt.Y("hub:N", sort=list(view["hub"]), title=None, axis=alt.Axis(labelLimit=220)),
            tooltip=[alt.Tooltip("hub:N", title="Hub"), alt.Tooltip("family:N", title="Hazard"),
                     alt.Tooltip("value:Q", title="Score", format=".1f")],
        )
        rects = cells.mark_rect(cornerRadius=4, stroke=colors["surface"], strokeWidth=2).encode(
            color=alt.Color("value:Q", scale=alt.Scale(domain=[0, 100], range=SEQUENTIAL),
                            legend=alt.Legend(title="Score", orient="bottom")))
        text = cells.transform_filter("isValid(datum.value)").mark_text(fontSize=12).encode(
            text=alt.Text("value:Q", format=".0f"),
            color=alt.condition(alt.datum.value > 55, alt.value("#ffffff"), alt.value("#0b0b0b")))
        st.altair_chart(rects + text, height=36 * max(len(view), 1) + 60)

with st.expander("Data table", icon=":material/table:"):
    st.dataframe(view.drop(columns=["hub_id", "score"]).rename(columns=FAMILIES), hide_index=True,
                 column_config={"overall": st.column_config.NumberColumn("Overall", format="%.1f")})

# -- hub comparison ------------------------------------------------------------------
st.subheader("Compare hubs", anchor=False)
picked = st.multiselect("Hubs (up to 3)", roster_order, default=roster_order[:2], max_selections=MAX_COMPARE)
picked_ordered = [h for h in roster_order if h in picked]
palette = dict(zip(picked_ordered, colors["series"]))
if picked_ordered:
    long = hubs[hubs["hub"].isin(picked_ordered)].melt(id_vars=["hub"], value_vars=list(FAMILIES),
                                                       var_name="family", value_name="value")
    long["family"] = long["family"].map(FAMILIES)
    with st.container(border=True):
        st.markdown("**Hazard-family scores**")
        chart = alt.Chart(long).mark_bar(cornerRadiusEnd=4, stroke=colors["surface"], strokeWidth=2).encode(
            x=alt.X("family:N", sort=list(FAMILIES.values()), title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("hub:N", sort=picked_ordered),
            y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 100]), title="Score (0–100)"),
            color=alt.Color("hub:N", scale=alt.Scale(domain=picked_ordered, range=list(palette.values())),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=[alt.Tooltip("hub:N", title="Hub"), alt.Tooltip("family:N", title="Hazard"),
                     alt.Tooltip("value:Q", title="Score", format=".1f")],
        )
        st.altair_chart(chart, height=320)

    # -- historical exposure metrics -------------------------------------------------
    yearly = api_client.get("/analytics/yearly") or {}
    if yearly:
        labels = next(iter(yearly.values()))["labels"]
        indicator = st.selectbox("Observed indicator", list(labels), format_func=labels.get)
        rows = [{"hub": name, "year": int(year), "days": counts[indicator]}
                for name in picked_ordered for year, counts in yearly[hub_ids[name]]["years"].items()]
        hist = pd.DataFrame(rows)
        with st.container(border=True):
            st.markdown(f"**{labels[indicator]} — per year**")
            line = alt.Chart(hist).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=70, filled=True)).encode(
                x=alt.X("year:O", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("days:Q", title="Days"),
                color=alt.Color("hub:N", scale=alt.Scale(domain=picked_ordered, range=list(palette.values())),
                                legend=alt.Legend(title=None, orient="top")),
                tooltip=[alt.Tooltip("hub:N", title="Hub"), alt.Tooltip("year:O", title="Year"),
                         alt.Tooltip("days:Q", title="Days")],
            )
            st.altair_chart(line, height=280)
        with st.expander("Data table", icon=":material/table:"):
            st.dataframe(hist.pivot(index="year", columns="hub", values="days"))

# -- one hub, explained ----------------------------------------------------------------
st.subheader("Why does a hub score what it does?", anchor=False)
hub_name = st.selectbox("Hub", roster_order, index=roster_order.index("Dallas") if "Dallas" in roster_order else 0)
explained = api_client.get(f"/analytics/hubs/{hub_ids[hub_name]}")
if explained:
    with st.container(border=True):
        st.markdown(explained["answer"])
        st.dataframe(pd.DataFrame(explained["table"]), hide_index=True)
        st.markdown("**Top drivers**")
        for point in explained["reasoning"]:
            st.markdown(f"- {point}")
        if explained["uncertainty"]["level"] != "low":
            st.caption("Data notes: " + " ".join(explained["uncertainty"]["reasons"]))
