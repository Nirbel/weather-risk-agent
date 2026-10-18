"""How scoring works: the strategy, the formulas, the per-hazard recipe and a worked example.

Thresholds, anchors and weights are read from the API (/scoring/config = config/scoring.yaml) and the
worked example from /analytics/hubs/{id}, so this page always matches the numbers chat cites.
"""

import pandas as pd
import streamlit as st

import api_client

LENSES = {"observed": "Observed weather (Open-Meteo ERA5)", "modeled": "Modeled loss (FEMA NRI)"}


def fmt(x: float) -> str:
    return f"{x:g}"


def md(text: str) -> str:
    """Escape "$", which Streamlit Markdown would otherwise read as LaTeX."""
    return text.replace("$", r"\$")


def per_million(rate: float) -> str:
    """Building loss rate → dollars per $1M of building value a year (Markdown-escaped)."""
    dollars = rate * 1e6
    return md(f"${dollars:,.0f}" if dollars >= 10 else f"${dollars:.2g}")


st.title("How scoring works", anchor=False)
st.markdown(
    "Every hub gets a **long-term weather exposure score from 0 to 100**, overall and for each of five hazard "
    "families. It answers one question for an investment decision: *how often, and how badly, does weather "
    "disrupt this location?* Higher means more exposed. This page shows exactly how the number is built."
)

cfg = api_client.get("/scoring/config")
if not cfg:
    st.error("The scoring configuration is unavailable right now.", icon=":material/error:")
    st.stop()
families = cfg["families"]
weights = cfg["lens_weights_by_family"]
equal = len(set(cfg["family_weights"].values())) == 1

# -- strategy ----------------------------------------------------------------------
st.header("The strategy", anchor=False)
# Cards on the left, a top-to-bottom pipeline on the right: readable on a wide screen, and the two
# columns stack on a phone, where a left-to-right diagram would shrink to illegible.
cards, pipeline = st.columns([3, 2], vertical_alignment="center")
with cards:
    with st.container(border=True):
        st.markdown("**:material/function: Numbers come from code**")
        st.caption("Scores, ranks and percentages are computed by deterministic Python. The language model only "
                   "turns your question into a query plan and explains the result. Every number it writes is "
                   "checked against the computed result.")
    with st.container(border=True):
        st.markdown("**:material/stacks: Two kinds of evidence**")
        st.caption("*Observed*: how many days a year the hub's weather crossed an operational threshold. "
                   "*Modeled*: FEMA's estimate of how much building damage the county's hazards cause. "
                   "They cover each other's blind spots.")
    with st.container(border=True):
        st.markdown("**:material/anchor: Stable and auditable**")
        st.caption("Indicators are scored on fixed anchors, not relative to the other hubs. Adding or removing a hub "
                   "never changes another hub's score, so a score change always means the data changed.")
with pipeline:
    st.mermaid_chart("""
flowchart TD
    Q[Your question] --> P[LLM planner<br/>query plan, schema-checked]
    P --> E[Python executor<br/>scores, ranks, stats]
    D[(Open-Meteo ERA5<br/>FEMA NRI)] --> E
    E --> X[LLM explainer<br/>numbers + rank claims checked]
    X --> A[Answer + assumptions,<br/>uncertainty, sources, window]
""")

# -- formulas ----------------------------------------------------------------------
st.header("From weather to a 0–100 score", anchor=False)
years = cfg["climatology"]["years"]
left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.markdown("**1 · Observed indicator** — days per year over the previous "
                    f"{years} completed calendar years, scored against a fixed anchor:")
        st.latex(r"\text{indicator} = \min\!\left(100,\; 100 \cdot \frac{\text{days per year}}{\text{anchor}}\right)")
        snow = families["winter"]["observed"][0]
        st.caption(f"Example: {snow['label'].lower()} scores 100 at {fmt(snow['full_score_at'])} days a year, so "
                   f"{fmt(snow['full_score_at'] / 2)} days a year scores 50. The observed lens is the mean of the "
                   "family's indicators.")
    with st.container(border=True):
        st.markdown("**2 · Modeled hazard** — the county's FEMA NRI *expected annual building-loss rate*, "
                    "scored on a log scale between two fixed anchors:")
        st.latex(r"\text{NRI score} = \min\!\left(100,\; \max\!\left(0,\; 100 \cdot "
                 r"\frac{\log_{10} \text{rate} - \log_{10} \text{floor}}{\log_{10} \text{full} - \log_{10} \text{floor}}"
                 r"\right)\right)")
        hrcn = cfg["nri_anchors"]["HRCN"]
        st.caption(f"Example: hurricane scores 0 at {per_million(hrcn['floor'])} and 100 at {per_million(hrcn['full'])} "
                   "lost per \\$1M of building value a year, and each tenfold rise in the rate adds the same number of "
                   "points. A hazard that does not apply to the county (e.g. hurricanes in Denver) scores 0. Several "
                   "NRI hazards in one family are averaged, or the maximum is taken for flood.")
with right:
    with st.container(border=True):
        st.markdown("**3 · Hazard family** — a weighted mean of the two lenses:")
        st.latex(r"F = \frac{w_{obs} \cdot O + w_{nri} \cdot M}{w_{obs} + w_{nri}}")
        st.caption("If one lens is missing by design (hurricane and severe storm have no observed lens), the weights "
                   "are renormalized over the lens that exists. Incomplete weather is *not* renormalized away: the "
                   "family then gets no score (see the coverage rule below).")
    with st.container(border=True):
        st.markdown("**4 · Overall exposure** — a weighted mean of the five families:")
        st.latex(r"S = \frac{\sum_f W_f \cdot F_f}{\sum_f W_f}")
        st.caption(("All family weights are equal by default, so the overall is the plain average. "
                    if equal else "Family weights come from config/scoring.yaml. ")
                   + "You can re-weight per question in chat, e.g. *“rank all hubs, weight flood double”*.")

# -- per-hazard recipe ---------------------------------------------------------------
st.header("What each hazard is made of", anchor=False)
st.caption("Read live from `config/scoring.yaml`, the single source of truth for every threshold, anchor and weight.")
rows = []
for fid, fam in families.items():
    observed = "  \n".join(f"{i['label']} → 100 pts at {fmt(i['full_score_at'])} days/yr" for i in fam["observed"])
    modeled = fam.get("modeled") or {}
    nri = ", ".join(cfg["nri_hazard_names"][c] for c in modeled.get("nri", []))
    w = weights[fid]
    rows.append({
        "hazard": fam["label"],
        "observed weather indicators": observed or "— (reanalysis can't resolve this hazard)",
        "FEMA NRI hazards": f"{nri} ({modeled['combine']})" if nri else "—",
        "NRI anchors (0 → 100 pts, per \\$1M a year)": "  \n".join(
            f"{cfg['nri_hazard_names'][c]}: {per_million(cfg['nri_anchors'][c]['floor'])} → "
            f"{per_million(cfg['nri_anchors'][c]['full'])}" for c in modeled.get("nri", [])) or "—",
        "weight: observed / NRI": f"{fmt(w['observed'])} / {fmt(w['modeled'])}" if fam["observed"]
        else "— / 1 (NRI only)",
        "family weight": fmt(cfg["family_weights"][fid]),
    })
st.table(pd.DataFrame(rows), hide_index=True)  # st.table wraps long indicator text; dataframes truncate it

# -- worked example ------------------------------------------------------------------
st.header("Worked example", anchor=False)
hubs = api_client.get("/hubs") or []
names = {h["name"]: h["id"] for h in hubs}
choice = st.selectbox("Hub", list(names), index=list(names).index("Denver") if "Denver" in names else 0,
                      key="scoring_hub")
with st.spinner("Computing the breakdown (a cold cache fetches 5 years of weather, up to a minute per hub)…"):
    detail = api_client.get(f"/analytics/hubs/{names[choice]}") if choice else None
if not detail or not detail.get("breakdown"):
    st.warning("The breakdown for this hub is unavailable right now.", icon=":material/warning:")
else:
    bd = detail["breakdown"]
    table = detail["table"]
    with st.container(horizontal=True):
        st.metric("Overall exposure", f"{bd['overall_score']}/100" if bd["overall_score"] is not None else "not scored",
                  border=True)
        st.metric("Rank among hubs", bd["overall_rank"], border=True)
        st.metric("Biggest contributor", table[0]["hazard"].split(" (")[0] if table else "—", border=True)
    st.markdown(detail["answer"])

    for name, fam in bd["hazards"].items():
        score = f"{fam['score']}/100" if fam["score"] is not None else "not scored"
        with st.expander(f"**{name}** — {score}", icon=":material/calculate:"):
            terms = []
            for lens, data in fam["lenses"].items():
                lens_score = f"{data['score']} pts" if data["score"] is not None else "missing"
                st.markdown(f"**{LENSES[lens]}** · {lens_score} · weight {fmt(data['weight'])}")
                for evidence in data["evidence"]:
                    st.markdown(f"- {md(evidence)}")
                if data["score"] is not None:
                    terms.append((data["weight"], data["score"]))
            if fam["score"] is not None and terms:
                total = sum(w for w, _ in terms)
                expr = " + ".join(f"{fmt(w)} × {s}" for w, s in terms)
                expr = expr if abs(total - 1) < 1e-9 else f"({expr}) / {fmt(total)}"
                st.markdown(f"Family score: {expr} = **{fam['score']}**")
            if fam["missing_evidence"]:
                st.caption("Missing evidence: " + ", ".join(LENSES[m] for m in fam["missing_evidence"]))

    scored = [r for r in table if r["score"] is not None]
    if bd["overall_score"] is not None and len(scored) == len(families) and equal:
        expr = " + ".join(str(r["score"]) for r in scored)
        st.markdown(f"**Overall:** ({expr}) / {len(scored)} = **{bd['overall_score']}**")
    st.caption("Inputs are shown rounded to one decimal; scores are computed from unrounded values, so the last "
               "digit can differ by 0.1.")

# -- trust ---------------------------------------------------------------------------
st.header("How far to trust a ranking", anchor=False)
cov, rob = cfg["coverage"], cfg["robustness"]
with st.container(horizontal=True):
    with st.container(border=True):
        st.markdown("**:material/rule: Coverage rule**")
        st.caption(f"A hub's weather counts as complete only if at least {fmt(cov['min_pct'])} % of days have data "
                   f"and no gap is longer than {cov['max_gap_days']} days. Otherwise its weather-based scores are "
                   "withheld and it is listed as *not ranked*, with the reason.")
    with st.container(border=True):
        st.markdown("**:material/balance: Near-ties**")
        st.caption(f"Hubs closer than {fmt(rob['tie_margin'])} points are flagged as effectively tied — the "
                   "data can't separate them that finely.")
    with st.container(border=True):
        st.markdown("**:material/tune: Weight sensitivity**")
        st.caption(f"Every ranking is recomputed {cfg['robustness_scenarios']} times, moving each family weight and "
                   f"each evidence weight ±{rob['perturbation']:.0%} one at a time. The answer says whether first "
                   "place and the top 3 survive.")

# -- rationale -----------------------------------------------------------------------
st.header("Why these choices", anchor=False)
with st.expander("Why fixed anchors instead of ranking the hubs against each other?", icon=":material/help:"):
    st.markdown("Min-max or rank scaling makes a hub's score depend on which other hubs are in the list: drop Miami "
                "and everyone else moves. With fixed anchors, removing a hub only changes *ranks*, never scores, and "
                "a score change over time always means the weather or FEMA data changed.")
with st.expander("Why FEMA's building-loss rate, not the headline NRI risk score?", icon=":material/help:"):
    st.markdown("The headline `RISK_SCORE` scales with the dollar value exposed in the county, so every metro hub "
                "scores 95–100. The composite percentiles include crop losses (Miami looks cold-wave exposed because "
                "of crop freezes). The building-loss *rate* measures how hazardous the place is for a facility, "
                "independent of county size.")
with st.expander("Why a log scale with fixed anchors for FEMA loss rates?", icon=":material/help:"):
    st.markdown("Loss rates span many orders of magnitude: Miami's hurricane rate is about 14,000 times Chicago's. "
                "A national percentile only measures order, so Chicago's tiny rate still scored 29 (over a quarter of "
                "counties have no hurricane loss at all) while Miami and Houston, 1.6× apart, sat 2 points apart. On "
                "a log scale every tenfold step is worth the same points. For each hazard the anchors are measured "
                "from the data: *floor* is the 10th and *full* the 99th percentile of the counties with any loss.")
with st.expander("Why do winter and heat lean on observed weather?", icon=":material/help:"):
    w, h = weights["winter"], weights["heat"]
    st.markdown(f"Snow, ice and heat disrupt hubs through operations — closures, slow handling, labour safety — not "
                f"building damage. So winter weighs observed weather {fmt(w['observed'])} and NRI {fmt(w['modeled'])}, "
                f"heat {fmt(h['observed'])} / {fmt(h['modeled'])}. Flood uses "
                f"{fmt(weights['flood']['observed'])} / {fmt(weights['flood']['modeled'])}.")
with st.expander("Why equal hazard weights?", icon=":material/help:"):
    st.markdown("Without your own loss history, equal weights are the most neutral default. The trade-off: a hub with "
                "one extreme hazard (Miami and hurricanes) is diluted overall. That's why answers always name each "
                "hub's top hazard, hazard questions rank by that family alone, and you can re-weight per question.")
with st.expander("Why hurricane and severe storm use FEMA NRI only", icon=":material/help:"):
    st.markdown("Reanalysis weather is a smoothed grid of daily values. It can't resolve tornadoes or hail, and it is not a "
                "hurricane-track record. Rather than fake an observed signal, those families rest on NRI alone, and "
                "every answer that uses them says so.")
with st.expander(f"Why the previous {years} completed years?", icon=":material/help:"):
    st.markdown(f"{years} full calendar years smooth out a single odd winter while staying recent. Partial years would "
                "bias seasonal counts, so the window always ends at the last completed year.")

st.header("Known limitations", anchor=False)
st.markdown(
    "- **NRI anchors are a snapshot.** They were measured on the December 2025 NRI release; a new release does not "
    "move them by itself, so they need re-measuring when FEMA updates the data.\n"
    "- **Reanalysis is modeled, not measured.** ERA5 smooths daily extremes; counts can differ from airport stations "
    "by several days a year. The anchors are set in reanalysis terms for that reason.\n"
    "- **Hub locations are assumed** (each metro's main airport logistics zone), and county-level hazard data stands "
    "in for the site.\n"
    "- **Thresholds and weights are judgment calls.** They're all in config and printed in every answer, and the "
    "sensitivity check shows how fragile an order is."
)
st.info("To change the scoring, edit `config/scoring.yaml` and restart the API — it is validated at startup. "
        "For a one-off view, ask in chat, e.g. *“rank all hubs, weight hurricane twice as much”*.",
        icon=":material/edit_note:")
