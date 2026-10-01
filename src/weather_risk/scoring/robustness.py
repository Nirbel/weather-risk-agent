"""How far to trust a ranking: one-at-a-time weight perturbations and near-ties."""

from weather_risk.scoring.exposure import family_score_from_lenses


def _scores(breakdowns: dict[str, dict], families: list[str], family_weights: dict[str, float],
            lens_factors: dict[str, float]) -> dict[str, float]:
    out = {}
    for hub, bd in breakdowns.items():
        pairs = []
        for f in families:
            s = family_score_from_lenses(bd["families"][f]["lenses"], lens_factors)
            if s is not None:
                pairs.append((family_weights.get(f, 0.0), s))
        total = sum(w for w, _ in pairs)
        if total > 0:
            out[hub] = sum(w * s for w, s in pairs) / total
    return out


def ranked(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=lambda h: (-scores[h], h))


def rank_stability(
    breakdowns: dict[str, dict],
    *,
    families: list[str],
    family_weights: dict[str, float],
    lenses: tuple[str, ...],
    perturbation: float,
    top_k: int,
) -> dict:
    """Re-rank with each family weight and each lens weight moved ±perturbation, one at a time."""
    baseline = ranked(_scores(breakdowns, families, family_weights, {}))
    scenarios: list[tuple[str, dict, dict]] = []
    for factor, sign in ((1 + perturbation, "+"), (1 - perturbation, "−")):
        for f in families:
            scenarios.append((f"{f} weight {sign}{perturbation:.0%}", family_weights | {f: family_weights.get(f, 0.0) * factor}, {}))
        for lens in lenses:
            scenarios.append((f"{lens} evidence weight {sign}{perturbation:.0%}", family_weights, {lens: factor}))

    top1_same = topk_same = 0
    flips = []
    for description, fw, lw in scenarios:
        order = ranked(_scores(breakdowns, families, fw, lw))
        if order[:1] == baseline[:1]:
            top1_same += 1
        elif len(flips) < 3:
            flips.append({"scenario": description, "first": order[0]})
        if set(order[:top_k]) == set(baseline[:top_k]):
            topk_same += 1
    return {
        "scenarios": len(scenarios),
        "top1_unchanged": top1_same,
        "topk_unchanged": topk_same,
        "top_k": top_k,
        "baseline": baseline,
        "flips": flips,
    }


def near_ties(ranked_scores: list[tuple[str, float]], margin: float, top_k: int) -> list[tuple[str, str, float]]:
    """Adjacent pairs, down to the top-k boundary, whose scores are closer than `margin` points."""
    ties = []
    for i in range(min(top_k, len(ranked_scores) - 1)):
        (a, sa), (b, sb) = ranked_scores[i], ranked_scores[i + 1]
        if sa - sb < margin:
            ties.append((a, b, round(sa - sb, 1)))
    return ties
