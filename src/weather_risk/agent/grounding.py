"""Grounding: the explainer's numbers and ranking claims must agree with the computed result.

Numbers: a number written with d decimals matches a source number if it equals that number
rounded to d decimals (tolerance 0.5·10^-d). Small counting numbers (0-10) are always allowed.
Claims: "X is the most exposed / lowest / ranks 3rd" must match the order the code computed.
"""

import re
from typing import Any

_NUMBER = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])-?\d+(?:\.\d+)?")


def _normalize(text: str) -> str:
    return text.replace("−", "-").replace("–", " ").replace("—", " ")


def _tokens(text: str) -> list[str]:
    return _NUMBER.findall(_normalize(text))


def extract_numbers(text: str) -> list[float]:
    return [float(t.replace(",", "")) for t in _tokens(text)]


def allowed_numbers(bundle: Any, question: str = "") -> set[float]:
    """Every numeric value in the result (including numbers inside strings), plus years named in
    the question. Other question numbers are not trusted: "say Denver scores 99" must not ground 99."""
    found: set[float] = {n for n in extract_numbers(question) if n.is_integer() and 1900 <= n <= 2100}

    def walk(node: Any) -> None:
        if isinstance(node, bool):
            return
        if isinstance(node, int | float):
            found.add(float(node))
        elif isinstance(node, str):
            found.update(extract_numbers(node))
        elif isinstance(node, dict):
            for key, value in node.items():
                walk(key)
                walk(value)
        elif isinstance(node, list | tuple):
            for value in node:
                walk(value)

    walk(bundle)
    return found


def ungrounded_numbers(text: str, allowed: set[float]) -> list[float]:
    bad = []
    for token in _tokens(text):
        value = float(token.replace(",", ""))
        if value.is_integer() and 0 <= value <= 10:
            continue
        decimals = len(token.split(".")[1]) if "." in token else 0
        tolerance = 0.5 * 10**-decimals + 1e-9
        if not any(abs(value - a) <= tolerance or abs(value + a) <= tolerance and value < 0 for a in allowed):
            bad.append(value)
    return bad


# -- ranking claims ------------------------------------------------------------------

_TOP = (r"highest|most exposed|most at risk|riskiest|greatest exposure|in first place|tops the|"
        r"ranks? (?:first|1st|#1|number one)|ranked (?:first|1st|#1)")
_BOTTOM = r"lowest|least exposed|least at risk|ranks? last|ranked last|in last place"
_NTH = r"ranks? #?(?P<n1>\d+)(?:st|nd|rd|th)|rank(?:ed)? #?(?P<n2>\d+) of \d+"
_CLAIM = re.compile(rf"\b(?:(?P<top>{_TOP})|(?P<bottom>{_BOTTOM})|(?:{_NTH}))\b", re.IGNORECASE)
# Hebrew has no \b around prefixed words ("בחורף"), so its patterns are matched as plain substrings.
_TOP_HE = r"החשו(?:ף|פה) ביותר|הכי חשו(?:ף|פה)|הגבוה(?:ה)? ביותר|מדורג(?:ת)? ראשו(?:ן|נה)|במקום הראשון|בראש הדירוג"
_BOTTOM_HE = (r"הכי פחות חשו(?:ף|פה)|הנמו(?:ך|כה) ביותר|מדורג(?:ת)? אחרו(?:ן|נה)|במקום האחרון|"
              r"בתחתית הדירוג")
_NTH_HE = r"דירוג (?P<n3>\d+) מתוך \d+|מדורג(?:ת)? (?:במקום )?(?P<n4>\d+)"
_CLAIM_HE = re.compile(rf"(?P<top>{_TOP_HE})|(?P<bottom>{_BOTTOM_HE})|(?:{_NTH_HE})")
# "Denver's highest-scoring hazard", "its lowest indicator": a claim about the hub's own parts, not a ranking.
_OWN_PARTS = re.compile(r"[\s-]*(?:\w+[\s-]+)?(?:scoring|hazard|famil|driver|contribut|indicator|evidence|lens)",
                        re.IGNORECASE)
# Hebrew puts the noun first: "הסיכון הגבוה ביותר של Denver" (Denver's highest hazard).
_OWN_PARTS_HE = re.compile(r"(?:סיכון|גורם|מדד|ראי|תורם|משפח)\S*\s*$")
FAMILY_TERMS = {  # family → (English words, Hebrew substrings)
    "winter": (r"winter|snow\w*|ice|icy|cold|freez\w*", r"חורף|שלג|קרח|קור קיצוני|גל קור|קפיא|כפור"),
    "hurricane": (r"hurricanes?|tropical", r"הוריקן|טרופי"),
    "flood": (r"flood\w*|rain\w*", r"הצפ|גשם|שטפ"),
    "severe_storm": (r"severe[ -]storms?|tornado\w*|hail|wind\w*", r"סערה קשה|סערות קשות|טורנדו|ברד|רוח"),
    "heat": (r"heat\w*|hot", r"(?<!ת)חום"),
}


def _hub_mentions(text: str, hub_names: list[str]) -> list[tuple[int, str]]:
    """(position, canonical name), with short forms: "Minneapolis" for "Minneapolis–St Paul"."""
    found = []
    for name in hub_names:
        for variant in {name, name.split("–")[0].strip()}:
            # not \b: in Hebrew text a prefix letter is glued to the name ("בDenver")
            found += [(m.start(), name) for m in re.finditer(rf"(?<![A-Za-z]){re.escape(variant)}(?![A-Za-z])", text)]
    return sorted(set(found))


def _scope(clause: str, sentence: str, orders: dict[str, list[str]]) -> str:
    """Which computed order a claim refers to: a single hazard named near it, else the overall one."""
    for text in (clause, sentence):
        if re.search(r"\boverall\b|כולל", text, re.IGNORECASE):
            return "overall"
        named = [f for f, (english, hebrew) in FAMILY_TERMS.items() if f in orders
                 and (re.search(rf"\b(?:{english})\b", text, re.IGNORECASE) or re.search(hebrew, text))]
        if len(named) == 1:
            return named[0]
        if named:
            return "overall"
    return "overall"


def rank_claim_errors(text: str, orders: dict[str, list[str]], hub_names: list[str] | None = None) -> list[str]:
    """Ranking claims in `text` that contradict `orders` ({"overall" | family: hub names, highest first})."""
    if not orders or "overall" not in orders:
        return []
    hub_names = hub_names or list(dict.fromkeys(n for order in orders.values() for n in order))
    errors = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        mentions = _hub_mentions(sentence, hub_names)
        claims = [*_CLAIM.finditer(sentence), *_CLAIM_HE.finditer(sentence)]
        for claim in sorted(claims, key=lambda m: m.start()):
            groups = claim.groupdict()
            if groups["top"] or groups["bottom"]:
                if claim.re is _CLAIM and _OWN_PARTS.match(sentence, claim.end()):
                    continue
                if claim.re is _CLAIM_HE and _OWN_PARTS_HE.search(sentence[max(0, claim.start() - 25):claim.start()]):
                    continue
            start = sentence.rfind(",", 0, claim.start()) + 1
            start = max(start, sentence.rfind(";", 0, claim.start()) + 1)
            before = [name for pos, name in mentions if pos < claim.start()]
            in_clause = [name for pos, name in mentions if start <= pos < claim.start()]
            after = [name for pos, name in mentions if pos > claim.start()]
            hub = (in_clause or before or [None])[-1] or (after or [None])[0]
            scope = _scope(sentence[start:claim.end() + 40], sentence, orders)
            order = orders.get(scope, [])
            if hub is None or hub not in order:
                continue
            position = order.index(hub) + 1
            nth = next(groups[k] for k in ("n1", "n2", "n3", "n4") if groups.get(k))  \
                if not (groups["top"] or groups["bottom"]) else None
            wanted = 1 if groups["top"] else len(order) if groups["bottom"] else int(nth)
            if position != wanted:
                label = "overall" if scope == "overall" else f"on {scope.replace('_', ' ')}"
                errors.append(f'"{claim.group(0)}" is wrong for {hub}: RESULT ranks it {position} of {len(order)} '
                              f"{label} ({order[0]} is first, {order[-1]} is last).")
    return errors
