"""Numeric grounding: every number the explainer writes must come from the computed result.

A number written with d decimals matches a source number if it equals that number
rounded to d decimals (tolerance 0.5·10^-d). Small counting numbers (0-10) are always allowed.
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
    """Every numeric value in the result (including numbers inside strings) plus the question."""
    found: set[float] = set(extract_numbers(question))

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
