"""LLM prompt strategies (the thing you tinker with)."""
from __future__ import annotations

from pathlib import Path

DEFAULT_STRATEGIES: dict[str, dict[str, str]] = {
    "Generous": {
        "summary": "Trusting: cooperates by default and forgives most defections.",
        "prompt": (
            "You are an optimistic, trusting player. Cooperate (C) by default and "
            "assume the opponent wants mutual cooperation. Forgive isolated "
            "defections. Only defect (D) if the opponent defected in each of the "
            "last 3 rounds."
        ),
    },
    "Pragmatic": {
        "summary": "Reciprocal (tit-for-tat flavour): mirrors the opponent's last move.",
        "prompt": (
            "You are a pragmatic, reciprocal player. Cooperate in the first round. "
            "After that, copy the opponent's previous move exactly: cooperate "
            "after a C, defect after a D. If the opponent has cooperated for 5 "
            "rounds in a row, keep cooperating even after a single defection."
        ),
    },
    "Merciless": {
        "summary": "Ruthless: defects, never forgives, exploits any cooperation.",
        "prompt": (
            "You are a merciless, self-interested player. Defect (D) every round. "
            "Never forgive and never cooperate: any cooperation from the opponent "
            "is weakness to exploit, and any defection proves they are hostile."
        ),
    },
    "Peacemaker": {
        "summary": "Olive branch: breaks defection spirals by offering cooperation.",
        "prompt": (
            "You are a peacemaker whose goal is to restore mutual cooperation. "
            "Cooperate (C) in the first round. If the previous round was mutual "
            "defection (D,D), cooperate (C) to offer peace. If the opponent "
            "defected while you cooperated twice in a row, defect (D) once as a "
            "warning, then go back to cooperating. Otherwise, mirror the "
            "opponent's last move."
        ),
    },
}
STRATEGIES_FILE = Path(__file__).with_name("llm_strategies.json")
MIN_STRATEGIES = 3  # the project requires at least 3 distinct prompt strategies
