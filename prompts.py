"""Prompts of the agents (static parts)."""
from __future__ import annotations

import axelrod as axl


def payoff_text() -> str:
    r, p, s, t = axl.Game().RPST()
    return (
        "Iterated Prisoner's Dilemma. Each round you and the opponent each play C "
        f"(cooperate) or D (defect). Points: both C = {r} each; both D = {p} each; "
        f"you D / they C = {t} for you and {s} for them; you C / they D = {s} for "
        f"you and {t} for them."
    )


def move_instruction(strategy_prompt: str) -> str:
    return (
        f"{payoff_text()}\n\n"
        "YOUR CURRENT STRATEGY (follow it faithfully):\n"
        f"{strategy_prompt}\n\n"
        "You will be given the state of the match. Answer with exactly one "
        "letter: C or D. No other text."
    )


ANALYZE_INSTRUCTION = (
    f"{payoff_text()}\n\n"
    "You are the ANALYZE stage of a MAPE-K loop that controls a Prisoner's "
    "Dilemma player. You receive monitoring data (JSON) about the match so far. "
    "Diagnose: (1) the opponent's behavioural profile (e.g. unconditional "
    "defector, unconditional cooperator, tit-for-tat-like, grudger, alternator, "
    "prober...), based on how its moves react to ours; (2) whether OUR "
    "performance is good, ok or poor *relative to what is achievable against "
    "this kind of opponent* (mutual defection = 1 point can be the best possible "
    "result against an unconditional defector). Be concrete and brief. Reply "
    'with JSON only: {"opponent_profile": "...", "performance": "good|ok|poor", '
    '"insight": "one or two sentences"}'
)

PLAN_INSTRUCTION = (
    f"{payoff_text()}\n\n"
    "You are the PLAN stage of a MAPE-K loop that controls a Prisoner's Dilemma "
    "player. You receive: the analysis, the monitoring data, the catalogue of "
    "available prompt strategies, the current strategy and the track record "
    "(Knowledge) of every strategy tried so far in this match. Pick the "
    "strategy to use for the next rounds. Keep the current one if it works or "
    "if there is too little evidence (fewer than ~3 rounds); switch when "
    "performance is poor and another strategy is likely to do better, or when "
    "one has clearly outperformed the current one. Consider the number of "
    "remaining rounds. Reply with JSON only: "
    '{"strategy": "<exact name from the catalogue>", "reason": "one short sentence"}'
)
