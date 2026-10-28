#!/usr/bin/env python3
"""
Axelrod's tournament: classic deterministic strategies vs. a self-adaptive,
LLM-guided strategy driven by a MAPE-K loop.

    pip install axelrod google-adk
    export GOOGLE_API_KEY=...        # or put it in a .env file
    python main.py

How the LLM player works
------------------------
Every round, an ADK agent plays C or D following the *active prompt strategy*
(Generous, Pragmatic, Merciless, Peacemaker ... you can edit/add them from the
menu). Every `mapek_every` rounds, a MAPE-K loop runs:

    Monitor  (python)  sliding-window score, opponent cooperation rate, trend
    Analyze  (LLM)     profiles the opponent, judges our performance
    Plan     (LLM)     picks the strategy for the next rounds, using the
                       Knowledge base (track record of each strategy so far)
    Execute  (python)  swaps the active prompt strategy
    Knowledge          opponent profile, analyses, per-strategy track record

The LLM never sees the opponent's name: it has to infer who it is facing.

Project layout
--------------
    main.py        entry point (menu loop)
    menus.py       interactive menus (1v1, tournament, strategy lab, settings)
    matches.py     running a 1v1 match / a tournament, results tables
    player.py      LLMPlayer: the Axelrod player with the MAPE-K loop
    brain.py       ADK wrapper: LLM calls, retries, quota/error reporting
    prompts.py     static prompts of the Analyze / Plan / move agents
    strategies.py  default LLM prompt strategies (edited ones: llm_strategies.json)
    records.py     Round / MatchRecord / Knowledge data classes
    display.py     live per-round display and MAPE-K trace
    terminal.py    colours, wrapped output, input helpers
    opponents.py   classic deterministic opponents
    config.py      Config dataclass + .env loading
    runtime.py     shared state (RT): config, strategies, brain, match records
"""
from __future__ import annotations

from menus import MENU, menu_1v1, menu_lab, menu_settings, menu_tournament
from terminal import ask


def main() -> None:
    print(MENU)
    actions = {"1": menu_1v1, "2": menu_tournament, "3": menu_lab, "4": menu_settings}
    while True:
        try:
            choice = ask("\n> ").lower()
            if choice in ("5", "q", "quit", "exit"):
                return
            if choice in actions:
                actions[choice]()
                print(MENU)
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return


if __name__ == "__main__":
    main()
