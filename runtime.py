"""Runtime state shared with the Axelrod player.

Axelrod clones/resets players, so config, prompts and the LLM client live in
one module-level object (``RT``) that every other module imports.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from config import Config
from strategies import DEFAULT_STRATEGIES, MIN_STRATEGIES, STRATEGIES_FILE

if TYPE_CHECKING:  # only for type hints: avoids circular imports at runtime
    from brain import Brain
    from records import MatchRecord


class Runtime:
    def __init__(self) -> None:
        self.cfg = Config()
        self.strategies: dict[str, dict[str, str]] = {}
        self.records: list["MatchRecord"] = []  # one per LLM match played
        self.ui = "rounds"  # "rounds" (1v1: every round) | "compact" (tournament)
        self.brain: "Brain | None" = None
        self.load_strategies()

    def load_strategies(self) -> None:
        self.strategies = json.loads(json.dumps(DEFAULT_STRATEGIES))
        if STRATEGIES_FILE.exists():
            try:
                saved = json.loads(STRATEGIES_FILE.read_text(encoding="utf-8"))
                valid = isinstance(saved, dict) and all(
                    isinstance(n, str) and n.strip()
                    and isinstance(s, dict)
                    and isinstance(s.get("summary"), str)
                    and isinstance(s.get("prompt"), str) and s["prompt"].strip()
                    for n, s in saved.items()
                )
                if valid and len(saved) >= MIN_STRATEGIES:
                    self.strategies = saved
                else:
                    print("  ! llm_strategies.json is invalid, using defaults")
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                print("  ! could not read llm_strategies.json, using defaults")
        if self.cfg.initial_strategy not in self.strategies:
            self.cfg.initial_strategy = next(iter(self.strategies))

    def save_strategies(self) -> None:
        STRATEGIES_FILE.write_text(
            json.dumps(self.strategies, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def reset_strategies(self) -> None:
        STRATEGIES_FILE.unlink(missing_ok=True)
        self.load_strategies()


RT = Runtime()
