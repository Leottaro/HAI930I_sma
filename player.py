"""The self-adaptive LLM player (plugs into Axelrod like any other strategy)."""
from __future__ import annotations

import json
import math
import re

import axelrod as axl
from axelrod import Action

from brain import parse_json
from display import ui_mapek, ui_match_end, ui_match_start, ui_round
from prompts import ANALYZE_INSTRUCTION, PLAN_INSTRUCTION, move_instruction
from records import Knowledge, MatchRecord, Round
from runtime import RT

C, D = Action.C, Action.D


class LLMPlayer(axl.Player):
    name = "LLM MAPE-K"
    classifier = {
        "memory_depth": float("inf"),
        "stochastic": True,
        "long_run_time": True,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
        "makes_use_of": set(),
    }

    def __init__(self) -> None:
        # Axelrod calls __init__ again on reset(), i.e. before every match:
        # this is where the per-match state starts from scratch.
        super().__init__()
        self.rec: MatchRecord | None = None
        self.kb = Knowledge()
        self.current = RT.cfg.initial_strategy

    # ── helpers ────────────────────────────────────────────────────────────
    def _game(self) -> axl.Game:
        return self.match_attributes.get("game") or axl.Game()

    def _length(self) -> int | None:
        n = self.match_attributes.get("length")
        return int(n) if n and math.isfinite(n) and n > 0 else None

    def _llm(self, kind: str, name: str, instruction: str, text: str) -> str | None:
        out = RT.brain.ask(kind, name, instruction, text)
        if out is not None and self.rec is not None and RT.cfg.model not in self.rec.models:
            self.rec.models.append(str(RT.cfg.model))  # the model may change mid-match (quota)
        return out

    # ── Axelrod hooks ──────────────────────────────────────────────────────
    def update_history(self, play, coplay) -> None:
        """Called by Axelrod after every round: log it and update the display."""
        super().update_history(play, coplay)
        if self.rec is None:
            return
        mine, theirs = self._game().score((play, coplay))
        self.rec.rounds.append(
            Round(str(play), str(coplay), int(mine), int(theirs), self.current)
        )
        ui_round(self.rec)
        if len(self.rec.rounds) == self.rec.length:
            ui_match_end(self.rec)

    def strategy(self, opponent: axl.Player) -> Action:
        t = len(self.history)
        if t == 0:
            self.rec = MatchRecord(opponent.name, self._length())
            RT.records.append(self.rec)
            ui_match_start(self.rec, self.current)
        elif RT.cfg.adaptive and t % RT.cfg.mapek_every == 0:
            self._mapek(opponent)
        return self._execute_move(opponent)

    # ── MAPE-K loop ────────────────────────────────────────────────────────
    def _mapek(self, opponent: axl.Player) -> None:
        mon = self._monitor()  # M
        ana = self._analyze(mon)  # A
        plan = self._plan(mon, ana)  # P
        self._execute_plan(mon, ana, plan)  # E  (K is updated along the way)

    def _monitor(self) -> dict:
        cfg, rounds = RT.cfg, self.rec.rounds
        w = cfg.window
        win, prev = rounds[-w:], rounds[-2 * w : -w]
        avg = lambda xs: round(sum(xs) / len(xs), 2) if xs else None
        my_win = avg([r.mine for r in win])
        alert = "POOR" if my_win < cfg.poor_threshold else "GOOD" if my_win >= 2.5 else "OK"
        a, b = self.rec.total()
        return {
            "round": len(rounds),
            "total_rounds": self.rec.length,
            "current_strategy": self.current,
            "my_total": a,
            "opp_total": b,
            "window": len(win),
            "my_avg_window": my_win,
            "opp_avg_window": avg([r.theirs for r in win]),
            "previous_window_my_avg": avg([r.mine for r in prev]),
            "opp_coop_window": avg([r.opp == "C" for r in win]),
            "opp_coop_overall": avg([r.opp == "C" for r in rounds]),
            "my_coop_overall": avg([r.me == "C" for r in rounds]),
            "recent_rounds_me_then_opp": " ".join(
                r.me + r.opp for r in rounds[-cfg.history_shown :]
            ),
            "alert": alert,
        }

    def _analyze(self, mon: dict) -> dict:
        payload = {
            "monitoring": mon,
            "previous_opponent_profile": self.kb.profile,
            "track_record_by_strategy": self.rec.track_record(),
        }
        raw = self._llm("analyze", "analyzer", ANALYZE_INSTRUCTION, json.dumps(payload))
        ana = parse_json(raw)
        if ana:
            self.kb.profile = str(ana.get("opponent_profile") or self.kb.profile)
            self.kb.analyses.append(ana)
        return ana

    def _plan(self, mon: dict, ana: dict) -> dict:
        payload = {
            "analysis": ana or {"insight": "analysis unavailable"},
            "monitoring": mon,
            "current_strategy": self.current,
            "strategy_catalogue": {n: s["summary"] for n, s in RT.strategies.items()},
            "track_record_by_strategy": self.rec.track_record(),
            "rounds_remaining": (self.rec.length - mon["round"]) if self.rec.length else None,
        }
        raw = self._llm("plan", "planner", PLAN_INSTRUCTION, json.dumps(payload))
        return parse_json(raw)

    def _execute_plan(self, mon: dict, ana: dict, plan: dict) -> None:
        old, wanted = self.current, str(plan.get("strategy") or self.current).strip()
        by_key = {n.casefold(): n for n in RT.strategies}
        new = by_key.get(wanted.casefold(), old)
        if new != old:
            self.rec.switches.append((mon["round"], old, new, str(plan.get("reason", ""))))
            self.current = new
        ui_mapek(mon["round"], mon, ana, plan, old, new)

    # ── Execute: play one round with the active prompt strategy ────────────
    def _execute_move(self, opponent: axl.Player) -> Action:
        cfg, rec = RT.cfg, self.rec
        t, L = len(rec.rounds), rec.length
        a, b = rec.total()
        lines = [f"Round {t + 1}" + (f" of {L}." if L else ".")]
        if t == 0:
            lines.append("This is the first round: there is no history yet.")
        else:
            lines += [
                f"Score so far: you {a}, opponent {b}.",
                f"The opponent cooperated in {sum(r.opp == 'C' for r in rec.rounds)} of the {t} previous rounds.",
                f"Last {min(t, cfg.history_shown)} rounds (your move then the opponent's), oldest first: "
                + " ".join(r.me + r.opp for r in rec.rounds[-cfg.history_shown :]),
            ]
        lines.append("Your move (C or D)?")
        instr = move_instruction(RT.strategies[self.current]["prompt"])
        raw = self._llm("move", self.current, instr, "\n".join(lines))
        if raw:
            up = raw.upper()
            m = (
                re.search(r"\b([CD])\b", up)
                or re.match(r"\W*([CD])", up)
                or re.search(r"\b(COOPERATE|DEFECT)", up)
            )
            if m:
                return C if m.group(1)[0] == "C" else D
        # Fallback (LLM hiccup): tit-for-tat, so the match can continue.
        return opponent.history[-1] if t else C
