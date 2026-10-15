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
"""
from __future__ import annotations

import asyncio
import getpass
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import axelrod as axl
from axelrod import Action
from google.adk import Agent
from google.adk.planners import BuiltInPlanner
from google.adk.runners import InMemoryRunner
from google.genai import types

ENV_FILE = Path(__file__).with_name(".env")

try:  # optional: load GOOGLE_API_KEY from a .env file
    from dotenv import load_dotenv

    load_dotenv(ENV_FILE)
except ImportError:
    pass

# ADK/genai log a full traceback for every failed model call, even when we
# retry it: Brain reports failures itself, in one line.
logging.getLogger("google_adk").setLevel(logging.CRITICAL)
logging.getLogger("google_genai").setLevel(logging.ERROR)

C, D = Action.C, Action.D


# ════════════════════════════════════════════════════════════════════════════
#  Configuration (everything here can be changed from the "Settings" menu)
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    model: Any = "gemini-3.1-flash-lite"
    temperature: float = 0.2
    turns: int = 10  # rounds per match (keep it small: 1 LLM call per round)
    repetitions: int = 1  # matches per pairing in a tournament
    seed: int = 1
    mapek_every: int = 10  # run the MAPE-K loop every N rounds
    window: int = 5  # Monitor: size of the sliding window (rounds)
    poor_threshold: float = 1.5  # avg points/round under which we flag "POOR"
    history_shown: int = 10  # rounds of history shown to the move agent
    initial_strategy: str = "Generous"  # starts naive so a comeback is visible
    adaptive: bool = True  # False = frozen on the initial strategy (baseline)
    llm_self_play: bool = False  # let the LLM also play against a copy of itself


# Classic *deterministic* opponents (name shown in menus -> class)
OPPONENTS = {
    cls().name: cls
    for cls in (
        axl.Cooperator,
        axl.Defector,
        axl.TitForTat,
        axl.TitFor2Tats,
        axl.SuspiciousTitForTat,
        axl.Grudger,
        axl.Alternator,
        axl.Prober,
        axl.WinStayLoseShift,
    )
}

# ════════════════════════════════════════════════════════════════════════════
#  LLM prompt strategies (the thing you tinker with)
# ════════════════════════════════════════════════════════════════════════════
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


# ════════════════════════════════════════════════════════════════════════════
#  Runtime state shared with the Axelrod player (Axelrod clones/resets players,
#  so config, prompts and the LLM client live in one module-level object)
# ════════════════════════════════════════════════════════════════════════════
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


# ════════════════════════════════════════════════════════════════════════════
#  Terminal helpers
# ════════════════════════════════════════════════════════════════════════════
USE_COLOR = sys.stdout.isatty() and not os.getenv("NO_COLOR")


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if USE_COLOR else s


bold = lambda s: _c("1", s)
dim = lambda s: _c("2", s)
red = lambda s: _c("31", s)
green = lambda s: _c("32", s)
yellow = lambda s: _c("33", s)
cyan = lambda s: _c("36", s)
move_col = lambda a: green(a) if a == "C" else red(a)


def width() -> int:
    return min(shutil.get_terminal_size((100, 20)).columns, 110)


def note(tag: str, text: str) -> None:
    """Print a labelled, wrapped line (used for the MAPE-K trace)."""
    pad = "     "
    print(
        textwrap.fill(
            f"{tag} {text}",
            width=width(),
            initial_indent=pad,
            subsequent_indent=pad + "  ",
        )
    )


def clear_status() -> None:
    if USE_COLOR:
        print("\r\033[K", end="")


def ask(prompt: str) -> str:
    return input(prompt).strip()


def read_multiline(prompt: str) -> str:
    print(prompt + " (finish with an empty line)")
    lines: list[str] = []
    while True:
        line = input("  | ")
        if not line.strip():
            break
        lines.append(line.strip())
    return " ".join(lines)


def spark(values: list[float], lo: float = 0.0, hi: float = 5.0) -> str:
    bars = "▁▂▃▄▅▆▇█"
    out = ""
    for v in values:
        i = int(round((min(max(v, lo), hi) - lo) / (hi - lo) * (len(bars) - 1)))
        out += bars[i]
    return out


# ════════════════════════════════════════════════════════════════════════════
#  Records of what happens in a match (feed both the UI and the Knowledge base)
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class Round:
    me: str
    opp: str
    mine: int
    theirs: int
    strategy: str


@dataclass
class MatchRecord:
    opponent: str
    length: int | None
    rounds: list[Round] = field(default_factory=list)
    switches: list[tuple[int, str, str, str]] = field(default_factory=list)

    def total(self) -> tuple[int, int]:
        return sum(r.mine for r in self.rounds), sum(r.theirs for r in self.rounds)

    def avg(self) -> tuple[float, float]:
        n = max(len(self.rounds), 1)
        a, b = self.total()
        return a / n, b / n

    def blocks(self, size: int) -> list[float]:
        pts = [r.mine for r in self.rounds]
        return [sum(pts[i : i + size]) / len(pts[i : i + size]) for i in range(0, len(pts), size)]

    def episodes(self) -> list[tuple[str, int, int, float]]:
        """(strategy, first round, last round, avg points) for each stretch."""
        eps: list[list[Any]] = []
        for i, r in enumerate(self.rounds, 1):
            if not eps or eps[-1][0] != r.strategy:
                eps.append([r.strategy, i, i, [r.mine]])
            else:
                eps[-1][2] = i
                eps[-1][3].append(r.mine)
        return [(s, a, b, sum(p) / len(p)) for s, a, b, p in eps]

    def track_record(self) -> dict[str, dict[str, float]]:
        """Knowledge: how each strategy performed so far in THIS match."""
        out: dict[str, dict[str, float]] = {}
        for s in {r.strategy for r in self.rounds}:
            rs = [r for r in self.rounds if r.strategy == s]
            out[s] = {
                "rounds": len(rs),
                "my_avg_points": round(sum(r.mine for r in rs) / len(rs), 2),
                "opp_avg_points": round(sum(r.theirs for r in rs) / len(rs), 2),
                "opp_cooperation_rate": round(sum(r.opp == "C" for r in rs) / len(rs), 2),
            }
        return out

    def verdict(self) -> str:
        w = RT.cfg.window
        pts = [r.mine for r in self.rounds]
        if len(pts) < 2 * w:
            return "—"
        first, last = sum(pts[:w]) / w, sum(pts[-w:]) / w
        if last - first >= 0.5:
            return green("↗ comeback") if first < RT.cfg.poor_threshold else green("↗ improved")
        if first - last >= 0.5:
            return red("↘ declined")
        return yellow("→ steady")


@dataclass
class Knowledge:
    """The K of MAPE-K (the per-round history itself lives in MatchRecord)."""

    profile: str = "unknown yet"
    analyses: list[dict] = field(default_factory=list)


# ════════════════════════════════════════════════════════════════════════════
#  Brain: thin synchronous wrapper around ADK agents
# ════════════════════════════════════════════════════════════════════════════
class LLMUnavailable(RuntimeError):
    pass


APP, USER = "axelrod_mapek", "player"


def http_code(e: Exception) -> int | None:
    code = getattr(e, "code", None)  # set by google.genai's APIError
    return code if isinstance(code, int) else None


def retry_delay(e: Exception, attempt: int) -> float:
    """Seconds to wait before retrying a failed call (honours the API's hint)."""
    m = re.search(r"retry in ([\d.]+)\s*s", str(e)) or re.search(
        r"retryDelay['\"]?:\s*['\"](\d+(?:\.\d+)?)s", str(e)
    )
    if m:
        return min(float(m.group(1)) + 1, 90)
    code = http_code(e)
    if code == 429:  # rate limited
        return 15 * (attempt + 1)
    if code in (500, 502, 503, 504):  # overloaded
        return 5 * (attempt + 1)
    return 1.5 * (attempt + 1)


def quota_info(e: Exception) -> dict | None:
    """Which quota a 429 hit, e.g. {"per": "day", "what": "requests", "limit": 20, "model": ...}."""
    if http_code(e) != 429:
        return None
    text = str(e)
    qid = re.search(r"quotaId['\"]?:\s*['\"]([^'\"]+)", text)
    qid = qid.group(1) if qid else ""
    per = "day" if "PerDay" in qid else "minute" if "PerMinute" in qid else None
    what = "tokens" if "Token" in qid else "requests"
    limit = re.search(r"limit: (\d+)", text)
    model = re.search(r"model: ([\w.-]+)", text)
    return {
        "per": per,
        "what": what,
        "limit": int(limit.group(1)) if limit else None,
        "model": model.group(1) if model else str(RT.cfg.model),
        "free": "FreeTier" in qid or "free_tier" in text,
    }


def short_error(e: Exception) -> str:
    q = quota_info(e)
    if q:
        tier = "free tier" if q["free"] else "your plan"
        if q["limit"] == 0:
            return f"429: {q['model']} is not available on the {tier} (paid model)"
        per = f"/{q['per']}" if q["per"] else ""
        limit = f"{q['limit']} {q['what']}{per}" if q["limit"] is not None else "quota"
        return f"429: {limit} exceeded for {q['model']} ({tier})"
    m = re.search(r"""'message':\s*(['"])(.*?)\1""", str(e))
    msg = m.group(2).replace("\\n", " ") if m else str(e).strip().splitlines()[-1] if str(e).strip() else ""
    code = http_code(e)
    return f"{type(e).__name__}{f' {code}' if code else ''}: {msg[:300]}"


class Brain:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()  # one loop for the whole program
        self._runners: dict[tuple, InMemoryRunner] = {}
        self.calls = 0
        self.failures = 0  # consecutive failed calls
        self.last_error = ""

    def runner(self, kind: str, name: str, instruction: str) -> InMemoryRunner:
        key = (kind, name, instruction, str(RT.cfg.model), RT.cfg.temperature)
        if key not in self._runners:
            slug = re.sub(r"\W", "_", f"{kind}_{name}")
            kwargs: dict[str, Any] = {}
            model = RT.cfg.model
            if isinstance(model, str) and "2.5-flash" in model:
                # No hidden "thinking" tokens: faster and cheaper for a live demo.
                # (2.5 Pro and Gemini 3+ reject a zero budget.)
                kwargs["planner"] = BuiltInPlanner(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
            agent = Agent(
                name=slug,
                model=model,
                # A callable instruction is used verbatim: no {placeholder}
                # substitution, so prompts may safely contain braces.
                instruction=lambda ctx, text=instruction: text,
                generate_content_config=types.GenerateContentConfig(
                    temperature=RT.cfg.temperature
                ),
                **kwargs,
            )
            self._runners[key] = InMemoryRunner(agent=agent, app_name=APP)
        return self._runners[key]

    async def _ask(self, runner: InMemoryRunner, text: str) -> str:
        session = await runner.session_service.create_session(app_name=APP, user_id=USER)
        message = types.Content(role="user", parts=[types.Part(text=text)])
        out = ""
        try:
            async for ev in runner.run_async(
                user_id=USER, session_id=session.id, new_message=message
            ):
                if ev.is_final_response() and ev.content and ev.content.parts:
                    out += "".join(p.text for p in ev.content.parts if p.text and not p.thought)
        finally:
            await runner.session_service.delete_session(
                app_name=APP, user_id=USER, session_id=session.id
            )
        return out.strip()

    def _cancel_pending(self) -> None:
        """After a Ctrl-C, don't leave a half-done call to resume later."""
        tasks = asyncio.all_tasks(self.loop)
        for t in tasks:
            t.cancel()
        if tasks:
            self.loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))

    def ask(self, runner: InMemoryRunner, text: str) -> str | None:
        err: Exception | None = None
        for attempt in range(3):
            try:
                out = self.loop.run_until_complete(self._ask(runner, text))
                self.calls += 1
                self.failures = 0
                return out
            except KeyboardInterrupt:
                self._cancel_pending()
                raise
            except Exception as e:  # network, quota, bad key ...
                err = e
                code = http_code(e)
                if code is not None and 400 <= code < 500 and code not in (408, 429):
                    # Bad key, unknown model, invalid request: retrying won't help.
                    raise LLMUnavailable(short_error(e)) from e
                q = quota_info(e)
                if q and (q["per"] == "day" or q["limit"] == 0):
                    # Daily quota used up / paid-only model: waiting a minute won't help.
                    hint = (
                        "pick a free model in Settings"
                        if q["limit"] == 0
                        else "daily quota resets at midnight Pacific time; pick another model in Settings"
                    )
                    raise LLMUnavailable(f"{short_error(e)}: {hint}") from e
                if attempt < 2:
                    delay = retry_delay(e, attempt)
                    clear_status()
                    print(dim(f"  ! LLM call failed ({short_error(e)}), retry in {delay:.0f}s"))
                    time.sleep(delay)
        self.failures += 1
        self.last_error = short_error(err)
        clear_status()
        print(yellow(f"  ! LLM call given up ({self.last_error})"))
        if self.failures >= 3:
            raise LLMUnavailable(self.last_error)
        return None


def parse_json(text: str | None) -> dict:
    """First JSON object found in the reply (models like to add prose/fences)."""
    if not text:
        return {}
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            data, _ = decoder.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return {}


# ════════════════════════════════════════════════════════════════════════════
#  Prompts of the agents (static parts)
# ════════════════════════════════════════════════════════════════════════════
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


# ════════════════════════════════════════════════════════════════════════════
#  Live display
# ════════════════════════════════════════════════════════════════════════════
def ui_match_start(rec: MatchRecord, strategy: str) -> None:
    clear_status()
    n = f"{rec.length} rounds" if rec.length else "unknown length"
    print(f"\n{bold('▶ LLM MAPE-K')} vs {bold(rec.opponent)}  {dim(f'({n}, starts as {strategy})')}")


def ui_round(rec: MatchRecord) -> None:
    r = rec.rounds[-1]
    a, b = rec.total()
    n, L = len(rec.rounds), rec.length
    if RT.ui == "rounds":
        print(
            f"  r{n:>2}/{L or '?'}  LLM {move_col(r.me)} · Opp {move_col(r.opp)}"
            f"   {r.mine:+d}/{r.theirs:+d}   total {a:>3} – {b:<3}  {dim('[' + r.strategy + ']')}"
        )
    elif USE_COLOR:  # compact: one live-updating status line
        print(f"\r\033[K  r{n}/{L or '?'} {dim('[' + r.strategy + ']')} {a} – {b}", end="", flush=True)


def ui_match_end(rec: MatchRecord) -> None:
    clear_status()
    a, b = rec.total()
    outcome = green("WIN") if a > b else red("LOSS") if a < b else yellow("DRAW")
    pa, pb = rec.avg()
    first_last = ""
    if len(rec.rounds) >= 2 * RT.cfg.window:
        pts = [r.mine for r in rec.rounds]
        w = RT.cfg.window
        first_last = f"  first {w}: {sum(pts[:w]) / w:.1f} → last {w}: {sum(pts[-w:]) / w:.1f}"
    print(f"  {bold('■')} {outcome}  LLM {a} – {b} {rec.opponent}   avg {pa:.2f} / {pb:.2f}{first_last}  {rec.verdict()}")
    eps = rec.episodes()
    if len(eps) > 1:
        line = "  →  ".join(f"{s} r{x}-{y} ({p:.1f})" for s, x, y, p in eps)
        print(dim(f"    timeline: {line}"))


def ui_mapek(round_no: int, mon: dict, ana: dict, plan: dict, old: str, new: str) -> None:
    clear_status()
    print(f"  {cyan('⟳ MAPE-K')} {dim(f'@ round {round_no}')}")
    alert = {"POOR": red, "OK": yellow, "GOOD": green}[mon["alert"]](mon["alert"])
    note(
        cyan("M"),
        f"last {mon['window']} rounds: {mon['my_avg_window']} pts/round vs "
        f"{mon['opp_avg_window']} for opp, opp cooperated {int(mon['opp_coop_window'] * 100)}%  → {alert}",
    )
    if ana:
        note(cyan("A"), f"{ana.get('opponent_profile') or '?'}. {ana.get('insight') or ''}")
    else:
        note(cyan("A"), dim("analysis unavailable"))
    arrow = f"{old} → {bold(new)}" if new != old else f"keep {bold(old)}"
    note(cyan("P/E"), f"{arrow}  {dim(str(plan.get('reason') or ''))}")


# ════════════════════════════════════════════════════════════════════════════
#  The self-adaptive LLM player (plugs into Axelrod like any other strategy)
# ════════════════════════════════════════════════════════════════════════════
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
        return RT.brain.ask(RT.brain.runner(kind, name, instruction), text)

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


# ════════════════════════════════════════════════════════════════════════════
#  Running a 1v1 match or a tournament
# ════════════════════════════════════════════════════════════════════════════
def ensure_ready() -> bool:
    if not os.getenv("GOOGLE_API_KEY") and isinstance(RT.cfg.model, str):
        api_key = getpass.getpass("Google API key (input hidden): ").strip()
        if not api_key:
            print(red("  ✗ No API key provided.\n"))
            return False
        os.environ["GOOGLE_API_KEY"] = api_key
        try:
            existing = ENV_FILE.read_text() if ENV_FILE.exists() else ""
            lines = [line for line in existing.splitlines() if not line.startswith("GOOGLE_API_KEY=")]
            lines.append(f"GOOGLE_API_KEY={api_key}")
            ENV_FILE.write_text("\n".join(lines) + "\n")
        except OSError as e:
            print(yellow(f"  ! could not save the key to {ENV_FILE} ({e}), using it for this session only"))
    if RT.brain is None:
        RT.brain = Brain()
    RT.brain.failures = 0
    RT.records.clear()
    return True


def estimate(n_matches: int) -> int:
    cfg = RT.cfg
    cycles = (cfg.turns // cfg.mapek_every) * 2 if cfg.adaptive else 0
    return n_matches * (cfg.turns + cycles)


def run_1v1(opp_cls: type) -> None:
    if not ensure_ready():
        return
    RT.ui = "rounds"
    cfg = RT.cfg
    print(dim(f"  ~{estimate(1)} LLM calls"))
    match = axl.Match((LLMPlayer(), opp_cls()), turns=cfg.turns, seed=cfg.seed)
    try:
        match.play()
    except LLMUnavailable as e:
        clear_status()
        print(red(f"\n  ✗ LLM unreachable, match aborted: {e}"))
        return
    except KeyboardInterrupt:
        clear_status()
        print(yellow("\n  ✗ match interrupted"))
        return
    print(dim(f"\n  LLM calls: {RT.brain.calls}"))


def run_tournament(opp_classes: list[type]) -> None:
    if not ensure_ready():
        return
    RT.ui = "compact"
    cfg = RT.cfg
    players = [LLMPlayer()] + [c() for c in opp_classes]
    n = len(players)
    edges = None
    if not cfg.llm_self_play:  # classic round robin, minus LLM-vs-LLM
        edges = [(i, j) for i in range(n) for j in range(i, n) if (i, j) != (0, 0)]
    llm_matches = (n - 1 + (1 if cfg.llm_self_play else 0)) * cfg.repetitions
    print(dim(f"  {llm_matches} LLM matches · ~{estimate(llm_matches)} LLM calls (matches between classic strategies are free)"))
    tournament = axl.Tournament(
        players, turns=cfg.turns, repetitions=cfg.repetitions, edges=edges, seed=cfg.seed
    )
    # Axelrod's own temp file is not removed when a match raises: use ours.
    with tempfile.TemporaryDirectory() as tmp:
        try:
            results = tournament.play(
                progress_bar=False, filename=str(Path(tmp) / "interactions.csv")
            )
        except LLMUnavailable as e:
            clear_status()
            print(red(f"\n  ✗ LLM unreachable, tournament aborted: {e}"))
            return
        except KeyboardInterrupt:
            clear_status()
            print(yellow("\n  ✗ tournament interrupted"))
            return
    show_results(results)


def show_results(results: Any) -> None:
    print(f"\n{bold('═══ Tournament ranking ═══')}")
    for rank, idx in enumerate(results.ranking, 1):
        name = results.ranked_names[rank - 1]
        scores = results.normalised_scores[idx]
        mean = sum(scores) / len(scores)
        tag = bold(name) if name == LLMPlayer.name else name
        print(f"  {rank:>2}. {tag:<{24 + (len(tag) - len(name))}} {mean:5.2f} pts/round")
    print(f"\n{bold('═══ LLM MAPE-K, match by match ═══')}")
    for rec in RT.records:
        a, b = rec.avg()
        trend = spark(rec.blocks(RT.cfg.window))
        sw = ", ".join(f"r{r}: {o}→{n}" for r, o, n, _ in rec.switches) or "no switch"
        print(f"  vs {rec.opponent:<22} {a:4.2f} / {b:4.2f}  {trend:<8} {rec.verdict():<22} {dim(sw)}")
    print(dim(f"\n  (trend = LLM points per {RT.cfg.window}-round block, scale 0–5)  ·  LLM calls: {RT.brain.calls}"))


# ════════════════════════════════════════════════════════════════════════════
#  Menus
# ════════════════════════════════════════════════════════════════════════════
def pick_from(title: str, names: list[str], many: bool = False) -> list[int]:
    print(f"\n{bold(title)}")
    for i, n in enumerate(names, 1):
        print(f"  {i:>2}) {n}")
    hint = "numbers separated by commas, Enter = all" if many else "number"
    while True:
        raw = ask(f"  [{hint}] > ")
        if many and not raw:
            return list(range(len(names)))
        try:
            picks = list(dict.fromkeys(int(x) - 1 for x in re.split(r"[,\s]+", raw) if x))
            if picks and all(0 <= p < len(names) for p in picks) and (many or len(picks) == 1):
                return picks
        except ValueError:
            pass
        print("  invalid choice")


def menu_1v1() -> None:
    names = list(OPPONENTS)
    (i,) = pick_from("Opponent for the LLM:", names)
    run_1v1(OPPONENTS[names[i]])


def menu_tournament() -> None:
    names = list(OPPONENTS)
    picks = pick_from("Opponents in the tournament:", names, many=True)
    run_tournament([OPPONENTS[names[i]] for i in picks])


def show_strategies() -> None:
    print(f"\n{bold('LLM prompt strategies')}  (initial: {RT.cfg.initial_strategy})")
    for i, (name, s) in enumerate(RT.strategies.items(), 1):
        print(f"  {i}) {bold(name)} — {s['summary']}")


def menu_lab() -> None:
    while True:
        show_strategies()
        print("  [v]iew prompt  [e]dit  [a]dd  [d]elete  [r]eset to defaults  [b]ack")
        choice = ask("  > ").lower()
        names = list(RT.strategies)
        if choice in ("b", ""):
            return
        if choice in ("v", "e", "d"):
            (i,) = pick_from("Which strategy?", names)
            name = names[i]
            if choice == "v":
                print(textwrap.fill(RT.strategies[name]["prompt"], width(), initial_indent="  ", subsequent_indent="  "))
            elif choice == "e":
                print(textwrap.fill(RT.strategies[name]["prompt"], width(), initial_indent="  current: ", subsequent_indent="           "))
                text = read_multiline("  New prompt (empty = keep)")
                if text:
                    RT.strategies[name]["prompt"] = text
                summary = ask("  New one-line summary (Enter = keep): ")
                if summary:
                    RT.strategies[name]["summary"] = summary
                RT.save_strategies()
            elif len(names) <= MIN_STRATEGIES:
                print(f"  need at least {MIN_STRATEGIES} strategies")
            else:
                del RT.strategies[name]
                if RT.cfg.initial_strategy == name:
                    RT.cfg.initial_strategy = next(iter(RT.strategies))
                RT.save_strategies()
        elif choice == "a":
            name = ask("  Name: ")
            if name and name.casefold() not in {n.casefold() for n in RT.strategies}:
                prompt = read_multiline("  Prompt (tell the LLM when to play C or D)")
                if prompt:
                    RT.strategies[name] = {"summary": ask("  One-line summary: ") or name, "prompt": prompt}
                    RT.save_strategies()
            else:
                print("  empty or already existing name")
        elif choice == "r":
            RT.reset_strategies()
            print("  defaults restored")


def menu_settings() -> None:
    cfg = RT.cfg
    labels = {
        "model": "Gemini model",
        "temperature": "temperature",
        "turns": "rounds per match",
        "repetitions": "repetitions per pairing (tournament)",
        "seed": "random seed",
        "mapek_every": "run MAPE-K every N rounds",
        "window": "Monitor window (rounds)",
        "poor_threshold": "'POOR' threshold (avg pts/round)",
        "history_shown": "rounds of history shown to the move agent",
        "initial_strategy": "initial prompt strategy",
        "adaptive": "adaptive (MAPE-K on; off = frozen baseline)",
        "llm_self_play": "LLM plays a copy of itself in tournaments",
    }
    keys = list(labels)
    while True:
        print(f"\n{bold('Settings')}")
        for i, k in enumerate(keys, 1):
            print(f"  {i:>2}) {labels[k]:<46} {cyan(str(getattr(cfg, k)))}")
        raw = ask("  Change which? (Enter = back) > ")
        if not raw:
            return
        if not raw.isdigit() or not 1 <= int(raw) <= len(keys):
            print("  invalid choice")
            continue
        key = keys[int(raw) - 1]
        cur = getattr(cfg, key)
        val = ask(f"  New value for '{key}' > ")
        try:
            if key == "initial_strategy":
                if val not in RT.strategies:
                    raise ValueError(f"choose one of: {', '.join(RT.strategies)}")
                new: Any = val
            elif isinstance(cur, bool):
                new = val.lower() in ("y", "yes", "true", "1", "on")
            elif isinstance(cur, (int, float)):
                new = type(cur)(val)
                if not math.isfinite(new):
                    raise ValueError("must be a finite number")
                if key == "temperature" and not 0 <= new <= 2:
                    raise ValueError("must be between 0 and 2")
                if key == "seed" and not 0 <= new < 2**32:
                    raise ValueError("must be between 0 and 2**32 - 1")
                if new <= 0 and key not in ("seed", "temperature"):
                    raise ValueError("must be positive")
            else:
                new = val or cur
            setattr(cfg, key, new)
        except ValueError as e:
            print(f"  invalid value ({e})")


MENU = f"""
{bold('Axelrod tournament · deterministic strategies vs self-adaptive LLM (MAPE-K)')}
  1) 1v1 match        LLM vs one classic strategy (round by round)
  2) Tournament       LLM + classic strategies, round robin
  3) Strategy lab     view / edit / add the LLM prompt strategies
  4) Settings         rounds, MAPE-K period, model, adaptive on/off ...
  5) Quit
"""


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