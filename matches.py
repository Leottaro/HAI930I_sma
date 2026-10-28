"""Running a 1v1 match or a tournament."""
from __future__ import annotations

import getpass
import os
import tempfile
from pathlib import Path
from typing import Any

import axelrod as axl

from brain import Brain, LLMUnavailable
from config import ENV_FILE
from player import LLMPlayer
from runtime import RT
from terminal import bold, clear_status, dim, red, spark, yellow


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
