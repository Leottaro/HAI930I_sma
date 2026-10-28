"""Interactive terminal menus."""
from __future__ import annotations

import math
import re
import textwrap
from typing import Any

from matches import run_1v1, run_tournament
from opponents import OPPONENTS
from runtime import RT
from strategies import MIN_STRATEGIES
from terminal import ask, bold, cyan, read_multiline, width

MENU = f"""
{bold('Axelrod tournament · deterministic strategies vs self-adaptive LLM (MAPE-K)')}
  1) 1v1 match        LLM vs one classic strategy (round by round)
  2) Tournament       LLM + classic strategies, round robin
  3) Strategy lab     view / edit / add the LLM prompt strategies
  4) Settings         rounds, MAPE-K period, model, adaptive on/off ...
  5) Results history  browse every saved 1v1 / tournament, export to CSV
  6) Data generator   sweep settings, run every combination as a 1v1 match
  7) Quit
"""

SETTING_LABELS = {
    "model": "Gemini model",
    "auto_switch_model": "switch to another free model on daily quota",
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


def check_setting(key: str, value: float) -> None:
    """Raise ValueError if a numeric setting is out of range."""
    if not math.isfinite(value):
        raise ValueError("must be a finite number")
    if key == "temperature" and not 0 <= value <= 2:
        raise ValueError("must be between 0 and 2")
    if key == "seed" and not 0 <= value < 2**32:
        raise ValueError("must be between 0 and 2**32 - 1")
    if value <= 0 and key not in ("seed", "temperature"):
        raise ValueError("must be positive")


def pick_from(
    title: str, names: list[str], many: bool = False, default: int | None = None
) -> list[int]:
    """Numbered prompt. `default` (0-based index) is chosen when the user just presses Enter."""
    print(f"\n{bold(title)}")
    for i, n in enumerate(names, 1):
        print(f"  {i:>2}) {n}")
    hint = "numbers separated by commas, Enter = all" if many else "number"
    if default is not None:
        hint += f", Enter = {default + 1}"
    while True:
        raw = ask(f"  [{hint}] > ")
        if many and not raw:
            return list(range(len(names)))
        if default is not None and not raw:
            return [default]
        try:
            picks = list(dict.fromkeys(int(x) - 1 for x in re.split(r"[,\s]+", raw) if x))
            if picks and all(0 <= p < len(names) for p in picks) and (many or len(picks) == 1):
                return picks
        except ValueError:
            pass
        print("  invalid choice")


def menu_1v1() -> None:
    names = list(OPPONENTS)

    # Base strategy: the prompt strategy the LLM starts the match with
    # (the MAPE-K loop may switch away from it later). Enter keeps the current one.
    strategies = list(RT.strategies)
    labels = [f"{n} — {s['summary']}" for n, s in RT.strategies.items()]
    (i,) = pick_from(
        f"LLM base strategy (currently: {RT.cfg.initial_strategy}):",
        labels,
        default=strategies.index(RT.cfg.initial_strategy),
    )
    RT.cfg.initial_strategy = strategies[i]  # also what Settings shows as "initial prompt strategy"

    (j,) = pick_from("Opponent for the LLM:", names)

    run_1v1(OPPONENTS[names[j]])


def menu_tournament() -> None:
    names = list(OPPONENTS)
    
    # Base strategy: the prompt strategy the LLM starts the match with
    # (the MAPE-K loop may switch away from it later). Enter keeps the current one.
    strategies = list(RT.strategies)
    labels = [f"{n} — {s['summary']}" for n, s in RT.strategies.items()]
    (i,) = pick_from(
        f"LLM base strategy (currently: {RT.cfg.initial_strategy}):",
        labels,
        default=strategies.index(RT.cfg.initial_strategy),
    )
    RT.cfg.initial_strategy = strategies[i]  # also what Settings shows as "initial prompt strategy"
    
    picks = pick_from("Opponents in the tournament:", names, many=True)
    run_tournament([OPPONENTS[names[j]] for j in picks])


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
    labels = SETTING_LABELS
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
                check_setting(key, new)
            else:
                new = val or cur
            setattr(cfg, key, new)
        except ValueError as e:
            print(f"  invalid value ({e})")
