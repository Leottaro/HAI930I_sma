"""Data generator: sweep settings, play every combination as a 1v1 match, save them all.

Each parameter takes one value, a list or a range; the generator plays the
cartesian product (times N repeats) and appends every finished match to
results_history.jsonl as its own 1v1 session, tagged with the same ``batch`` id.
"""
from __future__ import annotations

import itertools
import re
import time
from dataclasses import asdict, replace
from datetime import datetime
from typing import Any

from history import HISTORY_FILE
from matches import ensure_ready, llm_calls, run_1v1
from menus import SETTING_LABELS, check_setting, pick_from
from opponents import OPPONENTS
from runtime import RT
from terminal import ask, bold, clear_status, cyan, dim, yellow

# Settings that can be swept, in the order they are asked
# (repetitions / llm_self_play are tournament-only: every generated run is a 1v1)
NUMERIC_KEYS = ["turns", "mapek_every", "window", "poor_threshold", "history_shown", "temperature", "seed"]
MAX_VALUES = 1000  # per parameter, so a typo like 0:1:0.0001 doesn't explode
DEFAULT_SECONDS_PER_CALL = 2.0  # used for the estimate until a call has been timed

VALUES_HINT = "one value (10), a list (10, 20, 40) or a range start:stop[:step], stop included (10:50:10)"


# ── parsing ────────────────────────────────────────────────────────────────
def parse_values(raw: str, kind: type) -> list:
    """'10' · '10, 20, 40' · '10:50:10' · '0:1:0.25' (ranges and lists can be mixed)."""
    values: list = []
    for part in filter(None, re.split(r"[,\s]+", raw)):
        try:
            if ":" not in part:
                values.append(kind(part))
                continue
            bits = part.split(":")
            if len(bits) not in (2, 3):
                raise ValueError
            start, stop = kind(bits[0]), kind(bits[1])
            step = kind(bits[2]) if len(bits) == 3 else kind(1)
        except ValueError:
            raise ValueError(f"'{part}' is not a valid {kind.__name__} value, list or range") from None
        if step <= 0 or stop < start:
            raise ValueError(f"range '{part}' needs start ≤ stop and a positive step")
        n = int((stop - start) / step + 1e-9) + 1
        if n > MAX_VALUES:
            raise ValueError(f"range '{part}' has {n} values (max {MAX_VALUES})")
        values += [kind(round(start + i * step, 10)) for i in range(n)]
    return list(dict.fromkeys(values))


def ask_numeric(key: str) -> list:
    cur = getattr(RT.cfg, key)
    while True:
        raw = ask(f"  {SETTING_LABELS[key]:<44} [Enter = {cur}] > ")
        if not raw:
            return [cur]
        try:
            values = parse_values(raw, type(cur))
            for v in values:
                check_setting(key, v)
            if values:
                return values
        except ValueError as e:
            print(f"  invalid ({e})")


def ask_models() -> list[str]:
    cur = RT.cfg.model
    raw = ask(f"  {'Gemini model(s), comma-separated':<44} [Enter = {cur}] > ")
    return list(dict.fromkeys(m for m in re.split(r"[,\s]+", raw) if m)) or [cur]


def ask_adaptive() -> list[bool]:
    cur = RT.cfg.adaptive
    answers = {"y": [True], "yes": [True], "on": [True], "n": [False], "no": [False], "off": [False],
               "both": [True, False], "b": [True, False]}
    while True:
        raw = ask(f"  {'adaptive: y / n / both':<44} [Enter = {'y' if cur else 'n'}] > ").lower()
        if not raw:
            return [cur]
        if raw in answers:
            return answers[raw]
        print("  invalid choice")


def ask_repeats() -> int:
    while True:
        raw = ask(f"  {'repeat each combination N times':<44} [Enter = 1] > ")
        if not raw:
            return 1
        if raw.isdigit() and int(raw) >= 1:
            return int(raw)
        print("  must be a positive integer")


# ── estimate / display ─────────────────────────────────────────────────────
def duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {s % 3600 // 60:02d}m"


def seconds_per_call() -> tuple[float, str]:
    """Average duration of an LLM call, measured this session if possible."""
    b = RT.brain
    if b and b.calls:
        return b.seconds / b.calls, f"measured over {b.calls} calls this session"
    return DEFAULT_SECONDS_PER_CALL, "assumed: no call timed yet this session"


def show_plan(axes: dict[str, list], repeats: int, n_runs: int, calls: int) -> None:
    print(f"\n{bold('Data generator plan')}")
    swept = {k: v for k, v in axes.items() if len(v) > 1}
    fixed = {k: v[0] for k, v in axes.items() if len(v) == 1}
    for k, v in swept.items():
        print(f"  {k:<17} {cyan(', '.join(map(str, v)))}  {dim(f'({len(v)})')}")
    if fixed:
        print(dim("  fixed: " + ", ".join(f"{k}={v}" for k, v in fixed.items())))
    sizes = " × ".join(str(len(v)) for v in swept.values()) or "1"
    rep = f" × {repeats} repeats" if repeats > 1 else ""
    spc, how = seconds_per_call()
    print(
        f"\n  {sizes}{rep} = {bold(f'{n_runs} matches')} · ~{calls} LLM calls · "
        f"~{bold(duration(calls * spc))} {dim(f'({spc:.1f} s/call, {how})')}"
    )
    print(dim("  Rate limits and retries can make it longer (free tier: about 15 requests/minute)."))


# ── menu ───────────────────────────────────────────────────────────────────
def menu_generator() -> None:
    print(f"\n{bold('Data generator')}  {dim('every combination is played as a 1v1 match and saved')}")
    names = list(OPPONENTS)
    opponents = [names[i] for i in pick_from("Opponents:", names, many=True)]
    strategies = list(RT.strategies)
    picks = pick_from(
        "LLM base strategies:", [f"{n} — {s['summary']}" for n, s in RT.strategies.items()], many=True
    )

    print(f"\n{bold('Settings')}  {dim(VALUES_HINT)}")
    axes: dict[str, list[Any]] = {
        "opponent": opponents,
        "initial_strategy": [strategies[i] for i in picks],
        "model": ask_models(),
    }
    for key in NUMERIC_KEYS:
        axes[key] = ask_numeric(key)
    axes["adaptive"] = ask_adaptive()
    repeats = ask_repeats()

    keys = list(axes)
    runs = [dict(zip(keys, combo)) for combo in itertools.product(*axes.values()) for _ in range(repeats)]
    costs = [llm_calls(replace(RT.cfg, **{k: v for k, v in r.items() if k != "opponent"})) for r in runs]
    show_plan(axes, repeats, len(runs), sum(costs))
    if ask(f"\n  Play these {len(runs)} matches? [y/N] > ").lower() not in ("y", "yes"):
        print("  cancelled")
        return
    if not ensure_ready():
        return
    run_batch(runs, costs, [k for k in keys if len(axes[k]) > 1 and k not in ("opponent", "initial_strategy")])


def run_batch(runs: list[dict], costs: list[int], shown: list[str]) -> None:
    """Play every run in turn; stop at the first one that can't finish (quota, Ctrl-C ...)."""
    batch = datetime.now().strftime("gen-%Y%m%d-%H%M%S")
    saved_cfg = asdict(RT.cfg)  # restored at the end: the sweep must not change Settings
    calls0, t0 = RT.brain.calls, time.monotonic()
    done = 0
    try:
        for k, run in enumerate(runs):
            for key, value in run.items():
                if key != "opponent":
                    setattr(RT.cfg, key, value)
            used = RT.brain.calls - calls0
            spc = (time.monotonic() - t0) / used if used else seconds_per_call()[0]
            params = "  ".join(f"{key}={run[key]}" for key in shown)
            print(f"\n{bold(f'[{k + 1}/{len(runs)}]')} {params}  {dim(f'ETA {duration(sum(costs[k:]) * spc)}')}")
            if not run_1v1(OPPONENTS[run["opponent"]], ui="compact", batch=batch):
                break
            done += 1
    except KeyboardInterrupt:
        clear_status()
        print(yellow("\n  ✗ data generator interrupted"))
    finally:
        for key, value in saved_cfg.items():
            setattr(RT.cfg, key, value)
    took = duration(time.monotonic() - t0)
    msg = f"{done}/{len(runs)} matches saved to {HISTORY_FILE.name} (batch {batch}, {took})"
    print(f"\n  {bold(msg) if done == len(runs) else yellow('stopped early: ' + msg)}")
