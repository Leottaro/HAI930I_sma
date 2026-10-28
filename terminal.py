"""Terminal helpers: colours, wrapped output, input prompts, sparklines."""
from __future__ import annotations

import os
import shutil
import sys
import textwrap

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
