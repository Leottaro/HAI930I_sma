"""Browse the saved history: sessions, every match, round-by-round replays, CSV export."""
from __future__ import annotations

import textwrap
from typing import Any

from history import CSV_FILE, HISTORY_FILE, export_csv, load_sessions
from records import verdict_text
from terminal import ask, bold, cyan, dim, green, move_col, note, red, width, yellow

RECENT_SESSIONS = 20  # rows shown by default in the sessions list ([f] shows all)
RECENT_MATCHES = 30  # same for the "every match" list
OUTCOME_COLOUR = {"win": green, "loss": red, "draw": yellow}

# Column header shared by every table of matches (see match_cells)
MATCH_HEAD = (
    f"{'opponent':<22}  {'start':<12}  {'score':>7}  {'LLM / opp':>11}  {'result':<6}  {'trend':<11}  switches"
)


# ── small formatting helpers ───────────────────────────────────────────────
def cell(text: Any, w: int) -> str:
    """Fixed-width plain-text cell (cut with an ellipsis when too long)."""
    s = str(text)
    if len(s) > w:
        s = s[: w - 1] + "…"
    return f"{s:<{w}}"


def when(session: dict) -> str:
    return str(session.get("timestamp", ""))[:16].replace("T", " ")


def plural(n: int, word: str, suffix: str = "s") -> str:
    return f"{n} {word}{'' if n == 1 else suffix}"


def llm_avg(session: dict) -> float | None:
    ms = session["matches"]
    return sum(m.get("llm_avg", 0) for m in ms) / len(ms) if ms else None


def summary(session: dict) -> str:
    """One-line description of a session (plain text)."""
    ms = session["matches"]
    if session.get("kind") == "1v1" and ms:
        m = ms[0]
        return f"vs {m['opponent']} · {str(m.get('outcome', '')).upper()} {m['llm_score']}–{m['opp_score']}"
    ranking = session.get("ranking") or []
    llm = next((r for r in ranking if r.get("is_llm")), None)
    if session.get("status") == "complete" and llm:
        return f"LLM ranked {llm['rank']}/{len(ranking)} · {plural(len(ms), 'LLM match', 'es')}"
    return f"aborted · {plural(len(ms), 'LLM match', 'es')} played"


def match_cells(m: dict) -> str:
    """One table row for a match, from the opponent column onwards."""
    outcome = m.get("outcome", "draw")
    result = OUTCOME_COLOUR.get(outcome, dim)(cell(outcome.upper(), 6))
    score = f"{m.get('llm_score', 0)}–{m.get('opp_score', 0)}"
    avg = f"{m.get('llm_avg', 0):.2f} / {m.get('opp_avg', 0):.2f}"
    switches = ", ".join(f"r{x['round']}: {x['from']}→{x['to']}" for x in m.get("switches", []))
    return (
        f"{cell(m.get('opponent', '?'), 22)}  {cell(m.get('start_strategy', '?'), 12)}  "
        f"{score:>7}  {avg:>11}  {result}  {verdict_text(m.get('trend'), 11)}  {dim(switches or 'no switch')}"
    )


# ── views ──────────────────────────────────────────────────────────────────
def print_sessions(sessions: list[dict], show_all: bool) -> None:
    shown = sessions if show_all else sessions[-RECENT_SESSIONS:]
    head = f"{plural(len(sessions), 'session')} · {HISTORY_FILE.name}"
    print(f"\n{bold('Saved results')}  {dim(head)}")
    print(dim(f"  {'#':>3}  {cell('date', 16)}  {cell('kind', 10)}  {'LLM avg':>7}  {cell('model', 21)}  result"))
    room = max(20, width() - 69)
    for s in shown:
        avg = llm_avg(s)
        avg_txt = f"{avg:.2f}" if avg is not None else "—"
        model = s.get("config", {}).get("model", "?")
        print(
            f"  {s['id']:>3}  {cell(when(s), 16)}  {cell(s.get('kind', '?'), 10)}  "
            f"{avg_txt:>7}  {cell(model, 21)}  {cell(summary(s), room).rstrip()}"
        )
    if len(shown) < len(sessions):
        print(dim(f"  … {plural(len(sessions) - len(shown), 'older session')} hidden ([f] shows all)"))


def print_all_matches(sessions: list[dict], show_all: bool) -> None:
    rows = [(s, no, m) for s in sessions for no, m in enumerate(s["matches"], 1)]
    shown = rows if show_all else rows[-RECENT_MATCHES:]
    print(f"\n{bold('Every saved match')}  {dim(plural(len(rows), 'LLM match', 'es'))}")
    print(dim(f"  {'sess':>4}  {cell('date', 16)}  {MATCH_HEAD}"))
    for s, _no, m in shown:
        print(f"  {'#' + str(s['id']):>4}  {cell(when(s), 16)}  {match_cells(m)}")
    if len(shown) < len(rows):
        print(dim(f"  … {plural(len(rows) - len(shown), 'older match', 'es')} hidden ([f] shows all)"))


def show_match(s: dict, m: dict, no: int) -> None:
    """Round-by-round replay of one saved match."""
    title = f"Session #{s['id']} · match {no} · LLM vs {m['opponent']}"
    print(f"\n{bold(title)}")
    length = m.get("length") or len(m["rounds"])
    a = b = 0
    for n, r in enumerate(m["rounds"], 1):
        a, b = a + r["mine"], b + r["theirs"]
        tag = dim("[" + r["strategy"] + "]")
        print(
            f"  r{n:>2}/{length}  LLM {move_col(r['me'])} · Opp {move_col(r['opp'])}"
            f"   {r['mine']:+d}/{r['theirs']:+d}   total {a:>3} – {b:<3}  {tag}"
        )
    outcome = m.get("outcome", "draw")
    colour = OUTCOME_COLOUR.get(outcome, dim)
    print(
        f"  {bold('■')} {colour(outcome.upper())}  LLM {m['llm_score']} – {m['opp_score']} {m['opponent']}"
        f"   avg {m['llm_avg']:.2f} / {m['opp_avg']:.2f}  {verdict_text(m.get('trend'))}"
    )
    episodes = m.get("episodes", [])
    if len(episodes) > 1:
        line = "  →  ".join(f"{e['strategy']} r{e['first']}-{e['last']} ({e['avg']:.1f})" for e in episodes)
        print(dim(f"    timeline: {line}"))
    for x in m.get("switches", []):
        reason = f" — {x['reason']}" if x.get("reason") else ""
        note(cyan("↻"), f"r{x['round']}: {x['from']} → {x['to']}{reason}")


def show_session(s: dict) -> None:
    cfg = s.get("config", {})
    ms = s["matches"]
    status = "" if s.get("status") == "complete" else yellow("  (aborted: only the finished matches were kept)")
    print(f"\n{bold('Session #' + str(s['id']))} · {s.get('kind')} · {when(s)}{status}")
    mode = (
        f"MAPE-K every {cfg.get('mapek_every')} rounds (window {cfg.get('window')})"
        if cfg.get("adaptive")
        else "adaptive off (frozen strategy)"
    )
    settings = (
        f"{cfg.get('model')} · temperature {cfg.get('temperature')} · {cfg.get('turns')} rounds · "
        f"seed {cfg.get('seed')} · {mode} · base strategy: {cfg.get('initial_strategy')}"
    )
    print(dim(textwrap.fill(settings, width(), initial_indent="  ", subsequent_indent="  ")))

    ranking = s.get("ranking")
    if ranking:
        print(f"\n  {bold('Ranking')}")
        for row in ranking:
            name = row["name"]
            tag = bold(name) if row.get("is_llm") else name
            print(f"   {row['rank']:>2}. {tag:<{24 + (len(tag) - len(name))}} {row['score']:5.2f} pts/round")

    print(f"\n  {bold('LLM matches')}")
    print(dim(f"  {'#':>3}  {MATCH_HEAD}"))
    for no, m in enumerate(ms, 1):
        print(f"  {no:>3}  {match_cells(m)}")

    while True:
        raw = ask("\n  Match number for the round-by-round replay (Enter = back) > ")
        if not raw:
            return
        if raw.isdigit() and 1 <= int(raw) <= len(ms):
            show_match(s, ms[int(raw) - 1], int(raw))
        else:
            print("  invalid choice")


# ── menu ───────────────────────────────────────────────────────────────────
def menu_history() -> None:
    show_all = False
    while True:
        sessions, skipped = load_sessions()
        if not sessions:
            print(f"\n  No saved results yet ({HISTORY_FILE.name}): play a 1v1 or a tournament first.")
            return
        print_sessions(sessions, show_all)
        if skipped:
            print(yellow(f"  ! {plural(skipped, 'unreadable line')} in {HISTORY_FILE.name} ignored"))
        toggle = "show recent only" if show_all else "show full lists"
        print(f"  [number] open session  [a]ll matches  [e]xport CSV  [f] {toggle}  [b]ack")
        choice = ask("  > ").lower()
        if choice in ("b", ""):
            return
        if choice == "f":
            show_all = not show_all
        elif choice == "a":
            print_all_matches(sessions, show_all)
        elif choice == "e":
            try:
                n = export_csv(sessions)
                print(f"  exported {plural(n, 'match', 'es')} → {CSV_FILE.name}")
            except OSError as e:
                print(yellow(f"  ! could not write {CSV_FILE.name} ({e})"))
        elif choice.isdigit():
            found = next((s for s in sessions if s["id"] == int(choice)), None)
            if found:
                show_session(found)
            else:
                print("  no such session")
        else:
            print("  invalid choice")
