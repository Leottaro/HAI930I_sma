"""Live display of a match and of the MAPE-K trace."""
from __future__ import annotations

from records import MatchRecord
from runtime import RT
from terminal import USE_COLOR, bold, clear_status, cyan, dim, green, move_col, note, red, yellow


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
