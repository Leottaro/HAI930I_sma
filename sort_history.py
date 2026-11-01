#!/usr/bin/env python3
"""Sort one or several history files into sorted_history/, one folder per test condition.

Several people each have their own history file: sort them one after the other
(or all at once). Every run MERGES its file into what sorted_history/ already
holds; nothing is ever deleted:

    python sort_history.py                                # results_history.jsonl
    python sort_history.py results_history_prenom.jsonl   # a teammate's file
    python sort_history.py a.jsonl b.jsonl                # several files at once
    python sort_history.py --dry-run x.jsonl              # show what would change
    python visualizations.py sorted_history               # then the plots

Each session is tagged with `source` (the name of the file it came from). A
session already present (same content, id and timestamp, e.g. the same file
sorted twice, or lines shared by two files copied from git) is not added again.

A *condition* is everything that defines a test, except the two things
visualizations.py compares inside one file (the opponent and the LLM base
strategy): kind (1v1 / tournament), temperature, turns, repetitions, adaptive,
the MAPE-K settings, history_shown, seed, llm_self_play and the prompt
strategies. Sessions from different launches or different people that share a
condition end up in the same folder.

The model is NOT part of a condition: switching models is only a way around
free-tier quotas, not a deliberate parameter, so every model is assumed to play
the same. The models that answered are still listed in index.csv (column
`models`) and in the `model` column of each matches.csv.

Replay duplicates: a restarted launch replays matches that were already played.
Within ONE source file, a session whose content (settings, prompts, moves,
scores, switches and their reasons) is identical to one from another launch is
a replay duplicate, whichever model played it. Identical repetitions inside one
launch are genuine measurements (Generous vs Cooperator is always CC) and are
kept: for each content, as many copies are kept as the launch that holds the
most of them. Identical results from two different people are independent
measurements and are all kept. Replay duplicates go to duplicates_removed.jsonl.

Each condition folder holds history.jsonl (the sessions kept, same format as
results_history.jsonl plus `source`), matches.csv (same format as
results_matches.csv, plus a `source` column) and, if needed,
duplicates_removed.jsonl. These files are rewritten on each run with everything
already stored plus the new sessions; plots/ folders are left untouched (rerun
visualizations.py to update them). sorted_history/index.csv lists every condition.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from history import HISTORY_FILE, export_csv
from strategies import DEFAULT_STRATEGIES

HERE = Path(__file__).parent
OUT_DIR = HERE / "sorted_history"
MARKER = ".sort_history"  # JSON settings of the folder; marks a folder this script owns
FILES = ("history.jsonl", "duplicates_removed.jsonl")  # where the sessions of a condition are stored

# Settings that define a condition, in folder-name order.
CONDITION_FIELDS = [
    "kind", "temperature", "repetitions", "adaptive", "mapek_every",
    "turns", "window", "poor_threshold", "history_shown", "seed", "llm_self_play", "prompts",
]
ALWAYS_NAMED = {"kind", "temperature", "repetitions", "adaptive", "mapek_every"}
# The other settings appear in a folder name only when they differ from these
# values, so that a name never changes when new data arrives.
REFERENCE = {
    "turns": 10, "window": 5, "poor_threshold": 1.5, "history_shown": 10, "seed": 1,
    "llm_self_play": False,
}
# Not part of the content compared for replay duplicates: bookkeeping, and the
# models (all models are assumed to play the same, see above).
NOT_CONTENT = {"version", "id", "timestamp", "batch", "source"}
NOT_CONTENT_CFG = {"model", "auto_switch_model"}
NOT_CONTENT_MATCH = {"models"}


def short_hash(obj: object, n: int = 6) -> str:
    return hashlib.md5(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:n]


REFERENCE["prompts"] = short_hash(DEFAULT_STRATEGIES)


# ── reading ────────────────────────────────────────────────────────────────
def read_history(path: Path) -> tuple[list[dict], list[tuple[int, str]]]:
    """Valid sessions in file order, and the (line number, text) of the lines skipped."""
    sessions, skipped = [], []
    with path.open(encoding="utf-8", errors="replace") as f:
        for no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                skipped.append((no, line.strip()))
                continue
            if isinstance(s, dict) and isinstance(s.get("id"), int) and isinstance(s.get("matches"), list):
                sessions.append(s)
            else:
                skipped.append((no, line.strip()))
    return sessions, skipped


def resolve_input(name: str) -> Path:
    """The file as given, or else the file of that name next to this script."""
    p = Path(name)
    if not p.exists() and not p.is_absolute() and (HERE / p).exists():
        p = HERE / p
    if not p.is_file():
        sys.exit(f"{name} not found")
    return p


def fingerprint(s: dict) -> str:
    """Identity of a session (everything but its source): same fingerprint = same session."""
    return short_hash({k: v for k, v in s.items() if k != "source"}, 32)


# ── conditions ─────────────────────────────────────────────────────────────
def models_summary(sessions: list[dict]) -> str:
    """Information only: 'model (n matches); ...', most used first (older sessions: config's model)."""
    counts: Counter = Counter()
    for s in sessions:
        for m in s["matches"]:
            for x in m.get("models") or [s.get("config", {}).get("model")]:
                counts[str(x)] += 1
    return "; ".join(f"{name} ({n})" for name, n in counts.most_common())


def sources_summary(sessions: list[dict]) -> str:
    counts = Counter(s.get("source") or "?" for s in sessions for _ in s["matches"])
    return "; ".join(f"{name} ({n})" for name, n in counts.most_common())


def condition(s: dict, ignore: set[str]) -> dict:
    cfg = s.get("config", {})
    c = {
        "kind": s.get("kind"),
        "prompts": short_hash(s.get("strategies")),
        **{k: cfg.get(k) for k in CONDITION_FIELDS if k not in ("kind", "prompts")},
    }
    if "adaptive" not in ignore and not c["adaptive"]:
        c["mapek_every"] = None  # the MAPE-K period plays no part in a frozen baseline
    return {k: v for k, v in c.items() if k not in ignore}


def label(key: str, value: object) -> str:
    if key == "adaptive":
        return "adaptive" if value else "fixed"
    if key == "llm_self_play":
        return "selfplay" if value else "noselfplay"
    if value is None:
        return ""
    prefix = {
        "temperature": "T", "repetitions": "rep", "mapek_every": "mapek", "turns": "turns",
        "window": "w", "poor_threshold": "poor", "history_shown": "hist", "seed": "seed",
        "prompts": "prompts-",
    }.get(key, "")
    return f"{prefix}{value}"


def folder_name(c: dict) -> str:
    """Stable, readable name: the main settings, plus any other one that differs from REFERENCE."""
    parts = [
        label(k, c[k]) for k in CONDITION_FIELDS
        if k in c and (k in ALWAYS_NAMED or c[k] != REFERENCE.get(k))
    ]
    return re.sub(r"[^\w.-]", "-", "_".join(p for p in parts if p))


# ── replay duplicates ──────────────────────────────────────────────────────
def content_key(s: dict) -> str:
    content = {k: v for k, v in s.items() if k not in NOT_CONTENT}
    content["config"] = {k: v for k, v in s.get("config", {}).items() if k not in NOT_CONTENT_CFG}
    content["matches"] = [
        {k: v for k, v in m.items() if k not in NOT_CONTENT_MATCH} for m in s["matches"]
    ]
    return short_hash(content, 32)


def launch(s: dict) -> str:
    """One data-generator run (its batch id), or one manual 1v1 / tournament."""
    return s.get("batch") or f"session #{s['id']}"


def split_duplicates(sessions: list[dict]) -> tuple[list[dict], list[dict]]:
    """(kept, dropped) for the sessions of ONE source: replays from another launch are dropped."""
    per_launch: dict[str, Counter] = defaultdict(Counter)
    for s in sessions:
        per_launch[content_key(s)][launch(s)] += 1
    allowed = {k: max(c.values()) for k, c in per_launch.items()}
    seen: Counter = Counter()
    kept, dropped = [], []
    for s in sessions:
        k = content_key(s)
        (kept if seen[k] < allowed[k] else dropped).append(s)
        seen[k] += 1
    return kept, dropped


# ── output folder ──────────────────────────────────────────────────────────
def open_output(out: Path, ignore: list[str], dry_run: bool) -> list[dict]:
    """Check the output folder and return the sessions it already holds."""
    if not out.exists() or not any(out.iterdir()):
        if not dry_run:
            out.mkdir(parents=True, exist_ok=True)
            (out / MARKER).write_text(json.dumps({"ignore": sorted(ignore)}) + "\n")
        return []
    if not (out / MARKER).exists():
        sys.exit(f"{out} exists and was not created by this script: choose another --output")
    try:
        settings = json.loads((out / MARKER).read_text())
    except json.JSONDecodeError:  # marker of an older version (no --ignore was used)
        settings = {"ignore": []}
    if sorted(settings.get("ignore", [])) != sorted(ignore):
        was = " ".join(settings.get("ignore", [])) or "nothing"
        sys.exit(
            f"{out} was sorted with --ignore {was}: use the same --ignore, "
            "or another --output for a different grouping"
        )
    if not dry_run:
        (out / MARKER).write_text(json.dumps({"ignore": sorted(ignore)}) + "\n")
    stored = []
    for folder in sorted(p for p in out.iterdir() if p.is_dir()):
        for name in FILES:
            if (folder / name).exists():
                stored += read_history(folder / name)[0]
    return stored


def replace_file(path: Path, write) -> None:
    """Write through a temporary file, so an interrupted run never leaves a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def write_jsonl(path: Path, sessions: list[dict]) -> None:
    def write(tmp: Path) -> None:
        with tmp.open("w", encoding="utf-8") as f:
            for s in sessions:
                f.write(json.dumps(s, ensure_ascii=False, default=str) + "\n")
    replace_file(path, write)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "files", nargs="*", default=[HISTORY_FILE.name], metavar="FILE",
        help=f"history file(s) to add (default: {HISTORY_FILE.name})",
    )
    ap.add_argument("-o", "--output", type=Path, default=OUT_DIR, help="output folder (default: %(default)s)")
    ap.add_argument(
        "--ignore", nargs="+", default=[], metavar="FIELD",
        choices=[k for k in CONDITION_FIELDS if k != "kind"],
        help="settings that must NOT split the conditions (the same for every run on one "
             "output folder): " + ", ".join(CONDITION_FIELDS[1:]),
    )
    ap.add_argument("--keep-duplicates", action="store_true", help="keep replayed identical sessions")
    ap.add_argument("--dry-run", action="store_true", help="print what would change, write nothing")
    args = ap.parse_args()

    inputs = [resolve_input(f) for f in args.files]
    known: dict[str, dict] = {}  # fingerprint -> session (with its source)
    for s in open_output(args.output, args.ignore, args.dry_run):
        known.setdefault(fingerprint(s), s)
    before = set(known)

    # Add the new sessions, file by file.
    for path in inputs:
        sessions, skipped = read_history(path)
        for no, text in skipped:
            print(f"  {path.name}: line {no} skipped (not a session): {text[:60]}")
        added = present = 0
        for s in sessions:
            fp = fingerprint(s)
            if fp in known:
                present += 1
                if not known[fp].get("source"):  # stored by an older version: now we know
                    known[fp]["source"] = path.name
            else:
                known[fp] = {**s, "source": path.name}
                added += 1
        print(f"  {path.name}: {added} new sessions, {present} already in {args.output.name}/")
    new = set(known) - before

    # Group everything by condition, oldest first.
    ignore = set(args.ignore)
    groups: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for fp, s in known.items():
        groups[folder_name(condition(s, ignore))].append((fp, s))

    rows = []
    totals = Counter()
    for name in sorted(groups):
        items = sorted(groups[name], key=lambda x: str(x[1].get("timestamp")))
        sessions = [s for _, s in items]
        kept, dropped = [], []
        if args.keep_duplicates:
            kept = sessions
        else:  # replays are looked for inside each source file only
            by_source: dict[str, list[dict]] = defaultdict(list)
            for s in sessions:
                by_source[s.get("source") or "?"].append(s)
            for group in by_source.values():
                k, d = split_duplicates(group)
                kept += k
                dropped += d
            order = {id(s): i for i, s in enumerate(sessions)}
            kept.sort(key=lambda s: order[id(s)])
            dropped.sort(key=lambda s: order[id(s)])
        n_new = sum(len(s["matches"]) for fp, s in items if fp in new)
        row = {
            "folder": name, **{k: condition(sessions[0], ignore).get(k) for k in CONDITION_FIELDS},
            "sessions": len(kept), "matches": sum(len(s["matches"]) for s in kept),
            "new_matches": n_new, "duplicates_removed": sum(len(s["matches"]) for s in dropped),
            "sources": sources_summary(kept), "models": models_summary(kept),
            "launches": "; ".join(dict.fromkeys(f"{s.get('source')}:{launch(s)}" for s in sessions)),
            "first": sessions[0].get("timestamp"), "last": sessions[-1].get("timestamp"),
        }
        rows.append(row)
        totals.update(matches=row["matches"], new=n_new, dup=row["duplicates_removed"])
        if args.dry_run:
            continue
        folder = args.output / name
        folder.mkdir(exist_ok=True)
        write_jsonl(folder / "history.jsonl", kept)
        replace_file(folder / "matches.csv", lambda tmp: export_csv(kept, tmp))
        if dropped:
            write_jsonl(folder / "duplicates_removed.jsonl", dropped)
        elif (folder / "duplicates_removed.jsonl").exists():  # its sessions are now in history.jsonl
            (folder / "duplicates_removed.jsonl").unlink()

    if not args.dry_run and rows:
        def write_index(tmp: Path) -> None:
            with tmp.open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        replace_file(args.output / "index.csv", write_index)

    width = max((len(r["folder"]) for r in rows), default=10)
    print(f"\n  {'condition':<{width}}  matches  new  dup.removed  sources")
    for r in rows:
        n_src = len(r["sources"].split("; ")) if r["sources"] else 0
        new_txt = f"+{r['new_matches']}" if r["new_matches"] else ""
        print(f"  {r['folder']:<{width}}  {r['matches']:>7}  {new_txt:>3}  {r['duplicates_removed']:>11}  {n_src:>7}")
    where = "(dry run: nothing written)" if args.dry_run else f"→ {args.output}/"
    print(
        f"\n  {len(rows)} conditions · {totals['matches']} matches kept ({totals['new']} new) · "
        f"{totals['dup']} replayed duplicates set aside {where}"
    )
    if not args.dry_run and totals["new"]:
        print(f"  update the plots: python visualizations.py {args.output}")


if __name__ == "__main__":
    main()
