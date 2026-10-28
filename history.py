"""Persistent, append-only history of every finished LLM match (1v1 and tournaments).

One *session* = one 1v1 or one tournament run = one JSON line in
``results_history.jsonl``. Existing lines are never rewritten: each new run is
appended, so the file keeps every match you ever played. ``results_matches.csv``
is a flat export (one row per LLM match) generated on demand from that file.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from records import MatchRecord
from runtime import RT

HISTORY_FILE = Path(__file__).with_name("results_history.jsonl")
CSV_FILE = Path(__file__).with_name("results_matches.csv")
SCHEMA_VERSION = 1

CSV_COLUMNS = [
    "session_id", "timestamp", "kind", "status", "batch", "match_no",
    "opponent", "length", "start_strategy", "final_strategy",
    "llm_score", "opp_score", "llm_avg", "opp_avg", "outcome", "trend",
    "n_switches", "switches", "llm_rank", "n_players",
    "model", "temperature", "seed", "adaptive", "mapek_every", "window",
    "poor_threshold", "initial_strategy",
]


def match_to_dict(rec: MatchRecord) -> dict:
    """A finished match as plain JSON-able data (round by round + summary)."""
    a, b = rec.total()
    pa, pb = rec.avg()
    return {
        "opponent": rec.opponent,
        "length": rec.length,
        "start_strategy": rec.rounds[0].strategy,
        "final_strategy": rec.rounds[-1].strategy,
        "llm_score": a,
        "opp_score": b,
        "llm_avg": round(pa, 3),
        "opp_avg": round(pb, 3),
        "outcome": "win" if a > b else "loss" if a < b else "draw",
        "trend": rec.trend(),
        "models": rec.models,
        "switches": [
            {"round": r, "from": old, "to": new, "reason": why}
            for r, old, new, why in rec.switches
        ],
        "episodes": [
            {"strategy": s, "first": x, "last": y, "avg": round(p, 2)}
            for s, x, y, p in rec.episodes()
        ],
        "rounds": [
            {"me": r.me, "opp": r.opp, "mine": r.mine, "theirs": r.theirs, "strategy": r.strategy}
            for r in rec.rounds
        ],
    }


def load_sessions() -> tuple[list[dict], int]:
    """All saved sessions (oldest first) and how many unreadable lines were skipped."""
    sessions: list[dict] = []
    skipped = 0
    if not HISTORY_FILE.exists():
        return sessions, skipped
    with HISTORY_FILE.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if isinstance(s, dict) and isinstance(s.get("id"), int) and isinstance(s.get("matches"), list):
                sessions.append(s)
            else:
                skipped += 1
    return sessions, skipped


def save_session(
    kind: str,
    records: list[MatchRecord],
    ranking: list[dict] | None = None,
    status: str = "complete",
    batch: str | None = None,
) -> int:
    """Append one run to the history file and return its session number.

    kind    "1v1" or "tournament"
    ranking final tournament ranking (rows with rank / name / score / is_llm), if any
    status  "complete", or "aborted" when only the matches finished so far are kept
    batch   id shared by every match of one data-generator run (None otherwise)
    """
    sessions, _ = load_sessions()
    sid = max((s["id"] for s in sessions), default=0) + 1
    session = {
        "version": SCHEMA_VERSION,
        "id": sid,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "kind": kind,
        "status": status,
        "batch": batch,
        "config": asdict(RT.cfg),  # model, temperature, turns, seed, MAPE-K settings ...
        "strategies": RT.strategies,  # prompts as they were for this run
        "ranking": (
            [
                {"rank": r["rank"], "name": r["name"], "score": round(r["score"], 4), "is_llm": r["is_llm"]}
                for r in ranking
            ]
            if ranking
            else None
        ),
        "matches": [match_to_dict(r) for r in records],
    }
    line = json.dumps(session, ensure_ascii=False, default=str)

    # If a previous write was cut short (no final newline), start on a fresh line
    # so the new session is never glued to a broken one.
    needs_newline = False
    if HISTORY_FILE.exists() and HISTORY_FILE.stat().st_size:
        with HISTORY_FILE.open("rb") as f:
            f.seek(-1, 2)
            needs_newline = f.read(1) != b"\n"
    with HISTORY_FILE.open("a", encoding="utf-8") as f:
        f.write(("\n" if needs_newline else "") + line + "\n")
    return sid


def export_csv(sessions: list[dict]) -> int:
    """Write one row per saved LLM match to results_matches.csv; returns the row count."""
    n = 0
    with CSV_FILE.open("w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: Excel-friendly
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for s in sessions:
            cfg = s.get("config", {})
            ranking = s.get("ranking") or []
            llm = next((r for r in ranking if r.get("is_llm")), {})
            for no, m in enumerate(s["matches"], 1):
                switches = m.get("switches", [])
                writer.writerow(
                    {
                        "session_id": s["id"],
                        "timestamp": s.get("timestamp"),
                        "kind": s.get("kind"),
                        "status": s.get("status"),
                        "batch": s.get("batch"),
                        "match_no": no,
                        "opponent": m.get("opponent"),
                        "length": m.get("length"),
                        "start_strategy": m.get("start_strategy"),
                        "final_strategy": m.get("final_strategy"),
                        "llm_score": m.get("llm_score"),
                        "opp_score": m.get("opp_score"),
                        "llm_avg": m.get("llm_avg"),
                        "opp_avg": m.get("opp_avg"),
                        "outcome": m.get("outcome"),
                        "trend": m.get("trend"),
                        "n_switches": len(switches),
                        "switches": "; ".join(f"r{x['round']}: {x['from']}->{x['to']}" for x in switches),
                        "llm_rank": llm.get("rank"),
                        "n_players": len(ranking) or None,
                        "model": " + ".join(m.get("models") or []) or cfg.get("model"),
                        "temperature": cfg.get("temperature"),
                        "seed": cfg.get("seed"),
                        "adaptive": cfg.get("adaptive"),
                        "mapek_every": cfg.get("mapek_every"),
                        "window": cfg.get("window"),
                        "poor_threshold": cfg.get("poor_threshold"),
                        "initial_strategy": cfg.get("initial_strategy"),
                    }
                )
                n += 1
    return n
