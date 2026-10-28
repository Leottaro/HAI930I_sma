"""Records of what happens in a match (feed both the UI and the Knowledge base)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from runtime import RT
from terminal import green, red, yellow

# Trend of a match (how the LLM's score evolved), as saved in the history file.
TREND_LABELS = {
    "comeback": "↗ comeback",
    "improved": "↗ improved",
    "declined": "↘ declined",
    "steady": "→ steady",
}
_TREND_COLOUR = {"comeback": green, "improved": green, "declined": red, "steady": yellow}


def verdict_text(trend: str | None, pad: int = 0) -> str:
    """Coloured label for a trend key ('—' when unknown). `pad` pads the visible text."""
    label = f"{TREND_LABELS.get(trend, '—'):<{pad}}"
    colour = _TREND_COLOUR.get(trend)
    return colour(label) if colour else label


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

    def is_complete(self) -> bool:
        """True once every round of the match has been played."""
        return self.length is not None and len(self.rounds) >= self.length

    def trend(self) -> str | None:
        """'comeback' | 'improved' | 'declined' | 'steady', or None if the match is too short."""
        w = RT.cfg.window
        pts = [r.mine for r in self.rounds]
        if len(pts) < 2 * w:
            return None
        first, last = sum(pts[:w]) / w, sum(pts[-w:]) / w
        if last - first >= 0.5:
            return "comeback" if first < RT.cfg.poor_threshold else "improved"
        if first - last >= 0.5:
            return "declined"
        return "steady"

    def verdict(self) -> str:
        return verdict_text(self.trend())


@dataclass
class Knowledge:
    """The K of MAPE-K (the per-round history itself lives in MatchRecord)."""

    profile: str = "unknown yet"
    analyses: list[dict] = field(default_factory=list)
