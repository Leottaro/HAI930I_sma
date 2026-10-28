"""The classic deterministic opponents the LLM can face."""
from __future__ import annotations

import axelrod as axl

# Name shown in menus -> class
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
