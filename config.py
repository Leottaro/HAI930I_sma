"""Configuration (everything here can be changed from the "Settings" menu)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

ENV_FILE = Path(__file__).with_name(".env")

try:  # optional: load GOOGLE_API_KEY from a .env file
    from dotenv import load_dotenv

    load_dotenv(ENV_FILE)
except ImportError:
    pass


# Free-tier models, in order of preference: when the current model's daily quota
# is used up, Brain switches to the first one of these that is still available.
FREE_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-robotics-er-2-preview",
    "gemini-3.5-flash",
    "gemma-4-26b-a4b-it",
    "gemini-3.8-flash",
    "gemini-3-flash-preview",
    "gemini-3.6-flash",
    "gemma-4-31b-it",
]


@dataclass
class Config:
    model: Any = "gemini-3.1-flash-lite"
    auto_switch_model: bool = True  # daily quota hit → continue on another FREE_MODELS model
    temperature: float = 0.2
    turns: int = 10  # rounds per match (keep it small: 1 LLM call per round)
    repetitions: int = 1  # matches per pairing in a tournament
    seed: int = 1
    mapek_every: int = 5  # run the MAPE-K loop every N rounds
    window: int = 5  # Monitor: size of the sliding window (rounds)
    poor_threshold: float = 1.5  # avg points/round under which we flag "POOR"
    history_shown: int = 10  # rounds of history shown to the move agent
    initial_strategy: str = "Generous"  # starts naive so a comeback is visible
    adaptive: bool = True  # False = frozen on the initial strategy (baseline)
    llm_self_play: bool = False  # let the LLM also play against a copy of itself
