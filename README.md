# Axelrod's Tournament: deterministic strategies vs a self-adaptive LLM

A terminal program that reproduces Axelrod's Iterated Prisoner's Dilemma tournament with classic **deterministic** strategies (Tit-for-Tat, Always Cooperate...) and a dynamic **LLM-guided** player.

The LLM player runs a **MAPE-K loop**: it monitors the opponent and its own performance, and **switches between prompt strategies during a match**. The question of the demo: *can a self-adaptive LLM strategy make a comeback when it detects poor performance?*

Built with [Axelrod](https://github.com/Axelrod-Python/Axelrod) (tournament engine and classic strategies) and [Google ADK](https://github.com/google/adk-python) (LLM agents, Gemini).

---

## Quick start

You need Python 3.10+ and a [Gemini API key](https://aistudio.google.com/apikey).

### 1. Create a virtual environment (venv)

A venv is an isolated folder with its own Python and packages, so nothing is installed globally.

```sh
python -m venv .venv
```

### 2. Install the dependencies

You can install and run **without activating** the venv, by calling the executables inside it directly:

```sh
.venv/bin/pip install axelrod google-adk
.venv/bin/python main.py
```

Or **activate** the venv once, then use plain `pip` / `python`. The activation script must be *sourced*, not executed, so there is no `chmod +x` and no direct call:

```sh
source .venv/bin/activate        # Linux / macOS (bash, zsh)
pip install axelrod google-adk
python main.py
deactivate                       # when you are done
```

| Shell | Activation command |
|---|---|
| bash / zsh | `source .venv/bin/activate` |
| fish | `source .venv/bin/activate.fish` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows cmd | `.venv\Scripts\activate.bat` |

On Windows the executables live in `.venv\Scripts\` instead of `.venv/bin/`.

### 3. Set your API key

Either export it:

```sh
export GOOGLE_API_KEY="your-key"
```

or create a `.env` file next to `main.py` (loaded automatically):

```
GOOGLE_API_KEY=your-key
```

> Add `.venv/` and `.env` to your `.gitignore` so neither ends up in git.

### 4. Run

```sh
python main.py
```

---

## The menu

| Choice | What it does |
|---|---|
| **1) 1v1 match** | The LLM against one classic strategy, shown **round by round**, with the MAPE-K trace at every cycle. |
| **2) Tournament** | The LLM plus the classic strategies you pick, as a round robin. Ends with a ranking and a per-opponent table. |
| **3) Strategy lab** | View, edit, add or delete the LLM prompt strategies. Changes are saved to `llm_strategies.json`. At least 3 strategies must remain. |
| **4) Settings** | Rounds per match, MAPE-K period, "poor performance" threshold, model, starting strategy, adaptive on/off... |
| **5) Quit** | |

Opponents (all deterministic): Cooperator, Defector, Tit For Tat, Tit For 2 Tats, Suspicious Tit For Tat, Grudger, Alternator, Prober, Win-Stay Lose-Shift. To add one, add its Axelrod class to the `OPPONENTS` dict in `main.py`.

---

## How the LLM player works

Every round, an ADK agent plays `C` or `D` following the **active prompt strategy**. The LLM never sees the opponent's name: it has to infer who it is facing from the moves.

Every `mapek_every` rounds (5 by default), the MAPE-K loop runs:

| Stage | Done by | What happens |
|---|---|---|
| **Monitor** | Python | Computes the sliding-window score, the opponent's cooperation rate, the trend, and a `POOR` / `OK` / `GOOD` flag from the score threshold. |
| **Analyze** | LLM | Profiles the opponent (defector, grudger, tit-for-tat-like...) and judges our performance *relative to what is achievable against that opponent*. |
| **Plan** | LLM | Picks the strategy for the next rounds, using the catalogue of strategies and the track record of each one so far in this match. |
| **Execute** | Python | Swaps the active prompt strategy. |
| **Knowledge** | Python | Opponent profile, past analyses, per-strategy track record (points obtained, opponent cooperation rate). |

### Default prompt strategies

| Strategy | Behaviour |
|---|---|
| **Generous** | Trusting: cooperates by default, forgives most defections. |
| **Pragmatic** | Reciprocal (tit-for-tat flavour): mirrors the opponent's last move. |
| **Merciless** | Ruthless: defects every round, never forgives. |
| **Peacemaker** | Olive branch: breaks defection spirals by offering cooperation. |

The LLM starts as **Generous** on purpose: it gets exploited by hostile opponents first, which makes a comeback visible once the loop switches strategy.

### Reading the results

After each match you get a timeline and a verdict:

```
■ LOSS  LLM 15 – 40 Defector   avg 0.75 / 2.00  first 5: 0.0 → last 5: 1.0  ↗ comeback
  timeline: Generous r1-5 (0.0)  →  Merciless r6-20 (1.0)
```

(illustrative output, to show the format)

- `↗ comeback`: the first window was poor and the last window is clearly better.
- `↗ improved`, `→ steady`, `↘ declined`: the change between the first and the last window.
- In the tournament table, the trend bars show the LLM's points per 5-round block on a 0 to 5 scale.

---

## Tinkering

**Edit the prompts** (menu 3). Each strategy is a name, a one-line summary (shown to the Plan stage as the catalogue) and a prompt telling the LLM when to play C or D. Prompts may contain braces and any text.

**Compare adaptive vs frozen** (menu 4): turn **adaptive** off to keep the LLM locked on its starting strategy, then rerun the same match to see whether the MAPE-K loop actually made the difference.

**Settings** (defaults):

| Setting | Default | Meaning |
|---|---|---|
| `model` | `gemini-2.5-flash` | Gemini model used by every agent |
| `temperature` | `0.2` | Sampling temperature |
| `turns` | `30` | Rounds per match |
| `repetitions` | `1` | Matches per pairing in a tournament |
| `seed` | `1` | Random seed |
| `mapek_every` | `5` | Run the MAPE-K loop every N rounds |
| `window` | `5` | Monitor sliding window (rounds) |
| `poor_threshold` | `1.5` | Average points per round under which the flag is `POOR` |
| `history_shown` | `10` | Rounds of history shown to the move agent |
| `initial_strategy` | `Generous` | Strategy the LLM starts each match with |
| `adaptive` | `on` | MAPE-K on; off gives a frozen baseline |
| `llm_self_play` | `off` | Let the LLM also play a copy of itself in tournaments |

Payoffs are Axelrod's defaults: both cooperate = 3 each, both defect = 1 each, defect against a cooperator = 5 vs 0.

---

## Things to know

- **Cost and speed**: the LLM makes one call per round, plus 2 calls per MAPE-K cycle. The program prints an estimate before each run. Matches between classic strategies are free.
- **Rate limits**: the free Gemini tier has a low per-minute limit. Use fewer rounds or a paid key. Failed calls are retried with backoff, and a run aborts after 3 consecutive failures instead of silently playing something else.
- **Mutual defection can be optimal**: against an unconditional defector, 1 point per round is the best anyone can get. The `POOR` flag is only a score threshold, so the Analyze stage is told to judge performance relative to what is achievable.
- **LLM vs itself** is off by default because it doubles the cost.
- **Fallback**: if a single move call fails, the player falls back to tit-for-tat for that round so the match can continue.

---

## Files

| File | Purpose |
|---|---|
| `main.py` | The whole program: config, prompt strategies, ADK agents, MAPE-K player, menus |
| `llm_strategies.json` | Created when you edit strategies in the lab. Delete it (or use "reset") to get the defaults back |
| `.env` | Optional, holds `GOOGLE_API_KEY` |