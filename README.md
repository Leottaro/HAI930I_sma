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
.venv/bin/pip install axelrod google-adk python-dotenv
.venv/bin/python main.py
```

Or **activate** the venv once, then use plain `pip` / `python`. The activation script must be *sourced*, not executed, so there is no `chmod +x` and no direct call:

```sh
source .venv/bin/activate        # Linux / macOS (bash, zsh)
pip install axelrod google-adk python-dotenv
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

or create a `.env` file next to `main.py` (loaded automatically when `python-dotenv` is installed):

```
GOOGLE_API_KEY=your-key
```

If no key is found, the program asks for it (hidden input) at the first match and saves it to that same `.env` file.

> Add `.venv/` and `.env` to your `.gitignore` so neither ends up in git.

### 4. Run

```sh
python main.py
```

Then open **4) Settings → 1) Gemini model** and pick a model from [Choosing a model](#choosing-a-model): the default `gemini-2.5-flash` is no longer available to recently created API keys.

---

## Choosing a model

Any Gemini API text model works; set it in **Settings → Gemini model**. "Free" means the model has a free tier on the Gemini API: you pay nothing, but requests per minute and per day are limited (about 15 requests/minute for `gemini-3.5-flash-lite`, for example), and Google may use the data to improve its products. "Paid" means the model needs a key with billing enabled: on a free key it fails right away with `429 RESOURCE_EXHAUSTED ... limit: 0`.

Prices are the standard paid rates in USD per 1M text tokens (input / output), from the [official pricing page](https://ai.google.dev/gemini-api/docs/pricing) in October 2026. One call takes a few hundred tokens, so a 30-round match (about 42 calls) costs a fraction of a cent on any Flash model. The latencies were measured once per model on a free key and vary a lot with server load.

**Free tier available**

| Model | Latency per call | Paid price (in / out) | Notes |
|---|---|---|---|
| `gemini-3.1-flash-lite` | ~1-2 s | $0.25 / $1.50 | **Recommended for demos**: fastest, follows the strategies and returns valid JSON for the MAPE-K stages |
| `gemini-3.5-flash-lite` | ~3 s | $0.30 / $2.50 | Good default too |
| `gemini-flash-lite-latest` | ~1-3 s | same as the Flash-Lite it points to | Alias that follows the newest Flash-Lite (may share its quota) |
| `gemini-robotics-er-2-preview` | ~2-4 s | not listed | Robotics reasoning model, but handles the move and MAPE-K prompts fine; may refuse prompts phrased as judging a person |
| `gemini-3.5-flash` | ~5 s | not listed | Stronger reasoning in the Analyze / Plan stages |
| `gemini-3.8-flash` | ~8 s | $0.75 / $3.75 (from Jan 1, 2027: $1.50 / $7.50) | Most capable Flash, but often `503 UNAVAILABLE` (high demand) on the free tier |
| `gemini-3.6-flash` | ~20 s | $0.75 / $3.75 (from Jan 1, 2027: $1.50 / $7.50) | Slow: thinks before every answer |
| `gemini-3-flash-preview` | ~20 s | $0.50 / $3.00 | Slow, preview model |
| `gemma-4-26b-a4b-it` | ~6 s | no paid tier | Open model (Gemma), free only |
| `gemma-4-31b-it` | ~30 s | no paid tier | Open model (Gemma), free only, slow |
| `gemini-flash-latest` | ~30 s | same as the Flash it points to | Alias that follows the newest Flash; slow, not in `FREE_MODELS` |
| `gemini-3.7-flash` | ~160 s | not listed | Far too slow for a match, not in `FREE_MODELS` |

`gemini-3.1-flash-lite-preview` is free too, but it uses the same daily quota as `gemini-3.1-flash-lite`, so it is no use as a fallback.

**Paid only**

| Model | Paid price (in / out) | Notes |
|---|---|---|
| `gemini-3.1-pro-preview` | $2.00 / $12.00 (prompts ≤ 200k tokens) | Best reasoning, slower |
| `gemini-4-argon` | $2.00 / $10.00 (from Dec 1, 2026: $4.00 / $20.00) | Newest model |
| `gemini-pro-latest` | same as the Pro it points to | Alias that follows the newest Pro |
| `gemini-omni-1.1-flash`, `gemini-omni-flash-preview` | not listed | Omni models (`limit: 0` on a free key) |

**No longer available to new API keys** (`404 NOT_FOUND`): `gemini-2.5-flash` (the default), `gemini-2.5-flash-lite`, `gemini-2.5-pro`. Older keys may still use them.

The list of models your key can see changes over time. To print it:

```sh
.venv/bin/python -c "from dotenv import load_dotenv; load_dotenv('.env'); from google import genai; c = genai.Client(); [print(m.name) for m in c.models.list() if 'generateContent' in (m.supported_actions or [])]"
```

Your current limits and usage are shown at <https://ai.dev/rate-limit>.

---

## The menu

| Choice | What it does |
|---|---|
| **1) 1v1 match** | The LLM against one classic strategy, shown **round by round**, with the MAPE-K trace at every cycle. |
| **2) Tournament** | The LLM plus the classic strategies you pick, as a round robin. Ends with a ranking and a per-opponent table. |
| **3) Strategy lab** | View, edit, add or delete the LLM prompt strategies. Changes are saved to `llm_strategies.json`. At least 3 strategies must remain. |
| **4) Settings** | Rounds per match, MAPE-K period, "poor performance" threshold, model, starting strategy, adaptive on/off... Invalid values are rejected (temperature must be between 0 and 2, the seed between 0 and 2³²−1, counts must be positive). |
| **5) Results history** | Browse every saved 1v1 / tournament (`results_history.jsonl`), replay a match round by round, export one row per match to `results_matches.csv`. |
| **6) Data generator** | Pick the opponents and the base strategies, then give the model, temperature, rounds per match, MAPE-K period, window and threshold each one value, a list (`10, 20, 40`) or a range (`10:50:10`, stop included; Enter keeps the current setting). Adaptive takes `y`, `n` or `both`, and `repetitions` is how many matches are played for each combination. The seed and the history shown to the move agent are not swept: they come from Settings. It shows the number of matches, the LLM calls and an estimated duration, asks for confirmation, then plays every combination as a 1v1 match. Each finished match is appended to the history as its own session, with a shared `batch` id (also a column of the CSV export). The batch stops at the first match that can't finish (quota, `Ctrl-C`), keeping the ones already played, and your Settings are left unchanged. |
| **7) Quit** | |

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
| `model` | `gemini-2.5-flash` | Gemini model used by every agent. Unavailable to new keys: see [Choosing a model](#choosing-a-model) |
| `auto_switch_model` | `on` | When the model's daily quota is used up, continue on the next free model of `FREE_MODELS` (`config.py`) |
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
- **Rate limits and errors**: the free Gemini tier has a low per-minute limit. Use fewer rounds, a lighter model or a paid key. Each failed call is reported on one line (`! LLM call failed (...), retry in Ns`) and retried up to 3 times: on a `429` the message says which quota was hit (e.g. `15 requests/minute exceeded for gemini-3.5-flash-lite (free tier)`) and the program waits as long as the API asks. On a `503` (model overloaded) it waits a few seconds. A run aborts after 3 consecutive failed calls instead of silently playing something else. An error that retrying cannot fix (invalid key, unknown or unavailable model) aborts the run immediately with the API's message. A used-up daily quota (e.g. `500 requests/day`), a paid-only model (`limit: 0`) or an unknown model (`404`) switches automatically to the next free model of `FREE_MODELS` in `config.py` (`! ...: switching to gemini-3.5-flash-lite`) and the same call is retried there, so the match goes on. Models used up this way are remembered until the program quits, so no request is wasted on them again. The run aborts only when every free model is used up (quotas reset at midnight Pacific time). Turn `auto_switch_model` off in Settings to abort instead. Each saved match lists the models that actually answered (`models` in the history, `model` column of the CSV, e.g. `gemini-3.1-flash-lite + gemini-3.5-flash-lite`). In the data generator, a match planned on a used-up model is played on the fallback one.
- **Interrupting**: `Ctrl-C` during a match or a tournament stops it and returns to the menu. `Ctrl-C` at the menu quits.
- **Mutual defection can be optimal**: against an unconditional defector, 1 point per round is the best anyone can get. The `POOR` flag is only a score threshold, so the Analyze stage is told to judge performance relative to what is achievable.
- **LLM vs itself** is off by default because it doubles the cost.
- **Fallback**: if a single move call fails or its answer has neither `C` nor `D` in it, the player falls back to tit-for-tat for that round so the match can continue. The Analyze / Plan replies may come with extra text or code fences around the JSON, and the strategy name is matched case-insensitively. If the Plan names an unknown strategy, the current one is kept.

---

## Files

| File | Purpose |
|---|---|
| `main.py` | The whole program: config, prompt strategies, ADK agents, MAPE-K player, menus |
| `llm_strategies.json` | Created when you edit strategies in the lab. Delete it (or use "reset") to get the defaults back. If it is malformed, the defaults are used and a warning is printed |
| `.env` | Optional, holds `GOOGLE_API_KEY`. Read from, and written to, the folder of `main.py` |