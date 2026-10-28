"""Brain: thin synchronous wrapper around ADK agents (+ error/retry handling)."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from google.adk import Agent
from google.adk.planners import BuiltInPlanner
from google.adk.runners import InMemoryRunner
from google.genai import types

from runtime import RT
from terminal import clear_status, dim, yellow

# ADK/genai log a full traceback for every failed model call, even when we
# retry it: Brain reports failures itself, in one line.
logging.getLogger("google_adk").setLevel(logging.CRITICAL)
logging.getLogger("google_genai").setLevel(logging.ERROR)

APP, USER = "axelrod_mapek", "player"


class LLMUnavailable(RuntimeError):
    pass


def http_code(e: Exception) -> int | None:
    code = getattr(e, "code", None)  # set by google.genai's APIError
    return code if isinstance(code, int) else None


def retry_delay(e: Exception, attempt: int) -> float:
    """Seconds to wait before retrying a failed call (honours the API's hint)."""
    m = re.search(r"retry in ([\d.]+)\s*s", str(e)) or re.search(
        r"retryDelay['\"]?:\s*['\"](\d+(?:\.\d+)?)s", str(e)
    )
    if m:
        return min(float(m.group(1)) + 1, 90)
    code = http_code(e)
    if code == 429:  # rate limited
        return 15 * (attempt + 1)
    if code in (500, 502, 503, 504):  # overloaded
        return 5 * (attempt + 1)
    return 1.5 * (attempt + 1)


def quota_info(e: Exception) -> dict | None:
    """Which quota a 429 hit, e.g. {"per": "day", "what": "requests", "limit": 20, "model": ...}."""
    if http_code(e) != 429:
        return None
    text = str(e)
    qid = re.search(r"quotaId['\"]?:\s*['\"]([^'\"]+)", text)
    qid = qid.group(1) if qid else ""
    per = "day" if "PerDay" in qid else "minute" if "PerMinute" in qid else None
    what = "tokens" if "Token" in qid else "requests"
    limit = re.search(r"limit: (\d+)", text)
    model = re.search(r"model: ([\w.-]+)", text)
    return {
        "per": per,
        "what": what,
        "limit": int(limit.group(1)) if limit else None,
        "model": model.group(1) if model else str(RT.cfg.model),
        "free": "FreeTier" in qid or "free_tier" in text,
    }


def short_error(e: Exception) -> str:
    q = quota_info(e)
    if q:
        tier = "free tier" if q["free"] else "your plan"
        if q["limit"] == 0:
            return f"429: {q['model']} is not available on the {tier} (paid model)"
        per = f"/{q['per']}" if q["per"] else ""
        limit = f"{q['limit']} {q['what']}{per}" if q["limit"] is not None else "quota"
        return f"429: {limit} exceeded for {q['model']} ({tier})"
    m = re.search(r"""'message':\s*(['"])(.*?)\1""", str(e))
    msg = m.group(2).replace("\\n", " ") if m else str(e).strip().splitlines()[-1] if str(e).strip() else ""
    code = http_code(e)
    return f"{type(e).__name__}{f' {code}' if code else ''}: {msg[:300]}"


class Brain:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()  # one loop for the whole program
        self._runners: dict[tuple, InMemoryRunner] = {}
        self.calls = 0
        self.failures = 0  # consecutive failed calls
        self.last_error = ""

    def runner(self, kind: str, name: str, instruction: str) -> InMemoryRunner:
        key = (kind, name, instruction, str(RT.cfg.model), RT.cfg.temperature)
        if key not in self._runners:
            slug = re.sub(r"\W", "_", f"{kind}_{name}")
            kwargs: dict[str, Any] = {}
            model = RT.cfg.model
            if isinstance(model, str) and "2.5-flash" in model:
                # No hidden "thinking" tokens: faster and cheaper for a live demo.
                # (2.5 Pro and Gemini 3+ reject a zero budget.)
                kwargs["planner"] = BuiltInPlanner(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
            agent = Agent(
                name=slug,
                model=model,
                # A callable instruction is used verbatim: no {placeholder}
                # substitution, so prompts may safely contain braces.
                instruction=lambda ctx, text=instruction: text,
                generate_content_config=types.GenerateContentConfig(
                    temperature=RT.cfg.temperature
                ),
                **kwargs,
            )
            self._runners[key] = InMemoryRunner(agent=agent, app_name=APP)
        return self._runners[key]

    async def _ask(self, runner: InMemoryRunner, text: str) -> str:
        session = await runner.session_service.create_session(app_name=APP, user_id=USER)
        message = types.Content(role="user", parts=[types.Part(text=text)])
        out = ""
        try:
            async for ev in runner.run_async(
                user_id=USER, session_id=session.id, new_message=message
            ):
                if ev.is_final_response() and ev.content and ev.content.parts:
                    out += "".join(p.text for p in ev.content.parts if p.text and not p.thought)
        finally:
            await runner.session_service.delete_session(
                app_name=APP, user_id=USER, session_id=session.id
            )
        return out.strip()

    def _cancel_pending(self) -> None:
        """After a Ctrl-C, don't leave a half-done call to resume later."""
        tasks = asyncio.all_tasks(self.loop)
        for t in tasks:
            t.cancel()
        if tasks:
            self.loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))

    def ask(self, runner: InMemoryRunner, text: str) -> str | None:
        err: Exception | None = None
        for attempt in range(3):
            try:
                out = self.loop.run_until_complete(self._ask(runner, text))
                self.calls += 1
                self.failures = 0
                return out
            except KeyboardInterrupt:
                self._cancel_pending()
                raise
            except Exception as e:  # network, quota, bad key ...
                err = e
                code = http_code(e)
                if code is not None and 400 <= code < 500 and code not in (408, 429):
                    # Bad key, unknown model, invalid request: retrying won't help.
                    raise LLMUnavailable(short_error(e)) from e
                q = quota_info(e)
                if q and (q["per"] == "day" or q["limit"] == 0):
                    # Daily quota used up / paid-only model: waiting a minute won't help.
                    hint = (
                        "pick a free model in Settings"
                        if q["limit"] == 0
                        else "daily quota resets at midnight Pacific time; pick another model in Settings"
                    )
                    raise LLMUnavailable(f"{short_error(e)}: {hint}") from e
                if attempt < 2:
                    delay = retry_delay(e, attempt)
                    clear_status()
                    print(dim(f"  ! LLM call failed ({short_error(e)}), retry in {delay:.0f}s"))
                    time.sleep(delay)
        self.failures += 1
        self.last_error = short_error(err)
        clear_status()
        print(yellow(f"  ! LLM call given up ({self.last_error})"))
        if self.failures >= 3:
            raise LLMUnavailable(self.last_error)
        return None


def parse_json(text: str | None) -> dict:
    """First JSON object found in the reply (models like to add prose/fences)."""
    if not text:
        return {}
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            data, _ = decoder.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return {}
