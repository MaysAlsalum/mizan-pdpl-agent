"""Single entry point for every LLM call: disk cache, rate limiting, retries, JSON parsing, cost tracking."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import litellm
import yaml
from dotenv import load_dotenv

load_dotenv()
litellm.suppress_debug_info = True

ROOT = Path(__file__).resolve().parent.parent
CONFIG: dict = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
CACHE_DIR = ROOT / CONFIG.get("cache_dir", ".cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_RETRYABLE = tuple(
    getattr(litellm, name)
    for name in ("RateLimitError", "ServiceUnavailableError", "InternalServerError",
                 "APIConnectionError", "Timeout")
    if hasattr(litellm, name)
)


_EXHAUSTED: set[str] = set()  # models whose daily quota ran out during this process


def _is_quota(e: Exception) -> bool:
    """A per-day quota error (not a short per-minute limit)."""
    msg = str(e)
    return "RESOURCE_EXHAUSTED" in msg or "per day" in msg.lower() or "PerDay" in msg


def model_for(role: str) -> str:
    """Model name for a role in config.yaml (small, large, generator, judge_eval)."""
    return CONFIG["models"][role]


@dataclass
class LLMResult:
    text: str
    data: Any
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float
    cached: bool = False


class _Totals:
    """Thread-safe running totals across all calls."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.calls = self.cache_hits = self.prompt_tokens = self.completion_tokens = 0
        self.cost_usd = 0.0

    def add(self, r: LLMResult) -> None:
        with self._lock:
            self.calls += 1
            self.cache_hits += int(r.cached)
            self.prompt_tokens += r.prompt_tokens
            self.completion_tokens += r.completion_tokens
            self.cost_usd += r.cost_usd

    def summary(self) -> dict:
        return {"calls": self.calls, "cache_hits": self.cache_hits,
                "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "cost_usd": round(self.cost_usd, 6)}


TOTALS = _Totals()


class _RateLimiter:
    """Spaces requests per provider so we stay under the rpm limit in config.yaml."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next: dict[str, float] = {}

    def wait(self, provider: str) -> None:
        rpm = CONFIG.get("rate_limits_rpm", {}).get(provider)
        if not rpm:
            return
        interval = 60.0 / rpm
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next.get(provider, 0.0))
            self._next[provider] = slot + interval
        if slot > now:
            time.sleep(slot - now)


_LIMITER = _RateLimiter()


def _cache_path(model: str, messages: list, temperature: float, json_mode: bool) -> Path:
    payload = json.dumps({"m": model, "msgs": messages, "t": temperature, "j": json_mode},
                         ensure_ascii=False, sort_keys=True)
    key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return CACHE_DIR / key[:2] / f"{key}.json"


def parse_json(text: str) -> Any:
    """Parse JSON from a model reply, tolerating code fences and stray prose. Raises ValueError."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
    raise ValueError(f"Model did not return valid JSON: {text[:200]!r}")


def call(messages: str | list[dict], *, role: str | None = None, model: str | None = None,
         temperature: float = 0.0, json_mode: bool = False, use_cache: bool = True,
         max_retries: int = 5) -> LLMResult:
    """Call a model by role or name. Same inputs return the cached result at zero cost."""
    model = model or model_for(role or "small")
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    path = _cache_path(model, messages, temperature, json_mode)

    if use_cache and path.exists():
        rec = json.loads(path.read_text(encoding="utf-8"))
        result = LLMResult(**{**rec, "cached": True, "cost_usd": 0.0, "latency_s": 0.0})
        TOTALS.add(result)
        return result

    fallback = CONFIG.get("fallback_on_quota", {}).get(model)
    if model in _EXHAUSTED and fallback:
        # Known to be out of quota: go straight to the fallback instead of waiting for a rate-limit slot.
        return call(messages, model=fallback, temperature=temperature, json_mode=json_mode,
                    use_cache=use_cache, max_retries=max_retries)

    provider = model.split("/", 1)[0]
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if provider != "gemini":  # Gemini 3+ deprecates sampling params; it warns on every call
        kwargs["temperature"] = temperature
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    json_retried = False
    last_error: Exception | None = None
    for attempt in range(max_retries):
        _LIMITER.wait(provider)
        start = time.perf_counter()
        try:
            resp = litellm.completion(**kwargs)
        except _RETRYABLE as e:
            last_error = e
            if _is_quota(e):
                # Daily quota used up: waiting will not help. Switch model if one is configured.
                _EXHAUSTED.add(model)
                if fallback:
                    print(f"[llm] daily quota used up on {model}, switching to {fallback}")
                    return call(messages, model=fallback, temperature=temperature, json_mode=json_mode,
                                use_cache=use_cache, max_retries=max_retries)
                raise RuntimeError(f"daily quota used up on {model}: {e}") from e
            wait = min(60, 5 * 2 ** attempt)
            print(f"[llm] {type(e).__name__} on {model}, retrying in {wait}s")
            time.sleep(wait)
            continue
        latency = time.perf_counter() - start
        text = resp.choices[0].message.content or ""

        data = None
        if json_mode:
            try:
                data = parse_json(text)
            except ValueError as e:
                if json_retried:
                    raise
                json_retried, last_error = True, e
                kwargs["messages"] = messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": "Return ONLY valid JSON. No prose, no code fences."},
                ]
                continue

        try:
            cost = float(litellm.completion_cost(completion_response=resp))
        except Exception:
            cost = 0.0
        usage = getattr(resp, "usage", None)
        result = LLMResult(
            text=text, data=data, model=model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cost_usd=cost, latency_s=round(latency, 3),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(result), ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        TOTALS.add(result)
        return result

    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_error}")


if __name__ == "__main__":
    # Smoke test: every configured model once, then again to confirm the cache.
    prompt = 'Reply with JSON exactly like {"ok": true, "word": "<one Arabic greeting word>"}'
    for role in CONFIG["models"]:
        for run in (1, 2):
            try:
                r = call(prompt, role=role, json_mode=True)
                print(f"{role:10} run{run}  {r.model:32} cached={r.cached!s:5} data={r.data}")
            except Exception as e:
                print(f"{role:10} run{run}  FAILED: {type(e).__name__}: {str(e)[:150]}")
                break
    print("totals:", TOTALS.summary())