"""LLM agent driver: DeepSeek via its OpenAI-compatible Chat Completions API (ADR-0011).

Off by default. A real run needs `--agent llm`, an explicit confirmation flag and
DEEPSEEK_API_KEY in the environment. Tests use `FakeChatClient` (no network).

Design:
- one tool definition (`openai_tools()` from TOOL_SPECS);
- the model's tool calls are executed one at a time through the harness, and every
  tool call id gets exactly one `role: tool` reply, in order;
- model mistakes become feedback, not crashes: invalid JSON arguments and unknown
  tools go to the harness, which returns a tool error; a text-only reply gets a
  nudge (bounded), then the driver stops;
- every model call is logged as an `llm_call` event (new messages, full response
  incl. reasoning, usage, latency), so the exact transcript is in the trajectory;
- API failures: transient ones are retried by the client; persistent ones raise,
  which ends the episode as `agent_error` -> reward None (unscorable), never 0;
- the API key lives only in the HTTP client's request header; it is never put in
  messages, events or error texts (and the trajectory redactor is a second line).
"""

from __future__ import annotations

import itertools
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from rlenv.agents.base import AgentDriver, Observation, ToolCall
from rlenv.tools import INVALID_JSON_KEY, openai_tools

DEFAULT_BASE_URL = "https://api.deepseek.com"
API_KEY_ENV = "DEEPSEEK_API_KEY"
# Step 9a (ADR-0011): `deepseek-chat` / `deepseek-reasoner` were retired on 2026-07-24.
DEFAULT_MODEL = "deepseek-flash"
RETIRED_MODELS = {"deepseek-chat", "deepseek-reasoner"}
RETRY_STATUS = {429, 500, 502, 503, 504}
RETRY_FINISH_REASONS = {"insufficient_system_resource", "aborted"}  # server-side, retry the same request
MAX_FINISH_RETRIES = 2
MAX_TOOL_RESULT_CHARS = 6000
# Token budget (ADR-0011): the cost cap counts billable tokens (cache-miss input + output);
# the total-token hard cap is a loose backstop. Warn the model from WARN_FRACTION on.
DEFAULT_MAX_BILLABLE_TOKENS = 200_000
DEFAULT_MAX_TOTAL_TOKENS = 2_000_000
WARN_FRACTION = 0.8

SYSTEM_PROMPT = """You are an RL engineer working inside a sandboxed workspace. You solve classic
reinforcement-learning tasks with methods from Sutton & Barto, using ONLY the tools provided.

How to work:
- Write code with write_file, train with run_training, check results with evaluate.
- Every tool result tells you the remaining budget (wall clock, tool calls, training runs).
- The harness, not your code, computes all scores. Final scoring re-runs your train.py from
  scratch and evaluates the policy on held-out seeds you cannot see.
- When train.py, policy.py and REPORT.md are ready, call submit. If you run out of budget
  before submitting, whatever is in the workspace is scored.
Follow the task's file contract and report rules exactly."""


class LLMError(Exception):
    """A model/API failure that ends the episode (the episode becomes unscorable)."""


@dataclass
class ChatResponse:
    message: dict[str, Any]          # role, content, tool_calls?, reasoning_content?
    finish_reason: str | None
    usage: dict[str, Any] = field(default_factory=dict)
    model: str | None = None


ChatClient = Callable[[list[dict], list[dict], dict], ChatResponse]


# --------------------------------------------------------------------------- #
# HTTP client (stdlib only)
# --------------------------------------------------------------------------- #


class HttpChatClient:
    """POST {base_url}/chat/completions. Retries 429/5xx with exponential backoff."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, api_key: str | None = None, timeout_s: float = 300.0,
                 max_attempts: int = 4, backoff_s: float = 2.0,
                 urlopen: Callable = urllib.request.urlopen, sleep: Callable[[float], None] = time.sleep) -> None:
        self.base_url = base_url.rstrip("/")
        self._key = api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")
        if not self._key:
            raise LLMError(f"{API_KEY_ENV} is not set")
        self.timeout_s, self.max_attempts, self.backoff_s = timeout_s, max_attempts, backoff_s
        self._urlopen, self._sleep = urlopen, sleep

    def __call__(self, messages: list[dict], tools: list[dict], params: dict) -> ChatResponse:
        body = json.dumps({"messages": messages, "tools": tools, "tool_choice": "auto", "stream": False,
                           **params}).encode()
        last_error = ""
        for attempt in range(1, self.max_attempts + 1):
            req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, method="POST",
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": f"Bearer {self._key}"})
            try:
                with self._urlopen(req, timeout=self.timeout_s) as resp:
                    data = json.loads(resp.read().decode())
                choice = data["choices"][0]
                return ChatResponse(choice["message"], choice.get("finish_reason"), data.get("usage", {}),
                                    data.get("model"))
            except urllib.error.HTTPError as exc:
                detail = _safe_error_body(exc)
                if exc.code not in RETRY_STATUS:
                    raise LLMError(f"API error {exc.code}: {detail}") from None
                last_error = f"API error {exc.code}: {detail}"
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_error = f"network error: {type(exc).__name__}"
            except (KeyError, IndexError, json.JSONDecodeError) as exc:
                raise LLMError(f"malformed API response: {type(exc).__name__}") from None
            if attempt < self.max_attempts:
                self._sleep(self.backoff_s * 2 ** (attempt - 1))
        raise LLMError(f"giving up after {self.max_attempts} attempts: {last_error}")


    def list_models(self) -> list[str]:
        """GET /models: a cheap preflight that the configured model id exists (real runs only)."""
        req = urllib.request.Request(f"{self.base_url}/models", method="GET",
                                     headers={"Authorization": f"Bearer {self._key}"})
        try:
            with self._urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            raise LLMError(f"API error {exc.code}: {_safe_error_body(exc)}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise LLMError(f"network error: {type(exc).__name__}") from None
        return sorted(m.get("id", "") for m in data.get("data", []))


def _safe_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(exc.read().decode())
        msg = payload.get("error", {}).get("message") or str(payload)[:300]
    except Exception:  # noqa: BLE001 - best effort, never echo headers
        msg = exc.reason if isinstance(exc.reason, str) else "unreadable error body"
    return str(msg)[:300]


# --------------------------------------------------------------------------- #
# Fake client for offline tests
# --------------------------------------------------------------------------- #


class FakeChatClient:
    """Replays scripted responses (or a function of the conversation); records every request."""

    def __init__(self, responses: list[ChatResponse | Exception] | Callable[[list[dict]], ChatResponse]) -> None:
        self.responses = responses
        self.requests: list[dict] = []

    def __call__(self, messages: list[dict], tools: list[dict], params: dict) -> ChatResponse:
        self.requests.append({"messages": json.loads(json.dumps(messages)), "tools": tools, "params": params})
        if callable(self.responses):
            return self.responses(messages)
        if not self.responses:
            raise LLMError("fake client: no scripted responses left")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


_FAKE_IDS = itertools.count(1)


def tool_call_response(*calls: tuple[str, dict], content: str | None = None, reasoning: str | None = None,
                       usage: dict | None = None) -> ChatResponse:
    """Helper for fakes: an assistant message calling the given tools (ids unique, like the real API)."""
    msg: dict[str, Any] = {"role": "assistant", "content": content,
                           "tool_calls": [{"id": f"call_{next(_FAKE_IDS)}_{name}", "type": "function",
                                           "function": {"name": name, "arguments": json.dumps(args)}}
                                          for name, args in calls]}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return ChatResponse(msg, "tool_calls", usage or {"prompt_tokens": 100, "completion_tokens": 20,
                                                     "total_tokens": 120})


# --------------------------------------------------------------------------- #
# The driver
# --------------------------------------------------------------------------- #


class LLMAgent(AgentDriver):
    kind = "llm"

    def __init__(self, client: ChatClient, model: str = DEFAULT_MODEL, thinking: str = "enabled",
                 temperature: float | None = 0.0, max_tokens: int = 32_768,
                 max_billable_tokens: int = DEFAULT_MAX_BILLABLE_TOKENS,
                 max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS, warn_fraction: float = WARN_FRACTION,
                 max_nudges: int = 2, reasoning_policy: str = "pass_back") -> None:
        if model in RETIRED_MODELS:
            raise ValueError(f"{model!r} was retired by DeepSeek on 2026-07-24; use 'deepseek-flash' or "
                             "'deepseek-v4-pro' (ADR-0011)")
        if thinking not in ("enabled", "disabled"):
            raise ValueError("thinking must be 'enabled' or 'disabled'")
        if reasoning_policy not in ("pass_back", "strip_all"):
            raise ValueError(f"unknown reasoning_policy {reasoning_policy!r}")
        self.client, self.model, self.thinking = client, model, thinking
        self.temperature, self.max_tokens = temperature, max_tokens
        if max_billable_tokens <= 0 or max_total_tokens <= 0 or not 0 < warn_fraction < 1:
            raise ValueError("token budgets must be positive and 0 < warn_fraction < 1")
        self.max_billable_tokens, self.max_total_tokens = max_billable_tokens, max_total_tokens
        self.warn_fraction, self.max_nudges = warn_fraction, max_nudges
        self.reasoning_policy = reasoning_policy
        self.messages: list[dict] = []
        self.pending: list[dict] = []      # tool calls from the last response not yet executed
        self.awaiting: str | None = None   # tool_call id whose result we're waiting for
        self.tokens_used = 0      # total_tokens summed over calls (loose hard cap)
        self.billable_used = 0    # cache-miss input + output tokens (the cost budget)
        self.warned = False
        self.calls = 0
        self.nudges = 0
        self._logged_upto = 0
        self.stop_reason: str | None = None

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "provider": "deepseek", "model": self.model, "thinking": self.thinking,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens, "max_billable_tokens": self.max_billable_tokens,
                "max_total_tokens": self.max_total_tokens, "warn_fraction": self.warn_fraction,
                "reasoning_policy": self.reasoning_policy, "system_prompt": SYSTEM_PROMPT}

    def start(self, observation: Observation) -> None:
        task = {k: v for k, v in observation.task.items() if k != "instruction"}
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{observation.instruction.strip()}\n\nTask spec:\n"
                                        f"{json.dumps(task, sort_keys=True)}\n\nBudget: {json.dumps(observation.budget)}"},
        ]
        self.tools = openai_tools()

    # -- one harness step ---------------------------------------------------

    def next_action(self, last_result: dict[str, Any] | None) -> ToolCall | None:
        if self.awaiting is not None:
            self.messages.append({"role": "tool", "tool_call_id": self.awaiting,
                                  "content": _tool_content(last_result, self._budget_warning())})
            self.awaiting = None
        while not self.pending:
            exhausted = self._budget_exhausted()
            if exhausted:
                self.stop_reason = exhausted
                self.emit({"type": "llm_stop", "reason": self.stop_reason})
                return None
            response = self._call_model()
            msg = response.message
            calls = msg.get("tool_calls") or []
            if calls:
                self.pending = list(calls)
                break
            # Text-only reply: nudge (bounded), then give up.
            if self.nudges >= self.max_nudges:
                self.stop_reason = "model stopped calling tools"
                self.emit({"type": "llm_stop", "reason": self.stop_reason})
                return None
            self.nudges += 1
            self.messages.append({"role": "user", "content": "Please continue by calling one of the tools. "
                                                             "Call submit when train.py, policy.py and REPORT.md "
                                                             "are ready."})
        call = self.pending.pop(0)
        self.awaiting = call["id"]
        name = call.get("function", {}).get("name", "")
        raw = call.get("function", {}).get("arguments", "")
        try:
            args = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            args = {INVALID_JSON_KEY: raw[:2000]}  # the harness turns this into a clear tool error
        if not isinstance(args, dict):
            args = {"_invalid_arguments": args}
        return ToolCall(name, args)

    def _call_model(self) -> Any:
        params: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens,
                                  "thinking": {"type": self.thinking}}
        if self.thinking == "disabled" and self.temperature is not None:
            params["temperature"] = self.temperature  # ignored by the API in thinking mode (ADR-0011)
        request_messages = self._request_messages()
        t0 = time.monotonic()
        for attempt in range(MAX_FINISH_RETRIES + 1):
            response = self.client(request_messages, self.tools, params)
            if response.finish_reason not in RETRY_FINISH_REASONS:
                break
            self.emit({"type": "llm_retry", "finish_reason": response.finish_reason, "attempt": attempt + 1})
        else:
            raise LLMError(f"model kept returning finish_reason={response.finish_reason!r}")
        latency = time.monotonic() - t0
        if response.finish_reason == "content_filter":
            raise LLMError("response blocked by the provider's content filter")
        if response.finish_reason == "length" and not response.message.get("tool_calls"):
            raise LLMError(f"response truncated at max_tokens={self.max_tokens}")
        self.calls += 1
        self.tokens_used += int(response.usage.get("total_tokens", 0) or 0)
        self.billable_used += billable_tokens(response.usage)
        msg = {k: v for k, v in response.message.items() if v is not None}
        msg.setdefault("role", "assistant")
        self.messages.append(msg)
        new = self.messages[self._logged_upto:-1]
        self._logged_upto = len(self.messages)
        self.emit({"type": "llm_call", "n": self.calls, "model": response.model or self.model, "params": params,
                   "new_messages": new, "response": msg, "finish_reason": response.finish_reason,
                   "usage": response.usage, "latency_s": round(latency, 3), "tokens_used": self.tokens_used,
                   "billable_tokens_used": self.billable_used})
        return response

    # -- token budget (ADR-0011) ---------------------------------------------

    def _budget_fraction(self) -> float:
        return max(self.billable_used / self.max_billable_tokens, self.tokens_used / self.max_total_tokens)

    def _budget_exhausted(self) -> str | None:
        if self.billable_used >= self.max_billable_tokens:
            return f"token budget exhausted (billable {self.billable_used} >= {self.max_billable_tokens})"
        if self.tokens_used >= self.max_total_tokens:
            return f"token budget exhausted (total {self.tokens_used} >= hard cap {self.max_total_tokens})"
        return None

    def _budget_warning(self) -> str | None:
        """From `warn_fraction` on, every tool result tells the model to wrap up and submit."""
        frac = self._budget_fraction()
        if frac < self.warn_fraction:
            return None
        if not self.warned:
            self.warned = True
            self.emit({"type": "llm_budget_warning", "fraction": round(frac, 3),
                       "billable_tokens_used": self.billable_used, "tokens_used": self.tokens_used})
        return (f"Your LLM token budget is {min(frac, 1.0):.0%} used and the episode stops when it runs out. "
                "Wrap up now: make sure train.py, policy.py and REPORT.md are in the workspace and call submit. "
                "Do not start new experiments.")

    def _request_messages(self) -> list[dict]:
        """Messages as sent to the API (ADR-0011).

        pass_back (default, required by DeepSeek): when `tools` are sent, every earlier
        assistant message must be passed back unchanged, INCLUDING `reasoning_content`;
        omitting it is a 400 error. strip_all exists for OpenAI-compatible providers
        that reject the field."""
        if self.reasoning_policy == "strip_all":
            return [{k: v for k, v in m.items() if k != "reasoning_content"} for m in self.messages]
        return list(self.messages)


def billable_tokens(usage: dict[str, Any]) -> int:
    """Tokens that cost real money: output + cache-miss input (ADR-0011).

    Cache-hit input is ~50x cheaper and dominates `total_tokens` because every call re-sends
    the whole context. If the provider does not report cache usage, all prompt tokens count
    (fail-safe: never under-count)."""
    completion = int(usage.get("completion_tokens", 0) or 0)
    miss = usage.get("prompt_cache_miss_tokens")
    prompt = int(miss if miss is not None else usage.get("prompt_tokens", 0) or 0)
    if not completion and not prompt:  # only total_tokens reported
        return int(usage.get("total_tokens", 0) or 0)
    return completion + prompt


def _tool_content(result: dict[str, Any] | None, warning: str | None = None) -> str:
    payload = dict(result or {"ok": False, "error": "no result"})
    if warning:
        payload["WARNING"] = warning  # upper case sorts first: survives truncation, stays valid JSON
    text = json.dumps(payload, sort_keys=True)
    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = text[:MAX_TOOL_RESULT_CHARS] + f"... [truncated, {len(text)} chars]"
    return text
