"""LLM driver contract tests with a fake LLM (offline; no network, no key needed)."""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import pytest
from conftest import fast_task

from rlenv import cli
from rlenv.agents.llm import (
    API_KEY_ENV,
    ChatResponse,
    FakeChatClient,
    HttpChatClient,
    LLMAgent,
    LLMError,
    billable_tokens,
    tool_call_response,
)
from rlenv.export import export
from rlenv.harness import run_episode
from rlenv.task import load_tasks
from rlenv.trajectory import load_trajectory

ROOT = Path(__file__).resolve().parents[1]
FL = fast_task(load_tasks(ROOT / "tasks")["frozenlake-tabular"], eval_episodes=100)
SECRET = "sk-deepseek-test-0123456789abcdef"


def solver(reference_root: Path):
    """A fake model that solves FrozenLake like a competent agent, reading tool results."""
    files = {p.name: p.read_text() for p in sorted((reference_root / "frozenlake-tabular").glob("*.py"))}
    state = {"step": 0}

    def respond(messages):
        state["step"] += 1
        last_tool = next((json.loads(m["content"]) for m in reversed(messages) if m["role"] == "tool"), None)
        step = state["step"]
        think = f"step {step}: planning"
        if step == 1:
            return tool_call_response(*[("write_file", {"path": n, "content": c}) for n, c in files.items()],
                                      reasoning=think)
        if step == 2:
            return tool_call_response(("run_training", {}), reasoning=think)
        if step == 3:
            return tool_call_response(("evaluate", {"run_id": "r1"}), reasoning=think)
        if step == 4:
            mean = last_tool["data"]["mean_return"]
            report = ("## Method\nTabular Q-learning.\n## Hyperparameters\n-\n## Experiments\nr1\n"
                      f"## Results\nDev mean return **{mean:.3f}**\n## Limitations\n-\n")
            return tool_call_response(("write_file", {"path": "REPORT.md", "content": report}), reasoning=think)
        return tool_call_response(("submit", {"notes": "done"}), reasoning=think)

    return respond


def text(content: str) -> ChatResponse:
    return ChatResponse({"role": "assistant", "content": content}, "stop", {"total_tokens": 10})


# --------------------------------------------------------------------------- #
# End to end through the harness
# --------------------------------------------------------------------------- #


def test_fake_llm_solves_an_episode_end_to_end(tmp_path, fast_reference, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, SECRET)
    fake = FakeChatClient(solver(fast_reference))
    rec = run_episode(FL, LLMAgent(fake), tmp_path)
    assert rec.outcome.end_reason == "submitted" and rec.reward.status == "scored"
    assert rec.verification.report_ok  # the model's report cites the harness's numbers

    # Protocol: every tool call id answered exactly once, before the next request.
    for req in fake.requests[1:]:
        msgs = req["messages"]
        for i, m in enumerate(msgs):
            for tc in m.get("tool_calls") or []:
                replies = [x for x in msgs[i + 1:] if x.get("role") == "tool" and x.get("tool_call_id") == tc["id"]]
                assert len(replies) == 1, tc["id"]
    # DeepSeek rule (ADR-0011): with tools, all earlier reasoning_content is passed back.
    last = fake.requests[-1]["messages"]
    assert [m["reasoning_content"] for m in last if m.get("role") == "assistant"] == [
        f"step {k}: planning" for k in range(1, len(fake.requests))]
    assert all(r["params"]["thinking"] == {"type": "enabled"} and "temperature" not in r["params"]
               for r in fake.requests)
    assert fake.requests[0]["tools"][0]["function"]["name"] == "write_file"

    # Transcript in the trajectory, exact export, no secret anywhere.
    t = load_trajectory(rec.trajectory_dir)
    assert t.ok and len(t.of_type("llm_call")) == len(fake.requests)
    assert t.header["agent"]["model"] == "deepseek-flash"
    export([rec.trajectory_dir], tmp_path / "m.jsonl")
    row = json.loads((tmp_path / "m.jsonl").read_text())
    assert row["transcript"] == "exact" and row["messages"][0]["role"] == "system"
    assert row["messages"][-1]["role"] == "tool" and "submitted" in row["messages"][-1]["content"]
    assert row["loss_mask"] == [1 if m["role"] == "assistant" else 0 for m in row["messages"]]
    blob = "".join(p.read_text(errors="replace") for p in rec.trajectory_dir.rglob("*")
                   if p.is_file() and "workspace" not in p.parts and p.suffix != ".npy")
    assert SECRET not in blob + (tmp_path / "m.jsonl").read_text()


# --------------------------------------------------------------------------- #
# Model mistakes become feedback
# --------------------------------------------------------------------------- #


def run_with(responses, tmp_path, **kw):
    fake = FakeChatClient(responses)
    rec = run_episode(FL, LLMAgent(fake, **kw), tmp_path)
    return rec, fake


def tool_replies(fake, n):
    return [json.loads(m["content"]) for m in fake.requests[n]["messages"] if m["role"] == "tool"]


def test_invalid_json_and_unknown_tool_are_tool_errors(tmp_path):
    bad_json = ChatResponse({"role": "assistant", "content": None, "tool_calls": [
        {"id": "a", "type": "function", "function": {"name": "write_file", "arguments": "{not json"}},
        {"id": "b", "type": "function", "function": {"name": "rm_rf", "arguments": "{}"}}]}, "tool_calls", {})
    rec, fake = run_with([bad_json, tool_call_response(("submit", {}))], tmp_path)
    replies = tool_replies(fake, 1)
    assert not replies[0]["ok"] and "not valid JSON" in replies[0]["error"] and "{not json" in replies[0]["error"]
    assert not replies[1]["ok"] and "unknown tool" in replies[1]["error"]
    assert rec.outcome.end_reason == "submitted"


def test_text_only_replies_are_nudged_then_stop(tmp_path):
    rec, fake = run_with([text("I think we should..."), text("Hmm."), text("Done!")], tmp_path, max_nudges=2)
    assert rec.outcome.end_reason == "agent_stopped" and len(fake.requests) == 3
    assert "calling one of the tools" in fake.requests[1]["messages"][-1]["content"]
    assert rec.reward.reward == 0.0  # nothing submitted: a real (agent) failure, not unscorable


def test_api_failure_is_unscorable_not_zero(tmp_path):
    rec, _ = run_with([LLMError("API error 402: Insufficient Balance")], tmp_path)
    assert rec.outcome.end_reason == "agent_error"
    assert rec.reward.reward is None and rec.reward.failure_kind == "agent_driver_error"


def test_retryable_finish_reasons(tmp_path):
    busy = ChatResponse({"role": "assistant", "content": ""}, "insufficient_system_resource", {})
    rec, fake = run_with([busy, tool_call_response(("submit", {}))], tmp_path)
    assert rec.outcome.end_reason == "submitted" and len(fake.requests) == 2
    assert any(e["type"] == "llm_retry" for e in rec.outcome.events)
    rec2, _ = run_with([busy, busy, busy], tmp_path / "b")
    assert rec2.reward.reward is None and "kept returning" in rec2.outcome.agent_error


@pytest.mark.parametrize("finish,fragment", [("length", "truncated"), ("content_filter", "content filter")])
def test_fatal_finish_reasons(tmp_path, finish, fragment):
    rec, _ = run_with([ChatResponse({"role": "assistant", "content": "..."}, finish, {})], tmp_path)
    assert rec.reward.reward is None and fragment in rec.outcome.agent_error


def test_total_token_hard_cap_stops_the_agent(tmp_path):
    big = lambda: tool_call_response(("list_files", {}), usage={"total_tokens": 600})  # noqa: E731
    rec, fake = run_with([big(), big(), big(), big()], tmp_path, max_total_tokens=1000)
    assert rec.outcome.end_reason == "agent_stopped" and len(fake.requests) == 2
    assert any(e["type"] == "llm_stop" and "hard cap" in e["reason"] for e in rec.outcome.events)


def usage(hit=0, miss=0, out=0):
    return {"prompt_tokens": hit + miss, "prompt_cache_hit_tokens": hit, "prompt_cache_miss_tokens": miss,
            "completion_tokens": out, "total_tokens": hit + miss + out}


def test_billable_tokens_count_cache_misses_and_output_only():
    assert billable_tokens(usage(hit=40_000, miss=1_000, out=500)) == 1_500
    # No cache info: every prompt token counts (never under-count).
    assert billable_tokens({"prompt_tokens": 3_000, "completion_tokens": 100, "total_tokens": 3_100}) == 3_100
    assert billable_tokens({"total_tokens": 77}) == 77
    assert billable_tokens({}) == 0


def test_cached_context_does_not_exhaust_the_cost_budget(tmp_path):
    """The first real episode: ~97 % of prompt tokens were cache hits, so a total-token cap
    stopped the model before submit at ~$0.05. Billable tokens must ignore cache hits."""
    step = lambda: tool_call_response(("list_files", {}), usage=usage(hit=90_000, miss=500, out=500))  # noqa: E731
    rec, fake = run_with([step(), step(), step(), step(), tool_call_response(("submit", {}))], tmp_path,
                         max_billable_tokens=10_000)
    assert rec.outcome.end_reason == "submitted" and len(fake.requests) == 5  # 362k total tokens, 4k billable
    assert not any(e["type"] in ("llm_stop", "llm_budget_warning") for e in rec.outcome.events)


def test_billable_budget_stops_the_agent(tmp_path):
    step = lambda: tool_call_response(("list_files", {}), usage=usage(hit=10, miss=400, out=300))  # noqa: E731
    rec, fake = run_with([step() for _ in range(5)], tmp_path, max_billable_tokens=1_000)
    assert rec.outcome.end_reason == "agent_stopped" and len(fake.requests) == 2  # 700, then 1400 >= 1000
    stop = [e for e in rec.outcome.events if e["type"] == "llm_stop"]
    assert len(stop) == 1 and "billable 1400 >= 1000" in stop[0]["reason"]


def test_budget_warning_from_80_percent_lets_the_model_submit(tmp_path):
    step = lambda: tool_call_response(("list_files", {}), usage=usage(miss=200, out=100))  # noqa: E731
    responses = [step(), step(), step(), tool_call_response(("submit", {}), usage=usage(miss=10, out=10))]
    rec, fake = run_with(responses, tmp_path, max_billable_tokens=1_000)  # 300, 600, 900 (>= 800) ...
    warned = [any("WARNING" in r for r in tool_replies(fake, n)) for n in range(4)]
    assert warned == [False, False, False, True]
    reply = tool_replies(fake, 3)[-1]  # still valid JSON, the result itself intact
    assert "call submit" in reply["WARNING"] and "90%" in reply["WARNING"] and reply["ok"]
    assert [e["type"] for e in rec.outcome.events].count("llm_budget_warning") == 1
    assert rec.outcome.end_reason == "submitted"


def test_budget_warning_survives_truncation():
    from rlenv.agents.llm import MAX_TOOL_RESULT_CHARS, _tool_content

    text = _tool_content({"ok": True, "data": {"content": "x" * (2 * MAX_TOOL_RESULT_CHARS)}}, "wrap up")
    assert text.startswith('{"WARNING": "wrap up"') and "[truncated" in text


@pytest.mark.parametrize("kw", [{"max_billable_tokens": 0}, {"max_total_tokens": -1}, {"warn_fraction": 1.0}])
def test_bad_budget_settings_are_rejected(kw):
    with pytest.raises(ValueError):
        LLMAgent(FakeChatClient([]), **kw)


def test_strip_all_policy_and_non_thinking_params(tmp_path):
    r = tool_call_response(("list_files", {}), reasoning="secret plan")
    rec, fake = run_with([r, tool_call_response(("submit", {}))], tmp_path, reasoning_policy="strip_all",
                         thinking="disabled", temperature=0.0)
    second = fake.requests[1]
    assert not any("reasoning_content" in m for m in second["messages"])
    assert second["params"]["thinking"] == {"type": "disabled"} and second["params"]["temperature"] == 0.0


@pytest.mark.parametrize("model", ["deepseek-chat", "deepseek-reasoner"])
def test_retired_model_ids_are_rejected(model):
    with pytest.raises(ValueError, match="retired"):
        LLMAgent(FakeChatClient([]), model=model)


# --------------------------------------------------------------------------- #
# HTTP client (fake urlopen; never touches the network)
# --------------------------------------------------------------------------- #


class FakeHTTP:
    def __init__(self, outcomes):
        self.outcomes, self.requests = list(outcomes), []

    def __call__(self, req, timeout):
        self.requests.append(req)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return io.BytesIO(json.dumps(out).encode())


def http_error(code, message="boom"):
    return urllib.error.HTTPError("https://api.deepseek.com/chat/completions", code, "err", {},
                                  io.BytesIO(json.dumps({"error": {"message": message}}).encode()))


OK = {"model": "deepseek-flash", "choices": [{"message": {"role": "assistant", "content": "hi"},
                                               "finish_reason": "stop"}], "usage": {"total_tokens": 5}}


def test_http_client_request_shape_and_retry():
    http = FakeHTTP([http_error(503), OK])
    slept = []
    client = HttpChatClient(api_key=SECRET, urlopen=http, sleep=slept.append)
    resp = client([{"role": "user", "content": "x"}], [{"type": "function"}], {"model": "deepseek-flash"})
    assert resp.message["content"] == "hi" and resp.model == "deepseek-flash" and slept == [2.0]
    req = http.requests[0]
    body = json.loads(req.data)
    assert req.full_url == "https://api.deepseek.com/chat/completions"
    assert body["tool_choice"] == "auto" and body["stream"] is False and body["model"] == "deepseek-flash"
    assert req.get_header("Authorization") == f"Bearer {SECRET}"


@pytest.mark.parametrize("code", [400, 401, 402, 422])
def test_http_client_does_not_retry_client_errors_and_never_leaks_the_key(code):
    http = FakeHTTP([http_error(code, f"bad request; key={SECRET}?")])
    client = HttpChatClient(api_key=SECRET, urlopen=http, sleep=lambda s: None)
    with pytest.raises(LLMError) as exc:
        client([], [], {})
    assert len(http.requests) == 1 and str(code) in str(exc.value)
    # The server echoed the key in its message: our error text is truncated + redacted downstream,
    # but the client itself never adds the key or headers to error messages.
    assert "Authorization" not in str(exc.value) and "Bearer" not in str(exc.value)


def test_http_client_gives_up_after_network_errors():
    http = FakeHTTP([urllib.error.URLError("dns")] * 4)
    client = HttpChatClient(api_key=SECRET, urlopen=http, sleep=lambda s: None, max_attempts=4)
    with pytest.raises(LLMError, match="giving up after 4 attempts: network error"):
        client([], [], {})


def test_http_client_requires_a_key(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(LLMError, match="not set"):
        HttpChatClient()


def test_list_models_preflight():
    http = FakeHTTP([{"data": [{"id": "deepseek-v4-pro"}, {"id": "deepseek-flash"}]}])
    assert HttpChatClient(api_key=SECRET, urlopen=http).list_models() == ["deepseek-flash", "deepseek-v4-pro"]
    assert http.requests[0].full_url.endswith("/models") and http.requests[0].get_method() == "GET"


# --------------------------------------------------------------------------- #
# CLI gates (no network)
# --------------------------------------------------------------------------- #


def test_cli_gates(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv(API_KEY_ENV, SECRET)
    assert cli.main(["run", "frozenlake-tabular", "--tasks", str(ROOT / "tasks"), "--agent", "llm",
                     "--out", str(tmp_path)]) == 1
    assert "--allow-external-api" in capsys.readouterr().err
    assert cli.main(["llm-preflight"]) == 1
    assert cli.main(["run", "frozenlake-tabular", "--tasks", str(ROOT / "tasks"), "--agent", "llm",
                     "--allow-external-api", "--model", "deepseek-reasoner", "--out", str(tmp_path)]) == 1
    assert "retired" in capsys.readouterr().err
