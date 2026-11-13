"""Episode loop with the scripted agent (end-to-end) and with misbehaving agents."""

from __future__ import annotations

from pathlib import Path

import pytest

from rlenv import cli
from rlenv.agents.base import AgentDriver, ToolCall
from rlenv.agents.scripted import ScriptedAgent
from rlenv.episode import Episode
from rlenv.task import load_tasks

ROOT = Path(__file__).resolve().parents[1]
TASKS = load_tasks(ROOT / "tasks")


@pytest.mark.parametrize("task_id,episodes", [("frozenlake-tabular", 2000), ("mountaincar-tilecoding", 100)])
def test_scripted_agent_end_to_end(task_id, episodes, tmp_path):
    task = TASKS[task_id]
    out = Episode(task, ScriptedAgent(task_id, ["--episodes", str(episodes)]), tmp_path).run()
    assert out.end_reason == "submitted", out.events[-3:]
    assert out.terminated and not out.truncated
    calls = [e["tool"] for e in out.events if e["type"] == "tool_call"]
    assert calls[-4:] == ["run_training", "evaluate", "write_file", "submit"]
    assert len(out.runs) == 1 and out.runs[0].ok
    files = {p.name for p in out.workspace.iterdir()}
    assert {"train.py", "policy.py", "REPORT.md", "runs"} <= files

    # The report's numbers are the harness's numbers, not invented ones.
    dev = out.runs[0].dev_evals[0]
    report = (out.workspace / "REPORT.md").read_text()
    fmt = "{:.3f}" if task_id.startswith("frozenlake") else "{:.1f}"
    assert fmt.format(dev["mean_return"]) in report
    assert "## Limitations" in report and "{" not in report  # template fully filled

    # Events are ordered and complete.
    types = [e["type"] for e in out.events]
    assert types[0] == "start" and types[-1] == "end" and "run" in types
    assert all(e["t"] >= 0 for e in out.events)


class Crashing(AgentDriver):
    kind = "crashing"

    def start(self, observation):
        pass

    def next_action(self, last):
        raise RuntimeError("model API exploded")


class Looping(AgentDriver):
    kind = "looping"

    def start(self, observation):
        self.obs = observation

    def next_action(self, last):
        return ToolCall("list_files", {})


class Quitting(AgentDriver):
    kind = "quitting"

    def start(self, observation):
        self.obs = observation

    def next_action(self, last):
        return None


def test_agent_crash_ends_episode_cleanly(tmp_path):
    out = Episode(TASKS["frozenlake-tabular"], Crashing(), tmp_path).run()
    assert out.end_reason == "agent_error" and "model API exploded" in out.agent_error
    assert not out.terminated and not out.truncated  # neither: candidate for "unscorable" (step 7)
    assert any(e["type"] == "agent_error" for e in out.events)


def test_tool_call_budget_ends_episode(tmp_path):
    task = TASKS["frozenlake-tabular"]
    out = Episode(task, Looping(), tmp_path).run()
    assert out.end_reason == "budget:tool_calls" and out.tool_calls == task.budget.max_tool_calls
    assert out.truncated and not out.terminated


def test_wall_clock_budget_ends_episode(tmp_path):
    t = {"now": 0.0}

    class Slow(Looping):
        def next_action(self, last):
            t["now"] += 400  # the agent "thinks" for 400 s per step
            return super().next_action(last)

    out = Episode(TASKS["frozenlake-tabular"], Slow(), tmp_path, clock=lambda: t["now"]).run()
    assert out.end_reason == "budget:wall_clock" and out.tool_calls == 3  # 900 s budget


def test_agent_may_stop_without_submitting(tmp_path):
    out = Episode(TASKS["frozenlake-tabular"], Quitting(), tmp_path).run()
    assert out.end_reason == "agent_stopped" and out.tool_calls == 0


def test_agent_sees_no_eval_salt(tmp_path):
    agent = Quitting()
    Episode(TASKS["mountaincar-tilecoding"], agent, tmp_path).run()
    assert "heldout" not in str(agent.obs.task) and agent.obs.tools[0]["name"] == "write_file"


def test_sink_receives_every_event(tmp_path):
    seen = []
    out = Episode(TASKS["frozenlake-tabular"], Looping(), tmp_path, sink=seen.append).run()
    assert seen == out.events


@pytest.mark.slow
def test_cli_run(tmp_path, capsys, fast_reference):
    cli.main(["run", "frozenlake-tabular", "--tasks", str(ROOT / "tasks"), "--episodes", "500",
              "--out", str(tmp_path), "--reference-dir", str(fast_reference)])
    out = capsys.readouterr().out
    assert '"end_reason": "submitted"' in out and "evaluate" in out and "reward:" in out
