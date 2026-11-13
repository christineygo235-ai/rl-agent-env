"""OpenEnv-style facade (step 10a): same lifecycle, same records, same reward as the driver loop."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from conftest import fast_task

from rlenv.agents.base import Observation
from rlenv.agents.scripted import ScriptedAgent
from rlenv.harness import rescore, run_episode
from rlenv.openenv_facade import CLIENT_KIND, EnvObservation, RLTaskEnv, ToolAction
from rlenv.task import load_tasks
from rlenv.trajectory import load_trajectory

ROOT = Path(__file__).resolve().parents[1]
FL = fast_task(load_tasks(ROOT / "tasks")["frozenlake-tabular"], eval_episodes=100)

# Fields that legitimately differ between two runs of the same episode.
VOLATILE = {"t", "seq", "prev", "episode_id", "duration_s", "latency_s", "rerun_seconds", "created", "created_at",
            "started_at", "wall_clock_s", "workspace", "trajectory_dir", "head", "host", "pid"}


def drive(env: RLTaskEnv, agent) -> list[EnvObservation]:
    """An AgentDriver as an OpenEnv client: the episode is pushed through reset/step/close."""
    obs = env.reset(agent_info=agent.describe())
    agent.start(Observation(**obs.content))
    seen, last = [obs], None
    while not obs.done:
        call = agent.next_action(last)
        if call is None:
            env.close()
            break
        obs = env.step(ToolAction(call.name, call.args))
        seen.append(obs)
        last = obs.content
    return seen


def normalised(traj_dir: Path) -> list[dict]:
    traj = load_trajectory(traj_dir)
    assert traj.ok, traj.problems
    ids = {traj.records[0].get("episode_id", ""), traj_dir.name}

    def clean(x):
        if isinstance(x, dict):
            drop = VOLATILE | ({"bytes"} if x.get("path") == "REPORT.md" else set())  # its length varies too
            return {k: clean(v) for k, v in x.items() if k not in drop}
        if isinstance(x, list):
            return [clean(v) for v in x]
        if isinstance(x, str):
            for i in ids:
                if i:
                    x = x.replace(i, "<episode>")
            x = re.sub(r'"train_seconds": [0-9.]+', '"train_seconds": <t>', x)  # printed by train.py
            # The scripted agent's REPORT.md lists each run's wall-clock seconds ("Train s" column).
            return re.sub(r"(\| -?\d+ \| )\d+\.\d( \| \d+ \| )", r"\1<s>\2", x)
        return x

    return [clean(traj.inline(r)) for r in traj.records]  # compare file contents, not blob hashes


def test_facade_matches_driver_loop(tmp_path, fast_reference):
    args = ["--episodes", "2000"]
    loop = run_episode(FL, ScriptedAgent(FL.id, args, reference_root=fast_reference), tmp_path / "loop")
    env = RLTaskEnv(FL, tmp_path / "facade")
    seen = drive(env, ScriptedAgent(FL.id, args, reference_root=fast_reference))
    facade = env.record

    assert loop.reward.reward == 1.0 and loop.outcome.end_reason == "submitted"
    assert facade.reward.to_json() == loop.reward.to_json()
    assert seen[-1].done and seen[-1].reward == loop.reward.reward
    assert seen[-1].metadata["reward_status"] == "scored" and seen[-1].metadata["end_reason"] == "submitted"
    a, b = normalised(loop.trajectory_dir), normalised(facade.trajectory_dir)
    assert [r["type"] for r in a] == [r["type"] for r in b]
    for x, y in zip(a, b):
        assert x == y, json.dumps({"loop": x, "facade": y}, sort_keys=True)[:2000]
    assert rescore(facade.trajectory_dir).ok


def test_intermediate_observations_and_state(tmp_path):
    env = RLTaskEnv(FL, tmp_path)
    assert env.state.episode_id is None and env.state.step_count == 0
    obs = env.reset()
    assert not obs.done and obs.reward is None and obs.metadata["reward_status"] == "pending"
    assert set(obs.content) == {"instruction", "task", "tools", "budget"}
    assert "salt" not in json.dumps(obs.content)  # the client sees the agent view only
    obs = env.step(ToolAction("write_file", {"path": "notes.txt", "content": "hi"}))
    assert obs.content["ok"] and not obs.done and obs.reward is None
    obs = env.step(ToolAction("rm_rf", {}))  # unknown tool: a tool error, the episode goes on
    assert not obs.content["ok"] and "unknown tool" in obs.content["error"] and not obs.done
    st = env.state
    assert st.step_count == 2 and not st.done and st.episode_id == obs.metadata["episode_id"]
    env.close()
    assert env.state.done and env.state.end_reason == "agent_stopped"


def test_close_before_submit_verifies_and_writes(tmp_path):
    env = RLTaskEnv(FL, tmp_path)
    env.reset(agent_info={"kind": "my-client", "model": "x"})
    env.step(ToolAction("list_files"))
    env.close()
    rec = env.record
    assert rec.outcome.end_reason == "agent_stopped" and rec.reward.reward == 0.0
    assert rec.reward.status == "hard_fail"
    traj = load_trajectory(rec.trajectory_dir)
    assert traj.ok and traj.records[-1]["type"] == "final"
    assert traj.of_type("start")[0]["agent"] == {"kind": "my-client", "model": "x"}
    assert rescore(rec.trajectory_dir).ok


def test_budget_exhaustion_ends_on_the_last_step(tmp_path):
    task = FL.model_copy(update={"budget": FL.budget.model_copy(update={"max_tool_calls": 2})})
    env = RLTaskEnv(task, tmp_path)
    env.reset()
    assert not env.step(ToolAction("list_files")).done
    obs = env.step(ToolAction("list_files"))
    assert obs.done and obs.metadata["end_reason"] == "budget:tool_calls" and obs.metadata["truncated"]
    assert obs.reward == 0.0 and obs.content["ok"]  # the last call still ran and returned its result


def test_harness_crash_is_unscorable(tmp_path, monkeypatch):
    env = RLTaskEnv(FL, tmp_path)
    env.reset()

    def boom(*a, **k):
        raise OSError("disk on fire")

    monkeypatch.setattr(env._session.episode.tools, "dispatch", boom)
    obs = env.step(ToolAction("list_files"))
    assert obs.done and obs.reward is None and obs.metadata["reward_status"] == "unscorable"
    assert obs.metadata["failure_kind"] == "agent_driver_error" and "disk on fire" not in json.dumps(obs.content)
    assert any(r["type"] == "agent_error" for r in load_trajectory(env.record.trajectory_dir).records)


def test_reset_after_done_starts_a_new_episode(tmp_path):
    env = RLTaskEnv(FL, tmp_path)
    first = env.reset().metadata["episode_id"]
    assert env.step(ToolAction("submit", {})).done
    second = env.reset().metadata["episode_id"]
    assert second != first and env.state.step_count == 0 and env.record is None
    env.close()
    assert len([p for p in tmp_path.iterdir() if p.is_dir()]) == 2


def test_misuse_is_rejected(tmp_path):
    env = RLTaskEnv(FL, tmp_path)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(ToolAction("list_files"))
    env.reset()
    with pytest.raises(RuntimeError, match="in progress"):
        env.reset()
    with pytest.raises(TypeError):
        env.step({"tool_name": "list_files"})
    with pytest.raises(TypeError):
        env.step(ToolAction("list_files", ["not", "a", "dict"]))
    with pytest.raises(TypeError):
        RLTaskEnv(FL, tmp_path / "other").reset(agent_info="model-x")
    assert env.state.step_count == 0  # rejected calls did not reach the harness
    env.step(ToolAction("submit", {}))
    with pytest.raises(RuntimeError, match="done"):
        env.step(ToolAction("list_files"))
    env.close()
    env.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()
    with pytest.raises(RuntimeError, match="closed"):
        env.step(ToolAction("list_files"))


def test_context_manager_closes(tmp_path):
    with RLTaskEnv(FL, tmp_path) as env:
        env.reset()
    assert env.state.done and env.record.outcome.end_reason == "agent_stopped"
    assert env.record.outcome.workspace.exists() and CLIENT_KIND == "openenv_client"
