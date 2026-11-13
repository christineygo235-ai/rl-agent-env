"""ATIF export (Harbor v1.8): mapping from the native trajectory, and a validator that can fail."""

from __future__ import annotations

import copy
import json

import pytest
from test_llm import FL, SECRET, solver

from rlenv import cli
from rlenv.agents.llm import API_KEY_ENV, FakeChatClient, LLMAgent
from rlenv.agents.scripted import ScriptedAgent
from rlenv.atif import ATIF_VERSION, to_atif, validate_atif
from rlenv.harness import run_episode
from rlenv.trajectory import load_trajectory


@pytest.fixture(scope="module")
def llm_episode(tmp_path_factory, fast_reference):
    mp = pytest.MonkeyPatch()
    mp.setenv(API_KEY_ENV, SECRET)
    try:
        root = tmp_path_factory.mktemp("atif_llm")
        rec = run_episode(FL, LLMAgent(FakeChatClient(solver(fast_reference))), root)
    finally:
        mp.undo()
    return rec


def test_llm_episode_maps_to_valid_atif(llm_episode):
    traj = load_trajectory(llm_episode.trajectory_dir)
    doc = to_atif(traj)
    assert validate_atif(doc) == [], validate_atif(doc)
    assert doc["schema_version"] == ATIF_VERSION and doc["agent"]["model_name"] == "deepseek-flash"
    steps = doc["steps"]
    assert [s["source"] for s in steps[:2]] == ["system", "user"]
    calls = traj.of_type("llm_call")
    agent_steps = [s for s in steps if s["source"] == "agent"]
    assert len(agent_steps) == len(calls) and all(s["llm_call_count"] == 1 for s in agent_steps)
    # Every tool call is answered in the same step's observation (including the final submit).
    for s in agent_steps:
        ids = [tc["tool_call_id"] for tc in s.get("tool_calls", [])]
        got = [r["source_call_id"] for r in s.get("observation", {}).get("results", [])]
        assert got == ids
        assert "reasoning_content" in s and "metrics" in s
    assert agent_steps[-1]["tool_calls"][-1]["function_name"] == "submit"
    # write_file contents are inlined (the code is the sample), not blob references.
    first_write = agent_steps[0]["tool_calls"][0]
    assert first_write["function_name"] == "write_file" and "def " in first_write["arguments"]["content"]
    fm = doc["final_metrics"]
    assert fm["total_steps"] == len(steps) and fm["extra"]["llm_calls"] == len(calls)
    assert fm["total_prompt_tokens"] == sum(c["usage"].get("prompt_tokens", 0) for c in calls)
    fin = traj.of_type("final")[-1]
    assert doc["extra"]["rlenv"]["reward"] == fin["reward"] and doc["extra"]["rlenv"]["status"] == "scored"
    assert SECRET not in json.dumps(doc)


def test_non_llm_episode_maps_to_valid_atif(tmp_path, fast_reference):
    rec = run_episode(FL, ScriptedAgent(FL.id, ["--episodes", "500"], reference_root=fast_reference), tmp_path)
    traj = load_trajectory(rec.trajectory_dir)
    doc = to_atif(traj)
    assert validate_atif(doc) == []
    steps = doc["steps"]
    assert steps[0]["source"] == "user" and len(steps) == 1 + len(traj.of_type("tool_call"))
    assert all(s["llm_call_count"] == 0 and "metrics" not in s for s in steps[1:])
    assert "final_metrics" in doc and set(doc["final_metrics"]) == {"total_steps"}


MUTATIONS = [
    ("unknown root field", lambda d: d.update(reward=1.0), "unknown root field 'reward'"),
    ("step id gap", lambda d: d["steps"][1].update(step_id=7), "step_id must be 2"),
    ("bad source", lambda d: d["steps"][0].update(source="tool"), "source must be one of"),
    ("agent-only field on user step", lambda d: d["steps"][1].update(reasoning_content="x"), "only allowed on agent"),
    ("dangling tool reference", lambda d: _agent(d)["observation"]["results"][0].update(source_call_id="nope"),
     "matches no tool_call"),
    ("arguments not an object", lambda d: _agent(d)["tool_calls"][0].update(arguments="{}"), "must be an object"),
    ("metrics without an LLM call", lambda d: _agent(d).update(llm_call_count=0), "must be absent"),
    ("missing agent version", lambda d: d["agent"].pop("version"), "agent.version"),
]


def _agent(doc):
    return next(s for s in doc["steps"] if s["source"] == "agent" and s.get("observation"))


@pytest.mark.parametrize("name,mutate,expected", MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_validator_rejects_violations(llm_episode, name, mutate, expected):
    doc = copy.deepcopy(to_atif(load_trajectory(llm_episode.trajectory_dir)))
    mutate(doc)
    problems = validate_atif(doc)
    assert any(expected in p for p in problems), problems


def test_cli_exports_one_document_per_episode(llm_episode, tmp_path, capsys):
    out = tmp_path / "atif"
    assert cli.main(["export", str(llm_episode.trajectory_dir), "-o", str(out), "--format", "atif"]) == 0
    files = list(out.glob("*.json"))
    assert len(files) == 1 and files[0].stem == llm_episode.outcome.episode_id
    assert validate_atif(json.loads(files[0].read_text())) == []
