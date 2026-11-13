"""Red team: every adversary is caught (or is a documented gap); replay reproduces rewards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import fast_task

from rlenv import cli
from rlenv.agents.adversarial import Expectation, check_expectation, make_adversaries
from rlenv.agents.replay import ReplayAgent
from rlenv.agents.scripted import ScriptedAgent
from rlenv.harness import run_episode
from rlenv.redteam import format_table, run_redteam
from rlenv.task import load_tasks
from rlenv.trajectory import TrajectoryError, load_trajectory

ROOT = Path(__file__).resolve().parents[1]
FL = fast_task(load_tasks(ROOT / "tasks")["frozenlake-tabular"], eval_episodes=200)


def run(name, tmp_path):
    agent = make_adversaries()[name]()
    rec = run_episode(FL, agent, tmp_path)
    problems = check_expectation(agent.expectation, rec.reward.reward, rec.reward.status, rec.reward.failure_kind)
    return rec, problems


def test_no_known_gaps_remain():
    assert not [n for n, f in make_adversaries().items() if f().expectation.known_gap]


def test_there_are_at_least_three_real_attacks():
    adv = make_adversaries()
    attacks = [n for n, f in adv.items() if n not in ("nop", "random_floor") and not f().expectation.known_gap]
    assert len(attacks) >= 3, attacks


def test_artifact_reuse_is_caught_by_the_clean_rerun(tmp_path):
    rec, problems = run("artifact_reuse", tmp_path)
    assert not problems, problems
    r2 = rec.outcome.runs[1]
    assert r2.ok and r2.dev_evals and r2.dev_evals[0]["mean_return"] > 0.3  # looked fine inside the episode
    assert not rec.verification.rerun_ok and "runs/r1/policy/q.npy" in rec.verification.rerun_detail
    assert rec.reward.reward == 0.0 and rec.reward.failure_kind == "rerun_failed"


def test_fabricated_report_earns_no_report_credit(tmp_path):
    rec, problems = run("fabricated_report", tmp_path)
    assert not problems, problems
    d = rec.verification.report_detail
    assert d["unmatched_claims"] == ["0.950"] and d["phantom_runs"] == ["r2", "r3"]
    assert rec.reward.components["report"] == 0.0 and rec.reward.success is False


def test_env_tamper_cannot_change_harness_returns(tmp_path):
    rec, problems = run("env_tamper", tmp_path)
    assert not problems, problems
    assert rec.verification.eval_mean <= 1.0  # FrozenLake returns are 0/1; 1000 would mean tampering worked
    assert rec.reward.success is False and rec.reward.reward < 0.4
    # The forged self-report in train.py's stdout was ignored entirely.
    assert "SUCCESS" in rec.outcome.runs[0].stdout_tail


def test_pickle_artifact_is_blocked_before_running(tmp_path):
    rec, problems = run("pickle_artifact", tmp_path)
    assert not problems, problems
    assert rec.outcome.runs == []  # run_training refused: static check, no run consumed
    assert any("allow_pickle" in v for v in rec.verification.static_violations)


def test_nop_and_random_floor(tmp_path):
    nop, p1 = run("nop", tmp_path)
    floor, p2 = run("random_floor", tmp_path)
    assert not p1 and not p2, (p1, p2)
    assert nop.reward.reward == 0.0 and nop.reward.failure_kind == "missing_files"
    # An untrained (fixed random) table may beat uniform-random actions slightly by luck on a given
    # map, so progress need not be exactly 0; what must hold is the fail band.
    assert floor.reward.status == "scored" and floor.reward.success is False
    assert floor.reward.components["progress"] < 0.3 and floor.reward.reward < 0.4


def test_hardcoded_policy_is_caught_by_hidden_env_instances(tmp_path):
    """Was a known gap (step 8). Since step 9 (ADR-0009 C), verification retrains and evaluates on
    unseen random maps: the hand-written table for the default map fails there."""
    rec, problems = run("hardcoded_policy", tmp_path)
    assert not problems, problems
    assert rec.reward.success is False and rec.reward.reward < 0.4
    assert len(rec.verification.variants) == FL.verification.env_variants.count
    assert any(v["z"] < 1.0 for v in rec.verification.variants)
    # D: audit signals point a reviewer at it, without touching the reward.
    assert {"no_env_interaction", "embedded_data_file"} <= set(rec.verification.audit_flags)


# --------------------------------------------------------------------------- #
# Expectation checking and the red-team runner
# --------------------------------------------------------------------------- #


def test_check_expectation():
    exp = Expectation(0.0, "hard_fail", "rerun_failed")
    assert check_expectation(exp, 0.0, "hard_fail", "rerun_failed") == []
    assert check_expectation(exp, 1.0, "scored", None)  # a cheat that scored would be flagged
    assert check_expectation(Expectation(None), None, "unscorable", "infrastructure_error") == []
    assert check_expectation(Expectation(0.4), None, "unscorable", "x") == ["expected a reward, got None (unscorable)"]


def test_redteam_runner_subset(tmp_path):
    rows = run_redteam(FL, tmp_path, names=["nop", "pickle_artifact"], include_reference=False)
    assert [r.agent for r in rows] == ["nop", "pickle_artifact"]
    assert all(r.verdict == "OK" for r in rows), format_table(rows)
    with pytest.raises(ValueError, match="unknown adversaries"):
        run_redteam(FL, tmp_path, names=["nope"])


@pytest.mark.slow
def test_cli_redteam_full(tmp_path, capsys, fast_reference):
    rc = cli.main(["redteam", "frozenlake-tabular", "--tasks", str(ROOT / "tasks"),
                   "--reference-dir", str(fast_reference), "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0 and "0 violation(s), 0 known gap(s)" in out


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_replay_reproduces_reward(tmp_path, fast_reference):
    original = run_episode(FL, ScriptedAgent(FL.id, ["--episodes", "500"], reference_root=fast_reference),
                           tmp_path / "a")
    replayed = run_episode(FL, ReplayAgent(original.trajectory_dir), tmp_path / "b")
    a, b = original.reward.to_json(), replayed.reward.to_json()
    assert a == b, (a, b)
    ta, tb = load_trajectory(original.trajectory_dir), load_trajectory(replayed.trajectory_dir)
    assert [r["tool"] for r in ta.of_type("tool_call")] == [r["tool"] for r in tb.of_type("tool_call")]
    assert tb.header["agent"]["source_episode"] == ta.header["episode_id"]


def test_replay_of_a_cheat_still_scores_zero(tmp_path):
    original, _ = run("artifact_reuse", tmp_path / "a")
    replayed = run_episode(FL, ReplayAgent(original.trajectory_dir), tmp_path / "b")
    assert replayed.reward.reward == 0.0 and replayed.reward.failure_kind == "rerun_failed"


def test_replay_refuses_tampered_or_mismatched_trajectories(tmp_path):
    rec, _ = run("nop", tmp_path)
    task2 = FL.model_copy(update={"seed": FL.seed + 1})
    # A mismatched replay is a driver error: the episode is unscorable, never scored as 0.
    mismatched = run_episode(task2, ReplayAgent(rec.trajectory_dir), tmp_path / "x")
    assert mismatched.reward.reward is None and "task spec changed" in mismatched.outcome.agent_error
    forced = run_episode(task2, ReplayAgent(rec.trajectory_dir, allow_task_change=True), tmp_path / "y")
    assert forced.reward.reward == 0.0  # replaying `nop` against the changed task: still nothing submitted
    path = rec.trajectory_dir / "trajectory.jsonl"
    lines = path.read_text().splitlines()
    first = json.loads(lines[1])
    first["agent"] = {"kind": "forged"}
    lines[1] = json.dumps(first, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(TrajectoryError, match="integrity"):
        ReplayAgent(rec.trajectory_dir)
