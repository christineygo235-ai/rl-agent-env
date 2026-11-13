"""Held-out env instances (ADR-0009 C), audit signals (D), calibration."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from conftest import fast_task
from test_policy_check import GOOD_POLICY, GOOD_TRAIN
from test_verifier import FilesAgent

from rlenv import calibrate as cal
from rlenv.audit import audit_workspace
from rlenv.evaluator import evaluate
from rlenv.harness import run_episode
from rlenv.reference import REFERENCE_ROOT
from rlenv.reward import compute_reward
from rlenv.sandbox import SubprocessSandbox
from rlenv.seeds import dev_seeds
from rlenv.task import TaskError, load_tasks, parse_task
from rlenv.tools import Budget, ToolBox, dev_variants
from rlenv.trajectory import load_trajectory
from rlenv.variants import kwargs_sha256, optimal_policy, variant_kwargs
from rlenv.verifier import hidden_instances
from rlenv.workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]
TASKS = load_tasks(ROOT / "tasks")
FL = fast_task(TASKS["frozenlake-tabular"], eval_episodes=100, hidden_instances=2)
CANON = {"map_name": "4x4", "is_slippery": True}


# --------------------------------------------------------------------------- #
# Variant generation and the optimal policy
# --------------------------------------------------------------------------- #


def test_variants_are_deterministic_distinct_and_not_canonical():
    a = variant_kwargs("frozenlake_random_map", "FrozenLake-v1", CANON, "dev", "public", 1, 5, {"frozen_p": 0.8})
    b = variant_kwargs("frozenlake_random_map", "FrozenLake-v1", CANON, "dev", "public", 1, 5, {"frozen_p": 0.8})
    assert a == b and len({kwargs_sha256(k) for k in a}) == 5
    assert all(k["desc"] != ["SFFF", "FHFH", "FFFH", "HFFG"] and "map_name" not in k for k in a)
    assert all(k["is_slippery"] is True for k in a)


def test_dev_and_hidden_instances_differ():
    dev = {kwargs_sha256(k) for k in dev_variants(FL)}
    hidden = {kwargs_sha256(k) for k in hidden_instances(FL)}
    assert len(dev) == FL.verification.env_variants.dev_count
    assert len(hidden) == FL.verification.env_variants.count
    assert not dev & hidden


def test_optimal_policy_matches_known_optimum_on_canonical_map():
    pol = optimal_policy("frozenlake_random_map", "FrozenLake-v1", CANON)
    r = evaluate("FrozenLake-v1", CANON, dev_seeds(7, 2000), pol)
    assert 0.70 < r.mean < 0.78  # known optimum is about 0.74


def test_variants_only_for_supported_env_families():
    raw = yaml.safe_load((ROOT / "tasks" / "mountaincar-tilecoding.yaml").read_text())
    raw = copy.deepcopy(raw)
    raw["verification"] = {"env_variants": {"kind": "frozenlake_random_map"}}
    with pytest.raises(TaskError, match="not supported for MountainCar-v0"):
        parse_task(raw)
    raw["verification"] = {"env_variants": {"kind": "made_up"}}
    with pytest.raises(TaskError, match="not supported"):
        parse_task(raw)


# --------------------------------------------------------------------------- #
# Tools: --env-kwargs contract and public variants
# --------------------------------------------------------------------------- #


@pytest.fixture()
def box(tmp_path):
    return ToolBox(FL, Workspace(tmp_path / "ws"), tmp_path / "logs", Budget(FL))


def test_run_training_on_a_public_variant(box):
    box.dispatch("write_file", {"path": "train.py", "content": GOOD_TRAIN})
    box.dispatch("write_file", {"path": "policy.py", "content": GOOD_POLICY})
    r0 = box.dispatch("run_training", {})
    r2 = box.dispatch("run_training", {"variant": 2})
    assert r0.ok and r2.ok
    assert r0.data["env_kwargs"] == dict(FL.env_kwargs) and r2.data["env_kwargs"] == dev_variants(FL)[1]
    e = box.dispatch("evaluate", {"run_id": "r2"})
    assert e.ok and e.data["variant"] == 2


@pytest.mark.parametrize("args,fragment", [
    ({"variant": 4}, "does not exist"),
    ({"args": ["--env-kwargs", "{}"]}, "set by the harness"),
])
def test_variant_and_reserved_arg_errors(box, args, fragment):
    box.dispatch("write_file", {"path": "train.py", "content": GOOD_TRAIN})
    r = box.dispatch("run_training", args)
    assert not r.ok and fragment in r.error


# --------------------------------------------------------------------------- #
# Verification on hidden instances
# --------------------------------------------------------------------------- #


def test_hidden_maps_are_never_written_to_the_trajectory(tmp_path):
    rec = run_episode(FL, FilesAgent({"train.py": GOOD_TRAIN, "policy.py": GOOD_POLICY}), tmp_path)
    v = rec.verification
    assert len(v.variants) == 2 and all(len(e["env_kwargs_sha256"]) == 64 for e in v.variants)
    assert (v.random_mean, v.threshold) == (0.0, 1.0)  # normalised units: worst instance
    assert v.eval_mean == min(e["z"] for e in v.variants)
    text = (rec.trajectory_dir / "trajectory.jsonl").read_text()
    for kw in hidden_instances(FL):
        assert json.dumps(kw["desc"]) not in text and "".join(kw["desc"]) not in text


def test_missing_limits_on_any_instance_make_the_episode_unscorable(tmp_path, monkeypatch):
    calls = {"n": 0}
    orig = SubprocessSandbox.run_python

    def flaky_limits(self, *a, **kw):
        res = orig(self, *a, **kw)
        calls["n"] += 1
        if a[0] == "train.py" and calls["n"] >= 2:  # second hidden instance: CPU limit "not applied"
            res.limits_applied = {**res.limits_applied, "RLIMIT_CPU": False}
        return res

    monkeypatch.setattr(SubprocessSandbox, "run_python", flaky_limits)
    rec = run_episode(FL, FilesAgent({"train.py": GOOD_TRAIN, "policy.py": GOOD_POLICY}, train=False,
                                     evaluate=False, report="## Results\n**0.1**\n"), tmp_path)
    assert rec.reward.reward is None and rec.reward.failure_kind == "isolation_not_enforced"


# --------------------------------------------------------------------------- #
# Audit signals (D): recorded, never rewarded
# --------------------------------------------------------------------------- #


def test_references_raise_no_audit_flags():
    for task_id in TASKS:
        assert audit_workspace(REFERENCE_ROOT / task_id) == [], task_id


def test_audit_flags_do_not_change_the_reward(tmp_path):
    rec = run_episode(FL, FilesAgent({"train.py": GOOD_TRAIN, "policy.py": GOOD_POLICY}), tmp_path)
    flagged = replace(rec.verification, audit_flags=["no_env_interaction", "embedded_data_file"])
    clean = replace(rec.verification, audit_flags=[])
    assert compute_reward(flagged).to_json() == compute_reward(clean).to_json()
    assert load_trajectory(rec.trajectory_dir).of_type("verification")[0]["audit_flags"] == rec.verification.audit_flags


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def test_calibration_smoke(tmp_path):
    rep = cal.calibrate(TASKS["cliffwalking-sarsa"], n_seeds=2, workdir=tmp_path)
    assert rep.admitted and rep.deterministic and rep.pass_rate == 1.0 and rep.random_fails_everywhere
    js, md = cal.write_report(rep, tmp_path / "out")
    assert json.loads(js.read_text())["task_id"] == "cliffwalking-sarsa" and "ADMITTED" in md.read_text()


def test_calibration_rejects_a_task_the_reference_cannot_pass(tmp_path, monkeypatch):
    hard = TASKS["cliffwalking-sarsa"].model_copy(
        update={"success": TASKS["cliffwalking-sarsa"].success.model_copy(update={"threshold": -12.0})})
    rep = cal.calibrate(hard, n_seeds=2, workdir=tmp_path)
    assert not rep.admitted and any("passes on 0%" in r for r in rep.reasons)
