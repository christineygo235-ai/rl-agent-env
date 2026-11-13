"""Task spec model, validation and loading (offline)."""

from __future__ import annotations

import copy
from pathlib import Path

import gymnasium
import pytest
import yaml

from rlenv import cli
from rlenv.task import ALLOWED_ENVS, TaskError, load_task, load_tasks, parse_task

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks"


@pytest.fixture()
def raw() -> dict:
    return yaml.safe_load((TASKS / "mountaincar-tilecoding.yaml").read_text())


def set_path(data: dict, path: str, value):
    data = copy.deepcopy(data)
    *parents, leaf = path.split(".")
    node = data
    for key in parents:
        node = node[key]
    if value is ...:
        del node[leaf]
    else:
        node[leaf] = value
    return data


def invalid(data, *fragments):
    with pytest.raises(TaskError) as exc:
        parse_task(data)
    for f in fragments:
        assert f in str(exc.value), str(exc.value)


# --------------------------------------------------------------------------- #


def test_example_tasks_load():
    tasks = load_tasks(TASKS)
    assert set(tasks) == {"frozenlake-tabular", "mountaincar-tilecoding", "taxi-tabular", "cliffwalking-sarsa"}
    mc = tasks["mountaincar-tilecoding"]
    assert mc.env_id == "MountainCar-v0" and mc.success.threshold == -110.0  # calibrated (step 9)
    assert mc.methods.allow_torch is False  # numpy-only by default
    fl = tasks["frozenlake-tabular"]
    assert fl.env_kwargs == {"map_name": "4x4", "is_slippery": True}


def test_allowed_envs_are_registered_in_pinned_gymnasium():
    assert gymnasium.__version__ == "1.4.0"
    missing = [e for e in ALLOWED_ENVS if e not in gymnasium.envs.registry]
    assert missing == []


def test_example_envs_can_be_constructed():
    for task in load_tasks(TASKS).values():
        env = gymnasium.make(task.env_id, **task.env_kwargs)
        obs, _ = env.reset(seed=task.seed)
        assert env.observation_space.contains(obs)
        env.close()


def test_agent_view_hides_eval_salt(raw):
    task = parse_task(raw)
    view = task.agent_view()
    assert "eval_seed_salt" not in view["success"]
    assert "mountaincar-v0-heldout" not in str(view)
    assert task.model_dump()["success"]["eval_seed_salt"] == "mountaincar-v0-heldout"


def test_hashes_are_stable_and_sensitive(raw):
    a, b = parse_task(raw), parse_task(copy.deepcopy(raw))
    assert a.public_sha256() == b.public_sha256() and a.full_sha256() == b.full_sha256()
    salted = parse_task(set_path(raw, "success.eval_seed_salt", "other"))
    assert salted.public_sha256() == a.public_sha256()  # salt is not public
    assert salted.full_sha256() != a.full_sha256()
    harder = parse_task(set_path(raw, "success.threshold", -100.0))
    assert harder.public_sha256() != a.public_sha256()


def test_task_is_immutable(raw):
    task = parse_task(raw)
    with pytest.raises(Exception):
        task.seed = 1  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Rejections
# --------------------------------------------------------------------------- #


def test_unknown_key(raw):
    invalid(set_path(raw, "budget.wall_clok_s", 10), "budget.wall_clok_s", "Extra inputs")


def test_env_not_allowed(raw):
    invalid(set_path(raw, "env_id", "ALE/Pong-v5"), "not an allowed classic env")


def test_env_not_registered(raw, monkeypatch):
    monkeypatch.setattr("rlenv.task.ALLOWED_ENVS", frozenset({"Taxi-v3"}))
    invalid(set_path(raw, "env_id", "Taxi-v3"), "not registered in gymnasium 1.4.0")


def test_run_timeout_exceeds_episode_budget(raw):
    invalid(set_path(raw, "budget.train_timeout_s", 99999), "cannot exceed wall_clock_s")


@pytest.mark.parametrize("method", ["ppo", "a2c", "stable_baselines3"])
def test_non_classic_method_rejected(raw, method):
    invalid(set_path(raw, "methods.allowed", [method]), "methods.allowed.0")


def test_duplicate_methods(raw):
    invalid(set_path(raw, "methods.allowed", ["sarsa", "sarsa"]), "duplicate methods")


def test_torch_requires_dqn(raw):
    data = set_path(raw, "methods.allowed", ["sarsa"])
    invalid(set_path(data, "methods.allow_torch", True), "allow_torch")


def test_missing_salt(raw):
    invalid(set_path(raw, "success.eval_seed_salt", ...), "eval_seed_salt")


@pytest.mark.parametrize("bad_id", ["MountainCar", "a_b", "-x", ""])
def test_bad_id(raw, bad_id):
    invalid(set_path(raw, "id", bad_id), "id")


def test_unsupported_metric(raw):
    invalid(set_path(raw, "success.metric", "max_return"), "success.metric")


def test_env_kwargs_must_be_scalars(raw):
    invalid(set_path(raw, "env_kwargs", {"desc": ["SFFF", "FHFH"]}), "env_kwargs")


def test_id_must_match_file_name(raw, tmp_path):
    f = tmp_path / "something-else.yaml"
    f.write_text(yaml.safe_dump(raw))
    with pytest.raises(TaskError, match="must match the file name"):
        load_task(f)


def test_bad_yaml(tmp_path):
    f = tmp_path / "x.yaml"
    f.write_text("id: [unclosed\n")
    with pytest.raises(TaskError, match="invalid YAML"):
        load_task(f)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_list(capsys):
    assert cli.main(["list-tasks", "--tasks", str(TASKS)]) == 0
    out = capsys.readouterr().out
    assert "frozenlake-tabular" in out and "MountainCar-v0" in out


def test_cli_show_hides_salt_unless_full(capsys):
    assert cli.main(["show-task", "mountaincar-tilecoding", "--tasks", str(TASKS)]) == 0
    assert "heldout" not in capsys.readouterr().out
    assert cli.main(["show-task", "mountaincar-tilecoding", "--tasks", str(TASKS), "--full"]) == 0
    assert "heldout" in capsys.readouterr().out


def test_cli_error(tmp_path, capsys):
    (tmp_path / "bad.yaml").write_text("id: bad\n")
    assert cli.main(["list-tasks", "--tasks", str(tmp_path)]) == 1
    assert "invalid task" in capsys.readouterr().err
