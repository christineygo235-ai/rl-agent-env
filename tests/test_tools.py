"""Tool API: argument validation, reserved paths, budget accounting, run/evaluate semantics."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_policy_check import GOOD_POLICY, GOOD_TRAIN

from rlenv.task import load_tasks
from rlenv.tools import TOOL_NAMES, TOOL_SPECS, Budget, ToolArgError, ToolBox, openai_tools, validate_args
from rlenv.workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]
TASK = load_tasks(ROOT / "tasks")["frozenlake-tabular"]


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture()
def box(tmp_path):
    ws = Workspace(tmp_path / "ws")
    return ToolBox(TASK, ws, tmp_path / "logs", Budget(TASK))


def call(box, name, **args):
    return box.dispatch(name, args)


# --------------------------------------------------------------------------- #
# Schemas and validation
# --------------------------------------------------------------------------- #


def test_tool_specs_are_well_formed():
    assert TOOL_NAMES == ["write_file", "read_file", "list_files", "run_training", "evaluate", "submit"]
    for spec in TOOL_SPECS:
        assert set(spec) == {"name", "description", "input_schema"}
        assert spec["input_schema"]["type"] == "object" and spec["input_schema"]["additionalProperties"] is False


def test_openai_format_is_a_pure_conversion():
    tools = openai_tools()
    assert [t["function"]["name"] for t in tools] == TOOL_NAMES
    for t, spec in zip(tools, TOOL_SPECS):
        assert t["type"] == "function" and t["function"]["parameters"] is spec["input_schema"]


@pytest.mark.parametrize("name,args,fragment", [
    ("shell", {}, "unknown tool"),
    ("write_file", {"path": "a.py"}, "missing required"),
    ("write_file", {"path": "a.py", "content": "x", "mode": "w"}, "unknown argument"),
    ("write_file", {"path": 1, "content": "x"}, "must be of type string"),
    ("run_training", {"seed": True}, "must be of type integer"),
    ("run_training", {"seed": -1}, ">= 0"),
    ("run_training", {"args": ["a"] * 17}, "more than 16"),
    ("run_training", {"args": [1]}, "must be of type array"),
    ("read_file", {"path": "a", "max_bytes": 10**9}, "<= 262144"),
    ("evaluate", "r1", "must be an object"),
])
def test_argument_validation(name, args, fragment):
    with pytest.raises(ToolArgError, match=fragment):
        validate_args(name, args)


def test_bad_arguments_are_tool_errors_not_crashes(box):
    r = box.dispatch("write_file", {"path": "x.py"})
    assert not r.ok and "missing required" in r.error
    assert r.budget_remaining["tool_calls"] == TASK.budget.max_tool_calls - 1


# --------------------------------------------------------------------------- #
# File tools
# --------------------------------------------------------------------------- #


def test_write_read_list(box):
    assert call(box, "write_file", path="train.py", content="print(1)\n").ok
    r = call(box, "read_file", path="train.py")
    assert r.ok and r.data["content"] == "print(1)\n"
    assert [e["path"] for e in call(box, "list_files").data["entries"]] == ["train.py"]
    for root in ("", "."):  # models often pass "" for the root
        assert [e["path"] for e in call(box, "list_files", path=root).data["entries"]] == ["train.py"]


@pytest.mark.parametrize("path", ["runs/r1/policy/q.json", "runs/x.py", "./runs/evil.json", ".tmp/a.py", "../x.py"])
def test_agent_cannot_write_harness_paths_or_escape(box, path):
    r = call(box, "write_file", path=path, content="x")
    assert not r.ok


# --------------------------------------------------------------------------- #
# run_training / evaluate
# --------------------------------------------------------------------------- #


def test_static_violation_blocks_run_without_consuming_it(box):
    call(box, "write_file", path="train.py", content="import subprocess\n")
    r = call(box, "run_training")
    assert not r.ok and "static check failed" in r.error and r.data["static_check"]
    assert box.budget.runs == 0 and not box.runs


def test_successful_run_and_dev_evaluation(box):
    call(box, "write_file", path="train.py", content=GOOD_TRAIN)
    call(box, "write_file", path="policy.py", content=GOOD_POLICY)
    r = call(box, "run_training")
    assert r.ok, r.error
    assert r.data["run_id"] == "r1" and r.data["seed"] == TASK.seed
    assert r.data["artifacts"] == ["runs/r1/policy/meta.json", "runs/r1/policy/q.npy"]
    rec = box.runs["r1"]
    assert set(rec.code) == {"train.py", "policy.py"}  # artifacts are not code
    e = call(box, "evaluate", run_id="r1", episodes=10_000)
    assert e.ok and e.data["episodes"] == TASK.success.dev_eval_episodes  # capped
    assert 0.0 <= e.data["mean_return"] <= 1.0 and e.data["seeds"] == "dev"
    assert e.data["policy_sha256"] == rec.code["policy.py"]
    assert rec.dev_evals and rec.dev_evals[0]["error"] is None
    assert r.budget_remaining["runs"] == TASK.budget.max_runs - 1


def test_agent_chosen_seed_and_args(box):
    call(box, "write_file", path="train.py", content=GOOD_TRAIN)
    r = call(box, "run_training", seed=7)
    assert r.ok and r.data["seed"] == 7
    r = call(box, "run_training", args=["--seed", "1"])
    assert not r.ok and "set by the harness" in r.error
    r = call(box, "run_training", args=["--out=/tmp/x"])
    assert not r.ok and "set by the harness" in r.error


def test_failed_run(box):
    call(box, "write_file", path="train.py", content="raise SystemExit(4)\n")
    r = call(box, "run_training")
    assert not r.ok and "exit code 4" in r.error
    assert not call(box, "evaluate", run_id="r1").ok


def test_run_without_artifacts_is_not_evaluable(box):
    call(box, "write_file", path="train.py", content="print('trained!')\n")
    call(box, "write_file", path="policy.py", content=GOOD_POLICY)
    r = call(box, "run_training")
    assert not r.ok and "wrote nothing" in r.error
    e = call(box, "evaluate", run_id="r1")
    assert not e.ok and "did not succeed" in e.error


def test_run_limit(box):
    call(box, "write_file", path="train.py", content="import sys\nimport pathlib\n"
         "pathlib.Path(sys.argv[4]).mkdir(parents=True)\n(pathlib.Path(sys.argv[4]) / 'a.json').write_text('1')\n")
    for _ in range(TASK.budget.max_runs):
        assert call(box, "run_training").ok
    r = call(box, "run_training")
    assert not r.ok and "no training runs left" in r.error


def test_evaluate_unknown_run_and_bad_policy(box):
    assert "unknown run" in call(box, "evaluate", run_id="r9").error
    call(box, "write_file", path="train.py", content=GOOD_TRAIN)
    call(box, "write_file", path="policy.py", content="import socket\n")
    assert not call(box, "run_training").ok  # the whole workspace is checked before any run
    call(box, "write_file", path="policy.py", content=GOOD_POLICY)
    assert call(box, "run_training").ok
    call(box, "write_file", path="policy.py", content="import socket\n")  # swapped after training
    e = call(box, "evaluate", run_id="r1")
    assert not e.ok and "static check failed" in e.error


def test_submit_reports_missing_files(box):
    call(box, "write_file", path="train.py", content=GOOD_TRAIN)
    r = call(box, "submit")
    assert r.ok and box.submitted and r.data["missing_files"] == ["policy.py", "REPORT.md"]


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #


def test_budget_wall_clock_and_tool_calls():
    clock = FakeClock()
    b = Budget(TASK, clock)
    assert b.exhausted() is None and b.remaining()["wall_clock_s"] == TASK.budget.wall_clock_s
    clock.t = TASK.budget.wall_clock_s + 1
    assert b.exhausted() == "wall_clock" and b.wall_remaining() == 0
    b2 = Budget(TASK)
    b2.tool_calls = TASK.budget.max_tool_calls
    assert b2.exhausted() == "tool_calls"


def test_run_timeout_is_capped_by_remaining_wall_clock(tmp_path):
    clock = FakeClock()
    b = Budget(TASK, clock)
    box = ToolBox(TASK, Workspace(tmp_path / "ws"), tmp_path / "logs", b)
    call(box, "write_file", path="train.py", content="import time\ntime.sleep(30)\n")
    clock.t = TASK.budget.wall_clock_s - 1.5  # only 1.5 s left
    r = call(box, "run_training")
    assert not r.ok and r.data["killed_reason"] == "timeout" and r.data["duration_s"] < 5
