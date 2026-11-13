"""Trajectories: append-only hash chain, tamper detection, redaction, blobs, end-to-end recording."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml
from conftest import fast_task

from rlenv import cli
from rlenv.agents.base import AgentDriver, ToolCall
from rlenv.agents.scripted import ScriptedAgent
from rlenv.evaluator import evaluate
from rlenv.harness import run_episode
from rlenv.task import TaskError, load_tasks, parse_task
from rlenv.trajectory import (
    GENESIS,
    SCHEMA_VERSION,
    Recorder,
    Redactor,
    TrajectoryError,
    TrajectoryWriter,
    load_trajectory,
)

ROOT = Path(__file__).resolve().parents[1]
TASKS = load_tasks(ROOT / "tasks")
SECRET = "sk-live-0123456789abcdefSECRET"


def write_sample(d: Path, n: int = 5, redactor: Redactor | None = None) -> Path:
    w = TrajectoryWriter(d, redactor or Redactor({}))
    w.write({"type": "header", "schema": SCHEMA_VERSION, "episode_id": "e"})
    for i in range(n):
        w.write({"type": "event", "i": i})
    w.close()
    return d


def lines(d: Path) -> list[str]:
    return (d / "trajectory.jsonl").read_text().splitlines()


def rewrite(d: Path, new_lines: list[str]) -> None:
    (d / "trajectory.jsonl").write_text("".join(x + "\n" for x in new_lines))


# --------------------------------------------------------------------------- #
# Chain and tamper detection
# --------------------------------------------------------------------------- #


def test_roundtrip_chain_and_head(tmp_path):
    d = write_sample(tmp_path / "t")
    t = load_trajectory(d)
    assert t.ok and t.head_ok and len(t.records) == 6
    assert t.records[0]["prev"] == GENESIS and [r["seq"] for r in t.records] == list(range(6))
    assert t.records[1]["prev"] == hashlib.sha256(lines(d)[0].encode()).hexdigest()


def test_modified_middle_line_breaks_chain(tmp_path):
    d = write_sample(tmp_path / "t")
    ls = lines(d)
    rec = json.loads(ls[2])
    rec["i"] = 999
    ls[2] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    rewrite(d, ls)
    t = load_trajectory(d)
    assert not t.ok and any("line 4: hash chain broken" in p for p in t.problems)


def test_modified_last_line_caught_by_head(tmp_path):
    d = write_sample(tmp_path / "t")
    ls = lines(d)
    ls[-1] = ls[-1].replace('"i":4', '"i":5')
    rewrite(d, ls)
    t = load_trajectory(d)
    assert not t.ok and t.head_ok is False


def test_truncation_caught_by_head(tmp_path):
    d = write_sample(tmp_path / "t")
    rewrite(d, lines(d)[:-2])
    t = load_trajectory(d)
    assert not t.ok and any("truncated" in p for p in t.problems)


def test_removed_middle_line(tmp_path):
    d = write_sample(tmp_path / "t")
    ls = lines(d)
    del ls[3]
    rewrite(d, ls)
    t = load_trajectory(d)
    assert any("seq" in p for p in t.problems) and any("chain broken" in p for p in t.problems)


def test_invalid_json_line(tmp_path):
    d = write_sample(tmp_path / "t")
    ls = lines(d)
    ls[1] = "{not json"
    rewrite(d, ls)
    assert any("line 2: invalid JSON" in p for p in load_trajectory(d).problems)


def test_writer_is_append_only_per_episode(tmp_path):
    d = write_sample(tmp_path / "t")
    with pytest.raises(TrajectoryError, match="append-only"):
        TrajectoryWriter(d)


def test_unknown_schema(tmp_path):
    w = TrajectoryWriter(tmp_path / "t", Redactor({}))
    w.write({"type": "header", "schema": "something/v9"})
    w.close()
    assert any("unsupported schema" in p for p in load_trajectory(tmp_path / "t").problems)


# --------------------------------------------------------------------------- #
# Redaction and blobs
# --------------------------------------------------------------------------- #


def test_secrets_are_redacted_everywhere(tmp_path):
    red = Redactor({"DEEPSEEK_API_KEY": SECRET, "HOME": "/Users/x", "MY_TOKEN": "tok-abcdefgh"})
    w = TrajectoryWriter(tmp_path / "t", red)
    w.write({"type": "header", "schema": SCHEMA_VERSION})
    w.write({"type": "event", "msg": f"auth {SECRET}", "nested": ["tok-abcdefgh", "sk-otherKEYabcdefghijkl"],
             "cwd": "/Users/x/work"})
    ref = w.blob(f"key = '{SECRET}'\n")
    w.close()
    everything = "".join(p.read_text(errors="replace") for p in (tmp_path / "t").rglob("*") if p.is_file())
    assert SECRET not in everything and "tok-abcdefgh" not in everything and "sk-otherKEY" not in everything
    assert "[REDACTED:DEEPSEEK_API_KEY]" in everything and "[REDACTED:MY_TOKEN]" in everything
    assert "/Users/x" in everything  # non-secret env values are untouched
    assert load_trajectory(tmp_path / "t").ok and ref["blob"]


def test_blob_corruption_and_loss_detected(tmp_path):
    w = TrajectoryWriter(tmp_path / "t", Redactor({}))
    w.write({"type": "header", "schema": SCHEMA_VERSION})
    ref = w.blob("print('code')\n")
    ref2 = w.blob("print('more')\n")
    w.write({"type": "event", "a": ref, "b": ref2})
    w.close()
    blobs = tmp_path / "t" / "blobs"
    (blobs / ref["blob"]).write_text("print('evil')\n")
    (blobs / ref2["blob"]).unlink()
    problems = load_trajectory(tmp_path / "t").problems
    assert any("does not match its hash" in p for p in problems) and any("missing blob" in p for p in problems)


def test_recorder_moves_code_and_large_strings_to_blobs(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "train.py").write_text("print(1)\n")
    w = TrajectoryWriter(tmp_path / "t", Redactor({}))
    rec = Recorder(w, ws)
    w.write({"type": "header", "schema": SCHEMA_VERSION})
    rec({"type": "tool_call", "tool": "write_file", "args": {"path": "train.py", "content": "print(1)\n"}})
    sha = hashlib.sha256(b"print(1)\n").hexdigest()
    rec({"type": "run", "code": {"train.py": sha}, "stdout_tail": "x" * 20000})
    rec.close()
    t = load_trajectory(tmp_path / "t")
    call, run = t.records[1], t.records[2]
    assert t.blob_text(call["args"]["content"]) == "print(1)\n"
    assert run["code"]["train.py"] == {"sha256": sha, "blob": sha, "matches": True}
    assert run["stdout_tail"]["bytes"] == 20000


# --------------------------------------------------------------------------- #
# End-to-end recording
# --------------------------------------------------------------------------- #


def test_episode_trajectory_end_to_end(tmp_path, monkeypatch, fast_reference):
    monkeypatch.setenv("DEEPSEEK_API_KEY", SECRET)
    task = fast_task(TASKS["frozenlake-tabular"])
    rec = run_episode(task, ScriptedAgent(task.id, ["--episodes", "1000"], reference_root=fast_reference), tmp_path)
    t = load_trajectory(rec.trajectory_dir)
    assert t.ok, t.problems
    h = t.header
    assert h["schema"] == SCHEMA_VERSION and h["task_public_sha256"] == task.public_sha256()
    assert h["runtime"]["gymnasium"] == "1.4.0" and h["agent"]["kind"] == "scripted"
    types = [r["type"] for r in t.records]
    assert types[:2] == ["header", "start"] and types[-4:] == ["end", "outcome", "verification", "final"]
    assert t.of_type("outcome")[0]["end_reason"] == "submitted"

    # Every code version the agent wrote is recoverable and matches the workspace.
    ws = rec.trajectory_dir / "workspace"
    run = t.of_type("run")[0]
    for path, info in run["code"].items():
        assert info["matches"] and t.blob_text(info) == (ws / path).read_text()

    # No held-out salt and no secret anywhere in the episode directory (outside the workspace).
    text = "".join(p.read_text(errors="replace") for p in rec.trajectory_dir.rglob("*")
                   if p.is_file() and "workspace" not in p.parts and p.suffix != ".npy")
    assert task.success.eval_seed_salt not in text and SECRET not in text


class CrashAfterWrite(AgentDriver):
    kind = "crash-after-write"

    def start(self, observation):
        self.n = 0

    def next_action(self, last):
        self.n += 1
        if self.n == 1:
            return ToolCall("write_file", {"path": "train.py", "content": "print('hi')\n"})
        raise RuntimeError("driver died")


def test_crashing_agent_still_leaves_a_valid_trajectory(tmp_path):
    rec = run_episode(TASKS["frozenlake-tabular"], CrashAfterWrite(), tmp_path)
    t = load_trajectory(rec.trajectory_dir)
    assert t.ok and t.of_type("outcome")[0]["end_reason"] == "agent_error"
    assert t.of_type("agent_error")[0]["error"] == "RuntimeError: driver died"


@pytest.mark.slow
def test_cli_run_and_check_trajectory(tmp_path, capsys, fast_reference):
    assert cli.main(["run", "frozenlake-tabular", "--tasks", str(ROOT / "tasks"), "--episodes", "300",
                     "--out", str(tmp_path), "--reference-dir", str(fast_reference)]) in (0, 2)
    traj_line = [x for x in capsys.readouterr().out.splitlines() if x.startswith("trajectory: ")][0]
    d = Path(traj_line.split(": ", 1)[1]).parent
    assert cli.main(["check-trajectory", str(d)]) == 0
    assert "OK:" in capsys.readouterr().out
    (d / "trajectory.head").write_text("1 " + "0" * 64 + "\n")
    assert cli.main(["check-trajectory", str(d)]) == 2


# --------------------------------------------------------------------------- #
# Step limits (envs without a registered TimeLimit)
# --------------------------------------------------------------------------- #


def _cliff_task(raw_overrides: dict):
    raw = yaml.safe_load((ROOT / "tasks" / "frozenlake-tabular.yaml").read_text())
    raw = copy.deepcopy(raw)
    raw.pop("verification", None)  # FrozenLake's random-map variants don't apply to CliffWalking
    raw.update({"id": "cliff", "env_id": "CliffWalking-v1", "env_kwargs": {}, **raw_overrides})
    return raw


def test_env_without_step_limit_requires_max_episode_steps():
    with pytest.raises(TaskError, match="no registered step limit"):
        parse_task(_cliff_task({}))
    task = parse_task(_cliff_task({"max_episode_steps": 200}))
    assert task.episode_limit() == 200
    assert TASKS["mountaincar-tilecoding"].episode_limit() == 200  # from the env spec


def test_step_limit_truncates_a_policy_that_never_finishes():
    r = evaluate("CliffWalking-v1", {}, [1, 2], lambda obs: 0, max_episode_steps=50)  # always "up"
    assert r.ok and r.lengths == [50, 50] and r.returns == [-50.0, -50.0]


def test_home_directory_becomes_tilde(tmp_path):
    red = Redactor({}, home="/home/alice")
    assert red("/home/alice/proj/ws") == "~/proj/ws" and red("/home/alice") == "~"
    assert red("/home/alicex/ws") == "/home/alicex/ws" and red.count == 0  # not a secret, not counted
    assert Redactor({}, home="/")("/etc/x") == "/etc/x"


def test_episode_trajectory_has_no_local_paths(tmp_path, fast_reference):
    task = fast_task(load_tasks(Path(__file__).resolve().parents[1] / "tasks")["frozenlake-tabular"],
                     eval_episodes=50)
    rec = run_episode(task, ScriptedAgent(task.id, ["--episodes", "500"], reference_root=fast_reference),
                      tmp_path / "runs", redactor=Redactor({}, home=tmp_path))
    text = (rec.trajectory_dir / "trajectory.jsonl").read_text()
    assert str(tmp_path) not in text and "~/runs/" in text
    assert load_trajectory(rec.trajectory_dir).ok
