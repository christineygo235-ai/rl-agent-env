"""Exports: messages (OpenAI chat) and nemo-gym rows derived from verified trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import fast_task
from test_policy_check import GOOD_POLICY, GOOD_TRAIN
from test_verifier import FilesAgent

from rlenv import cli
from rlenv.agents.base import AgentDriver
from rlenv.export import export
from rlenv.harness import run_episode
from rlenv.task import load_tasks
from rlenv.tools import TOOL_NAMES, openai_tools
from rlenv.trajectory import TrajectoryWriter, load_trajectory

ROOT = Path(__file__).resolve().parents[1]
FL = fast_task(load_tasks(ROOT / "tasks")["frozenlake-tabular"], eval_episodes=50)
SECRET = "sk-exportleak-0123456789abcdef"


class Dies(AgentDriver):
    kind = "dies"

    def start(self, observation):
        pass

    def next_action(self, last):
        raise ConnectionError("LLM API unreachable")


@pytest.fixture(scope="module")
def episodes(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("DEEPSEEK_API_KEY", SECRET)
    root = tmp_path_factory.mktemp("episodes")
    files = {"train.py": GOOD_TRAIN, "policy.py": GOOD_POLICY}
    scored = [run_episode(FL, FilesAgent(dict(files)), root) for _ in range(2)]
    unscorable = run_episode(FL, Dies(), root)
    mp.undo()
    return root, scored, unscorable


def rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_messages_export(episodes, tmp_path):
    root, scored, _ = episodes
    rep = export([r.trajectory_dir for r in scored], tmp_path / "m.jsonl", "messages")
    assert rep.written == 2 and not rep.skipped_invalid
    row = rows(tmp_path / "m.jsonl")[0]
    assert row["tools"] == openai_tools() and row["status"] == "scored" and isinstance(row["reward"], float)
    msgs = row["messages"]
    assert msgs[0]["role"] == "user" and "FrozenLake" in msgs[0]["content"]
    assert [m["role"] for m in msgs[1:]] == ["assistant", "tool"] * ((len(msgs) - 1) // 2)
    assert row["loss_mask"] == [1 if m["role"] == "assistant" else 0 for m in msgs]
    calls = [m["tool_calls"][0] for m in msgs if m["role"] == "assistant"]
    assert {c["function"]["name"] for c in calls} <= set(TOOL_NAMES)
    # The code the model wrote is inlined back, byte-for-byte.
    write = next(c for c in calls if json.loads(c["function"]["arguments"]).get("path") == "train.py")
    assert json.loads(write["function"]["arguments"])["content"] == GOOD_TRAIN
    tool_ids = [m["tool_call_id"] for m in msgs if m["role"] == "tool"]
    assert tool_ids == [c["id"] for c in calls]
    # Provenance: the head hash ties the row to an intact source trajectory.
    head = (scored[0].trajectory_dir / "trajectory.head").read_text().split()[1]
    assert {r["source"]["trajectory_head_sha256"] for r in rows(tmp_path / "m.jsonl")} >= {head}


def test_group_id_shared_by_episodes_of_one_task(episodes, tmp_path):
    _, scored, _ = episodes
    export([r.trajectory_dir for r in scored], tmp_path / "m.jsonl")
    gids = {r["group_id"] for r in rows(tmp_path / "m.jsonl")}
    assert len(gids) == 1 and gids.pop().startswith(f"frozenlake-tabular@v{FL.version}:")


def test_nemo_gym_export(episodes, tmp_path):
    _, scored, _ = episodes
    export([r.trajectory_dir for r in scored], tmp_path / "n.jsonl", "nemo-gym")
    a, b = rows(tmp_path / "n.jsonl")
    assert {i["type"] for i in a["response"]["output"]} == {"function_call", "function_call_output"}
    assert a["mask_sample"] is False and isinstance(a["reward"], float) and a["rlenv"]["status"] == "scored"
    assert a["rlenv"]["group_id"] == b["rlenv"]["group_id"] and a["rlenv"]["id"] != b["rlenv"]["id"]
    assert (a["_ng_task_index"], a["_ng_rollout_index"], b["_ng_task_index"], b["_ng_rollout_index"]) == (0, 0, 0, 1)
    assert a["responses_create_params"]["tools"][0]["name"] == "write_file"
    assert "reward_success" in a and a["reward_components"]["success"] == a["reward_success"]


def test_unscorable_excluded_by_default_and_masked_on_request(episodes, tmp_path):
    root, scored, unscorable = episodes
    dirs = [r.trajectory_dir for r in scored] + [unscorable.trajectory_dir]
    rep = export(dirs, tmp_path / "m.jsonl")
    assert rep.written == 2 and rep.skipped_unscorable == 1
    export(dirs, tmp_path / "m2.jsonl", include_unscorable=True)
    bad = [r for r in rows(tmp_path / "m2.jsonl") if r["status"] == "unscorable"][0]
    assert bad["reward"] is None and bad["failure_kind"] == "agent_driver_error"
    export(dirs, tmp_path / "n.jsonl", "nemo-gym", include_unscorable=True)
    nbad = [r for r in rows(tmp_path / "n.jsonl") if r["mask_sample"]][0]
    assert nbad["reward"] == 0.0 and nbad["failure_kind"] == "agent_driver_error"


def test_invalid_and_incomplete_trajectories_are_never_exported(episodes, tmp_path):
    root, scored, _ = episodes
    import shutil

    tampered = tmp_path / "tampered"
    shutil.copytree(scored[0].trajectory_dir, tampered)
    path = tampered / "trajectory.jsonl"
    lines = path.read_text().splitlines()
    lines[-1] = lines[-1].replace('"status":"scored"', '"status":"SCORED"')
    path.write_text("\n".join(lines) + "\n")
    w = TrajectoryWriter(tmp_path / "partial")
    w.write({"type": "header", "schema": "rlenv.trajectory/v1"})
    w.close()
    rep = export([tampered, tmp_path / "partial", scored[1].trajectory_dir], tmp_path / "m.jsonl")
    assert rep.written == 1 and len(rep.skipped_invalid) == 1 and len(rep.skipped_incomplete) == 1


def test_no_secrets_in_exports(episodes, tmp_path):
    root, scored, unscorable = episodes
    dirs = [r.trajectory_dir for r in scored] + [unscorable.trajectory_dir]
    export(dirs, tmp_path / "m.jsonl", include_unscorable=True)
    export(dirs, tmp_path / "n.jsonl", "nemo-gym", include_unscorable=True)
    assert SECRET not in (tmp_path / "m.jsonl").read_text() + (tmp_path / "n.jsonl").read_text()
    assert FL.success.eval_seed_salt not in (tmp_path / "m.jsonl").read_text()


def test_final_record_has_group_and_failure_vocabulary(episodes):
    _, scored, unscorable = episodes
    fin = load_trajectory(unscorable.trajectory_dir).of_type("final")[0]
    assert fin["failure_kind"] == "agent_driver_error" and fin["reward"] is None
    assert load_trajectory(scored[0].trajectory_dir).header["runtime"]["git_dirty"] in (True, False)


def test_cli_export_root_directory(episodes, tmp_path, capsys):
    root, _, _ = episodes
    assert cli.main(["export", str(root), "-o", str(tmp_path / "all.jsonl"), "--format", "nemo-gym"]) == 0
    assert "wrote 2 row(s)" in capsys.readouterr().out
