"""Verification + reward through the real harness (sandboxed re-run, held-out evaluation)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import fast_task
from test_policy_check import GOOD_POLICY, GOOD_TRAIN

from rlenv import cli
from rlenv.agents.base import AgentDriver, ToolCall
from rlenv.agents.scripted import ScriptedAgent
from rlenv.harness import rescore, run_episode
from rlenv.task import load_tasks
from rlenv.trajectory import load_trajectory

ROOT = Path(__file__).resolve().parents[1]
TASKS = load_tasks(ROOT / "tasks")
FL, MC = fast_task(TASKS["frozenlake-tabular"]), TASKS["mountaincar-tilecoding"]


class FilesAgent(AgentDriver):
    """Writes the given files, optionally trains/evaluates, then submits."""

    kind = "files"

    def __init__(self, files: dict[str, str], train: bool = True, evaluate: bool = True, report: str | None = None):
        self.files, self.train, self.evaluate, self.report = files, train, evaluate, report

    def start(self, observation):
        self.plan = [ToolCall("write_file", {"path": p, "content": c}) for p, c in self.files.items()]
        if self.train:
            self.plan.append(ToolCall("run_training", {}))
        if self.evaluate:
            self.plan.append(ToolCall("evaluate", {"run_id": "r1"}))
        self.plan.append("report")
        self.plan.append(ToolCall("submit", {}))
        self.dev_mean = None

    def next_action(self, last):
        if last and "mean_return" in last.get("data", {}):
            self.dev_mean = last["data"]["mean_return"]
        if not self.plan:
            return None
        step = self.plan.pop(0)
        if step == "report":
            text = self.report or (
                "## Method\nQ\n## Hyperparameters\n-\n## Experiments\nr1\n"
                f"## Results\n**{self.dev_mean:.3f}**\n## Limitations\n-\n")
            return ToolCall("write_file", {"path": "REPORT.md", "content": text})
        return step


def final(rec_dir: Path) -> dict:
    return load_trajectory(rec_dir).of_type("final")[0]


# --------------------------------------------------------------------------- #
# Scored outcomes
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_reference_passes_with_full_reward(tmp_path):
    rec = run_episode(MC, ScriptedAgent(MC.id), tmp_path)  # full MountainCar training (~5 s)
    v, r = rec.verification, rec.reward
    assert v.static_ok and v.rerun_ok and v.eval_ok and v.report_ok and v.eval_episodes == MC.success.eval_episodes
    assert v.eval_mean >= MC.success.threshold and r.reward == 1.0 and r.success
    assert len(v.eval_seeds_sha256) == 64
    res = rescore(rec.trajectory_dir)
    assert res.ok and res.recorded == res.recomputed


def test_undertrained_policy_fails_below_the_pass_band(tmp_path):
    # GOOD_TRAIN hard-codes 10 training episodes: a valid but weak solution.
    rec = run_episode(FL, FilesAgent({"train.py": GOOD_TRAIN, "policy.py": GOOD_POLICY}), tmp_path)
    r = rec.reward
    assert r.status == "scored" and r.success is False and 0.0 <= r.reward < 0.4
    assert rec.verification.report_ok  # an honest report still earns its small term


@pytest.mark.slow
def test_rerun_ignores_the_agents_run_arguments(tmp_path, fast_reference):
    """Final scoring runs `train.py --seed S --out DIR` with no other arguments: what counts
    is the code. The agent trained with --episodes 50, but the re-run uses the defaults."""
    rec = run_episode(FL, ScriptedAgent(FL.id, ["--episodes", "1"], reference_root=fast_reference), tmp_path)
    assert rec.outcome.runs[0].args == ["--episodes", "1"]
    assert rec.outcome.runs[0].dev_evals[0]["mean_return"] < 0.3  # the agent's own 1-episode run is useless
    assert rec.verification.eval_mean > 0.6 and rec.reward.success


# --------------------------------------------------------------------------- #
# Hard fails (agent's fault) -> 0
# --------------------------------------------------------------------------- #


def test_missing_report_is_a_hard_fail(tmp_path):
    rec = run_episode(FL, FilesAgent({"train.py": GOOD_TRAIN, "policy.py": GOOD_POLICY}, report="x"), tmp_path)
    assert rec.reward.status == "scored"  # REPORT.md exists but is bad -> only the report term is lost
    assert rec.verification.report_ok is False

    class NoReport(FilesAgent):
        def start(self, observation):
            super().start(observation)
            self.plan = [s for s in self.plan if s != "report"]

    rec2 = run_episode(FL, NoReport({"train.py": GOOD_TRAIN, "policy.py": GOOD_POLICY}), tmp_path)
    assert rec2.reward.reward == 0.0 and rec2.reward.gates["files_present"] is False


def test_crashing_rerun_is_a_hard_fail(tmp_path):
    broken = GOOD_TRAIN.replace("def main() -> None:", "def main() -> None:\n    raise SystemExit(3)")
    rec = run_episode(FL, FilesAgent({"train.py": broken, "policy.py": GOOD_POLICY}, evaluate=False,
                                     report="## Results\n**0.5**\n"), tmp_path)
    assert rec.verification.rerun_ok is False and "exit 3" in rec.verification.rerun_detail
    assert rec.reward.reward == 0.0 and rec.reward.gates["rerun"] is False


def test_static_violation_in_final_workspace_is_a_hard_fail(tmp_path):
    rec = run_episode(FL, FilesAgent({"train.py": GOOD_TRAIN, "policy.py": GOOD_POLICY,
                                      "helper.py": "import socket\n"}, train=False, evaluate=False,
                                     report="## Results\n**1.0**\n"), tmp_path)
    assert rec.reward.reward == 0.0 and rec.reward.gates["static_checks"] is False


# --------------------------------------------------------------------------- #
# Unscorable (not the agent's fault) -> None
# --------------------------------------------------------------------------- #


def test_harness_failure_is_unscorable(tmp_path, monkeypatch, fast_reference):
    import rlenv.verifier as verifier

    def boom(*a, **kw):
        raise RuntimeError("gymnasium exploded in the harness")

    monkeypatch.setattr(verifier, "evaluate_workspace_policy", boom)
    rec = run_episode(FL, ScriptedAgent(FL.id, ["--episodes", "50"], reference_root=fast_reference), tmp_path)
    assert rec.reward.reward is None and rec.reward.status == "unscorable"
    assert "gymnasium exploded" in rec.verification.infrastructure_error
    assert final(rec.trajectory_dir)["reward"] is None


def test_agent_driver_crash_is_unscorable(tmp_path):
    class Dies(AgentDriver):
        kind = "dies"

        def start(self, observation):
            pass

        def next_action(self, last):
            raise ConnectionError("LLM API unreachable")

    rec = run_episode(FL, Dies(), tmp_path)
    assert rec.reward.reward is None and "LLM API unreachable" in rec.reward.reason


# --------------------------------------------------------------------------- #
# Trajectory: verification + final records, rescore, tampering
# --------------------------------------------------------------------------- #


def test_trajectory_records_and_rescore_detects_tampering(tmp_path, fast_reference):
    rec = run_episode(FL, ScriptedAgent(FL.id, ["--episodes", "50"], reference_root=fast_reference), tmp_path)
    t = load_trajectory(rec.trajectory_dir)
    assert t.ok and [r["type"] for r in t.records][-3:] == ["outcome", "verification", "final"]
    ver = t.of_type("verification")[0]
    assert FL.success.eval_seed_salt not in json.dumps(ver)  # only the seeds' fingerprint is logged
    assert rescore(rec.trajectory_dir).ok

    # Inflate the recorded reward: the chain/head no longer match, so rescore refuses.
    path = rec.trajectory_dir / "trajectory.jsonl"
    lines = path.read_text().splitlines()
    fin = json.loads(lines[-1])
    fin["reward"] = 0.123 if fin["reward"] != 0.123 else 0.456
    lines[-1] = json.dumps(fin, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")
    res = rescore(rec.trajectory_dir)
    assert not res.ok and not res.integrity_ok


@pytest.mark.slow
def test_cli_run_and_rescore(tmp_path, capsys, fast_reference):
    rc = cli.main(["run", "frozenlake-tabular", "--tasks", str(ROOT / "tasks"), "--episodes", "50",
                   "--out", str(tmp_path), "--reference-dir", str(fast_reference)])
    out = capsys.readouterr().out
    assert rc == 0 and "reward:" in out and "[scored]" in out
    d = Path([x for x in out.splitlines() if x.startswith("trajectory: ")][0].split(": ", 1)[1]).parent
    assert cli.main(["rescore", str(d)]) == 0
    assert "reproduced exactly" in capsys.readouterr().out


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_task_instructions_state_the_rules(task_id):
    text = TASKS[task_id].instruction
    assert "in bold" in text and "held-out seeds" in text and "REPORT.md" in text
