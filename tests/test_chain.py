"""The end-to-end chain, grown step by step:
workspace -> static check -> sandboxed training -> harness-owned evaluation."""

from __future__ import annotations

import numpy as np
from test_policy_check import GOOD_POLICY, GOOD_TRAIN

from rlenv.evaluator import evaluate_workspace_policy, random_baseline
from rlenv.policy_check import check_workspace
from rlenv.sandbox import SandboxLimits, SubprocessSandbox
from rlenv.seeds import dev_seeds
from rlenv.workspace import Workspace


def test_agent_code_trains_in_sandbox_and_writes_artifact(tmp_path):
    ws = Workspace(tmp_path / "ws")
    ws.write_text("train.py", GOOD_TRAIN)
    ws.write_text("policy.py", GOOD_POLICY)
    report = check_workspace(ws.root, required=("train.py", "policy.py"))
    assert report.ok, report.summary()

    result = SubprocessSandbox().run_python(
        "train.py", ["--seed", "1234", "--out", "runs/r1/policy"], ws.root, tmp_path / "logs",
        SandboxLimits(timeout_s=60))
    assert result.ok, result.stderr_tail
    q = np.load(ws.root / "runs/r1/policy/q.npy")
    assert q.shape == (16, 4)
    assert set(ws.snapshot()) == {"train.py", "policy.py"}  # artifacts are not code

    # Harness-owned evaluation of the trained artifact (policy.py runs sandboxed).
    seeds = dev_seeds(1234, 20)
    result = evaluate_workspace_policy("FrozenLake-v1", {"is_slippery": True}, seeds, ws.root,
                                       "runs/r1/policy", tmp_path / "evallogs", SandboxLimits(timeout_s=60))
    assert result.ok, result.error
    assert result.episodes == 20 and 0.0 <= result.mean <= 1.0
    assert random_baseline("FrozenLake-v1", {"is_slippery": True}, seeds).ok
