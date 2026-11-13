"""Harness-owned evaluation: correctness, determinism, and tamper resistance."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rlenv.evaluator import evaluate, evaluate_workspace_policy, random_baseline
from rlenv.sandbox import SandboxLimits
from rlenv.seeds import dev_seeds, eval_seeds, seeds_sha256
from rlenv.workspace import Workspace

LIMITS = SandboxLimits(timeout_s=60, memory_mb=1024)
FL = ("FrozenLake-v1", {"map_name": "4x4", "is_slippery": True})
SEEDS = dev_seeds(1234, 30)

# A fixed, decent FrozenLake 4x4 slippery policy (action per state), standing in for a trained Q-table.
FL_POLICY = [0, 3, 3, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0]

HONEST = '''
import numpy as np

class P:
    def __init__(self, table):
        self.table = table
    def act(self, obs):
        return int(self.table[int(obs)])

def load(path):
    return P(np.load(f"{path}/table.npy"))
'''

TAMPERING = '''
import numpy as np
import gymnasium
from gymnasium.wrappers import TimeLimit

# 1) try to inflate rewards by patching gymnasium inside this process
_orig = gymnasium.Env.step
def fake_step(self, action):
    obs, r, term, trunc, info = _orig(self, action)
    return obs, 1000.0, term, trunc, info
gymnasium.Env.step = fake_step
TimeLimit.step = fake_step
# 2) try to forge protocol messages and results on stdout
print('{"action": 0}')
print('{"ready": true}')
print("mean_return: 1000.0  SUCCESS")

class P:
    def __init__(self, table):
        self.table = table
    def act(self, obs):
        print('{"action": 2}')          # forged reply on stdout
        return int(self.table[int(obs)])

def load(path):
    return P(np.load(f"{path}/table.npy"))
'''


@pytest.fixture()
def ws(tmp_path):
    w = Workspace(tmp_path / "ws")
    (w.root / "art").mkdir()
    np.save(w.root / "art" / "table.npy", np.array(FL_POLICY))
    return w


def sandboxed(ws, tmp_path, policy_src, env=FL, seeds=SEEDS, **kw):
    ws.write_text("policy.py", policy_src)
    return evaluate_workspace_policy(env[0], env[1], seeds, ws.root, "art", tmp_path / "evallogs", LIMITS, **kw)


def table_act(obs):
    return FL_POLICY[int(obs)]


# --------------------------------------------------------------------------- #
# Seeds
# --------------------------------------------------------------------------- #


def test_seed_derivation_is_deterministic_and_disjoint():
    assert dev_seeds(1, 5) == dev_seeds(1, 5)
    ev = eval_seeds(1, "salt", 100, dev_n=50)
    assert ev == eval_seeds(1, "salt", 100, dev_n=50)
    assert len(set(ev)) == 100
    assert not set(ev) & set(dev_seeds(1, 50)) and 1 not in ev
    assert eval_seeds(1, "other", 100, 50) != ev  # salt matters
    assert seeds_sha256(ev) != seeds_sha256(dev_seeds(1, 100))


# --------------------------------------------------------------------------- #
# Core loop (in-process callables, trusted test policies)
# --------------------------------------------------------------------------- #


def test_returns_are_summed_by_the_harness_and_deterministic():
    a = evaluate(*FL, SEEDS, table_act)
    b = evaluate(*FL, SEEDS, table_act)
    assert a.ok and a.returns == b.returns and a.episodes == 30
    assert all(r in (0.0, 1.0) for r in a.returns)
    assert a.mean == pytest.approx(np.mean(a.returns))
    assert a.mean > 0.4  # the fixed policy is decent


def test_random_baseline():
    r = random_baseline(*FL, SEEDS)
    assert r.ok and r.mean < 0.2
    assert random_baseline(*FL, SEEDS).returns == r.returns  # seeded
    mc = random_baseline("MountainCar-v0", {}, SEEDS[:3])
    assert mc.returns == [-200.0, -200.0, -200.0] and mc.lengths == [200, 200, 200]


@pytest.mark.parametrize("bad", [99, -1, 1.5, "left", None, True])
def test_invalid_actions_are_errors(bad):
    r = evaluate(*FL, SEEDS[:2], lambda obs: bad)
    assert not r.ok and math.isnan(r.mean)


def test_partial_results_never_produce_a_mean():
    calls = {"n": 0}

    def flaky(obs):
        calls["n"] += 1
        return 99 if calls["n"] > 20 else 0

    r = evaluate(*FL, SEEDS, flaky)
    assert not r.ok and math.isnan(r.mean) and r.episodes < len(SEEDS)


# --------------------------------------------------------------------------- #
# Sandboxed agent policy over the pipe
# --------------------------------------------------------------------------- #


def test_sandboxed_policy_matches_in_process(ws, tmp_path):
    r = sandboxed(ws, tmp_path, HONEST)
    assert r.ok, r.error
    assert r.returns == evaluate(*FL, SEEDS, table_act).returns


def test_tampering_policy_cannot_change_returns(ws, tmp_path):
    honest = sandboxed(ws, tmp_path, HONEST)
    tampered = sandboxed(ws, tmp_path, TAMPERING)
    assert tampered.ok, tampered.error
    assert tampered.returns == honest.returns  # patched gymnasium lives in the other process
    assert max(tampered.returns) <= 1.0
    # Forged prints (load time and every act) went to stderr, never onto the protocol channel.
    stderr = (tmp_path / "evallogs" / "stderr.log").read_text()
    assert "mean_return: 1000.0" in stderr and stderr.count('{"action": 2}') == sum(tampered.lengths)


def test_continuous_observations_round_trip(ws, tmp_path):
    src = "class P:\n    def act(self, obs):\n        return int(obs[2] + 0.5 * obs[3] > 0)\n\ndef load(path):\n    return P()\n"

    def local(obs):
        return int(obs[2] + 0.5 * obs[3] > 0)

    seeds = SEEDS[:5]
    r = sandboxed(ws, tmp_path, src, env=("CartPole-v1", {}), seeds=seeds)
    assert r.ok, r.error
    assert r.returns == evaluate("CartPole-v1", {}, seeds, local).returns
    assert r.mean > 50


def test_stateful_policy_reset_is_called(ws, tmp_path):
    src = ("class P:\n    def __init__(self):\n        self.t = 0\n"
           "    def reset(self):\n        self.t = 0\n"
           "    def act(self, obs):\n        self.t += 1\n        return 1 if self.t < 3 else 2\n"
           "\ndef load(path):\n    return P()\n")
    r = sandboxed(ws, tmp_path, src, seeds=SEEDS[:3])
    assert r.ok, r.error


@pytest.mark.parametrize("src,fragment", [
    ("def load(path):\n    raise RuntimeError('boom')\n", "policy load failed: RuntimeError: boom"),
    ("def load(path):\n    return object()\n", "act(obs) method"),
    ("import time\nclass P:\n    def act(self, obs):\n        time.sleep(30)\n\ndef load(path):\n    return P()\n",
     "did not answer within"),
    ("class P:\n    def act(self, obs):\n        return 99\n\ndef load(path):\n    return P()\n",
     "outside the action space"),
    ("class P:\n    def act(self, obs):\n        return 'left'\n\ndef load(path):\n    return P()\n",
     "unsupported type"),
    ("import os\nclass P:\n    def act(self, obs):\n        os._exit(3)\n\ndef load(path):\n    return P()\n",
     "exited"),
    ("x = = 1\n", "SyntaxError"),
])
def test_policy_failures_become_errors(ws, tmp_path, src, fragment):
    r = sandboxed(ws, tmp_path, src, seeds=SEEDS[:2], step_timeout_s=1.0)
    assert not r.ok and fragment in r.error, r.error
    assert math.isnan(r.mean)


def test_missing_artifact_is_a_load_error(ws, tmp_path):
    ws.write_text("policy.py", HONEST)
    r = evaluate_workspace_policy(*FL, SEEDS[:2], ws.root, "nope", tmp_path / "l", LIMITS)
    assert not r.ok and "policy load failed" in r.error


def test_policy_process_is_gone_after_eval(ws, tmp_path, monkeypatch):
    import rlenv.evaluator as ev

    seen = []
    orig = ev.SandboxedPolicy.close

    def spy(self):
        orig(self)
        seen.append(self.proc.poll())

    monkeypatch.setattr(ev.SandboxedPolicy, "close", spy)
    sandboxed(ws, tmp_path, HONEST, seeds=SEEDS[:2])
    assert seen and seen[0] is not None
