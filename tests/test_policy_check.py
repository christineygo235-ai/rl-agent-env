"""Static checks: a normal tabular solution passes; every rule can fail."""

from __future__ import annotations

import os

import pytest

from rlenv.policy_check import check_source, check_workspace

GOOD_TRAIN = '''
import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--env-kwargs", default="{}")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    env = gym.make("FrozenLake-v1", **json.loads(args.env_kwargs))
    q = np.zeros((env.observation_space.n, env.action_space.n))
    for ep in range(10):
        s, _ = env.reset(seed=args.seed + ep)
        done = False
        while not done:
            a = int(rng.integers(env.action_space.n)) if rng.random() < 0.1 else int(q[s].argmax())
            s2, r, term, trunc, _ = env.step(a)
            q[s, a] += 0.1 * (r + 0.99 * q[s2].max() * (not term) - q[s, a])
            s, done = s2, term or trunc
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "q.npy", q)
    (out / "meta.json").write_text(json.dumps({"episodes": 10}))


if __name__ == "__main__":
    main()
'''

GOOD_POLICY = '''
import numpy as np


class Greedy:
    def __init__(self, q):
        self.q = q

    def act(self, obs):
        return int(self.q[obs].argmax())


def load(path):
    return Greedy(np.load(f"{path}/q.npy"))
'''


def violations(src, **kw):
    report = check_source(src, "train.py", **kw)
    return [(v.rule, v.message) for v in report.violations]


def test_good_solution_passes():
    assert check_source(GOOD_TRAIN).ok
    assert check_source(GOOD_POLICY).ok


@pytest.mark.parametrize("src,rule,fragment", [
    ("import subprocess", "forbidden-import", "starting processes"),
    ("from subprocess import run", "forbidden-import", "starting processes"),
    ("import socket", "forbidden-import", "network"),
    ("import urllib.request", "forbidden-import", "network"),
    ("import pickle", "forbidden-import", "artifacts must be data"),
    ("import importlib", "forbidden-import", "dynamic imports"),
    ("import ctypes", "forbidden-import", "FFI"),
    ("import threading", "forbidden-import", "threads"),
    ("from stable_baselines3 import PPO", "forbidden-import", "RL libraries"),
    ("import torch", "forbidden-import", "torch is not enabled"),
    ("import pandas", "import-not-allowed", "not on the allowlist"),
])
def test_forbidden_imports(src, rule, fragment):
    found = violations(src)
    assert any(r == rule and fragment in m for r, m in found), found


@pytest.mark.parametrize("src", [
    "eval('1+1')",
    "exec('x=1')",
    "__import__('subprocess')",
    "compile('x', 'f', 'exec')",
    "import os\nos.system('curl evil')",
    "import os as o\no.popen('ls')",
    "from os import system",
    "from os import *",
    "import os\nos.execv('/bin/sh', [])",
    "import os\nos.symlink('/etc/passwd', 'x')",
])
def test_forbidden_calls(src):
    assert violations(src), src


@pytest.mark.parametrize("src,rule", [
    ("import os\ngetattr(os, 'sys' + 'tem')('id')", "dynamic-attribute"),
    ("().__class__.__bases__[0].__subclasses__()", "dunder-access"),
    ("f = lambda: 0\nf.__globals__['os']", "dunder-access"),
    ("__builtins__['eval']('1')", "dunder-access"),
    ("import numpy as np\nnp.load('q.npy', allow_pickle=True)", "pickle"),
    ("open('/etc/passwd').read()", "absolute-path"),
])
def test_escape_hatches(src, rule):
    assert rule in [r for r, _ in violations(src)], violations(src)


def test_constant_getattr_and_allow_pickle_false_are_fine():
    assert check_source("import numpy as np\nx = getattr(np, 'float32')\nnp.load('q.npy', allow_pickle=False)").ok


def test_torch_allowed_only_when_enabled():
    assert not check_source("import torch").ok
    assert check_source("import torch", allow_torch=True).ok


def test_syntax_error():
    assert violations("def broken(:\n  pass") == [("syntax", "invalid syntax")]


def test_workspace_check_local_modules_and_required(tmp_path):
    (tmp_path / "train.py").write_text("import policy\nimport helpers\n" + GOOD_TRAIN)
    (tmp_path / "policy.py").write_text(GOOD_POLICY)
    (tmp_path / "helpers.py").write_text("X = 1\n")
    report = check_workspace(tmp_path, required=("train.py", "policy.py"))
    assert report.ok, report.summary()
    assert sorted(report.files) == ["helpers.py", "policy.py", "train.py"]


def test_workspace_missing_required_file(tmp_path):
    (tmp_path / "train.py").write_text(GOOD_TRAIN)
    report = check_workspace(tmp_path, required=("train.py", "policy.py"))
    assert not report.ok and report.violations[0].rule == "missing-file"


def test_workspace_violation_in_any_file(tmp_path):
    (tmp_path / "train.py").write_text(GOOD_TRAIN)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "sneaky.py").write_text("import socket\n")
    report = check_workspace(tmp_path)
    assert not report.ok
    assert "sub/sneaky.py:1: [forbidden-import]" in report.summary()


def test_workspace_rejects_symlinked_source(tmp_path):
    outside = tmp_path.parent / "outside_src.py"
    outside.write_text("import subprocess\n")
    ws = tmp_path / "ws"
    ws.mkdir()
    os.symlink(outside, ws / "train.py")
    report = check_workspace(ws)
    assert [v.rule for v in report.violations] == ["symlink"]
