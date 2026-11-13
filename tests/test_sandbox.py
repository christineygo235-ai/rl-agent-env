"""Sandbox runner: timeouts, process-group kill, rlimits, scrubbed env.

These tests run real subprocesses but stay fast (< ~10 s total)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from rlenv.sandbox import SandboxLimits, SubprocessSandbox
from rlenv.workspace import Workspace


@pytest.fixture()
def env(tmp_path):
    ws = Workspace(tmp_path / "ws")
    return SubprocessSandbox(), ws, tmp_path / "logs"


def run(env, code: str, **limits):
    sb, ws, logs = env
    ws.write_text("main.py", code)
    return sb.run_python("main.py", [], ws.root, logs, SandboxLimits(**{"timeout_s": 10, **limits}))


def test_success_and_output(env):
    r = run(env, "import sys; print('hello'); print('err', file=sys.stderr)")
    assert r.ok and r.exit_code == 0 and r.signal is None
    assert r.stdout_tail.strip() == "hello" and "err" in r.stderr_tail
    assert r.argv[1:4] == ["-E", "-s", "-B"]


def test_nonzero_exit(env):
    r = run(env, "raise SystemExit(3)")
    assert r.exit_code == 3 and not r.ok


def test_exception_traceback_in_stderr(env):
    r = run(env, "1/0")
    assert r.exit_code == 1 and "ZeroDivisionError" in r.stderr_tail


def test_timeout_kills_process(env):
    r = run(env, "import time\nwhile True: time.sleep(0.1)", timeout_s=1)
    assert r.timed_out and r.killed_reason == "timeout" and not r.ok and r.exit_code is None
    assert r.duration_s < 5


def test_timeout_kills_grandchildren(env, tmp_path):
    sb, ws, logs = env
    pidfile = ws.root / "child.pid"
    code = (
        "import os, time\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    while True: time.sleep(0.1)\n"
        f"open({str(pidfile)!r}, 'w').write(str(pid))\n"
        "while True: time.sleep(0.1)\n"
    )
    r = run(env, code, timeout_s=1)
    assert r.timed_out
    child = int(pidfile.read_text())
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)  # the grandchild is gone too


def test_environment_is_scrubbed(env, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-should-not-leak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-should-not-leak")
    r = run(env, "import os, json; print(json.dumps(dict(os.environ)))")
    child_env = json.loads(r.stdout_tail)
    assert "ANTHROPIC_API_KEY" not in child_env and "AWS_SECRET_ACCESS_KEY" not in child_env
    assert "DEEPSEEK_API_KEY" not in child_env and "sk-deepseek" not in r.stdout_tail
    assert child_env["CUDA_VISIBLE_DEVICES"] == "" and child_env["OMP_NUM_THREADS"] == "1"
    assert child_env["HOME"] == str(env[1].root)


def test_cwd_is_workspace_and_pinned_libs_importable(env):
    r = run(env, "import os, gymnasium, numpy; print(os.getcwd()); print(gymnasium.__version__)")
    lines = r.stdout_tail.split()
    assert os.path.realpath(lines[0]) == str(env[1].root)
    assert lines[1] == "1.4.0"


def test_cpu_limit_kills_busy_loop(env):
    r = run(env, "while True: pass", timeout_s=20, cpu_limit_s=1)
    assert r.signal in ("SIGXCPU", "SIGKILL") and not r.timed_out
    assert r.duration_s < 10


def test_file_size_limit(env):
    r = run(env, "open('big.txt','wb').write(b'x' * (3 * 1024 * 1024))", file_size_mb=1)
    assert r.signal == "SIGXFSZ" or r.exit_code != 0


def test_output_spam_is_capped(env):
    r = run(env, "import sys\nfor i in range(200000): print('spam line', i)")
    assert len(r.stdout_tail) <= 4000 and r.stdout_bytes > 1_000_000
    assert "199999" in r.stdout_tail


def test_memory_limit(env):
    # Touch the pages so they count as resident memory; then hold them.
    code = "import time\nx = bytearray(600 * 1024 * 1024)\nx[::4096] = b'x' * len(x[::4096])\ntime.sleep(10)"
    r = run(env, code, memory_mb=256, timeout_s=15)
    assert not r.ok
    # Linux: RLIMIT_AS -> MemoryError; macOS: the RSS watchdog kills the group.
    assert "MemoryError" in r.stderr_tail or (r.killed_reason or "").startswith("memory")
    assert r.duration_s < 10


def test_limits_applied_are_reported(env):
    r = run(env, "pass")
    assert {"RLIMIT_CPU", "RLIMIT_FSIZE", "RLIMIT_NOFILE", "RLIMIT_CORE", "RLIMIT_AS"} <= set(r.limits_applied)
    assert r.limits_applied["RLIMIT_CPU"] and r.limits_applied["RLIMIT_FSIZE"]


def test_relative_paths_work(tmp_path, monkeypatch):
    """Regression (first real LLM episode): the child chdirs to the workspace before
    preexec_fn runs, so a relative logs dir (CLI default `--out runs`) made every run fail."""
    monkeypatch.chdir(tmp_path)
    Workspace(tmp_path / "ws").write_text("main.py", "print('ok')")
    r = SubprocessSandbox().run_python("main.py", [], Path("ws"), Path("runs/logs"), SandboxLimits(timeout_s=10))
    assert r.ok and r.stdout_tail.strip() == "ok"
    assert r.limits_applied["RLIMIT_CPU"] and (tmp_path / "runs/logs/.limits").exists()
