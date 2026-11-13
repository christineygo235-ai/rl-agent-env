"""Run untrusted agent code in a subprocess with limits (PLAN D1, option A).

What this gives:
- wall-clock timeout; the whole process group is killed (grandchildren too);
- rlimits: CPU seconds, file size, open files, no core dumps, address space
  (address space is enforced on Linux; macOS ignores it, recorded in
  `limits_applied`);
- a memory watchdog: the harness polls the process group's RSS (`ps`) and
  kills it above `memory_mb`; this works where RLIMIT_AS does not (macOS);
- a **scrubbed environment**: nothing is inherited from the harness (no API
  keys, no cloud credentials); CPU-only (`CUDA_VISIBLE_DEVICES=""`) and
  single-threaded BLAS for determinism;
- `python -E -s -B`: ignore PYTHON* env vars and user site-packages, no .pyc;
- cwd = the workspace; HOME/TMPDIR point inside the workspace;
- stdout/stderr go to files outside the workspace (capped by the file-size
  rlimit), and the tool result only gets the tail.

What this does NOT give (documented in ADR-0002): network isolation and
filesystem isolation. Use a container backend (`--network none`, read-only
root) for untrusted agents beyond local development.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

TAIL_BYTES = 4000
MEMORY_POLL_S = 0.25


def group_rss_mb(pgid: int) -> float:
    """Resident memory of all processes in a process group (MB), via `ps` (macOS + Linux)."""
    try:
        out = subprocess.run(["ps", "-A", "-o", "pgid=,rss="], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return 0.0
    total_kb = 0
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == str(pgid):
            total_kb += int(parts[1])
    return total_kb / 1024


@dataclass(frozen=True)
class SandboxLimits:
    timeout_s: float
    memory_mb: int = 2048
    file_size_mb: int = 64
    open_files: int = 256
    cpu_limit_s: int | None = None  # default: derived from the wall-clock timeout

    @property
    def cpu_s(self) -> int:
        if self.cpu_limit_s is not None:
            return self.cpu_limit_s
        # Single-threaded code can't use much more CPU time than wall time; small slack.
        return int(self.timeout_s) + 5


@dataclass
class RunResult:
    argv: list[str]
    exit_code: int | None
    signal: str | None
    timed_out: bool
    killed_reason: str | None  # "timeout" | "memory (...)" | None
    duration_s: float
    stdout_tail: str
    stderr_tail: str
    stdout_bytes: int
    stderr_bytes: int
    limits_applied: dict[str, bool] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and self.killed_reason is None

    def to_json(self) -> dict:
        return asdict(self)


def scrubbed_env(workspace: Path) -> dict[str, str]:
    tmp = workspace / ".tmp"
    tmp.mkdir(exist_ok=True, mode=0o700)
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(workspace),
        "TMPDIR": str(tmp),
        "LANG": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        # CPU-only, single-threaded numerics (determinism, fair budgets).
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


def _absolute(path: Path) -> Path:
    """The child chdirs to the workspace *before* preexec_fn runs, so every path handed to the
    child (logs, .limits) must be absolute; a relative `--out` once broke every sandbox run."""
    return Path(path).resolve()


def _apply_limits(limits: SandboxLimits, applied_path: str) -> None:  # runs in the child
    import resource

    applied = {}

    def setl(name: str, value: int) -> None:
        res = getattr(resource, name, None)
        if res is None:
            applied[name] = False
            return
        try:
            resource.setrlimit(res, (value, value))
            applied[name] = True
        except (ValueError, OSError):
            applied[name] = False

    setl("RLIMIT_CPU", limits.cpu_s)
    setl("RLIMIT_FSIZE", limits.file_size_mb * 1024 * 1024)
    setl("RLIMIT_NOFILE", limits.open_files)
    setl("RLIMIT_CORE", 0)
    setl("RLIMIT_AS", limits.memory_mb * 1024 * 1024)
    # Report which limits took effect (the harness cannot observe this otherwise).
    with open(applied_path, "w") as fh:
        fh.write(",".join(f"{k}={int(v)}" for k, v in applied.items()))


def _tail(path: Path) -> tuple[str, int]:
    size = path.stat().st_size if path.exists() else 0
    if not size:
        return "", 0
    with open(path, "rb") as fh:
        fh.seek(max(0, size - TAIL_BYTES))
        return fh.read().decode("utf-8", errors="replace"), size


class SubprocessSandbox:
    kind = "subprocess"

    def __init__(self, python: str | None = None) -> None:
        # The harness venv's interpreter: it has the pinned gymnasium/numpy.
        self.python = python or sys.executable

    def run_python(self, script: str, args: list[str], workspace: Path, logs_dir: Path,
                   limits: SandboxLimits, stdin_data: bytes | None = None) -> RunResult:
        argv = [self.python, "-E", "-s", "-B", script, *args]
        return self.run(argv, workspace, logs_dir, limits, stdin_data)

    def spawn_python(self, script: str, args: list[str], workspace: Path, logs_dir: Path,
                     limits: SandboxLimits) -> subprocess.Popen:
        """Start a long-lived interactive child (stdin/stdout pipes) with the same isolation
        as `run`. The caller owns its lifetime and must call `terminate_group(proc)`."""
        workspace, logs_dir = _absolute(workspace), _absolute(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        err = open(logs_dir / "stderr.log", "wb")  # noqa: SIM115 - closed in terminate_group
        proc = subprocess.Popen(
            [self.python, "-E", "-s", "-B", script, *args],
            cwd=workspace,
            env=scrubbed_env(workspace),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=err,
            start_new_session=True,
            preexec_fn=lambda: _apply_limits(limits, str(logs_dir / ".limits")),
            close_fds=True,
        )
        proc._rlenv_stderr = err  # type: ignore[attr-defined]
        return proc

    def run(self, argv: list[str], workspace: Path, logs_dir: Path, limits: SandboxLimits,
            stdin_data: bytes | None = None) -> RunResult:
        workspace, logs_dir = _absolute(workspace), _absolute(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        out_path, err_path = logs_dir / "stdout.log", logs_dir / "stderr.log"
        applied_path = logs_dir / ".limits"
        start = time.monotonic()
        with open(out_path, "wb") as out, open(err_path, "wb") as err:
            proc = subprocess.Popen(
                argv,
                cwd=workspace,
                env=scrubbed_env(workspace),
                stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,  # own process group: kill everything on timeout
                preexec_fn=lambda: _apply_limits(limits, str(applied_path)),
                close_fds=True,
            )
            timed_out = False
            killed_reason = None
            try:
                if stdin_data is not None:
                    proc.stdin.write(stdin_data)
                    proc.stdin.close()
                deadline = start + limits.timeout_s
                next_mem_check = 0.0
                while proc.poll() is None:
                    now = time.monotonic()
                    if now >= deadline:
                        timed_out, killed_reason = True, "timeout"
                        break
                    if now >= next_mem_check:
                        # RLIMIT_AS is not enforced on macOS: watch the group's RSS instead.
                        rss_mb = group_rss_mb(proc.pid)
                        if rss_mb > limits.memory_mb:
                            killed_reason = f"memory ({rss_mb:.0f} MB > {limits.memory_mb} MB)"
                            break
                        next_mem_check = now + MEMORY_POLL_S
                    time.sleep(0.02)
            finally:
                _kill_group(proc.pid)
                proc.wait()
        duration = time.monotonic() - start
        rc = proc.returncode
        sig = signal.Signals(-rc).name if rc is not None and rc < 0 else None
        out_tail, out_bytes = _tail(out_path)
        err_tail, err_bytes = _tail(err_path)
        applied = {}
        if applied_path.exists():
            for item in filter(None, applied_path.read_text().split(",")):
                k, v = item.split("=")
                applied[k] = v == "1"
        return RunResult(argv=argv, exit_code=None if killed_reason else rc, signal=sig, timed_out=timed_out,
                         killed_reason=killed_reason,
                         duration_s=round(duration, 3), stdout_tail=out_tail, stderr_tail=err_tail,
                         stdout_bytes=out_bytes, stderr_bytes=err_bytes, limits_applied=applied)


def terminate_group(proc: subprocess.Popen) -> None:
    """Kill an interactive child's whole process group and release its pipes."""
    _kill_group(proc.pid)
    proc.wait()
    for stream in (proc.stdin, proc.stdout, getattr(proc, "_rlenv_stderr", None)):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass


def _kill_group(pid: int) -> None:
    """Kill the child's whole process group (no-op if it already exited)."""
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
