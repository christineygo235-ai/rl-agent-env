# ADR-0002: Workspace confinement, subprocess sandbox, static checks

Status: accepted (step 2). Implements PLAN D1 option A.

## Context
Agent code is untrusted: it may be buggy (infinite loops, memory blow-ups,
output spam) or adversarial (reading secrets, reaching the network, gaming
the reward). Runs must be local, CPU-only and dependency-free. Docker is not
available on the development machine.

## Decision
Three independent layers. None of them is trusted alone.

### 1. Workspace (`rlenv/workspace.py`): the only door for tool file operations
- Relative paths only; no `..`, NUL or `~`; the **fully resolved path (symlinks
  followed) must stay inside the workspace**. Agent code runs inside the
  workspace and can create symlinks to `/etc/passwd`; `read_file` must not
  follow them out.
- Writes use `O_NOFOLLOW`, text suffixes only, a 256 KB cap and a file-count cap.
- `list` never follows symlinks. `snapshot` hashes regular text files (the code
  version). `copy_clean` copies only regular source files into a fresh
  directory for verification, dropping symlinks and stale artifacts.

### 2. Sandbox runner (`rlenv/sandbox.py`)
| Control | How | Enforced on |
|---|---|---|
| Wall-clock timeout | harness loop, then `killpg(SIGKILL)` on the child's own session | macOS + Linux |
| Grandchildren | `start_new_session=True`, so the whole group is killed (tested with `fork`) | macOS + Linux |
| CPU time | `RLIMIT_CPU` | macOS + Linux |
| File size / open files / core | `RLIMIT_FSIZE`, `RLIMIT_NOFILE`, `RLIMIT_CORE=0` | macOS + Linux |
| Memory | `RLIMIT_AS` **plus** an RSS watchdog (`ps -A -o pgid=,rss=` every 0.25 s) | `RLIMIT_AS`: Linux only. The watchdog covers macOS (tested: 600 MB allocation killed at 256 MB) |
| Secrets | environment **not inherited**: minimal PATH, HOME/TMPDIR in the workspace (tested: `ANTHROPIC_API_KEY` and AWS keys absent) | macOS + Linux |
| Determinism / CPU-only | `CUDA_VISIBLE_DEVICES=""`, single-threaded BLAS, `PYTHONHASHSEED=0`, `python -E -s -B` | macOS + Linux |
| Output | stdout/stderr to files outside the workspace (capped by `RLIMIT_FSIZE`); the tool result gets a 4 KB tail | macOS + Linux |

Each run records `limits_applied` and `killed_reason` (`timeout` / `memory (...)`).

### 3. Static checks (`rlenv/policy_check.py`): a policy filter, not a security boundary
- AST-based import allowlist (numpy, gymnasium, stdlib basics). Local workspace
  modules are allowed. torch only if the task enables it.
- Explicitly forbidden, each with a clear message: process, thread, network,
  FFI, pickle-like, dynamic import, introspection, `shutil`, RL libraries,
  other ML frameworks.
- Forbidden calls: `eval`/`exec`/`compile`/`__import__`, `os.system`/`popen`/
  `exec*`/`spawn*`/`fork`/`kill`/`symlink`, and `from os import *`.
- Escape hatches: `getattr` with a computed name, dunder access
  (`__subclasses__`, `__globals__`, `__builtins__`, …), `allow_pickle=True`,
  `open()` of absolute paths.
- Rationale: they enforce the *task rules* (classic methods, numpy-only,
  data-only artifacts), give the agent a fast and clear error before burning
  budget, and stop casual misuse.

## Known limits (stated plainly)
- **No network isolation and no filesystem isolation** in the subprocess
  backend. Python is dynamic enough that static checks can be bypassed (e.g.
  via a numpy C-extension corner or string tricks that the AST can't see).
  Malicious code could read files the user can read.
- Mitigation path: the `Sandbox` interface allows a container backend (Docker
  `--network none --read-only --memory`, or `bwrap`/`nsjail` on Linux). It is
  required before running untrusted third-party agents. The local dev default is
  acceptable because the agent is our own and the harness never trusts its numbers
  (evaluation stays harness-owned, step 3).
- The RSS watchdog polls, so a very fast allocation spike can briefly exceed
  the limit before it is killed.

## Alternatives considered
| Option | Why not (now) |
|---|---|
| Docker per run | not installed here; slower; planned as a backend |
| macOS `sandbox-exec` | deprecated, platform-specific profile language |
| RestrictedPython / in-process exec | in-process means agent code shares memory with the harness, so the reward can be tampered with |
| psutil for memory | extra dependency; `ps` is available on macOS and Linux |
