"""Harness-owned evaluation (PLAN D2, option C).

The harness creates the environment, resets it with seeds it chose, steps it,
and sums the rewards itself. A policy is just `act(obs) -> action`:
- `SandboxedPolicy`: the agent's `policy.py` in a sandboxed subprocess,
  queried over a pipe (it never sees the env, the seeds or the returns);
- `random_policy`: the baseline for reward normalisation.
Every action is validated against the action space; an invalid action, a
crash, or a slow answer ends the evaluation with an error (no partial credit
is invented).
"""

from __future__ import annotations

import json
import math
import os
import select
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from rlenv.sandbox import SandboxLimits, SubprocessSandbox, group_rss_mb, terminate_group
from rlenv.seeds import seeds_sha256

POLICY_SERVER = str(Path(__file__).with_name("policy_server.py"))
DEFAULT_STEP_TIMEOUT_S = 5.0
LOAD_TIMEOUT_S = 30.0
HARD_STEP_CAP = 100_000  # safety net beyond the env's own TimeLimit
MAX_MESSAGE_BYTES = 1_000_000


class PolicyError(Exception):
    """The policy failed: crashed, timed out, or returned an invalid action."""


@dataclass
class EvalResult:
    env_id: str
    episodes: int
    returns: list[float]
    lengths: list[int]
    mean: float
    std: float
    seeds_sha256: str
    duration_s: float
    error: str | None = None
    policy_stderr_tail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_json(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Observation / action encoding (mirrors policy_server.py)
# --------------------------------------------------------------------------- #


def encode_obs(obs: Any) -> Any:
    if isinstance(obs, np.ndarray):
        return {"t": "a", "v": obs.tolist(), "dtype": str(obs.dtype), "shape": list(obs.shape)}
    if isinstance(obs, np.generic):
        return obs.item()
    return obs


def decode_action(enc: Any, space: gym.Space) -> Any:
    if isinstance(enc, dict) and enc.get("t") == "a":
        action = np.asarray(enc["v"], dtype=space.dtype).reshape(enc["shape"])
    elif isinstance(space, gym.spaces.Discrete):
        if isinstance(enc, bool) or not isinstance(enc, (int, float)) or float(enc) != int(enc):
            raise PolicyError(f"invalid action {enc!r} for {space}")
        action = np.int64(int(enc))
    else:
        action = np.asarray(enc, dtype=space.dtype)
    if isinstance(action, np.ndarray) and action.dtype.kind == "f" and not np.all(np.isfinite(action)):
        raise PolicyError(f"non-finite action {enc!r}")
    if not space.contains(action):
        raise PolicyError(f"action {enc!r} is outside the action space {space}")
    return action


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #


class SandboxedPolicy:
    """The agent's policy.py behind a pipe, in its own sandboxed process."""

    def __init__(self, workspace: Path, policy_dir: str, logs_dir: Path, limits: SandboxLimits,
                 sandbox: SubprocessSandbox | None = None, step_timeout_s: float = DEFAULT_STEP_TIMEOUT_S) -> None:
        self.limits = limits
        self.step_timeout_s = step_timeout_s
        self.logs_dir = logs_dir
        self.proc = (sandbox or SubprocessSandbox()).spawn_python(
            POLICY_SERVER, ["--policy-dir", policy_dir], workspace, logs_dir, limits)
        self.calls = 0
        self._buf = b""
        msg = self._read(LOAD_TIMEOUT_S)
        if "error" in msg:
            raise PolicyError(msg["error"])
        if msg != {"ready": True}:
            raise PolicyError(f"unexpected handshake {msg!r}")

    def _readline(self, timeout: float) -> bytes:
        """Read one line from the raw pipe fd with a deadline (own buffer, so select()
        never misses data sitting in a Python-level buffer)."""
        deadline = time.monotonic() + timeout
        fd = self.proc.stdout.fileno()
        while b"\n" not in self._buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PolicyError(f"policy did not answer within {timeout:g}s")
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                raise PolicyError(f"policy process exited (code {self.proc.poll()})")
            self._buf += chunk
            if len(self._buf) > MAX_MESSAGE_BYTES:
                raise PolicyError("policy message too large")
        line, _, self._buf = self._buf.partition(b"\n")
        return line

    def _read(self, timeout: float) -> dict:
        line = self._readline(timeout)
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PolicyError(f"malformed policy message: {line[:200]!r}") from exc
        if not isinstance(msg, dict):
            raise PolicyError(f"malformed policy message: {line[:200]!r}")
        return msg

    def _request(self, req: dict) -> dict:
        try:
            self.proc.stdin.write((json.dumps(req) + "\n").encode())
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise PolicyError(f"policy process is gone: {exc}") from exc
        msg = self._read(self.step_timeout_s)
        if "error" in msg:
            raise PolicyError(msg["error"])
        self.calls += 1
        if self.calls % 500 == 0 and group_rss_mb(self.proc.pid) > self.limits.memory_mb:
            raise PolicyError(f"policy exceeded {self.limits.memory_mb} MB")
        return msg

    def reset(self) -> None:
        self._request({"op": "reset"})

    def act(self, obs: Any) -> Any:
        msg = self._request({"op": "act", "obs": encode_obs(obs)})
        if "action" not in msg:
            raise PolicyError(f"malformed act reply {msg!r}")
        return msg["action"]

    def stderr_tail(self, n: int = 2000) -> str:
        path = self.logs_dir / "stderr.log"
        return path.read_text(errors="replace")[-n:] if path.exists() else ""

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self.proc.stdin.write(b'{"op": "close"}\n')
                self.proc.stdin.flush()
        except OSError:
            pass
        terminate_group(self.proc)

    def __enter__(self) -> SandboxedPolicy:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def random_policy(space: gym.Space, seed: int) -> Callable[[Any], Any]:
    space.seed(seed)

    def act(_obs: Any) -> Any:
        a = space.sample()
        return a.item() if isinstance(a, np.generic) else encode_obs(a)

    return act


# --------------------------------------------------------------------------- #
# The evaluation loop (harness side)
# --------------------------------------------------------------------------- #


def make_env(env_id: str, env_kwargs: dict, max_episode_steps: int | None = None) -> gym.Env:
    """The harness's env: registered wrappers, plus an explicit TimeLimit when given."""
    if max_episode_steps is not None:
        return gym.make(env_id, max_episode_steps=max_episode_steps, **env_kwargs)
    return gym.make(env_id, **env_kwargs)


def evaluate(env_id: str, env_kwargs: dict, seeds: list[int], act: Callable[[Any], Any],
             reset: Callable[[], None] | None = None, max_duration_s: float = 600.0,
             max_episode_steps: int | None = None) -> EvalResult:
    """Run one episode per seed; the harness owns the env and sums the rewards."""
    start = time.monotonic()
    env = make_env(env_id, env_kwargs, max_episode_steps)
    returns: list[float] = []
    lengths: list[int] = []
    error = None
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            if reset is not None:
                reset()
            total, steps, done = 0.0, 0, False
            while not done:
                if time.monotonic() - start > max_duration_s:
                    raise PolicyError(f"evaluation exceeded {max_duration_s:g}s")
                action = decode_action(act(obs), env.action_space)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                steps += 1
                done = terminated or truncated or steps >= HARD_STEP_CAP
            returns.append(total)
            lengths.append(steps)
    except PolicyError as exc:
        error = str(exc)
    finally:
        env.close()
    complete = error is None and len(returns) == len(seeds)
    mean = float(np.mean(returns)) if complete else math.nan
    std = float(np.std(returns)) if complete else math.nan
    return EvalResult(env_id=env_id, episodes=len(returns), returns=returns, lengths=lengths, mean=mean, std=std,
                      seeds_sha256=seeds_sha256(seeds), duration_s=round(time.monotonic() - start, 3), error=error)


def evaluate_workspace_policy(env_id: str, env_kwargs: dict, seeds: list[int], workspace: Path,
                              policy_dir: str, logs_dir: Path, limits: SandboxLimits,
                              step_timeout_s: float = DEFAULT_STEP_TIMEOUT_S,
                              max_duration_s: float = 600.0, max_episode_steps: int | None = None) -> EvalResult:
    """Evaluate the agent's policy.py + artifact. Any failure is an EvalResult with `error`."""
    start = time.monotonic()
    try:
        policy = SandboxedPolicy(workspace, policy_dir, logs_dir, limits, step_timeout_s=step_timeout_s)
    except PolicyError as exc:
        return EvalResult(env_id, 0, [], [], math.nan, math.nan, seeds_sha256(seeds),
                          round(time.monotonic() - start, 3), error=str(exc),
                          policy_stderr_tail=_tail(logs_dir / "stderr.log"))
    with policy:
        result = evaluate(env_id, env_kwargs, seeds, policy.act, policy.reset, max_duration_s, max_episode_steps)
    result.policy_stderr_tail = policy.stderr_tail()
    return result


def random_baseline(env_id: str, env_kwargs: dict, seeds: list[int],
                    max_episode_steps: int | None = None) -> EvalResult:
    env = make_env(env_id, env_kwargs, max_episode_steps)
    space = env.action_space
    env.close()
    return evaluate(env_id, env_kwargs, seeds, random_policy(space, seeds[0] if seeds else 0),
                    max_episode_steps=max_episode_steps)


def _tail(path: Path, n: int = 2000) -> str:
    return path.read_text(errors="replace")[-n:] if path.exists() else ""
