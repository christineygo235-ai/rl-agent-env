"""Reference solutions: prove each task is solvable through the real pipeline.

`check_reference` runs a reference solution exactly the way verification will:
fresh workspace -> static checks -> sandboxed `train.py --seed <task seed>` within
the task's train timeout -> harness-owned evaluation on the held-out seeds.
The scripted agent (step 5) replays these files through the tools.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from rlenv.evaluator import evaluate_workspace_policy, random_baseline
from rlenv.policy_check import check_workspace
from rlenv.sandbox import SandboxLimits, SubprocessSandbox
from rlenv.seeds import eval_seeds
from rlenv.task import Task
from rlenv.tools import train_argv
from rlenv.workspace import Workspace

REFERENCE_ROOT = Path(__file__).resolve().parents[1] / "reference"
REQUIRED = ("train.py", "policy.py")


@dataclass
class ReferenceCheck:
    task_id: str
    static_ok: bool
    train_ok: bool
    train_seconds: float
    eval_ok: bool
    eval_mean: float | None
    eval_std: float | None
    random_mean: float | None
    threshold: float
    passed: bool
    detail: str = ""

    def to_json(self) -> dict:
        return asdict(self)


def reference_files(task_id: str, root: Path = REFERENCE_ROOT) -> dict[str, str]:
    ref = root / task_id
    if not ref.is_dir():
        raise FileNotFoundError(f"no reference solution for {task_id!r} in {root}")
    return {p.name: p.read_text() for p in sorted(ref.glob("*.py"))}


def check_reference(task: Task, episodes: int | None = None, workdir: Path | None = None,
                    eval_episodes: int | None = None) -> ReferenceCheck:
    """Run the reference end-to-end. `episodes`/`eval_episodes` shrink it for fast tests."""
    base = Path(workdir or tempfile.mkdtemp(prefix=f"ref-{task.id}-"))
    ws = Workspace(base / "ws")
    for name, text in reference_files(task.id).items():
        ws.write_text(name, text)

    def result(**kw) -> ReferenceCheck:
        fields = dict(task_id=task.id, static_ok=False, train_ok=False, train_seconds=0.0, eval_ok=False,
                      eval_mean=None, eval_std=None, random_mean=None, threshold=task.success.threshold,
                      passed=False)
        fields.update(kw)
        return ReferenceCheck(**fields)

    report = check_workspace(ws.root, allow_torch=task.methods.allow_torch, required=REQUIRED)
    if not report.ok:
        return result(detail=report.summary())

    args = train_argv(task.seed, "policy_out", dict(task.env_kwargs),
                      ["--episodes", str(episodes)] if episodes is not None else None)
    limits = SandboxLimits(timeout_s=task.budget.train_timeout_s, memory_mb=task.limits.memory_mb)
    start = time.monotonic()
    run = SubprocessSandbox().run_python("train.py", args, ws.root, base / "train_logs", limits)
    train_s = round(time.monotonic() - start, 2)
    if not run.ok:
        return result(static_ok=True, train_seconds=train_s,
                      detail=f"train failed ({run.killed_reason or run.exit_code}): {run.stderr_tail[-500:]}")

    n = eval_episodes or task.success.eval_episodes
    seeds = eval_seeds(task.seed, task.success.eval_seed_salt, n, task.success.dev_eval_episodes)
    ev = evaluate_workspace_policy(task.env_id, dict(task.env_kwargs), seeds, ws.root, "policy_out",
                                   base / "eval_logs", SandboxLimits(timeout_s=600, memory_mb=task.limits.memory_mb),
                                   max_episode_steps=task.episode_limit())
    if not ev.ok:
        return result(static_ok=True, train_ok=True, train_seconds=train_s, detail=f"evaluation failed: {ev.error}")
    rnd = random_baseline(task.env_id, dict(task.env_kwargs), seeds, task.episode_limit())
    return result(static_ok=True, train_ok=True, train_seconds=train_s, eval_ok=True, eval_mean=ev.mean,
                  eval_std=ev.std, random_mean=rnd.mean, passed=ev.mean >= task.success.threshold,
                  detail=run.stdout_tail[-500:])
