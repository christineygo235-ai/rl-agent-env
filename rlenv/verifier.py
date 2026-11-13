"""Verifier: establish the facts the reward depends on, independently of the agent.

1. required files present (train.py, policy.py, REPORT.md);
2. copy_clean into a fresh directory: only regular source files survive (no
   symlinks, no planted runs/ artifacts, no caches);
3. static checks on the clean copy;
4. fresh re-run of `train.py --seed <task seed> --out DIR --env-kwargs JSON`
   (no other arguments) in the sandbox, task timeout;
5. harness-owned evaluation on the held-out seeds (never shown to the agent);
6. random baseline on the same seeds;
7. report checks against the runs the harness logged;
8. audit flags (never part of the reward).

With env variants (ADR-0009 option C), steps 4–6 run once per HIDDEN instance:
retrain on it, evaluate on it, and compute the optimal policy's return on the same
seeds. Each instance is normalised, z = (agent - random) / (fraction*optimal - random),
and the record stores the WORST instance in normalised units (random 0,
threshold 1). So success requires every hidden instance to pass, and the reward
function stays unchanged. Per-instance details are kept in `variants`; maps are
logged only as sha256 fingerprints.

Agent-caused failures become gate failures (reward 0). Exceptions inside the
harness become `infrastructure_error` (reward None).
"""

from __future__ import annotations

import traceback
from pathlib import Path

from rlenv.audit import audit_workspace
from rlenv.evaluator import evaluate, evaluate_workspace_policy, random_baseline
from rlenv.policy_check import check_workspace
from rlenv.report import check_report
from rlenv.reward import VerificationRecord
from rlenv.sandbox import SandboxLimits, SubprocessSandbox
from rlenv.seeds import eval_seeds
from rlenv.task import Task
from rlenv.tools import RunRecord, train_argv
from rlenv.variants import kwargs_sha256, optimal_policy, variant_kwargs
from rlenv.workspace import Workspace

REQUIRED_FILES = ("train.py", "policy.py", "REPORT.md")
VERIFY_POLICY_DIR = "verify_policy"
EVAL_TIMEOUT_S = 900.0


def hidden_instances(task: Task) -> list[dict]:
    v = task.verification.env_variants
    if v is None:
        return []
    return variant_kwargs(v.kind, task.env_id, dict(task.env_kwargs), "verify", task.success.eval_seed_salt,
                          task.seed, v.count, {"frozen_p": v.frozen_p})


def verify(task: Task, workspace: Path, runs: list[RunRecord], work_dir: Path,
           agent_driver_error: str | None = None, sandbox: SubprocessSandbox | None = None) -> VerificationRecord:
    rec = VerificationRecord(task_id=task.id, task_version=task.version, threshold=task.success.threshold,
                             agent_driver_error=agent_driver_error)
    try:
        _verify(task, Workspace(workspace, create=False), runs, Path(work_dir), rec, sandbox or SubprocessSandbox())
    except Exception as exc:  # harness-side failure -> unscorable, never 0
        rec.infrastructure_error = f"{type(exc).__name__}: {exc}"
        rec.report_detail.setdefault("traceback", traceback.format_exc()[-2000:])
    return rec


def _verify(task: Task, ws: Workspace, runs: list[RunRecord], work_dir: Path, rec: VerificationRecord,
            sandbox: SubprocessSandbox) -> None:
    present = {e.path for e in ws.list() if e.kind == "file"}
    rec.files_missing = [f for f in REQUIRED_FILES if f not in present]

    dev_means = [e["mean_return"] for r in runs for e in r.dev_evals if e.get("mean_return") is not None]
    report_text = (ws.root / "REPORT.md").read_text(errors="replace") if "REPORT.md" in present else None
    rec.report_ok, rec.report_detail = check_report(report_text, dev_means, [r.run_id for r in runs])
    rec.audit_flags = audit_workspace(ws.root)
    if rec.files_missing:
        return

    clean = ws.copy_clean(work_dir / "clean")
    static = check_workspace(clean.root, allow_torch=task.methods.allow_torch, required=REQUIRED_FILES[:2])
    rec.static_ok = static.ok
    rec.static_violations = [f"{v.file}:{v.line}: [{v.rule}] {v.message}" for v in static.violations][:50]
    if not static.ok:
        return

    seeds = eval_seeds(task.seed, task.success.eval_seed_salt, task.success.eval_episodes,
                       task.success.dev_eval_episodes)
    instances = hidden_instances(task)
    if not instances:
        _verify_instance(task, clean, work_dir, rec, sandbox, dict(task.env_kwargs), seeds, VERIFY_POLICY_DIR)
        return

    fraction = task.verification.env_variants.threshold_fraction_of_optimal
    z_scores = []
    applied: list[dict[str, bool]] = []
    for i, kw in enumerate(instances):
        sub = VerificationRecord(task_id=task.id, task_version=task.version, threshold=0.0)
        _verify_instance(task, clean, work_dir / f"variant{i}", sub, sandbox, kw, seeds, f"{VERIFY_POLICY_DIR}_{i}")
        applied.append(sub.limits_applied)
        # Fail closed: a limit counts as applied only if it was applied to EVERY instance's re-run.
        rec.limits_applied = {k: all(a.get(k, False) for a in applied) for k in applied[0]}
        entry = {"index": i, "env_kwargs_sha256": kwargs_sha256(kw), "rerun_ok": sub.rerun_ok,
                 "rerun_seconds": sub.rerun_seconds,
                 "eval_ok": sub.eval_ok, "agent_mean": sub.eval_mean, "random_mean": sub.random_mean,
                 "eval_error": sub.eval_error}
        if not sub.rerun_ok or not sub.eval_ok:
            rec.variants.append(entry)
            rec.rerun_ok, rec.rerun_detail = sub.rerun_ok, f"variant {i}: {sub.rerun_detail}"
            rec.eval_ok, rec.eval_error = sub.eval_ok, sub.eval_error and f"variant {i}: {sub.eval_error}"
            rec.eval_seeds_sha256 = sub.eval_seeds_sha256
            return
        opt = evaluate(task.env_id, kw, seeds, optimal_policy(task.verification.env_variants.kind, task.env_id, kw),
                       max_episode_steps=task.episode_limit())
        threshold_i = fraction * opt.mean
        span = threshold_i - sub.random_mean
        z = (sub.eval_mean - sub.random_mean) / span if span > 0 else (1.0 if sub.eval_mean >= threshold_i else 0.0)
        entry.update({"optimal_mean": opt.mean, "threshold": threshold_i, "z": z})
        rec.variants.append(entry)
        z_scores.append(z)

    # Worst hidden instance, in normalised units: random = 0, threshold = 1.
    rec.rerun_ok = rec.eval_ok = True
    rec.rerun_detail = f"{len(instances)} hidden instance(s) re-trained"
    rec.eval_mean, rec.eval_std = min(z_scores), None
    rec.random_mean, rec.threshold = 0.0, 1.0
    rec.eval_episodes = len(seeds) * len(instances)
    rec.eval_seeds_sha256 = sub.eval_seeds_sha256


def _verify_instance(task: Task, clean: Workspace, work_dir: Path, rec: VerificationRecord,
                     sandbox: SubprocessSandbox, env_kwargs: dict, seeds: list[int], out_dir: str) -> None:
    limits = SandboxLimits(timeout_s=task.budget.train_timeout_s, memory_mb=task.limits.memory_mb)
    run = sandbox.run_python("train.py", train_argv(task.seed, out_dir, env_kwargs), clean.root,
                             work_dir / "rerun_logs", limits)
    rec.limits_applied = dict(run.limits_applied)
    artifacts = clean.root / out_dir
    has_artifacts = artifacts.is_dir() and any(p.is_file() for p in artifacts.rglob("*"))
    rec.rerun_ok = run.ok and has_artifacts
    rec.rerun_detail = (f"exit {run.exit_code}, killed {run.killed_reason}, {run.duration_s:.1f}s, "
                        f"artifacts {'yes' if has_artifacts else 'no'}; stderr: {run.stderr_tail[-500:]}")
    rec.rerun_seconds = run.duration_s
    if not rec.rerun_ok:
        return
    ev = evaluate_workspace_policy(task.env_id, env_kwargs, seeds, clean.root, out_dir, work_dir / "eval_logs",
                                   SandboxLimits(timeout_s=EVAL_TIMEOUT_S, memory_mb=task.limits.memory_mb),
                                   max_duration_s=EVAL_TIMEOUT_S, max_episode_steps=task.episode_limit())
    rec.eval_seeds_sha256 = ev.seeds_sha256  # fingerprint only; the seeds stay secret
    rec.eval_ok, rec.eval_error, rec.eval_episodes = ev.ok, ev.error, ev.episodes
    if ev.ok:
        rec.eval_mean, rec.eval_std = ev.mean, ev.std
    rnd = random_baseline(task.env_id, env_kwargs, seeds, task.episode_limit())
    if rnd.ok:
        rec.random_mean = rnd.mean
