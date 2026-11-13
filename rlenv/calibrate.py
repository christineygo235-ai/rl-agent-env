"""Calibration (PLAN step 9, ADR-0010): is a task solvable, non-trivial and reproducible?

For N task seeds, the reference solution goes through the real verification path
(fresh re-run in a clean copy, held-out seeds, hidden env instances if enabled).
Admission criteria for the task suite:
  1. the reference passes on >= 90 % of seeds;
  2. the random policy fails (its mean is below the threshold) on every seed;
  3. verification is deterministic (same seed twice -> identical numbers).
The report also suggests a train timeout (3x the slowest reference re-run) and, for
tasks with env variants, shows the reference's return as a fraction of the optimal
return per hidden instance, which is what `threshold_fraction_of_optimal` is set from.
"""

from __future__ import annotations

import json
import math
import statistics
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import gymnasium

from rlenv.reference import REFERENCE_ROOT, reference_files
from rlenv.reward import compute_reward
from rlenv.task import Task
from rlenv.verifier import verify
from rlenv.workspace import Workspace

PASS_RATE_REQUIRED = 0.9
SEED_STRIDE = 1000


@dataclass
class SeedResult:
    seed: int
    success: bool | None
    reward: float | None
    eval_mean: float | None        # task units, or worst normalised z with env variants
    random_mean: float | None
    threshold: float
    rerun_seconds: float | None
    variants: list[dict] = field(default_factory=list)
    detail: str = ""


@dataclass
class CalibrationReport:
    task_id: str
    task_version: int
    seeds: int
    has_variants: bool
    threshold: float
    gym_reward_threshold: float | None
    results: list[SeedResult]
    pass_rate: float
    random_fails_everywhere: bool
    deterministic: bool
    max_rerun_seconds: float
    suggested_train_timeout_s: int
    eval_means: dict[str, float]
    optimal_fraction: dict[str, float] | None
    admitted: bool
    reasons: list[str]

    def to_json(self) -> dict:
        return asdict(self)


def _reference_workspace(task: Task, base: Path, reference_root: Path) -> Workspace:
    ws = Workspace(base)
    for name, text in reference_files(task.id, reference_root).items():
        ws.write_text(name, text)
    ws.write_text("REPORT.md", "## Method\ncalibration\n## Hyperparameters\n-\n## Experiments\n-\n"
                               "## Results\n-\n## Limitations\n-\n")
    return ws


def _one(task: Task, work: Path, reference_root: Path) -> SeedResult:
    ws = _reference_workspace(task, work / "ws", reference_root)
    rec = verify(task, ws.root, [], work / "verify")
    r = compute_reward(rec)
    return SeedResult(task.seed, r.success, r.reward, rec.eval_mean, rec.random_mean, rec.threshold,
                      rec.rerun_seconds if not rec.variants else
                      max((v.get("rerun_seconds") or 0) for v in rec.variants) or rec.rerun_seconds,
                      rec.variants, r.reason)


def calibrate(task: Task, n_seeds: int = 10, workdir: Path | None = None,
              reference_root: Path = REFERENCE_ROOT) -> CalibrationReport:
    base = Path(workdir or tempfile.mkdtemp(prefix=f"calibrate-{task.id}-"))
    results = []
    for k in range(n_seeds):
        t = task.model_copy(update={"seed": task.seed + SEED_STRIDE * k})
        results.append(_one(t, base / f"seed{k}", reference_root))
    repeat = _one(task, base / "repeat0", reference_root)

    def same(a: SeedResult, b: SeedResult) -> bool:
        strip = lambda vs: [{k: v for k, v in x.items() if k != "rerun_seconds"} for x in vs]  # noqa: E731
        return a.eval_mean == b.eval_mean and a.random_mean == b.random_mean and strip(a.variants) == strip(b.variants)

    passes = [r for r in results if r.success]
    pass_rate = len(passes) / len(results)
    random_fails = all(r.random_mean is not None and r.random_mean < r.threshold for r in results)
    deterministic = same(results[0], repeat)
    rerun_s = [r.rerun_seconds for r in results if r.rerun_seconds] or [0.0]
    means = [r.eval_mean for r in results if r.eval_mean is not None and math.isfinite(r.eval_mean)]
    fractions = [v["agent_mean"] / v["optimal_mean"] for r in results for v in r.variants
                 if v.get("optimal_mean")]
    reasons = []
    if pass_rate < PASS_RATE_REQUIRED:
        reasons.append(f"reference passes on {pass_rate:.0%} of seeds (< {PASS_RATE_REQUIRED:.0%})")
    if not random_fails:
        reasons.append("the random policy meets the threshold on some seed")
    if not deterministic:
        reasons.append("verification is not deterministic (same seed, different numbers)")
    spec = gymnasium.spec(task.env_id)
    return CalibrationReport(
        task_id=task.id, task_version=task.version, seeds=n_seeds,
        has_variants=task.verification.env_variants is not None, threshold=task.success.threshold,
        gym_reward_threshold=spec.reward_threshold, results=results, pass_rate=pass_rate,
        random_fails_everywhere=random_fails, deterministic=deterministic, max_rerun_seconds=max(rerun_s),
        suggested_train_timeout_s=int(max(30, math.ceil(3 * max(rerun_s) / 10) * 10)),
        eval_means={"min": min(means), "median": statistics.median(means), "max": max(means)} if means else {},
        optimal_fraction=({"min": min(fractions), "median": statistics.median(fractions), "max": max(fractions)}
                          if fractions else None),
        admitted=not reasons, reasons=reasons)


def to_markdown(rep: CalibrationReport) -> str:
    unit = "worst hidden instance, normalised (random 0, threshold 1)" if rep.has_variants else "mean return"
    lines = [
        f"# Calibration: {rep.task_id} v{rep.task_version}",
        "",
        f"**{'ADMITTED' if rep.admitted else 'NOT ADMITTED'}**"
        + ("" if rep.admitted else f": {'; '.join(rep.reasons)}"),
        "",
        "| Criterion | Value |",
        "|---|---|",
        f"| Seeds | {rep.seeds} |",
        f"| Reference pass rate | {rep.pass_rate:.0%} (required ≥ {PASS_RATE_REQUIRED:.0%}) |",
        f"| Random policy fails on every seed | {rep.random_fails_everywhere} |",
        f"| Deterministic (same seed twice) | {rep.deterministic} |",
        f"| Score unit | {unit} |",
        f"| Threshold (task) | {rep.threshold:g}" + (" (1.0 = fraction × optimal per instance)" if rep.has_variants else "")
        + " |",
        f"| Gymnasium `reward_threshold` (canonical env) | {rep.gym_reward_threshold} |",
        f"| Reference score min / median / max | " + (" / ".join(f"{rep.eval_means[k]:.3f}" for k in
                                                           ("min", "median", "max")) if rep.eval_means else "n/a") + " |",
    ]
    if rep.optimal_fraction:
        f = rep.optimal_fraction
        lines.append(f"| Reference / optimal per hidden instance min / median / max | "
                     f"{f['min']:.3f} / {f['median']:.3f} / {f['max']:.3f} |")
    lines += [
        f"| Slowest reference re-run | {rep.max_rerun_seconds:.1f} s |",
        f"| Suggested train timeout (3×, rounded) | {rep.suggested_train_timeout_s} s |",
        "",
        "| Seed | Pass | Reward | Score | Random | Re-run s |",
        "|---|---|---|---|---|---|",
    ]
    for r in rep.results:
        score = "n/a" if r.eval_mean is None else f"{r.eval_mean:.3f}"
        rnd = "n/a" if r.random_mean is None else f"{r.random_mean:.3f}"
        rew = "None" if r.reward is None else f"{r.reward:.3f}"
        lines.append(f"| {r.seed} | {r.success} | {rew} | {score} | {rnd} | {r.rerun_seconds or 0:.1f} |")
    return "\n".join(lines) + "\n"


def write_report(rep: CalibrationReport, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    js, md = out_dir / f"{rep.task_id}.json", out_dir / f"{rep.task_id}.md"
    js.write_text(json.dumps(rep.to_json(), indent=2, default=str) + "\n")
    md.write_text(to_markdown(rep))
    return js, md
