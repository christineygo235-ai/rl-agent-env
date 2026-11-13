"""Reward function: a pure, versioned function of the verification record (ADR-0007).

    reward = None                         if the episode is unscorable (infrastructure failure)
           = 0.0                          if any hard gate fails (integrity)
           = 0.6*success + 0.3*progress + 0.1*report   otherwise

Ordering guarantee (asserted at import time and property-tested):
every passing submission scores at least the success term, 0.6, while every
failing one scores at most progress + report = 0.3 + 0.1 = 0.4. So **any pass
ranks above any fail**, no matter how good the report or how close the miss.
(Progress is 1 once the threshold is met, so a pass actually scores >= 0.9.)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

REWARD_VERSION = "v1"
W_SUCCESS, W_PROGRESS, W_REPORT = 0.6, 0.3, 0.1

# Ordering guarantee, in its strong form: the success term alone (0.6) must exceed the
# largest score a failing submission can reach (progress + report = 0.4). This holds
# for ANY definition of progress in [0, 1], so changing the dense term later can't
# break the ranking. (With the current progress definition a pass is in fact >= 0.9.)
assert W_SUCCESS > W_PROGRESS + W_REPORT, "ordering guarantee: success must outweigh progress + report"
assert abs(W_SUCCESS + W_PROGRESS + W_REPORT - 1.0) < 1e-12, "reward range must be [0, 1]"

# Low-cardinality failure vocabulary (NeMo Gym style): lets pipelines count and filter
# failures without parsing free-text reasons.
FAILURE_KINDS = {
    # unscorable (reward None): not evidence about the policy
    "infrastructure_error", "agent_driver_error", "isolation_not_enforced", "baseline_unavailable",
    # hard fail (reward 0): the agent's submission is invalid
    "missing_files", "static_violation", "rerun_failed", "evaluation_failed",
}
_GATE_KIND = {"files_present": "missing_files", "static_checks": "static_violation",
              "rerun": "rerun_failed", "evaluation": "evaluation_failed"}

# Sandbox limits that must have been applied to the verification re-run (fail closed).
# RLIMIT_AS is absent on purpose: macOS ignores it; the RSS watchdog covers memory there.
REQUIRED_LIMITS = ("RLIMIT_CPU", "RLIMIT_FSIZE", "RLIMIT_NOFILE")


@dataclass
class VerificationRecord:
    """Facts established by the verifier. Everything the reward depends on is in here,
    so `rescore` can recompute the reward from the trajectory alone."""

    task_id: str
    task_version: int
    threshold: float
    files_missing: list[str] = field(default_factory=list)
    static_ok: bool = False
    static_violations: list[str] = field(default_factory=list)
    rerun_ok: bool = False
    rerun_detail: str = ""
    limits_applied: dict[str, bool] = field(default_factory=dict)
    eval_ok: bool = False
    eval_mean: float | None = None
    eval_std: float | None = None
    eval_episodes: int = 0
    eval_seeds_sha256: str = ""
    eval_error: str | None = None
    random_mean: float | None = None
    report_ok: bool = False
    report_detail: dict[str, Any] = field(default_factory=dict)
    infrastructure_error: str | None = None  # harness-side failure: not the agent's fault
    agent_driver_error: str | None = None    # the driver (e.g. LLM API) crashed mid-episode
    # Informational fields: recorded for audit/analysis, NEVER read by compute_reward.
    rerun_seconds: float | None = None
    variants: list[dict[str, Any]] = field(default_factory=list)  # per hidden env instance (ADR-0009 C)
    audit_flags: list[str] = field(default_factory=list)          # ADR-0009 D: audit signal, not reward

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> VerificationRecord:
        return cls(**data)


@dataclass
class RewardResult:
    reward: float | None
    status: str  # scored | hard_fail | unscorable
    success: bool | None
    gates: dict[str, bool]
    components: dict[str, float]
    reason: str
    failure_kind: str | None = None  # one of FAILURE_KINDS, None when scored
    version: str = REWARD_VERSION

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def progress(mean: float, random_mean: float, threshold: float) -> float:
    """Dense signal: how far from the random baseline to the threshold, clipped to [0, 1].
    Works for negative returns (MountainCar: random -200, threshold -110)."""
    span = threshold - random_mean
    if span <= 0:  # degenerate task: random already meets the threshold
        return 1.0 if mean >= threshold else 0.0
    return min(1.0, max(0.0, (mean - random_mean) / span))


def compute_reward(rec: VerificationRecord) -> RewardResult:
    # 1) Unscorable: failures that say nothing about the policy. Never 0 (that would teach noise).
    if rec.infrastructure_error:
        return RewardResult(None, "unscorable", None, {}, {}, f"infrastructure: {rec.infrastructure_error}",
                            "infrastructure_error")
    if rec.agent_driver_error:
        return RewardResult(None, "unscorable", None, {}, {}, f"agent driver: {rec.agent_driver_error}",
                            "agent_driver_error")
    missing_limits = [k for k in REQUIRED_LIMITS if not rec.limits_applied.get(k, False)]
    if rec.rerun_ok and missing_limits:
        return RewardResult(None, "unscorable", None, {}, {},
                            f"isolation not enforced for the re-run: {missing_limits}", "isolation_not_enforced")

    # 2) Hard gates: integrity failures zero the reward; they are not weighed against anything.
    gates = {
        "files_present": not rec.files_missing,
        "static_checks": rec.static_ok,
        "rerun": rec.rerun_ok,
        "evaluation": rec.eval_ok and rec.eval_mean is not None and math.isfinite(rec.eval_mean),
        "baseline": rec.random_mean is not None and math.isfinite(rec.random_mean),
    }
    failed = [g for g, ok in gates.items() if not ok]
    if failed:
        if "baseline" in failed and len(failed) == 1:  # harness could not compute its own baseline
            return RewardResult(None, "unscorable", None, gates, {}, "infrastructure: random baseline unavailable",
                                "baseline_unavailable")
        first = next(g for g in gates if not gates[g] and g in _GATE_KIND)
        return RewardResult(0.0, "hard_fail", False, gates, {}, f"hard gate(s) failed: {failed}", _GATE_KIND[first])

    # 3) Weighted sum.
    success = rec.eval_mean >= rec.threshold
    prog = progress(rec.eval_mean, rec.random_mean, rec.threshold)
    report = 1.0 if rec.report_ok else 0.0
    components = {"success": 1.0 if success else 0.0, "progress": prog, "report": report}
    reward = W_SUCCESS * components["success"] + W_PROGRESS * prog + W_REPORT * report
    reward = round(reward, 12)  # stable across platforms for bit-for-bit rescoring
    return RewardResult(reward, "scored", success, gates, components,
                        f"{'pass' if success else 'fail'}: mean {rec.eval_mean:.4g} vs threshold {rec.threshold:g}")
