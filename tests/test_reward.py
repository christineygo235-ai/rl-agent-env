"""Reward function: ordering guarantee, hard gates, unscorable cases, progress."""

from __future__ import annotations

import json
import random

import pytest

from rlenv.reward import (
    FAILURE_KINDS,
    REWARD_VERSION,
    W_PROGRESS,
    W_REPORT,
    W_SUCCESS,
    VerificationRecord,
    compute_reward,
    progress,
)
from rlenv.report import check_report

LIMITS = {"RLIMIT_CPU": True, "RLIMIT_FSIZE": True, "RLIMIT_NOFILE": True, "RLIMIT_AS": False}


def rec(mean, threshold=-130.0, random_mean=-200.0, report=True, **kw) -> VerificationRecord:
    base = dict(task_id="t", task_version=1, threshold=threshold, static_ok=True, rerun_ok=True,
                limits_applied=dict(LIMITS), eval_ok=True, eval_mean=mean, eval_std=1.0, eval_episodes=100,
                random_mean=random_mean, report_ok=report)
    base.update(kw)
    return VerificationRecord(**base)


# --------------------------------------------------------------------------- #
# Ordering guarantee
# --------------------------------------------------------------------------- #


def test_weights_encode_the_ordering_guarantee():
    assert W_SUCCESS > W_PROGRESS + W_REPORT
    assert (W_SUCCESS, W_PROGRESS, W_REPORT) == (0.6, 0.3, 0.1)


def test_any_pass_ranks_above_any_fail_property():
    rng = random.Random(0)
    passes, fails = [], []
    for _ in range(5000):
        threshold = rng.uniform(-300, 300)
        random_mean = threshold - rng.uniform(0.01, 300)
        report = rng.random() < 0.5
        if rng.random() < 0.5:
            r = compute_reward(rec(threshold + rng.uniform(0, 100), threshold, random_mean, report))
            passes.append(r.reward)
        else:
            r = compute_reward(rec(threshold - rng.uniform(1e-9, 400), threshold, random_mean, report))
            fails.append(r.reward)
    assert min(passes) >= W_SUCCESS + W_PROGRESS > 0.4 > max(fails)
    assert min(passes) > max(fails)


def test_boundaries():
    assert compute_reward(rec(-130.0)).success is True  # exactly at threshold passes
    best_fail = compute_reward(rec(-130.0000001, report=True)).reward
    worst_pass = compute_reward(rec(-130.0, report=False)).reward
    assert best_fail < 0.4 and worst_pass == pytest.approx(0.9) and worst_pass > best_fail


def test_examples():
    assert compute_reward(rec(-99.7)).reward == 1.0
    assert compute_reward(rec(-99.7, report=False)).reward == pytest.approx(0.9)
    halfway = compute_reward(rec(-165.0))  # halfway from random (-200) to threshold (-130)
    assert halfway.components["progress"] == pytest.approx(0.5)
    assert halfway.reward == pytest.approx(0.3 * 0.5 + 0.1)
    assert compute_reward(rec(-250.0, report=False)).reward == 0.0  # worse than random: no progress


# --------------------------------------------------------------------------- #
# Hard gates: integrity failures zero the reward, nothing is weighed against them
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("override,gate,kind", [
    ({"files_missing": ["REPORT.md"]}, "files_present", "missing_files"),
    ({"static_ok": False}, "static_checks", "static_violation"),
    ({"rerun_ok": False}, "rerun", "rerun_failed"),
    ({"eval_ok": False, "eval_mean": None}, "evaluation", "evaluation_failed"),
    ({"eval_mean": float("nan")}, "evaluation", "evaluation_failed"),
])
def test_hard_gates_zero_the_reward(override, gate, kind):
    r = compute_reward(rec(-99.7, report=True, **override))  # would be a perfect 1.0 otherwise
    assert r.reward == 0.0 and r.status == "hard_fail" and r.gates[gate] is False
    assert r.failure_kind == kind and kind in FAILURE_KINDS
    assert r.components == {}  # not weighed


# --------------------------------------------------------------------------- #
# Unscorable: None, never 0
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("override,fragment", [
    ({"infrastructure_error": "RuntimeError: harness bug"}, "infrastructure"),
    ({"agent_driver_error": "APIConnectionError: timeout"}, "agent driver"),
    ({"limits_applied": {"RLIMIT_CPU": True}}, "isolation not enforced"),
    ({"random_mean": None}, "baseline unavailable"),
])
def test_unscorable_is_none_not_zero(override, fragment):
    r = compute_reward(rec(-99.7, **override))
    assert r.reward is None and r.status == "unscorable" and fragment in r.reason
    assert r.failure_kind in {"infrastructure_error", "agent_driver_error", "isolation_not_enforced",
                              "baseline_unavailable"}
    assert compute_reward(rec(-99.7)).failure_kind is None


def test_unscorable_takes_precedence_over_gates():
    r = compute_reward(rec(-99.7, static_ok=False, infrastructure_error="disk full"))
    assert r.reward is None


# --------------------------------------------------------------------------- #
# Progress and serialisation
# --------------------------------------------------------------------------- #


def test_progress_handles_negative_returns_and_degenerate_tasks():
    assert progress(-200, -200, -130) == 0.0 and progress(-130, -200, -130) == 1.0
    assert progress(0.35, 0.0, 0.70) == pytest.approx(0.5)
    assert progress(9.0, 10.0, 8.0) == 1.0 and progress(7.0, 10.0, 8.0) == 0.0  # random above threshold


def test_record_roundtrip_and_determinism():
    r = rec(-140.123456789)
    again = VerificationRecord.from_json(json.loads(json.dumps(r.to_json())))
    assert compute_reward(again).to_json() == compute_reward(r).to_json()
    assert compute_reward(r).version == REWARD_VERSION


# --------------------------------------------------------------------------- #
# Report checks
# --------------------------------------------------------------------------- #

GOOD_REPORT = """# Report
## Method
Q-learning.
## Hyperparameters
| a | b |
## Experiments
| r1 | 0.735 |
## Results
Dev mean return: **0.735** (threshold 0.7).
## Limitations
None.
"""


def test_report_ok():
    ok, detail = check_report(GOOD_REPORT, [0.7353], ["r1"])
    assert ok, detail


@pytest.mark.parametrize("text,logged,runs,key", [
    (GOOD_REPORT.replace("## Limitations\nNone.\n", ""), [0.7353], ["r1"], "missing_sections"),
    (GOOD_REPORT.replace("**0.735**", "0.735"), [0.7353], ["r1"], "error"),
    (GOOD_REPORT.replace("**0.735**", "**0.95**"), [0.7353], ["r1"], "unmatched_claims"),
    (GOOD_REPORT, [], ["r1"], "unmatched_claims"),                     # no evaluation was ever run
    (GOOD_REPORT.replace("| r1 |", "| r7 |"), [0.7353], ["r1"], "phantom_runs"),
])
def test_report_failures(text, logged, runs, key):
    ok, detail = check_report(text, logged, runs)
    assert not ok and detail.get(key), detail


def test_report_precision_matching():
    for claim, ok in (("0.74", True), ("0.7353", True), ("0.7", True), ("0.8", False), ("0.736", False)):
        text = GOOD_REPORT.replace("**0.735**", f"**{claim}**")
        assert check_report(text, [0.7353], ["r1"])[0] is ok, claim
    neg = GOOD_REPORT.replace("**0.735**", "**-99.7**")
    assert check_report(neg, [-99.71], ["r1"])[0]


def test_missing_report():
    assert check_report(None, [1.0], ["r1"]) == (False, {"error": "REPORT.md missing"})
