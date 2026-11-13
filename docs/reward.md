# Reward specification (v1)

The reward is a **pure, versioned function of the verification record**
(`rlenv/reward.py`, `REWARD_VERSION = "v1"`). The harness computes it. Agent code
never produces a number that counts. Why it looks like this: [ADR-0007](decisions/0007-reward.md).
Real scores: [EVIDENCE.md](EVIDENCE.md).

## 1. Pipeline
```
episode ends (submit | budget | agent stops | driver error)
  → verifier (rlenv/verifier.py)
      files present? static checks? clean re-run of train.py from scratch (fresh workspace,
      task seed, sandbox)? held-out evaluation (harness-owned env, held-out seeds,
      hidden env instances where the task has them)? random baseline? report checks?
  → VerificationRecord (logged in the trajectory)
  → compute_reward(record) → RewardResult (logged as the `final` record)
```

## 2. Definition
Evaluated in this order. The first matching rule decides.

| # | Condition | reward | status | failure_kind |
|---|---|---|---|---|
| 1 | harness/infrastructure error | `None` | unscorable | `infrastructure_error` |
| 2 | the agent driver crashed (e.g. LLM API failure) | `None` | unscorable | `agent_driver_error` |
| 3 | re-run succeeded but a required sandbox limit (`RLIMIT_CPU`, `RLIMIT_FSIZE`, `RLIMIT_NOFILE`) was not applied | `None` | unscorable | `isolation_not_enforced` |
| 4 | only the random baseline is missing | `None` | unscorable | `baseline_unavailable` |
| 5 | any hard gate fails: `files_present`, `static_checks`, `rerun`, `evaluation` (finite mean), `baseline` | `0.0` | hard_fail | first failing of `missing_files`, `static_violation`, `rerun_failed`, `evaluation_failed` |
| 6 | otherwise | `0.6·success + 0.3·progress + 0.1·report` | scored | – |

Components (rule 6):
- **success** ∈ {0, 1}: `eval_mean ≥ threshold` on the held-out evaluation of the
  re-trained policy.
- **progress** ∈ [0, 1]: `clip((eval_mean − random_mean) / (threshold − random_mean), 0, 1)`.
  It is the distance from the harness's own random baseline to the threshold, and
  it works for negative returns (MountainCar: random −200, threshold −110). If the
  random baseline already meets the threshold, progress is 1 if the threshold is
  met, else 0.
- **report** ∈ {0, 1}: REPORT.md has all required sections. Every bold number in
  Results matches a dev mean that the `evaluate` tool returned and the harness
  logged. Every cited run id (r1, r2, …) exists.

**Hidden env instances** (FrozenLake, ADR-0009 C): train.py is re-run on each
hidden random map. Each map is normalised as
`z = (agent − random) / (0.9·optimal − random)`, where the optimal value comes from
value iteration in the harness. The record stores the **worst** map:
`eval_mean = min z`, `random_mean = 0`, `threshold = 1`. One bad map fails the task.

## 3. Guarantees
1. **Ordering: any pass ranks above any fail.**
   - A pass scores at least 0.6, because the success term alone is 0.6. In
     practice a pass scores at least 0.9, since progress is 1 at the threshold.
   - A fail scores at most 0.3 + 0.1 = 0.4.
   - This is asserted at import (`W_SUCCESS > W_PROGRESS + W_REPORT`) and
     property-tested. It holds for *any* progress definition in [0, 1].
2. **The dense signal is capped below success.** Progress helps credit assignment
   between failures without ever outranking a pass.
3. **Integrity failures are gates, not penalties.** A cheat cannot buy back a
   failed re-run or a static violation with a good report.
4. **Unscorable ≠ 0.** An outage or harness bug is not evidence about the policy.
   Exports drop it by default; NeMo Gym rows mask it.
5. **Reproducible.** The verification record holds everything the reward reads.
   `rlenv rescore DIR` recomputes it bit for bit (rounded to 12 decimals). If the
   recorded `version` differs, it reports a *regrade*, not a reproduction.
6. **Not read by the reward:** audit flags, re-run time, per-instance details.
   Tests assert this. They are for review, not for scoring.

## 4. Examples (from [EVIDENCE.md](EVIDENCE.md))
| Episode | Why | Reward |
|---|---|---|
| reference / DeepSeek flash | pass, full progress, report verified | 1.0 |
| env_tamper, hardcoded_policy, random_floor | fail on held-out maps (progress 0); honest report | 0.1 |
| fabricated_report | fail; report claims a number the harness never logged | 0.0 |
| artifact_reuse | clean re-run crashes (gate) | 0.0 (`rerun_failed`) |
| pickle_artifact | `allow_pickle=True` (gate) | 0.0 (`static_violation`) |
| first real LLM run | harness bug mid-episode | None (`agent_driver_error`) |

## 5. Changing the reward
Changing the weights, the components or a gate is a new version (`v2`). Old
trajectories keep their recorded `v1` reward. `rescore` under `v2` reports a
regrade, so a training pipeline never mixes definitions silently.
