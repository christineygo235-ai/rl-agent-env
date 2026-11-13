# ADR-0010: Calibration, the train.py contract, and held-out env instances

Status: accepted (step 9).

## Context
Draft thresholds and budgets were guesses. A task is only useful if:
- the allowed method solves it reliably within its budget;
- a random policy does not;
- verification gives the same numbers twice.

Separately, the red team (ADR-0009) showed that a hand-written policy could
pass a fixed environment, and the user chose held-out env instances (C) plus
audit signals (D) as the fix.

## Decision

### 1. The train.py contract (all tasks)
```
python train.py --seed SEED --out DIR --env-kwargs JSON   [+ the agent's extra args, in-episode only]
```
- `train.py` must call `gymnasium.make(env_id, **json.loads(JSON))`.
- The harness always passes `--env-kwargs`; like `--seed` and `--out`, it is
  reserved and the agent cannot override it.
- Verification passes no other arguments.
- All task versions were bumped and every instruction states the contract.

### 2. Held-out env instances (`verification.env_variants`, opt-in per task)
- Accepted only for supported env families (`rlenv/variants.py` registry). Today
  that is `frozenlake_random_map`; task validation rejects other combinations.
- **Hidden instances** (`count`) are derived from the secret salt. Verification
  re-trains and evaluates on each one. **Public dev instances** (`dev_count`) are
  derived from the task seed, and the agent can train on them via
  `run_training(variant=k)`. The two sets are disjoint, and neither equals the
  canonical map.
- Per instance, threshold = `threshold_fraction_of_optimal × optimal return`. The
  optimal return comes from exact value iteration on the env's transition model,
  evaluated on the same held-out seeds (fair, same noise realisation).
- The **worst** instance decides: each is normalised to
  `z = (agent − random)/(threshold − random)`, and the record stores `min z` with
  random = 0 and threshold = 1. The reward function is unchanged (v1): success
  means every hidden instance reaches its threshold, and progress comes from the
  worst one.
- Only `sha256(env_kwargs)` per hidden instance is logged. The maps themselves
  never reach trajectories or exports, so training on trajectories can't leak them
  into the model.
- Isolation fails closed across instances: a limit counts as applied only if it was
  applied to every instance's re-run.

### 3. Calibration (`rlenv calibrate`, `make calibrate`)
For 10 task seeds (each with its own held-out seeds and hidden maps), the
reference goes through the real verification path. Admission criteria:
1. reference pass rate ≥ 90 %;
2. the random policy fails on every seed;
3. same seed twice gives identical numbers.

Reports are committed in `calibration/<task>.md|json`.

| Task (version) | Threshold | Reference score min / median / max (10 seeds) | Random | Slowest re-run | Train timeout | Admitted |
|---|---|---|---|---|---|---|
| frozenlake-tabular (v3) | 0.9 × optimal on each of 2 hidden maps | z 1.091 / 1.112 / 1.132; reference/optimal ≥ 0.977 | z = 0 | 13.1 s | 120 s | yes |
| mountaincar-tilecoding (v3) | **-110** (was -130) | -106.8 / -100.7 / -99.1 | -200 | 4.8 s | **120 s** (was 300) | yes |
| taxi-tabular (v1, new) | **7.5** | 7.765 / 7.975 / 8.185 | ≈ -770 | 2.7 s | 120 s | yes |
| cliffwalking-sarsa (v1, new) | -17 (SARSA's safe path; Q-learning's -13 also passes) | -17 / -17 / -17 | far below | 0.9 s | 60 s | yes |

Threshold choices:
- **MountainCar -110**: Gymnasium's own `reward_threshold`. All 10 seeds pass,
  with 3.2 of margin on the worst; the draft -130 was too loose to separate good
  from mediocre.
- **Taxi 7.5, not Gymnasium's 8**: the reference is essentially optimal (it
  converged to the same policy on every seed), but the mean over 1000 held-out
  episodes varies with the start-state distribution (7.77–8.19). At 8 the reference
  itself would pass on only 2 of 10 seeds, so criterion 1 fails. At 7.5 all pass
  with 0.27 of margin, and random sits about 780 below.
- **FrozenLake 0.9 × optimal**: the reference reaches ≥ 0.977 of optimal on all 20
  hidden maps, so 0.9 leaves margin for an honest but less tuned learner, while
  a policy written for another map fails.
- **Train timeouts**: references are tuned. LLM-written code is typically slower, so
  timeouts are roughly 10–25× the reference re-run rather than the mechanical 3×.

### 4. Deferred: CartPole-v1 and Acrobot-v1 (step 9b)
They need tile-coding/DQN references and longer calibration runs. They are
scheduled as step 9b in PLAN.md, not dropped.

## Alternatives considered
| Option | Why not |
|---|---|
| Fixed threshold on random maps | optimal success on random 4x4 maps ranges from 0.25 to 0.999 (measured), so a fixed threshold is unfair on some maps and trivial on others |
| Average over hidden maps instead of the worst | a policy that solves one map and fails another would pass on average |
| Show hidden maps' contents in trajectories | trajectories feed training; the maps would leak into the model across episodes |
| Use Gymnasium's `reward_threshold` everywhere | Taxi's 8 is within evaluation noise of optimal; FrozenLake's 0.70 doesn't apply to other maps |
| Calibrate once on the task seed only | hides seed-to-seed variance (Taxi spans 0.4 across seeds) |

## Consequences
- Every task in the suite has a committed calibration report backing its numbers.
- `make test` grew to about 50 s because FrozenLake verification now re-trains on
  hidden maps. Tests use one hidden instance; production tasks use two.
