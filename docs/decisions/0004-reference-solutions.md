# ADR-0004: Reference solutions prove solvability through the real pipeline

Status: accepted (step 4). Two of six planned references (end-to-end first);
the rest arrive with calibration (step 9). **Update (step 11):** four references
exist and are calibrated: FrozenLake (tabular Q-learning), MountainCar (tile
coding + SARSA), Taxi (tabular Q-learning) and CliffWalking (SARSA). CartPole and
Acrobot (step 9b) are on the roadmap.

## Context
A task is only useful if it is solvable with the allowed methods inside its
budget, and unsolvable by chance. The harness also needs known-good agent
behaviour for the scripted agent (step 5) and for calibration.

## Decision
- `reference/<task-id>/` holds `train.py`, `policy.py` (+ local modules) and
  `REPORT.template.md`. They are written exactly like an agent would write them:
  numpy + gymnasium only, data artifacts, the same CLI contract.
- `rlenv check-reference <id|all>` runs them **through the verification path**:
  fresh workspace → static checks → sandboxed `train.py --seed <task seed>`
  under the task's train timeout → harness evaluation on the **held-out seeds** →
  compare with the threshold and the random baseline.
- `make test` runs fast smoke versions (few episodes, 10 eval episodes). Full
  runs are `make reference-check`.
- Report templates contain the five required sections. Numbers are filled from
  harness results at run time (step 5), never hard-coded.

## Results (dev machine, task seed 1234, full held-out evaluation)

| Task | Method | Held-out mean | Threshold | Random | Train time |
|---|---|---|---|---|---|
| frozenlake-tabular | tabular Q-learning, 30k episodes | **0.735** (1000 eps) | 0.70 | 0.006 | 8.4 s (limit 120 s) |
| mountaincar-tilecoding | tile coding + semi-gradient SARSA, 1000 episodes | **-99.7** (100 eps) | -130 | -200 | 4.3 s (limit 300 s) |

## Findings while building them (worth keeping)
- **FrozenLake**: Q-learning with γ ∈ {0.99, 0.999} converged to the same
  greedy policy on 3 seeds (0.749 on 2000 test episodes; known optimum ≈ 0.74).
- **MountainCar, constant step size**: 500 episodes → -121…-126, but 1000
  episodes → -128…-153. More training made the policy *worse* (oscillating
  weights). A linearly decaying step size (0.5/8 → 0.1/8) plus checkpoint
  selection on training-side seeds gave -100…-108 over 5 seeds.
- **Tile offsets must wrap** (`% 1`): S&B's asymmetric displacement (1, 3)/n is a
  fraction of a tile width. Without the wrap, velocity indices left the weight
  array (regression test added).

## Consequences
- MountainCar's draft threshold (-130) looks loose relative to the reference
  (-100…-108). Calibration (step 9) will reconsider it (e.g. -115) using several
  seeds, so that a mediocre agent can't pass by luck.
- FrozenLake's margin is about 2.5 standard errors of the 1000-episode mean. Calibration
  checks it across seeds.
