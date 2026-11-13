# ADR-0007: Reward = unscorable check → hard gates → ordered weighted sum

Status: accepted (step 7). Implemented in `rlenv/reward.py` (pure, versioned
`v1`). Facts come from `rlenv/verifier.py`.

## Context
The reward trains a model. It must:
1. rank outcomes the way a human reviewer would;
2. give a learning signal before the model can solve tasks;
3. be impossible to raise by anything other than a working policy;
4. never turn the harness's own failures into training signal.

It must also be recomputable from the logged verification record (`rlenv rescore`).

## Decision

```
if unscorable(record):           reward = None       # infrastructure failure, not the agent's
elif any hard gate fails:        reward = 0.0        # integrity failure; nothing is weighed
else:
    success  = mean_return >= threshold                                  # 0 or 1
    progress = clip((mean - random_mean) / (threshold - random_mean), 0, 1)
    report   = 1 if report checks pass else 0
    reward   = 0.6 * success + 0.3 * progress + 0.1 * report
```

All three numbers (`mean_return`, `random_mean`, `threshold`) come from the
harness. `mean_return` is evaluated on held-out seeds after a fresh re-run of
`train.py`. `random_mean` is a seeded random policy on the same seeds.

### 1. The core property: an ordering guarantee
**Every passing submission scores above every failing submission.**

- A pass gets at least the success term: **≥ 0.6**.
- A fail can get at most progress + report: **≤ 0.3 + 0.1 = 0.4**.
- 0.6 > 0.4, so no report, however good, and no near miss, however close,
  can lift a failure above a success.

This is enforced in code, not by convention. `reward.py` asserts
`W_SUCCESS > W_PROGRESS + W_REPORT` at import time, and a property test draws
5,000 random records (random thresholds, baselines, returns, report flags)
and checks `min(pass) > max(fail)`.

The inequality is deliberately the **strong form**. It holds for *any*
progress definition in [0, 1], so a later change to the dense term cannot
break the ranking. With the current definition progress is 1 at the threshold,
so passes actually land in [0.9, 1.0] and fails in [0, 0.4). The gap between
0.4 and 0.9 is empty by construction.

Why ordering matters for training: group-relative methods (GRPO and friends)
only use *relative* rewards within a group. If some failing rollout could
out-score some passing one, the model would be pushed towards the failing
behaviour in that group.

### 2. Progress: a dense signal, capped below success
Binary success alone gives zero gradient until the model first solves a task,
and classic-control tasks are rarely solved on the first try. `progress`
measures how far the evaluated return moved from the random baseline towards
the threshold:
- it works for negative returns (MountainCar: random -200, threshold -130,
  so -165 gives progress 0.5);
- it is 0 for anything no better than random, so "do nothing clever" earns nothing;
- it is clipped at 1, so there is no reward for overshooting the threshold.
  Overshooting isn't the task, and capping removes the incentive to farm return
  at the expense of everything else.

Its weight (0.3) is large enough to rank failing attempts by how close they
came. Its ceiling, together with the report term (0.4), still sits below the
success floor (0.6). Progress can rank failures among themselves, never against a pass.

### 3. Report: a hygiene term, so the smallest weight
The report proves the agent documented what it did. It does **not** prove the
policy works. Its checks (`rlenv/report.py`) are mechanical:
- the five required sections are present;
- every bold number in Results matches a dev-evaluation mean the harness
  logged during this episode;
- every cited run id really ran.

That catches made-up numbers and phantom runs, but it doesn't grade prose. A
large weight would reward writing over solving, so it gets 0.1: enough to
prefer honest documentation between otherwise equal outcomes, never enough to
matter across the pass/fail line.

### 4. Integrity failures: hard gates that zero the reward
These are not weighed against anything; any one of them makes the reward 0.0:

| Gate | Fails when | failure_kind |
|---|---|---|
| `files_present` | `train.py`, `policy.py` or `REPORT.md` missing | `missing_files` |
| `static_checks` | forbidden import, call or escape hatch in the final code | `static_violation` |
| `rerun` | the fresh `train.py --seed S --out DIR` (no extra args) crashes, times out or writes no artifact | `rerun_failed` |
| `evaluation` | the policy crashes, hangs, returns invalid actions or yields a non-finite mean on held-out seeds | `evaluation_failed` |

Why gates instead of penalties: a submission that only works next to stale
artifacts, or that breaks the rules, is not a slightly worse solution. It is
not a solution. Weighing it (e.g. "success 1, rules −0.5") would let a big
enough success outweigh cheating, which is exactly the trade we must not
offer. Gated outcomes carry no components (`components == {}`), so nothing
downstream can accidentally average them in.

### 5. Infrastructure failures: `None`, never 0
`reward = None` (status `unscorable`) when the outcome says nothing about the policy:

| Cause | failure_kind |
|---|---|
| exception inside the harness (verifier, evaluator, gym on the harness side) | `infrastructure_error` |
| the agent driver crashed (e.g. LLM API unreachable) | `agent_driver_error` |
| required sandbox limits were not applied to the re-run (CPU, file size, open files) | `isolation_not_enforced` (fail closed) |
| the harness could not compute its own random baseline | `baseline_unavailable` |

Recording these as 0 would teach the model that a network outage or a harness
bug means its behaviour was bad: confident, wrong signal. `None` can't be
averaged by accident. Exports exclude unscorable episodes by default, and the
NeMo Gym export encodes them as `mask_sample: true`. Unscorable checks run
**before** the gates, so a harness failure is never misattributed to the agent.

The verifier separates the two kinds of failure. Policy crashes and invalid
actions (`PolicyError`) are the agent's failures and fail the `evaluation`
gate. Any other exception in verification is the harness's failure and is
`infrastructure_error`.

### 6. Versioning and rescoring
- `reward.version = "v1"` is written in every final record.
- `rlenv rescore` recomputes the reward from the verification record. It fails
  if the trajectory's integrity checks fail, and it calls a version difference
  a **regrade**, not a reproduction.
- Changing weights or rules means bumping the version and regrading old data
  explicitly (Harbor's "minor = regrade"). Changing the task (threshold, seeds,
  env) means bumping `task.version` and rerunning ("major = rerun").

## Alternatives considered
| Option | Why not |
|---|---|
| Binary success only | no signal before the first success; classic control rarely solves on the first try |
| Return-based reward only (normalised mean) | no ordering: a high-return fail can out-score a just-passing success; rewards overshooting |
| Penalties instead of gates | lets enough success buy rule violations |
| Large report weight / LLM-judged report | rewards prose over solving; not verifiable |
| 0 for infrastructure failures (verl's timeout behaviour) | teaches noise; misattributes harness failures to the model |
| Reward on dev-seed results | the agent saw those seeds; scoring must be held-out |

## Consequences
- A pass is worth ≥ 0.9, a fail < 0.4, and gated or unscorable outcomes are 0 or None.
  Pipelines can read the band directly from the number.
- The report rules are part of every task instruction (bold result, real run ids,
  how final scoring works), so the hygiene term is not a hidden trap.
- Because verification ignores the agent's run arguments, the reward depends on
  the submitted code alone. A test checks this.
