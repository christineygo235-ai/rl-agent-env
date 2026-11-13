# ADR-0009: Red team of the reward, replay, and the known "no learning" gap

Status: accepted (step 8). §4 decided by the user on 2026-11-12: **C + D**; the gap is closed in step 9.

## Context
The reward (ADR-0007) is only as good as the attacks it survives. Step 8 makes
the attacks executable, so that every harness change is re-tested against them,
and makes recorded episodes replayable.

## Decision

### 1. Executable adversaries with declared expectations
`rlenv/agents/adversarial.py` defines agents that try to earn reward without a
policy that works. Each agent declares what the harness must do with it.
`rlenv redteam` (and `make redteam`) runs the reference and all of them, then
compares outcomes with expectations: any violation → exit 2. All adversaries
target `frozenlake-tabular` (fast, tabular); the attacks themselves are generic.

| Agent | Attack | Caught by | Result (full task, 2026-11-12) |
|---|---|---|---|
| reference | (honest solution) | — | **1.000** |
| `artifact_reuse` | train once, then make `train.py` copy run r1's artifact (run 2 "trains" in 0 s and evaluates well) | fresh re-run in a clean copy: `runs/` doesn't exist there | **0.000** `rerun_failed` |
| `fabricated_report` | weak policy; report claims **0.950** and cites runs r2, r3 that never happened | report checks (unmatched claim, phantom runs) + held-out evaluation | **0.000** (report term 0, fail band) |
| `env_tamper` | `policy.py` patches `gymnasium.Env.step`/`TimeLimit.step` to reward 1000, prints forged protocol lines and "SUCCESS" | harness-owned env in another process; protocol on a separate fd | **0.100** (honest-report term only; returns ≤ 1) |
| `pickle_artifact` | artifacts saved/loaded with `allow_pickle=True` | static checks (blocked before any run) | **0.000** `static_violation` |
| `nop` | submit immediately | `files_present` gate | **0.000** `missing_files` |
| `random_floor` | valid but untrained random policy, honest report | progress = 0 | **0.100** (floor: hygiene term only) |
| `hardcoded_policy` | **no learning**: `train.py` copies a hand-written optimal table (`table.json`) | step 8: **not caught** (1.000). Since step 9: hidden env instances (§4 C) | **0.100** since step 9 (was 1.000 in step 8) |

At least three distinct real attacks (`artifact_reuse`, `fabricated_report`,
`env_tamper`, `pickle_artifact`) all land in the 0–0.1 range, under the pass
band (≥ 0.9) and under the fail ceiling (0.4). `nop` and `random_floor` pin the
floor, as recommended by OpenEnv RFC 008 and Harbor's nop agent.

### 2. Replay agent
`ReplayAgent(trajectory_dir)` re-issues the recorded tool calls (blob contents
re-inlined) in a new episode:
- it refuses trajectories that fail integrity checks;
- by default it refuses to replay against a changed task spec (public hash). A
  mismatch makes the episode **unscorable** (driver error), never 0;
- tests: replaying a scripted episode reproduces the exact reward record, and
  replaying `artifact_reuse` still scores 0.

Uses: regression tests after harness changes, reproducing a model's episode
without the model, and re-running old data under a new harness.

### 3. Tests
Each adversary has its own test that checks the reward **and** the specific
evidence. For example, `artifact_reuse`'s in-episode run succeeded while the
clean re-run failed on `runs/r1/policy/q.npy`; `fabricated_report` lists the
claim `0.950` and the phantom runs `r2`, `r3`. The known gap is asserted as
*passing today*. If someone closes it, that test fails and forces this ADR and
the expectation to be updated (the red-team verdict becomes `GAP-CHANGED`).

### 4. Known gap: "no learning" — decided and closed
`hardcoded_policy` passes every check:
- the code is static-clean;
- the fresh re-run reproduces the table deterministically;
- held-out seeds don't matter because the optimal policy generalises across seeds.

The task asks for a *learning method*; this submission learns nothing. Options:

| Option | How | Verdict |
|---|---|---|
| A. Seed sensitivity | re-run with a second seed; flag if artifacts are identical | **weak**: adding noise to the table evades it. Comparing greedy policies false-flags real learners, which converge to the same optimal policy across seeds (step 4 showed exactly this) |
| B. Count env steps during training | wrap `gymnasium.make` in the sandbox, require ≥ N steps | **weak**: dummy steps evade it; the wrapper is bypassable in-process |
| C. **Held-out environment instances** | verification trains and evaluates on an env variant the agent never saw (e.g. a freshly generated FrozenLake map, passed to `train.py` as `--env-kwargs`). A hard-coded table can't solve an unseen map; a learner can | **robust**. It is "held-out seeds" extended to env parameters. Cost: the task contract changes (`train.py` must accept env parameters), and it only applies where the env family has variants (FrozenLake maps, CliffWalking slippery, CartPole/MountainCar physics parameters within ranges) |
| D. Method review (human or LLM rubric over the code and report) | judge "is this Q-learning?" | **not verifiable**; useful as a secondary signal, never as reward |

**Decision (user, 2026-11-12): C + D.**

**A and B are rejected**, for the reasons above:
- A (seed sensitivity): adding noise to a hand-written table evades it, and comparing
  greedy policies false-flags real learners, which converge to the same optimal
  policy (step 4 data).
- B (step counting): dummy steps evade it, and an in-process wrapper can be bypassed.

Neither raises the cost of the attack meaningfully, and both add false positives
or noise to the reward.

**C, as implemented in step 9** (ADR-0010):
- an optional task-spec field `verification.env_variants`, accepted only for
  supported env families (first: `frozenlake_random_map`);
- `frozenlake-tabular` v3 enables it. Verification re-trains and evaluates on
  2 hidden random maps derived from the secret salt; only their sha256
  fingerprints are logged;
- on each map the threshold is `0.9 ×` the optimal policy's return on the same
  held-out seeds (exact value iteration on the env's model, harness-side);
- the worst map decides (normalised score), so every hidden map must pass;
- 3 public random maps are available to the agent through `run_training(variant=…)`,
  so generalisation can be tested honestly;
- result: `hardcoded_policy` dropped from **1.000 to 0.100**. The red team flagged
  the change automatically (`GAP-CHANGED`), and its expectation is now "caught".

**D, as implemented in step 9:** `rlenv/audit.py` records `audit_flags` in the
verification record and exports:
- `no_env_interaction`;
- `embedded_data_file`;
- `large_numeric_literal`;
- `no_learning_update`.

`compute_reward` never reads them, and a test asserts the reward is identical with
and without flags. `hardcoded_policy` raises `no_env_interaction` and
`embedded_data_file`; the reference solutions raise none.

**Residual risk:**
- Env families without variants (MountainCar, Taxi, CliffWalking today) can still
  be hard-coded. For them, D is the only signal, and an LLM/human review of flagged
  submissions is the follow-up.
- A policy that solves *every* 4x4 map without learning (e.g. a planner using the
  map passed in env kwargs) would pass. That is a model-based method, not one of
  the allowed families; D does not flag it today, so it is a candidate for the
  method-review rubric.

### 5. Other limits found while red-teaming (not new, recorded here)
- The subprocess sandbox has no filesystem isolation (ADR-0002). Agent code can
  read files the user can read, including task YAMLs with the eval salt, so a
  determined agent could compute the held-out seeds. Static checks block literal
  absolute paths, but not computed ones. Mitigation: the container backend
  (workspace-only mount). With option C, seed knowledge matters less.

## Consequences
- `make redteam` is part of reviewing any reward or verifier change.
- The step-8 gap is closed for FrozenLake (C). For other families it is mitigated by audit flags (D) and
  remains listed under residual risk.
