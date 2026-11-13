# rl-agent-env

An environment for an LLM agent to solve classic RL problems on Gymnasium
with Sutton & Barto methods (tabular Q-learning, SARSA, tile coding, small DQN).
The agent writes code and runs experiments through tools. The harness verifies
the result independently and computes a verifiable reward. Every episode is
logged as a trajectory. See [PLAN.md](PLAN.md).

Local, CPU-only and offline by default.

**Start here**
| | |
|---|---|
| [docs/EVIDENCE.md](docs/EVIDENCE.md) | scoreboard, red team, calibration, a real DeepSeek episode, export samples, known limitations. Generated from committed real artifacts (`make evidence`) |
| [docs/reward.md](docs/reward.md) | reward specification (gates, components, guarantees) |
| [docs/decisions/](docs/decisions/README.md) | ADR index (12 decisions) |
| [samples/](samples/README.md) | real LLM trajectories + `messages`, `nemo-gym` and `atif` exports |
| [docs/reference-survey.md](docs/reference-survey.md) | OpenEnv, Harbor, Gymnasium, miles, verl, NeMo Gym: what we borrow and what we don't |

## Users and usage

Three kinds of users touch this environment. Each one drives specific design decisions.

### 1. Task authors (researchers writing tasks)
**How they use it:** write `tasks/<id>.yaml` (instruction, env id, threshold,
seed, budget, allowed methods), add a reference solution under `reference/<id>/`,
then run `rlenv list-tasks`, `rlenv show-task <id>` (what the agent will see)
and `rlenv check-reference <id>` (the reference is solvable through the real
pipeline).

**Design decisions this drives:**
- A **strict, declarative task schema**: unknown keys, unregistered env ids and
  inconsistent budgets fail at load time with path-qualified messages, so a typo
  can't silently become a different task.
- **A task is only valid with a passing reference solution** that runs through
  the same verification path as the agent (fresh workspace, sandbox, held-out
  seeds). Calibration (step 9) adds "reference passes on ≥ 90 % of seeds, random fails".
- **The agent view is explicit** (`show-task`): authors see exactly what the model
  sees, and the held-out salt is provably excluded.
- **Versioned tasks**: thresholds, seeds or budgets can't change silently;
  `version` and the task hashes go into every trajectory.

### 2. Training pipelines (batch consumers of trajectories)
**How they use it:** run many episodes (`rlenv run`, later in parallel), then
stream `runs/<episode>/trajectory.jsonl` into a training or analysis job:
filter by task, success and reward, join on task version, and replay code
versions from `blobs/`.

**Design decisions this drives:**
- **Self-describing, versioned records**: every trajectory starts with a header
  (schema version, task hashes and version, agent and model id, harness version,
  git commit, library versions). Files can be consumed without this repo's runtime state.
- **Append-only JSONL with a hash chain**: streamable line by line, and truncation
  or tampering is detectable before data enters training.
- **Rewards are recomputable** (`rlenv rescore`) from the verification record,
  and the reward function is versioned: a pipeline can re-weight old data
  explicitly instead of mixing reward definitions silently.
- **Reward breakdown, not just a scalar**: success flag, progress and report
  terms are stored separately so pipelines can choose their own objective.
- **Ready-made exports with grouping**: `rlenv export` writes OpenAI-chat
  `messages` rows (with a message-level loss mask) or NeMo Gym rollout rows,
  with a `group_id` per task version for group-relative methods, a
  `failure_kind` vocabulary, and unscorable episodes excluded by default.
- **Content-addressed code blobs**: deduplicated across versions and episodes;
  diffs between the agent's code versions are cheap.
- **No secrets in data**: API keys are redacted before anything is written,
  so trajectories can be shared.
- **A push API for trainers that own the rollout loop** (OpenEnv/Gymnasium
  shape, ADR-0012). The reward arrives only on the final observation, and
  `reward_status` tells "pending" apart from "unscorable":

  ```python
  from rlenv.openenv_facade import RLTaskEnv, ToolAction
  with RLTaskEnv(task, root="runs") as env:
      obs = env.reset(agent_info={"model": "my-policy"})    # instruction, task, tools, budget
      while not obs.done:
          name, args = my_policy(obs)                        # your model picks a tool call
          obs = env.step(ToolAction(name, args))             # one tool call -> its result
      obs.reward, obs.metadata["reward_status"]              # e.g. 1.0, "scored"
  ```

### 3. The model being trained (the agent)
**How it uses it:** receives the instruction, the agent view of the task and
the tool schemas, then calls `write_file`, `run_training`, `evaluate`, … and
`submit`.

**Design decisions this drives:**
- **Every tool result is structured and actionable**: `{ok, data, error,
  budget_remaining}`. Static-check failures list file, line and rule, and don't
  consume a training run. The remaining budget is visible at every step.
- **No hidden traps, no hidden help**: the instruction states the file contract
  (`train.py`, `policy.py`, `REPORT.md`) and that final scoring re-runs
  training on held-out seeds. Dev evaluation is honest but cannot be overfit to
  the final score.
- **The only path to reward is a policy that works**: the harness owns the env and
  the returns, re-runs training from scratch and checks report numbers against
  logged runs. Optimising anything else (printing results, planting artifacts,
  patching gym) earns nothing.
- **Same tools for every driver**: one tool definition serves scripted, replayed
  and LLM agents (DeepSeek `deepseek-flash` via its OpenAI-compatible API), so behaviour seen in
  offline tests is the behaviour the model gets.

## Quick start

```bash
make test      # creates .venv (gymnasium 1.4.0, numpy, pydantic), fast offline tests (~1 min)
make test-all  # + slow end-to-end runs (real training, full held-out evaluation)
make tasks     # validate and list task specs
.venv/bin/rlenv show-task mountaincar-tilecoding   # what the agent sees (no held-out salt)
make reference-check   # reference solutions vs thresholds on held-out seeds (~20 s)
make calibrate         # reference over 10 seeds per task -> calibration/<task>.md (several minutes)
.venv/bin/rlenv run frozenlake-tabular   # one scripted-agent episode through the tools (+ trajectory)
.venv/bin/rlenv check-trajectory runs/<episode_id>   # verify hash chain, head and blobs
.venv/bin/rlenv rescore runs/<episode_id>            # recompute the reward from the verification record
.venv/bin/rlenv export runs -o rows.jsonl [--format messages|nemo-gym]   # training rows
.venv/bin/rlenv export runs -o atif/ --format atif   # Harbor ATIF v1.8, one JSON per episode
make evidence  # regenerate docs/EVIDENCE.md from evidence/, samples/, calibration/ (checks + rescores them)
make redteam   # reference + nop + random floor + adversaries vs expected rewards
# Real LLM episode (calls DeepSeek, costs money; needs DEEPSEEK_API_KEY):
.venv/bin/rlenv llm-preflight --allow-external-api
.venv/bin/rlenv run frozenlake-tabular --agent llm --allow-external-api
.venv/bin/rlenv run frozenlake-tabular --agent replay --replay runs/<episode_id>   # re-issue recorded calls
```

## Status

Steps 1–11 (with 9a and 10a) of [PLAN.md §10](PLAN.md#10-build-order-small-steps-each-ends-with-tests-a-commit-and-user-review):
task model + example tasks; workspace confinement, subprocess sandbox and static checks;
harness-owned evaluation (policy in a sandboxed process over a pipe), random baseline, seeds;
reference solutions for FrozenLake (tabular Q), MountainCar (tile coding + SARSA), Taxi and CliffWalking;
tool API, episode loop with budgets, scripted agent (first full episode end to end);
append-only hash-chained trajectories with content-addressed code blobs and secret redaction.
verification (fresh re-run, held-out evaluation, report checks) and the reward (ordered: any pass > any fail;
unscorable = None), `rescore`, exports for training pipelines (`messages`, `nemo-gym`);
red team (executable adversaries vs expected rewards) and a replay agent;
calibrated task suite (FrozenLake with hidden random maps, MountainCar, Taxi, CliffWalking),
audit signals, `rlenv calibrate` ([calibration/](calibration/));
LLM driver for DeepSeek (`deepseek-flash`; contract-tested with a fake LLM; real calls opt-in; first real episode scored 1.0, ADR-0011);
OpenEnv-style facade `RLTaskEnv` (reset/step/state/close over the same episode lifecycle, ADR-0012);
docs, ATIF export, committed evidence (red team + real LLM episodes) and the generated EVIDENCE.md (step 11).
Roadmap: CartPole and Acrobot (9b), see [PLAN.md](PLAN.md).
Reference survey (OpenEnv, Harbor, Gymnasium, miles, verl, NeMo Gym): [docs/reference-survey.md](docs/reference-survey.md).

| Path | What |
|---|---|
| `tasks/` | task specs (YAML), one per file |
| `rlenv/task.py` | task model + validation + loading |
| `rlenv/workspace.py` | per-episode workspace, path confinement (no `..`, no symlink escapes) |
| `rlenv/sandbox.py` | subprocess runner: timeout + group kill, rlimits, RSS watchdog, scrubbed env, CPU-only |
| `rlenv/policy_check.py` | AST static checks: import allowlist, forbidden calls, escape hatches, no pickle |
| `rlenv/seeds.py` | dev seeds (public) and held-out eval seeds (secret salt), disjoint |
| `rlenv/evaluator.py` | harness owns env + returns; validates actions; random baseline |
| `rlenv/policy_server.py` | runs in the sandbox: loads `policy.py`, answers obs -> action |
| `rlenv/reference.py` | runs a reference solution through the verification path |
| `reference/<task>/` | reference `train.py`, `policy.py`, `REPORT.template.md` |
| `rlenv/tools.py` | tool specs (defined once; OpenAI function format for the DeepSeek driver), validation, budget, the 6 tools |
| `rlenv/episode.py` | episode loop: agent turns, budget checks, events |
| `rlenv/agents/` | driver interface + scripted agent (reference solution through the tools) |
| `rlenv/trajectory.py` | trajectory writer/reader: hash chain, head, blobs, redaction |
| `rlenv/harness.py` | runs an episode and records its trajectory |
| `rlenv/verifier.py` | fresh re-run in a clean copy, held-out evaluation, baseline, report checks |
| `rlenv/reward.py` | reward: unscorable → hard gates → 0.6/0.3/0.1 (ordering guarantee) |
| `rlenv/report.py` | REPORT.md checks: sections, bold results match logged dev means, real run ids |
| `rlenv/export.py` | `messages` and `nemo-gym` exports (integrity-checked, unscorable excluded by default) |
| `rlenv/atif.py` | Harbor ATIF v1.8 export + `validate_atif` (RFC 0001 rules) |
| `rlenv/evidence.py` | builds docs/EVIDENCE.md from committed artifacts (fails on tampered/missing ones) |
| `rlenv/openenv_facade.py` | OpenEnv-style push API (`reset`/`step`/`state`/`close`) over the same lifecycle |
| `rlenv/agents/llm.py` | DeepSeek driver (stdlib HTTP client, tool loop, transcript, cost guard) + fake client |
| `rlenv/agents/adversarial.py` | adversaries + floor agents with declared expectations |
| `rlenv/agents/replay.py` | replays a verified trajectory's tool calls |
| `rlenv/redteam.py` | runs them all and checks outcomes against expectations |
| `rlenv/variants.py` | held-out env instances (FrozenLake random maps) + exact optimal policy |
| `rlenv/audit.py` | audit flags for review (never part of the reward) |
| `rlenv/calibrate.py` | calibration: pass rate, random floor, determinism, suggested budgets |
| `calibration/` | committed calibration reports per task |
| `evidence/redteam/` | committed red-team trajectories (reference, floors, 5 cheats) |
| `samples/` | committed real DeepSeek trajectories + exports |
| `docs/` | EVIDENCE, reward spec, ADRs, reference survey |
| `rlenv/cli.py` | `rlenv list-tasks`, `rlenv show-task` |
| `docs/decisions/` | ADRs |
| `docs/reference-survey.md` | what we borrow from OpenEnv, Harbor and Gymnasium, and what we don't |
