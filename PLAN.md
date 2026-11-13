# PLAN.md: LLM agent environment for classic RL (Gymnasium)

Status: **plan approved (2026-11-11)**; steps 1–7 done (end-to-end chain complete, merged to main 2026-11-12); step 8 done (red team, ADR-0009); step 9 done (calibration, held-out env instances closed the 'no learning' gap, audit flags, Taxi + CliffWalking, ADR-0010); steps 1–9 merged to main. **Order from here (user, 2026-11-12): 9a → 10 → 10a → 11, then 9b only if time allows (otherwise it stays on the roadmap).** 9a done (ADR-0011: `deepseek-reasoner`/`deepseek-chat` retired; default `deepseek-flash`; reasoning must be passed back). Step 10: driver done and contract-tested offline; first real episodes run (flash: reward 1.0 on FrozenLake; a relative-path sandbox bug found and fixed; token budget then switched to billable tokens (cache-miss input + output) with a 2M total hard cap and an 80 % wrap-up warning, ADR-0011). Real DeepSeek calls in step 10 only after the fake-LLM contract tests pass and the user confirms. Step 10 merged to main (2026-11-12). 10a done: OpenEnv-style facade `RLTaskEnv` over an explicit episode lifecycle shared with the driver loop (ADR-0012); same reward and trajectory as the loop (tested). 10a merged to main (2026-11-12). **Step 11 done:** docs/reward.md, ADR index + tidy-up, ATIF export (`rlenv/atif.py`, validator), committed real artifacts (`evidence/redteam/`, `samples/`), generated docs/EVIDENCE.md (`make evidence`, drift-tested), `rlenv timeline`, home-path scrubbing in trajectories. Remaining: 9b (CartPole, Acrobot) only if time allows, otherwise roadmap. Pending and scheduled: 9a (DeepSeek tool-calling check, before 10), 10a (OpenEnv facade, before 11).

## 1. Goal

An **environment for an LLM agent** (in the RL-for-LLMs sense). The agent gets
a task ("solve MountainCar-v0 with a classic method"). It uses tools to write
code, run training experiments and evaluate results. It must end with:

1. a runnable `train.py` that trains a policy for a classic-control/toy-text
   problem using **Sutton & Barto methods only**: tabular Q-learning, SARSA,
   tile coding with linear function approximation, or a small DQN; and
2. an experiment report `REPORT.md`.

The harness then **independently verifies** the result:
- it re-runs `train.py` from scratch;
- it evaluates the produced policy with its own evaluator;
- it computes a **verifiable reward**.

Every episode is logged as a **trajectory**: instruction, tool calls, code
versions, run results, success flag and reward breakdown. A reward can be
recomputed from the trajectory plus its stored artifacts.

Defaults: **dry, local, CPU-only, no external services.** The real LLM adapter
exists only behind an explicit flag; tests and demos use scripted or replayed agents.

## 2. Terminology

| Term | Meaning |
|---|---|
| Task | instruction + env id + success threshold + seed + time budget (+ limits) |
| Episode | one agent attempt at one task, from instruction to `submit` (or budget exhausted) |
| Workspace | per-episode sandbox directory the agent writes into |
| Run | one execution of agent code via `run_training` |
| Policy artifact | file(s) produced by `train.py` that the harness can load and evaluate |
| Verification | the harness's own re-run of `train.py` plus evaluation on held-out seeds |
| Trajectory | append-only JSONL log of one episode + content-addressed code store |

## 3. Architecture

```
            ┌────────────────────────── harness process (trusted) ───────────────────────────┐
 tasks/*.yaml ─► TaskLoader ─► Episode ─► AgentDriver ◄─► ToolRouter ─► Tools               │
            │                      │          (scripted |          │       write_file / read_file│
            │                      │           replay |            │       list_files / run_training
            │                      │           llm*)               │       evaluate / submit     │
            │                      │                               ▼                             │
            │                      │                        Sandbox runner (subprocess, limits)  │
            │                      │                        workspace/ (untrusted agent code)    │
            │                      ▼                                                             │
            │                TrajectoryLogger ─► runs/<episode>/trajectory.jsonl + blobs/        │
            │                      │                                                             │
            │   on submit / budget end:                                                          │
            │                Verifier ─► fresh re-run of train.py (task seed, time limit)        │
            │                    │       ─► Evaluator: env stepped HERE, policy queried          │
            │                    │          over a pipe in a sandboxed subprocess                │
            │                    ▼                                                               │
            │                RewardFunction ─► reward + breakdown + success flag                 │
            └─────────────────────────────────────────────────────────────────────────────────────┘
 * llm driver: off by default; needs --agent llm + API key env; never used in tests
```

Proposed layout (all inside `rl-agent-env/`):

```
rl-agent-env/
├── PLAN.md, README.md, Makefile, pyproject.toml
├── tasks/                       # task specs (YAML), one per file
├── rlenv/
│   ├── task.py                  # Task model + validation (pydantic)
│   ├── workspace.py             # per-episode workspace, path confinement
│   ├── sandbox.py               # subprocess runner: timeouts, rlimits, env scrubbing, CPU-only
│   ├── policy_check.py          # static checks: import allowlist, forbidden calls, DQN size
│   ├── tools.py                 # tool API (schemas + implementations)
│   ├── evaluator.py             # harness-owned env loop; policy over a pipe
│   ├── policy_server.py         # runs inside the sandbox: loads agent policy, answers obs->action
│   ├── verifier.py              # fresh re-run + evaluation + report checks
│   ├── reward.py                # reward function (pure, versioned)
│   ├── trajectory.py            # JSONL logger + content-addressed blob store + rescore
│   ├── agents/
│   │   ├── base.py              # AgentDriver interface (observation -> tool call)
│   │   ├── scripted.py          # reference solutions (offline, deterministic)
│   │   ├── replay.py            # replays recorded tool calls from a trajectory
│   │   └── llm.py               # LLM adapter (OFF by default)
│   ├── episode.py               # the loop: budget accounting, tool routing, logging
│   └── cli.py                   # rlenv run | verify | rescore | calibrate | list-tasks
├── reference/                   # reference train.py per method (used by the scripted agent)
└── tests/
```

## 4. Task format

```yaml
# tasks/mountaincar-tilecoding.yaml
id: mountaincar-tilecoding
version: 1
instruction: |
  Write train.py that solves MountainCar-v0 with a classic method from Sutton & Barto
  (tile coding + SARSA/Q-learning recommended). train.py must save a policy that
  policy.py can load. Then write REPORT.md describing method, hyperparameters and results.
env_id: MountainCar-v0
env_kwargs: {}
seed: 1234                    # training seed passed to train.py (--seed)
success:
  metric: mean_return         # mean undiscounted return over eval episodes
  threshold: -130.0
  eval_episodes: 100
  eval_seed_salt: "v1"        # held-out eval seeds derived from seed + salt; never shown to the agent
budget:
  wall_clock_s: 1800          # whole episode (agent think time + all runs)
  train_timeout_s: 300        # per run_training call AND for the verification re-run
  max_tool_calls: 60
  max_runs: 15
methods:                      # allowed families; anything else = violation
  allowed: [tabular_q_learning, sarsa, expected_sarsa, tile_coding, small_dqn]
limits:
  dqn_max_params: 50000
  memory_mb: 2048
```

Validation: known env id (allowlist), threshold sanity (between the random
baseline and the best known return, calibrated in step 9), budgets > 0,
`train_timeout_s` ≤ `wall_clock_s`.

**Initial task suite** (thresholds are placeholders until calibration):

| Env | Natural method | Threshold (draft) | Note |
|---|---|---|---|
| FrozenLake-v1 (4x4, slippery) | tabular Q / SARSA | mean success ≥ 0.70 | optimal ≈ 0.74 |
| Taxi-v4 | tabular Q / SARSA | mean return ≥ 7.0 | optimal ≈ 8 |
| CliffWalking-v1 | SARSA vs Q-learning | mean return ≥ -17 | the textbook S&B example |
| CartPole-v1 | tile coding / small DQN | mean return ≥ 195 | 475 likely too slow for CPU budgets |
| MountainCar-v0 | tile coding + SARSA | mean return ≥ -130 | S&B chapter 10 |
| Acrobot-v1 | tile coding / small DQN | mean return ≥ -100 | stretch |

Env ids are confirmed against the pinned Gymnasium 1.4.0 (step 1): it registers
`CliffWalking-v1` and `Taxi-v4` (not the older `-v0`/`-v3`). Task validation
rejects ids the installed version doesn't register.

A task enters the suite only if, in calibration, (a) the reference solution
passes in ≥ 90 % of seeds within the budget and (b) a random policy fails.

## 5. Contract for the agent's code

The workspace must contain:

| File | Contract |
|---|---|
| `train.py` | `python train.py --seed S --out policy/` trains and writes the artifact into `policy/`. Must finish within `train_timeout_s`. May print metrics. |
| `policy.py` | `load(path) -> Policy` where `Policy.act(obs) -> action` (greedy, deterministic). Pure inference: no training, no env access. |
| `REPORT.md` | required sections: Method, Hyperparameters, Experiments (table of runs), Results, Limitations |

Artifacts must be data, e.g. `.npy`/`.npz` arrays or JSON, not pickle.
Pickle is forbidden because unpickling can execute code inside the evaluator.
The `policy.py` server runs sandboxed anyway, but data-only artifacts keep
policies inspectable.

## 6. Tool API

JSON-schema tools (defined once; converted to the OpenAI function-calling format for the DeepSeek driver). Every call is
logged. Every result is truncated to a fixed size and carries
`{ok, data, error, budget_remaining}`.

| Tool | Args | Effect | Notes |
|---|---|---|---|
| `write_file` | `path`, `content` | write a file in the workspace | path confined to the workspace; size cap; stores a code version (sha256) |
| `read_file` | `path`, `max_bytes?` | read a workspace file | |
| `list_files` | `path?` | list the workspace tree | |
| `run_training` | `args?`, `timeout_s?` | static check, then `python train.py --seed <task seed or given> --out runs/<id>/policy` in the sandbox | returns exit code, stdout/stderr tail, duration, artifact list; consumes the time budget; counts toward `max_runs` |
| `evaluate` | `run_id`, `episodes?` | harness evaluator on **dev seeds** (not the held-out eval seeds) | returns mean/std return, per-episode returns; used for iteration |
| `submit` | `notes?` | ends the episode and triggers verification | without `submit`, verification runs when the budget is exhausted |

Deliberately **not** offered: shell access, network, package install,
reading harness files or held-out seeds.

## 7. Verification and reward

### 7.1 Verification (only the harness's numbers count)

1. **Static checks** on the final workspace:
   - `train.py`, `policy.py` and `REPORT.md` exist;
   - import allowlist (`numpy`, `gymnasium`, `math`, `random`, `json`, `argparse`, `pathlib`, …;
     `torch` only if the small-DQN extra is enabled);
   - forbidden: `subprocess`, `socket`, `os.system`, `ctypes`, `importlib` tricks, pickle,
     RL libraries (`stable_baselines3`, …);
   - DQN parameter count ≤ `dqn_max_params` (checked on the loaded artifact).
2. **Fresh re-run**: clean copy of the workspace → `train.py --seed <task seed>`
   within `train_timeout_s`, in the sandbox. The agent's earlier runs are never
   trusted.
3. **Evaluation**: the harness steps `gymnasium.make(env_id)` itself over
   `eval_episodes` held-out seeds. Only observations go to the policy server
   (a sandboxed subprocess running `policy.py`), and only actions come back.
   Returns are summed by the harness, so agent code cannot touch the env,
   the RNG or the return accounting.
4. **Report checks**: required sections present; every numeric result claimed
   in the report's results table must match a logged run within tolerance
   (catches made-up numbers).

### 7.2 Reward (pure function of the verification record; versioned)

```
hard_fail = missing files | static violation | re-run crash/timeout | evaluation error
if hard_fail:                        reward = 0.0
else:
    s      = clip((R_eval - R_random) / (R_threshold - R_random), 0, 1)   # progress, works for negative returns
    success= R_eval >= R_threshold                                         # the binary success flag
    report = 1 if report checks pass else 0
    reward = 0.6 * success + 0.3 * s + 0.1 * report
```

- `R_random`: mean return of a uniform random policy on the same eval seeds
  (computed by the harness and cached per task version).
- The success flag is **binary and verifiable**. The shaped term `s` gives
  partial credit for learning signal. The report term is small, so a report
  can never compensate for a policy that doesn't work.
- Alternatives (pure binary; binary + efficiency bonus for budget left) are
  configurable via `reward.version` and recorded in each trajectory. Changing
  weights never rewrites old scores; `rescore` makes it explicit.

## 8. Trajectory format

`runs/<episode_id>/trajectory.jsonl`, append-only, one event per line:

```json
{"type":"header","episode_id":"…","task":{…full spec, minus eval salt…},"task_sha256":"…","agent":{"kind":"scripted","version":"…"},"harness_version":"…","git_commit":"…","reward_version":"v1","started":"…"}
{"type":"tool_call","i":3,"tool":"write_file","args":{"path":"train.py","sha256":"ab12…"},"t":12.4}
{"type":"tool_result","i":3,"ok":true,"data":{"bytes":2841},"budget_remaining":{"wall_clock_s":1787,"runs":15,"tool_calls":56}}
{"type":"run","run_id":"r2","code":{"train.py":"ab12…","policy.py":"cd34…"},"exit":0,"duration_s":41.2,"dev_eval":{"mean":-142.1,"std":18.3,"episodes":20}}
{"type":"verification","static":{…},"rerun":{…},"eval":{"mean":-118.4,"std":9.7,"episodes":100,"eval_seeds_sha256":"…"},"report":{…}}
{"type":"final","success":true,"reward":0.97,"breakdown":{"success":1,"progress":1.0,"report":0.7},"ended":"…"}
```

- Code versions are stored content-addressed in `runs/<episode_id>/blobs/<sha256>`,
  so every version the agent wrote can be diffed.
- `rlenv rescore <trajectory>` recomputes the reward from the verification
  record, which proves the reward is reproducible. `rlenv verify <workspace>`
  re-runs the verification itself.
- The LLM driver additionally logs model id, prompts and responses (token counts) for training use.

## 9. Key decisions (options → proposal)

### D1. Sandbox for untrusted agent code
| Option | Pros | Cons |
|---|---|---|
| A. Subprocess + `resource` rlimits (CPU, memory, files), timeouts, scrubbed env, fresh workspace, `CUDA_VISIBLE_DEVICES=""` | no dependencies, works on macOS/Linux, fast | weak isolation: no network or FS isolation on macOS |
| B. Docker container per run (`--network none`, read-only root, mem/cpu limits) | strong isolation, network off | Docker not installed here; slower; heavier |
| C. macOS `sandbox-exec` / Linux `bwrap`/nsjail | network + FS isolation without containers | platform-specific |

**Proposal:** A now, behind a `Sandbox` interface; B as a drop-in for
shared/production use. Static checks (no `socket`/`subprocess`) reduce risk
but are not a security boundary. This is documented as such.

### D2. Who computes the return?
| Option | Pros | Cons |
|---|---|---|
| A. Agent's own `evaluate.py` prints the return | trivial | reward hacking: the agent can print anything |
| B. Harness imports `policy.py` in-process | simple | agent code can monkeypatch gym or the evaluator |
| C. Harness owns the env; the policy runs in a sandboxed subprocess over a pipe | agent cannot touch env, seeds or returns | IPC overhead (fine for classic control) |

**Proposal:** C.

### D3. DQN framework
| Option | Pros | Cons |
|---|---|---|
| A. numpy-only (agent writes a small MLP + backprop) | tiny deps, deterministic, fast install, easy sandbox allowlist | harder for the agent to write correctly |
| B. PyTorch CPU (optional extra) | natural for DQN | ~200 MB install, thread nondeterminism, larger attack surface |

**Proposal:** numpy-only by default (tabular and tile coding need nothing
else); `torch` as an opt-in extra per task (`methods.allow_torch: true`), with
`torch.set_num_threads(1)` and deterministic flags enforced in the sandbox.

### D4. Agent drivers
Scripted (reference solutions; makes the env testable and calibratable),
replay (re-executes a logged trajectory: regression tests and debugging), and
LLM (DeepSeek via its OpenAI-compatible API: off by default; needs `--agent llm`, `DEEPSEEK_API_KEY`
and a confirmation flag; never in tests).

### D5. Budget accounting
Wall-clock for the whole episode (realistic for agents), plus hard caps on
tool calls and runs (deterministic, independent of machine speed). Calibration
runs record machine speed; budgets in the task are for this reference machine.

## 10. Build order (small steps; each ends with tests, a commit and user review)

| Step | Deliverable | Offline check |
|---|---|---|
| 0 | `PLAN.md` | — |
| 1 | Project skeleton, venv (`gymnasium`, `numpy`, `pydantic`, `pytest`), task model + YAML validation + 2 example tasks | `make test` |
| 2 | Workspace (path confinement) + sandbox runner (timeouts, rlimits, scrubbed env) + static policy checks | tests: escape attempts, timeout, forbidden imports |
| 3 | Evaluator + policy server (pipe protocol), random baseline | tests: a tampering policy cannot change returns; determinism with seeds |
| 4 | Reference solutions: tabular Q (FrozenLake, Taxi, CliffWalking), tile-coding SARSA (MountainCar, CartPole) | each passes its draft threshold locally |
| 5 | Tools + episode loop + budget accounting + scripted agent | end-to-end episode with the scripted agent |
| 6 | Trajectory logger (append-only JSONL, hash chain + head) + blob store + secret redaction + `check-trajectory` (rescore moves to step 7, where the reward exists) | tamper/truncation/blob-loss detected; no secrets on disk; crash leaves a valid trajectory |
| 7 | Verifier (fresh re-run, held-out eval) + reward function (gates + weighted sum, named components, `None` = unscorable for infrastructure failures, fail closed on missing limits) + report checks + `rescore` | tests for each hard-fail path and reward edge cases; rescore reproduces the reward |
| 8 | Replay agent + adversarial agents (≥ 3 reward-hacking attempts) + nop agent (floor) | each hack scores 0 / is caught; nop scores 0 |
| 9 | Calibration: thresholds and budgets from reference runs × seeds (Gymnasium `reward_threshold` as anchors); random baselines; determinism across repeats | calibration report, final task suite |
| **9b** | CartPole-v1 (tile coding or small numpy DQN) and Acrobot-v1 tasks: references, calibration as in step 9 (deferred from step 9 to keep it reviewable) | calibration reports; reference admitted |
| **9a** | **DeepSeek tool-calling check (required before step 10):** confirm from DeepSeek's official docs whether `deepseek-reasoner` (thinking mode) supports function/tool calling, how `reasoning_content` must be handled across tool turns, and which model id the driver uses (`deepseek-chat` vs `deepseek-reasoner`). Record the conclusion with sources in an ADR. No API calls unless the user approves them | written conclusion + ADR; step 10 may not start without it |
| 10 | LLM driver: DeepSeek (OpenAI-compatible), off by default; prompt, tool schemas, transcript logging; then real episodes with the user's key to generate trajectories | contract tests with a fake LLM; real runs only on explicit go |
| **10a** | **OpenEnv-compatible facade (required before step 11):** thin adapter mapping `reset()` → start an episode and return the first observation, `step(action)` → one tool call returning an observation with `done`/`reward` (reward only on the final step, `None` = unscorable), `state` → episode id/step count, `close()` → end the episode, verify, write the trajectory. No server or Docker; same harness underneath | tests: a scripted agent through the facade gets the same reward and trajectory as through the driver loop; reset/step/close misuse is rejected |
| 11 | Docs: README, reward spec, sample trajectories, ADRs; ATIF export (the gym/OpenEnv-style adapter is step 10a) | review |

## 11. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Reward hacking: fake prints, editing the evaluator, overfitting to eval seeds, pickled payloads | inflated reward | harness-owned eval over a pipe (D2), held-out seeds, fresh re-run, data-only artifacts; step 8 adversarial tests |
| Weak sandbox on macOS (no network/FS isolation) | untrusted code could reach the network or read files | static checks, scrubbed env, workspace-only cwd, rlimits; Docker backend for real use; documented limitation |
| Non-determinism (Gymnasium versions, numpy/BLAS threads, torch) | verification differs from the agent's own runs | pin versions; seed everything; single-threaded BLAS (`OMP_NUM_THREADS=1`); evaluation over many episodes; record versions in the trajectory |
| Thresholds too easy/hard for CPU budgets | useless signal | calibration step: reference passes ≥ 90 %, random fails |
| "Classic methods only" is hard to verify mechanically | agent uses e.g. PPO written in numpy | import allowlist + DQN size cap + method declared in the report; a reviewer rubric for what can't be checked mechanically |
| Slow tasks make tests slow | slow feedback | tests use tiny budgets and toy-text envs; long runs only in `make calibrate` |
| Gymnasium API/version drift | env ids or behaviour change | pin `gymnasium` version; task spec carries `env_id` with version suffix |
| Report checks too strict or too loose | noisy reward | small weight (0.1); only structural + number-consistency checks |
| Real task differs from this plan | rework | modular tools/drivers/reward; steps are independent |

## 12. Decisions confirmed by the user (2026-11-11)

1. Location: `rl-agent-env/` subdirectory of the existing repo.
2. Reward: `0.6·success + 0.3·progress + 0.1·report` (hard fails → 0).
3. DQN: numpy-only by default; torch opt-in per task (`methods.allow_torch`).
4. Task suite: FrozenLake, Taxi, CliffWalking, CartPole, MountainCar; Acrobot as stretch.
5. LLM driver: Anthropic API, key from the user's environment (`ANTHROPIC_API_KEY`);
   still off by default and enabled only on explicit request.
   **Changed (2026-11-12): DeepSeek via its OpenAI-compatible API**; the key comes
   only from `DEEPSEEK_API_KEY`, is never written to the repo, trajectories or logs,
   and is never visible to sandboxed agent code. Step 10 uses it for real episodes
   that generate trajectories.

**Priority:** the end-to-end chain first (task → sandbox → harness → verification →
reward → trajectory → report). Steps 1–7 are built as thin, working slices before any
breadth: step 4 starts with two reference solutions (FrozenLake tabular,
MountainCar tile coding); the remaining references come with calibration (step 9).
Step 8 keeps at least three reward-hacking agents.
