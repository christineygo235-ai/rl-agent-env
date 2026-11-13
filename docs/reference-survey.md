# Reference survey: OpenEnv, Harbor, Gymnasium, miles, verl, NeMo Gym

Date: 2026-11-11/12. Purpose: decide what rl-agent-env borrows from existing
environment frameworks and formats, and what it deliberately does not.
Facts were collected from primary sources (repos, official docs, release notes;
links inline). Items that could not be confirmed are marked *unconfirmed*.
Decisions are recorded in [ADR-0005](decisions/0005-tools-episode-loop.md)
(API), [ADR-0006](decisions/0006-trajectories.md) (trajectory format),
[ADR-0007](decisions/0007-reward.md) (reward) and
[ADR-0008](decisions/0008-export-formats.md) (exports).

Status legend: **done** = implemented, **step N** = planned in PLAN.md,
**no** = deliberately not adopted.

---

## 1. Gymnasium 1.4.0 (Farama)

**What it is:** the reference single-agent RL environment API:
- `reset(seed, options) -> (obs, info)` and `step(a) -> (obs, reward, terminated, truncated, info)`;
- `EnvSpec` registry with `max_episode_steps` and `reward_threshold`;
- `gym.make` adds `PassiveEnvChecker`, `OrderEnforcing` and `TimeLimit`.

Sources: [Env API](https://gymnasium.farama.org/api/env/),
[migration guide](https://gymnasium.farama.org/introduction/migration_guide/),
[release notes](https://gymnasium.farama.org/gymnasium_release_notes/).
Facts that matter here (read from the 1.4.0 wheel):

| Env | max_episode_steps | reward_threshold (spec) |
|---|---|---|
| FrozenLake-v1 (4x4) | 100 | 0.70 (optimum ≈ 0.74) |
| Taxi-v4 | 200 | 8 (optimum ≈ 8.46) |
| CliffWalking-v1 | **None** | **None** |
| CartPole-v1 | 500 | 475 |
| MountainCar-v0 | 200 | -110 |
| Acrobot-v1 | 500 | -100 |

- `Taxi-v4` exists because v1.3.0 fixed the rainy/fickle variants. `CliffWalking-v1`
  added a slippery option, and the plain `CliffWalking-v0` id is gone from 1.4.0.
- The spec does **not** define over how many episodes `reward_threshold` is
  averaged; that protocol is the user's choice.
- Gymnasium has no LLM/text-agent convention beyond the `Text` space.
  Gym-style LLM projects exist (TextArena, BALROG, GEM), with differing shapes.

### Borrow
| What | Why | Status |
|---|---|---|
| Use the real API in the harness: `reset(seed=...)`, the 5-tuple `step`, wrappers from `gym.make` | the evaluator must see exactly what a user of the env sees | **done** (step 3) |
| **Explicit step limit when the spec has none** | CliffWalking-v1 has no `TimeLimit`: a policy that never reaches the goal never ends. Tasks for such envs must set `max_episode_steps` (validated), and the evaluator passes it to `gym.make` | **done** (this step; found by the survey) |
| `terminated` vs `truncated` for agent episodes | `submit`/deliberate stop = terminated, budget exhaustion = truncated. Training pipelines need the distinction for bootstrapping, exactly as in Gymnasium | **done** (outcome record) |
| Official `reward_threshold` values as calibration anchors | MountainCar's official -110 is tighter than our draft -130, and our reference reaches -99.7. FrozenLake's 0.70 matches ours | **step 9** |
| Pin the library and log versions | 1.x bumps env versions (Taxi v3→v4) and seeding details between releases | **done** (`gymnasium==1.4.0`, versions in every trajectory header) |

### Do not borrow
| What | Why not |
|---|---|
| Make the **agent-facing** episode a `gymnasium.Env` as the core abstraction | Gymnasium defines no LLM convention. Tool calls with JSON-schema arguments, structured results and a single delayed reward don't fit numeric spaces. The driver loop (`start`/`next_action`) stays the core. A thin gym-style adapter (`reset() -> observation`, `step(tool_call) -> (result, reward, terminated, truncated, info)`) is cheap to add for trainers that expect it (**step 11**, optional) |
| "Seed once, then reset without a seed" | Evaluation seeds every episode explicitly (`reset(seed=s_i)`), so each held-out episode is independently reproducible and fingerprintable. Training code may follow the Gymnasium advice |
| Vector envs / autoreset modes, `check_env` for our agent episode | irrelevant to a single sequential agent episode; the harness's own tests cover the episode contract |

---

## 2. OpenEnv (openenv/OpenEnv; Meta-PyTorch, Hugging Face and others)

**What it is:** "an end-to-end framework for creating, deploying and using isolated
execution environments for agentic RL, with a simple Gymnasium-style API".
- BSD-3, PyPI `openenv` 0.12.0.
- Self-described as early development: "APIs may still change".
- Environments are FastAPI servers in Docker images, with a WebSocket client.
- Typed `Action`/`Observation`/`State`; `Observation` carries `done` and `reward`.
- The agent sees only MCP tools; `reset`/`step` are control-plane only.
- Server-side composable rubrics.
- RFC 008 auto-validation (oracle/floor/leakage/determinism checks).
- An `openenv.yaml` manifest.

Sources: [repo](https://github.com/openenv/OpenEnv), [docs](https://huggingface.co/docs/openenv/index),
[RFC 002](https://github.com/openenv/OpenEnv/blob/main/rfcs/002-env-spec.md),
[RFC 008](https://github.com/openenv/OpenEnv/blob/main/rfcs/008-environment-auto-validation.md),
[TRL integration](https://huggingface.co/docs/trl/openenv).

### Borrow
| What | Why | Status |
|---|---|---|
| **Agent vs control-plane split**: reset, step and scoring are never agent tools | the agent cannot reset, re-seed or score itself. Our `submit` only ends the turn; verification and reward are harness-only | **done** (steps 3/5) |
| The environment, not the client, computes the reward | same principle as harness-owned evaluation (ADR-0003) | **done** |
| **Rubric composition: hard gates first, then a weighted sum**, reward range declared as [0, 1] | matches our reward (hard fails → 0, then 0.6/0.3/0.1). Naming it as gates + weighted terms keeps it inspectable and versioned | **step 7** |
| **"Unscorable" (`None`) ≠ 0** | an infrastructure failure (LLM API outage, harness crash) is not evidence about the policy. Recording it as 0 would teach the model from noise. Agent-caused failures stay 0 | **step 7** (the outcome already separates `agent_error` from terminated/truncated) |
| **RFC 008 validation ideas**: the oracle must reach the max, doing nothing must sit well below it (floor margin), no solution leakage in observations, canary trajectories stay at the floor, determinism across repeats | turns "is this task any good?" into automated checks. Fits our reference check, the adversarial agents and calibration | **steps 8–9** (nop agent, oracle score, leakage check: the agent view has no salt, **done**) |
| OpenAI-format `messages` + `tool_calls` + `reward` per record for LLM rollouts (`openenv collect`) | DeepSeek is OpenAI-compatible, so the LLM driver logs its transcript in that shape, which SFT/RL tooling already reads | **done** (step 10: exact transcript export) |
| **`reset`/`step`/`state`/`close` with `Observation(done, reward)` and `State(episode_id, step_count)`** | trainers that own the rollout loop expect this shape. A thin facade over the same lifecycle gives it without a second harness | **done** (step 10a, ADR-0012: `RLTaskEnv`; the reward only on the final observation; `reward_status` disambiguates `None`) |

### Do not borrow
| What | Why not |
|---|---|
| FastAPI/WebSocket server + Docker image per environment | our requirement is local, offline, no services, no Docker on the dev machine. An in-process harness with subprocess sandboxes is enough for CPU-scale classic control. The `Sandbox` interface leaves room for a container backend |
| MCP as the tool transport | tools are in-process function calls with JSON-schema args; MCP would add a protocol hop without adding isolation. If a remote agent needs it, an MCP adapter over `TOOL_SPECS` is mechanical |
| Depending on the `openenv` package | pre-1.0, weekly releases, RFCs "In Review": API churn risk for a pinned, reproducible environment. We copy ideas, not code |
| HF Hub packaging (`openenv push`, Spaces) | no external services; tasks are versioned in git |
| `LLMJudge` rubrics | not verifiable; our reward must be recomputable from logged facts |

---

## 3. Harbor (harbor-framework/harbor; Terminal-Bench team)

**What it is:** "a framework for evaluating and improving agents", Apache-2.0,
v0.25.0. A task is a directory:
- `instruction.md`;
- `task.toml` with timeouts, resources, network policy and artifacts;
- `environment/` (Dockerfile);
- `solution/solve.sh` (oracle);
- `tests/test.sh`, which writes `/logs/verifier/reward.txt` or a `reward.json` with named metrics.

The verifier can run in a **separate sandbox** that receives only declared,
sanitised artifacts. Datasets are manifests of task pointers, published as
versioned snapshots: patch = reuse trials, minor = regrade, major = rerun.
Trajectories use ATIF v1.8, one JSON document per trial. Reward-hacking
safeguards: `harbor lint`, non-root agent, unprivileged code runner, stale
reward files deleted, `harbor analyze`.

Sources: [repo](https://github.com/harbor-framework/harbor),
[task format](https://docs.harborframework.com/tasks/overview.md),
[configuration](https://docs.harborframework.com/tasks/configuration.md),
[verifier](https://docs.harborframework.com/tasks/verifier.md),
[separate verifier](https://docs.harborframework.com/tasks/separate-verifier.md),
[ATIF](https://docs.harborframework.com/agents/atif.md),
[reward hacking](https://docs.harborframework.com/news/reward-hacking.md).

### Borrow
| What | Why | Status |
|---|---|---|
| Task = instruction + config + verifier + **oracle solution** | our `tasks/<id>.yaml` (instruction, budget, threshold) + `reference/<id>/` (oracle) + the harness verifier. A task without a passing oracle is invalid | **done** (step 4), enforced in calibration (**step 9**) |
| **Separate verifier fed only declared artifacts, sanitised** | verification copies only regular source files into a fresh workspace (`copy_clean`: no symlinks, no stale artifacts) and re-runs training there. Nothing the agent left behind (planted `runs/`, caches) is trusted | **done** (`copy_clean`) / **step 7** (fresh re-run) |
| **Named reward components** (`reward.json`-style) next to the scalar | our breakdown (success, progress, report, gates) is stored alongside the scalar so pipelines can choose objectives | **step 7** |
| **Version semantics**: patch = reuse, minor = regrade, major = rerun | maps onto our two versions: a reward-function change → `rescore` (regrade from the record); a task change (threshold, seeds, env) → bump `task.version` and rerun | **step 7** (documented with `rescore`) |
| oracle + **nop** agents as task validation | a do-nothing agent must score 0 and the oracle must pass | **step 8** (nop as one of the adversarial agents) |
| Fail closed when isolation can't be enforced | Harbor rejects a trial if the provider can't enforce the network policy. Our sandbox records `limits_applied`; verification will refuse to issue a reward if a required limit wasn't applied | **step 7** |
| ATIF field names for an **export** (`steps[{source, message, tool_calls[{tool_call_id, function_name, arguments}], observation}]`) | interoperability with Harbor tooling and viewers | **step 11** (`rlenv export --format atif`) |

### Do not borrow
| What | Why not |
|---|---|
| ATIF as the **native** trajectory format | it is one JSON document per trial: not appendable while the episode runs, no hash chain, no verification record. A crash mid-episode would leave nothing valid. We keep append-only hash-chained JSONL as our own format and export ATIF |
| A Docker image per task, cloud sandboxes (Daytona, Modal, E2B, …) | all tasks share one pinned Python + Gymnasium environment; no system dependencies; offline/local requirement |
| TOML task configs, `instruction.md` as a separate file | YAML was chosen in step 1 (ADR-0001) and is validated by one pydantic model. The instruction is short and versioned with its thresholds. Revisit if tasks get long prose |
| Shell `test.sh` verifiers that write reward files | our verifier is harness code with typed results; a reward file written by a script is one more thing that could be forged |
| 40 agent adapters | three drivers (scripted, replay, LLM via DeepSeek) cover our users |

---

## 4. Trajectory formats in RL trainers: miles, verl, NeMo Gym

Added 2026-11-12, specifically to review our
trajectory and export formats. Facts come from each repo's `main` (read-only
source reads on 2026-11-11) and the official docs.

### 4.1 miles (radixark/miles)
**What it is:**
- RL post-training for LLMs/VLMs at scale (SGLang rollouts, Megatron/FSDP training);
  "forked from and co-evolving with slime"; Apache-2.0; v0.1.3, actively developed.
- One mutable, **token-level** `Sample` dataclass (`miles/utils/types.py`): `prompt`,
  `tokens`, `response`, `loss_mask`, `rollout_log_probs`, `reward: float | dict | None`,
  `status` ∈ {pending, completed, truncated, aborted, failed}, `group_index`, `metadata`,
  `lineage`.
- Multi-turn tool calls are one concatenated token stream; tool outputs get `loss_mask` 0.
  Messages are only kept for debugging in `metadata["messages"]`.
- Dumps: `--dump-details` writes `.pt` files plus a `trajectory/{id}.jsonl` with
  `{sample_index, group_index, status, reward, prompt, messages}`.
- A `None` reward means "not scored" and those groups are filtered out, but eval metrics
  treat None as 0.0.
- Provenance: `EnvReport` (versions, pip list, git state with a hashed diff).
- No hash chain, no content addressing.

| Borrow | Why | Status |
|---|---|---|
| A status vocabulary next to the reward | pipelines filter on it. We have `status` (scored, hard_fail, unscorable) + `end_reason` + `terminated`/`truncated` | **done** |
| Reward as scalar **and** dict, with a designated primary | our `reward` + `reward_components` | **done** |
| "None = not scored → filter, never average as 0" | the exact rule we want. Exports exclude unscorable episodes by default | **done** (export) |
| Group index for group-relative advantages | `group_id` on every final record and export row | **done** |
| Environment provenance including a dirty working tree | runtime header now records `git_dirty` next to `git_commit` | **done** |
| Role-tagged message JSONL alongside the heavy data | our `messages` export | **done** |

| Don't borrow | Why not |
|---|---|
| Token-level samples (`tokens`, `loss_mask`, `log_probs`) as the native record | we are the environment, not the trainer. DeepSeek's API doesn't return token ids, and token masks depend on the trainer's tokenizer. We export message-level `loss_mask`; trainers tokenize |
| Mutable sample objects and `torch.save` dumps | not append-only, not inspectable without torch, pickle-based (the class of risk we forbid for agent artifacts) |
| Treating None as 0 in metrics | that is exactly the confusion our `unscorable` status exists to prevent |

### 4.2 verl (verl-project/verl)
**What it is:**
- RL training library from the ByteDance Seed team (HybridFlow); Apache-2.0; v0.9.1;
  the agent loop still lives in `verl/experimental/`.
- Rollouts live in `DataProto` (`batch` tensors + `non_tensor_batch` + `meta_info`).
- Multi-turn agent output: `prompts`, `responses`, and a `response_mask` with 1 for LLM
  tokens and 0 for tool tokens.
- The reward is placed on the last response token (`rm_scores`). `uid` groups samples
  for GRPO.
- `ToolAgentLoop` returns tool errors to the model as text rather than aborting.
- `compute_score` returns a float or a dict (`score` + extras → `reward_extra_info`).
- Scoring timeouts become reward 0.0. Exclusion is done separately by group filters.
- Data: parquet with `data_source, prompt, ability, reward_model{style, ground_truth},
  extra_info`. `rollout_data_dir` dumps `{step}.jsonl` with `input/output/gts/score`.
- No version headers, hashing or integrity features in rollout dumps.

| Borrow | Why | Status |
|---|---|---|
| Errors as observations, never episode aborts | already our tool contract (`{ok:false, error}`); it lets the model learn from mistakes | **done** (step 5) |
| Loss mask: model turns = 1, tool outputs = 0 | message-level `loss_mask` in the `messages` export; verl derives token masks from it | **done** |
| `uid`-style grouping | `group_id` = task id + version + spec hash prefix: same prompt → same group, and a silent spec change → a new group | **done** |
| Score as dict with extras | `reward_components` alongside `reward` | **done** |
| OpenAI function schemas for tools | `openai_tools()` | **done** (step 6) |

| Don't borrow | Why not |
|---|---|
| Scoring failures → 0.0 | an infrastructure failure would become a confident negative signal. We use `None` + `failure_kind` |
| Reward placed on the last token as the stored representation | a trainer detail; we store the episode reward explicitly with its components |
| Parquet training-data schema as our native format | prompt datasets are verl's input, not an environment's trajectory. A converter from the `messages` export is trivial if needed |

### 4.3 NeMo Gym (NVIDIA-NeMo/Gym)
**What it is:**
- "A library for evaluating and improving models and agents using environments";
  Apache-2.0; v0.7.0, beta.
- FastAPI servers: environment, agent, model (OpenAI Responses API) and resources
  (tools + `/seed_session` + `/verify` + `/close_session`).
- It ships a **Gymnasium adapter** (`/reset`, `/step` returning the 5-tuple).
- Rollout row = the input row + `response` (Responses API `output[]` items
  `function_call` / `function_call_output`) + `reward` (float),
  `reward_components` (multi-reward), `mask_sample` (for infrastructure failures),
  `failure_kind` (shared low-cardinality vocabulary), `_ng_task_index`,
  `_ng_rollout_index`.
- `reverify` recomputes rewards from stored rollouts.
- `RunManifest` digests; an ATIF export.
- No per-event hash chain and no content-addressed code.

| Borrow | Why | Status |
|---|---|---|
| **`failure_kind` vocabulary** | pipelines count and filter failures without parsing prose. Ours: unscorable kinds (`infrastructure_error`, `agent_driver_error`, `isolation_not_enforced`, `baseline_unavailable`) and hard-fail kinds (`missing_files`, `static_violation`, `rerun_failed`, `evaluation_failed`) | **done** (final record + exports) |
| **`mask_sample`** for unscorable samples | the NeMo Gym spelling of our `None`; used in the `nemo-gym` export (with their reward 0.0 convention) | **done** |
| Responses API items for multi-turn tool calls | `nemo-gym` export emits `function_call` / `function_call_output` | **done** |
| Task and rollout indices | `_ng_task_index` (per group) and `_ng_rollout_index` (within group) in the export | **done** |
| Reward components also at top level | NeMo metrics only see top-level fields: export adds `reward_<name>` | **done** |
| `reverify` (recompute rewards from stored data) | our `rescore`, which also refuses when integrity fails | **done** (step 7) |
| Gymnasium `/reset` `/step` contract | supports the optional gym-style adapter planned for step 11 | **step 11** |

| Don't borrow | Why not |
|---|---|
| Server architecture (4 HTTP services, aiohttp, session cookies) | local, offline, single process with sandboxed children; no network services |
| Reward 0.0 + mask as the *native* representation of unscorable | `None` is safer as the native form (it can't be averaged by accident); the 0.0 + mask form only exists in the NeMo export, where consumers know the convention |
| Model-call capture files as the source of truth | our trajectory is the source; model transcripts will be part of it (step 10) |

### 4.4 Review of our trajectory and export design
**Unchanged (validated by the review):**
- The native format stays **event-level, append-only, hash-chained JSONL with
  content-addressed blobs**. None of the three has integrity or content addressing.
  All three separate the trainer's token data from the environment's record.
- `None` = unscorable in our own format. miles half-implements it; verl lacks it; NeMo Gym
  encodes it as `mask_sample`.

**Changed in this step** ([ADR-0008](decisions/0008-export-formats.md)):
1. The final record gets `group_id` and `failure_kind`; the runtime header gets `git_dirty`.
2. `rlenv export` produces derived rows:
   - `messages`: OpenAI chat, message-level `loss_mask`, tools, reward, components,
     status, failure kind, provenance;
   - `nemo-gym`: Responses items, `mask_sample`, `_ng_*` indices.
3. Exports run integrity checks first. Invalid or incomplete trajectories are skipped and
   reported, and every row carries the source trajectory's head hash.
4. Unscorable episodes are excluded by default (`--include-unscorable` to keep them, masked).
5. Code written by the model is re-inlined from blobs, so a training row is self-contained.

**Still planned:**
- ATIF export (Harbor, also produced by NeMo Gym) and the gym-style adapter: step 11.
- Recording the LLM's exact prompt and transcript inside the trajectory: step 10.
  Until then, exports reconstruct the prompt from the instruction and the agent view.

## 5. Summary of changes caused by this survey

1. **Done now:**
   - `max_episode_steps` is required for envs without a step limit (CliffWalking-v1), and the evaluator enforces it;
   - `terminated`/`truncated` in the episode outcome;
   - `openai_tools()` converter (one tool definition, OpenAI format for DeepSeek).
2. **Step 7:**
   - gates + weighted sum (rubric style);
   - `None` = unscorable for infrastructure failures;
   - named reward components;
   - fail closed if required limits weren't applied;
   - regrade (`rescore`) vs rerun semantics.
3. **Step 8:** nop agent (floor), oracle score check, canary trajectories.
4. **Step 9:** official `reward_threshold` as calibration anchors; determinism across repeats.
5. **Step 10:** OpenAI-format transcript in the LLM driver's records.
6. **Step 11 (optional):** gym-style adapter for the agent episode; ATIF export.
7. **Step 7, from §4 (miles, verl, NeMo Gym):**
   - `group_id`, `failure_kind` and `git_dirty` in records;
   - `rlenv export` (`messages`, `nemo-gym`), integrity-checked, excluding unscorable episodes by default.
