# ADR-0005: Tool API, episode loop and budget accounting

Status: accepted (step 5)

## Context
The agent acts only through tools. Their shape determines what the agent can
do, what it can game, and what ends up in the trajectory. The same definitions
must serve the offline drivers (scripted, replay) and the LLM driver.

## Decision

### One definition of the tools
`rlenv/tools.py: TOOL_SPECS` uses the Anthropic tool format (`name`,
`description`, `input_schema`). The harness validates every call against it
(`validate_args`: types, required keys, no extra keys, bounds; `bool` is not an
integer). The LLM driver will send the same list to the model.

| Tool | Notes |
|---|---|
| `write_file(path, content)` | workspace-confined, text only; **`runs/` and `.tmp/` are harness-owned**, so artifacts can only come from `run_training` |
| `read_file(path, max_bytes?)`, `list_files(path?)` | read-only; run outputs under `runs/` are readable |
| `run_training(seed?, args?, timeout_s?)` | static check of the **whole workspace** first. On violation the code is not executed and **no run is consumed**. `--seed`/`--out` are reserved. The timeout is `min(requested, task limit, wall clock left)`. Exit 0 without artifacts is a failure. Each run records the code sha256 snapshot it executed |
| `evaluate(run_id, episodes?)` | **dev seeds only**, at most `dev_eval_episodes`; harness-owned evaluator (ADR-0003); records the `policy.py` sha256 used, because the agent may change `policy.py` after training |
| `submit(notes?)` | ends the agent's turn; reports missing required files; verification follows (step 7) |

Every result is `{ok, data, error, budget_remaining}`. Agent mistakes (bad
arguments, forbidden paths) are tool errors, never exceptions.

### Budget
- `wall_clock_s` covers the whole episode, including agent think time (realistic
  for LLM agents). `max_tool_calls` and `max_runs` are deterministic caps,
  independent of machine speed.
- Checked before every agent step. Exhaustion ends the episode with
  `end_reason = budget:<kind>`. Verification still runs on whatever is in the
  workspace.
- Every tool call costs one tool call, including invalid ones; otherwise
  spamming invalid calls would be free.

### Episode loop (`rlenv/episode.py`)
- `start(observation)`: the observation is the instruction, the agent view of the
  task (no eval salt), the tool specs and the budget.
- Then `next_action(last_result)` until `submit`, budget exhaustion, the agent
  returning `None` (`agent_stopped`), or an agent exception (`agent_error`). A
  crashing agent ends its own episode, never the harness.
- Every step emits events (`start`, `tool_call`, `tool_result`, `run`,
  `agent_error`, `end`) to a sink with a relative timestamp. Step 6 persists
  them as the trajectory.

### Scripted agent (`rlenv/agents/scripted.py`)
Replays a reference solution through the tools: write files → train →
evaluate → write `REPORT.md` from the template **with the numbers the harness
returned** → submit. It is deterministic and offline. It is the regression
test for the whole loop and the basis for calibration.

## Observed
The first full episode (FrozenLake, scripted, full training) took 6 tool calls
and 9.1 s. The dev evaluation gave 0.800 on 200 episodes, against 0.735 on 1000
held-out episodes for the same artifact. That gap is why final scoring uses
held-out seeds and more episodes than the agent's own `evaluate`.

## Alternatives considered
| Option | Why not |
|---|---|
| A generic `shell` tool | unbounded attack surface; can't be checked or budgeted per action |
| Let `evaluate` use the held-out seeds | the agent would overfit to them, and the final score would no longer be held-out |
| Agent-writable `runs/` | the agent could plant an artifact that `train.py` never produced |
| Free invalid calls | spamming invalid calls would cost nothing |

---

## Revisited after the reference survey (2026-11-12)

See [reference-survey.md](../reference-survey.md) (Gymnasium 1.4.0, OpenEnv, Harbor).
The core choice stands: **an in-process tool API driven by an agent loop, with
reset, step and scoring owned by the harness**. It is the same split OpenEnv
makes between agent tools and the control plane. The following changed or
was confirmed.

| Decision | Change | Reason |
|---|---|---|
| Canonical tool schema | unchanged: one `TOOL_SPECS` (name, description, JSON-schema `input_schema`) | single source; provider formats are pure conversions |
| Provider format | **added `openai_tools()`** (OpenAI function-calling) | the LLM driver uses DeepSeek's OpenAI-compatible API (user decision 2026-11-12); a test asserts the conversion is lossless |
| Episode end semantics | **added `terminated` / `truncated`** to the outcome (submit/stop = terminated, budget = truncated, agent/infrastructure error = neither) | Gymnasium semantics; pipelines need it for bootstrapping; "neither" feeds the unscorable case below |
| Reward for infrastructure failures | **planned: `None` (unscorable), distinct from 0** (step 7) | OpenEnv/TRL practice: an API outage says nothing about the policy |
| Reward structure | **planned: gates, then a weighted sum, range [0, 1], named components stored** (step 7) | OpenEnv rubrics and Harbor `reward.json`: inspectable, versioned, lets pipelines choose objectives |
| Episode length of harness envs | **`max_episode_steps` required when the env spec has none**; the evaluator passes it to `gym.make` | CliffWalking-v1 has no TimeLimit in Gymnasium 1.4.0 |
| Gymnasium-style agent API | **not the core**; optional adapter later (step 11) | Gymnasium has no LLM convention; JSON-schema tool calls and one delayed reward don't map onto numeric spaces |
| Transport (HTTP/WebSocket/MCP) | **not adopted** | local, offline, no services; an MCP or HTTP adapter over `TOOL_SPECS` is mechanical if a remote agent ever needs it |
| Dependency on `openenv` / `harbor` | **not adopted** | both are pre-1.0 with fast release cadences; we copy ideas, not code, to keep the environment pinned and reproducible |
