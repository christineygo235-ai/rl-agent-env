# ADR-0003: Harness-owned evaluation with a sandboxed policy over a pipe

Status: accepted (step 3). Implements PLAN D2 option C.

## Context
The reward depends on the evaluated return, so whoever computes that number
controls the reward. Agent code is untrusted and may try to inflate it:
- print fake results;
- patch Gymnasium (rewards, termination);
- find and overfit to the evaluation seeds;
- return out-of-range actions;
- hang or crash to dodge a bad score.

## Decision

### Ownership
- The **harness** process calls `gymnasium.make`, chooses the seeds, calls
  `reset`/`step`, validates actions, and **sums the rewards** itself
  (`rlenv/evaluator.py: evaluate`).
- The agent's `policy.py` runs in a **separate sandboxed process**
  (`rlenv/policy_server.py`, started with `SubprocessSandbox.spawn_python`:
  same scrubbed env, rlimits and process group as training runs). It only ever
  receives observations and returns actions.

### Protocol
- Line-delimited JSON: `{"ready"}` handshake after `load()`, `reset`, `act`, `close`.
- Arrays are encoded as `{t:"a", v, dtype, shape}`.
- The server duplicates the original stdout fd for the protocol, then points
  fd 1 at stderr. Every `print` from agent code (at import or in `act`) lands
  in stderr and can never be parsed as a protocol message. A test counts the
  forged lines: exactly one per step went to stderr.
- The harness reads the raw fd with its own line buffer and a per-message
  deadline (`select`). Using `select` on a buffered reader can miss data
  already buffered and report spurious timeouts.

### Validation and failure semantics
- Every action must satisfy `action_space.contains` (bools, floats that aren't
  integers, strings, NaN and out-of-range values are rejected).
- Per-step timeout (default 5 s), load timeout (30 s), total evaluation
  deadline, message-size cap, RSS check every 500 calls, plus a hard step cap
  beyond the env's own `TimeLimit`.
- Any failure (crash, hang, invalid action, exit) yields an `EvalResult` with
  `error` and `mean = NaN`. **No partial mean is ever produced**, so a policy
  cannot dodge bad episodes by crashing.

### Seeds
- `eval_seeds(task_seed, salt, n, dev_n)`: sha256-derived from a salt the agent
  never sees. They are disjoint from the dev seeds (used by the agent's `evaluate`
  tool) and from the training seed.
- Trajectories record only `seeds_sha256`, not the seeds.

### Baseline
`random_baseline` uses a seeded `action_space.sample()` on the same seeds. It
is the `R_random` used to normalise progress in the reward.

## Evidence (tests)
- A tampering `policy.py` patches `gymnasium.Env.step` and `TimeLimit.step`
  (reward 1000) and prints forged protocol lines and a fake "mean_return:
  1000". Its returns are **identical** to the honest policy's.
- Sandboxed and in-process evaluations of the same policy give identical
  returns (FrozenLake discrete obs; CartPole float32 arrays).
- Seven failure modes each produce a clear error: load crash, missing `act`,
  hang, out-of-space action, wrong type, `os._exit`, syntax error.
- The policy process is gone after evaluation.

## Cost
Measured on the dev machine: about 12k steps/s sandboxed vs about 114k steps/s
in-process. 100 CartPole episodes (≤ 50k steps) take about 4 s. Fine for classic
control; batching observations would be the fix for larger evaluations.

## Alternatives considered
| Option | Why not |
|---|---|
| Agent's evaluation script reports the return | trivially forgeable |
| Import `policy.py` into the harness | agent code shares the process: it could patch gym, the evaluator or the reward |
| Ship the Q-table only, harness implements `act` | robust, but forbids any policy beyond argmax-over-table (tile coding, DQN); `policy.py` keeps the contract general |
| gRPC / sockets | more moving parts; network-capable code is exactly what we forbid |

## Consequences
- Determinism depends on the policy being deterministic. A stochastic `act`
  is allowed but its variance shows up in the score (the instruction asks for
  greedy policies).
- Evaluation needs `policy.py` to pass the static checks first; the verifier
  enforces that order (step 7).
