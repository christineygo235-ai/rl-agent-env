# ADR-0001: Task spec format and validation

Status: accepted (step 1)

## Context
Every episode starts from a task. The task has to tell the agent what to do
and tell the harness how to verify it. Verification depends on details the
agent must not see (held-out seeds). Tasks must be reviewable, versioned and
impossible to misconfigure silently.

## Decision
- One YAML file per task in `tasks/<id>.yaml`; the `id` must equal the file name.
- Fields:
  - `instruction`, `env_id`, `env_kwargs` (scalars only), `seed`;
  - `success` (metric `mean_return`, `threshold`, `eval_episodes`,
    `dev_eval_episodes`, `eval_seed_salt`);
  - `budget` (`wall_clock_s`, `train_timeout_s`, `max_tool_calls`, `max_runs`);
  - `methods` (`allowed` from a closed set of Sutton & Barto families, `allow_torch`);
  - `limits` (`dqn_max_params`, `memory_mb`);
  - `version`.
- Validated by a pydantic model (`rlenv/task.py`):
  - unknown keys are errors, and models are frozen;
  - `env_id` must be on the classic allowlist **and** registered in the
    pinned Gymnasium (1.4.0 registers `Taxi-v4` and `CliffWalking-v1`, not
    `-v3`/`-v0`; the check caught this while planning);
  - `train_timeout_s ≤ wall_clock_s`;
  - `allow_torch` requires `small_dqn`.
- **Two views of a task**:
  - `agent_view()` omits `eval_seed_salt`;
  - `public_sha256()` hashes the agent view and goes into trajectory headers;
  - `full_sha256()` (including the salt) identifies the exact verification setup.
- `version` is bumped whenever the threshold, budget or seeds change, so old
  scores stay comparable within a version.

## Alternatives considered
| Option | Why not |
|---|---|
| JSON | Less readable for multi-line instructions; YAML is safe-loaded anyway |
| Python task classes | Not data; harder to diff and review; risks code in specs |
| Salt in a separate secrets file | More moving parts; the agent never reads `tasks/` anyway (workspace confinement, step 2). Revisit if tasks are shared with agents |

## Consequences
- Thresholds are drafts until calibration (step 9).
- Adding an env requires touching the allowlist, a deliberate review point.
