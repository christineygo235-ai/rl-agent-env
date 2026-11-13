# Architecture decision records

| ADR | Decision | Step |
|---|---|---|
| [0001](0001-task-spec.md) | Task spec: YAML validated by strict pydantic models; versioned; agent view excludes secrets (eval salt) | 1 |
| [0002](0002-workspace-sandbox-static-checks.md) | Workspace confinement, subprocess sandbox (rlimits, scrubbed env, process-group kill), static AST checks; no OS-level network isolation (container backend is the upgrade path) | 2 |
| [0003](0003-harness-owned-evaluation.md) | Harness-owned evaluation: the env and the return sum live in the harness; the policy runs sandboxed behind a line-JSON pipe | 3 |
| [0004](0004-reference-solutions.md) | Reference solutions prove each task solvable through the real pipeline | 4 |
| [0005](0005-tools-episode-loop.md) | One tool schema (`TOOL_SPECS`) for harness and drivers; episode loop; budgets | 5 |
| [0006](0006-trajectories.md) | Trajectories: append-only hash-chained JSONL, content-addressed blobs, redaction at the source | 6 |
| [0007](0007-reward.md) | Reward: unscorable → hard gates → 0.6/0.3/0.1, with the ordering guarantee ([spec](../reward.md)) | 7 |
| [0008](0008-export-formats.md) | Derived exports: `messages`, `nemo-gym`, `atif`; unscorable excluded by default | 7, 11 |
| [0009](0009-red-team.md) | Red team with declared expectations; the "no learning" gap closed by C (hidden env instances) + D (audit flags) | 8 |
| [0010](0010-calibration-and-variants.md) | Calibration admits tasks; train.py contract; held-out env instances | 9 |
| [0011](0011-deepseek-driver.md) | DeepSeek driver (9a findings), transcript in the trajectory, billable-token budget | 9a, 10 |
| [0012](0012-openenv-facade.md) | OpenEnv-style facade over one shared episode lifecycle | 10a |

Reading order for a reviewer: 0003 → 0007 → 0009 → 0010 (why the reward is hard to
game), then 0006/0008 (data for training pipelines), then 0011/0012 (integration).
