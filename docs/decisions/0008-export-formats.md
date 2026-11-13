# ADR-0008: Trajectory review and derived export formats

Status: accepted (step 7). Follows the review of miles, verl and NeMo Gym in
[reference-survey.md §4](../reference-survey.md). Amends
[ADR-0006](0006-trajectories.md).

## Context
We picked three RL training stacks (miles, verl, NeMo Gym) as
references for trajectory formats. Our trajectories feed training pipelines, so
the formats must fit how those stacks consume rollouts, without giving up what
our native format guarantees: integrity, self-description, no secrets, and
recomputable rewards.

## Findings that drive the decision
- All three **separate the trainer's token-level data from the environment's
  record**. Token ids, loss masks and log-probs are produced inside the trainer
  with its tokenizer and model.
- All three **group** samples for group-relative advantages: verl `uid`,
  miles `group_index`, NeMo Gym `_ng_task_index`.
- **None of them distinguishes "unscorable" cleanly in its core type**:
  - verl turns scoring timeouts into 0.0;
  - miles filters `None` out of training but treats it as 0 in eval metrics;
  - NeMo Gym uses `reward 0.0` plus `mask_sample: true` and a `failure_kind` vocabulary.
- **None** has per-event hash chaining or content-addressed code.
- Consumers read OpenAI-style messages (miles `metadata.messages`, verl agent
  loop, TRL, OpenEnv `collect`) or NeMo Gym's Responses-API rollout rows.

## Decision

### Native format: unchanged in structure, three additions
It stays event-level, append-only, hash-chained JSONL plus content-addressed blobs
(ADR-0006). Additions:
- `final.group_id = "<task_id>@v<version>:<public_sha256[:12]>"`: same task version
  means same group, and a silent spec change shows up as a new group.
- `final.failure_kind` from a fixed vocabulary (ADR-0007): pipelines filter
  without parsing prose.
- `header.runtime.git_dirty`: the commit alone doesn't describe the code if the
  tree had local edits (miles records a hashed diff for the same reason).

We do **not** store tokens in our own format. We are the environment, DeepSeek's API
doesn't return token ids, and masks depend on the trainer's tokenizer.

### Exports are derived, one row per episode (`rlenv export`)
| | `messages` (default) | `nemo-gym` |
|---|---|---|
| Conversation | OpenAI chat: `user` prompt, `assistant` with `tool_calls`, `tool` results | Responses API `output[]`: `function_call` / `function_call_output` |
| Tools | `openai_tools()` | `responses_create_params.tools` |
| Training mask | message-level `loss_mask` (assistant = 1, tool = 0) | (trainer derives from item types) |
| Reward | `reward` (float or null), `reward_components` | `reward` (float), `reward_components` + `reward_<name>` at top level |
| Unscorable | excluded by default; with `--include-unscorable`: `reward: null` | excluded by default; with the flag: `reward: 0.0, mask_sample: true` (NeMo convention) |
| Failure | `status`, `failure_kind`, `failure_reason` | `failure_kind`, `failure_reason`, `mask_sample` |
| Grouping | `group_id` | `_ng_task_index` (per group), `_ng_rollout_index` (within group) |
| Provenance | `source{episode_id, trajectory_head_sha256, schema, reward_version, export_version, agent, runtime}` | the same, under `rlenv.source` |

Rules for both formats:
1. **Integrity first.** A trajectory that fails the chain, head or blob checks, or
   has no head, is skipped and reported. Incomplete ones (no `final`) are skipped.
   Nothing unverified reaches training data. The CLI exits 2 if anything was invalid.
2. **Unscorable is filtered by default**, not scored as 0 (miles/verl practice;
   ADR-0007 §5).
3. **Self-contained rows.** `write_file` contents and other blob references are
   re-inlined. The code the model wrote is the most important part of the sample.
4. **Traceable rows.** The source head hash ties each row to an intact trajectory,
   which `rlenv check-trajectory` / `rescore` can re-verify.
5. Secrets cannot appear: exports are built from already-redacted trajectories
   (tested with `DEEPSEEK_API_KEY` set).

## Alternatives considered
| Option | Why not |
|---|---|
| Make a trainer format native (miles `Sample`, verl `DataProto`) | token-level and trainer-specific; loses the event record, the integrity chain and the verification facts |
| NeMo Gym's reward 0.0 + mask as the native unscorable | a 0.0 can be averaged by accident; `None` can't. Kept only in the NeMo export, where consumers expect it |
| verl parquet schema as an export | it's a prompt dataset (`ground_truth`), not a rollout; trivial to derive from `messages` if a pipeline wants it |
| Include unscorable rows by default | pipelines that ignore the status would train on noise |
| ATIF now | still planned (step 11): one JSON document per trial, useful for Harbor/NeMo viewers but not for streaming training |

## Consequences
- Both exports can be produced from the same trajectories at any time. Adding
  ATIF or a parquet view later is a pure function over verified trajectories.
- Until step 10 records the LLM's exact prompt, exports reconstruct the user
  prompt from the instruction and the agent view of the task.

## Addendum (step 11): ATIF export
`rlenv export --format atif -o DIR` writes one Harbor ATIF v1.8 document per episode
(`rlenv/atif.py`):
- LLM episodes map from the exact transcript. Each model call becomes an `agent`
  step with reasoning, tool calls, token metrics and the harness's tool results as
  `observation.results`.
- Other episodes become one agent step per tool call, with `llm_call_count: 0`.
- The reward has no ATIF field, so it goes to `extra.rlenv`.

`validate_atif` implements the RFC 0001 rules we rely on:
- field names and required fields;
- sequential step ids;
- `source` restrictions;
- tool-call references.

It is tested to fail on each violation. Harbor's own validator is not a
dependency. ATIF stays an export, never the native format (ADR-0006).
