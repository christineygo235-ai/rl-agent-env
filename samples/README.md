# Sample trajectories (real DeepSeek episodes)

Both were recorded on 2026-11-12 with `rlenv run frozenlake-tabular --agent llm
--allow-external-api --model deepseek-flash`, after the user approved the calls.
They are committed verbatim: trajectory, head file and blobs. Anyone can check
them offline:

```bash
.venv/bin/rlenv check-trajectory samples/deepseek-flash-frozenlake   # hash chain, head, blobs
.venv/bin/rlenv rescore samples/deepseek-flash-frozenlake            # recompute the reward
.venv/bin/rlenv run frozenlake-tabular --agent replay --replay samples/deepseek-flash-frozenlake  # re-issue the calls
```

| Directory | What happened | Reward |
|---|---|---|
| `deepseek-flash-frozenlake/` | 17 model calls, 29 tool calls, 10 training runs. Expected SARSA with 3 independent learners trained one after another and the best table selected by rollouts; verified on hidden random maps; the report numbers match the harness logs | **1.0** |
| `deepseek-flash-unscorable/` | the first attempt: 5 tool calls, then a harness bug (sandbox + relative `--out`) ended the episode | **None** (`agent_driver_error`): an infrastructure fault is not scored as 0 |

Contents of `deepseek-flash-frozenlake/`:
- `trajectory.jsonl`: every event, including each model call (`llm_call`: the new
  messages, the full response with `reasoning_content`, token usage and latency),
  every tool call and result, the verification record and the reward;
- `final/`: train.py, policy.py and REPORT.md as submitted (also in `blobs/`,
  content-addressed);
- `export/messages.jsonl`: OpenAI chat format with the exact transcript and a
  loss mask;
- `export/nemo-gym.jsonl`: NeMo Gym rollout row;
- `export/atif/`: Harbor ATIF v1.8 document.

The API key never appears in these files; tests grep for it. These two samples
predate home-path scrubbing (ADR-0006 addendum), so they keep absolute local
paths: the workspace path, and in the unscorable one also the error traceback.
Editing them would break the hash chain.

Numbers and per-map results: [docs/EVIDENCE.md §4](../docs/EVIDENCE.md).
