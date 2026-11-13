# ADR-0006: Trajectories as append-only, hash-chained JSONL with content-addressed blobs

Status: accepted (step 6)

## Context
Trajectories are the product for the training pipeline (README "Users and
usage"). They must be:
- streamable;
- self-describing;
- safe to share (no secrets);
- verifiable (no silent corruption, truncation or editing);
- complete enough to recompute the reward;
- valid even if an episode or the harness crashes midway.

## Decision

### Layout
```
runs/<episode_id>/
  trajectory.jsonl     # one record per line, written while the episode runs
  trajectory.head      # "<n_records> <sha256 of last line>", written on close
  blobs/<sha256>       # code versions and large strings (content-addressed)
  workspace/           # the agent's workspace (not part of the trajectory format)
  logs/                # sandbox stdout/stderr per run
```

### Records
- `header`:
  - schema `rlenv.trajectory/v1`, episode id, created;
  - **agent view of the task** (no eval salt) and task id/version;
  - `task_public_sha256` and `task_full_sha256`;
  - agent description;
  - runtime (rlenv, python, gymnasium, numpy, platform, git commit).
- Episode events, in order:
  - `start`;
  - `tool_call` / `tool_result`;
  - `run` (code snapshot, exit, duration, dev evals);
  - `agent_error`;
  - `end`.
- Final records:
  - `outcome`: end reason, `terminated`/`truncated`, runs;
  - from step 7, `verification` and `final` (reward + breakdown).

### Integrity
- Every record has `seq` and `prev` = sha256 of the previous line (genesis = 64
  zeros). The reader reports the first broken link with its line number.
- `trajectory.head` pins the record count and the last line's hash, so
  truncation and edits to the last line are caught too.
- Blobs are verified against their names on read; missing or altered blobs are reported.
- The reader **collects problems instead of raising**, so a pipeline can
  quarantine bad files and keep streaming. `rlenv check-trajectory` returns exit 2 for invalid files.
- The writer refuses to reuse an existing file (`open(..., "x")`): one trajectory
  per episode, never appended to later.

This is tamper-*evident*, not tamper-*proof*: someone who can rewrite the whole
file can recompute the chain. Signing the head (e.g. HMAC with a harness key)
would close that gap; it is not needed while trajectories stay on the harness host.

### Code versions
- `write_file` content goes to `blobs/` and the event keeps only `{blob, bytes}`.
- Each `run` stores the exact bytes of every file it executed (from the
  workspace at run time), with `matches` = the sha recorded by the tool.
- Strings over 8 KB (long stdout tails, file reads) become blob references,
  which keeps lines small for streaming.

### Secrets
Before any line or blob is written, a `Redactor` replaces:
- values of env vars named like secrets (`DEEPSEEK_API_KEY`, `*_TOKEN`,
  `*_SECRET`, …, at least 8 characters);
- `sk-…` style keys.

Replacements look like `[REDACTED:<NAME>]`. Tests set `DEEPSEEK_API_KEY` and
assert its value appears nowhere in the episode directory. The sandbox never
receives the key in the first place (scrubbed env).

### Crash behaviour
Records are written and flushed as the episode runs, so a crashing agent still
leaves a valid trajectory ending in `outcome: agent_error`. `harness.run_episode`
closes the writer in `finally`.

## Alternatives considered
| Option | Why not |
|---|---|
| One JSON document per episode (e.g. Harbor ATIF) | not appendable during the run; a crash leaves nothing; no chain. Planned as an **export** instead (step 11) |
| SQLite per episode / central DB | not streamable with standard tools; harder to ship to a training pipeline; more moving parts |
| Inline code in events | large and duplicated (the same file in `write_file` and in each `run`); blobs deduplicate |
| Logging raw secrets and redacting downstream | one missed consumer leaks the key; redaction must happen at the source |
| Signed records | worth it once trajectories cross trust boundaries; not needed for a local harness |

## Addendum (step 11): local paths are scrubbed at the source
Trajectories are shared as evidence (`evidence/`, `samples/`), so the `Redactor`
also replaces the user's home directory with `~`. This happens when a record is
written, before it is hashed, so the chain stays valid. The harness never reads
paths back from a trajectory. `samples/deepseek-flash-*` were recorded before
this change and keep absolute local paths (workspace, and a traceback in the
unscorable one): rewriting them would break the
hash chain they are evidence for.
