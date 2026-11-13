# ADR-0012: OpenEnv-style facade (step 10a)

Status: accepted (2026-11-12).

## Context
RL training stacks drive environments with a push API: `reset()` → observation;
`step(action)` → observation with `done` and `reward`; `state`; `close()`. That is
the Gymnasium shape, and OpenEnv's typed variant (survey §2). Our episode loop is a
pull API: `Episode.run()` asks an `AgentDriver` for the next action. A trainer that
owns its own rollout loop (TRL/OpenEnv style, a vLLM sampler, a remote agent) needs
the push shape. It must not get a second, subtly different harness.

## Decision
1. **One lifecycle, two front-ends.** `Episode` exposes its lifecycle
   explicitly: `begin` → (`check_budget` → `act`)\* → `end`, plus `fail`.
   `harness.EpisodeSession` owns the trajectory (open → `finish`: outcome,
   verification, reward, final record).
   - `run_episode` (driver loop) is now just `EpisodeSession` + `Episode.run()`.
   - The facade (`rlenv/openenv_facade.py`, `RLTaskEnv`) calls the same methods.
     It has no logic of its own beyond translating calls and rejecting misuse.
2. **API** (OpenEnv concepts, our own small dataclasses):

   | Call | Maps to | Returns |
   |---|---|---|
   | `reset(agent_info=None)` | new `EpisodeSession`, `Episode.begin` | `EnvObservation(done=False, reward=None, content={instruction, task, tools, budget})`, the agent view only |
   | `step(ToolAction(tool_name, arguments))` | `Episode.act` (one tool call through the ToolBox), then `submit` / budget check | the tool result. On the last step: `done=True`, `reward`, and metadata (end reason, terminated/truncated, status, components, failure kind, trajectory dir) |
   | `state` | episode id, step count | `EnvState(episode_id, step_count, done, end_reason)` |
   | `close()` | if still open: end as `agent_stopped`, verify, write | idempotent; the instance cannot be reset afterwards |

3. **Reward semantics.**
   - The reward appears only on the final observation.
   - `reward=None` is ambiguous on its own, so `metadata["reward_status"]` says
     which case it is: `pending` (not finished), `scored`, `hard_fail`, or
     `unscorable` (`None` means an infrastructure error, ADR-0007).
   - Intermediate rewards are deliberately absent. Dense shaping would let a
     client optimise the progress term mid-episode, and the reward is only
     defined on a verified final workspace.
4. **Same end conditions as the loop.**
   - The episode ends on `submit`.
   - It also ends when a budget is exhausted. The budget is checked *after* each
     action, which equals the loop's check before the next action, so wall-clock
     accounting matches too.
   - It ends on a harness-side exception: `agent_error` → unscorable. The client
     only gets "internal error"; the traceback goes to the trajectory.
   - Finally, it ends on `close()` before either of those (`agent_stopped`,
     like a driver returning None).
   - Unknown tools and bad arguments are tool errors and the episode continues,
     as in the loop.
5. **Misuse raises** and never reaches the harness. Rejected:
   - `step` before `reset`, after `done`, or after `close`;
   - `reset` while an episode is in progress (it would silently drop a trajectory
     without a final record);
   - `reset` after `close`;
   - non-`ToolAction` actions and non-dict arguments or `agent_info`.

## Evidence (tests/test_openenv_facade.py)
- The scripted reference agent driven through the facade yields the **same reward
  JSON and the same trajectory records** as the driver loop. Only timestamps,
  episode ids and the training script's own timing output differ. `rescore`
  reproduces the facade's reward.
- Mutation checks:
  - ignoring `agent_info` makes the equivalence test fail;
  - dropping the post-action budget check makes the budget test fail.
- Further tests:
  - intermediate observations and state;
  - unknown tool = tool error;
  - close before submit (hard fail 0, trajectory complete);
  - budget truncation on the last step;
  - harness crash = unscorable;
  - reset after done starts a new episode;
  - all misuse cases;
  - the context manager.

## Not done (and why)
| Option | Why not |
|---|---|
| Depend on the `openenv` package and subclass its `Environment` | pre-1.0 API churn (survey §2); one adapter file is enough if a trainer requires its exact base classes |
| HTTP/WebSocket server, Docker image | offline/local requirement; the facade is in-process. A server would wrap `RLTaskEnv` unchanged |
| Run the driver loop in a thread and bridge with queues | two sources of truth for end conditions, plus hard-to-test concurrency. An explicit lifecycle is simpler and testable |
| Gymnasium `gym.Env` subclass with `spaces` | the action is a JSON tool call, not a tensor. `Text`/`Dict` spaces add nothing but friction |
| Auto-close the running episode on `reset` | hides client bugs and costs a full verification. An explicit error is cheaper and clearer |
