"""OpenEnv-style facade over the episode lifecycle (step 10a, ADR-0012).

    env = RLTaskEnv(task, root="runs")
    obs = env.reset()                                    # start an episode: instruction, task, tools, budget
    obs = env.step(ToolAction("write_file", {...}))      # one tool call -> its result
    ...                                                  # the final observation has done=True and the reward
    env.state                                            # episode id, step count, done, end reason
    env.close()                                          # end the episode if still open, verify, write

It is a thin adapter. The same `EpisodeSession`/`Episode` lifecycle runs underneath
as in `run_episode`, so a client driving the facade gets the same trajectory records
and the same reward as the driver loop (tested).

Semantics:
- The reward appears only on the final observation. `reward is None` with
  `metadata["reward_status"] == "unscorable"` means the episode could not be
  scored (e.g. a harness/infrastructure error). Intermediate observations have
  `reward_status == "pending"`.
- The client is the agent: it sees what an AgentDriver sees and nothing else
  (no seeds, no salt, no verification internals before the end). Scoring
  happens only inside the harness when the episode ends.
- An episode ends on `submit`, on an exhausted budget (checked after every action,
  exactly like the driver loop), on a harness error, or on `close()` before
  either of those ("agent_stopped", like a driver returning None).

Not borrowed from OpenEnv (survey §2): no HTTP/WebSocket server, no Docker image,
no dependency on the `openenv` package. Names follow its concepts (Action,
Observation with done/reward, State with episode_id/step_count). Not thread-safe:
one episode at a time per instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rlenv.agents.base import ToolCall
from rlenv.harness import EpisodeRecord, EpisodeSession
from rlenv.task import Task
from rlenv.trajectory import Redactor

CLIENT_KIND = "openenv_client"


@dataclass(frozen=True)
class ToolAction:
    """One tool call: a name from TOOL_SPECS and its JSON arguments."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvObservation:
    done: bool
    reward: float | None          # only on the final observation; None = unscorable (or pending)
    content: dict[str, Any]       # reset: instruction/task/tools/budget; step: the tool result
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvState:
    episode_id: str | None
    step_count: int
    done: bool
    end_reason: str | None = None


class RLTaskEnv:
    def __init__(self, task: Task, root: str | Path, redactor: Redactor | None = None) -> None:
        self.task, self.root, self.redactor = task, Path(root), redactor
        self._session: EpisodeSession | None = None
        self._done = False
        self._closed = False
        self.record: EpisodeRecord | None = None  # the last finished episode

    # -- OpenEnv-style API -------------------------------------------------

    def reset(self, agent_info: dict[str, Any] | None = None) -> EnvObservation:
        """Start a new episode. `agent_info` (model id, config...) goes into the trajectory
        header, like AgentDriver.describe()."""
        if self._closed:
            raise RuntimeError("environment is closed")
        if self._session is not None and not self._done:
            raise RuntimeError("an episode is in progress; finish it (submit) or call close() first")
        if agent_info is not None and not isinstance(agent_info, dict):
            raise TypeError("agent_info must be a dict")
        info = dict(agent_info) if agent_info else {"kind": CLIENT_KIND}
        self._session = EpisodeSession(self.task, None, info, self.root, self.redactor)
        self._done, self.record = False, None
        obs = self._session.episode.begin(info)
        content = {"instruction": obs.instruction, "task": obs.task, "tools": obs.tools, "budget": obs.budget}
        end = self._session.episode.check_budget()
        if end:  # a budget that is exhausted before the first action (driver loop: same check)
            return self._finish(end, content)
        return EnvObservation(False, None, content, self._pending_meta())

    def step(self, action: ToolAction) -> EnvObservation:
        """Execute one tool call. Unknown tools or bad arguments are tool errors (the episode
        continues), exactly as in the driver loop; only misuse of the facade raises."""
        self._require_active()
        if not isinstance(action, ToolAction):
            raise TypeError(f"step() takes a ToolAction, got {type(action).__name__}")
        if not isinstance(action.tool_name, str) or not isinstance(action.arguments, dict):
            raise TypeError("ToolAction needs a str tool_name and a dict of arguments")
        episode = self._session.episode
        try:
            result = episode.act(ToolCall(action.tool_name, dict(action.arguments)))
        except Exception as exc:  # harness-side crash: the episode ends, reward is unscorable
            return self._finish("agent_error", {"ok": False, "error": "internal error; episode ended"},
                                agent_error=episode.fail(exc))
        if episode.tools.submitted:
            return self._finish("submitted", result)
        end = episode.check_budget()
        if end:
            return self._finish(end, result)
        return EnvObservation(False, None, result, self._pending_meta())

    @property
    def state(self) -> EnvState:
        if self._session is None:
            return EnvState(None, 0, False)
        ep = self._session.episode
        return EnvState(ep.id, ep.step_count, self._done, ep.outcome.end_reason if ep.outcome else None)

    def close(self) -> None:
        """End the current episode if it is still open ("agent_stopped"), verify the workspace,
        write the trajectory. Idempotent; the environment cannot be reset afterwards."""
        if self._closed:
            return
        try:
            if self._session is not None and not self._done:
                self._finish("agent_stopped", {})
        finally:
            self._closed = True

    def __enter__(self) -> RLTaskEnv:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- internals ---------------------------------------------------------

    def _require_active(self) -> None:
        if self._closed:
            raise RuntimeError("environment is closed")
        if self._session is None:
            raise RuntimeError("call reset() before step()")
        if self._done:
            raise RuntimeError("episode is done; call reset() to start a new one")

    def _pending_meta(self) -> dict[str, Any]:
        ep = self._session.episode
        return {"episode_id": ep.id, "step": ep.step_count, "reward_status": "pending"}

    def _finish(self, end_reason: str, content: dict[str, Any], agent_error: str | None = None) -> EnvObservation:
        self._done = True  # even if verification raises, this episode cannot be stepped again
        outcome = self._session.episode.end(end_reason, agent_error)
        self.record = record = self._session.finish(outcome)
        reward = record.reward
        meta = {"episode_id": outcome.episode_id, "step": self._session.episode.step_count, "end_reason": end_reason,
                "terminated": outcome.terminated, "truncated": outcome.truncated,
                "reward_status": reward.status, "success": reward.success, "components": reward.components,
                "failure_kind": reward.failure_kind, "reason": reward.reason, "reward_version": reward.version,
                "trajectory_dir": str(record.trajectory_dir)}
        return EnvObservation(True, reward.reward, content, meta)
