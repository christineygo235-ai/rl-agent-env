"""Replay agent: re-issues the tool calls of a recorded trajectory.

Uses: regression tests (the same calls must reach the same reward), debugging
(re-run a model's episode without the model), and re-running old episodes
after a harness change. Results are ignored: the replay does not adapt, so
it shows exactly what those calls produce *now*.

Safety: the source trajectory must pass its integrity checks, and by default
it must have been recorded against the same task spec (public hash).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rlenv.agents.base import AgentDriver, Observation, ToolCall
from rlenv.task import _sha256_json
from rlenv.trajectory import TrajectoryError, load_trajectory


class ReplayAgent(AgentDriver):
    kind = "replay"

    def __init__(self, trajectory_dir: str | Path, allow_task_change: bool = False) -> None:
        traj = load_trajectory(trajectory_dir)
        if not traj.ok:
            raise TrajectoryError(f"refusing to replay a trajectory that fails integrity checks: {traj.problems[:3]}")
        self.source = str(trajectory_dir)
        self.source_episode = traj.header["episode_id"]
        self.source_task_sha = traj.header["task_public_sha256"]
        self.allow_task_change = allow_task_change
        self.calls = [ToolCall(r["tool"], traj.inline(r.get("args", {}))) for r in traj.of_type("tool_call")]
        self._queue: list[ToolCall] = []

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "source_episode": self.source_episode, "calls": len(self.calls)}

    def start(self, observation: Observation) -> None:
        # Same hash as Task.public_sha256(): the observation carries the agent view of the task.
        if not self.allow_task_change and _sha256_json(observation.task) != self.source_task_sha:
            raise TrajectoryError("the task spec changed since this trajectory was recorded "
                                  "(pass allow_task_change=True to replay anyway)")
        self._queue = list(self.calls)

    def next_action(self, last_result: dict[str, Any] | None) -> ToolCall | None:
        return self._queue.pop(0) if self._queue else None
