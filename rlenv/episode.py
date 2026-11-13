"""The episode: instruction -> tool calls -> submit (or budget exhausted).

Every step emits events to `sink` (the append-only trajectory). Verification
and reward are applied afterwards (rlenv/harness.py): this module only decides
when the agent's turn is over and why.

The lifecycle is explicit so that two front-ends share one implementation:
- `run()`: the driver loop, the episode *pulls* actions from an AgentDriver;
- the OpenEnv-style facade (rlenv/openenv_facade.py): a client *pushes*
  actions with `step()`.
Both call `begin` -> (`check_budget` -> `act`)* -> `end`, in the same order.
"""

from __future__ import annotations

import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from rlenv.agents.base import AgentDriver, Observation, ToolCall
from rlenv.task import Task
from rlenv.tools import TOOL_SPECS, Budget, RunRecord, ToolBox
from rlenv.workspace import Workspace

Event = dict[str, Any]
Sink = Callable[[Event], None]


@dataclass
class EpisodeOutcome:
    episode_id: str
    task_id: str
    end_reason: str  # submitted | budget:<kind> | agent_stopped | agent_error
    tool_calls: int
    runs: list[RunRecord]
    workspace: Path
    duration_s: float
    submit_notes: str = ""
    agent_error: str | None = None
    events: list[Event] = field(default_factory=list)

    @property
    def terminated(self) -> bool:
        """Gymnasium semantics: the agent ended the episode itself (submit or deliberate stop)."""
        return self.end_reason in ("submitted", "agent_stopped")

    @property
    def truncated(self) -> bool:
        """Gymnasium semantics: an outside limit ended it (budget), not the agent."""
        return self.end_reason.startswith("budget:")

    def summary(self) -> dict:
        return {"episode_id": self.episode_id, "task_id": self.task_id, "end_reason": self.end_reason,
                "terminated": self.terminated, "truncated": self.truncated,
                "tool_calls": self.tool_calls, "runs": len(self.runs), "duration_s": round(self.duration_s, 2)}


def new_episode_id(task: Task) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{task.id}-{stamp}-{uuid.uuid4().hex[:6]}"


class Episode:
    def __init__(self, task: Task, agent: AgentDriver | None, root: Path, episode_id: str | None = None,
                 sink: Sink | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        self.task = task
        self.agent = agent
        self.id = episode_id or new_episode_id(task)
        self.root = Path(root) / self.id
        self.workspace = Workspace(self.root / "workspace")
        self.budget = Budget(task, clock)
        self.tools = ToolBox(task, self.workspace, self.root / "logs", self.budget)
        self.events: list[Event] = []
        self._sink = sink
        self._t0 = clock()
        self._clock = clock
        self._steps = 0
        self.outcome: EpisodeOutcome | None = None

    @property
    def step_count(self) -> int:
        return self._steps

    def emit(self, event: Event) -> None:
        event = {"t": round(self._clock() - self._t0, 3), **event}
        self.events.append(event)
        if self._sink:
            self._sink(event)

    # -- lifecycle (shared by the driver loop and the facade) -----------------

    def begin(self, agent_info: dict[str, Any]) -> Observation:
        """Log the start and return what the agent sees first."""
        self.emit({"type": "start", "episode_id": self.id, "agent": agent_info})
        return Observation(instruction=self.task.instruction, task=self.task.agent_view(), tools=TOOL_SPECS,
                           budget=self.budget.remaining())

    def check_budget(self) -> str | None:
        """`budget:<kind>` if a budget is exhausted (checked before every action)."""
        exhausted = self.budget.exhausted()
        return f"budget:{exhausted}" if exhausted else None

    def act(self, call: ToolCall) -> dict[str, Any]:
        """Execute one tool call through the harness and log it. Returns the tool result
        (JSON). After `submit`, `self.tools.submitted` is set and the caller ends the episode."""
        self._steps += 1
        i = self._steps
        self.emit({"type": "tool_call", "i": i, "tool": call.name, "args": call.args})
        result = self.tools.dispatch(call.name, call.args)
        last = result.to_json()
        self.emit({"type": "tool_result", "i": i, **last})
        if call.name == "run_training" and result.data.get("run_id") in self.tools.runs:
            self.emit({"type": "run", **self.tools.runs[result.data["run_id"]].to_json()})
        return last

    def fail(self, exc: BaseException) -> str:
        """Log a crash of the acting side (agent or client); returns the error text."""
        agent_error = f"{type(exc).__name__}: {exc}"
        self.emit({"type": "agent_error", "error": agent_error, "traceback": traceback.format_exc()[-4000:]})
        return agent_error

    def end(self, end_reason: str, agent_error: str | None = None) -> EpisodeOutcome:
        if self.outcome is not None:
            raise RuntimeError(f"episode {self.id} already ended ({self.outcome.end_reason})")
        self.outcome = EpisodeOutcome(
            episode_id=self.id, task_id=self.task.id, end_reason=end_reason, tool_calls=self.budget.tool_calls,
            runs=list(self.tools.runs.values()), workspace=self.workspace.root,
            duration_s=self._clock() - self._t0, submit_notes=self.tools.submit_notes, agent_error=agent_error,
            events=self.events)
        self.emit({"type": "end", **self.outcome.summary()})
        return self.outcome

    # -- the driver loop ---------------------------------------------------------

    def run(self) -> EpisodeOutcome:
        obs = self.begin(self.agent.describe())
        self.agent.bind_emitter(self.emit)
        end_reason, agent_error = None, None
        last: dict | None = None
        try:
            self.agent.start(obs)
            while end_reason is None:
                end_reason = self.check_budget()
                if end_reason:
                    break
                call = self.agent.next_action(last)
                if call is None:
                    end_reason = "agent_stopped"
                    break
                last = self.act(call)
                if self.tools.submitted:
                    end_reason = "submitted"
        except Exception as exc:  # a crashing agent ends its own episode, not the harness
            end_reason, agent_error = "agent_error", self.fail(exc)
        return self.end(end_reason, agent_error)


def outcome_to_json(outcome: EpisodeOutcome) -> dict:
    data = asdict(outcome)
    data["workspace"] = str(outcome.workspace)
    data["terminated"], data["truncated"] = outcome.terminated, outcome.truncated
    data.pop("events")
    return data
