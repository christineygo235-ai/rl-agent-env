"""Agent driver interface.

The episode loop calls `start` once, then `next_action` repeatedly with the
result of the previous tool call. Returning None means the agent stops
without calling `submit` (the harness still verifies whatever is in the workspace).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    """What the agent sees at the start: the instruction, the agent view of the task, the tools."""

    instruction: str
    task: dict[str, Any]
    tools: list[dict[str, Any]]
    budget: dict[str, Any]


class AgentDriver(ABC):
    kind = "abstract"
    _emit = None

    def describe(self) -> dict[str, Any]:
        """Logged in the trajectory header (model id, version, config...)."""
        return {"kind": self.kind}

    def bind_emitter(self, emit) -> None:
        """The episode passes its event emitter, so drivers can log their own records
        (e.g. the LLM transcript) into the trajectory. Default: drivers emit nothing."""
        self._emit = emit

    def emit(self, event: dict[str, Any]) -> None:
        if self._emit is not None:
            self._emit(event)

    @abstractmethod
    def start(self, observation: Observation) -> None: ...

    @abstractmethod
    def next_action(self, last_result: dict[str, Any] | None) -> ToolCall | None: ...
