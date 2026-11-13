"""Task specs: model, validation, loading.

A task is what the agent is asked to do plus everything the harness needs
to verify it: env id, success threshold, seeds, budgets and allowed methods.
Specs live in `tasks/<id>.yaml`.

Rules enforced here:
- unknown keys are errors, models are immutable;
- the env id must be on the classic-control/toy-text allowlist AND registered
  in the installed (pinned) Gymnasium;
- budgets are consistent (a single run fits in the episode budget);
- the held-out evaluation salt is never part of what the agent sees
  (`agent_view`), nor of the trajectory header.
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

# Classic problems this environment is meant for (Sutton & Barto scale).
ALLOWED_ENVS = frozenset({
    "FrozenLake-v1",
    "FrozenLake8x8-v1",
    "Taxi-v4",
    "CliffWalking-v1",
    "CartPole-v1",
    "MountainCar-v0",
    "Acrobot-v1",
})

_ID_RE = re.compile(r"^[a-z0-9]([-a-z0-9]{0,62}[a-z0-9])?$")


class TaskError(Exception):
    """User-facing task loading/validation error."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Method(str, Enum):
    tabular_q_learning = "tabular_q_learning"
    sarsa = "sarsa"
    expected_sarsa = "expected_sarsa"
    tile_coding = "tile_coding"
    small_dqn = "small_dqn"


class Success(StrictModel):
    metric: str = Field(default="mean_return", pattern=r"^mean_return$")
    threshold: float
    eval_episodes: int = Field(default=100, ge=1, le=10_000)
    # Secret-ish: derives the held-out eval seeds. Never shown to the agent.
    eval_seed_salt: str = Field(min_length=1)
    dev_eval_episodes: int = Field(default=20, ge=1, le=1_000)


class Budget(StrictModel):
    wall_clock_s: float = Field(gt=0, le=24 * 3600)
    train_timeout_s: float = Field(gt=0)
    max_tool_calls: int = Field(ge=1, le=1000)
    max_runs: int = Field(ge=1, le=200)

    @model_validator(mode="after")
    def _consistent(self) -> Budget:
        if self.train_timeout_s > self.wall_clock_s:
            raise ValueError("train_timeout_s cannot exceed wall_clock_s")
        return self


class Methods(StrictModel):
    allowed: list[Method] = Field(min_length=1)
    allow_torch: bool = False  # numpy-only by default (PLAN D3)

    @field_validator("allowed")
    @classmethod
    def _unique(cls, v: list[Method]) -> list[Method]:
        if len(set(v)) != len(v):
            raise ValueError("duplicate methods")
        return v

    @model_validator(mode="after")
    def _torch_needs_dqn(self) -> Methods:
        if self.allow_torch and Method.small_dqn not in self.allowed:
            raise ValueError("allow_torch only makes sense when small_dqn is allowed")
        return self


class EnvVariants(StrictModel):
    """Held-out env instances (ADR-0009 option C). Final verification trains and evaluates on
    `count` hidden instances; the threshold on each is a fraction of the optimal return there."""

    kind: str
    count: int = Field(default=2, ge=1, le=10)            # hidden instances used by verification
    dev_count: int = Field(default=3, ge=0, le=10)        # public instances the agent may train on
    frozen_p: float = Field(default=0.8, gt=0.0, le=1.0)  # FrozenLake: probability a tile is frozen
    threshold_fraction_of_optimal: float = Field(default=0.9, gt=0.0, le=1.0)


class Verification(StrictModel):
    env_variants: EnvVariants | None = None


class Limits(StrictModel):
    dqn_max_params: int = Field(default=50_000, ge=100, le=1_000_000)
    memory_mb: int = Field(default=2048, ge=256, le=16_384)


class Task(StrictModel):
    id: str
    version: int = Field(ge=1)
    instruction: str = Field(min_length=20)
    env_id: str
    env_kwargs: dict[str, str | int | float | bool] = Field(default_factory=dict)
    seed: int = Field(ge=0, lt=2**31)
    success: Success
    budget: Budget
    methods: Methods
    limits: Limits = Limits()
    verification: Verification = Verification()
    # Required when the registered env has no step limit (CliffWalking-v1 in gymnasium 1.4.0):
    # without it, a policy that never reaches the goal never ends its episode.
    max_episode_steps: int | None = Field(default=None, ge=1, le=100_000)

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not _ID_RE.match(v):
            raise ValueError(f"{v!r} is not a valid task id (lowercase, digits, '-')")
        return v

    @field_validator("env_id")
    @classmethod
    def _env(cls, v: str) -> str:
        if v not in ALLOWED_ENVS:
            raise ValueError(f"{v!r} is not an allowed classic env; allowed: {sorted(ALLOWED_ENVS)}")
        import gymnasium

        if v not in gymnasium.envs.registry:
            raise ValueError(f"{v!r} is not registered in gymnasium {gymnasium.__version__}")
        return v

    @model_validator(mode="after")
    def _step_limit(self) -> Task:
        import gymnasium

        if self.max_episode_steps is None and gymnasium.spec(self.env_id).max_episode_steps is None:
            raise ValueError(f"{self.env_id} has no registered step limit; set max_episode_steps")
        return self

    @model_validator(mode="after")
    def _variants_supported(self) -> Task:
        from rlenv.variants import supported

        v = self.verification.env_variants
        if v is not None and not supported(v.kind, self.env_id):
            raise ValueError(f"env variant kind {v.kind!r} is not supported for {self.env_id}")
        return self

    def episode_limit(self) -> int:
        """Step limit used everywhere the harness evaluates (task value, else the env spec)."""
        import gymnasium

        return self.max_episode_steps or gymnasium.spec(self.env_id).max_episode_steps

    def agent_view(self) -> dict[str, Any]:
        """What the agent may see: everything except the held-out eval salt."""
        data = self.model_dump(mode="json")
        data["success"].pop("eval_seed_salt")
        return data

    def public_sha256(self) -> str:
        """Hash of the agent-visible spec (safe to log)."""
        return _sha256_json(self.agent_view())

    def full_sha256(self) -> str:
        """Hash of the complete spec incl. the salt (identifies the exact verification setup)."""
        return _sha256_json(self.model_dump(mode="json"))


def _sha256_json(data: Any) -> str:
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _format(exc: ValidationError) -> str:
    return "\n".join(
        f"  - {'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg'].removeprefix('Value error, ')}"
        for e in exc.errors()
    )


def parse_task(data: Any, source: str = "<data>") -> Task:
    if not isinstance(data, dict):
        raise TaskError(f"{source}: top level must be a mapping")
    try:
        return Task.model_validate(data)
    except ValidationError as exc:
        raise TaskError(f"{source}: invalid task\n{_format(exc)}") from exc


def load_task(path: str | Path) -> Task:
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text())
    except OSError as exc:
        raise TaskError(f"cannot read {path}: {exc.strerror}") from exc
    except yaml.YAMLError as exc:
        raise TaskError(f"{path}: invalid YAML: {exc}") from exc
    task = parse_task(data, str(path))
    if task.id != path.stem:
        raise TaskError(f"{path}: task id {task.id!r} must match the file name {path.stem!r}")
    return task


def load_tasks(directory: str | Path) -> dict[str, Task]:
    tasks: dict[str, Task] = {}
    for path in sorted(Path(directory).glob("*.yaml")):
        task = load_task(path)
        tasks[task.id] = task  # unique by construction: id == file name
    return tasks
