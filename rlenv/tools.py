"""The agent's tool API (single source for the harness and the LLM driver).

TOOL_SPECS uses the Anthropic tool format (name, description, input_schema).
`ToolBox.dispatch` validates arguments against the schema, charges the budget,
runs the tool and returns a `ToolResult` (always with `budget_remaining`).

Deliberately not offered: shell access, network, package installs, reading
harness files or held-out seeds. `runs/` belongs to the harness: the agent
cannot write there, so artifacts can only come from `run_training`.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from rlenv.evaluator import evaluate_workspace_policy
from rlenv.policy_check import check_workspace
from rlenv.sandbox import SandboxLimits, SubprocessSandbox
from rlenv.seeds import dev_seeds
from rlenv.task import Task
from rlenv.workspace import Workspace, WorkspaceError

RESERVED_PREFIXES = ("runs/", ".tmp/")
RESERVED_ARGS = ("--seed", "--out", "--env-kwargs")


def train_argv(seed: int, out: str, env_kwargs: dict, extra: list[str] | None = None) -> list[str]:
    """The train.py contract: `--seed S --out DIR --env-kwargs JSON` (+ the agent's extra args in-episode).
    Verification calls it with no extra args."""
    return ["--seed", str(seed), "--out", out, "--env-kwargs", json.dumps(env_kwargs, sort_keys=True), *(extra or [])]


def dev_variants(task: Task) -> list[dict]:
    """Public env instances the agent may train/evaluate on (variant 1..N); variant 0 is canonical."""
    v = task.verification.env_variants
    if v is None or v.dev_count == 0:
        return []
    from rlenv.variants import variant_kwargs

    return variant_kwargs(v.kind, task.env_id, dict(task.env_kwargs), "dev", "public", task.seed, v.dev_count,
                          {"frozen_p": v.frozen_p})
MAX_EXTRA_ARGS = 16
EVAL_TIMEOUT_S = 300.0

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "write_file",
        "description": "Create or overwrite a text file in your workspace (.py, .md, .json, .yaml, .txt, ...). "
                       "Paths are relative to the workspace. You cannot write into runs/.",
        "input_schema": {"type": "object", "additionalProperties": False, "required": ["path", "content"],
                         "properties": {"path": {"type": "string"}, "content": {"type": "string"}}},
    },
    {
        "name": "read_file",
        "description": "Read a text file from your workspace (including run outputs under runs/).",
        "input_schema": {"type": "object", "additionalProperties": False, "required": ["path"],
                         "properties": {"path": {"type": "string"},
                                        "max_bytes": {"type": "integer", "minimum": 1, "maximum": 262144}}},
    },
    {
        "name": "list_files",
        "description": "List files in your workspace (recursively). `path` defaults to the workspace root.",
        "input_schema": {"type": "object", "additionalProperties": False,
                         "properties": {"path": {"type": "string"}}},
    },
    {
        "name": "run_training",
        "description": "Statically check your code, then run `python train.py --seed SEED --out runs/<run_id>/policy "
                       "--env-kwargs JSON [args]` in the sandbox (CPU only, no network). train.py must pass the "
                       "env kwargs to gymnasium.make. `variant` 0 is the task's environment; 1..N are public "
                       "variants (e.g. other FrozenLake maps) for testing generalisation, if the task has them. "
                       "Returns the run id, exit status, duration and the tail of stdout/stderr. Consumes one "
                       "training run and wall-clock budget.",
        "input_schema": {"type": "object", "additionalProperties": False,
                         "properties": {
                             "seed": {"type": "integer", "minimum": 0},
                             "variant": {"type": "integer", "minimum": 0},
                             "args": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_EXTRA_ARGS},
                             "timeout_s": {"type": "number", "minimum": 1}}},
    },
    {
        "name": "evaluate",
        "description": "Evaluate the artifact of a successful run with your current policy.py on development "
                       "seeds, in the same environment variant the run trained on. The harness runs the "
                       "environment and computes returns. Final scoring uses different, held-out seeds (and, if "
                       "the task has variants, unseen variants) and a fresh re-run of train.py.",
        "input_schema": {"type": "object", "additionalProperties": False, "required": ["run_id"],
                         "properties": {"run_id": {"type": "string"},
                                        "episodes": {"type": "integer", "minimum": 1}}},
    },
    {
        "name": "submit",
        "description": "Finish the episode. Your workspace must contain train.py, policy.py and REPORT.md. "
                       "The harness then re-runs train.py from scratch and evaluates it on held-out seeds.",
        "input_schema": {"type": "object", "additionalProperties": False,
                         "properties": {"notes": {"type": "string"}}},
    },
]
TOOL_NAMES = [t["name"] for t in TOOL_SPECS]


def openai_tools() -> list[dict[str, Any]]:
    """The same tools in OpenAI function-calling format (DeepSeek's OpenAI-compatible API).
    TOOL_SPECS stays the single source; this is a pure conversion."""
    return [{"type": "function",
             "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in TOOL_SPECS]
_SCHEMAS = {t["name"]: t["input_schema"] for t in TOOL_SPECS}


class ToolArgError(Exception):
    pass


INVALID_JSON_KEY = "_invalid_json_arguments"  # set by LLM drivers when the model's arguments don't parse


def validate_args(name: str, args: Any) -> dict:
    """Minimal JSON-schema check for the subset used above (types, required, extra keys, bounds)."""
    if name not in _SCHEMAS:
        raise ToolArgError(f"unknown tool {name!r}; available: {TOOL_NAMES}")
    if not isinstance(args, dict):
        raise ToolArgError("arguments must be an object")
    if INVALID_JSON_KEY in args:
        raise ToolArgError(f"arguments are not valid JSON: {str(args[INVALID_JSON_KEY])[:200]!r}")
    schema = _SCHEMAS[name]
    props = schema.get("properties", {})
    missing = [k for k in schema.get("required", []) if k not in args]
    if missing:
        raise ToolArgError(f"missing required argument(s): {missing}")
    extra = [k for k in args if k not in props]
    if extra:
        raise ToolArgError(f"unknown argument(s): {extra}")
    for key, value in args.items():
        spec = props[key]
        t = spec["type"]
        ok = {
            "string": isinstance(value, str),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "array": isinstance(value, list) and all(isinstance(v, str) for v in value),
        }[t]
        if not ok:
            raise ToolArgError(f"argument {key!r} must be of type {t}")
        if "minimum" in spec and value < spec["minimum"]:
            raise ToolArgError(f"argument {key!r} must be >= {spec['minimum']}")
        if "maximum" in spec and value > spec["maximum"]:
            raise ToolArgError(f"argument {key!r} must be <= {spec['maximum']}")
        if "maxItems" in spec and len(value) > spec["maxItems"]:
            raise ToolArgError(f"argument {key!r} has more than {spec['maxItems']} items")
    return args


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #


class Budget:
    def __init__(self, task: Task, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.started = clock()
        self.wall_clock_s = task.budget.wall_clock_s
        self.max_tool_calls = task.budget.max_tool_calls
        self.max_runs = task.budget.max_runs
        self.tool_calls = 0
        self.runs = 0

    def elapsed(self) -> float:
        return self.clock() - self.started

    def wall_remaining(self) -> float:
        return max(0.0, self.wall_clock_s - self.elapsed())

    def exhausted(self) -> str | None:
        if self.wall_remaining() <= 0:
            return "wall_clock"
        if self.tool_calls >= self.max_tool_calls:
            return "tool_calls"
        return None

    def remaining(self) -> dict:
        return {"wall_clock_s": round(self.wall_remaining(), 1),
                "tool_calls": self.max_tool_calls - self.tool_calls,
                "runs": self.max_runs - self.runs}


# --------------------------------------------------------------------------- #
# Results and records
# --------------------------------------------------------------------------- #


@dataclass
class ToolResult:
    ok: bool
    data: dict = field(default_factory=dict)
    error: str | None = None
    budget_remaining: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class RunRecord:
    run_id: str
    seed: int
    args: list[str]
    code: dict[str, str]  # path -> sha256 of the code that was run
    ok: bool
    exit_code: int | None
    killed_reason: str | None
    duration_s: float
    stdout_tail: str
    stderr_tail: str
    artifacts: list[str]
    dev_evals: list[dict] = field(default_factory=list)
    variant: int = 0
    env_kwargs: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


class ToolBox:
    def __init__(self, task: Task, workspace: Workspace, logs_dir: Path, budget: Budget,
                 sandbox: SubprocessSandbox | None = None) -> None:
        self.task = task
        self.ws = workspace
        self.logs = logs_dir
        self.budget = budget
        self.sandbox = sandbox or SubprocessSandbox()
        self.runs: dict[str, RunRecord] = {}
        self.submitted = False
        self.submit_notes = ""

    def dispatch(self, name: str, args: Any) -> ToolResult:
        """Charge one tool call, validate, run. Never raises for agent mistakes."""
        self.budget.tool_calls += 1
        try:
            args = validate_args(name, args)
            result = getattr(self, f"_tool_{name}")(**args)
        except (ToolArgError, WorkspaceError) as exc:
            result = ToolResult(False, error=str(exc))
        result.budget_remaining = self.budget.remaining()
        return result

    # ------------------------------------------------------------ file tools

    def _tool_write_file(self, path: str, content: str) -> ToolResult:
        rel = self.ws.rel(self.ws.resolve(path))
        if rel.startswith(RESERVED_PREFIXES) or rel in ("runs", ".tmp"):
            return ToolResult(False, error=f"{path!r}: runs/ and .tmp/ are managed by the harness")
        p = self.ws.write_text(path, content)
        return ToolResult(True, {"path": self.ws.rel(p), "bytes": len(content.encode())})

    def _tool_read_file(self, path: str, max_bytes: int = 65536) -> ToolResult:
        text, truncated = self.ws.read_text(path, max_bytes)
        return ToolResult(True, {"path": path, "content": text, "truncated": truncated})

    def _tool_list_files(self, path: str = ".") -> ToolResult:
        entries = [asdict(e) for e in self.ws.list(path or ".") if not e.path.startswith(".tmp")]
        return ToolResult(True, {"entries": entries[:500], "truncated": len(entries) > 500})

    # -------------------------------------------------------- run_training

    def _tool_run_training(self, seed: int | None = None, args: list[str] | None = None,
                           timeout_s: float | None = None, variant: int = 0) -> ToolResult:
        args = args or []
        if any(a.split("=")[0] in RESERVED_ARGS for a in args):
            return ToolResult(False, error="--seed, --out and --env-kwargs are set by the harness; "
                                           "use the `seed` and `variant` arguments")
        variants = dev_variants(self.task)
        if variant > len(variants):
            return ToolResult(False, error=f"variant {variant} does not exist; this task has variants 0..{len(variants)}")
        env_kwargs = dict(self.task.env_kwargs) if variant == 0 else variants[variant - 1]
        if self.budget.runs >= self.budget.max_runs:
            return ToolResult(False, error=f"no training runs left (max {self.budget.max_runs})")
        report = check_workspace(self.ws.root, allow_torch=self.task.methods.allow_torch, required=("train.py",))
        if not report.ok:  # not executed, no run consumed
            return ToolResult(False, {"static_check": [asdict(v) for v in report.violations]},
                              error="static check failed; the code was not run:\n" + report.summary())
        timeout = min(timeout_s or self.task.budget.train_timeout_s, self.task.budget.train_timeout_s,
                      self.budget.wall_remaining())
        if timeout < 1:
            return ToolResult(False, error="not enough wall-clock budget left for a run")

        self.budget.runs += 1
        run_id = f"r{self.budget.runs}"
        seed = self.task.seed if seed is None else seed
        out = f"runs/{run_id}/policy"
        code = self.ws.snapshot()
        res = self.sandbox.run_python(
            "train.py", train_argv(seed, out, env_kwargs, args), self.ws.root, self.logs / run_id,
            SandboxLimits(timeout_s=timeout, memory_mb=self.task.limits.memory_mb))
        out_dir = self.ws.root / out
        artifacts = sorted(p.relative_to(self.ws.root).as_posix() for p in out_dir.rglob("*") if p.is_file()) \
            if out_dir.is_dir() else []
        ok = res.ok and bool(artifacts)
        rec = RunRecord(run_id, seed, args, code, ok, res.exit_code, res.killed_reason, res.duration_s,
                        res.stdout_tail, res.stderr_tail, artifacts, variant=variant, env_kwargs=env_kwargs)
        self.runs[run_id] = rec
        data = {"run_id": run_id, "seed": seed, "variant": variant, "env_kwargs": env_kwargs,
                "exit_code": res.exit_code, "killed_reason": res.killed_reason,
                "duration_s": res.duration_s, "artifacts": artifacts,
                "stdout_tail": res.stdout_tail, "stderr_tail": res.stderr_tail}
        if res.ok and not artifacts:
            return ToolResult(False, data, error=f"train.py exited 0 but wrote nothing to {out}")
        return ToolResult(res.ok, data, None if res.ok else
                          f"training failed ({res.killed_reason or f'exit code {res.exit_code}'})")

    # ------------------------------------------------------------ evaluate

    def _tool_evaluate(self, run_id: str, episodes: int | None = None) -> ToolResult:
        rec = self.runs.get(run_id)
        if rec is None:
            return ToolResult(False, error=f"unknown run {run_id!r}; runs: {sorted(self.runs)}")
        if not rec.ok:
            return ToolResult(False, error=f"run {run_id} did not succeed")
        report = check_workspace(self.ws.root, allow_torch=self.task.methods.allow_torch, required=("policy.py",))
        if not report.ok:
            return ToolResult(False, error="static check failed:\n" + report.summary())
        n = min(episodes or self.task.success.dev_eval_episodes, self.task.success.dev_eval_episodes)
        seeds = dev_seeds(self.task.seed, n)
        ev = evaluate_workspace_policy(
            self.task.env_id, dict(rec.env_kwargs), seeds, self.ws.root, f"runs/{run_id}/policy",
            self.logs / f"{run_id}-eval{len(rec.dev_evals) + 1}",
            SandboxLimits(timeout_s=EVAL_TIMEOUT_S, memory_mb=self.task.limits.memory_mb),
            max_duration_s=min(EVAL_TIMEOUT_S, max(1.0, self.budget.wall_remaining())),
            max_episode_steps=self.task.episode_limit())
        summary = {"run_id": run_id, "variant": rec.variant, "episodes": ev.episodes,
                   "mean_return": ev.mean if ev.ok else None,
                   "std_return": ev.std if ev.ok else None,
                   "min_return": min(ev.returns) if ev.returns else None,
                   "max_return": max(ev.returns) if ev.returns else None,
                   "policy_sha256": self.ws.snapshot().get("policy.py"), "seeds": "dev",
                   "duration_s": ev.duration_s}
        rec.dev_evals.append({**summary, "error": ev.error})
        if not ev.ok:
            return ToolResult(False, {**summary, "policy_stderr_tail": ev.policy_stderr_tail},
                              error=f"evaluation failed: {ev.error}")
        return ToolResult(True, summary)

    # -------------------------------------------------------------- submit

    def _tool_submit(self, notes: str = "") -> ToolResult:
        self.submitted = True
        self.submit_notes = notes[:2000]
        files = {e.path for e in self.ws.list() if e.kind == "file"}
        missing = [f for f in ("train.py", "policy.py", "REPORT.md") if f not in files]
        return ToolResult(True, {"submitted": True, "missing_files": missing,
                                 "note": "verification (fresh re-run + held-out evaluation) starts now"})
