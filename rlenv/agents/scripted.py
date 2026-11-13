"""Scripted agent: replays a reference solution through the tools.

It behaves like a competent agent would:
1. write the reference files;
2. run training;
3. evaluate the run on dev seeds;
4. write REPORT.md from the template, filled with the numbers the harness returned;
5. submit.
It is deterministic and offline, which makes the whole environment testable and calibratable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rlenv.agents.base import AgentDriver, Observation, ToolCall
from rlenv.reference import REFERENCE_ROOT


def runs_table(runs: list[dict]) -> str:
    lines = ["| Run | Seed | Extra args | Exit | Train s | Dev episodes | Dev mean return |",
             "|---|---|---|---|---|---|---|"]
    for r in runs:
        mean = "n/a" if r.get("dev_mean") is None else f"{r['dev_mean']:.3f}"
        lines.append(f"| {r['run_id']} | {r['seed']} | {' '.join(r['args']) or '-'} | {r['exit_code']} | "
                     f"{r['duration_s']:.1f} | {r.get('dev_episodes', 0)} | {mean} |")
    return "\n".join(lines)


class ScriptedAgent(AgentDriver):
    kind = "scripted"

    def __init__(self, task_id: str, extra_args: list[str] | None = None, reference_root: Path = REFERENCE_ROOT):
        self.task_id = task_id
        self.extra_args = list(extra_args or [])
        self.ref = reference_root / task_id
        if not self.ref.is_dir():
            raise FileNotFoundError(f"no reference solution for {task_id!r}")
        self.files = {p.name: p.read_text() for p in sorted(self.ref.glob("*.py"))}
        self.template = (self.ref / "REPORT.template.md").read_text()
        self.plan: list[str] = []
        self.runs: list[dict] = []
        self.task: dict[str, Any] = {}

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "task_id": self.task_id, "extra_args": self.extra_args,
                "files": sorted(self.files)}

    def start(self, observation: Observation) -> None:
        self.task = observation.task
        self.plan = [f"write:{n}" for n in self.files] + ["train", "evaluate", "report", "submit"]

    def next_action(self, last_result: dict[str, Any] | None) -> ToolCall | None:
        self._absorb(last_result)
        if not self.plan:
            return None
        step = self.plan.pop(0)
        if step.startswith("write:"):
            name = step.split(":", 1)[1]
            return ToolCall("write_file", {"path": name, "content": self.files[name]})
        if step == "train":
            return ToolCall("run_training", {"args": self.extra_args} if self.extra_args else {})
        if step == "evaluate":
            if not self.runs or self.runs[-1]["exit_code"] != 0:
                self.plan = [p for p in self.plan if p != "evaluate"]
                return self.next_action(None)
            return ToolCall("evaluate", {"run_id": self.runs[-1]["run_id"]})
        if step == "report":
            return ToolCall("write_file", {"path": "REPORT.md", "content": self._report()})
        if step == "submit":
            return ToolCall("submit", {"notes": f"reference solution for {self.task_id}"})
        raise AssertionError(step)

    def _absorb(self, result: dict[str, Any] | None) -> None:
        """Remember run and evaluation results, exactly as the tools returned them."""
        if not result:
            return
        data = result.get("data", {})
        if "artifacts" in data and "run_id" in data:
            self.runs.append({"run_id": data["run_id"], "seed": data["seed"], "args": self.extra_args,
                              "exit_code": data["exit_code"], "duration_s": data["duration_s"]})
        if result.get("ok") and "mean_return" in data:
            for r in self.runs:
                if r["run_id"] == data["run_id"]:
                    r["dev_mean"], r["dev_episodes"] = data["mean_return"], data["episodes"]

    def _report(self) -> str:
        best = max((r for r in self.runs if r.get("dev_mean") is not None),
                   key=lambda r: r["dev_mean"], default={"dev_mean": float("nan"), "dev_episodes": 0})
        return self.template.format(train_seed=self.task.get("seed"), runs_table=runs_table(self.runs),
                                    dev_episodes=best["dev_episodes"], dev_mean=best["dev_mean"],
                                    threshold=self.task.get("success", {}).get("threshold"))
