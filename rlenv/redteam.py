"""Red team: run the reference, the floor agents and every adversary on a task,
and check each outcome against its declared expectation.

    rlenv redteam [TASK] [--out DIR]   -> table; exit 2 if any expectation is violated

Known gaps are reported as GAP (documented, not a failure). If a known gap
suddenly stops passing (someone fixed it), it is flagged so docs get updated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rlenv.agents.adversarial import Expectation, check_expectation, make_adversaries
from rlenv.agents.scripted import ScriptedAgent
from rlenv.harness import run_episode
from rlenv.reference import REFERENCE_ROOT
from rlenv.task import Task

REFERENCE_EXPECTATION = Expectation(1.0, "scored", min_reward=0.9, note="the reference solution must pass")


@dataclass
class RedTeamRow:
    agent: str
    reward: float | None
    status: str
    failure_kind: str | None
    components: dict
    expectation: Expectation
    problems: list[str]
    trajectory_dir: Path

    @property
    def verdict(self) -> str:
        if self.expectation.known_gap:
            return "GAP" if not self.problems else "GAP-CHANGED"
        return "OK" if not self.problems else "VIOLATION"


def run_redteam(task: Task, root: str | Path, names: list[str] | None = None, include_reference: bool = True,
                reference_root: Path = REFERENCE_ROOT) -> list[RedTeamRow]:
    if not task.env_id.startswith("FrozenLake"):
        raise ValueError("the adversaries target FrozenLake-v1 (tabular); use frozenlake-tabular")
    root = Path(root)
    adversaries = make_adversaries()
    selected = names or list(adversaries)
    unknown = [n for n in selected if n not in adversaries]
    if unknown:
        raise ValueError(f"unknown adversaries {unknown}; known: {sorted(adversaries)}")
    agents = []
    if include_reference:
        agents.append(("reference", ScriptedAgent(task.id, reference_root=reference_root), REFERENCE_EXPECTATION))
    for n in selected:
        agent = adversaries[n]()
        agents.append((n, agent, agent.expectation))
    rows = []
    for name, agent, exp in agents:
        rec = run_episode(task, agent, root)
        r = rec.reward
        rows.append(RedTeamRow(name, r.reward, r.status, r.failure_kind, r.components, exp,
                               check_expectation(exp, r.reward, r.status, r.failure_kind), rec.trajectory_dir))
    return rows


def format_table(rows: list[RedTeamRow]) -> str:
    lines = [f"{'agent':<18} {'reward':>7} {'status':<10} {'failure_kind':<17} {'verdict':<11} note",
             "-" * 100]
    for r in rows:
        reward = "None" if r.reward is None else f"{r.reward:.3f}"
        note = "; ".join(r.problems) if r.problems else r.expectation.note
        lines.append(f"{r.agent:<18} {reward:>7} {r.status:<10} {str(r.failure_kind):<17} {r.verdict:<11} {note}")
    return "\n".join(lines)
