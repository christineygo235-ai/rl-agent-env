"""Harness entry point: episode -> verification -> reward -> trajectory.

    run_episode(task, agent, root) -> EpisodeRecord
    rescore(trajectory_dir)        -> RescoreResult

The trajectory is written while the episode runs (a crash leaves a valid prefix),
then the verification record and the final reward are appended. `rescore`
recomputes the reward from the verification record alone, proving the reward
is a pure function of logged facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rlenv.agents.base import AgentDriver
from rlenv.episode import Episode, EpisodeOutcome, new_episode_id, outcome_to_json
from rlenv.reward import REWARD_VERSION, RewardResult, VerificationRecord, compute_reward
from rlenv.task import Task
from rlenv.trajectory import Recorder, Redactor, TrajectoryWriter, load_trajectory
from rlenv.verifier import verify


def group_id(task: Task) -> str:
    """Episodes of the same task version form a group (GRPO-style advantages, like verl's uid,
    miles' group_index or NeMo Gym's task index). The hash makes silent spec changes visible."""
    return f"{task.id}@v{task.version}:{task.public_sha256()[:12]}"


@dataclass
class EpisodeRecord:
    outcome: EpisodeOutcome
    verification: VerificationRecord
    reward: RewardResult
    trajectory_dir: Path


class EpisodeSession:
    """One episode with its trajectory: open -> (episode lifecycle) -> finish.

    Shared by `run_episode` (driver loop) and the OpenEnv-style facade, so both write
    the same records in the same order and score with the same verifier and reward."""

    def __init__(self, task: Task, agent: AgentDriver | None, agent_info: dict, root: str | Path,
                 redactor: Redactor | None = None) -> None:
        root = Path(root)
        self.task = task
        self.episode_id = new_episode_id(task)
        self.dir = root / self.episode_id
        self.writer = TrajectoryWriter(self.dir, redactor)
        self.recorder = Recorder(self.writer, self.dir / "workspace")
        self.record: EpisodeRecord | None = None
        try:
            self.recorder.header(self.episode_id, task, agent_info)
            self.episode = Episode(task, agent, root, episode_id=self.episode_id, sink=self.recorder)
        except BaseException:
            self.recorder.close()
            raise

    def finish(self, outcome: EpisodeOutcome) -> EpisodeRecord:
        """Verify the workspace, compute the reward, append outcome/verification/final, close."""
        try:
            self.writer.write({"type": "outcome", **outcome_to_json(outcome)})
            verification = verify(self.task, outcome.workspace, outcome.runs, self.dir / "verify",
                                  agent_driver_error=outcome.agent_error)
            self.writer.write({"type": "verification", **verification.to_json()})
            reward = compute_reward(verification)
            self.writer.write({"type": "final", "episode_id": self.episode_id, "group_id": group_id(self.task),
                               "task_id": self.task.id, "task_version": self.task.version,
                               "end_reason": outcome.end_reason, "terminated": outcome.terminated,
                               "truncated": outcome.truncated, **reward.to_json()})
        finally:
            self.close()
        self.record = EpisodeRecord(outcome, verification, reward, self.dir)
        return self.record

    def close(self) -> None:
        self.recorder.close()


def run_episode(task: Task, agent: AgentDriver, root: str | Path, redactor: Redactor | None = None) -> EpisodeRecord:
    session = EpisodeSession(task, agent, agent.describe(), root, redactor)
    try:
        outcome = session.episode.run()
    except BaseException:
        session.close()
        raise
    return session.finish(outcome)


@dataclass
class RescoreResult:
    ok: bool                 # trajectory intact and recomputed reward == recorded reward
    integrity_ok: bool
    recorded: dict | None
    recomputed: dict | None
    detail: str


def rescore(trajectory_dir: str | Path) -> RescoreResult:
    traj = load_trajectory(trajectory_dir)
    ver, fin = traj.of_type("verification"), traj.of_type("final")
    if not ver or not fin:
        return RescoreResult(False, traj.ok, None, None, "trajectory has no verification/final record")
    data = {k: v for k, v in ver[-1].items() if k not in ("seq", "prev", "type", "t")}
    recomputed = compute_reward(VerificationRecord.from_json(data)).to_json()
    recorded = {k: fin[-1][k] for k in recomputed}
    if recorded.get("version") != REWARD_VERSION:
        return RescoreResult(False, traj.ok, recorded, recomputed,
                             f"recorded with reward {recorded.get('version')}, current is {REWARD_VERSION}: "
                             "this is a regrade, not a reproduction")
    same = recorded == recomputed
    detail = "reward reproduced exactly" if same else "recomputed reward differs from the recorded one"
    if not traj.ok:
        detail += f"; integrity problems: {traj.problems[:3]}"
    return RescoreResult(same and traj.ok, traj.ok, recorded, recomputed, detail)
