"""Export trajectories for training pipelines (ADR-0008).

The native trajectory (hash-chained, event-level JSONL) stays the source of truth.
Exports are derived, one row per episode:

- `messages`: OpenAI chat format (assistant `tool_calls`, `tool` results) plus a
  message-level `loss_mask` (1 = model-generated). verl / miles / TRL / OpenEnv-style
  consumers derive token masks with their own tokenizer.
- `nemo-gym`: NeMo Gym rollout rows (Responses API `function_call` /
  `function_call_output` items, `reward`, `mask_sample`, `failure_kind`,
  `_ng_task_index`, `_ng_rollout_index`).
- `atif`: Harbor ATIF v1.8 documents, one JSON file per episode in the output
  directory (rlenv/atif.py; reward in `extra.rlenv`).

Rules shared by all formats:
- integrity first: a trajectory that fails the chain/head/blob checks is skipped and
  reported, never exported;
- **unscorable episodes are excluded by default** (as verl/miles filter them rather
  than score 0). With `include_unscorable`, they are emitted with reward null
  (`messages`) or reward 0.0 + `mask_sample: true` (NeMo Gym's own convention);
- write_file contents are re-inlined from blobs: the code the model wrote is the
  most important part of the sample;
- every row carries `source` provenance: episode id, trajectory head sha256, schema
  and reward version, so a training example can be traced back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from rlenv.tools import TOOL_SPECS, openai_tools
from rlenv.trajectory import Trajectory, TrajectoryError, load_trajectory

FORMATS = ("messages", "nemo-gym", "atif")
EXPORT_VERSION = "rlenv.export/v1"


@dataclass
class ExportReport:
    written: int = 0
    skipped_unscorable: int = 0
    skipped_invalid: list[str] = field(default_factory=list)
    skipped_incomplete: list[str] = field(default_factory=list)


def _inline(traj: Trajectory, value: Any) -> Any:
    return traj.inline(value)


def _strip(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k not in ("seq", "prev", "t", "type")}


def _turns(traj: Trajectory) -> list[tuple[dict, dict]]:
    calls = {r["i"]: r for r in traj.of_type("tool_call")}
    results = {r["i"]: r for r in traj.of_type("tool_result")}
    return [(calls[i], results.get(i, {"ok": False, "error": "no result recorded"})) for i in sorted(calls)]


def _prompt(traj: Trajectory) -> str:
    h = traj.header
    task = h["task"]
    return (f"{task['instruction'].strip()}\n\nTask spec (agent view):\n"
            f"{json.dumps({k: v for k, v in task.items() if k != 'instruction'}, sort_keys=True)}")


def _common(traj: Trajectory) -> dict | None:
    finals = traj.of_type("final")
    if not finals:
        return None
    fin, h = finals[-1], traj.header
    head = (traj.directory / "trajectory.head").read_text().split()[1]
    return {
        "id": h["episode_id"],
        "group_id": fin.get("group_id"),
        "task_id": h["task_id"],
        "task_version": h["task_version"],
        "reward": fin["reward"],
        "reward_components": fin.get("components", {}),
        "success": fin.get("success"),
        "status": fin["status"],
        "failure_kind": fin.get("failure_kind"),
        "failure_reason": fin.get("reason"),
        "terminated": fin.get("terminated"),
        "truncated": fin.get("truncated"),
        "end_reason": fin.get("end_reason"),
        "source": {"episode_id": h["episode_id"], "trajectory_head_sha256": head, "schema": h["schema"],
                   "reward_version": fin.get("version"), "export_version": EXPORT_VERSION,
                   "agent": h.get("agent"), "runtime": h.get("runtime")},
    }


def llm_transcript(traj: Trajectory) -> list[dict] | None:
    """The exact conversation an LLM driver had (from `llm_call` events), or None.

    Each llm_call logs the messages added since the previous call plus the model's
    response. Tool results after the last call (e.g. submit's) are appended from
    the episode's own tool_result records, matched to the last response's call ids."""
    calls = [r for r in traj.records if r.get("type") == "llm_call"]
    if not calls:
        return None
    messages: list[dict] = []
    for c in calls:
        messages.extend(traj.inline(c.get("new_messages", [])))
        messages.append(traj.inline(c["response"]))
    last_seq = calls[-1]["seq"]
    tail_calls = [r for r in traj.of_type("tool_call") if r["seq"] > last_seq]
    results = {r["i"]: r for r in traj.of_type("tool_result")}
    ids = [tc["id"] for tc in (messages[-1].get("tool_calls") or [])]
    for call_id, call in zip(ids, tail_calls):
        res = results.get(call["i"])
        if res is not None:
            messages.append({"role": "tool", "tool_call_id": call_id,
                             "content": json.dumps(traj.inline(_strip(res)), sort_keys=True)})
    return messages


def to_messages_row(traj: Trajectory) -> dict | None:
    row = _common(traj)
    if row is None:
        return None
    exact = llm_transcript(traj)
    if exact is not None:  # an LLM episode: export what the model actually saw and said
        row.update({"tools": openai_tools(), "messages": exact, "transcript": "exact",
                    "loss_mask": [1 if m.get("role") == "assistant" else 0 for m in exact]})
        return row
    messages: list[dict] = [{"role": "user", "content": _prompt(traj)}]
    for call, result in _turns(traj):
        call_id = f"call_{call['i']}"
        args = _inline(traj, call.get("args", {}))
        messages.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": call["tool"], "arguments": json.dumps(args)}}]})
        messages.append({"role": "tool", "tool_call_id": call_id,
                         "content": json.dumps(_inline(traj, _strip(result)), sort_keys=True)})
    row.update({"tools": openai_tools(), "messages": messages, "transcript": "reconstructed",
                "loss_mask": [1 if m["role"] == "assistant" else 0 for m in messages]})
    return row


def to_nemo_gym_row(traj: Trajectory, task_index: int, rollout_index: int) -> dict | None:
    row = _common(traj)
    if row is None:
        return None
    output: list[dict] = []
    for call, result in _turns(traj):
        call_id = f"call_{call['i']}"
        output.append({"type": "function_call", "call_id": call_id, "name": call["tool"],
                       "arguments": json.dumps(_inline(traj, call.get("args", {})))})
        output.append({"type": "function_call_output", "call_id": call_id,
                       "output": json.dumps(_inline(traj, _strip(result)), sort_keys=True)})
    unscorable = row["reward"] is None
    nemo = {
        "responses_create_params": {
            "input": [{"role": "user", "content": _prompt(traj)}],
            "tools": [{"type": "function", "name": t["name"], "description": t["description"],
                       "parameters": t["input_schema"]} for t in TOOL_SPECS]},
        "response": {"output": output},
        "reward": 0.0 if unscorable else row["reward"],  # NeMo Gym convention: masked samples carry 0.0
        "mask_sample": unscorable,
        "failure_kind": row["failure_kind"],
        "failure_reason": row["failure_reason"],
        "reward_components": row["reward_components"],
        **{f"reward_{k}": v for k, v in row["reward_components"].items()},  # top level, for NeMo metrics
        "terminated": row["terminated"],
        "truncated": row["truncated"],
        "_ng_task_index": task_index,
        "_ng_rollout_index": rollout_index,
        "rlenv": {k: row[k] for k in ("id", "group_id", "task_id", "task_version", "status", "success", "source")},
    }
    return nemo


def export(trajectory_dirs: Iterable[str | Path], out_path: str | Path, fmt: str = "messages",
           include_unscorable: bool = False) -> ExportReport:
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; choose from {FORMATS}")
    report = ExportReport()
    loaded: list[Trajectory] = []
    for d in trajectory_dirs:
        try:
            traj = load_trajectory(d)
        except TrajectoryError as exc:
            report.skipped_invalid.append(f"{d}: {exc}")
            continue
        if not traj.ok or traj.head_ok is not True:
            report.skipped_invalid.append(f"{d}: {traj.problems[:2] or 'no head file'}")
            continue
        if not traj.of_type("final"):
            report.skipped_incomplete.append(str(d))
            continue
        loaded.append(traj)

    # Stable indices: tasks in order of first appearance, rollouts in order within their group.
    loaded.sort(key=lambda t: t.header["episode_id"])
    task_index: dict[str, int] = {}
    rollout_counter: dict[str, int] = {}
    out_path = Path(out_path)
    if fmt == "atif":  # one document per episode, as Harbor expects
        from rlenv.atif import to_atif

        out_path.mkdir(parents=True, exist_ok=True)
        for traj in loaded:
            if traj.of_type("final")[-1]["reward"] is None and not include_unscorable:
                report.skipped_unscorable += 1
                continue
            doc = to_atif(traj)
            (out_path / f"{traj.header['episode_id']}.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
            report.written += 1
        return report
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for traj in loaded:
            fin = traj.of_type("final")[-1]
            if fin["reward"] is None and not include_unscorable:
                report.skipped_unscorable += 1
                continue
            gid = fin.get("group_id") or traj.header["task_id"]
            ti = task_index.setdefault(gid, len(task_index))
            ri = rollout_counter.get(gid, 0)
            rollout_counter[gid] = ri + 1
            row = to_messages_row(traj) if fmt == "messages" else to_nemo_gym_row(traj, ti, ri)
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            report.written += 1
    return report
