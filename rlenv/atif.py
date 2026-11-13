"""ATIF export (Harbor's Agent Trajectory Interchange Format, v1.8; ADR-0008 addendum).

One JSON document per episode, derived from the native trajectory (which stays
the source of truth: ATIF has no hash chain and cannot be appended while an
episode runs, survey §3).

Mapping:
- LLM episodes use the exact transcript (`llm_call` events): system/user messages
  become system/user steps; every model call becomes one `agent` step with
  `model_name`, `reasoning_content`, `tool_calls`, `metrics` (prompt, completion
  and cached tokens) and `llm_call_count: 1`; the harness's tool results become that
  step's `observation.results`, linked by `source_call_id`.
- Other episodes (scripted, replay, adversaries) have no model calls: one user step
  with the instruction, then one `agent` step per tool call with
  `llm_call_count: 0` (so, per the RFC, no metrics and no reasoning).
- ATIF has no reward field: the reward, its components, status, failure kind and
  provenance go to the root `extra.rlenv` (Harbor rejects undeclared fields, so
  nothing custom lives outside `extra`).

`validate_atif` checks the RFC 0001 rules we rely on (field names, required fields,
sequential step ids, source restrictions, tool-call references). It is a local
stand-in for `python -m harbor.utils.trajectory_validator`, which needs Harbor
installed (not a dependency here).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from rlenv.tools import openai_tools
from rlenv.trajectory import Trajectory

ATIF_VERSION = "ATIF-v1.8"

ROOT_FIELDS = {"schema_version", "session_id", "trajectory_id", "agent", "steps", "notes", "final_metrics",
               "continued_trajectory_ref", "extra", "subagent_trajectories"}
AGENT_FIELDS = {"name", "version", "model_name", "tool_definitions", "extra"}
STEP_FIELDS = {"step_id", "timestamp", "source", "model_name", "reasoning_effort", "message", "reasoning_content",
               "tool_calls", "observation", "metrics", "extra", "llm_call_count", "is_copied_context"}
AGENT_ONLY = {"model_name", "reasoning_effort", "reasoning_content", "tool_calls", "metrics"}
TOOL_CALL_FIELDS = {"tool_call_id", "function_name", "arguments", "extra"}
RESULT_FIELDS = {"source_call_id", "content", "subagent_trajectory_ref", "extra"}
METRIC_FIELDS = {"prompt_tokens", "completion_tokens", "cached_tokens", "cost_usd", "prompt_token_ids",
                 "completion_token_ids", "logprobs", "extra"}
FINAL_METRIC_FIELDS = {"total_prompt_tokens", "total_completion_tokens", "total_cached_tokens", "total_cost_usd",
                       "total_steps", "extra"}
SOURCES = {"system", "user", "agent"}


def _compact(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None and v != {} and v != []}


def _arguments(raw: Any) -> tuple[dict, dict | None]:
    """ATIF wants an object; OpenAI-style calls carry a JSON string."""
    if isinstance(raw, dict):
        return raw, None
    try:
        parsed = json.loads(raw) if raw else {}
    except (TypeError, json.JSONDecodeError):
        return {}, {"raw_arguments": str(raw)[:2000], "invalid_json": True}
    return (parsed, None) if isinstance(parsed, dict) else ({}, {"raw_arguments": str(raw)[:2000]})


def to_atif(traj: Trajectory) -> dict | None:
    from rlenv.export import _common, _inline, _prompt, _strip, _turns, llm_transcript

    row = _common(traj)
    if row is None:
        return None
    h = traj.header
    created = datetime.fromisoformat(h["created"])

    def ts(t: Any) -> str | None:
        return (created + timedelta(seconds=float(t))).isoformat() if isinstance(t, (int, float)) else None

    steps: list[dict] = []

    def add(source: str, message: Any, **fields: Any) -> dict:
        step = _compact({"step_id": len(steps) + 1, "source": source, **fields})
        step["message"] = message if message is not None else ""
        steps.append(step)
        return step

    calls = [r for r in traj.records if r.get("type") == "llm_call"]
    totals = {"prompt": 0, "completion": 0, "cached": 0, "reasoning": 0}
    if calls:
        transcript = llm_transcript(traj) or []
        call_iter = iter(calls)
        current: dict | None = None
        for m in transcript:
            role = m.get("role")
            if role in ("system", "user"):
                add(role, m.get("content") or "")
                current = None
            elif role == "assistant":
                c = next(call_iter)
                usage = c.get("usage") or {}
                details = usage.get("completion_tokens_details") or {}
                tool_calls = []
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    args, extra = _arguments(fn.get("arguments"))
                    tool_calls.append(_compact({"tool_call_id": tc.get("id", ""), "function_name": fn.get("name", ""),
                                                "arguments": args, "extra": extra}) | {"arguments": args})
                metrics = _compact({
                    "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                    "cached_tokens": usage.get("prompt_cache_hit_tokens"),
                    "extra": _compact({"reasoning_tokens": details.get("reasoning_tokens"),
                                       "cache_miss_tokens": usage.get("prompt_cache_miss_tokens"),
                                       "finish_reason": c.get("finish_reason"), "latency_s": c.get("latency_s")})})
                for key, src in (("prompt", "prompt_tokens"), ("completion", "completion_tokens"),
                                 ("cached", "prompt_cache_hit_tokens")):
                    totals[key] += int(usage.get(src) or 0)
                totals["reasoning"] += int(details.get("reasoning_tokens") or 0)
                current = add("agent", m.get("content") or "", timestamp=ts(c.get("t")),
                              model_name=c.get("model"), reasoning_content=m.get("reasoning_content"),
                              tool_calls=tool_calls, metrics=metrics, llm_call_count=1)
            elif role == "tool" and current is not None:
                current.setdefault("observation", {"results": []})["results"].append(
                    {"source_call_id": m.get("tool_call_id", ""), "content": m.get("content") or ""})
    else:
        add("user", _prompt(traj))
        for call, result in _turns(traj):
            cid = f"call_{call['i']}"
            add("agent", "", timestamp=ts(call.get("t")), llm_call_count=0,
                tool_calls=[{"tool_call_id": cid, "function_name": call["tool"],
                             "arguments": _inline(traj, call.get("args") or {})}],
                observation={"results": [{"source_call_id": cid,
                                          "content": json.dumps(_inline(traj, _strip(result)), sort_keys=True)}]})

    agent = h.get("agent") or {}
    final_metrics = {"total_steps": len(steps)}
    if calls:
        final_metrics.update({"total_prompt_tokens": totals["prompt"], "total_completion_tokens": totals["completion"],
                              "total_cached_tokens": totals["cached"],
                              "extra": {"llm_calls": len(calls), "total_reasoning_tokens": totals["reasoning"]}})
    rlenv_extra = {k: row[k] for k in ("id", "group_id", "task_id", "task_version", "reward", "reward_components",
                                       "success", "status", "failure_kind", "failure_reason", "terminated",
                                       "truncated", "end_reason", "source")}
    return _compact({
        "schema_version": ATIF_VERSION,
        "session_id": row["id"],
        "trajectory_id": row["id"],
        "agent": _compact({"name": f"rlenv-{agent.get('kind', 'agent')}",
                           "version": str((h.get("runtime") or {}).get("rlenv", "unknown")),
                           "model_name": agent.get("model"), "tool_definitions": openai_tools(),
                           "extra": agent}),
        "steps": steps,
        "notes": "Exported from an rlenv trajectory (hash-chained JSONL is the source of truth). "
                 "The reward computed by the harness is in extra.rlenv.",
        "final_metrics": final_metrics,
        "extra": {"rlenv": rlenv_extra},
    })


def validate_atif(doc: Any) -> list[str]:
    """Problems with an ATIF document (empty = valid for the rules we implement)."""
    p: list[str] = []
    if not isinstance(doc, dict):
        return ["document must be an object"]
    p += [f"unknown root field {k!r}" for k in sorted(set(doc) - ROOT_FIELDS)]
    if not str(doc.get("schema_version", "")).startswith("ATIF-v1."):
        p.append("schema_version must be ATIF-v1.x")
    agent = doc.get("agent")
    if not isinstance(agent, dict):
        p.append("agent is required")
    else:
        p += [f"unknown agent field {k!r}" for k in sorted(set(agent) - AGENT_FIELDS)]
        p += [f"agent.{k} must be a string" for k in ("name", "version") if not isinstance(agent.get(k), str)]
    steps = doc.get("steps")
    if not isinstance(steps, list) or not steps:
        return p + ["steps must be a non-empty list"]
    for n, s in enumerate(steps, 1):
        where = f"step {n}"
        if not isinstance(s, dict):
            p.append(f"{where}: must be an object")
            continue
        p += [f"{where}: unknown field {k!r}" for k in sorted(set(s) - STEP_FIELDS)]
        if s.get("step_id") != n:
            p.append(f"{where}: step_id must be {n} (sequential from 1), got {s.get('step_id')!r}")
        src = s.get("source")
        if src not in SOURCES:
            p.append(f"{where}: source must be one of {sorted(SOURCES)}, got {src!r}")
        if "message" not in s or not isinstance(s["message"], (str, list)):
            p.append(f"{where}: message (string or array) is required")
        if src != "agent":
            p += [f"{where}: {k!r} is only allowed on agent steps" for k in sorted(AGENT_ONLY & set(s))]
        if "observation" in s and src not in ("agent", "system"):
            p.append(f"{where}: observation is only allowed on agent and system steps")
        if src == "agent" and s.get("llm_call_count") == 0:
            p += [f"{where}: {k!r} must be absent when llm_call_count is 0"
                  for k in ("metrics", "reasoning_content") if k in s]
        if "timestamp" in s:
            try:
                datetime.fromisoformat(str(s["timestamp"]))
            except ValueError:
                p.append(f"{where}: timestamp must be ISO 8601")
        call_ids = set()
        for tc in s.get("tool_calls") or []:
            p += [f"{where}: unknown tool_call field {k!r}" for k in sorted(set(tc) - TOOL_CALL_FIELDS)]
            if not isinstance(tc.get("tool_call_id"), str) or not isinstance(tc.get("function_name"), str):
                p.append(f"{where}: tool_call needs string tool_call_id and function_name")
            if not isinstance(tc.get("arguments"), dict):
                p.append(f"{where}: tool_call arguments must be an object")
            call_ids.add(tc.get("tool_call_id"))
        obs = s.get("observation")
        if obs is not None:
            if not isinstance(obs, dict) or not isinstance(obs.get("results"), list) or set(obs) - {"results"}:
                p.append(f"{where}: observation must be {{'results': [...]}}")
            else:
                for r in obs["results"]:
                    p += [f"{where}: unknown result field {k!r}" for k in sorted(set(r) - RESULT_FIELDS)]
                    if "source_call_id" in r and r["source_call_id"] not in call_ids:
                        p.append(f"{where}: source_call_id {r['source_call_id']!r} matches no tool_call in the step")
        metrics = s.get("metrics")
        if metrics is not None:
            p += [f"{where}: unknown metrics field {k!r}" for k in sorted(set(metrics) - METRIC_FIELDS)]
    fm = doc.get("final_metrics")
    if fm is not None:
        p += [f"unknown final_metrics field {k!r}" for k in sorted(set(fm) - FINAL_METRIC_FIELDS)]
        if fm.get("total_steps") not in (None, len(steps)):
            p.append("final_metrics.total_steps does not match the number of steps")
    return p
