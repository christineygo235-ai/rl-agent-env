"""Read-only, one-line-per-event view of a trajectory (for review and demos).

    rlenv timeline DIR [--width N] [--reasoning]

Shows model calls (tokens, finish reason, first words of the reasoning), tool calls
and their results, training runs, the driver's stop reason, the verification facts
and the final reward. Nothing is executed; the integrity check runs first.
"""

from __future__ import annotations

import json
from typing import Any

from rlenv.trajectory import Trajectory


def _short(value: Any, width: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _args(traj: Trajectory, args: dict, width: int) -> str:
    shown = {}
    for k, v in (args or {}).items():
        if isinstance(v, dict) and set(v) == {"blob", "bytes"}:
            shown[k] = f"<{v['bytes']} bytes, blob {v['blob'][:8]}>"
        else:
            shown[k] = v
    return _short(shown, width)


def render(traj: Trajectory, width: int = 110, reasoning: bool = False) -> list[str]:
    h = traj.header
    rt = h.get("runtime") or {}
    lines = [f"episode {h.get('episode_id')}  task {h.get('task_id')} v{h.get('task_version')}  "
             f"agent {_short(h.get('agent', {}).get('kind'), 20)} "
             f"{h.get('agent', {}).get('model', '')}  gymnasium {rt.get('gymnasium')} numpy {rt.get('numpy')}  "
             f"integrity {'OK' if traj.ok else 'BROKEN'}"]
    for r in traj.records:
        t = r.get("t")
        ts = f"{t:7.1f}s" if isinstance(t, (int, float)) else " " * 8
        kind = r.get("type")
        if kind == "llm_call":
            u = r.get("usage") or {}
            resp = traj.inline(r.get("response") or {})
            calls = [c.get("function", {}).get("name") for c in resp.get("tool_calls") or []]
            line = (f"{ts} LLM #{r.get('n')}  {r.get('finish_reason')}  prompt {u.get('prompt_tokens', 0)} "
                    f"(cached {u.get('prompt_cache_hit_tokens', 0)}) out {u.get('completion_tokens', 0)}  "
                    f"-> {', '.join(calls) or 'text'}")
            lines.append(line)
            if reasoning and resp.get("reasoning_content"):
                lines.append(" " * 10 + "thinks: " + _short(resp["reasoning_content"], width - 18))
            if resp.get("content"):
                lines.append(" " * 10 + "says:   " + _short(resp["content"], width - 18))
        elif kind == "tool_call":
            lines.append(f"{ts}   [{r.get('i'):>2}] {r.get('tool')} {_args(traj, r.get('args'), width - 30)}")
        elif kind == "tool_result":
            body = r.get("data") if r.get("ok") else {"error": r.get("error")}
            lines.append(" " * 10 + ("ok    " if r.get("ok") else "ERROR ") + _short(traj.inline(body), width - 16))
        elif kind == "llm_budget_warning":
            lines.append(f"{ts} ! budget warning at {r.get('fraction'):.0%}")
        elif kind in ("llm_stop", "agent_error"):
            lines.append(f"{ts} ! {kind}: {_short(r.get('reason') or r.get('error'), width - 20)}")
        elif kind == "end":
            lines.append(f"{ts} END {r.get('end_reason')}  tool calls {r.get('tool_calls')}  runs {r.get('runs')}")
        elif kind == "verification":
            lines.append(f"{'':8} VERIFY files_missing={r.get('files_missing')} static={r.get('static_ok')} "
                         f"rerun={r.get('rerun_ok')} held-out={r.get('eval_mean')} vs {r.get('threshold')} "
                         f"report={r.get('report_ok')} audit={r.get('audit_flags')}")
            for v in r.get("variants") or []:
                lines.append(f"{'':10} hidden map #{v.get('index')}: agent {v.get('agent_mean')} optimal "
                             f"{v.get('optimal_mean')} random {v.get('random_mean')} z {v.get('z')}")
        elif kind == "final":
            lines.append(f"{'':8} REWARD {r.get('reward')} [{r.get('status')}] {r.get('components')} "
                         f"{r.get('failure_kind') or ''}  ({r.get('reason')})")
    return lines
