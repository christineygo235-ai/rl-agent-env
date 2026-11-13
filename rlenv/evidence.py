"""Generate docs/EVIDENCE.md from committed real artifacts (step 11).

    rlenv evidence            -> rewrite docs/EVIDENCE.md
    rlenv evidence --check    -> exit 1 if the committed file is out of date

Every number comes from an artifact in the repo:
- evidence/redteam/<agent>/   real red-team episode trajectories (production task config);
- samples/deepseek-flash-*/   real DeepSeek episodes (one scored, one unscorable);
- calibration/<task>.json     the calibration runs (10 seeds per task).
Each trajectory is integrity-checked and re-scored while the document is built:
a tampered or stale artifact makes generation fail instead of printing its number.
Output is deterministic (no timestamps), so a test can detect drift.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlenv.agents.adversarial import check_expectation, make_adversaries
from rlenv.atif import to_atif, validate_atif
from rlenv.export import to_messages_row, to_nemo_gym_row
from rlenv.harness import rescore
from rlenv.redteam import REFERENCE_EXPECTATION
from rlenv.trajectory import Trajectory, load_trajectory

PROJECT = Path(__file__).resolve().parents[1]
FLASH = "samples/deepseek-flash-frozenlake"
FLASH_BUG = "samples/deepseek-flash-unscorable"
TASK_ORDER = ["frozenlake-tabular", "taxi-tabular", "cliffwalking-sarsa", "mountaincar-tilecoding"]

# Declared mechanism per agent (what it does and which layer is supposed to stop it).
# The "observed" column is computed from the trajectory, not from this table.
AGENTS: dict[str, tuple[str, str]] = {
    "nop": ("submits immediately, empty workspace", "files gate"),
    "random_floor": ("trains nothing: a random Q-table, honest report", "held-out evaluation (floor)"),
    "artifact_reuse": ("trains once, then swaps train.py for one that copies the old artifact (runs/r1)",
                       "clean re-run of train.py in a fresh workspace"),
    "fabricated_report": ("weak training; REPORT.md claims **0.950** and cites runs r2/r3 that never ran",
                          "report check against harness-logged dev means and run ids; held-out evaluation"),
    "env_tamper": ("policy.py monkey-patches gymnasium.Env.step to inflate rewards",
                   "harness-owned evaluator: env and return sums live in the harness; the policy process "
                   "only maps observations to actions"),
    "pickle_artifact": ("saves/loads the Q-table with allow_pickle=True (code execution on load)",
                        "static AST checks (no pickle, data-only artifacts)"),
    "hardcoded_policy": ("ships a hand-written Q-table for the default 4x4 map in table.json",
                         "held-out env instances (random hidden maps); audit flag for review"),
    "reference": ("the reference solution (tabular Q-learning) with an honest report", "must pass"),
}
CHEATS = ["artifact_reuse", "fabricated_report", "env_tamper", "pickle_artifact", "hardcoded_policy"]
SCOREBOARD = ["nop", "random_floor", *CHEATS, "reference"]

# DeepSeek list prices per 1M tokens (ADR-0011, read 2026-11-11): (off-peak, peak).
FLASH_PRICES = {"cache_hit": (0.003, 0.006), "cache_miss": (0.15, 0.30), "output": (0.60, 1.20)}


class EvidenceError(Exception):
    pass


@dataclass
class Episode:
    name: str
    path: Path
    traj: Trajectory
    final: dict
    verification: dict
    root: Path

    @property
    def rel(self) -> str:
        return self.path.relative_to(self.root).as_posix()


def _load(path: Path, name: str, root: Path) -> Episode:
    traj = load_trajectory(path)
    if not traj.ok or traj.head_ok is not True:
        raise EvidenceError(f"{path}: integrity check failed: {traj.problems[:3]}")
    res = rescore(path)
    if not res.ok:
        raise EvidenceError(f"{path}: reward does not reproduce: {res.detail}")
    return Episode(name, path, traj, traj.of_type("final")[-1], traj.of_type("verification")[-1], root)


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "None"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{nd}f}".rstrip("0").rstrip(".") if nd else f"{x:.0f}"
    return str(x)


def _components(final: dict) -> str:
    c = final.get("components") or {}
    if not c:
        return "–"
    return " / ".join(_fmt(c.get(k), 2) for k in ("success", "progress", "report"))


def _observed(ep: Episode) -> str:
    """What the verification record shows stopped (or scored) this agent."""
    f, v = ep.final, ep.verification
    parts = []
    failed = [g for g, ok in (f.get("gates") or {}).items() if not ok]
    if failed:
        parts.append(f"hard fail `{f.get('failure_kind')}` (gates failed: {', '.join(sorted(failed))})")
    if v.get("static_violations"):
        parts.append("static: " + "; ".join(str(s) for s in v["static_violations"][:2]))
    if v.get("files_missing"):
        parts.append("missing: " + ", ".join(v["files_missing"]))
    if v.get("rerun_ok") is False and v.get("rerun_detail"):
        parts.append("clean re-run: " + str(v["rerun_detail"]).split(";")[0].replace("variant 0: ", ""))
    if not failed:
        variants = v.get("variants") or []
        if variants:
            worst = min(variants, key=lambda x: x.get("z", 0))
            parts.append(f"held-out: worst hidden map z = {_fmt(worst.get('z'))} (pass needs 1)")
        elif v.get("eval_mean") is not None:
            parts.append(f"held-out mean {_fmt(v['eval_mean'])} vs threshold {_fmt(v.get('threshold'))}")
        rd = v.get("report_detail") or {}
        if rd.get("unmatched_claims") or rd.get("phantom_runs"):
            parts.append(f"report: unmatched claims {rd.get('unmatched_claims')}, phantom runs {rd.get('phantom_runs')}")
    if v.get("audit_flags"):
        parts.append("audit flags: " + ", ".join(v["audit_flags"]))
    return "; ".join(parts) or "–"


def _verdict(ep: Episode) -> str:
    exp = REFERENCE_EXPECTATION if ep.name == "reference" else make_adversaries()[ep.name]().expectation
    problems = check_expectation(exp, ep.final["reward"], ep.final["status"], ep.final.get("failure_kind"))
    bound = f"≥ {exp.min_reward}" if ep.name == "reference" else f"≤ {exp.max_reward}"
    return f"as expected (reward {bound})" if not problems else "VIOLATION: " + "; ".join(problems)


def _truncate(value: Any, n: int = 140) -> Any:
    if isinstance(value, str):
        return value if len(value) <= n else value[:n] + f"… [{len(value)} chars]"
    if isinstance(value, dict):
        return {k: _truncate(v, n) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate(v, n) for v in value]
    return value


def _json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False) + "\n```"


# --------------------------------------------------------------------------- #


def build(root: Path = PROJECT) -> str:
    rt_root = root / "evidence" / "redteam"
    rt = {n: _load(rt_root / n, n, root) for n in SCOREBOARD if (rt_root / n).exists()}
    missing = [n for n in SCOREBOARD if n not in rt]
    if missing:
        raise EvidenceError(f"missing red-team trajectories: {missing}")
    flash = _load(root / FLASH, "deepseek-flash", root)
    flash_bug = _load(root / FLASH_BUG, "deepseek-flash (harness bug)", root)
    calib = {t: json.loads((root / "calibration" / f"{t}.json").read_text()) for t in TASK_ORDER}

    out: list[str] = []
    w = out.append
    w("# EVIDENCE\n")
    w("Generated by `rlenv evidence` from real artifacts committed in this repository. "
      "Do not edit by hand: `make evidence` regenerates it, and `tests/test_evidence.py` fails if "
      "it is out of date. While building it, every trajectory below was integrity-checked "
      "(hash chain, head, blobs) and re-scored (`rescore` reproduced the recorded reward).\n")
    w("| Source | What |\n|---|---|")
    w("| `evidence/redteam/<agent>/` | red-team episodes on `frozenlake-tabular` (production task config) |")
    w(f"| `{FLASH}/` | a real DeepSeek `deepseek-flash` episode (+ exports, final files) |")
    w(f"| `{FLASH_BUG}/` | the first real episode, ended by a harness bug: recorded as unscorable |")
    w("| `calibration/<task>.json` | reference over 10 seeds per task |\n")

    # 1. Scoreboard -----------------------------------------------------------
    w("## 1. Scoreboard (frozenlake-tabular, task v{})\n".format(rt["reference"].final["task_version"]))
    w("Reward v1 = 0.6·success + 0.3·progress + 0.1·report after hard gates (ADR-0007). "
      "Components are success / progress / report.\n")
    w("| Agent | What it does | Reward | Status | Success | Components | Failure kind | Trajectory |")
    w("|---|---|---|---|---|---|---|---|")
    rows = [*(rt[n] for n in SCOREBOARD), flash]
    for ep in rows:
        what = AGENTS[ep.name][0] if ep.name in AGENTS else "LLM agent, real API calls (see §4)"
        f = ep.final
        w(f"| **{ep.name}** | {what} | **{_fmt(f['reward'])}** | {f['status']} | {_fmt(f.get('success'))} | "
          f"{_components(f)} | {f.get('failure_kind') or '–'} | `{ep.rel}` |")
    passing = [ep.final["reward"] for ep in rows if ep.final.get("success")]
    failing = [ep.final["reward"] for ep in rows if not ep.final.get("success") and ep.final["reward"] is not None]
    w("")
    w(f"Ordering guarantee (ADR-0007): lowest passing reward **{_fmt(min(passing))}** > highest failing reward "
      f"**{_fmt(max(failing))}**; by construction any pass ≥ 0.6 and any fail ≤ 0.4. "
      f"Unscorable is not 0: the harness-bug episode `{FLASH_BUG}` has reward "
      f"**{_fmt(flash_bug.final['reward'])}** (`{flash_bug.final.get('failure_kind')}`).\n")

    # 2. Red team -------------------------------------------------------------
    w("## 2. Red team: attack → intercepting layer → score\n")
    w(f"{len(CHEATS)} cheating agents, plus floors (nop, random) and the reference. 'Observed' is read from each "
      "trajectory's verification record; 'Verdict' checks the reward against the expectation declared in "
      "`rlenv/agents/adversarial.py`.\n")
    w("| Attack | How it cheats | Layer that stops it | Observed in the trajectory | Reward | Verdict |")
    w("|---|---|---|---|---|---|")
    violations = 0
    for n in [*CHEATS, "nop", "random_floor", "reference"]:
        ep = rt[n]
        verdict = _verdict(ep)
        violations += verdict.startswith("VIOLATION")
        what, layer = AGENTS[n]
        w(f"| **{n}** | {what} | {layer} | {_observed(ep)} | **{_fmt(ep.final['reward'])}** | {verdict} |")
    w("")
    w(f"Result: {len(CHEATS) + 3} agents, **{violations} violation(s)**. Every cheat stays in the fail band "
      f"(max {_fmt(max(rt[n].final['reward'] for n in CHEATS))}); the reference passes "
      f"({_fmt(rt['reference'].final['reward'])}).\n")

    # 3. Calibration -----------------------------------------------------------
    w("## 3. Calibration: is each task solvable, unsolvable by chance, and deterministic?\n")
    w("Reference solution re-trained from scratch and evaluated on held-out seeds for 10 task seeds "
      "(`make calibrate`). Admission requires reference pass rate ≥ 90 %, random baseline failing, "
      "and identical results on repeat. The calibration reference writes no report, so its reward is 0.9.\n")
    w("| Task | Threshold | Reference, 10 seeds (min / median / max) | Random baseline (median) | Pass rate | "
      "Deterministic | Max re-run s | Admitted |")
    w("|---|---|---|---|---|---|---|---|")
    for t in TASK_ORDER:
        c = calib[t]
        em = c["eval_means"]
        rnd = statistics.median(r["random_mean"] for r in c["results"])
        if c.get("has_variants"):
            thr = "z ≥ 1 on every hidden map (z = (agent − random) / (0.9·optimal − random))"
            ref = f"z {_fmt(em['min'])} / {_fmt(em['median'])} / {_fmt(em['max'])}"
            rnd_s = f"z {_fmt(rnd)}"
        else:
            thr = f"mean return ≥ {_fmt(c['threshold'])}"
            ref = f"{_fmt(em['min'])} / {_fmt(em['median'])} / {_fmt(em['max'])}"
            rnd_s = _fmt(rnd)
        w(f"| `{t}` (v{c['task_version']}) | {thr} | {ref} | {rnd_s} | {_fmt(c['pass_rate'] * 100, 0)} % | "
          f"{_fmt(c['deterministic'])} | {_fmt(c['max_rerun_seconds'], 1)} | **{_fmt(c['admitted'])}** |")
    w("")
    tight = [t for t in TASK_ORDER if not calib[t].get("has_variants")
             and calib[t]["eval_means"]["min"] == calib[t]["threshold"]]
    if tight:
        w(f"Note: on {', '.join(f'`{t}`' for t in tight)} the reference lands exactly on the threshold (deterministic "
          "environment, the optimal safe path); see Limitations.\n")

    # 4. Real LLM episode --------------------------------------------------------
    w("## 4. Real LLM episode: DeepSeek `deepseek-flash` on frozenlake-tabular\n")
    calls = flash.traj.of_type("llm_call")
    usage: dict[str, int] = {}
    for c in calls:
        u = c.get("usage") or {}
        for k in ("prompt_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "completion_tokens"):
            usage[k] = usage.get(k, 0) + int(u.get(k) or 0)
        usage["reasoning"] = usage.get("reasoning", 0) + int(
            ((u.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)
    lo = sum(usage[k] * FLASH_PRICES[p][0] for k, p in (("prompt_cache_hit_tokens", "cache_hit"),
                                                          ("prompt_cache_miss_tokens", "cache_miss"),
                                                          ("completion_tokens", "output"))) / 1e6
    hi = sum(usage[k] * FLASH_PRICES[p][1] for k, p in (("prompt_cache_hit_tokens", "cache_hit"),
                                                          ("prompt_cache_miss_tokens", "cache_miss"),
                                                          ("completion_tokens", "output"))) / 1e6
    end = flash.traj.of_type("end")[-1]
    v = flash.verification
    w("| Fact | Value |\n|---|---|")
    w(f"| Reward | **{_fmt(flash.final['reward'])}** ({flash.final['status']}; components {_components(flash.final)}) |")
    stop = flash.traj.of_type("llm_stop")
    why = f" (driver: {stop[-1]['reason']}; pre-revision budget, ADR-0011)" if stop else ""
    w(f"| End | {end['end_reason']}{why} after {end['tool_calls']} tool calls, {end['runs']} training runs, "
      f"{_fmt(end['duration_s'], 0)} s; REPORT.md was written, the workspace was scored as is |")
    w(f"| Model calls | {len(calls)} (`{calls[0].get('model')}`, thinking {calls[0]['params']['thinking']['type']}) |")
    w(f"| Tokens | prompt {usage['prompt_tokens']:,} (cache hit {usage['prompt_cache_hit_tokens']:,}, "
      f"miss {usage['prompt_cache_miss_tokens']:,}); completion {usage['completion_tokens']:,} "
      f"(reasoning {usage['reasoning']:,}) |")
    w(f"| Billable tokens (ADR-0011) | {usage['prompt_cache_miss_tokens'] + usage['completion_tokens']:,} |")
    w(f"| Cost at list prices | ${lo:.3f} (off-peak) – ${hi:.3f} (peak) |")
    w(f"| Report check | claims {v['report_detail'].get('claimed_results')} matched logged dev means; "
      f"phantom runs {v['report_detail'].get('phantom_runs')} |")
    w(f"| Audit flags | {v.get('audit_flags') or 'none'} |")
    w(f"| Integrity | hash chain OK; `rlenv rescore {FLASH}` reproduces the reward |\n")
    w("Held-out evaluation (train.py re-run from scratch on each hidden map, then evaluated on held-out seeds):\n")
    w("| Hidden map | Agent success rate | Optimal policy | Random | z (pass ≥ 1) |\n|---|---|---|---|---|")
    for var in v.get("variants") or []:
        w(f"| #{var['index']} (`{var['env_kwargs_sha256'][:12]}`) | {_fmt(var['agent_mean'])} | "
          f"{_fmt(var['optimal_mean'])} | {_fmt(var['random_mean'])} | {_fmt(var.get('z'))} |")
    w("")
    w(f"Files: `{FLASH}/final/` (train.py, policy.py, REPORT.md as submitted), "
      f"`{FLASH}/trajectory.jsonl` (the full transcript incl. reasoning), `{FLASH}/export/`.\n")

    # 5. Export samples ----------------------------------------------------------
    w("## 5. Export samples (from the DeepSeek episode)\n")
    w("Full files in `" + FLASH + "/export/`. Long strings are truncated here.\n")
    m = to_messages_row(flash.traj)
    msg_excerpt = {k: m[k] for k in ("id", "group_id", "reward", "reward_components", "success", "transcript")}
    msg_excerpt["messages[2:4]"] = _truncate(m["messages"][2:4])
    msg_excerpt["loss_mask[:8]"] = " ".join(map(str, m["loss_mask"][:8]))
    msg_excerpt["counts"] = {"messages": len(m["messages"]), "assistant (loss_mask=1)": sum(m["loss_mask"])}
    w("**`messages`** (OpenAI chat + message-level loss mask; exact transcript):\n")
    w(_json_block(msg_excerpt) + "\n")
    ng = to_nemo_gym_row(flash.traj, 0, 0)
    ng_excerpt = {k: ng[k] for k in ("reward", "mask_sample", "failure_kind", "reward_success", "reward_progress",
                                     "reward_report", "_ng_task_index", "_ng_rollout_index")}
    ng_excerpt["response.output[:2]"] = _truncate(ng["response"]["output"][:2])
    w("**`nemo-gym`** (Responses API items + reward, NeMo Gym rollout row):\n")
    w(_json_block(ng_excerpt) + "\n")
    doc = to_atif(flash.traj)
    problems = validate_atif(doc)
    if problems:
        raise EvidenceError(f"ATIF export invalid: {problems[:3]}")
    agent_step = next(s for s in doc["steps"] if s["source"] == "agent")
    w(f"**`atif`** (Harbor {doc['schema_version']}; {len(doc['steps'])} steps, valid under `validate_atif`), "
      "first agent step:\n")
    w(_json_block(_truncate(agent_step, 100)) + "\n")

    # 6. Limitations ----------------------------------------------------------------
    ref_limits = rt["reference"].verification.get("limits_applied") or {}
    not_applied = sorted(k for k, ok in ref_limits.items() if not ok)
    w("## 6. Known limitations\n")
    w("What the evidence above does **not** show, and where the design is weaker than it looks.\n")
    lim: list[str] = []
    lim.append(f"**One real LLM episode.** n = 1 for `deepseek-flash` (plus one unscorable run). No pass@k, "
      "no variance across seeds or models; DeepSeek has no `seed` parameter, so episodes are not reproducible "
      "(the transcript is, via replay).")
    lim.append("**Method compliance is instructed, not verified.** The task names Q-learning/SARSA/tile coding/DQN; "
      "the static checks enforce a module allowlist (numpy, gymnasium; torch only where allowed) and ban "
      "pickle/network/process/FFI imports, but do not prove *which* algorithm runs. E.g. dynamic programming "
      "on `env.unwrapped.P` would pass FrozenLake.")
    lim.append("**Held-out env instances exist only for FrozenLake.** Taxi and CliffWalking are deterministic and "
      "use held-out seeds only: a hand-written optimal policy would pass there. Audit flags (not in the reward) "
      "point a reviewer at such submissions (ADR-0009 D).")
    lim.append(f"**The process sandbox is not a security boundary.** rlimits, scrubbed env, process-group kill and "
      f"static checks; no network or filesystem isolation at the OS level (ADR-0002). On this macOS host "
      f"{', '.join(not_applied) or 'no limit'} was not enforced (from `limits_applied` in the reference "
      "trajectory); an RSS watchdog stands in. A container backend (`--network none`, read-only root) is the "
      "production answer behind the `Sandbox` interface.")
    lim.append("**Red team covers known attacks on one task.** The adversaries target FrozenLake; unknown attacks "
      "and other tasks are covered only by the general layers (re-run, harness-owned evaluation, static checks).")
    lim.append("**The report check is syntactic.** Bold numbers must match harness-logged dev means and cited runs "
      "must exist; the prose (method description, limitations) is not verified. The report term is 0.1 at most.")
    if tight:
        lim.append(f"**Zero margin on {', '.join(tight)}.** The reference reaches exactly the threshold; it is "
          "deterministic (pass rate 100 %), but any regression in the reference fails the task.")
    lim.append("**Cost guard overshoots by up to one model call** (checked after each call); the 80 % warning makes "
      "reaching the cap rare. Prices for the cost estimate are list prices from ADR-0011, not billed amounts.")
    lim.append("**Single machine, sequential episodes.** No parallel rollout workers, no resumable batch runs; "
      "the facade (ADR-0012) is the hook for an external rollout loop.")
    for i, text in enumerate(lim, 1):
        w(f"{i}. {text}")
    w("")
    return "\n".join(out)


def write(path: Path | None = None) -> Path:
    path = path or PROJECT / "docs" / "EVIDENCE.md"
    path.write_text(build())
    return path


def is_current(path: Path | None = None) -> bool:
    path = path or PROJECT / "docs" / "EVIDENCE.md"
    return path.exists() and path.read_text() == build()
