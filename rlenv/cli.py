"""rlenv command line.

  rlenv list-tasks [--tasks DIR]              validate every task and print a summary
  rlenv show-task ID [--tasks DIR] [--full]   print the agent view (or the full spec)
  rlenv check-reference ID|all [--episodes N] [--eval-episodes N]
                                              run the reference solution end-to-end (slow)

  rlenv run ID [--agent scripted|replay|llm|nop|ADVERSARY] [--replay DIR] [--episodes N] [--out DIR]
                                              run one episode: trajectory + verification + reward
      --agent llm needs --allow-external-api and DEEPSEEK_API_KEY (calls DeepSeek; costs money)
  rlenv timeline DIR [--reasoning]            read-only event timeline of a trajectory
  rlenv evidence [--check]                     regenerate docs/EVIDENCE.md from committed artifacts
  rlenv llm-preflight [--model M] --allow-external-api
                                              GET /models: check the key and model id (no episode)
  rlenv calibrate ID|all [--seeds N] [--out calibration]
                                              reference over N seeds: pass rate, determinism, budgets
  rlenv redteam [ID] [--only NAME ...] [--out DIR]
                                              reference + floor + adversaries vs expectations
  rlenv check-trajectory DIR                  verify a trajectory (hash chain, head, blobs)
  rlenv rescore DIR                           recompute the reward from the verification record
  rlenv export DIR... -o FILE|DIR [--format messages|nemo-gym|atif] [--include-unscorable]
                                              derived training rows (integrity-checked first)
Later steps add: verify, rescore, calibrate.
Exit codes: 0 ok, 1 error, 2 a reference solution missed its threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rlenv.task import TaskError, load_tasks

DEFAULT_TASKS = "tasks"


def _cmd_list(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks)
    if not tasks:
        print(f"no tasks in {args.tasks}", file=sys.stderr)
        return 1
    for t in tasks.values():
        s, b = t.success, t.budget
        print(f"{t.id:<28} v{t.version}  {t.env_id:<16} {s.metric} >= {s.threshold:g} "
              f"over {s.eval_episodes} eps | {b.wall_clock_s:g}s, run<={b.train_timeout_s:g}s, "
              f"{b.max_runs} runs | {', '.join(m.value for m in t.methods.allowed)}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks)
    if args.id not in tasks:
        print(f"unknown task {args.id!r}; known: {sorted(tasks)}", file=sys.stderr)
        return 1
    task = tasks[args.id]
    data = task.model_dump(mode="json") if args.full else task.agent_view()
    print(json.dumps(data, indent=2))
    return 0


def _cmd_check_reference(args: argparse.Namespace) -> int:
    from rlenv.reference import check_reference

    tasks = load_tasks(args.tasks)
    ids = list(tasks) if args.id == "all" else [args.id]
    failed = 0
    for tid in ids:
        if tid not in tasks:
            print(f"unknown task {tid!r}", file=sys.stderr)
            return 1
        r = check_reference(tasks[tid], episodes=args.episodes, eval_episodes=args.eval_episodes)
        verdict = "PASS" if r.passed else "FAIL"
        mean = "n/a" if r.eval_mean is None else f"{r.eval_mean:.3f}"
        rnd = "n/a" if r.random_mean is None else f"{r.random_mean:.3f}"
        print(f"{verdict} {tid}: held-out mean {mean} (threshold {r.threshold:g}, random {rnd}), "
              f"train {r.train_seconds:.1f}s")
        if not r.passed:
            failed += 1
            print(f"  {r.detail}")
    return 0 if failed == 0 else 2


def _cmd_run(args: argparse.Namespace) -> int:
    from rlenv.agents.scripted import ScriptedAgent
    from rlenv.harness import run_episode

    tasks = load_tasks(args.tasks)
    if args.id not in tasks:
        print(f"unknown task {args.id!r}; known: {sorted(tasks)}", file=sys.stderr)
        return 1
    from rlenv.agents.adversarial import make_adversaries
    from rlenv.agents.replay import ReplayAgent

    adversaries = make_adversaries()
    if args.agent == "scripted":
        extra = ["--episodes", str(args.episodes)] if args.episodes else []
        agent = ScriptedAgent(args.id, extra_args=extra,
                              **({"reference_root": Path(args.reference_dir)} if args.reference_dir else {}))
    elif args.agent == "replay":
        if not args.replay:
            print("--agent replay needs --replay DIR", file=sys.stderr)
            return 1
        agent = ReplayAgent(args.replay)
    elif args.agent == "llm":
        from rlenv.agents.llm import API_KEY_ENV, HttpChatClient, LLMAgent, LLMError

        if not args.allow_external_api:
            print("--agent llm calls the DeepSeek API (network, costs money); pass --allow-external-api "
                  "to confirm", file=sys.stderr)
            return 1
        try:
            client = HttpChatClient()
        except LLMError as exc:
            print(f"ERROR: {exc} (export {API_KEY_ENV}; it is never written to disk)", file=sys.stderr)
            return 1
        try:
            agent = LLMAgent(client, model=args.model, thinking=args.thinking,
                             max_billable_tokens=args.max_billable_tokens, max_total_tokens=args.max_total_tokens)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    elif args.agent in adversaries:
        agent = adversaries[args.agent]()
    else:
        print(f"unknown agent {args.agent!r}; choose scripted, replay or one of {sorted(adversaries)}",
              file=sys.stderr)
        return 1
    record = run_episode(tasks[args.id], agent, root=args.out)
    outcome = record.outcome
    for e in outcome.events:
        if e["type"] == "tool_call":
            shown = {k: (v if k != "content" else f"<{len(v)} chars>") for k, v in e["args"].items()}
            print(f"  [{e['i']:>2}] {e['tool']} {shown}")
        elif e["type"] == "tool_result":
            d = e["data"]
            info = d.get("mean_return", d.get("run_id", d.get("path", "")))
            print(f"       -> {'ok' if e['ok'] else 'ERROR'} {info if e['ok'] else e['error']}")
    print(json.dumps(outcome.summary()))
    v, r = record.verification, record.reward
    print(f"verification: files_missing={v.files_missing} static={v.static_ok} rerun={v.rerun_ok} "
          f"held-out mean={v.eval_mean} (threshold {v.threshold:g}, random {v.random_mean}) report={v.report_ok}")
    print(f"reward: {r.reward} [{r.status}] {r.reason} components={r.components}")
    print(f"workspace: {outcome.workspace}")
    print(f"trajectory: {record.trajectory_dir / 'trajectory.jsonl'}")
    return 0 if r.status == "scored" else 2


def _cmd_rescore(args: argparse.Namespace) -> int:
    from rlenv.harness import rescore
    from rlenv.trajectory import TrajectoryError

    try:
        res = rescore(args.dir)
    except TrajectoryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"{'OK' if res.ok else 'MISMATCH'}: {res.detail}")
    print(f"  recorded:   {res.recorded}")
    print(f"  recomputed: {res.recomputed}")
    return 0 if res.ok else 2


def _cmd_export(args: argparse.Namespace) -> int:
    from rlenv.export import export

    dirs: list[Path] = []
    for d in args.dirs:
        p = Path(d)
        # Accept an episode directory or a root containing many episode directories.
        dirs.extend(sorted(x.parent for x in p.glob("*/trajectory.jsonl")) or [p])
    rep = export(dirs, args.output, args.format, args.include_unscorable)
    print(f"wrote {rep.written} row(s) to {args.output} ({args.format}); "
          f"skipped {rep.skipped_unscorable} unscorable, {len(rep.skipped_invalid)} invalid, "
          f"{len(rep.skipped_incomplete)} incomplete")
    for item in rep.skipped_invalid:
        print(f"  ! invalid: {item}")
    return 0 if not rep.skipped_invalid else 2


def _cmd_evidence(args: argparse.Namespace) -> int:
    from rlenv.evidence import EvidenceError, is_current, write

    try:
        if args.check:
            ok = is_current()
            print("docs/EVIDENCE.md is up to date" if ok else "docs/EVIDENCE.md is OUT OF DATE (run: rlenv evidence)")
            return 0 if ok else 1
        print(f"wrote {write()}")
    except EvidenceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


def _cmd_llm_preflight(args: argparse.Namespace) -> int:
    from rlenv.agents.llm import RETIRED_MODELS, HttpChatClient, LLMError

    if not args.allow_external_api:
        print("llm-preflight calls the DeepSeek API; pass --allow-external-api to confirm", file=sys.stderr)
        return 1
    try:
        models = HttpChatClient().list_models()
    except LLMError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"available models: {models}")
    if args.model in RETIRED_MODELS:
        print(f"{args.model!r} is a retired id (ADR-0011)", file=sys.stderr)
        return 2
    ok = args.model in models
    print(f"{args.model!r}: {'available' if ok else 'NOT available'}")
    return 0 if ok else 2


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from rlenv.calibrate import calibrate, write_report

    tasks = load_tasks(args.tasks)
    ids = list(tasks) if args.id == "all" else [args.id]
    rc = 0
    for tid in ids:
        if tid not in tasks:
            print(f"unknown task {tid!r}", file=sys.stderr)
            return 1
        rep = calibrate(tasks[tid], args.seeds,
                        **({"reference_root": Path(args.reference_dir)} if args.reference_dir else {}))
        js, md = write_report(rep, Path(args.out))
        state = "ADMITTED" if rep.admitted else "NOT ADMITTED: " + "; ".join(rep.reasons)
        frac = (f", ref/optimal min {rep.optimal_fraction['min']:.3f}" if rep.optimal_fraction else "")
        print(f"{tid}: {state} | pass {rep.pass_rate:.0%}, deterministic {rep.deterministic}, "
              f"score min {rep.eval_means.get('min', float('nan')):.3f}{frac}, "
              f"slowest re-run {rep.max_rerun_seconds:.1f}s -> {md}")
        rc = rc or (0 if rep.admitted else 2)
    return rc


def _cmd_redteam(args: argparse.Namespace) -> int:
    from rlenv.redteam import format_table, run_redteam

    tasks = load_tasks(args.tasks)
    if args.id not in tasks:
        print(f"unknown task {args.id!r}", file=sys.stderr)
        return 1
    rows = run_redteam(tasks[args.id], args.out, names=args.only,
                       include_reference=not args.no_reference,
                       **({"reference_root": Path(args.reference_dir)} if args.reference_dir else {}))
    print(format_table(rows))
    violations = [r for r in rows if r.verdict in ("VIOLATION", "GAP-CHANGED")]
    gaps = [r for r in rows if r.verdict == "GAP"]
    print(f"\n{len(rows)} agent(s): {len(violations)} violation(s), {len(gaps)} known gap(s)")
    return 2 if violations else 0


def _cmd_check_trajectory(args: argparse.Namespace) -> int:
    from rlenv.trajectory import TrajectoryError, load_trajectory

    try:
        traj = load_trajectory(args.dir)
    except TrajectoryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    h = traj.header
    kinds: dict[str, int] = {}
    for r in traj.records:
        kinds[r.get("type", "?")] = kinds.get(r.get("type", "?"), 0) + 1
    print(f"{'OK' if traj.ok else 'INVALID'}: {len(traj.records)} records, schema {h.get('schema')}, "
          f"episode {h.get('episode_id')}, task {h.get('task_id')} v{h.get('task_version')}")
    print(f"  records: {kinds}")
    for problem in traj.problems:
        print(f"  ! {problem}")
    return 0 if traj.ok else 2


def _cmd_timeline(args: argparse.Namespace) -> int:
    from rlenv.timeline import render
    from rlenv.trajectory import TrajectoryError, load_trajectory

    try:
        traj = load_trajectory(args.dir)
    except TrajectoryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("\n".join(render(traj, args.width, args.reasoning)))
    return 0 if traj.ok else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlenv", description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("list-tasks", help="validate and list tasks")
    p.add_argument("--tasks", default=DEFAULT_TASKS)
    p.set_defaults(func=_cmd_list)
    p = sub.add_parser("show-task", help="show a task as the agent sees it")
    p.add_argument("id")
    p.add_argument("--tasks", default=DEFAULT_TASKS)
    p.add_argument("--full", action="store_true", help="include harness-only fields (eval salt)")
    p.set_defaults(func=_cmd_show)
    p = sub.add_parser("check-reference", help="run a reference solution end-to-end (slow)")
    p.add_argument("id", help="task id or 'all'")
    p.add_argument("--tasks", default=DEFAULT_TASKS)
    p.add_argument("--episodes", type=int, help="override training episodes (fast smoke runs)")
    p.add_argument("--eval-episodes", type=int, help="override held-out evaluation episodes")
    p.set_defaults(func=_cmd_check_reference)
    p = sub.add_parser("run", help="run one agent episode")
    p.add_argument("id")
    p.add_argument("--tasks", default=DEFAULT_TASKS)
    p.add_argument("--agent", default="scripted",
                   help="scripted | replay | nop | random_floor | artifact_reuse | fabricated_report | "
                        "env_tamper | pickle_artifact | hardcoded_policy")
    p.add_argument("--replay", help="trajectory directory to replay (--agent replay)")
    p.add_argument("--model", default="deepseek-flash", help="--agent llm: DeepSeek model id")
    p.add_argument("--thinking", choices=["enabled", "disabled"], default="enabled",
                   help="--agent llm: DeepSeek thinking mode")
    p.add_argument("--max-billable-tokens", type=int, default=200_000,
                   help="--agent llm: cost budget in cache-miss input + output tokens")
    p.add_argument("--max-total-tokens", type=int, default=2_000_000,
                   help="--agent llm: loose hard cap on total tokens (incl. cached input)")
    p.add_argument("--allow-external-api", action="store_true",
                   help="--agent llm: confirm that this run calls the DeepSeek API")
    p.add_argument("--episodes", type=int, help="scripted agent: pass --episodes N to train.py (fast runs)")
    p.add_argument("--out", default="runs", help="episode root directory")
    p.add_argument("--reference-dir", help="scripted agent: alternative reference solutions (tests, demos)")
    p.set_defaults(func=_cmd_run)
    p = sub.add_parser("check-trajectory", help="verify a trajectory directory")
    p.add_argument("dir")
    p.set_defaults(func=_cmd_check_trajectory)
    p = sub.add_parser("timeline", help="one line per event: model calls, tool calls, verification, reward")
    p.add_argument("dir")
    p.add_argument("--width", type=int, default=110)
    p.add_argument("--reasoning", action="store_true", help="also show the model's reasoning (first words)")
    p.set_defaults(func=_cmd_timeline)
    p = sub.add_parser("rescore", help="recompute the reward of a trajectory")
    p.add_argument("dir")
    p.set_defaults(func=_cmd_rescore)
    p = sub.add_parser("export", help="export trajectories as training rows")
    p.add_argument("dirs", nargs="+", help="episode directories, or roots containing them")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--format", choices=["messages", "nemo-gym", "atif"], default="messages",
                   help="atif: -o is a directory, one ATIF v1.8 JSON document per episode")
    p.add_argument("--include-unscorable", action="store_true",
                   help="also emit unscorable episodes (reward null / mask_sample true)")
    p.set_defaults(func=_cmd_export)
    p = sub.add_parser("evidence", help="regenerate docs/EVIDENCE.md from committed real artifacts")
    p.add_argument("--check", action="store_true", help="exit 1 if docs/EVIDENCE.md is out of date")
    p.set_defaults(func=_cmd_evidence)
    p = sub.add_parser("llm-preflight", help="check the DeepSeek key and model id (one cheap API call)")
    p.add_argument("--model", default="deepseek-flash")
    p.add_argument("--allow-external-api", action="store_true")
    p.set_defaults(func=_cmd_llm_preflight)
    p = sub.add_parser("calibrate", help="calibrate a task with its reference solution over many seeds (slow)")
    p.add_argument("id", help="task id or 'all'")
    p.add_argument("--tasks", default=DEFAULT_TASKS)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--out", default="calibration")
    p.add_argument("--reference-dir", help="alternative reference solutions (tests)")
    p.set_defaults(func=_cmd_calibrate)
    p = sub.add_parser("redteam", help="run reference, floor and adversarial agents against the reward")
    p.add_argument("id", nargs="?", default="frozenlake-tabular")
    p.add_argument("--tasks", default=DEFAULT_TASKS)
    p.add_argument("--only", nargs="+", help="run only these adversaries")
    p.add_argument("--no-reference", action="store_true")
    p.add_argument("--reference-dir", help="alternative reference solutions (tests)")
    p.add_argument("--out", default="runs/redteam")
    p.set_defaults(func=_cmd_redteam)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except TaskError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
