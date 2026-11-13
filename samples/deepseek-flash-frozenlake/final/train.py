"""Tabular RL on FrozenLake-v1 -- Expected SARSA + greedy refinement phase.

    python train.py --seed SEED --out DIR --env-kwargs JSON [options]

The map comes from --env-kwargs ({"desc": [...]} for random maps, or
{"map_name": "4x4"}), so the same code handles the default map and unseen maps.

Artefacts written into DIR (no pickle):
    q.npy     float64 Q-table of shape (n_states, n_actions)
    meta.json map description + hyper-parameters + measured candidate scores

Method (tabular, numpy only)
---------------------------
* Expected SARSA control (Sutton & Barto sec. 6.6) with an epsilon-greedy
  behaviour policy:
      Q(s,a) <- Q(s,a) + alpha * [r + gamma * sum_a' pi(a'|s') Q(s',a') - Q(s,a)]
* Two phases on the same table:
    A) exploration: epsilon linearly 1.0 -> eps_end, alpha = 0.1
    B) refinement : epsilon ~ 0.01 with a small alpha, i.e. almost on-policy
                    evaluation of the greedy policy interleaved with greedy
                    improvement.  This removes the epsilon-soft bias of the
                    phase-A values, which matters on narrow maps where one
                    wrong action is fatal.
* `learners` independent learners (different RNG seeds per learner) are
  trained; every phase-A and phase-B table plus the mean of the phase-B tables
  is a candidate policy, and the candidate with the best success rate on fresh
  rollout episodes (common/paired evaluation seeds) is saved.
* Alternatives `tabular_q_learning` and `sarsa` are available via --method.
* A soft wall-clock guard skips later learners if the budget runs low, and a
  try/except fallback guarantees that a usable Q-table is always written.
"""
import argparse
import json
import os
import time

import numpy as np
import gymnasium as gym


# ---------------------------------------------------------------- utilities
def make_env(env_kwargs, seed=None):
    """gymnasium FrozenLake; `seed` only makes the rollouts repeatable."""
    env = gym.make("FrozenLake-v1", disable_env_checker=True, **env_kwargs)
    if seed is not None:
        env.reset(seed=int(seed))
        env.action_space.seed(int(seed) + 1)
    return env


def epsilon_at(ep, episodes, eps_start, eps_end, frac):
    """Linear decay eps_start -> eps_end over frac*episodes, then constant."""
    span = max(1.0, frac * episodes)
    f = min(1.0, ep / span)
    return eps_start + (eps_end - eps_start) * f


def update(Q, s, a, r, s2, term, trunc, eps, gamma, alpha, method, rng):
    """One tabular TD update of Q[s, a] in place."""
    if term:
        target = r
    else:
        qrow = Q[s2]
        if method == "tabular_q_learning":
            target = r + gamma * qrow.max()
        elif method == "sarsa":
            if rng.random() < eps:
                a2 = int(rng.integers(qrow.shape[0]))
            else:
                a2 = int(np.argmax(qrow))
            target = r + gamma * qrow[a2]
        else:  # expected_sarsa (default)
            target = r + gamma * ((1.0 - eps) * qrow.max()
                                  + (eps / qrow.shape[0]) * qrow.sum())
    Q[s, a] += alpha * (target - Q[s, a])


def run_phase(env, Q, rng, episodes, alpha, gamma, method, eps_start, eps_end,
              decay_frac, deadline=None):
    """Run epsilon-greedy control episodes, updating Q in place."""
    nA = Q.shape[1]
    for ep in range(episodes):
        if deadline is not None and time.time() > deadline:
            break
        eps = epsilon_at(ep, episodes, eps_start, eps_end, decay_frac)
        s, _ = env.reset()
        s = int(s)
        done = False
        while not done:
            if rng.random() < eps:
                a = int(rng.integers(nA))
            else:
                a = int(np.argmax(Q[s]))
            s2, r, term, trunc, _ = env.step(a)
            s2 = int(s2)
            update(Q, s, a, r, s2, term, trunc, eps, gamma, alpha, method, rng)
            s = s2
            done = term or trunc


def train_one(env_kwargs, seed, episodes, alpha, gamma, eps_start, eps_end,
              decay_frac, method, refine_episodes, alpha_b, eps_b,
              deadline=None):
    """One learner -> (Q after phase A, Q after phase B)."""
    env = make_env(env_kwargs, seed=seed)
    Q = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)
    rng = np.random.default_rng(seed)
    run_phase(env, Q, rng, episodes, alpha, gamma, method, eps_start, eps_end,
              decay_frac, deadline)
    QA = Q.copy()
    if refine_episodes > 0:
        run_phase(env, Q, rng, refine_episodes, alpha_b, gamma, method, eps_b,
                  eps_b, 1.0, deadline)
    env.close()
    return QA, Q


def greedy_success(Q, env_kwargs, n=600, seed=50000):
    """Fraction of successful episodes for the greedy policy on Q."""
    env = make_env(env_kwargs, seed=seed)
    wins = 0
    for _ in range(n):
        s, _ = env.reset()
        done = False
        while not done:
            s, r, term, trunc, _ = env.step(int(np.argmax(Q[int(s)])))
            s = int(s)
            done = term or trunc
            if term and r > 0:
                wins += 1
    env.close()
    return wins / n


def select_and_meta(env_kwargs, args, deadline):
    """Train the learners, pick the best candidate; returns (Q, info dict)."""
    cands = {}
    n_done = 0
    for k in range(max(1, args.learners)):
        if n_done >= 1 and time.time() > deadline:
            print(f"time guard: skipping learner {k}")
            break
        tk = time.time()
        QA, QB = train_one(env_kwargs, args.seed + 1000 * k, args.episodes,
                           args.alpha, args.gamma, args.eps_start,
                           args.eps_end, args.decay_frac, args.method,
                           args.refine_episodes, args.alpha_b, args.eps_b,
                           deadline)
        print(f"learner {k}: {time.time()-tk:.1f}s")
        cands[f"a{k}"] = QA
        cands[f"b{k}"] = QB
        n_done += 1
    if n_done > 1:
        cands["meanB"] = np.mean(np.stack([cands[f"b{k}"]
                                           for k in range(n_done)], 0), 0)

    # rollout-based policy selection with common (paired) evaluation seeds
    scores = {}
    for name in sorted(cands):
        scores[name] = greedy_success(cands[name], env_kwargs,
                                      n=args.select_episodes, seed=50000)
        print(f"select {name}: {scores[name]:.3f}")
    best_name = max(scores, key=lambda k: scores[k])
    print(f"chosen={best_name} score={scores[best_name]:.3f}")
    return cands[best_name], {"chosen": best_name,
                              "scores": {k: float(v) for k, v in scores.items()},
                              "learners": n_done}


def fallback_train(env_kwargs, args, deadline):
    """Last-resort single Expected-SARSA learner (same update rule)."""
    QA, QB = train_one(env_kwargs, args.seed, args.episodes, args.alpha,
                       args.gamma, args.eps_start, args.eps_end,
                       args.decay_frac, args.method, args.refine_episodes,
                       args.alpha_b, args.eps_b, deadline)
    return QB, {"chosen": "fallback", "scores": {}, "learners": 1}


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--env-kwargs", type=str, default="{}")
    # phase A: exploration
    p.add_argument("--episodes", type=int, default=40000)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--decay-frac", type=float, default=0.4)
    # phase B: greedy refinement
    p.add_argument("--refine-episodes", type=int, default=20000)
    p.add_argument("--alpha-b", type=float, default=0.02)
    p.add_argument("--eps-b", type=float, default=0.01)
    # bookkeeping / safety
    p.add_argument("--learners", type=int, default=3)
    p.add_argument("--method", type=str, default="expected_sarsa")
    p.add_argument("--select-episodes", type=int, default=800)
    p.add_argument("--max-train-seconds", type=float, default=80.0)
    args = p.parse_args()

    env_kwargs = json.loads(args.env_kwargs) if args.env_kwargs else {}
    env_kwargs.setdefault("is_slippery", True)
    if not env_kwargs.get("desc"):
        env_kwargs.pop("desc", None)
        env_kwargs.setdefault("map_name", "4x4")

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    deadline = t0 + args.max_train_seconds

    try:
        best_Q, info = select_and_meta(env_kwargs, args, deadline)
    except Exception as exc:  # never leave the caller without a policy
        print("fallback after error:", repr(exc))
        best_Q, info = fallback_train(env_kwargs, args, deadline)

    np.save(os.path.join(args.out, "q.npy"), np.asarray(best_Q, dtype=np.float64))

    desc = None
    try:
        tmp = make_env(env_kwargs)
        desc = ["".join(map(str, row)) for row in tmp.unwrapped.desc]
        tmp.close()
    except Exception:
        desc = None

    meta = {
        "method": args.method,
        "seed": args.seed,
        "episodes": args.episodes,
        "refine_episodes": args.refine_episodes,
        "alpha": args.alpha,
        "alpha_b": args.alpha_b,
        "gamma": args.gamma,
        "eps_start": args.eps_start,
        "eps_end": args.eps_end,
        "eps_b": args.eps_b,
        "decay_frac": args.decay_frac,
        "nS": int(best_Q.shape[0]),
        "nA": int(best_Q.shape[1]),
        "desc": desc,
        "env_kwargs": {k: v for k, v in env_kwargs.items() if k != "desc"},
    }
    meta.update(info)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("total train time %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
