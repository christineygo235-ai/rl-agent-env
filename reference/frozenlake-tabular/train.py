"""Tabular Q-learning for FrozenLake-v1 (4x4, slippery). Sutton & Barto, section 6.5.

    python train.py --seed SEED --out DIR --env-kwargs JSON [--episodes N]

Trains on gymnasium.make("FrozenLake-v1", **env_kwargs), so it works on any map
(the task's map or an unseen one). Writes DIR/q.npy (n_states x n_actions) and
DIR/metrics.json.
"""

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

GAMMA = 0.99
ALPHA_START, ALPHA_END = 0.2, 0.01
EPS_END = 0.05
EXPLORE_FRACTION = 0.6  # epsilon decays linearly over the first 60 % of episodes
MAX_STEPS = 100


def greedy(q_row, rng):
    best = np.flatnonzero(q_row == q_row.max())
    return int(rng.choice(best))  # random tie-breaking: avoids a bias towards action 0 early on


def train(seed: int, episodes: int, env_kwargs: dict):
    rng = np.random.default_rng(seed)
    env = gym.make("FrozenLake-v1", **env_kwargs).unwrapped  # speed; own step cap below
    n_s, n_a = env.observation_space.n, env.action_space.n
    q = np.zeros((n_s, n_a))
    successes = []
    for ep in range(episodes):
        eps = max(EPS_END, 1.0 - ep / (EXPLORE_FRACTION * episodes))
        alpha = max(ALPHA_END, ALPHA_START * (1.0 - ep / episodes))
        s, _ = env.reset(seed=seed + ep)
        reward = 0.0
        for _ in range(MAX_STEPS):
            a = int(rng.integers(n_a)) if rng.random() < eps else greedy(q[s], rng)
            s2, reward, terminated, _, _ = env.step(a)
            target = reward + (0.0 if terminated else GAMMA * q[s2].max())
            q[s, a] += alpha * (target - q[s, a])
            s = s2
            if terminated:
                break
        successes.append(reward)
    return q, successes


def evaluate_own(q, seed: int, env_kwargs: dict, episodes: int = 500) -> float:
    """Quick greedy check on training-side seeds (the harness evaluates independently)."""
    env = gym.make("FrozenLake-v1", **env_kwargs)
    total = 0.0
    for i in range(episodes):
        s, _ = env.reset(seed=seed * 7919 + i)
        done = False
        while not done:
            s, r, terminated, truncated, _ = env.step(int(q[s].argmax()))
            total += r
            done = terminated or truncated
    return total / episodes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--env-kwargs", default='{"map_name": "4x4", "is_slippery": true}')
    ap.add_argument("--episodes", type=int, default=30000)
    args = ap.parse_args()
    env_kwargs = json.loads(args.env_kwargs)

    start = time.time()
    q, successes = train(args.seed, args.episodes, env_kwargs)
    own = evaluate_own(q, args.seed, env_kwargs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "q.npy", q)
    metrics = {
        "method": "tabular_q_learning",
        "episodes": args.episodes,
        "gamma": GAMMA,
        "train_success_last_1000": float(np.mean(successes[-1000:])),
        "greedy_success_own_seeds": own,
        "train_seconds": round(time.time() - start, 2),
        "policy": q.argmax(axis=1).tolist(),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
