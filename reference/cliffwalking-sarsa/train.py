"""SARSA for CliffWalking-v1 (Sutton & Barto, Example 6.6).

    python train.py --seed SEED --out DIR --env-kwargs JSON [--episodes N]

On-policy SARSA learns the "safe" path away from the cliff (greedy return -17),
unlike Q-learning's optimal but risky cliff-edge path (-13); both are reasonable
classic answers. Writes DIR/q.npy (48x4) and DIR/metrics.json.
"""

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

GAMMA = 1.0
ALPHA = 0.5
EPS_END = 0.0
EXPLORE_FRACTION = 0.8
MAX_STEPS = 200  # the task's step limit (CliffWalking-v1 has none of its own)


def train(seed: int, episodes: int, env_kwargs: dict):
    rng = np.random.default_rng(seed)
    env = gym.make("CliffWalking-v1", **env_kwargs).unwrapped
    q = np.zeros((env.observation_space.n, env.action_space.n))

    def pick(s, eps):
        if rng.random() < eps:
            return int(rng.integers(q.shape[1]))
        return int(rng.choice(np.flatnonzero(q[s] == q[s].max())))

    returns = []
    for ep in range(episodes):
        eps = max(EPS_END, 1.0 - ep / (EXPLORE_FRACTION * episodes))
        s, _ = env.reset(seed=seed + ep)
        a = pick(s, eps)
        total = 0.0
        for _ in range(MAX_STEPS):
            s2, r, terminated, _, _ = env.step(a)
            a2 = pick(s2, eps)
            target = r + (0.0 if terminated else GAMMA * q[s2, a2])
            q[s, a] += ALPHA * (target - q[s, a])
            s, a, total = s2, a2, total + r
            if terminated:
                break
        returns.append(total)
    return q, returns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--env-kwargs", default="{}")
    ap.add_argument("--episodes", type=int, default=1000)
    args = ap.parse_args()
    start = time.time()
    q, returns = train(args.seed, args.episodes, json.loads(args.env_kwargs))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "q.npy", q)
    metrics = {"method": "sarsa", "episodes": args.episodes, "alpha": ALPHA, "gamma": GAMMA,
               "train_return_last_100": float(np.mean(returns[-100:])), "train_seconds": round(time.time() - start, 2)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
