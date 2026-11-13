"""Semi-gradient SARSA with tile coding for MountainCar-v0 (Sutton & Barto, section 10.1).

    python train.py --seed SEED --out DIR --env-kwargs JSON [--episodes N]

Writes DIR/weights.npy, DIR/tiles.json (coder config) and DIR/metrics.json.

Choices:
- 8 tilings of 8x8 with asymmetric offsets, as in S&B Example 10.1;
- epsilon = 0: zero-initialised weights are optimistic because every reward is -1;
- the step size decays linearly from 0.5/8 to 0.1/8 (a constant step size oscillates late);
- every 50 episodes the greedy policy is scored on 10 training-side seeds and the
  best weights are kept (checkpoint selection; the harness evaluates on its own seeds).
"""

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from tiles import TileCoder

N_TILINGS, N_TILES = 8, 8
ALPHA_START, ALPHA_END = 0.5, 0.1  # divided by N_TILINGS
GAMMA = 1.0
CHECK_EVERY, CHECK_EPISODES = 50, 10
MAX_STEPS = 1000


def greedy(qv, rng):
    return int(rng.choice(np.flatnonzero(qv == qv.max())))


def greedy_return(env, coder, w, seeds) -> float:
    returns = []
    for sd in seeds:
        x, _ = env.reset(seed=int(sd))
        total = 0.0
        for _ in range(200):  # the evaluation TimeLimit
            x, r, terminated, _, _ = env.step(int(coder.q_values(w, x).argmax()))
            total += r
            if terminated:
                break
        returns.append(total)
    return float(np.mean(returns))


def train(seed: int, episodes: int, env_kwargs: dict):
    rng = np.random.default_rng(seed)
    coder = TileCoder(N_TILINGS, N_TILES)
    env = gym.make("MountainCar-v0", **env_kwargs).unwrapped  # no TimeLimit during training (S&B setting)
    n_actions = env.action_space.n
    w = np.zeros(coder.shape(n_actions))
    check_seeds = rng.integers(0, 2**31 - 1, size=CHECK_EPISODES)
    best_score, best_w, best_ep = -np.inf, w.copy(), 0
    lengths, checks = [], []
    for ep in range(episodes):
        alpha = (ALPHA_START + (ALPHA_END - ALPHA_START) * ep / episodes) / N_TILINGS
        x, _ = env.reset(seed=seed + ep)
        q = coder.q_values(w, x)
        a = greedy(q, rng)
        for step in range(MAX_STEPS):
            x2, r, terminated, _, _ = env.step(a)
            rows, i, j = coder.indices(x)
            if terminated:
                w[rows, i, j, a] += alpha * (r - q[a])
                break
            q2 = coder.q_values(w, x2)
            a2 = greedy(q2, rng)
            w[rows, i, j, a] += alpha * (r + GAMMA * q2[a2] - q[a])
            x, a, q = x2, a2, q2
        lengths.append(step + 1)
        if (ep + 1) % CHECK_EVERY == 0:
            score = greedy_return(env, coder, w, check_seeds)
            checks.append([ep + 1, score])
            if score > best_score:
                best_score, best_w, best_ep = score, w.copy(), ep + 1
    return coder, best_w, {"lengths": lengths, "checks": checks, "best_score": best_score, "best_episode": best_ep}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--env-kwargs", default="{}")
    ap.add_argument("--episodes", type=int, default=1000)
    args = ap.parse_args()

    start = time.time()
    coder, w, info = train(args.seed, args.episodes, json.loads(args.env_kwargs))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "weights.npy", w)
    (out / "tiles.json").write_text(json.dumps({"n_tilings": coder.n_tilings, "n_tiles": coder.n_tiles}))
    metrics = {
        "method": "tile_coding + semi-gradient SARSA",
        "episodes": args.episodes,
        "tilings": f"{N_TILINGS} x {N_TILES}x{N_TILES}",
        "alpha": f"{ALPHA_START}/{N_TILINGS} -> {ALPHA_END}/{N_TILINGS}",
        "train_mean_steps_last_100": float(np.mean(info["lengths"][-100:])),
        "best_greedy_return_own_seeds": info["best_score"],
        "best_checkpoint_episode": info["best_episode"],
        "checks": info["checks"],
        "train_seconds": round(time.time() - start, 2),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "checks"}))


if __name__ == "__main__":
    main()
