"""Held-out environment instances (ADR-0009 option C).

A task may opt in to env variants: final verification then trains AND evaluates
on instances of the env family that the agent has never seen. A policy that
was hand-written for the canonical instance cannot pass; a learning method can.
This is "held-out seeds" extended to env parameters.

Supported families (registry below): FrozenLake random maps. Adding a family =
a generator for env kwargs + (optionally) an exact optimal policy, so that
thresholds can be expressed relative to what is achievable on each instance.

- dev variants (purpose "dev"): public, derived from the task seed; the agent can
  train/evaluate on them through the tools to test generalisation.
- verify variants (purpose "verify"): derived from the secret eval salt; never shown
  to the agent; only their sha256 fingerprints are logged.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable

import gymnasium as gym
import numpy as np

SEED_SPACE = 2**31 - 1


def _seed(purpose: str, salt: str, base_seed: int, i: int) -> int:
    digest = hashlib.sha256(f"variant|{purpose}|{salt}|{base_seed}|{i}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % SEED_SPACE


def kwargs_sha256(env_kwargs: dict) -> str:
    return hashlib.sha256(json.dumps(env_kwargs, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class VariantFamily:
    name: str
    env_ids: frozenset[str]
    generate: Callable[[dict, int, dict], dict]           # (canonical kwargs, seed, params) -> env kwargs
    optimal_policy: Callable[[str, dict], Callable[[Any], Any]] | None


# --------------------------------------------------------------------------- #
# FrozenLake random maps
# --------------------------------------------------------------------------- #

_MAP_SIZES = {"4x4": 4, "8x8": 8}


def _canonical_desc(env_id: str, kwargs: dict) -> list[str]:
    env = gym.make(env_id, **kwargs)
    desc = env.unwrapped.desc.astype(str).tolist()
    env.close()
    return ["".join(row) for row in desc]


def _frozenlake_generate(canonical: dict, seed: int, params: dict) -> dict:
    from gymnasium.envs.toy_text.frozen_lake import generate_random_map

    size = _MAP_SIZES.get(canonical.get("map_name", "4x4"), 4)
    desc = generate_random_map(size=size, p=float(params.get("frozen_p", 0.8)), seed=seed)
    kwargs = {k: v for k, v in canonical.items() if k != "map_name"}
    kwargs["desc"] = desc
    return kwargs


def _frozenlake_optimal(env_id: str, kwargs: dict, gamma: float = 0.999, tol: float = 1e-10):
    """Exact optimal policy by value iteration on the env's own transition model (harness-side)."""
    env = gym.make(env_id, **kwargs)
    P, n_s, n_a = env.unwrapped.P, env.observation_space.n, env.action_space.n
    env.close()
    v = np.zeros(n_s)
    for _ in range(100_000):
        q = np.array([[sum(p * (r + gamma * v[s2] * (not done)) for p, s2, r, done in P[s][a])
                       for a in range(n_a)] for s in range(n_s)])
        v_new = q.max(axis=1)
        if np.max(np.abs(v_new - v)) < tol:
            break
        v = v_new
    policy = q.argmax(axis=1)
    return lambda obs: int(policy[int(obs)])


FAMILIES: dict[str, VariantFamily] = {
    "frozenlake_random_map": VariantFamily(
        "frozenlake_random_map", frozenset({"FrozenLake-v1", "FrozenLake8x8-v1"}),
        _frozenlake_generate, _frozenlake_optimal),
}


def supported(kind: str, env_id: str) -> bool:
    return kind in FAMILIES and env_id in FAMILIES[kind].env_ids


def variant_kwargs(kind: str, env_id: str, canonical: dict, purpose: str, salt: str, base_seed: int,
                   count: int, params: dict) -> list[dict]:
    """`count` distinct instances, all different from the canonical one (and from each other)."""
    fam = FAMILIES[kind]
    canon_desc = _canonical_desc(env_id, canonical)
    out, seen, i = [], set(), 0
    while len(out) < count:
        kw = fam.generate(canonical, _seed(purpose, salt, base_seed, i), params)
        i += 1
        key = kwargs_sha256(kw)
        if kw.get("desc") == canon_desc or key in seen:
            continue
        seen.add(key)
        out.append(kw)
    return out


def optimal_policy(kind: str, env_id: str, kwargs: dict) -> Callable[[Any], Any] | None:
    fam = FAMILIES[kind]
    return fam.optimal_policy(env_id, kwargs) if fam.optimal_policy else None
