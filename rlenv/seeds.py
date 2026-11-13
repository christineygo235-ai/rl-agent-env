"""Seed derivation for evaluation.

- held-out eval seeds: derived from (task seed, secret salt). The agent never
  sees the salt, so it cannot compute them or overfit to them.
- dev seeds: derived from the task seed with a public purpose string. The
  agent's `evaluate` tool uses them for iteration.
The two sets are disjoint (overlaps are re-drawn deterministically).
"""

from __future__ import annotations

import hashlib

SEED_SPACE = 2**31 - 1


def _draw(purpose: str, salt: str, base_seed: int, i: int) -> int:
    digest = hashlib.sha256(f"{purpose}|{salt}|{base_seed}|{i}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % SEED_SPACE


def derive_seeds(purpose: str, salt: str, base_seed: int, n: int, exclude: frozenset[int] = frozenset()) -> list[int]:
    seeds: list[int] = []
    seen = set(exclude)
    i = 0
    while len(seeds) < n:
        s = _draw(purpose, salt, base_seed, i)
        i += 1
        if s not in seen:
            seen.add(s)
            seeds.append(s)
    return seeds


def dev_seeds(base_seed: int, n: int) -> list[int]:
    return derive_seeds("dev", "public", base_seed, n)


def eval_seeds(base_seed: int, salt: str, n: int, dev_n: int) -> list[int]:
    """Held-out seeds, guaranteed disjoint from the first `dev_n` dev seeds and the training seed."""
    exclude = frozenset(dev_seeds(base_seed, dev_n)) | {base_seed}
    return derive_seeds("eval", salt, base_seed, n, exclude)


def seeds_sha256(seeds: list[int]) -> str:
    """Fingerprint for trajectories: proves which seeds were used without revealing them."""
    return hashlib.sha256(",".join(map(str, seeds)).encode()).hexdigest()
