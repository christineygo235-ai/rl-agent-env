"""Audit signals (ADR-0009 option D): heuristics for human/LLM review, NEVER part of the reward.

They point a reviewer at submissions that may satisfy the checks without
learning (e.g. a hand-written policy). They are recorded in the verification
record and in exports; `compute_reward` does not read them (tested).

Flags:
- no_env_interaction: train.py never calls gym/gymnasium.make or .step(...)
- embedded_data_file: the workspace contains non-code data files (json/csv/txt/yaml)
  that train.py opens or mentions by name (could be a hand-written table)
- large_numeric_literal: a list/tuple literal with >= 16 numbers in the code
- no_learning_update: train.py has no augmented assignment (+=, -=, ...) and no
  subscript assignment on an array (no visible update rule)
"""

from __future__ import annotations

import ast
from pathlib import Path

DATA_SUFFIXES = {".json", ".csv", ".txt", ".yaml", ".yml", ".toml", ".cfg"}
LARGE_LITERAL = 16


def _calls(tree: ast.AST) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute):
                names.add(f.attr)
            elif isinstance(f, ast.Name):
                names.add(f.id)
    return names


def audit_workspace(root: str | Path) -> list[str]:
    root = Path(root)
    train = root / "train.py"
    if not train.is_file():
        return []
    source = train.read_text(errors="replace")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ["unparseable_train_py"]
    flags = []
    calls = _calls(tree)
    if not ({"make", "step"} <= calls):
        flags.append("no_env_interaction")
    data_files = [p.name for p in root.iterdir()
                  if p.is_file() and p.suffix in DATA_SUFFIXES and p.name not in ("REPORT.md",)]
    if any(name in source for name in data_files):
        flags.append("embedded_data_file")
    for p in root.rglob("*.py"):
        if p.is_symlink() or ".tmp" in p.parts or "runs" in p.parts:
            continue
        try:
            t = ast.parse(p.read_text(errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(t):
            if isinstance(node, (ast.List, ast.Tuple)) and \
                    sum(isinstance(e, ast.Constant) and isinstance(e.value, (int, float)) for e in node.elts) \
                    >= LARGE_LITERAL:
                flags.append("large_numeric_literal")
                break
        if "large_numeric_literal" in flags:
            break
    has_update = any(isinstance(n, ast.AugAssign) for n in ast.walk(tree)) or any(
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Subscript) for t in n.targets) for n in ast.walk(tree))
    if not has_update:
        flags.append("no_learning_update")
    return sorted(set(flags))
