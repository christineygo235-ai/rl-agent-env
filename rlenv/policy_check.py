"""Static checks on agent code (AST based).

Purpose: enforce the task's rules (classic methods, numpy-only, data-only
artifacts) and reject obviously dangerous code before it runs. This is a
policy filter, NOT a security boundary: Python is too dynamic to prove safety
statically. The sandbox and the harness-owned evaluator do the real work.

Rules:
- imports must be on the allowlist, or be a module in the workspace itself;
  torch only when the task allows it;
- explicitly forbidden: process/network/FFI/serialisation/dynamic-import
  modules and RL libraries (clear message for each);
- forbidden calls: eval, exec, compile, __import__, os.system and friends;
- forbidden escape hatches: getattr/setattr/delattr with a non-constant name,
  dunder attribute access such as __subclasses__ / __globals__ / __builtins__;
- `allow_pickle=True` anywhere (artifacts must be data).
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from pathlib import Path

ALLOWED_MODULES = frozenset({
    "__future__", "abc", "argparse", "bisect", "collections", "copy", "csv", "dataclasses",
    "datetime", "enum", "functools", "gymnasium", "hashlib", "heapq", "itertools", "json",
    "logging", "math", "numpy", "os", "pathlib", "random", "re", "statistics", "string",
    "struct", "sys", "time", "typing", "warnings",
})
TORCH_MODULES = frozenset({"torch"})

FORBIDDEN_MODULES = {
    "subprocess": "starting processes is not allowed",
    "multiprocessing": "starting processes is not allowed",
    "threading": "threads are not allowed (determinism, fair budgets)",
    "concurrent": "threads/processes are not allowed",
    "socket": "network access is not allowed",
    "ssl": "network access is not allowed",
    "http": "network access is not allowed",
    "urllib": "network access is not allowed",
    "requests": "network access is not allowed",
    "asyncio": "network/event loops are not allowed",
    "ctypes": "native code (FFI) is not allowed",
    "cffi": "native code (FFI) is not allowed",
    "pickle": "pickle is not allowed: artifacts must be data (npy/npz/json)",
    "dill": "pickle-like serialisation is not allowed",
    "joblib": "pickle-like serialisation is not allowed",
    "marshal": "code serialisation is not allowed",
    "shelve": "pickle-like serialisation is not allowed",
    "importlib": "dynamic imports are not allowed",
    "builtins": "patching builtins is not allowed",
    "inspect": "introspection of the harness is not allowed",
    "gc": "introspection of the harness is not allowed",
    "signal": "signal handling is not allowed",
    "resource": "changing resource limits is not allowed",
    "shutil": "bulk file operations are not allowed",
    "stable_baselines3": "RL libraries are not allowed: implement the method yourself",
    "sb3_contrib": "RL libraries are not allowed: implement the method yourself",
    "ray": "RL libraries are not allowed: implement the method yourself",
    "tianshou": "RL libraries are not allowed: implement the method yourself",
    "cleanrl": "RL libraries are not allowed: implement the method yourself",
    "tensorflow": "only numpy (and torch where the task allows it) may be used",
    "jax": "only numpy (and torch where the task allows it) may be used",
    "sklearn": "only numpy (and torch where the task allows it) may be used",
}

FORBIDDEN_BUILTINS = {"eval", "exec", "compile", "__import__", "breakpoint", "globals", "vars"}
FORBIDDEN_OS_ATTRS = {
    "system", "popen", "fork", "forkpty", "kill", "killpg", "putenv", "setuid", "setgid", "chroot",
    "symlink", "link", "execl", "execle", "execlp", "execlpe", "execv", "execve", "execvp", "execvpe",
    "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe",
    "posix_spawn", "posix_spawnp",
}
FORBIDDEN_DUNDERS = {
    "__subclasses__", "__globals__", "__builtins__", "__code__", "__closure__", "__mro__",
    "__bases__", "__base__", "__dict__", "__class__", "__getattribute__", "__loader__", "__spec__",
}
DYNAMIC_ATTR_FUNCS = {"getattr", "setattr", "delattr", "hasattr"}


@dataclass(frozen=True)
class Violation:
    file: str
    line: int
    rule: str
    message: str


@dataclass
class CheckReport:
    ok: bool
    files: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)

    def summary(self, limit: int = 20) -> str:
        if self.ok:
            return f"ok: {len(self.files)} file(s), imports {sorted(set(self.imports))}"
        lines = [f"{v.file}:{v.line}: [{v.rule}] {v.message}" for v in self.violations[:limit]]
        if len(self.violations) > limit:
            lines.append(f"... {len(self.violations) - limit} more")
        return "\n".join(lines)


class _Visitor(ast.NodeVisitor):
    def __init__(self, file: str, allowed: frozenset[str], local_modules: frozenset[str]) -> None:
        self.file, self.allowed, self.local = file, allowed, local_modules
        self.violations: list[Violation] = []
        self.imports: list[str] = []
        self.os_aliases: set[str] = set()  # names bound to the os module
        self.os_funcs: set[str] = set()    # names imported from os that are forbidden

    def _v(self, node: ast.AST, rule: str, msg: str) -> None:
        self.violations.append(Violation(self.file, getattr(node, "lineno", 0), rule, msg))

    def _check_module(self, node: ast.AST, module: str) -> None:
        top = module.split(".")[0]
        self.imports.append(top)
        if top in FORBIDDEN_MODULES:
            self._v(node, "forbidden-import", f"import {module}: {FORBIDDEN_MODULES[top]}")
        elif top in TORCH_MODULES and top not in self.allowed:
            self._v(node, "forbidden-import", f"import {module}: torch is not enabled for this task (numpy only)")
        elif top not in self.allowed and top not in self.local:
            self._v(node, "import-not-allowed", f"import {module}: not on the allowlist")

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_module(node, alias.name)
            if alias.name == "os":
                self.os_aliases.add(alias.asname or "os")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:  # relative import inside the workspace package
            self.generic_visit(node)
            return
        module = node.module or ""
        self._check_module(node, module)
        if module == "os":
            for alias in node.names:
                if alias.name in FORBIDDEN_OS_ATTRS or alias.name == "*":
                    self._v(node, "forbidden-call", f"from os import {alias.name}: not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in FORBIDDEN_BUILTINS:
                self._v(node, "forbidden-call", f"{func.id}() is not allowed")
            if func.id in DYNAMIC_ATTR_FUNCS and len(node.args) >= 2 and not isinstance(node.args[1], ast.Constant):
                self._v(node, "dynamic-attribute", f"{func.id}() with a computed attribute name is not allowed")
            if func.id == "open" and node.args and isinstance(node.args[0], ast.Constant) \
                    and str(node.args[0].value).startswith("/"):
                self._v(node, "absolute-path", "open() of an absolute path is not allowed")
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) \
                and func.value.id in self.os_aliases and func.attr in FORBIDDEN_OS_ATTRS:
            self._v(node, "forbidden-call", f"os.{func.attr}() is not allowed")
        for kw in node.keywords:
            if kw.arg == "allow_pickle" and not (isinstance(kw.value, ast.Constant) and kw.value.value is False):
                self._v(node, "pickle", "allow_pickle=True is not allowed: artifacts must be data")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in FORBIDDEN_DUNDERS:
            self._v(node, "dunder-access", f"access to {node.attr} is not allowed")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in ("__builtins__", "__loader__", "__spec__"):
            self._v(node, "dunder-access", f"access to {node.id} is not allowed")
        self.generic_visit(node)


def check_source(source: str, file: str = "<string>", allow_torch: bool = False,
                 local_modules: frozenset[str] = frozenset()) -> CheckReport:
    allowed = ALLOWED_MODULES | (TORCH_MODULES if allow_torch else frozenset())
    try:
        tree = ast.parse(source, filename=file)
    except SyntaxError as exc:
        return CheckReport(False, [file], [], [Violation(file, exc.lineno or 0, "syntax", str(exc.msg))])
    visitor = _Visitor(file, allowed, local_modules)
    visitor.visit(tree)
    return CheckReport(not visitor.violations, [file], visitor.imports, visitor.violations)


def check_workspace(root: str | Path, allow_torch: bool = False,
                    required: tuple[str, ...] = ()) -> CheckReport:
    """Check every .py file in the workspace (symlinks are rejected, not followed)."""
    root = Path(root)
    report = CheckReport(True)
    py_files = sorted(p for p in root.rglob("*.py") if ".tmp" not in p.parts)
    local = frozenset(p.stem for p in py_files if not p.is_symlink())
    for req in required:
        if not (root / req).is_file():
            report.violations.append(Violation(req, 0, "missing-file", f"{req} is required"))
    for path in py_files:
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            report.violations.append(Violation(rel, 0, "symlink", "symlinked source files are not allowed"))
            continue
        sub = check_source(path.read_text(encoding="utf-8", errors="replace"), rel, allow_torch, local)
        report.files.append(rel)
        report.imports.extend(sub.imports)
        report.violations.extend(sub.violations)
    report.ok = not report.violations
    return report
