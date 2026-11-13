"""Per-episode workspace: the only place agent tools may read or write.

Every agent-supplied path goes through `Workspace.resolve`:
- relative paths only, no NUL, no `..` components;
- the fully resolved path (symlinks followed) must stay inside the workspace.
  This matters because agent code runs inside the workspace and could create
  a symlink to /etc/passwd or ~/.ssh, then ask `read_file` to follow it.
- writes refuse to follow a symlink in the final component (O_NOFOLLOW).
Tool calls and agent code never run concurrently (runs are synchronous), so
there is no window to swap a checked path between check and use.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

TEXT_SUFFIXES = frozenset({".py", ".md", ".txt", ".json", ".yaml", ".yml", ".cfg", ".toml", ".csv"})
MAX_WRITE_BYTES = 256 * 1024
MAX_READ_BYTES = 256 * 1024
MAX_FILES = 500


class WorkspaceError(Exception):
    """A rejected workspace operation (message is safe to show to the agent)."""


@dataclass(frozen=True)
class Entry:
    path: str
    kind: str  # file | dir | symlink | other
    size: int


class Workspace:
    def __init__(self, root: str | Path, create: bool = True) -> None:
        root = Path(root)
        if create:
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root = root.resolve(strict=True)

    # ------------------------------------------------------------------ paths

    def resolve(self, rel: str) -> Path:
        if not isinstance(rel, str) or not rel or "\x00" in rel:
            raise WorkspaceError("path must be a non-empty string without NUL bytes")
        pure = PurePosixPath(rel)
        if pure.is_absolute() or rel.startswith("~"):
            raise WorkspaceError(f"{rel!r}: absolute paths are not allowed; use a path relative to the workspace")
        if any(part == ".." for part in pure.parts):
            raise WorkspaceError(f"{rel!r}: '..' is not allowed")
        target = (self.root / pure).resolve(strict=False)
        if target != self.root and self.root not in target.parents:
            raise WorkspaceError(f"{rel!r} resolves outside the workspace")
        return target

    def rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    # --------------------------------------------------------------- file ops

    def write_text(self, rel: str, content: str) -> Path:
        path = self.resolve(rel)
        if path.suffix not in TEXT_SUFFIXES:
            raise WorkspaceError(f"{rel!r}: only text files can be written ({', '.join(sorted(TEXT_SUFFIXES))})")
        data = content.encode("utf-8")
        if len(data) > MAX_WRITE_BYTES:
            raise WorkspaceError(f"{rel!r}: {len(data)} bytes exceeds the {MAX_WRITE_BYTES} byte limit")
        if not path.exists() and self.count_files() >= MAX_FILES:
            raise WorkspaceError(f"workspace already has {MAX_FILES} files")
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.root not in path.parent.resolve().parents and path.parent.resolve() != self.root:
            raise WorkspaceError(f"{rel!r}: parent directory resolves outside the workspace")
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags, 0o600)
        except OSError as exc:
            raise WorkspaceError(f"{rel!r}: cannot write ({exc.strerror})") from exc
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        return path

    def read_text(self, rel: str, max_bytes: int = MAX_READ_BYTES) -> tuple[str, bool]:
        """Return (text, truncated)."""
        path = self.resolve(rel)
        if not path.is_file():
            raise WorkspaceError(f"{rel!r}: no such file")
        with open(path, "rb") as fh:
            data = fh.read(max_bytes + 1)
        truncated = len(data) > max_bytes
        return data[:max_bytes].decode("utf-8", errors="replace"), truncated

    def list(self, rel: str = ".") -> list[Entry]:
        base = self.resolve(rel)
        if not base.is_dir():
            raise WorkspaceError(f"{rel!r}: not a directory")
        out = []
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            dirnames.sort()
            for name in sorted(dirnames) + sorted(filenames):
                p = Path(dirpath) / name
                st = p.lstat()
                kind = ("symlink" if stat.S_ISLNK(st.st_mode) else "dir" if stat.S_ISDIR(st.st_mode)
                        else "file" if stat.S_ISREG(st.st_mode) else "other")
                out.append(Entry(p.relative_to(self.root).as_posix(), kind, st.st_size if kind == "file" else 0))
        return out

    def count_files(self) -> int:
        return sum(1 for e in self.list() if e.kind == "file")

    # -------------------------------------------------------------- snapshots

    def snapshot(self, suffixes: frozenset[str] = TEXT_SUFFIXES,
                 exclude_prefixes: tuple[str, ...] = ("runs/", ".tmp/")) -> dict[str, str]:
        """sha256 of every regular (non-symlink) text file outside harness-owned
        directories: this is the code version."""
        result = {}
        for e in self.list():
            if e.path.startswith(exclude_prefixes):
                continue
            if e.kind == "file" and PurePosixPath(e.path).suffix in suffixes:
                result[e.path] = hashlib.sha256((self.root / e.path).read_bytes()).hexdigest()
        return result

    def copy_clean(self, dest: str | Path, suffixes: frozenset[str] = TEXT_SUFFIXES) -> Workspace:
        """Copy only regular source files into a fresh workspace (for verification).
        Symlinks, artifacts and anything else the agent left behind are dropped."""
        dest = Path(dest)
        if dest.exists():
            raise WorkspaceError(f"{dest} already exists")
        dest.mkdir(parents=True, mode=0o700)
        for e in self.list():
            if e.kind == "file" and PurePosixPath(e.path).suffix in suffixes:
                target = dest / e.path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.root / e.path, target, follow_symlinks=False)
        return Workspace(dest, create=False)
