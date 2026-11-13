"""Trajectories: append-only, hash-chained JSONL + content-addressed blobs.

Layout per episode:
    <root>/<episode_id>/trajectory.jsonl   one record per line
    <root>/<episode_id>/trajectory.head    sha256 of the last line (written on close)
    <root>/<episode_id>/blobs/<sha256>     code versions and large strings

Every record carries `seq` and `prev` = sha256 of the previous line's bytes
(64 zeros for the header). Changing any line breaks the chain at the next one;
the head file catches truncation and changes to the last line.

Before anything is written, secret values are redacted: values of env vars
whose names look secret (DEEPSEEK_API_KEY, *_TOKEN, ...) and `sk-...` style
keys. Trajectories can therefore be shared with training pipelines.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rlenv

SCHEMA_VERSION = "rlenv.trajectory/v1"
GENESIS = "0" * 64
LARGE_STRING = 8192  # strings longer than this go to the blob store
_SECRET_NAME = re.compile(r"(API_KEY|SECRET|TOKEN|PASSWORD|ACCESS_KEY|PRIVATE_KEY)", re.I)
_SECRET_PATTERNS = [re.compile(r"sk-[A-Za-z0-9_\-]{16,}")]
ALWAYS_SECRET = ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")


class TrajectoryError(Exception):
    pass


# --------------------------------------------------------------------------- #
# Redaction
# --------------------------------------------------------------------------- #


class Redactor:
    def __init__(self, environ: dict[str, str] | None = None, home: str | Path | None = None) -> None:
        env = dict(os.environ if environ is None else environ)
        # Local paths: the user's home directory becomes "~" (trajectories are shared as evidence;
        # the harness never reads paths back from them).
        h = str(Path.home() if home is None else home).rstrip("/")
        self.home = h if len(h) > 1 else None
        self.secrets: dict[str, str] = {}
        for name, value in env.items():
            if (name in ALWAYS_SECRET or _SECRET_NAME.search(name)) and value and len(value) >= 8:
                self.secrets[value] = name
        self.count = 0

    def __call__(self, text: str) -> str:
        for value, name in sorted(self.secrets.items(), key=lambda kv: -len(kv[0])):
            if value in text:
                self.count += text.count(value)
                text = text.replace(value, f"[REDACTED:{name}]")
        for pat in _SECRET_PATTERNS:
            text, n = pat.subn("[REDACTED:pattern]", text)
            self.count += n
        if self.home and self.home in text:
            text = text.replace(self.home + "/", "~/")
            if text == self.home:
                text = "~"
        return text


# --------------------------------------------------------------------------- #
# Blob store
# --------------------------------------------------------------------------- #


class BlobStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, data: str | bytes) -> str:
        raw = data.encode("utf-8") if isinstance(data, str) else data
        digest = hashlib.sha256(raw).hexdigest()
        path = self.root / digest
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise TrajectoryError(f"blob {digest} is corrupted")
            return digest
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(raw)
        tmp.replace(path)
        return digest

    def get(self, digest: str) -> bytes:
        path = self.root / digest
        if not path.exists():
            raise TrajectoryError(f"missing blob {digest}")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise TrajectoryError(f"blob {digest} does not match its hash")
        return raw


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #


def _git_commit() -> str | None:
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5,
                             cwd=Path(__file__).resolve().parent)
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout.strip() if res.returncode == 0 else None


def _git_dirty() -> bool | None:
    try:
        res = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True,
                             timeout=5, cwd=Path(__file__).resolve().parent)
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(res.stdout.strip()) if res.returncode == 0 else None


def runtime_versions() -> dict[str, Any]:
    import gymnasium
    import numpy

    return {"rlenv": rlenv.__version__, "python": platform.python_version(), "gymnasium": gymnasium.__version__,
            "numpy": numpy.__version__, "platform": f"{sys.platform}-{platform.machine()}",
            "git_commit": _git_commit(), "git_dirty": _git_dirty()}


class TrajectoryWriter:
    def __init__(self, directory: Path, redactor: Redactor | None = None) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "trajectory.jsonl"
        if self.path.exists():
            raise TrajectoryError(f"{self.path} already exists; trajectories are append-only and never reused")
        self.blobs = BlobStore(self.dir / "blobs")
        self.redact = redactor or Redactor()
        self.seq = 0
        self.prev = GENESIS
        self._fh = open(self.path, "x", encoding="utf-8")  # noqa: SIM115 - closed in close()
        self.closed = False

    def blob(self, text: str) -> dict:
        clean = self.redact(text)
        return {"blob": self.blobs.put(clean), "bytes": len(clean.encode())}

    def write(self, record: dict[str, Any]) -> str:
        if self.closed:
            raise TrajectoryError("trajectory is closed")
        body = {"seq": self.seq, "prev": self.prev, **record}
        line = self.redact(json.dumps(body, sort_keys=True, separators=(",", ":"), default=str))
        self._fh.write(line + "\n")
        self._fh.flush()
        self.prev = hashlib.sha256(line.encode()).hexdigest()
        self.seq += 1
        return self.prev

    def close(self) -> None:
        if self.closed:
            return
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        (self.dir / "trajectory.head").write_text(f"{self.seq} {self.prev}\n")
        self.closed = True


# --------------------------------------------------------------------------- #
# Recorder: episode events -> trajectory records
# --------------------------------------------------------------------------- #


class Recorder:
    """Episode sink that turns events into trajectory records (code -> blobs, big strings -> blobs)."""

    def __init__(self, writer: TrajectoryWriter, workspace: Path) -> None:
        self.w = writer
        self.workspace = Path(workspace)

    def header(self, episode_id: str, task, agent: dict[str, Any]) -> None:
        self.w.write({
            "type": "header",
            "schema": SCHEMA_VERSION,
            "episode_id": episode_id,
            "created": datetime.now(timezone.utc).isoformat(),
            "task": task.agent_view(),
            "task_id": task.id,
            "task_version": task.version,
            "task_public_sha256": task.public_sha256(),
            "task_full_sha256": task.full_sha256(),
            "agent": agent,
            "runtime": runtime_versions(),
        })

    def __call__(self, event: dict[str, Any]) -> None:
        event = dict(event)
        kind = event.get("type")
        if kind == "tool_call" and event.get("tool") == "write_file" and "content" in event.get("args", {}):
            args = dict(event["args"])
            args["content"] = self.w.blob(args.pop("content"))
            event["args"] = args
        elif kind == "run":
            event["code"] = self._code_blobs(event.get("code", {}))
        self.w.write(self._shrink(event))

    def _code_blobs(self, code: dict[str, str]) -> dict[str, dict]:
        """Store the exact bytes of each file that a run executed."""
        out = {}
        for rel, sha in code.items():
            path = self.workspace / rel
            raw = path.read_bytes() if path.is_file() else b""
            stored = self.w.blobs.put(self.w.redact(raw.decode("utf-8", errors="replace")))
            out[rel] = {"sha256": sha, "blob": stored, "matches": hashlib.sha256(raw).hexdigest() == sha}
        return out

    def _shrink(self, value: Any) -> Any:
        if isinstance(value, str) and len(value) > LARGE_STRING:
            return self.w.blob(value)
        if isinstance(value, dict):
            return {k: self._shrink(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._shrink(v) for v in value]
        return value

    def close(self) -> None:
        self.w.close()


# --------------------------------------------------------------------------- #
# Reader
# --------------------------------------------------------------------------- #


@dataclass
class Trajectory:
    directory: Path
    records: list[dict[str, Any]]
    head_ok: bool | None
    problems: list[str] = field(default_factory=list)

    @property
    def header(self) -> dict[str, Any]:
        return self.records[0]

    @property
    def ok(self) -> bool:
        return not self.problems

    def of_type(self, kind: str) -> list[dict[str, Any]]:
        return [r for r in self.records if r.get("type") == kind]

    def blob_text(self, ref: dict) -> str:
        return BlobStore(self.directory / "blobs").get(ref["blob"]).decode("utf-8")

    def inline(self, value: Any) -> Any:
        """Replace {"blob": sha, "bytes": n} references by their text (exports, replay)."""
        if isinstance(value, dict):
            if set(value) == {"blob", "bytes"}:
                return self.blob_text(value)
            return {k: self.inline(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.inline(v) for v in value]
        return value


def load_trajectory(directory: str | Path, check_blobs: bool = True) -> Trajectory:
    """Parse and verify a trajectory. Problems are collected (with line numbers), not raised,
    so a pipeline can quarantine bad files and keep going."""
    directory = Path(directory)
    path = directory / "trajectory.jsonl"
    if not path.exists():
        raise TrajectoryError(f"no trajectory at {path}")
    problems: list[str] = []
    records: list[dict] = []
    prev = GENESIS
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.rstrip("\n")
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                problems.append(f"line {lineno}: invalid JSON ({exc.msg})")
                prev = hashlib.sha256(line.encode()).hexdigest()
                continue
            if rec.get("seq") != lineno - 1:
                problems.append(f"line {lineno}: seq {rec.get('seq')} != {lineno - 1}")
            if rec.get("prev") != prev:
                problems.append(f"line {lineno}: hash chain broken (previous line was modified or removed)")
            prev = hashlib.sha256(line.encode()).hexdigest()
            records.append(rec)
    if not records or records[0].get("type") != "header":
        problems.append("line 1: missing header")
    elif records[0].get("schema") != SCHEMA_VERSION:
        problems.append(f"unsupported schema {records[0].get('schema')!r}")
    head_ok = None
    head = directory / "trajectory.head"
    if head.exists():
        n, digest = head.read_text().split()
        head_ok = int(n) == len(records) and digest == prev
        if not head_ok:
            problems.append("head mismatch: the trajectory was truncated or its last line was changed")
    if check_blobs:
        store = BlobStore(directory / "blobs")
        for ref in _blob_refs(records):
            try:
                store.get(ref)
            except TrajectoryError as exc:
                problems.append(str(exc))
    return Trajectory(directory, records, head_ok, problems)


def _blob_refs(value: Any):
    if isinstance(value, dict):
        if isinstance(value.get("blob"), str) and len(value["blob"]) == 64:
            yield value["blob"]
        for v in value.values():
            yield from _blob_refs(v)
    elif isinstance(value, list):
        for v in value:
            yield from _blob_refs(v)
