"""Workspace path confinement and file operations."""

from __future__ import annotations

import os

import pytest

from rlenv.workspace import MAX_WRITE_BYTES, Workspace, WorkspaceError


@pytest.fixture()
def ws(tmp_path):
    return Workspace(tmp_path / "ws")


@pytest.fixture()
def secret(tmp_path):
    p = tmp_path / "secret.txt"
    p.write_text("TOP SECRET")
    return p


def test_write_read_list(ws):
    ws.write_text("train.py", "print('hi')\n")
    ws.write_text("pkg/util.py", "X = 1\n")
    text, truncated = ws.read_text("train.py")
    assert text == "print('hi')\n" and not truncated
    assert [(e.path, e.kind) for e in ws.list()] == [("pkg", "dir"), ("train.py", "file"), ("pkg/util.py", "file")]
    assert oct((ws.root / "train.py").stat().st_mode & 0o777) == "0o600"


@pytest.mark.parametrize("path", ["../escape.py", "a/../../x.py", "/etc/passwd", "~/x.py", "", "a\x00b.py"])
def test_path_escapes_rejected(ws, path):
    with pytest.raises(WorkspaceError):
        ws.resolve(path)


def test_symlink_to_outside_cannot_be_read_or_written(ws, secret):
    # Agent code (running in the workspace) could create this symlink.
    os.symlink(secret, ws.root / "leak.txt")
    with pytest.raises(WorkspaceError, match="outside the workspace"):
        ws.read_text("leak.txt")
    with pytest.raises(WorkspaceError, match="outside the workspace"):
        ws.write_text("leak.txt", "overwrite")
    assert secret.read_text() == "TOP SECRET"


def test_symlinked_directory_to_outside(ws, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, ws.root / "d")
    with pytest.raises(WorkspaceError, match="outside the workspace"):
        ws.write_text("d/x.py", "x")
    assert not (outside / "x.py").exists()


def test_list_does_not_follow_symlinks(ws, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "hidden.py").write_text("x")
    os.symlink(outside, ws.root / "d")
    entries = {e.path: e.kind for e in ws.list()}
    assert entries == {"d": "symlink"}


def test_only_text_files_and_size_cap(ws):
    with pytest.raises(WorkspaceError, match="only text files"):
        ws.write_text("policy/q.npy", "x")
    with pytest.raises(WorkspaceError, match="byte limit"):
        ws.write_text("big.py", "x" * (MAX_WRITE_BYTES + 1))


def test_read_truncates(ws):
    ws.write_text("a.txt", "abcdef")
    assert ws.read_text("a.txt", max_bytes=3) == ("abc", True)


def test_snapshot_hashes_only_regular_text_files(ws, secret):
    ws.write_text("train.py", "a")
    os.symlink(secret, ws.root / "link.py")
    (ws.root / "q.npy").write_bytes(b"\x00\x01")
    snap = ws.snapshot()
    assert list(snap) == ["train.py"]
    assert len(snap["train.py"]) == 64


def test_copy_clean_drops_symlinks_and_artifacts(ws, secret, tmp_path):
    ws.write_text("train.py", "a")
    ws.write_text("policy.py", "b")
    os.symlink(secret, ws.root / "evil.py")
    (ws.root / "runs").mkdir()
    (ws.root / "runs" / "q.npy").write_bytes(b"\x00")
    clean = ws.copy_clean(tmp_path / "verify")
    assert sorted(e.path for e in clean.list()) == ["policy.py", "train.py"]
    with pytest.raises(WorkspaceError, match="already exists"):
        ws.copy_clean(tmp_path / "verify")
