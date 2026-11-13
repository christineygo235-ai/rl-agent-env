"""docs/EVIDENCE.md is generated from committed real artifacts and must stay in sync with them."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from rlenv.atif import validate_atif
from rlenv.evidence import FLASH, PROJECT, EvidenceError, build, is_current
from rlenv.harness import rescore
from rlenv.trajectory import load_trajectory

ARTIFACTS = sorted(p for p in [*(PROJECT / "evidence" / "redteam").iterdir(), *(PROJECT / "samples").iterdir()]
                   if p.is_dir())


def test_committed_evidence_is_current():
    assert is_current(), "docs/EVIDENCE.md is out of date: run `make evidence` and commit"


@pytest.mark.parametrize("path", ARTIFACTS, ids=[p.name for p in ARTIFACTS])
def test_every_artifact_is_intact_and_rescores(path):
    traj = load_trajectory(path)
    assert traj.ok and traj.head_ok is True, traj.problems[:3]
    assert rescore(path).ok


def test_committed_artifacts_hold_no_secrets_or_new_local_paths():
    text = "".join(p.read_text(errors="replace") for root in ("evidence", "samples")
                   for p in (PROJECT / root).rglob("*") if p.is_file())
    assert "DEEPSEEK_API_KEY=" not in text and "Bearer " not in text
    import re
    assert not re.search(r"sk-[A-Za-z0-9]{20,}", text)
    # Red-team evidence was recorded with home-directory scrubbing (see Redactor).
    rt = "".join(p.read_text() for p in (PROJECT / "evidence").rglob("trajectory.jsonl"))
    assert "/Users/" not in rt and "/home/" not in rt


def test_sample_exports_are_valid():
    atif = list((PROJECT / FLASH / "export" / "atif").glob("*.json"))
    assert len(atif) == 1 and validate_atif(json.loads(atif[0].read_text())) == []
    row = json.loads((PROJECT / FLASH / "export" / "messages.jsonl").read_text().splitlines()[0])
    assert row["transcript"] == "exact" and row["reward"] == 1.0


def test_tampered_artifact_breaks_generation(tmp_path):
    for sub in ("evidence", "samples", "calibration"):
        shutil.copytree(PROJECT / sub, tmp_path / sub)
    assert "**0 violation(s)**" in build(tmp_path)  # the copy builds
    traj = tmp_path / "evidence" / "redteam" / "env_tamper" / "trajectory.jsonl"
    lines = traj.read_text().splitlines()
    final = json.loads(lines[-1])
    final["reward"] = 1.0  # someone "improves" a cheat's score
    lines[-1] = json.dumps(final, sort_keys=True, separators=(",", ":"))
    traj.write_text("\n".join(lines) + "\n")
    with pytest.raises(EvidenceError, match="integrity|reproduce"):
        build(tmp_path)


def test_missing_artifact_breaks_generation(tmp_path):
    for sub in ("evidence", "samples", "calibration"):
        shutil.copytree(PROJECT / sub, tmp_path / sub)
    shutil.rmtree(tmp_path / "evidence" / "redteam" / "pickle_artifact")
    with pytest.raises(EvidenceError, match="missing"):
        build(tmp_path)


def test_evidence_numbers_come_from_artifacts():
    text = (PROJECT / "docs" / "EVIDENCE.md").read_text()
    fin = load_trajectory(PROJECT / FLASH).of_type("final")[-1]
    assert fin["reward"] == 1.0 and "| **deepseek-flash** |" in text
    assert "**0 violation(s)**" in text and "| **yes** |" in text


def test_timeline_renders_the_sample(capsys):
    from rlenv import cli

    assert cli.main(["timeline", str(PROJECT / FLASH), "--reasoning"]) == 0
    out = capsys.readouterr().out
    assert "LLM #17" in out and "thinks:" in out and "llm_stop: token budget exhausted" in out
    assert "hidden map #1" in out and "REWARD 1.0 [scored]" in out
    assert cli.main(["timeline", str(PROJECT / "samples" / "deepseek-flash-unscorable")]) == 0
    assert "agent_error" in capsys.readouterr().out
