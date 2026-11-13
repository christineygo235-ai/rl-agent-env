"""Reference solutions: fast smoke runs through the real pipeline.

Full-length runs against the thresholds are `make reference-check` (slow)."""

from __future__ import annotations

import importlib.util
import string
import sys
from pathlib import Path

import numpy as np
import pytest

from rlenv import cli
from rlenv.policy_check import check_workspace
from rlenv.reference import REFERENCE_ROOT, check_reference
from rlenv.task import load_tasks

ROOT = Path(__file__).resolve().parents[1]
TASKS = load_tasks(ROOT / "tasks")
REPORT_SECTIONS = ("## Method", "## Hyperparameters", "## Experiments", "## Results", "## Limitations")


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_every_task_has_a_reference_that_passes_static_checks(task_id):
    ref = REFERENCE_ROOT / task_id
    report = check_workspace(ref, allow_torch=TASKS[task_id].methods.allow_torch, required=("train.py", "policy.py"))
    assert report.ok, report.summary()
    assert set(report.imports) <= {"argparse", "json", "time", "pathlib", "gymnasium", "numpy", "tiles"}


@pytest.mark.parametrize("task_id,episodes", [("frozenlake-tabular", 300), ("mountaincar-tilecoding", 50)])
def test_reference_pipeline_smoke(task_id, episodes, tmp_path):
    r = check_reference(TASKS[task_id], episodes=episodes, workdir=tmp_path, eval_episodes=10)
    assert r.static_ok and r.train_ok and r.eval_ok, r.detail
    assert r.eval_mean is not None and r.random_mean is not None
    assert r.train_seconds < TASKS[task_id].budget.train_timeout_s


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_report_template_has_required_sections_and_fields(task_id):
    text = (REFERENCE_ROOT / task_id / "REPORT.template.md").read_text()
    for section in REPORT_SECTIONS:
        assert section in text
    fields = {f for _, f, _, _ in string.Formatter().parse(text) if f}
    assert fields == {"train_seed", "runs_table", "dev_episodes", "dev_mean", "threshold"}


def _load_tiles():
    path = REFERENCE_ROOT / "mountaincar-tilecoding" / "tiles.py"
    spec = importlib.util.spec_from_file_location("ref_tiles", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ref_tiles"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_tile_indices_stay_in_bounds_at_the_corners():
    """Regression: unwrapped (1, 3) offsets pushed velocity indices out of range."""
    tiles = _load_tiles()
    coder = tiles.TileCoder(8, 8)
    w = np.zeros(coder.shape(3))
    for x in ([-1.2, -0.07], [0.6, 0.07], [-1.2, 0.07], [0.6, -0.07], [0.0, 0.0]):
        _, i, j = coder.indices(np.array(x))
        assert i.min() >= 0 and j.min() >= 0 and i.max() <= 8 and j.max() <= 8
        assert coder.q_values(w, np.array(x)).shape == (3,)


def test_tilings_are_offset_from_each_other():
    tiles = _load_tiles()
    coder = tiles.TileCoder(8, 8)
    # A point whose fractional tile coordinates are tiny lands in the same cell in every
    # tiling (that's how offset tilings generalise), so test over many points instead.
    rng = np.random.default_rng(0)
    points = np.column_stack([rng.uniform(-1.2, 0.6, 500), rng.uniform(-0.07, 0.07, 500)])
    distinct = [len(set(zip(*map(np.ndarray.tolist, coder.indices(p)[1:])))) for p in points]
    assert np.mean([d > 1 for d in distinct]) > 0.8


def test_cli_check_reference_smoke(capsys):
    rc = cli.main(["check-reference", "frozenlake-tabular", "--tasks", str(ROOT / "tasks"),
                   "--episodes", "200", "--eval-episodes", "10"])
    out = capsys.readouterr().out
    assert rc in (0, 2) and "frozenlake-tabular: held-out mean" in out
