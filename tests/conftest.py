

import shutil as _shutil
from pathlib import Path as _Path

import pytest as _pytest

_REF = _Path(__file__).resolve().parents[1] / "reference"


@_pytest.fixture(scope="session")
def fast_reference(tmp_path_factory):
    """The reference solutions with smaller default training budgets. Verification re-runs
    train.py with its defaults, so this keeps end-to-end tests fast without changing logic."""
    root = tmp_path_factory.mktemp("fast_reference")
    for d in _REF.iterdir():
        _shutil.copytree(d, root / d.name)
    fl = root / "frozenlake-tabular" / "train.py"
    fl.write_text(fl.read_text().replace("default=30000", "default=5000"))
    return root


def fast_task(task, eval_episodes: int = 200, hidden_instances: int = 1):
    """Same task with fewer held-out evaluation episodes and hidden env instances
    (tests of mechanics, not of quality; production tasks are unchanged)."""
    update = {"success": task.success.model_copy(update={"eval_episodes": eval_episodes})}
    if task.verification.env_variants is not None:
        variants = task.verification.env_variants.model_copy(update={"count": hidden_instances})
        update["verification"] = task.verification.model_copy(update={"env_variants": variants})
    return task.model_copy(update=update)
