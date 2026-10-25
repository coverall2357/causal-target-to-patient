"""Tests for labelling-run file names and the overwrite guard."""

from datetime import date
from pathlib import Path

import pytest

from causal_target_to_patient.labelling.runs import (
    refuse_overwrite,
    run_name,
    run_outputs,
)


def test_run_name_from_file_or_date():
    assert run_name(Path("labels/dev_set_20261022.csv"), date(2026, 10, 22)) == "dev_set_20261022"
    assert run_name(None, date(2026, 10, 22)) == "adhoc_20261022"


def test_run_outputs():
    out = run_outputs(Path("labels"), "dev_set_20261022")
    assert out == {"labels": Path("labels/dev_set_20261022_labels.csv"),
                   "review": Path("labels/dev_set_20261022_review.csv"),
                   "blind": Path("labels/dev_set_20261022_blind.csv")}


def test_refuse_overwrite(tmp_path):
    existing, new = tmp_path / "a.csv", tmp_path / "b.csv"
    existing.write_text("x")
    refuse_overwrite([new])                                # nothing there: fine
    refuse_overwrite([existing, new], force=True)          # forced: fine
    with pytest.raises(FileExistsError, match="a.csv.*--force"):
        refuse_overwrite([existing, new])
