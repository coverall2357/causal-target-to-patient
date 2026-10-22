"""Tests for finding dated data files."""

import pytest

from causal_target_to_patient.files import latest_dated, stamp_of


def test_latest_dated_ignores_backups_and_other_prefixes(tmp_path):
    for name in ("ra_trials_20261009.csv", "ra_trials_20261013.csv",
                 "ra_trials_20261013_before_b1.csv", "ra_trials_backup.csv",
                 "ra_trials_extra_20261099.csv", "ra_trial_contrasts_20261201.csv"):
        (tmp_path / name).write_text("x")
    latest = latest_dated(tmp_path, "ra_trials")
    assert latest.name == "ra_trials_20261013.csv"
    assert stamp_of(latest) == "20261013"


def test_latest_dated_errors():
    with pytest.raises(FileNotFoundError):
        latest_dated("/nonexistent", "ra_trials")
    with pytest.raises(ValueError):
        stamp_of("ra_trials_before.csv")
