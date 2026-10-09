"""Tests for AACT helpers that don't need a database connection."""

import pytest

from causal_target_to_patient.sources.aact import assign_trial_set


@pytest.mark.parametrize(
    "phase, status, expected",
    [
        ("PHASE2", "COMPLETED", "labelling"),
        ("PHASE3", "TERMINATED", "labelling"),
        ("PHASE1/PHASE2", "UNKNOWN", "labelling"),
        ("PHASE2/PHASE3", "SUSPENDED", "labelling"),
        ("PHASE4", "COMPLETED", "context"),
        ("PHASE3", "RECRUITING", "ongoing"),
        ("PHASE4", "ACTIVE_NOT_RECRUITING", "ongoing"),  # ongoing takes precedence
    ],
)
def test_assign_trial_set(phase, status, expected):
    assert assign_trial_set(phase, status) == expected
