"""Tests for AACT helpers that don't need a database connection."""

import pandas as pd
import pytest

from causal_target_to_patient.sources.aact import (
    assign_intervention_role,
    assign_trial_set,
    flag_trials_for_review,
)


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


@pytest.mark.parametrize(
    "name, in_experimental, in_comparator, expected",
    [
        ("Tocilizumab", True, False, "investigational"),
        ("Methotrexate", True, True, "background"),
        ("Adalimumab", False, True, "comparator"),
        ("Leflunomide", False, False, "unassigned"),
        ("Placebo", False, True, "placebo"),
        ("Placebo to match tofacitinib", True, False, "placebo"),  # name overrides arms
        ("PLACEBO", True, True, "placebo"),
    ],
)
def test_assign_intervention_role(name, in_experimental, in_comparator, expected):
    assert assign_intervention_role(name, in_experimental, in_comparator) == expected


def test_flag_trials_for_review():
    interventions = pd.DataFrame({
        "nct_id": ["A", "A", "B", "C", "C"],
        "role": ["investigational", "background",       # A: one tested drug
                 "comparator",                          # B: nothing tested
                 "investigational", "investigational"], # C: two, single arm
    })
    arms = pd.DataFrame({
        "nct_id": ["A", "B", "C"],
        "n_arms": [2, 2, 1],
        "n_experimental_arms": [1, 0, 1],
        "n_comparator_arms": [1, 2, 0],
    })
    flagged = flag_trials_for_review(interventions, arms).set_index("nct_id")
    assert flagged.loc["A", "n_investigational"] == 1
    assert not flagged.loc["A", "needs_review"]
    assert flagged.loc["B", "needs_review"]   # no investigational intervention
    assert flagged.loc["C", "needs_review"]   # several, with no comparator arm


def test_apply_candidate_fallback():
    from causal_target_to_patient.sources.aact import apply_candidate_fallback

    interventions = pd.DataFrame({
        "nct_id": ["A", "A", "B", "B"],
        "role": ["investigational", "placebo", "comparator", "placebo"],
    })
    flagged = pd.DataFrame({"nct_id": ["A", "B"], "n_investigational": [1, 0]})
    out = apply_candidate_fallback(interventions, flagged)
    assert list(out.role) == ["investigational", "placebo", "candidate", "placebo"]
    assert list(out.arm_role) == ["investigational", "placebo", "comparator", "placebo"]


@pytest.mark.parametrize("name", ["Placebos", "Vehicle control", "Sham injection",
                                  "Placebo to match baricitinib"])
def test_inactive_controls_are_placebo(name):
    assert assign_intervention_role(name, True, False) == "placebo"


def test_vehicular_names_not_placebo():
    # Word boundaries: an unrelated word containing "sham" is not a control.
    assert assign_intervention_role("Shampoo", True, False) == "investigational"
