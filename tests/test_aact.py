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


@pytest.mark.parametrize(
    "n_outcomes, n_p_values, n_pubs, why_stopped, expected",
    [
        (2, 1, 0, None, "primary analysis"),
        (1, 0, 3, None, "primary results"),         # posted results beat publications
        (0, 0, 1, "Lack of efficacy", "publication"),
        (0, 0, 0, "Lack of efficacy", "stop reason"),
        (0, 0, 0, float("nan"), "none"),            # missing why_stopped from CSV
        (0, 0, 0, "  ", "none"),
        (0, 2, 0, None, "none"),                    # p-values without an outcome: ignored
    ],
)
def test_evidence_tier(n_outcomes, n_p_values, n_pubs, why_stopped, expected):
    from causal_target_to_patient.sources.aact import evidence_tier

    assert evidence_tier(n_outcomes, n_p_values, n_pubs, why_stopped) == expected



@pytest.mark.parametrize(
    "direct, ancestor, text, adult, expected",
    [
        (True, True, True, True, "mesh_direct"),       # strongest route wins
        (False, False, True, True, "text_adult"),
        (False, False, True, False, "text_juvenile"),  # "juvenile rheumatoid arthritis"
        (False, True, False, False, "mesh_ancestor"),  # e.g. Sjogren's syndrome
        (False, False, False, False, "none"),
    ],
)
def test_ra_match_route(direct, ancestor, text, adult, expected):
    from causal_target_to_patient.sources.aact import ra_match_route

    assert ra_match_route(direct, ancestor, text, adult) == expected


@pytest.mark.parametrize("conditions, expected", [
    ("Rheumatoid Arthritis", False),
    ("Juvenile Idiopathic Arthritis | Rheumatoid Arthritis", True),
    ("Polyarticular JIA", True),
    ("Pediatric Rheumatology", True),
    (None, False),
])
def test_mentions_juvenile(conditions, expected):
    from causal_target_to_patient.sources.aact import mentions_juvenile

    assert mentions_juvenile(conditions) is expected



def test_ra_definition_uses_direct_mesh_and_drops_juvenile_only():
    from causal_target_to_patient.sources.aact import (
        BROAD_RA_STUDIES_CTE, ELIGIBLE_CTE, RA_STUDIES_CTE)

    assert "LOWER(mesh_type) = 'mesh-list'" in RA_STUDIES_CTE   # no ancestor matches
    assert "juvenile_only" in RA_STUDIES_CTE and "EXCEPT" in RA_STUDIES_CTE
    assert ELIGIBLE_CTE.startswith(RA_STUDIES_CTE)               # the diagnostic swaps it
    assert "mesh_type" not in BROAD_RA_STUDIES_CTE


def test_apply_excluded_trials():
    from causal_target_to_patient.sources.aact import apply_excluded_trials

    trials = pd.DataFrame({"nct_id": ["A", "B"], "trial_set": ["labelling", "context"]})
    excluded = pd.DataFrame({"nct_id": ["B", "Z"], "reason": ["treats RA-ILD", "not pulled"]})
    out = apply_excluded_trials(trials, excluded).set_index("nct_id")
    assert out.loc["A", "trial_set"] == "labelling" and pd.isna(out.loc["A", "exclusion_reason"])
    assert (out.loc["B", "trial_set"], out.loc["B", "exclusion_reason"]) == ("excluded", "treats RA-ILD")
    assert len(out) == 2                                          # nothing dropped or added
