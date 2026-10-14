"""Tests for mechanism-level coverage comparison."""

import pandas as pd

from causal_target_to_patient.coverage import (
    chembl_mechanisms,
    compare,
    expected_mechanisms,
    pipeline_mechanisms,
    to_mechanisms,
)

GROUPS = {"IL6": "IL-6 pathway", "IL6R": "IL-6 pathway", "JAK1": "JAK family",
          "TYK2": "JAK family", "TNF": "TNF", "IL17A": "IL-17 pathway"}


def test_to_mechanisms_groups_and_defaults():
    assert to_mechanisms("IL6R;IL6", GROUPS) == {"IL-6 pathway"}
    assert to_mechanisms("JAK1;CD22", GROUPS) == {"JAK family", "CD22"}   # ungrouped gene
    assert to_mechanisms("", GROUPS, "Melanocortin receptors") == {"Melanocortin receptors"}
    assert to_mechanisms("CD22", GROUPS, "B-cell receptor CD22") == {"CD22"}
    assert to_mechanisms(None, GROUPS) == set()


def test_pipeline_mechanisms_counts_trials_once_per_mechanism():
    contrasts = pd.DataFrame({
        "nct_id": ["A", "B", "C", "D"],
        "trial_set": ["labelling", "labelling", "labelling", "context"],
        "contrast_class": ["target_contrast", "head_to_head", "same_target", "target_contrast"],
        "contrast_targets": ["T_JAK1;T_TYK2", "T_IL6R;T_TNF", "T_TNF", "T_TNF"],
    })
    targets = pd.DataFrame({
        "target_chembl_id": ["T_JAK1", "T_TYK2", "T_IL6R", "T_TNF"],
        "gene_symbols": ["JAK1", "TYK2", "IL6R", "TNF"],
        "target_pref_name": ["JAK1", "TYK2", "IL6R", "TNF"],
    })
    out = pipeline_mechanisms(contrasts, targets, GROUPS).set_index("mechanism").n_trials
    # A counts once for JAK family despite two JAK targets; C and D are excluded.
    assert out.to_dict() == {"IL-6 pathway": 1, "JAK family": 1, "TNF": 1}


def test_expected_combination_rows_decompose():
    expected = pd.DataFrame({
        "mechanism": ["TNF + IL-17 (bispecific)", "TNF"],
        "gene_symbols": ["TNF;IL17A", "TNF"],
        "status_in_ra": ["failed: not superior to TNF inhibitors", "approved"],
    })
    out = expected_mechanisms(expected, GROUPS).set_index("mechanism").reference_detail
    assert set(out.index) == {"TNF", "IL-17 pathway"}
    assert "TNF: approved" in out["TNF"] and "bispecific" in out["TNF"]


def test_compare_categories():
    pipeline = pd.DataFrame({"mechanism": ["TNF", "CD22"], "n_trials": [90, 1]})
    reference = pd.DataFrame({"mechanism": ["TNF", "CCR1"], "reference_detail": ["a", "b"]})
    out = compare(pipeline, reference).set_index("mechanism").category.to_dict()
    assert out == {"TNF": "both", "CD22": "pipeline_only", "CCR1": "reference_only"}


def test_chembl_mechanisms_lists_molecules_by_phase():
    ref = pd.DataFrame({
        "molecule_pref_name": ["TOFACITINIB", "DECERNOTINIB"],
        "max_phase_for_ra": [4.0, 2.0],
        "gene_symbols": ["JAK1", "JAK1"], "target_pref_name": ["JAK1", "JAK1"],
    })
    detail = chembl_mechanisms(ref, GROUPS).set_index("mechanism").reference_detail["JAK family"]
    assert detail == "TOFACITINIB (phase 4); DECERNOTINIB (phase 2)"


def test_multigene_target_without_groups_is_one_mechanism():
    assert to_mechanisms("PSMA1;PSMB5;PSMC1", GROUPS, "26S proteasome") == {"26S proteasome"}
    assert to_mechanisms("JAK1;ROCK1", GROUPS, "x") == {"JAK family", "ROCK1"}  # some grouped


def test_search_terms():
    from causal_target_to_patient.coverage import search_terms

    assert search_terms("rozibafusp alfa (AMG 570); CPL409116") == [
        "rozibafusp alfa", "amg 570", "cpl409116"]
    assert search_terms("(unnamed in source)") == []
    assert search_terms("mavrilimumab (named in source references)") == ["mavrilimumab"]


def test_locate_and_diagnose():
    from causal_target_to_patient.coverage import diagnose, locate_in_pipeline

    name_molecules = pd.DataFrame({"intervention_key": ["losmapimod"],
                                   "parent_chembl_id": ["P_LOS"], "mechanism_key": ["P_LOS"]})
    interventions = pd.DataFrame({
        "nct_id": ["A", "B", "C"],
        "intervention_key": ["losmapimod", "mk0812 / 12 weeks", "other"],
        "role": ["investigational", "investigational", "investigational"]})
    contrasts = pd.DataFrame({"nct_id": ["A", "B", "C"],
                              "trial_set": ["context", "labelling", "labelling"],
                              "contrast_class": ["target_contrast", "unmapped", "target_contrast"]})
    found = locate_in_pipeline(["losmapimod"], {"P_LOS"}, name_molecules, interventions, contrasts)
    assert diagnose(found) == "in pull, not counted: context/target_contrast x1"
    found = locate_in_pipeline(["MK-0812"], set(), name_molecules, interventions, contrasts)
    assert diagnose(found) == "in pull, name not mapped to a molecule"
    found = locate_in_pipeline(["efalizumab"], {"P_EFA"}, name_molecules, interventions, contrasts)
    assert diagnose(found) == "not in AACT pull"


def test_name_match_on_mapped_name_is_not_reported_as_unmapped():
    from causal_target_to_patient.coverage import diagnose, locate_in_pipeline

    name_molecules = pd.DataFrame({"intervention_key": ["peresolimab"],
                                   "molecule_chembl_id": ["P_PER"], "mechanism_key": ["P_PER"]})
    interventions = pd.DataFrame({"nct_id": ["A"], "intervention_key": ["peresolimab"],
                                  "role": ["investigational"]})
    contrasts = pd.DataFrame({"nct_id": ["A"], "trial_set": ["labelling"],
                              "contrast_class": ["unmapped"]})
    found = locate_in_pipeline(["peresolimab"], set(), name_molecules, interventions, contrasts)
    assert diagnose(found) == "in pull, not counted: labelling/unmapped x1"
