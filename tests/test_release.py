"""Tests for building the published trial-to-target map."""

import pandas as pd

from causal_target_to_patient.release import (
    TARGET_COLUMNS,
    TRIAL_COLUMNS,
    contrast_genes,
    summary,
    trial_table,
    trial_target_table,
)

TARGETS = pd.DataFrame([
    {"parent_chembl_id": "CHEMBL_ADA", "mechanism_of_action": "TNF inhibitor",
     "action_type": "INHIBITOR", "target_chembl_id": "CHEMBL_TNF",
     "target_pref_name": "TNF-alpha", "target_type": "SINGLE PROTEIN",
     "gene_symbols": "TNF", "mechanism_source": "chembl"},
    {"parent_chembl_id": "CHEMBL_MTX", "mechanism_of_action": "DHFR inhibitor",
     "action_type": "INHIBITOR", "target_chembl_id": "CHEMBL_DHFR",
     "target_pref_name": "DHFR", "target_type": "SINGLE PROTEIN",
     "gene_symbols": "DHFR", "mechanism_source": "chembl"},
    {"parent_chembl_id": "MANUAL:drugx", "mechanism_of_action": None,
     "action_type": None, "target_chembl_id": "CHEMBL_X",
     "target_pref_name": "Target X", "target_type": None,
     "gene_symbols": "TGTX", "mechanism_source": "manual: https://example.org/x"},
])

MOLECULES = pd.DataFrame([
    {"intervention_key": "adalimumab", "component": 0, "excluded": False,
     "excluded_therapy": False, "matched_on": "adalimumab",
     "parent_chembl_id": "CHEMBL_ADA", "molecule_pref_name": "ADALIMUMAB",
     "molecule_type": "Antibody", "max_phase": 4.0, "mechanism_key": "CHEMBL_ADA"},
    {"intervention_key": "mtx", "component": 0, "excluded": False,
     "excluded_therapy": True, "matched_on": "alias:mtx",
     "parent_chembl_id": "CHEMBL_MTX", "molecule_pref_name": "METHOTREXATE",
     "molecule_type": "Small molecule", "max_phase": 4.0, "mechanism_key": "CHEMBL_MTX"},
    {"intervention_key": "drug x", "component": 0, "excluded": False,
     "excluded_therapy": False, "matched_on": "override:drugx",
     "parent_chembl_id": None, "molecule_pref_name": None,
     "molecule_type": None, "max_phase": None, "mechanism_key": "MANUAL:drugx"},
    {"intervention_key": "biologics", "component": 0, "excluded": True,
     "excluded_therapy": True, "matched_on": "override:biologics",
     "parent_chembl_id": None, "molecule_pref_name": None,
     "molecule_type": None, "max_phase": None, "mechanism_key": None},
    {"intervention_key": "herb", "component": 0, "excluded": False,
     "excluded_therapy": False, "matched_on": None,
     "parent_chembl_id": None, "molecule_pref_name": None,
     "molecule_type": None, "max_phase": None, "mechanism_key": None},
])

INTERVENTIONS = pd.DataFrame([
    {"nct_id": "NCT1", "intervention_type": "DRUG", "intervention_name": "Adalimumab",
     "role": "investigational"},
    {"nct_id": "NCT1", "intervention_type": "DRUG", "intervention_name": "MTX",
     "role": "background"},
    {"nct_id": "NCT1", "intervention_type": "DRUG", "intervention_name": "Placebo",
     "role": "placebo"},
    {"nct_id": "NCT2", "intervention_type": "DRUG", "intervention_name": "Drug X",
     "role": "investigational"},
    {"nct_id": "NCT2", "intervention_type": "DRUG", "intervention_name": "Biologics",
     "role": "comparator"},
    {"nct_id": "NCT3", "intervention_type": "DRUG", "intervention_name": "Herb",
     "role": "investigational"},
])


def _trials():
    base = {c: None for c in ("brief_title", "phase", "overall_status", "why_stopped",
                              "start_date", "completion_date", "enrollment",
                              "lead_sponsor", "sponsor_class", "conditions",
                              "exclusion_reason")}
    trials = pd.DataFrame([dict(base, nct_id=n, trial_set=s)
                           for n, s in (("NCT2", "labelling"), ("NCT1", "labelling"),
                                        ("NCT3", "context"))])
    contrasts = pd.DataFrame([
        {"nct_id": "NCT1", "contrast_class": "target_contrast",
         "contrast_targets": "CHEMBL_TNF", "has_unmapped_test": False},
        {"nct_id": "NCT2", "contrast_class": "head_to_head",
         "contrast_targets": "CHEMBL_X;GENE:ABC", "has_unmapped_test": False},
        {"nct_id": "NCT3", "contrast_class": "unmapped",
         "contrast_targets": None, "has_unmapped_test": True},
    ])
    evidence = pd.DataFrame({"nct_id": ["NCT1", "NCT2", "NCT3"],
                             "evidence_tier": ["primary analysis", "none", "none"]})
    return trial_table(trials, contrasts, evidence, TARGETS)


def test_contrast_genes_handles_chembl_ids_gene_targets_and_blanks():
    genes = {"T1": "CD80, CD86", "T2": "CD86"}
    assert contrast_genes("T1;T2;GENE:FOO", genes) == "CD80;CD86;FOO"
    assert contrast_genes(None, genes) == ""
    assert contrast_genes("UNKNOWN", genes) == ""


def test_trial_table_columns_order_and_genes():
    table = _trials()
    assert tuple(table.columns) == TRIAL_COLUMNS
    assert table.nct_id.tolist() == ["NCT1", "NCT2", "NCT3"]
    assert table.set_index("nct_id").contrast_genes.to_dict() == {
        "NCT1": "TNF", "NCT2": "ABC;TGTX", "NCT3": ""}


def test_trial_target_table_statuses_and_provenance():
    table = trial_target_table(INTERVENTIONS, MOLECULES, TARGETS)
    assert tuple(table.columns) == TARGET_COLUMNS
    rows = table.set_index("intervention_name")
    assert rows.mapping_status.to_dict() == {
        "Adalimumab": "mapped", "MTX": "excluded_therapy", "Placebo": "placebo",
        "Drug X": "mapped", "Biologics": "excluded_name", "Herb": "unmapped"}
    assert rows.loc["Adalimumab", "mapping_basis"] == "chembl_name"
    assert rows.loc["MTX", "mapping_basis"] == "chembl_synonym"
    assert rows.loc["Drug X", "mapping_basis"] == "curated_override"
    assert rows.loc["Drug X", "mechanism_source"] == "curated"
    assert rows.loc["Drug X", "mechanism_reference"] == "https://example.org/x"
    assert rows.loc["Adalimumab", "mechanism_source"] == "chembl"
    assert rows.loc["Adalimumab", "mechanism_reference"] == ""
    assert pd.isna(rows.loc["Placebo", "target_chembl_id"])


def test_summary_counts_tests_in_labelling_trials_only():
    counts = summary(_trials(), trial_target_table(INTERVENTIONS, MOLECULES, TARGETS))
    assert counts["trials"] == 3
    assert counts["by_trial_set"] == {"labelling": 2, "context": 1}
    assert counts["labelling_tests"] == 2
    assert counts["targets_tested"] == 3          # CHEMBL_TNF, CHEMBL_X, GENE:ABC
    assert counts["genes_tested"] == 3            # TNF, TGTX, ABC
    assert counts["chembl_molecules_mapped"] == 1  # adalimumab; drug X has no ChEMBL ID
    assert counts["curated_mechanism_rows"] == 1
