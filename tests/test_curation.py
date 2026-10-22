"""Unit tests for the provenance checks and the curation worklist."""

import pandas as pd

from causal_target_to_patient.curation import (
    check_manual_mechanisms,
    check_no_conflicts,
    check_overrides,
    check_unmapped_decisions,
    curation_worklist,
    missing_batch_logs,
)

GOOD_MECHANISM = {
    "name": "drugx", "target_chembl_id": "CHEMBL1", "target_pref_name": "Target X",
    "gene_symbols": "TGTX", "source": "https://example.org/drugx", "source_type": "paper",
    "evidence": "Drug X inhibits target X", "confidence": "high", "notes": "",
    "batch": "2026-10-22-b1",
}


def test_good_mechanism_passes():
    assert check_manual_mechanisms(pd.DataFrame([GOOD_MECHANISM])) == []


def test_mechanism_problems_are_named():
    bad = dict(GOOD_MECHANISM, source="see paper", confidence="certain", evidence="",
               gene_symbols="")
    problems = "\n".join(check_manual_mechanisms(pd.DataFrame([bad])))
    assert "source is not a URL" in problems
    assert "confidence not allowed" in problems
    assert "evidence is empty" in problems
    assert "gene_symbols is empty" in problems
    assert "line 2 (drugx)" in problems


def test_legacy_batch_may_omit_evidence_and_notes_explain_missing_genes():
    legacy = dict(GOOD_MECHANISM, evidence="", batch="2026-10-13-initial",
                  gene_symbols="", notes="subtypes not confirmed")
    assert check_manual_mechanisms(pd.DataFrame([legacy])) == []


def test_duplicates_and_missing_columns():
    df = pd.DataFrame([GOOD_MECHANISM, dict(GOOD_MECHANISM, name="DrugX")])
    assert any("duplicate" in p for p in check_manual_mechanisms(df))
    assert check_manual_mechanisms(df.drop(columns="batch")) == [
        "manual_mechanisms.csv: missing column batch"]


def test_self_evident_overrides_need_no_url():
    base = {"intervention_name": "mtx", "maps_to": "methotrexate", "note": "abbreviation",
            "source": "", "source_type": "self-evident", "confidence": "high",
            "batch": "2026-10-13-initial"}
    assert check_overrides(pd.DataFrame([base])) == []
    sourced = dict(base, source_type="company")
    assert any("not a URL" in p for p in check_overrides(pd.DataFrame([sourced])))


def test_unmapped_decisions_and_conflicts():
    decision = {"name": "drugx", "reason": "no defined target", "source": "https://e.org",
                "source_type": "drug_database", "evidence": "quote", "batch": "b1"}
    assert check_unmapped_decisions(pd.DataFrame([decision])) == []
    assert check_no_conflicts(pd.DataFrame([GOOD_MECHANISM]), pd.DataFrame([decision])) == [
        "drugx: in both manual_mechanisms.csv and unmapped_decisions.csv"]


def test_missing_batch_logs(tmp_path):
    (tmp_path / "b1.md").write_text("log")
    assert missing_batch_logs({"b1", "b2"}, tmp_path) == [
        f"no curation log for batch b2: expected {tmp_path / 'b2.md'}"]


def test_curation_worklist():
    interventions = pd.DataFrame({
        "nct_id": ["T1", "T1", "T2", "T3", "T3", "T4"],
        "intervention_name": ["Drug A", "Placebo", "Drug A", "Code-9", "MTX", "Drug B"],
        "role": ["investigational", "placebo", "investigational", "candidate",
                 "investigational", "investigational"],
    })
    name_molecules = pd.DataFrame({
        "intervention_key": ["drug a", "code-9", "mtx", "drug b"],
        "lookup_name": ["drug a", "code-9", "mtx", "drug b"],
        "excluded": [False, False, False, False],
        "excluded_therapy": [False, False, True, False],
        "molecule_chembl_id": ["C_A", None, "C_MTX", "C_B"],
        "parent_chembl_id": ["C_A", None, "C_MTX", "C_B"],
        "molecule_pref_name": ["DRUG A", None, "METHOTREXATE", "DRUG B"],
        "molecule_type": ["Small molecule", None, "Small molecule", "Antibody"],
        "mechanism_key": ["C_A", None, "C_MTX", "C_B"],
    })
    molecule_targets = pd.DataFrame({"parent_chembl_id": ["C_B", "C_MTX"]})
    contrasts = pd.DataFrame({"nct_id": ["T1", "T2", "T3", "T4"],
                              "trial_set": ["labelling", "context", "labelling", "labelling"]})

    wl = curation_worklist(interventions, name_molecules, molecule_targets, contrasts)
    assert list(zip(wl.kind, wl["item"])) == [("no_mechanism", "drug a"),
                                              ("no_molecule", "code-9")]
    drug_a = wl.iloc[0]
    assert (drug_a.n_trials, drug_a.n_labelling_trials) == (2, 1)   # T1 labelling, T2 context
    # Drug B has a mechanism, MTX is excluded, placebo isn't tested: none listed.

    decided = curation_worklist(interventions, name_molecules, molecule_targets, contrasts,
                                decided={"DRUG A"})
    assert list(decided["item"]) == ["code-9"]


def test_check_excluded_trials():
    from causal_target_to_patient.curation import check_excluded_trials

    good = {"nct_id": "NCT00000001", "reason": "treats RA-ILD, not RA activity",
            "source": "https://clinicaltrials.gov/study/NCT00000001",
            "source_type": "trial_registry", "evidence": "Drug X in RA-ILD", "batch": "b2"}
    assert check_excluded_trials(pd.DataFrame([good])) == []
    bad = dict(good, nct_id="12345", evidence="")
    problems = "\n".join(check_excluded_trials(pd.DataFrame([bad])))
    assert "not an NCT number" in problems and "evidence is empty" in problems
