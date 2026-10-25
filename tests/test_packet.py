"""Tests for evidence-packet assembly, using small hand-made AACT tables."""

import json

import numpy as np
import pandas as pd

from causal_target_to_patient.labelling.packet import (
    build_packet,
    intervention_mapping,
    packet_counts,
    render_markdown,
)

NCT = "NCT00000001"


def tables():
    return {
        "studies": pd.DataFrame({
            "nct_id": [NCT, "NCT99999999"], "brief_title": ["Drug X in RA", "Other"],
            "phase": ["PHASE3", "PHASE2"], "overall_status": ["COMPLETED"] * 2,
            "why_stopped": [np.nan, np.nan], "enrollment": [np.int64(300), 10],
            "start_date": [pd.Timestamp("2015-03-01"), pd.NaT],
        }),
        "brief_summaries": pd.DataFrame({"nct_id": [NCT], "description": ["A trial."]}),
        "eligibilities": pd.DataFrame({
            "nct_id": [NCT], "criteria": ["Inclusion: inadequate response to MTX."]}),
        "design_groups": pd.DataFrame({
            "id": [1, 2], "nct_id": [NCT, NCT],
            "group_type": ["EXPERIMENTAL", "PLACEBO_COMPARATOR"],
            "title": ["Drug X + MTX", "Placebo + MTX"], "description": ["", None],
        }),
        "design_group_interventions": pd.DataFrame({
            "nct_id": [NCT] * 4, "design_group_id": [1, 1, 2, 2],
            "intervention_id": [10, 12, 11, 12],
        }),
        "interventions": pd.DataFrame({
            "id": [10, 11, 12], "nct_id": [NCT] * 3,
            "name": ["Drug X", "Placebo", "Methotrexate"],
        }),
        "design_outcomes": pd.DataFrame({
            "nct_id": [NCT, NCT], "outcome_type": ["primary", "secondary"],
            "measure": ["ACR20 response", "HAQ-DI"], "time_frame": ["Week 12", "Week 24"],
        }),
        "outcomes": pd.DataFrame({
            "id": [100, 101], "nct_id": [NCT, NCT], "outcome_type": ["PRIMARY", "SECONDARY"],
            "title": ["ACR20 at Week 12", "HAQ-DI"], "units": ["percentage", "units"],
        }),
        "result_groups": pd.DataFrame({
            "id": [500, 501], "nct_id": [NCT, NCT], "title": ["Drug X", "Placebo"],
        }),
        "outcome_measurements": pd.DataFrame({
            "nct_id": [NCT] * 3, "outcome_id": [100, 100, 101],
            "result_group_id": [500, 501, 500], "param_value": ["66.0", "30.0", "-0.5"],
        }),
        "outcome_analyses": pd.DataFrame({
            "id": [900], "nct_id": [NCT], "outcome_id": [100],
            "method": ["Cochran-Mantel-Haenszel"], "p_value": [0.0001],
            "p_value_modifier": ["<"],
        }),
        "outcome_analysis_groups": pd.DataFrame({
            "nct_id": [NCT, NCT], "outcome_analysis_id": [900, 900],
            "result_group_id": [500, 501],
        }),
        "study_references": pd.DataFrame({
            "nct_id": [NCT, NCT], "pmid": [222.0, 111.0],
            "reference_type": ["derived", "RESULT"], "citation": ["Cit B", "Cit A"],
        }),
    }


ABSTRACTS = {"111": {"pmid": "111", "title": "Main result", "abstract": "RESULTS: met."}}


def test_build_packet_core_sections():
    packet = build_packet(NCT, tables(), abstracts=ABSTRACTS)
    trial = packet["trial"]
    assert trial["brief_title"] == "Drug X in RA"
    assert "why_stopped" not in trial               # missing values dropped
    assert trial["enrollment"] == 300               # numpy scalar converted
    assert trial["start_date"] == "2015-03-01"
    assert packet["summary"] == "A trial."

    arms = {a["title"]: a for a in packet["arms"]}
    assert arms["Drug X + MTX"]["interventions"] == ["Drug X", "Methotrexate"]
    assert "description" not in arms["Drug X + MTX"]  # blank string dropped

    assert packet["planned_primary_outcomes"] == [
        {"measure": "ACR20 response", "time_frame": "Week 12"}]


def test_posted_primary_results_only_primary_with_groups():
    posted = build_packet(NCT, tables())["posted_primary_results"]
    assert len(posted) == 1                          # secondary outcome excluded
    outcome = posted[0]
    assert [m["group"] for m in outcome["measurements"]] == ["Drug X", "Placebo"]
    (analysis,) = outcome["analyses"]
    assert analysis["groups"] == "Drug X vs. Placebo"
    assert analysis["p_value"] == 0.0001


def test_publications_ordered_and_joined_to_abstracts():
    pubs = build_packet(NCT, tables(), abstracts=ABSTRACTS)["publications"]
    assert [p["pmid"] for p in pubs] == ["111", "222"]   # RESULT before DERIVED
    assert pubs[0]["abstract"] == "RESULTS: met."
    assert pubs[1]["reference_type"] == "DERIVED"
    assert "abstract" not in pubs[1]


def test_trial_without_results_or_publications():
    packet = build_packet("NCT00000002", tables())
    assert packet["posted_primary_results"] == []
    assert packet["publications"] == []
    assert packet["trial"] == {"nct_id": "NCT00000002"}


def test_packet_is_json_serialisable_and_renders():
    packet = build_packet(NCT, tables(), abstracts=ABSTRACTS)
    json.dumps(packet)
    md = render_markdown(packet)
    assert "ACR20 at Week 12" in md and "PMID 111 (RESULT)" in md
    assert packet_counts(packet) == {
        "nct_id": NCT, "arms": 2, "planned_primary": 1, "posted_primary": 1,
        "analyses": 1, "publications": 2, "abstracts": 1}


def test_intervention_mapping():
    interventions = pd.DataFrame({
        "nct_id": [NCT, NCT], "intervention_name": ["Drug X", "Methotrexate"],
        "role": ["investigational", "background"],
        "arm_role": ["investigational", "background"],
    })
    name_molecules = pd.DataFrame({
        "intervention_key": ["drug x", "methotrexate"], "lookup_name": ["drug x", "methotrexate"],
        "molecule_pref_name": ["DRUG X", "METHOTREXATE"], "excluded": [False, False],
        "excluded_therapy": [False, True], "mechanism_key": ["CHEMBL1", "CHEMBL2"],
    })
    molecule_targets = pd.DataFrame({
        "parent_chembl_id": ["CHEMBL1"], "gene_symbols": ["TNF"],
        "target_pref_name": ["Tumor necrosis factor"],
    })
    contrasts = pd.DataFrame({"nct_id": [NCT], "contrast_class": ["target_contrast"],
                              "contrast_targets": ["CHEMBL1825"]})
    mapping = intervention_mapping(NCT, interventions, name_molecules,
                                   molecule_targets, contrasts)
    assert mapping["contrast_class"] == "target_contrast"
    drug, mtx = mapping["interventions"]
    assert drug["components"][0]["targets"] == ["TNF (Tumor necrosis factor)"]
    assert mtx["components"][0]["excluded_therapy"] is True
    json.dumps(mapping)


def test_long_measurement_lists_are_capped():
    from causal_target_to_patient.labelling import packet as packet_module

    t = tables()
    n = packet_module.MAX_MEASUREMENTS + 5
    t["outcome_measurements"] = pd.DataFrame({
        "nct_id": [NCT] * n, "outcome_id": [100] * n,
        "result_group_id": [500] * n, "param_value": [str(i) for i in range(n)],
    })
    outcome = build_packet(NCT, t)["posted_primary_results"][0]
    assert len(outcome["measurements"]) == packet_module.MAX_MEASUREMENTS
    assert outcome["measurements_truncated"] == (
        f"showing the first {packet_module.MAX_MEASUREMENTS} of {n} rows")


def test_reference_without_pmid_renders():
    t = tables()
    t["study_references"] = pd.DataFrame({
        "nct_id": [NCT], "pmid": [np.nan], "reference_type": ["RESULT"],
        "citation": ["Conference abstract"]})
    md = render_markdown(build_packet(NCT, t))
    assert "### No PMID (RESULT)" in md and "PMID None" not in md


def test_eligibility_criteria_included_and_capped():
    from causal_target_to_patient.labelling import packet as packet_module

    packet = build_packet(NCT, tables())
    assert packet["eligibility_criteria"] == "Inclusion: inadequate response to MTX."
    assert "## Eligibility criteria" in render_markdown(packet)
    assert build_packet("NCT00000002", tables())["eligibility_criteria"] is None

    t = tables()
    n = packet_module.MAX_CRITERIA_CHARS + 10
    t["eligibilities"] = pd.DataFrame({"nct_id": [NCT], "criteria": ["x" * n]})
    capped = build_packet(NCT, t)["eligibility_criteria"]
    assert capped.startswith("x" * packet_module.MAX_CRITERIA_CHARS)
    assert capped.endswith(f"[truncated: showing the first "
                           f"{packet_module.MAX_CRITERIA_CHARS} of {n} characters]")
