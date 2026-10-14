"""Tests for name normalisation and target-contrast classification."""

import pandas as pd
import pytest

from causal_target_to_patient.sources.chembl import (
    classify_target_contrast,
    name_variants,
)


@pytest.mark.parametrize(
    "raw, first_variants",
    [
        ("Tocilizumab", ["tocilizumab"]),
        ("tocilizumab [roactemra/actemra]", ["tocilizumab", "roactemra", "actemra"]),
        ("cp-690,550", ["cp-690,550", "cp-690550", "cp690550"]),
        ("Tofacitinib 5 mg BID", ["tofacitinib"]),
        ("Methotrexate (MTX)", ["methotrexate", "mtx"]),
        ("Adalimumab subcutaneous injection", ["adalimumab"]),
    ],
)
def test_name_variants_single(raw, first_variants):
    components = name_variants(raw)
    assert len(components) == 1
    assert components[0][: len(first_variants)] == first_variants


def test_name_variants_combination():
    components = name_variants("Tocilizumab + Methotrexate")
    assert [c[0] for c in components] == ["tocilizumab", "methotrexate"]


def _rows(*records):
    return pd.DataFrame(records, columns=["nct_id", "intervention_key", "role",
                                          "target_chembl_id"])


@pytest.mark.parametrize(
    "rows, expected_class, expected_targets",
    [
        # New drug vs placebo, background methotrexate.
        (_rows(("T", "drugx", "investigational", "X"),
               ("T", "placebo", "placebo", None),
               ("T", "methotrexate", "background", "DHFR")),
         "target_contrast", "X"),
        # Biosimilar vs originator: same target in both arms.
        (_rows(("T", "sb5", "investigational", "TNF"),
               ("T", "humira", "comparator", "TNF")),
         "same_target", ""),
        # Add-on: atacicept tested on top of rituximab.
        (_rows(("T", "atacicept", "investigational", "BAFF"),
               ("T", "rituximab", "background", "CD20")),
         "target_contrast", "BAFF"),
        # Double-dummy: adalimumab placebo listed under the drug's name.
        (_rows(("T", "upadacitinib", "investigational", "JAK1"),
               ("T", "adalimumab", "background", "TNF")),
         "target_contrast", "JAK1"),
        # Fallback, head to head between mechanisms.
        (_rows(("T", "abatacept", "candidate", "CD80"),
               ("T", "adalimumab", "candidate", "TNF")),
         "head_to_head", "CD80;TNF"),
        # Fallback, biosimilar with names that differ.
        (_rows(("T", "bcd-020", "candidate", "CD20"),
               ("T", "mabthera", "candidate", "CD20")),
         "same_target", ""),
        # Fallback, old placebo-controlled trial with no arm records.
        (_rows(("T", "drugx", "candidate", "X"),
               ("T", "placebo", "placebo", None)),
         "target_contrast", "X"),
        # Fallback, one therapeutic and nothing else.
        (_rows(("T", "drugx", "candidate", "X")),
         "single_intervention", "X"),
        # Tested intervention not mapped.
        (_rows(("T", "art621", "investigational", None),
               ("T", "placebo", "placebo", None)),
         "unmapped", ""),
        # Nothing tested.
        (_rows(("T", "adalimumab", "comparator", "TNF")),
         "no_test", ""),
    ],
)
def test_classify_target_contrast(rows, expected_class, expected_targets):
    result = classify_target_contrast(rows).iloc[0]
    assert result.contrast_class == expected_class
    assert result.contrast_targets == expected_targets


def test_unmapped_test_flag():
    rows = _rows(("T", "drugx", "investigational", "X"),
                 ("T", "drugy", "investigational", None))
    assert classify_target_contrast(rows).iloc[0].has_unmapped_test


class FakeClient:
    """Stands in for ChemblClient: known hits, a pre-filled cache, a request log."""

    def __init__(self, hits, cached):
        self.hits = hits          # {(name, field): hit}
        self.cached = cached      # set of (name, field) already in the cache
        self.requests = []        # live requests made

    def find_molecule(self, name, field, cached_only=False):
        if (name, field) not in self.cached:
            if cached_only:
                return None
            self.requests.append((name, field))
        return self.hits.get((name, field))


def test_lookup_uses_cached_hit_without_requests():
    from causal_target_to_patient.sources.chembl import PREF_NAME_FIELD, lookup_component

    hit = {"molecule_chembl_id": "CHEMBL1"}
    client = FakeClient({("drugx", PREF_NAME_FIELD): hit}, {("drugx", PREF_NAME_FIELD)})
    assert lookup_component(["drugx"], client) == (hit, "drugx")
    assert client.requests == []


def test_lookup_unmatched_costs_one_request_per_variant_plus_one():
    from causal_target_to_patient.sources.chembl import (
        PREF_NAME_FIELD, SYNONYM_FIELD, lookup_component,
    )

    client = FakeClient({}, set())
    assert lookup_component(["a", "b", "c"], client) == (None, None)
    assert client.requests == [("a", SYNONYM_FIELD), ("b", SYNONYM_FIELD),
                               ("c", SYNONYM_FIELD), ("a", PREF_NAME_FIELD)]


@pytest.mark.parametrize(
    "raw, expected_first",
    [
        ("Humira®", ["humira"]),
        ("Subcutaneous (SC) Abatacept", ["abatacept"]),          # "sc" not searched
        ("Comparator: Diclofenac sodium", ["diclofenac sodium"]),
        ("EU-approved RoActemra", ["roactemra"]),
    ],
)
def test_name_variants_cleaning(raw, expected_first):
    components = name_variants(raw)
    assert len(components) == 1
    assert components[0][: len(expected_first)] == expected_first
    assert "sc" not in components[0]


def test_name_variants_comma_list_but_not_code():
    assert [c[0] for c in name_variants("MK0663, etoricoxib")] == ["mk0663", "etoricoxib"]
    assert len(name_variants("cp-690,550")) == 1


def test_short_or_numeric_brackets_not_searched():
    variants = name_variants("p38 inhibitor (4) 50mg")[0]
    assert "4" not in variants


def test_map_names_overrides():
    from causal_target_to_patient.sources.chembl import SYNONYM_FIELD, map_names

    hit = {"molecule_chembl_id": "CHEMBL34259", "parent_chembl_id": "CHEMBL34259"}
    client = FakeClient({("methotrexate", SYNONYM_FIELD): hit}, set())
    out = map_names(
        ["MTX", "Tc99m-tilmanocept", "abp 501"],
        client,
        {"mtx": "methotrexate", "tc99m-tilmanocept": "EXCLUDE", "abp 501": "CHEMBL1201580"},
        progress=False,
    ).set_index("intervention_key")
    assert out.loc["mtx", "matched_on"] == "alias:methotrexate"
    assert out.loc["mtx", "molecule_chembl_id"] == "CHEMBL34259"
    assert out.loc["tc99m-tilmanocept", "excluded"]
    assert out.loc["abp 501", "parent_chembl_id"] == "CHEMBL1201580"


def test_mechanisms_queried_by_parent_and_deduplicated(tmp_path):
    import json
    from causal_target_to_patient.sources.chembl import ChemblClient

    client = ChemblClient(tmp_path / "cache.json")
    params = {"parent_molecule_chembl_id": "P1", "limit": 100}
    mech = {"mechanism_of_action": "JAK1 inhibitor", "action_type": "INHIBITOR",
            "target_chembl_id": "T1"}
    # The same mechanism recorded on the parent and on its salt form.
    client.cache["mechanism?" + json.dumps(params, sort_keys=True)] = {
        "mechanisms": [dict(mech, molecule_chembl_id="P1"),
                       dict(mech, molecule_chembl_id="P1_SALT")]}
    assert client.mechanisms("P1") == [dict(mech, parent_chembl_id="P1")]


def test_excluded_therapy_class():
    rows = _rows(("T", "methotrexate", "investigational", "DHFR"),
                 ("T", "placebo", "placebo", None))
    rows["excluded_therapy"] = [True, False]
    assert classify_target_contrast(rows).iloc[0].contrast_class == "excluded_therapy"

    # Excluded background doesn't mask a real test.
    rows = _rows(("T", "drugx", "investigational", "X"),
                 ("T", "methotrexate", "investigational", "DHFR"))
    rows["excluded_therapy"] = [False, True]
    result = classify_target_contrast(rows).iloc[0]
    assert (result.contrast_class, result.contrast_targets) == ("target_contrast", "X")


def test_flag_excluded_therapies():
    from causal_target_to_patient.sources.chembl import flag_excluded_therapies

    molecules = pd.DataFrame({
        "intervention_key": ["mtx", "diclofenac", "tocilizumab", "vaccine", "unknown"],
        "excluded": [False, False, False, True, None],      # None: unmatched row
        "molecule_pref_name": ["METHOTREXATE", "DICLOFENAC SODIUM", "TOCILIZUMAB",
                               None, None],
    })
    flags = flag_excluded_therapies(molecules, {"METHOTREXATE", "DICLOFENAC"})
    assert list(flags) == [True, True, False, True, False]


def test_fallback_excluded_therapy_acts_as_control():
    # "New drug vs methotrexate" with uninformative arms.
    rows = _rows(("T", "drugx", "candidate", "X"),
                 ("T", "methotrexate", "candidate", "DHFR"))
    rows["excluded_therapy"] = [False, True]
    result = classify_target_contrast(rows).iloc[0]
    assert (result.contrast_class, result.contrast_targets) == ("target_contrast", "X")


def test_quoted_names_cleaned():
    assert name_variants('"baricitinib", "olumiant®"')[0][0] == "baricitinib"


def test_dose_only_component_dropped():
    assert [c[0] for c in name_variants("AP1189, 100 mg")] == ["ap1189"]


def test_component_level_overrides():
    from causal_target_to_patient.sources.chembl import resolve_components

    overrides = {"rc18": "telitacicept", "mtx": "methotrexate", "mm-093": "EXCLUDE"}
    resolved = resolve_components("RC18 160 mg plus MTX", overrides)
    assert [(k, v[0], via) for k, v, via in resolved] == [
        ("lookup", "telitacicept", "alias:"), ("lookup", "methotrexate", "alias:")]
    assert resolve_components("MM-093", overrides) == [("exclude", None, "override:mm-093")]


def test_apply_manual_mechanisms():
    from causal_target_to_patient.sources.chembl import apply_manual_mechanisms

    molecules = pd.DataFrame({
        "intervention_key": ["tofacitinib", "tll-018", "natrunix", "old drug"],
        "lookup_name": ["tofacitinib", "tll-018", "natrunix", "old drug"],
        "molecule_pref_name": ["TOFACITINIB", None, "NATRUNIX", "OLDDRUG"],
        "parent_chembl_id": ["P_TOFA", None, "P_NAT", "P_OLD"],
    })
    targets = pd.DataFrame({"parent_chembl_id": ["P_TOFA"], "target_chembl_id": ["JAK1"],
                            "target_pref_name": ["JAK1"], "gene_symbols": ["JAK1"]})
    manual = pd.DataFrame({
        "name": ["TLL-018", "tll-018", "natrunix", "tofacitinib"],
        "target_chembl_id": ["JAK1", "TYK2", "GENE:IL1A", "SHOULD_NOT_APPLY"],
        "target_pref_name": ["JAK1", "TYK2", "IL-1a", "x"],
        "gene_symbols": ["JAK1", "TYK2", "IL1A", "x"],
        "source": ["s1", "s1", "s2", "s3"],
    })
    mol, tgt = apply_manual_mechanisms(molecules, targets, manual)
    keys = dict(zip(mol.intervention_key, mol.mechanism_key))
    assert keys == {"tofacitinib": "P_TOFA", "tll-018": "MANUAL:tll-018",
                    "natrunix": "P_NAT", "old drug": "P_OLD"}
    by_key = tgt.groupby("parent_chembl_id").target_chembl_id.apply(set).to_dict()
    assert by_key["MANUAL:tll-018"] == {"JAK1", "TYK2"}
    assert by_key["P_NAT"] == {"GENE:IL1A"}            # ChEMBL molecule, no mechanism
    assert by_key["P_TOFA"] == {"JAK1"}                # ChEMBL mechanism kept, manual ignored
    assert set(tgt.loc[tgt.parent_chembl_id == "P_NAT", "mechanism_source"]) == {"manual: s2"}


class FakeIndicationClient:
    """Two pages of indication records for one filter, nothing for the other."""

    def __init__(self):
        self.calls = []

    def drug_indications(self, **filters):
        self.calls.append(filters)
        if "mesh_id" in filters:
            return [
                {"parent_molecule_chembl_id": "P1", "max_phase_for_ind": "4.0"},
                {"parent_molecule_chembl_id": "P1", "max_phase_for_ind": "2.0"},
                {"parent_molecule_chembl_id": "P2", "max_phase_for_ind": "1.0"},
                {"molecule_chembl_id": "P3", "max_phase_for_ind": "2"},
                {"parent_molecule_chembl_id": "P4", "max_phase_for_ind": None},
            ]
        return [{"parent_molecule_chembl_id": "P3", "max_phase_for_ind": "3.0"}]


def test_ra_indication_molecules():
    from causal_target_to_patient.sources.chembl import ra_indication_molecules

    client = FakeIndicationClient()
    out = ra_indication_molecules(client, min_phase=2).set_index("parent_chembl_id")
    assert out.max_phase_for_ra.to_dict() == {"P1": 4.0, "P3": 3.0}   # P2 phase 1, P4 no phase
    assert len(client.calls) == 2                                      # MeSH and EFO


def test_drug_indications_pagination(tmp_path):
    import json
    from causal_target_to_patient.sources.chembl import ChemblClient

    client = ChemblClient(tmp_path / "cache.json")
    def key(offset):
        return "drug_indication?" + json.dumps(
            {"mesh_id": "D001172", "limit": 1000, "offset": offset}, sort_keys=True)
    client.cache[key(0)] = {"drug_indications": [{"a": 1}, {"a": 2}],
                            "page_meta": {"next": "/next"}}
    client.cache[key(2)] = {"drug_indications": [{"a": 3}], "page_meta": {"next": None}}
    assert [r["a"] for r in client.drug_indications(mesh_id="D001172")] == [1, 2, 3]


def test_single_intervention_with_control_arm_is_contrast():
    rows = _rows(("T", "efalizumab", "candidate", "LFA1"))
    assert classify_target_contrast(rows).iloc[0].contrast_class == "single_intervention"
    result = classify_target_contrast(rows, controlled_trials={"T"}).iloc[0]
    assert (result.contrast_class, result.contrast_targets) == ("target_contrast", "LFA1")
