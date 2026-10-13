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
