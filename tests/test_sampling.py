"""Tests for choosing the dev and test sets."""

import pandas as pd
import pytest

from causal_target_to_patient.labelling.sampling import (
    eligible_for_labelling,
    select_stratified,
)


def evidence():
    rows = []
    for tier, n in (("primary analysis", 10), ("publication", 6), ("none", 3)):
        rows += [{"nct_id": f"NCT{tier[:3]}{i:03d}", "evidence_tier": tier,
                  "trial_set": "labelling", "contrast_class": "target_contrast"}
                 for i in range(n)]
    rows += [{"nct_id": "NCTctx", "evidence_tier": "none", "trial_set": "context",
              "contrast_class": "target_contrast"},
             {"nct_id": "NCTunm", "evidence_tier": "none", "trial_set": "labelling",
              "contrast_class": "unmapped"}]
    return pd.DataFrame(rows)


COUNTS = {"primary analysis": 4, "publication": 2, "none": 1}


def test_eligible_for_labelling_keeps_labelling_contrast_tests():
    pool = eligible_for_labelling(evidence())
    assert "NCTctx" not in set(pool.nct_id) and "NCTunm" not in set(pool.nct_id)
    assert len(pool) == 19


def test_select_stratified_counts_order_and_reproducibility():
    pool = eligible_for_labelling(evidence())
    a = select_stratified(pool, COUNTS, seed=1)
    assert a.evidence_tier.value_counts().to_dict() == COUNTS
    assert list(dict.fromkeys(a.evidence_tier)) == list(COUNTS)   # tiers in order
    assert a.nct_id.tolist() == select_stratified(pool, COUNTS, seed=1).nct_id.tolist()
    assert a.nct_id.is_unique


def test_select_stratified_excludes_and_never_overlaps():
    pool = eligible_for_labelling(evidence())
    dev = select_stratified(pool, COUNTS, seed=1)
    test = select_stratified(pool, COUNTS, seed=2, exclude=set(dev.nct_id))
    assert not set(dev.nct_id) & set(test.nct_id)


def test_select_stratified_raises_when_a_tier_is_short():
    pool = eligible_for_labelling(evidence())
    with pytest.raises(ValueError, match="tier 'none' has 3 trials, 4 requested"):
        select_stratified(pool, {"none": 4}, seed=1)


def test_pilot_trials_are_five_distinct_nct_ids():
    from causal_target_to_patient.labelling.sampling import PILOT_TRIALS

    assert len(set(PILOT_TRIALS)) == 5
    assert all(n.startswith("NCT") and len(n) == 11 for n in PILOT_TRIALS)
