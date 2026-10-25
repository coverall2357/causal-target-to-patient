"""Choosing trials for hand labelling: a development set and a held-out test set.

Both are drawn from the labelling trials that test a contrast (target contrasts
and head-to-heads), stratified by evidence tier so that every kind of evidence
the labeller meets is represented. Draws are seeded, so the same inputs give
the same set, and trials already in another set are excluded, so the dev and
test sets never overlap.
"""

import pandas as pd

TEST_CLASSES = ("target_contrast", "head_to_head")

# The five trials of the first labelling prototype (2026-10-20). They are kept
# out of the dev and test sets. Named explicitly rather than read from the
# labels folder, so the same inputs always give the same sets.
PILOT_TRIALS = ("NCT02889796", "NCT00413660", "NCT00106574", "NCT01679951", "NCT01430507")

# Trials per evidence tier. The dev set over-samples the harder tiers
# (publication only, stop reason, none) relative to their share of the 420.
DEV_SET_COUNTS = {
    "primary analysis": 6,
    "primary results": 5,
    "publication": 5,
    "stop reason": 2,
    "none": 2,
}


def eligible_for_labelling(evidence: pd.DataFrame) -> pd.DataFrame:
    """Labelling trials that test a contrast, from ra_trial_evidence_*.csv."""
    keep = evidence.trial_set.eq("labelling") & evidence.contrast_class.isin(TEST_CLASSES)
    return evidence[keep]


def select_stratified(
    trials: pd.DataFrame,
    counts: dict[str, int],
    seed: int,
    exclude: set[str] = frozenset(),
    tier_column: str = "evidence_tier",
) -> pd.DataFrame:
    """Draw counts[tier] trials from each tier, without the excluded nct_ids.

    Raises ValueError if a tier has fewer trials than requested. Returns the
    chosen rows sorted by tier (in the order of `counts`) and nct_id.
    """
    pool = trials[~trials.nct_id.isin(exclude)]
    chosen = []
    for i, (tier, n) in enumerate(counts.items()):
        rows = pool[pool[tier_column] == tier]
        if len(rows) < n:
            raise ValueError(f"tier {tier!r} has {len(rows)} trials, {n} requested")
        chosen.append(rows.sample(n=n, random_state=seed + i).assign(_order=i))
    out = pd.concat(chosen).sort_values(["_order", "nct_id"]).drop(columns="_order")
    return out.reset_index(drop=True)
