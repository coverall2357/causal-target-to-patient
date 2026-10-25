"""Choose the development set of trials for hand labelling.

Draws trials from the labelling trials that test a contrast, stratified by
evidence tier (counts in labelling/sampling.py), with a fixed seed. The five
pilot trials (PILOT_TRIALS) are left out. The same evidence file and seed
always give the same set.

Reads the latest ra_trial_evidence_*.csv and the ra_trials_*.csv with the same
date stamp from --data-dir. Writes, with today's date:
    labels/dev_set_YYYYMMDD.csv  nct_id, evidence tier, contrast class, phase,
                                 status, title

An existing dev set is never replaced without --force: once drawn, it is fixed,
and its trials may already be hand-labelled. The held-out test set will be
drawn the same way later, excluding this set.

Usage:
    python scripts/select_dev_set.py --data-dir ../causal-data/aact
"""

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from causal_target_to_patient.files import latest_dated, stamp_of
from causal_target_to_patient.labelling.runs import refuse_overwrite
from causal_target_to_patient.labelling.sampling import (
    DEV_SET_COUNTS,
    PILOT_TRIALS,
    eligible_for_labelling,
    select_stratified,
)

SEED = 20261022


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true",
                        help="replace an existing dev set")
    args = parser.parse_args()
    labels_dir = args.data_dir / "labels"
    refuse_overwrite(sorted(labels_dir.glob("dev_set_*.csv")), force=args.force)

    source = latest_dated(args.data_dir, "ra_trial_evidence")
    stamp = stamp_of(source)
    evidence = pd.read_csv(source)
    titles = pd.read_csv(args.data_dir / f"ra_trials_{stamp}.csv",
                         usecols=["nct_id", "brief_title"])
    print(f"Read {source.name}; leaving out the {len(PILOT_TRIALS)} pilot trials")

    pool = eligible_for_labelling(evidence)
    print("Labelling trials that test a contrast, by evidence tier:\n",
          pool.evidence_tier.value_counts().to_string())

    dev = select_stratified(pool, DEV_SET_COUNTS, seed=args.seed, exclude=set(PILOT_TRIALS))
    dev = dev.merge(titles, on="nct_id", how="left")[
        ["nct_id", "evidence_tier", "contrast_class", "phase", "overall_status", "brief_title"]]

    labels_dir.mkdir(parents=True, exist_ok=True)
    out = labels_dir / f"dev_set_{date.today():%Y%m%d}.csv"
    dev.to_csv(out, index=False)
    print(f"\nDev set ({len(dev)} trials, seed {args.seed}):\n", dev.to_string(index=False))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
