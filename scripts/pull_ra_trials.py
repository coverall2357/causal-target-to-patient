"""Pull the RA trial universe from AACT and write it as raw data.

Writes two tables, dated with today's AACT snapshot (AACT refreshes daily):
    ra_trials_YYYYMMDD.csv         one row per eligible trial, with trial set,
                                   arm counts, and review flag
    ra_interventions_YYYYMMDD.csv  one row per drug or biologic intervention,
                                   with its role in the trial
                                   (arm_role keeps the role before fallback)

Usage:
    python scripts/pull_ra_trials.py --out-dir ../causal-data/aact
"""

import argparse
from datetime import date
from pathlib import Path

from causal_target_to_patient.sources.aact import (
    apply_candidate_fallback,
    flag_trials_for_review,
    get_engine,
    pull_ra_trial_arms,
    pull_ra_trial_interventions,
    pull_ra_trials,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    engine = get_engine()
    trials = pull_ra_trials(engine)
    interventions = pull_ra_trial_interventions(engine)
    arms = flag_trials_for_review(interventions, pull_ra_trial_arms(engine))
    trials = trials.merge(arms, on="nct_id", how="left")
    interventions = apply_candidate_fallback(interventions, arms)

    print(f"Eligible trials: {len(trials)}\n")
    print("By trial set and phase:\n",
          trials.groupby(["trial_set", "phase"]).size().to_string(), "\n")

    print("Intervention roles:\n", interventions.role.value_counts().to_string(), "\n")
    print("Trials with no arm records:", int((trials.n_arms == 0).sum()))
    print("Trials by number of investigational interventions:\n",
          trials.n_investigational.value_counts().sort_index().to_string(), "\n")
    print("Trials flagged for review, by trial set:\n",
          trials[trials.needs_review].trial_set.value_counts().to_string(), "\n")

    investigational = interventions[interventions.role == "investigational"]
    print(f"Investigational: {len(investigational)} rows, "
          f"{investigational.intervention_name.str.lower().nunique()} distinct names")
    print("Most frequent investigational names:\n",
          investigational.intervention_name.str.lower()
          .value_counts().head(30).to_string(), "\n")
    print("Most frequent background names:\n",
          interventions[interventions.role == "background"].intervention_name
          .str.lower().value_counts().head(10).to_string())

    stamp = date.today().strftime("%Y%m%d")
    trials.to_csv(args.out_dir / f"ra_trials_{stamp}.csv", index=False)
    interventions.to_csv(args.out_dir / f"ra_interventions_{stamp}.csv", index=False)
    print(f"\nWritten to {args.out_dir}")


if __name__ == "__main__":
    main()
