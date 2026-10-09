"""Pull the RA trial universe from AACT and write it as raw data.

Writes two tables, dated with today's AACT snapshot (AACT refreshes daily):
    ra_trials_YYYYMMDD.csv         one row per eligible trial, with trial_set
    ra_interventions_YYYYMMDD.csv  one row per drug or biologic intervention

Usage:
    python scripts/pull_ra_trials.py --out-dir /path/to/causal-data/aact
"""

import argparse
from datetime import date
from pathlib import Path

from causal_target_to_patient.sources.aact import (
    get_engine,
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

    print(f"Eligible trials: {len(trials)}\n")
    print("By trial set and phase:\n",
          trials.groupby(["trial_set", "phase"]).size().to_string(), "\n")
    print(f"Interventions: {len(interventions)} rows, "
          f"{interventions.intervention_name.str.lower().nunique()} distinct names\n")
    print("Most frequent intervention names:\n",
          interventions.intervention_name.str.lower().value_counts().head(30).to_string())

    stamp = date.today().strftime("%Y%m%d")
    trials.to_csv(args.out_dir / f"ra_trials_{stamp}.csv", index=False)
    interventions.to_csv(args.out_dir / f"ra_interventions_{stamp}.csv", index=False)
    print(f"\nWritten to {args.out_dir}")


if __name__ == "__main__":
    main()
