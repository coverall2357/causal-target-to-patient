"""Count the outcome evidence available for each RA trial.

Read-only check before outcome labelling: how many trials can be labelled from
AACT's structured results, and how many need a publication or the stop reason.

Reads the latest ra_trial_contrasts_*.csv and the matching ra_trials_*.csv from
--data-dir. Evidence counts are pulled live from AACT, so they can be a few
days newer than the trial pull. Writes, with today's date:
    ra_trial_evidence_YYYYMMDD.csv  per eligible trial: evidence counts and tier

Usage:
    python scripts/check_evidence_availability.py --data-dir ../causal-data/aact
"""

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from causal_target_to_patient.files import latest_dated, stamp_of
from causal_target_to_patient.sources.aact import (
    EVIDENCE_TIERS,
    evidence_tier,
    get_engine,
    pull_ra_trial_evidence,
)

TEST_CLASSES = ("target_contrast", "head_to_head")
COUNT_COLUMNS = ("n_primary_outcomes", "n_primary_p_values",
                 "n_result_refs", "n_derived_refs")


def _share(mask: pd.Series) -> str:
    return f"{int(mask.sum()):>4}  ({mask.mean():.0%})"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    source = latest_dated(args.data_dir, "ra_trial_contrasts")
    stamp = stamp_of(source)
    contrasts = pd.read_csv(source)
    trials = pd.read_csv(args.data_dir / f"ra_trials_{stamp}.csv",
                         usecols=["nct_id", "overall_status", "why_stopped",
                                  "results_first_submitted_date"])
    print(f"Read {source.name} and ra_trials_{stamp}.csv")

    evidence = pull_ra_trial_evidence(get_engine())
    df = (contrasts[["nct_id", "trial_set", "phase", "contrast_class"]]
          .merge(trials, on="nct_id", how="left")
          .merge(evidence, on="nct_id", how="left"))
    for col in COUNT_COLUMNS:
        df[col] = df[col].fillna(0).astype(int)
    df["n_publications"] = df.n_result_refs + df.n_derived_refs
    df["evidence_tier"] = [
        evidence_tier(o, p, n, w) for o, p, n, w in zip(
            df.n_primary_outcomes, df.n_primary_p_values,
            df.n_publications, df.why_stopped)
    ]
    df["evidence_tier"] = pd.Categorical(df.evidence_tier, categories=EVIDENCE_TIERS,
                                         ordered=True)

    tested = df[(df.trial_set == "labelling") & df.contrast_class.isin(TEST_CLASSES)]
    print(f"\nLabelling trials that test a contrast: {len(tested)}\n")
    print("Evidence held in AACT (a trial can have several):")
    print("  results posted          ", _share(tested.results_first_submitted_date.notna()))
    print("  primary outcome posted  ", _share(tested.n_primary_outcomes > 0))
    print("  primary p-value         ", _share(tested.n_primary_p_values > 0))
    print("  results publication     ", _share(tested.n_result_refs > 0))
    print("  PubMed-linked (derived) ", _share(tested.n_derived_refs > 0))
    print("  any publication         ", _share(tested.n_publications > 0))
    print("  reason for stopping     ", _share(tested.why_stopped.notna()))
    print("  primary outcome posted, no publication ",
          _share((tested.n_primary_outcomes > 0) & (tested.n_publications == 0)))
    print("  publication, no primary outcome posted ",
          _share((tested.n_primary_outcomes == 0) & (tested.n_publications > 0)))

    print("\nBest evidence tier by overall status:\n",
          pd.crosstab(tested.evidence_tier, tested.overall_status,
                      margins=True, dropna=False).to_string())
    print("\nBest evidence tier by phase:\n",
          pd.crosstab(tested.evidence_tier, tested.phase,
                      margins=True, dropna=False).to_string())

    out = args.data_dir / f"ra_trial_evidence_{date.today():%Y%m%d}.csv"
    df.to_csv(out, index=False)
    print(f"\nAll {len(df)} eligible trials written to {out}")


if __name__ == "__main__":
    main()
