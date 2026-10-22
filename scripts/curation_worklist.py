"""List the tested interventions that still need curating.

Reads the latest mapping outputs from --data-dir (ra_trial_contrasts_*.csv and
the ra_interventions, ra_name_molecules, and ra_molecule_targets files with the
same date stamp) and curated/trials/unmapped_decisions.csv. Writes:
    curation_worklist_YYYYMMDD.csv   (same stamp as the mapping outputs)

Two kinds of item, most labelling trials first:
    no_mechanism  ChEMBL molecule found, but ChEMBL gives no mechanism
    no_molecule   no ChEMBL molecule found for the cleaned name

Drugs already recorded in unmapped_decisions.csv are left out.

Usage:
    python scripts/curation_worklist.py --data-dir ../causal-data/aact
"""

import argparse
from pathlib import Path

import pandas as pd

from causal_target_to_patient.files import latest_dated, stamp_of
from causal_target_to_patient.curation import curation_worklist

REPO_ROOT = Path(__file__).resolve().parents[1]
DECISIONS = REPO_ROOT / "curated" / "trials" / "unmapped_decisions.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    source = latest_dated(args.data_dir, "ra_trial_contrasts")
    stamp = stamp_of(source)
    read = lambda name: pd.read_csv(args.data_dir / f"{name}_{stamp}.csv")  # noqa: E731
    decided = set(pd.read_csv(DECISIONS).name) if DECISIONS.exists() else set()

    worklist = curation_worklist(read("ra_interventions"), read("ra_name_molecules"),
                                 read("ra_molecule_targets"), pd.read_csv(source), decided)

    print(f"Mapping outputs: {stamp}. Already decided (left out): {len(decided)}\n")
    summary = worklist.groupby("kind").agg(items=("item", "size"),
                                           in_labelling_trials=("n_labelling_trials",
                                                                lambda s: int((s > 0).sum())))
    print(summary.to_string(), "\n")
    pd.set_option("display.width", 200, "display.max_colwidth", 50)
    print("Top items by labelling trials:\n",
          worklist.head(30)[["kind", "item", "n_labelling_trials", "n_trials",
                             "intervention_names"]].to_string(index=False))

    out = args.data_dir / f"curation_worklist_{stamp}.csv"
    worklist.to_csv(out, index=False)
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
