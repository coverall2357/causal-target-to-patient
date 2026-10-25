"""Export the trial-to-target map to data/ for publication.

Reads the latest dated outputs of pull_ra_trials.py, map_ra_interventions.py and
check_evidence_availability.py from --data-dir, all with the same date stamp,
and writes:
    data/ra_trials.csv         one row per trial
    data/ra_trial_targets.csv  one row per trial, intervention, component, target
    data/summary.json          headline counts and the source date stamp

The column definitions, licence and attribution are in data/README.md. The
summary feeds the MVP diagram (docs/make_mvp_diagram.py), so published numbers
come from one place.

Usage:
    python scripts/export_map.py --data-dir ../causal-data/aact
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from causal_target_to_patient.files import latest_dated, stamp_of
from causal_target_to_patient.release import summary, trial_table, trial_target_table

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data")
    args = parser.parse_args()

    stamp = stamp_of(latest_dated(args.data_dir, "ra_trial_contrasts"))

    def read(prefix):
        path = args.data_dir / f"{prefix}_{stamp}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path.name} is missing; rerun the pipeline for {stamp}")
        return pd.read_csv(path)

    targets = read("ra_molecule_targets")
    trials = trial_table(read("ra_trials"), read("ra_trial_contrasts"),
                         read("ra_trial_evidence"), targets)
    trial_targets = trial_target_table(read("ra_interventions"),
                                       read("ra_name_molecules"), targets)
    counts = {"source_stamp": stamp, **summary(trials, trial_targets)}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trials.to_csv(args.out_dir / "ra_trials.csv", index=False)
    trial_targets.to_csv(args.out_dir / "ra_trial_targets.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(counts, indent=2) + "\n")

    print(f"Source outputs: *_{stamp}.csv")
    print(f"ra_trials.csv: {len(trials)} trials; "
          f"ra_trial_targets.csv: {len(trial_targets)} rows")
    print(f"Labelling trials: {counts['by_trial_set'].get('labelling', 0)}; "
          f"contrast tests: {counts['labelling_tests']}; "
          f"targets tested: {counts['targets_tested']}")
    print(f"Written to {args.out_dir}")


if __name__ == "__main__":
    main()
