"""Compare the pipeline's tested mechanisms with ChEMBL and the curated expected list.

Reads from --data-dir the latest ra_trial_contrasts_*, ra_molecule_targets_*
(same date stamp), and chembl_ra_indications_*; from curated/trials the
expected list and mechanism groups. Labelling trials with a target contrast or
head-to-head comparison count as tests of a mechanism.

Writes mechanism_coverage_YYYYMMDD.csv (stamp of the contrasts file) with one
row per reference and mechanism, and prints the gaps.

Usage:
    python scripts/compare_mechanism_coverage.py --data-dir ../causal-data/aact
"""

import argparse
from pathlib import Path

import pandas as pd

from causal_target_to_patient.files import latest_dated, stamp_of
from causal_target_to_patient.coverage import (
    PIPELINE_ONLY,
    REFERENCE_ONLY,
    chembl_mechanisms,
    compare,
    expected_mechanisms,
    load_groups,
    pipeline_mechanisms,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CURATED = REPO_ROOT / "curated" / "trials"


def report(name: str, table: pd.DataFrame) -> None:
    counts = table.category.value_counts()
    print(f"\n=== {name} ===")
    print(counts.to_string())
    gaps = table[table.category == REFERENCE_ONLY]
    print(f"\nIn {name} but not tested in the pipeline ({len(gaps)}):")
    print(gaps[["mechanism", "reference_detail"]].to_string(index=False, max_colwidth=90)
          if len(gaps) else "  none")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    contrasts_path = latest_dated(args.data_dir, "ra_trial_contrasts")
    stamp = stamp_of(contrasts_path)
    contrasts = pd.read_csv(contrasts_path)
    targets = pd.read_csv(args.data_dir / f"ra_molecule_targets_{stamp}.csv")
    chembl = pd.read_csv(latest_dated(args.data_dir, "chembl_ra_indications"))
    groups = load_groups(pd.read_csv(CURATED / "mechanism_groups.csv"))
    expected = pd.read_csv(CURATED / "expected_mechanisms.csv")

    pipeline = pipeline_mechanisms(contrasts, targets, groups)
    print(f"Mechanisms tested in labelling trials: {len(pipeline)}")

    tables = []
    for name, ref in [("expected list", expected_mechanisms(expected, groups)),
                      ("ChEMBL RA indications", chembl_mechanisms(chembl, groups))]:
        table = compare(pipeline, ref)
        report(name, table)
        tables.append(table.assign(reference=name))

    # Pipeline mechanisms in neither reference, for a sanity check.
    in_any = set().union(*(set(t.loc[t.category != PIPELINE_ONLY, "mechanism"]) for t in tables))
    unexpected = pipeline[~pipeline.mechanism.isin(in_any)].sort_values("n_trials", ascending=False)
    print(f"\nTested in the pipeline, in neither reference ({len(unexpected)}):")
    print(unexpected.to_string(index=False) if len(unexpected) else "  none")

    out = args.data_dir / f"mechanism_coverage_{stamp}.csv"
    pd.concat(tables, ignore_index=True).to_csv(out, index=False)
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
