"""Explain each coverage gap: where the missing mechanism's drugs sit in the AACT pull.

For every mechanism a reference lists but the pipeline did not count, finds the
reference's drugs among the pulled trials, by ChEMBL molecule and by name, and
reports why they did not count: not in the pull at all, name not mapped, or
present but in another trial set or contrast class.

Reads the latest mechanism_coverage_*, ra_interventions_*, ra_name_molecules_*,
ra_trial_contrasts_* and chembl_ra_indications_* from --data-dir, plus the
curated expected list and mechanism groups. Writes coverage_gaps_YYYYMMDD.csv.

Usage:
    python scripts/diagnose_coverage_gaps.py --data-dir ../causal-data/aact
"""

import argparse
from pathlib import Path

import pandas as pd

from causal_target_to_patient.coverage import (
    REFERENCE_ONLY,
    diagnose,
    load_groups,
    locate_in_pipeline,
    search_terms,
    to_mechanisms,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CURATED = REPO_ROOT / "curated" / "trials"


def latest(data_dir: Path, prefix: str) -> Path:
    files = sorted(data_dir.glob(f"{prefix}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No {prefix}_*.csv in {data_dir}")
    return files[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    d = args.data_dir

    coverage_path = latest(d, "mechanism_coverage")
    stamp = coverage_path.stem.rsplit("_", 1)[-1]
    coverage = pd.read_csv(coverage_path)
    interventions = pd.read_csv(d / f"ra_interventions_{stamp}.csv")
    interventions["intervention_key"] = interventions.intervention_name.str.strip().str.lower()
    name_molecules = pd.read_csv(d / f"ra_name_molecules_{stamp}.csv")
    contrasts = pd.read_csv(d / f"ra_trial_contrasts_{stamp}.csv")
    chembl = pd.read_csv(latest(d, "chembl_ra_indications"))
    expected = pd.read_csv(CURATED / "expected_mechanisms.csv")
    groups = load_groups(pd.read_csv(CURATED / "mechanism_groups.csv"))

    gaps = coverage[coverage.category == REFERENCE_ONLY]
    rows = []
    for gap in gaps.itertuples(index=False):
        if gap.reference == "ChEMBL RA indications":
            hits = chembl[[gap.mechanism in to_mechanisms(g, groups, n)
                           for g, n in zip(chembl.gene_symbols, chembl.target_pref_name)]]
            drugs = [(name, {pid}, [name]) for pid, name in
                     hits[["parent_chembl_id", "molecule_pref_name"]].drop_duplicates().itertuples(index=False)]
        else:
            hits = expected[[gap.mechanism in to_mechanisms(g, groups, m)
                             for g, m in zip(expected.gene_symbols, expected.mechanism)]]
            drugs = [(ex, set(), search_terms(ex)) for ex in hits.example_therapeutics]
        for label, parent_ids, terms in drugs:
            found = locate_in_pipeline(terms, parent_ids, name_molecules, interventions, contrasts)
            rows.append({
                "reference": gap.reference, "mechanism": gap.mechanism, "drug": label,
                "diagnosis": diagnose(found) if (terms or parent_ids) else "no named drug to search",
                "nct_ids": ";".join(sorted(found.nct_id.unique())),
                "intervention_names": ";".join(sorted(found.intervention_key.unique())),
            })

    # One row per drug: a multi-target drug (dipyridamole) leaves several gaps.
    out = (pd.DataFrame(rows)
           .groupby(["reference", "drug", "diagnosis", "nct_ids", "intervention_names"],
                    as_index=False, dropna=False)
           .agg(mechanism=("mechanism", lambda m: "; ".join(sorted(set(m)))))
           [["reference", "mechanism", "drug", "diagnosis", "nct_ids", "intervention_names"]])
    print(f"Gaps diagnosed: {len(out)} drugs\n")
    print(out.diagnosis.str.split(":").str[0].value_counts().to_string(), "\n")
    pd.set_option("display.width", 200)
    print(out[["reference", "mechanism", "drug", "diagnosis"]]
          .to_string(index=False, max_colwidth=60))
    path = d / f"coverage_gaps_{stamp}.csv"
    out.to_csv(path, index=False)
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()
