"""Pull ChEMBL's RA drug indications (phase 2+) with mechanisms and targets.

A reference for the coverage check, independent of the AACT trial pull except
where ChEMBL itself draws on ClinicalTrials.gov. Conventional and symptomatic
therapies on curated/trials/excluded_therapies.csv are dropped.

Writes chembl_ra_indications_YYYYMMDD.csv to --data-dir: one row per molecule
and target. Reuses the ChEMBL response cache in --data-dir.

Usage:
    python scripts/pull_chembl_ra_indications.py --data-dir ../causal-data/aact
"""

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from causal_target_to_patient.sources.chembl import (
    ChemblClient,
    is_excluded_name,
    map_targets,
    ra_indication_molecules,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = REPO_ROOT / "curated" / "trials" / "excluded_therapies.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    client = ChemblClient(args.data_dir / "chembl_cache.json")
    try:
        molecules = ra_indication_molecules(client, min_phase=2)
        print(f"Parent molecules with an RA indication at phase 2+: {len(molecules)}")
        names = [client.molecule(pid)["molecule_pref_name"]
                 for pid in tqdm(molecules.parent_chembl_id, desc="Fetching names")]
        molecules["molecule_pref_name"] = names
        excluded_names = set(pd.read_csv(EXCLUDED).molecule_pref_name.str.upper())
        excluded = molecules.molecule_pref_name.map(lambda n: is_excluded_name(n, excluded_names))
        molecules = molecules[~excluded]
        print(f"After dropping excluded therapies: {len(molecules)}")
        targets = map_targets(sorted(molecules.parent_chembl_id), client)
    finally:
        client.save()

    reference = molecules.merge(targets, on="parent_chembl_id", how="left")
    no_mechanism = reference.target_chembl_id.isna().sum()
    print(f"Molecules with no ChEMBL mechanism: {no_mechanism}")
    print("\nBy highest phase for RA:\n",
          molecules.max_phase_for_ra.value_counts().sort_index(ascending=False).to_string())

    stamp = date.today().strftime("%Y%m%d")
    out = args.data_dir / f"chembl_ra_indications_{stamp}.csv"
    reference.to_csv(out, index=False)
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
