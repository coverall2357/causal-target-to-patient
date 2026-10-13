"""Map RA trial interventions to ChEMBL targets and classify each trial's contrast.

Reads the latest ra_interventions_YYYYMMDD.csv from --data-dir and writes, with
the same date stamp:
    ra_name_molecules_*.csv    intervention name -> ChEMBL molecule
    ra_molecule_targets_*.csv  molecule -> mechanism -> target (gene symbols)
    ra_trial_contrasts_*.csv   per trial: contrast class and contrasted targets
    ra_unmatched_names_*.csv   names with no ChEMBL match, by number of trials

Curated inputs (committed, edited by hand):
    curated/trials/intervention_overrides.csv  name -> name to look up, ChEMBL ID,
                                               or EXCLUDE
    curated/trials/excluded_therapies.csv      conventional and symptomatic
                                               therapies never counted as tested
    curated/trials/manual_mechanisms.csv       targets, with sources, for molecules
                                               ChEMBL lacks or has no mechanism for

Usage:
    python scripts/map_ra_interventions.py --data-dir ../causal-data/aact
"""

import argparse
from pathlib import Path

import pandas as pd

from causal_target_to_patient.sources.chembl import (
    apply_manual_mechanisms,
    ChemblClient,
    classify_target_contrast,
    flag_excluded_therapies,
    map_names,
    map_targets,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CURATED = REPO_ROOT / "curated" / "trials"
OVERRIDES = CURATED / "intervention_overrides.csv"
EXCLUDED = CURATED / "excluded_therapies.csv"
MANUAL = CURATED / "manual_mechanisms.csv"



def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    source = sorted(args.data_dir.glob("ra_interventions_*.csv"))[-1]
    stamp = source.stem.rsplit("_", 1)[-1]
    interventions = pd.read_csv(source)
    interventions["intervention_key"] = interventions.intervention_name.str.strip().str.lower()
    trials = pd.read_csv(args.data_dir / f"ra_trials_{stamp}.csv")
    print(f"Read {source.name}: {len(interventions)} rows")

    overrides_df = pd.read_csv(OVERRIDES)
    overrides = dict(zip(overrides_df.intervention_name.str.strip().str.lower(),
                         overrides_df.maps_to))
    excluded_names = set(pd.read_csv(EXCLUDED).molecule_pref_name.str.upper())

    to_map = interventions[interventions.role != "placebo"]
    names = sorted(to_map.intervention_key.unique())
    print(f"Mapping {len(names)} distinct names (cached responses are reused)...")

    client = ChemblClient(args.data_dir / "chembl_cache.json")
    try:
        molecules = map_names(names, client, overrides)
        parents = sorted(molecules.parent_chembl_id.dropna().unique())
        print(f"Fetching mechanisms for {len(parents)} molecules...")
        targets = map_targets(parents, client)
    finally:
        client.save()

    molecules["excluded_therapy"] = flag_excluded_therapies(molecules, excluded_names)
    no_mechanism = set(parents) - set(targets.parent_chembl_id) if len(targets) else set(parents)
    molecules, targets = apply_manual_mechanisms(molecules, targets, pd.read_csv(MANUAL))

    # One row per intervention-target pair, for the contrast.
    pairs = (
        interventions[["nct_id", "intervention_key", "role"]]
        .merge(molecules[["intervention_key", "mechanism_key", "excluded_therapy"]],
               on="intervention_key", how="left")
        .merge(targets[["parent_chembl_id", "target_chembl_id"]]
               .rename(columns={"parent_chembl_id": "mechanism_key"}),
               on="mechanism_key", how="left")
    )
    contrasts = classify_target_contrast(pairs).merge(
        trials[["nct_id", "trial_set", "phase"]], on="nct_id", how="left"
    )

    excluded_name = molecules.excluded.astype("boolean").fillna(False).astype(bool)
    # Resolved: a ChEMBL molecule, a manual mechanism, or an explicit exclusion.
    resolved = molecules.mechanism_key.notna() | excluded_name
    unmatched = molecules[~resolved]
    unmatched_counts = (
        to_map[to_map.intervention_key.isin(unmatched.intervention_key)]
        .groupby("intervention_key").nct_id.nunique()
        .sort_values(ascending=False).rename("n_trials").reset_index()
    )
    fully_resolved = resolved.groupby(molecules.intervention_key).all()
    print(f"\nNames fully resolved: {fully_resolved.sum()} of {len(fully_resolved)}")
    print(f"Molecules with no ChEMBL mechanism: {len(no_mechanism)}")
    print(f"Components given a manual mechanism: "
          f"{int(molecules.mechanism_key.isin(targets.loc[targets.mechanism_source != 'chembl', 'parent_chembl_id']).sum())}")
    print(f"Name components on the excluded list: {int(molecules.excluded_therapy.sum())}")
    print("\nContrast class by trial set:\n",
          pd.crosstab(contrasts.contrast_class, contrasts.trial_set, margins=True).to_string())
    print("\nMost frequent unmatched names (by trials):\n",
          unmatched_counts.head(30).to_string(index=False))

    labelling = contrasts[(contrasts.trial_set == "labelling") &
                          contrasts.contrast_class.isin(["target_contrast", "head_to_head"])]
    top = (labelling.contrast_targets.str.split(";").explode()
           .value_counts().head(30).rename_axis("target_chembl_id").rename("n_trials").reset_index()
           .merge(targets.drop_duplicates("target_chembl_id")
                  [["target_chembl_id", "target_pref_name", "gene_symbols"]],
                  on="target_chembl_id", how="left"))
    print("\nMost frequently contrasted targets in labelling trials:\n",
          top.to_string(index=False))

    molecules.to_csv(args.data_dir / f"ra_name_molecules_{stamp}.csv", index=False)
    targets.to_csv(args.data_dir / f"ra_molecule_targets_{stamp}.csv", index=False)
    contrasts.to_csv(args.data_dir / f"ra_trial_contrasts_{stamp}.csv", index=False)
    unmatched_counts.to_csv(args.data_dir / f"ra_unmatched_names_{stamp}.csv", index=False)
    print(f"\nWritten to {args.data_dir}")


if __name__ == "__main__":
    main()
