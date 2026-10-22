"""Check how each eligible trial matched the RA definition (read-only).

For every eligible trial, pulls from AACT whether 'Arthritis, Rheumatoid' is
mapped directly from its conditions or only as an ancestor term, and whether
its condition text names adult or juvenile RA. Joins the trial set and contrast
class from the latest ra_trials_*.csv and ra_trial_contrasts_*.csv in
--data-dir. Writes, with today's date:
    ra_definition_check_YYYYMMDD.csv   one row per eligible trial

Usage:
    python scripts/check_ra_definition.py --data-dir ../causal-data/aact
"""

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from causal_target_to_patient.files import latest_dated
from causal_target_to_patient.sources.aact import (
    get_engine,
    mentions_juvenile,
    pull_ra_condition_matches,
    ra_match_route,
)

TEST_CLASSES = ("target_contrast", "head_to_head")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    trials_file = latest_dated(args.data_dir, "ra_trials")
    contrasts_file = latest_dated(args.data_dir, "ra_trial_contrasts")
    trials = pd.read_csv(trials_file, usecols=["nct_id", "brief_title", "trial_set"])
    contrasts = pd.read_csv(contrasts_file, usecols=["nct_id", "contrast_class"])

    m = pull_ra_condition_matches(get_engine())
    if not m.mesh_type_available.iloc[0]:
        print("NOTE: browse_conditions has no mesh_type column; "
              "direct and ancestor MeSH matches can't be told apart.\n")
    m["route"] = [ra_match_route(d, a, t, ad) for d, a, t, ad in
                  zip(m.mesh_direct, m.mesh_ancestor, m.text_match, m.text_adult)]
    m["mentions_juvenile"] = m.conditions.map(mentions_juvenile)
    m["multi_condition"] = m.n_conditions > 1
    df = m.merge(trials, on="nct_id", how="left").merge(contrasts, on="nct_id", how="left")
    df["contrast_test"] = (df.trial_set == "labelling") & df.contrast_class.isin(TEST_CLASSES)
    print(f"Eligible trials pulled now: {len(df)}; in {trials_file.name}: {len(trials)}\n")

    print("Route into the RA definition, by trial set:\n",
          pd.crosstab(df.route, df.trial_set.fillna("not in current pull"),
                      margins=True).to_string(), "\n")
    print("Labelling contrast tests by route:\n",
          df[df.contrast_test].route.value_counts().to_string(), "\n")

    pd.set_option("display.width", 220, "display.max_colwidth", 90)
    for route in ("mesh_ancestor", "text_juvenile", "text_adult"):
        sub = df[df.route == route]
        print(f"--- {route}: {len(sub)} trials "
              f"({int(sub.contrast_test.sum())} labelling contrast tests). Examples:")
        print(sub[["nct_id", "trial_set", "contrast_class", "conditions"]]
              .head(25).to_string(index=False), "\n")

    multi = df[df.multi_condition & df.route.isin(["mesh_direct", "text_adult"])]
    print(f"--- Kept routes but several conditions listed: {len(multi)} trials "
          f"({int(multi.contrast_test.sum())} labelling contrast tests). "
          "Most conditions first:")
    print(multi.sort_values("n_conditions", ascending=False)
          [["nct_id", "n_conditions", "contrast_class", "conditions"]]
          .head(25).to_string(index=False), "\n")
    juv = df[df.mentions_juvenile & df.route.isin(["mesh_direct", "text_adult"])]
    print(f"--- Kept routes but a juvenile condition also listed: {len(juv)} trials")
    print(juv[["nct_id", "trial_set", "conditions"]].head(15).to_string(index=False))

    out = args.data_dir / f"ra_definition_check_{date.today():%Y%m%d}.csv"
    df.to_csv(out, index=False)
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
