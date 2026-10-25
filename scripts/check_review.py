"""Check the hand labels in a review or blind sheet for values outside the allowed sets.

Run after filling in a sheet, before agreement is scored, so that typos
(for example "mett" or "MTX-IR" instead of "mtx_ir") are fixed rather than
counted as disagreements. Prints each problem and a count of rows labelled.

Allowed values (blank is always allowed):
    your_label       met, failed, mixed, untested, terminated_futility,
                     terminated_other, no_results
    your_population  mtx_naive, mtx_ir, csdmard_ir, biologic_ir, mixed,
                     other, not_stated
    agree            yes, no
    issue            one or more of wrong_label, wrong_population,
                     wrong_comparison, wrong_endpoint, wrong_effect, bad_quote,
                     missed_evidence, efficacy_flag_wrong, other (separated by ;)

Usage:
    python scripts/check_review.py ../causal-data/aact/labels/dev_set_20261022_review.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from causal_target_to_patient.labelling.review import check_hand_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sheet", type=Path)
    args = parser.parse_args()

    sheet = pd.read_csv(args.sheet, dtype=str, keep_default_na=False)
    labelled = sheet.your_label.str.strip().ne("") | sheet.agree.str.strip().ne("")
    print(f"{args.sheet.name}: {int(labelled.sum())} of {len(sheet)} rows have hand labels")
    problems = check_hand_labels(sheet)
    if problems:
        print(f"{len(problems)} problem(s):")
        print("\n".join(f"  {p}" for p in problems))
        sys.exit(1)
    print("All entries are allowed values.")


if __name__ == "__main__":
    main()
