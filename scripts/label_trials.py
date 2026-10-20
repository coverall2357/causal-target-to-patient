"""Label trial efficacy outcomes with an LLM, from evidence packets.

Reads <data-dir>/evidence/NCT*.md (built by build_evidence_packets.py) and asks
the model for a structured label per trial. Needs ANTHROPIC_API_KEY in .env.
Writes, with today's date:
    labels/NCT*.json                   the model's answer with run metadata
    labels/llm_labels_YYYYMMDD.csv     all answers, one row per trial
    labels/dev_review_YYYYMMDD.csv     answers plus checks and empty review
                                       columns (your_label, agree, issue,
                                       review_notes)

Usage:
    python scripts/label_trials.py --data-dir ../causal-data/aact \
        --nct-ids NCT02889796 NCT00413660 NCT00106574 NCT01679951 NCT01430507
"""

import argparse
import json
from datetime import date
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv

from causal_target_to_patient.labelling.extract import (
    DEFAULT_MODEL,
    extract_label,
    review_row,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--nct-ids", nargs="+", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    evidence_dir = args.data_dir / "evidence"
    out_dir = args.data_dir / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    load_dotenv()
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    labels, rows = [], []
    for nct_id in args.nct_ids:
        packet_text = (evidence_dir / f"{nct_id}.md").read_text()
        label = extract_label(client, packet_text, model=args.model)
        (out_dir / f"{nct_id}.json").write_text(json.dumps(label, indent=2))
        labels.append({"nct_id": nct_id, **label})
        rows.append(review_row(nct_id, label, packet_text))
        print(f"{nct_id}: {label['label']} ({label['basis']}, {label['confidence']}); "
              f"{label['input_tokens']} in / {label['output_tokens']} out tokens")

    stamp = f"{date.today():%Y%m%d}"
    pd.DataFrame(labels).to_csv(out_dir / f"llm_labels_{stamp}.csv", index=False)
    review = pd.DataFrame(rows)
    review.to_csv(out_dir / f"dev_review_{stamp}.csv", index=False)

    print(f"\nModel: {args.model}")
    print("Quotes found in packet:", int(review.quote_found.sum()), "of", len(review))
    if review.problems.str.len().gt(0).any():
        print("Structural problems:\n", review.loc[review.problems != "",
                                                   ["nct_id", "problems"]].to_string(index=False))
    print(f"Tokens: {sum(l['input_tokens'] for l in labels)} in, "
          f"{sum(l['output_tokens'] for l in labels)} out")
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()
