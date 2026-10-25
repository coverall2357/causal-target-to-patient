"""Label trial efficacy outcomes with an LLM, from evidence packets.

Reads <data-dir>/evidence/NCT*.md and NCT*.json (built by
build_evidence_packets.py) and asks the model for a structured label per trial.
Needs ANTHROPIC_API_KEY in .env. Writes to <data-dir>/labels:
    NCT*.json            the model's answer with run metadata, one per trial
    <run>_labels.csv     all answers, one row per trial
    <run>_review.csv     the review sheet: context (title, investigational
                         drug, evidence tier, packet path), then the columns to
                         fill in (your_label, your_population, agree, issue,
                         review_notes), then the model's answer and checks
where <run> is the --nct-file name (e.g. dev_set_20261022) or adhoc_YYYYMMDD.

With --blind, no model calls are made and only <run>_blind.csv is written: the
context and the columns to fill in, without any model answers. Use it for the
held-out test set, which is labelled by hand first.

With --reuse-saved, no model calls are made: the answers are read from the
saved NCT*.json files and the run files are rebuilt from them.

Existing run files are not overwritten without --force, since a sheet may
already hold hand labels. The labelling rules are in docs/labelling.md.

Usage:
    python scripts/label_trials.py --data-dir ../causal-data/aact \
        --nct-file ../causal-data/aact/labels/dev_set_20261022.csv
    python scripts/label_trials.py --data-dir ../causal-data/aact \
        --nct-file ../causal-data/aact/labels/dev_set_20261022.csv --reuse-saved --force
    python scripts/label_trials.py --data-dir ../causal-data/aact \
        --nct-file ../causal-data/aact/labels/test_set_YYYYMMDD.csv --blind
"""

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from causal_target_to_patient.labelling.extract import (
    DEFAULT_MODEL,
    extract_label,
    review_row,
)
from causal_target_to_patient.labelling.review import blind_sheet, review_sheet, trial_context
from causal_target_to_patient.labelling.runs import (
    refuse_overwrite,
    run_name,
    run_outputs,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    ids = parser.add_mutually_exclusive_group(required=True)
    ids.add_argument("--nct-ids", nargs="+")
    ids.add_argument("--nct-file", type=Path,
                     help="CSV with an nct_id column, such as a dev or test set")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--reuse-saved", action="store_true",
                      help="rebuild the run files from saved NCT*.json; no model calls")
    mode.add_argument("--blind", action="store_true",
                      help="write only a blind sheet for hand labelling; no model calls")
    parser.add_argument("--force", action="store_true",
                        help="replace existing run files")
    args = parser.parse_args()

    trials = (pd.read_csv(args.nct_file) if args.nct_file
              else pd.DataFrame({"nct_id": args.nct_ids}))
    nct_ids = trials["nct_id"].tolist()
    tiers = (dict(zip(trials.nct_id, trials.evidence_tier))
             if "evidence_tier" in trials.columns else {})
    evidence_dir = args.data_dir / "evidence"
    out_dir = args.data_dir / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    name = run_name(args.nct_file, date.today())
    outputs = run_outputs(out_dir, name)

    context = []
    for nct_id in nct_ids:
        packet = json.loads((evidence_dir / f"{nct_id}.json").read_text())
        context.append(trial_context(nct_id, packet,
                                     (evidence_dir / f"{nct_id}.md").resolve(),
                                     tiers.get(nct_id)))

    if args.blind:
        refuse_overwrite([outputs["blind"]], force=args.force)
        blind_sheet(context).to_csv(outputs["blind"], index=False)
        print(f"Written {outputs['blind'].name} ({len(nct_ids)} trials) to {out_dir}")
        return

    refuse_overwrite([outputs["labels"], outputs["review"]], force=args.force)
    if args.reuse_saved:
        missing = [n for n in nct_ids if not (out_dir / f"{n}.json").exists()]
        if missing:
            raise FileNotFoundError(f"no saved label for {', '.join(missing)}")
        client = None
    else:
        import anthropic

        load_dotenv()
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    labels, rows = [], []
    for nct_id in nct_ids:
        packet_text = (evidence_dir / f"{nct_id}.md").read_text()
        if args.reuse_saved:
            label = json.loads((out_dir / f"{nct_id}.json").read_text())
        else:
            label = extract_label(client, packet_text, model=args.model)
            (out_dir / f"{nct_id}.json").write_text(json.dumps(label, indent=2))
        labels.append({"nct_id": nct_id, **label})
        rows.append(review_row(nct_id, label, packet_text))
        print(f"{nct_id}: {label['label']} ({label['basis']}, {label['confidence']}; "
              f"{label.get('population')})"
              + ("" if args.reuse_saved else
                 f"; {label['input_tokens']} in / {label['output_tokens']} out tokens"))

    pd.DataFrame(labels).to_csv(outputs["labels"], index=False)
    review = review_sheet(context, rows)
    review.to_csv(outputs["review"], index=False)

    models = sorted({str(l.get("model")) for l in labels})
    print(f"\nRun: {name}. Model: {', '.join(models)}"
          + (" (saved answers reused)" if args.reuse_saved else ""))
    print("Quotes found in packet:", int(review.quote_found.sum()), "of", len(review))
    missing = review.effect_numbers_missing.fillna("").str.len().gt(0)
    if missing.any():
        print("Effect numbers not found in packet:\n",
              review.loc[missing, ["nct_id", "effect_numbers_missing"]].to_string(index=False))
    problems = review.problems.fillna("").str.len().gt(0)
    if problems.any():
        print("Structural problems:\n",
              review.loc[problems, ["nct_id", "problems"]].to_string(index=False))
    if not args.reuse_saved:
        print(f"Tokens: {sum(l['input_tokens'] for l in labels)} in, "
              f"{sum(l['output_tokens'] for l in labels)} out")
    print(f"\nWritten {outputs['labels'].name} and {outputs['review'].name} to {out_dir}")


if __name__ == "__main__":
    main()
