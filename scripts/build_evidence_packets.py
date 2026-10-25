"""Build evidence packets for LLM outcome labelling.

For each trial, pulls the registered design, the registered and posted primary
outcomes, and the linked publications from AACT, fetches the abstracts from
PubMed, and adds our intervention mapping. Writes to --out-dir (default
<data-dir>/evidence):
    NCT*.json  the packet, exactly as the LLM will receive it
    NCT*.md    the same packet rendered for review

Reads the latest ra_trial_contrasts_*.csv from --data-dir and the mapping files
with the same date stamp. PubMed records are cached in <data-dir>/pubmed_cache.json.

Usage:
    python scripts/build_evidence_packets.py --data-dir ../causal-data/aact \
        --nct-ids NCT02889796 NCT03120949 NCT00106574 NCT01679951 NCT01430507
    python scripts/build_evidence_packets.py --data-dir ../causal-data/aact \
        --nct-file ../causal-data/aact/labels/dev_set_20261022.csv
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from causal_target_to_patient.files import latest_dated, stamp_of
from causal_target_to_patient.labelling.packet import (
    build_packet,
    intervention_mapping,
    packet_counts,
    render_markdown,
)
from causal_target_to_patient.sources.aact import get_engine, pull_trial_evidence
from causal_target_to_patient.sources.pubmed import PubmedClient


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    ids = parser.add_mutually_exclusive_group(required=True)
    ids.add_argument("--nct-ids", nargs="+")
    ids.add_argument("--nct-file", type=Path,
                     help="CSV with an nct_id column, such as a dev or test set")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    nct_ids = (args.nct_ids if args.nct_ids
               else pd.read_csv(args.nct_file)["nct_id"].tolist())
    out_dir = args.out_dir or args.data_dir / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)

    source = latest_dated(args.data_dir, "ra_trial_contrasts")
    stamp = stamp_of(source)
    contrasts = pd.read_csv(source)
    interventions = pd.read_csv(args.data_dir / f"ra_interventions_{stamp}.csv")
    name_molecules = pd.read_csv(args.data_dir / f"ra_name_molecules_{stamp}.csv")
    molecule_targets = pd.read_csv(args.data_dir / f"ra_molecule_targets_{stamp}.csv")
    print(f"Mapping files: {stamp}. Trials: {len(nct_ids)}")

    tables = pull_trial_evidence(get_engine(), nct_ids)
    print("AACT rows pulled:", {k: len(v) for k, v in tables.items()})

    refs = tables["study_references"]
    pmids = [str(int(float(p))) for p in refs.pmid.dropna()]
    load_dotenv()
    pubmed = PubmedClient(args.data_dir / "pubmed_cache.json",
                          api_key=os.environ.get("NCBI_API_KEY"))
    abstracts = pubmed.abstracts(pmids)
    print(f"PubMed records: {len(abstracts)} of {len(set(pmids))} linked PMIDs")

    counts = []
    for nct_id in nct_ids:
        mapping = intervention_mapping(nct_id, interventions, name_molecules,
                                       molecule_targets, contrasts)
        packet = build_packet(nct_id, tables, mapping, abstracts)
        (out_dir / f"{nct_id}.json").write_text(json.dumps(packet, indent=2))
        (out_dir / f"{nct_id}.md").write_text(render_markdown(packet))
        counts.append(packet_counts(packet))

    print("\n", pd.DataFrame(counts).to_string(index=False))
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()
