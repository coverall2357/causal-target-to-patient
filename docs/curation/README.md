# Curation

The mapping from trial interventions to targets is mostly automatic (ChEMBL), but some steps need judgement. Those judgements live in four hand-edited files under `curated/trials/`. This page sets out the rules for editing them. Each curation batch has its own log in this folder.

## Files

| File | One row per | What it does |
|---|---|---|
| `manual_mechanisms.csv` | drug × target | Gives a target to a drug that ChEMBL lacks, or that ChEMBL has without a mechanism. Used only when ChEMBL records no mechanism. |
| `intervention_overrides.csv` | intervention name | Maps a trial's intervention name to a name to look up (aliases, development codes, brand names), or `EXCLUDE`s it (drug classes, procedures, non-RA trials). |
| `excluded_therapies.csv` | molecule | Conventional and symptomatic therapies that are never counted as the thing a trial tests: csDMARDs, glucocorticoids, NSAIDs, supportive care. |
| `unmapped_decisions.csv` | drug | Drugs deliberately left without a target, and why, so absence is a recorded decision rather than a gap. |

`expected_mechanisms.csv` and `mechanism_groups.csv` are curated too, but they serve the coverage check and the mechanism grouping, and have their own provenance columns.

## Provenance columns

| Column | Content |
|---|---|
| `source` | URL of the source. Required, except for overrides whose `source_type` is `self-evident`. |
| `source_type` | One of: `paper`, `review`, `conference_abstract`, `trial_registry`, `company`, `news`, `drug_database`, `target_database`, `vendor`, `encyclopedia`; and, for overrides only, `self-evident` (abbreviations, drug classes, name cleaning). |
| `evidence` | A verbatim quote from the source, or a structured field such as a database's target entry, supporting the row. A page title counts when it states the claim; the `notes` say so. Required in `manual_mechanisms.csv` and `unmapped_decisions.csv` for every batch after `2026-10-13-initial`. |
| `confidence` | `high`: a clear, specific statement from a paper, trial record, or company. `medium`: a specific statement from a database, vendor, or news source, or an inference from a clear one (for example, a gene chosen for "MK2 pathway"). `low`: a proposed or contested mechanism, or an isoform-unspecified target from a database alone. |
| `notes` | Reasoning and caveats: why this gene, what the source leaves unsaid, code-to-name links. |
| `batch` | The curation batch that added the row, matching a log `docs/curation/<batch>.md`. |

`tests/test_curated.py` enforces these rules, so a row without provenance fails the test suite. The checks themselves are in `src/causal_target_to_patient/curation.py`.

## Standing rules

1. **What counts as a mechanism.** A molecular target the drug acts on directly: an enzyme, receptor, ligand, or cell-surface protein. Pathway effects with no named target ("NF-κB regulation"), non-protein mechanisms (scavengers), and binding to a broad class of molecules are not mapped. They are recorded in `unmapped_decisions.csv`.
2. **Identifiers.** Use the ChEMBL target ID when ChEMBL has the target. Otherwise use `GENE:<symbol>`, or `GENE:<family>` for a family.
3. **Unspecified isoforms.** When a source names a family without the isoform ("JAK", "HDAC", "CaMKII"), list the family's genes, or the family symbol if the members are unclear. Say so in `notes`, and lower the confidence when only a database supports it.
4. **Multiple targets.** One row per target: dual inhibitors, and antibody-drug conjugates (antibody target plus payload target).
5. **Code-to-name links.** When a source uses a development code, confirm the code belongs to the drug through ChEMBL's synonyms, and say so in `notes`.
6. **Source preference.** In order: paper, trial registry, or company; then review, conference abstract, or news; then drug or target database; then vendor. Use the best source found, and let `confidence` reflect what it is.
7. **Non-RA trials.** An intervention whose only trial is in another disease is `EXCLUDE`d in `intervention_overrides.csv`, with the trial as the source.
8. **LLM assistance.** Research may be done with an LLM (Claude). Every row is checked against its cited source and accepted by a human curator before it is added, and each batch log records this.

## Workflow for a batch

1. Run `python scripts/curation_worklist.py --data-dir ../causal-data/aact` to list what still needs curating, most labelling trials first.
2. Research the top items. Draft rows with full provenance.
3. Review: the curator checks each row against its source.
4. Add the rows with a new `batch` value, and write `docs/curation/<batch>.md`: the worklist used, the method, every decision including corrections, and anything deferred.
5. Run `pytest -q tests`, rerun the mapping, and record the effect on the contrast classes in the log.
6. Commit the curated files and the log together, naming the batch in the commit message.
