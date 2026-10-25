"""Build the published trial-to-target map from the pipeline's dated outputs.

The map has two tables:
- trials: one row per RA trial, with its trial set, what it compares at the
  target level (contrast class and targets) and how much outcome evidence it
  has (evidence tier);
- trial_targets: one row per trial, intervention, name component and target,
  showing how each intervention name was mapped and where each mechanism came
  from (ChEMBL or a curated source).

summary() gives the headline counts, which the data README and the MVP diagram
use, so the numbers quoted anywhere come from the same tables.
"""

import pandas as pd

TEST_CLASSES = ("target_contrast", "head_to_head")

TRIAL_COLUMNS = (
    "nct_id", "brief_title", "phase", "overall_status", "why_stopped",
    "start_date", "completion_date", "enrollment", "lead_sponsor", "sponsor_class",
    "conditions", "trial_set", "exclusion_reason", "contrast_class",
    "contrast_targets", "contrast_genes", "has_unmapped_test", "evidence_tier",
)

TARGET_COLUMNS = (
    "nct_id", "intervention_name", "intervention_type", "role", "component",
    "mapping_status", "mapping_basis", "molecule_chembl_id", "molecule_pref_name",
    "molecule_type", "max_phase", "mechanism_of_action", "action_type",
    "target_chembl_id", "target_pref_name", "target_type", "gene_symbols",
    "mechanism_source", "mechanism_reference",
)


def _target_genes(targets: pd.DataFrame) -> dict[str, str]:
    """Target ID -> gene symbols, from the molecule-target table."""
    known = targets.dropna(subset=["target_chembl_id"]).drop_duplicates("target_chembl_id")
    return dict(zip(known.target_chembl_id, known.gene_symbols.fillna("")))


def contrast_genes(contrast_targets, genes: dict[str, str]) -> str:
    """Gene symbols for a ';'-separated list of contrasted targets.

    Targets curated without a ChEMBL ID are written GENE:<symbol>. Genes are
    de-duplicated and sorted, so the same set always reads the same way.
    """
    if not isinstance(contrast_targets, str) or not contrast_targets:
        return ""
    found = []
    for target in contrast_targets.split(";"):
        symbols = target[5:] if target.startswith("GENE:") else genes.get(target, "")
        found += [s.strip() for s in str(symbols).replace(",", ";").split(";") if s.strip()]
    return ";".join(sorted(set(found)))


def trial_table(trials: pd.DataFrame, contrasts: pd.DataFrame, evidence: pd.DataFrame,
                targets: pd.DataFrame) -> pd.DataFrame:
    """One row per trial: registry fields, contrast class and targets, evidence tier."""
    table = (
        trials
        .merge(contrasts[["nct_id", "contrast_class", "contrast_targets", "has_unmapped_test"]],
               on="nct_id", how="left", validate="one_to_one")
        .merge(evidence[["nct_id", "evidence_tier"]], on="nct_id", how="left",
               validate="one_to_one")
    )
    genes = _target_genes(targets)
    table["contrast_genes"] = table.contrast_targets.map(lambda t: contrast_genes(t, genes))
    return table[list(TRIAL_COLUMNS)].sort_values("nct_id").reset_index(drop=True)


def _mapping_basis(matched_on) -> str:
    """How a name was matched: ChEMBL name, ChEMBL alias, or a curated override."""
    if not isinstance(matched_on, str) or not matched_on:
        return ""
    kind = matched_on.split(":", 1)[0]
    return {"alias": "chembl_synonym", "override": "curated_override"}.get(kind, "chembl_name")


def _mapping_status(row) -> str:
    if row["role"] == "placebo":
        return "placebo"
    if row["excluded"] is True:
        return "excluded_name"
    if row["excluded_therapy"] is True:
        return "excluded_therapy"
    if isinstance(row["target_chembl_id"], str):
        return "mapped"
    if isinstance(row["mechanism_key"], str):
        return "no_mechanism"
    return "unmapped"


def _as_bool(series: pd.Series) -> pd.Series:
    return series.map(lambda v: v is True or str(v).strip().lower() == "true")


def trial_target_table(interventions: pd.DataFrame, molecules: pd.DataFrame,
                       targets: pd.DataFrame) -> pd.DataFrame:
    """One row per trial, intervention, name component and target.

    Placebos are kept, unmapped. Names that resolve to several molecules (for
    example combinations) have one row per component; molecules with several
    targets have one row per target.
    """
    iv = interventions.copy()
    iv["intervention_key"] = iv.intervention_name.str.strip().str.lower()
    mol = molecules.copy()
    for col in ("excluded", "excluded_therapy"):
        mol[col] = _as_bool(mol[col])
    tgt = targets.rename(columns={"parent_chembl_id": "mechanism_key"})
    mol_cols = ["intervention_key", "component", "excluded", "excluded_therapy",
                "matched_on", "parent_chembl_id", "molecule_pref_name", "molecule_type",
                "max_phase", "mechanism_key"]
    table = (
        iv[["nct_id", "intervention_name", "intervention_type", "role", "intervention_key"]]
        .merge(mol[mol_cols], on="intervention_key", how="left")
        .merge(tgt[["mechanism_key", "mechanism_of_action", "action_type", "target_chembl_id",
                    "target_pref_name", "target_type", "gene_symbols", "mechanism_source"]],
               on="mechanism_key", how="left")
    )
    placebo = table.role.eq("placebo")
    # Placebo names are never mapped; drop any match their names picked up elsewhere.
    for col in mol_cols[1:] + ["mechanism_of_action", "action_type", "target_chembl_id",
                                "target_pref_name", "target_type", "gene_symbols",
                                "mechanism_source"]:
        table.loc[placebo, col] = None
    table = table.drop_duplicates(["nct_id", "intervention_key", "component", "target_chembl_id"])
    table["excluded"] = table.excluded.map(lambda v: v is True)
    table["excluded_therapy"] = table.excluded_therapy.map(lambda v: v is True)
    table["mapping_status"] = table.apply(_mapping_status, axis=1)
    table["mapping_basis"] = table.matched_on.map(_mapping_basis)
    source = table.mechanism_source.fillna("")
    table["mechanism_reference"] = source.str.extract(r"^manual:\s*(.*)$")[0].fillna("")
    table["mechanism_source"] = source.map(
        lambda s: "curated" if s.startswith("manual") else s)
    table = table.rename(columns={"parent_chembl_id": "molecule_chembl_id"})
    table["component"] = table.component.astype("Int64")
    return (table[list(TARGET_COLUMNS)]
            .sort_values(["nct_id", "intervention_name", "component", "target_chembl_id"],
                         na_position="last")
            .reset_index(drop=True))


def summary(trials: pd.DataFrame, trial_targets: pd.DataFrame) -> dict:
    """Headline counts of the map, as plain ints, for the README and diagram."""
    labelling = trials[trials.trial_set == "labelling"]
    tests = labelling[labelling.contrast_class.isin(TEST_CLASSES)]
    tested_targets = tests.contrast_targets.dropna().str.split(";").explode()
    tested_genes = tests.contrast_genes.replace("", pd.NA).dropna().str.split(";").explode()
    mapped = trial_targets[trial_targets.mapping_status == "mapped"]
    return {
        "trials": int(len(trials)),
        "by_trial_set": {k: int(v) for k, v in trials.trial_set.value_counts().items()},
        "labelling_by_contrast_class": {
            k: int(v) for k, v in labelling.contrast_class.value_counts().items()},
        "labelling_tests": int(len(tests)),
        "labelling_tests_by_class": {
            k: int(v) for k, v in tests.contrast_class.value_counts().items()},
        "labelling_tests_by_evidence_tier": {
            k: int(v) for k, v in tests.evidence_tier.value_counts().items()},
        "targets_tested": int(tested_targets.nunique()),
        "genes_tested": int(tested_genes.nunique()),
        "chembl_molecules_mapped": int(mapped.molecule_chembl_id.nunique()),
        "curated_mechanism_rows": int((mapped.mechanism_source == "curated").sum()),
    }
