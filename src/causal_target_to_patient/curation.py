"""Provenance checks for the curated files, and the curation worklist.

The curated files under curated/trials/ are hand-edited inputs to the mapping.
Each row records where its claim comes from (source, source_type, evidence),
how sure the curator is (confidence), and which curation batch added it
(batch, with a log in docs/curation/<batch>.md). The check_* functions return
a list of problems; tests/test_curated.py requires every list to be empty.

curation_worklist lists what still needs curating: tested interventions that
resolve to no molecule, or to a molecule ChEMBL gives no mechanism for.
"""

from pathlib import Path

import pandas as pd

SOURCE_TYPES = (
    "paper", "review", "conference_abstract", "trial_registry", "company", "news",
    "drug_database", "target_database", "vendor", "encyclopedia",
    "self-evident",  # overrides only: abbreviations, drug classes, name cleaning
)
CONFIDENCE = ("high", "medium", "low")

# Batches curated before evidence quotes were recorded. Their rows may leave
# `evidence` empty; every later batch must quote its source.
LEGACY_BATCHES = ("2026-10-13-initial",)

TEST_ROLES = ("investigational", "candidate")


def _blank(value) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or not str(value).strip()


def _is_url(value) -> bool:
    return not _blank(value) and str(value).strip().startswith(("http://", "https://"))


def _row_problems(df: pd.DataFrame, file: str, key: str, check) -> list[str]:
    problems = []
    for i, row in enumerate(df.to_dict("records"), start=2):  # line 1 is the header
        for issue in check(row):
            problems.append(f"{file} line {i} ({row.get(key)}): {issue}")
    return problems


def _missing_columns(df, file, columns) -> list[str]:
    missing = [c for c in columns if c not in df.columns]
    return [f"{file}: missing column {c}" for c in missing]


def _duplicates(df, file, columns) -> list[str]:
    keys = df[list(columns)].astype(str).apply(lambda s: s.str.strip().str.lower())
    dup = keys[keys.duplicated(keep=False)].drop_duplicates()
    return [f"{file}: duplicate {', '.join(columns)}: {tuple(r)}" for r in dup.itertuples(index=False)]


MANUAL_COLUMNS = ("name", "target_chembl_id", "target_pref_name", "gene_symbols", "source",
                  "source_type", "evidence", "confidence", "notes", "batch")


def check_manual_mechanisms(df: pd.DataFrame, file: str = "manual_mechanisms.csv") -> list[str]:
    """Each mechanism needs a URL source, a source type, a confidence, a batch,
    an evidence quote (outside legacy batches), and gene symbols or a note
    explaining their absence."""
    problems = _missing_columns(df, file, MANUAL_COLUMNS)
    if problems:
        return problems

    def check(r):
        for col in ("name", "target_chembl_id", "target_pref_name", "batch"):
            if _blank(r[col]):
                yield f"{col} is empty"
        if not _is_url(r["source"]):
            yield "source is not a URL"
        if r["source_type"] not in SOURCE_TYPES or r["source_type"] == "self-evident":
            yield f"source_type not allowed: {r['source_type']!r}"
        if r["confidence"] not in CONFIDENCE:
            yield f"confidence not allowed: {r['confidence']!r}"
        if _blank(r["evidence"]) and r["batch"] not in LEGACY_BATCHES:
            yield "evidence is empty"
        if _blank(r["gene_symbols"]) and _blank(r["notes"]):
            yield "gene_symbols is empty with no note explaining why"

    return problems + _row_problems(df, file, "name", check) + _duplicates(
        df, file, ("name", "target_chembl_id"))


OVERRIDE_COLUMNS = ("intervention_name", "maps_to", "note", "source", "source_type",
                    "confidence", "batch")


def check_overrides(df: pd.DataFrame, file: str = "intervention_overrides.csv") -> list[str]:
    """Each override needs a note and a batch; a URL source unless self-evident."""
    problems = _missing_columns(df, file, OVERRIDE_COLUMNS)
    if problems:
        return problems

    def check(r):
        for col in ("intervention_name", "maps_to", "note", "batch"):
            if _blank(r[col]):
                yield f"{col} is empty"
        if r["source_type"] not in SOURCE_TYPES:
            yield f"source_type not allowed: {r['source_type']!r}"
        elif r["source_type"] != "self-evident" and not _is_url(r["source"]):
            yield "source is not a URL (required unless self-evident)"
        if r["confidence"] not in CONFIDENCE:
            yield f"confidence not allowed: {r['confidence']!r}"

    return problems + _row_problems(df, file, "intervention_name", check) + _duplicates(
        df, file, ("intervention_name",))


EXCLUSION_COLUMNS = ("molecule_pref_name", "category", "reason", "batch")


def check_exclusions(df: pd.DataFrame, file: str = "excluded_therapies.csv") -> list[str]:
    """Exclusions are policy decisions: each needs a category, a reason, and a batch."""
    problems = _missing_columns(df, file, EXCLUSION_COLUMNS)
    if problems:
        return problems

    def check(r):
        for col in EXCLUSION_COLUMNS:
            if _blank(r[col]):
                yield f"{col} is empty"

    return problems + _row_problems(df, file, "molecule_pref_name", check) + _duplicates(
        df, file, ("molecule_pref_name",))


DECISION_COLUMNS = ("name", "reason", "source", "source_type", "evidence", "batch")


def check_unmapped_decisions(df: pd.DataFrame, file: str = "unmapped_decisions.csv") -> list[str]:
    """A decision not to map needs a reason, a URL source, evidence, and a batch."""
    problems = _missing_columns(df, file, DECISION_COLUMNS)
    if problems:
        return problems

    def check(r):
        for col in ("name", "reason", "evidence", "batch"):
            if _blank(r[col]):
                yield f"{col} is empty"
        if not _is_url(r["source"]):
            yield "source is not a URL"
        if r["source_type"] not in SOURCE_TYPES or r["source_type"] == "self-evident":
            yield f"source_type not allowed: {r['source_type']!r}"

    return problems + _row_problems(df, file, "name", check) + _duplicates(df, file, ("name",))


def check_no_conflicts(manual: pd.DataFrame, decisions: pd.DataFrame) -> list[str]:
    """A drug can't be both given a mechanism and recorded as deliberately unmapped."""
    both = set(manual["name"].str.lower()) & set(decisions["name"].str.lower())
    return [f"{n}: in both manual_mechanisms.csv and unmapped_decisions.csv" for n in sorted(both)]


def missing_batch_logs(batches, log_dir: Path) -> list[str]:
    """Batches with no log file docs/curation/<batch>.md."""
    return [f"no curation log for batch {b}: expected {Path(log_dir) / (b + '.md')}"
            for b in sorted(set(batches)) if not (Path(log_dir) / f"{b}.md").exists()]


def curation_worklist(
    interventions: pd.DataFrame,
    name_molecules: pd.DataFrame,
    molecule_targets: pd.DataFrame,
    contrasts: pd.DataFrame,
    decided: set[str] = frozenset(),
) -> pd.DataFrame:
    """Tested interventions still needing curation, one row per item.

    kind "no_mechanism": resolved to a ChEMBL molecule with no mechanism; the
    item is the molecule's preferred name. kind "no_molecule": resolved to no
    molecule; the item is the cleaned name that was looked up. Excluded
    therapies, components with a manual mechanism, and names in `decided`
    (case-insensitive) are left out. Sorted by the number of labelling trials.
    """
    iv = interventions[interventions.role.isin(TEST_ROLES)].copy()
    iv["intervention_key"] = iv.intervention_name.str.strip().str.lower()
    m = iv.merge(name_molecules, on="intervention_key").merge(
        contrasts[["nct_id", "trial_set"]], on="nct_id", how="left")
    excluded = (m.excluded.astype("boolean").fillna(False)
                | m.excluded_therapy.astype("boolean").fillna(False)).astype(bool)
    m = m[~excluded]
    has_target = m.mechanism_key.isin(set(molecule_targets.parent_chembl_id))
    has_molecule = m.molecule_chembl_id.notna()
    m = m[~has_target].copy()
    m["kind"] = ["no_mechanism" if h else "no_molecule" for h in has_molecule[~has_target]]
    m["item"] = [str(p).lower() if k == "no_mechanism" else str(l).lower()
                 for k, p, l in zip(m.kind, m.molecule_pref_name, m.lookup_name)]
    decided = {d.strip().lower() for d in decided}
    m = m[~m["item"].isin(decided) & ~m.lookup_name.str.lower().isin(decided)]

    def labelling(s):
        return s[m.loc[s.index, "trial_set"].eq("labelling")].nunique()

    out = (m.groupby(["kind", "item"], dropna=False)
           .agg(parent_chembl_id=("parent_chembl_id", "first"),
                molecule_type=("molecule_type", "first"),
                n_trials=("nct_id", "nunique"),
                n_labelling_trials=("nct_id", labelling),
                intervention_names=("intervention_name",
                                    lambda s: "; ".join(sorted(set(s)))[:200]),
                example_trials=("nct_id", lambda s: "; ".join(sorted(set(s))[:5])))
           .reset_index()
           .sort_values(["n_labelling_trials", "n_trials", "item"],
                        ascending=[False, False, True]))
    return out.reset_index(drop=True)
