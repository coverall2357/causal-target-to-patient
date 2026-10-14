"""Compare the trial pipeline's targets with reference lists, at the mechanism level.

Genes map to mechanisms through curated/trials/mechanism_groups.csv; a gene not
listed is its own mechanism, named by its symbol. A combination (a bispecific,
or a multi-target drug) counts towards every mechanism it touches.
"""

import pandas as pd

BOTH, REFERENCE_ONLY, PIPELINE_ONLY = "both", "reference_only", "pipeline_only"


def load_groups(groups: pd.DataFrame) -> dict[str, str]:
    """Gene symbol (upper case) -> mechanism name."""
    return dict(zip(groups.gene_symbol.str.strip().str.upper(), groups.mechanism.str.strip()))


def to_mechanisms(gene_symbols, groups: dict[str, str], target_name=None) -> set[str]:
    """Mechanisms for one target's ';'-separated gene list.

    Grouped genes map to their mechanism. A target whose genes are all
    ungrouped is one mechanism: named by its single gene, or, for a multi-gene
    target (a complex or family such as the proteasome), by the target name.
    With no gene symbols, the target name is used if given.
    """
    genes = ([g.strip().upper() for g in gene_symbols.split(";") if g.strip()]
             if isinstance(gene_symbols, str) else [])
    name = target_name if isinstance(target_name, str) and target_name else None
    if not genes:
        return {name} if name else set()
    if len(genes) > 1 and name and not any(g in groups for g in genes):
        return {name}
    return {groups.get(g, g) for g in genes}


def pipeline_mechanisms(
    contrasts: pd.DataFrame, targets: pd.DataFrame, groups: dict[str, str],
    trial_set: str = "labelling",
    classes: tuple[str, ...] = ("target_contrast", "head_to_head"),
) -> pd.DataFrame:
    """Mechanisms tested by pipeline trials, with the number of trials for each."""
    lookup = targets.drop_duplicates("target_chembl_id").set_index("target_chembl_id")
    keep = contrasts[(contrasts.trial_set == trial_set) & contrasts.contrast_class.isin(classes)]
    pairs = set()
    for nct_id, contrast in zip(keep.nct_id, keep.contrast_targets.fillna("")):
        for tid in filter(None, contrast.split(";")):
            if tid in lookup.index:
                row = lookup.loc[tid]
                mechs = to_mechanisms(row.get("gene_symbols"), groups, row.get("target_pref_name"))
            else:
                mechs = {tid}
            pairs.update((nct_id, m) for m in mechs)
    out = pd.DataFrame(sorted(pairs), columns=["nct_id", "mechanism"])
    return (out.groupby("mechanism").nct_id.nunique().rename("n_trials").reset_index()
            if len(out) else pd.DataFrame(columns=["mechanism", "n_trials"]))


def expected_mechanisms(expected: pd.DataFrame, groups: dict[str, str]) -> pd.DataFrame:
    """Expected list decomposed into mechanisms, keeping each status and source row."""
    rows = []
    for r in expected.itertuples(index=False):
        for m in to_mechanisms(r.gene_symbols, groups, r.mechanism):
            rows.append({"mechanism": m, "reference_detail": f"{r.mechanism}: {r.status_in_ra}"})
    out = pd.DataFrame(rows)
    return out.groupby("mechanism").reference_detail.apply("; ".join).reset_index()


def chembl_mechanisms(reference: pd.DataFrame, groups: dict[str, str]) -> pd.DataFrame:
    """ChEMBL RA reference decomposed into mechanisms, with molecules and max phase."""
    rows = []
    for r in reference.itertuples(index=False):
        for m in to_mechanisms(r.gene_symbols, groups, r.target_pref_name):
            rows.append({"mechanism": m, "molecule": r.molecule_pref_name,
                         "phase": r.max_phase_for_ra})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["mechanism", "reference_detail"])
    out = out.sort_values("phase", ascending=False)
    return (out.groupby("mechanism")
            .apply(lambda g: "; ".join(f"{m} (phase {p:g})" for m, p in
                                       zip(g.molecule, g.phase)), include_groups=False)
            .rename("reference_detail").reset_index())


def compare(pipeline: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Outer join on mechanism, with a coverage category for each."""
    out = pipeline.merge(reference, on="mechanism", how="outer", indicator=True)
    out["category"] = out["_merge"].map(
        {"both": BOTH, "left_only": PIPELINE_ONLY, "right_only": REFERENCE_ONLY}).astype(str)
    out["n_trials"] = out.n_trials.fillna(0).astype(int)
    return (out.drop(columns="_merge")
            .sort_values(["category", "n_trials"], ascending=[True, False])
            .reset_index(drop=True))


# --- Gap diagnosis ----------------------------------------------------------

import re  # noqa: E402

COUNTED_CLASSES = ("target_contrast", "head_to_head")


def _normalise(text: str) -> str:
    """Lower case, letters and digits only: "MK-0812" and "mk0812" compare equal."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def search_terms(example_therapeutics: str) -> list[str]:
    """Drug names and codes from an expected-list cell, for text search.

    "rozibafusp alfa (AMG 570); CPL409116" -> ["rozibafusp alfa", "amg 570", "cpl409116"].
    Placeholders such as "(unnamed in source)" and terms under 4 characters
    (once normalised) are dropped to avoid spurious matches.
    """
    terms = []
    for part in str(example_therapeutics).split(";"):
        inside = re.findall(r"\(([^)]*)\)", part)
        outside = re.sub(r"\([^)]*\)", "", part)
        for t in [outside, *inside]:
            t = t.strip().lower()
            if t and "unnamed" not in t and "named in source" not in t and len(_normalise(t)) >= 4:
                terms.append(t)
    return list(dict.fromkeys(terms))


def locate_in_pipeline(
    terms: list[str],
    parent_ids: set[str],
    name_molecules: pd.DataFrame,
    interventions: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> pd.DataFrame:
    """Pipeline trials involving a drug, found by ChEMBL molecule or by name text.

    Returns one row per trial and intervention: nct_id, intervention_key, role,
    trial_set, contrast_class, and how it was found ("molecule" or "name").
    """
    by_molecule = set()
    if parent_ids:
        hit = (name_molecules.get("parent_chembl_id", pd.Series(dtype=object)).isin(parent_ids)
               | name_molecules.get("mechanism_key", pd.Series(dtype=object)).isin(parent_ids))
        by_molecule = set(name_molecules.loc[hit, "intervention_key"])
    norm_terms = [_normalise(t) for t in terms if len(_normalise(t)) >= 4]
    keys = interventions.intervention_key.astype(str)
    by_name = set(keys[keys.map(lambda k: any(t in _normalise(k) for t in norm_terms))]) \
        if norm_terms else set()
    found = interventions[keys.isin(by_molecule | by_name)][["nct_id", "intervention_key", "role"]]
    mapped = set(name_molecules.loc[name_molecules.get(
        "mechanism_key", pd.Series(index=name_molecules.index, dtype=object)).notna()
        | name_molecules.get("molecule_chembl_id", pd.Series(
            index=name_molecules.index, dtype=object)).notna(), "intervention_key"])
    found = found.assign(found_by=found.intervention_key.map(
        lambda k: "molecule" if k in by_molecule
        else "name (mapped)" if k in mapped else "name"))
    return found.merge(contrasts[["nct_id", "trial_set", "contrast_class"]],
                       on="nct_id", how="left").drop_duplicates()


def diagnose(found: pd.DataFrame) -> str:
    """One-line explanation of why a drug's trials did not count as a test."""
    if found.empty:
        return "not in AACT pull"
    if (found.found_by == "name").all():
        return "in pull, name not mapped to a molecule"
    counted = found[(found.trial_set == "labelling") & found.contrast_class.isin(COUNTED_CLASSES)]
    if len(counted):
        return "counted, but under a different mechanism"
    parts = (found.drop_duplicates("nct_id")
             .assign(location=lambda d: d.trial_set.fillna("?") + "/" + d.contrast_class.fillna("?"))
             .location.value_counts())
    return "in pull, not counted: " + ", ".join(f"{k} x{v}" for k, v in parts.items())
