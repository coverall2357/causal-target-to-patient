"""Map trial intervention names to ChEMBL molecules, mechanisms, and targets.

Uses the ChEMBL REST API (https://www.ebi.ac.uk/chembl/api/data). Responses are
cached in a JSON file, so reruns and interrupted runs don't repeat requests.
"""

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"

# Molecule lookup fields. Synonyms usually include the preferred name, so the
# preferred-name field is only a last resort.
SYNONYM_FIELD = "molecule_synonyms__molecule_synonym__iexact"
PREF_NAME_FIELD = "pref_name__iexact"

# Dose and everything after it: "tofacitinib 5 mg bid" -> "tofacitinib".
_DOSE = re.compile(
    r"\s\d+(\.\d+)?\s*(mg/kg|mg|mcg|µg|ug|g|ml|iu|units?)\b.*$", re.IGNORECASE
)
# A dose at the start of a name: "50 mg etanercept" -> "etanercept".
_LEADING_DOSE = re.compile(
    r"^\d+(\.\d+)?\s*(mg/kg|mg|mcg|µg|ug|g|ml|iu|units?)\s+", re.IGNORECASE
)
# Dose levels named in words, and everything after: "yra-1909 low dose",
# "gsk3196165 dose 2" -> the drug name.
_DOSE_WORDS = re.compile(
    r"\s((low|mid|middle|medium|high)[\s-]dose|dose\s*(level\s*)?\d+)\b.*$",
    re.IGNORECASE,
)
# Trailing formulation, device, and route words, and everything after them.
_FORMULATION = re.compile(
    r"\s(tablets?|capsules?|injection|infusion|oral|subcutaneous|intravenous|"
    r"sc|iv|s\.c\.|i\.v\.|pre-?filled|pfs|pens?|auto-?injectors?|syringes?|via)\b.*$",
    re.IGNORECASE,
)
# Arm words, only at the very end: "etanercept treatment" -> "etanercept", but
# "combined therapy of iguratimod" keeps its drug name.
_TRAILING_WORDS = re.compile(r"(\s(treatment|therapy|administration|group))+$",
                             re.IGNORECASE)
# A component that is only a device or arm word, left over from splitting
# "adalimumab pfs and pen".
_DEVICE_ONLY = re.compile(
    r"(pre-?filled\s*)?(syringes?|pens?|pfs|auto-?injectors?(\s+[a-z])?)"
    r"|treatment|therapy|(low|mid|middle|medium|high)[\s-]dose",
    re.IGNORECASE,
)
_BRACKETED = re.compile(r"[\[(]([^\])]*)[\])]")
# Combinations ("+", "plus", "and"), and comma lists with a space ("mk0663,
# etoricoxib"); a comma without a space is part of a code ("cp-690,550").
_COMBINATION = re.compile(r"\s*\+\s*|\s+plus\s+|\s+and\s+|,\s+", re.IGNORECASE)
_SYMBOLS = re.compile(r"[®™©\"]")
# Leading arm labels, regulatory qualifiers, and routes.
_PREFIX = re.compile(
    r"^((active\s+)?comparator|experimental|control)\s*:\s*"
    r"|^arm\s+\w+\s*:\s*"
    r"|^(treatment|combination|combined\s+(therapy|treatment))\s+(of|with)\s+"
    r"|^(eu|us)[-\s](approved|licensed|sourced)\s+"
    r"|^(subcutaneous|intravenous|oral|sc|iv)\s+",
    re.IGNORECASE,
)


# A component that is only a dose, left over from a comma list.
_DOSE_ONLY = re.compile(r"\d+(\.\d+)?\s*(mg/kg|mg|mcg|µg|ug|g|ml|iu|units?)", re.IGNORECASE)


def _clean(text: str) -> str:
    text = _SYMBOLS.sub("", text)
    previous = None
    while previous != text:  # prefixes can stack: "comparator: iv abatacept"
        previous = text
        text = _PREFIX.sub("", text.strip())
    text = _LEADING_DOSE.sub("", text)
    text = _DOSE.sub("", text)
    text = _DOSE_WORDS.sub("", text)
    text = _FORMULATION.sub("", text)
    text = _TRAILING_WORDS.sub("", text)
    return re.sub(r"\s+", " ", text).strip(" -,;/")


def _usable_alternative(text: str) -> bool:
    """Bracketed text worth searching: at least 3 characters, with a letter.

    Excludes route abbreviations ("sc", "iv") and labels like "(4)".
    """
    return len(text) >= 3 and bool(re.search(r"[a-z]", text))


def _usable_component(text: str) -> bool:
    """A searchable component: at least 3 letters or digits.

    Drops fragments left by splitting ("a", "db") but keeps numeric
    development codes ("552-02").
    """
    return len(re.sub(r"\W", "", text)) >= 3


def _spellings(text: str) -> list[str]:
    """Spellings of one name: as is, without commas, and without punctuation.

    Development codes vary: "cp-690,550" is "CP-690550" in ChEMBL.
    """
    forms = [text, text.replace(",", ""), re.sub(r"[\s,\-]", "", text)]
    return list(dict.fromkeys(f for f in forms if f))


def name_variants(raw: str) -> list[list[str]]:
    """Return search variants for each component of an intervention name.

    A combination ("tocilizumab + methotrexate") gives one list per component.
    Bracketed or parenthesised text (brand names, abbreviations) is tried after
    the main name: "tocilizumab [roactemra/actemra]" -> tocilizumab, roactemra,
    actemra. A slash outside brackets also separates alternative names
    ("mabthera/rituxan"); each is tried, so a slash never adds a component.
    """
    text = _SYMBOLS.sub("", raw.strip().lower())
    alternatives = [
        _clean(part)
        for inner in _BRACKETED.findall(text)
        for part in re.split(r"[/;]", inner)
    ]
    alternatives = [a for a in alternatives if _usable_alternative(a)]
    base = _BRACKETED.sub(" ", text)
    components = []
    for part in _COMBINATION.split(base):
        part = _clean(part)
        if (not _usable_component(part) or _DOSE_ONLY.fullmatch(part)
                or _DEVICE_ONLY.fullmatch(part)):
            continue  # nothing searchable: a dose, a device word, or a fragment
        first, *others = [_clean(n) for n in re.split(r"\s*/\s*", part)]
        others = [n for n in others
                  if _usable_alternative(n) and not _DEVICE_ONLY.fullmatch(n)]
        keep_first = _usable_component(first) and not _DEVICE_ONLY.fullmatch(first)
        names = ([first] if keep_first else []) + others
        names = names or [part]
        variants = [v for n in names for v in _spellings(n)]
        for alt in alternatives:
            variants += _spellings(alt)
        components.append(list(dict.fromkeys(variants)))
    return components


class ChemblClient:
    """Minimal ChEMBL REST client with a JSON response cache."""

    def __init__(self, cache_path: Path, pause: float = 0.1):
        self.cache_path = Path(cache_path)
        self.pause = pause
        self.cache = (
            json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        )
        self._new = 0

    def _get(
        self, endpoint: str, params: dict | None = None, cached_only: bool = False
    ) -> dict | None:
        """Return a response, from the cache if present.

        With cached_only, return None instead of making a request.
        """
        params = params or {}
        key = endpoint + "?" + json.dumps(params, sort_keys=True)
        if key not in self.cache:
            if cached_only:
                return None
            response = requests.get(
                f"{CHEMBL_API}/{endpoint}.json", params=params, timeout=60
            )
            response.raise_for_status()
            self.cache[key] = response.json()
            self._new += 1
            if self._new % 50 == 0:
                self.save()
            time.sleep(self.pause)
        return self.cache[key]

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache))

    def find_molecule(
        self, name: str, field: str = SYNONYM_FIELD, cached_only: bool = False
    ) -> dict | None:
        """Exact, case-insensitive match of `name` on one ChEMBL molecule field."""
        data = self._get("molecule", {field: name, "limit": 20}, cached_only)
        hits = data["molecules"] if data else []
        if not hits:
            return None
        # Prefer the parent form over salts and other child forms.
        parents = [
            m for m in hits
            if (m.get("molecule_hierarchy") or {}).get("parent_chembl_id")
            in (None, m["molecule_chembl_id"])
        ]
        m = (parents or hits)[0]
        parent = (m.get("molecule_hierarchy") or {}).get("parent_chembl_id")
        return {
            "molecule_chembl_id": m["molecule_chembl_id"],
            "parent_chembl_id": parent or m["molecule_chembl_id"],
            "molecule_pref_name": m.get("pref_name"),
            "molecule_type": m.get("molecule_type"),
            "max_phase": m.get("max_phase"),
        }

    def mechanisms(self, parent_chembl_id: str) -> list[dict]:
        """Mechanisms of action recorded for a parent molecule or any of its forms.

        ChEMBL often records the mechanism on the marketed salt form, so the
        query is by parent, which groups all forms of a molecule.
        """
        data = self._get(
            "mechanism", {"parent_molecule_chembl_id": parent_chembl_id, "limit": 100}
        )
        rows = [
            {
                "parent_chembl_id": parent_chembl_id,
                "mechanism_of_action": m.get("mechanism_of_action"),
                "action_type": m.get("action_type"),
                "target_chembl_id": m.get("target_chembl_id"),
            }
            for m in data["mechanisms"]
        ]
        # Several forms can carry the same mechanism; keep one of each.
        return [dict(t) for t in dict.fromkeys(tuple(r.items()) for r in rows)]

    def drug_indications(self, **filters) -> list[dict]:
        """All drug-indication records matching filters, following pagination."""
        records, offset = [], 0
        while True:
            data = self._get("drug_indication", {**filters, "limit": 1000, "offset": offset})
            batch = data.get("drug_indications", [])
            records += batch
            meta = data.get("page_meta") or {}
            if not batch or not meta.get("next"):
                return records
            offset += len(batch)

    def molecule(self, chembl_id: str) -> dict:
        """One molecule record: preferred name and parent."""
        m = self._get(f"molecule/{chembl_id}")
        parent = (m.get("molecule_hierarchy") or {}).get("parent_chembl_id")
        return {"molecule_chembl_id": chembl_id,
                "parent_chembl_id": parent or chembl_id,
                "molecule_pref_name": m.get("pref_name")}

    def target(self, target_chembl_id: str) -> dict:
        """Target name, type, and gene symbols of its components."""
        t = self._get(f"target/{target_chembl_id}")
        symbols = sorted({
            syn["component_synonym"]
            for comp in t.get("target_components", [])
            for syn in comp.get("target_component_synonyms", [])
            if syn.get("syn_type") == "GENE_SYMBOL"
        })
        return {
            "target_chembl_id": target_chembl_id,
            "target_pref_name": t.get("pref_name"),
            "target_type": t.get("target_type"),
            "gene_symbols": ";".join(symbols),
        }


def lookup_component(variants: list[str], client) -> tuple[dict | None, str | None]:
    """Find the molecule for one name component, trying its variants in order.

    1. Any hit already in the cache, from either field (no requests).
    2. Synonym match for each variant.
    3. Preferred-name match for the main variant only.

    An unmatched component costs one request per variant plus one, rather
    than two per variant.
    """
    for v in variants:
        for field in (SYNONYM_FIELD, PREF_NAME_FIELD):
            hit = client.find_molecule(v, field, cached_only=True)
            if hit:
                return hit, v
    for v in variants:
        hit = client.find_molecule(v, SYNONYM_FIELD)
        if hit:
            return hit, v
    hit = client.find_molecule(variants[0], PREF_NAME_FIELD)
    return (hit, variants[0]) if hit else (None, None)


def _apply_override(key: str, maps_to) -> list[tuple]:
    """Expand one override into (kind, value, via) entries.

    kind is "exclude", "id" (a ChEMBL molecule ID), or "lookup" (search variants).
    """
    value = str(maps_to).strip()
    if value.upper() == "EXCLUDE":
        return [("exclude", None, f"override:{key}")]
    if re.fullmatch(r"CHEMBL\d+", value.upper()):
        return [("id", value.upper(), "override:id")]
    return [("lookup", variants, "alias:") for variants in name_variants(value)]


def resolve_components(raw: str, overrides: dict[str, str]) -> list[tuple]:
    """Resolve an intervention name into components, applying overrides.

    Overrides match the whole lower-case name, or any single component after
    cleaning: "rc18 160 mg plus mtx" uses the overrides for "rc18" and "mtx".
    """
    key = raw.strip().lower()
    if key in overrides:
        return _apply_override(key, overrides[key])
    out = []
    for variants in name_variants(raw):
        component_key = variants[0]
        if component_key in overrides:
            out += _apply_override(component_key, overrides[component_key])
        else:
            out.append(("lookup", variants, ""))
    return out


def map_names(
    names: list[str],
    client: ChemblClient,
    overrides: dict[str, str] | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Map intervention names to molecules, one row per name component.

    `overrides` maps a lower-case name (whole, or one cleaned component) to:
        a ChEMBL molecule ID   used directly ("CHEMBL1201580")
        EXCLUDE                not a therapeutic (imaging agent, vaccine, ...)
        any other text         a name to look up instead ("mtx" -> "methotrexate")
    Each row records `lookup_name`, the cleaned name searched for, which manual
    mechanisms can key on. Unmatched components get an empty molecule ID.
    """
    overrides = overrides or {}
    rows = []
    for raw in tqdm(names, desc="Matching names", disable=not progress):
        key = raw.strip().lower()
        for i, (kind, value, via) in enumerate(resolve_components(raw, overrides)):
            row = {"intervention_key": key, "component": i, "excluded": False}
            if kind == "exclude":
                row.update(matched_on=via, excluded=True)
            elif kind == "id":
                row.update(matched_on=via, molecule_chembl_id=value, parent_chembl_id=value)
            else:
                hit, matched = lookup_component(value, client)
                row.update(lookup_name=value[0],
                           matched_on=f"{via}{matched}" if matched else None,
                           **(hit or {}))
            rows.append(row)
    return pd.DataFrame(rows)


def apply_manual_mechanisms(
    molecules: pd.DataFrame, targets: pd.DataFrame, manual: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add curated mechanisms where ChEMBL has none.

    `manual` columns: name, target_chembl_id, target_pref_name, gene_symbols,
    source. `name` matches a component's lookup_name or the molecule's ChEMBL
    preferred name, case-insensitively. Used only when ChEMBL records no
    mechanism for the component.

    Returns molecules with a `mechanism_key` column (the parent ChEMBL ID, or
    "MANUAL:<name>" for components ChEMBL lacks) and targets extended with the
    manual rows under the same key, each marked with mechanism_source.
    """
    molecules = molecules.copy()
    targets = targets.copy()
    if "mechanism_source" not in targets:
        targets["mechanism_source"] = "chembl"
    for col in ("lookup_name", "molecule_pref_name", "parent_chembl_id"):
        if col not in molecules:
            molecules[col] = None
    manual = manual.copy()
    manual["name"] = manual["name"].str.strip().str.lower()
    manual_names = set(manual["name"])
    has_mechanism = set(targets.parent_chembl_id) if len(targets) else set()

    keys, added = [], []
    for row in molecules.itertuples(index=False):
        parent = row.parent_chembl_id if isinstance(row.parent_chembl_id, str) else None
        candidates = [str(n).strip().lower() for n in (row.lookup_name, row.molecule_pref_name)
                      if isinstance(n, str)]
        manual_name = next((n for n in candidates if n in manual_names), None)
        if parent in has_mechanism or manual_name is None:
            keys.append(parent)
            continue
        key = parent or f"MANUAL:{manual_name}"
        keys.append(key)
        for m in manual[manual["name"] == manual_name].itertuples(index=False):
            added.append({"parent_chembl_id": key, "mechanism_of_action": None,
                          "action_type": None, "target_chembl_id": m.target_chembl_id,
                          "target_pref_name": m.target_pref_name, "target_type": None,
                          "gene_symbols": m.gene_symbols,
                          "mechanism_source": f"manual: {m.source}"})
    molecules["mechanism_key"] = keys
    if added:
        targets = pd.concat([targets, pd.DataFrame(added)], ignore_index=True)
        targets = targets.drop_duplicates(["parent_chembl_id", "target_chembl_id"])
    return molecules, targets


def map_targets(
    parent_ids: list[str], client: ChemblClient, progress: bool = True
) -> pd.DataFrame:
    """Mechanisms and targets for each parent molecule, one row per mechanism."""
    rows = []
    for pid in tqdm(parent_ids, desc="Fetching targets", disable=not progress):
        for mech in client.mechanisms(pid):
            target = client.target(mech["target_chembl_id"]) if mech["target_chembl_id"] else {}
            rows.append({**mech, **target})
    return pd.DataFrame(rows)


def is_excluded_name(pref_name, excluded_names: set[str]) -> bool:
    """True if a ChEMBL preferred name is an excluded therapy or one of its salts.

    "DICLOFENAC SODIUM" matches "DICLOFENAC".
    """
    if not isinstance(pref_name, str):
        return False
    name = pref_name.upper()
    return any(name == e or name.startswith(e + " ") for e in excluded_names)


def flag_excluded_therapies(
    molecules: pd.DataFrame, excluded_names: set[str]
) -> pd.Series:
    """Boolean Series: name excluded by override, or molecule on the excluded list."""
    by_override = (
        molecules["excluded"].astype("boolean").fillna(False).astype(bool)
        if "excluded" in molecules else pd.Series(False, index=molecules.index)
    )
    pref_names = (
        molecules["molecule_pref_name"]
        if "molecule_pref_name" in molecules else pd.Series(None, index=molecules.index)
    )
    by_list = pref_names.map(lambda n: is_excluded_name(n, excluded_names)).astype(bool)
    return by_override | by_list


# Rheumatoid arthritis in ChEMBL's indication records.
RA_INDICATION_FILTERS = ({"mesh_id": "D001172"}, {"efo_id": "EFO_0000685"})


def ra_indication_molecules(client, min_phase: float = 2) -> pd.DataFrame:
    """Parent molecules with an RA indication at or above min_phase, one row each.

    Queries by MeSH and EFO identifiers, keeps the highest phase per parent.
    """
    rows = []
    for filters in RA_INDICATION_FILTERS:
        for rec in client.drug_indications(**filters):
            phase = pd.to_numeric(rec.get("max_phase_for_ind"), errors="coerce")
            parent = rec.get("parent_molecule_chembl_id") or rec.get("molecule_chembl_id")
            if parent and pd.notna(phase) and phase >= min_phase:
                rows.append({"parent_chembl_id": parent, "max_phase_for_ra": float(phase)})
    if not rows:
        return pd.DataFrame(columns=["parent_chembl_id", "max_phase_for_ra"])
    return (pd.DataFrame(rows).groupby("parent_chembl_id").max_phase_for_ra.max()
            .reset_index())


TEST_ROLES = ("investigational", "candidate")
CONTROL_ROLES = ("comparator", "background")


def classify_target_contrast(
    rows: pd.DataFrame, controlled_trials: set[str] | None = None
) -> pd.DataFrame:
    """Classify what each trial tests at the target level.

    `rows`: one row per intervention-target pair, with columns nct_id,
    intervention_key, role, target_chembl_id (missing if unmapped), and
    optionally excluded_therapy (True for conventional or symptomatic
    therapies, which are never counted as tested).

    Classes:
        target_contrast     tested targets differ from control targets
        head_to_head        fallback trial comparing distinct targets
        same_target         every arm acts on the same targets (biosimilar,
                            dose, formulation, tapering, strategy)
        single_intervention fallback trial, one tested therapeutic, no placebo,
                            and no evidence of a control arm
        uncontrolled        arm-based trial with no control: no placebo, no
                            comparator or background intervention, and no
                            comparator arm (single-arm, extension, and
                            uncontrolled dose studies)
        excluded_therapy    every tested therapeutic is on the excluded list
        unmapped            no tested intervention mapped to a target
        no_test             no tested intervention at all

    `controlled_trials`: trials whose arm records show a control (two or more
    arms, at least one a comparator arm). A trial with one tested therapeutic
    counts as a target contrast if it is in this set: the control arm existed,
    its intervention just wasn't listed.

    A trial without any control is never a target contrast. If its tested
    therapeutics act on different targets it is a head-to-head comparison;
    otherwise it is uncontrolled (or single_intervention in fallback trials).
    """
    controlled_trials = controlled_trials or set()
    rows = rows.copy()
    if "excluded_therapy" not in rows:
        rows["excluded_therapy"] = False
    rows["excluded_therapy"] = rows.excluded_therapy.fillna(False).astype(bool)
    out = []
    for nct_id, trial in rows.groupby("nct_id"):
        all_test = trial[trial.role.isin(TEST_ROLES)]
        test = all_test[~all_test.excluded_therapy]
        control = trial[trial.role.isin(CONTROL_ROLES)]
        test_targets = set(test.target_chembl_id.dropna())
        control_targets = set(control.target_chembl_id.dropna())
        mapped_test = test.dropna(subset=["target_chembl_id"])
        unmapped_test = bool(
            set(test.intervention_key) - set(mapped_test.intervention_key)
        )
        is_fallback = (trial.role == "candidate").any()
        has_placebo = (trial.role == "placebo").any()
        per_drug = {frozenset(g.target_chembl_id) for _, g in
                    mapped_test.groupby("intervention_key")}
        # In fallback trials, a conventional therapy alongside the tested one
        # acts as the control ("new drug vs methotrexate").
        has_control = has_placebo or all_test.excluded_therapy.any()
        # In arm-based trials, a control is a placebo, a comparator or
        # background intervention, or a comparator arm in the arm records.
        has_arm_control = has_placebo or not control.empty or nct_id in controlled_trials

        if all_test.empty:
            cls, contrast = "no_test", set()
        elif test.empty:
            cls, contrast = "excluded_therapy", set()
        elif not test_targets:
            cls, contrast = "unmapped", set()
        elif is_fallback:
            n_therapeutics = mapped_test.intervention_key.nunique()
            if len(per_drug) > 1:
                cls, contrast = "head_to_head", test_targets
            elif has_control:
                cls, contrast = "target_contrast", test_targets
            elif n_therapeutics > 1:
                cls, contrast = "same_target", set()
            elif nct_id in controlled_trials:
                cls, contrast = "target_contrast", test_targets
            else:
                cls, contrast = "single_intervention", test_targets
        elif has_arm_control:
            contrast = test_targets - control_targets
            cls = "target_contrast" if contrast else "same_target"
        elif len(per_drug) > 1:
            cls, contrast = "head_to_head", test_targets
        else:
            cls, contrast = "uncontrolled", test_targets

        out.append({
            "nct_id": nct_id,
            "contrast_class": cls,
            "contrast_targets": ";".join(sorted(contrast)),
            "has_unmapped_test": unmapped_test,
        })
    return pd.DataFrame(out)
