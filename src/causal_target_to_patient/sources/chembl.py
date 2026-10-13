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
# Trailing formulation and route words.
_FORMULATION = re.compile(
    r"\s(tablets?|capsules?|injection|infusion|oral|subcutaneous|intravenous|"
    r"sc|iv|s\.c\.|i\.v\.)\b.*$",
    re.IGNORECASE,
)
_BRACKETED = re.compile(r"[\[(]([^\])]*)[\])]")
# Combinations, and comma lists with a space ("mk0663, etoricoxib"); a comma
# without a space is part of a code ("cp-690,550") and is left alone.
_COMBINATION = re.compile(r"\s*\+\s*|\s+plus\s+|,\s+", re.IGNORECASE)
_SYMBOLS = re.compile(r"[®™©\"]")
# Leading arm labels, regulatory qualifiers, and routes.
_PREFIX = re.compile(
    r"^((active\s+)?comparator|experimental|control)\s*:\s*"
    r"|^(eu|us)[-\s](approved|licensed|sourced)\s+"
    r"|^(subcutaneous|intravenous|oral|sc|iv)\s+",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    text = _SYMBOLS.sub("", text)
    previous = None
    while previous != text:  # prefixes can stack: "comparator: iv abatacept"
        previous = text
        text = _PREFIX.sub("", text.strip())
    text = _DOSE.sub("", text)
    text = _FORMULATION.sub("", text)
    return re.sub(r"\s+", " ", text).strip(" -,;/")


def _usable_alternative(text: str) -> bool:
    """Bracketed text worth searching: at least 3 characters, with a letter.

    Excludes route abbreviations ("sc", "iv") and labels like "(4)".
    """
    return len(text) >= 3 and bool(re.search(r"[a-z]", text))


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
    actemra.
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
        if not part:
            continue
        variants = _spellings(part)
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


def map_names(
    names: list[str],
    client: ChemblClient,
    overrides: dict[str, str] | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Map intervention names to molecules, one row per name component.

    `overrides` maps a lower-case raw name to one of:
        a ChEMBL molecule ID   used directly ("CHEMBL1201580")
        EXCLUDE                not a therapeutic (imaging agent, vaccine, ...)
        any other text         a name to look up instead ("mtx" -> "methotrexate")
    Unmatched components get a row with an empty molecule ID.
    """
    overrides = overrides or {}
    rows = []
    for raw in tqdm(names, desc="Matching names", disable=not progress):
        key = raw.strip().lower()
        target = str(overrides.get(key, raw)).strip()
        if target.upper() == "EXCLUDE":
            rows.append({"intervention_key": key, "component": 0,
                         "matched_on": "override:EXCLUDE", "excluded": True})
            continue
        if re.fullmatch(r"CHEMBL\d+", target.upper()):
            rows.append({"intervention_key": key, "component": 0,
                         "matched_on": "override:id", "excluded": False,
                         "molecule_chembl_id": target.upper(),
                         "parent_chembl_id": target.upper()})
            continue
        via = "alias:" if key in overrides else ""
        for i, variants in enumerate(name_variants(target)):
            hit, matched = lookup_component(variants, client)
            rows.append({"intervention_key": key, "component": i,
                         "matched_on": f"{via}{matched}" if matched else None,
                         "excluded": False, **(hit or {})})
    return pd.DataFrame(rows)


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


TEST_ROLES = ("investigational", "candidate")
CONTROL_ROLES = ("comparator", "background")


def classify_target_contrast(rows: pd.DataFrame) -> pd.DataFrame:
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
        single_intervention fallback trial, one tested therapeutic, no placebo
        excluded_therapy    every tested therapeutic is on the excluded list
        unmapped            no tested intervention mapped to a target
        no_test             no tested intervention at all
    """
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
        # In fallback trials, a conventional therapy alongside the tested one
        # acts as the control ("new drug vs methotrexate").
        has_control = (trial.role == "placebo").any() or all_test.excluded_therapy.any()

        if all_test.empty:
            cls, contrast = "no_test", set()
        elif test.empty:
            cls, contrast = "excluded_therapy", set()
        elif not test_targets:
            cls, contrast = "unmapped", set()
        elif is_fallback:
            n_therapeutics = mapped_test.intervention_key.nunique()
            per_drug = {frozenset(g.target_chembl_id) for _, g in
                        mapped_test.groupby("intervention_key")}
            if len(per_drug) > 1:
                cls, contrast = "head_to_head", test_targets
            elif has_control:
                cls, contrast = "target_contrast", test_targets
            elif n_therapeutics > 1:
                cls, contrast = "same_target", set()
            else:
                cls, contrast = "single_intervention", test_targets
        else:
            contrast = test_targets - control_targets
            cls = "target_contrast" if contrast else "same_target"

        out.append({
            "nct_id": nct_id,
            "contrast_class": cls,
            "contrast_targets": ";".join(sorted(contrast)),
            "has_unmapped_test": unmapped_test,
        })
    return pd.DataFrame(out)
