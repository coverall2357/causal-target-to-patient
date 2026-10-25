"""Evidence packets for LLM outcome labelling.

A packet holds everything the labeller may use for one trial: the registered
design and eligibility criteria, the registered and posted primary outcomes,
our intervention mapping, and the linked publications. The LLM receives the packet as JSON; a reviewer
reads the same packet rendered as markdown.
"""

import datetime as dt
import math

import pandas as pd

TRIAL_COLUMNS = (
    "nct_id", "brief_title", "official_title", "phase", "overall_status",
    "why_stopped", "enrollment", "start_date", "completion_date",
    "results_first_submitted_date",
)
ARM_COLUMNS = ("group_type", "title", "description")
PLANNED_OUTCOME_COLUMNS = ("measure", "time_frame", "description")
OUTCOME_COLUMNS = (
    "title", "description", "time_frame", "population", "units",
    "param_type", "dispersion_type",
)
MEASUREMENT_COLUMNS = (
    "group", "classification", "category", "title", "param_value",
    "dispersion_value", "dispersion_lower_limit", "dispersion_upper_limit",
    "explanation_of_na",
)
ANALYSIS_COLUMNS = (
    "groups", "groups_description", "non_inferiority_type",
    "non_inferiority_description", "method", "method_description",
    "param_type", "param_value", "dispersion_type", "dispersion_value",
    "ci_percent", "ci_n_sides", "ci_lower_limit", "ci_upper_limit",
    "p_value_modifier", "p_value", "p_value_description",
    "estimate_description", "other_analysis_description",
)
# Posted measurements kept per outcome. Safety outcomes broken down by adverse-
# event term can run to hundreds of rows, which would swamp the labeller.
MAX_MEASUREMENTS = 60

# Eligibility criteria kept per trial. They say who was enrolled (for example,
# inadequate response to methotrexate or to TNF inhibitors), which the
# population label needs; the full text can run to several thousand characters.
MAX_CRITERIA_CHARS = 4000

# Results publications first, then PubMed's NCT-number links, then background.
REFERENCE_ORDER = {"RESULT": 0, "DERIVED": 1, "BACKGROUND": 2}


def _clean(value):
    """A JSON-ready scalar, or None for missing and blank values."""
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    if hasattr(value, "item"):  # numpy scalar
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
        return value.isoformat()[:10]
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _records(df: pd.DataFrame, columns) -> list[dict]:
    """Rows as dicts with the given columns that exist, dropping missing values."""
    present = [c for c in columns if c in df.columns]
    out = []
    for row in df[present].to_dict("records"):
        record = {k: _clean(v) for k, v in row.items()}
        out.append({k: v for k, v in record.items() if v is not None})
    return out


def _rows(tables: dict[str, pd.DataFrame], name: str, nct_id: str) -> pd.DataFrame:
    df = tables.get(name)
    if df is None or df.empty or "nct_id" not in df.columns:
        return pd.DataFrame()
    return df[df.nct_id == nct_id]


def _is_primary(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper() == "PRIMARY"


def _pmid(value) -> str | None:
    value = _clean(value)
    if value is None:
        return None
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def _arms(tables, nct_id) -> list[dict]:
    groups = _rows(tables, "design_groups", nct_id)
    links = _rows(tables, "design_group_interventions", nct_id)
    interventions = _rows(tables, "interventions", nct_id)
    names = (dict(zip(interventions["id"], interventions["name"]))
             if {"id", "name"} <= set(interventions.columns) else {})
    arms = []
    for _, group in groups.iterrows():
        arm = _records(group.to_frame().T, ARM_COLUMNS)[0]
        if {"design_group_id", "intervention_id"} <= set(links.columns):
            ids = links.loc[links.design_group_id == group["id"], "intervention_id"]
            arm["interventions"] = sorted(str(names[i]) for i in ids if i in names)
        arms.append(arm)
    return arms


def _posted_primary_results(tables, nct_id) -> list[dict]:
    outcomes = _rows(tables, "outcomes", nct_id)
    if outcomes.empty or "outcome_type" not in outcomes.columns:
        return []
    outcomes = outcomes[_is_primary(outcomes.outcome_type)]
    groups = _rows(tables, "result_groups", nct_id)
    group_title = (dict(zip(groups["id"], groups["title"]))
                   if {"id", "title"} <= set(groups.columns) else {})
    measurements = _rows(tables, "outcome_measurements", nct_id)
    analyses = _rows(tables, "outcome_analyses", nct_id)
    analysis_groups = _rows(tables, "outcome_analysis_groups", nct_id)

    results = []
    for _, outcome in outcomes.iterrows():
        record = _records(outcome.to_frame().T, OUTCOME_COLUMNS)[0]
        if "outcome_id" in measurements.columns:
            m = measurements[measurements.outcome_id == outcome["id"]].copy()
            if "result_group_id" in m.columns:
                m["group"] = m.result_group_id.map(group_title)
            record["measurements"] = _records(m.head(MAX_MEASUREMENTS), MEASUREMENT_COLUMNS)
            if len(m) > MAX_MEASUREMENTS:
                record["measurements_truncated"] = (
                    f"showing the first {MAX_MEASUREMENTS} of {len(m)} rows")
        if "outcome_id" in analyses.columns:
            a = analyses[analyses.outcome_id == outcome["id"]].copy()
            if {"outcome_analysis_id", "result_group_id"} <= set(analysis_groups.columns):
                compared = (analysis_groups.assign(title=analysis_groups.result_group_id
                                                   .map(group_title))
                            .groupby("outcome_analysis_id").title
                            .apply(lambda t: " vs. ".join(str(x) for x in t)))
                a["groups"] = a["id"].map(compared)
            record["analyses"] = _records(a, ANALYSIS_COLUMNS)
        results.append(record)
    return results


def _publications(tables, nct_id, abstracts: dict[str, dict]) -> list[dict]:
    refs = _rows(tables, "study_references", nct_id)
    pubs = []
    for _, ref in refs.iterrows():
        pmid = _pmid(ref.get("pmid"))
        kind = str(ref.get("reference_type") or "").upper()
        pub = {"pmid": pmid, "reference_type": kind or None,
               "citation": _clean(ref.get("citation"))}
        if pmid in abstracts:
            pub.update({k: v for k, v in abstracts[pmid].items() if k != "pmid"})
        pubs.append({k: v for k, v in pub.items() if v not in (None, "", [])})
    return sorted(pubs, key=lambda p: (REFERENCE_ORDER.get(p.get("reference_type"), 3),
                                       p.get("pmid") or ""))


def intervention_mapping(
    nct_id: str,
    interventions: pd.DataFrame,
    name_molecules: pd.DataFrame,
    molecule_targets: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> dict:
    """Our pipeline's view of a trial: roles, mapped molecules, targets, contrast.

    `interventions` is ra_interventions_*; `name_molecules`, `molecule_targets`
    and `contrasts` are the outputs of map_ra_interventions.py.
    """
    contrast = contrasts[contrasts.nct_id == nct_id]
    out = (_records(contrast, ("contrast_class", "contrast_targets"))[0]
           if len(contrast) else {})
    targets_by_key = {
        key: sorted(f"{t.gene_symbols} ({t.target_pref_name})"
                    for t in g.itertuples() if isinstance(t.gene_symbols, str))
        for key, g in molecule_targets.groupby("parent_chembl_id")
    }
    rows = []
    for row in interventions[interventions.nct_id == nct_id].itertuples():
        key = str(row.intervention_name).strip().lower()
        comps = name_molecules[name_molecules.intervention_key == key]
        components = []
        for c in comps.itertuples():
            comp = {"lookup_name": c.lookup_name, "molecule": c.molecule_pref_name,
                    "excluded_therapy": bool(c.excluded_therapy) or bool(c.excluded),
                    "targets": targets_by_key.get(c.mechanism_key, [])}
            components.append({k: _clean(v) for k, v in comp.items()
                               if _clean(v) not in (None, [])})
        rows.append({"name": row.intervention_name, "role": row.role,
                     "arm_role": getattr(row, "arm_role", None),
                     "components": components})
    out["interventions"] = rows
    return out


def build_packet(
    nct_id: str,
    tables: dict[str, pd.DataFrame],
    mapping: dict | None = None,
    abstracts: dict[str, dict] | None = None,
) -> dict:
    """Assemble the evidence packet for one trial from AACT tables and abstracts."""
    studies = _rows(tables, "studies", nct_id)
    summaries = _rows(tables, "brief_summaries", nct_id)
    eligibility = _rows(tables, "eligibilities", nct_id)
    planned = _rows(tables, "design_outcomes", nct_id)
    if "outcome_type" in planned.columns:
        planned = planned[_is_primary(planned.outcome_type)]
    return {
        "nct_id": nct_id,
        "trial": _records(studies, TRIAL_COLUMNS)[0] if len(studies) else {"nct_id": nct_id},
        "summary": (_clean(summaries["description"].iloc[0])
                    if "description" in summaries.columns and len(summaries) else None),
        "eligibility_criteria": _criteria(eligibility),
        "arms": _arms(tables, nct_id),
        "our_mapping": mapping or {},
        "planned_primary_outcomes": _records(planned, PLANNED_OUTCOME_COLUMNS),
        "posted_primary_results": _posted_primary_results(tables, nct_id),
        "publications": _publications(tables, nct_id, abstracts or {}),
    }


def _criteria(eligibility: pd.DataFrame) -> str | None:
    """The trial's eligibility criteria text, capped at MAX_CRITERIA_CHARS."""
    if "criteria" not in eligibility.columns or not len(eligibility):
        return None
    text = _clean(eligibility["criteria"].iloc[0])
    if not text:
        return None
    if len(text) > MAX_CRITERIA_CHARS:
        text = (text[:MAX_CRITERIA_CHARS]
                + f"\n[truncated: showing the first {MAX_CRITERIA_CHARS} of {len(text)} characters]")
    return text


def packet_counts(packet: dict) -> dict:
    """Summary counts for a packet, for the build script's report."""
    posted = packet["posted_primary_results"]
    pubs = packet["publications"]
    return {
        "nct_id": packet["nct_id"],
        "arms": len(packet["arms"]),
        "planned_primary": len(packet["planned_primary_outcomes"]),
        "posted_primary": len(posted),
        "analyses": sum(len(o.get("analyses", [])) for o in posted),
        "publications": len(pubs),
        "abstracts": sum(1 for p in pubs if p.get("abstract")),
    }


def _fields(record: dict, skip=()) -> list[str]:
    return [f"- **{k}:** {v}" for k, v in record.items() if k not in skip]


def render_markdown(packet: dict) -> str:
    """Render a packet as markdown for human review. Same content as the JSON."""
    trial = packet["trial"]
    lines = [f"# {packet['nct_id']}: {trial.get('brief_title', '')}", ""]
    lines += _fields(trial, skip=("nct_id", "brief_title")) + [""]
    if packet.get("summary"):
        lines += ["## Summary", "", packet["summary"], ""]
    if packet.get("eligibility_criteria"):
        lines += ["## Eligibility criteria", "", packet["eligibility_criteria"], ""]

    lines += ["## Arms", ""]
    for arm in packet["arms"]:
        lines.append(f"- **{arm.get('title', '')}** ({arm.get('group_type', '')}): "
                     f"{', '.join(arm.get('interventions', []))}")
        if arm.get("description"):
            lines.append(f"  {arm['description']}")
    lines.append("")

    mapping = packet.get("our_mapping") or {}
    lines += ["## Our mapping", "",
              f"- **contrast class:** {mapping.get('contrast_class')}",
              f"- **contrast targets:** {mapping.get('contrast_targets')}"]
    for i in mapping.get("interventions", []):
        comps = "; ".join(
            f"{c.get('molecule') or c.get('lookup_name')}"
            + (" [excluded]" if c.get("excluded_therapy") else "")
            + (f" → {', '.join(c['targets'])}" if c.get("targets") else "")
            for c in i["components"]) or "no molecule"
        lines.append(f"- {i['name']}: role {i['role']} (arms: {i.get('arm_role')}); {comps}")
    lines.append("")

    lines += ["## Planned primary outcomes (registered)", ""]
    for o in packet["planned_primary_outcomes"] or [{"measure": "none registered"}]:
        lines += [f"- **{o.get('measure')}** ({o.get('time_frame', 'no time frame')})"]
        if o.get("description"):
            lines.append(f"  {o['description']}")
    lines.append("")

    lines += ["## Posted primary results", ""]
    if not packet["posted_primary_results"]:
        lines += ["None posted.", ""]
    for o in packet["posted_primary_results"]:
        lines += [f"### {o.get('title')}", ""]
        lines += _fields(o, skip=("title", "measurements", "analyses")) + [""]
        if o.get("measurements"):
            lines += ["**Measurements**", ""]
            for m in o["measurements"]:
                lines.append("- " + "; ".join(f"{k}: {v}" for k, v in m.items()))
            lines.append("")
        if o.get("analyses"):
            lines += ["**Analyses**", ""]
            for a in o["analyses"]:
                lines.append("- " + "; ".join(f"{k}: {v}" for k, v in a.items()))
            lines.append("")

    lines += ["## Publications", ""]
    if not packet["publications"]:
        lines += ["None linked.", ""]
    for p in packet["publications"]:
        label = f"PMID {p['pmid']}" if p.get("pmid") else "No PMID"
        lines += [f"### {label} ({p.get('reference_type')})", ""]
        if p.get("title"):
            lines.append(f"**{p['title']}** {p.get('journal', '')} {p.get('year', '')}")
        if p.get("publication_types"):
            lines.append(f"*{', '.join(p['publication_types'])}*")
        lines += ["", p.get("abstract") or p.get("citation") or "(no abstract)", ""]
    return "\n".join(lines)
