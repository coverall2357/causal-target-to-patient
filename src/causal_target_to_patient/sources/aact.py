"""Access to the AACT cloud database (ClinicalTrials.gov as PostgreSQL).

Credentials are read from a .env file in the repo root (never committed):
    AACT_USER=your_username
    AACT_PASSWORD=your_password

Free accounts: https://aact.ctti-clinicaltrials.org/users/sign_up
"""

import os
import re

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

AACT_HOST = "aact-db.ctti-clinicaltrials.org"
AACT_PORT = 5432
AACT_DB = "aact"

# Definition of an RA study, used throughout the trial universe. Two routes,
# combined: the MeSH term AACT derives from each study's conditions (catches
# synonyms), and a direct text match on the listed condition (catches studies
# whose MeSH mapping is missing). Prepend to a query and select from `ra`.
RA_STUDIES_CTE = """
WITH ra AS (
    SELECT nct_id FROM ctgov.browse_conditions
    WHERE mesh_term = 'Arthritis, Rheumatoid'
    UNION
    SELECT nct_id FROM ctgov.conditions
    WHERE downcase_name LIKE '%rheumatoid arthritis%'
)
"""

# --- Trial-universe inclusion criteria (see docs/inclusion_criteria.md) -----

# Phases used for efficacy and safety labels. Phase 1/2 is kept because it is
# often the first test of efficacy in patients.
LABELLING_PHASES = ("PHASE1/PHASE2", "PHASE2", "PHASE2/PHASE3", "PHASE3")

# Post-approval trials: rarely change a target's label, but include the
# biomarker-stratified trials relevant to the therapeutic -> patient link.
CONTEXT_PHASES = ("PHASE4",)

# Withdrawn trials never enrolled anyone, so they hold no evidence.
EXCLUDED_STATUSES = ("WITHDRAWN",)

# Trials without results yet; kept to show which targets are being tested now.
ONGOING_STATUSES = (
    "RECRUITING",
    "NOT_YET_RECRUITING",
    "ACTIVE_NOT_RECRUITING",
    "ENROLLING_BY_INVITATION",
)

# A study qualifies if at least one intervention is a drug or biologic. Studies
# also listing OTHER are kept, because placebo is often coded as OTHER.
THERAPEUTIC_INTERVENTION_TYPES = ("DRUG", "BIOLOGICAL")

# Arm types that mark an intervention as given to the control group. AACT's
# coding has changed over time ("Active Comparator" vs "ACTIVE_COMPARATOR"),
# so arm types are normalised to upper case with underscores before matching.
EXPERIMENTAL_ARM_TYPES = ("EXPERIMENTAL",)
COMPARATOR_ARM_TYPES = (
    "ACTIVE_COMPARATOR",
    "PLACEBO_COMPARATOR",
    "SHAM_COMPARATOR",
    "NO_INTERVENTION",
)

# Inactive controls: placebo or placebos, vehicle, and sham.
PLACEBO_PATTERN = re.compile(r"\b(placebos?|vehicle|sham)\b", re.IGNORECASE)


def _sql_list(values: tuple[str, ...]) -> str:
    """Format a tuple of constants as a SQL IN-list."""
    return ", ".join(f"'{v}'" for v in values)


# Extends RA_STUDIES_CTE with `eligible`: the nct_ids that pass the criteria.
ELIGIBLE_CTE = RA_STUDIES_CTE + f""",
eligible AS (
    SELECT s.nct_id
    FROM ra JOIN ctgov.studies s USING (nct_id)
    WHERE s.study_type ILIKE 'interventional'
      AND s.phase IN ({_sql_list(LABELLING_PHASES + CONTEXT_PHASES)})
      AND s.overall_status NOT IN ({_sql_list(EXCLUDED_STATUSES)})
      AND EXISTS (
          SELECT 1 FROM ctgov.interventions i
          WHERE i.nct_id = s.nct_id
            AND i.intervention_type IN ({_sql_list(THERAPEUTIC_INTERVENTION_TYPES)})
      )
)
"""


def get_engine() -> Engine:
    """Return a SQLAlchemy engine for the AACT cloud database."""
    load_dotenv()
    user = os.environ["AACT_USER"]
    password = os.environ["AACT_PASSWORD"]
    url = f"postgresql+psycopg2://{user}:{password}@{AACT_HOST}:{AACT_PORT}/{AACT_DB}"
    return create_engine(url)


def query(engine: Engine, sql: str, params: dict | None = None) -> pd.DataFrame:
    """Run a SQL query against AACT and return the result as a DataFrame.

    `params` binds named parameters in the SQL (":name"), e.g. a list of trial
    IDs for "nct_id = ANY(:ids)".
    """
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def query_ra(engine: Engine, sql: str) -> pd.DataFrame:
    """Run a query with the RA study definition available as the CTE `ra`."""
    return query(engine, RA_STUDIES_CTE + sql)


def assign_trial_set(phase: str, status: str) -> str:
    """Assign an eligible trial to the labelling, context, or ongoing set.

    Ongoing status takes precedence: a trial without results cannot inform a
    label, whatever its phase.
    """
    if status in ONGOING_STATUSES:
        return "ongoing"
    if phase in CONTEXT_PHASES:
        return "context"
    return "labelling"


def pull_ra_trials(engine: Engine) -> pd.DataFrame:
    """Return eligible RA trials, one row per trial, with their trial set."""
    trials = query(engine, ELIGIBLE_CTE + """
        SELECT s.nct_id, s.brief_title, s.phase, s.overall_status,
               s.why_stopped, s.start_date, s.completion_date, s.enrollment,
               s.results_first_submitted_date,
               sp.name AS lead_sponsor, sp.agency_class AS sponsor_class
        FROM eligible
        JOIN ctgov.studies s USING (nct_id)
        LEFT JOIN ctgov.sponsors sp
               ON sp.nct_id = s.nct_id AND sp.lead_or_collaborator ILIKE 'lead'
        ORDER BY s.nct_id
    """)
    trials = trials.drop_duplicates(subset="nct_id")
    trials["trial_set"] = [
        assign_trial_set(p, s) for p, s in zip(trials.phase, trials.overall_status)
    ]
    return trials


def assign_intervention_role(
    name: str, in_experimental: bool, in_comparator: bool
) -> str:
    """Classify an intervention's role in its trial from the arms it appears in.

    investigational: experimental arms only, the therapeutic being tested.
    background:      experimental and comparator arms, given to everyone.
    comparator:      comparator arms only, the active control.
    unassigned:      not linked to an experimental or comparator arm.
    placebo:         an inactive control (placebo, vehicle, sham), whatever its arms.
    """
    if PLACEBO_PATTERN.search(name):
        return "placebo"
    if in_experimental and in_comparator:
        return "background"
    if in_experimental:
        return "investigational"
    if in_comparator:
        return "comparator"
    return "unassigned"


# Normalised arm type, e.g. "Active Comparator" -> "ACTIVE_COMPARATOR".
_ARM_TYPE_SQL = "UPPER(REPLACE(dg.group_type, ' ', '_'))"


def pull_ra_trial_interventions(engine: Engine) -> pd.DataFrame:
    """Return drug and biologic interventions of eligible RA trials with their role.

    One row per intervention, with flags for the arm types it appears in and
    a role from assign_intervention_role.
    """
    interventions = query(engine, ELIGIBLE_CTE + f"""
        , arms AS (
            SELECT dgi.intervention_id, {_ARM_TYPE_SQL} AS arm_type
            FROM ctgov.design_group_interventions dgi
            JOIN ctgov.design_groups dg ON dg.id = dgi.design_group_id
        )
        SELECT i.id AS intervention_id, i.nct_id, i.intervention_type,
               i.name AS intervention_name,
               COALESCE(BOOL_OR(a.arm_type IN ({_sql_list(EXPERIMENTAL_ARM_TYPES)})),
                        FALSE) AS in_experimental,
               COALESCE(BOOL_OR(a.arm_type IN ({_sql_list(COMPARATOR_ARM_TYPES)})),
                        FALSE) AS in_comparator
        FROM eligible
        JOIN ctgov.interventions i USING (nct_id)
        LEFT JOIN arms a ON a.intervention_id = i.id
        WHERE i.intervention_type IN ({_sql_list(THERAPEUTIC_INTERVENTION_TYPES)})
        GROUP BY i.id, i.nct_id, i.intervention_type, i.name
        ORDER BY i.nct_id, i.name
    """)
    interventions["role"] = [
        assign_intervention_role(n, e, c)
        for n, e, c in zip(
            interventions.intervention_name,
            interventions.in_experimental,
            interventions.in_comparator,
        )
    ]
    return interventions


def pull_ra_trial_arms(engine: Engine) -> pd.DataFrame:
    """Return arm counts per eligible RA trial: total, experimental, comparator."""
    return query(engine, ELIGIBLE_CTE + f"""
        SELECT e.nct_id,
               COUNT(dg.id) AS n_arms,
               COUNT(dg.id) FILTER (
                   WHERE {_ARM_TYPE_SQL} IN ({_sql_list(EXPERIMENTAL_ARM_TYPES)})
               ) AS n_experimental_arms,
               COUNT(dg.id) FILTER (
                   WHERE {_ARM_TYPE_SQL} IN ({_sql_list(COMPARATOR_ARM_TYPES)})
               ) AS n_comparator_arms
        FROM eligible e
        LEFT JOIN ctgov.design_groups dg ON dg.nct_id = e.nct_id
        GROUP BY e.nct_id
        ORDER BY e.nct_id
    """)


def flag_trials_for_review(
    interventions: pd.DataFrame, arms: pd.DataFrame
) -> pd.DataFrame:
    """Add investigational counts and a review flag to the per-trial arm table.

    Flagged trials: no investigational intervention (nothing identified as
    tested), or several investigational interventions with no comparator arm
    (background therapy may be mislabelled as investigational).
    """
    n_inv = (
        interventions[interventions.role == "investigational"]
        .groupby("nct_id").size().rename("n_investigational")
    )
    out = arms.merge(n_inv, on="nct_id", how="left")
    out["n_investigational"] = out.n_investigational.fillna(0).astype(int)
    out["needs_review"] = (out.n_investigational == 0) | (
        (out.n_investigational > 1) & (out.n_comparator_arms == 0)
    )
    return out


def apply_candidate_fallback(
    interventions: pd.DataFrame, flagged_arms: pd.DataFrame
) -> pd.DataFrame:
    """Mark non-placebo interventions as candidates in trials with no investigational one.

    Arm records are missing or uninformative in these trials (no arms, all arms
    coded as comparators, or the same drug in every arm). Their interventions
    become candidates, and the target contrast decides what the trial tests.
    The arm-based role is kept in `arm_role` for traceability.
    """
    no_investigational = set(
        flagged_arms.loc[flagged_arms.n_investigational == 0, "nct_id"]
    )
    out = interventions.copy()
    out["arm_role"] = out.role
    mask = out.nct_id.isin(no_investigational) & (out.role != "placebo")
    out.loc[mask, "role"] = "candidate"
    return out


# --- Outcome evidence ---------------------------------------------------------

# Best available source for a trial's efficacy outcome, most structured first.
EVIDENCE_TIERS = (
    "primary analysis",  # posted primary outcome with a statistical test (p-value)
    "primary results",   # posted primary outcome, no p-value
    "publication",       # no posted primary outcome, but a linked publication
    "stop reason",       # only a reason for stopping early
    "none",
)


def evidence_tier(
    n_primary_outcomes: int,
    n_primary_p_values: int,
    n_publications: int,
    why_stopped: object,
) -> str:
    """Assign a trial the most structured source available for its efficacy outcome."""
    if n_primary_outcomes > 0 and n_primary_p_values > 0:
        return "primary analysis"
    if n_primary_outcomes > 0:
        return "primary results"
    if n_publications > 0:
        return "publication"
    if isinstance(why_stopped, str) and why_stopped.strip():
        return "stop reason"
    return "none"


def pull_ra_trial_evidence(engine: Engine) -> pd.DataFrame:
    """Return, per eligible RA trial, counts of the outcome evidence AACT holds.

    n_primary_outcomes:  primary outcomes with posted results
    n_primary_p_values:  analyses of those outcomes that report a p-value
    n_result_refs:       publications the sponsor linked as reporting results
    n_derived_refs:      publications PubMed linked by NCT number; these can
                         include secondary analyses, not only the main report
    """
    return query(engine, ELIGIBLE_CTE + """
        , primary_outcomes AS (
            SELECT o.nct_id,
                   COUNT(DISTINCT o.id) AS n_primary_outcomes,
                   COUNT(DISTINCT oa.id) FILTER (WHERE oa.p_value IS NOT NULL)
                       AS n_primary_p_values
            FROM ctgov.outcomes o
            LEFT JOIN ctgov.outcome_analyses oa ON oa.outcome_id = o.id
            WHERE UPPER(o.outcome_type) = 'PRIMARY'
            GROUP BY o.nct_id
        ),
        refs AS (
            SELECT nct_id,
                   COUNT(DISTINCT pmid) FILTER (WHERE UPPER(reference_type) = 'RESULT')
                       AS n_result_refs,
                   COUNT(DISTINCT pmid) FILTER (WHERE UPPER(reference_type) = 'DERIVED')
                       AS n_derived_refs
            FROM ctgov.study_references
            GROUP BY nct_id
        )
        SELECT e.nct_id,
               COALESCE(p.n_primary_outcomes, 0) AS n_primary_outcomes,
               COALESCE(p.n_primary_p_values, 0) AS n_primary_p_values,
               COALESCE(r.n_result_refs, 0) AS n_result_refs,
               COALESCE(r.n_derived_refs, 0) AS n_derived_refs
        FROM eligible e
        LEFT JOIN primary_outcomes p USING (nct_id)
        LEFT JOIN refs r USING (nct_id)
        ORDER BY e.nct_id
    """)


# --- Evidence for outcome labelling --------------------------------------------

# AACT tables pulled per trial for an evidence packet. All columns are kept
# (SELECT *), so a renamed column in AACT doesn't break the pull; the packet
# builder picks the columns it needs.
EVIDENCE_TABLES = (
    "studies",
    "brief_summaries",
    "design_groups",
    "design_group_interventions",
    "interventions",
    "design_outcomes",
    "outcomes",
    "result_groups",
    "outcome_measurements",
    "outcome_analyses",
    "outcome_analysis_groups",
    "study_references",
)


def pull_trial_evidence(engine: Engine, nct_ids: list[str]) -> dict[str, pd.DataFrame]:
    """Return the AACT tables needed for evidence packets, filtered to `nct_ids`.

    One DataFrame per table in EVIDENCE_TABLES, all rows for the given trials.
    """
    return {
        table: query(
            engine,
            f"SELECT * FROM ctgov.{table} WHERE nct_id = ANY(:ids)",
            params={"ids": list(nct_ids)},
        )
        for table in EVIDENCE_TABLES
    }
