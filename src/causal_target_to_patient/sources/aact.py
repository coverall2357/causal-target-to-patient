"""Access to the AACT cloud database (ClinicalTrials.gov as PostgreSQL).

Credentials are read from a .env file in the repo root (never committed):
    AACT_USER=your_username
    AACT_PASSWORD=your_password

Free accounts: https://aact.ctti-clinicaltrials.org/users/sign_up
"""

import os

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


def query(engine: Engine, sql: str) -> pd.DataFrame:
    """Run a SQL query against AACT and return the result as a DataFrame."""
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


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


def pull_ra_trial_interventions(engine: Engine) -> pd.DataFrame:
    """Return drug and biologic interventions of eligible RA trials, one row each."""
    return query(engine, ELIGIBLE_CTE + f"""
        SELECT i.nct_id, i.intervention_type, i.name AS intervention_name
        FROM eligible
        JOIN ctgov.interventions i USING (nct_id)
        WHERE i.intervention_type IN ({_sql_list(THERAPEUTIC_INTERVENTION_TYPES)})
        ORDER BY i.nct_id, i.name
    """)
