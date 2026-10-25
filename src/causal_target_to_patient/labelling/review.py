"""Review sheets for hand labelling, and checks on the hand labels entered.

A review sheet has one row per trial, in three blocks:
1. context: trial, title, the investigational drug(s), evidence tier, and the
   path of the evidence packet to read;
2. the reviewer's columns, to fill in (REVIEW_COLUMNS);
3. the model's answer and the automatic checks on it.

A blind sheet has only the first two blocks. It is for the held-out test set,
which is labelled by hand before the model's answers are seen.

check_hand_labels() lists entries outside the allowed values (for example a
typo such as "mett"), so they can be fixed before agreement is scored rather
than silently counted as disagreements.
"""

from pathlib import Path

import pandas as pd

from causal_target_to_patient.labelling.extract import LABELS, POPULATIONS, REVIEW_COLUMNS

CONTEXT_COLUMNS = ("nct_id", "brief_title", "investigational", "evidence_tier", "packet")
TEST_ROLES = ("investigational", "candidate")
AGREE = ("yes", "no")
ISSUE_CODES = (
    "wrong_label", "wrong_population", "wrong_comparison", "wrong_endpoint",
    "wrong_effect", "bad_quote", "missed_evidence", "efficacy_flag_wrong", "other",
)


def trial_context(nct_id: str, packet: dict, packet_path: Path, evidence_tier=None) -> dict:
    """Context columns for one trial, from its evidence packet (the JSON)."""
    interventions = (packet.get("our_mapping") or {}).get("interventions", [])
    tested = [i["name"] for i in interventions if i.get("role") in TEST_ROLES]
    return {
        "nct_id": nct_id,
        "brief_title": (packet.get("trial") or {}).get("brief_title"),
        "investigational": "; ".join(dict.fromkeys(tested)),
        "evidence_tier": evidence_tier,
        "packet": str(packet_path),
    }


def review_sheet(context: list[dict], rows: list[dict]) -> pd.DataFrame:
    """Context, then the reviewer's columns, then the model's answer and checks.

    `rows` are review_row() outputs, matched to `context` on nct_id.
    """
    model = pd.DataFrame(rows).drop(columns=list(REVIEW_COLUMNS), errors="ignore")
    sheet = pd.DataFrame(context).merge(model, on="nct_id", how="left")
    for col in REVIEW_COLUMNS:
        sheet[col] = ""
    model_columns = [c for c in model.columns if c != "nct_id"]
    return sheet[list(CONTEXT_COLUMNS) + list(REVIEW_COLUMNS) + model_columns]


def blind_sheet(context: list[dict]) -> pd.DataFrame:
    """Context and the reviewer's columns only: no model answers."""
    sheet = pd.DataFrame(context)
    for col in REVIEW_COLUMNS:
        sheet[col] = ""
    return sheet[list(CONTEXT_COLUMNS) + list(REVIEW_COLUMNS)]


def _blank(value) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or not str(value).strip()


def check_hand_labels(sheet: pd.DataFrame) -> list[str]:
    """Entries outside the allowed values, one message per problem.

    Blank entries are allowed (not yet labelled, or not needed). `issue` may
    hold several codes separated by semicolons.
    """
    allowed = {"your_label": LABELS, "your_population": POPULATIONS, "agree": AGREE}
    problems = []
    for row in sheet.to_dict("records"):
        for col, values in allowed.items():
            value = row.get(col)
            if not _blank(value) and str(value).strip() not in values:
                problems.append(f"{row['nct_id']}: {col} not allowed: {str(value).strip()!r}")
        issue = row.get("issue")
        if not _blank(issue):
            for code in (c.strip() for c in str(issue).split(";")):
                if code and code not in ISSUE_CODES:
                    problems.append(f"{row['nct_id']}: issue code not allowed: {code!r}")
    return problems
