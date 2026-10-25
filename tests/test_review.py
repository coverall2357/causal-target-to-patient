"""Tests for review sheets and the hand-label check."""

from pathlib import Path

import pandas as pd

from causal_target_to_patient.labelling.extract import REVIEW_COLUMNS
from causal_target_to_patient.labelling.review import (
    CONTEXT_COLUMNS,
    blind_sheet,
    check_hand_labels,
    review_sheet,
    trial_context,
)

PACKET = {
    "trial": {"brief_title": "Drug X in RA"},
    "our_mapping": {"interventions": [
        {"name": "Drug X 5 mg", "role": "investigational"},
        {"name": "Drug X 5 mg", "role": "investigational"},     # repeated arm
        {"name": "Placebo", "role": "placebo"},
        {"name": "Methotrexate", "role": "background"},
    ]},
}


def context():
    return [trial_context("NCT1", PACKET, Path("/data/evidence/NCT1.md"), "primary analysis")]


def test_trial_context():
    (c,) = context()
    assert c == {"nct_id": "NCT1", "brief_title": "Drug X in RA",
                 "investigational": "Drug X 5 mg", "evidence_tier": "primary analysis",
                 "packet": "/data/evidence/NCT1.md"}
    assert trial_context("NCT2", {}, Path("p"))["investigational"] == ""


def test_review_sheet_column_order():
    rows = [{"nct_id": "NCT1", "label": "met", "quote_found": True,
             **{c: "" for c in REVIEW_COLUMNS}}]
    sheet = review_sheet(context(), rows)
    n_context, n_review = len(CONTEXT_COLUMNS), len(REVIEW_COLUMNS)
    assert list(sheet.columns[:n_context]) == list(CONTEXT_COLUMNS)
    assert list(sheet.columns[n_context:n_context + n_review]) == list(REVIEW_COLUMNS)
    assert list(sheet.columns[n_context + n_review:]) == ["label", "quote_found"]
    assert sheet.loc[0, "label"] == "met" and sheet.loc[0, "your_label"] == ""


def test_blind_sheet_has_no_model_columns():
    sheet = blind_sheet(context())
    assert list(sheet.columns) == list(CONTEXT_COLUMNS) + list(REVIEW_COLUMNS)


def test_check_hand_labels():
    sheet = pd.DataFrame([
        {"nct_id": "NCT1", "your_label": "met", "your_population": "mtx_ir",
         "agree": "yes", "issue": ""},
        {"nct_id": "NCT2", "your_label": "mett", "your_population": "MTX-IR",
         "agree": "Y", "issue": "wrong_label; typo_code"},
        {"nct_id": "NCT3", "your_label": "", "your_population": None,
         "agree": "no", "issue": "efficacy_flag_wrong"},
    ])
    assert check_hand_labels(sheet) == [
        "NCT2: your_label not allowed: 'mett'",
        "NCT2: your_population not allowed: 'MTX-IR'",
        "NCT2: agree not allowed: 'Y'",
        "NCT2: issue code not allowed: 'typo_code'",
    ]
