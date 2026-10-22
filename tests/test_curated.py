"""The real curated files must pass the provenance checks.

If this fails after editing curated/trials/, the message names the file,
line, and problem. See docs/curation/README.md for the rules.
"""

from pathlib import Path

import pandas as pd
import pytest

from causal_target_to_patient.curation import (
    check_excluded_trials,
    check_exclusions,
    check_manual_mechanisms,
    check_no_conflicts,
    check_overrides,
    check_unmapped_decisions,
    missing_batch_logs,
)

REPO = Path(__file__).resolve().parents[1]
CURATED = REPO / "curated" / "trials"
LOGS = REPO / "docs" / "curation"

FILES = {
    "manual_mechanisms.csv": check_manual_mechanisms,
    "intervention_overrides.csv": check_overrides,
    "excluded_therapies.csv": check_exclusions,
    "excluded_trials.csv": check_excluded_trials,
    "unmapped_decisions.csv": check_unmapped_decisions,
}


def read(name):
    return pd.read_csv(CURATED / name, dtype=str, keep_default_na=False)


@pytest.mark.parametrize("name", sorted(FILES))
def test_curated_file_has_provenance(name):
    problems = FILES[name](read(name), name)
    assert not problems, "\n".join(problems)


def test_no_drug_both_mapped_and_deliberately_unmapped():
    problems = check_no_conflicts(read("manual_mechanisms.csv"), read("unmapped_decisions.csv"))
    assert not problems, "\n".join(problems)


def test_every_batch_has_a_log():
    batches = set()
    for name in FILES:
        batches |= set(read(name)["batch"])
    problems = missing_batch_logs(batches, LOGS)
    assert not problems, "\n".join(problems)
