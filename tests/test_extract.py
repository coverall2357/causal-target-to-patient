"""Tests for LLM extraction, with a fake client (no network, no API key)."""

import json
from types import SimpleNamespace

import pytest

from causal_target_to_patient.labelling.extract import (
    LABEL_SCHEMA,
    MAX_QUOTE_CHARS,
    REVIEW_COLUMNS,
    SYSTEM_PROMPT,
    extract_label,
    number_found,
    quote_found,
    review_row,
    validate_label,
)

PACKET = """# NCT1: Drug X in RA

## Posted primary results

- groups: Drug X vs. Placebo; param_value: 36.0; ci_lower_limit: 24.50; ci_upper_limit: 47.5; p_value_modifier: <; p_value: 0.001

## Publications

**Drug X in rheumatoid arthritis.** Lancet 2019

RESULTS: ACR20 was 66% vs 30% (p<0.001). Drug X was well tolerated.
"""

GOOD = {
    "primary_endpoint": "ACR20 at Week 12", "primary_is_efficacy": True,
    "comparison": "Drug X vs. placebo",
    "population": "mtx_ir", "population_description": "MTX-IR adults",
    "label": "met", "dose_results": "", "dose_basis": "single_dose",
    "lowest_effective_dose": None, "dose_taken_forward": None,
    "effect_direction": "favours_investigational",
    "effect_measure": "difference in ACR20 response rate, percentage points",
    "effect_value": 36.0, "effect_ci_lower": 24.5, "effect_ci_upper": 47.5,
    "p_value": "<0.001",
    "basis": "posted_analysis", "source": "AACT results",
    "main_publication_pmid": None,
    "quote": "p_value_modifier: <; p_value: 0.001",
    "confidence": "high", "notes": "",
}


@pytest.mark.parametrize("quote, expected", [
    ("p_value_modifier: <; p_value: 0.001", True),
    ("ACR20 was 66% vs 30%   (p<0.001).", True),            # whitespace differs
    ("Drug X in rheumatoid arthritis.", True),               # bold markers ignored
    ("ACR20 was 66% vs 30% (p<0.001). Drug X was well tolerated.", True),
    ("ACR20 was 70% vs 30%", False),                         # altered number
    ("ACR20 was 66% vs 30% ... well tolerated", False),      # elided passage
    ("", False),
    (None, False),
])
def test_quote_found(quote, expected):
    assert quote_found(quote, PACKET) is expected


def test_quote_found_normalises_typographic_punctuation():
    assert quote_found("Drug X’s effect – large", "Drug X's effect - large")


def test_validate_label():
    assert validate_label(GOOD) == []
    bad = dict(GOOD, label="success", confidence="certain", population="tnf")
    del bad["quote"]
    problems = validate_label(bad)
    assert "missing quote" in problems
    assert "label not allowed: 'success'" in problems
    assert "confidence not allowed: 'certain'" in problems
    assert "population not allowed: 'tnf'" in problems


def test_validate_label_consistency_and_quote_length():
    no_evidence = dict(GOOD, label="no_results", basis="none", source="none",
                       effect_direction="not_applicable")
    assert validate_label(no_evidence) == []
    assert "basis is none but source is 'AACT results'" in validate_label(
        dict(GOOD, basis="none"))
    assert "source is none but basis is 'posted_analysis'" in validate_label(
        dict(GOOD, source="none"))
    assert "label no_results but basis is 'posted_analysis'" in validate_label(
        dict(GOOD, label="no_results", effect_direction="not_applicable"))
    long_quote = dict(GOOD, quote="x" * (MAX_QUOTE_CHARS + 1))
    assert f"quote longer than {MAX_QUOTE_CHARS} characters ({MAX_QUOTE_CHARS + 1})" in (
        validate_label(long_quote))


@pytest.mark.parametrize("value, expected", [
    (36.0, True), (36, True),          # trailing zero ignored
    (24.5, True),                      # packet says 24.50
    (-47.5, True),                     # sign ignored
    (66, True), (0.001, True),
    (36.1, False), (3.6, False),
    (None, False), (float("nan"), False),
])
def test_number_found(value, expected):
    assert number_found(value, PACKET) is expected


def test_number_found_does_not_match_inside_other_numbers():
    assert number_found(12, "NCT0012345 version 3.12") is False
    assert number_found(3.12, "NCT0012345 version 3.12") is True


class FakeMessages:
    def __init__(self, content, stop_reason="tool_use"):
        self.content, self.stop_reason, self.calls = content, stop_reason, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=self.content, stop_reason=self.stop_reason,
                               usage=SimpleNamespace(input_tokens=1200, output_tokens=150))


def fake_client(content, stop_reason="tool_use"):
    return SimpleNamespace(messages=FakeMessages(content, stop_reason))


def test_extract_label_uses_schema_and_returns_metadata():
    client = fake_client([SimpleNamespace(type="text", text=json.dumps(GOOD))])
    out = extract_label(client, PACKET, model="test-model")
    call = client.messages.calls[0]
    assert call["output_config"] == {"format": {"type": "json_schema",
                                                "schema": LABEL_SCHEMA}}
    assert "tools" not in call and "tool_choice" not in call
    assert "temperature" not in call          # rejected by current models
    assert PACKET in call["messages"][0]["content"]
    assert out["label"] == "met" and out["model"] == "test-model"
    assert (out["input_tokens"], out["output_tokens"]) == (1200, 150)


def test_extract_label_bad_answers_raise():
    client = fake_client([SimpleNamespace(type="text", text="I think it met.")], "end_turn")
    with pytest.raises(ValueError, match="not valid JSON"):
        extract_label(client, PACKET)
    with pytest.raises(ValueError, match="No text"):
        extract_label(fake_client([], "max_tokens"), PACKET)


def test_schema_requires_every_field_and_allows_no_extras():
    assert set(LABEL_SCHEMA["required"]) == set(LABEL_SCHEMA["properties"])
    assert LABEL_SCHEMA["additionalProperties"] is False


def test_review_row():
    row = review_row("NCT1", GOOD, PACKET)
    assert row["quote_found"] is True and row["problems"] == ""
    assert row["effect_numbers_missing"] == ""
    assert row["population"] == "mtx_ir" and row["effect_value"] == 36.0
    assert all(row[col] == "" for col in REVIEW_COLUMNS)
    made_up = review_row("NCT1", dict(GOOD, quote="ACR20 was 80%", effect_value=44.0,
                                      effect_ci_upper=None), PACKET)
    assert made_up["quote_found"] is False
    assert made_up["effect_numbers_missing"] == "effect_value"


def test_prompt_states_the_rules_it_is_checked_against():
    assert f"at most {MAX_QUOTE_CHARS} characters" in SYSTEM_PROMPT
    assert '"none" when there is no outcome evidence' in SYSTEM_PROMPT
    assert "Dose rule" in SYSTEM_PROMPT and "highest registered dose" in SYSTEM_PROMPT
    for population in LABEL_SCHEMA["properties"]["population"]["enum"]:
        assert f"- {population}:" in SYSTEM_PROMPT


def test_validate_label_untested_and_direction_rules():
    untested = dict(GOOD, label="untested", basis="posted_values",
                    effect_value=None, p_value=None)
    assert validate_label(untested) == []
    assert validate_label(dict(untested, basis="publication")) == []
    assert "label met rests on values without a test; use untested" in validate_label(
        dict(GOOD, basis="posted_values"))
    assert "label untested but basis is 'posted_analysis'" in validate_label(
        dict(untested, basis="posted_analysis"))
    assert "label untested needs an effect_direction" in validate_label(
        dict(untested, effect_direction="not_applicable"))
    stopped = dict(GOOD, label="terminated_futility", basis="stop_reason", source="why_stopped")
    assert "label terminated_futility but effect_direction is 'favours_investigational'" in (
        validate_label(stopped))
    assert validate_label(dict(stopped, effect_direction="not_applicable")) == []
    assert "effect_direction not allowed: 'up'" in validate_label(
        dict(GOOD, effect_direction="up"))


def test_prompt_orders_tests_before_untested_values():
    a = SYSTEM_PROMPT.index("a. a posted statistical analysis")
    b = SYSTEM_PROMPT.index("b. the main results publication's test")
    c = SYSTEM_PROMPT.index("c. primary-endpoint values for each arm without a test")
    assert a < b < c
    assert "Never judge statistical significance yourself" in SYSTEM_PROMPT
    assert "primary_is_efficacy" in SYSTEM_PROMPT


def test_validate_label_dose_fields():
    multi = dict(GOOD, dose_basis="highest_dose_fallback", dose_results="5 mg met; 1 mg failed",
                 lowest_effective_dose="5 mg", dose_taken_forward=None)
    assert validate_label(multi) == []
    assert validate_label(dict(multi, dose_basis="prespecified")) == []
    assert "dose_basis not allowed: 'best_dose'" in validate_label(dict(multi, dose_basis="best_dose"))
    assert "single_dose but a lowest effective or forward dose is given" in validate_label(
        dict(GOOD, lowest_effective_dose="5 mg"))


def test_prompt_dose_rule_prefers_prespecified_and_records_doses():
    assert SYSTEM_PROMPT.index("prespecified primary analysis") < SYSTEM_PROMPT.index(
        "highest_dose_fallback")
    assert "not a claim that the highest dose is the best dose" in SYSTEM_PROMPT
    assert "lowest_effective_dose" in SYSTEM_PROMPT and "dose_taken_forward" in SYSTEM_PROMPT
    assert "Never infer it from outside knowledge" in SYSTEM_PROMPT
    assert "descriptive: they never decide the label" in " ".join(SYSTEM_PROMPT.split())
