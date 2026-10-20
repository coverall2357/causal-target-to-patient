"""Tests for LLM extraction, with a fake client (no network, no API key)."""

import json
from types import SimpleNamespace

import pytest

from causal_target_to_patient.labelling.extract import (
    LABEL_SCHEMA,
    REVIEW_COLUMNS,
    extract_label,
    quote_found,
    review_row,
    validate_label,
)

PACKET = """# NCT1: Drug X in RA

## Posted primary results

- groups: Drug X vs. Placebo; p_value_modifier: <; p_value: 0.001

## Publications

**Drug X in rheumatoid arthritis.** Lancet 2019

RESULTS: ACR20 was 66% vs 30% (p<0.001). Drug X was well tolerated.
"""

GOOD = {
    "primary_endpoint": "ACR20 at Week 12", "comparison": "Drug X vs. placebo",
    "label": "met", "basis": "posted_analysis", "source": "AACT results",
    "main_publication_pmid": None,
    "quote": "groups: Drug X vs. Placebo; p_value_modifier: <; p_value: 0.001",
    "confidence": "high", "notes": "",
}


@pytest.mark.parametrize("quote, expected", [
    ("groups: Drug X vs. Placebo; p_value_modifier: <; p_value: 0.001", True),
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
    bad = dict(GOOD, label="success", confidence="certain")
    del bad["quote"]
    problems = validate_label(bad)
    assert "missing quote" in problems
    assert "label not allowed: 'success'" in problems
    assert "confidence not allowed: 'certain'" in problems


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
    assert all(row[col] == "" for col in REVIEW_COLUMNS)
    made_up = review_row("NCT1", dict(GOOD, quote="ACR20 was 80%"), PACKET)
    assert made_up["quote_found"] is False
