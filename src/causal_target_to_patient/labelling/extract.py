"""LLM extraction of a trial's efficacy outcome from its evidence packet.

The model reads one packet (the rendered markdown, which is also what a
reviewer reads) and answers in JSON constrained to LABEL_SCHEMA (structured
outputs), so its output is always structured. The quote it gives is then checked against the packet: a
quote that isn't in the packet is flagged, as the main guard against made-up
evidence.

Current models don't accept sampling settings such as temperature, so answers
can vary between runs. Each run's answers are saved, and repeat runs measure
how stable the labels are.
"""

import json
import re
import unicodedata

DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 2000

LABELS = ("met", "failed", "mixed", "terminated_futility", "terminated_other", "no_results")
BASES = ("posted_analysis", "posted_values", "publication", "stop_reason", "none")
CONFIDENCE = ("high", "medium", "low")

SYSTEM_PROMPT = """\
You label the efficacy outcome of one rheumatoid arthritis clinical trial from \
an evidence packet. Use only the packet. Do not use outside knowledge of the \
trial, the drug, or its later history. If the packet does not show the \
outcome, say so: answering no_results is better than guessing.

Work in this order.

1. Primary endpoint. Find the registered primary endpoint and its timepoint \
under "Planned primary outcomes". If none of the registered primary endpoints \
measures efficacy (for example, all are adverse-event counts), say so in \
notes, label the main efficacy endpoint the evidence reports, and set \
confidence to low.

2. Comparison. Identify the investigational therapeutic (role investigational \
or candidate under "Our mapping") and its control: placebo, or the active \
comparator. Background therapy given to every arm (such as methotrexate) is \
not the comparison.

3. Evidence, in this order of preference:
   a. a posted statistical analysis of the primary endpoint;
   b. posted primary-endpoint values for each arm, without a test;
   c. the main results publication: the paper reporting this trial's primary \
result. Post hoc, pooled, subgroup, extension, biomarker, and genetic \
analyses are not the main results publication, even when they repeat numbers;
   d. the reason for stopping.
If you rely only on posted values without a statistical test, judge from the \
size of the difference and any confidence intervals, set confidence to medium \
at most, and explain in notes.

4. Label, using these rules:
- met: the primary endpoint is statistically significant in favour of the \
investigational therapeutic over the control at the registered timepoint. For \
a non-inferiority design, non-inferiority is shown. With several doses, met \
if the highest registered dose meets every co-primary endpoint, even if lower \
doses do not (record the dose results in notes).
- mixed: the highest dose fails but another dose succeeds; co-primary \
endpoints disagree; or the therapeutic beats placebo but not the active \
comparator, when superiority to both was a primary objective.
- failed: the trial reached its primary readout and the primary endpoint was \
not met.
- terminated_futility: stopped early for lack of efficacy or futility, \
without a completed primary readout in the packet.
- terminated_other: stopped early for business, enrolment, safety or other \
reasons, without an efficacy readout in the packet.
- no_results: the packet holds no outcome evidence for the primary endpoint.

5. Quote. Copy, character for character, one contiguous passage of at most \
300 characters from the packet that supports the label: a line of posted \
analysis, a line of posted values, a sentence from an abstract, or the stop \
reason. Do not paraphrase, join separate passages, or add ellipses. For \
no_results, quote the line showing what is missing (for example "None \
posted.").

Answer with JSON matching the schema."""

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "primary_endpoint": {
            "type": "string",
            "description": "Registered primary endpoint and timepoint.",
        },
        "comparison": {
            "type": "string",
            "description": "Arm(s) vs. control that the label refers to.",
        },
        "label": {"type": "string", "enum": list(LABELS)},
        "basis": {"type": "string", "enum": list(BASES)},
        "source": {
            "type": "string",
            "description": "'AACT results', 'PMID:<number>', or 'why_stopped'.",
        },
        "main_publication_pmid": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "PMID of the main results publication, if any.",
        },
        "quote": {
            "type": "string",
            "description": "Verbatim passage from the packet supporting the label.",
        },
        "confidence": {"type": "string", "enum": list(CONFIDENCE)},
        "notes": {
            "type": "string",
            "description": "What made the call hard; dose-level results.",
        },
    },
    "required": ["primary_endpoint", "comparison", "label", "basis", "source",
                 "main_publication_pmid", "quote", "confidence", "notes"],
    "additionalProperties": False,
}

REQUIRED = tuple(LABEL_SCHEMA["required"])
REVIEW_COLUMNS = ("your_label", "agree", "issue", "review_notes")

_PUNCTUATION = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                              "–": "-", "—": "-", "−": "-", " ": " "})


def normalise_text(text: str) -> str:
    """Text for quote matching: Unicode-normalised, plain punctuation,
    markdown emphasis removed, whitespace collapsed."""
    text = unicodedata.normalize("NFKC", text).translate(_PUNCTUATION)
    text = text.replace("**", "").replace("*", "")
    return re.sub(r"\s+", " ", text).strip()


def quote_found(quote: str | None, packet_text: str) -> bool:
    """True if the quote appears in the packet, after normalisation."""
    if not quote or not quote.strip():
        return False
    return normalise_text(quote) in normalise_text(packet_text)


def validate_label(label: dict) -> list[str]:
    """Problems with a label's structure: missing fields or values outside the enums."""
    problems = [f"missing {field}" for field in REQUIRED if field not in label]
    for field, allowed in (("label", LABELS), ("basis", BASES), ("confidence", CONFIDENCE)):
        if field in label and label[field] not in allowed:
            problems.append(f"{field} not allowed: {label[field]!r}")
    return problems


def build_messages(packet_text: str) -> list[dict]:
    return [{"role": "user",
             "content": f"<evidence_packet>\n{packet_text}\n</evidence_packet>"}]


def extract_label(client, packet_text: str, model: str = DEFAULT_MODEL) -> dict:
    """Ask the model for a label and return it with run metadata.

    `client` is an anthropic.Anthropic instance (or anything with the same
    messages.create interface). Returns the parsed JSON answer plus model,
    stop_reason, input_tokens, and output_tokens.
    """
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        output_config={"format": {"type": "json_schema", "schema": LABEL_SCHEMA}},
        messages=build_messages(packet_text),
    )
    texts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
    if not texts:
        raise ValueError(f"No text in response (stop_reason={response.stop_reason})")
    try:
        answer = json.loads(texts[0])
    except json.JSONDecodeError as err:
        raise ValueError(f"Answer is not valid JSON (stop_reason="
                         f"{response.stop_reason}): {texts[0][:200]}") from err
    return {
        **answer,
        "model": model,
        "stop_reason": response.stop_reason,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def review_row(nct_id: str, label: dict, packet_text: str) -> dict:
    """One row of the review table: the model's answer, checks, and empty
    columns for the reviewer."""
    row = {"nct_id": nct_id}
    row.update({field: label.get(field) for field in REQUIRED})
    row["quote_found"] = quote_found(label.get("quote"), packet_text)
    row["problems"] = "; ".join(validate_label(label))
    row.update({col: "" for col in REVIEW_COLUMNS})
    return row
