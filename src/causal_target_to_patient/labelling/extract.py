"""LLM extraction of a trial's efficacy outcome from its evidence packet.

The model reads one packet (the rendered markdown, which is also what a
reviewer reads) and answers in JSON constrained to LABEL_SCHEMA (structured
outputs), so its output is always structured. The answer is then checked
against the packet:
- the quote must appear in the packet, as the main guard against made-up
  evidence;
- every number reported for the effect (estimate, confidence limits) must
  appear in the packet too.

The labelling rules are in SYSTEM_PROMPT and are documented for readers in
docs/labelling.md; keep the two in step.

Current models don't accept sampling settings such as temperature, so answers
can vary between runs. Each run's answers are saved, and repeat runs measure
how stable the labels are.
"""

import json
import math
import re
import unicodedata

DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 3000
MAX_QUOTE_CHARS = 400

LABELS = ("met", "failed", "mixed", "untested", "terminated_futility", "terminated_other",
          "no_results")
BASES = ("posted_analysis", "posted_values", "publication", "stop_reason", "none")
CONFIDENCE = ("high", "medium", "low")
DOSE_BASES = ("single_dose", "prespecified", "highest_dose_fallback")
DIRECTIONS = ("favours_investigational", "favours_control", "no_difference", "not_applicable")
POPULATIONS = ("mtx_naive", "mtx_ir", "csdmard_ir", "biologic_ir", "mixed", "other", "not_stated")

SYSTEM_PROMPT = f"""\
You label the efficacy outcome of one rheumatoid arthritis clinical trial from \
an evidence packet. Use only the packet. Do not use outside knowledge of the \
trial, the drug, or its later history. If the packet does not show the \
outcome, say so: answering no_results is better than guessing.

Work in this order.

1. Primary endpoint. Find the registered primary endpoint and its timepoint \
under "Planned primary outcomes". Set primary_is_efficacy to true if at least \
one registered primary endpoint measures efficacy (disease activity, response, \
remission, function, or structural damage). Set it to false if they all \
measure something else, such as pharmacokinetics, safety, adverse events, \
imaging of vessels, or a mechanistic biomarker; then label the main efficacy \
endpoint the evidence reports, if any, say so in notes, and set confidence to \
low.

2. Comparison. Identify the investigational therapeutic (role investigational \
or candidate under "Our mapping") and its control: placebo, or the active \
comparator. Background therapy given to every arm (such as methotrexate) is \
not the comparison.

3. Population. From the title, summary and eligibility criteria, classify who \
was enrolled:
- mtx_naive: no prior methotrexate (including early or DMARD-naive RA);
- mtx_ir: inadequate response or intolerance to methotrexate;
- csdmard_ir: inadequate response to conventional DMARDs, not specifically \
methotrexate;
- biologic_ir: inadequate response or intolerance to TNF inhibitors or other \
biologics;
- mixed: the trial explicitly enrolls more than one of these groups;
- other: a defined population that fits none of these;
- not_stated: the packet does not say.
Describe the population in a few words in population_description.

4. Evidence, in this order of preference. A statistical test, posted or \
published, always outranks values without one.
   a. a posted statistical analysis of the primary endpoint;
   b. the main results publication's test of the primary endpoint. The main \
results publication is the paper reporting this trial's primary result. Post \
hoc, pooled, subgroup, extension, biomarker, and genetic analyses are not the \
main results publication, even when they repeat numbers;
   c. primary-endpoint values for each arm without a test, posted or \
published;
   d. the reason for stopping.
Never judge statistical significance yourself from values without a test: \
that case is labelled untested.

5. Label, using these rules:
- met: the primary endpoint is statistically significant in favour of the \
investigational therapeutic over the control at the registered timepoint. For \
a non-inferiority design, non-inferiority is shown.
- mixed: co-primary endpoints disagree; or the therapeutic beats placebo but \
not the active comparator, when superiority to both was a primary objective; \
or, with several doses, the dose rule below gives mixed.
- failed: the trial reached its primary readout and the primary endpoint was \
not met.
- untested: the primary endpoint's values are reported for each arm, but no \
statistical test of the comparison is posted or published. Record the \
direction in effect_direction and the per-arm values in notes.
- terminated_futility: stopped early for lack of efficacy or futility, \
without a completed primary readout in the packet.
- terminated_other: stopped early for business, enrolment, safety or other \
reasons, without an efficacy readout in the packet.
- no_results: the packet holds no outcome evidence for the primary endpoint.

Dose rule, for trials testing several doses or regimens of the \
investigational therapeutic against one control:
- If the packet shows the trial's prespecified primary analysis (which \
comparison, pooled doses, trend test, or testing order was primary), label \
that analysis and set dose_basis to prespecified. If the trial reports that \
its primary endpoint was met under its prespecified testing strategy, that is \
met.
- Otherwise, label the highest registered dose and set dose_basis to \
highest_dose_fallback: met if the highest dose meets every co-primary \
endpoint, even if lower doses do not; mixed if the highest dose fails but \
another dose succeeds; failed if no dose succeeds. This is a convention for \
the label, not a claim that the highest dose is the best dose.
- Summarise the result for each dose in dose_results.
- Set lowest_effective_dose to the lowest dose that met the primary endpoint, \
or null if none did or the evidence does not say.
- Set dose_taken_forward to the dose the packet says was selected for later \
development or recommended, quoting the packet's wording; if the packet does \
not say, set it to null. Never infer it from outside knowledge.
lowest_effective_dose and dose_taken_forward are descriptive: they never \
decide the label.
For a single dose, set dose_basis to single_dose, leave dose_results empty, \
and set lowest_effective_dose and dose_taken_forward to null.

6. Effect. Set effect_direction from the evidence: favours_investigational, \
favours_control, or no_difference (values equal, or the reported test is \
clearly null); not_applicable when there is no readout (terminated or \
no_results). For the comparison the label refers to, report the primary \
endpoint's effect against control as the evidence states it: effect_measure \
(for example "difference in ACR20 response rate, percentage points" or \
"difference in mean change in DAS28-CRP"), effect_value, the confidence \
interval limits, and the p-value as written (for example "<0.001"). Copy the \
numbers exactly as they appear in the packet; do not compute or round them. \
If the evidence reports only each arm's value, give those in notes and leave \
effect_value null. Use null for anything the packet does not state.

7. Source and quote. source is "AACT results", "PMID:<number>", \
"why_stopped", or "none" when there is no outcome evidence (basis none). The \
quote is one contiguous passage of at most {MAX_QUOTE_CHARS} characters, copied \
character for character from the packet, that contains the decisive evidence \
for the label: the line or sentence with the p-value, the effect estimate or \
the per-arm results, or the stop reason. A sentence that only describes the \
design is not enough. Do not paraphrase, join separate passages, or add \
ellipses. For no_results, quote the line showing what is missing (for example \
"None posted.").

Answer with JSON matching the schema."""

_NUMBER_OR_NULL = {"anyOf": [{"type": "number"}, {"type": "null"}]}
_STRING_OR_NULL = {"anyOf": [{"type": "string"}, {"type": "null"}]}

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
        "primary_is_efficacy": {
            "type": "boolean",
            "description": "Whether any registered primary endpoint measures efficacy.",
        },
        "population": {"type": "string", "enum": list(POPULATIONS)},
        "population_description": {
            "type": "string",
            "description": "Who was enrolled, in a few words.",
        },
        "label": {"type": "string", "enum": list(LABELS)},
        "dose_results": {
            "type": "string",
            "description": "Result for each dose; empty for a single dose.",
        },
        "dose_basis": {"type": "string", "enum": list(DOSE_BASES)},
        "lowest_effective_dose": {
            **_STRING_OR_NULL,
            "description": "Lowest dose that met the primary endpoint, if any.",
        },
        "dose_taken_forward": {
            **_STRING_OR_NULL,
            "description": "Dose the packet says was selected or recommended; null if not stated.",
        },
        "effect_direction": {"type": "string", "enum": list(DIRECTIONS)},
        "effect_measure": {
            **_STRING_OR_NULL,
            "description": "What the effect estimate measures, with units.",
        },
        "effect_value": {**_NUMBER_OR_NULL, "description": "Estimate vs. control."},
        "effect_ci_lower": {**_NUMBER_OR_NULL, "description": "Lower confidence limit."},
        "effect_ci_upper": {**_NUMBER_OR_NULL, "description": "Upper confidence limit."},
        "p_value": {**_STRING_OR_NULL, "description": "P-value as written, e.g. '<0.001'."},
        "basis": {"type": "string", "enum": list(BASES)},
        "source": {
            "type": "string",
            "description": "'AACT results', 'PMID:<number>', 'why_stopped', or 'none'.",
        },
        "main_publication_pmid": {
            **_STRING_OR_NULL,
            "description": "PMID of the main results publication, if any.",
        },
        "quote": {
            "type": "string",
            "description": "Verbatim passage from the packet with the decisive evidence.",
        },
        "confidence": {"type": "string", "enum": list(CONFIDENCE)},
        "notes": {
            "type": "string",
            "description": "What made the call hard; per-arm values if no estimate.",
        },
    },
    "required": ["primary_endpoint", "primary_is_efficacy", "comparison", "population",
                 "population_description", "label", "dose_results", "dose_basis",
                 "lowest_effective_dose", "dose_taken_forward", "effect_direction",
                 "effect_measure", "effect_value",
                 "effect_ci_lower", "effect_ci_upper", "p_value", "basis", "source",
                 "main_publication_pmid", "quote", "confidence", "notes"],
    "additionalProperties": False,
}

REQUIRED = tuple(LABEL_SCHEMA["required"])
EFFECT_NUMBERS = ("effect_value", "effect_ci_lower", "effect_ci_upper")
REVIEW_COLUMNS = ("your_label", "your_population", "agree", "issue", "review_notes")

_PUNCTUATION = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                              "–": "-", "—": "-", "−": "-", " ": " "})
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


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


def number_found(value, packet_text: str) -> bool:
    """True if the number appears in the packet, ignoring sign and trailing
    zeros (26.7 matches "26.70"; -0.5 matches "−0.5" or "0.5")."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    target = abs(float(value))
    for token in _NUMBER.findall(normalise_text(packet_text)):
        if math.isclose(abs(float(token)), target, rel_tol=0, abs_tol=1e-9):
            return True
    return False


def validate_label(label: dict) -> list[str]:
    """Problems with a label's structure and consistency: missing fields,
    values outside the enums, an over-long quote, and source or basis that
    disagree with each other."""
    problems = [f"missing {field}" for field in REQUIRED if field not in label]
    for field, allowed in (("label", LABELS), ("basis", BASES), ("confidence", CONFIDENCE),
                           ("population", POPULATIONS), ("effect_direction", DIRECTIONS),
                           ("dose_basis", DOSE_BASES)):
        if field in label and label[field] not in allowed:
            problems.append(f"{field} not allowed: {label[field]!r}")
    quote = label.get("quote") or ""
    if len(quote) > MAX_QUOTE_CHARS:
        problems.append(f"quote longer than {MAX_QUOTE_CHARS} characters ({len(quote)})")
    basis, source = label.get("basis"), label.get("source")
    if basis == "none" and source != "none":
        problems.append(f"basis is none but source is {source!r}")
    if source == "none" and basis not in (None, "none"):
        problems.append(f"source is none but basis is {basis!r}")
    lab = label.get("label")
    if lab == "no_results" and basis not in (None, "none"):
        problems.append(f"label no_results but basis is {basis!r}")
    if lab in ("met", "failed", "mixed") and basis == "posted_values":
        problems.append(f"label {lab} rests on values without a test; use untested")
    if lab == "untested" and basis not in (None, "posted_values", "publication"):
        problems.append(f"label untested but basis is {basis!r}")
    direction = label.get("effect_direction")
    no_readout = ("no_results", "terminated_futility", "terminated_other")
    if lab in no_readout and direction not in (None, "not_applicable"):
        problems.append(f"label {lab} but effect_direction is {direction!r}")
    if lab == "untested" and direction == "not_applicable":
        problems.append("label untested needs an effect_direction")
    if label.get("dose_basis") == "single_dose" and (
            label.get("lowest_effective_dose") or label.get("dose_taken_forward")):
        problems.append("single_dose but a lowest effective or forward dose is given")
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
    columns for the reviewer.

    effect_numbers_found lists any reported effect number missing from the
    packet ("" when all are found or none were reported).
    """
    row = {"nct_id": nct_id}
    row.update({field: label.get(field) for field in REQUIRED})
    row["quote_found"] = quote_found(label.get("quote"), packet_text)
    missing = [f for f in EFFECT_NUMBERS
               if label.get(f) is not None and not number_found(label[f], packet_text)]
    row["effect_numbers_missing"] = "; ".join(missing)
    row["problems"] = "; ".join(validate_label(label))
    row.update({col: "" for col in REVIEW_COLUMNS})
    return row
