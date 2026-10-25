# Outcome labelling

Each RA trial that tests a contrast between targets gets an efficacy label. An LLM reads the trial's evidence packet and proposes the label; the label is checked against the packet and scored against hand labels. The rules below are the ones in the model's instructions (`SYSTEM_PROMPT` in `src/causal_target_to_patient/labelling/extract.py`); the two are kept in step.

## Evidence packet

One per trial, built by `scripts/build_evidence_packets.py` from AACT and PubMed:
- the registered design: title, phase, status, enrolment, dates, summary, **eligibility criteria** (capped at 4,000 characters), arms
- our intervention mapping: each intervention's role, molecule and targets
- the registered primary outcomes
- the posted primary results: values per arm and statistical analyses
- the linked publications, with abstracts

The labeller uses only the packet, not outside knowledge of the trial or the drug.

## Labels

| Label | Meaning |
|---|---|
| `met` | The primary endpoint is significant in favour of the investigational therapeutic at the registered timepoint (or non-inferiority is shown, for a non-inferiority design). |
| `mixed` | Co-primary endpoints disagree; or the therapeutic beats placebo but not an active comparator it was also meant to beat; or the dose rule gives mixed. |
| `failed` | The primary readout was reached and the endpoint was not met. |
| `untested` | Per-arm values for the primary endpoint are reported, but no statistical test of the comparison is posted or published. The direction is recorded; the model never judges significance itself. |
| `terminated_futility` | Stopped early for lack of efficacy, with no completed primary readout. |
| `terminated_other` | Stopped early for other reasons, with no efficacy readout. |
| `no_results` | No outcome evidence for the primary endpoint in the packet. |

**Evidence, in order of preference.** A statistical test, posted or published, always outranks values without one:
1. a posted statistical analysis of the primary endpoint
2. the main results publication's test (not post hoc, pooled, subgroup, extension or biomarker papers)
3. per-arm values without a test, posted or published, giving `untested`
4. the reason for stopping

**Trials whose primary endpoint isn't efficacy** (pharmacokinetics, safety, vascular imaging, mechanistic biomarkers) are flagged with `primary_is_efficacy: false`. They are excluded from the efficacy benchmark through `curated/trials/excluded_trials.csv`; the flag is how new ones are found.

**Planned:** for `untested` trials where AACT posts each arm's size and dispersion or counts, the test will be computed in code, with the method recorded, rather than by the model.

**Dose rule.** For trials testing several doses or regimens against one control:
1. **The trial's own prespecified primary analysis, when the packet shows it** (which comparison, pooled doses, trend test or testing order was primary). If the trial reports its primary endpoint met under its prespecified testing strategy, the label is `met`. This follows how trials are judged under ICH E9 and the FDA's guidance on multiple endpoints: by the prespecified primary hypothesis, with multiplicity controlled. "Any dose succeeded" is not used, because unadjusted comparisons across doses inflate false positives.
2. **Otherwise, the highest registered dose**, as a labelling convention:
   - `met` if the highest dose meets every co-primary endpoint, even if lower doses don't
   - `mixed` if the highest dose fails but another dose succeeds
   - `failed` if no dose succeeds

   Testing doses in a fixed order from the highest is a common way to control multiplicity, and the highest dose is the closest to maximal target engagement, which is what the benchmark asks about. It is **not** a claim that the highest dose is the best dose: the clinically preferred dose is often the lowest effective one.

Which rule was used is recorded in `dose_basis` (`prespecified`, `highest_dose_fallback`, or `single_dose`). Every multi-dose label also records:
- `dose_results`: the result for each dose
- `lowest_effective_dose`: the lowest dose that met the primary endpoint
- `dose_taken_forward`: the dose the packet says was selected for later development or recommended. It is null when the packet doesn't say, which makes "unclear" explicit. It is never inferred from outside knowledge.

Both are **descriptive and never decide the label**:
- The dose taken forward is chosen after seeing this trial's results. Labelling by it would favour whichever dose happened to look best, and would bring the drug's later history into the label.
- The lowest effective dose often rests on unadjusted per-dose comparisons, so it can be a chance finding. Neither field counts towards agreement scoring.

## Fields

| Field | Content |
|---|---|
| `primary_endpoint`, `primary_is_efficacy`, `comparison` | The registered primary endpoint and timepoint; whether any primary endpoint measures efficacy; the arm(s) and control the label refers to |
| `population` | Who was enrolled: `mtx_naive`, `mtx_ir`, `csdmard_ir`, `biologic_ir`, `mixed`, `other`, `not_stated`; with `population_description` |
| `label` | As above |
| `dose_basis`, `dose_results`, `lowest_effective_dose`, `dose_taken_forward` | Which dose rule decided the label, each dose's result, the lowest dose that worked, and the dose the packet says was taken forward (null if not stated) |
| `effect_direction` | `favours_investigational`, `favours_control`, `no_difference`, or `not_applicable` (no readout) |
| `effect_measure`, `effect_value`, `effect_ci_lower`, `effect_ci_upper`, `p_value` | The primary endpoint's effect against control, copied exactly as the evidence states it; null where not stated |
| `basis`, `source` | Which evidence decided the label: `AACT results`, `PMID:<n>`, `why_stopped`, or `none` |
| `main_publication_pmid` | The main results publication, if any |
| `quote` | One verbatim passage of at most 400 characters containing the decisive evidence: the p-value, the effect estimate or per-arm results, or the stop reason |
| `confidence`, `notes` | How sure, and what made the call hard |

## Checks on every label

- **Quote found:** the quote must appear in the packet (after normalising whitespace and punctuation).
- **Effect numbers found:** the effect estimate and confidence limits must each appear in the packet.
- **Consistency:**
  - values inside the allowed sets; quote within 400 characters
  - `basis` and `source` both `none` or neither; `no_results` only with basis `none`
  - `met`, `failed` or `mixed` never rest on values without a test (`posted_values`)
  - `untested` rests on posted or published values and has a direction
  - no readout (terminated, `no_results`) means direction `not_applicable`
  - a single-dose trial has no lowest effective or forward dose

## Known limitations

- **Dose limits.** The label can't tell whether the doses tested engaged the target enough. If toxicity capped the dose, a failure says more about the molecule than about the target. Only exposure or target-engagement data could separate the two, and the packets don't hold them.
- **Highest-dose fallback.** When the trial's plan isn't in the packet, the label rests on a convention (`dose_basis: highest_dose_fallback`). Results are reported with and without these trials.

## Development and test sets

Drawn by `labelling/sampling.py` from the labelling trials that test a contrast, stratified by evidence tier, with a fixed seed. The two sets never overlap.

- **Development set** (`scripts/select_dev_set.py`): 20 trials, 6 with a posted primary analysis, 5 with posted primary results only, 5 with a publication only, 2 with a stop reason only and 2 with no evidence. The five pilot trials are left out. Labelled by hand alongside the model's labels, and used to refine the rules. Once drawn, the set is fixed: the script won't replace it without `--force`.
- **Test set** (later): 40–80 trials, labelled by hand **before** seeing the model's labels, and used once to report accuracy and repeat-run stability.

**Files.** Each run's outputs are named after its trial list:
- `labels/dev_set_YYYYMMDD_labels.csv`: all the model's answers
- `labels/dev_set_YYYYMMDD_review.csv`: the review sheet, in three blocks:
  1. context: trial, title, investigational drug(s), evidence tier, and the path of the packet to read
  2. the columns to fill in: `your_label`, `your_population`, `agree`, `issue`, `review_notes`
  3. the model's answer and the automatic checks
- `labels/<set>_blind.csv` (`--blind`): blocks 1 and 2 only, for the test set, which is labelled by hand before the model's answers are seen

Existing run files are never overwritten without `--force`. `scripts/label_trials.py --reuse-saved` rebuilds them from the saved per-trial answers without calling the model.

**Checking hand labels.** `scripts/check_review.py <sheet>` lists any entry outside the allowed values, so typos are fixed before scoring rather than counted as disagreements. `issue` takes one or more codes, separated by semicolons: `wrong_label`, `wrong_population`, `wrong_comparison`, `wrong_endpoint`, `wrong_effect`, `bad_quote`, `missed_evidence`, `efficacy_flag_wrong`, `other`.

**Bar for moving on from the development set:** the model's label agrees with the hand label on at least 80% of the dev-set trials whose primary endpoint is efficacy. For the others, the check is that the model flags them (`primary_is_efficacy: false`).
