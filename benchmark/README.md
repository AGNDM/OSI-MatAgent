# Instrument benchmarks: dataset construction and evaluation

This directory contains two synthetic instrument-operation benchmarks and the
local evaluation code used in this project:

| Track | Dataset | What the model does | Main outcome |
| --- | --- | --- | --- |
| Static MCQA | 388 questions with 2–4 offered answers | Select one supplied answer or complete plan | Answer accuracy on 114 held-out questions |
| Interactive sequential | 120 cases across 14 workflows | Choose successive actions and receive native simulator feedback | Autonomous outcome completion and, separately, teacher-forced proposal accuracy |

The main workflow experiment crosses an **expert LLM** with a **decision model
or method** inside the actual application graph. The MCQA experiment measures
isolated answer selection and confidence. They use related instrument scenarios
but have different inputs, interaction protocols and scores.

This README describes the implemented construction and evaluation procedures.
[PLAN.md](PLAN.md) also contains research targets; a planned method, declared
matrix cell or submitted job is not evidence of a completed experiment.

The next planned full workflow study, **D**, prioritizes **22 cells × 120 cases
in `teacher_forced` mode**, seed zero; it has not started and awaits the corrected
pilot gate. Original-proposal accuracy and correction rate are its primary
measures. Autonomous B and the completed 64-episode teacher-forced C pilot are
development studies. Completion after reference corrections is assisted
completion, separate from autonomous performance.

## 1. Source, authorship and frozen release

ChemLab-Assets supplies instrument contracts and native Python simulators. This
project authored the scenarios, questions, candidate answers and grading
criteria; the questions are not an existing ChemLab question bank or records of
real laboratory incidents.

Scenarios and wording templates were developed with coding-agent assistance.
The generator instantiates them deterministically in Python and makes no LLM or
hosted API calls. Reference labels are established by native execution and
explicit authored rules, rather than by an LLM judge.

The portable release is version **1.1.1**, pinned to ChemLab-Assets 1.3.0:

| Component | Pinned revision |
| --- | --- |
| Simulator code | `123cf78593469871e41b8dbd63a11dfb7c5ca251` |
| Instrument data | `4ca4a42dab2ca0a884e449d202e9ad96625d5648` |

The eight benchmark instruments are the Binder FD 56 oven, Eppendorf CellXpert
C170i incubator, Esco fume hood, IKA RCT digital hotplate, Nabertherm L9 furnace,
New Era NE-1000 syringe pump, Peak Precision Hydrogen generator and Vaisala
DMT152 dew-point transmitter. They span four instrument domains.

The [release manifest](instrument/manifest.json) records packaged files and
checksums. The [original MCQA manifest](instrument/mcqa_388/source-release-manifest.json)
preserves the generation configuration, source hashes, counts and audit.

## 2. How the 388 MCQA questions were made

### Scenario construction

The original generator and scenario definitions are included in the portable
runtime:

- [scenarios.py](instrument/sequential_120/runtime/vendor/instrument_bench/scenarios.py):
  authored starting states, goals, faults, candidate answers and plans.
- [generator.py](instrument/sequential_120/runtime/vendor/instrument_bench/generator.py):
  simulator replay, answer labeling, grouping, presentation and release audit.
- [schema.py](instrument/sequential_120/runtime/vendor/instrument_bench/schema.py):
  scenario structure and frozen generation settings.

The construction procedure is:

1. **Define a scenario family.** Specify the instrument, public observations,
   starting fixture, operational goal, candidate choices, any scheduled
   interruption and the criterion for a correct outcome.
2. **Instantiate parameter variants.** The release uses master seed 42, eight
   parameter/simulation seeds per family and two presentation variants per
   retained base case. Selected temperatures, rates, durations and other
   authored parameters vary; this is not exhaustive sampling of simulator states.
3. **Run each candidate independently.** Recreate the same initial fixture in a
   fresh simulator for each candidate, execute its probe or plan, and save the
   returned evidence and final-state checks. Choices cannot contaminate one
   another's starting state.
4. **Check answerability and distractors.** Every retained question must have an
   accepted candidate and an incorrect alternative. Distractors represent
   authored mistakes such as missing prerequisites, wrong operation order,
   insufficient waiting, unsupported conclusions or unjustified escalation.
5. **Create presentation variants.** Vary the authored wording frame and
   deterministically shuffle choice order using the task ID. Assign A/B/C/D
   after shuffling. The reference answer follows that permutation.
6. **Deduplicate and group related inputs.** Collapse duplicate physical cases
   and join semantically equivalent public decision inputs into common groups.
   Keep both presentation variants and related cases in the same split.
7. **Freeze public questions and evaluator answers separately.** Store source
   identities, seeds, counts and hashes with the release.

The resulting corpus contains **388 questions, 194 base cases, 112 semantic
scenario groups and 42 authored scenario families**. Thus 388 questions do not
mean 388 independent physical scenarios. Families include healthy controls,
setup conditions and faults; they are not 42 distinct hardware failures.
The frozen corpus includes 146 healthy-control questions.

### How the reference answer is determined

The verification depends on the authored task type:

| Task type | Reference criterion |
| --- | --- |
| Command or outcome prediction | Execute the proposed probe and inspect its native status, declared error or operational-goal outcome. |
| Diagnosis | Apply explicit authored predicates to observable simulator state under the question's stated candidate hypotheses. |
| Evidence sufficiency | Construct paired fixtures with identical initial public observations but different probe outcomes; this supports an insufficient-information answer. |
| Diagnostic test selection | Execute the offered test in paired fixtures and check whether its returned evidence distinguishes those hypotheses. |
| Procedure or recovery selection | Execute the candidate plan and check its final operational predicate, or a justified maintenance escalation when required. |
| Interruption response | Apply scheduled disturbances during execution and check the same declared completion or escalation criteria. |

For example, an oven recording plan must actually produce the required recording
state or row; accepting a recording command alone is insufficient. Correctness
follows the declared final predicate, not a blanket requirement that every
intermediate command succeed.

Diagnosis labels depend on the authored observable rules and restricted
hypotheses. Paired diagnostic tests establish discrimination between those
hypotheses, not globally optimal diagnosis. The precise claim is **native
execution plus explicit authored evaluation rules**.

All 388 released questions have exactly one accepted answer. The
`correct_option_ids` list format supports multiple accepted answers in a future
release. The compact [answer file](instrument/mcqa_388/answers.jsonl) omits the
private fixtures and full native traces needed during generation.

### What is public to the model

[Prompt export](instrument/eval/prompts.py) uses the
`public-menu-union-v2` projection: question, operational objective, public
observations/history, relevant manual assumptions, offered answers and public
contract descriptions. Documentation is collected symmetrically from **all
offered candidates**, not from the correct answer.

Answer keys, hidden fixture state, fault-injection controls and evaluator
metadata are withheld. Raw question records also contain split/group bookkeeping;
give the model the exported prompt, retaining its opaque task ID only for
matching predictions.

Some MCQA choices contain an entire action sequence, and the legacy record
format can call these `sequential`. They remain **static plan-selection
questions**, separate from the 120 interactive cases.

## 3. Categories, answer counts and splits

### Legacy L1–L5 categories

These labels describe the kind of decision being tested. They are **not an
established increasing difficulty ranking**.

| Label | Task category | All questions | Held out |
| --- | --- | ---: | ---: |
| L1 | Single-action checks: feasibility, arguments and immediate errors | 158 | 44 |
| L2 | Direct diagnosis and simple recovery | 50 | 20 |
| L3 | Evidence assessment, testing and escalation | 88 | 18 |
| L4 | Procedure and service plan selection | 62 | 22 |
| L5 | Interruption response plan selection | 30 | 10 |

The frozen field is named `difficulty`; preserve its values when joining
results. L4/L5 select supplied plans and do not demonstrate autonomous planning.

T1/T2/T3 are a separate construction taxonomy: visible-state checks,
hidden-state/evidence tasks and goal-oriented tasks, respectively. Their full
counts are 208/96/84. Failure mechanism, instrument, condition, workflow,
`decision_kind` and scenario `group_id` are additional, distinct labels.
A single fault mechanism can appear in several L categories.

### Number of offered answers

| Offered choices | All questions | Held out | Uniform guessing expectation |
| --- | ---: | ---: | ---: |
| 2 | 182 | 64 | 50% |
| 3 | 136 | 26 | 33.3% |
| 4 | 70 | 24 | 25% |

These are menu sizes, not semantic classes. Their task mixtures differ: the
held-out two-choice subset is L1/L2, while 20 of the 24 four-choice questions are
L4. Comparing these subgroup accuracies does not isolate the effect of adding
answer choices.

### Training, validation and held-out evaluation

| Split | Questions | Use |
| --- | ---: | --- |
| `train` | 232 | Fit frozen-feature classifiers, kNN memory and label priors |
| `validation` | 42 | Fit the cosine temperature; select validation-only reporting thresholds where used |
| `test` | 60 | Ordinary held-out scenario groups |
| `test_template` | 36 | Reserved top-level wording frames |
| `test_fault` | 8 | Reserved `sensor_failure` fault family |
| `test_instrument` | 10 | Reserved incubator instrument |

The primary comparison in the published JSON files is the union of the four
`test*` splits: **114 questions**. All-388 scores include training and validation
diagnostics and are labeled separately.

Split assignment preserves complete semantic groups. The audit checks
public/private ID agreement, valid answer IDs, duplicate decision inputs,
cross-split group leakage and separation of held-out template IDs. The wording
protocol uses two ordinary frames and two reserved frames; it is a limited
template-shift test, not broad language-robustness evidence.

**No pretrained neural model was fine-tuned for this benchmark.** Qwen,
OpenJev, APUS and MiniLM weights remain frozen. Small classifiers and calibration
parameters are fitted separately using the designated data, which is why an
evaluation split is needed.

## 4. How static MCQA evaluation works

The model receives one public question with its offered answers and returns a
choice. The evaluator compares the selected option ID with the frozen answer
key. It does not let the model operate the simulator during this static track;
native simulation established the reference answers during construction.

A minimal prediction is:

```json
{"task_id": "opaque-question-id", "selected_option_id": "B"}
```

The [minimal scorer](instrument/eval/mcqa.py) computes:

```text
accuracy = correct answers / every question in the selected cohort
response coverage = valid offered choices with status ok / cohort size
```

Missing predictions, non-menu answers and explicit non-`ok` statuses count as
incorrect. An answer that fails the model-output schema becomes an invalid-result
record; structurally malformed prediction JSONL, duplicate task IDs and unknown
IDs are rejected rather than silently repaired. For saved comparison records,
nested backend answers are normalized before scoring. The submitted selected
answer is scored; the evaluator does not replace it with a probability argmax.

Attempt coverage is separate from response coverage: an explicit context-limit
or model-error record is an attempted question but not a valid answer. Every
subgroup retains missing/invalid/error questions in its denominator. A group
outside a reference method's scope has `n: 0` and `accuracy: null`.

The T1 rule reference is intentionally restricted to **208 full-corpus questions
and 64 held-out questions**. Its scoped score must not be presented as accuracy
over all 388 or all 114 questions.

The [T1 rules](comparison/t1_rules.py) use public observations, schemas,
preconditions and stated diagnostic hypotheses, without reading answer keys or
invoking the native runtime. This differs from the privileged sequential
reference policy described below.

### Decision methods in the current local comparison

The result snapshot contains **five pretrained checkpoints and 34 configurations**:

| Configuration family | Count | Mechanism |
| --- | ---: | --- |
| Qwen2.5-3B-Instruct, HF | 7 | Choice, token logits, verbal confidence, cited confidence, three sampling seeds |
| Qwen3-4B-Instruct-2507, HF | 7 | Same methods |
| Qwen2.5-3B-Instruct, Ollama | 6 | Choice, verbal confidence, cited confidence, three sampling seeds; no complete-menu logits |
| Native decision heads | 3 | OpenJev-8B; APUS-Reflex 4B low/high |
| Frozen-feature methods | 4 | MiniLM cosine, logistic classifier and kNN; Qwen hidden-feature logistic classifier |
| Expert then decision model | 4 | Qwen2.5 expert followed by Qwen2.5, OpenJev, APUS-low or APUS-high |
| References | 3 | Uniform random, training-label prior and scoped T1 rules |

The exact pretrained identities are `Qwen/Qwen2.5-3B-Instruct`,
`Qwen/Qwen3-4B-Instruct-2507`, `alanhuangya/OpenJev-8B`,
`apus-ailab/APUS-Reflex` subfolder `4B-5949`, and
`sentence-transformers/all-MiniLM-L6-v2`. Runtimes, seeds and effort depths are
configurations, not additional models.

[Backend implementation](comparison/backends.py) and
[comparison guide](comparison/README.md) define the readouts:

- **Direct choice:** greedy generation must parse as exactly one offered label
  after whitespace stripping. It supplies no confidence estimate.
- **Token probabilities:** require distinct single-token option labels, normalize
  the next-token logits over those labels only, and choose the highest. These
  probabilities are conditional on the offered label set.
- **Verbal confidence:** parse an offered answer and finite numeric confidence
  in `[0,1]`. The cited variant also parses explicit cause/evidence identifiers;
  their correctness is evaluated separately from schema validity.
- **Sampling:** draw ten independently seeded full answers at temperature 0.8,
  use answer frequencies as probabilities, and select the vote winner.
  Three seeds are separate runs. Ties follow menu order. Under this strict
  comparison protocol, any malformed draw invalidates the whole decision;
  invalid draws are not dropped to improve the vote.
- **OpenJev/APUS:** use their released learned decision heads. APUS low/high use
  trained depths 16/32 of the same checkpoint. Their probabilities are not
  assumed calibrated on this instrument dataset.
- **Frozen features:** encode context and candidates without updating encoder
  weights. Fit logistic classifiers on the 232 training questions; kNN uses
  training examples. Fit the MiniLM cosine softmax temperature on the 42
  validation questions. Complete text contributes through declared chunks.
  Probe/kNN have no extra post-fit temperature calibration; validation-only
  selective thresholds are a separate reporting analysis. No sequential case
  labels are used for fitting these methods.

The four earlier expert/chooser configurations are a comparison-owned adapter,
separate from the actual application graph. Their expert must return
`min(4, offered choices)` distinct candidates with diagnostic/rationale fields.
Because MCQA menus have only 2–4 choices, valid expert output includes the whole
static menu. It becomes a genuine subset for larger sequential menus. Its gate
checks schema and public membership, not reference correctness. Conditional
expert/chooser confidence is not treated as a full-menu probability vector.

No hosted MiniMax/Jev calls are included in these 34 local configurations.
Stronger Qwen14B/DeepSeek graph experiments and the colleague's hosted API
results are separate studies.

### Confidence, errors and resource metrics

The [study reporter](comparison/study_report.py) and
[metric implementation](comparison/study_metrics.py) distinguish:

| Metric | Target and availability |
| --- | --- |
| Accuracy | Correct choice over the complete declared cohort |
| Valid-response and attempt coverage | Separately, valid outputs / cohort and recorded attempts / cohort |
| ECE | Ten fixed-width bins of available selected-answer confidences versus correctness; confidence coverage is reported |
| Selected-answer Brier | Squared error of a supplied confidence against selected-answer correctness |
| Multiclass Brier | Sum of squared errors of a genuine full-menu probability vector against exactly one accepted reference label; omit multi-answer rows |
| Selective accuracy and AUROC | Confidence-based correctness ranking, only where defined and supported |
| Latency and tokens | Recorded decision timing and known token usage, with coverage and hardware identified |
| Monetary compute cost | Estimate from retained Slurm allocation hours and installed site tariffs; actual invoiced cost remains unavailable |

#### Which metrics were actually recorded?

The table above defines evaluator metrics; it does not mean every metric is
available for every configuration or included in the three exported JSON files.
For the static snapshot documented in Section 7:

| Metric | Recorded result availability | In the three exported MCQA JSON files? |
| --- | --- | --- |
| Accuracy and attempt/valid-response coverage | All 34 configurations, with the T1 reference's restricted cohort explicit | Yes |
| ECE | 31 configurations; the three plain-choice methods supply no confidence | Yes, with confidence counts; unsupported values are null |
| Multiclass Brier | 21 configurations with genuine probability vectors | Yes, with distribution counts; unsupported values are null |
| Selected-answer binary Brier | 31 configurations with valid confidence values | Yes, with confidence counts and coverage |
| Selective accuracy and correctness AUROC | Curves for 31 configurations; AUROC for 30, since it requires both correct and incorrect confidence-bearing answers | Yes, with eligible counts and unavailable reasons |
| Latency and input/output tokens | Measured timing for 32 configurations; input/output counts include known-usage coverage | Yes; uniform/majority reference timing placeholders are excluded |
| Monetary compute cost | Estimated for all 34 configurations from 272 retained MCQA allocation records and installed SCITAS tariffs | Yes for full MCQA allocations; held-out/subgroup allocations and invoice amounts remain null |

The metric augmentation uses the same authenticated predictions, not the older
partial research report. Missing confidence/usage stays uncovered rather than
being interpreted as zero. The uniform and majority references have explicit
zero-latency placeholders; these are excluded from measured timing.

Direct-backend wall timing normally excludes model loading, but the first
expert-to-native-chooser pipeline call can include lazy chooser initialization.
Feature timings include content-cache behavior and CPU decision work. Sampling
token totals already include all draws; expert/chooser totals already include
both stages. Nested usage is not added a second time. These wall timings are
not active GPU time or total experiment runtime.

Allocation estimates use terminal Slurm elapsed time multiplied by allocated
GPU units and the installed site partition tariff. A MIG unit is a slice, not
a full-GPU equivalent. The archived tariff configuration specifies CHF per
allocated unit-hour: H100 0.5174, L40S 0.2141, MIG12GB 0.0740 and MIG24GB 0.1725.
Its source path, hash and rate-version header are retained in each JSON.

The retained MCQA allocations total approximately **CHF 2.19**. This includes
startup/loading and allocation overhead for the retained MCQA shard jobs,
but excludes environment setup, downloads, feature fitting, discarded attempts,
and sequential/graph runs. It is an estimate, not an invoice or total project
cost. Held-out questions and subgroups share allocations with other questions;
their GPU hours and monetary cost remain null rather than receive invented
latency-based prorations. Actual active GPU time and invoiced cost are unknown.

Unavailable confidence/distribution metrics stay `null`; a verbal confidence
scalar is not expanded into an invented answer distribution. Malformed output,
context overflow, unsupported menu size, runtime errors and budget exhaustion
retain separate diagnostics. Evidence and choices are not silently truncated to
make a model fit. Native model context/menu limits therefore affect coverage.

The larger reporter supports grouped uncertainty and paired analyses. The three
JSON exports linked below are **descriptive snapshots** and do not establish
statistical significance. Citation agreement measures agreement with declared
cause/evidence identifiers, not proof of causal reasoning. Wall latency is not
active GPU time, and zero hosted API usage does not mean computation is free.

## 5. How the 120 sequential cases were made and scored

The interactive corpus contains **120 parameterized cases across 14 workflow
templates and eight instruments**, including **27 matched healthy/failure pairs**.
It combines 72 cases from six workflows and 48 from eight additional workflows.
There are healthy controls, visible setup problems and persistent or scheduled
faults. These are not 120 distinct failure mechanisms.

The authored [case builders](instrument/sequential_120/runtime/vendor/stepwise_expanded/tasks.py)
join the [original variants](instrument/sequential_120/runtime/vendor/stepwise_expanded/variants.py)
and [additional workflows](instrument/sequential_120/runtime/vendor/stepwise_expanded/new_workflows.py).
Each case specifies an initial fixture, objective, checkpoints, acceptance rules,
prerequisites, fault schedule and budget.

Public [cases](instrument/sequential_120/cases.jsonl) and
[manuals](instrument/sequential_120/public-contracts.json) are separate from
[private evaluator specifications](instrument/sequential_120/evaluator/case-specs.jsonl).
The latter are never model inputs. The sequential corpus has **no assigned
train/test split** and is not a held-out generalization result. L1–L5 labels
belong to the static corpus, not these sequential cases.

### Menus and native feedback

A shared finite bank supplies **1,954 schema-valid argument presets for 146
public commands**. It combines authored public argument values with public
schema representatives. Presets are shared across cases; originating case,
correct checkpoint and reference order are omitted. This finite bank does not
cover every possible numeric/string argument.

The first choice selects among eight instruments. Later choices expose public
commands, observations/manual reads, waiting, switching machines, completion
and escalation. Commands remain available even when their operating
prerequisites are unmet. Machine/command/argument label selections may require
multiple model calls but form one scored native action.

State persists across actions. Waiting advances simulated time. Readings retain
their sampling times until an actual observation-producing action/read updates
them; a model cannot obtain a fresh measurement merely by reusing a cached value.

The [native grader](instrument/sequential_120/runtime/vendor/stepwise_bench/engine.py)
checks a declared partial-order checkpoint curriculum using prerequisites and
state conditions, including deterministic shadow replays of proposals. It
allows specified alternative actions, eligible orders and parameters; it is
not matching one fixed action string, an optimal planner, or proof that all
physically valid alternative procedures earn credit.

### Autonomous and teacher-forced modes

| Mode | What executes | What its score means |
| --- | --- | --- |
| `free_run` | The model's proposed action, without evaluator correction | Autonomous verified outcome completion |
| `teacher_forced` | A correct proposal, or an admissible reference replacement after an incorrect proposal | Original-proposal correctness, corrections and assisted completion |

In both modes the original proposal is graded. In teacher-forced mode, the next
policy call receives the original proposal, executed action, correction flag
and actual native response. **Corrections do not earn proposal credit or update
model weights.** They keep the trajectory progressing.

For a teacher-forced comparison, report correct original proposals divided by
graded original proposals, and reference interventions divided by the same
native-decision denominator. Retain both counts and coverage. The scorer's
`corrected_trajectory_outcome_met` field describes assisted completion after
reference corrections; it must not be reported as autonomous task success.

A successful outcome requires the declared goal conditions, accepted completion
evidence and appropriate termination. Cases requiring maintenance instead need
a justified escalation with the required evidence. Merely reaching a target
state or saying “done” is insufficient if required checkpoints are unmet.

The [sequential evaluator](instrument/eval/sequential.py) reports:

- For a full-cohort report, outcome completion over **all 120 declared cases per
  mode**, retaining missing and infrastructure-error episodes in the denominator.
- Original-proposal accuracy, control-action accuracy excluding auxiliary
  queries, query admissibility, checkpoint/stage performance and corrections.
- Separate workflow, instrument and condition breakdowns, with observed
  decisions and episode coverage explicit.

Step accuracy uses observed records from completed valid episodes; it does not
invent decisions for unvisited steps. Partial aborted traces remain diagnostics
rather than silently becoming completed episodes. A completed-episode-only rate
has a different denominator and must accompany coverage. Offline scoring
aggregates trusted native grades; an arbitrary saved success flag is not a
substitute for native replay.

A privileged reference policy is an evaluator check, not a learned-agent
baseline. Its successful replay demonstrates execution/rubric coverage, not
model intelligence.

## 6. End-to-end expert × decision-model evaluation

[graph/](graph/README.md) bridges local experts and decision methods to the actual
`app.graph.build_graph` workflow:

```text
diagnosis → proposed operations → operation selection → procedure
          → deterministic gate → native execution and feedback → verification
```

The adapter pauses after the typed gate, lets the native evaluator execute the
action, checks the matching feedback and resumes the same graph. It evaluates
the same 120 cases separately in `free_run` and `teacher_forced`.

Keep the cases, public evidence, graph, repair policy and budgets fixed when
comparing expert/chooser cells. Report schema/gate acceptance, chooser
reachability, native execution coverage, failure causes and usage alongside
outcome success. A specialist failure before chooser invocation does not
establish a ranking between choosers.

The application's configured call counter is not a universal count of all
backend calls: router and specialist calls increment it, while selections by
some dedicated decision backends do not. The separate ledger records actual
backend attempts. Report observed calls and tokens as well as configured limits.

The installed [graph package](graph/README.md) retains the **study v2** default,
`actual-repository-langgraph-native-bridge-v1`. Its optional protocol-v5
transports use public menu IDs for expert proposals and bounded current-decision
views for choosers. The selected action is bound exactly to a public preset;
native execution and grading are unchanged. Older history is explicitly
summarized for the chooser while full public evidence remains in the trace and
specialist input. Existing frozen runs remain separate. Matrix examples declare
experiments, not results; a working tokenizer/schema boundary does not establish
successful model control. Do not pool graph protocols or equate the earlier
comparison-owned adapter with this workflow.

Internal graph routes have no authoritative native route-answer oracle, so
route confidence alone does not justify route ECE/Brier claims. Context,
schema, deadline and bridge failures are explicit episode failures; teacher
forcing does not silently repair those infrastructure boundaries.

## 7. Result files and how to read them

The [workflow combination inventory](../results/120/workflow-combinations.json)
records 36 local expert/chooser pairs with separate `free_run` and
`teacher_forced` counts. Eighteen pairs belong to the original study-v2
protocol; eighteen belong to the stronger-expert v4 follow-up. The latter is a
saved-progress snapshot, not a final report. The export preserves source paths,
hashes, native completion and abort counts, and identifies the two unlaunched
strong-expert self-chooser controls. Do not pool protocols or interpret complete
attempt coverage as successful workflow execution.

The [protocol-v5 repair checks](../results/120/workflow-repair-validation.json)
record the bounded-input tokenizer replay, affected code checks, actual LLM and
OpenJev routing/operation readouts, and separate queued workflow pilots. These
interface checks are not new 120-case performance results.

The [cancellation audit](../results/120/workflow-cancellation.json) records the
35 remaining old v4 workers stopped at the user's request, the preserved saved
attempts, and the first replacement pilot launch. The
[full-case pilot evidence](../results/120/workflow-pilot-results.json) keeps
initial and corrected v5 snapshots separate, with raw episode/trace hashes,
actual model selections, deterministic single-option shortcuts, native actions,
errors and evaluator outcomes. These runs use one selected development case per
combination; they do not estimate accuracy across the 120 workflows.

The [stopped protocol-v7 A results](../results/120/workflow-deadline-v7-stopped-results.json)
preserve **75 closed records: 19 completed native episodes, 56 aborts and zero
verified goals**. The declared study contained 22 cells × 120 cases, seed zero,
autonomous `free_run` only. Another 2,565 declared episodes have no closed record;
all 2,640 scheduler slots are terminal. Completed native episodes can fail their
goals, and missing records do not become scored attempts. This stopped,
incomplete study is separate from development smokes and earlier protocols.

The autonomous B development pilot uses grouped readable public menus
(`menu-ids-v3`) and the same lean chooser view (`decision-view-v3`). Group fields
plus exact choice arguments reconstruct every offered action; explicit named
references remove identical observations and repeated receipts. Input parameter
schemas are omitted only from specialist model text, where all offered preset
arguments are supplied; source schemas, native validation and grading remain
unchanged. Previous verifier feedback is marked historical, and the verifier is
told that `supported` makes a FINISH claim for the entire goal. Its model verdict
does not itself award success.

The declared DeepSeek context is restored to 163,840 tokens; Qwen retains
32,768. The common limits remain 300 seconds per episode, 128 configured calls,
120 seconds per call, 2,048 graph steps and 2,048 specialist output tokens. The
fixed development qualification declares eight priority cells × eight public
instrument cases (64 episodes), with its outputs excluded from production.
B's saved autonomous episodes do not establish full-120 performance. Later
interface changes require final checks and a fresh source freeze; the newer
native `decision-view-v4` layout is a separate protocol, not a relabeling of B.

The [teacher-forced C pilot export](../results/120/workflow-teacher-forced-pilot-results.json)
contains **64 closed attempts: 41 reached the goal after reference corrections
and 23 aborted**. Its logged native decisions contain **228 correct original
proposals out of 535 (42.6%)**, with **307 reference corrections**. These totals
include saved decisions from closed completed and aborted attempts; the export
keeps completed-episode-only native aggregates separately. These are development
pilot results on eight predeclared public cases per priority cell, not full-120
or autonomous performance. C's outcomes and original source remain preserved.

The next planned full study, **D**, is **22 cells × 120 cases, seed zero,
`teacher_forced` only**. It has not started and awaits the corrected pilot gate
and final source freeze. No D performance results are claimed. Its primary measures are
original-proposal accuracy and reference-correction rate, accompanied by native
episode/decision coverage and explicit model/interface failures. Goal completion
on a reference-corrected trajectory is assisted completion, even when the final
native goal holds. Preserve B's autonomous pilot results separately and do not
pool its inputs, outcomes or denominators with this new study. See
[the graph protocol guide](graph/README.md) for exact view, qualification and
teacher-forcing semantics.

The [canonical deadline results path](../results/120/workflow-deadline-results.json)
remains separate from stopped A. Read the embedded study identity, status and
coverage before citing it. Each production cell retains its full 120-case
denominator, including aborted, missing and deadline-censored cases.

The exported MCQA summaries are:

| File | Contents |
| --- | --- |
| [mcqa-results.json](../results/388/mcqa-results.json) | Held-out scores, calibration, AUROC/selective curves, timing/token coverage, full-cohort allocation estimates and provenance for all 34 configurations |
| [mcqa-results-by-category.json](../results/388/mcqa-results-by-category.json) | The same supported metrics per L1–L5 category, with full-cohort and held-out sections |
| [mcqa-results-by-option-count.json](../results/388/mcqa-results-by-option-count.json) | The same supported metrics per 2/3/4-choice subgroup, with full-cohort and held-out sections |

Accuracy and coverage are fractions: `0.9298` means `92.98%`. Subgroup records
include `n`, `correct`, invalid/missing counts and coverage. An empty scoped
reference group is `n: 0, accuracy: null`, not zero accuracy.

These files share the static snapshot generated on **2026-10-22 at 10:16 UTC
(12:16 Europe/Zurich)**. The group exports were generated in Slurm job
`4639272` from **272 authenticated prediction files**, using their recorded
hashes. All 34 held-out summaries matched the original results; group counts
reconciled and the independent portable scorer agreed. No new model inference
was performed for that grouping.

Schema 1.1 adds selected-answer Brier, correctness AUROC, full selective curves
and timing/token coverage in Slurm job `4640085`. All original score fields and
group denominators were checked before augmentation. Known-case checks cover
Brier arithmetic, AUROC ties, selective-threshold ties, missing usage, empty
groups and reference timing placeholders. All 272 dedicated MCQA raw allocation
IDs matched Slurm accounting; generic GPU units are counted once, without
adding their typed-GPU TRES entries again. No inference was rerun.

In the overall file, use `heldout_binary_brier_selected_correctness`,
`heldout_correctness_auroc`, `heldout_selective_curve` and `heldout_resources`
within each method. The group files store their unprefixed counterparts within
each `overall`/`groups` summary. `compute_accounting` describes complete retained
MCQA allocations, separate from held-out/subgroup resources; its currency
amounts are labeled estimates, not measured per-question costs.

Use `methods[].heldout.groups` for the main subgroup comparison and
`methods[].all_questions.groups` for full-cohort diagnostics in the two group
files. Preserve experiment IDs and seeds; do not average subgroup percentages
without accounting for their denominators. The overall JSON stores held-out
accuracy; it does not supply all-388 accuracy values.

Earlier baseline scores, later local comparisons, hosted colleague runs and
graph protocols have separate source/configuration identities. Cite the exact
snapshot rather than combining their scores. Static result availability does
not establish completion of the end-to-end graph study.

## 8. Reproduction, execution and validation

| Folder | Role |
| --- | --- |
| [instrument/](instrument/README.md) | Frozen public datasets, private answers/specifications, native runtime and portable graders |
| [comparison/](comparison/README.md) | Local HF/Ollama readouts, native heads, frozen features and supporting studies |
| [graph/](graph/README.md) | Actual application-graph adapter and expert × chooser study preparation/reporting |
| [review/](review/README.md) | 150-question human-review package |

On SCITAS, installation, imports/validation, tests, simulation, fitting,
inference and report rendering run **through Slurm on compute nodes**. Login
nodes are used for editing, job submission, status queries and plain log
inspection. A source/configuration change requires a new output directory and
matching provenance.

Start with the portable instrument package:

```bash
cd benchmark/instrument
mkdir -p output
sbatch slurm/setup.sbatch
# After setup succeeds:
sbatch slurm/check.sbatch
sbatch slurm/sequential.sbatch --policy reference --mode both
```

The following are **compute-job payloads**, run from that same directory inside
an allocated compute job:

```bash
.venv/bin/python eval/prompts.py --split all --output output/mcqa-prompts.jsonl
.venv/bin/python eval/mcqa.py --predictions output/model-predictions.jsonl \
  --split heldout --output output/model-heldout.json
.venv/bin/python eval/sequential.py score --episodes output/RUN/episodes.jsonl
```

Give a model only each exported prompt and save its selected original option
ID. Stage model weights separately; the portable data/scorer package does not
download them. Use the [comparison guide](comparison/README.md) for inference
environments and method configurations, and the [graph guide](graph/README.md)
for actual workflow setup, freezing, arrays and reports.

[Portable regeneration](graph/tools/regenerate_mcqa.py) authenticates the
vendored generator/runtime, restores the frozen generation configuration,
replays candidates and compares the original public-question and private-oracle
hashes. Run it inside a compute job with a fresh output directory. Release
maintenance is documented [here](instrument/tools/README.md); do not casually
edit manifest-protected data, answers, grader or runtime files.

Validation evidence is separated by scope:

- [Instrument validation](instrument/validation.json) distinguishes static
  integrity checks, isolated-copy tests, limited packaging smoke replays and
  preserved prior full native sequential replay.
- [Comparison validation](comparison-validation.json) records supporting-runner
  checks.
- [Graph validation](graph-validation.json),
  [installed-copy validation](graph-installed-validation.json) and the
  [graph manifest](graph-manifest.json) identify the installed graph package.
- [Human-review compatibility](review/compatibility.json) verifies the frozen
  public task content used in the review interfaces.

A passed packaging test is not a new full native replay or completed neural
evaluation. Historical MCQA-only version-1.0 receipts describe the earlier
sequential placeholder, not the current combined release.

## 9. Reporting boundaries

Independent expert validity review is **still pending**. The review package
contains 150 blinded questions covering all 112 scenario groups; browser-local
quality ratings and a blank CSV template are not evidence that independent
review has been completed.

Report these as authored simulator-operation benchmarks. Simulation consistency
does not establish real-equipment reliability, physical safety, chemistry or
material-transfer outcomes. Related variants reduce the effective number of
independent cases; use scenario groups when analyzing uncertainty. L labels
are not validated difficulty levels, and the small template/fault/instrument
holdouts do not demonstrate broad generalization.

For the sequential track, report assisted completion separately from autonomous
success and preserve missing/error coverage. Keep models, prompts, decoding,
native runtime, datasets, hardware and evaluation version identifiable. Retain
the release [licenses](instrument/LICENSE),
[asset terms](instrument/LICENSE-ASSETS.md), [notices](instrument/NOTICE.md)
and [citation](instrument/CITATION.cff) when sharing.
