# 388-question results: evaluation protocol

These files report **static multiple-choice answer selection** on the frozen
instrument benchmark. The primary comparison uses **114 held-out questions**.
The 388-question corpus also contains training and validation examples; an
all-388 score must be labelled separately from held-out performance.

## Result files

| File | Contents |
| --- | --- |
| [mcqa-results.json](mcqa-results.json) | One flat held-out summary per configuration in `methods[]`, including accuracy, coverage, confidence metrics and resource accounting. |
| [mcqa-results-by-category.json](mcqa-results-by-category.json) | Overall and L1–L5 category metrics, separately under `methods[].heldout` and `methods[].all_questions`. |
| [mcqa-results-by-option-count.json](mcqa-results-by-option-count.json) | The same metrics grouped by 2, 3 or 4 offered choices, with separate held-out and all-question sections. |

For the primary score, read `methods[].heldout_accuracy` and `heldout_n` in the
first file. In the grouped files, read `methods[].heldout.overall` or
`methods[].heldout.groups`. Full-corpus accuracy is available in the grouped
files under `methods[].all_questions.overall`; the first file does not contain
an all-388 accuracy field. Rates are fractions: `0.75` means 75%. An unavailable
metric is `null`, not zero.

## Dataset, reference answers and held-out split

The corpus contains **388 questions, 194 base cases and 112 semantic scenario
groups** across eight instrument simulators. Questions and answer candidates
were authored from instrument contracts and parameterized simulator scenarios.
Reference answers were established by native candidate execution and explicit
authored predicates, not an LLM judge. All released questions have exactly one
accepted option. Some options are complete procedures or recovery plans, but
the model still selects a supplied answer rather than executing that plan.

The model receives the question, operating goal, public observations, relevant
manual/contract evidence and shuffled choices through the
`public-menu-union-v2` prompt projection. Documentation is collected from all
candidates symmetrically. Answer keys, private simulator state, fault controls
and split/group bookkeeping are withheld. There is no interactive simulator
feedback or teacher-forcing correction during MCQA evaluation.

| Cohort | Questions | Use |
| --- | ---: | --- |
| Training | 232 | Fit feature classifiers, kNN memory and the label-prior reference. |
| Validation | 42 | Fit the cosine temperature; keep calibration/model selection separate from held-out evaluation. |
| Held out | 114 | Primary evaluation: `test` 60, `test_template` 36, `test_fault` 8 and `test_instrument` 10. |

Related semantic cases and presentation variants remain together when assigning
splits. **Pretrained model weights stayed frozen.** Logistic classifiers were
trained on the training split, kNN retained labelled training examples, and the
MiniLM cosine temperature was fitted on validation data. Thus the feature
methods use supervised fitting even though their neural encoders are frozen.

The public T1-rule reference is a scoped exception: **208 corpus questions and
64 held-out questions**, not 388 and 114. Always use its recorded denominator.
An out-of-scope subgroup has `n: 0` and `accuracy: null`.

## Configurations and answer generation

The export contains **34 configurations using five pretrained checkpoints**:
Qwen2.5-3B-Instruct, Qwen3-4B-Instruct-2507, OpenJev-8B, APUS-Reflex
`4B-5949`, and the all-MiniLM-L6-v2 embedding encoder.

| Configuration family | Count | Evaluation method |
| --- | ---: | --- |
| Qwen2.5-3B with Hugging Face | 7 | Greedy choice, menu-token logprobs, verbal confidence, cited confidence, three sampling seeds. |
| Qwen3-4B with Hugging Face | 7 | The same methods. |
| Qwen2.5-3B with Ollama | 6 | Choice, verbal/cited confidence and three sampling seeds; no full-menu logprobs. |
| Native decision heads | 3 | OpenJev and APUS low/high, using released learned readouts; APUS uses trained depths 16/32. |
| Frozen features | 4 | MiniLM cosine, logistic classifier and kNN; Qwen2.5 hidden-feature logistic classifier. |
| Expert then chooser | 4 | Qwen2.5 expert followed by Qwen2.5, OpenJev or APUS low/high. |
| References | 3 | Uniform random, training-label prior and scoped public T1 rules. |

Greedy choice must parse as one offered label. Logprobs normalize next-token
scores over the offered labels. Verbal methods supply an answer and a finite
confidence in `[0,1]`. Sampling generates **ten answers at temperature 0.8**,
uses vote frequencies as probabilities and resolves ties by menu order; under
the strict comparison protocol, any malformed draw invalidates that decision.
Sampling seeds 0, 1 and 2 remain separate configurations, not an averaged score.
Shards are partitions of a cohort, not additional independent seeds.

The four `expert_decision` rows use the earlier comparison-owned shortlist
adapter, identified as `old_shortlist_adapter`. Their expert proposes distinct
offered candidates and a chooser selects among them. On 2–4-choice MCQA, a
valid `min(4, menu size)` shortlist includes the whole menu. These rows are
separate from the application-graph experiment in [../120](../120/README.md).

## Scoring and confidence metrics

The evaluator joins predictions to frozen task IDs and scores the **submitted
selected option**, without replacing it with a probability argmax. Missing
answers, non-menu answers and non-`ok` outcomes count as incorrect in the full
declared cohort. Invalid model replies and infrastructure-error attempts stay
visible. Structurally malformed prediction files, duplicate IDs and unknown
task IDs are rejected rather than silently discarded.

| Metric | Definition and denominator |
| --- | --- |
| Accuracy | Correct selected options / all questions in the declared cohort or subgroup, including invalid/missing cases. |
| Attempt coverage | Recorded attempts / cohort size; an error can be an attempt without being a valid response. |
| Response coverage | Valid offered answers with status `ok` / cohort size. |
| Confidence coverage | Eligible selected-answer confidence rows / cohort size. |
| ECE | Ten fixed-width confidence bins; confidence-row-weighted absolute difference between mean confidence and correctness in each bin. |
| Selected-answer Brier | Mean `(confidence - correctness)^2` over eligible confidence rows. |
| Multiclass Brier | Mean sum of squared errors over all menu labels, requiring a genuine full-menu probability vector and one accepted label. A confidence scalar or unvalidated shortlist vector is insufficient. |
| Correctness AUROC | Confidence ranking of correct versus incorrect answers, with half credit for ties; undefined without both classes. |
| Selective curve | Keep answers at or above each confidence threshold, including ties. Accuracy uses accepted answers; coverage uses the full subgroup size. |

The held-out export supports ECE and selected-answer Brier for 31
configurations, full-vector Brier for 21, selective curves for 31, and AUROC for
30. Plain-choice methods have no supplied confidence; unsupported metrics stay
null. Curves are descriptive held-out analyses, not deployment guarantees from
thresholds fitted on held-out data. Check eligible counts and coverage alongside
every calibration or ranking score.

## Category and option-count breakdowns

The legacy field `difficulty` encodes the following **task categories**, not an
established increasing difficulty scale:

| Label | Category | All questions | Held out |
| --- | --- | ---: | ---: |
| L1 | Single-action checks | 158 | 44 |
| L2 | Direct diagnosis and simple recovery | 50 | 20 |
| L3 | Evidence assessment, testing and escalation | 88 | 18 |
| L4 | Procedure and service-plan selection | 62 | 22 |
| L5 | Interruption-response plan selection | 30 | 10 |

Choice-count groups contain **182/136/70** corpus questions and **64/26/24**
held-out questions for **2/3/4 choices**, respectively. These are menu sizes,
not semantic classes. Their task mixtures differ, so subgroup accuracy does not
isolate the effect of adding answer choices. Each group keeps its own invalid,
missing and confidence-availability counts.

## Timing, tokens, cost and verification

Recorded latency is backend/pipeline wall time, not active GPU time. It normally
excludes loading, although the first expert-to-native call can include lazy
chooser initialization. Feature timing includes cache and CPU work. Uniform and
majority reference zero-time placeholders are excluded from measured timing;
32 configurations have observed timing. Token totals retain known-usage
coverage; sampling totals already include all draws and expert/chooser totals
already include both stages. Missing usage is not replaced by zero.

`compute_accounting` uses **272 retained MCQA shard allocations**: Slurm elapsed
hours times allocated GPU units, multiplied by the archived installed-site
tariff. These are full-cohort allocation estimates, including startup/loading,
not invoices or active kernel costs. A MIG unit is a slice. Preparation, model
downloads, feature fitting, other tracks and discarded attempts are excluded.
Held-out/category/choice subgroups share allocations, so their allocation costs
remain null rather than being prorated from inference latency. Hosted API spend
was zero; that does not mean local compute was free.

The JSONs preserve the **2026-10-22** snapshot and schema-1.1 metric augmentation.
`provenance`, `metrics_augmentation` and `verification` record source paths,
hashes, jobs and checks: all 272 prediction hashes matched, existing scores and
denominators were preserved, grouped counts reconciled, and an independent
portable accuracy scorer agreed. Added metrics were computed from those saved
predictions without new model inference. No new significance tests or
confidence intervals are claimed. Related questions are not independent
physical experiments, and independent domain-expert review remains pending.

See the [dataset guide](../../benchmark/instrument/mcqa_388/README.md),
[comparison implementation guide](../../benchmark/comparison/README.md), and
[construction/evaluation report](../../benchmark/README.md) for source details
and reproduction instructions. Raw prediction paths inside the JSON provenance
refer to the original cluster checkout; these summary files do not embed every
raw prediction or model checkpoint.
