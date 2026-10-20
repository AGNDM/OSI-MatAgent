# Decision Mechanisms for Lab Agents (working title)

**How should an autonomous lab agent make operational decisions, and when should it act, escalate, or measure first?**

This project builds a benchmark of instrument-operation decisions from the [ChemLab-Assets](https://github.com/mbzuai-nlp/ChemLab-Assets) simulator and uses it to compare *mechanisms* for producing a decision and its confidence: generative LLMs, token probabilities, sampling, linear probes, embedding similarity, and dedicated decision models (Jev / OpenJev family), both local and API-based.

No model is trained in this project. Linear probes and a single temperature parameter for embedding similarity are the only fitted components, and they are fitted on frozen features.

---

## Contents

1. [Goal and research questions](#1-goal-and-research-questions)
2. [Testbed: ChemLab-Assets](#2-testbed-chemlab-assets)
3. [Part 1: Decision-task generator](#3-part-1-decision-task-generator)
4. [Part 2: Methods](#4-part-2-methods)
5. [Metrics](#5-metrics)
6. [Failure feedback](#6-failure-feedback)
7. [Splits and evaluation protocol](#7-splits-and-evaluation-protocol)
8. [Expected outputs](#8-expected-outputs)
9. [Limitations](#9-limitations)
10. [First steps](#10-first-steps)
11. [References and resources](#11-references-and-resources)

---

## 1. Goal and research questions

The goal is not to name a single best method. We expect trade-offs between accuracy, calibration, cost, and explainability, and the contribution is to map them on decisions with exact ground truth.

| | Research question | Answered by |
|---|---|---|
| **RQ1** | **Calibration vs. accuracy.** Which mechanism produces confidences that can be used to automate decisions safely? | Selective accuracy, ECE, Brier, confident errors on hazardous actions |
| **RQ2** | **Cost.** What does each level of reliability cost, local vs. API? | Latency, cost per 1,000 decisions |
| **RQ3** | **Knowing what you don't know.** When decisive information is hidden, which methods recognize it and ask to measure instead of guessing? | Information-request quality on hidden-state tasks |
| **RQ4** | **Explaining failures.** When decisions fail, can the method say why, and is the stated reason the true one? | Rationale faithfulness against contract-derived causes |
| **RQ5** | **Generalization.** Do method rankings hold on instrument domains not seen when fitting probes or thresholds? | Leave-domain-out generalization gap |

---

## 2. Testbed: ChemLab-Assets

ChemLab-Assets (MBZUAI) provides 198 simulation-ready chemistry lab instruments in 16 capability domains. Each instrument ships:

- **A contract** (`asset.contract.json`): components, typed state slots with units, ranges and visibility (agent / evaluator / debug), and capabilities with JSON-Schema arguments, preconditions, state effects and declared errors.
- **A behaviour runtime** (`scripts/asset_runtime/<id>.py`): a time-stepped process model behind `describe` / `read` / `invoke` / `poll`, with agent and evaluator views and seeded noise.
- **Geometry** (URDF, GLB, meshes) and Isaac Lab conversions. These are not used in this project.

**The dataset contains no decision tasks.** It contains instrument specifications and simulators. The decision tasks are constructed in Part 1, and their construction is a core contribution of this work.

Three properties make it suitable:

- **Labels by execution.** Whether a decision was right is determined by running it in the runtime. No annotation is needed.
- **Built-in partial observability.** Evaluator-only slots hold true values the agent cannot see (e.g., the true face velocity of the fume hood), which allows "measure first" decisions.
- **Groundable failure causes.** Preconditions and declared errors give an exact reason for each failure.

---

## 3. Part 1: Decision-task generator

### 3.1 Pipeline

For each instrument:

1. **Sample a state.** Configure the runtime (door positions, setpoints, calibration status, injected faults) from a recorded seed.
2. **Pose a question.** Instantiate a decision type (Section 3.2) with its answer options.
3. **Label by execution.** Snapshot the state, try each option in the runtime, record the outcome, and restore. The correct answer is whatever the simulator says happens.
4. **Record the cause.** Store the precondition, declared error, or hidden slot that determined the outcome. This is the ground truth for rationale faithfulness and failure categories.
5. **Control the mix.** Balance labels, decision types, formats, difficulty tiers and domains so that no metric is dominated by easy or majority-class cases.

### 3.2 Decision types

| Contract element | Decision | Example | Label source |
|---|---|---|---|
| Capability preconditions | **Can I do this now?** | Can the balance start a measurement with its draft-shield door open? | Invoke and check success |
| Declared errors | **Will this fail, and why?** | Which error does raising the sash past its limit trigger? | Runtime raises the error |
| State effects | **Which action reaches the goal?** | Which capability returns the airflow alarm to "none"? | Apply each candidate, check resulting state |
| Argument schemas | **Are these parameters valid?** | Is this setpoint within the allowed range? | Schema validation plus runtime |
| Readouts vs. evaluator-only slots | **Can I trust this reading, or must I measure?** | Monitor reads 0.28 m/s while true face velocity is hidden; calibrate first? | Compare readout with hidden true value |
| Alarms and state | **What caused this alarm?** | Low-airflow alarm: sash position, uncalibrated monitor, or real fault? | Injected fault is known |
| Capability sequences | **What is the next protocol step?** | During hood calibration: open sash, measure, or calibrate? | Terminal state reached or not |

### 3.3 Decision formats

| Format | Example | Options |
|---|---|---|
| Binary | Can capability X run now? | 2 |
| Binary + abstain | Is this reading trustworthy? (yes / no / `insufficient_information`) | 3 |
| Multiple choice, fixed set | Which declared error will this trigger? | Fixed per instrument |
| Multiple choice, variable set | Which capability reaches the goal? | Varies, up to 50+ |
| Diagnosis | What caused this alarm? | Candidate causes |
| Sequential | A chain of decisions through a protocol | Many per episode, judged by final state |

### 3.4 Difficulty tiers

| Tier | Requires | Expected behaviour |
|---|---|---|
| **T1: Visible state** | Preconditions, parameter validity, error prediction from agent-visible slots | Mostly solvable by the rule checker; serves as a calibration sanity check |
| **T2: Hidden state or noise** | Trusting readings, deciding when to measure, diagnosing alarms | Where methods should differ most and calibration matters most |
| **T3: Goal understanding** | Choosing actions for natural-language goals, next protocol step | Where LLMs should outperform probes and rules |

**Viability check:** if the rule checker solves most generated tasks, the benchmark is too easy. A substantial share of tasks must be T2 or T3.

### 3.5 Validity measures

- **Snapshot and restore.** Step 3 requires copying and restoring runtime state, or deterministic replay from a seed. Verify this first.
- **Expert audit.** A chemist reviews a random sample of 100–200 generated tasks: would an operator face this decision, and is the simulator's answer right?
- **Wording diversity.** Several phrasings per decision type (LLM-paraphrased, then validated), with some templates held out for testing, so that probes cannot win by learning templates.
- **Release the generator.** Publish the generator alongside a fixed test set so that others can regenerate harder splits.

---

## 4. Part 2: Methods

All methods receive the same state serialization, question wording, and answer space, including an `insufficient_information` option where applicable.

| # | Method | Probability source | Variants |
|---|---|---|---|
| 1 | **Verbalized confidence** | LLM states its answer and confidence in text | Local and API LLMs |
| 2 | **Token log-probabilities** | Output constrained to single-token option labels; probability read from logprobs and renormalized over valid options | Local LLMs; API models that expose logprobs |
| 3 | **Sampling frequency** | Vote share over N ≥ 10 sampled answers (chat-model OpenJev approach) | Local and API LLMs |
| 4 | **Dedicated decision models** | Native calibrated output, used off the shelf | Jev (API), OpenJev-8B (local), APUS-OpenJev low / high effort (local) |
| 5 | **Linear probes** | Logistic regression on frozen features; variable-option choices scored per candidate, then softmaxed | Local-LLM hidden states; embedding models |
| 6 | **Embedding similarity** | Cosine between state and answer options with one fitted temperature; kNN over labeled examples | Embedding models |
| 7 | **LLM with tool access** | Methods 1 or 2, with `describe` / `read` calls allowed before deciding | Reported as a separate track (more information available) |
| 8 | **Reference baselines** | Rule-based contract checker (upper bound, T1 only); majority-class prior (lower bound) | — |

**Notes**

- The rule checker is a reference, not a competitor. It is exact on fully observable tasks and undefined on hidden-state tasks, so it is reported only on T1.
- The tool-access track is ranked separately because it receives more information than single-shot methods.
- Generative methods (1, 2 with rationale, 3, 7) must cite the slot or precondition ID their decision rests on, so faithfulness can be scored without an LLM judge.

---

## 5. Metrics

| # | Metric | Measures | Applies to |
|---|---|---|---|
| 1 | **Balanced accuracy** | Correctness, robust to label imbalance | All |
| 2 | **Brier score / Brier skill score** | Probability quality, absolute and relative to the class prior (multi-class form for >2 options) | All probabilistic methods |
| 3 | **Expected calibration error (ECE)** | Whether confidence matches accuracy (on the chosen option's confidence); shown with reliability diagrams | All probabilistic methods; embedding similarity only after temperature scaling |
| 4 | **AUROC** | Ranking quality of confidence for correct vs. incorrect answers | All; primary calibration-free metric for raw similarity scores |
| 5 | **Selective accuracy vs. coverage** | Share of decisions automatable at fixed precision (e.g., coverage at 95% and 99%) | All |
| 6 | **Confident-error rate on hazardous actions** | High-confidence mistakes on irreversible or hazardous actions, weighted separately from safe reads | All |
| 7 | **Information-request quality** | Precision and recall of choosing `insufficient_information` or measuring first | T2 tasks |
| 8 | **Rationale faithfulness** | Exact match between cited slot / precondition ID and the contract-derived cause | Generative methods |
| 9 | **Failure-category distribution** | Share of errors per category (Section 6) | All |
| 10 | **Latency per decision** | Wall-clock time | All |
| 11 | **Cost per 1,000 decisions** | API dollars and local GPU-seconds, reported separately with hardware named | All |
| 12 | **Label efficiency** | Performance vs. number of labeled examples | Probes, kNN |
| 13 | **Generalization gap** | In-domain vs. leave-domain-out performance | All; most relevant for fitted methods |

**Reporting conventions**

- Results reported overall and broken down by decision format, difficulty tier, and domain.
- 95% bootstrap confidence intervals over decisions.
- Paired randomization tests between methods on the same tasks, Holm-corrected.
- Multiple seeds for sampling-based methods.
- Label ratio controlled at generation time; Brier skill score reported against the prior.

---

## 6. Failure feedback

Every incorrect decision is classified using the contract, for every method:

| Category | Meaning | Feedback to the user |
|---|---|---|
| **Violated precondition / declared error** | The runtime names the cause | Exact cause, e.g. "sash above 457 mm; stability check requires it closed" |
| **Unobservable** | The decisive slot was evaluator-only; no method could have known | "Measure X first"; the method should have been uncertain |
| **Confident and wrong** | Information was visible and the method was confident | Flagged as the dangerous class |
| **Uncertain and wrong** | Low confidence but acted anyway | Threshold-setting issue; should have escalated |

Distinguishing *unobservable* from *confident and wrong* separates "the agent needed more information" from "the method is unreliable". The agent / evaluator visibility split makes this distinction exact.

For generative methods, the stated rationale is additionally checked against the true cause (metric 8). This exposes the trade-off between methods that explain but may explain wrongly and methods that are calibrated but give no explanation.

---

## 7. Splits and evaluation protocol

- **Leave-domain-out.** Probes, kNN and confidence thresholds are fitted on some capability domains and tested on held-out domains. In-domain results are reported only alongside the held-out ones.
- **Held-out templates.** Some question phrasings appear only in the test split.
- **Label budgets.** Probes are evaluated at several training-set sizes to produce learning curves.
- **Fixed inputs.** Identical serialization, wording, and answer space for every method.
- **Unsupported capabilities excluded.** Capabilities marked unsupported in the contracts are filtered out of the task pool.

---

## 8. Expected outputs

1. **Task generator** with documented decision types, formats and tiers, plus a fixed released test set.
2. **Expert-audited validity sample.**
3. **Comparison of decision mechanisms** across all metrics, by format, tier and domain.
4. **Practical recommendations**, e.g. "for reversible reads use X; for hazardous actions use Y with threshold Z and escalate below it".
5. **Failure-feedback module** that classifies and explains failed decisions from the contract.

Suitable venues: datasets-and-benchmarks tracks, where the generator and its validation carry as much weight as the comparison.

---

## 9. Limitations

- **Simulator fidelity.** The assets reproduce documented instrument behaviour but are not calibrated against physical units. The benchmark tests decision mechanisms against documented instrument models, not real-lab correctness.
- **Operations, not chemistry.** The runtimes model instrument behaviour, not reaction outcomes.
- **Unsupported capabilities.** 786 capabilities are declared but marked unsupported, and none claims robot-contact or real-hardware validation.
- **Maturity.** The repository is young; expect rough edges. Contract and protocol specifications are in Chinese.
- **Licence.** Assets, contracts and data are CC BY-NC 4.0 (code is MIT). Suitable for research, not commercial use.
- **API access.** Not all APIs expose logprobs, which limits method 2 for some models.

---

## 10. First steps

1. **Audit (half a day).** Download the Reaction & Molecular Synthesis domain plus a few analytical instruments (NMR, HPLC, GC), run `pytest`, and count per instrument:
   - capabilities with non-trivial preconditions,
   - declared errors,
   - evaluator-only slots,
   - unsupported capabilities.
2. **Check snapshot / restore** or deterministic replay in the runtimes.
3. **Check material flow** between instruments (ports) to decide whether multi-instrument workflows are feasible or need a glue layer.
4. **Prototype one instrument end to end** (the fume hood): generate T1–T3 tasks, run methods 1, 2 and 8, compute metrics 1–5.
5. **Decide the task mix** based on how many T2 / T3 tasks the audit shows the dataset can support.

---

## 11. References and resources

- **ChemLab-Assets:** [GitHub](https://github.com/mbzuai-nlp/ChemLab-Assets) · [Hugging Face dataset](https://huggingface.co/datasets/MBZUAI/ChemLab-Assets)
- **Jev (TypeSafe):** [OpenRouter](https://openrouter.ai/typesafe/jev-1.13)
- **OpenJev-8B:** [Hugging Face](https://huggingface.co/alanhuangya/OpenJev-8B)
- **APUS-OpenJev-v1:** [Hugging Face](https://huggingface.co/apus-ailab/APUS-OpenJev-v1)
- **SiliconLabAI/OpenJev:** chat-model wrapper implementing the Jev decision contract
- **OpenJev-RLCD:** [GitHub](https://github.com/ZimmyGao/openjev-rlcd) · [paper](https://arxiv.org/pdf/2609.38850)

```bibtex
@misc{song2026chemlabassets,
  title        = {ChemLab-Assets: Simulation-Ready Chemistry Laboratory
                  Instruments with Executable Behaviour Contracts},
  author       = {Song, Zirui and Zhang, Jiadong and Zhang, Jiahao and
                  Chang, Xiaojun and Ma, Xiaosong and Chen, Xiuying},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/MBZUAI/ChemLab-Assets}},
  note         = {Dataset, release 1.0}
}
```
