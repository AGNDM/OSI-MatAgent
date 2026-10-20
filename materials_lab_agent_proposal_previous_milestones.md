# Materials Lab Agent: A Two-Day, Milestone-Driven Hackathon Proposal

**Revised 19 October 2026 · Build a working loop first; choose the scientific case from verified capabilities.**

## 1. Destination: the demo we want to have

By the end of the hackathon, demonstrate an agent investigating a material-processing result that does not meet expectations. It observes the laboratory state, retrieves relevant materials knowledge, requests a useful check, chooses a corrective action, and verifies the new result. A second episode should lead to a different decision when the evidence changes.

The scientific theme is **processing → material characterization → diagnosis → corrective action → verification**. Heat treatment and XRD are the preferred direction, but the exact compound, instrument and failure are selected after inspecting the available runtime. MgAl₂O₄ is one possible example, not a project requirement.

The presentation should make three things visible: what happened in the virtual lab, what evidence changed the agent's decision, and whether the correction actually worked in the declared simulation. Materials knowledge must influence the chosen check or repair; simply explaining a generic alarm is insufficient.

The two-day completion target is a small, honest vertical slice. Use time limits and fallback decisions to protect that target. This plan assumes one primary builder, approximately eight focused hours per day, a usable Python environment and access to an LLM. It does not rely on overnight development, model training or a second team. Model access and dependency failures are handled at explicit checkpoints rather than allowed to consume the event.

## 2. Start here: dataset and repository

- **Dataset:** [MBZUAI/ChemLab-Assets](https://huggingface.co/datasets/MBZUAI/ChemLab-Assets).
- **Official code repository:** [mbzuai-nlp/ChemLab-Assets](https://github.com/mbzuai-nlp/ChemLab-Assets).
- **Project site:** [ChemLab-Assets](https://mbzuai-nlp.github.io/ChemLab-Assets/).

The repository README was checked for this revision. It contains runtimes, examples, tests, a catalog and an explorer; models/data are hosted on Hugging Face. It documents `describe/read/invoke/poll`, agent/evaluator views and time-stepped behavior. This is a verified documentation claim, not a local execution result. Exact signatures and selected instrument behavior still need inspection.

Start by reading the README, `examples/`, `catalog/`, the selected runtime and its tests. The repository documents `tools/download_assets.py`; inspect its options before downloading. Fetch only the assets needed for the first example and eventual demo. Record the code and dataset revisions so they can be reproduced together.

The first working example may be the supplied fume-hood example. Its purpose is to learn the runtime and permissions. The final materials demo must go beyond it.

## 3. Two-day schedule and stopping rules

All times below are focused working hours, with breaks outside the table. The schedule totals **16 hours**, including **2 hours of contingency** and **1 hour of rehearsal**. Each milestone leaves a usable artifact. At a time limit, take the stated fallback instead of extending scope.

| Day / hours | Milestone | Small question to answer | Artifact / exit condition |
|---|---|---|---|
| Day 1, 0–1.5 | M0: Run upstream | Can we operate one asset locally? | Reproducible command and before/after runtime state |
| Day 1, 1.5–2.5 | M1: Expose tools | Can our workflow safely call that asset? | Minimal adapter and one logged tool sequence |
| Day 1, 2.5–3.5 | M2: Choose the case | Which materials question fits what actually works? | One-page scenario card; one primary cause and one alternative |
| Day 1, 3.5–6 | M3: Build the smallest loop | Can a check distinguish causes and a repair change the outcome? | Scripted observe/check/repair/verify loop with an independent score |
| Day 1, 6–7 | M4: Add agent decisions | Can the LLM choose tools from observations? | One complete agent-run episode in the terminal |
| Day 1, 7–8 | Contingency | What blocks the working loop? | Repair the core; no new features |
| Day 2, 0–1.5 | M5: Add materials retrieval | Does retrieved evidence improve the next decision? | Small cited corpus and retrieval integrated into the loop |
| Day 2, 1.5–2.5 | M6: Add System 1 routing | Can Laya handle one routine decision and escalate the rest? | Actual routing call, or clearly labeled deterministic fallback |
| Day 2, 2.5–4 | M7: Make it visible | Can an audience follow the evidence and action? | Simple lab view, result chart and action timeline |
| Day 2, 4–6 | M8: Verify and freeze | Does it work beyond a single happy path? | Small regression matrix, baseline comparison and frozen demo |
| Day 2, 6–7 | Contingency | Are there remaining demo-breaking issues? | Stabilize; no architecture changes |
| Day 2, 7–8 | Rehearse and package | Can we show it clearly in five minutes? | Live script, run instructions and labeled replay backup |

**Day 1 gate:** a complete loop exists, even with a minimal interface and deterministic helper logic. **Day 2 feature freeze:** after hour 6, only fix failures and rehearse. If a task finishes early, use the saved time for stability before adding optional scope.

## 4. Milestones as actionable mini-projects

### M0 — Make one upstream example work (90 minutes)

Questions: What has to be installed? Where are asset data expected? Which principal can read or change which state? How is simulation time advanced?

- [ ] Obtain the repository and required asset subset; save revisions and installation notes.
- [ ] Run an existing example unchanged before modifying it.
- [ ] Save an initial observation, an accepted action and a changed observation.
- [ ] Inspect one relevant test and the distinction between agent-visible and evaluator-only state.

**Done:** the same command works again after resetting the runtime. **Fallback:** after 60 minutes of installation trouble, reduce to the smallest documented runtime example and omit 3D dependencies. If no native runtime works by the 90-minute limit, log that blocker and use a labeled custom stub temporarily. Do not later present that stub as executed ChemLab behavior. A full ChemLab-backed claim requires restoring at least one relevant native action by the Day 1 gate.

### M1 — Give the workflow a few reliable tools (60 minutes)

Question: Can a small Python wrapper expose observations and actions without exposing hidden answers?

- [ ] Implement only reset-for-developer, agent observation, supported action invocation and bounded waiting.
- [ ] Preserve upstream errors and units; log inputs, outputs and simulation timestamps.
- [ ] Keep evaluator setup, fault configuration and hidden state outside the agent tool list.
- [ ] Demonstrate one rejected invalid action.

**Done:** the same manually chosen sequence runs through the wrapper. Use direct Python calls initially; a separate HTTP/MCP service is unnecessary for this demo. Normalize `describe/read/invoke/poll` only where appropriate; do not invent native signatures or asynchronous jobs. The scheduler calls `sim.step` according to the actual runtime contract.

**Fallback:** support one instrument and two useful actions. Avoid building a general-purpose laboratory API.

### M2 — Select the smallest meaningful materials case (60 minutes)

Question: What can we demonstrate scientifically without building a new simulator for most of the event?

Inspect candidate synthesis/thermal and characterization assets, their implementations and limitations. Then choose one row below. Prefer the case with accessible evidence and the fewest missing behaviors, not the most impressive title.

| Candidate | Useful materials question | Minimum additional work |
|---|---|---|
| Heat treatment + phase evidence | Did processing fail, or is the characterization unreliable? | Two plausible observation states and one discriminating check; XRD fixtures if needed |
| Precursor preparation + downstream result | Is a composition error responsible for residual precursor/undesired phase? | Mass-to-mole calculation, sample record and bounded outcome rules |
| Another supported materials workflow | Which process variable explains a measured material-quality mismatch? | A material-specific evidence source and two different corrective choices |

The preferred starting case is **“unexpected post-treatment phase evidence: insufficient treatment or measurement/sample-preparation problem?”** Select the actual material only if its reference data are available within this hour. MgAl₂O₄ remains an optional candidate. Do not spend the event surveying many compounds.

Create a scenario card with six fields: target outcome; first unexpected observation; two hypotheses; available discriminating check; two corresponding actions; verification criterion. Add a short ownership table marking each behavior as native, imported data or custom rule.

**Done:** someone can understand the complete episode from this one page. **Fallback:** retain a material-processing case but use explicitly synthetic reference observations. Name the example material only when its scientific assumptions can be supported. Freeze the scenario at the end of this milestone.

### M3 — Close the loop before adding intelligence (150 minutes)

Question: Does the environment respond consistently to the chosen checks and actions?

- [ ] Track one sample ID, its processing history and current observation state.
- [ ] Implement two hidden scenarios that begin with a similar apparent failure but require different actions.
- [ ] Add a check that produces evidence about the cause rather than returning its name.
- [ ] Make the correct action change the appropriate state; make the wrong action fail to resolve the cause.
- [ ] Require a new observation after the action and score it with a separate evaluator function.

Use the cheapest adequate material model: a small deterministic transition table is acceptable. It must depend on actual state and action, not the agent's final explanation. For example, a remount changes measurement quality but cannot change phase composition; repeating treatment cannot repair a modeled missing precursor. Record the assumptions as simulation rules.

If XRD is chosen, first use a few fixed, labeled synthetic or openly licensed patterns with fixed acquisition metadata. Generating patterns from public CIFs is optional if time remains. Avoid quantitative phase-fraction claims from normalized mixture plots. Prefer a simple multi-peak check plus unexplained-residual indicator.

**Done:** both manually driven branches pass, and at least one deliberately wrong repair fails. **Fallback:** reduce to one fault plus a measurement-only alternative; remove continuous kinetics, detailed noise and arbitrary recipes. Keep the distinction between process failure and observation failure.

### M4 — Let the LLM choose the next tool (60 minutes)

Question: Can the agent complete the existing loop without a scripted action sequence?

Implement a short LangGraph flow: observe → decide → tool → observe → verify/report. Give the model a small menu, structured arguments, a maximum of six decision steps and one retry for malformed output. It should state the hypothesis being tested and cite evidence IDs in its final report.

**Done:** one terminal episode includes a model-selected check, correction and fresh verification. **Fallback:** if graph integration consumes 30 minutes without progress, use a simple Python state loop behind the same tools and mark LangGraph integration pending. A fixed narrated sequence does not count as agent control.

### M5 — Add a small materials evidence library (90 minutes)

Question: Can domain evidence change the selected test or correction?

Index roughly 5–10 short, sourced passages: the selected material/process assumptions, the relevant instrument limitation, and the chosen characterization artifact. Use the existing ModernBERT Materials Encoder if its inference and pooling interface are already usable. Return passage text, source URL and section with every hit. Keep hidden scenario answers out of the corpus.

**Done:** show one useful retrieval and the decision it supports. **Fallback:** cap checkpoint integration at 30 minutes; use a simple lexical search over the same corpus for the runnable demo, label it, and retain the encoder adapter as an integration point. Do not train or fine-tune an encoder during the event.

### M6 — Give Laya one bounded job (60 minutes)

Question: Can System 1 make a routine routing decision while System 2 handles uncertain diagnosis?

Use Laya to choose among a small menu such as `continue_known_step`, `request_more_evidence`, and `escalate_to_llm`. Feed structured observations and short evidence summaries. Keep hard preconditions in code. The LLM handles competing explanations and repair selection.

**Done:** log a real Laya decision and an escalation path. **Fallback:** after 30 minutes of loading/inference problems, keep a deterministic router and label Laya as pending. Do not let model infrastructure consume the final demo. Any probability cutoff is a prototype setting, not a calibrated reliability guarantee.

### M7 — Make the workflow understandable (90 minutes)

Question: Can a judge see the relationship between laboratory events and reasoning?

Build one page with: a selected GLB asset or existing explorer view, the current measurement/result chart, and the latest action/evidence timeline. Reuse the upstream explorer where practical; inspect its setup before committing. Show sample ID, simulation time and backend labels.

**Done:** the displayed state updates from logged backend events. A static GLB with live state badges is enough. **Fallback:** after 30 minutes on 3D integration, use the upstream viewer separately and keep the workflow page simple; if unavailable, use a labeled asset image. Articulated URDF animation and robotic transfer are optional. Do not add Isaac Lab or a physics engine to the critical path.

### M8 — Check the claims and freeze (120 minutes)

Question: Does the demo respond to evidence, and can it avoid false success?

Run six resettable cases: two variants of process failure, two of measurement failure, one no-fault control and one unresolved/contradictory case. The last case may be a small observation fixture; its expected outcome is uncertainty, not automatic repair. Include the wrong-repair negative control from M3.

Record diagnosis, chosen action, verified outcome, false-success count, tool count and wall-clock time. Compare the main agent with one simple baseline, such as “always repeat treatment.” If time allows, add an LLM-only routing comparison using the same episodes. Six cases support a demo check, not a statistical benchmark or a claim of general superiority.

**Done:** two contrasting branches work from reset, the wrong repair fails, and success cannot be declared without a new measurement. Package one start command, pinned inputs, a saved trace and a labeled replay/video backup. Use remaining time to fix failures rather than expand the evaluation suite.

## 5. Minimal architecture and capability boundaries

```text
Simple viewer / result chart / event timeline
                    |
Workflow state loop (LangGraph preferred)
       |            |             |
  Laya router   LLM planner    Materials retrieval
                                  |
                        Existing ModernBERT encoder
                    |
             Small tool adapter
              /             \
Verified ChemLab actions   Custom sample + observation rules
              \             /
                Event log
                    |
          Separate evaluator function
```

Every event should carry an action ID, sample ID, backend label, simulation time, observed result and error status. Use JSON logs and simple in-memory state; a database, deployment service and distributed architecture are unnecessary.

| Capability | Boundary for the demo |
|---|---|
| Instrument operation | Claim native behavior only after executing the selected runtime and inspecting its contract |
| Material state changes | Reuse only if verified; otherwise explicitly author a small rule table |
| XRD or other characterization | Distinguish native acquisition behavior, imported observations and custom spectrum generation |
| Sample flow | Maintain a custom sample record; do not imply physical robot validation |
| Fault injection | Use a supported native hook if available, otherwise a labeled custom scenario initializer |
| Evaluation | Keep truth outside model inputs; evaluate the observed trajectory independently |
| GLB/URDF display | Treat visual assets and articulation as presentation; they do not prove chemical fidelity |

Illustrative application tools, **not promised upstream method signatures**:

```text
observe(sample_id)
read_process_history(sample_id)
run_check(sample_id, check_id)
apply_action(sample_id, approved_action_id)
measure_again(sample_id)
search_materials_evidence(query)
```

Expose only tools the frozen scenario needs. The agent cannot reset the world, inject faults, read evaluator truth or set a success flag.

## 6. Scope choices that keep two days realistic

**Required demo floor:** one materials question, two explanations requiring different decisions, a bounded live agent loop, one materials evidence lookup, a new verification observation, a simple visual trace and a negative control. For a ChemLab-backed demo, include at least one relevant executed native action; if that is unavailable, explicitly call the result a prototype using ChemLab assets.

**Target integrations:** Laya routing, the existing ModernBERT encoder, LangGraph and one GLB view. Attempt each within its milestone timebox. Report exactly which integrations are live; an unavailable model is not silently replaced under its name.

**Optional only after the core works:** CIF-generated XRD, a third fault, multiple instruments, animated joints, a more continuous material model or an extra ablation.

**After the hackathon:** calibrated reaction kinetics, detailed phase quantification, broad benchmark datasets, fine-tuning, real hardware, robot physics, multiple chemistries and deployment infrastructure.

If behind schedule, cut in this order: animation → third fault → continuous simulation → extra model integration → extra ablations. Preserve the material-specific question, contrasting decisions, truthful component labels and verification.

## 7. Five-minute presentation

1. **0:00–0:40:** Show the intended material outcome and unexpected measurement. Explain which parts come from ChemLab and which were added.
2. **0:40–2:00:** Run the agent. Show a materials evidence passage, competing explanations and the selected check.
3. **2:00–3:15:** Apply the chosen correction, acquire a fresh result and reveal the independent evaluation.
4. **3:15–4:15:** Reset to the alternative case. Show why the new evidence requires a different action; a saved trace may shorten this part if labeled replay.
5. **4:15–5:00:** Show the six-case table, one wrong-repair failure, and the limits of the prototype.

The central claim is: **“We built an evidence-driven materials troubleshooting loop on top of an existing virtual lab, and the next action changes when the evidence changes.”**

## 8. Sources, assumptions and the first task to do

The official dataset and GitHub README were inspected on 19 October 2026; no local runtime execution was performed while writing this proposal. The repository reports both implemented and unsupported capabilities, so the first milestone must verify the selected ones. Code and asset/data licenses differ; keep upstream attribution and check the relevant files before redistributing the demo bundle.

Optional implementation references:

- [COD](https://cod.psdi.ac.uk/) for public crystal structures. Save exact accession IDs and provenance only if used.
- [pymatgen diffraction documentation](https://pymatgen.org/pymatgen.analysis.diffraction.html) for calculated reference patterns. Structure-based patterns do not supply reaction kinetics.
- [Laya model card](https://huggingface.co/convaiinnovations/laya) and [runtime repository](https://github.com/WhoIsMrSentry/laya-system-one) for the System 1 integration.
- [LangGraph documentation](https://docs.langchain.com/oss/python/langgraph/overview) for workflow implementation.
- User-provided ModernBERT checkpoint and existing materials corpus; availability and interface remain to be confirmed during the timeboxed integration.

**Start with M0:** run one upstream example and save the before/action/after trace. Then inspect only the few assets needed to choose a materials question. The next implementation decision should come from that evidence, while the final five-minute demo remains the destination.
