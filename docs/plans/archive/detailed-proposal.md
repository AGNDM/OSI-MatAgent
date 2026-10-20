# PhaseLoop: Evidence-Driven Troubleshooting for a Virtual Materials Laboratory

**Hackathon proposal · 19 October 2026 · Design specification, not an implemented system**

## 1. Objective and demonstration claim

Build an agent workflow that investigates an unsuccessful solid-state synthesis, chooses discriminating measurements, repairs the process when justified, and verifies the resulting material. The demonstration asks: **“Why did the post-anneal XRD pattern fail to show predominantly MgAl₂O₄ spinel, and what is the least costly defensible next experiment?”**

This is a materials reasoning task: composition constraints, diffusion-limited conversion, phase identification, measurement artifacts, and experimental design matter. A normal instrument status is not proof that the target material formed. The result is a reproducible, bounded virtual laboratory benchmark, not a general predictive digital twin or evidence of real synthesis performance.

Combine ChemLab-Assets for available instrument behavior and geometry; a separately authored material/sample simulator; Laya for routine typed decisions; an LLM for planning and difficult diagnosis; the user's existing ModernBERT Materials Encoder for retrieval; and LangGraph plus a permission-enforcing Tool Adapter. The former low-airflow hood scenario becomes an optional instrument integration smoke test, not the scientific centerpiece.

## 2. What has actually been checked

The referenced conversation was retrieved, but its assistant responses resolve to content-reference placeholders. Only the user requests are readable. No unseen previous technical claims are treated as evidence. The destination directory contains only `OSI-hack.md`, a folder-index note; no existing proposal was found or modified.

The official [ChemLab-Assets dataset card](https://huggingface.co/datasets/MBZUAI/ChemLab-Assets) documents GLB/URDF assets, contracts and local Python runtimes. Its hood example separates agent and evaluator sessions and advances time with `sim.step`. Contracts describe state visibility, capabilities and limitations. The card labels the dataset CC-BY-NC-4.0. These observations establish an integration starting point, not coverage of every proposed furnace or XRD behavior.

Direct protocol-file retrieval failed through the available browser, and command-line downloading failed because network DNS was unavailable. No runtime was installed or executed in this session. The precise `describe/read/invoke/poll` signatures, asynchronous behavior, available fault hooks, selected furnace/XRD contracts, and their verification levels remain integration checks. An asset appearing in a catalog does not prove the needed behavior exists.

The official [pymatgen diffraction documentation](https://pymatgen.org/pymatgen.analysis.diffraction.html) supports calculating powder diffraction from structures. [COD](https://cod.psdi.ac.uk/) supplies public-domain crystal structures. These enable reference-pattern construction; neither supplies reaction kinetics. [Laya's model card](https://huggingface.co/convaiinnovations/laya/blob/00c37c405e3c3ad73ee070227614c89cda06b99e/README.md?code=true) and [runtime repository](https://github.com/WhoIsMrSentry/laya-system-one) provide the System 1 integration starting point. Pin versions before implementation; performance and calibration must be measured locally.

## 3. Scientific scenario and boundaries

Use the ideal reaction:

**MgO + Al₂O₃ → MgAl₂O₄**

A mixed powder batch receives a nominal thermal program. Its first powder XRD scan contains precursor reflections and weaker-than-expected target reflections. The agent must distinguish incomplete reaction from a composition ceiling and from an unreliable scan. Exact thermal windows must be selected from an openly accessible, reviewed synthesis source before freezing the benchmark; this proposal deliberately does not prescribe an unverified laboratory recipe.

Why this system: a small crystalline phase library, transparent elemental balance, and a diffusion/contact-limited reaction model are enough to create meaningful decisions. Raising the temperature indiscriminately cannot fix a batch missing one precursor. A poor mount can alter intensities without changing chemistry. A controller setpoint does not establish the sample's thermal history.

| Hypothesis | Discriminating evidence available through tools | Appropriate response |
|---|---|---|
| Insufficient effective thermal dose | Independent sample-temperature witness and elapsed dwell disagree with the intended program; composition records are plausible | Correct the program or temperature offset after checking instrument status; regrind and re-anneal a tracked aliquot |
| Incorrect precursor molar ratio | Weighing ledger, precursor identity/purity basis, and optional independent composition check disagree with the intended 1:1 oxide molar ratio | Prefer a correctly weighed fresh batch in MVP; do not pretend extra annealing fixes elemental imbalance |
| Poor mixing/contact | Adequate thermal witness and ratio, but precursor persistence; a paired regrind treatment improves conversion | Regrind and repeat a bounded thermal treatment; use only in stretch scope |
| Mounting/measurement artifact | Standard scan or remount changes offset/intensity behavior; repeat aliquot conflicts with the first scan | Recalibrate or remount, then rescan before chemically altering the sample |
| Wrong atmosphere configuration | Logged gas configuration and independent witness conflict | Stop and validate gas path if required by protocol; do not infer it from weak target peaks alone |
| Wrong sample or unknown phase | Sample lineage mismatch or large unexplained full-pattern residual | Quarantine, repeat identity checks, or abstain; never force a known-phase answer |

For MgAl₂O₄ from these oxides, atmosphere is not assigned a fictitious “oxygen leak destroys the target phase” rule. It can be a checked confounder with no material effect in the MVP. A genuinely atmosphere-dependent chemistry, such as a multivalent oxide system, is a separate stretch scenario requiring its own published basis and model. The hypothesis space includes “none of the modeled causes.”

## 4. Ownership of simulation capabilities

| Layer | Reuse or build | Evidence boundary |
|---|---|---|
| Instrument geometry | Reuse selected ChemLab GLB/URDF after checking files | Geometry and animation do not establish process fidelity |
| Instrument runtime | Reuse only operations and state explicitly declared and implemented by the selected contract/runtime | Verify heating, readouts, preconditions, timing and fault hooks individually; do not infer them from the hood example |
| Runtime access | Wrap local Python sessions behind a tool service | HTTP/MCP transport is our wrapper, not an assumed hosted ChemLab service |
| Material evolution | **Build** stoichiometry and thermal-dose model | Not claimed as built into ChemLab |
| XRD acquisition and artifacts | **Build/import** CIF reference generation, mixture rendering and acquisition effects | Any asset-side XRD behavior must be separately inspected; our spectrum generator remains separately labeled |
| Sample lifecycle | **Build** immutable IDs, aliquots, mass, mount state, location and treatment history | A visual object transfer is not validated robotic manipulation |
| Faults and scoring | **Build** scenario injector and evaluator; reuse a native fault only if its exact support is verified | No blanket assertion of native fault coverage |

If no suitable high-temperature runtime exists, implement a small custom furnace state machine with explicit `custom simulator` labeling. Do not misuse a low-temperature drying oven as a synthesis furnace. The demo must remain runnable with this fallback while clearly reporting how much ChemLab behavior was actually reused.

## 5. Reproducible material and XRD engine

### Controlled reaction model

Store hidden phase amounts in moles. With initial `n_MgO` and `n_Al2O3`, define `extent = min(n_MgO, n_Al2O3) * conversion`, where `conversion` lies in [0,1]. Subtract extent from both reactants and add extent to spinel. Check Mg, Al and O conservation after every transition; derive mass fractions from formula masses.

Use an interpretable conversion law such as `conversion = 1 - exp(-contact * dose)`, with dose accumulated from a documented monotonic temperature-rate lookup over **sample temperature**, not merely setpoint. The lookup and contact factors are authored benchmark parameters. They are not measured kinetic constants, equilibrium predictions, or a phase diagram. Bound the temperature range; reject unsupported extrapolation. Composition deficiency naturally imposes residual precursor even at complete conversion. Regrinding changes contact for subsequent reaction but cannot create missing elements.

Keep controller temperature, independent witness observations, and latent sample temperature separate. A sensor-bias fault must affect the instrument observation/trajectory consistently; a scenario-service bias must be labeled as custom. Measurement-only faults leave material state untouched. Replaying an event must not apply conversion twice.

### Pattern pipeline

1. Curate CIFs for spinel MgAl₂O₄, periclase MgO and corundum Al₂O₃ from COD. Confirm phase identity, occupancy, provenance and duplicates. Exact accession IDs remain a build gate; do not invent them.
2. Generate fixed-radiation reference peak lists using a pinned pymatgen version. Record wavelength, angular range, grid, broadening and normalization in a manifest.
3. Render mixtures with explicit phase scale factors, broadening, background, small angular offset, preferred-orientation perturbations and seeded counting noise. **A linear combination of individually normalized patterns is an illustrative signal model, not calibrated quantitative phase analysis.**
4. Fit constrained nonnegative phase scales plus bounded nuisance parameters; inspect multiple diagnostic reflections and the full-pattern residual. Report ambiguity and detection limits. Mask labels and hidden fractions from the agent.
5. Calibrate any phase-fraction proxy on synthetic mixtures with held-out nuisance settings. In MVP report target evidence, precursor evidence and estimated crystalline composition with uncertainty, not an asserted Rietveld mass fraction.

The simulator generates a new observation only after a valid acquisition action. It cannot return a “success spectrum” merely because the agent chose a repair. Realistic amorphous fractions, microstrain, microabsorption and nonstoichiometric solid solutions are excluded from MVP and named limitations.

## 6. Architecture and control flow

```text
3D viewer + XRD chart + evidence / workflow panel
                    |
        LangGraph persistent episode state
                    |
      Laya routine gate <-> LLM diagnostic planner
                    |                |
                    |       ModernBERT Materials RAG
                    |                |
        Tool Adapter / schema + permission + safety checks
                    |
        +-----------+--------------------------+
        |                                      |
ChemLab local sessions                 Custom sample/material/XRD service
(contract-supported behavior)           (explicit simulation model)
        +---------------- simulation clock / event log --------+

Evaluator-only process: scenario setup, fault truth, independent scoring
No evaluator tools, credentials or truth fields in the agent tool registry.
```

LangGraph nodes: observe → validate measurement → retrieve evidence → update hypotheses → choose discriminating test → execute/poll → reassess → propose repair → verify → report or abstain. Persist checkpoints, action IDs, observations, retrieved passage IDs and concise decision rationales. Log hypotheses and predicted outcomes rather than requesting hidden chain-of-thought.

**Laya/System 1:** classify scan validity status from structured summaries, select among a short allowlisted next-step menu, detect uncertainty/escalation and route routine polling. Its probabilities are uncalibrated until validated on development episodes. Never give it sole authority to certify success or bypass interlocks. Set thresholds on held-out development data; test option-order sensitivity.

**LLM/System 2:** construct and revise competing explanations, request RAG passages, predict what each check would discriminate, interpret conflicting evidence, and propose a bounded corrective experiment. Numerical phase matching and mass balance remain deterministic services.

**ModernBERT Materials Encoder/RAG:** reuse the user's checkpoint as a retrieval encoder if its pooling, tokenizer and embedding interface support that use. Confirm the actual checkpoint and corpus; no access or quality is assumed. Index curated synthesis notes, CIF provenance, instrument limits and artifact guides. Return passages with source, page/section and document hash. Do not mix this encoder's role with Laya's decision role or train on held-out fault solutions. Compare against a basic retrieval baseline.

**Tool Adapter:** enforce argument schemas, units, finite values, allowed ranges, role filtering, sample lineage, prerequisites, budgets and idempotency. Simulation-time stepping belongs to the scheduler/evaluator; the agent may request a bounded wait but cannot directly advance time arbitrarily to defeat the budget. Safety checks are deterministic even when both models approve an action.

## 7. Tool interface sketch

Everything below is a **proposed adapter interface**, not a verified ChemLab native signature. The integration task is to map operations to inspected runtime contracts; unsupported operations return `unsupported_capability` or use an explicitly identified custom backend.

```python
# Agent-facing wrapper; no evaluator session is accessible here.
lab.describe(asset_id="furnace_01")  # filtered state + supported action schemas
lab.read(asset_id="furnace_01", fields=["status", "controller_temperature_K"])
job = lab.invoke(
    asset_id="furnace_01", capability="run_validated_program",
    arguments={"program_id": "approved_spinel_v1", "sample_id": "batch17-a"},
    idempotency_key="episode42-action08")
lab.poll(job_id=job["job_id"])  # normalized pending/completed/failed

samples.history(sample_id="batch17-a")
measurements.acquire_xrd(sample_id="batch17-a", protocol_id="powder_v1")
analysis.match_phases(scan_id="scan09", library_id="spinel_library_v1")
knowledge.search(query="spinel precursor persistence vs composition imbalance", k=4)
```

A tool result includes `episode_id`, `action_id`, `backend`, `sample_id`, `sim_time_s`, status, units, evidence IDs and errors. Expected errors include invalid state, hot sample, unknown sample, duplicate action, invalid arguments and budget exceeded. Failed calls must not silently change state.

```python
# Evaluator/scheduler process only — custom API sketch.
world.reset(seed=42, scenario_manifest="thermal_bias_v1")
world.inject_fault(kind="sample_temperature_bias", parameters={"delta_K": -80})
# Advance through the scheduler; delegate to native sim.step only where supported.
world.advance(dt_s=10)
score = evaluator.score(event_log, hidden_truth, frozen_rubric)
```

Use `describe/read/invoke/poll` as a normalization vocabulary. First inspect the upstream protocol and tests, including whether jobs are actually asynchronous and which principal may call `sim.step`. Do not expose principal registration, fault injection, file access to manifests, or evaluator readouts to the agent. UI evaluator access is a separate authenticated mode.

## 8. Live demo: six-minute main path and branches

**0:00–0:40 — A failed material, not an alarm.** Show a tracked powder batch moving through a virtual furnace and onto an XRD stage. Display the first synthetic scan over reference ticks: residual precursor peaks, weak target evidence. Show the user goal, cost budget and simulation badge. Physical hours are accelerated simulation time, explicitly labeled.

**0:40–1:40 — Competing explanations.** Laya routes an uncertain diagnosis to the LLM. The agent retrieves materials evidence and lists at least composition imbalance, insufficient thermal dose and measurement artifact, with a predicted distinguishing observation for each. It checks sample lineage and scan metadata before altering chemistry.

**1:40–2:40 — A discriminating sequence.** Run the standard/remount check to test measurement reliability; audit mass-to-mole calculations and precursor IDs; retrieve an independent temperature witness. In the main episode, the standard is acceptable, the ratio is correct, and the witness indicates an insufficient effective dwell/temperature despite the normal controller display. The witness is noisy, not a direct root-cause label.

**2:40–4:30 — Bounded repair.** The agent proposes a corrected validated program and regrinding of a reserved aliquot, predicts reduced precursor evidence, checks allowed thermal limits and executes. Instrument state gates loading, heating and unloading; the material engine integrates the resulting sample history. The 3D animation follows accepted events. It cannot override them.

**4:30–5:30 — Independent verification.** Acquire a fresh scan under a fixed verification protocol and analyze the full pattern. Check precursor evidence, target support, fit residual, lineage and instrument status. An evaluator-only panel reveals the injected fault and final hidden composition after the agent commits its report.

**5:30–6:00 — Change one cause.** Reset to a prepared alternate seed with similar first-scan appearance. Show that the agent now chooses a different action; never describe a prerecorded trace as a live run.

| Branch | Reveal through observations | Expected workflow difference |
|---|---|---|
| Composition-deficient batch | Thermal witness passes; ledger/composition check fails | Make a corrected fresh batch; preserve failed batch provenance. Re-annealing alone must fail the evaluator |
| Measurement-only fault | Standard fails or remount resolves apparent failure | Recalibrate/remount and reacquire; avoid unnecessary synthesis |
| Adequate heat, poor contact | Thermal and composition checks pass; regrind aliquot responds | Stretch: paired aliquot test before committing the full batch |
| Unknown impurity / contradictory evidence | Persistent unexplained residual | Report unresolved diagnosis and request an additional measurement; do not claim recovery |
| Atmosphere discrepancy | Witness differs from configuration | Validate the gas path; for MVP spinel, do not automatically call this the chemical root cause |

Include one negative-control run in rehearsal: the same repair applied to the wrong cause must not pass. Choose checks for expected information gain per cost over the current hypothesis set; the displayed policy may use an interpretable heuristic rather than an unvalidated claim of optimal Bayesian planning.

## 9. Scope and delivery plan

**MVP:** one chemistry; three crystalline phases; one tracked batch with aliquots; three single-fault families (thermal history, stoichiometry, measurement); a no-fault control; seeded noise; deterministic constrained phase matching; actual Laya and LLM calls; the user's encoder when accessible; LangGraph checkpoints; a small 3D scene; evaluator-separated replay. A deterministic router may keep development moving, but must be labeled as a fallback and excluded from claims about Laya.

**Stretch:** mixing/contact faults, paired experiments, two simultaneous faults, unknown-phase challenge, validated atmosphere-sensitive second chemistry, measured open XRD scans with redistribution rights, improved quantitative fitting, and deeper native runtime coupling. Do not add general robot control, real hardware execution or unconstrained materials discovery to the hackathon scope.

| Timebox | Work and exit gate |
|---|---|
| Day 1 morning | Inspect selected contracts/runtime/tests and upstream protocol; record revision and licenses; choose supported equipment or custom fallback. Confirm encoder access and Laya inference on demo hardware |
| Day 1 afternoon | Select CIF accessions and synthesis evidence; freeze manifest. Build mass-conserving material model, sample ledger, three fault families and XRD renderer |
| Day 2 morning | Implement filtered adapter, scheduler, structured errors, replay and numerical phase matching; test invariants and role boundaries |
| Day 2 afternoon | Connect RAG, Laya and LLM through LangGraph; implement hypothesis/test/repair/verify loop; freeze decision thresholds on development seeds |
| Day 3 morning | Attach GLB viewer, runtime event overlays, scan chart and evidence timeline; run blinded evaluations and ablations |
| Day 3 afternoon | Freeze versions, rehearse alternate branches, cache public data, record a clearly labeled fallback video, export report and trace |

If time contracts, reduce visual animation and fault count before weakening role separation or verification. Gate the public demo on two independently successful branches and one correctly rejected false recovery. This schedule assumes a small team and existing model access; staffing and compute remain unconfirmed.

## 10. Evaluation and ablations

Freeze development/test scenario seeds and parameter ranges before policy tuning. Use paired episodes across systems, vary hidden fault severity and nuisance parameters, and keep a test-only ambiguous/unknown condition. An initial target is 30 development episodes and 100 held-out episodes across controls and fault families; this is a proposed evaluation budget, not an achieved result. Split by fault configuration and severity rather than changing only random noise. Report per-family counts and bootstrap intervals; use repeated agent runs for stochastic variability.

| Metric | Definition / safeguard |
|---|---|
| Verified recovery rate | Hidden target crystalline mass fraction meets a preset benchmark threshold (e.g. ≥0.90), conservation and lineage checks pass, and a fresh verification scan supports recovery; the threshold is an authored benchmark criterion, not a universal purity standard |
| Root-cause diagnosis | Correct fault family and supporting observations against evaluator truth; score separately from recovery |
| False-success rate | Agent declares success when hidden state or verification requirements fail; denominator is all success declarations |
| Diagnostic efficiency | Measurement cost and count before a justified diagnosis; compare against an enumerated bounded-policy reference where feasible |
| End-to-end cost | Total Laya, encoder, retrieval and LLM latency/cost, tool time, simulated thermal cost and failed calls; include all routing overhead |
| Safety and tool correctness | Preconditions violated, unauthorized calls attempted/succeeded, duplicate effects, invalid units, and untracked sample transitions |
| Uncertainty behavior | Coverage versus error for abstention; calibration/Brier score for defined typed decisions, reliability by fault family |
| Scientific evidence quality | Cited passages support the selected test/repair; full-pattern residual and unexplained peaks are reported; blind rubric review |
| Reproducibility | Same pinned simulator, manifest, seed and actions reproduce observations and score; agent variability reported separately |

Run paired ablations: full hybrid; LLM-only with identical tools/budgets; Laya replaced by deterministic routing; no materials RAG; generic retrieval instead of the materials encoder; no measurement-validity check; fixed “always re-anneal” repair; and no independent verification. For the last ablation, compute hidden-state outcomes anyway to expose false success. Compare phase matching against a single-peak shortcut. Keep the same evidence availability, maximum actions and timeout for fair comparisons. Do not claim the hybrid is faster or better until measured; include option-order and truncated-context stress tests for Laya.

## 11. Frontend and 3D presentation

Use a lightweight browser GLB viewer (for example Three.js) as a display client, not the simulator of record. URDF is optional for articulated links; no Isaac/robot physics requirement in MVP. Load only the chosen assets and obey their axis metadata. Joint animation must bind to supported state or to clearly marked custom scene events.

Arrange four synchronized views: laboratory scene with sample IDs and temperature/status badges; XRD pattern with reference ticks and residual; LangGraph progress with System 1/System 2 routing; and an evidence/hypothesis panel showing uncertainty and the next test's purpose. Provide simulation-time versus wall-time indicators, a manifest/version drawer, pause/replay and a separate post-run evaluator reveal. Use generic custom geometry when an appropriate asset is unavailable and label it. Never let the normal agent screen display latent phase fractions or fault names before commitment.

## 12. Risks, anti-shortcuts and assumptions

- **Circular success:** the agent cannot write phase fractions or success labels. A separate evaluator reads hidden state and acquisition events. It does not score the eloquence of the final explanation.
- **Fault leakage:** hide scenario names, file paths, IDs encoding causes and evaluator channels; vary visible IDs. Audit prompts, RAG corpus and logs for answer leakage.
- **Toy-model overclaim:** publish the transition rules and label all spectra synthetic. This evaluates reasoning inside a controlled model, not transfer to a real laboratory. Withheld model perturbations expose overfitting but do not establish physical validity.
- **XRD non-identifiability:** overlapping reflections, preferred orientation and incomplete phase libraries require residual checks and abstention. XRD alone cannot prove bulk chemical purity or exclude amorphous material.
- **Unjustified kinetic recipes:** a materials reviewer must approve the chosen references, temperature domain and reaction assumptions before they become benchmark facts. Otherwise keep numerical thermal settings explicitly synthetic.
- **Model access:** the exact ModernBERT checkpoint, embedding interface, training data and license are user-supplied assumptions. No assumption that it already performs reliable phase diagnosis. Laya's short-menu routing needs local calibration.
- **Permission fidelity:** session separation must be enforced server-side, including websocket/event filtering and error messages. Test direct unauthorized calls, not just hidden buttons.
- **License/data availability:** ChemLab's noncommercial license needs review for the event and any later commercial reuse. Attribute assets; retain each file's license. Do not bundle proprietary ICDD/ICSD patterns without rights.
- **Live failure:** cache pinned assets/CIFs and references locally, preflight model loading, cap retries and timeout. A replay/video fallback must display its status clearly.

## 13. Source manifest and outstanding verification gates

Sources were consulted on 19 October 2026. Links below are starting points, not a claim that all integration code or individual structures have been verified.

1. [MBZUAI/ChemLab-Assets official dataset and card](https://huggingface.co/datasets/MBZUAI/ChemLab-Assets): runtime/geometry starting point and asset license. Pin an exact dataset commit during implementation.
2. [Upstream protocol path](https://huggingface.co/datasets/MBZUAI/ChemLab-Assets/blob/main/skills/chemistry-lab-assets/references/protocol.md): referenced by the card; inaccessible in this session. Inspect locally before writing the native adapter.
3. [Crystallography Open Database](https://cod.psdi.ac.uk/): public-domain structure source. Record exact accession, original publication, download date and SHA-256 for every selected CIF. The phase-specific shortlist is still pending.
4. [pymatgen diffraction API](https://pymatgen.org/pymatgen.analysis.diffraction.html): structure-to-reference-pattern calculation. Pin package version and generated reference files.
5. [Laya model card, pinned page](https://huggingface.co/convaiinnovations/laya/blob/00c37c405e3c3ad73ee070227614c89cda06b99e/README.md?code=true) and [runtime repository](https://github.com/WhoIsMrSentry/laya-system-one): typed-decision component; verify local signatures, checkpoint revision and license.
6. [LangGraph official documentation](https://docs.langchain.com/oss/python/langgraph/overview): proposed orchestration framework; exact version/API selection remains an implementation task and was not audited here.

Before implementation acceptance, add: selected instrument IDs and inspected capability table; upstream runtime revision and passing relevant tests; phase-specific CIF accession list; an openly accessible synthesis reference with section/page evidence; checkpoint hashes; a lockfile; scenario JSON schemas and seeds; evaluator rubric; and an attribution/redistribution manifest. The reproducibility bundle should contain one command to reset and run an episode, an independent scoring command, and saved traces for all three main branches.

**Deliverable promise:** a reviewable materials troubleshooting workflow with explicit physics limits, traceable evidence, distinct corrective branches and independent verification. All new material behavior is identified as authored simulation rather than attributed to ChemLab-Assets.
