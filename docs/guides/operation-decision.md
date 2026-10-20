# Online next-operation decisions

This implementation adds action selection to the existing LangGraph. The offline decision benchmark is deferred at the user's request. There is no new task generator, all-option label replay, model-ranking study or calibration experiment.

## Execution path

Observe → diagnostic candidates → **operation decision** → specialist router → procedure → deterministic gate → native execution → fresh observation → verification.

The existing router still chooses a specialist. The operation adapter chooses a laboratory action. Both can call the same local OpenJev service through independent prompts and trace records; no additional model weights are needed. The gate remains the only path to tool execution.

When enabled, diagnosis returns `OperationOptions` with hypotheses and one to four distinct `ActionCandidate` records. Each has a stable candidate ID, a typed intent, rationale and observation citations. The diagnostic expert does not return the final selected intent. A single candidate is allowed when only initial inspection is justified. Candidate capability names, argument schemas and evidence IDs are checked before selection; native preconditions remain enforced by the gate and executor.

The chooser sees public observations, target/policy, hypotheses, capabilities, and candidate records. It returns a candidate ID plus a complete probability distribution. The chosen ID must be supplied and have maximum probability; missing/invalid probability mass is rejected. The graph copies that candidate's intent into the procedure input. Procedure cannot replace it, and the gate independently checks the match even if a profile omits specialist-output validation. The decision's revision is checked against the current lab revision before handoff and execution. Candidate payloads are copied so a backend cannot change arguments in place.

Titania observations now identify the current aliquot's completed thermal receipt explicitly. After replacement, an old aliquot's scan is not exposed as the new aliquot's current measurement. Candidate prompting states inspection presence explicitly, and the titania profile rejects reinspection of unchanged valid evidence.

Selection failures stop unresolved with an explicit error; there is no silent operation fallback to an expert or rule policy. Existing specialist-router fallback remains separately logged. Local operational calls count against the shared per-episode model-call limit, as do cloud calls and specialist format repairs; operation call count is also reported separately. Router calls retain the existing separate routing counters. Action/time limits still apply. Every attempted selection is logged, including errors.

OpenJev probabilities are raw T=1 token probabilities conditioned on the supplied candidate letters. They are **not calibrated confidence**, and cannot indicate whether diagnosis omitted a better action. A singleton menu necessarily normalizes to 1.0. No new probability threshold is used to claim reliability. Inspection/measurement alternatives can be proposed explicitly; this version does not implement a general human-escalation protocol.

## Run the titania demo

Defaults retain the prior workflow. Enable the new node explicitly:

```bash
# No model/API calls: scripted candidates and scripted selection are test doubles.
python -m bench.titania --operation-decision scripted \
  --output results/titania-operation-scripted-smoke

# Same local OpenJev service for specialist routing and action selection.
# Candidate generation and verification remain explicitly scripted test doubles.
python -m bench.titania --router openjev --operation-decision openjev \
  --output results/titania-operation-local-smoke

# Real DeepSeek generates candidates/procedure/verification records.
# Existing shared token ledger meters all SiliconFlow requests.
python -m bench.titania --backend deepseek --router openjev \
  --operation-decision openjev --output results/titania-operation-real-smoke
```

`--operation-decision deepseek` selects the existing budgeted SiliconFlow expert transport as the action chooser instead. `--operation-decision off` retains direct diagnostic intent selection. Never delete the durable budget ledger to restart the 5M allowance. No OpenRouter requests are made.

The two titania branches remain independent reset worlds. A wrong thermal history calls for reference treatment of a retained dried sister aliquot followed by XRD; a truncated scan after correct treatment calls for a full scan without replacement. Physics/chemistry boundaries remain as declared in the [titania scenario](../demos/titania/scenario.md).

## Browser demonstration

The local drying UI adds two explicit modes:

- **Scripted candidates + operation test policy (regression)**: no model requests.
- **Scripted candidates + local OpenJev router and operation chooser**: two separate uses of the local service; specialists remain test doubles.

The graph/timeline includes `Choose operation`, and a separate panel displays candidate actions, selected ID and raw probabilities. Historical recording and full reset/action replay continue to work. Non-local expert configurations are rejected before output/transport creation. The browser does not enable cloud experts or the titania scene; real titania specialist runs use the CLI above.

## Implementation and validation

- `app/models.py`: candidate-set and operation-decision records.
- `app/operation_decision.py`: local OpenJev, budgeted-expert and explicitly scripted adapters.
- `app/graph.py`: mandatory candidate-selection handoff when enabled, including budget and stale-state checks.
- `bench/titania.py`, `app/demo.py`: independently configured specialist router and operation chooser.
- `tests/test_operation_decision.py`: selection, handoff/gate, malformed candidates, probabilities, freshness, mutation isolation, budgets/deadline and UI/replay regressions.

The [workflow illustration](../plans/assets/workflow-operational-decision-v1.png) depicts the proposed architecture including the deferred benchmark panel. Its main online path is now implemented as an opt-in mode; the offline panel remains unimplemented. See [execution evidence](../evidence/operation-decision-validation.md) for the observed runs and limitations.

Release review: [Astra findings, fixes and final acceptance](../evidence/operation-feature-review.md). Run summaries separate operation-disabled and operation-enabled configurations and retain the chooser backend.
