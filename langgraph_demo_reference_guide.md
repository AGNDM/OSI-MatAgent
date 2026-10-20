# LangGraph Laboratory Demo: Implementation References and Build Notes

**Companion to `materials_lab_agent_proposal.md` · checked 19 October 2026**

This guide prioritizes primary sources and identifies what to implement from each. Documentation access was verified; local execution and package compatibility were not. Pin versions during setup rather than combining snippets from different releases. The application designs below are our proposed glue code, not upstream examples or completed implementation.

## 1. Read in this order

| Priority / timebox | Read | Concrete result |
|---|---|---|
| First 15 minutes | R1–R3: ChemLab repository, protocol and example | Know how to instantiate, observe, invoke and advance one asset |
| Next 20 minutes | R4–R6: Graph API, workflows and multi-agent patterns | Draw three role nodes and their typed handoffs |
| Next 15 minutes | R7–R10: subgraphs, persistence, streaming, structured output | Decide state ownership, trace filtering and schema validation |
| Next 15 minutes | R13–R15: chosen decision implementation | Run one typed routing request; record exact backend |
| Before benchmark freeze | R11–R12 and R16–R18 | Isolated test runner, deterministic score and error rubric |

Reading is bounded. Use the relevant sections and runnable examples; there is no need to complete every tutorial before starting.

## 2. Environment: inspect the real protocol before wrapping it

**R1 — [ChemLab-Assets official repository](https://github.com/mbzuai-nlp/ChemLab-Assets)** and **[dataset](https://huggingface.co/datasets/MBZUAI/ChemLab-Assets)**. Start from the documented example, selected runtime, catalog and corresponding tests. Deliverable: one native action trace and code/data revision manifest. The full asset collection is not required for one episode.

**R2 — [ChemLab logical call protocol](https://github.com/mbzuai-nlp/ChemLab-Assets/blob/main/skills/chemistry-lab-assets/references/protocol.md)**. Implement visibility filtering, observation revisions and request deduplication. Reads do not advance time. A successful setpoint command does not establish physical stabilization. Inspect operation status before proceeding. Host identity controls access; callers cannot grant themselves evaluator privileges. Deliverable: wrapper tests for stale observations, denied access and repeated requests. These are protocol requirements; verify selected adapters implement them before relying on the behavior.

**R3 — [Official fume-hood quickstart](https://github.com/mbzuai-nlp/ChemLab-Assets/blob/main/examples/quickstart_fume_hood.py)**. Use it as the initial smoke-test path identified in the README; inspect the file locally and run it before adapting it. The final scenario should involve materials evidence. The [contract reference](https://github.com/mbzuai-nlp/ChemLab-Assets/blob/main/skills/chemistry-lab-assets/references/contract.md) is a follow-on inspection target for capability bindings and verification levels; its full text was not audited for this guide.

## 3. LangGraph: the official references that map to our design

**R4 — [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api).** Read state, reducers, nodes and conditional edges. Use explicit state fields and route validation. The documentation warns that graph private channels can still appear in streams; never store evaluator truth there. Deliverable: a compiled graph with a finite route map and explicit termination paths.

**R5 — [Workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents).** Focus on routing, orchestrator-worker and evaluator-optimizer patterns. Our design combines routing with bounded verification; it does not require dynamic worker creation. Deliverable: a graph sketch explaining which decisions are model-driven and which are deterministic.

**R6 — [Multi-agent patterns](https://docs.langchain.com/oss/python/langchain/multi-agent).** Read the context and role-separation discussion. Use it to define each agent's allowed evidence and responsibilities. Deliverable: three role prompts and three output schemas. Several nodes sharing a prompt are not meaningful specialization.

**R7 — [Subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs).** Useful if a role needs its own small loop or message context. Start with ordinary role nodes; add a subgraph only where needed. Deliverable: one explicit parent/role state mapping. Avoid introducing a second layer of agent orchestration before the main loop works.

**R8 — [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence).** Read checkpoint and thread behavior. Assign a unique thread per independent run. Our additional requirement is that simulator state must match graph state: a saved conversation cannot restore a physical-world side effect. Deliverable: isolated runs and reset/replay, plus a documented resume limitation.

**R9 — [Streaming](https://docs.langchain.com/oss/python/langgraph/streaming).** Use state updates/events to drive the timeline. Emit selected public fields and concise evidence references rather than dumping whole state. Deliverable: a UI event for role entry, tool result, routing and termination. Token streaming is optional.

**R10 — [Structured output](https://docs.langchain.com/oss/python/langchain/structured-output).** Read provider strategy versus tool strategy and validation behavior. Our benchmark must record which mechanism each model uses. Deliverable: typed role responses, one bounded format-repair policy, and retained raw output for diagnosing invalid responses.

**R11 — [Testing LangGraph applications](https://docs.langchain.com/oss/python/langgraph/test).** Use fresh state/checkpoint context and deterministic test dependencies. Deliverable: tests for invalid route, stale action proposal, pending operation, repeated mutation and budget termination. These are consequential integration tests, not checks that simply reproduce implementation details.

**R12 — [LangSmith evaluation](https://docs.langchain.com/langsmith/evaluation).** Useful for datasets, evaluators and experiment comparison if the team already uses it. Local JSONL plus a scoring script is sufficient for this event. Deliverable: run-level records and reproducible summaries; do not spend the deadline setting up hosted observability.

## 4. Decision-model references: choose an identity, not a name

**R13 — [ejhshen/OpenJev](https://github.com/ejhshen/OpenJev).** Initial candidate: `shenjunhao/OpenJev-4B`, loaded through its own `DecisionEngine`. Inspect `examples/infer.py` and the inference guide linked by the README. Its documented setup uses a compatible CUDA stack; availability on the demo machine remains untested. Pin checkpoint and code revisions and preserve output probabilities.

**R14 — [razorback16/openjev](https://github.com/razorback16/openjev).** A separate implementation offering a System One server. Read deployment requirements if considering it; do not assume it shares dependencies, weights or performance with R13. This is a comparison reference, not a second required integration.

**R15 — [Laya model card](https://huggingface.co/convaiinnovations/laya) and [runtime repository](https://github.com/WhoIsMrSentry/laya-system-one).** Alternative if already runnable on available hardware. Use the same application-level routing contract, while adapting actual request/response fields. Keep decision-model inference separate from the user's Materials Encoder retrieval. Do not attribute one project's API, training or speed claims to another.

Backend-neutral output expected by our graph:

```json
{
  "backend_id": "repository + checkpoint revision",
  "choice": "diagnose",
  "probabilities": {"diagnose": 0.65, "prepare_action": 0.10, "verify": 0.10, "deliberate": 0.15},
  "latency_ms": 0,
  "fallback_used": false
}
```

Numbers are illustrative placeholders. Validate option keys and finite probabilities; preserve the raw response separately. A malformed response or timeout is an integration event, not a high-confidence choice. Never fabricate decision probabilities for a rule-based fallback. Display its backend as `rules` and probabilities as unavailable.

## 5. A small graph scaffold to adapt

This illustrates **our topology only**. It requires the real role, router, gate, executor and termination functions; it has not been run. Each role returns validated partial state updates. Keep actual lab writes inside `execute`, never inside routing functions.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class LabState(TypedDict, total=False):
    episode_id: str
    sample_id: str
    observation_revision: int
    observations: dict
    hypotheses: list
    evidence_refs: list
    intent: dict
    proposal: dict
    pending_operation: dict | None
    verification: dict
    route: str
    gate_result: str
    stop_result: str
    llm_call_count: int
    action_count: int
    terminal_reason: str | None

def build_graph(nodes, checkpointer):
    # nodes is an application-supplied mapping of validated functions.
    graph = StateGraph(LabState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)
    graph.add_edge(START, "observe")
    graph.add_edge("observe", "router")
    destinations = {
        "diagnose": "diagnostic",
        "prepare_action": "procedure",
        "verify": "verification",
        "deliberate": "deliberation",
        "stop": "report",  # code-enforced budget exit, not model choice
    }
    graph.add_conditional_edges("router", lambda s: s["route"], destinations)
    graph.add_edge("diagnostic", "router")
    graph.add_edge("procedure", "gate")
    graph.add_conditional_edges(
        "gate", lambda s: s["gate_result"],
        {"execute": "execute", "reject": "router", "stop": "report"},
    )
    graph.add_edge("execute", "observe")
    # Deliberation chooses a role directly; do not re-query the router here.
    graph.add_conditional_edges(
        "deliberation", lambda s: s["route"],
        {k: v for k, v in destinations.items() if k != "deliberate"},
    )
    graph.add_edge("verification", "termination")
    graph.add_conditional_edges(
        "termination", lambda s: s["stop_result"],
        {"continue": "router", "stop": "report"},
    )
    graph.add_edge("report", END)
    return graph.compile(checkpointer=checkpointer)
```

Before using this scaffold:

- `router` and `gate` check action/call/deadline limits before invoking a model or tool. Set a separate graph recursion cap.
- `deliberation` validates its route and returns unresolved on an invalid response after the shared retry limit.
- `execute` resolves a pending operation first; it must not create a duplicate mutation on each pass.
- `observe` reports observations only. Scenario reset occurs in the test harness before invocation.
- Clear consumed intent/proposal fields explicitly after their intended lifecycle. Bind proposals to an observation revision and reject stale ones.
- Keep the external evaluator outside `nodes`. The graph's final report is an artifact to score, not an authority.

Recommended role records: `HypothesisSet(hypotheses, supporting_ids, contradicting_ids, next_check)`; `ActionProposal(capability_id, arguments, request_id, expected_revision, postcondition)`; `VerificationReport(status, evidence_ids, missing_evidence)`. Validate these records using the same schema across model backends.

## 6. Benchmark and failure-analysis foundations

**R16 — [Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657)**, Cemri et al. Read the failure taxonomy and annotation methodology. It motivates distinguishing system design, coordination and verification failures instead of attributing every failure to model intelligence. Our E1–E9 labels are a simpler laboratory-specific adaptation. Deliverable: a codebook with one positive and one boundary example per label; reviewers cite exact trace events.

**R17 — [τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://arxiv.org/abs/2406.12045)** and [original repository](https://github.com/sierra-research/tau-bench). Borrow state-based evaluation and repeated-trial reliability. Do not import its domain scores or claim our pilot is τ-bench. Deliverable: evaluator assertions, isolated resets and a repeat-consistency column. “Both repeats passed” differs from “one of two attempts passed.”

**R18 — [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)**. Background reading for a single-agent observe/act baseline; not inspected in full in this revision. Deliverable, if time permits: one baseline using the same tools and overall budgets. There is no need to expose private model reasoning; log action rationales and cited observations.

## 7. Evaluation artifacts you can create immediately

Create a config manifest before running any benchmark:

```yaml
# Illustrative schema, not executable framework configuration.
experiment_id: pilot_v1
agent_model: EXACT_PROVIDER_MODEL_ID
agent_model_revision: RECORD_IF_AVAILABLE
roles_share_model: true
decision_backend: EXACT_REPOSITORY_AND_CHECKPOINT
router_mode: decision  # alternate: deterministic
prompt_version: FROZEN_HASH
scenario_version: FROZEN_HASH
retrieval_corpus_version: FROZEN_HASH
limits:
  lab_actions: 6
  llm_calls: 12
  episode_seconds: 180
  format_repairs: 1
  graph_steps: 60
```

Record generation parameters and per-role input/output caps in the real manifest. Choose caps in development and use the same values across the main model comparison; document unsupported parameters rather than forcing an invalid provider request.

For every event, save run/case/repeat ID, role, model ID, timestamp, observation revision, prompt/schema hash, evidence references, request/response or their artifact paths, operation status, token usage, elapsed time, error code and fallback status. Save secrets nowhere. Keep evaluator records in a separate artifact not available to agents.

Generate three outputs:

1. **`summary.csv`:** configuration, attempted/completed runs, family success, recovery, false success, abstention, consistency, latency and total cost.
2. **`errors.csv`:** run/step, primary error, component, evidence, downstream labels and reviewer agreement status.
3. **`traces/`:** two short case studies with the same starting task, different model behavior and the first consequential divergence highlighted.

Do not evaluate only the final answer. Do not choose a “representative” trace solely because it flatters the preferred model. Include a failure and one case where the baseline matches or exceeds the hybrid when observed. If results are tied or noisy, report that directly.

## 8. What remains to be verified at kickoff

Exact model access and hardware, code/checkpoint revisions, selected native instrument behavior, scenario evidence, the existing Materials Encoder interface, provider schema compatibility and expected run cost remain local checks. New chemistry/XRD features are not necessary to complete the architecture and benchmark objectives. Keep the first day focused on a working graph and reliable test harness; the second day should produce evidence about that system.
