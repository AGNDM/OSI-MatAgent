# Architecture and development

## Execution path

1. **Discover:** `device_catalog` exposes public contracts; `device_selection` binds a proposed instrument for native tasks.
2. **Diagnose and propose:** general experts produce typed intents, operation candidates and procedures from current evidence.
3. **Choose:** the decision backend selects the route and operation using legal argmax. Candidate probabilities are conditional scores, not calibrated success estimates.
4. **Gate and execute:** schemas, scope, revision, intent and replay checks precede native invocation.
5. **Observe and verify:** fresh native evidence returns to the graph. Independent task predicates or the benchmark evaluator determine outcomes.

| Module | Responsibility |
|---|---|
| `app/graph.py` | LangGraph transitions, verification and optional execute boundary |
| `app/single_instrument.py` | Public instrument facade, predicates and evidence lifecycle |
| `app/native_lifecycle.py` | Native clock, async poll and cancellation adaptation |
| `app/operation_decision.py`, `app/decision_backend.py` | Candidate choice contracts and routing policy |
| `app/openrouter_jev.py` | Hosted Jev / Luna decision transport |
| `app/expert_config.py`, `app/token_budget.py` | Expert transport, request validation and durable metering |
| `bench/instrument_eval_langgraph.py` | Persistent graph adapter to external simulator callbacks |
| `bench/multi_disturbance_eval.py` | Environment-side disturbance injection and scoring |

The standalone solver owns one native instrument. The benchmark adapter instead yields actions to the external evaluator, which owns its simulator. Scene adapters are injected explicitly; task-specific reference sequences are not runtime agent inputs. Offline replay fixtures under `tests/fixtures/` exercise deterministic integration paths only.

## Checks

```bash
python tools/fetch_device_runtime.py
python -m pytest -q
```

When changing catalog inputs (registry, adapter, lifecycle or runtime manifests), rebuild the public contract cache before testing:

```bash
python -c 'from app.device_catalog import build_cache; build_cache()'
```

The cache contains native interface declarations, not benchmark answers or experiment results. `load_catalog` rejects stale source hashes. Tests use offline model doubles; do not put credentials in test fixtures.

The disturbance harness consumes declarative files under `cases/evals/`. Environment schedules and scoring expectations stay outside model prompts. Native `host.step` drives simulated time. Saved model outputs are never used as a runtime solution script.

## Contribution boundary

Commit reusable code, tests, configuration, task inputs and maintained documentation. Keep API traces, ledgers, per-case outputs, temporary analysis, credentials and intermediate figures under ignored `results/`. Publish only the final report and assets it references. Retain upstream licenses and attribution. Run tests and check relative documentation links before opening a PR.

Legacy scripted demos remain available for regression. They do not establish autonomous synthesis or multi-instrument material transfer.
