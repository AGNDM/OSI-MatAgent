# Native acceptance and task-conditioned knowledge

Date: 21 October 2026. **Historical source snapshot:** the later [broad lifecycle update](native-lifecycle-coverage.md) removes the four-device whitelist and supersedes its adapter limitations. Baseline commit: `b46a09a`; the implementation described here is uncommitted development work. This report records all ten live attempts across three separately frozen stages. It is development integration evidence, not a held-out benchmark, chemistry validation or evidence of routing superiority. The [machine-readable companion](../evidence/native-acceptance-and-knowledge.json) contains exact source hashes, task/oracle manifests, actions/readbacks, passage queries, citations, request usage and artifact hashes. Full raw traces and frozen source snapshots remain in ignored local `results/` directories named there.

## Approved scope

The user selected direct maximum-probability routing without low-confidence review, asked for diverse instrument choices and knowledge retrieval, deferred the scientific UI, and required that real execution not depend on hand-authored task solution scripts, including the old TiO2 example. A later instruction removed token-amount concerns for OpenRouter. Astra reviewed the design, implementation and revised accounting before the new paired runs.

The new live path is goal JSON → public device discovery → Jev device choice → generic native `InstrumentLab` → diagnostic candidates → Jev operation choice → procedure → deterministic gate → native execution/readback → direct wait or verification. The first expert sees all 194 registered device IDs/names/domains. It can search and describe up to four public contracts within six rounds before proposing candidates. Goals and constraints remain immutable. Jev receives candidate rationales and identity/domain/lifecycle summaries: it does **not** independently inspect the full detailed contract during device choice. Native compatibility and action validation remain host-enforced.

Default `argmax-v1` accepts a legal maximum, including diffuse probabilities, and stops on invalid/backend output. No threshold, deliberate escalation or silent fallback is active. Conditional choice probabilities are not calibrated reliability.

No new real run imports the device reference tasks, test doubles, Titania profile or `thermal.prepare_reference` / `xrd.acquire` macros. Developer offline tests still use explicit scripted doubles. The historical TiO2 runner now requires `--legacy-demo`; its earlier success remains historical and does not satisfy the new generic-execution requirement. General multi-device sample transport and material chemistry are still absent. Unsupported material-phase goals terminate without inventing state.

## Diverse automatic-selection acceptance

All four goals used `device: "auto"`; no task-specific device shortlist or action sequence was supplied. Each selected device was appropriate for its fixed public-state goal.

| Goal | Bound device | Goal met | Protocol correct | Combined tokens |
|---|---|---:|---:|---:|
| balance-filter | `mt_xpr204e` | False | True | 61,534 |
| ika-stirring | `ika_plate_rctdigital` | True | True | 194,271 |
| binder-timer | `binder_fd056` | True | True | 175,090 |
| furnace-program | `nabertherm_l9` | True | True | 99,210 |

Three goals completed, four protocol outcomes were correct, and there was no false success. This is a small development sample. Directory visibility is not complete scientific coverage of 194 instruments; many historical device fixtures only test power control.

The balance failure is retained: the expert proposed a valid filter-setting candidate together with `workflow.await`, which the current adapter did not expose. The complete candidate set was rejected; the one allowed format repair repeated the unavailable option and the run ended `role_error`. The valid filter action was not silently extracted and executed. Follow-up inspection established that the native XPR204E runtime does support host-owned `sim.step`: a direct native `step(10)` advanced its public clock from 0 to 10 seconds and readback succeeded. Our hard-coded `NativeLifecycle` bridge whitelist omits this runtime, so `can_step` is false and the wrapper is absent from the model menu. The model violated that exposed menu, but the adapter coverage gap is a substantive workflow limitation; waiting itself is a valid native simulation operation. The earlier attribution primarily to model candidate generation was incomplete. This follow-up made no API requests and did not change the retained trajectory or source code.

The stirring goal is the native control setpoint of 100 rpm, represented as `10.471975511965978 rad/s`, with power on. The native 5 rpm grid makes the earlier proposed 10 rad/s exact-equality goal unsuitable; the reachable task was corrected **before all real acceptance attempts**. This verifies configured speed, not actual loaded-fluid mixing. Binder verifies a native one-minute timer expiry; the furnace verifies its empty-furnace program, not synthesis.

The native-only offline pack passed 28/28 declared checks: six cases repeated twice with explicitly offline scripted fixtures, four wrong-control negatives, and 12 protocol checks. Cases include an already-satisfied control and an unsupported material-state goal. Both repeats use the same test-role implementation; this is repeatability coverage, not a two-policy matched baseline. This pack does not certify the old authored TiO2 scientific demo. The full project suite passed **193 tests in 86.78 seconds**, and all **570** pinned upstream file hashes matched.

## Matched knowledge comparison

`KnowledgeInstrumentProfile` forms lexical queries from the public objective, predicates, constraints, current native observations, capability IDs and errors. It returns up to three ranked passages plus a common evidence protocol. Eight corpus passages cover instrument facts, evidence quality, simulator limitations and one scoped published TiO2 result. No passage contains a task action sequence or branch-to-repair mapping. `material_evidence_ids` cite retrieved facts; current native observation IDs remain necessary for success. No Materials Encoder is integrated.

Both comparison arms use the same native executor, oracle, prompt family and lossless `$public_meta` contract encoding. Repeated metadata is deduplicated while parameter schemas remain inline; tests verify exact reconstruction. This reduces repeated context without hiding capabilities. Device discovery happens before this knowledge augmentation.

The new fixed order was furnace off/on, then IKA on/off. Evidence checks confirm the same actual device, task/oracle, initial public values/units/quality, full public contract, setup, clock bridge, model identities and outer episode caps within each pair. Discovery and execution share the 48-call and 360-second allowance, so remaining execution time naturally differs by discovery latency. These are single pairs without stochastic replication, not a controlled estimate of general benefit.

| Task | Knowledge | Goal / protocol | All model calls | Wall seconds | Combined tokens |
|---|---|---|---:|---:|---:|
| furnace-program | off | True / True | 35 | 32.96 | 109,190 |
| furnace-program | on | True / True | 35 | 35.01 | 113,801 |
| ika-stirring | on | True / True | 20 | 37.19 | 181,761 |
| ika-stirring | off | True / True | 20 | 44.31 | 177,850 |

All four goals completed with supported final reports, one final verification each and no false success. Model-call counts include discovery, Jev routing/operation/device selection, expert requests and any repairs. Combined tokens include both expert and hosted Jev requests; elapsed time includes selection and initialization.

The exact furnace sequence was identical in both arms:

| Order | Proposed call | Arguments / effect |
|---|---|---|
| 1 | `workflow.inspect` | Obtain initial public state. |
| 2 | `nabertherm_l9::power.set` | `{"enabled": true}` |
| 3 | `nabertherm_l9::thermal.configure_program` | `{"segments": [{"temperature_K": 323.15, "ramp_rate_K_s": 0.1, "hold_s": 10}]}` |
| 4 | `nabertherm_l9::thermal.start_program` | `{}`; accepted start is not completion. |
| 5–13 | `workflow.await` | Nine separate `{"duration_s": 30.0}` proposals, each gated and followed by fresh native evidence. |
| Final assessment | Verification | Public phase is completed and hold elapsed is 10 s; the declared segment predicates also match. |

The IKA sequence was also identical: `workflow.inspect`, `ika_plate_rctdigital::power.set({"on": true})`, then `ika_plate_rctdigital::stirring.set_speed({"speed_rad_s": 10.471975511965978})`, followed by fresh-state verification. These are agent-generated runtime proposals reproduced from the trace, not inputs supplied as a solution script.

Knowledge-on furnace hypotheses cited `furnace-program-lifecycle` and `furnace-units`; IKA cited `ika-speed-units` and `native-read-quality`. Off arms cited only the shared evidence protocol. **No selected action changed, and model-call counts did not decrease.** Knowledge added 4,611 tokens for the furnace and 3,911 for IKA. IKA wall time was lower with knowledge and furnace wall time higher, but these single runs cannot establish a latency effect. The result establishes functioning retrieval/citation integration, with no demonstrated decision or efficiency benefit. It does not close M5's substantive materials-decision requirement.

## Historical censored knowledge stage

Before the user removed OpenRouter amount concerns, two separately frozen furnace attempts used the old 1M incremental reservation window. Knowledge-off succeeded with 109,697 tokens. Knowledge-on stopped with 83,429 actual tokens when a **pre-send** Jev reservation needed 283,224 tokens but the remaining local window was 276,769. The furnace was configured but still idle. The historical terminal label is `operation_decision_error`; its nested cause is `TokenBudgetExceeded`, and `request_sent` is false.

This is a local accounting-policy interruption, not an account-quota response or evidence that the model/knowledge could not solve the task. No IKA attempts ran in that stage. Those two records remain unchanged, are not spliced into the new pairs, and are not counted as retrieval degradation. The renewed four-run phase was explicitly separate and persistently capped at four attempts; no repeat-until-success runs were made.

## Accounting and provenance

| Stage | Attempts | Reported tokens |
|---|---:|---:|
| Initial diverse acceptance | 4 | 530,105 |
| Historical budget-censored knowledge stage | 2 | 193,126 |
| New fixed OpenRouter record-only knowledge stage | 4 | 582,602 |
| Total new development usage across these stages | 10 | 1,305,833 |

The new four-run phase reconciles **110 unique API reservations / 110 model calls / 582,602 reported tokens**, all settled with complete usage. Its source hashes matched after every attempt. Ledger totals after it are 714 reserved/settled requests, 2,907,900 reported and 3,263,264 conservatively charged tokens. Eight older unknown-usage requests remain in that historical difference; there were zero new unknowns, zero in-flight requests and no accounting violation.

Record-only applies only to `provider=openrouter` with exact IDs `deepseek/deepseek-v4.1-flash` and `typesafe/jev-1.14`. It records reservations as estimates and settles actual usage without a token-amount cap. The original limit/window markers remain immutable. New exempt calls do not drain the non-OpenRouter allowance: its charged total stayed 2,680,662 and remaining allowance 2,319,338. Action/call/time/output bounds remain; missing usage can stop for metering integrity. Individual Jev manifests still describe conservative reservation sizing; the phase freeze and policy marker distinguish that estimate from an enforced amount cap.

The corpus's TiO2 passage is scoped to the published feedstock and separate aliquots, not general kinetics or sequential reheating. Its presence does not add native chemistry state. Sources and locators are retained in the [corpus](../../corpus/instrument-knowledge-v1.json). Compact evidence does not include credentials or private model reasoning.

## Review and remaining work

Astra passed the provider-specific accounting and current code design in read-only reviews. It confirmed the public directory, immutable goals, native-only live path, separate knowledge/observation citations and argmax fail-closed behavior. It identified two nonblocking limits: Jev does not independently receive detailed device contracts during binding, and incompatible selected-device initialization currently shares the broad `initialization_error` label. Parent test/hash verification is separately reported above; Astra did not claim to rerun those checks.

Remaining acceptance gaps are meaningful material-decision improvement, a generic multi-instrument TiO2 execution/transfer/oracle, broader acquisition/lifecycle coverage and formal held-out evaluation. A second independent general expert and human failure review are not completed. The scientific UI and offline decision benchmark remain deferred. See the [current checklist](../status/checklist.md).


Final Astra report/evidence re-review: **PASS** after clarifying that the offline fixtures repeat one implementation. The reviewer independently reconciled all 110 new requests, 70 original artifact hashes and 83 source hashes in the reviewed pre-lifecycle snapshot against the v4 freeze. It confirmed one final verification per new run, identical paired action parameters and the absence of a demonstrated retrieval benefit. No new model requests were made by the review.
