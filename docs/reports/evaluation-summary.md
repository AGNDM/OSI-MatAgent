# Recorded Evaluation Snapshot

Recorded on 20 October 2026. This summary preserves the development outcome table for the initial repository commit. Detailed request/response traces remain in the ignored local `results/` directory; this document does not claim that a new clone contains those traces or can reproduce hosted model outputs bit-for-bit.

## Device workflow development

| Phase | Passed / attempted | Reported API tokens | Role-output error/repair events | Gate rejections | Routing fallbacks |
|---|---:|---:|---:|---:|---:|
| Complete offline device regression | 194/194 | 0 | 0 | 0 | 0 |
| First real API pilot, before handoff fix | 0/3 | 53,232 | 3 | 0 | 0 |
| Second real API pilot | 2/3 | 42,929 | 3 | 2 | 0 |
| Predeclared real DeepSeek category sample | 20/20 | 199,969 | 9 | 0 | 0 |
| Local OpenJev wiring, scripted specialists | 3/3 | 0 | 0 | 0 | 0 |
| Real DeepSeek specialists + local OpenJev | 2/2 | 14,892 | 1 | 0 | 0 |

All device-debugging API requests, including both failed pilots, repairs and repeat debugging, consumed **311,022 tokens across 206 requests**. The device ledger ceiling was 5,000,000 tokens. All requests had complete usage, no reservations remained outstanding, and all returned model IDs matched `deepseek-ai/DeepSeek-V4.1-Flash`. No OpenRouter requests were made for this expansion. Provider monetary billing was unavailable; no currency amount is inferred.

The first pilot repeatedly acted on an old inspection after a native command. A subsequent guard-only recovery also failed on an oven power prerequisite. The implemented fix returns a native public readback immediately after successful commands and counts both the command and read against the action budget. Specialist inputs include the actual last-action receipt. The declared 20-device sample then passed; its intermediate errors were not discarded.

## What coverage means

The pinned official catalog has 203 entries. Native sessions were established for 198, and 194 have independently reset control/readback tasks spanning 16 categories. **116 tasks verify power control only; 78 verify other controls/readbacks.** A device passes only its declared task and exposed capability subset. These are authored integration fixtures, not an external scientific benchmark.

Examples include balance filter selection, oven/hotplate temperature setpoints, rotary-evaporator speed, stirrer speed, sash height and door/lid position. The powder-XRD task verifies a door angle, not diffraction acquisition. The syringe-pump example verifies power, not fluid delivery.

The offline test uses a scripted specialist policy/reference sequence. Its success demonstrates integration regression, not LLM competence. The real category sample is a small declared development set, not a held-out performance study. No result here establishes routing benefit, scientific fidelity or multi-device experiment completion.

## Unaccepted entries

| Entry | Remaining integration work |
|---|---|
| CHEM-181 Bronkhorst P702CV | Supply and validate the explicit constructor boundary/physics parameters. |
| CHEM-182 VICI EUTA valve | Adapt its different public-interface/session contract. |
| Legacy `agilent_7890b_ssl_fid` | The pinned catalog lacks a runtime path; no alternative implementation was accepted. |
| Legacy `ohaus_px224e` | The pinned catalog lacks a runtime path; no alternative implementation was accepted. |
| Legacy `ohaus_spx2201` | The pinned catalog lacks a runtime path; no alternative implementation was accepted. |
| CHEM-012 BINDER VD56 | Native session works; the current generator has no accepted state-changing task. |
| CHEM-096 Agilent GC8890B | Native session works; the current generator has no accepted state-changing task. |
| CHEM-120 Biolin Sigma700 | Native session works; the current generator has no accepted state-changing task. |
| CHEM-179 Ziath DP5 | Native session works; the current generator has no accepted state-changing task. |

A generator limitation does not imply that upstream has no corresponding capability. These entries were not replaced with synthetic device stubs to increase the accepted count.

## Verification and retained evidence

The project passed **73 tests**, and all **570 pinned upstream hashes** matched. The artifact audit found no copy of the actual credential value in 745 scanned files. These are recorded execution checks, not claims about an immutable future environment.

- [Machine-readable phase summary](../evidence/device-expansion-summary.json)
- [Per-device coverage table](../evidence/device-coverage.csv)
- [Raw-trace hash index](../evidence/trace-manifest.json), identifying the local traces without committing their request bodies
- [Reproduction and budget instructions](../guides/device-workflow.md)

The earlier three-case drying expert pilot is separate: 2/3 passed, with 42,802 reported tokens and a retained verification failure. Formal two-expert comparison, materials retrieval benefit and independent human causal review remain pending.

## Paper-grounded titania development slice (21 October 2026)

Two new thermal-treatment/XRD branches passed scripted and real DeepSeek-plus-OpenJev execution (2/2 each). Both wrong-repair controls failed independent scoring as intended. Real runs required two evidence-ID repairs total and no routing fallback. API usage: 22,617 tokens across 14 requests, bringing the shared ledger to 333,639/5,000,000. All 80 project tests passed.

Native furnace and XRD acquisition execute, with a common clock and logical aliquot identity. Chemistry endpoints and transfers remain authored; curves are reference-derived synthetic acquisitions. Fixed paper context and explicit repair policy make this a development demonstration rather than a blind held-out benchmark. Full provenance, paths and limitations: [titania execution evidence](../evidence/titania-validation.md).
