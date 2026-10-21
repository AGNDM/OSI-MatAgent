# Native lifecycle coverage without a device whitelist

Date: 21 October 2026. Uncommitted development work on `development`, baseline `b46a09a`. This supersedes the earlier four-device lifecycle limitation in the [native acceptance report](native-acceptance-and-knowledge.md). Original model trajectories and their frozen sources are unchanged.

## Corrected failure attribution

The original balance task stopped because our adapter omitted the native XPR204E clock API from a hard-coded whitelist. Native `sim.step` was legal: `XPR204ERuntime.step(10)` advanced its public clock from 0 to 10 seconds. The expert's proposed wait was outside the menu our adapter exposed, but waiting itself was not an invented or unavailable native operation. The adapter gap is the central integration defect; attributing the failure mainly to model candidate generation was incomplete.

The user explicitly requested broad native-device adaptation without arbitrary device restrictions. The whitelist has been removed. The new live path still receives goals and contracts rather than reference solution scripts.

## Implementation

[NativeLifecycle](../../app/native_lifecycle.py) derives clock binding from native method signatures instead of module/class names. It supports relative `step(dt)` / `step(dt_s)` / `host_step(dt_s)`, absolute `host_step(time_s)` / `host_advance(time_s)`, combined `host_step(dt_s, world_time_s=...)`, and native `host_set_clock` methods. Initial native time is preserved. Optional physical inputs retain native defaults; the adapter does not invent loads, samples or scene callbacks. Clock-only setters are labeled `clock_only`, not physical motion.

Every recognized native clock becomes available through `workflow.await` in the existing Lab gate/executor. Public native instrument tools remain exposed by their actual visibility/availability contracts. Pending operation polling and cancellation use declared asynchronous capabilities and the native session signatures, including APIs that do not accept `instance_id`. Some native sessions return only operation ID, status and result: missing outer receipt fields are obtained from a fresh read of the same session, with the original payload retained. Existing contradictory identity or clock fields are not overwritten.

Unacquired native measurement fields can legitimately have no sample timestamp. Those unrelated fields no longer prevent the entire Lab from initializing. Raw values, qualities and timestamps are preserved; goal/constraint evidence is evaluated field by field, and stale or missing-timestamp goal evidence cannot establish success. Operation identity, native clock/revision, schema and replay checks remain in effect.

## Registry-wide execution checks

The same offline audit ran every registered device through the public-tool adapter. For each device with a native clock entry point it performed inspect → await 10 s → await 5 s, checked the actual readback clock and snapshot revision, then replayed the last request and checked that time did not advance again. These are universal adapter probes, not instrument-specific solver plans. No model/API requests were made by this audit.

| Check | Result |
|---|---:|
| Registered devices inspected | 194 |
| Public native tools exposed, matching native contracts | 2,855 |
| Actual clock/readback/replay checks passed | 191 |
| Native clock API absent | 3 |
| Remaining integration failures | 0 |
| Native-dynamics bindings | 189 |
| Clock-only bindings | 2 |
| Declared asynchronous poll/cancel interfaces exposed | 149 |
| Complete project tests | 201 passed in 89.02 s |
| Fixed-version upstream files verified | 571 |

The three devices without a native host clock entry point are `hach_dr6000`, `jasco_ftir4700` and `agilent_hs8697`. Their public instrument tools remain exposed; there is no device ban. Their current native host APIs provide no supported method to advance time, so no artificial clock mutation is added.

The two clock-only bindings belong to the Opentrons Flex and PreciseFlex robot runtimes. Updating their clock does not establish movement, pipetting, attachment or sample transfer. The 149 asynchronous declarations establish adapter availability; they do not mean 149 complete real acquisitions were run. A focused integration test executes a real native HX204 acquisition through accepted start, bare poll, actual time progression, terminal completion and independent goal readback. Other tests cover balance waiting, all clock modes from a nonzero initial time and preservation of conflicting native identities.

## Retained first audit and dependency repair

The first audit passed 186 devices and retained five failures, plus the same three native-clock absences. Four failures came from the old blanket timestamp check rejecting unrelated unacquired measurement fields. The fifth was BioTek 405 TS attempting to read a missing `wash_liquid_library.json` during native stepping. Both integration gaps were corrected before the second audit.

The missing library was restored from the already pinned ChemLab-Assets dataset revision `4ca4a42dab2ca0a884e449d202e9ad96625d5648` and added to the existing data manifest with SHA-256 `7ccac4c2d6fd152847ab0ddfa255d2396b3ec6d44dad8de2cd0eb34067b6df9b`. No upstream runtime code was edited. The earlier 570-file verification remains a historical snapshot; the current pinned subset contains 571 files.

The [compact evidence](../evidence/native-lifecycle-coverage.json) records all devices, bindings, counts, source hashes and both audit artifact hashes. Full local audit artifacts are ignored under `results/native-lifecycle-all-devices-v1` and `results/native-lifecycle-all-devices-v2`. Reproduce without model calls:

```bash
python -m bench.device_lifecycle --output results/my-native-clock-audit
python -B -m pytest -q
python tools/fetch_device_runtime.py
```

Astra reviewed the code and audit read-only and returned PASS. It confirmed that native contracts drive adaptation, original bare responses are retained, and missing measurement timestamps no longer block unrelated work while stale goal evidence remains rejected. Test-suite and 571-file verification were performed by the parent agent, not claimed as independently rerun by Astra.

## Original balance task rerun

One new real OpenRouter DeepSeek + hosted Jev attempt completed the original `device: auto` goal with the ordinary native profile, no knowledge augmentation and no reference solution script. Exactly one new attempt ran; the original failure remains immutable.

| Check | Result |
|---|---|
| Selected device | `mt_xpr204e` from the full 194-device directory |
| Final native filter | `High` |
| Agent report / independent goal | supported / met |
| Protocol / false success | correct / false |
| Model requests / reported tokens | 20 / 63,106; all usage complete and settled |
| Wall time including selection | 18.69 s |
| Executed proposals | inspect → `power.set(enabled=true)` → `balance.set_filter(level="High")` |

After power-on, DeepSeek again proposed `workflow.await`, this time with `duration_s:30`, alongside the direct filter-setting option. The candidate batch passed validation; Jev selected `balance.set_filter`. Thus the previously rejected wait is now a legal workflow option. No wait was actually executed in this model run because the goal only requires the selected filter setting; native wait execution is separately exercised by the balance integration test and 191-device clock audit. This single rerun is not a causal planning-quality or efficiency benchmark.

The shared ledger increased by exactly 63,106 reported tokens across 20 settled requests, with zero new unknown usage or outstanding reservations. Authorized OpenRouter metering remains record-only. Full result/manifests, generated candidates, native readbacks and all 20 request records are retained in the compact evidence.

Final Astra report/evidence review: **PASS**. It independently reconciled both audit artifacts, the new dependency, 32 execution-source hashes, 53 selection-stage source hashes and 20 uniquely settled model requests. It confirmed that the 30-second wait was an accepted candidate, while only inspection, power-on and filter-setting were executed. The review made no API calls or device changes.
