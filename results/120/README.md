# 120-case interactive instrument evaluation

This folder contains results for multi-step control of simulated laboratory instruments.
The final production study evaluates **22 expert/chooser configurations on the same
120 cases**, once each with **seed 0**, in **`teacher_forced` mode**. Its 2,640
usable closed records include successful episodes and saved failed attempts.
**Goal completion is assisted success after reference corrections, not autonomous success.**

## Which files to use

| File | Contents |
| --- | --- |
| [workflow-teacher-forced-status.json](workflow-teacher-forced-status.json) | Compact totals and one row per configuration: processed cases, assisted goals, aborts, original-action accuracy and corrections. |
| [workflow-teacher-forced-results.json](workflow-teacher-forced-results.json) | Full results, case outcomes, completed-episode native scores and group breakdowns, all-closed-attempt action metrics, usage, errors and provenance. |
| [workflow-teacher-forced-final-audit.json](workflow-teacher-forced-final-audit.json) | Independent reconciliation of raw records, saved native grades, full/compact exports, failures, source identities and actual hardware. |

The final full export was generated on **2026-10-22 at 21:27 Zurich (19:27 UTC)**;
the final audit passed at **21:28 Zurich (19:28 UTC)**. Other JSON files describe earlier protocols,
stopped runs, inventories or development pilots. In particular, `workflow-autonomous-pilot-results.json`,
`workflow-teacher-forced-pilot-results.json` and `workflow-deadline-v7-stopped-results.json`
have different cohorts or protocols. Do not pool them with this production study.
The static 388-question experiment is documented separately in [../388/](../388/README.md).

## Dataset and public inputs

The dataset comprises **120 authored, parameterized cases across 14 workflow families
and eight native instrument simulators**. These are synthetic workflows built from
ChemLab public contracts and runtimes, rather than recorded laboratory failures.

| Instrument | Cases | Workflow families |
| --- | ---: | --- |
| Binder FD 56 oven | 18 | USB recording; standby restoration |
| Nabertherm L9 furnace | 18 | Heating program; door-pause recovery |
| IKA RCT digital hotplate | 18 | Unlock/stirring; motor diagnosis |
| New Era NE-1000 syringe pump | 24 | Delivery; timed monitoring |
| Esco fume hood | 18 | Illumination; sash access and resecure |
| Vaisala DMT152 dew-point transmitter | 12 | Startup/measurement diagnosis; offline calibration |
| Peak Precision Hydrogen generator | 6 | Startup diagnosis |
| Eppendorf CellXpert C170i incubator | 6 | Sensor-cover access |

Cases include 75 healthy-labeled cases and setup/fault variants, with 27 matched
healthy/failure pairs. There is **no assigned train/test split** for this sequential
dataset, and no L1–L5 classification or established increasing difficulty order.

The model receives the public goal, initial context, manuals, offered actions and
observations. Private evaluator fixtures contain checkpoints, prerequisites,
accepted alternatives, fault schedules and outcome rules; they are excluded from
model inputs. A shared bank offers **1,954 argument presets for 146 public commands**,
without filtering the menu by the current case's private answer. The first native
choice selects an instrument. Later actions can operate it, read observations or
manuals, wait, switch instruments, finish or escalate. State persists; waiting
advances simulated time, and cached readings keep their original sampling times.

See the [dataset guide](../../benchmark/instrument/sequential_120/README.md) and
[dataset manifest](../../benchmark/instrument/sequential_120/manifest.json).

## Agent workflow and configurations

The evaluation uses the actual application `app.graph.build_graph`:

```text
diagnosis → proposed operations → chooser selection → procedure
          → deterministic gate → native execution/feedback → verification
```

The expert produces diagnostic and verification records and proposes public action
IDs. The chooser selects graph routes and operations. A deterministic procedure
envelope binds the selected operation to its exact offered arguments; the application
gate validates it before native execution. The bridge returns native feedback to
the continuing graph. A model's completion claim alone does not award success.

Each of **Qwen2.5-14B-Instruct** and **DeepSeek-V2-Lite-Chat** is paired with these
ten chooser configurations, giving 20 configurations:

| Chooser | Decision method |
| --- | --- |
| Same model as expert (`*-self`) | HF next-token softmax over offered answer labels |
| OpenJev8B | Released native decision head |
| APUS Reflex 4B, low effort | Released native decision head |
| APUS Reflex 4B, high effort | Same checkpoint, different declared effort |
| Qwen2.5-3B-Instruct | HF next-token softmax over offered answer labels |
| Qwen3-4B-Instruct-2507 | HF next-token softmax over offered answer labels |
| all-MiniLM-L6-v2, cosine | Frozen embedding similarity with previously fitted temperature |
| all-MiniLM-L6-v2, linear | Previously fitted linear readout on frozen features |
| all-MiniLM-L6-v2, kNN | Previously fitted nearest-neighbor readout |
| Qwen2.5-3B hidden features, linear | Previously fitted linear readout on frozen features |

Two additional self controls use Qwen2.5-3B and Qwen3-4B as both expert and chooser,
giving **22 configurations**. Linear/kNN readouts use the static dataset's training
partition; cosine temperature uses its validation partition. Their manifests declare
**no sequential labels used**. All are reused unchanged, with no online updates.

The frozen protocol is `deadline-rescue-qwen-yarn64k-menu-ids-v3-decision-view-v4`.
Experts use schema-guided JSON with exact offered menu IDs. Grouped menus and
explicit references remove repeated fields without changing the offered native
actions. Role-specific current-state views omit older history; the complete original
public payload remains in the trace. Input parameter schemas are omitted from expert
model text where exact preset arguments are supplied; native validation is unchanged.
Choosers use the declared current-decision view with a 32,000-byte preliminary cap,
plus their own tokenizer/context limits. No private correctness ranking is inserted.

Common limits are **300 seconds per episode, 128 configured graph calls,
120 seconds per call, 2,048 graph steps and 2,048 specialist output tokens**.
Each case also enforces the native action, query and simulated-time budgets in
its public `initial_context.budget`.
The configured counter differs from actual inference attempts for some backends;
use saved usage as well as the limit. Qwen2.5-14B uses explicit YaRN factor 4 with
a 65,536-token operational cap; DeepSeek uses 163,840, weak Qwen controls 32,768,
and native heads retain their released 4k/8k limits. Exact settings and dtypes are
in each full-report cell's `configuration` and the frozen configuration below.
Model checkpoint weights remain frozen; YaRN changes the context configuration.

## Grading, assistance and denominators

The native grader checks an authored partial-order checkpoint curriculum, including
prerequisites, state conditions and specified alternative actions/parameters.
It grades each **original native proposal before correction**. Machine, command
and argument selections can require several internal calls but count as one native
proposal. A correct proposal executes unchanged; an incorrect one is replaced by
an offered acceptable action. The next decision receives the original proposal,
executed action, correction flag and actual native response.

Teacher forcing starts only after the graph returns a native action. Routing,
expert records and internal format/schema checks remain unassisted; these can fail
or exhaust a budget before a correctable proposal exists. Successful termination
requires native goal conditions, accepted evidence and required checkpoints;
maintenance cases may require justified escalation.

| Measure | Definition and JSON location |
| --- | --- |
| Original-action accuracy, all closed attempts | Correct original proposals / observed original proposals, including auxiliary queries and saved decisions before an abort. Full report: `original_action_metrics_all_closed_attempts`, globally and per cell. Compact: `totals.original` and `cells[].original`. |
| Reference corrections | Actual corrected native actions, using the same observed-action scope. Correction rate is corrections / observed proposals; retain both counts. |
| Assisted goal completion | Native verified goals / **120 declared cases per configuration**, or / 2,640 globally. Full report: `verified_outcome_successes` and `assisted_goal_completion_rate` per cell; compact: `assisted_goals` and `planned`. |
| Completed-episode diagnostics | `cells[].native_score` and top-level `progress` proposal/correction totals use completed native episodes only. Control accuracy excludes auxiliary queries; query admissibility is separate. Checkpoint/stage and workflow/instrument/condition breakdowns retain their stated scopes. |

Episode completion, record coverage and goal success are distinct. Saved aborts
count as processed attempts and remain unsuccessful in the full-case goal denominator.
Missing/unusable attempts would also remain in that denominator; no unobserved
actions are invented for an aborted or missing trajectory. A zero observed-action
denominator gives **`null` accuracy**, not zero accuracy. This occurs in the four
MiniLM cosine/linear configurations despite all their attempt records being present.
The legacy `native_score.infrastructure_failures` field counts abort envelopes;
it does not by itself establish a hardware or infrastructure fault. Consult the
recorded failure evidence. Micro accuracy weights observed actions; episode/workflow
macro fields use different weighting and must not be substituted for it.

Final totals are **1,344 assisted goals / 2,640 attempts (50.91%)**, **1,296
model-output or budget aborts**, and **zero missing/unusable records**. All-closed
original accuracy is **5,477 / 12,466 (43.94%)**, with **6,989 corrections** and
complete proposal/correction-count coverage. Completed-episode-only accuracy is
**3,719 / 9,851 (37.75%)**, with 6,132 corrections; it has a different scope.

These exports do not report graph-route ECE/Brier calibration: internal routes
have no authoritative native route-answer oracle. The confidence analyses in
[../388](../388/README.md) concern the separate labelled static questions.

## Usage, provenance and limits

`usage` reports saved episode wall time, token counts and graph-call counters with
coverage. Episode timing excludes model initialization outside the episode and
is not GPU kernel time. Unavailable totals remain `null`; partial token sums are
lower bounds with explicit coverage. Raw zero placeholders require their
`*_known` flags and must not be interpreted as measured zero usage.
Inference was local: `hosted_api_cost_usd: 0` does not mean compute was free.
GPU-active seconds and actual monetary compute cost were not measured and remain
`null`. Actual allocations include H100 and L40S GPUs; compare latency and
time-budget failures with each allocation's hardware provenance, not as a uniform
hardware experiment.

The frozen study is [workflow-teacher-forced-v11-20261022-e](../workflow-teacher-forced-v11-20261022-e/):
[configuration](../workflow-teacher-forced-v11-20261022-e/configuration.json),
[scope](../workflow-teacher-forced-v11-20261022-e/deadline-scope.json),
[source manifest](../workflow-teacher-forced-v11-20261022-e/source-manifest.json),
public/native sources under `snapshot/`, and raw episodes/traces under `runs/`.
The [saved-record scorer](../workflow-teacher-forced-v11-20261022-e/snapshot/instrument/eval/sequential.py) aggregates trusted native grades without rerunning models or simulation.
Application revision: `e116c6b42b3dffbd7870d0e5e98ca965328064d3`; exact file hashes
are authoritative. Public exports retain source paths/hashes. The final audit
reconciles all 22 × 120 raw records, original grades/corrections, completed native
scores including groups, usage, runtime/hardware identities and both public exports;
it found no unresolved technical/startup/unknown failures or source-read errors.
The frozen study, raw traces and checkpoint paths are local cluster artifacts,
normally ignored by Git. A checkout with only the published JSON files does not
contain all inputs needed to repeat the source-authentication audit.

This is one seed, with finite presets, authored acceptance rules and no sequential
holdout. Independent domain-expert review remains pending; valid procedures outside
the authored curriculum may receive no credit. Instruments are independent
simulators, without cross-instrument material transfer or real hardware validation.
These scores do not establish autonomous success or an exact reproduction of a
hosted experiment. Further protocol details: [benchmark guide](../../benchmark/README.md)
and [graph guide](../../benchmark/graph/README.md).
