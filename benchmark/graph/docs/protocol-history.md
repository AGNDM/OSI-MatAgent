> Historical protocol and study record. Earlier milestones and site paths are retained for provenance. Software tests and scheduler scripts have been removed; use [the current Python package guide](../README.md) for all new execution.

# Local instrument benchmark through the application graph

This package adapts local models to the existing `app.graph.build_graph`. Keep
the repository's single `app/` directory and the unchanged sibling
`benchmark/instrument/` and `benchmark/comparison/` packages. The adapter has no
hosted model provider. Model weights and fitted features must already exist
locally. No application source is vendored in this tracked package.

The default model/graph core retains the **study v2** interface, with adapter
identity `actual-repository-langgraph-native-bridge-v1`. Explicit optional
`menu-ids-v1` and `decision-view-v1` transports form a separate protocol-v5
experiment, described below. Existing frozen studies are not modified by these
options. The portable preparation CLI and launchers snapshot the chosen source.

The primary experiment crosses an expert model with a chooser inside the actual
diagnostic → operation selection → procedure → gate → native feedback →
verification workflow. `examples/matrix.json` declares the original two Qwen
expert rows and nine local chooser columns. It is an editable **input template**,
not a result. The provider-neutral research design can include expert families
such as Qwen, DeepSeek and MiniMax and chooser families such as Jev/OpenJev;
only explicitly staged local implementations are accepted here. A declaration
does not establish that a provider was executed or that it supports the graph's
specialist schemas.

`examples/stronger-matrix.json` uses the staged Qwen2.5-14B and
DeepSeek-V2-Lite-Chat rows with the same nine chooser columns. It declares
experiments rather than asserting schema capability or completed trajectories.
`examples/provider-design.json` documents the broader Qwen/DeepSeek/MiniMax ×
Jev/OpenJev research design; the MiniMax and hosted Jev placeholders are explicitly
unrun and are not executable plans.

The runner supports all 120 original sequential cases in autonomous `free_run`
mode and in separate `teacher_forced` runs; the default mode setting selects
both. Teacher forcing grades the original
proposal before executing the reference replacement and returning its actual
feedback. It does not train the model or make assisted completion autonomous
success. The 388 MCQA cases are the supporting direct-decision experiment in
`benchmark/comparison`; they are not additional graph trajectories.

The full study **E has launched**, covering **22 cells × 120 cases in
`teacher_forced` mode**, seed zero. Its source and Qwen2.5-14B YaRN profile are
frozen, and the final technical qualification passed. Scheduler recovery uses
arrays `4644408` (core), `4644409` (weak controls), `4644410` (support) and late
core recovery `4644566`, retaining 27 original native attempts and the submission
audit below. This is a launch
record, not a claim that all 2,640 episodes have completed. Autonomous B
and the teacher-forced C/D development pilots remain separate. E's primary
results are original-proposal accuracy and correction rate; reference-corrected
goal completion is assisted completion.

Model-format and schema failures, native context/menu limits, runtime failures,
deadlines and model budgets need separate failure counts over the full declared
cohort. Raw abort envelopes retain their diagnostics. Unknown token counts keep
false `*_tokens_known` flags; numeric zero placeholders are not measured usage.
Graph route probabilities have no authoritative route oracle, so full-graph
route ECE/Brier calibration is not claimed. Confidence analyses belong to the
supporting labeled MCQA experiment. Compare latency within hardware groups;
MIG slices, full H100 and L40S are different resource allocations.

This uses the unchanged application plus a new public benchmark adapter. It is
a follow-up to the PDF experiment, not an exact reproduction of its hosted
models or adapter. Independent human validity review remains outstanding.

## Historical execution environment

The study was executed in genuine Slurm allocations, with pinned graph and model
interpreters and separately authenticated APUS dependencies. Scheduler scripts
and software tests used during qualification are no longer included. Historical
job IDs and receipts below remain evidence of those runs; current execution
uses the direct Python commands in [the package guide](../README.md).

## Historical protocol development

### Bounded public decisions: protocol v5

The opt-in expert specification uses `record_protocol: "menu-ids-v1"` and
`structured_output: "json_schema_lmfe_v4"`. The chooser specification uses
`context_protocol: "decision-view-v1"`. These flags must be recorded in a fresh
study; results cannot be pooled with earlier full-context protocols.

The expert proposes one to four IDs from the complete currently offered public
menu, with brief rationales, postconditions and current evidence citations. The
schema-guided decoder constrains IDs to that public menu; it receives no answer
key or private evaluator state. The adapter binds each ID to the exact offered
native action, preserving its arguments. Diagnostic and verification citation
fields admit only current public inspection/measurement wrapper IDs. With no
observations the wire field is omitted and binds to an empty citation list;
shared-reference names are never valid evidence IDs. The chooser still selects using its
actual released probabilities or conditional option logits. The procedure
envelope copies the selected action and current request/revision fields without
model inference. The original app gate, native executor and grader retain their
authority; selecting a legal operation does not establish that it is correct.

Specialist context uses reversible shared references/tables to avoid repeating
contracts and unchanged structures. The router and operation chooser receive an
explicit current-decision projection: goal, native readings including quality
and timestamps, wrapper evidence IDs, selected intent, errors, exact candidates,
relevant contracts and recent/relevant native receipts. Older history is
represented by counts and hashes; this projection is **not lossless**. Its
omissions are declared, and the complete public source remains in the trace and
specialist input. No model-generated summary or evaluator ranking is inserted.

The default 16,000-byte decision-view cap is a preliminary check. A fresh study
may explicitly set `context_max_bytes` (the v6 pilot uses 32,000). Each backend
also enforces its actual tokenizer/context limit, rejecting overflow rather than
cutting text or candidates. A replay of the original 22,977-token OpenJev
failure input fits in 1,755 state tokens under the new current-state projection;
that is a tokenizer check, not a model accuracy or successful episode result.

`context_protocol: "decision-view-v2"` declares a separate native request
layout. Native OpenJev/APUS heads receive the current public state in their state
input and exact operation candidates in their dedicated choice criteria. The
state text does not repeat the candidate option list. HF choosers retain the
explicit option list because their conditional-logit readout consumes one
prompt. Routing likewise keeps native route descriptions in choice criteria.
The full original public input, projected view, actual state messages and exact
criteria remain recorded. Version 1 retains its original layout; v6 results
must remain separate from earlier snapshots. This removes duplication without
dropping observations, candidate actions or their arguments, and does not raise
the released heads' strict state/branch token limits.

Guided generation requires `lm-format-enforcer==0.11.3` and
`interegular==0.3.3` in the recorded environment. The validated local interpreter
is `/scratch/dsegura/git/osi_hackathon_AGMAT/.venv-langgraph-guided-v4/bin/python`.
Default unguided runs do not import these optional libraries. Actual inference
calls and usage are recorded independently of the app's configured role-call
budget: the latter still counts deterministic procedure envelopes as role calls.
The menu-ID transport limits consecutive JSON formatting whitespace to 16
characters. This prevents an observed greedy-decoding whitespace loop without
changing offered actions or evidence. The original structural-guidance transport
keeps its previous whitespace setting; each fresh snapshot records the actual
setting and source hashes. Full-case pilot failures and corrections must remain
separate from benchmark results.
Verifier wire records bound missing-evidence prose to four strings of at most
240 characters each, preventing an observed repeated-sentence output loop.
These fields still pass through the original application verification schema;
all three verdicts and empty/current-ID citations remain available. The adapter
never supplies a verification verdict or native outcome.

### Stopped deadline study A: readable expert inputs, protocol v7

The stopped deadline study A used `record_protocol: "menu-ids-v2"` and
`context_protocol: "decision-view-v3"`. The expert sees literal public actions
with their exact menu IDs and preset text. Current native receipts appear before
cached readings; original values, quality and sample times remain unchanged.
The view retains the latest six receipts, latest successful and latest attempted
receipt per command, latest error/read, current manuals and input contracts.
It explicitly omits duplicated initial context, older history and full historical
contracts; the complete original source and projection hashes stay in the trace.
This role view is not lossless. No physical state, recovery action or goal verdict
is supplied by the adapter. Diagnostic prose is bounded to 160-character
assessments and 120-character explanations; verifier gaps are at most four
160-character strings. Exact menu binding and the native grader are unchanged.

Decision v3 retains all v2 decision content but omits projection bookkeeping
from model text: audit hashes, descriptive selection policies and historical
contract-omission hashes. The full projected view, actual model view and original
public input remain separately recorded. Native heads receive candidates once
in question criteria; HF choosers retain their explicit option list. Actual
released token limits remain enforced without truncation.

The declared deadline matrix has 22 cells and all 120 release cases per cell,
seed zero and autonomous `free_run` only. All cells use the same 300-second
episode limit, 128 configured calls, 120-second per-call timeout and 2,048-token
specialist output budget. Eight strong-expert self/OpenJev/APUS cells receive
scheduling priority, followed by two weak self controls and twelve supporting
cells. Missing/deadline-censored questions remain in each full-120 denominator.
This shorter-budget study must stay separate from prior protocols and the
colleague's hosted, mixed-budget report. A valid failed native episode is a
model result; it is not evidence of task success.

The [final stopped-A export](../../../results/120/workflow-deadline-v7-stopped-results.json)
contains **75 closed records: 19 completed native episodes, 56 aborts and zero
verified goals**. A completed episode can fail its goal. The other 2,565 declared
episodes have no closed record; they retain their place in the 2,640-episode
cohort. All scheduler slots are terminal: 74 allocations completed and 2,566
were cancelled. A total of 106 workers received allocations, so allocation
starts and saved episode counts are different measures. These incomplete-study
results do not establish full-grid performance. Development smokes are excluded.

### Autonomous development pilot B: grouped public menus, protocol v8

The v8 configuration selects `record_protocol: "menu-ids-v3"` with the same
`context_protocol: "decision-view-v3"`. B uses this separate protocol for
autonomous development qualification. Saved pilot episodes are not a completed
full-120 production study, and later source changes require fresh validation
and a new frozen study.

The specialist sees readable action groups with literal common native fields
(`kind`, instrument and capability) and a choice-specific menu ID, remaining
fields and exact arguments. Combining those fields reconstructs every offered
native action. Only JSON preset text that exactly repeats those arguments is
omitted. Identical cached observation maps and repeated native receipts use
explicit named references; history indices, order, values, quality, sample
times and JSON numeric types are preserved. Identical proposed/executed actions
can use a receipt-local reference. These references remove exact duplicates;
they are not evidence IDs or inferred observations. The existing current-state
projection still omits older public history, so the complete role view is not
lossless; the original public payload remains in the trace.

To avoid repeating parameter definitions alongside fixed preset arguments, v8
omits input parameter schemas from specialist model text. Every offered preset
action and its exact arguments remain available. Current descriptions,
preconditions, observation declarations and slot metadata remain visible;
original schemas remain in the source and native/app validation is unchanged.
This changes neither the public preset menu nor the evaluator's accepted actions.

The diagnostic view explicitly identifies previous verifier feedback as
historical, retaining its original observation ID and evidence signature. Its
missing-evidence critique can guide the next proposal; old citations do not
become fresh evidence. The verifier prompt explicitly states that `supported`
immediately makes a **FINISH claim for the entire stated goal**. It does not
merely acknowledge a valid observation or successful command. The model still
chooses its verdict, and the native evaluator alone awards outcome credit.

The declared model contexts are restored to 163,840 tokens for DeepSeek and
remain 32,768 for Qwen. Exact tokenizer checks must include the 2,048-token
output reserve; declared limits do not prove that a particular input fits. The
common 300-second episode limit, 128 configured calls, 120-second call timeout,
2,048 graph-step limit and 2,048-token specialist output limit are unchanged.
Native chooser limits and the 32,000-byte preliminary decision-view cap remain
enforced without truncation.

The autonomous B qualification crosses the eight priority cells with the
first public release case for each of eight instruments: zero-based indices
`[0, 12, 24, 36, 48, 72, 78, 84]`, for 64 fixed development episodes. Case
selection uses release order rather than outcomes. Qualification outputs belong
under `development-qualification/` and are excluded from the production
`runs/` aggregate. Qualification must authenticate its case/cell/source bindings
and report missing records and technical failures before a readiness gate can
authorize production. A passed technical gate would not establish positive
goal performance or full-cohort reliability. The canonical
[deadline results path](../../../results/120/workflow-deadline-results.json) is
separate from the preserved stopped-A report; read its study identity and status
before citing it.

### Teacher-forced development pilot C

The [C pilot export](../../../results/120/workflow-teacher-forced-pilot-results.json)
records all **64 closed attempts** across eight priority cells and eight
predeclared public instrument cases per cell. **41 reached the goal after
reference corrections and 23 aborted.** This is development qualification,
not the complete 22-cell × 120-case study or autonomous performance.

The pilot records **228 correct original proposals out of 535 observed native
proposals (42.6%)**, with **307 reference corrections**. These observed-proposal
totals include saved decisions from closed completed and aborted attempts;
unfinished or unvisited decisions are not invented. The export separately
retains the native scorer's completed-episode-only aggregates. Reaching the
goal on a corrected trajectory does not earn credit for the original mistakes.

C remains preserved with its own source identity. Subsequent source repairs,
including exact references for duplicate individual cached readings, require a
new frozen study and fresh qualification; they do not change C's outcomes.

### Teacher-forced regression pilot D

The [D qualification receipt](../../../results/workflow-teacher-forced-v10-20261022-d/development-qualification/readiness.json)
records **24 teacher-forced attempts: 15 assisted goals and 9 aborts**. It uses
the eight priority cells on public release indices `[48, 72, 84]` (hood,
incubator and dewpoint goals). The technical gate failed; context-boundary and
malformed generated-JSON failures remain explicit. This targeted regression
pilot is separate from C, and neither pilot is a full-120 result.

### Launched full study E: explicit Qwen YaRN profile

Study E declares **22 expert/chooser cells × 120 release cases**, seed zero,
in `teacher_forced` mode. The frozen study is
`results/workflow-teacher-forced-v11-20261022-e`. Its
[final qualification](../../../results/workflow-teacher-forced-v11-20261022-e/development-qualification/readiness.json)
passed with all 24 pilot attempts recorded, 11 completed native episodes and
11 assisted goals. Passing the technical gate does not turn the remaining
model failures into successes or estimate full-cohort accuracy. These development
episodes remain excluded from production results.

The [production launch receipt](../../../results/workflow-teacher-forced-v11-20261022-e/production-launch-receipt.json)
records successful submission at **2026-10-22 16:35 UTC (18:35 Zurich)**:

| Batch | Original array | Declared episode slots | Recovery array | Replacement slots |
| --- | --- | --- | --- | ---: |
| Core | `4644053` | 0–959 | `4644408` | 936 |
| Weak controls | `4644054` | 960–1199 | `4644409` | 235 |
| Support | `4644055` | 1200–2639 | `4644410` | 1,440 |
| Core, late binding failures | `4644053` | 27, 28 | `4644566` | 2 |

An initial production failure was traced to a controller `srun` CPU-binding mask
exported through `sbatch --export=ALL`; affected worker steps failed before
Python started. The [scheduler recovery manifest](../../../results/workflow-teacher-forced-v11-20261022-e/scheduler-recovery/manifest.json)
records **2,611 safe replacement slots**: 282 proven pre-Python binding failures
and 2,329 original pending slots cancelled before starting. It initially retained
29 original slots. Of these, 27 produced native attempts and two subsequently
failed before Python with the same binding error. The separate
[late recovery manifest](../../../results/workflow-teacher-forced-v11-20261022-e/scheduler-recovery-late/manifest.json)
records array `4644566` for exactly indices 27 and 28. The first manifest remains
unchanged: combined recovery covers **2,613 safe unstarted replacements**, with
**27 original native attempts retained**. Recovery preserves E's frozen model
source, configuration and study identity. Original failures remain auditable
and are not completed native episodes.

The [independent recovery verification](../../../results/workflow-teacher-forced-v11-20261022-e/scheduler-recovery/verification-4644461.json)
passed and records actual model loading/worker identity after the scheduler
override. Model loading is evidence that Python started, not a completed task
or a goal score.

E now uses **mixed H100/L40S hardware**. The
[16-task trial](../../../results/workflow-teacher-forced-v11-20261022-e/hardware-scheduling/4645654-8710f327f764/finished.json)
(`4645654`, completed 2026-10-22 17:32:38 UTC) updated and released 16 pending
Qwen14 → MiniLM tasks, with actual L40S workers observed afterward. Qualified
L40S allocations request **one GPU, 8 CPUs and 46G host memory**. Default
allocator evidence covers Qwen14 with MiniLM cosine/linear/kNN and the
[shared Qwen14 self pair](../../../results/.deadline-rescue/l40s-focused/4645446-0.json).
Initial controller `4645674` reached its 15-minute limit; cleanup `4645909`
released and verified its three remaining holds. Self controller `4645693`
stopped before any changes because of conflicting scheduler accounting. The
[MiniLM continuation](../../../results/workflow-teacher-forced-v11-20261022-e/hardware-scheduling/4645910-76231213f826/finished.json)
updated and released **95** further requests, followed by **37** in the
[bounded self continuation](../../../results/workflow-teacher-forced-v11-20261022-e/hardware-scheduling/4646135-37468faf9ffd/finished.json).
These counts exclude earlier committed updates. The remaining two MiniLM
controllers completed with no eligible work. All resource controllers have
terminated without abandoned holds.

The append-only [hardware audit](../../../results/workflow-teacher-forced-v11-20261022-e/hardware-scheduling/)
and [override links](../../../results/workflow-teacher-forced-v11-20261022-e/scheduler-overrides/)
record proof hashes and before/after commands. Each group contains at most
eight exact tasks, held and rechecked before resource updates, then released.
Only pending, unstarted tasks with no dispatch/output are eligible. Array
ownership, frozen models/dtypes, data, seed and the **300-second/128-model-call
budget** are unchanged; no episodes are resubmitted. Compare latency and
time-budget aborts using actual saved allocations, since E is no longer a
uniform hardware profile.

The [original continuation schedule](../../../results/workflow-teacher-forced-v11-20261022-e/hardware-scheduling/continuation-schedule-20261022.json)
and [accounting/continuation update](../../../results/workflow-teacher-forced-v11-20261022-e/hardware-scheduling/continuation-update-20261022T1749Z.json)
record the serialized dependencies, including bounded self controller `4646135`
between MiniLM `4645910` and `4645911`. Bounded continuations select at most
96 unmatched pending requests, use a 45-minute allocation and preserve cleanup
time before a new chunk. These jobs continue resource scheduling; native
benchmark episodes keep their original ownership and outputs.

Production execution has started. Submitted slots are not completed episodes;
retain aborts, missing records and deadline-censored cases in each 120-case
denominator. Read the embedded study identity, generation time and coverage in
the [teacher-forced production export](../../../results/120/workflow-teacher-forced-results.json)
before citing progress. Its study must identify E rather than an earlier pilot.
The frozen source inventory is recorded in E's `source-manifest.json`. The native
`decision-view-v4` layout is separate from B's v3 chooser view; a passed context
boundary check alone does not establish successful task performance.

The [report schedule](../../../results/workflow-teacher-forced-v11-20261022-e/report-schedule.json)
records exports at 19:00, 20:00, 21:00 and 22:00 Zurich (`4644293`, `4644295`,
`4644297`, `4644299`) and a 22:30 snapshot (`4644020`). Final export `4644452`
waits for all seven original/recovery arrays, including `4644566`, to terminate. Scheduled reports do
not establish completion; each export retains its timestamp, coverage and full
2,640-episode denominator.
The later 21:00/22:00/22:30 snapshots were superseded by the five-minute monitor
and cancelled before starting to protect the final canonical files from late writes.

The [independent final audit](../../../results/120/workflow-teacher-forced-final-audit.json)
ran after the final export and all-cell saved-failure audit; see its
[dependency schedule](../../../results/workflow-teacher-forced-v11-20261022-e/final-audits/schedule-20261022.json).
Writer-drain job `4646828` first verifies all model arrays are terminal, then
drains the exact monitor chain under its shared lock. Final export `4644452`
requires that barrier to succeed (`afterok`), preventing late monitor writes.
It requires all **22 × 120** closed teacher-forced cases and reconciles complete
saved-score groups/checkpoints, original proposals and corrections, usage,
runtime/hardware identities, raw-source hashes, and the full/compact exports.
Missing cases, explicit initialization failures, technical or unknown failures,
and inconsistent evidence block completion. Model-output and budget aborts are
closed results. A partial cohort correctly produces `not_ready`.

The final E audit **passed on 2026-10-22 at 19:28 UTC (21:28 Zurich)**: all
**22 × 120 = 2,640** cases have usable closed records, comprising **1,344 assisted
goals (50.91%)** and **1,296 model-output or budget aborts**. No missing records
or unresolved technical/startup faults remain. Across all closed attempts,
original native proposals were correct in **5,477 / 12,466 decisions (43.94%)**,
including auxiliary queries, with **6,989 reference corrections**. The
[full export](../../../results/120/workflow-teacher-forced-results.json) and
[compact summary](../../../results/120/workflow-teacher-forced-status.json) retain
each combination's 120-case denominator and observed-decision counts.

E explicitly sets `context_extension: "qwen25-yarn-v1"` for Qwen2.5-14B and its
shared self-chooser. The [checkpoint's long-context recipe](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct#processing-long-texts)
uses YaRN factor **4.0**, original positional range **32,768**, and extended
configuration **131,072**. E uses a lower operational context cap of **65,536**,
including the reserved output tokens. The adapter clones the configuration in
memory; checkpoint files, tokenizer and weights remain unchanged, and no
training occurs. Identity records retain the original configuration/weight
hashes plus the actual loaded configuration, RoPE settings and extension receipt.
Default loads without the flag retain their original behavior.

The weak Qwen2.5-3B and Qwen3 controls remain at **32,768**; DeepSeek remains at
**163,840**. Native heads retain their released **4k/8k** limits. The common
300-second episode, 128-call, 120-second call and 2,048-token output budgets
remain separately declared. E's Qwen positional configuration differs from the
32k Qwen configuration used in C and D, even though its weight files are the
same. Report E separately; do not pool those model profiles or present a
tokenizer/configuration check as a full model result.

For each observed native decision, the evaluator grades the model's original
proposal first. When it is incorrect, the evaluator executes an offered
acceptable reference replacement and returns that action's real native feedback
before the next proposal. Graph routing, specialist records and internal schema
checks remain unassisted: teacher forcing applies after the graph returns a
native action to the evaluator. A chooser can repeatedly request diagnosis and
exhaust its episode or model-call budget before producing a correctable proposal,
so assisted completion remains sensitive to the chooser.

Report these primary measures together:

- **Original-proposal accuracy:** correct original proposals divided by graded
  original proposals, using the native `decision_accuracy_all_proposals` records.
- **Correction rate:** reference interventions divided by those native decisions;
  retain the intervention count and decision denominator.
- **Coverage:** valid native episodes and observed decisions, plus aborts and
  missing records, against all 120 declared cases per cell.

Also report control-action accuracy separately from auxiliary queries and
workflow/instrument breakdowns. The existing native scorer derives step metrics
from valid completed episodes; incomplete traces remain separate diagnostics.
Teacher forcing cannot replace a response that failed before a valid native
proposal existed, so context, schema, transport and budget failures remain
explicit failures.

`corrected_trajectory_outcome_met` is **assisted goal completion after reference
corrections**. It is not autonomous model performance: the model may reach a goal
only because its incorrect choices were replaced. Keep it separate from B's
`free_run` outcomes and from the original-proposal metrics. Preserve the earlier
autonomous attempts and their source identities rather than relabeling them as
teacher-forced runs.

## Historical preparation and reports

Historical preparation froze the application, public adapter, comparison package,
original instrument bundle, input plan and environment receipt. The declared
worker manifest owned disjoint case IDs; source or configuration changes required
a fresh study. Model, interface and budget aborts remained in cohort denominators.

The matrix reporter was qualified in Slurm job `4635920` and the reachability
appendix in `4635941`. Their historical qualification receipts remain under
`docs/provenance/`; the software tests and scheduler scripts have been removed.
Current implementations live under `graph_local/reporting/`, with direct Python
entry points documented in [the package guide](../README.md).

Reports preserve original grades and distinguish routing, candidate selection,
procedure/gate progress and native handoffs. Different profile, policy, backend
or application graph protocols require separate reports. Missing records and
closed aborts retain distinct coverage states.

The original 388 public tasks and private oracle hashes were matched during
regeneration in Slurm job `4635805`. Regeneration preserves private oracle file
mode 0600 and does not alter the canonical instrument release.
