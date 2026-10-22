# Study history and raw evidence

This dated record preserves the pilots, production launch, scheduler recovery
and final closure of the 2026-10-22 studies. Earlier progress statements describe
their point in the run history. **Final study E completed all 2,640 records and
passed its closure audit at 21:28 Zurich.** Use the published
[120 results guide](../../results/120/README.md) and
[388 results guide](../../results/388/README.md) for the final results.

Paths to ignored raw studies, recovery proofs and local helper directories below
are archival references on the original cluster and will not resolve in a clean
clone. Reusable reporting code is now in [evaluation/](../evaluation/README.md);
it does not move or rewrite the original evidence.

## Result files and how to read them

The [workflow combination inventory](../../results/120/workflow-combinations.json)
records 36 local expert/chooser pairs with separate `free_run` and
`teacher_forced` counts. Eighteen pairs belong to the original study-v2
protocol; eighteen belong to the stronger-expert v4 follow-up. The latter is a
saved-progress snapshot, not a final report. The export preserves source paths,
hashes, native completion and abort counts, and identifies the two unlaunched
strong-expert self-chooser controls. Do not pool protocols or interpret complete
attempt coverage as successful workflow execution.

The [protocol-v5 repair checks](../../results/120/workflow-repair-validation.json)
record the bounded-input tokenizer replay, affected code checks, actual LLM and
OpenJev routing/operation readouts, and separate queued workflow pilots. These
interface checks are not new 120-case performance results.

The [cancellation audit](../../results/120/workflow-cancellation.json) records the
35 remaining old v4 workers stopped at the user's request, the preserved saved
attempts, and the first replacement pilot launch. The
[full-case pilot evidence](../../results/120/workflow-pilot-results.json) keeps
initial and corrected v5 snapshots separate, with raw episode/trace hashes,
actual model selections, deterministic single-option shortcuts, native actions,
errors and evaluator outcomes. These runs use one selected development case per
combination; they do not estimate accuracy across the 120 workflows.

The [stopped protocol-v7 A results](../../results/120/workflow-deadline-v7-stopped-results.json)
preserve **75 closed records: 19 completed native episodes, 56 aborts and zero
verified goals**. The declared study contained 22 cells × 120 cases, seed zero,
autonomous `free_run` only. Another 2,565 declared episodes have no closed record;
all 2,640 scheduler slots are terminal. Completed native episodes can fail their
goals, and missing records do not become scored attempts. This stopped,
incomplete study is separate from development smokes and earlier protocols.

The autonomous B development pilot uses grouped readable public menus
(`menu-ids-v3`) and the same lean chooser view (`decision-view-v3`). Group fields
plus exact choice arguments reconstruct every offered action; explicit named
references remove identical observations and repeated receipts. Input parameter
schemas are omitted only from specialist model text, where all offered preset
arguments are supplied; source schemas, native validation and grading remain
unchanged. Previous verifier feedback is marked historical, and the verifier is
told that `supported` makes a FINISH claim for the entire goal. Its model verdict
does not itself award success.

The declared DeepSeek context is restored to 163,840 tokens; Qwen retains
32,768. The common limits remain 300 seconds per episode, 128 configured calls,
120 seconds per call, 2,048 graph steps and 2,048 specialist output tokens. The
fixed development qualification declares eight priority cells × eight public
instrument cases (64 episodes), with its outputs excluded from production.
B's saved autonomous episodes do not establish full-120 performance. Later
interface changes require final checks and a fresh source freeze; the newer
native `decision-view-v4` layout is a separate protocol, not a relabeling of B.

The [teacher-forced C pilot export](../../results/120/workflow-teacher-forced-pilot-results.json)
contains **64 closed attempts: 41 reached the goal after reference corrections
and 23 aborted**. Its logged native decisions contain **228 correct original
proposals out of 535 (42.6%)**, with **307 reference corrections**. These totals
include saved decisions from closed completed and aborted attempts; the export
keeps completed-episode-only native aggregates separately. These are development
pilot results on eight predeclared public cases per priority cell, not full-120
or autonomous performance. C's outcomes and original source remain preserved.

The [D qualification receipt](../../results/workflow-teacher-forced-v10-20261022-d/development-qualification/readiness.json)
records a targeted **24-episode teacher-forced pilot: 15 assisted goals and 9
aborts** on public indices `[48, 72, 84]` across the eight priority cells. Its
technical gate failed; context-boundary and malformed generated-JSON failures
remain reported. It does not establish full-120 performance.

Full study **E** is **22 cells × 120 cases, seed zero, `teacher_forced` only**,
frozen at `results/workflow-teacher-forced-v11-20261022-e`. Its
[final technical qualification](../../results/workflow-teacher-forced-v11-20261022-e/development-qualification/readiness.json)
passed with 24 recorded development attempts, 11 completed native episodes and
11 assisted goals. The remaining model failures stay failures; pilot attempts
are excluded from production. The
[production launch receipt](../../results/workflow-teacher-forced-v11-20261022-e/production-launch-receipt.json)
records submission on **2026-10-22 at 16:35 UTC (18:35 Zurich)** of core array
`4644053`, weak-control array `4644054` and support array `4644055`, covering
declared slots 0–959, 960–1199 and 1200–2639 respectively. Production has started;
this is not a completed 2,640-episode result. For progress, read the generation
time, coverage and E study identity in the
[teacher-forced production export](../../results/120/workflow-teacher-forced-results.json).

The initial production launch exposed an inherited controller CPU-binding mask:
some worker `srun` steps failed before Python started. The
[scheduler recovery manifest](../../results/workflow-teacher-forced-v11-20261022-e/scheduler-recovery/manifest.json)
records replacement arrays **`4644408` (936 core slots), `4644409` (235 weak
slots) and `4644410` (1,440 support slots)**. These cover 282 proven pre-Python
binding failures and 2,329 pending slots cancelled before starting. Of the 29
initially retained original slots, 27 produced native attempts and two later
failed before Python. The separate
[late recovery manifest](../../results/workflow-teacher-forced-v11-20261022-e/scheduler-recovery-late/manifest.json)
records array **`4644566` for exactly indices 27 and 28**, leaving the first
manifest unchanged. Combined recovery covers **2,613 safe unstarted
replacements**, with **27 original native attempts retained**. Original IDs and
failures stay auditable; E's frozen model source and study identity are unchanged.
The [recovery verification](../../results/workflow-teacher-forced-v11-20261022-e/scheduler-recovery/verification-4644461.json)
passed and observed real model loading; it does not claim completed tasks or
goal scores.

E now executes on a mixture of **H100 and L40S allocations**. The
[16-task resource trial](../../results/workflow-teacher-forced-v11-20261022-e/hardware-scheduling/4645654-8710f327f764/finished.json)
(`4645654`, completed 2026-10-22 17:32:38 UTC) updated and released 16 pending
Qwen2.5-14B → MiniLM tasks; actual workers then started on L40S. The qualified
L40S request is **one GPU, 8 CPUs and 46G host memory**. Default-allocator
memory checks cover the three Qwen14 → MiniLM readouts and the shared
[Qwen14 self pair](../../results/.deadline-rescue/l40s-focused/4645446-0.json).
The initial bulk controller `4645674` reached its 15-minute limit; cleanup
`4645909` released and verified its three remaining holds. The first self
controller `4645693` stopped before any changes because of conflicting scheduler
accounting. The subsequent
[MiniLM continuation](../../results/workflow-teacher-forced-v11-20261022-e/hardware-scheduling/4645910-76231213f826/finished.json)
updated and released **95** further requests; the
[bounded self continuation](../../results/workflow-teacher-forced-v11-20261022-e/hardware-scheduling/4646135-37468faf9ffd/finished.json)
updated and released **37**. These are continuation counts, excluding earlier
committed updates. The two remaining MiniLM controllers completed with no
eligible work. All resource controllers have terminated without abandoned holds;
they did not create additional benchmark attempts.

The append-only [hardware audit](../../results/workflow-teacher-forced-v11-20261022-e/hardware-scheduling/)
and [override links](../../results/workflow-teacher-forced-v11-20261022-e/scheduler-overrides/)
retain proof hashes and exact commands for each group of at most eight tasks.
Only pending, unstarted tasks without dispatch/output are eligible. These
changes preserve array ownership, frozen model/dtype/source, seed and cases;
they do not resubmit episodes. The **300-second episode budget and 128 model-call
limit remain unchanged**. Read latency and time-budget failures with the saved
hardware allocation rather than treating E as a uniform GPU profile.

The [original schedule](../../results/workflow-teacher-forced-v11-20261022-e/hardware-scheduling/continuation-schedule-20261022.json)
and [accounting/continuation update](../../results/workflow-teacher-forced-v11-20261022-e/hardware-scheduling/continuation-update-20261022T1749Z.json)
retain the serialized controller dependencies, including the bounded self retry
`4646135` between MiniLM `4645910` and `4645911`. Bounded continuations select
at most 96 unmatched pending requests, use a 45-minute allocation and preserve
cleanup time before a new chunk. Native episodes retain their original ownership
and outputs.

The [report schedule](../../results/workflow-teacher-forced-v11-20261022-e/report-schedule.json)
records 19:00/20:00/21:00/22:00 Zurich exports (`4644293`, `4644295`, `4644297`,
`4644299`), a 22:30 snapshot (`4644020`) and final export `4644452` after all seven
original/recovery arrays, including `4644566`, terminate. Scheduled exports retain incomplete
coverage rather than claiming all 2,640 episodes have finished.
The later 21:00/22:00/22:30 jobs were superseded by the five-minute monitor and
cancelled before starting, preventing them from overwriting audited final files.

The [independent completion audit](../../results/120/workflow-teacher-forced-final-audit.json)
ran after the final export and saved-failure audit; its
[schedule](../../results/workflow-teacher-forced-v11-20261022-e/final-audits/schedule-20261022.json)
records the dependencies. Before export, writer-drain job `4646828` verifies
all model arrays are terminal and drains the exact monitor chain under its lock;
final export `4644452` requires its successful exit (`afterok`). This prevents a
late poll from overwriting verified files. Completion requires all **22 × 120** closed cases,
matching raw scores and usage, source-bound failure evidence, runtime/hardware
provenance, and coherent full/compact exports. Missing cases, initialization or
technical faults, unknown failures, and inconsistent evidence block completion.
Recorded model-output and budget aborts remain valid results. The audit reports
`not_ready` while these full-cohort conditions are unmet.

E completed its final audit on **2026-10-22 at 19:28 UTC (21:28 Zurich)**.
All **22 combinations × 120 cases = 2,640** have usable closed records: **1,344
assisted goals (50.91%)** and **1,296 model-output or budget aborts**, with no
missing records or unresolved technical/startup faults. Original native proposals
were correct in **5,477 / 12,466 decisions (43.94%)**, with **6,989 reference
corrections**. These decision counts include closed aborted attempts and auxiliary
queries. See the [full results](../../results/120/workflow-teacher-forced-results.json)
and [compact per-combination summary](../../results/120/workflow-teacher-forced-status.json).

Qwen2.5-14B explicitly uses the checkpoint-documented YaRN configuration
(`context_extension: "qwen25-yarn-v1"`, factor 4, original 32,768 and extended
131,072 positional configuration), with a **65,536-token operational cap** that
includes output reserve. The configuration is cloned in memory; checkpoint
files and weights are unchanged. Identity records retain original hashes and
the actual loaded configuration/RoPE settings. Weak Qwen controls stay at 32,768,
DeepSeek at 163,840, and native heads at their released 4k/8k boundaries. C and D
used the earlier 32k Qwen profile; report E separately rather than pooling them.

E's primary measures are
original-proposal accuracy and reference-correction rate, accompanied by native
episode/decision coverage and explicit model/interface failures. Goal completion
on a reference-corrected trajectory is assisted completion, even when the final
native goal holds. Preserve B's autonomous pilot results separately and do not
pool its inputs, outcomes or denominators with this new study. See
[the graph protocol guide](../graph/README.md) for exact view, qualification and
teacher-forcing semantics.

Teacher forcing applies after the graph returns a native action to the evaluator.
Graph routing, specialist records and internal schema checks remain unassisted.
A chooser can repeatedly request diagnosis and exhaust its episode or model-call
budget before producing a correctable proposal, so assisted completion remains
sensitive to the chooser.

The [canonical deadline results path](../../results/120/workflow-deadline-results.json)
remains separate from stopped A. Read the embedded study identity, status and
coverage before citing it. Each production cell retains its full 120-case
denominator, including aborted, missing and deadline-censored cases.

The exported MCQA summaries are:

| File | Contents |
| --- | --- |
| [mcqa-results.json](../../results/388/mcqa-results.json) | Held-out scores, calibration, AUROC/selective curves, timing/token coverage, full-cohort allocation estimates and provenance for all 34 configurations |
| [mcqa-results-by-category.json](../../results/388/mcqa-results-by-category.json) | The same supported metrics per L1–L5 category, with full-cohort and held-out sections |
| [mcqa-results-by-option-count.json](../../results/388/mcqa-results-by-option-count.json) | The same supported metrics per 2/3/4-choice subgroup, with full-cohort and held-out sections |

Accuracy and coverage are fractions: `0.9298` means `92.98%`. Subgroup records
include `n`, `correct`, invalid/missing counts and coverage. An empty scoped
reference group is `n: 0, accuracy: null`, not zero accuracy.

These files share the static snapshot generated on **2026-10-22 at 10:16 UTC
(12:16 Europe/Zurich)**. The group exports were generated in Slurm job
`4639272` from **272 authenticated prediction files**, using their recorded
hashes. All 34 held-out summaries matched the original results; group counts
reconciled and the independent portable scorer agreed. No new model inference
was performed for that grouping.

Schema 1.1 adds selected-answer Brier, correctness AUROC, full selective curves
and timing/token coverage in Slurm job `4640085`. All original score fields and
group denominators were checked before augmentation. Known-case checks cover
Brier arithmetic, AUROC ties, selective-threshold ties, missing usage, empty
groups and reference timing placeholders. All 272 dedicated MCQA raw allocation
IDs matched Slurm accounting; generic GPU units are counted once, without
adding their typed-GPU TRES entries again. No inference was rerun.

In the overall file, use `heldout_binary_brier_selected_correctness`,
`heldout_correctness_auroc`, `heldout_selective_curve` and `heldout_resources`
within each method. The group files store their unprefixed counterparts within
each `overall`/`groups` summary. `compute_accounting` describes complete retained
MCQA allocations, separate from held-out/subgroup resources; its currency
amounts are labeled estimates, not measured per-question costs.

Use `methods[].heldout.groups` for the main subgroup comparison and
`methods[].all_questions.groups` for full-cohort diagnostics in the two group
files. Preserve experiment IDs and seeds; do not average subgroup percentages
without accounting for their denominators. The overall JSON stores held-out
accuracy; it does not supply all-388 accuracy values.

Earlier baseline scores, later local comparisons, hosted colleague runs and
graph protocols have separate source/configuration identities. Cite the exact
snapshot rather than combining their scores. Static result availability does
not establish completion of the end-to-end graph study.
