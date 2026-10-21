# Documentation

- [Astra code/report review](evidence/single-instrument-code-review.md): fixed verification liveness/citation blockers, independent re-review and commit validation.

## Start here

- [Native lifecycle coverage without a device whitelist](reports/native-lifecycle-coverage.md): all 194 devices,191 actual clock checks,149 async interfaces and retained failures.

- [Current native acceptance and knowledge report](reports/native-acceptance-and-knowledge.md): automatic instrument selection, argmax routing, no reference solution script, all outcomes and retrieval limits.
- [Native single-instrument agent](guides/single-instrument-agent.md): the primary live entry point.

- [Current progress](status/progress.md) and [delivery checklist](status/checklist.md)
- [Titania scientific scenario](demos/titania/scenario.md) and [reproduction guide](demos/titania/workflow.md)
- [Recorded evaluation summary](reports/evaluation-summary.md)
- [Binder timer and furnace demo for benchmark development](reports/single-instrument-demo-for-benchmark.md): complete model decisions, native calls, public state transitions, scoring and benchmark design requirements.
- [Detailed demo evidence](evidence/single-instrument-demo-details.json): machine-readable tasks, per-step snapshots, requests, probabilities and provenance hashes.

## Guides

- [Online next-operation decisions](guides/operation-decision.md): candidate generation, action chooser, gate handoff and demo modes; offline benchmark deferred.
- [Native-device workflow and API budget](guides/device-workflow.md)
- [Local OpenJev deployment](guides/local-openjev.md)
- [Evaluation commands and interpretation](guides/experiments.md)
- [Drying demonstration](demos/drying.md)

## Plans and reference

- [Collaborator's decision-mechanism benchmark proposal](plans/decision-mechanisms-proposal.md): proposed research direction; the generator and comparison are pending.
- [Astra review and staged decision-benchmark plan](plans/decision-benchmark-review-and-plan.md): proposed alignment, feasibility gates and evaluation safeguards; no benchmark adoption or execution claimed.
- [Existing workflow proposal](plans/workflow-proposal.md): implementation background.
- [Historical milestone plan](plans/archive/milestones.md) and [earlier detailed proposal](plans/archive/detailed-proposal.md): archived design history, not current completion claims.
- [Implementation reference guide](reference/implementation-guide.md)
- [Error annotation codebook](reference/error-codebook.md)
- [Third-party components and provenance](reference/third-party.md)
- [Astra feature review and release fixes](evidence/operation-feature-review.md)
- [Execution evidence index](evidence/README.md)

Run all project commands from the repository root. Generated traces and the active token ledger stay under the ignored `results/` directory; compact publication evidence is under `docs/evidence/`.
