# Scenario card: missing rutile evidence after titania processing

**Question.** Did the batch receive the wrong thermal treatment, or did characterization miss the relevant diffraction range? Recover qualitative evidence for both anatase and rutile while preserving a correctly processed aliquot.

**Published anchor.** Catauro et al., *Materials* **11** (2018), 2364, [doi:10.3390/ma11122364](https://doi.org/10.3390/ma11122364), [open paper](https://res.mdpi.com/d_attachment/materials/materials-11-02364/article_deploy/materials-11-02364.pdf). Section 2.2 and Tables 1–2 compare sister aliquots from a dried sol–gel batch: 9 °C/min ramps and two-hour holds at 400 °C produced anatase, while 600 °C produced anatase plus rutile. Section 2.3.2 uses Cu Kα XRD. We model only this thermal-treatment/characterization substep; wet synthesis and drying are prerequisites outside the workflow.

**Common initial alarm.** Expected rutile reflections were not confirmed. Inspect delivered furnace history, current aliquot identity, scan range and acquired peaks before choosing a repair. The expert receives neither the branch name nor the host's phase fractions.

| Branch | Public distinguishing evidence | Appropriate intervention | Wrong-repair control |
|---|---|---|---|
| Thermal history | Delivered 400 °C; complete 20–60° scan lacks rutile peaks | Prepare a retained dried sister aliquot with the reference program, cool, then acquire a new full scan | A full rescan of the original aliquot cannot create missing phase evidence |
| Scan coverage | Delivered reference treatment; earlier scan spans only 20–26° | Preserve the processed aliquot and acquire 20–60° | Extra heat with the truncated method still cannot establish the target; unnecessary replacement also violates preservation policy |

**Runnable path.** Existing LangGraph: diagnose → procedure → schema/identity/revision gate → execute → observe → verify. The local OpenJev router selects graph transitions; DeepSeek supplies specialist records. `thermal.prepare_reference` composes native furnace commands and bounded public-state polling. `xrd.acquire` composes mounting/configuration/start/wait/read commands. Aliquot transfer is logical, and furnace and XRD share one simulation clock.

**Acceptance.** Fresh, identity-matched, complete 801-point scan over 20–60°; at least three rutile and two anatase reference-window peak matches; cited acquisition evidence; branch-specific preservation policy. Independent evaluation re-reads native acquired points rather than accepting the expert's claim. A wrong repair must fail even with an optimistic verifier.

**Evidence boundary.** Furnace dynamics are native but describe an empty furnace. A host-authored endpoint lookup supplies the XRD response from pinned RRUFF references (anatase R060277, rutile R040049). The curve, profile assumptions and phase mixing are synthetic. Neither paper replication, calibrated kinetics nor quantitative phase fractions are established. Sister-aliquot preparation avoids inventing a sequential-reheating result. The scan range, matching thresholds and 313.15 K transfer guard are authored demonstration choices. Prompts contain an explicit repair policy; this is a workflow demonstration, not a blind reasoning benchmark.

Run instructions and acquisition audit: [titania-workflow.md](titania-workflow.md).
