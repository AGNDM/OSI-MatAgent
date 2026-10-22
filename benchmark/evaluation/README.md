# Evaluation and reporting tools

These tools process explicitly supplied local runs. They include no model weights, prediction journals or published result JSON. On SCITAS, their existing guards require a genuine Slurm compute allocation.

| Path | Purpose |
|---|---|
| `mcqa/export_overall.py`, `mcqa/export_groups.py` | Aggregate static predictions with original scorers |
| `mcqa/metrics.py`, `mcqa/accounting.py`, `mcqa/enrich.py` | Calibration, usage and resource summaries |
| `workflow/render_plan.py` | Resolve explicit model/artifact path bindings |
| `workflow/report_grid.py`, `workflow/compact_status.py` | Summarize saved workflow cells |
| `workflow/audit_failures.py`, `workflow/final_audit.py` | Inspect supplied execution evidence |
| `prepare_profiles.py`, `prepare_static_profiles.py` | Build portable profiles from authenticated source runs |

The retained plans are inputs for new studies: `configs/workflow-e-plan.json` and both `configs/static-388/*-plan.json` files. Fill a copy of the corresponding bindings example, then render to a fresh output path:

```bash
python benchmark/evaluation/workflow/render_plan.py \
  --plan benchmark/evaluation/configs/workflow-e-plan.json \
  --bindings /absolute/path/to/bindings.json \
  --output /absolute/path/to/fresh-plan.json
```

Use the rendered workflow plan with [graph preparation](../graph/README.md) and explicitly select `teacher_forced`. Each exporter's `--help` lists required snapshots, metadata and evidence inputs. Older audit utilities intentionally retain their original study-specific validation rules; use them only with compatible evidence. The dated tariff file is an optional input, not a current price quote.
