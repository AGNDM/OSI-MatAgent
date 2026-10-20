"""Paired, fixed-observation routing probe; annotations never enter model input."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from app.decision_backend import RulesBackend, select_route
from app.local_decision import LocalOpenJevBackend

ROUTES = ["diagnose", "prepare_action", "verify", "deliberate"]


def snapshots(split):
    records = []
    for route in ROUTES:
        for variant in range(3):
            summary = {"stage": "diagnose", "has_intent": False, "has_proposal": False,
                "has_inspection": variant > 0, "has_measurement": False,
                "measurement_quality": "unknown", "measurement_fresh": False,
                "pending_operation": False, "verifier_disagreement": False,
                "actions_used": variant if split == "dev" else variant + 2,
                "llm_calls_used": variant if split == "dev" else variant + 4}
            acceptable = [route]
            if route == "prepare_action":
                summary.update(stage="prepare", has_intent=True, has_inspection=True)
            elif route == "verify":
                summary.update(stage="verify", has_inspection=True, has_measurement=True,
                    measurement_fresh=True, measurement_quality=["valid", "suspect", "unavailable"][variant])
            elif route == "deliberate":
                summary.update(stage="verify", has_inspection=True, has_measurement=True,
                    measurement_fresh=variant != 1, measurement_quality="suspect",
                    verifier_disagreement=True)
                # With conflicting evidence both diagnostic reinspection and bounded escalation are defensible.
                acceptable = ["deliberate", "diagnose"]
            records.append({"id": f"{split}-{route}-{variant+1}", "primary_route": route,
                "acceptable_routes": acceptable, "summary": summary,
                "annotation_status": "single-author protocol annotation; no independent human review"})
    return records


def run_probe(output, splits):
    output.mkdir(parents=True, exist_ok=False)
    corpus = {s: snapshots(s) for s in splits}
    frozen = json.dumps(corpus, sort_keys=True, indent=2) + "\n"
    (output / "snapshots.json").write_text(frozen)
    model = LocalOpenJevBackend()
    (output / "manifest.json").write_text(json.dumps({"snapshot_hash": sha256(frozen.encode()).hexdigest(),
        "source_hash": sha256(Path(__file__).read_bytes()).hexdigest(), "decision": model.manifest(),
        "thresholds": {"min_top_probability": .6, "min_margin": .15},
        "test_policy": "Authored before model calls, no tuning on held-out results; test varies counters and freshness, narrow generalization"}, indent=2))
    records = []
    for split, items in corpus.items():
        for item in items:
            for backend in [RulesBackend(), model]:
                for order, allowed in [("original", ROUTES), ("reversed", list(reversed(ROUTES)))]:
                    decision, error = select_route(backend, item["summary"], allowed)
                    probabilities = decision.probabilities
                    argmax = max(probabilities, key=probabilities.get) if probabilities else None
                    record = {"split": split, "snapshot_id": item["id"], "backend": backend.backend_id,
                        "option_order": order, "allowed_routes": allowed, "decision": decision.model_dump(),
                        "raw_argmax": argmax, "correct": decision.choice in item["acceptable_routes"],
                        "error": error, "raw": getattr(backend, "last_call", None)}
                    records.append(record)
                    with (output / "decisions.jsonl").open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
            print(f"{item['id']} recorded", flush=True)
    summary = []
    for split in splits:
        for backend in [RulesBackend.backend_id, model.backend_id]:
            selected = [r for r in records if r["split"] == split and r["backend"] == backend]
            pairs = [[r for r in selected if r["snapshot_id"] == i["id"]] for i in corpus[split]]
            summary.append({"split": split, "backend": backend, "attempted": len(selected),
                "acceptable_count": sum(r["correct"] for r in selected),
                "fallback_count": sum(r["decision"]["fallback_used"] for r in selected),
                "valid_model_readouts": sum(r["raw_argmax"] is not None and not r["decision"]["fallback_used"] for r in selected),
                "order_invariant_pairs": sum(a["decision"]["choice"] == b["decision"]["choice"] for a, b in pairs),
                "paired_snapshots": len(pairs),
                "choices": dict(Counter(r["decision"]["choice"] for r in selected))})
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["dev", "test", "both"], default="both")
    args = p.parse_args()
    run_probe(args.output, ["dev", "test"] if args.split == "both" else [args.split])


if __name__ == "__main__":
    main()
