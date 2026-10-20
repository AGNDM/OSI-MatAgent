"""Development-only local specialist/router comparison; not the proposal two-LLM test matrix."""
import argparse
import json
from pathlib import Path

from app.decision_backend import RulesBackend
from app.local_decision import LocalOpenJevBackend
from app.local_roles import LocalLlamaRoles
from app.models import Limits
from app.scenarios import case_by_id
from bench.matrix import fingerprint, write_summary
from bench.run import manifest, run_episode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    roles = LocalLlamaRoles()
    decisions = [LocalOpenJevBackend(), RulesBackend()]
    frozen = fingerprint()
    provenance = manifest(Limits(), "dev", 1, [roles], ["dev-01", "dev-03"], decisions[0])
    provenance.update(experiment="four-episode local development comparison; not two independent specialist LLMs",
        baseline=decisions[1].backend_id, frozen_fingerprint=frozen)
    (args.output / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    records = []
    try:
        for index, case_id in enumerate(["dev-01", "dev-03"]):
            for decision in decisions[::1 if index == 0 else -1]:
                if fingerprint() != frozen:
                    raise RuntimeError("Frozen sources changed during comparison")
                r = run_episode(case_by_id(case_id), args.output, roles=roles, decision_backend=decision)
                records.append(r)
                write_summary(args.output, records)
                print(f"{len(records)}/4 {case_id} {decision.backend_id}: success={r['score']['task_success']} cause_covered={r['score'].get('expected_cause_covered')} reason={r['report']['terminal_reason']}", flush=True)
    finally:
        (args.output / "status.json").write_text(json.dumps({"planned": 4, "attempted": len(records),
            "complete": len(records) == 4, "task_success": sum(r["score"]["task_success"] for r in records)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
