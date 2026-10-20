import argparse
import json
from pathlib import Path
from uuid import uuid4

from app.scenarios import ROOT, case_by_id, load_cases
from bench.run import manifest, run_episode
from app.models import Limits
from app.roles import ScriptedRoles


def main():
    parser = argparse.ArgumentParser(description="Run the LangGraph lab skeleton with scripted roles")
    parser.add_argument("--case", default="dev-01")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backend", choices=["scripted", "openrouter", "local", "configured"], default="scripted")
    parser.add_argument("--expert-config", type=Path, help="ExpertPlan JSON, with a configured reference endpoint")
    parser.add_argument("--model", help="Explicit :free model ID; required with --backend openrouter")
    parser.add_argument("--router", choices=["rules", "openjev"], default="rules")
    parser.add_argument("--decision-url", default="http://127.0.0.1:8792")
    args = parser.parse_args()
    if args.list:
        for case in load_cases():
            print(case.case_id, case.family)
        return
    output = args.output or ROOT / "results" / f"demo-{uuid4().hex[:10]}"
    if output.exists() and any(output.iterdir()):
        parser.error("output directory must be new or empty")
    roles = None
    from app.decision_backend import RulesBackend
    decision = RulesBackend()
    if args.router == "openjev":
        from app.local_decision import LocalOpenJevBackend
        decision = LocalOpenJevBackend(args.decision_url)
    if args.backend == "openrouter":
        if not args.model:
            parser.error("--model is required for OpenRouter; paid model IDs are refused")
        from app.openrouter import OpenRouterRoles
        roles = OpenRouterRoles(args.model)
    elif args.backend == "local":
        from app.local_roles import LocalLlamaRoles
        roles = LocalLlamaRoles(args.decision_url)
    elif args.backend == "configured":
        if not args.expert_config:
            parser.error("--expert-config is required for configured general specialists")
        from app.expert_config import CompatibleExpert, load_plan
        roles = CompatibleExpert(load_plan(args.expert_config).reference)
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest(Limits(), "dev", 1, [roles or ScriptedRoles()], [args.case], decision), indent=2) + "\n")
    result = run_episode(case_by_id(args.case), output, roles=roles, decision_backend=decision)
    print(("LIVE LLM" if roles else "Scripted role test doubles") + " + " + decision.backend_id + " + native oven")
    print(json.dumps(result, indent=2))
    if args.router == "openjev" and not result["decision_integration_ok"]:
        print("Local decision-model integration did not pass; fallback is not a model success.")
        raise SystemExit(2)
    if not result["score"]["task_success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
