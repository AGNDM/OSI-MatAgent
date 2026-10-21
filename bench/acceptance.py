"""Frozen development acceptance. Offline first; explicit bounded live execution."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time

from app.models import Limits,ActionProposal
from app.single_instrument import InstrumentTask
from app.token_budget import TokenLedger

WINDOW='demo-materials-argmax-20261021'
INCREMENTAL_LIMIT=1_000_000


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def hashes(directory,pattern):
    return {str(p):sha256(p.read_bytes()).hexdigest() for p in sorted(Path(directory).rglob(pattern))}


def fingerprint():
    return {**hashes('app','*.py'),**hashes('bench','*.py'),**hashes('tests','*.py'),**hashes('cases/acceptance','*.json'),
            **hashes('cases/dev','*.json'),**hashes('corpus','*.json'),**hashes('configs','*.json'),
            'cases/devices/registry.json':sha256(Path('cases/devices/registry.json').read_bytes()).hexdigest(),
            'cases/devices/public-catalog.json':sha256(Path('cases/devices/public-catalog.json').read_bytes()).hexdigest(),
            **hashes('vendor/chemlab','*manifest.json')}


def offline(output):
    from tests.native_acceptance_fixtures import run_checks
    rows=run_checks(output)
    tests=[
      'tests/test_catalog_and_budget.py::test_catalog_pages_cover_all_registered_devices_without_reference_fields',
      'tests/test_catalog_and_budget.py::test_unknown_device_paths_are_not_resolved',
      'tests/test_catalog_and_budget.py::test_expert_directory_candidates_feed_jev_without_changing_oracle',
      'tests/test_catalog_and_budget.py::test_argmax_failure_stops_without_rules_fallback',
      'tests/test_catalog_and_budget.py::test_budget_window_survives_restart_and_directory_changes',
      'tests/test_router_wait.py::test_stale_router_wait_cannot_advance_clock',
      'tests/test_router_wait.py::test_direct_wait_obeys_action_budget_before_native_step',
      'tests/test_router_wait.py::test_idle_process_and_unrelated_fan_cannot_offer_wait',
      'tests/test_verification_progress.py::test_redundant_reads_and_diagnosis_cannot_reverify_same_semantic_evidence',
      'tests/test_verification_progress.py::test_supported_without_citation_gets_bounded_repair_before_success',
      'tests/test_cloud_jev_and_verification.py::test_cloud_failure_does_not_execute_rules_fallback',
      'tests/test_single_instrument.py::test_public_contract_and_optional_native_arguments']
    for test in tests:
        result=subprocess.run([sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',test],capture_output=True,text=True)
        rows.append({'id':test,'track':'offline-protocol','accepted':result.returncode==0,'output':result.stdout[-1500:]})
    return rows


def backends(ledger):
    from app.expert_config import load_plan
    from app.token_budget import BudgetedExpert
    from app.openrouter_jev import OpenRouterJevBackend,OpenRouterJevOperationBackend
    config=load_plan(Path('configs/openrouter-deepseek-v4.1-flash.json')).reference
    config.max_tokens=6144
    return BudgetedExpert(config,ledger),OpenRouterJevBackend(ledger),OpenRouterJevOperationBackend(ledger)


def live(phase,output,frozen):
    ledger=TokenLedger(window_id=WINDOW,window_limit=INCREMENTAL_LIMIT)
    ledger.enable_openrouter_record_only()
    initial=ledger.snapshot();dump(output/'budget-before.json',initial)
    if initial['inflight_or_crashed'] or initial['accounting_violation'] or initial['window']['unknown_usage_detected']:raise RuntimeError('Unresolved ledger state; no requests sent')
    rows=[]
    if phase=='acceptance':
        jobs=[('balance-filter',None),('ika-stirring',None),('binder-timer',None),('furnace-program',None)]
    else:
        jobs=[('furnace-program',False),('furnace-program',True),('ika-stirring',True),('ika-stirring',False)]
    for name,enabled in jobs:
        if fingerprint()!=frozen:raise RuntimeError('Frozen sources changed; remaining attempts stopped')
        if not ledger.record_only_openrouter and ledger.snapshot()['window']['remaining_tokens'] < 262144+8192+24000:
            dump(output/'stop.json',{'reason':'insufficient_full_hosted_reservation_for_next_episode'})
            break
        attempt=ledger.claim_episode('knowledge-record-only-v4' if phase=='retrieval' else phase)
        roles,router,operation=backends(ledger)
        started=time.monotonic();directory=output/f'{name}-{enabled if enabled is not None else "live"}'
        try:
            from app.instrument_cli import run
            from app.material_retrieval import KnowledgeInstrumentProfile
            task=InstrumentTask.model_validate_json(Path(f'cases/acceptance/v1/{name}.json').read_text())
            profile=KnowledgeInstrumentProfile(enabled) if phase=='retrieval' else None
            result=run(task,directory,roles,router=router,operation=operation,profile=profile,
                limits=Limits(lab_actions=32,llm_calls=48,episode_seconds=360,graph_steps=260))
            row={'id':attempt,'task':name,'retrieval_enabled':enabled,'track':'live-development',
                 'result':result,'elapsed_seconds':time.monotonic()-started}
        except Exception as exc:
            row={'id':attempt,'task':name,'retrieval_enabled':enabled,'track':'live-development',
                 'error':{'type':type(exc).__name__,'message':str(exc)[:1000]}}
        row['source_match_after']=fingerprint()==frozen
        rows.append(row);dump(output/'runs.json',rows)
        after=ledger.snapshot();dump(output/'budget-after.json',after)
        # Stop on unknown/crashed accounting or actual provider errors. Known
        # model/protocol failures remain distinct and retain their outcome.
        provider_failure=False
        for path in directory.rglob('*.jsonl') if directory.exists() else []:
            for line in path.read_text().splitlines():
                event=json.loads(line);m=event.get('llm_metadata') or event.get('backend_metadata') or {}
                provider_failure |= bool(m.get('http_status',200)>=400 or m.get('response_classification')=='expert_provider_error')
        if (after['unmetered_requests']>initial['unmetered_requests'] or after['inflight_or_crashed'] or
                after['accounting_violation'] or provider_failure or 'error' in row or not row['source_match_after']):
            dump(output/'stop.json',{'reason':'provider_or_budget_or_infrastructure_failure','attempt':attempt})
            break
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['offline','acceptance','retrieval'],default='offline')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.phase!='offline' and not args.execute:parser.error('Live phases require --execute')
    if args.output.exists():parser.error('Use a fresh output directory; historical attempts are immutable')
    args.output.mkdir(parents=True);frozen=fingerprint()
    dump(args.output/'freeze.json',{'version':'acceptance-native-v2' if args.phase!='retrieval' else 'acceptance-native-v4-knowledge-record-only',
        'parent_version':'acceptance-v1-live + native-v2-offline' if args.phase=='retrieval' else None,'phase':args.phase,'source_hashes':frozen,
        'routing_policy':'argmax-v1','budget_window':WINDOW,'historical_incremental_cap':INCREMENTAL_LIMIT,'openrouter_budget_policy':'record-only by user instruction',
        'non_openrouter_budget_cap':5000000,'new_knowledge_attempt_cap':4,
        'historical_phases_total_cap':8,'historical_phases_per_phase_cap':4,
        'new_phase':'knowledge-record-only-v4' if args.phase=='retrieval' else None,
        'new_phase_attempt_cap':4 if args.phase=='retrieval' else None,
        'policy_change_reason':'User removed OpenRouter quota restrictions; retain fixed episode and phase limits',
        'record_only_scope':{'provider':'openrouter','models':['deepseek/deepseek-v4.1-flash','typesafe/jev-1.14']},
        'execution_settings':{'instrument_limits':Limits(lab_actions=32,llm_calls=48,episode_seconds=360,graph_steps=260).model_dump(),
                              'expert_output_tokens':6144,'jev_timeout_seconds':45,'transport_retries':0},
        'models':{'expert':'deepseek/deepseek-v4.1-flash','router_and_chooser':'typesafe/jev-1.14'},
        'scope':'Development acceptance, not a released or held-out benchmark; UI deferred'})
    import shutil
    for name in frozen:
        target=args.output/'frozen-source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
    rows=offline(args.output) if args.phase=='offline' else live(args.phase,args.output,frozen)
    dump(args.output/'summary.json',{'phase':args.phase,'attempted':len(rows),'runs':rows,
        'accepted_checks':sum(r.get('accepted',False) for r in rows) if args.phase=='offline' else None})
    print(json.dumps({'phase':args.phase,'attempted':len(rows),'output':str(args.output)}))


if __name__=='__main__':main()
