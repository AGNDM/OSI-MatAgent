"""Run two paper-grounded titania branches; no API calls unless explicitly selected."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4

from app.graph import build_graph, initial_state
from app.models import ActionProposal, Limits
from app.trace import Trace
from app.titania import BRANCHES, PAPER, TitaniaLab, TitaniaProfile, ScriptedTitaniaRoles, public_reference


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def action(lab, name, args=None):
    return lab.execute(ActionProposal(sample_id=lab.sample_id, expected_revision=lab.revision,
        capability_id=name, arguments=args or {}, request_id=str(uuid4()), postcondition='Probe native evidence.'))


def summarize_usage(events):
    """Missing provider usage stays visibly incomplete; durable ledger charges it."""
    keys = ['prompt_tokens', 'completion_tokens', 'total_tokens']
    total = dict.fromkeys(keys, 0)
    complete = True
    for event in events:
        metadata = event.get('llm_metadata') or {}
        if not metadata.get('request_sent'):
            continue
        usage = metadata.get('usage')
        valid = (isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0 for k in keys)
                 and usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'])
        if not valid:
            complete = False
            continue
        for key in keys:
            total[key] += usage[key]
    return total, complete


def run(branch, output, roles=None, decision=None, operation=None):
    lab = TitaniaLab(branch)
    run_id = str(uuid4())
    trace = Trace(run_id, output/'traces'/(run_id+'.jsonl'))
    dump(output/(branch+'-reset.json'), {'fixture_events': lab.fixture_events,
         'thermal_runs': lab.thermal_runs, 'initial_analysis': lab.initial_analysis,
         'initial_pattern': lab._pattern, 'template_references': {p: public_reference(p) for p in ['anatase', 'rutile']}})
    backend = roles or ScriptedTitaniaRoles()
    limits = Limits(lab_actions=8, llm_calls=16, episode_seconds=240, graph_steps=120)
    graph = build_graph(lab, trace, limits, roles=backend, decision_backend=decision, profile=TitaniaProfile(), operation_backend=operation)
    final = graph.invoke(initial_state(lab, run_id, limits),
        {'configurable': {'thread_id': run_id}, 'recursion_limit': limits.graph_steps})
    score = lab.score(final['report'])
    trace.emit('evaluator', 'titania_score', score=score)
    usage, usage_complete = summarize_usage(trace.events)
    row = {'branch': branch, 'backend': backend.backend_id,
           'router': decision.backend_id if decision else 'rules-v1',
           'operation_backend': operation.backend_id if operation else None,
           'report': final['report'], 'score': score, 'usage': usage, 'usage_complete': usage_complete,
           'router_fallbacks': sum(e.get('decision', {}).get('fallback_used', False) for e in trace.events),
           'actions': [e['proposal']['capability_id'] for e in lab.history],
           'trace': str(trace.path), 'final_analysis': lab._measurement['analysis'] if lab._measurement else None}
    dump(output/(branch+'-run.json'), row)
    dump(output/(branch+'-final-pattern.json'), lab._pattern)
    print(json.dumps(row), flush=True)
    return row


def negative_controls(output):
    rows = []
    for branch in BRANCHES:
        lab = TitaniaLab(branch)
        action(lab, 'workflow.inspect')
        if branch == 'scan_coverage':
            action(lab, 'thermal.prepare_reference')  # needless heat cannot repair a truncated method
        action(lab, 'xrd.acquire', {'stop_deg': 60 if branch == 'thermal_history' else 26})
        # Deliberately optimistic verifier to demonstrate independent rejection.
        report = {'status': 'supported', 'evidence_ids': [lab._measurement['id']]}
        rows.append({'branch': branch, 'deliberately_wrong_repair': [e['proposal']['capability_id'] for e in lab.history],
                     'score': lab.score(report), 'final_analysis': lab._measurement['analysis']})
    dump(output/'negative-controls.json', rows)
    return rows


def audit(output):
    lab = TitaniaLab('thermal_history')
    action(lab, 'workflow.inspect')
    action(lab, 'xrd.acquire', {'stop_deg': 60})
    denied = lab.x.read(instance_id='xrd-001', state_ids=['template_sample_truth'])
    result = {'paper': PAPER, 'native_furnace': 'NaberthermL9Runtime / CHEM-185',
              'native_xrd': 'RigakuMiniFlex600Runtime / CHEM-160',
              'scene_adapter': 'In-memory callback receipts, no visual scene or physical hardware.',
              'furnace_program_completion': lab.thermal_runs[0]['completion'],
              'furnace_program_poll_count': len(lab.thermal_runs[0]['polls']),
              'safe_transfer_temperature_k': lab.thermal_runs[0]['transfer_temperature_k'],
              'scan_start': lab.last_scan_states[0],
              'scan_partial': lab.last_scan_states[len(lab.last_scan_states)//2],
              'scan_end': lab.last_scan_states[-1],
              'clock_s': {'scheduler': lab.time_s, 'furnace': lab.furnace.time_s, 'xrd': lab.xrd.time_s},
              'agent_truth_query': denied,
              'limitations': ['Furnace is an empty-furnace thermal model; temperature is a sample proxy.',
                  'Finite chemistry endpoints and logical aliquot transfer are authored fixtures.',
                  'XRD integrates a host-loaded RRUFF-derived profile; acquired data are synthetic.',
                  'Only qualitative phase evidence is scored; no calibrated fractions, mass balance or kinetics.',
                  'Native async start is awaited inside a bounded macro, not a general asynchronous graph scheduler.']}
    dump(output/'acquisition-audit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--backend', choices=['scripted', 'deepseek'], default='scripted')
    parser.add_argument('--router', choices=['rules', 'openjev'], default='rules')
    parser.add_argument('--branch', choices=BRANCHES, action='append')
    parser.add_argument('--controls', action='store_true')
    parser.add_argument('--operation-decision', choices=['off', 'scripted', 'openjev', 'deepseek'], default='off',
                        help='Separate next-operation chooser; expert router remains independent')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory; attempts are retained.')
    roles = ledger = None
    if args.backend == 'deepseek' or args.operation_decision == 'deepseek':
        from app.expert_config import load_plan
        from app.token_budget import BudgetedExpert, TokenLedger
        ledger = TokenLedger()
        config = load_plan(Path('configs/openrouter-deepseek-v4.1-flash.json')).reference
        if args.backend == 'deepseek':
            roles = BudgetedExpert(config, ledger)
    decision = None
    if args.router == 'openjev':
        from app.local_decision import LocalOpenJevBackend
        decision = LocalOpenJevBackend()
    operation = None
    if args.operation_decision != 'off':
        from app.operation_decision import LocalOperationBackend, ExpertOperationBackend, ScriptedOperationBackend
        if args.operation_decision == 'openjev':
            operation = LocalOperationBackend()
        elif args.operation_decision == 'scripted':
            operation = ScriptedOperationBackend()
        else:
            operation = ExpertOperationBackend(BudgetedExpert(config, ledger))
    args.output.mkdir(parents=True)
    manifest = {'paper': PAPER, 'branches': args.branch or list(BRANCHES), 'backend': args.backend,
                'router': args.router, 'operation_decision': args.operation_decision, 'budget_before': ledger.snapshot() if ledger else None,
                'source_hashes': {str(p): sha256(p.read_bytes()).hexdigest()
                    for p in [Path('app/titania.py'), Path('bench/titania.py'), Path('app/graph.py'),
                              Path('app/operation_decision.py'), Path('app/local_decision.py'), Path('app/models.py'),
                              Path('vendor/chemlab/expansion-code-manifest.json'), Path('vendor/chemlab/expansion-data-manifest.json')]}}
    dump(args.output/'manifest.json', manifest)
    rows = [run(branch, args.output, roles, decision, operation) for branch in manifest['branches']]
    controls = negative_controls(args.output) if args.controls else []
    if args.controls:
        audit(args.output)
    summary = {'runs': rows, 'negative_controls': controls,
               'budget_after': ledger.snapshot() if ledger else None}
    dump(args.output/'summary.json', summary)


if __name__ == '__main__':
    main()
