"""Run the complete native benchmark through isolated repository LangGraph workers."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time


def read_rows(path):
    if not path.exists():return []
    lines=path.read_text().splitlines();rows=[]
    for index,line in enumerate(lines):
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index!=len(lines)-1:raise
    return rows


def aggregate(output,bundle):
    sys.path.insert(0,str(bundle))
    from eval.sequential import score_episodes
    specs=read_rows(bundle/'sequential_120/evaluator/case-specs.jsonl')
    rows=[]
    for folder in sorted(output.glob('worker-*')):rows.extend(read_rows(folder/'episodes.jsonl'))
    ids=[r['task_id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate native episode in full cohort')
    summary=score_episodes(specs,rows,['free_run'])
    summary.update(cohort_size=len(specs),attempted_cases=len(rows),unattempted_cases=len(specs)-len(rows),
                   all_cases_attempted=len(rows)==len(specs),full_run_completed=len(rows)==len(specs),
                   final_native_scoring_coverage=summary['by_mode']['free_run']['coverage'])
    temp=output/'native-summary.tmp';temp.write_text(json.dumps(summary,indent=2)+'\n');temp.replace(output/'native-summary.json')
    temp=output/'episodes.tmp';temp.write_text(''.join(json.dumps(r)+'\n' for r in rows));temp.replace(output/'episodes.jsonl')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--expert-config',type=Path,default=Path('configs/openrouter-minimax-m3.1-flash-low.json'))
    parser.add_argument('--decision-model',choices=['typesafe/jev-1.14','openai/gpt-6-luna-decisions'],default='typesafe/jev-1.14')
    parser.add_argument('--rate-state',type=Path)
    args=parser.parse_args();bundle=args.bundle.resolve();output=args.output.resolve()
    repo=Path(__file__).resolve().parents[1]
    from app.expert_config import load_plan
    expert_config=args.expert_config.resolve()
    expert_settings=load_plan(expert_config).reference
    if output.exists():raise ValueError('Use a fresh batch output directory')
    if not 1<=args.workers<=16:raise ValueError('Worker count must be in [1,16]')
    tasks=read_rows(bundle/'sequential_120/cases.jsonl')
    ids=sorted(t['task_id'] for t in tasks)
    if len(ids)!=120 or len(ids)!=len(set(ids)):raise ValueError('Expected frozen 120-case cohort')
    random.Random(20261022).shuffle(ids)
    output.mkdir(parents=True)
    execution_ids=list(ids)
    frozen=output/'execution-source';frozen.mkdir()
    source_hashes={}
    for directory in ['app','bench','configs']:
        (frozen/directory).mkdir()
        for p in sorted((repo/directory).glob('*.py' if directory!='configs' else '*.json')):
            destination=frozen/directory/p.name;shutil.copy2(p,destination)
            source_hashes[str(p.relative_to(repo))]=sha256(p.read_bytes()).hexdigest()
    frozen_config=frozen/'configs'/'batch-selected-expert.json'
    shutil.copy2(expert_config,frozen_config)
    source_hashes['configs/batch-selected-expert.json']=sha256(frozen_config.read_bytes()).hexdigest()
    config=dict(schema='osi-full-langgraph-instrument-eval-v1',dataset_commit='22a8192e4e40f3a7d37e1164a8755b006f932d60',
                started_at_utc=datetime.now(timezone.utc).isoformat(),cohort_size=120,all_public_case_ids=ids,
                instrument_families=8,workflow_templates=14,mode='free_run',workers=args.workers,
                expert=expert_settings.model,router_and_operation=args.decision_model,
                expert_timeout_seconds=900,expert_max_tokens=expert_settings.max_tokens,transport_retries=0,
                output_policy='Provider-advertised maximum completion allowance; no local 8192-token cap; no separate reasoning token limit',
                new_execution_case_ids=execution_ids,
                shared_minimax_request_interval_seconds=3.4,selection_seed=20261022,
                filters='None; all 120 sequential cases; 388 static MCQA outside this workflow run',
                source_hashes=source_hashes,source_execution='Frozen source copy; isolated subprocess per worker; independent graph per case',
                failure_policy='Continue after individual policy errors; stop new cases after provider-wide daily quota, unavailable endpoint or payment error',
                paid_fallback=False,token_accounting='Shared durable record-only ledger; missing usage remains unknown')
    (output/'cohort.json').write_text(json.dumps(config,indent=2)+'\n')
    workers=[]
    for index in range(args.workers):
        target=output/f'worker-{index:02d}';subset=execution_ids[index::args.workers]
        command=[sys.executable,'-B','-m','bench.instrument_eval_langgraph','--bundle',str(bundle),
                 '--execution','local-cpu','--mode','free_run','--continue-on-error','--expert-timeout','900',
                 '--expert-config',str(frozen_config),'--decision-model',args.decision_model,
                 '--rate-state',str(args.rate_state.resolve() if args.rate_state else output/'expert-rate-state.txt'),'--rate-interval','3.4',
                 '--ledger',str(output/'api-ledger.jsonl'),'--provider-stop',str(output/'provider-stop.json'),
                 '--output',str(target)]
        for key in subset:command.extend(['--case-id',key])
        stream=(output/f'worker-{index:02d}.log').open('w')
        process=subprocess.Popen(command,cwd=frozen,stdout=stream,stderr=subprocess.STDOUT)
        workers.append((process,stream,index,subset))
    previous=None
    try:
        while True:
            summary=aggregate(output,bundle)
            progress=dict(attempted=summary['attempted_cases'],total=120,unattempted=summary['unattempted_cases'],
                          active_workers=sum(p.poll() is None for p,_,_,_ in workers),
                          native_completed=summary['by_mode']['free_run']['completed'],
                          goals=summary['by_mode']['free_run']['goals_reached'],
                          protocol_failures=summary['by_mode']['free_run']['infrastructure_failures'],
                          provider_stopped=(output/'provider-stop.json').exists())
            (output/'batch-progress.json').write_text(json.dumps(progress,indent=2)+'\n')
            if progress!=previous:print(json.dumps(progress),flush=True);previous=progress
            if not progress['active_workers']:break
            time.sleep(3)
    except BaseException:
        for process,_,_,_ in workers:
            if process.poll() is None:process.terminate()
        raise
    finally:
        for _,stream,_,_ in workers:stream.close()
    final=aggregate(output,bundle)
    (output/'worker-exits.json').write_text(json.dumps([dict(worker=i,exit_code=p.returncode,assigned_ids=subset) for p,_,i,subset in workers],indent=2)+'\n')
    (output/'finished.json').write_text(json.dumps(dict(finished_at_utc=datetime.now(timezone.utc).isoformat(),all_cases_attempted=final['all_cases_attempted']))+'\n')
    return 0 if final['all_cases_attempted'] else 2


if __name__=='__main__':raise SystemExit(main())
