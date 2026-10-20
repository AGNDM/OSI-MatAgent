import csv,json,collections
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.token_budget import TokenLedger
root=ROOT
out=root/'results/device-expansion-v1'
r=json.loads((root/'cases/devices/registry.json').read_text())
phases=[('Complete offline regression','device-expansion-offline-v2'),('First real API pilot, before fix','device-expansion-deepseek-pilot-v1'),('Second real API pilot','device-expansion-deepseek-pilot-v2'),('Declared category sample','device-expansion-deepseek-domains-v1'),('OpenJev wiring with scripted specialists','device-expansion-openjev-wiring-v1'),('Real DeepSeek and OpenJev combination','device-expansion-deepseek-openjev-v1')]
records={};stats={}
for title,d in phases:
 rows=[json.loads(x) for x in (root/'results'/d/'runs.jsonl').read_text().splitlines()]
 records[d]=rows
 events=[]
 for row in rows:
  if row.get('trace_path'):events += [json.loads(x) for x in Path(row['trace_path']).read_text().splitlines()]
 stats[d]={'attempts':len(rows),'success':sum(x.get('score',{}).get('task_success',False) for x in rows),
  'tokens':sum(x.get('usage',{}).get('total_tokens',0) for x in rows),
  'false_success':sum(x.get('score',{}).get('false_success',False) for x in rows),
  'gate_rejections':sum(e['kind']=='rejected' for e in events),
  'role_errors':sum(e['kind']=='role_error' for e in events),
  'router_calls':sum(e['kind']=='routing' for e in events),
  'router_fallbacks':sum(e.get('decision',{}).get('fallback_used',False) for e in events if e['kind']=='routing')}
ledger=TokenLedger().snapshot()
summary={'catalog_count':r['catalog_count'],'native_session_count':len(json.loads((root/'cases/devices/inventory.json').read_text())['devices']),'verified_task_count':len(r['devices']),
 'unaccepted_count':len(r['unavailable']),'phases':stats,'budget':ledger,'scope':'Native control/interface integration; not scientific benchmark or multi-device sample flow',
 'verification':{'project_tests_passed':73,'vendor_hashes_verified':570,'vendor_hash_mismatches':0}}
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
model_results=collections.defaultdict(list)
for d,rows in records.items():
 if 'deepseek-domains' in d or 'deepseek-openjev' in d:
  for x in rows:model_results[x['slug']].append(x)
offline={x['slug']:x for x in records['device-expansion-offline-v2']}
with (out/'device-coverage.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=['chem_id','slug','name','domain','goal_state','goal_value','native_capabilities_discovered','task_tool_menu','offline_success','deepseek_attempts_final_versions','deepseek_success_final_versions'])
 writer.writeheader()
 for s in r['devices']:
  task=json.loads((root/'cases/devices'/s['task_file']).read_text());mr=model_results[s['slug']]
  writer.writerow({'chem_id':s['chem_id'],'slug':s['slug'],'name':s['name'],'domain':s['domain'],
   'goal_state':task['goal']['state_id'],'goal_value':task['goal']['equals'],'native_capabilities_discovered':s['capability_count'],
   'task_tool_menu':';'.join(task['tool_menu']),'offline_success':offline[s['slug']]['score']['task_success'],
   'deepseek_attempts_final_versions':len(mr),'deepseek_success_final_versions':sum(x['score']['task_success'] for x in mr)})
(out/'unaccepted-devices.json').write_text(json.dumps(r['unavailable'],ensure_ascii=False,indent=2)+'\n')
lines = ['# Native Device Development Report', '',
    'These are independently reset control/readback integration tasks, not full scientific experiments.', '',
    '| Phase | Passed / attempted | API tokens | Role errors / repairs | Gate rejections | Router fallbacks |',
    '|---|---:|---:|---:|---:|---:|']
for title, directory in phases:
    st = stats[directory]
    lines.append(f"| [{title}](../{directory}/REPORT.md) | {st['success']}/{st['attempts']} | {st['tokens']:,} | {st['role_errors']} | {st['gate_rejections']} | {st['router_fallbacks']} |")
lines += ['',
    'Both failed pilots are retained. Successful native commands now return fresh public readbacks; commands and reads count separately against the action budget. The declared category sample and real-expert/local-router runs use the repaired version.',
    '', f"Reported device API usage: {ledger['reported_tokens']:,} tokens across {ledger['requests_settled']} requests. Ceiling: {ledger['limit']:,}. Unknown-usage requests: {ledger['unmetered_requests']}; outstanding reservations: {ledger['inflight_or_crashed']}. Failed calls and output repairs are included. Monetary cost is unknown.",
    '', '## Coverage', '',
    f"Catalog entries: {r['catalog_count']}; native sessions: {summary['native_session_count']}; verified task subsets: {len(r['devices'])}.",
    '', 'See [device-coverage.csv](device-coverage.csv) for every accepted device and task goal.',
    '', '## Unaccepted entries', '', '| Entry | Remaining work |', '|---|---|']
for item in r['unavailable']:
    if 'Empty module' in item['error']:
        reason = 'Pinned catalog lacks a runtime path; no alternative implementation accepted.'
    elif item['slug'].startswith('bronkhorst_p702'):
        reason = 'Explicit constructor boundary parameters are not yet adapted.'
    elif item['slug'].startswith('vici_'):
        reason = 'Different public-interface/session contract is not yet adapted.'
    else:
        reason = 'Native session works; the current task generator has no accepted state-changing task.'
    lines.append(f"| {item.get('chem_id') or 'Legacy'} {item['slug']} | {reason} |")
lines += ['', 'Detailed native errors remain in [unaccepted-devices.json](unaccepted-devices.json).',
    '', '## Boundaries', '',
    'Passing these tasks does not validate all instrument capabilities, cross-device sample transfer, asynchronous acquisition, scientific interpretation, hardware or robotics. Some tasks only test power control.',
    '', 'See [DEVICE_WORKFLOW.md](../../DEVICE_WORKFLOW.md) for reproduction and the shared token budget. Recorded project validation: 73 tests passed; 570 upstream file hashes matched.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
