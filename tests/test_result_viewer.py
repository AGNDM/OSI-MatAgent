import json
from pathlib import Path

import pytest

from app.result_viewer import load_run, list_runs, response_for, sanitize


def write_run(root):
    run=root/'local-live-example'
    run.mkdir()
    (run/'result.json').write_text(json.dumps({'task_id':'example','score':{'task_success':True},'report':{'llm_call_count':1}}),encoding='utf-8')
    (run/'manifest.json').write_text(json.dumps({'task':{'objective':'A task'}}),encoding='utf-8')
    event={'event_id':'call-1','timestamp':'2026-10-22T00:00:00Z','kind':'role_output','component':'diagnostic',
           'llm_metadata':{'request_sent':True,'budget_reservation_id':'reservation-1','usage':{'cost':.01},
                           'request':{'messages':[{'role':'user','content':'A question'}]},
                           'response':{'choices':[{'message':{'content':'An answer'}}]}}}
    (run/'selection-trace.jsonl').write_text(json.dumps(event)+'\n'+json.dumps(event)+'\n{broken',encoding='utf-8')
    return run


def test_preserves_questions_answers_and_deduplicates_cost(tmp_path):
    run=write_run(tmp_path)
    data=load_run(tmp_path,run.name)
    assert data['entries'][0]['event']['llm_metadata']['request']['messages'][0]['content']=='A question'
    assert data['entries'][0]['event']['llm_metadata']['response']['choices'][0]['message']['content']=='An answer'
    assert data['summary']['logged_model_calls']==1
    assert data['summary']['cost_usd']==.01
    assert len(data['warnings'])==1


@pytest.mark.parametrize('run_id',['../.env','..%2f.env','/absolute','x/../run','..'])
def test_rejects_traversal(tmp_path,run_id):
    with pytest.raises((ValueError,FileNotFoundError)):
        load_run(tmp_path,run_id)


def test_masks_credentials_without_dropping_token_usage():
    data=sanitize({'api_key':'secret','prompt':'Bearer sensitive-token sk-or-v1-testkey','usage':{'total_tokens':100},
                   'message':{'reasoning_content':'not displayed','content':'visible answer'}})
    assert data['api_key']=='[redacted]'
    assert 'sensitive-token' not in data['prompt'] and 'testkey' not in data['prompt']
    assert data['usage']['total_tokens']==100 and data['message']['content']=='visible answer'
    assert data['message']['reasoning_content']=='[redacted]'


def test_list_and_api_only_expose_runs(tmp_path):
    write_run(tmp_path)
    (tmp_path/'.env').write_text('API_KEY=secret')
    assert len(list_runs(tmp_path))==1
    code,kind,body=response_for('/api/runs/local-live-example',tmp_path,tmp_path)
    assert code==200 and json.loads(body)['task']['objective']=='A task'
    assert response_for('/.env',tmp_path,tmp_path) is None
    assert response_for('/api/runs/../../.env',tmp_path,tmp_path)[0]==400


def test_missing_trace_remains_explicit(tmp_path):
    run=write_run(tmp_path)
    (run/'selection-trace.jsonl').unlink()
    data=load_run(tmp_path,run.name)
    assert data['entries']==[] and data['summary']['cost_usd'] is None
    assert data['summary']['has_trace'] is False


def test_fresh_readbacks_are_associated_with_actions(tmp_path):
    run=write_run(tmp_path)
    (run/'selection-trace.jsonl').unlink()
    events=[]
    for seconds,expired in [(0,False),(30,False),(60,True)]:
        events.extend([
            {'timestamp':f'{seconds:03}-a','component':'execute','kind':'tool_result','result':{},'proposal':{'capability_id':'workflow.await'}},
            {'timestamp':f'{seconds:03}-b','component':'observe','kind':'observation','observations':{'measurement':{'id':f'obs-{seconds}','native':{'clock':{'time_s':seconds},'observations':{'timer.state':{'value':{'set_s':60,'expired':expired}}}}}}},
        ])
    (run/'trace.jsonl').write_text('\n'.join(map(json.dumps,events)),encoding='utf-8')
    rows=load_run(tmp_path,run.name)['entries']
    assert [r['snapshot']['native']['clock']['time_s'] for r in rows]==[0,30,60]
    assert rows[-1]['snapshot']['native']['observations']['timer.state']['value']['expired'] is True
    assert rows[-1]['snapshot_source']['line']==6


def test_larger_trace_uses_separate_bounded_limit(tmp_path,monkeypatch):
    import app.result_viewer as viewer
    run=write_run(tmp_path)
    monkeypatch.setattr(viewer,'MAX_FILE_BYTES',1024)
    monkeypatch.setattr(viewer,'MAX_TRACE_BYTES',2048)
    event={'component':'report','kind':'agent_report','report':{'text':'x'*1200}}
    trace=run/'selection-trace.jsonl'
    trace.write_text(json.dumps(event),encoding='utf-8')
    assert len(viewer.load_run(tmp_path,run.name)['entries'])==1
    trace.write_text(json.dumps(event)*2,encoding='utf-8')
    with pytest.raises(ValueError,match='not loaded'):
        viewer.load_run(tmp_path,run.name)
