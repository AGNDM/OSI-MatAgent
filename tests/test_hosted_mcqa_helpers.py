import json
from bench.experiment import Experiment, digest


def test_provider_stop_preserves_first_failure_and_stops_new_calls(tmp_path):
    exp = Experiment(tmp_path, None, tmp_path/'pace.json')
    assert not exp.stopped()
    exp.record_stop({'http_status': 404, 'response': 'Unavailable endpoint'})
    first = json.loads(exp.stop.read_text())
    assert exp.stopped() and first['reason'] == 'provider_unavailable_404'
    exp.record_stop({'http_status': 402, 'response': 'Payment required'})
    assert json.loads(exp.stop.read_text()) == first
    assert first['paid_fallback'] is False


def test_transient_errors_do_not_mark_a_provider_wide_stop(tmp_path):
    exp = Experiment(tmp_path, None, tmp_path/'pace.json')
    exp.record_stop({'http_status': 429, 'response': 'Requests too frequent'})
    assert not exp.stopped()
    exp.record_stop({'http_status': 429, 'response': 'Daily quota exceeded'})
    assert exp.stopped()
    assert digest({'a': 1, 'b': 2}) == digest({'b': 2, 'a': 1})


def test_paid_mcqa_direct_runs_without_open_question_module(tmp_path, monkeypatch):
    import asyncio
    from argparse import Namespace
    from pathlib import Path
    from bench import instrument_mcqa_minimax as direct
    from bench.experiment import save
    from bench.instrument_eval_langgraph import SharedRequestPacer
    from app.token_budget import BudgetedExpert

    monkeypatch.setenv('OPENROUTER_API_KEY', 'offline-test-key')
    prompt = {'options': [{'option_id': 'A', 'text': 'Inspect'}, {'option_id': 'B', 'text': 'Guess'}]}
    (tmp_path/'public-prompts.jsonl').write_text(json.dumps({'task_id': 'q', 'prompt': json.dumps(prompt)})+'\n')

    async def no_wait(self):
        return 0

    async def request(self, body, timeout):
        assert body['model'] == 'minimax/minimax-m3.1-flash'
        assert body['reasoning']['effort'] == 'low'
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({
            'selected_option_id': 'A', 'justification': 'Use public evidence'})}}]}

    monkeypatch.setattr(SharedRequestPacer, 'wait', no_wait)
    monkeypatch.setattr(BudgetedExpert, '_request', request)
    asyncio.run(direct.run(Namespace(config=Path('configs/openrouter-minimax-m3.1-flash-low.json'),
        output=tmp_path, workers=1, limit=1, bundle=None)))
    result=json.loads((tmp_path/'cases/q/prediction.json').read_text())
    assert result['status']=='ok' and result['selected_option_id']=='A'
