import json
from pathlib import Path

import pytest

from app.expert_config import load_plan
from app.token_budget import BudgetedExpert, TokenLedger
from bench.instrument_open_question import carry_forward, parse_answer, save


@pytest.mark.parametrize('config_name', [
    'free-low', 'free-medium', 'free-max', 'low', 'max',
])
def test_effort_preserved_through_transport_and_metering(tmp_path, monkeypatch, config_name):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'offline-test-key')
    config = load_plan(Path('configs/openrouter-minimax-m3.1-flash-' + config_name + '.json')).reference
    effort = config.reasoning_effort
    assert config.model == ('minimax/minimax-m3.1-flash:free' if config_name.startswith('free-') else 'minimax/minimax-m3.1-flash')
    ledger = TokenLedger(tmp_path / 'ledger.jsonl', limit=100)
    expert = BudgetedExpert(config, ledger)
    body = expert.prepare_request({'model': config.model, 'messages': [{'role': 'user', 'content': 'evidence'}],
                                  'max_tokens': config.max_tokens, 'stream': False})
    assert body['reasoning'] == {'effort': effort, 'exclude': True}
    assert body['max_tokens'] == 524288
    key = ledger.reserve(body, provider='openrouter')
    ledger.settle(key, {'prompt_tokens': 500, 'completion_tokens': 1000, 'total_tokens': 1500})
    assert ledger.snapshot()['budget_charged_tokens'] == 0
    with pytest.raises(ValueError):
        ledger.reserve({**body, 'reasoning': {'effort': effort, 'exclude': True, 'enabled': False}}, provider='openrouter')


def test_separate_justification_is_preserved_and_duplicate_keys_rejected():
    response = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({
        'candidates': [{'answer': str(i), 'justification': 'Evidence ' + str(i)} for i in range(4)]})}}]}
    result = parse_answer(response, True)
    assert len(result) == 4
    assert result[2] == {'answer': '2\n\nJustification: Evidence 2'}
    response['choices'][0]['message']['content'] = '{"answer":"first","answer":"second"}'
    with pytest.raises(ValueError, match='Duplicate JSON key'):
        parse_answer(response)


def test_controlled_continuation_preserves_failures_and_defers_inflight(tmp_path):
    old, new = tmp_path / 'old', tmp_path / 'new'
    for folder in (old, new):
        folder.mkdir()
        (folder / 'public-questions.jsonl').write_text('{"task_id":"a"}\n')
    save(old / 'protocol.json', {'expert': {'model': 'minimax'}})
    save(old / 'provider-stop.json', {'reason': 'user_requested_reasoning_effort_change'})
    save(old / 'cases/a/pair.json', {'task_id': 'a', 'direct': {'status': 'error'},
         'generated': {'status': 'ok'}, 'selected': {'status': 'error'}})
    save(old / 'cases/b/pair.json', {'task_id': 'b', 'direct': {'status': 'ok'},
         'generated': {'status': 'not_run_provider_stop'}})
    save(old / 'cases/c/generated-started.json', {'at': 'earlier'})
    first = carry_forward(old, new)
    assert first['carried_ids'] == ['a']
    assert first['restarted_after_control_stop'] == ['b']
    assert first['deferred_inflight_ids'] == ['c']
    assert json.loads((new / 'cases/a/pair.json').read_text())['direct']['status'] == 'error'
    save(old / 'cases/c/pair.json', {'task_id': 'c', 'direct': {'status': 'ok'}, 'generated': {'status': 'ok'}})
    save(old / 'finished.json', {'done': True})
    final = carry_forward(old, new)
    assert final['carried_ids'] == ['a', 'c']
    assert final['deferred_inflight_ids'] == []


def test_continuation_refuses_uncontrolled_active_source(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'target'
    source.mkdir(); target.mkdir()
    with pytest.raises(ValueError, match='explicit effort change'):
        carry_forward(source, target)
