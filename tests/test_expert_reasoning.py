import json
from pathlib import Path

import pytest

from app.expert_config import load_plan
from app.token_budget import BudgetedExpert, TokenLedger


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
