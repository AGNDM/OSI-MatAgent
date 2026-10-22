import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from bench.instrument_open_question import (
    comparison_graph, parse_answer, public_evidence, stop_reason,
)


def test_options_removed_before_projection_and_inputs_not_mutated():
    source = {'question': 'Return the offered response supported by that evidence.',
              'options': [{'text': 'SECRET ORIGINAL OPTION'}], 'contract': {'all': True}}

    def project(value, interactive):
        assert interactive is True
        assert 'options' not in value
        return {key: value[key] for key in ('question', 'contract')}

    view = public_evidence(source, project)
    assert 'SECRET ORIGINAL OPTION' not in json.dumps(view)
    assert 'options' in source
    assert view['contract'] == source['contract']
    assert 'open response' in view['question']


def response(content, reason='stop'):
    return {'choices': [{'finish_reason': reason, 'message': {'content': content}}]}


def test_long_answer_not_truncated_locally_and_invalid_candidates_rejected():
    text = 'valid evidence ' * 10000
    assert parse_answer(response(json.dumps({'answer': text})))['answer'] == text
    with pytest.raises(ValueError, match='four'):
        parse_answer(response(json.dumps({'candidates': [{'answer': 'a'}] * 3})), True)
    with pytest.raises(ValueError, match='Non-complete'):
        parse_answer(response('{"answer":"partial"}', 'length'))


def test_paired_graph_keeps_generation_independent():
    calls = []

    async def direct(state):
        assert 'generated' not in state
        calls.append('direct')
        return {'direct': {'status': 'ok', 'answer': 'Direct answer'}}

    async def generate(state):
        assert 'direct' not in state
        assert set(state) == {'task_id', 'evidence'}
        calls.append('generate')
        return {'generated': {'status': 'ok', 'candidates': [{'answer': str(i)} for i in range(4)]}}

    async def select(state):
        calls.append('select')
        assert len(state['generated']['candidates']) == 4
        return {'selected': {'status': 'ok', 'answer': '2'}}

    result = asyncio.run(comparison_graph(direct, generate, select).ainvoke(
        {'task_id': 'public-id', 'evidence': {'question': 'Public question'}}))
    assert sorted(calls) == ['direct', 'generate', 'select']
    assert result['direct']['answer'] == 'Direct answer'
    assert result['selected']['answer'] == '2'


def test_provider_stop_does_not_treat_temporary_rpm_as_daily_exhaustion():
    assert stop_reason({'http_status': 429, 'response': 'daily limit exceeded'})
    assert stop_reason({'http_status': 429, 'response': 'rate limit exceeded, retry after 3 s'}) is None
    assert stop_reason({'http_status': 404, 'response': 'No endpoints'})
    assert stop_reason({'http_status': 200}) is None
