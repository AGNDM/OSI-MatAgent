import asyncio
import hashlib
import json

from bench.instrument_mcqa_jev import decision_input, run, save


def test_original_choices_and_structured_steps_are_preserved():
    view = {'question': 'Choose a plan', 'options': [
        {'option_id': 'B', 'text': 'Read\nthen wait', 'steps': [{'kind': 'wait', 'dt_s': 10}]},
        {'option_id': 'A', 'text': 'Stop', 'steps': [{'kind': 'escalate'}]},
    ]}
    prompt = json.dumps(view)
    text, mapping = decision_input(prompt)
    assert prompt in text
    assert list(mapping) == ['B', 'A']
    criteria = {line[1]: json.loads(line[4:]) for line in text.splitlines()
                if line.startswith(('[A] ', '[B] '))}
    assert criteria['B'] == view['options'][0]
    assert criteria['A'] == view['options'][1]


def test_inference_uses_native_tie_choice_and_keeps_errors(tmp_path, monkeypatch):
    from bench import instrument_mcqa_jev as module
    prompt = json.dumps({'question': 'Select', 'options': [
        {'option_id': 'A', 'text': 'One'}, {'option_id': 'B', 'text': 'Two'}]})
    path = tmp_path / 'public-prompts.jsonl'
    path.write_text(json.dumps({'task_id': 'secret_task_id', 'prompt': prompt}) + '\n')
    save(tmp_path / 'input-manifest.json', {'prompts_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})

    class Backend:
        def __init__(self, *args, **kwargs): pass
        async def _infer_options(self, text, options, timeout):
            assert 'secret_task_id' not in text
            assert 'correct_option_ids' not in text
            self.last_call['response'] = {'answers': {'next_action': {'choice': 'B'}}}
            return {'A': .5, 'B': .5}
    monkeypatch.setattr(module, 'OpenRouterJevBackend', Backend)
    asyncio.run(run(tmp_path, workers=1))
    prediction = json.loads((tmp_path / 'predictions.jsonl').read_text())
    assert prediction['selected_option_id'] == 'B'
    assert prediction['status'] == 'ok'
    # No reference key is created or needed anywhere in this inference run.
    assert not (tmp_path / 'answers.jsonl').exists()
