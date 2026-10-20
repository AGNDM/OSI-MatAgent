"""Choose a lab action, independently of specialist routing and execution authority."""
import asyncio
from copy import deepcopy
import json
import math
import time
from typing import Protocol

from app.local_decision import LocalOpenJevBackend
from app.models import ActionCandidate, OperationDecision

CANDIDATE_INSTRUCTION = (
    ' This run uses a separate operational decision maker. Override the instruction to select one intent: '
    'return hypotheses and 1-4 distinct candidates matching OperationOptions, not a final intent. '
    'Each candidate has candidate_id, intent, rationale and evidence_ids. Propose meaningful alternative '
    'next operations with exact schema-valid arguments and honest evidence/conditions; do not rank or '
    'mark a preferred candidate. At the initial state without inspection, propose inspection only. After a correction, prioritize a fresh measurement before another mutation. '
    'Use only current public evidence; citations must be observation IDs, never document IDs. '
    'The decision maker, not the diagnostic expert, chooses which candidate proceeds to procedure.'
)

DECISION_PROMPT = (
    'Choose the next laboratory operation from the supplied diagnostic candidates, using public '
    'observations and target policy. Hypotheses and candidate rationales are proposals, not verified '
    'facts. A current thermal receipt belongs to the current aliquot; do not reprocess it because '
    'a different historical aliquot failed. A missing new measurement does not mean inspection is absent. '
    'Inspect first only if inspection is missing or stale; preserve correctly processed material and seek a '
    'discriminating measurement when needed. Do not invent capabilities, arguments or candidate IDs. '
    'Return choice as one candidate_id and a complete normalized probability distribution over exactly '
    'the supplied candidate IDs. Choice must be a maximum-probability candidate. Do not execute tools.'
)


class OperationBackend(Protocol):
    backend_id: str
    is_llm: bool

    def choose(self, context: dict, candidates: list[dict]) -> OperationDecision: ...


def validate_decision(raw, candidates):
    decision = OperationDecision.model_validate(raw)
    ids = [ActionCandidate.model_validate(c).candidate_id for c in candidates]
    p = decision.probabilities
    if decision.choice not in ids or set(p) != set(ids):
        raise ValueError('Decision must reference exactly the supplied candidate IDs')
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in p.values()) or abs(sum(p.values())-1) > 1e-5:
        raise ValueError('Invalid operation probability distribution')
    if p[decision.choice] != max(p.values()):
        raise ValueError('Selected operation must maximize the supplied probabilities')
    return decision


def operation_payload(state):
    """No host internals, evaluator labels, or candidate reference ranking."""
    return deepcopy({k: state.get(k) for k in [
        'sample_id', 'observation_revision', 'target', 'observations', 'hypotheses',
        'capabilities', 'last_error']})


class LocalOperationBackend(LocalOpenJevBackend):
    """Same local model/transport as router, separate question, calls and records."""
    is_llm = True

    def choose(self, context, candidates):
        options = {chr(65+i): c['candidate_id'] for i, c in enumerate(candidates)}
        if not 1 <= len(options) <= 4:
            raise ValueError('Operational decision supports 1-4 candidates')
        lines = [f'[{letter}] '+json.dumps(c, sort_keys=True, allow_nan=False)
                 for letter, c in zip(options, candidates)]
        text = (DECISION_PROMPT+'\nPublic context:\n'+json.dumps(context, sort_keys=True, allow_nan=False)
                +'\nCandidate operations:\n'+'\n'.join(lines)
                +'\nAnswer with the letter of the best candidate only.')
        start = time.monotonic()
        timeout = min(self.timeout_seconds, self.deadline-start)
        self.last_call = {'backend_id': self.backend_id, 'purpose': 'operational_action_selection',
                          'context': deepcopy(context), 'candidates': deepcopy(candidates),
                          'probability_semantics': self.manifest()['probability_semantics']}
        if timeout <= 0:
            raise TimeoutError('Deadline expired before operational inference')
        async def bounded():
            return await asyncio.wait_for(self._infer_options(text, options, timeout), timeout)
        try:
            p = asyncio.run(bounded())
            return OperationDecision(backend_id=self.backend_id, choice=max(p, key=p.get),
                                     probabilities=p, latency_ms=(time.monotonic()-start)*1000)
        except Exception as exc:
            self.last_call['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            self.last_call['elapsed_ms'] = (time.monotonic()-start)*1000


class ExpertOperationBackend:
    """Adapter for an existing budgeted general expert transport; no new endpoint."""
    is_llm = True

    def __init__(self, expert):
        self.expert = expert
        self.backend_id = expert.backend_id
        self.deadline = float('inf')

    @property
    def last_call(self):
        return self.expert.last_call

    def choose(self, context, candidates):
        self.expert.deadline = self.deadline
        raw = self.expert.invoke('operation_decision', DECISION_PROMPT,
            {'context': context, 'candidates': candidates, 'backend_id': self.backend_id}, OperationDecision)
        return OperationDecision.model_validate(raw)


class ScriptedOperationBackend:
    """Public-evidence integration-test policy, NOT a model or calibrated scores."""
    backend_id = 'scripted-operation-test-double-v1'
    is_llm = False
    last_call = None

    def choose(self, context, candidates):
        obs = context['observations']
        inspection = obs.get('inspection')
        if obs.get('last_mutation_step') and not obs.get('measurement') and any(
                c['intent']['capability_id'] == 'balance.measure' for c in candidates):
            cap = 'balance.measure'
        elif not inspection:
            cap = 'workflow.inspect'
        elif 'thermal_history' in inspection:
            cap = ('thermal.prepare_reference' if inspection['thermal_history'][-1]['temperature_c'] != 600
                   else 'xrd.acquire')
        else:
            # Test fixture for the existing drying demo.
            check = inspection['balance_check']
            setpoint = inspection['native']['observations']['temperature.set_k']['value']
            cap = ('balance.measure' if obs.get('last_mutation_step') or not check['available'] else
                   'balance.clean' if check['pan'] == 'contaminated' else
                   'temperature.set' if abs(setpoint-context['target']['temperature_k']) > .5 else 'balance.measure')
        selected = next(c for c in candidates if c['intent']['capability_id'] == cap)
        return OperationDecision(backend_id=self.backend_id, choice=selected['candidate_id'],
            probabilities={c['candidate_id']: float(c is selected) for c in candidates})


def scripted_options(record, payload):
    """Candidate fixture for integration runs; never presented as model reasoning."""
    value = record.model_dump() if hasattr(record, 'model_dump') else deepcopy(record)
    inspection = payload['observations'].get('inspection')
    current = value.pop('intent')
    actions = [(current['capability_id'], current['arguments'])]
    if inspection and 'thermal_history' in inspection:
        actions = [('thermal.prepare_reference', {}), ('xrd.acquire', {'stop_deg': 60})]
    elif inspection and 'balance_check' in inspection:
        actions = [('workflow.inspect', {}), ('balance.measure', {}), ('balance.clean', {}),
                   ('temperature.set', {'temperature_k': payload['target']['temperature_k']})]
    candidates = []
    for index, (cap, args) in enumerate(actions):
        candidates.append({'candidate_id': f'candidate-{index+1}',
            'intent': {'capability_id': cap, 'arguments': args,
                'discriminating_observation': 'Compare public processing and measurement evidence.',
                'postcondition': 'Apply the operation only when justified by the public target and evidence.'},
            'rationale': 'Consider this operation against the current public observations and target policy.',
            'evidence_ids': [inspection['id']] if inspection else []})
    return {'hypotheses': value['hypotheses'], 'candidates': candidates}
