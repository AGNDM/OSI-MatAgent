"""Durable, process-safe reservations for this user-authorized API budget.

Missing usage, transport failures and crashed requests retain the entire input
upper-bound reservation. No estimate is silently treated as reported usage.
"""
from contextlib import contextmanager
from app.file_lock import exclusive_file_lock
import json
import os
from pathlib import Path
from uuid import uuid4

from app.expert_config import CompatibleExpert

HARD_LIMIT = 5_000_000
RECORD_ONLY_MODELS={'deepseek/deepseek-v4.1-flash','typesafe/jev-1.14'}
DEFAULT_LEDGER = Path(__file__).resolve().parents[1] / 'results/device-expansion-budget-v1/ledger.jsonl'


def valid_usage(usage):
    return (isinstance(usage,dict) and all(type(usage.get(k)) is int and usage[k]>=0
        for k in ['prompt_tokens','completion_tokens','total_tokens'])
        and usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens'])


class TokenBudgetExceeded(RuntimeError):
    code='token_budget_exceeded'


class MeteringUnavailable(TokenBudgetExceeded):
    code='metering_unavailable'


class TokenLedger:
    def __init__(self,path=DEFAULT_LEDGER,limit=HARD_LIMIT,*,window_id=None,window_limit=None):
        if type(limit) is not int or not 0 < limit <= HARD_LIMIT:
            raise ValueError('Budget must be between 1 and 5,000,000 tokens')
        if (window_id is None) != (window_limit is None):
            raise ValueError('Window identity and limit must be supplied together')
        if window_id is not None and (not isinstance(window_id,str) or not window_id or
                type(window_limit) is not int or not 0 < window_limit <= limit):
            raise ValueError('Invalid incremental budget window')
        self.record_only_openrouter=False
        self.window_id,self.window_limit=window_id,window_limit
        self.path,self.limit=Path(path),limit
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self._locked() as entries:
            if not entries:
                self._append({'kind':'limit','limit':limit})
            elif entries[0] != {'kind':'limit','limit':limit}:
                raise ValueError('Existing ledger limit must not be reset or increased')
            if window_id is not None:
                if not entries:
                    entries=[{'kind':'limit','limit':limit}]
                existing=next((e for e in entries if e['kind']=='window' and e['id']==window_id),None)
                if existing and existing['limit'] != window_limit:
                    raise ValueError('Existing window limit must not be changed')
                if not existing:
                    state=self._state(entries)
                    if state['inflight_or_crashed']:
                        raise ValueError('Cannot start a budget window with outstanding reservations')
                    self._append({'kind':'window','id':window_id,'limit':window_limit,
                                  'start_charged_tokens':state['charged_tokens'],
                                  'start_unmetered_requests':state['unmetered_requests']})

    def enable_openrouter_record_only(self):
        with self._locked() as entries:
            if not any(e['kind']=='policy' and e.get('id')=='openrouter-record-only-v1' for e in entries):
                self._append({'kind':'policy','id':'openrouter-record-only-v1','provider':'openrouter',
                    'models':sorted(RECORD_ONLY_MODELS),'effective_after_reservations':self._state(entries)['requests_reserved'],
                    'reason':'User removed OpenRouter quota concerns; retain accounting and finite experiment plans'})
            self.record_only_openrouter=True

    @contextmanager
    def _locked(self):
        with exclusive_file_lock(self.path.with_suffix('.lock')):
            entries=[json.loads(x) for x in self.path.read_text().splitlines()] if self.path.exists() else []
            yield entries

    def _append(self,value):
        with self.path.open('a') as stream:
            stream.write(json.dumps(value,allow_nan=False)+'\n')
            stream.flush();os.fsync(stream.fileno())

    @staticmethod
    def _state(entries):
        reserves={e['id']:e for e in entries if e['kind']=='reserve'}
        settled={e['id']:e for e in entries if e['kind']=='settle'}
        reported=sum(e.get('reported_tokens',0) for e in settled.values())
        charged=sum(settled[k]['charged_tokens'] if k in settled else r['tokens'] for k,r in reserves.items())
        budget_charged=sum(settled[k]['charged_tokens'] if k in settled else r['tokens']
            for k,r in reserves.items() if not r.get('record_only',False))
        return {'limit':entries[0]['limit'],'reported_tokens':reported,'charged_tokens':charged,
            'budget_charged_tokens':budget_charged,'remaining_tokens':max(0,entries[0]['limit']-budget_charged),'requests_reserved':len(reserves),
            'requests_settled':len(settled),'unmetered_requests':sum(not s['usage_complete'] for s in settled.values()),
            'inflight_or_crashed':len(reserves)-len(settled),
            'accounting_violation':any(s.get('accounting_violation',False) for s in settled.values())}

    def _window_state(self, entries, state):
        record=next(e for e in entries if e['kind']=='window' and e['id']==self.window_id)
        charged=max(0,state['charged_tokens']-record['start_charged_tokens'])
        baseline=record.get('start_unmetered_requests')
        if baseline is None:
            baseline=self._state(entries[:entries.index(record)])['unmetered_requests']
        prefix=self._state(entries[:entries.index(record)])
        budget_delta=max(0,state['budget_charged_tokens']-prefix['budget_charged_tokens'])
        return {**record,'charged_tokens':charged,'budget_charged_tokens':budget_delta,'remaining_tokens':max(0,record['limit']-budget_delta),
                'unknown_usage_detected':state['unmetered_requests']>baseline}

    def snapshot(self):
        with self._locked() as entries:
            state=self._state(entries)
            if self.window_id is not None:
                state['window']=self._window_state(entries,state)
            state['openrouter_budget_policy']='record-only' if self.record_only_openrouter else 'bounded'
            return state

    def claim_episode(self, phase):
        """Persist finite phase attempts, including the explicitly renewed four-pair plan."""
        if self.window_id is None or phase not in {'acceptance','retrieval','knowledge-record-only-v4'}:
            raise ValueError('A named window and known phase are required')
        with self._locked() as entries:
            attempts=[e for e in entries if e['kind']=='episode_start' and e['window_id']==self.window_id]
            if ((phase=='knowledge-record-only-v4' and sum(e['phase']==phase for e in attempts)>=4) or
                    (phase!='knowledge-record-only-v4' and (len([e for e in attempts if e['phase']!='knowledge-record-only-v4'])>=8 or sum(e['phase']==phase for e in attempts)>=4))):
                raise TokenBudgetExceeded('Live episode allowance exhausted; attempt not started')
            key=str(uuid4())
            self._append({'kind':'episode_start','window_id':self.window_id,'phase':phase,'id':key})
            return key

    def reserve(self,body,*,provider=None):
        # Byte-level tokenization cannot exceed UTF-8 byte count for these plain
        # text messages. Add a generous 8192 framing allowance and output ceiling.
        messages=body.get('messages',[])
        if not messages or any(not isinstance(m.get('content'),str) for m in messages):
            raise ValueError('Only plain text requests have a defined input bound')
        thinking_off = (body.get('enable_thinking') is False or body.get('reasoning') == {'enabled': False})
        conflicting = (body.get('enable_thinking') is True or
                       ('reasoning' in body and body['reasoning'] != {'enabled': False}))
        if not thinking_off or conflicting or body.get('stream') is not False:
            raise ValueError('Budgeted debug requires bounded non-thinking, non-streaming calls')
        maximum=body.get('max_tokens')
        if type(maximum) is not int or not 1 <= maximum <= 8192:
            raise ValueError('A bounded output token ceiling is required')
        bound=len(json.dumps(body,ensure_ascii=False).encode('utf-8'))+8192+maximum
        return self._reserve_bound(body.get('model'), bound,provider=provider)

    def reserve_decision(self, body,*,provider=None):
        # One closed choice, plain text only. Reserve the entire documented model
        # context plus framing/input bytes since this API has no max_tokens knob.
        if body.get('model') != 'typesafe/jev-1.14' or not isinstance(body.get('state'), str):
            raise ValueError('Only the authorized text Jev decision model is supported')
        questions=body.get('questions', {})
        if len(questions) != 1:
            raise ValueError('Only one bounded decision question is allowed')
        question=next(iter(questions.values()))
        criteria=question.get('criteria', {})
        if (question.get('type') != 'choice' or not isinstance(criteria,dict) or not 1 <= len(criteria) <= 4
                or any(not isinstance(k,str) or not 1 <= len(k) <= 64 or not isinstance(v,str) for k,v in criteria.items())):
            raise ValueError('Decision requires 1-4 text choices')
        size=len(json.dumps(body,ensure_ascii=False,allow_nan=False).encode('utf-8'))
        if size > 24000:
            raise ValueError('Decision input exceeds the conservative debug limit')
        return self._reserve_bound(body['model'], 262144+8192+size,provider=provider)

    def _reserve_bound(self, model, bound,*,provider=None):
        with self._locked() as entries:
            state=self._state(entries)
            record_only=self.record_only_openrouter and provider=='openrouter' and model in RECORD_ONLY_MODELS
            if self.window_id is not None and self._window_state(entries,state)['unknown_usage_detected']:
                raise MeteringUnavailable('New unknown usage prevents accurate metering; request not sent')
            if state['accounting_violation'] or (not record_only and (bound>state['remaining_tokens'] or
                    (self.window_id is not None and bound>self._window_state(entries,state)['remaining_tokens']))):
                raise TokenBudgetExceeded('Insufficient reserved token budget; request not sent')
            key=str(uuid4())
            self._append({'kind':'reserve','id':key,'tokens':bound,'model':model,'provider':provider,'record_only':record_only})
        return key

    def settle(self,key,usage):
        with self._locked() as entries:
            reservation=next(e for e in entries if e['kind']=='reserve' and e['id']==key)
            if any(e['kind']=='settle' and e['id']==key for e in entries):
                raise ValueError('Reservation already settled')
            complete=valid_usage(usage)
            reported=usage['total_tokens'] if complete else 0
            violation=reported>reservation['tokens'] and not reservation.get('record_only',False)
            self._append({'kind':'settle','id':key,'reported_tokens':reported,'usage_complete':complete,
                'charged_tokens':reported if complete else reservation['tokens'],'accounting_violation':violation})
            if violation:
                raise TokenBudgetExceeded('Provider usage exceeded input/output reservation; all further requests blocked')


class BudgetedExpert(CompatibleExpert):
    def __init__(self,config,ledger):
        super().__init__(config)
        if (config.provider, config.model) not in {('siliconflow', 'deepseek-ai/DeepSeek-V4.1-Flash'),
                ('openrouter', 'deepseek/deepseek-v4.1-flash')}:
            raise ValueError('Only the explicitly authorized DeepSeek expert is allowed')
        self.ledger=ledger
        if config.provider=='openrouter':ledger.enable_openrouter_record_only()

    async def _request(self,body,timeout):
        self.last_call['request_sent']=False
        key=self.ledger.reserve(body,provider=self.config.provider)
        self.last_call['request_sent']=True
        self.last_call['budget_reservation_id']=key
        usage=None
        try:
            response=await super()._request(body,timeout)
            usage=response.get('usage')
            return response
        finally:
            self.ledger.settle(key,usage)
