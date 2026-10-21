"""Durable, process-safe reservations for this user-authorized API budget.

Missing usage, transport failures and crashed requests retain the entire input
upper-bound reservation. No estimate is silently treated as reported usage.
"""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
from uuid import uuid4

from app.expert_config import CompatibleExpert

HARD_LIMIT = 5_000_000
DEFAULT_LEDGER = Path(__file__).resolve().parents[1] / 'results/device-expansion-budget-v1/ledger.jsonl'


def valid_usage(usage):
    return (isinstance(usage,dict) and all(type(usage.get(k)) is int and usage[k]>=0
        for k in ['prompt_tokens','completion_tokens','total_tokens'])
        and usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens'])


class TokenBudgetExceeded(RuntimeError):
    pass


class TokenLedger:
    def __init__(self,path=DEFAULT_LEDGER,limit=HARD_LIMIT):
        if type(limit) is not int or not 0 < limit <= HARD_LIMIT:
            raise ValueError('Budget must be between 1 and 5,000,000 tokens')
        self.path,self.limit=Path(path),limit
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self._locked() as entries:
            if not entries:
                self._append({'kind':'limit','limit':limit})
            elif entries[0] != {'kind':'limit','limit':limit}:
                raise ValueError('Existing ledger limit must not be reset or increased')

    @contextmanager
    def _locked(self):
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            try:
                entries=[json.loads(x) for x in self.path.read_text().splitlines()] if self.path.exists() else []
                yield entries
            finally:
                fcntl.flock(lock,fcntl.LOCK_UN)

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
        return {'limit':entries[0]['limit'],'reported_tokens':reported,'charged_tokens':charged,
            'remaining_tokens':max(0,entries[0]['limit']-charged),'requests_reserved':len(reserves),
            'requests_settled':len(settled),'unmetered_requests':sum(not s['usage_complete'] for s in settled.values()),
            'inflight_or_crashed':len(reserves)-len(settled),
            'accounting_violation':any(s.get('accounting_violation',False) for s in settled.values())}

    def snapshot(self):
        with self._locked() as entries:
            return self._state(entries)

    def reserve(self,body):
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
        return self._reserve_bound(body.get('model'), bound)

    def reserve_decision(self, body):
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
        return self._reserve_bound(body['model'], 262144+8192+size)

    def _reserve_bound(self, model, bound):
        with self._locked() as entries:
            state=self._state(entries)
            if state['accounting_violation'] or bound>state['remaining_tokens']:
                raise TokenBudgetExceeded('Insufficient reserved token budget; request not sent')
            key=str(uuid4())
            self._append({'kind':'reserve','id':key,'tokens':bound,'model':model})
        return key

    def settle(self,key,usage):
        with self._locked() as entries:
            reservation=next(e for e in entries if e['kind']=='reserve' and e['id']==key)
            if any(e['kind']=='settle' and e['id']==key for e in entries):
                raise ValueError('Reservation already settled')
            complete=valid_usage(usage)
            reported=usage['total_tokens'] if complete else 0
            violation=reported>reservation['tokens']
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

    async def _request(self,body,timeout):
        self.last_call['request_sent']=False
        key=self.ledger.reserve(body)
        self.last_call['request_sent']=True
        self.last_call['budget_reservation_id']=key
        usage=None
        try:
            response=await super()._request(body,timeout)
            usage=response.get('usage')
            return response
        finally:
            self.ledger.settle(key,usage)
