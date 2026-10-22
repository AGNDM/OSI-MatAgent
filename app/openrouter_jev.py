"""Typed hosted Jev decisions, reusing the public routing/operation contracts."""
import json
import math
import os

import httpx

from app.local_decision import LocalOpenJevBackend
from app.operation_decision import LocalOperationBackend

MODEL = 'typesafe/jev-1.14'
AUTHORIZED_MODELS = (MODEL, 'openai/gpt-6-luna-decisions')
ENDPOINT = 'https://openrouter.ai/api/alpha/decisions'


def normalize_usage(usage):
    if not isinstance(usage,dict):
        return None
    a,b=usage.get('input_tokens'),usage.get('output_tokens')
    if type(a) is not int or type(b) is not int or min(a,b)<0:
        return None
    return {'prompt_tokens':a,'completion_tokens':b,'total_tokens':a+b,
            **({'cost':usage['cost']} if 'cost' in usage else {})}


class OpenRouterJevBackend(LocalOpenJevBackend):
    """No chat generation, fabricated probabilities, or silent rules fallback."""
    backend_id = 'openrouter-decisions/'+MODEL
    is_llm = True
    allow_rules_fallback = False

    def __init__(self,ledger,timeout_seconds=45,context_byte_limit=24000,model=MODEL,force_singleton=False):
        if model not in AUTHORIZED_MODELS:
            raise ValueError("Unsupported hosted decision model")
        self.force_singleton=force_singleton
        self.model=model
        self.backend_id="openrouter-decisions/"+model
        self.context_byte_limit=context_byte_limit
        self.ledger=ledger
        ledger.enable_openrouter_record_only()
        self.timeout_seconds=timeout_seconds
        self.deadline=float('inf')
        self.last_call=None
        self._key=os.environ.get('OPENROUTER_API_KEY')
        if not self._key:
            raise ValueError('OPENROUTER_API_KEY is missing')

    def manifest(self):
        return {'backend_id':self.backend_id,'model':self.model,'endpoint':ENDPOINT,
            'probability_semantics':'Native typed choice probabilities returned by the selected hosted decision model; not calibrated on these instrument tasks',
            'transport_retries':0,'rules_fallback':False,'context_byte_limit':self.context_byte_limit,
            'reservation_policy':'Full 262144-token model context plus request bytes and framing; shared durable ledger'}

    def redact(self,value):
        if isinstance(value,str):return value.replace(self._key,'[REDACTED]')
        if isinstance(value,dict):return {k:self.redact(v) for k,v in value.items()}
        if isinstance(value,list):return [self.redact(v) for v in value]
        return value

    async def _infer_options(self,text,options,timeout):
        # A singleton is a forced transition, not a model decision. OpenAI's
        # Decisions endpoint rejects fewer than two choices. Apply this same
        # no-inference behavior to both hosted models for comparable control flow.
        if len(options)==1 and (self.force_singleton or self.model=="openai/gpt-6-luna-decisions"):
            self.last_call.update(request_sent=False, decision_source='single_legal_option',
                                  options=options, usage=None)
            return {next(iter(options.values())): 1.0}
        lines={line[1]:line[4:] for line in text.splitlines() if len(line)>4 and line[0]=='[' and line[2:4]=='] '}
        criteria={letter:lines.get(letter,meaning) for letter,meaning in options.items()}
        body={'model':self.model,'state':text,'questions':{'next_action':{'type':'choice',
            'instructions':'Choose the best next option using the instructions, current public state and option descriptions supplied in state.',
            'criteria':criteria}}}
        self.last_call.update(request=body,request_sent=False)
        key=self.ledger.reserve_decision(body,provider="openrouter",context_byte_limit=self.context_byte_limit)
        self.last_call.update(request_sent=True,budget_reservation_id=key)
        usage=None
        try:
            async with httpx.AsyncClient(timeout=timeout,trust_env=False) as client:
                response=await client.post(ENDPOINT,json=body,headers={'Authorization':'Bearer '+self._key})
                self.last_call['http_status']=response.status_code
                if response.is_error:
                    self.last_call['response']=self.redact(response.text)
                    raise RuntimeError('Jev HTTP '+str(response.status_code)+': '+self.redact(response.text)[:1000])
                raw=self.redact(response.json())
            usage=normalize_usage(raw.get('usage'))
            self.last_call.update(response=raw,usage=usage,options=options)
            actual=raw.get('model','')
            if actual != self.model and not actual.startswith(self.model+'-'):
                raise ValueError('Jev returned an unexpected model identity')
            answer=raw['answers']['next_action']
            p=answer['probabilities']
            if answer.get('type')!='choice' or set(p)!=set(options):
                raise ValueError('Jev decision did not cover the supplied choices')
            if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in p.values()):
                raise ValueError('Invalid native Jev probability distribution')
            choice=answer['choice']
            if choice not in p or p[choice]!=max(p.values()):
                raise ValueError('Jev choice disagrees with its probabilities')
            self.last_call['native_confidence']=answer.get('confidence')
            return {options[k]:v for k,v in p.items()}
        finally:
            self.ledger.settle(key,usage)


class OpenRouterJevOperationBackend(OpenRouterJevBackend):
    choose = LocalOperationBackend.choose
