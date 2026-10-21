"""Task-conditioned knowledge only; no experiment macros or material transitions."""
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import re

from app.single_instrument import InstrumentProfile

CORPUS=Path(__file__).resolve().parents[1]/'corpus/instrument-knowledge-v1.json'
PROTOCOL={'id':'protocol-public-evidence','source':'app/models.py; app/single_instrument.py',
    'locator':'Public observation and execution contracts','ownership':'Authored protocol',
    'text':'Use public tool schemas, units and current sample identity. A command receipt is not process completion. '
           'Knowledge citations support reasoning; only fresh native observations establish goal satisfaction.'}


def tokens(text):return set(re.findall(r'[a-z0-9]+',text.lower()))


def retrieve_materials(role,payload,enabled=True):
    target=payload.get('target',{});obs=payload.get('observations',{})
    measurement=obs.get('measurement') or obs.get('inspection') or {}
    native=measurement.get('native',{}).get('observations',{})
    public={'objective':target.get('objective'),'predicates':target.get('predicates'),
        'constraints':target.get('constraints'),
        'observations':{k:{f:v.get(f) for f in ['value','unit','quality']} for k,v in sorted(native.items())},
        'capability_ids':sorted(payload.get('capabilities',{})), 'last_error':payload.get('last_error')}
    query=(role+' '+json.dumps(public,sort_keys=True,allow_nan=False))[:16000]
    terms=tokens(query);data=json.loads(CORPUS.read_text());doc_terms={d['id']:tokens(d['title']+' '+d['text']) for d in data}
    def score(d):
        return sum(1+math.log((len(data)+1)/(1+sum(t in ts for ts in doc_terms.values())))
                   for t in terms & doc_terms[d['id']])
    ranked=sorted(data,key=lambda d:(-score(d),d['id']))
    hits=[deepcopy(PROTOCOL)]+([deepcopy(d) for d in ranked[:3] if score(d)>0] if enabled else [])
    return {'query':query,'hits':hits,'enabled':enabled,'backend':'public-context-lexical-v1','top_k':3,
            'corpus_sha256':sha256(CORPUS.read_bytes()).hexdigest(),
            'scope':'Instrument knowledge grounding; no scientific outcome or task-specific action sequence'}


class KnowledgeInstrumentProfile(InstrumentProfile):
    prompts={**InstrumentProfile.prompts,
        'diagnostic':InstrumentProfile.prompts['diagnostic']+
            ' Use relevant retrieved knowledge as evidence for reasoning, not as a plan. '
            'Cite retrieved passage IDs separately in material_evidence_ids; observation citations stay in supporting_ids/contradicting_ids.'}
    candidate_prompt=(prompts['diagnostic']+' Propose 1-4 distinct schema-valid operation candidates. '
        'Do not rank candidates. Use material_evidence_ids for relevant supplied passages. '
        'At the initial state without inspection propose only workflow.inspect. Keep rationale and intent fields brief.')

    @staticmethod
    def projection(role,state,request_id):
        from app.public_contracts import compact_capabilities
        payload=InstrumentProfile.projection(role,state,request_id)
        if 'capabilities' in payload:
            payload=dict(payload)
            payload['capabilities'],payload['public_contract_definitions']=compact_capabilities(payload['capabilities'])
            payload['contract_encoding']='Resolve $public_meta references using public_contract_definitions; all parameter schemas remain inline.'
        return payload

    def __init__(self,enabled=True):self.enabled=enabled

    def retrieve_context(self,role,payload):return retrieve_materials(role,payload,self.enabled)
