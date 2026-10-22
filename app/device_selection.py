"""Bounded expert discovery -> Jev device choice -> one native instance binding."""
from copy import deepcopy
import json
import time
from uuid import uuid4
from typing import Literal

from pydantic import Field,model_validator
from app.models import Record,ActionCandidate,Intent
from app.operation_decision import validate_decision
from app import device_catalog


class DeviceCandidate(Record):
    candidate_id:str=Field(min_length=1,max_length=64)
    device_id:str=Field(min_length=1,max_length=100)
    rationale:str=Field(min_length=1,max_length=500)


class DiscoveryRequest(Record):
    action:Literal['search','describe','propose','stop']
    query:str=Field(default='',max_length=200)
    reason:str=Field(default='',max_length=500)
    device_ids:list[str]=Field(default_factory=list,max_length=4)
    candidates:list[DeviceCandidate]=Field(default_factory=list,max_length=4)

    @model_validator(mode='after')
    def shape(self):
        if self.action=='stop' and (not self.reason or self.device_ids or self.candidates):raise ValueError('Stop needs an honest reason only')
        if self.action=='describe' and (not self.device_ids or self.candidates):raise ValueError('Describe needs device IDs only')
        if self.action=='propose' and (not self.candidates or self.device_ids):raise ValueError('Propose needs device candidates only')
        if self.action=='search' and (self.device_ids or self.candidates):raise ValueError('Search needs a query only')
        ids=[c.candidate_id for c in self.candidates];devices=[c.device_id for c in self.candidates]
        if len(set(ids))!=len(ids) or len(set(devices))!=len(devices):raise ValueError('Duplicate device candidates')
        return self


PROMPT=('Select a native instrument for the supplied objective and fixed public-state predicates. '
    'The complete registered public device directory is supplied; no task-specific shortlist or solution sequence is supplied. '
    'Use search or describe to inspect public contracts. Describe any device before proposing it. '
    'Propose 1-4 distinct devices and short evidence-based rationales; a separate Jev model chooses. '
    'Never change goal predicates, invent a module/path, or execute tools. Prefer actual capability and state-unit compatibility. '
    'Only one instrument will be bound; cross-device sample transfer is unavailable. '
    'If no native device can satisfy the declared public goal, use stop with an honest reason. Return exactly DiscoveryRequest JSON. Stop searching by proposing candidates within the query budget.')


class UnsupportedObjective(RuntimeError):pass


def select_device(task,expert,chooser,trace,limits,*,catalog=None,public_task=None,prompt=None,describe_device=None):
    catalog=catalog or device_catalog.load_catalog()
    for predicate in task.predicates+task.constraints:
        if not any(d['state_slots'].get(predicate.state_id,{}).get('unit')==predicate.unit for d in catalog['devices']):
            raise UnsupportedObjective('No registered public native state supports '+predicate.state_id+' with unit '+predicate.unit)
    start=time.monotonic();deadline=start+limits.episode_seconds
    calls=0;described={};history=[]
    public_task=task.model_dump(exclude={'device'}) if public_task is None else deepcopy(public_task)
    prompt=prompt or PROMPT
    describe_device=describe_device or device_catalog.describe
    for turn in range(6):
        payload={'task':public_task,'device_directory':device_catalog.index(catalog),
                 'discovered_contracts':deepcopy(described),'query_history':history,
                 'remaining_discovery_rounds':6-turn,'remaining_descriptions':4-len(described)}
        request=None
        for attempt in range(1+limits.format_repairs):
            if calls>=limits.llm_calls or time.monotonic()>=deadline:raise RuntimeError('device_discovery_budget')
            expert.deadline=deadline;calls+=int(expert.is_llm)
            try:
                raw=expert.invoke('device_discovery',prompt,payload,DiscoveryRequest)
                request=DiscoveryRequest.model_validate(raw)
                if request.action=='describe':
                    if len(set(described)|set(request.device_ids))>4:raise ValueError('At most four detailed device contracts')
                    for device in request.device_ids:device_catalog.describe(catalog,device)
                if request.action=='propose':
                    if any(c.device_id not in described for c in request.candidates):raise ValueError('Describe candidate contracts before proposing')
                trace.emit('device_discovery','role_output',input=payload,output=request.model_dump(),
                           llm_metadata=getattr(expert,'last_call',None))
                break
            except Exception as exc:
                trace.emit('device_discovery','role_error',input=payload,error=type(exc).__name__+': '+str(exc)[:1000],
                           llm_metadata=getattr(expert,'last_call',None))
                if not isinstance(exc,ValueError) or attempt>=limits.format_repairs:raise
                payload=dict(payload,format_repair={'validation_error':str(exc)[:1000]})
        if request.action=='stop':raise UnsupportedObjective(request.reason)
        if request.action=='search':history.append(device_catalog.search(catalog,request.query,limit=30))
        elif request.action=='describe':
            described.update({d:describe_device(catalog,d) for d in request.device_ids})
        else:
            candidates=[ActionCandidate(candidate_id=c.candidate_id,
                intent=Intent(capability_id='catalog.bind',arguments={'device_id':c.device_id},
                              discriminating_observation='Public native contract compatibility',postcondition='Bind exactly one fresh runtime'),
                rationale=c.rationale,evidence_ids=[]).model_dump() for c in request.candidates]
            if calls>=limits.llm_calls or time.monotonic()>=deadline:raise RuntimeError('device_selection_budget')
            chooser.deadline=deadline;calls+=int(chooser.is_llm)
            try:
                decision=validate_decision(chooser.choose({'task':public_task,'purpose':'device_binding',
                    'described_devices':[{k:described[c.device_id][k] for k in ['id','name','domain','lifecycle']} for c in request.candidates]},candidates),candidates)
                if decision.backend_id!=chooser.backend_id:raise ValueError('Device chooser identity mismatch')
            except Exception as exc:
                trace.emit('device_selection','selection_error',error=str(exc)[:1000],backend_metadata=getattr(chooser,'last_call',None));raise
            selected=next(c for c in request.candidates if c.candidate_id==decision.choice)
            trace.emit('device_selection','device_selected',candidates=candidates,decision=decision.model_dump(),
                       backend_metadata=getattr(chooser,'last_call',None),routing_policy='argmax-v1')
            return task.model_copy(update={'device':selected.device_id}),{
                'selection_id':str(uuid4()),'selected_device':selected.device_id,'model_calls':calls,
                'elapsed_seconds':time.monotonic()-start,'catalog_count':len(catalog['devices']),
                'catalog_version':catalog['version'],'described_device_ids':list(described),
                'reference_scripts_used':False,'decision':decision.model_dump()}
    raise RuntimeError('device_discovery_round_limit')
