from copy import deepcopy

import pytest

from app.graph import build_graph,initial_state
from app.models import Limits,RouteDecision,VerificationReport
from app.single_instrument import InstrumentProfile
from app.trace import Trace
from tests.test_single_instrument import fake_lab,execute,LifecycleTestRoles


class RepeatVerifierRouter:
    is_decision_model=True
    backend_id='offline-repeat-verifier'
    def __init__(self):self.summaries=[]
    def decide(self,summary,allowed):
        self.summaries.append(deepcopy(summary))
        chosen='prepare_action' if summary.get('has_intent') else 'verify' if 'verify' in allowed else 'diagnose'
        return RouteDecision(backend_id=self.backend_id,choice=chosen,
            probabilities={k:float(k==chosen) for k in allowed})


class ReadbackRoles(LifecycleTestRoles):
    is_llm=True
    def invoke(self,role,prompt,payload,schema):
        if role=='diagnostic':
            return schema.model_validate({'hypotheses':[],'intent':{
                'capability_id':'workflow.readback','arguments':{},
                'discriminating_observation':'New read attempt','postcondition':'Read current state'}}).model_dump()
        return super().invoke(role,prompt,payload,schema)


def test_redundant_reads_and_diagnosis_cannot_reverify_same_semantic_evidence():
    lab,native=fake_lab();execute(lab,'workflow.inspect')
    router=RepeatVerifierRouter();trace=Trace('duplicate');limits=Limits(lab_actions=7,llm_calls=30,graph_steps=100)
    graph=build_graph(lab,trace,limits,roles=ReadbackRoles(),decision_backend=router,profile=InstrumentProfile())
    final=graph.invoke(initial_state(lab,'duplicate',limits),{'configurable':{'thread_id':'duplicate'},'recursion_limit':100})
    reports=[e for e in trace.events if e['kind']=='role_output' and e['component']=='verification']
    assert len(reports)==1 and reports[0]['output']['status']=='contradicted'
    assert native.time_s==0 and final['report']['terminal_reason']=='action_budget'
    assert len(final['verified_evidence_signatures'])==1
    # The graph clears transient verification on diagnosis and execution, while
    # persistent feedback and consumption remain available to later planning.
    diagnoses=[e for e in trace.events if e['kind']=='role_output' and e['component']=='diagnostic']
    assert diagnoses and all(e['input']['last_verification']['status']=='contradicted' for e in diagnoses)
    assert any(s['instrument_context']['current_evidence_already_verified'] for s in router.summaries)
    assert any(s['stage']=='plan_or_wait' for s in router.summaries)
    assert len({e['result']['measurement']['id'] for e in trace.events if e['kind']=='tool_result'})>1


def test_signature_changes_for_evidence_not_bookkeeping():
    lab,native=fake_lab();execute(lab,'workflow.inspect')
    state=initial_state(lab,'sig',Limits());state['observations']=lab.observe()
    before=lab.verification_signature(state)
    execute(lab,'workflow.readback');state['observations']=lab.observe()
    assert lab.verification_signature(state)==before
    # Even a native no-op invocation can increase both revision counters.
    execute(lab,'binder_fd056::act');state['observations']=lab.observe()
    assert lab.verification_signature(state)==before
    for field,value in [('value',True),('quality','estimated'),('unit','K')]:
        changed=deepcopy(state)
        changed['observations']['measurement']['native']['observations']['goal'][field]=value
        assert lab.verification_signature(changed)!=before
    for modification in ['clock','pending','constraint','stale']:
        changed=deepcopy(state)
        if modification=='clock':changed['observations']['measurement']['native']['clock']['time_s']=1
        if modification=='pending':changed['observations']['pending_operation']={'operation_id':'op','status':'running'}
        if modification=='stale':changed['observations']['measurement']['revision']=-1
        if modification=='constraint':
            lab.task.constraints=list(lab.task.predicates)
            assert lab.verification_signature(changed)!=before
            lab.task.constraints=[]
            continue
        assert lab.verification_signature(changed)!=before


@pytest.mark.parametrize('fallback',[False,True])
def test_duplicate_guard_also_applies_to_deliberation_and_rules_fallback(fallback):
    lab,native=fake_lab();execute(lab,'workflow.inspect')
    lab.routing_policy='legacy-threshold-v1'
    limits=Limits(lab_actions=5,llm_calls=20,graph_steps=100);state=initial_state(lab,'boundary',limits)
    state['observations']=lab.observe()
    sig=lab.verification_signature(state)
    state['verified_evidence_signatures']=[sig]
    state['last_verification_feedback']={'status':'contradicted','evidence_ids':['obs-1'],
        'missing_evidence':[],'evidence_signature':sig,'observation_id':'obs-1'}
    class BoundaryRouter(RepeatVerifierRouter):
        def decide(self,summary,allowed):
            if fallback:raise RuntimeError('offline outage')
            # Simulate an adapter exposing the optional deliberation branch.
            return RouteDecision(backend_id=self.backend_id,choice='deliberate')
    class WithDeliberation(InstrumentProfile):pass
    if not fallback:
        original=lab.route_context
        def route_context(state,summary,allowed):
            value,routes=original(state,summary,allowed)
            if 'deliberate' not in routes:routes.append('deliberate')
            return value,routes
        lab.route_context=route_context
    class BoundaryRoles(ReadbackRoles):
        def invoke(self,role,prompt,payload,schema):
            if role=='deliberation':
                assert 'verify' not in payload['allowed_routes']
                assert payload['summary']['instrument_context']['last_verification']['status']=='contradicted'
                return schema.model_validate({'backend_id':self.backend_id,'choice':'diagnose'}).model_dump()
            return super().invoke(role,prompt,payload,schema)
    trace=Trace('boundary')
    graph=build_graph(lab,trace,limits,roles=BoundaryRoles(),decision_backend=BoundaryRouter(),profile=WithDeliberation())
    final=graph.invoke(state,{'configurable':{'thread_id':'boundary'},'recursion_limit':100})
    assert not any(e['component']=='verification' and e['kind']=='role_start' for e in trace.events)
    assert final['report']['terminal_reason'] in {'action_budget','llm_budget','invalid_deliberation'}


@pytest.mark.parametrize('value,status',[(True,'supported'),(False,'contradicted')])
def test_evaluable_verdict_requires_current_measurement_citation(value,status):
    lab,native=fake_lab();native.goal=value;execute(lab,'workflow.inspect')
    state=initial_state(lab,'citation',Limits());state['observations']=lab.observe()
    payload=InstrumentProfile.projection('verification',state,'request')
    for evidence in [[],['obs-other']]:
        report=VerificationReport(status=status,evidence_ids=evidence,missing_evidence=[])
        with pytest.raises(ValueError,match='current measurement'):
            InstrumentProfile.validate_role_contract('verification',report,payload)
    InstrumentProfile.validate_role_contract('verification',VerificationReport(
        status=status,evidence_ids=[payload['observations']['measurement']['id']],missing_evidence=[]),payload)


def test_supported_without_citation_gets_bounded_repair_before_success():
    lab,native=fake_lab();native.goal=True;execute(lab,'workflow.inspect')
    class CitationRepairRoles(ReadbackRoles):
        verification_calls=0
        def invoke(self,role,prompt,payload,schema):
            if role=='verification':
                self.verification_calls+=1
                ids=[] if self.verification_calls==1 else [payload['observations']['measurement']['id']]
                return schema.model_validate({'status':'supported','evidence_ids':ids,'missing_evidence':[]}).model_dump()
            return super().invoke(role,prompt,payload,schema)
    roles=CitationRepairRoles();trace=Trace('repair');limits=Limits()
    graph=build_graph(lab,trace,limits,roles=roles,decision_backend=RepeatVerifierRouter(),profile=InstrumentProfile())
    final=graph.invoke(initial_state(lab,'repair',limits),{'configurable':{'thread_id':'repair'}})
    assert roles.verification_calls==2 and final['report']['terminal_reason']=='agent_supported'
    assert lab.score(final['report'])['false_success'] is False
    assert len(final['verified_evidence_signatures'])==1
    assert sum(e['kind']=='role_error' for e in trace.events)==1
