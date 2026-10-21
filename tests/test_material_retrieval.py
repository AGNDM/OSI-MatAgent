from copy import deepcopy
import inspect
import json
import pytest
from app.material_retrieval import KnowledgeInstrumentProfile,retrieve_materials
from app.models import HypothesisSet,VerificationReport,Limits
from app.graph import validate_evidence,EvidenceReferenceError
from app.single_instrument import InstrumentTask,InstrumentLab
from app.instrument_cli import run_episode
from tests.test_single_instrument import LifecycleTestRoles


def payload():
    return {'target':{'objective':'Set stirring control to 100 rpm','predicates':[{'state_id':'stirring.set_rad_s','unit':'rad/s','value':10.471975511965978}]},
            'observations':{'inspection':{'id':'obs-1','native':{'observations':{
                'stirring.set_rad_s':{'value':0,'unit':'rad/s','quality':'valid'}}}}},'capabilities':{'stirring.set_speed':{}}}


def test_query_uses_public_context_only_and_is_deterministic():
    p=payload();r=retrieve_materials('diagnostic',p)
    p['branch']='hidden-label';p['expected_repair']='secret';p['evaluator']={'phase_fraction':'secret'};p['target']['truth']='secret'
    assert retrieve_materials('diagnostic',p)==r
    changed=deepcopy(p);changed['observations']['inspection']['native']['observations']['stirring.set_rad_s']['value']=10
    assert retrieve_materials('diagnostic',changed)['query']!=r['query']
    assert len(r['hits'])<=4 and all('source' in h and 'locator' in h for h in r['hits'])
    assert 'hidden-label' not in json.dumps(r)


def test_paired_control_changes_only_retrieval_evidence():
    p=payload();before=deepcopy(p)
    on=retrieve_materials('diagnostic',p,True);off=retrieve_materials('diagnostic',p,False)
    assert p==before and on['query']==off['query'] and on['corpus_sha256']==off['corpus_sha256']
    assert off['hits']==[on['hits'][0]] and 'ika-speed-units' in [h['id'] for h in on['hits']]


def test_material_and_observation_citation_namespaces_are_distinct():
    p=payload();p['retrieval_evidence']=retrieve_materials('diagnostic',p)['hits'];doc=p['retrieval_evidence'][1]['id']
    h=HypothesisSet(hypotheses=[{'cause':'hypothesis','supporting_ids':['obs-1'],'contradicting_ids':[],'material_evidence_ids':[doc]}],intent=None)
    validate_evidence('diagnostic',h,p)
    h.hypotheses[0].material_evidence_ids=['not-retrieved']
    with pytest.raises(EvidenceReferenceError):validate_evidence('diagnostic',h,p)
    with pytest.raises(EvidenceReferenceError):validate_evidence('verification',VerificationReport(status='supported',evidence_ids=[doc],missing_evidence=[]),p)


def test_new_profile_has_no_task_specific_execution_dependency():
    import app.material_retrieval as module
    source=inspect.getsource(module)
    assert 'app.titania' not in source and 'TitaniaLab' not in source
    assert not {'execute','_heat','_acquire'} & set(KnowledgeInstrumentProfile.__dict__)
    assert 'thermal.prepare_reference' not in source and 'xrd.acquire' not in source


def test_unavailable_chemistry_goal_stops_without_models_or_macro(tmp_path):
    task=InstrumentTask(task_id='native-only-chemistry',device='auto',objective='Establish TiO2 rutile formation using native public evidence',
        predicates=[{'state_id':'material.rutile_present','unit':'1','value':True}])
    class Never:
        is_llm=True;backend_id='offline-never-called'
        def invoke(self,*args):raise AssertionError('Unavailable native state needs no paid model')
        def choose(self,*args):raise AssertionError('No fabricated device or macro')
    result=run_episode(task,tmp_path/'unsupported',Never(),router=Never(),operation=Never())
    assert result['report']['terminal_reason']=='unsupported_objective'
    assert result['report']['llm_call_count']==0 and not result['score']['goal_met']
    assert not list((tmp_path/'unsupported').glob('execution/*'))


def test_contract_metadata_compaction_is_lossless_and_preserves_all_tools():
    from app.device_catalog import load_catalog,describe
    from app.public_contracts import compact_capabilities,expand_capabilities
    original=describe(load_catalog(),'ika_plate_rctdigital')['capabilities']
    compact,definitions=compact_capabilities(original)
    assert expand_capabilities(compact,definitions)==original
    assert set(compact)==set(original)
    assert all(compact[k]['input_schema']==original[k]['input_schema'] for k in original)
    assert len(json.dumps({'capabilities':compact,'definitions':definitions}))<len(json.dumps(original))
