from copy import deepcopy
import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.decision_backend import RulesBackend, select_route
from app.graph import build_graph, initial_state
from app.lab_adapter import GateRejected, LabAdapter
from app.models import ActionProposal, Limits, RouteDecision
from app.roles import ScriptedRoles
from app.scenarios import case_by_id, load_cases
from app.trace import Trace
from bench.replay import replay
from bench.run import run_episode, summarize_runs
from bench.score import score


def proposal(lab, capability, arguments=None, request_id=None):
    return ActionProposal(sample_id=lab.sample_id, expected_revision=lab.revision,
        capability_id=capability, arguments=arguments or {}, request_id=request_id or str(uuid4()),
        postcondition="test invariant")


def act(lab, capability, arguments=None):
    return lab.execute(proposal(lab, capability, arguments))


def claim(lab, measurement, status="supported"):
    return {"sample_id": lab.sample_id, "status": status,
        "terminal_reason": "agent_supported" if status == "supported" else "agent_unresolved",
        "evidence_ids": [measurement["id"]] if measurement else [], "hypotheses": []}


@pytest.mark.parametrize("case", load_cases(), ids=lambda c: c.case_id)
def test_four_families_complete_with_public_scripted_roles(case):
    r = run_episode(case)
    assert r["completed"] and r["score"]["task_success"]
    assert r["report"]["llm_call_count"] == 0
    assert r["score"]["expected_cause_covered"]
    assert r["report"]["action_count"] <= 3


def test_native_heating_changes_sample_and_requires_fresh_mass():
    fixture = case_by_id("dev-01")
    lab = LabAdapter(fixture)
    before = lab.evaluator_snapshot()["sample"]["liquid_kg"]
    act(lab, "workflow.inspect")
    event = act(lab, "temperature.set", {"temperature_k": fixture.target_temperature_k})
    assert event["result"]["native_response"]["status"] == "succeeded"
    assert lab.evaluator_snapshot()["sample"]["liquid_kg"] < before
    assert lab.evaluator_snapshot()["sample"]["liquid_kg"] <= fixture.residual_limit_kg
    assert not score(fixture, lab.evaluator_snapshot(), claim(lab, None))["task_success"]
    measurement = act(lab, "balance.measure")["result"]
    assert score(fixture, lab.evaluator_snapshot(), claim(lab, measurement))["task_success"]


@pytest.mark.parametrize("case_id,wrong_action", [("dev-01", "balance.clean"), ("dev-02", "temperature.set")])
def test_wrong_correction_and_false_success_are_rejected(case_id, wrong_action):
    fixture = case_by_id(case_id)
    lab = LabAdapter(fixture)
    act(lab, "workflow.inspect")
    args = {"temperature_k": fixture.target_temperature_k} if wrong_action == "temperature.set" else {}
    act(lab, wrong_action, args)
    m = act(lab, "balance.measure")["result"]
    result = score(fixture, lab.evaluator_snapshot(), claim(lab, m))
    assert not result["task_success"]
    assert result["false_success"]


def test_mutation_replay_preserves_identity_and_does_not_advance_world():
    lab = LabAdapter(case_by_id("dev-01"))
    act(lab, "workflow.inspect")
    p = proposal(lab, "temperature.set", {"temperature_k": 378.15})
    first = lab.execute(p)
    world = lab.evaluator_snapshot()
    assert lab.execute(p) == first
    assert lab.evaluator_snapshot() == world
    with pytest.raises(GateRejected, match="REQUEST_CONFLICT"):
        lab.execute(p.model_copy(update={"arguments": {"temperature_k": 383.15}}))


def test_stale_proposal_wrong_sample_and_missing_prerequisite_do_not_mutate():
    lab = LabAdapter(case_by_id("dev-01"))
    p = proposal(lab, "temperature.set", {"temperature_k": 378.15})
    with pytest.raises(GateRejected, match="INSPECTION_REQUIRED"):
        lab.execute(p)
    act(lab, "workflow.inspect")
    act(lab, "balance.clean")
    before = lab.evaluator_snapshot()
    with pytest.raises(GateRejected, match="STALE_PROPOSAL"):
        lab.execute(p)
    with pytest.raises(GateRejected, match="WRONG_SAMPLE"):
        lab.execute(proposal(lab, "balance.measure").model_copy(update={"sample_id": "other-sample"}))
    assert lab.evaluator_snapshot() == before


def test_native_evaluator_access_is_denied_and_public_state_has_no_truth():
    lab = LabAdapter(case_by_id("dev-01"))
    result = lab._agent.invoke(instance_id=lab.instance_id, capability_id="truth.inspect",
                              arguments={}, request_id="denied")
    assert result["error"]["code"] == "ACCESS_DENIED"
    with pytest.raises(ValueError, match="ACCESS_DENIED"):
        lab._agent.read(instance_id=lab.instance_id, state_ids=["truth.process"])
    trace = Trace("public-only")
    graph = build_graph(lab, trace)
    config = {"configurable": {"thread_id": "public-only"}, "recursion_limit": 60}
    final = graph.invoke(initial_state(lab, "public-only", Limits()), config)
    # Inspect every actual checkpoint and every role input, not just a fixture view.
    for snapshot in graph.get_state_history(config):
        public = json.dumps(snapshot.values)
        assert "truth.process" not in public and "initial_liquid_kg" not in public and '"family"' not in public
    for e in trace.events:
        if "input" in e:
            public = json.dumps(e["input"])
            assert "truth.process" not in public and '"family"' not in public
    assert final["report"]["status"] == "supported"


def test_old_or_wrong_sample_measurement_cannot_support_success():
    fixture = case_by_id("dev-03")
    lab = LabAdapter(fixture)
    act(lab, "workflow.inspect")
    m = act(lab, "balance.measure")["result"]
    act(lab, "balance.clean")
    assert score(fixture, lab.evaluator_snapshot(), claim(lab, m))["false_success"]
    m = act(lab, "balance.measure")["result"]
    world = lab.evaluator_snapshot()
    world["measurement"]["sample_id"] = "another-sample"
    assert score(fixture, world, claim(lab, m))["false_success"]


def test_action_budget_zero_stops_before_any_tool():
    record = run_episode(case_by_id("dev-01"), limits=Limits(lab_actions=0))
    assert record["report"]["terminal_reason"] == "action_budget"
    assert record["report"]["action_count"] == 0
    assert not record["score"]["task_success"]


def test_llm_budget_counts_attempts_including_transport_failure():
    class FailingLLM(ScriptedRoles):
        is_llm = True
        backend_id = "failing-test-transport"

        def invoke(self, *args):
            raise TimeoutError("test timeout")

    no_calls = run_episode(case_by_id("dev-01"), roles=FailingLLM(), limits=Limits(llm_calls=0))
    assert no_calls["report"]["terminal_reason"] == "llm_budget"
    assert no_calls["report"]["llm_call_count"] == 0
    failed = run_episode(case_by_id("dev-01"), roles=FailingLLM())
    assert failed["report"]["llm_call_count"] == 1
    assert not failed["score"]["task_success"]


def test_pending_and_deadline_stop_without_mutation():
    for fields, reason in [({"pending_operation": {"operation_id": "known-id"}}, "pending_operation_unresolved"),
                           ({"deadline": 0}, "deadline")]:
        lab = LabAdapter(case_by_id("dev-01"))
        before = lab.evaluator_snapshot()
        with pytest.raises(GateRejected, match="PENDING_OPERATION_UNRESOLVED"):
            lab.validate(proposal(lab, "balance.measure"), {"operation_id": "known-id"})
        state = initial_state(lab, reason, Limits())
        state.update(fields)
        final = build_graph(lab, Trace(reason)).invoke(state,
            {"configurable": {"thread_id": reason}, "recursion_limit": 60})
        assert final["report"]["terminal_reason"] == reason
        assert lab.evaluator_snapshot() == before


def test_invalid_router_stops_without_fallback(tmp_path):
    class InvalidRouter:
        backend_id = "invalid-test-decision-backend"

        def decide(self, summary, allowed):
            return {"backend_id": self.backend_id, "choice": "invented-route"}

    record = run_episode(case_by_id("dev-01"), tmp_path, decision_backend=InvalidRouter())
    assert not record["score"]["task_success"] and record["fallback_count"] == 0
    assert record["report"]["terminal_reason"] == "decision_model_error"
    events = [json.loads(line) for line in Path(record["trace_path"]).read_text().splitlines()]
    assert not any(e["kind"] == "tool_result" for e in events)


def test_low_confidence_keeps_argmax_without_deliberation(tmp_path):
    class UncertainRouter:
        backend_id = "uncertain-test-decision-backend"

        def decide(self, summary, allowed):
            choice="prepare_action" if summary['has_intent'] else "verify" if summary['has_measurement'] else "diagnose"
            p={r:(.51 if r==choice else .49/(len(allowed)-1)) for r in allowed} if len(allowed)>1 else {choice:1.}
            return RouteDecision(backend_id=self.backend_id, choice=choice, probabilities=p)

    record = run_episode(case_by_id("dev-03"), tmp_path, decision_backend=UncertainRouter())
    assert record["score"]["task_success"]
    assert record["fallback_count"] == 0


def test_graph_loop_budget_is_retained_as_failed_attempt(tmp_path):
    class LoopRouter:
        backend_id = "loop-test-decision-backend"

        def decide(self, summary, allowed):
            return RouteDecision(backend_id=self.backend_id, choice="diagnose")

    record = run_episode(case_by_id("dev-01"), tmp_path, limits=Limits(graph_steps=5), decision_backend=LoopRouter())
    assert record["attempted"] and not record["completed"]
    assert record["report"]["terminal_reason"] == "graph_budget"
    assert not record["score"]["task_success"]
    assert len((tmp_path / "runs.jsonl").read_text().splitlines()) == 1


def test_recorded_native_actions_replay_exactly_after_reset(tmp_path):
    record = run_episode(case_by_id("dev-01"), tmp_path)
    assert replay("dev-01", Path(record["trace_path"])) == record["report"]["action_count"]


def test_consistency_uses_all_repeats_including_failures():
    good = run_episode(case_by_id("dev-03"))
    failed = deepcopy(good)
    failed["score"]["task_success"] = False
    failed["completed"] = False
    row = summarize_runs([good, failed])[-1]
    assert row["attempted"] == 2 and row["completed"] == 1
    assert row["task_success_rate"] == 0.5 and row["repeat_consistency"] == 0


def test_verification_can_return_to_diagnosis_before_correcting_process():
    class MeasureFirstRoles(ScriptedRoles):
        def invoke(self, role, prompt, payload, schema):
            result = super().invoke(role, prompt, payload, schema)
            if role == "diagnostic":
                obs = payload["observations"]
                if obs["inspection"] and not obs["measurement"] and not obs["last_mutation_step"]:
                    result["intent"]["capability_id"] = "balance.measure"
                    result["intent"]["arguments"] = {}
            return result

    record = run_episode(case_by_id("dev-01"), roles=MeasureFirstRoles())
    assert record["score"]["task_success"]
    assert record["report"]["action_count"] == 4


def test_one_format_repair_counts_against_the_same_llm_budget(tmp_path):
    class MalformedOnceLLM(ScriptedRoles):
        is_llm = True
        backend_id = "malformed-test-llm"

        def __init__(self):
            self.calls = 0

        def invoke(self, role, prompt, payload, schema):
            self.calls += 1
            if self.calls == 1:
                return {"wrong": "shape"}
            return super().invoke(role, prompt, payload, schema)

    record = run_episode(case_by_id("dev-03"), tmp_path, roles=MalformedOnceLLM())
    assert record["score"]["task_success"]
    assert record["report"]["llm_call_count"] == 6
    budget = run_episode(case_by_id("dev-03"), roles=MalformedOnceLLM(), limits=Limits(llm_calls=1))
    assert budget["report"]["llm_call_count"] == 1
    assert budget["report"]["terminal_reason"] == "llm_budget"


def test_model_summaries_keep_backend_denominators_separate():
    first = run_episode(case_by_id("dev-03"))
    second = deepcopy(first)
    second["role_backend"] = "another-backend"
    second["score"]["task_success"] = False
    overall = [r for r in summarize_runs([first, second]) if r["family"] == "overall"]
    assert len(overall) == 2 and {r["task_success_rate"] for r in overall} == {0, 1}


def test_invented_evidence_is_rejected_before_actions(tmp_path):
    class InventedEvidenceLLM(ScriptedRoles):
        is_llm = True
        backend_id = "invented-evidence-test"

        def invoke(self, role, prompt, payload, schema):
            result = super().invoke(role, prompt, payload, schema)
            if role == "diagnostic":
                result["hypotheses"][0]["supporting_ids"] = ["nonexistent-observation"]
            return result

    record = run_episode(case_by_id("dev-01"), tmp_path, roles=InventedEvidenceLLM())
    assert record["report"]["action_count"] == 0
    assert record["report"]["llm_call_count"] == 2
    events = [json.loads(line) for line in Path(record["trace_path"]).read_text().splitlines()]
    assert any(e.get("error_code") == "E1" for e in events)


def test_missing_temperature_intent_is_repaired_before_procedure_handoff(tmp_path):
    class MissingTemperatureOnce(ScriptedRoles):
        is_llm = True
        backend_id = "missing-temperature-test"
        broken = False

        def invoke(self, role, prompt, payload, schema):
            result = super().invoke(role, prompt, payload, schema)
            if role == "diagnostic" and result.get("intent", {}).get("capability_id") == "temperature.set" and not self.broken:
                self.broken = True
                result["intent"]["arguments"] = {}
            if role == "procedure":
                assert payload["target"]["temperature_k"] == 378.15
                if payload["intent"]["capability_id"] == "temperature.set":
                    assert payload["intent"]["arguments"] == {"temperature_k": 378.15}
            return result
    record = run_episode(case_by_id("dev-01"), tmp_path, roles=MissingTemperatureOnce())
    assert record["score"]["task_success"]
    assert record["report"]["llm_call_count"] == 8
    assert record["report"]["action_count"] == 3


def test_procedure_argument_drift_cannot_reach_executor(tmp_path):
    class DriftingProcedure(ScriptedRoles):
        is_llm = True
        backend_id = "drifting-procedure-test"

        def invoke(self, role, prompt, payload, schema):
            result = super().invoke(role, prompt, payload, schema)
            if role == "procedure" and result["capability_id"] == "temperature.set":
                result["arguments"] = {"temperature_k": 323.15}
            return result
    record = run_episode(case_by_id("dev-01"), tmp_path, roles=DriftingProcedure())
    assert not record["score"]["task_success"]
    assert record["report"]["terminal_reason"] == "role_error"
    assert record["report"]["action_count"] == 1
    events = [json.loads(s) for s in Path(record["trace_path"]).read_text().splitlines()]
    assert len([e for e in events if e["kind"] == "role_error"]) == 2
    assert not any(e["kind"] == "tool_result" and e["proposal"]["capability_id"] == "temperature.set" for e in events)
