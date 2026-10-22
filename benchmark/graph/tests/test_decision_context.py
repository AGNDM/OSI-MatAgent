"""Pure public-view tests; run with the graph package checks inside Slurm."""
from copy import deepcopy
import json

import pytest

from graph_local.decision_context import (
    DecisionContextError, DecisionContextLimitError,
    build_decision_context, serialize_decision_context,
)


def fixture():
    reading = {"value": 313.15, "unit": "K", "quality": "estimated",
               "sample_time_s": 5, "sample_window_s": [4, 5]}
    source = {
        "has_intent": True, "actions_used": 2, "last_error": {"code": "LOCKED"},
        "selected_intent": {"capability_id": "benchmark.invoke", "arguments": {"action": {
            "kind": "invoke", "instrument_id": "oven", "capability_id": "temperature.set",
            "arguments": {"temperature_K": 320}}}},
        "observations": {"revision": 2, "last_mutation_step": 1,
                         "measurement": {"id": "obs-2", "quality": "valid", "revision": 2,
                                         "captured_at_step": 2}},
        "benchmark_public_context": {
            "wrapper_revision": 2,
            "public_task": {"task_id": "public-task", "initial_context": {
                "machine_catalog": ["initial state is not current"], "private": "HIDDEN_ORACLE"}},
            "current_public_view": {
                "goal": "Obtain fresh native evidence at 320 K", "selected_instrument": "oven",
                "observation": {"clock": 9, "observations": {"temperature": reading}},
                "contract": {"capabilities": {
                    "temperature.set": {"input_schema": {"type": "object"},
                                        "preconditions": ["lock.off"]},
                    "unrelated.command": {"description": "UNRELATED_CONTRACT"}},
                    "state_slots": {"lock.off": {"unit": "1"}}},
                "manual": ["Readings retain their sample times."],
                "budget": {"max_decisions": 64, "decisions_used": 2},
                "elapsed_sim_time_s": 9, "mode": "free_run"},
            "complete_native_history": [],
            "complete_offered_commands_and_presets": [{"text": "UNRELATED_MENU"}],
        },
    }
    candidates = [{"candidate_id": "exact-id-7", "intent": deepcopy(source["selected_intent"]),
                   "rationale": "Compare the exact goal and supplied evidence.", "evidence_ids": ["obs-2"]}]
    return source, candidates


def test_exact_candidates_reading_times_citations_and_public_goal_preserved():
    source, candidates = fixture()
    before = deepcopy(source)
    result = build_decision_context(source, purpose="operational_action_selection", candidates=candidates)
    assert result["candidates"] == candidates
    assert result["state"]["current_observation"] == source["benchmark_public_context"]["current_public_view"]["observation"]
    assert result["state"]["observation_wrappers"]["measurement"]["id"] == "obs-2"
    assert result["state"]["goal"] == "Obtain fresh native evidence at 320 K"
    assert result["state"]["last_error"] == {"code": "LOCKED"}
    assert source == before
    text = json.dumps(result)
    assert "HIDDEN_ORACLE" not in text
    assert "UNRELATED_MENU" not in text
    assert "UNRELATED_CONTRACT" not in text
    assert "temperature.set" in result["candidate_contracts"]["capabilities"]
    assert "lock.off" in result["candidate_contracts"]["state_slots"]


def test_router_does_not_need_entire_menu_or_operation_candidates():
    source, _ = fixture()
    result = build_decision_context(source, purpose="routing", allowed_routes=["diagnose", "prepare_action"])
    assert result["allowed_routes"] == ["diagnose", "prepare_action"]
    assert "candidates" not in result
    assert result["state"]["selected_intent"] == source["selected_intent"]
    assert result["state"]["graph"]["has_intent"] is True


def test_history_growth_uses_explicit_progress_and_latest_relevant_receipts():
    source, candidates = fixture()
    action = candidates[0]["intent"]["arguments"]["action"]
    failed = {"executed_action": deepcopy(action), "proposed_action": deepcopy(action),
              "corrected": False, "response": {"status": "rejected", "error": {"code": "LOCKED"}}}
    succeeded = dict(failed, response={"status": "succeeded"})
    source["benchmark_public_context"]["complete_native_history"] = [failed] + [succeeded] * 999
    result = build_decision_context(source, purpose="operational_action_selection", candidates=candidates)
    history = result["history"]
    assert history["count"] == 1000
    assert history["omitted_receipt_count"] == 997
    assert [item["history_index"] for item in history["receipts"]] == [0, 998, 999]
    assert history["related_command_counts"][0]["count"] == 1000
    assert len(serialize_decision_context(source, purpose="operational_action_selection", candidates=candidates)) < 8000


def test_repeated_current_readings_are_lossless_explicit_references():
    source, _ = fixture()
    current = source["benchmark_public_context"]["current_public_view"]["observation"]["observations"]
    source["benchmark_public_context"]["complete_native_history"] = [{
        "executed_action": {"kind": "read", "instrument_id": "oven", "state_ids": None},
        "response": {"status": "succeeded", "observations": deepcopy(current)}}]
    result = build_decision_context(source, purpose="routing", allowed_routes=["diagnose", "verify"])
    assert result["history"]["receipts"][0]["receipt"]["response"]["observations"] == {
        "$decision_ref": "#/state/current_observation/observations"}
    assert result["state"]["current_observation"]["observations"] == current


def test_budget_overflow_is_explicit_and_never_changes_candidate_actions():
    source, candidates = fixture()
    before = deepcopy(candidates)
    with pytest.raises(DecisionContextLimitError, match="no evidence, candidates or text were truncated"):
        build_decision_context(source, purpose="operational_action_selection", candidates=candidates, max_bytes=100)
    assert candidates == before


def test_private_ranking_or_reference_fields_are_rejected():
    source, candidates = fixture()
    candidates[0]["correct"] = True
    with pytest.raises(DecisionContextError, match="unsupported fields"):
        build_decision_context(source, purpose="operational_action_selection", candidates=candidates)
    source["last_error"] = {"reference_action": {"kind": "finish"}}
    with pytest.raises(DecisionContextError, match="Non-public"):
        build_decision_context(source, purpose="routing", allowed_routes=["diagnose"])


def test_duplicate_menu_ids_nonfinite_input_and_invalid_routes_fail_closed():
    source, candidates = fixture()
    with pytest.raises(DecisionContextError, match="distinct"):
        build_decision_context(source, purpose="operational_action_selection", candidates=candidates * 2)
    with pytest.raises(DecisionContextError):
        build_decision_context(source, purpose="routing", allowed_routes=["diagnose", "diagnose"])
    source["last_error"] = float("nan")
    with pytest.raises(DecisionContextError, match="finite JSON"):
        build_decision_context(source, purpose="routing", allowed_routes=["diagnose"])


def test_deterministic_source_audit_changes_when_omitted_history_changes():
    source, _ = fixture()
    first = build_decision_context(source, purpose="routing", allowed_routes=["diagnose"])
    assert first == build_decision_context(deepcopy(source), purpose="routing", allowed_routes=["diagnose"])
    source["benchmark_public_context"]["complete_offered_commands_and_presets"].append({"text": "Another unrelated public option"})
    second = build_decision_context(source, purpose="routing", allowed_routes=["diagnose"])
    assert first["state"] == second["state"]
    assert first["audit"]["input_sha256"] != second["audit"]["input_sha256"]
