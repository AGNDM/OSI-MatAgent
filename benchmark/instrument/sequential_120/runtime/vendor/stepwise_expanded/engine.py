"""Expanded curriculum checks over the frozen pilot's public native world.

The rollout loop below is retained from stepwise_bench.engine, with a separate
Curriculum binding so queued pilot experiments retain their original source.
Delayed actuator commands may declare required arguments as well as immediate
observable conditions; final actuator position still requires native waits and
observed verification.
"""
from __future__ import annotations

import copy
import time
from collections import defaultdict

from instrument_bench.adapters import conditions_hold
from stepwise_bench.engine import (Curriculum as PilotCurriculum,
                                   ReferencePolicy as PilotReferencePolicy,
                                   World, require_compute_node)
from stepwise_bench.policies import ResumeMismatchError


class Curriculum(PilotCurriculum):
    def trial(self, raw_action):
        judgement = super().trial(raw_action)
        node = self.nodes.get(judgement.get("checkpoint_id"))
        if judgement.get("correct") and node:
            required = node["accept"].get("arguments")
            if required is not None:
                actual = judgement["action"].get("arguments", {})
                if any(key not in actual or actual[key] != value for key, value in required.items()):
                    return {"correct": False, "action": judgement["action"],
                            "reason": "Command arguments do not satisfy the active actuator checkpoint",
                            "auxiliary": False}
        return judgement


class ReferencePolicy(PilotReferencePolicy):
    def __init__(self, case):
        super().__init__(case)
        self.curriculum = Curriculum(case)


def run_case(case, policy, mode="teacher_forced", on_step=None):
    require_compute_node()
    if mode not in {"teacher_forced", "free_run"}:
        raise ValueError("Unknown evaluation mode")
    curriculum = Curriculum(case)
    world = curriculum.world
    history, records = [], []
    node_scores, node_stages = defaultdict(list), {}
    termination = "decision_budget"
    corrections = query_attempts = query_valid = 0
    for index in range(case["budget"].get("max_decisions", 24)):
        start = time.perf_counter()
        proposal_error = None
        try:
            proposal = policy.next_action(copy.deepcopy(world.view(mode, index)), copy.deepcopy(history))
        except ResumeMismatchError:
            raise
        except Exception as error:
            proposal = None
            proposal_error = f"{type(error).__name__}: {error}"
        policy_latency = time.perf_counter() - start
        grade_started = time.perf_counter()
        judgement = (curriculum.trial(proposal) if proposal_error is None else
                     {"correct": False, "auxiliary": False, "reason": proposal_error})
        grader_latency = time.perf_counter() - grade_started
        correct = judgement["correct"]
        checkpoint = stage = None
        if judgement.get("auxiliary"):
            query_attempts += 1
            query_valid += int(correct)
        else:
            if "__machine__" not in curriculum.done:
                checkpoint, stage = "__machine__", "machine_selection"
            elif judgement.get("checkpoint_id"):
                checkpoint, stage = judgement["checkpoint_id"], judgement["stage"]
            else:
                nodes = curriculum.eligible()
                checkpoint = nodes[0]["checkpoint_id"] if nodes else "__no_continuation__"
                stage = nodes[0]["stage"] if nodes else "no_continuation"
            node_scores[checkpoint].append(bool(correct))
            node_stages[checkpoint] = stage
        corrected = mode == "teacher_forced" and not correct
        action = judgement.get("action")
        executed_judgement = judgement
        if corrected:
            try:
                action = curriculum.reference()
                executed_judgement = curriculum.trial(action)
                if not executed_judgement["correct"]:
                    raise RuntimeError("Declared correction is not natively admissible")
            except Exception as error:
                termination = "invalid_curriculum_correction"
                proposal_error = f"{type(error).__name__}: {error}"
                action = None
            corrections += 1
        if action is None:
            response = {"status": "rejected", "error": {"code": "INVALID_PUBLIC_ACTION"}}
        else:
            response = curriculum.commit(action, executed_judgement)
        public_history = {"proposed_action": copy.deepcopy(proposal), "executed_action": copy.deepcopy(action),
                          "corrected": corrected, "response": copy.deepcopy(response)}
        history.append(public_history)
        record = {"decision_index": index + 1, **copy.deepcopy(public_history),
                  "proposal_correct": bool(correct), "auxiliary_query": bool(judgement.get("auxiliary")),
                  "checkpoint_id": checkpoint, "stage": stage,
                  "grader_reason": judgement.get("reason"), "policy_error": proposal_error,
                  "policy_latency_s": policy_latency, "grader_latency_s": grader_latency,
                  "usage": copy.deepcopy(getattr(policy, "last_action_usage", {})),
                  "elapsed_sim_time_s": world.elapsed, "completed_checkpoints": sorted(curriculum.done)}
        records.append(record)
        if on_step:
            on_step(copy.deepcopy(record))
        if termination == "invalid_curriculum_correction":
            break
        if action and action["kind"] in {"finish", "escalate"}:
            termination = "finished" if action["kind"] == "finish" else "escalated"
            break
    goal_met = bool(case.get("goal_conditions")) and conditions_hold(world.adapters[world.target], case["goal_conditions"])
    escalated = termination == "escalated"
    escalation_evidence = any(node["accept"]["kind"] == "escalate" and node["checkpoint_id"] in curriculum.done for node in case["checkpoints"])
    justified = escalated and not goal_met and escalation_evidence and conditions_hold(world.adapters[world.target], case.get("escalation_conditions", []))
    finish_evidence = any(node["accept"]["kind"] == "finish" and node["checkpoint_id"] in curriculum.done for node in case["checkpoints"])
    outcome = bool(justified if case.get("escalation_expected") else
                   goal_met and termination == "finished" and finish_evidence)
    checkpoint_scores = {key: float(all(scores)) for key, scores in node_scores.items()}
    stages = defaultdict(list)
    for key, score in checkpoint_scores.items():
        stages[node_stages[key]].append(score)
    stage_accuracy = {stage: sum(scores) / len(scores) for stage, scores in stages.items()}
    usages = [row["usage"] for row in records]
    return {"task_id": case["task_id"], "group_id": case["group_id"], "instrument_id": case["instrument_id"],
            "mode": mode, "privileged_policy": bool(getattr(policy, "privileged", False)),
            "termination": termination, "goal_met": bool(goal_met), "escalated": escalated,
            "escalation_justified": bool(justified), "trajectory_outcome_met": outcome,
            "autonomous_success": outcome if mode == "free_run" else None,
            "corrected_trajectory_outcome_met": outcome if mode == "teacher_forced" else None,
            "all_original_decisions_admissible": outcome and bool(checkpoint_scores) and all(checkpoint_scores.values()),
            "checkpoint_accuracy": sum(checkpoint_scores.values()) / len(checkpoint_scores) if checkpoint_scores else None,
            "checkpoint_opportunities": len(checkpoint_scores), "checkpoint_scores": checkpoint_scores,
            "stage_accuracy": stage_accuracy, "machine_choice_correct": checkpoint_scores.get("__machine__"),
            "interventions": corrections, "decisions": len(records), "query_attempts": query_attempts,
            "decision_accuracy_all_proposals": sum(row["proposal_correct"] for row in records) / len(records) if records else None,
            "control_decision_accuracy": sum(row["proposal_correct"] for row in records if not row["auxiliary_query"]) / sum(not row["auxiliary_query"] for row in records) if any(not row["auxiliary_query"] for row in records) else None,
            "query_valid": query_valid, "automatic_observation_checkpoints": sorted(curriculum.auto_observation_nodes),
            "elapsed_sim_time_s": world.elapsed, "history": history, "decision_records": records,
            "final_public_observation": world.adapters[world.target].read(),
            "false_completion_claim": termination == "finished" and not outcome,
            "premature_finish_proposals": sum(isinstance(row["proposed_action"], dict) and
                row["proposed_action"].get("kind") == "finish" and not row["proposal_correct"] for row in records),
            "unjustified_escalation_proposals": sum(isinstance(row["proposed_action"], dict) and
                row["proposed_action"].get("kind") == "escalate" and not row["proposal_correct"] for row in records),
            "policy_latency_s": sum(row["policy_latency_s"] for row in records),
            "input_tokens": sum(usage.get("input_tokens", 0) for usage in usages),
            "output_tokens": sum(usage.get("output_tokens", 0) for usage in usages),
            "token_accounting_complete": bool(usages) and all(usage.get("input_tokens_known") and usage.get("output_tokens_known") for usage in usages),
            "api_cost_usd": sum(usage["api_cost_usd"] for usage in usages) if usages and all(usage.get("api_cost_usd") is not None for usage in usages) else None,
            "gpu_seconds": sum(usage["gpu_seconds"] for usage in usages) if usages and all(usage.get("gpu_seconds") is not None for usage in usages) else None,
            "limitations": ["Checkpoint admissibility uses a declared partial-order curriculum, not a globally optimal planner.",
                            "Machines are independent runtimes; material transfers and cross-instrument chemistry are not modeled.",
                            "Oracle-corrected completion is not autonomous agent success."]}
