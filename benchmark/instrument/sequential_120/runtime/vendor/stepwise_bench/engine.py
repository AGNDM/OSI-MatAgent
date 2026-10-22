"""Execute public instrument actions with optional evaluator-owned corrections.

The curriculum is a declared partial-order admissibility rubric, not a globally
optimal planning oracle. Proposed actions are tested on deterministic shadow
replays before an intervention; they never mutate the corrected live world.
"""
from __future__ import annotations

import copy
import math
import os
import time
from collections import defaultdict

from instrument_bench.adapters import INSTRUMENTS, apply_steps, conditions_hold, make_adapter
from instrument_bench.generator import execute_with_events

from .tasks import machine_catalog
from .policies import ResumeMismatchError


def require_compute_node():
    if not (os.environ.get("SLURMD_NODENAME") and os.environ.get("SLURM_JOB_ID")):
        raise RuntimeError("Run this evaluation inside a Slurm compute allocation")


def _contract(adapter):
    description = adapter.describe()
    capabilities = {}
    for name, spec in description["capabilities"].items():
        if spec.get("visibility", "agent") != "agent" or spec.get("availability", "implemented") != "implemented":
            continue
        capabilities[name] = {key: copy.deepcopy(spec[key]) for key in (
            "input_schema", "output_schema", "description_zh", "function_zh",
            "preconditions", "preconditions_zh", "errors", "observes") if key in spec}
    slots = {name: {key: copy.deepcopy(spec[key]) for key in (
        "unit", "value_schema", "origin", "visibility", "access") if key in spec}
        for name, spec in description["state_slots"].items()
        if spec.get("visibility", "agent") == "agent"}
    assert "fault.inject" not in capabilities and not any(name.startswith("truth.") for name in slots)
    return {"capabilities": capabilities, "state_slots": slots}


class World:
    """Independent instruments, a shared elapsed-time budget, and public caches."""

    def __init__(self, case):
        self.case = case
        self.target = case["instrument_id"]
        self.adapters, self.contracts, self.observations = {}, {}, {}
        self.selected = None
        adapter = make_adapter(self.target, case.get("seed", 42))
        apply_steps(adapter, case.get("setup_steps", []), actor="fixture", require_success=True)
        if not conditions_hold(adapter, case.get("fixture_conditions", [])):
            raise RuntimeError("Native setup does not establish its declared fixture")
        self.adapters[self.target] = adapter
        self.start_s = adapter.time_s
        self.contracts[self.target] = _contract(adapter)
        self.observations[self.target] = adapter.read(case.get("observation_ids"))
        if self.observations[self.target].get("status") != "succeeded":
            raise RuntimeError("Initial observation must contain only supported public readings")

    @property
    def elapsed(self):
        return self.adapters[self.target].time_s - self.start_s

    def get(self, instrument_id):
        if instrument_id not in INSTRUMENTS:
            raise ValueError("Unknown instrument")
        if instrument_id not in self.adapters:
            adapter = make_adapter(instrument_id, self.case.get("seed", 42))
            if self.elapsed:
                adapter.wait(self.elapsed)
            self.adapters[instrument_id] = adapter
            self.contracts[instrument_id] = _contract(adapter)
            self.observations[instrument_id] = adapter.read()
        return self.adapters[instrument_id]

    def normalize(self, action):
        if not isinstance(action, dict):
            raise ValueError("A proposed action must be one JSON object")
        kind = action.get("kind")
        keys = {"kind", "instrument_id"}
        extras = {"invoke": {"capability_id", "arguments"}, "read": {"state_ids"},
                  "wait": {"dt_s"}, "select_instrument": set(), "describe": set(),
                  "finish": set(), "escalate": set()}
        if kind not in extras or set(action) - keys - extras.get(kind, set()):
            raise ValueError("Unknown action kind or non-public action fields")
        machine = action.get("instrument_id", self.selected)
        if kind == "wait":
            dt = action.get("dt_s")
            if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
                raise ValueError("Wait duration must be finite and positive")
            if dt > 86400 or self.elapsed + dt > self.case["budget"]["max_sim_time_s"] + 1e-8:
                raise ValueError("Simulation-time budget exceeded")
            return {"kind": kind, "dt_s": float(dt)}
        if kind in {"finish", "escalate"}:
            return {"kind": kind}
        adapter = self.get(machine)
        clean = {"kind": kind, "instrument_id": machine}
        if kind == "invoke":
            cap, arguments = action.get("capability_id"), action.get("arguments", {})
            if cap not in self.contracts[machine]["capabilities"] or not isinstance(arguments, dict):
                raise ValueError("Command must be implemented and public, with object arguments")
            clean.update(capability_id=cap, arguments=copy.deepcopy(arguments))
        elif kind == "read":
            slots = action.get("state_ids")
            if slots is not None and (not isinstance(slots, list) or len(slots) != len(set(slots))
                                      or any(not isinstance(slot, str) or slot not in self.contracts[machine]["state_slots"] for slot in slots)):
                raise ValueError("Only public state identifiers may be read")
            clean["state_ids"] = copy.deepcopy(slots)
        return clean

    def execute(self, action):
        action = self.normalize(action)
        kind = action["kind"]
        if kind == "select_instrument":
            self.selected = action["instrument_id"]
            return {"status": "succeeded", "instrument_id": self.selected,
                    "contract": copy.deepcopy(self.contracts[self.selected]),
                    "observations": copy.deepcopy(self.observations[self.selected].get("observations", {}))}
        if kind == "describe":
            return {"status": "succeeded", "instrument_id": action["instrument_id"],
                    "contract": copy.deepcopy(self.contracts[action["instrument_id"]])}
        if kind in {"finish", "escalate"}:
            return {"status": "finished" if kind == "finish" else "escalated"}
        target_adapter = self.adapters[self.target]
        if kind == "wait":
            results, _ = execute_with_events(target_adapter, [action], self.case.get("scheduled_events", []))
            for name, adapter in self.adapters.items():
                if name != self.target:
                    adapter.wait(action["dt_s"])
            result, machine = results[0], self.target
        else:
            machine = action["instrument_id"]
            adapter = self.get(machine)
            results, _ = execute_with_events(adapter, [action], self.case.get("scheduled_events", []) if machine == self.target else [])
            result = results[0]
        if result.get("status") == "succeeded" and result.get("observations"):
            cached = self.observations[machine]
            cached.setdefault("observations", {}).update(copy.deepcopy(result["observations"]))
            for key in ("clock", "state_revision"):
                if key in result:
                    cached[key] = copy.deepcopy(result[key])
        return result

    def view(self, mode, decisions):
        selected = self.selected
        return {"goal": self.case["goal"], "machine_catalog": copy.deepcopy(machine_catalog()),
                "selected_instrument": selected,
                "contract": copy.deepcopy(self.contracts[selected]) if selected else {},
                "observation": copy.deepcopy(self.observations[selected]) if selected else {},
                "manual": copy.deepcopy(self.case.get("manual", [])),
                "allowed_action_kinds": ["select_instrument", "describe", "invoke", "read", "wait", "finish", "escalate"],
                "budget": {**copy.deepcopy(self.case["budget"]), "decisions_used": decisions},
                "elapsed_sim_time_s": self.elapsed,
                "observation_policy": "Cached public readings retain their sample times. Commands return their native responses; explicit reads consume a decision. Select an instrument before operating it.",
                "mode": mode}


class Curriculum:
    def __init__(self, case):
        self.case, self.world = case, World(case)
        self.done = set()
        self.waited = defaultdict(float)
        self.executed = []
        self.queries = 0
        self.auto_observation_nodes = set()
        self.nodes = {node["checkpoint_id"]: node for node in case["checkpoints"]}

    def eligible(self):
        return [node for node in self.case["checkpoints"]
                if node["checkpoint_id"] not in self.done
                and set(node.get("depends_on", [])) <= self.done
                and conditions_hold(self.world.adapters[self.world.target], node.get("when_conditions", []))]

    def reference(self):
        if "__machine__" not in self.done:
            return {"kind": "select_instrument", "instrument_id": self.world.target}
        eligible = self.eligible()
        if not eligible:
            raise RuntimeError("Curriculum has no admissible continuation")
        node = eligible[0]
        action = copy.deepcopy(node["reference_action"])
        if action["kind"] not in {"wait", "finish", "escalate"}:
            action.setdefault("instrument_id", self.world.target)
        if action["kind"] == "wait":
            remaining = self.case["budget"]["max_sim_time_s"] - self.world.elapsed
            action["dt_s"] = min(action["dt_s"], remaining)
            if action["dt_s"] <= 0:
                raise RuntimeError("No simulation time remains for a correction")
        return action

    def shadow(self):
        other = World(self.case)
        for action in self.executed:
            other.execute(action)
        return other

    def trial(self, raw_action):
        trial = self.shadow()
        try:
            action = trial.normalize(raw_action)
        except Exception as error:
            return {"correct": False, "action": None, "reason": str(error), "auxiliary": False}
        kind, machine = action["kind"], action.get("instrument_id")
        if "__machine__" not in self.done and kind == "select_instrument":
            return {"correct": machine == self.world.target, "action": action,
                    "checkpoint_id": "__machine__", "stage": "machine_selection", "auxiliary": False}
        if kind == "select_instrument" and machine == self.world.target and self.queries < self.case["budget"].get("max_queries", 4):
            return {"correct": True, "action": action, "stage": "information_query", "auxiliary": True}
        if kind in {"describe", "read"} and self.queries < self.case["budget"].get("max_queries", 4):
            auxiliary = True
        else:
            auxiliary = False
        if "__machine__" not in self.done and not auxiliary:
            return {"correct": False, "action": action, "reason": "Select the instrument first", "auxiliary": False}
        result = trial.execute(action)
        succeeded = result.get("status") in {"succeeded", "finished", "escalated"}
        for node in self.eligible() if "__machine__" in self.done else []:
            accept = node["accept"]
            required_kind = {"conditions": "invoke"}.get(accept["kind"], accept["kind"])
            if kind != required_kind or (kind not in {"wait", "finish", "escalate"} and machine != self.world.target):
                continue
            if kind == "invoke" and action.get("capability_id") != node["reference_action"].get("capability_id"):
                continue
            if not succeeded:
                continue
            complete = conditions_hold(trial.adapters[self.world.target], accept.get("conditions", []))
            if kind == "finish":
                complete = complete and bool(self.case.get("goal_conditions")) and conditions_hold(
                    trial.adapters[self.world.target], self.case["goal_conditions"])
            if kind == "escalate":
                complete = complete and bool(self.case.get("escalation_expected")) and conditions_hold(
                    trial.adapters[self.world.target], self.case.get("escalation_conditions", []))
            if kind == "read":
                complete = complete and set(accept.get("state_ids", [])) <= set(result.get("observations", {}))
            if kind == "wait":
                complete = complete and self.waited[node["checkpoint_id"]] + action["dt_s"] >= accept.get("min_dt_s", 0)
                # A bounded positive wait is admissible progress while this
                # declared time-dependent stage is active; completion is separate.
                return {"correct": True, "action": action, "checkpoint_id": node["checkpoint_id"],
                        "stage": node["stage"], "complete": complete, "auxiliary": False}
            if complete:
                return {"correct": True, "action": action, "checkpoint_id": node["checkpoint_id"],
                        "stage": node["stage"], "complete": True, "auxiliary": False}
        if auxiliary and succeeded:
            return {"correct": True, "action": action, "stage": "information_query", "auxiliary": True}
        return {"correct": False, "action": action, "reason": "Does not satisfy an available curriculum checkpoint", "auxiliary": False}

    def commit(self, action, judgement):
        result = self.world.execute(action)
        self.executed.append(copy.deepcopy(action))
        if judgement.get("auxiliary"):
            self.queries += 1
        checkpoint = judgement.get("checkpoint_id")
        if judgement.get("correct") and checkpoint:
            if checkpoint == "__machine__" or judgement.get("complete"):
                self.done.add(checkpoint)
            if action["kind"] == "wait":
                self.waited[checkpoint] += action["dt_s"]
        # Native commands may already return the requested observation. Do not
        # force an extra read if the public evidence is genuinely available.
        observed = set(result.get("observations", {})) if action["kind"] != "select_instrument" else set()
        for node in self.eligible():
            accept = node["accept"]
            if accept["kind"] == "read" and observed and set(accept.get("state_ids", [])) <= observed:
                if conditions_hold(self.world.adapters[self.world.target], accept.get("conditions", [])):
                    self.done.add(node["checkpoint_id"])
                    self.auto_observation_nodes.add(node["checkpoint_id"])
        return result


class ReferencePolicy:
    """Privileged curriculum diagnostic. Never report as a learned agent."""
    privileged = True

    def __init__(self, case):
        self.curriculum = Curriculum(case)
        self.position = 0
        self.last_action_usage = {"input_tokens": 0, "output_tokens": 0,
                                  "input_tokens_known": True, "output_tokens_known": True,
                                  "api_cost_usd": 0.0, "gpu_seconds": 0.0}

    def next_action(self, public_view, history):
        while self.position < len(history):
            action = history[self.position]["executed_action"]
            if action is not None:
                judgement = self.curriculum.trial(action)
                self.curriculum.commit(action, judgement)
            self.position += 1
        return self.curriculum.reference()


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
