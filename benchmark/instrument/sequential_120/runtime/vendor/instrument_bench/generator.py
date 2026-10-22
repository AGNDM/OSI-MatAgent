"""Replay native simulators to generate public tasks and separate private oracles."""
from __future__ import annotations

from collections import Counter
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess

from . import SCHEMA_VERSION
from .adapters import SOURCE, apply_step, apply_steps, conditions_hold, make_adapter, values
from .scenarios import build_scenarios, family_inventory
from .schema import DatasetConfig

PROJECT = Path(__file__).resolve().parents[1]


def opaque_id(*parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:24]


def execute_with_events(adapter, steps, scheduled_events=()):
    """Run steps with persistent action offsets and exact simulated-time events.

    Call this repeatedly with a single solver step for interactive policies.
    The adapter retains applied-event IDs and its one-based action counter.
    Time events split long waits, so a policy cannot skip a mid-run fault.
    """
    results, events = [], []
    applied = getattr(adapter, "_scheduled_event_ids", set())
    adapter._scheduled_event_ids = applied

    def inject(event):
        event_id = opaque_id(event)
        responses = apply_steps(adapter, event["steps"], actor="fixture", require_success=True)
        applied.add(event_id)
        events.append({"event_id": event_id, "after_action": adapter._solver_action_index,
                       "time_s": adapter.time_s, "responses": responses})

    for step in steps:
        adapter._solver_action_index = getattr(adapter, "_solver_action_index", 0) + 1
        if step["kind"] == "wait":
            duration = step["dt_s"]
            if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 0 <= duration <= 86400:
                raise ValueError("Wait must be finite and between 0 and 86400 seconds")
            target = adapter.time_s + duration
            pending = sorted((event for event in scheduled_events
                              if "trigger_time_s" in event and opaque_id(event) not in applied
                              and adapter.time_s <= event["trigger_time_s"] <= target),
                             key=lambda event: event["trigger_time_s"])
            for event in pending:
                adapter.wait(max(0.0, event["trigger_time_s"] - adapter.time_s))
                inject(event)
            results.append(adapter.wait(max(0.0, target - adapter.time_s)))
        else:
            results.append(apply_step(adapter, step))
        for event in scheduled_events:
            if opaque_id(event) in applied:
                continue
            if (event.get("trigger_action_index") == adapter._solver_action_index
                    or ("trigger_time_s" in event and event["trigger_time_s"] <= adapter.time_s)):
                inject(event)
    return results, events


def fresh_case(case, seed, paired=False):
    adapter = make_adapter(case.instrument_id, seed)
    setup = case.paired_setup_steps if paired else case.setup_steps
    apply_steps(adapter, setup or [], actor="fixture", require_success=True)
    if not conditions_hold(adapter, case.fixture_conditions):
        raise AssertionError(f"Fixture does not establish its stated assumptions: {case.family_id}")
    return adapter


def _reading_results(results):
    # Time/revision metadata cannot be used to distinguish hidden fault fixtures.
    evidence = []
    for result in results:
        if result.get("observations"):
            evidence.append({"observations": values(result)})
        if result.get("result"):
            evidence.append({"result": result["result"]})
    return evidence


def _public_contract(adapter):
    description = adapter.describe()
    caps = {}
    for name, capability in description["capabilities"].items():
        if capability.get("availability", "implemented") != "implemented":
            continue
        if capability.get("visibility", "agent") != "agent":
            continue
        keys = ("input_schema", "output_schema", "preconditions", "preconditions_zh",
                "description_zh", "function_zh", "errors", "observes")
        caps[name] = {key: value for key, value in capability.items() if key in keys}
    slots = {key: value for key, value in description["state_slots"].items()
             if value.get("visibility", "agent") == "agent"}
    if any(key.startswith("truth.") for key in slots) or "fault.inject" in caps:
        raise AssertionError("Evaluator fields escaped native public description")
    return {"capabilities": caps, "state_slots": slots}


def _probe_outcome(adapter, case):
    responses, _ = execute_with_events(adapter, case.probe_steps)
    if case.goal_conditions:
        return "goal_met" if conditions_hold(adapter, case.goal_conditions) else "goal_not_met"
    if not responses:
        raise ValueError(f"Prediction scenario needs a probe: {case.family_id}")
    if case.prediction_target == "error_code":
        return (responses[-1].get("error") or {}).get("code", "NO_ERROR")
    return responses[-1]["status"]


def _evaluate_option(case, seed, candidate):
    adapter = fresh_case(case, seed)
    start = adapter.time_s
    probe_outcome = _probe_outcome(adapter, case) if case.task_type == "predict" else None
    results, events = execute_with_events(adapter, candidate["steps"], case.scheduled_events)
    answers = [result["answer"] for result in results if result.get("status") == "answered"]
    escalated = any(result.get("status") == "escalated" for result in results)
    goal_met = conditions_hold(adapter, case.goal_conditions) if case.goal_conditions else False
    informative = False
    if case.task_type == "predict":
        expected = "insufficient_information" if case.paired_setup_steps is not None else probe_outcome
        valid = answers == [expected]
        label_class = expected
    elif case.task_type == "diagnose":
        matches = [label for label, criteria in case.answer_conditions.items()
                   if conditions_hold(adapter, criteria)]
        if not matches:
            raise ValueError(f"No observable diagnosis rule matched: {case.family_id}")
        valid = len(answers) == 1 and answers[0] in matches
        label_class = "|".join(sorted(matches))
    elif case.task_type == "measure":
        paired = fresh_case(case, seed, paired=True)
        paired_results, _ = execute_with_events(paired, candidate["steps"], case.scheduled_events)
        actual_readings, paired_readings = _reading_results(results), _reading_results(paired_results)
        informative = bool(actual_readings and paired_readings and actual_readings != paired_readings)
        valid = informative and not escalated and not answers
        label_class = "information_request"
    else:
        valid = (escalated and not goal_met and conditions_hold(adapter, case.escalation_conditions)
                 if case.escalation_expected else goal_met and not escalated)
        label_class = "escalate" if case.escalation_expected else "operational_goal"
    # A choice that reaches a goal after a rejected action is still a valid
    # recovery sequence; the independent end-state predicate is authoritative.
    return {"valid": valid, "label_class": label_class, "goal_met": goal_met,
            "escalated": escalated, "informative": informative,
            "probe_outcome": probe_outcome, "responses": results, "events": events,
            "final_public_observation": adapter.read(),
            "elapsed_sim_time_s": adapter.time_s - start, "trace": adapter.trace}


def _split(case, group_id, config):
    if case.instrument_id in config.heldout_instruments:
        return "test_instrument"
    if case.fault_family in config.heldout_fault_families:
        return "test_fault"
    fraction = int(hashlib.sha256(group_id.encode()).hexdigest()[:12], 16) / float(16**12)
    if fraction < config.template_test_fraction:
        return "test_template"
    if fraction < config.template_test_fraction + config.test_fraction:
        return "test"
    if fraction < config.template_test_fraction + config.test_fraction + config.validation_fraction:
        return "validation"
    return "train"


def render_question(case, variant, held_out=False):
    """Distinct LLM-authored prompt frames, with whole groups held out.

    The structured objective and physical evidence remain identical; this is a
    question-format generalization protocol, not a claim of human paraphrases.
    """
    instructions = {
        "predict": "Assess the proposed steps in goal.proposed_steps against the current readout and contract. Select the predicted result, using insufficient information when the evidence cannot establish it.",
        "diagnose": "Classify the incident from the readout under the candidate hypotheses and operating assumptions stated in goal.objective.",
        "measure": "Select an inspection that distinguishes the unresolved hypotheses in goal.objective by producing new instrument evidence.",
        "recover": "Select an action sequence that establishes the operational conditions in goal, or request maintenance for persistent hardware failure.",
        "adapt": "Choose a response that completes the operational goal after a change during simulated execution; verify the resulting state.",
    }[case.task_type]
    if held_out:
        if variant % 2 == 0:
            return ("SHIFT HANDOVER\nThe pending operation is specified in goal.objective. "
                    "The recorded instrument state is the evidence available at handover. " + instructions,
                    "heldout_shift_handover_v1")
        return ("INSTRUMENT REVIEW CARD\nEvidence: the attached readout and public contract. "
                "Acceptance target: the structured goal. " + instructions
                + " Return the offered response supported by that evidence.", "heldout_review_card_v1")
    if variant % 2 == 0:
        return case.question, "scenario_question_v1"
    return ("Operator procedure review. " + instructions
            + " Explain the choice using the supplied operational assumptions.", "procedure_review_v1")


def _identity_plan(config):
    """Union contrast groups that share an indistinguishable public query.

    Private physical parameters can differ while the actual solver input stays
    the same. Such queries must share a split, and identical decision inputs
    contribute only one base case. Public option order and wording are ignored.
    """
    parent, by_base, by_public, records = {}, {}, {}, []

    def root(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = root(parent[key])
        return parent[key]

    def union(first, second):
        a, b = root(first), root(second)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for index in range(config.seeds_per_family):
        seed = int(opaque_id(config.seed, index)[:7], 16)
        cases = build_scenarios(seed)
        signatures = {}
        for case in cases:
            signatures.setdefault(case.family_id, []).append(asdict(case))
        for case in cases:
            group = opaque_id("group", case.instrument_id, case.family_id, signatures[case.family_id])
            base = opaque_id("base", case.instrument_id, asdict(case))
            root(group)
            if base in by_base:
                union(group, by_base[base]["original_group"])
                continue
            adapter = fresh_case(case, seed)
            observation = adapter.read(case.observation_ids)
            history = ([record for record in adapter.trace[case.history_from_step:] if record["actor"] in ("agent", "host")]
                       if case.history_from_step is not None else [])
            options = sorted(json.dumps(option["steps"], sort_keys=True) for option in case.options)
            projection = {"instrument_id": case.instrument_id, "objective": case.question,
                          "observation": observation, "history": history, "options": options,
                          "manual": case.manual, "proposed_steps": case.probe_steps,
                          "goal_conditions": [condition for condition in case.goal_conditions if condition.get("visibility", "agent") == "agent"]}
            semantic = opaque_id("public-query", projection)
            keep = semantic not in by_public
            if not keep:
                union(group, by_public[semantic]["original_group"])
            record = {"base_id": base, "original_group": group, "keep": keep}
            by_base[base] = record
            by_public.setdefault(semantic, record)
            records.append(record)
    components = {}
    for original in parent:
        components.setdefault(root(original), []).append(original)
    for record in records:
        record["group_id"] = opaque_id("semantic-equivalence-group", sorted(components[root(record["original_group"])]))
    return by_base


def generate_records(config=None):
    """Yield (public_task, private_oracle); every option uses a fresh runtime."""
    config = config or DatasetConfig()
    if config.seeds_per_family < 1 or config.variants_per_case < 1:
        raise ValueError("Seed and variant counts must be positive")
    if not 0 <= config.test_fraction + config.validation_fraction + config.template_test_fraction < 1:
        raise ValueError("Validation/test fractions must sum to less than one")
    identities = _identity_plan(config)
    seen_base_cases = set()
    for index in range(config.seeds_per_family):
        simulation_seed = int(opaque_id(config.seed, index)[:7], 16)
        seed_cases = build_scenarios(simulation_seed)
        signatures = {}
        for candidate in seed_cases:
            signatures.setdefault(candidate.family_id, []).append(asdict(candidate))
        for case in seed_cases:
            # Noise seeds alone must not put identical physical fixtures into
            # different splits, or count copies as independent benchmark tasks.
            group_id = opaque_id("group", case.instrument_id, case.family_id, signatures[case.family_id])
            base_id = opaque_id("base", case.instrument_id, asdict(case))
            identity = identities[base_id]
            if base_id in seen_base_cases or not identity["keep"]:
                continue
            seen_base_cases.add(base_id)
            group_id = identity["group_id"]
            adapter = fresh_case(case, simulation_seed)
            observation = adapter.read(case.observation_ids)
            contract = _public_contract(adapter)
            initial_trace = copy.deepcopy(adapter.trace)
            witness = None
            if case.paired_setup_steps is not None:
                paired = fresh_case(case, simulation_seed, paired=True)
                paired_observation = paired.read(case.observation_ids)
                same = observation == paired_observation
                if not same:
                    raise AssertionError(f"Hidden-state pair is observably different: {case.family_id}")
                first, second = fresh_case(case, simulation_seed), fresh_case(case, simulation_seed, paired=True)
                probe_first = _probe_outcome(first, case) if case.probe_steps or case.goal_conditions else None
                probe_second = _probe_outcome(second, case) if case.probe_steps or case.goal_conditions else None
                if case.task_type == "predict" and probe_first == probe_second:
                    raise AssertionError("Insufficient-information witness has identical outcomes")
                witness = {"same_public_values": True, "same_public_envelope": True,
                           "public_observation": observation, "paired_public_observation": paired_observation,
                           "public_values": values(observation),
                           "paired_public_values": values(paired_observation), "outcomes": [probe_first, probe_second],
                           "paired_setup_steps": case.paired_setup_steps}
            candidates = [_evaluate_option(case, simulation_seed, option) for option in case.options]
            good_indices = [i for i, result in enumerate(candidates) if result["valid"]]
            if not good_indices:
                raise AssertionError(f"Scenario has no executable valid answer: {case.family_id}")
            if len(good_indices) == len(candidates):
                raise AssertionError(f"Scenario has no informative distractor: {case.family_id}")
            if case.task_type in ("recover", "adapt"):
                reference = fresh_case(case, simulation_seed)
                reference_results, _ = execute_with_events(reference, case.reference_steps, case.scheduled_events)
                reference_valid = (any(result.get("status") == "escalated" for result in reference_results)
                                   if case.escalation_expected else conditions_hold(reference, case.goal_conditions))
                if not reference_valid:
                    raise AssertionError(f"Reference policy failed: {case.family_id}")
            for variant in range(config.variants_per_case):
                task_id = opaque_id("task", base_id, variant)
                order = list(range(len(case.options)))
                random.Random(int(task_id[:12], 16)).shuffle(order)
                options, correct, candidate_results = [], [], []
                for position, original_index in enumerate(order):
                    option_id = chr(ord("A") + position)
                    options.append({"option_id": option_id, **copy.deepcopy(case.options[original_index])})
                    candidate_results.append({"option_id": option_id, **copy.deepcopy(candidates[original_index])})
                    if original_index in good_indices:
                        correct.append(option_id)
                split = _split(case, group_id, config)
                question, template_id = render_question(case, variant, held_out=split == "test_template")
                if case.task_type in ("recover", "adapt"):
                    task_format = "sequential"
                elif case.task_type == "measure":
                    task_format = "variable_choice"
                elif any(step.get("answer") == "insufficient_information"
                         for option in options for step in option["steps"]):
                    task_format = "binary_abstain"
                else:
                    task_format = "binary" if len(options) == 2 else "multiple_choice"
                public_history = []
                if case.history_from_step is not None:
                    public_history = [record for record in initial_trace[case.history_from_step:]
                                      if record["actor"] in ("agent", "host")]
                task = {"schema_version": SCHEMA_VERSION, "task_id": task_id, "group_id": group_id,
                        "instrument_id": case.instrument_id, "domain": case.domain,
                        "task_type": case.task_type, "difficulty": case.difficulty, "tier": case.tier,
                        "decision_kind": case.decision_kind,
                        "difficulty_axes": case.difficulty_axes, "format": task_format,
                        "options": options, "observation": observation, "history": public_history,
                        "question": question, "instrument_contract": contract, "manual": case.manual,
                        "contract": {**contract, "operational_notes": case.manual},
                        "goal": {"objective": case.question,
                                 "conditions": [criteria for criteria in case.goal_conditions if criteria.get("visibility", "agent") == "agent"],
                                 "proposed_steps": case.probe_steps},
                        "proposed_steps": case.probe_steps,
                        "goal_conditions": [criteria for criteria in case.goal_conditions if criteria.get("visibility", "agent") == "agent"],
                        "allowed_actions": list(contract["capabilities"]) + ["read", "wait", "escalate"],
                        "budget": case.budget, "split": split,
                        "template_id": template_id, "seed": simulation_seed, "risk": case.risk}
                oracle = {"schema_version": SCHEMA_VERSION, "task_id": task_id, "group_id": group_id,
                          "base_case_id": base_id, "instrument_id": case.instrument_id, "seed": simulation_seed,
                          "family_id": case.family_id, "fault_family": case.fault_family,
                          "decision_kind": case.decision_kind, "prediction_target": case.prediction_target,
                          "evidence_condition": case.evidence_condition,
                          "setup_steps": case.setup_steps, "reference_steps": case.reference_steps,
                          "fixture_conditions": case.fixture_conditions,
                          "goal_conditions": case.goal_conditions, "correct_option_ids": correct,
                          "causes": case.causes, "scheduled_events": case.scheduled_events,
                          "escalation_expected": case.escalation_expected,
                          "escalation_conditions": case.escalation_conditions,
                          "healthy_control": case.healthy_control, "candidate_results": candidate_results,
                          "label_class": candidates[good_indices[0]]["label_class"],
                          "expected_information_request": case.task_type == "measure" or witness is not None,
                          "information_witness": witness, "fixture_trace": initial_trace,
                          "label_source": {
                              "predict": "matched_observation_counterfactual_execution" if witness else "native_probe_execution",
                              "diagnose": "observable_diagnostic_rules_with_restricted_hypotheses",
                              "measure": "paired_hypothesis_readout_discrimination_by_execution",
                              "recover": "native_trajectory_and_explicit_operational_predicates",
                              "adapt": "native_trajectory_and_explicit_operational_predicates",
                          }[case.task_type]}
                yield task, oracle


def generate_dataset(output_dir, config=None, *, seeds_per_family=None, seed=None, variants_per_case=None):
    """Write tasks.jsonl, private/oracles.jsonl and a reproducibility manifest."""
    config = config or DatasetConfig()
    for field, value in (("seeds_per_family", seeds_per_family), ("seed", seed), ("variants_per_case", variants_per_case)):
        if value is not None:
            setattr(config, field, value)
    output_dir = Path(output_dir)
    lock = json.loads((PROJECT / "environment/chemlab-source.json").read_text())
    actual_commit = subprocess.run(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], check=True,
                                   text=True, capture_output=True).stdout.strip()
    if actual_commit != lock["source_revision"]:
        raise RuntimeError("Runtime checkout differs from the recorded source pin")
    private_dir = output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    task_path, oracle_path = output_dir / "tasks.jsonl", private_dir / "oracles.jsonl"
    task_staging, oracle_staging = output_dir / ".tasks.jsonl.tmp", private_dir / ".oracles.jsonl.tmp"
    counts = {key: Counter() for key in ("instrument", "domain", "task_type", "decision_kind", "difficulty", "format", "split", "label_class")}
    task_count, healthy_count, unique_correct = 0, 0, 0
    with task_staging.open("w") as public_file, oracle_staging.open("w") as private_file:
        oracle_staging.chmod(0o600)
        for task, oracle in generate_records(config):
            public_file.write(json.dumps(task, ensure_ascii=False, allow_nan=False) + "\n")
            private_file.write(json.dumps(oracle, ensure_ascii=False, allow_nan=False) + "\n")
            task_count += 1
            healthy_count += int(oracle["healthy_control"])
            unique_correct += int(len(oracle["correct_option_ids"]) == 1)
            for field in counts:
                source = task.get({"instrument": "instrument_id"}.get(field, field), oracle.get(field))
                counts[field][source] += 1
    audit = audit_release_files(task_staging, oracle_staging)
    task_staging.replace(task_path)
    oracle_staging.replace(oracle_path)
    manifest = {"schema_version": SCHEMA_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source": {**lock, "actual_source_revision": actual_commit}, "config": asdict(config),
                "generator_sha256": {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in sorted((PROJECT / "instrument_bench").glob("*.py"))
                                     if path.name in ("schema.py", "adapters.py", "scenarios.py", "generator.py")},
                "task_count": task_count, "healthy_control_count": healthy_count,
                "audit": audit,
                "deduplicated_variant_count": len(build_scenarios(0)) * config.seeds_per_family * config.variants_per_case - task_count,
                "unique_correct_count": unique_correct, "counts": {key: dict(value) for key, value in counts.items()},
                "families": family_inventory(), "files": {"tasks": str(task_path), "oracles": str(oracle_path)},
                "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (task_path, oracle_path)},
                "limitations": ["Synthetic instrument-operation behavior, not real-equipment reliability or chemical outcomes.",
                                "Door movement uses a Python scene callback; no robot-contact validation is claimed.",
                                "Information-request labels test discrimination between the explicitly paired lamp hypotheses, not global diagnostic optimality.",
                                "Question templates are LLM-authored and not independently human-validated; held-out template IDs occur only in test_template groups.",
                                "Diagnosis labels use explicit observable rules within the stated candidate hypotheses; these are not general causal-identifiability proofs.",
                                "Difficulty levels are construction strata; empirical ordering must be measured."]}
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def audit_release_files(task_path, oracle_path):
    """Reject duplicate decision inputs and split leakage before publication."""
    groups, queries, bases, task_ids, templates = {}, {}, set(), set(), {}
    for line in Path(task_path).open():
        task = json.loads(line)
        if task["task_id"] in task_ids:
            raise AssertionError("Duplicate task identifier")
        task_ids.add(task["task_id"])
        previous = groups.setdefault(task["group_id"], task["split"])
        if previous != task["split"]:
            raise AssertionError("A semantic equivalence group crosses splits")
        templates.setdefault(task["split"], set()).add(task["template_id"])
    oracle_records = {record["task_id"]: record for record in map(json.loads, Path(oracle_path).open())}
    if task_ids != set(oracle_records):
        raise AssertionError("Public/private identifiers do not match")
    for line in Path(task_path).open():
        task = json.loads(line)
        oracle = oracle_records[task["task_id"]]
        bases.add(oracle["base_case_id"])
        canonical = {key: task[key] for key in ("instrument_id", "goal", "observation", "history", "manual", "contract", "allowed_actions", "budget")}
        canonical["options"] = sorted(json.dumps(option["steps"], sort_keys=True) for option in task["options"])
        signature = opaque_id("canonical-public-input", canonical)
        if queries.setdefault(signature, oracle["base_case_id"]) != oracle["base_case_id"]:
            raise AssertionError("Equivalent public queries have different base-case identities")
        offered = {option["option_id"] for option in task["options"]}
        if not set(oracle["correct_option_ids"]) or not set(oracle["correct_option_ids"]) <= offered:
            raise AssertionError("Invalid oracle choice identifiers")
    heldout = templates.get("test_template", set())
    ordinary = set().union(*(ids for split, ids in templates.items() if split != "test_template"))
    if heldout & ordinary:
        raise AssertionError("Held-out question templates leaked into another cohort")
    return {"task_count": len(task_ids), "base_case_count": len(bases), "equivalence_group_count": len(groups),
            "distinct_public_queries": len(queries), "cross_split_groups": 0,
            "duplicate_public_queries_across_base_cases": 0, "heldout_template_overlap": 0}
