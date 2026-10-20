"""Distinct public input projections and validated specialist output records.

ScriptedRoles is a deterministic integration-test double. Replace this backend
through the protocol when real model endpoints are configured.
"""
from typing import Protocol

from pydantic import BaseModel

from app.models import ActionProposal, Hypothesis, HypothesisSet, Intent, RouteDecision, VerificationReport

PROMPTS = {
    "diagnostic": "Compare process failure, measurement failure, and an already acceptable sample. Use only cited public observations. Choose a discriminating check before correction. Return hypotheses and one intent using only the listed capabilities; never execute tools. Cause labels: undetermined, process_mismatch, pan_contamination, none, missing_evidence. If inspection is absent and no correction has occurred, inspect first. At inspection compare native.observations['temperature.set_k'].value (Kelvin) against target.temperature_k: a mismatch is process_mismatch evidence and may justify bounded temperature.set to the target. Pan contamination justifies balance.clean. After any correction, obtain a fresh balance.measure before further intervention. If the balance is unavailable, obtain an unavailable measurement so verification can finish unresolved. A clean pan and matching setpoint require a fresh weighing, not another correction. Evidence IDs are inspection/measurement id fields; use an empty list when no evidence exists. Arguments for custom capabilities are empty objects. temperature.set takes temperature_k in K.",
    "procedure": "Translate the diagnostic intent to exactly one supported capability and its arguments. Copy the supplied sample_id, expected_revision and request_id exactly. Respect prerequisites. Return an action proposal; never execute it. Custom capabilities take empty argument objects. Do not add units or sample IDs inside arguments unless the capability requests them.",
    "verification": "Check drying completion using the public mass criterion: total_mass_kg <= target.tare_mass_kg + target.dry_mass_kg + target.residual_limit_kg. The tare is part of the measured total, so add it to the threshold. A measurement is fresh when captured_at_step > last_mutation_step and revision equals observations.revision. If a valid fresh measurement of the correct sample meets the mass criterion, return supported and cite its id. If a valid fresh measurement exceeds the threshold, return contradicted. If the measurement is missing, suspect, unavailable, stale, or for another sample, return unresolved and explain the missing evidence. Do not demand an additional measurement when the supplied valid measurement already establishes the criterion. Never assign the official score.",
    "deliberation": "Resolve routing ambiguity using the supplied summary and allowed routes. Choose exactly one specialist; do not execute tools.",
}


class RoleBackend(Protocol):
    backend_id: str
    is_llm: bool

    def invoke(self, role: str, prompt: str, payload: dict, schema: type[BaseModel]) -> dict: ...


def projection(role: str, state: dict, request_id: str) -> dict:
    if role == "diagnostic":
        return {"target": state["target"], "observations": state["observations"],
            "capabilities": state["capabilities"], "previous_hypotheses": state.get("hypotheses", []),
            "last_error": state.get("last_error")}
    if role == "procedure":
        return {"intent": state["intent"], "sample_id": state["sample_id"],
            "target": state["target"],
            "expected_revision": state["observation_revision"], "request_id": request_id,
            "capabilities": state["capabilities"]}
    if role == "verification":
        return {"target": state["target"], "observations": state["observations"],
            "hypotheses": state.get("hypotheses", [])}
    raise ValueError(role)


class ScriptedRoles:
    backend_id = "scripted-test-double-v1"
    is_llm = False

    def invoke(self, role: str, prompt: str, payload: dict, schema: type[BaseModel]) -> dict:
        if role == "procedure":
            intent = Intent.model_validate(payload["intent"])
            record = ActionProposal(sample_id=payload["sample_id"], expected_revision=payload["expected_revision"],
                request_id=payload["request_id"], capability_id=intent.capability_id,
                arguments=intent.arguments, postcondition=intent.postcondition)
        elif role == "verification":
            obs, target = payload["observations"], payload["target"]
            m = obs.get("measurement")
            fresh = bool(m and m["quality"] == "valid" and m["sample_id"] == target["sample_id"]
                and m["revision"] == obs["revision"] and m["captured_at_step"] > obs["last_mutation_step"])
            if not fresh:
                record = VerificationReport(status="unresolved", evidence_ids=[m["id"]] if m else [],
                    missing_evidence=["reliable fresh sample weighing"])
            else:
                threshold = target["tare_mass_kg"] + target["dry_mass_kg"] + target["residual_limit_kg"]
                record = VerificationReport(status="supported" if m["total_mass_kg"] <= threshold else "contradicted",
                    evidence_ids=[m["id"]], missing_evidence=[])
        elif role == "diagnostic":
            obs, target = payload["observations"], payload["target"]
            inspection = obs.get("inspection")
            cause, evidence, args = "undetermined", [], {}
            if obs["last_mutation_step"] and not obs.get("measurement"):
                cap, cause = "balance.measure", "correction_requires_verification"
            elif not inspection:
                cap = "workflow.inspect"
            else:
                evidence = [inspection["id"]]
                check = inspection["balance_check"]
                setpoint = inspection["native"]["observations"]["temperature.set_k"]["value"]
                if not check["available"]:
                    cap, cause = "balance.measure", "missing_evidence"
                elif check["pan"] == "contaminated":
                    cap, cause = "balance.clean", "pan_contamination"
                elif abs(setpoint - target["temperature_k"]) > 0.5:
                    cap, cause = "temperature.set", "process_mismatch"
                    args = {"temperature_k": target["temperature_k"]}
                else:
                    cap, cause = "balance.measure", "none"
            previous = payload["previous_hypotheses"]
            hypotheses = ([Hypothesis.model_validate(h) for h in previous]
                if cause == "correction_requires_verification" and previous else
                [Hypothesis(cause=cause, supporting_ids=evidence, contradicting_ids=[])])
            record = HypothesisSet(hypotheses=hypotheses,
                intent=Intent(capability_id=cap, arguments=args,
                    discriminating_observation="Compare observed setpoint/pan quality and a fresh sample mass.",
                    postcondition="Obtain sufficient evidence for a fresh outcome check."))
        elif role == "deliberation":
            route = "prepare_action" if payload["summary"]["has_intent"] else "verify" if payload["summary"]["has_measurement"] else "diagnose"
            record = RouteDecision(backend_id=self.backend_id, choice=route)
        else:
            raise ValueError(role)
        return schema.model_validate(record.model_dump()).model_dump()
