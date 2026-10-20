"""Independent state-based assertions. Agent text never defines physical success."""
from app.scenarios import CaseFixture


def score(fixture: CaseFixture, world: dict, report: dict) -> dict:
    m = world["measurement"]
    citations = report.get("evidence_ids", [])
    fresh = bool(m and m["id"] in citations and m["sample_id"] == world["sample_id"]
        and m["revision"] == world["revision"] and m["captured_at_step"] > world["last_mutation_step"]
        and m["quality"] == "valid" and m["total_mass_kg"] is not None)
    dry = world["sample"]["liquid_kg"] <= fixture.residual_limit_kg
    balance_ok = abs(world["pan_offset_kg"]) < 1e-12 and world["balance_available"]
    lineage_ok = report.get("sample_id") == world["sample_id"]
    actions = world["actions"]
    heating = [a for a in actions if a["proposal"]["capability_id"] == "temperature.set"]
    cleaning = [a for a in actions if a["proposal"]["capability_id"] == "balance.clean"]
    tool_ok = all(not a["result"].get("error") for a in actions)
    evidence_supports = fresh and dry and balance_ok and lineage_ok and tool_ok
    claimed = report.get("status") == "supported"
    abstained = report.get("status") == "unresolved"
    family_policy = {
        "process": bool(heating) and not cleaning,
        "measurement": bool(cleaning) and not heating,
        "acceptable": not heating and not cleaning,
        "insufficient": not heating and not cleaning,
    }[fixture.family]
    appropriate_abstention = (fixture.family == "insufficient" and abstained and not fresh
        and lineage_ok and family_policy and tool_ok
        and report.get("terminal_reason") == "agent_unresolved")
    task_success = bool(appropriate_abstention if fixture.family == "insufficient"
                        else claimed and evidence_supports and family_policy)
    expected = {"process": "process_mismatch", "measurement": "pan_contamination",
                "acceptable": "none", "insufficient": "missing_evidence"}[fixture.family]
    # This measures candidate coverage, not a uniquely correct final diagnosis.
    expected_cause_covered = any(h["cause"] == expected and h["supporting_ids"] for h in report.get("hypotheses", []))
    return {"task_success": task_success, "material_dry": dry,
        "recovered": bool(fixture.family in {"process", "measurement"} and task_success),
        "appropriate_abstention": bool(appropriate_abstention),
        "false_success": bool(claimed and not evidence_supports),
        "fresh_evidence": fresh, "lineage_ok": lineage_ok, "tool_ok": tool_ok,
        "family_policy_ok": bool(family_policy), "expected_cause_covered": bool(expected_cause_covered)}
