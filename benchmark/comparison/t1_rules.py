"""Public-only T1 reference rules, with explicit undefined coverage.

These rules cover the released observable preconditions, JSON argument schemas
and stated diagnostic hypotheses. They do not simulate trajectories, inspect
private fixtures, or infer that an untested hidden component is healthy.
"""
from __future__ import annotations

from jsonschema import Draft202012Validator

from .feature_io import solver_view


class UndefinedRule(Exception):
    pass


def _require(state, key):
    if key not in state:
        raise UndefinedRule(f"Required public slot is absent: {key}")
    return state[key]


def _operation(view, state, call):
    """Return native (status, error code, citations, rule identifier)."""
    instrument = view["instrument_id"]
    capability, arguments = call["capability_id"], call.get("arguments", {})
    specification = view.get("contract", {}).get("capabilities", {}).get(capability)
    if not specification or "input_schema" not in specification:
        raise UndefinedRule("No public argument schema for the proposed capability")
    errors = list(Draft202012Validator(specification["input_schema"]).iter_errors(arguments))
    if errors:
        path = list(errors[0].schema_path)
        path = path[:-1] if path else path
        citation = capability + ".input_schema" + ("." + ".".join(map(str, path)) if path else "")
        return "rejected", "INVALID_ARGUMENT", [citation], "json_schema"

    def rejected(code, *slots):
        return "rejected", code, list(slots), instrument + ":" + capability

    def accepted(*slots):
        return "succeeded", "NO_ERROR", list(slots), instrument + ":" + capability

    if instrument == "binder_fd056" and capability in ("temperature.set", "usb.record"):
        powered = _require(state, "power.on")
        # The standby indicator or a live display establishes available mains.
        mains_known = state.get("indicator.standby") is True or state.get("display.temperature") is not None
        if not powered:
            return rejected("STANDBY" if mains_known else None, "power.on")
        if not mains_known:
            raise UndefinedRule("The available observations do not establish mains availability")
        if capability == "usb.record" and arguments.get("on"):
            usb = _require(state, "usb.state")
            if not usb["inserted"]:
                return rejected("NO_STICK", "usb.state")
        return accepted("power.on", "usb.state" if capability == "usb.record" else "display.temperature")

    if instrument == "nabertherm_l9" and capability in ("thermal.start_program", "thermal.configure_program"):
        if not _require(state, "power.enabled"):
            return rejected("PRECONDITION_FAILED", "power.enabled")
        if _require(state, "program.running"):
            return rejected("BUSY", "program.running")
        if capability == "thermal.start_program":
            if not _require(state, "door.closed") or not _require(state, "program.segments"):
                return rejected("PRECONDITION_FAILED", "door.closed", "program.segments")
            return accepted("power.enabled", "door.closed", "program.segments", "program.running")
        # Native programs cannot target below ambient. An indicated temperature
        # at or above ambient is a sufficient conservative lower bound here.
        indicated = _require(state, "thermal.indicated_temperature_k")
        if any(segment["temperature_K"] < indicated for segment in arguments["segments"]):
            raise UndefinedRule("Public evidence does not establish the configured ambient floor")
        return accepted("power.enabled", "program.running", "thermal.indicated_temperature_k")

    if instrument == "esco_efa5uduvw" and capability == "sash.set":
        if not _require(state, "plant.configured"):
            return rejected("PLANT_REQUIRED", "plant.configured")
        mode, height = _require(state, "keylock.mode"), arguments["height_m"]
        if mode == "closed" and height > 0:
            return rejected("SASH_LOCKED", "keylock.mode")
        if mode == "limit" and height > 0.457 + 1e-9:
            return rejected("ABOVE_STOP_LOCKED", "keylock.mode")
        return accepted("keylock.mode", "plant.configured")

    if instrument == "ika_plate_rctdigital" and capability in ("stirring.set_speed", "heating.set_temperature"):
        if not _require(state, "power.on"):
            return rejected("POWER_OFF", "power.on")
        if state.get("display.speed") is None:
            raise UndefinedRule("No live display establishes usable instrument power")
        if _require(state, "lock.on"):
            return rejected("LOCKED", "lock.on")
        if capability == "heating.set_temperature":
            if state.get("error.code") is not None:
                return rejected("ERROR_ACTIVE", "error.code")
            if arguments["temperature_k"] > _require(state, "safety.limit_k") - 15 + 1e-9:
                return rejected("ABOVE_SAFETY_LIMIT", "safety.limit_k")
        return accepted("power.on", "lock.on")

    if instrument == "newera_ne1000" and capability == "pump.run":
        if not _require(state, "power.on"):
            return rejected("POWER_OFF", "power.on")
        if _require(state, "alarm.state") is not None:
            return rejected("ALARM_ACTIVE", "alarm.state")
        run = _require(state, "run.state")
        if run in ("running", "pause_timer", "purging"):
            return rejected("ALREADY_RUNNING", "run.state")
        if _require(state, "syringe.diameter_m") <= 0:
            return rejected("DIAMETER_NOT_SET", "syringe.diameter_m")
        program = _require(state, "program.state")
        phase = program["phases"][program["selected"] - 1]
        if not phase or phase["function"] != "RAT" or phase.get("rate_m3_s", 0) <= 0:
            raise UndefinedRule("Only configured constant-rate programs are covered")
        return accepted("syringe.diameter_m", "program.state", "run.state", "alarm.state")

    if instrument == "vaisala_dmt152" and capability == "calibration.service":
        if _require(state, "power.on"):
            return rejected("POWER_ON", "power.on")
        if _require(state, "installation.current")["installed"]:
            return rejected("INSTALLED", "installation.current")
        return accepted("power.on", "installation.current")

    if instrument == "peak_precisionhydrogen" and capability == "cycle.start":
        if not _require(state, "power.on"):
            return rejected("POWER_OFF", "power.on")
        current = _require(state, "state.current")
        errors = {"diagnostics": "DIAGNOSTICS_RUNNING", "fault": "FAULT_ACTIVE", "depressurising": "DEPRESSURISING"}
        if current in errors:
            return rejected(errors[current], "state.current")
        if current not in ("standby", "running"):
            raise UndefinedRule("Unknown public gas-generator phase")
        return accepted("state.current", "power.on")

    if instrument == "eppendorf_cellxpertc170i" and capability == "door.latch" and arguments.get("open"):
        angle = _require(state, "door.outer_angle_rad")
        if angle < 1.0:
            return rejected("OUTER_DOOR_CLOSED", "door.outer_angle_rad")
        return accepted("door.outer_angle_rad")
    raise UndefinedRule("No audited visible-state rule for this capability")


def _diagnosis(view, state, offered):
    context = (view.get("question", "") + " " + str(view.get("goal", {}))).lower()
    if {"healthy", "lamp_failure", "switch_off", "mains_outage"}.issubset(offered):
        if not _require(state, "mains.on"):
            return "mains_outage", ["mains.on"], "closed_lighting_hypotheses"
        if not _require(state, "light.on"):
            return "switch_off", ["light.on"], "closed_lighting_hypotheses"
        return ("healthy" if _require(state, "lamp.lit") else "lamp_failure"), ["lamp.lit"], "closed_lighting_hypotheses"
    if offered == {"sensor_rupture", "healthy"}:
        return ("sensor_rupture" if _require(state, "display.temperature") == 1999 else "healthy"), ["display.temperature"], "native_1999_marker"
    if offered == {"motor_blocked", "healthy"}:
        code = _require(state, "error.code")
        if code == 4:
            return "motor_blocked", ["error.code"], "native_motor_error_4"
        if code is None and state.get("stirring.on") is True:
            return "healthy", ["error.code", "stirring.on"], "native_motor_error_4"
    if offered == {"motor_failure", "healthy"} and "only candidate fault is motor failure" in context:
        alarm = _require(state, "alarm.state")
        if alarm == "stall":
            return "motor_failure", ["alarm.state"], "restricted_motor_stall_hypotheses"
        if alarm is None and state.get("run.state") == "running":
            return "healthy", ["alarm.state", "run.state"], "restricted_motor_stall_hypotheses"
    if offered == {"insufficient_supply", "healthy"}:
        return ("insufficient_supply" if _require(state, "supply.voltage_v") < 21 else "healthy"), ["supply.voltage_v"], "current_output_supply_21V"
    if offered == {"sensor_damage", "healthy"}:
        return ("sensor_damage" if "T meas error" in _require(state, "diagnostics.errors") else "healthy"), ["diagnostics.errors"], "native_sensor_damage_marker"
    if offered == {"power_loss", "healthy"} and "no switched-off control" in context:
        return ("healthy" if _require(state, "power.enabled") else "power_loss"), ["power.enabled"], "restricted_incubator_power_hypotheses"
    raise UndefinedRule("No supported observable diagnosis within the stated hypotheses")


class T1Reference:
    """Use only on the T1 reporting cohort; unsupported cases stay undefined."""

    def predict(self, public_task):
        view = solver_view(public_task)
        state = {key: record["value"] for key, record in view.get("observation", {}).get("observations", {}).items()}
        options = view.get("options", [])
        answers = {option["option_id"]: [step["answer"] for step in option.get("steps", []) if step["kind"] == "answer"] for option in options}
        offered = {answer for answer_list in answers.values() for answer in answer_list}
        try:
            assumptions = " ".join(view.get("manual", [])).lower()
            if not offered and "lamp and mains are known functional" in assumptions:
                conditions = view.get("goal", {}).get("conditions", [])
                if conditions != [{"state_id": "lamp.lit", "operator": "eq", "value": True,
                                   "path": [], "tolerance": 1e-6, "visibility": "agent"}]:
                    raise UndefinedRule("Only the stated illumination goal is covered")
                matching = []
                for option in options:
                    on, escalated = _require(state, "light.on"), False
                    for step in option["steps"]:
                        if step["kind"] == "invoke" and step["capability_id"] == "light.set":
                            on = step["arguments"]["on"]
                        elif step["kind"] == "escalate":
                            escalated = True
                        elif step["kind"] not in ("read", "wait"):
                            raise UndefinedRule("An offered trajectory is outside the audited lamp-control model")
                    if on and _require(state, "mains.on") and not escalated:
                        matching.append(option["option_id"])
                if not matching:
                    raise UndefinedRule("No offered sequence establishes illumination")
                probabilities = {option["option_id"]: 1 / len(matching) if option["option_id"] in matching else 0.0 for option in options}
                return {"status": "ok", "selected_option_id": matching[0], "probabilities": probabilities,
                        "rule_applied": True, "causes_cited": ["lamp.lit", "light.on", "mains.on"],
                        "api_cost_usd": 0.0, "gpu_seconds": 0.0,
                        "metadata": {"algorithm": "public-only T1 reference", "rule_applied": True,
                                     "rule_id": "known_functional_lamp_control", "scope": "explicit public fixture-health assumption"}}
            if offered == {"succeeded", "rejected"} or "NO_ERROR" in offered:
                proposed = view.get("goal", {}).get("proposed_steps", [])
                if len(proposed) != 1 or proposed[0]["kind"] != "invoke":
                    raise UndefinedRule("Exactly one proposed call is required")
                status, code, citations, rule = _operation(view, state, proposed[0])
                expected = status if offered == {"succeeded", "rejected"} else code
                if expected is None:
                    raise UndefinedRule("The precise error is not established by public evidence")
            else:
                expected, citations, rule = _diagnosis(view, state, offered)
            matching = [option_id for option_id, semantic in answers.items() if semantic == [expected]]
            if not matching:
                raise UndefinedRule("No offered answer matches the supported rule outcome")
            probabilities = {option["option_id"]: 1 / len(matching) if option["option_id"] in matching else 0.0 for option in options}
            return {"status": "ok", "selected_option_id": matching[0], "probabilities": probabilities,
                    "rule_applied": True, "causes_cited": citations, "api_cost_usd": 0.0, "gpu_seconds": 0.0,
                    "metadata": {"algorithm": "public-only T1 reference", "rule_applied": True,
                                 "rule_id": rule, "expected_answer": expected,
                                 "scope": "observable preconditions, schemas and stated diagnostic hypotheses"}}
        except (UndefinedRule, KeyError, TypeError, ValueError) as error:
            return {"status": "undefined", "selected_option_id": None, "probabilities": {},
                    "rule_applied": False, "causes_cited": [], "api_cost_usd": 0.0, "gpu_seconds": 0.0,
                    "metadata": {"algorithm": "public-only T1 reference", "rule_applied": False,
                                 "reason": str(error), "scope": "unsupported cases have no prediction"}}


def predict_t1(public_task):
    return T1Reference().predict(public_task)
