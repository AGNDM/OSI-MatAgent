"""Additional authored sequential fixtures grounded in ChemLab native runtimes.

These are private evaluator specifications, not ChemLab-provided episodes or
recorded laboratory incidents.  A paired fixture shares its complete initial
public view with its control; faults/events are introduced only during a later
native wait.  Parameter variants share a workflow template, not independence.
"""
from __future__ import annotations

import copy
import math

from stepwise_bench.tasks import (
    _case, _condition, _finish_node, _invoke, _node, _read_node,
)


def _wait(checkpoint, duration, conditions=(), *, depends_on=(), when_conditions=()):
    return _node(checkpoint, "monitoring", {"kind": "wait", "dt_s": float(duration)},
                 conditions, depends_on=depends_on, kind="wait", min_dt_s=duration,
                 when_conditions=when_conditions)


def _fault(name):
    return dict(_invoke("fault.inject", fault=name, active=True), actor="evaluator")


def _annotate(case, variant, condition, *, pair=None, observability="visible",
              timing="initial", recovery="setup", parameter=None):
    case["expansion"] = {
        "template_id": case["workflow_id"], "variant_id": variant,
        "condition": condition, "pair_id": pair,
        "difficulty_axes": {
            "checkpoint_count": len(case["checkpoints"]) + 1,
            "observability": observability, "event_timing": timing,
            "response_requirement": recovery,
            "parameterization": parameter,
        },
    }
    return case


def _escalate(checkpoint, depends_on, evidence):
    return _node(checkpoint, "completion", {"kind": "escalate"},
                 depends_on=depends_on, kind="escalate", when_conditions=evidence)


def build_new_workflows(seed=42):
    """Construct 48 cases: eight native workflow templates, six variants each."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    cases = []

    # 1. Access mechanics are time dependent; command acceptance is distinct
    # from a door or latch physically reaching its requested position.
    for initial in ("secured", "outer_open", "both_open"):
        for cover in (False, True):
            variant = f"{initial}-cover-{int(cover)}"
            setup = [
                _invoke("door.outer", open=True), {"kind": "wait", "dt_s": 2.0},
                _invoke("door.latch", open=True), {"kind": "wait", "dt_s": 1.0},
                _invoke("door.inner", open=True), {"kind": "wait", "dt_s": 2.0},
                _invoke("rh_sensor.cover", on=not cover),
            ]
            if initial != "both_open":
                setup += [_invoke("door.inner", open=False), {"kind": "wait", "dt_s": 2.0},
                          _invoke("door.latch", open=False), {"kind": "wait", "dt_s": 1.0}]
            if initial == "secured":
                setup += [_invoke("door.outer", open=False), {"kind": "wait", "dt_s": 2.0}]
            nodes, predecessor = [], []
            if initial == "secured":
                nodes += [
                    _node("outer_open", "access", _invoke("door.outer", open=True),
                          [_condition("door.outer_state", "opening")]),
                    _wait("outer_ready", 2, [_condition("door.outer_state", "open")],
                          depends_on=["outer_open"]),
                ]
                predecessor = ["outer_ready"]
            if initial != "both_open":
                latch = _node("unlatch", "access", _invoke("door.latch", open=True),
                              depends_on=predecessor)
                # The latch has no public target slot.  Expanded evaluation
                # must check the public command arguments in addition to its
                # later observed angle; instantaneous angle alone is ambiguous.
                latch["accept"]["arguments"] = {"open": True}
                nodes += [latch,
                          _wait("latch_ready", 1,
                                [_condition("door.latch_angle_rad", math.pi / 2)],
                                depends_on=["unlatch"]),
                          _node("inner_open", "access", _invoke("door.inner", open=True),
                                [_condition("door.inner_state", "opening")],
                                depends_on=["latch_ready"]),
                          _wait("inner_ready", 2, [_condition("door.inner_state", "open")],
                                depends_on=["inner_open"])]
                predecessor = ["inner_ready"]
            latch_close = _node("latch_close", "securing", _invoke("door.latch", open=False),
                                depends_on=["inner_closed"])
            latch_close["accept"]["arguments"] = {"open": False}
            goals = [_condition("rh_sensor.cover_on", cover),
                     _condition("door.inner_state", "closed"),
                     _condition("door.outer_state", "closed"),
                     _condition("door.latched", True)]
            nodes += [
                _node("cover", "maintenance", _invoke("rh_sensor.cover", on=cover),
                      [_condition("rh_sensor.cover_on", cover)], depends_on=predecessor),
                _node("inner_close", "securing", _invoke("door.inner", open=False),
                      [_condition("door.inner_state", "closing")], depends_on=["cover"]),
                _wait("inner_closed", 2, [_condition("door.inner_state", "closed")],
                      depends_on=["inner_close"]), latch_close,
                _wait("latch_closed", 1, [_condition("door.latched", True)],
                      depends_on=["latch_close"]),
                _node("outer_close", "securing", _invoke("door.outer", open=False),
                      [_condition("door.outer_state", "closing")], depends_on=["latch_closed"]),
                _wait("secured", 2, goals, depends_on=["outer_close"]),
                _read_node("verify", ["rh_sensor.cover_on", "door.inner_state",
                                     "door.outer_state", "door.latched"], ["secured"]),
                _finish_node(["verify"]),
            ]
            case = _case("incubator_sensor_cover_access", variant, "eppendorf_cellxpertc170i",
                "Select the incubation chamber. With the chamber cool and empty, "
                f"set the humidity-sensor protective cover {'on' if cover else 'off'}, "
                "then leave both doors closed and the inner door latched. Observe the final states.",
                nodes, goals, seed=seed, setup_steps=setup,
                observation_ids=["power.enabled", "door.outer_state", "door.inner_state",
                                 "door.latch_angle_rad", "door.latched", "rh_sensor.cover_on"],
                manual=["No cultures or material transformations are involved.",
                        "Opening the inner door requires the outer door fully open and the latch fully raised.",
                        "Doors take about 1.5 simulated seconds to move; the latch takes about 0.5 seconds.",
                        "Cover access requires both doors open and a cool chamber. Secure the inner door before closing the outer door."])
            cases.append(_annotate(case, variant, initial, recovery="ordered_access",
                                   parameter={"cover_on": cover, "initial_access": initial}))

    # 2. A common startup test reveals either successful readiness or a native
    # cell fault.  Merely acknowledging power does not establish readiness.
    for pressure in (300000.0, 400000.0, 500000.0):
        for failed in (False, True):
            variant = f"pressure-{int(pressure)}-{'cell_fault' if failed else 'healthy'}"
            pair = f"hydrogen-startup-{int(pressure)}-seed-{seed}"
            ready = [_condition("state.current", "standby"), _condition("alarms.active", [])]
            fault = [_condition("state.current", "fault"),
                     _condition("alarms.active", "CELL_FAULT", operator="contains")]
            goals = [_condition("state.current", "running"), _condition("alarms.active", []),
                     _condition("pressure.set_pa", pressure)]
            nodes = [
                _node("power", "readiness", _invoke("power.set", on=True), [_condition("power.on", True)]),
                _node("pressure", "configuration", _invoke("settings.configure", pressure_set_pa=pressure),
                      [_condition("pressure.set_pa", pressure)], depends_on=["power"]),
                _wait("diagnostics", 900, [_condition("state.current", ["standby", "fault"], operator="in")],
                      depends_on=["pressure"]),
                _read_node("diagnostic_result", ["state.current", "alarms.active"], ["diagnostics"]),
                _node("start", "execution", _invoke("cycle.start"), [_condition("state.current", "running")],
                      depends_on=["diagnostic_result"], when_conditions=ready),
                _wait("running_check", 5, goals, depends_on=["start"]),
                _read_node("verify", ["state.current", "alarms.active", "pressure.set_pa"], ["running_check"]),
                _finish_node(["verify"]),
                _escalate("maintenance", ["diagnostic_result"], fault),
            ]
            case = _case("hydrogen_startup_diagnosis", variant, "peak_precisionhydrogen",
                "Select the hydrogen generator. Complete startup diagnostics and establish "
                f"the running delivery state with pressure setpoint {pressure:g} Pa gauge. "
                "If startup instead reports a persistent CELL_FAULT, request maintenance using that evidence.",
                nodes, goals, seed=seed,
                scheduled_events=[{"trigger_time_s": 2, "steps": [_fault("cell_failure")]}] if failed else [],
                observation_ids=["power.on", "state.current", "state.remaining_s", "pressure.set_pa", "alarms.active"],
                manual=["The generator has adequate deionized feed water and an available electrical supply.",
                        "Power-up includes water checks, pressurisation and a leak test. Wait for the actual public readiness state.",
                        "A CELL_FAULT cannot be repaired by changing the pressure or muting the alarm.",
                        "This goal verifies instrument state and setpoint; it does not certify gas purity or downstream chemistry."],
                escalation_expected=failed, escalation_conditions=fault)
            cases.append(_annotate(case, variant, "cell_failure" if failed else "healthy", pair=pair,
                                   observability="hidden_until_startup_test", timing="during_diagnostics",
                                   recovery="continue_or_escalate", parameter={"pressure_set_pa": pressure}))

    # 3. Six visible prerequisite combinations for a supported offline service.
    for installed, powered in ((True, True), (True, False), (False, True)):
        for stored in (False, True):
            variant = f"installed-{int(installed)}-power-{int(powered)}-store-{int(stored)}"
            setup = [_invoke("cap.remove")]
            if installed:
                setup += [_invoke("installation.set", mode="direct")]
            if powered:
                setup += [_invoke("power.set", on=True, voltage_v=24.0)]
            setup += [{"kind": "wait", "dt_s": 10.0}]
            nodes, prerequisites = [], []
            if powered:
                nodes.append(_node("power_off", "isolation", _invoke("power.set", on=False),
                                   [_condition("power.on", False)]))
                prerequisites.append("power_off")
            if installed:
                nodes.append(_node("remove", "isolation", _invoke("installation.remove"),
                                   [_condition("installation.current", False, path=["installed"])]))
                prerequisites.append("remove")
            goals = [_condition("power.on", False),
                     _condition("installation.current", False, path=["installed"]),
                     _condition("calibration.age_s", 0.0), _condition("cap.fitted", stored)]
            nodes.append(_node("service", "maintenance", _invoke("calibration.service"),
                               [_condition("calibration.age_s", 0.0)], depends_on=prerequisites))
            last = "service"
            if stored:
                nodes.append(_node("protect", "storage", _invoke("cap.fit"),
                                   [_condition("cap.fitted", True)], depends_on=["service"]))
                last = "protect"
            nodes += [_read_node("verify", ["power.on", "installation.current", "calibration.age_s", "cap.fitted"], [last]),
                      _finish_node(["verify"])]
            case = _case("dewpoint_offline_calibration", variant, "vaisala_dmt152",
                "Select the dewpoint transmitter and complete its offline calibration service. "
                "Leave it unpowered and removed from the process "
                f"with the transport cap {'fitted for storage' if stored else 'removed for subsequent bench work'}. "
                "Verify the calibration age has reset to zero.", nodes, goals, seed=seed, setup_steps=setup,
                observation_ids=["power.on", "installation.current", "calibration.age_s", "cap.fitted"],
                manual=["The service command abstracts completed factory calibration; it is not an online calibration procedure.",
                        "Calibration service requires power off and the transmitter removed from its process installation.",
                        "A transport cap may only be fitted after removal. No change in gas composition is requested."])
            cases.append(_annotate(case, variant, "visible_service_prerequisites", recovery="offline_service",
                                   parameter={"initially_installed": installed, "initially_powered": powered,
                                              "cap_on_at_finish": stored}))

    # 4. Public keylock state changes which access step is needed.  The
    # raised-access period must be followed by closure and re-engaging the lock.
    for initial_lock in ("closed", "limit", "off"):
        for height in (0.30, 0.55):
            variant = f"lock-{initial_lock}-height-{height:g}"
            setup = [_invoke("sash.set", height_m=0.0), _invoke("keylock.set", mode=initial_lock)]
            nodes, prerequisites = [], []
            needs_unlock = initial_lock == "closed" or (initial_lock == "limit" and height > 0.457)
            if needs_unlock:
                nodes.append(_node("unlock", "readiness", _invoke("keylock.set", mode="off"),
                                   [_condition("keylock.mode", "off")]))
                prerequisites = ["unlock"]
            nodes += [_node("raise", "access", _invoke("sash.set", height_m=height, hold=True),
                            [_condition("sash.height_m", height)], depends_on=prerequisites),
                      _read_node("access_evidence", ["sash.height_m"], ["raise"]),
                      _node("close", "securing", _invoke("sash.set", height_m=0.0),
                            [_condition("sash.height_m", 0.0)], depends_on=["access_evidence"]),
                      _node("lock", "securing", _invoke("keylock.set", mode="closed"),
                            [_condition("keylock.mode", "closed")], depends_on=["close"])]
            goals = [_condition("sash.height_m", 0.0), _condition("keylock.mode", "closed")]
            nodes += [_read_node("verify", ["sash.height_m", "keylock.mode"], ["lock"]), _finish_node(["verify"])]
            case = _case("hood_sash_access_resecure", variant, "esco_efa5uduvw",
                "Select the fume hood. For an empty-cabinet access check, establish and observe "
                f"a sash opening of {height:g} m; then return the sash fully closed and set "
                "the keylock to closed. Verify the final secured state.", nodes, goals, seed=seed, setup_steps=setup,
                observation_ids=["sash.height_m", "sash.held", "keylock.mode"],
                manual=["This is an empty-cabinet mechanical access test; containment performance is not being certified.",
                        "The closed keylock prevents raising the sash. The limit mode prevents raising above 0.457 m.",
                        "Above 0.457 m, hold=True maintains the opening; an unheld sash lowers during a subsequent wait.",
                        "The closed keylock can only be engaged with the sash fully closed."])
            cases.append(_annotate(case, variant, "visible_keylock", recovery="access_and_resecure",
                                   parameter={"initial_keylock": initial_lock, "access_height_m": height}))

    # 5. Native door opening pauses an existing furnace program.  Closing
    # resumes it; a new start is inappropriate while the program remains running.
    for temperature, hold in ((303.15, 3), (308.15, 5), (313.15, 7)):
        segment = {"temperature_K": temperature, "ramp_rate_K_s": 0.05, "hold_s": hold}
        for interrupted in (False, True):
            variant = f"target-{temperature:g}-{'door_interruption' if interrupted else 'healthy'}"
            pair = f"furnace-door-{temperature:g}-seed-{seed}"
            goals = [_condition("program.phase", "completed"), _condition("program.running", False),
                     _condition("program.hold_elapsed_s", hold, operator="ge"),
                     _condition("program.segments", [segment]), _condition("door.closed", True)]
            nodes = [
                _wait("monitor", 3, depends_on=[]),
                _read_node("inspect", ["door.closed", "program.phase", "program.running"], ["monitor"]),
                _node("resume", "recovery", _invoke("door.close"), [_condition("door.closed", True)],
                      depends_on=["inspect"], when_conditions=[_condition("door.closed", False),
                                                               _condition("program.phase", "paused_door")]),
                _node("complete", "monitoring", {"kind": "wait", "dt_s": 400.0}, goals,
                      depends_on=["inspect"], kind="wait", when_conditions=[_condition("door.closed", True)]),
                _read_node("verify", ["program.phase", "program.running", "program.hold_elapsed_s", "door.closed"], ["complete"]),
                _finish_node(["verify"]),
            ]
            case = _case("furnace_door_pause_recovery", variant, "nabertherm_l9",
                "Select the programmable furnace already running an empty-chamber program "
                f"at {temperature:g} K, ramp 0.05 K/s and hold {hold} s. Monitor for at least "
                "three simulated seconds, inspect its status, resolve any recoverable door "
                "pause, and verify that the original program completes.", nodes, goals, seed=seed,
                setup_steps=[_invoke("power.set", enabled=True), _invoke("door.close"),
                             _invoke("thermal.configure_program", segments=[segment]), _invoke("thermal.start_program")],
                scheduled_events=[{"trigger_time_s": 2, "steps": [_invoke("door.open")]}] if interrupted else [],
                observation_ids=["power.enabled", "door.closed", "program.phase", "program.running",
                                 "program.segments", "program.hold_elapsed_s"],
                manual=["This native runtime models an empty furnace; no chemical transformation is evaluated.",
                        "Opening the door pauses the still-running program and cuts heater permission; closing resumes it automatically.",
                        "Use observations after monitoring to determine whether an intervention is needed."])
            cases.append(_annotate(case, variant, "door_interruption" if interrupted else "healthy", pair=pair,
                                   observability="hidden_until_monitoring", timing="mid_program",
                                   recovery="resume_if_paused", parameter={"temperature_K": temperature, "hold_s": hold}))

    # 6. Standby is an operator interruption, not damage or a repairable sensor
    # error.  An intact setpoint must be retained when normal operation resumes.
    for temperature in (303.15, 313.15, 323.15):
        for interrupted in (False, True):
            variant = f"target-{temperature:g}-{'standby' if interrupted else 'healthy'}"
            pair = f"oven-standby-{temperature:g}-seed-{seed}"
            goals = [_condition("power.on", True), _condition("temperature.set_k", temperature),
                     _condition("fan.running", True), _condition("alarm.state", None)]
            nodes = [
                _wait("monitor", 3),
                _read_node("inspect", ["power.on", "indicator.standby", "temperature.set_k", "alarm.state"], ["monitor"]),
                _node("resume", "recovery", _invoke("power.set", on=True), [_condition("power.on", True)],
                      depends_on=["inspect"], when_conditions=[_condition("power.on", False)]),
                _wait("operating", 1, goals, depends_on=["inspect"], when_conditions=[_condition("power.on", True)]),
                _read_node("verify", ["power.on", "temperature.set_k", "fan.running", "alarm.state"], ["operating"]),
                _finish_node(["verify"]),
            ]
            case = _case("oven_standby_restoration", variant, "binder_fd056",
                "Select the drying oven already configured with a temperature setpoint "
                f"of {temperature:g} K. Monitor at least three simulated seconds, inspect "
                "the controller, restore normal powered operation if it has entered standby, "
                "and verify that the original setpoint is retained, the fan runs and no alarm is active.",
                nodes, goals, seed=seed,
                setup_steps=[_invoke("power.set", on=True), _invoke("door.set", angle_rad=0.0),
                             _invoke("temperature.set", temperature_k=temperature)],
                scheduled_events=[{"trigger_time_s": 2, "steps": [_invoke("power.set", on=False)]}] if interrupted else [],
                observation_ids=["power.on", "indicator.standby", "temperature.set_k", "door.open", "fan.running", "alarm.state"],
                manual=["The chamber is empty and the electrical mains remain available; no sample drying outcome is modeled.",
                        "The controller may enter standby during monitoring. Standby preserves its temperature setpoint.",
                        "The requested outcome concerns restored operation and controls, not attainment of thermal equilibrium."])
            cases.append(_annotate(case, variant, "operator_standby" if interrupted else "healthy", pair=pair,
                                   observability="hidden_until_monitoring", timing="mid_operation",
                                   recovery="restore_if_standby", parameter={"temperature_set_k": temperature}))

    # 7. The same public speed test either stays operational or returns native
    # Err.4.  This does not infer stir-bar coupling or liquid properties.
    for rpm in (200, 400, 600):
        speed = rpm * 2 * math.pi / 60
        for failed in (False, True):
            variant = f"speed-{rpm}-{'blocked' if failed else 'healthy'}"
            pair = f"hotplate-motor-{rpm}-seed-{seed}"
            goals = [_condition("stirring.on", True), _condition("stirring.set_rad_s", speed),
                     _condition("error.code", None)]
            evidence = [_condition("error.code", 4), _condition("stirring.on", False)]
            nodes = [
                _node("speed", "configuration", _invoke("stirring.set_speed", speed_rad_s=speed),
                      [_condition("stirring.set_rad_s", speed)]),
                _node("start", "diagnosis", _invoke("stirring.set", on=True), [_condition("stirring.on", True)],
                      depends_on=["speed"]),
                _wait("test", 3, depends_on=["start"]),
                _read_node("observe", ["stirring.on", "stirring.set_rad_s", "error.code"], ["test"]),
                _finish_node(["observe"], when_conditions=goals),
                _escalate("maintenance", ["observe"], evidence),
            ]
            case = _case("hotplate_motor_diagnostic", variant, "ika_plate_rctdigital",
                f"Select the hotplate stirrer. Test stirring at {speed:.15g} rad/s ({rpm} rpm) "
                "for at least three simulated seconds and verify normal operation; if the "
                "test instead reports Err.4 with stirring shut down, request motor maintenance.",
                nodes, goals, seed=seed, setup_steps=[_invoke("power.set", on=True)],
                scheduled_events=[{"trigger_time_s": 2, "steps": [_fault("motor_blocked")]}] if failed else [],
                observation_ids=["power.on", "lock.on", "stirring.on", "stirring.set_rad_s", "error.code"],
                manual=["The controls are unlocked and heating is not requested. There is no sample in this motor-function test.",
                        "A successful stirring command is not proof of continued operation after simulated time advances.",
                        "Err.4 represents the native blocked-motor fault. Persistent blockage requires maintenance, not an alarm acknowledgement."],
                escalation_expected=failed, escalation_conditions=evidence)
            cases.append(_annotate(case, variant, "motor_blocked" if failed else "healthy", pair=pair,
                                   observability="hidden_until_motor_test", timing="during_test",
                                   recovery="finish_or_escalate", parameter={"speed_rad_s": speed, "rpm": rpm}))

    # 8. Waiting through startup purge and checking diagnostic errors separates
    # a live measurement from a frozen or failed output, without true-gas access.
    for voltage in (21.0, 24.0, 28.0):
        for failed in (False, True):
            variant = f"supply-{voltage:g}-{'sensor_damage' if failed else 'healthy'}"
            pair = f"dewpoint-startup-{voltage:g}-seed-{seed}"
            goals = [_condition("reading.frozen", False), _condition("reading.tdf_k", None, operator="ne"),
                     _condition("diagnostics.errors", [])]
            evidence = [_condition("diagnostics.errors", "T meas error", operator="contains"),
                        _condition("diagnostics.errors", "F meas error", operator="contains"),
                        _condition("reading.tdf_k", None)]
            nodes = [
                _node("power", "readiness", _invoke("power.set", on=True, voltage_v=voltage),
                      [_condition("power.on", True), _condition("supply.voltage_v", voltage)]),
                _wait("startup", 1200, [_condition("reading.frozen", False)], depends_on=["power"]),
                _read_node("diagnose", ["reading.frozen", "reading.tdf_k", "diagnostics.errors"], ["startup"]),
                _finish_node(["diagnose"], when_conditions=goals),
                _escalate("maintenance", ["diagnose"], evidence),
            ]
            process = dict(_invoke("process.set", gas="nitrogen", water_umol_mol=10.0,
                                   pressure_pa=101325.0, temperature_k=293.15), actor="evaluator")
            case = _case("dewpoint_startup_measurement_diagnosis", variant, "vaisala_dmt152",
                "Select the dewpoint transmitter already installed in its supplied nitrogen line. "
                f"Enable its {voltage:g} V supply, wait through startup purge, and verify an "
                "unfrozen non-null dewpoint reading without errors. If the instrument instead "
                "reports persistent T meas error and F meas error with no reading, request service.",
                nodes, goals, seed=seed,
                setup_steps=[_invoke("cap.remove"), _invoke("installation.set", mode="direct"), process],
                scheduled_events=[{"trigger_time_s": 2, "steps": [_fault("sensor_damage")]}] if failed else [],
                observation_ids=["power.on", "supply.voltage_v", "state.current", "state.remaining_s",
                                 "reading.frozen", "reading.tdf_k", "diagnostics.errors"],
                manual=["The transport cap has been removed and the process installation is valid.",
                        "Power-up purge freezes output; a frozen or null startup value is not yet a usable measurement.",
                        "The supplied voltage is within the permitted range for the configured outputs.",
                        "T meas error together with F meas error indicates sensor damage that public controls cannot repair.",
                        "This checks instrument readiness and diagnostic trust; it does not independently validate the true gas dewpoint."],
                escalation_expected=failed, escalation_conditions=evidence)
            cases.append(_annotate(case, variant, "sensor_damage" if failed else "healthy", pair=pair,
                                   observability="hidden_until_startup", timing="during_purge",
                                   recovery="trust_or_escalate", parameter={"supply_voltage_v": voltage}))

    return copy.deepcopy(cases)
