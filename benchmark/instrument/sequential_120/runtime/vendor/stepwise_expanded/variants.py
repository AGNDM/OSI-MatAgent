"""Seventy-two private sequential cases for six native workflow templates.

The scenarios are authored from ChemLab's implemented commands and states.
They are not ChemLab-provided question/answer examples.  Numeric parameters,
readiness conditions and branch timing vary; changing ``seed`` alone is never
used to inflate the case count.  Only the trusted evaluator loads these specs.
"""
from __future__ import annotations

import copy
import hashlib
import math

from stepwise_bench.tasks import (
    _case, _condition, _finish_node, _invoke, _node, _read_node,
)


def _private_condition(state_id, value, *, path=(), tolerance=1e-12):
    """Verify a declared setup quantity that the native public read omits."""
    condition = _condition(state_id, value, path=path, tolerance=tolerance)
    condition["visibility"] = "evaluator"
    return condition


def _expanded(case, variant_id, *, condition="healthy", pair_id=None,
              difficulty_axes=None, fixture_conditions=()):
    family = case["workflow_id"]
    key = f"stepwise-expanded-v1:{family}:{variant_id}:seed-{case['seed']}"
    identity = hashlib.sha256(key.encode()).hexdigest()[:24]
    case.update(task_id=identity, case_id=identity,
                group_id=f"stepwise-expanded-v1:{family}:{pair_id or variant_id}")
    case["fixture_conditions"] = list(fixture_conditions)
    case["expansion"] = {
        "template_id": family, "variant_id": variant_id,
        "condition": condition, "pair_id": pair_id,
        "difficulty_axes": dict(difficulty_axes or {}),
        "source": "authored_native_workflow_variant",
        "difficulty_is_empirically_validated": False,
    }
    return case


def _furnace_cases(seed):
    profiles = [
        [(303.15, .05, 2)], [(308.15, .1, 4)], [(313.15, .1, 6)],
        [(303.15, .05, 2), (308.15, .05, 3)],
        [(308.15, .1, 4), (318.15, .1, 5)],
        [(303.15, .1, 2), (308.15, .1, 3), (313.15, .1, 4)],
    ]
    cases = []
    for index, profile in enumerate(profiles):
        segments = [{"temperature_K": target, "ramp_rate_K_s": ramp, "hold_s": hold}
                    for target, ramp, hold in profile]
        previous, nominal_duration = 298.15, 0.0
        for target, ramp, hold in profile:
            nominal_duration += abs(target - previous) / ramp + hold
            previous = target
        wait_s = math.ceil(nominal_duration + 120)
        goals = [_condition("program.phase", "completed"),
                 _condition("program.running", False),
                 _condition("program.segment_index", len(segments)),
                 _condition("program.hold_elapsed_s", segments[-1]["hold_s"], operator="ge"),
                 _condition("program.segments", segments)]
        for ready in (False, True):
            setup, nodes, prerequisites = [], [], []
            if ready:
                setup = [_invoke("power.set", enabled=True), _invoke("door.close")]
            else:
                setup = [_invoke("door.open")]
                nodes.extend([
                    _node("power", "readiness", _invoke("power.set", enabled=True),
                          [_condition("power.enabled", True)]),
                    _node("door", "readiness", _invoke("door.close"),
                          [_condition("door.closed", True)]),
                ])
                prerequisites = ["power", "door"]
            nodes.extend([
                _node("configure", "configuration", _invoke("thermal.configure_program", segments=segments),
                      [_condition("program.segments", segments)], depends_on=[] if ready else ["power"]),
                _node("start", "execution", _invoke("thermal.start_program"),
                      [_condition("program.running", True)], depends_on=prerequisites + ["configure"]),
                _node("complete_program", "monitoring", {"kind": "wait", "dt_s": wait_s},
                      goals, depends_on=["start"], kind="wait"),
                _read_node("verify", ["program.phase", "program.running", "program.segment_index",
                                      "program.hold_elapsed_s"], ["complete_program"]),
                _finish_node(["verify"]),
            ])
            program_text = "; then ".join(
                f"{target:g} K at {ramp:g} K/s with a {hold:g}-second hold"
                for target, ramp, hold in profile)
            state_text = ("Power is on and the door is closed." if ready else
                          "Power is off and the door is open.")
            variant = f"profile-{index + 1}-{'ready' if ready else 'unprepared'}"
            case = _case("furnace_program", variant, "nabertherm_l9",
                "Select the programmable furnace. Complete this empty-chamber temperature program, "
                f"in the stated segment order: {program_text}. Observe and verify completion of all segments.",
                nodes, goals, seed=seed, setup_steps=setup,
                observation_ids=["power.enabled", "door.closed", "thermal.indicated_temperature_k",
                                 "program.segments", "program.phase", "program.running",
                                 "program.segment_index", "program.hold_elapsed_s"],
                manual=[state_text,
                        "The chamber is empty; sample transformations and chemical quality are not modeled.",
                        "Power is required for configuration; starting also requires the door closed.",
                        "The hold clock starts after reaching the segment target. An accepted start does not establish completion.",
                        "A completed program reports segment_index equal to its number of segments."])
            case["budget"]["max_sim_time_s"] = max(1200.0, 2 * wait_s)
            cases.append(_expanded(case, variant, difficulty_axes={
                "program_segments": len(segments), "readiness_actions": 0 if ready else 2,
                "feedback_branching": False, "verification_required": True},
                fixture_conditions=[_condition("power.enabled", ready), _condition("door.closed", ready)]))
    return cases


def _oven_cases(seed):
    cases = []
    for interval_min in (1, 2, 5):
        for readiness, (power_on, inserted) in enumerate(((False, False), (True, False),
                                                        (False, True), (True, True))):
            rows = 1 + readiness % 3
            interval_s = 60.0 * interval_min
            wait_s = interval_s * (rows - 1) + 1.0
            goals = [_condition("usb.state", True, path=["inserted"]),
                     _condition("usb.state", True, path=["recording"]),
                     _condition("usb.state", interval_s, path=["interval_s"]),
                     _condition("usb.state", rows, operator="ge", path=["rows"])]
            setup, nodes, dependencies = [], [], []
            for checkpoint, already, action, conditions in (
                ("power", power_on, _invoke("power.set", on=True), [_condition("power.on", True)]),
                ("media", inserted, _invoke("usb.set_stick", inserted=True),
                 [_condition("usb.state", True, path=["inserted"])]),
            ):
                if already:
                    setup.append(action)
                else:
                    nodes.append(_node(checkpoint, "readiness", action, conditions))
                    dependencies.append(checkpoint)
            nodes.extend([
                _node("record", "configuration", _invoke("usb.record", on=True, interval_min=interval_min),
                      goals[:3], depends_on=dependencies),
                _node("collect_rows", "monitoring", {"kind": "wait", "dt_s": wait_s},
                      goals, depends_on=["record"], kind="wait"),
                _read_node("verify", ["usb.state"], ["collect_rows"]),
                _finish_node(["verify"]),
            ])
            variant = f"interval-{interval_min}-readiness-{readiness}"
            case = _case("oven_usb_recording", variant, "binder_fd056",
                f"Select the drying oven with USB logging. Record at a {interval_min}-minute interval "
                f"and verify at least {rows} recorded row(s), with the USB stick inserted and recording still enabled.",
                nodes, goals, seed=seed, setup_steps=setup,
                observation_ids=["power.on", "usb.state"],
                manual=[f"The oven starts {'powered on' if power_on else 'in standby'}; "
                        f"a USB stick is {'already inserted' if inserted else 'not inserted'}.",
                        "Recording requires active power and an inserted stick.",
                        "The first row is written on the next simulation grid step, then rows follow the selected interval.",
                        "A successful recording command alone does not prove that any row was collected."])
            case["budget"]["max_sim_time_s"] = max(1200.0, 2 * wait_s)
            cases.append(_expanded(case, variant, difficulty_axes={
                "readiness_actions": int(not power_on) + int(not inserted),
                "recording_interval_min": interval_min, "required_rows": rows,
                "feedback_branching": False, "verification_required": True},
                fixture_conditions=[_condition("power.on", power_on),
                                    _condition("usb.state", inserted, path=["inserted"]),
                                    _condition("usb.state", False, path=["recording"])]))
    return cases


def _plate_cases(seed):
    cases = []
    readiness_profiles = [(True, True, False), (True, False, False),
                          (False, False, False), (True, True, True)]
    for rpm in (100, 300, 600):
        speed = rpm * 2 * math.pi / 60
        for readiness, (power_on, locked, running) in enumerate(readiness_profiles):
            monitor_s = float(2 + readiness)
            goals = [_condition("stirring.on", True), _condition("stirring.set_rad_s", speed),
                     _condition("error.code", None)]
            setup, nodes, unlock_dependencies = [], [], []
            if power_on:
                setup.append(_invoke("power.set", on=True))
            else:
                nodes.append(_node("power", "readiness", _invoke("power.set", on=True),
                                   [_condition("power.on", True)]))
                unlock_dependencies = ["power"]
            if running:
                setup.extend([_invoke("stirring.set_speed", speed_rad_s=50 * 2 * math.pi / 60),
                              _invoke("stirring.set", on=True)])
            if locked:
                setup.append(_invoke("lock.toggle"))
                nodes.append(_node("unlock", "readiness", _invoke("lock.toggle"),
                                   [_condition("lock.on", False)], depends_on=unlock_dependencies))
                speed_dependencies = ["unlock"]
            else:
                speed_dependencies = unlock_dependencies
            nodes.append(_node("speed", "configuration", _invoke("stirring.set_speed", speed_rad_s=speed),
                               [_condition("stirring.set_rad_s", speed)], depends_on=speed_dependencies))
            if not running:
                nodes.append(_node("stir", "execution", _invoke("stirring.set", on=True),
                                   [_condition("stirring.on", True)], depends_on=["speed"]))
            nodes.extend([
                _node("monitor", "monitoring", {"kind": "wait", "dt_s": monitor_s}, goals,
                      depends_on=["speed" if running else "stir"], kind="wait", min_dt_s=monitor_s),
                _read_node("verify", ["stirring.on", "stirring.set_rad_s", "error.code"], ["monitor"]),
                _finish_node(["verify"]),
            ])
            variant = f"rpm-{rpm}-readiness-{readiness}"
            case = _case("hotplate_unlock_stirring", variant, "ika_plate_rctdigital",
                f"Select the magnetic hotplate stirrer. Establish stirring at {speed:.15g} rad/s "
                f"({rpm} rpm), monitor it for at least {monitor_s:g} simulated seconds, "
                "then verify stirring at the requested setpoint without an alarm.",
                nodes, goals, seed=seed, setup_steps=setup,
                observation_ids=["power.on", "lock.on", "stirring.on", "stirring.set_rad_s", "error.code"],
                manual=[f"Power starts {'on' if power_on else 'off'}, the keypad is "
                        f"{'locked' if locked else 'unlocked'}, and stirring is "
                        f"{'already enabled at 50 rpm' if running else 'off'}.",
                        "The motor is functional; heating is not requested.",
                        "Changing stirring controls requires power and an unlocked keypad.",
                        "The speed control uses rad/s and rounds to native 10 rpm increments."])
            cases.append(_expanded(case, variant, difficulty_axes={
                "readiness_actions": int(not power_on) + int(locked), "initial_running": running,
                "target_rpm": rpm, "monitoring_s": monitor_s,
                "feedback_branching": False, "verification_required": True},
                fixture_conditions=[_condition("power.on", power_on), _condition("lock.on", locked),
                                    _condition("stirring.on", running)]))
    return cases


def _pump_cases(seed):
    # Values are documented in the native syringe_templates.json catalog.
    profiles = [("bd_1ml", .004699, .7, 1), ("bd_3ml", .008585, 2, 5),
                ("bd_5ml", .01199, 3, 10), ("bd_10ml", .01443, 7, 20),
                ("bd_20ml", .01905, 12, 40), ("bd_60ml", .02659, 40, 100)]
    cases = []
    for index, (syringe, diameter, fill_ml, rate_ml_h) in enumerate(profiles):
        rate, fill, length = rate_ml_h * 1e-6 / 3600, fill_ml * 1e-6, .2 + .1 * index
        load = _invoke("syringe.load", syringe=syringe, liquid_id="water", fill_volume_m3=fill)
        outlet = _invoke("outlet.connect", tubing="ptfe_0p8mm", tubing_length_m=length,
                         needle="none", backpressure_pa=0.0)
        load_conditions = [_condition("syringe.state", syringe, path=["type"]),
                           _condition("syringe.state", "water", path=["liquid_id"]),
                           _private_condition("truth.process", fill, path=["syringe", "liquid_m3"])]
        outlet_conditions = [_condition("outlet.state", "ptfe_0p8mm", path=["tubing"]),
                             _condition("outlet.state", length, path=["tubing_length_m"]),
                             _condition("outlet.state", "none", path=["needle"]),
                             _condition("outlet.state", 0.0, path=["backpressure_pa"])]
        diameter_condition = _condition("syringe.diameter_m", diameter, tolerance=1e-8)
        rate_condition = _condition("program.state", rate, path=["phases", 0, "rate_m3_s"], tolerance=1e-12)
        goals = [_condition("run.state", "running"), _condition("alarm.state", None),
                 _condition("volume.dispensed", 0, operator="gt", path=["infused_m3"]),
                 *load_conditions[:2], diameter_condition, rate_condition, *outlet_conditions]
        for preloaded in (False, True):
            setup, nodes = [], []
            if preloaded:
                setup.extend([load, outlet])
                nodes.append(_node("power", "readiness", _invoke("power.set", on=True),
                                   [_condition("power.on", True)]))
                preparation, power_dependencies = [], ["power"]
            else:
                setup.append(_invoke("power.set", on=True))
                nodes.extend([_node("load", "preparation", load, load_conditions),
                              _node("outlet", "preparation", outlet, outlet_conditions)])
                preparation, power_dependencies = ["load", "outlet"], []
            nodes.extend([
                _node("diameter", "configuration", _invoke("syringe.set_diameter", diameter_m=diameter),
                      [diameter_condition], depends_on=power_dependencies),
                _node("rate", "configuration", _invoke("rate.set", rate_m3_s=rate, units="MH"),
                      [rate_condition], depends_on=["diameter"]),
                _node("run", "execution", _invoke("pump.run"), [_condition("run.state", "running")],
                      depends_on=preparation + ["diameter", "rate"]),
                _node("deliver", "monitoring", {"kind": "wait", "dt_s": 3.0 + index}, goals,
                      depends_on=["run"], kind="wait"),
                _read_node("verify", ["run.state", "volume.dispensed", "alarm.state"], ["deliver"]),
                _finish_node(["verify"]),
            ])
            variant = f"{syringe}-{'preloaded-power-off' if preloaded else 'unloaded-power-on'}"
            case = _case("pump_delivery", variant, "newera_ne1000",
                f"Select the syringe pump. Use a {syringe} syringe loaded with {fill_ml:g} mL water, "
                f"set its inside diameter to {diameter:g} m and infusion rate to {rate_ml_h:g} mL/hour, "
                f"and connect {length:g} m of ptfe_0p8mm tubing without a needle or backpressure. "
                "Establish running delivery and verify positive reported infused volume without an alarm.",
                nodes, goals, seed=seed, setup_steps=setup,
                observation_ids=["power.on", "syringe.state", "outlet.state", "syringe.diameter_m",
                                 "program.state", "run.state", "volume.dispensed", "alarm.state"],
                manual=[("The correct filled syringe and outlet are already installed; power is off."
                         if preloaded else "Power is on; no syringe is loaded and the requested outlet must be connected."),
                        "The motor is functional, the tubing is unobstructed, and the collar is parked.",
                        f"The selected syringe's documented inside diameter is {diameter:g} m.",
                        f"{rate_ml_h:g} mL/hour equals {rate:.15g} m3/s; the native rate unit MH denotes mL/hour.",
                        "volume.dispensed uses configured diameter and is not an independent true-flow measurement."])
            # A generous short monitoring budget avoids accepting a long wait
            # that would empty a healthy syringe and manufacture another fault.
            case["budget"]["max_sim_time_s"] = 120.0
            fixture = [_condition("power.on", not preloaded)]
            if preloaded:
                fixture.extend(load_conditions + outlet_conditions)
            else:
                fixture.append(_condition("syringe.state", None))
            cases.append(_expanded(case, variant, difficulty_axes={
                "initial_syringe_loaded": preloaded, "readiness_actions": 1 if preloaded else 2,
                "syringe_template": syringe, "diameter_m": diameter, "rate_ml_h": rate_ml_h,
                "feedback_branching": False, "verification_required": True}, fixture_conditions=fixture))
    return cases


def _hood_cases(seed):
    # Six meaningful public tasks, each paired with a native lamp failure and a
    # healthy control.  The initial switch is off in both, hiding lamp condition.
    profiles = [
        (None, 0.0, "off", 0), (0.0, .3, "off", 0),
        (.2, 0.0, "off", 0), (.4, 0.0, "closed", 0),
        (None, 0.0, "off", 2), (.3, 0.0, "closed", 3),
    ]
    cases = []
    lamp_working = [_condition("lamp.lit", True)]
    lamp_failed = [_condition("mains.on", True), _condition("light.on", True), _condition("lamp.lit", False)]
    for index, (target_sash, initial_sash, keylock, monitor_s) in enumerate(profiles):
        pair_id = f"illumination-pair-{index + 1}"
        sash_goal = [] if target_sash is None else [_condition("sash.height_m", target_sash)]
        goal = "Select the fume hood. "
        if target_sash is not None:
            goal += f"Set its sash opening to {target_sash:g} m, then "
        goal += ("restore illumination or request maintenance only after testing shows persistent lamp failure. "
                 "The mains supply is functional; a working lamp and a failed lamp are both possible before testing.")
        if monitor_s:
            goal += f" Allow at least {monitor_s:g} simulated seconds after switching on before the final check."
        manual = ["The light switch is initially off. Switching it on tests the lamp; reading it while switched off cannot identify failure.",
                  "A light.set response may already return lamp.lit. A persistent unlit lamp with functional mains requires maintenance.",
                  "This is an empty-enclosure controls task; sash position does not establish chemical containment or safe airflow."]
        if keylock == "closed":
            manual.append("The sash keylock starts in closed mode. Set keylock mode off before opening the sash.")
        for failed in (False, True):
            setup = [_invoke("light.set", on=False), _invoke("sash.set", height_m=initial_sash)]
            if keylock != "off":
                setup.append(_invoke("keylock.set", mode=keylock))
            if failed:
                setup.append(dict(_invoke("fault.inject", fault="lamp_failure", active=True), actor="evaluator"))
            nodes, preparation = [], []
            if keylock == "closed" and target_sash is not None and target_sash > 0:
                nodes.append(_node("unlock_sash", "readiness", _invoke("keylock.set", mode="off"),
                                   [_condition("keylock.mode", "off")]))
                preparation = ["unlock_sash"]
            if target_sash is not None:
                nodes.append(_node("sash", "preparation", _invoke("sash.set", height_m=target_sash),
                                   sash_goal, depends_on=preparation))
                preparation = ["sash"]
            nodes.append(_node("test_light", "diagnosis", _invoke("light.set", on=True),
                               [_condition("light.on", True)], depends_on=preparation))
            diagnostic_dependencies = ["test_light"]
            if monitor_s:
                nodes.append(_node("persist", "monitoring", {"kind": "wait", "dt_s": float(monitor_s)},
                                   depends_on=diagnostic_dependencies, kind="wait", min_dt_s=monitor_s))
                diagnostic_dependencies = ["persist"]
            nodes.extend([
                _read_node("observe_lamp", ["lamp.lit", "mains.on"], diagnostic_dependencies),
                _finish_node(["observe_lamp"], when_conditions=lamp_working),
                _node("maintenance", "completion", {"kind": "escalate"}, depends_on=["observe_lamp"],
                      kind="escalate", when_conditions=lamp_failed),
            ])
            variant = f"{pair_id}-{'failed' if failed else 'healthy'}"
            case = _case("hood_illumination", variant, "esco_efa5uduvw", goal, nodes,
                lamp_working + sash_goal, seed=seed, setup_steps=setup,
                observation_ids=["light.on", "lamp.lit", "mains.on", "lighting.illuminance_lux",
                                 "sash.height_m", "keylock.mode"], manual=manual,
                escalation_expected=failed, escalation_conditions=lamp_failed + sash_goal)
            cases.append(_expanded(case, variant, condition="lamp_failure" if failed else "healthy",
                pair_id=pair_id, difficulty_axes={
                    "readiness_actions": int(target_sash is not None) + int(keylock == "closed"),
                    "minimum_monitoring_s": monitor_s, "latent_fault": True,
                    "feedback_branching": True, "verification_required": True},
                fixture_conditions=[_condition("light.on", False), _condition("lamp.lit", False),
                                    _condition("mains.on", True), _condition("sash.height_m", initial_sash),
                                    _condition("keylock.mode", keylock)]))
    return cases


def _timed_pump_cases(seed):
    cases = []
    for index, (rate_ml_h, monitoring_s, trigger_s) in enumerate(
            [(20, 3, 2), (50, 4, 3), (80, 5, 4), (100, 6, 4), (120, 8, 6), (150, 10, 8)]):
        pair_id, rate = f"monitor-pair-{index + 1}", rate_ml_h * 1e-6 / 3600
        setup = [_invoke("power.set", on=True),
                 _invoke("syringe.load", syringe="bd_60ml", liquid_id="water", fill_volume_m3=40e-6),
                 _invoke("outlet.connect", tubing="ptfe_0p8mm", tubing_length_m=.5,
                         needle="none", backpressure_pa=0.0),
                 _invoke("syringe.set_diameter", diameter_m=.02659),
                 _invoke("rate.set", rate_m3_s=rate, units="MH"), _invoke("pump.run")]
        healthy = [_condition("run.state", "running"), _condition("alarm.state", None),
                   _condition("volume.dispensed", 0, operator="gt", path=["infused_m3"])]
        stalled = [_condition("alarm.state", "stall"), _condition("run.state", "paused")]
        goal = ("Select the syringe pump already delivering water at "
                f"{rate_ml_h:g} mL/hour. Monitor for at least {monitoring_s:g} simulated seconds, then inspect delivery. "
                "Confirm continued delivery only if it is still running with positive reported infused volume and no alarm; "
                "request maintenance if a persistent motor stall is observed.")
        manual = ["The syringe contains water, tubing is unobstructed, and the collar is parked.",
                  "A motor failure may occur during monitoring; an initial running state does not establish whether one will occur.",
                  "Public controls cannot repair a persistent motor fault. Inspect the alarm before deciding to continue or request maintenance.",
                  "Waiting advances simulated time; it does not automatically return updated alarm readings."]
        for failed in (False, True):
            nodes = [
                _node("monitor", "monitoring", {"kind": "wait", "dt_s": float(monitoring_s)},
                      depends_on=[], kind="wait", min_dt_s=monitoring_s),
                _read_node("observe_delivery", ["run.state", "alarm.state", "volume.dispensed"], ["monitor"]),
                _node("maintenance", "completion", {"kind": "escalate"}, depends_on=["observe_delivery"],
                      kind="escalate", when_conditions=stalled),
                _finish_node(["observe_delivery"], when_conditions=healthy),
            ]
            event = {"trigger_time_s": float(trigger_s), "steps": [
                dict(_invoke("fault.inject", fault="motor_failure", active=True), actor="evaluator")]}
            variant = f"{pair_id}-{'motor-failure' if failed else 'healthy'}"
            case = _case("pump_timed_monitoring", variant, "newera_ne1000", goal, nodes, healthy,
                seed=seed, setup_steps=copy.deepcopy(setup), scheduled_events=[event] if failed else [],
                observation_ids=["run.state", "alarm.state", "volume.dispensed", "power.on"], manual=manual,
                escalation_expected=failed, escalation_conditions=stalled)
            case["budget"]["max_sim_time_s"] = 120.0
            cases.append(_expanded(case, variant, condition="timed_motor_failure" if failed else "healthy",
                pair_id=pair_id, difficulty_axes={
                    "monitoring_s": monitoring_s, "possible_failure_timing": "during_monitoring",
                    "latent_fault": True, "feedback_branching": True, "verification_required": True},
                fixture_conditions=[_condition("run.state", "running"), _condition("alarm.state", None),
                                    _condition("power.on", True)]))
    return cases


def build_variants(seed=42):
    """Return exactly 72 independent specs: 12 per native workflow template."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    cases = []
    for builder in (_furnace_cases, _oven_cases, _plate_cases, _pump_cases,
                    _hood_cases, _timed_pump_cases):
        cases.extend(builder(seed))
    return copy.deepcopy(cases)
