"""Private native workflow fixtures for corrected and free action rollouts.

This is a separate pilot corpus.  It does not modify the frozen multiple-choice
release.  Importing this module does not create an instrument or load a model.
Only the evaluator may inspect the dictionaries returned by ``build_tasks``;
checkpoint actions, fault fixtures and event schedules are not solver inputs.
"""
from __future__ import annotations

import copy
import hashlib
import math


def machine_catalog():
    """Public selection menu; these are independent machines, not a material chain."""
    return [
        {"instrument_id": "esco_efa5uduvw", "name": "Esco fume hood",
         "purpose": "Ventilated laboratory enclosure with sash and illumination controls.",
         "capabilities": ["light.set", "sash.set", "anemometer.measure"]},
        {"instrument_id": "binder_fd056", "name": "Binder FD 56 drying oven",
         "purpose": "Temperature-controlled drying oven with timer and USB recording.",
         "capabilities": ["power.set", "temperature.set", "timer.set", "usb.set_stick", "usb.record"]},
        {"instrument_id": "nabertherm_l9", "name": "Nabertherm L 9 programmable furnace",
         "purpose": "Controlled temperature programs with ramp, hold and door interlock.",
         "capabilities": ["power.set", "door.close", "thermal.configure_program", "thermal.start_program"]},
        {"instrument_id": "ika_plate_rctdigital", "name": "IKA RCT digital hotplate stirrer",
         "purpose": "Magnetic stirring and heating with keypad lock and safety controls.",
         "capabilities": ["power.set", "lock.toggle", "stirring.set_speed", "stirring.set"]},
        {"instrument_id": "newera_ne1000", "name": "New Era NE-1000 syringe pump",
         "purpose": "Programmed syringe infusion with geometry, rate and stall monitoring.",
         "capabilities": ["syringe.load", "syringe.set_diameter", "rate.set", "pump.run", "pump.stop"]},
        {"instrument_id": "vaisala_dmt152", "name": "Vaisala DMT152 dewpoint transmitter",
         "purpose": "Dewpoint measurement, output diagnostics and calibration service.",
         "capabilities": ["power.set", "installation.remove", "calibration.service"]},
        {"instrument_id": "peak_precisionhydrogen", "name": "Peak Precision Hydrogen generator",
         "purpose": "Hydrogen delivery after startup diagnostics and pressure configuration.",
         "capabilities": ["power.set", "settings.configure", "cycle.start", "alarm.mute"]},
        {"instrument_id": "eppendorf_cellxpertc170i", "name": "Eppendorf CellXpert C170i incubator",
         "purpose": "Controlled incubation chamber with outer door, latch and inner door.",
         "capabilities": ["power.set", "door.outer", "door.latch", "door.inner"]},
    ]


def _invoke(capability_id, **arguments):
    return {"kind": "invoke", "capability_id": capability_id, "arguments": arguments}


def _condition(state_id, value, *, operator="eq", path=(), tolerance=1e-6):
    return {"state_id": state_id, "operator": operator, "value": value,
            "path": list(path), "tolerance": tolerance, "visibility": "agent"}


def _node(checkpoint_id, stage, action, conditions=(), *, depends_on=(),
          kind="conditions", state_ids=(), min_dt_s=0, when_conditions=()):
    return {"checkpoint_id": checkpoint_id, "stage": stage,
            "depends_on": list(depends_on), "reference_action": action,
            "accept": {"kind": kind, "conditions": list(conditions),
                       "state_ids": list(state_ids), "min_dt_s": min_dt_s},
            "when_conditions": list(when_conditions)}


def _read_node(checkpoint_id, state_ids, depends_on):
    return _node(checkpoint_id, "verification", {"kind": "read", "state_ids": list(state_ids)},
                 depends_on=depends_on, kind="read", state_ids=state_ids)


def _finish_node(depends_on, *, when_conditions=()):
    return _node("finish", "completion", {"kind": "finish"},
                 depends_on=depends_on, kind="finish", when_conditions=when_conditions)


def _case(family, variant, instrument_id, goal, checkpoints, goal_conditions, *,
          seed, setup_steps=(), scheduled_events=(), observation_ids=(), manual=(),
          escalation_expected=False, escalation_conditions=()):
    # Opaque identifiers prevent a fixture label such as "failed" leaking into
    # a solver prompt.  Paired fixtures still have a shared private split group.
    identity = hashlib.sha256(f"stepwise-v1:{family}:{variant}:{seed}".encode()).hexdigest()[:24]
    return {"task_id": identity, "case_id": identity,
            "group_id": f"stepwise-v1:{family}:seed-{seed}",
            "workflow_id": family, "instrument_id": instrument_id, "seed": seed,
            "goal": goal, "setup_steps": list(setup_steps),
            "scheduled_events": list(scheduled_events),
            "goal_conditions": list(goal_conditions),
            "escalation_expected": escalation_expected,
            "escalation_conditions": list(escalation_conditions),
            "observation_ids": list(observation_ids), "manual": list(manual),
            "budget": {"max_decisions": 32, "max_actions": 32,
                       "max_sim_time_s": 3600.0, "max_queries": 4},
            "checkpoints": checkpoints}


def build_tasks(seed=0):
    """Return seven private fixtures spanning six native workflow groups.

    The hood fixtures have identical public goals and initial observations, but
    require different decisions after the same diagnostic action.  Other cases
    test prerequisites, parameter semantics, simulated time and observed goals.
    Seed controls native runtime noise; it does not invent unimplemented faults.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    tasks = []

    # Preparation actions form a DAG: closing the door and enabling power may
    # be performed in either order, and configuration need not follow closing.
    segment = {"temperature_K": 308.15, "ramp_rate_K_s": 0.05, "hold_s": 4}
    furnace_goals = [_condition("program.phase", "completed"),
                     _condition("program.running", False),
                     _condition("program.hold_elapsed_s", 4, operator="ge"),
                     _condition("program.segments", [segment])]
    furnace = [
        _node("power", "readiness", _invoke("power.set", enabled=True),
              [_condition("power.enabled", True)]),
        _node("door", "readiness", _invoke("door.close"), [_condition("door.closed", True)]),
        _node("configure", "configuration", _invoke("thermal.configure_program", segments=[segment]),
              [_condition("program.segments", [segment])], depends_on=["power"]),
        _node("start", "execution", _invoke("thermal.start_program"),
              [_condition("program.running", True)], depends_on=["power", "door", "configure"]),
        _node("complete_program", "monitoring", {"kind": "wait", "dt_s": 224.0},
              furnace_goals, depends_on=["start"], kind="wait", min_dt_s=0),
        _read_node("verify", ["program.phase", "program.running", "program.hold_elapsed_s"],
                   ["complete_program"]),
        _finish_node(["verify"]),
    ]
    tasks.append(_case("furnace_program", "ordinary", "nabertherm_l9",
        "Select the programmable furnace. Complete an empty-chamber program at 308.15 K "
        "with ramp rate 0.05 K/s and a 4-second hold, then observe and verify completion.",
        furnace, furnace_goals, seed=seed, setup_steps=[_invoke("door.open")],
        observation_ids=["power.enabled", "door.closed", "thermal.indicated_temperature_k", "program.segments",
                         "program.phase", "program.running", "program.hold_elapsed_s"],
        manual=["The chamber is empty; no chemical transformation or sample quality is modeled.",
                "Configuration requires power. Starting also requires the door closed.",
                "Waiting advances simulated time; an accepted start does not establish completion."]))

    oven_goals = [_condition("usb.state", True, path=["inserted"]),
                  _condition("usb.state", True, path=["recording"]),
                  _condition("usb.state", 60.0, path=["interval_s"]),
                  _condition("usb.state", 1, operator="ge", path=["rows"])]
    oven = [
        _node("power", "readiness", _invoke("power.set", on=True), [_condition("power.on", True)]),
        _node("media", "preparation", _invoke("usb.set_stick", inserted=True),
              [_condition("usb.state", True, path=["inserted"])]),
        _node("record", "configuration", _invoke("usb.record", on=True, interval_min=1),
              [_condition("usb.state", True, path=["recording"]),
               _condition("usb.state", 60.0, path=["interval_s"])], depends_on=["power", "media"]),
        _node("collect_row", "monitoring", {"kind": "wait", "dt_s": 60.0}, oven_goals,
              depends_on=["record"], kind="wait", min_dt_s=0),
        _read_node("verify", ["usb.state"], ["collect_row"]),
        _finish_node(["verify"]),
    ]
    tasks.append(_case("oven_usb_recording", "ordinary", "binder_fd056",
        "Select the drying oven with USB logging. Start recording at a one-minute interval "
        "and verify that at least one row exists while recording remains enabled.",
        oven, oven_goals, seed=seed,
        observation_ids=["power.on", "usb.state"],
        manual=["USB recording requires the oven powered on and a USB stick inserted.",
                "The first row is produced on a simulation grid step; later rows follow the configured interval.",
                "A successful recording command alone does not prove that a row was collected."]))

    speed = 300 * 2 * math.pi / 60
    plate_goals = [_condition("stirring.on", True), _condition("stirring.set_rad_s", speed),
                   _condition("error.code", None)]
    plate = [
        _node("unlock", "readiness", _invoke("lock.toggle"), [_condition("lock.on", False)]),
        _node("speed", "configuration", _invoke("stirring.set_speed", speed_rad_s=speed),
              [_condition("stirring.set_rad_s", speed)], depends_on=["unlock"]),
        _node("stir", "execution", _invoke("stirring.set", on=True),
              [_condition("stirring.on", True)], depends_on=["unlock", "speed"]),
        _node("monitor", "monitoring", {"kind": "wait", "dt_s": 2.0}, plate_goals,
              depends_on=["stir"], kind="wait", min_dt_s=2),
        _read_node("verify", ["stirring.on", "stirring.set_rad_s", "error.code"], ["monitor"]),
        _finish_node(["verify"]),
    ]
    tasks.append(_case("hotplate_unlock_stirring", "ordinary", "ika_plate_rctdigital",
        f"Select the magnetic hotplate stirrer. Establish stirring at {speed:.15g} rad/s "
        "(300 rpm), monitor it for at least two simulated seconds, and verify stirring without an alarm.",
        plate, plate_goals, seed=seed,
        setup_steps=[_invoke("power.set", on=True), _invoke("lock.toggle")],
        observation_ids=["power.on", "lock.on", "stirring.on", "stirring.set_rad_s", "error.code"],
        manual=["The instrument is powered on with its keypad locked; the motor is functional.",
                "Unlocking permits changes to the stirring controls. Heating is not requested."]))

    rate = 100e-6 / 3600
    diameter = 0.02659
    load = _invoke("syringe.load", syringe="bd_60ml", liquid_id="water", fill_volume_m3=40e-6)
    outlet = _invoke("outlet.connect", tubing="ptfe_0p8mm", tubing_length_m=0.5,
                     needle="none", backpressure_pa=0.0)
    diameter_action = _invoke("syringe.set_diameter", diameter_m=diameter)
    rate_action = _invoke("rate.set", rate_m3_s=rate, units="MH")
    pump_setup = [_invoke("power.set", on=True), load, outlet, diameter_action, rate_action]
    pump_goals = [_condition("run.state", "running"),
                  _condition("volume.dispensed", 0, operator="gt", path=["infused_m3"]),
                  _condition("alarm.state", None),
                  _condition("syringe.diameter_m", diameter, tolerance=1e-8),
                  _condition("program.state", rate, path=["phases", 0, "rate_m3_s"], tolerance=1e-12)]
    pump = [
        _node("load", "preparation", load, [_condition("syringe.state", "bd_60ml", path=["type"])]),
        _node("outlet", "preparation", outlet,
              [_condition("outlet.state", "ptfe_0p8mm", path=["tubing"]),
               _condition("outlet.state", 0.5, path=["tubing_length_m"]),
               _condition("outlet.state", "none", path=["needle"]),
               _condition("outlet.state", 0.0, path=["backpressure_pa"])]),
        _node("diameter", "configuration", diameter_action,
              [_condition("syringe.diameter_m", diameter, tolerance=1e-8)]),
        _node("rate", "configuration", rate_action,
              [_condition("program.state", rate, path=["phases", 0, "rate_m3_s"], tolerance=1e-12)],
              depends_on=["diameter"]),
        _node("run", "execution", _invoke("pump.run"), [_condition("run.state", "running")],
              depends_on=["load", "outlet", "diameter", "rate"]),
        _node("deliver", "monitoring", {"kind": "wait", "dt_s": 3.0}, pump_goals,
              depends_on=["run"], kind="wait", min_dt_s=0),
        _read_node("verify", ["run.state", "volume.dispensed", "alarm.state"], ["deliver"]),
        _finish_node(["verify"]),
    ]
    tasks.append(_case("pump_delivery", "ordinary", "newera_ne1000",
        "Select the syringe pump. Load a BD 60 mL syringe with 40 mL water, configure "
        "diameter 0.02659 m and infusion rate 100 mL/hour, connect 0.5 m of ptfe_0p8mm "
        "tubing with no needle or backpressure, then establish running delivery and "
        "verify a positive reported infused volume without an alarm.",
        pump, pump_goals, seed=seed, setup_steps=[_invoke("power.set", on=True)],
        observation_ids=["power.on", "syringe.state", "outlet.state", "syringe.diameter_m",
                         "program.state", "run.state", "volume.dispensed", "alarm.state"],
        manual=["The pump is powered on. The tubing is unobstructed, the motor functional and the collar parked.",
                "The specified BD 60 mL syringe has the known native geometry, with inside diameter 0.02659 m.",
                "100 mL/hour equals 2.77777777777778e-8 m3/s; native rate units MH denote mL/hour.",
                "volume.dispensed is computed using configured diameter; it is not an independent true-flow measurement."]))

    hood_goal = ("Select the fume hood. Restore its illumination, or request maintenance "
                 "only after testing shows a persistent lamp failure. The switch is initially "
                 "off; the mains supply is functional. A working lamp and a failed lamp "
                 "are both possible before the test.")
    lamp_on = [_condition("light.on", True)]
    lamp_working = [_condition("lamp.lit", True)]
    lamp_failed = [_condition("mains.on", True), _condition("light.on", True),
                   _condition("lamp.lit", False)]
    for failed in (False, True):
        hood = [
            _node("test_light", "diagnosis", _invoke("light.set", on=True), lamp_on),
            _read_node("observe_lamp", ["lamp.lit"], ["test_light"]),
            _finish_node(["observe_lamp"], when_conditions=lamp_working),
            _node("maintenance", "completion", {"kind": "escalate"},
                  depends_on=["observe_lamp"], kind="escalate", when_conditions=lamp_failed),
        ]
        fault = dict(_invoke("fault.inject", fault="lamp_failure", active=True), actor="evaluator")
        tasks.append(_case("hood_illumination", "failed" if failed else "healthy", "esco_efa5uduvw",
            hood_goal, hood, lamp_working, seed=seed,
            setup_steps=[_invoke("light.set", on=False)] + ([fault] if failed else []),
            observation_ids=["light.on", "lamp.lit", "mains.on", "lighting.illuminance_lux"],
            manual=["Switching on is the diagnostic intervention; reading a lamp while switched off cannot distinguish the possibilities.",
                    "An accepted light.set response may already return lamp.lit. Use that returned observation when present.",
                    "A lamp that remains unlit with functional mains and its switch on requires maintenance; repeated toggling does not repair it."],
            escalation_expected=failed, escalation_conditions=lamp_failed))

    # A timed native fault tests a decision after an observation, rather than
    # giving the model a preselected plan.  The possible failure is public, its
    # event schedule and the actual injected fault are evaluator-only.
    stalled = [_condition("alarm.state", "stall"), _condition("run.state", "paused")]
    monitor = [
        _node("monitor", "monitoring", {"kind": "wait", "dt_s": 3.0},
              depends_on=[], kind="wait", min_dt_s=3),
        _read_node("observe_delivery", ["run.state", "alarm.state", "volume.dispensed"], ["monitor"]),
        _node("maintenance", "completion", {"kind": "escalate"},
              depends_on=["observe_delivery"], kind="escalate", when_conditions=stalled),
        _finish_node(["observe_delivery"], when_conditions=[_condition("run.state", "running"),
                                                            _condition("alarm.state", None)]),
    ]
    motor_fault = dict(_invoke("fault.inject", fault="motor_failure", active=True), actor="evaluator")
    tasks.append(_case("pump_timed_monitoring", "midrun_failure", "newera_ne1000",
        "Select the syringe pump already delivering water. Monitor for at least three "
        "simulated seconds, then inspect delivery. Continue only if it is still running "
        "without an alarm; request maintenance if a persistent motor stall is observed.",
        monitor, [_condition("run.state", "running"), _condition("alarm.state", None)],
        seed=seed, setup_steps=pump_setup + [_invoke("pump.run")],
        scheduled_events=[{"trigger_time_s": 2.0, "steps": [motor_fault]}],
        observation_ids=["run.state", "alarm.state", "volume.dispensed", "power.on"],
        manual=["The syringe contains water, tubing is unobstructed and the collar is parked.",
                "Motor failure is possible during monitoring, but an initial running state does not establish a future stall.",
                "Once an actual stall is observed, public controls cannot repair the persistent motor fault.",
                "Waiting advances simulated time. Inspecting the alarm is a separate action unless a tool response returns it."],
        escalation_expected=True, escalation_conditions=stalled))

    return copy.deepcopy(tasks)
