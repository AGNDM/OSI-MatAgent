"""Native, deterministic scenario families; chemical outcome labels are excluded."""
from __future__ import annotations

import math
import json
import copy
from dataclasses import replace
import random

from .adapters import catalog_domain
from .schema import Condition, Scenario, Step


def command(capability_id, actor="agent", **arguments):
    return Step(capability_id=capability_id, arguments=arguments, actor=actor).to_dict()


def wait(seconds):
    return Step(kind="wait", dt_s=float(seconds), actor="host").to_dict()


def read(*state_ids):
    return Step(kind="read", state_ids=list(state_ids) or None).to_dict()


def answer(value):
    return Step(kind="answer", answer=value).to_dict()


def escalate():
    return Step(kind="escalate").to_dict()


def condition(slot, value, *, path=(), operator="eq", visibility="agent"):
    return Condition(slot, operator, value, list(path), visibility=visibility).to_dict()


def option(text, *steps):
    return {"text": text, "steps": list(steps)}


def fault(name, active=True):
    return command("fault.inject", actor="evaluator", fault=name, active=active)


def build_scenarios(seed=0):
    """Build paired controls and independent operational failure mechanisms.

    Seeds vary physical values, deadlines, rates and durations. Wording variants
    are generated separately and stay in the same latent-case split group.
    """
    rng = random.Random(seed)
    cases = []

    def add(family, instrument, task_type, question, options, **kwargs):
        difficulty = kwargs.pop("difficulty", "L1")
        cases.append(Scenario(family, instrument, catalog_domain(instrument), task_type,
                              difficulty, kwargs.pop("tier", "T1"), question, options, **kwargs))

    def predict(family, instrument, question, setup, probe, *, controls=(), **kwargs):
        # Probe outcomes are computed by replay, never guessed from a contract.
        choices = [option("The call succeeds.", answer("succeeded")),
                   option("The call is rejected.", answer("rejected"))]
        add(family, instrument, "predict", question, choices, setup_steps=setup,
            probe_steps=probe, healthy_control=bool(controls), **kwargs)

    def recover(family, instrument, question, setup, reference, goals, distractors, **kwargs):
        choices = [option("Apply this sequence of controls, waits, and observations.", *reference)] + distractors
        add(family, instrument, "recover", question, choices, setup_steps=setup,
            reference_steps=reference, goal_conditions=goals, difficulty=kwargs.pop("difficulty", "L4"),
            tier=kwargs.pop("tier", "T3"), **kwargs)

    def diagnosis(family, instrument, question, setup, rules, causes, **kwargs):
        choices = [option(text, answer(label)) for label, text, _ in rules]
        add(family, instrument, "diagnose", question, choices, setup_steps=setup,
            answer_conditions={label: conds for label, _, conds in rules}, causes=causes,
            difficulty=kwargs.pop("difficulty", "L2"), **kwargs)

    def information_pair(family, instrument, setup, paired_setup, probe, goals,
                         observation_ids, objective, diagnostic_steps, fault_family, causes):
        add(family + "_uncertainty", instrument, "predict", objective,
            [option("The operational outcome is established.", answer("goal_met")),
             option("The operational outcome will not hold.", answer("goal_not_met")),
             option("The recorded evidence is insufficient; inspect first.", answer("insufficient_information"))],
            setup_steps=setup, paired_setup_steps=paired_setup, probe_steps=probe,
            diagnostic_steps=diagnostic_steps, evidence_condition="untested",
            goal_conditions=goals, observation_ids=observation_ids, difficulty="L3", tier="T2",
            fault_family=fault_family, causes=causes,
            difficulty_axes={"unresolved_hypotheses": 2, "diagnostic_horizon": len(diagnostic_steps)})
        add(family + "_measurement", instrument, "measure",
            "Choose evidence that distinguishes the two possibilities in this objective: " + objective,
            [option("Repeat the existing readout.", read(*observation_ids)),
             option("Perform an independent diagnostic test.", *diagnostic_steps),
             option("Assume the outcome is established.", answer("goal_met")),
             option("Assume the outcome is impossible.", answer("goal_not_met"))],
            setup_steps=setup, paired_setup_steps=paired_setup, observation_ids=observation_ids,
            difficulty="L3", tier="T2", fault_family=fault_family, causes=causes,
            difficulty_axes={"unresolved_hypotheses": 2, "diagnostic_horizon": len(diagnostic_steps)})

    # Hood: light-switch, electrical supply, native lamp failure and hidden
    # fault counterfactuals are distinct cases, rather than renamed templates.
    hood = "esco_efa5uduvw"
    lamp_rules = [
        ("healthy", "No illumination failure is present.", [condition("lamp.lit", True)]),
        ("switch_off", "The light switch is off.", [condition("light.on", False), condition("mains.on", True)]),
        ("lamp_failure", "The lamp has failed.", [condition("light.on", True), condition("mains.on", True), condition("lamp.lit", False)]),
        ("mains_outage", "The electrical supply is unavailable.", [condition("mains.on", False)]),
    ]
    for state in ("healthy", "switch_off", "lamp_failure", "mains_outage"):
        setup = [command("light.set", on=state != "switch_off")]
        if state in ("lamp_failure", "mains_outage"):
            setup.append(fault(state))
        diagnosis("hood_lighting_diagnosis", hood,
                  "The possible causes are a switched-off light, a failed lamp, or lost mains; otherwise the light is healthy. Which diagnosis fits the observations?",
                  setup, lamp_rules, [{"state_id": "lamp.lit"}, {"state_id": "light.on"}, {"state_id": "mains.on"}],
                  healthy_control=state == "healthy", fault_family="lighting")
    recover("hood_switch_recovery", hood, "Restore illumination and verify that the lamp is lit.",
            [], [command("light.set", on=True), read("lamp.lit", "lighting.illuminance_lux")],
            [condition("lamp.lit", True)],
            [option("Wait for illumination without changing the switch.", wait(10), read("lamp.lit")),
             option("Switch the lamp off.", command("light.set", on=False)),
             option("Request maintenance immediately.", escalate())], difficulty="L2", tier="T1",
            manual=["The lamp and mains are known functional in this fixture; only the light switch is initially off."],
            fixture_conditions=[condition("truth.faults", [], visibility="evaluator"), condition("mains.on", True)])
    add("hood_lamp_service", hood, "recover", "Restore illumination, or request maintenance if public controls cannot restore it. A persistent lamp failure requires maintenance.",
        [option("Turn the light off and on.", command("light.set", on=False), command("light.set", on=True)),
         option("Wait and retry the light switch.", wait(5), command("light.set", on=True)),
         option("Request maintenance for the persistent failure.", escalate())],
        setup_steps=[command("light.set", on=True), fault("lamp_failure")],
        goal_conditions=[condition("lamp.lit", True)], reference_steps=[escalate()],
        escalation_expected=True, difficulty="L3", tier="T2", fault_family="lighting",
        causes=[{"state_id": "lamp.lit"}, {"state_id": "truth.faults", "visibility": "evaluator"}])
    hidden_ids = ["light.on", "lamp.lit", "mains.on", "lighting.illuminance_lux"]
    add("hood_hidden_lamp_prediction", hood, "predict",
        "Will switching the light on establish illumination, given the recorded evidence? Select the supported operational outcome, or insufficient information when neither outcome follows from that evidence.",
        [option("Yes, it will illuminate.", answer("goal_met")),
         option("No, it will not illuminate.", answer("goal_not_met")),
         option("The available evidence is insufficient; test the light first.", answer("insufficient_information"))],
        setup_steps=[], paired_setup_steps=[fault("lamp_failure")], observation_ids=hidden_ids,
        probe_steps=[command("light.set", on=True)], goal_conditions=[condition("lamp.lit", True)],
        diagnostic_steps=[command("light.set", on=True), read("lamp.lit", "lighting.illuminance_lux")], evidence_condition="untested",
        difficulty="L3", tier="T2", fault_family="lighting", acceptable_answers=["insufficient_information"],
        causes=[{"state_id": "truth.faults", "visibility": "evaluator"}])
    add("hood_informative_light_test", hood, "measure",
        "Determine whether the unlit lamp is functional. Select a diagnostic sequence that distinguishes a healthy lamp from a persistent lamp fault.",
        [option("Read the lamp while its switch remains off.", read("lamp.lit")),
         option("Switch the light on, then read its lamp and illumination state.", command("light.set", on=True), read("lamp.lit", "lighting.illuminance_lux")),
         option("Wait, then read the lamp without operating the switch.", wait(5), read("lamp.lit")),
         option("Declare a lamp fault now.", answer("lamp_failure"))],
        setup_steps=[], paired_setup_steps=[fault("lamp_failure")], observation_ids=hidden_ids,
        difficulty="L3", tier="T2", fault_family="lighting",
        causes=[{"state_id": "lamp.lit"}, {"state_id": "truth.faults", "visibility": "evaluator"}])
    from asset_runtime import esco_efa5uduvw as hood_runtime
    hood_height = rng.choice([0.35, 0.4, 0.457])
    safe_velocity, unsafe_velocity = rng.choice([0.38, 0.4, 0.42]), rng.choice([0.22, 0.24, 0.26])

    def monitor_fixture(velocity):
        return [command("exhaust.configure", actor="evaluator", mode="constant_volume", flow_m3_s=0.4 * hood_runtime.a_eff(hood_height)),
                command("sash.set", height_m=hood_height), wait(30),
                command("monitor.calibrate", reference_m_s=0.4),
                command("exhaust.configure", actor="evaluator", mode="constant_volume", flow_m3_s=velocity * hood_runtime.a_eff(hood_height)),
                wait(30), command("monitor.calibrate", reference_m_s=0.32)]

    information_pair("hood_independent_airflow", hood,
        monitor_fixture(safe_velocity), monitor_fixture(unsafe_velocity), [],
        [condition("truth.process", 0.3, path=["face_velocity_m_s"], operator="ge", visibility="evaluator")],
        ["monitor.reading_m_s", "monitor.settings", "sash.height_m"],
        "Is the actual face velocity at least 0.3 m/s, according to the recorded evidence? An independent anemometer measurement may corroborate or contradict the monitor's calibration.",
        [command("anemometer.measure")], "sensor_bias",
        [{"state_id": "monitor.reading_m_s"}, {"state_id": "truth.process", "path": ["face_velocity_m_s"], "visibility": "evaluator"}])
    for requested_height in (rng.choice([0.2, 0.3, 0.4]), rng.choice([0.5, 0.55, 0.6])):
        predict("hood_sash_lock_interlock", hood,
                f"Will setting the sash height to {requested_height} m be accepted with the 0.457 m travel-stop lock enabled?",
                [command("keylock.set", mode="limit")], [command("sash.set", height_m=requested_height, hold=True)],
                controls=(True,) if requested_height < 0.457 else (), causes=[{"state_id": "keylock.mode"}],
                difficulty_axes={"constraint_margin_m": round(abs(requested_height - 0.457), 3)})

    # Oven: accepted settings, missing recording media, timers, sensor hardware.
    oven = "binder_fd056"
    temperature = rng.choice([308.15, 318.15, 338.15, 358.15])
    minutes = rng.randint(1, 4)
    for powered in (False, True):
        predict("oven_power_precondition", oven,
                f"Will temperature.set with temperature_k={temperature} succeed in this state?",
                [command("power.set", on=powered)], [command("temperature.set", temperature_k=temperature)],
                controls=(powered,) if powered else (), causes=[{"state_id": "power.on"}])
    for inserted in (False, True):
        predict("oven_recording_media", oven, "Will enabling USB recording succeed now?",
                [command("power.set", on=True), command("usb.set_stick", inserted=inserted)],
                [command("usb.record", on=True, interval_min=minutes)], controls=(inserted,) if inserted else (),
                causes=[{"state_id": "usb.state", "path": ["inserted"]}])
    for powered in (False, True):
        add("oven_declared_error_prediction", oven, "predict",
            f"Predict the declared error from temperature.set(temperature_k={temperature}) in the recorded power state; choose NO_ERROR if it succeeds.",
            [option("STANDBY", answer("STANDBY")), option("INVALID_ARGUMENT", answer("INVALID_ARGUMENT")),
             option("NO_ERROR", answer("NO_ERROR"))],
            setup_steps=[command("power.set", on=powered)], probe_steps=[command("temperature.set", temperature_k=temperature)],
            prediction_target="error_code", decision_kind="error_prediction", difficulty="L1", tier="T1",
            healthy_control=powered, causes=[{"state_id": "power.on"}])
    record_ref = [command("power.set", on=True), command("usb.set_stick", inserted=True),
                  command("usb.record", on=True, interval_min=minutes), wait(60 * minutes), read("usb.state")]
    recover("oven_recording_recovery", oven, f"Collect at least one USB recording row with interval {minutes} minute(s); simply accepting a recording command is insufficient.",
            [], record_ref, [condition("usb.state", True, path=["recording"]), condition("usb.state", 1, path=["rows"], operator="ge")],
            [option("Power on and start recording without media.", command("power.set", on=True), command("usb.record", on=True), wait(60 * minutes)),
             option("Insert media and wait without starting recording.", command("usb.set_stick", inserted=True), wait(60 * minutes)),
             option("Request maintenance.", escalate())], causes=[{"state_id": "usb.state"}, {"state_id": "power.on"}])
    for broken in (False, True):
        setup = [command("power.set", on=True)] + ([fault("sensor_rupture")] if broken else [])
        diagnosis("oven_sensor_diagnosis", oven, "The display uses 1999 to signal a ruptured temperature sensor. Is this a sensor fault or a normal reading?",
                  setup, [("sensor_rupture", "The temperature sensor is ruptured.", [condition("display.temperature", 1999)]),
                          ("healthy", "No sensor rupture is indicated.", [condition("display.temperature", 1999, operator="ne")])],
                  [{"state_id": "display.temperature"}], fault_family="sensor_failure", healthy_control=not broken)
    timer_ref = [command("power.set", on=True), command("timer.set", days=0, hours=0, minutes=minutes, fan_after="off"),
                 wait(minutes * 60), read("timer.state")]
    recover("oven_timer_completion", oven, f"Run a {minutes}-minute delayed-off timer through expiration and preserve its configured duration.", [], timer_ref,
            [condition("timer.state", True, path=["expired"]), condition("timer.state", minutes * 60, path=["set_s"]), condition("power.on", True)],
            [option("Set the timer and declare completion immediately.", *timer_ref[:2]),
             option("Wait without configuring or powering the timer.", wait(minutes * 60))],
            causes=[{"state_id": "timer.state"}], healthy_control=True)
    oven_adapt_ref = [wait(3), read("power.on", "timer.state"), command("power.set", on=True), wait(minutes * 60), read("timer.state", "power.on")]
    add("oven_midrun_standby_adaptation", oven, "adapt",
        f"Complete the {minutes}-minute timer and finish with the oven powered on despite an external standby interruption during execution.",
        [option("Inspect and restore operation before completing the timer.", *oven_adapt_ref),
         option("Wait without inspecting the power state.", wait(minutes * 60 + 5)),
         option("Set the power state now and then wait without rechecking.", command("power.set", on=True), wait(minutes * 60 + 5))],
        setup_steps=timer_ref[:2], reference_steps=oven_adapt_ref,
        goal_conditions=[condition("timer.state", True, path=["expired"]), condition("timer.state", minutes * 60, path=["set_s"]), condition("power.on", True)],
        scheduled_events=[{"trigger_time_s": 2.0, "steps": [command("power.set", actor="evaluator", on=False)]}],
        difficulty="L5", tier="T3", fault_family="operator_interruption", causes=[{"state_id": "power.on"}],
        difficulty_axes={"matched_control_family": "oven_timer_completion", "interruption_count": 1, "required_restore_actions": 1})

    # Furnace: preconditions, changing doors, program ramp/hold and deadlines.
    furnace = "nabertherm_l9"
    target, rate, hold = rng.choice([303.15, 308.15, 318.15, 323.15]), rng.choice([0.05, 0.1, 0.2]), rng.randint(2, 12)
    segment = {"temperature_K": target, "ramp_rate_K_s": rate, "hold_s": hold}
    duration = math.ceil((target - 298.15) / rate + hold + 20)
    configure = command("thermal.configure_program", segments=[segment])
    for closed in (False, True):
        predict("furnace_door_interlock", furnace, "Will starting the configured temperature program succeed with the current door state?",
                [command("power.set", enabled=True), configure, command("door.close" if closed else "door.open")],
                [command("thermal.start_program")], controls=(closed,) if closed else (), causes=[{"state_id": "door.closed"}])
    for supplied_rate in (rate, rng.choice([0.0, -0.1, 1.5])):
        schema_segment = dict(segment, ramp_rate_K_s=supplied_rate)
        predict("furnace_program_argument_schema", furnace,
                f"Does this program satisfy the typed argument schema: {schema_segment}? Predict whether configuration is accepted.",
                [command("power.set", enabled=True)], [command("thermal.configure_program", segments=[schema_segment])],
                controls=(True,) if supplied_rate > 0 else (), causes=[{"schema_path": "thermal.configure_program.input_schema.properties.segments.items.properties.ramp_rate_K_s"}],
                difficulty_axes={"constraint_kind": "argument_schema"})
    furnace_ref = [command("power.set", enabled=True), command("door.close"), configure,
                   command("thermal.start_program"), wait(duration), read("program.phase", "program.hold_elapsed_s")]
    goals = [condition("program.phase", "completed"), condition("program.running", False),
             condition("program.hold_elapsed_s", hold, operator="ge"), condition("program.segments", [segment])]
    recover("furnace_program_recovery", furnace,
            f"Complete the configured empty-furnace program at {target} K, {rate} K/s, holding for {hold} s. Verify completion, not command acceptance.",
            [command("door.open")], furnace_ref, goals,
            [option("Power on, configure and start without closing the door.", furnace_ref[0], configure, command("thermal.start_program"), wait(duration)),
             option("Configure, close and start but do not advance time.", *furnace_ref[:4]),
             option("Request maintenance.", escalate())], causes=[{"state_id": "door.closed"}, {"state_id": "program.phase"}])
    adapt_ref = [wait(2), read("door.closed", "program.phase"), command("door.close"), wait(duration), read("program.phase")]
    add("furnace_midrun_door_adaptation", furnace, "adapt", "Finish the running program despite a door interruption that can occur during execution. Observe and restore the door interlock before resuming.",
        [option("Advance briefly, inspect the door, close it and allow the program to finish.", *adapt_ref),
         option("Wait for the whole program without inspecting or correcting the door.", wait(duration)),
         option("Stop the program and declare success.", command("thermal.stop_program"))],
        setup_steps=[command("power.set", enabled=True), configure, command("thermal.start_program")],
        scheduled_events=[{"trigger_time_s": 2.0, "steps": [command("door.open", actor="evaluator")]}],
        reference_steps=adapt_ref, goal_conditions=goals, difficulty="L5", tier="T3",
        fault_family="interlock", causes=[{"state_id": "door.closed"}], history_from_step=0,
        difficulty_axes={"temporal_change": True, "horizon": len(adapt_ref), "active_faults": 1})

    # Hotplate: power, lock, safety margin, and an actual motor-blockage alarm.
    plate = "ika_plate_rctdigital"
    speed = rng.choice([150, 250, 400, 600]) * 2 * math.pi / 60
    for locked in (False, True):
        setup = [command("power.set", on=True)] + ([command("lock.toggle")] if locked else [])
        predict("hotplate_lock_precondition", plate, f"Will setting the stirrer speed to {speed:.8g} rad/s succeed now?",
                setup, [command("stirring.set_speed", speed_rad_s=speed)], controls=(not locked,) if not locked else (), causes=[{"state_id": "lock.on"}])
    plate_ref = [command("lock.toggle"), command("stirring.set_speed", speed_rad_s=speed), command("stirring.set", on=True), wait(2), read("stirring.on")]
    recover("hotplate_unlock_recovery", plate, f"Restore stirring at {speed:.8g} rad/s while the keypad is locked.",
            [command("power.set", on=True), command("lock.toggle")], plate_ref,
            [condition("stirring.on", True), condition("stirring.set_rad_s", speed)],
            [option("Retry the speed setting without unlocking.", command("stirring.set_speed", speed_rad_s=speed), command("stirring.set", on=True)),
             option("Switch the motor off.", command("stirring.set", on=False)), option("Request maintenance.", escalate())], causes=[{"state_id": "lock.on"}])
    for blocked in (False, True):
        setup = [command("power.set", on=True), command("stirring.set_speed", speed_rad_s=speed), command("stirring.set", on=True)]
        if blocked:
            setup.append(fault("motor_blocked"))
        setup.append(wait(1))
        diagnosis("hotplate_motor_diagnosis", plate, "Error 4 denotes a blocked motor and stops stirring. Which operational diagnosis is supported?", setup,
                  [("motor_blocked", "The motor is blocked.", [condition("error.code", 4)]),
                   ("healthy", "Stirring is running without this fault.", [condition("error.code", None), condition("stirring.on", True)])],
                  [{"state_id": "error.code"}], fault_family="motor_failure", healthy_control=not blocked)
    idle_plate = [command("power.set", on=True), command("stirring.set_speed", speed_rad_s=speed)]
    information_pair("hotplate_idle_motor", plate, idle_plate, idle_plate + [fault("motor_blocked")],
        [command("stirring.set", on=True), wait(1)], [condition("stirring.on", True), condition("error.code", None)],
        ["power.on", "stirring.on", "stirring.set_rad_s", "error.code"],
        "After requesting stirring for one second, will it remain active without an error? Decide from the recorded evidence, or choose insufficient information.",
        [command("stirring.set", on=True), wait(1), read("stirring.on", "error.code")], "motor_failure",
        [{"state_id": "error.code"}, {"state_id": "truth.faults", "visibility": "evaluator"}])
    predict("hotplate_safety_margin", plate, "Will a 463.15 K heating setpoint be accepted when the safety limit is 473.15 K and the required margin is 15 K?",
            [command("power.set", on=True), command("safety.set_limit", limit_k=473.15)],
            [command("heating.set_temperature", temperature_k=463.15)], causes=[{"state_id": "safety.limit_k"}], risk="heating")

    # Syringe pump: configured geometry, actual motion and an unrecoverable motor.
    pump = "newera_ne1000"
    pump_rate = rng.choice([50, 75, 100, 125]) * 1e-6 / 3600
    diameter = 0.02659  # Native bd_60ml table value, independently documented in upstream tests.
    pump_setup = [command("power.set", on=True), command("syringe.load", syringe="bd_60ml", liquid_id="water", fill_volume_m3=40e-6),
                  command("outlet.connect", tubing="ptfe_0p8mm", tubing_length_m=0.5, needle="none", backpressure_pa=0.0),
                  command("syringe.set_diameter", diameter_m=diameter), command("rate.set", rate_m3_s=pump_rate, units="MH")]
    for configured in (False, True):
        predict("pump_geometry_precondition", pump, "Will starting the pump succeed with its current syringe-diameter setting?",
                pump_setup if configured else [command("power.set", on=True)], [command("pump.run")],
                controls=(configured,) if configured else (), causes=[{"state_id": "syringe.diameter_m"}])
    pump_ref = pump_setup[1:] + [command("pump.run"), wait(3), read("run.state", "volume.dispensed")]
    recover("pump_geometry_recovery", pump, "Configure a loaded 60 mL syringe correctly and establish actual running delivery, rather than only accepting a rate setting.",
            [command("power.set", on=True)], pump_ref, [condition("run.state", "running"), condition("volume.dispensed", 0, path=["infused_m3"], operator="gt")],
            [option("Run without configuring syringe geometry.", command("pump.run"), wait(3)),
             option("Configure geometry but leave the pump stopped.", *pump_setup[1:]), option("Request maintenance.", escalate())], causes=[{"state_id": "syringe.diameter_m"}])
    for broken in (False, True):
        setup = pump_setup + ([fault("motor_failure")] if broken else []) + [command("pump.run"), wait(2)]
        diagnosis("pump_stall_diagnosis", pump,
                  "In this fixture the tubing is unobstructed, the collar is parked and the syringe contains liquid. The only candidate fault is motor failure. Which state is supported?", setup,
                  [("motor_failure", "The motor has failed and the pump stalled.", [condition("alarm.state", "stall")]),
                   ("healthy", "The pump is running without a stall.", [condition("alarm.state", None), condition("run.state", "running")])],
                  [{"state_id": "alarm.state"}], fault_family="motor_failure", healthy_control=not broken)
    information_pair("pump_idle_motor", pump, pump_setup, pump_setup + [fault("motor_failure")],
        [command("pump.run"), wait(2)], [condition("run.state", "running"), condition("alarm.state", None)],
        ["run.state", "alarm.state", "syringe.diameter_m", "pump.rate_m3_s", "power.on"],
        "After requesting delivery for two seconds, will the pump remain running without a stall? Decide from the recorded evidence, or choose insufficient information.",
        [command("pump.run"), wait(2), read("run.state", "alarm.state", "volume.dispensed")], "motor_failure",
        [{"state_id": "alarm.state"}, {"state_id": "truth.faults", "visibility": "evaluator"}])
    add("pump_motor_service", pump, "recover", "Establish running delivery or escalate a persistent motor failure; changing rate or dismissing the stall cannot repair hardware.",
        [option("Dismiss the stall and restart.", command("keypad.press", key="rate"), command("pump.run"), wait(2)),
         option("Power cycle, reconfigure and retry.", command("power.set", on=False), *pump_setup, command("pump.run"), wait(2)),
         option("Request maintenance for the motor.", escalate())],
        setup_steps=pump_setup + [fault("motor_failure"), command("pump.run"), wait(2)],
        reference_steps=[escalate()], goal_conditions=[condition("run.state", "running"), condition("alarm.state", None)],
        escalation_expected=True, difficulty="L4", tier="T2", fault_family="motor_failure", causes=[{"state_id": "alarm.state"}])
    pump_adapt_ref = [wait(3), read("alarm.state", "run.state"), escalate()]
    add("pump_midrun_motor_adaptation", pump, "adapt",
        "Monitor delivery during a possible mid-run motor failure. Continue only while operation remains healthy; request maintenance once an actual persistent stall is observed.",
        [option("Observe after running briefly, then escalate a stall.", *pump_adapt_ref),
         option("Request maintenance before observing a failure.", escalate()),
         option("Wait and claim successful running delivery.", wait(5), answer("goal_met"))],
        setup_steps=pump_setup + [command("pump.run")], reference_steps=pump_adapt_ref,
        scheduled_events=[{"trigger_time_s": 2.0, "steps": [fault("motor_failure")]}],
        goal_conditions=[condition("run.state", "running"), condition("alarm.state", None)],
        escalation_conditions=[condition("alarm.state", "stall")], escalation_expected=True,
        difficulty="L5", tier="T3", fault_family="motor_failure", causes=[{"state_id": "alarm.state"}],
        difficulty_axes={"interruption_count": 1, "diagnostic_horizon": 3, "fault_timing_s": 2})

    # Dewpoint sensor: supply, output validity, and service prerequisites.
    dew = "vaisala_dmt152"
    for powered in (False, True):
        predict("dewpoint_service_power", dew, "Will calibration.service succeed? Service requires power off and the sensor removed from the process.",
                [command("power.set", on=powered)], [command("calibration.service")],
                controls=(not powered,) if not powered else (), causes=[{"state_id": "power.on"}])
    for voltage in (12.0, 24.0):
        setup = [command("power.set", on=True, voltage_v=voltage), wait(1)]
        diagnosis("dewpoint_output_supply", dew,
                  "A 4–20 mA channel requires at least 21 V. Diagnose whether supply voltage is sufficient for the current output, even if digital communication can work.", setup,
                  [("insufficient_supply", "Supply is insufficient for the current output.", [condition("supply.voltage_v", 21, operator="lt")]),
                   ("healthy", "The supply meets the current-output voltage requirement.", [condition("supply.voltage_v", 21, operator="ge")])],
                  [{"state_id": "supply.voltage_v"}], fault_family="supply", healthy_control=voltage == 24)
    for damaged in (False, True):
        setup = [command("power.set", on=True, voltage_v=24)] + ([fault("sensor_damage")] if damaged else []) + [wait(1)]
        diagnosis("dewpoint_sensor_damage", dew, "T meas error indicates sensor damage. A startup frozen output alone does not establish damage. Which diagnosis is supported?", setup,
                  [("sensor_damage", "The sensor reports measurement damage.", [condition("diagnostics.errors", "T meas error", operator="contains")]),
                   ("healthy", "No damage diagnostic is present; startup may still be in progress.", [condition("diagnostics.errors", "T meas error", operator="not_contains")])],
                  [{"state_id": "diagnostics.errors"}], fault_family="sensor_failure", healthy_control=not damaged)
    service_ref = [command("power.set", on=False), command("installation.remove"), command("calibration.service"), read("calibration.age_s")]
    recover("dewpoint_service_recovery", dew, "Prepare the installed sensor for calibration service: disconnect power, remove it, then calibrate and verify its calibration age reset.",
            [command("cap.remove"), command("installation.set", mode="direct"), command("power.set", on=True)], service_ref,
            [condition("calibration.age_s", 0), condition("power.on", False), condition("installation.current", False, path=["installed"])],
            [option("Attempt calibration while powered and installed.", command("calibration.service")),
             option("Power off but leave the sensor installed.", command("power.set", on=False), command("calibration.service"))], causes=[{"state_id": "installation.current"}, {"state_id": "power.on"}])

    # Hydrogen generator: startup diagnostics are not a fault; maintenance is.
    hydrogen = "peak_precisionhydrogen"
    for ready in (False, True):
        setup = [command("power.set", on=True)] + ([wait(600)] if ready else [])
        predict("hydrogen_startup_precondition", hydrogen, "Will cycle.start succeed now? It requires completed startup diagnostics and standby.", setup,
                [command("cycle.start")], controls=(ready,) if ready else (), causes=[{"state_id": "state.current"}])
    pressure = rng.choice([250000.0, 350000.0, 450000.0])
    h_ref = [wait(600), command("settings.configure", pressure_set_pa=pressure), command("cycle.start"), wait(5), read("state.current")]
    recover("hydrogen_startup_recovery", hydrogen, f"Finish startup checks, set {pressure} Pa delivery pressure and start supplying hydrogen.",
            [command("power.set", on=True)], h_ref, [condition("state.current", "running"), condition("pressure.set_pa", pressure)],
            [option("Start immediately during diagnostics.", command("cycle.start")),
             option("Wait but do not start supplying gas.", wait(600)), option("Mute alarms without starting.", command("alarm.mute"))],
            causes=[{"state_id": "state.current"}], risk="pressurized_gas")
    add("hydrogen_cell_service", hydrogen, "recover", "Resume gas delivery or escalate a persistent cell-failure alarm. Muting acknowledges sound and does not repair the cell.",
        [option("Mute the alarm and retry delivery.", command("alarm.mute"), command("cycle.start")),
         option("Wait for a spontaneous hardware repair.", wait(20)), option("Request maintenance.", escalate())],
        setup_steps=[command("power.set", on=True), wait(600), fault("cell_failure"), wait(1)],
        reference_steps=[escalate()], goal_conditions=[condition("state.current", "running")], escalation_expected=True,
        difficulty="L4", tier="T2", fault_family="cell_failure", risk="pressurized_gas", causes=[{"state_id": "state.current"}])

    # Incubator: independent outer door/latch/inner door temporal interlocks.
    incubator = "eppendorf_cellxpertc170i"
    for outer_open in (False, True):
        setup = [command("power.set", enabled=True)] + ([command("door.outer", open=True), wait(2)] if outer_open else [])
        predict("incubator_latch_interlock", incubator, "Will releasing the inner-door latch succeed? The outer door must first be fully open.",
                setup, [command("door.latch", open=True)], controls=(outer_open,) if outer_open else (), causes=[{"state_id": "door.outer_state"}])
    door_ref = [command("door.outer", open=True), wait(2), command("door.latch", open=True), wait(1), command("door.inner", open=True), wait(2), read("door.inner_state")]
    recover("incubator_ordered_access", incubator, "Open the inner chamber door through the mechanical sequence; commands need time to complete before the next interlock is released.",
            [command("power.set", enabled=True)], door_ref, [condition("door.inner_state", "open")],
            [option("Open the inner door directly.", command("door.inner", open=True)),
             option("Issue all opening commands without waiting for motion.", command("door.outer", open=True), command("door.latch", open=True), command("door.inner", open=True)),
             option("Open only the outer door.", command("door.outer", open=True), wait(2))], causes=[{"state_id": "door.outer_state"}, {"state_id": "door.latched"}])
    for failed in (False, True):
        setup = [command("power.set", enabled=True)] + ([fault("power_loss")] if failed else []) + [wait(1)]
        diagnosis("incubator_mains_diagnosis", incubator, "Determine whether actual power is available; this fixture has no switched-off control and permits only a mains loss or normal operation.", setup,
                  [("power_loss", "Mains power has been lost.", [condition("power.enabled", False)]),
                   ("healthy", "Power remains available.", [condition("power.enabled", True)])],
                  [{"state_id": "power.enabled"}], fault_family="supply", healthy_control=not failed)

    # Match every ambiguous decision with native-tested healthy and faulted
    # controls. Only available evidence changes: question, menu and probe remain
    # identical. These three cases share a family/split equivalence group.
    answerable_controls = []
    for case in cases:
        if case.task_type != "predict" or case.paired_setup_steps is None:
            continue
        for healthy, base_setup in ((True, case.setup_steps), (False, case.paired_setup_steps)):
            answerable_controls.append(replace(
                case, setup_steps=copy.deepcopy(base_setup + case.diagnostic_steps),
                paired_setup_steps=None, acceptable_answers=[], history_from_step=0,
                healthy_control=healthy, evidence_condition="tested_healthy" if healthy else "tested_fault",
                options=copy.deepcopy(case.options), manual=list(case.manual),
                difficulty_axes={**case.difficulty_axes, "diagnostic_evidence_available": True},
            ))
    cases.extend(answerable_controls)

    for case in cases:
        if not case.decision_kind:
            if "argument_schema" in case.family_id:
                case.decision_kind = "argument_validity"
            elif "independent_airflow" in case.family_id:
                case.decision_kind = "reading_trust"
            elif case.task_type == "diagnose":
                case.decision_kind = "alarm_diagnosis"
            elif case.task_type == "predict":
                case.decision_kind = "precondition"
            elif case.task_type == "adapt" or "ordered_access" in case.family_id:
                case.decision_kind = "protocol_step"
            elif case.task_type == "measure":
                case.decision_kind = "reading_trust"
            else:
                case.decision_kind = "action_selection"
        if case.task_type in ("recover", "adapt", "measure"):
            for candidate in case.options:
                descriptions = []
                for step in candidate["steps"]:
                    if step["kind"] == "invoke":
                        descriptions.append(step["capability_id"] + " " + json.dumps(step["arguments"], sort_keys=True))
                    elif step["kind"] == "wait":
                        descriptions.append(f"wait {step['dt_s']:g} s")
                    elif step["kind"] == "read":
                        descriptions.append("read " + ", ".join(step["state_ids"] or ["public state"]))
                    elif step["kind"] == "escalate":
                        descriptions.append("request maintenance")
                    else:
                        descriptions.append("report " + str(step["answer"]))
                candidate["text"] = "Execute: " + "; ".join(descriptions)
        case.manual += [
            "Agent commands and measurements are public. Fault injection, evaluator truth and world fixtures are unavailable to the solver.",
            "A successful command means it was accepted; time-dependent completion requires a subsequent observation.",
        ]
        case.difficulty_axes = {"horizon": max(1, len(case.reference_steps)),
                                "partial_observability": case.paired_setup_steps is not None,
                                "diagnostic_hypotheses": 2 if case.paired_setup_steps is not None else 1,
                                "temporal_change": bool(case.scheduled_events),
                                "option_count": len(case.options), **case.difficulty_axes}
    return cases


def family_inventory():
    inventory = {}
    for case in build_scenarios(0):
        inventory.setdefault(case.family_id, {"family_id": case.family_id,
            "instrument_id": case.instrument_id, "domain": case.domain,
            "task_type": case.task_type, "difficulty": case.difficulty,
            "decision_kind": case.decision_kind,
            "tier": case.tier, "fault_family": case.fault_family})
    return list(inventory.values())
