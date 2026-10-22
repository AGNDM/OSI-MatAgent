"""Uniform public sessions and private fixture access for eight native runtimes.

The trusted evaluator owns this object. Solver inputs must use only describe,
read, invoke and bounded wait responses, never the host or evaluator session.
"""
from __future__ import annotations

import copy
import csv
import importlib
import math
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[1] / "external" / "ChemLab-Assets"
sys.path.insert(0, str(SOURCE / "scripts"))

INSTRUMENTS = {
    "esco_efa5uduvw": ("asset_runtime.esco_efa5uduvw", "EscoEfa5uduvwRuntime", "lab_utilities"),
    "binder_fd056": ("asset_runtime.binder_fd056", "BinderFd056Runtime", "thermal_processing"),
    "nabertherm_l9": ("basic40_runtime_thermal", "NaberthermL9Runtime", "thermal_processing"),
    "ika_plate_rctdigital": ("asset_runtime.ika_plate_rctdigital", "IkaPlateRctdigitalRuntime", "mixing_heating"),
    "newera_ne1000": ("asset_runtime.newera_ne1000", "NeweraNe1000Runtime", "liquid_delivery"),
    "vaisala_dmt152": ("asset_runtime.vaisala_dmt152", "VaisalaDmt152Runtime", "gas_metrology"),
    "peak_precisionhydrogen": ("asset_runtime.peak_precisionhydrogen", "PeakPrecisionhydrogenRuntime", "gas_generation"),
    "eppendorf_cellxpertc170i": ("asset_runtime.eppendorf_cellxpertc170i", "EppendorfCellxpertC170iRuntime", "environmental_control"),
}


def catalog_domain(instrument_id):
    with (SOURCE / "catalog/catalog.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["id"] == "instrument." + instrument_id:
                return row["domain_name"]
    raise KeyError(instrument_id)


def values(envelope):
    if "observations" in envelope:
        return {key: value["value"] for key, value in envelope["observations"].items()}
    return envelope


class InstrumentAdapter:
    def __init__(self, instrument_id, seed=0):
        module_name, class_name, self.subdomain = INSTRUMENTS[instrument_id]
        self.domain = catalog_domain(instrument_id)
        self.instrument_id, self.seed = instrument_id, int(seed)
        self.module = importlib.import_module(module_name)
        constructor = getattr(self.module, class_name)
        self.instance_id = "benchmark-instrument"
        self.trace, self._request, self._agent_request, self._public_revision = [], 0, 0, 0
        if instrument_id == "nabertherm_l9":
            # Pure-Python scene mirror: door commands are tested, no physical
            # geometry or robot-contact validation is implied by this callback.
            self.scene = {}
            self.host = constructor(scene_adapter=lambda instance, updates: self.scene.update(copy.deepcopy(updates)))
            self.host.create_instance(self.instance_id)
            self.host.register_principal("agent", visibility="agent")
            self.host.register_principal("evaluator", visibility="evaluator")
        else:
            self.host = constructor(self.instance_id)
            self.host.register_principal("agent", "agent")
            self.host.register_principal("evaluator", "evaluator")
        self.agent, self.evaluator = self.host.session("agent"), self.host.session("evaluator")
        if hasattr(self.module, "PLANT_EXAMPLE"):
            plant = dict(self.module.PLANT_EXAMPLE, noise_seed=int(seed) % (2**31))
            response = self.setup("plant.configure", plant)
            if response["status"] != "succeeded":
                raise RuntimeError(f"Cannot configure {instrument_id}: {response}")

    @property
    def time_s(self):
        return float(self.host.time_s)

    def describe(self):
        return self.agent.describe(instance_id=self.instance_id)

    def read(self, state_ids=None):
        response = self.agent.read(instance_id=self.instance_id, state_ids=state_ids)
        response["state_revision"] = self._public_revision
        return response

    def truth(self):
        return values(self.evaluator.read(instance_id=self.instance_id))

    def _invoke(self, session, capability_id, arguments, actor):
        self._request += 1
        if actor == "agent":
            self._agent_request += 1
        request_id = f"agent-{self._agent_request}" if actor == "agent" else f"fixture-{self._request}"
        response = session.invoke(instance_id=self.instance_id, capability_id=capability_id,
                                  arguments=copy.deepcopy(arguments), request_id=request_id)
        if actor == "agent":
            if response.get("status") == "succeeded":
                self._public_revision += 1
            response["state_revision"] = self._public_revision
        self.trace.append({"actor": actor, "kind": "invoke", "capability_id": capability_id,
                           "arguments": copy.deepcopy(arguments), "response": copy.deepcopy(response),
                           "time_s": self.time_s})
        return response

    def invoke(self, capability_id, arguments=None):
        return self._invoke(self.agent, capability_id, arguments or {}, "agent")

    def setup(self, capability_id, arguments=None):
        return self._invoke(self.evaluator, capability_id, arguments or {}, "evaluator")

    def wait(self, dt_s):
        if isinstance(dt_s, bool) or not math.isfinite(dt_s) or not 0 <= dt_s <= 86400:
            raise ValueError("Wait must be finite and between 0 and 86400 seconds")
        self.host.step(float(dt_s))
        self._public_revision += int(dt_s > 0)
        response = {"status": "succeeded", "clock": {"domain": "simulation", "time_s": self.time_s}}
        self.trace.append({"actor": "host", "kind": "wait", "dt_s": float(dt_s),
                           "response": response, "time_s": self.time_s})
        return response


def make_adapter(instrument_id, seed=0):
    return InstrumentAdapter(instrument_id, seed)


def apply_step(adapter, step, actor="agent"):
    step = step.to_dict() if hasattr(step, "to_dict") else step
    kind = step["kind"]
    trusted = actor in ("fixture", "evaluator")
    requested_actor = step.get("actor", "agent")
    if requested_actor == "evaluator" and not trusted:
        raise PermissionError("Solver actions cannot select an evaluator identity")
    actor = requested_actor if trusted else "agent"
    if kind == "invoke":
        call = adapter.setup if actor == "evaluator" else adapter.invoke
        return call(step["capability_id"], step.get("arguments", {}))
    if kind == "wait":
        return adapter.wait(step["dt_s"])
    if kind == "read":
        return adapter.read(step.get("state_ids"))
    if kind == "answer":
        return {"status": "answered", "answer": step["answer"]}
    if kind == "escalate":
        return {"status": "escalated"}
    raise ValueError(f"Unknown action kind: {kind}")


def apply_steps(adapter, steps, actor="agent", require_success=False):
    results = []
    for step in steps:
        result = apply_step(adapter, step, actor=actor)
        results.append(result)
        if require_success and result.get("status") not in ("succeeded", "answered", "escalated"):
            raise RuntimeError(f"Fixture/reference step failed: {step}: {result}")
    return results


def condition_holds(state, condition):
    condition = condition.to_dict() if hasattr(condition, "to_dict") else condition
    state = values(state)
    try:
        actual = state[condition["state_id"]]
        for key in condition.get("path", []):
            actual = actual[key]
    except (KeyError, IndexError, TypeError):
        return condition["operator"] == "absent"
    target, operator = condition.get("value"), condition["operator"]
    if operator == "eq":
        if isinstance(actual, (int, float)) and not isinstance(actual, bool) and isinstance(target, (int, float)):
            return math.isclose(actual, target, rel_tol=0, abs_tol=condition.get("tolerance", 1e-6))
        return actual == target
    if operator == "ne":
        return actual != target
    if operator == "contains":
        return target in actual
    if operator == "not_contains":
        return target not in actual
    if operator == "in":
        return actual in target
    if operator == "absent":
        return False
    return {"lt": lambda: actual < target, "le": lambda: actual <= target,
            "gt": lambda: actual > target, "ge": lambda: actual >= target}[operator]()


def conditions_hold(adapter, conditions):
    public, private = None, None
    for condition in conditions:
        if condition.get("visibility", "agent") == "agent":
            public = adapter.read() if public is None else public
            state = public
        else:
            private = adapter.truth() if private is None else private
            state = private
        if not condition_holds(state, condition):
            return False
    return True
