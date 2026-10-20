"""Native oven + explicitly authored balance; all mutation authority stays here.

The authored balance sensor converts native sample total mass to an observation.
There is no authored drying/success transition. Roles receive public snapshots only.
"""
from copy import deepcopy
import json
import math

from app.models import ActionProposal
from app.native import oven_runtime as rt
from app.scenarios import CaseFixture


class GateRejected(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class LabAdapter:
    def __init__(self, fixture: CaseFixture):
        self.sample_id = "sample-001"
        self.instance_id = "oven-001"
        self._world = rt.BinderFd056Runtime(self.instance_id)
        self._world.register_principal("host", "evaluator")
        self._world.register_principal("policy", "agent")
        self._host = self._world.session("host")
        self._agent = self._world.session("policy")
        self._host_request = 0
        self._requests: dict[str, tuple[str, dict]] = {}
        self.history: list[dict] = []
        self.action_count = 0
        self.revision = 0
        self.last_mutation_step = 0
        self._offset = fixture.balance_offset_kg
        self._available = fixture.balance_available
        self._scheduler_seconds = fixture.scheduler_seconds
        self._target_k = fixture.target_temperature_k
        self._dry_mass = fixture.dry_mass_kg
        self._limit = fixture.residual_limit_kg
        self._inspection = None
        self._measurement = None
        self._host_call("plant.configure", **dict(rt.PLANT_EXAMPLE, noise_seed=fixture.seed))
        # Scenario initialization is harness-owned and excluded from policy budgets.
        for cap, args in [
            ("power.set", {"on": True}),
            ("door.set", {"angle_rad": 1.57}),
            ("sample.place", dict(rt.SAMPLE_EXAMPLE, sample_id=self.sample_id,
                position=fixture.position, dry_mass_kg=fixture.dry_mass_kg,
                liquid_mass_kg=fixture.initial_liquid_kg)),
            ("door.set", {"angle_rad": 0.0}),
            ("temperature.set", {"temperature_k": fixture.initial_setpoint_k}),
        ]:
            self._host_call(cap, **args)
        sample = self._truth_sample()
        self._tare = sample["total_mass_kg"] - fixture.dry_mass_kg - fixture.initial_liquid_kg

    def _host_call(self, capability_id: str, **arguments):
        self._host_request += 1
        response = self._host.invoke(instance_id=self.instance_id,
            capability_id=capability_id, arguments=arguments,
            request_id=f"host-{self._host_request}")
        if response["status"] != "succeeded":
            raise RuntimeError(response)
        return response["result"]

    def _truth_sample(self):
        return self._world.read("evaluator")["truth.process"]["samples"][self.sample_id]

    def target(self):
        return {"task": "Check the earlier excess-mass report. Obtain a reliable fresh weighing; correct drying or the balance only if justified.",
            "sample_id": self.sample_id, "temperature_k": self._target_k,
            "tare_mass_kg": self._tare, "dry_mass_kg": self._dry_mass,
            "residual_limit_kg": self._limit}

    def capabilities(self):
        return {
            "workflow.inspect": {"source": "custom read bundle: native public oven read + authored balance self-check", "arguments": {}},
            "temperature.set": {"source": "native ChemLab CHEM-011", "arguments": {"temperature_k": "number, K; exactly task target"},
                "post_action": f"host advances native simulator {self._scheduler_seconds} s synchronously"},
            "balance.clean": {"source": "authored pan-contamination fixture", "arguments": {}},
            "balance.measure": {"source": "authored sensor over native sample mass", "arguments": {}},
        }

    def observe(self):
        # Plain copies form the only environment view sent to specialists.
        return deepcopy({"revision": self.revision, "sample_id": self.sample_id,
            "symptom": "An earlier weighing reported excess mass; its freshness and reliability are unconfirmed.",
            "inspection": self._inspection, "measurement": self._measurement,
            "last_mutation_step": self.last_mutation_step})

    def validate(self, proposal: ActionProposal, pending_operation=None):
        if pending_operation:
            raise GateRejected("PENDING_OPERATION_UNRESOLVED")
        if proposal.sample_id != self.sample_id:
            raise GateRejected("WRONG_SAMPLE")
        digest = json.dumps(proposal.model_dump(), sort_keys=True, allow_nan=False)
        if proposal.request_id in self._requests:
            if self._requests[proposal.request_id][0] != digest:
                raise GateRejected("REQUEST_CONFLICT")
            return  # A known completed request is replayed even after the revision changed.
        if proposal.expected_revision != self.revision:
            raise GateRejected("STALE_PROPOSAL")
        cap = proposal.capability_id
        if cap not in self.capabilities():
            raise GateRejected("UNSUPPORTED_CAPABILITY")
        if cap == "temperature.set":
            try:
                rt.validate(rt.CAPS[cap]["input_schema"], proposal.arguments)
            except (ValueError, TypeError, KeyError):
                raise GateRejected("INVALID_ARGUMENT") from None
            value = proposal.arguments["temperature_k"]
            if isinstance(value, bool) or not math.isfinite(value) or abs(value - self._target_k) > 1e-9:
                raise GateRejected("OUTSIDE_CORRECTION_BOUND")
            if not self._inspection:
                raise GateRejected("INSPECTION_REQUIRED")
        elif proposal.arguments:
            raise GateRejected("INVALID_ARGUMENT")
        if cap == "balance.clean" and not self._inspection:
            raise GateRejected("INSPECTION_REQUIRED")

    def execute(self, proposal: ActionProposal):
        self.validate(proposal)
        if proposal.request_id in self._requests:
            return deepcopy(self._requests[proposal.request_id][1])
        cap, args = proposal.capability_id, proposal.arguments
        self.action_count += 1
        step = self.action_count
        result = {}
        if cap == "workflow.inspect":
            fields = ["temperature.set_k", "display.temperature", "unit", "power.on", "door.open", "alarm.state"]
            native = self._agent.read(instance_id=self.instance_id, state_ids=fields)
            result = {"id": f"obs-{step}", "sample_id": self.sample_id,
                "revision": self.revision, "captured_at_step": step,
                "native": native,
                "balance_check": {"available": self._available,
                    "pan": "contaminated" if self._offset else "clean",
                    "source": "authored balance self-check"}}
            self._inspection = result
        elif cap == "temperature.set":
            response = self._agent.invoke(instance_id=self.instance_id,
                capability_id=cap, arguments=args, request_id=proposal.request_id)
            result = {"native_response": response}
            if response["status"] == "succeeded":
                # Record possible mutation before waiting. Never retry with a new ID.
                self.revision += 1
                self.last_mutation_step = step
                self._inspection = None
                self._measurement = None
                self._host_call("sim.step", dt_s=float(self._scheduler_seconds))
                result["scheduler"] = {"status": "succeeded", "dt_s": self._scheduler_seconds}
            else:
                result["error"] = response["error"]
        elif cap == "balance.clean":
            self._offset = 0.0
            self.revision += 1
            self.last_mutation_step = step
            self._inspection = None  # a consumed diagnostic record cannot authorize another correction
            self._measurement = None
            result = {"pan": "clean", "source": "authored balance fixture"}
        elif cap == "balance.measure":
            result = {"id": f"obs-{step}", "sample_id": self.sample_id,
                "revision": self.revision, "captured_at_step": step,
                "quality": "valid" if self._available and not self._offset else "suspect" if self._available else "unavailable",
                "total_mass_kg": self._truth_sample()["total_mass_kg"] + self._offset if self._available else None,
                "unit": "kg", "source": "authored balance sensor over native sample mass"}
            self._measurement = result
        event = {"step": step, "proposal": proposal.model_dump(), "result": deepcopy(result),
            "revision": self.revision, "source": self.capabilities()[cap]["source"]}
        self.history.append(event)
        self._requests[proposal.request_id] = (
            json.dumps(proposal.model_dump(), sort_keys=True, allow_nan=False), deepcopy(event))
        return deepcopy(event)

    def evaluator_snapshot(self):
        """Harness/evaluator only. This record never enters the graph or role prompts."""
        return {"sample_id": self.sample_id, "sample": deepcopy(self._truth_sample()),
            "pan_offset_kg": self._offset, "balance_available": self._available,
            "measurement": deepcopy(self._measurement), "revision": self.revision,
            "last_mutation_step": self.last_mutation_step,
            "actions": deepcopy(self.history)}
